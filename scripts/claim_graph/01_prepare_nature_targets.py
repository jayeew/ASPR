#!/usr/bin/env python3
"""从本地 Nature manifest 构建 Claim Graph Phase 1 目标论文表。"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path("/mnt/d/aspr_nature_markdown/manifest.jsonl")
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "claim_graph" / "nature_targets.parquet"
DEFAULT_FAILURES = PROJECT_ROOT / "data" / "claim_graph" / "nature_target_failures.csv"
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
TITLE_PATTERN = re.compile(r"^#{1,2}\s+(.+?)\s*$", re.MULTILINE)
DATE_PATTERN = re.compile(r"\b(Received|Accepted):\s*(\d{1,2}\s+\w+\s+\d{4})")
TARGET_SCHEMA = pa.schema(
    [
        pa.field("article_id", pa.string(), nullable=False),
        pa.field("doi", pa.string(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("year", pa.int16(), nullable=False),
        pa.field("journal_id", pa.string()),
        pa.field("pair_key", pa.string()),
        pa.field("paper_markdown_path", pa.string(), nullable=False),
        pa.field("peer_review_markdown_path", pa.string()),
        pa.field("received_date_text", pa.string()),
        pa.field("accepted_date_text", pa.string()),
    ]
)
FAILURE_COLUMNS = (
    "line_number",
    "article_id",
    "paper_markdown_path",
    "reason",
    "detail",
)


def setup_logging(log_path: Path, verbose: bool) -> logging.Logger:
    """Create Chinese console and file logging for this standalone stage."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("claim_graph.phase1")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    for handler, level in (
        (logging.StreamHandler(sys.stdout), logging.DEBUG if verbose else logging.INFO),
        (logging.FileHandler(log_path, encoding="utf-8", mode="a"), logging.DEBUG),
    ):
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def normalize_doi(value: object) -> str | None:
    """Normalize DOI URLs and remove trailing punctuation."""
    text = unquote(str(value or "")).strip().lower()
    if not text:
        return None
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi\s*:\s*", "", text)
    match = DOI_PATTERN.search(text)
    if match is None:
        return None
    return match.group(0).rstrip(".,;:)]}>\\'")


def extract_doi(markdown: str, article_id: str) -> str | None:
    """Prefer the DOI belonging to this Nature article, then use the legacy fallback."""
    expected_suffix = article_id.lower()
    for match in DOI_PATTERN.finditer(markdown):
        candidate = normalize_doi(match.group(0))
        if candidate and candidate.rsplit("/", 1)[-1] == expected_suffix:
            return candidate
    for line in markdown.splitlines():
        if "doi.org/" in line.lower():
            doi = normalize_doi(line)
            if doi:
                return doi
    return normalize_doi(markdown)


def clean_title(value: str) -> str:
    """Collapse Markdown title spacing without rewriting title content."""
    text = re.sub(r"\s+", " ", value).strip()
    return text.strip("# ")


def extract_title(markdown: str) -> str | None:
    """Extract the first article title heading before section headings begin."""
    ignored = {"abstract", "introduction", "results", "methods", "article"}
    for match in TITLE_PATTERN.finditer(markdown):
        title = clean_title(match.group(1))
        if title and title.lower() not in ignored:
            return title
    return None


def extract_date_texts(markdown: str) -> tuple[str | None, str | None]:
    """Preserve optional Received/Accepted text for future temporal audits."""
    values = {label.lower(): text for label, text in DATE_PATTERN.findall(markdown)}
    return values.get("received"), values.get("accepted")


def failure_row(
    line_number: int, article_id: str, markdown_path: str, reason: str, detail: str
) -> dict[str, str | int]:
    return {
        "line_number": line_number,
        "article_id": article_id,
        "paper_markdown_path": markdown_path,
        "reason": reason,
        "detail": detail,
    }


def iter_manifest(path: Path) -> Iterable[tuple[int, dict[str, Any] | None, str | None]]:
    """Yield parsed manifest rows while keeping malformed lines recoverable."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                yield line_number, None, "空行"
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                yield line_number, None, f"JSON 无法解析：{error.msg}"
                continue
            if not isinstance(payload, dict):
                yield line_number, None, "JSONL 记录不是对象"
                continue
            yield line_number, payload, None


def build_targets(
    manifest: Path,
    output: Path,
    failures: Path,
    min_year: int,
    max_year: int,
    markdown_max_bytes: int,
    limit: int | None,
    progress_every: int,
    excluded_article_ids: set[str],
    overwrite: bool,
    verbose: bool,
) -> dict[str, int]:
    """Materialize validated Nature targets and a separate failure table."""
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest 不存在：{manifest}")
    if markdown_max_bytes <= 0:
        raise ValueError("--markdown-max-bytes 必须为正数")
    if min_year > max_year:
        raise ValueError("--min-year 不能大于 --max-year")
    if limit is not None and limit <= 0:
        raise ValueError("--limit 必须为正数")
    if progress_every <= 0:
        raise ValueError("--progress-every 必须为正数")
    if (output.exists() or failures.exists()) and not overwrite:
        raise FileExistsError("输出已存在；如需重建请显式传入 --overwrite")

    output.parent.mkdir(parents=True, exist_ok=True)
    failures.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output.parent / "logs" / "phase1_prepare_targets.log", verbose)
    logger.info("[步骤 1/5] 校验输入与输出路径")
    logger.info("manifest：%s", manifest)
    logger.info("目标表：%s", output)
    logger.info("失败表：%s", failures)
    logger.info("年份范围：%d-%d；读取前段字节数：%d", min_year, max_year, markdown_max_bytes)
    logger.info("显式排除 article ID：%d 篇", len(excluded_article_ids))
    logger.info("[步骤 2/5] 开始逐行读取 Nature manifest")
    records: list[dict[str, Any]] = []
    failures_rows: list[dict[str, str | int]] = []
    seen_article_ids: set[str] = set()
    processed = 0

    for line_number, payload, parse_error in iter_manifest(manifest):
        if limit is not None and processed >= limit:
            break
        processed += 1
        if parse_error is not None or payload is None:
            failures_rows.append(failure_row(line_number, "", "", "manifest_invalid", parse_error or "未知错误"))
            logger.debug("第 %d 行 manifest 失败：%s", line_number, parse_error)
            continue

        article_id = str(payload.get("article_id") or "").strip()
        markdown_path_text = str(payload.get("paper_markdown_path") or "").strip()
        if article_id in excluded_article_ids:
            logger.info("论文 %s 被显式排除", article_id)
            continue
        if not article_id or not markdown_path_text:
            failures_rows.append(
                failure_row(line_number, article_id, markdown_path_text, "missing_required_field", "缺少 article_id 或 paper_markdown_path")
            )
            logger.debug("论文 %s 失败：缺少必要字段", article_id or "<unknown>")
            continue
        if article_id in seen_article_ids:
            failures_rows.append(failure_row(line_number, article_id, markdown_path_text, "duplicate_article_id", "article_id 重复"))
            logger.debug("论文 %s 失败：article_id 重复", article_id)
            continue
        seen_article_ids.add(article_id)

        try:
            year = int(payload.get("year"))
        except (TypeError, ValueError):
            failures_rows.append(failure_row(line_number, article_id, markdown_path_text, "invalid_year", "year 不是整数"))
            logger.debug("论文 %s 失败：year 不是整数", article_id)
            continue
        if not min_year <= year <= max_year:
            failures_rows.append(
                failure_row(line_number, article_id, markdown_path_text, "year_out_of_range", f"year={year}，期望范围={min_year}-{max_year}")
            )
            logger.debug("论文 %s 失败：year=%d 超出范围", article_id, year)
            continue

        markdown_path = Path(markdown_path_text)
        if not markdown_path.is_file():
            failures_rows.append(failure_row(line_number, article_id, markdown_path_text, "markdown_missing", "论文 Markdown 不存在"))
            logger.debug("论文 %s 失败：Markdown 不存在", article_id)
            continue
        try:
            with markdown_path.open("r", encoding="utf-8", errors="replace") as handle:
                markdown = handle.read(markdown_max_bytes)
        except OSError as error:
            failures_rows.append(failure_row(line_number, article_id, markdown_path_text, "markdown_unreadable", str(error)))
            logger.debug("论文 %s 失败：Markdown 无法读取：%s", article_id, error)
            continue

        doi = extract_doi(markdown, article_id)
        title = extract_title(markdown)
        if doi is None:
            failures_rows.append(failure_row(line_number, article_id, markdown_path_text, "doi_not_found", "前段 Markdown 未找到 DOI"))
            logger.debug("论文 %s 失败：未找到 DOI", article_id)
            continue
        if title is None:
            failures_rows.append(failure_row(line_number, article_id, markdown_path_text, "title_not_found", "前段 Markdown 未找到标题 heading"))
            logger.debug("论文 %s 失败：未找到标题", article_id)
            continue
        received, accepted = extract_date_texts(markdown)
        records.append(
            {
                "article_id": article_id,
                "doi": doi,
                "title": title,
                "year": year,
                "journal_id": str(payload.get("journal_id") or "") or None,
                "pair_key": str(payload.get("pair_key") or "") or None,
                "paper_markdown_path": str(markdown_path),
                "peer_review_markdown_path": str(payload.get("peer_review_markdown_path") or "") or None,
                "received_date_text": received,
                "accepted_date_text": accepted,
            }
        )
        if processed % progress_every == 0:
            logger.info(
                "[步骤 2/5] 进度：已读取=%d，成功=%d，失败=%d，当前=%s",
                processed,
                len(records),
                len(failures_rows),
                article_id,
            )

    logger.info("[步骤 3/5] 整理并写入目标论文 Parquet 表")
    records.sort(key=lambda row: row["article_id"])
    table = pa.Table.from_pylist(records, schema=TARGET_SCHEMA)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    pq.write_table(table, temporary_output, compression="zstd")
    temporary_output.replace(output)
    logger.info("[步骤 4/5] 写入失败记录 CSV")
    with failures.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_COLUMNS)
        writer.writeheader()
        writer.writerows(failures_rows)
    logger.info("[步骤 5/5] Phase 1 完成：读取=%d，成功=%d，失败=%d", processed, len(records), len(failures_rows))
    logger.info("目标论文表：%s", output)
    logger.info("失败记录：%s", failures)
    return {"processed": processed, "success": len(records), "failure": len(failures_rows)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--min-year", type=int, default=2023)
    parser.add_argument("--max-year", type=int, default=2025)
    parser.add_argument("--markdown-max-bytes", type=int, default=262_144)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--exclude-article-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        build_targets(
            manifest=args.manifest,
            output=args.output,
            failures=args.failures,
            min_year=args.min_year,
            max_year=args.max_year,
            markdown_max_bytes=args.markdown_max_bytes,
            limit=args.limit,
            progress_every=args.progress_every,
            excluded_article_ids={value.strip() for value in args.exclude_article_id if value.strip()},
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
    except (OSError, ValueError, pa.ArrowException) as error:
        print(f"Phase 1 构建失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
