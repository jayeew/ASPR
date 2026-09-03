#!/usr/bin/env python3
"""从 Nature Markdown 并行抽取摘要与带原文位置的句子（Claim Graph Phase 3）。"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGETS = PROJECT_ROOT / "data" / "claim_graph" / "nature_targets.parquet"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "claim_graph"
DEFAULT_CANONICAL_TARGETS = PROJECT_ROOT / "data" / "claim_graph" / "canonical_target_works.parquet"
DEFAULT_OPENALEX_BASE_URL = "https://api.openalex.org"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
ABSTRACT_SCHEMA = pa.schema([
    pa.field("article_id", pa.string(), False), pa.field("doi", pa.string(), False),
    pa.field("title", pa.string(), False), pa.field("year", pa.int16(), False),
    pa.field("paper_markdown_path", pa.string(), False), pa.field("abstract_text", pa.string(), False),
    pa.field("abstract_start_char", pa.int64(), False), pa.field("abstract_end_char", pa.int64(), False),
    pa.field("extraction_method", pa.string(), False),
])
SENTENCE_SCHEMA = pa.schema([
    pa.field("sentence_id", pa.string(), False), pa.field("article_id", pa.string(), False),
    pa.field("sentence_index", pa.int16(), False), pa.field("sentence_text", pa.string(), False),
    pa.field("markdown_start_char", pa.int64(), False), pa.field("markdown_end_char", pa.int64(), False),
])
FAILURE_COLUMNS = ("article_id", "paper_markdown_path", "reason", "detail")
EXPLICIT_ABSTRACT = re.compile(r"(?im)^#{1,3}\s*abstract\s*$")
HEADING = re.compile(r"(?m)^(#{1,3})\s+(.+?)\s*$")
BODY_HEADINGS = {"introduction", "results", "methods", "method", "discussion", "conclusions", "references"}
NOISE_LINE = re.compile(r"(?im)^\s*(?:\*\*==> picture .*? omitted <==\*\*|nature communications\s*\||article|https?://doi\.org/\S+)\s*$(?:\n|$)")
PICTURE_BLOCK = re.compile(r"(?is)\*\*----- Start of picture text -----\*\*.*?\*\*----- End of picture text -----\*\*")
CHECK_FOR_UPDATES = re.compile(r"(?i)\bcheck for updates\b")
ACCEPTED_DATE = re.compile(r"(?i)\baccepted:\s*\d{1,2}\s+\w+\s+\d{4}")
AUTHOR_AFFILIATION_MARKER = re.compile(r"\b[A-Z][A-Za-zÀ-ÿ'’\-]+\s+\d+(?:,\d+)?\b")
WORD = re.compile(r"[A-Za-z]{2,}|[\u3400-\u9fff]")
LEADING_ABSTRACT = re.compile(r"(?i)^\s*abstract\s+")


def setup_logging(path: Path, verbose: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("claim_graph.phase3")
    logger.handlers.clear(); logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler, level in ((logging.StreamHandler(sys.stdout), logging.DEBUG if verbose else logging.INFO), (logging.FileHandler(path, encoding="utf-8", mode="a"), logging.DEBUG)):
        handler.setLevel(level); handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def mask(match: re.Match[str]) -> str:
    """Replace noise without changing offsets inside the original Markdown."""
    return "".join("\n" if char == "\n" else " " for char in match.group(0))


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def cleaned_preserving_offsets(text: str) -> str:
    text = PICTURE_BLOCK.sub(mask, text)
    text = NOISE_LINE.sub(mask, text)
    return CHECK_FOR_UPDATES.sub(mask, text)


def front_body_end(text: str) -> int:
    for match in HEADING.finditer(text):
        heading = normalized(match.group(2)).lower().strip("# ")
        if heading in BODY_HEADINGS:
            return match.start()
    return min(len(text), 80_000)


def next_heading_end(text: str, start: int, level: int) -> int:
    for match in HEADING.finditer(text, pos=start):
        if len(match.group(1)) <= level:
            return match.start()
    return len(text)


def paragraph_spans(text: str, start: int, end: int) -> Iterable[tuple[int, int]]:
    cursor = start
    for match in re.finditer(r"\n\s*\n", text[start:end]):
        paragraph_end = start + match.start()
        if paragraph_end > cursor:
            yield cursor, paragraph_end
        cursor = start + match.end()
    if cursor < end:
        yield cursor, end


def plausible_abstract(candidate: str) -> bool:
    text = normalized(candidate)
    words = len(WORD.findall(text))
    if words < 45 or len(text) < 280 or "@" in text:
        return False
    if text.startswith(">") or text.lower().startswith("received:"):
        return False
    return any(mark in text for mark in (".", "?", "!"))


def looks_like_author_block(candidate: str) -> bool:
    """Reject author lines that became long enough to resemble an abstract."""
    return " & " in candidate and len(AUTHOR_AFFILIATION_MARKER.findall(candidate)) >= 3


def locate_abstract(raw: str, title: str) -> tuple[int, int, str] | None:
    """Use explicit Abstract first, then an author-following long paragraph fallback."""
    explicit = EXPLICIT_ABSTRACT.search(raw)
    if explicit:
        level = len(explicit.group(0).split()[0])
        start = explicit.end(); end = next_heading_end(raw, start, level)
        if plausible_abstract(cleaned_preserving_offsets(raw[start:end])):
            return start, end, "explicit_heading"
    body_end = front_body_end(raw)
    title_end = raw.lower().find(title.lower())
    search_start = title_end + len(title) if 0 <= title_end < body_end else 0
    masked = cleaned_preserving_offsets(raw[:body_end])
    for start, end in paragraph_spans(masked, search_start, body_end):
        candidate_start = start
        accepted = ACCEPTED_DATE.search(raw[start:end])
        if accepted:
            candidate_start = start + accepted.end()
        candidate = masked[candidate_start:end]
        if not looks_like_author_block(candidate) and plausible_abstract(candidate):
            return candidate_start, end, "front_matter_fallback"
    return None


def sentence_spans(masked_segment: str, absolute_start: int) -> list[tuple[str, int, int]]:
    """A deterministic, lightweight splitter retaining source offsets."""
    spans: list[tuple[str, int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?](?=\s+[A-Z0-9]|\s*$)", masked_segment):
        end = match.end(); text = normalized(masked_segment[start:end])
        if text:
            spans.append((text, absolute_start + start, absolute_start + end))
        start = end
    tail = normalized(masked_segment[start:])
    if tail:
        spans.append((tail, absolute_start + start, absolute_start + len(masked_segment)))
    return spans


def reconstruct_openalex_abstract(inverted_index: Any) -> str:
    """Recover one plain-text abstract from OpenAlex's positional inverted index."""
    if not isinstance(inverted_index, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, indexes in inverted_index.items():
        if isinstance(word, str) and isinstance(indexes, list):
            positions.extend((index, word) for index in indexes if isinstance(index, int) and index >= 0)
    if not positions:
        return ""
    positions.sort()
    return LEADING_ABSTRACT.sub("", normalized(" ".join(word for _, word in positions)))


def fetch_openalex_batch(
    targets: list[dict[str, Any]], base_url: str, api_key: str, timeout_seconds: int, retries: int
) -> dict[str, str]:
    """Fetch 1--50 canonical works and return recoverable abstracts keyed by Work ID."""
    work_ids = "|".join(str(target["work_id"]) for target in targets)
    parameters = {"filter": f"openalex:{work_ids}", "per-page": 100, "select": "id,abstract_inverted_index"}
    if api_key:
        parameters["api_key"] = api_key
    query = urlencode(parameters)
    url = f"{base_url.rstrip('/')}/works?{query}"
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urlopen(url, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            result: dict[str, str] = {}
            for record in payload.get("results", []):
                work_id = str(record.get("id") or "").rstrip("/").rsplit("/", 1)[-1].upper()
                abstract = reconstruct_openalex_abstract(record.get("abstract_inverted_index"))
                if work_id and abstract:
                    result[work_id] = abstract
            return result
        except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(2 ** (attempt + 1), 20))
    raise RuntimeError(f"OpenAlex 批请求失败：{last_error}")


def fetch_openalex_abstracts(
    targets: list[dict[str, Any]], args: argparse.Namespace, logger: logging.Logger
) -> dict[str, str]:
    """Parallelize bounded OpenAlex batches and retain only plain-text abstracts."""
    batches = [targets[index : index + args.openalex_batch_size] for index in range(0, len(targets), args.openalex_batch_size)]
    api_keys = load_openalex_api_keys(args.openalex_api_keys_env, args.env_file)
    workers = args.openalex_workers or min(64, max(16, (os.cpu_count() or 1) * 2))
    logger.info("[步骤 2/5] 开始请求 OpenAlex：论文=%d，批次=%d，每批最多=%d，并发=%d，密钥=%d 个", len(targets), len(batches), args.openalex_batch_size, workers, len(api_keys))
    found: dict[str, str] = {}
    failures = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="openalex") as pool:
        futures = {
            pool.submit(fetch_openalex_batch, batch, args.openalex_base_url, api_keys[(number - 1) % len(api_keys)], args.openalex_timeout_seconds, args.openalex_retries): number
            for number, batch in enumerate(batches, 1)
        }
        for future in as_completed(futures):
            number = futures[future]
            try:
                found.update(future.result())
            except RuntimeError as error:
                failures += 1
                logger.warning("OpenAlex 第 %d/%d 批失败，将对该批使用 Markdown 兜底：%s", number, len(batches), error)
            if number % 20 == 0 or number == len(batches):
                logger.info("[步骤 2/5] OpenAlex 批次进度：已完成=%d/%d，已恢复摘要=%d，失败批次=%d", number, len(batches), len(found), failures)
    logger.info("[步骤 2/5] OpenAlex 完成：恢复摘要=%d/%d，未恢复=%d", len(found), len(targets), len(targets) - len(found))
    return found


def read_env_value(path: Path, key: str) -> str:
    """Read one .env KEY=VALUE entry without exposing or modifying secret material."""
    if not path.is_file():
        return ""
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            name, separator, value = line.partition("=")
            if separator and name.strip() == key:
                return value.strip().strip("\"'")
    except OSError as error:
        raise ValueError(f"无法读取环境变量文件：{path} ({error})") from error
    return ""


def load_openalex_api_keys(key_name: str, env_file: Path) -> list[str]:
    """Load comma-separated API keys from the process environment or the project .env."""
    raw = os.environ.get(key_name, "").strip() or read_env_value(env_file, key_name)
    keys = [key.strip() for key in raw.split(",") if key.strip()]
    return keys or [""]


def build_complete_result(
    target: dict[str, Any], abstract: str, method: str, sentence_offsets: list[tuple[str, int, int]]
) -> dict[str, Any]:
    """Build a Phase 3 success record with stable sentence IDs and source metadata."""
    article_id = str(target["article_id"])
    path = str(target["paper_markdown_path"])
    return {
        "article_id": article_id,
        "paper_markdown_path": path,
        "status": "complete",
        "abstract": {
            "article_id": article_id,
            "doi": target["doi"],
            "title": target["title"],
            "year": target["year"],
            "paper_markdown_path": path,
            "abstract_text": abstract,
            "abstract_start_char": -1 if method == "openalex_api" else sentence_offsets[0][1],
            "abstract_end_char": -1 if method == "openalex_api" else sentence_offsets[-1][2],
            "extraction_method": method,
        },
        "sentences": [
            {
                "sentence_id": f"{article_id}::S{number:02d}",
                "article_id": article_id,
                "sentence_index": number,
                "sentence_text": text,
                "markdown_start_char": -1 if method == "openalex_api" else source_start,
                "markdown_end_char": -1 if method == "openalex_api" else source_end,
            }
            for number, (text, source_start, source_end) in enumerate(sentence_offsets, 1)
        ],
    }


def extract_one(index: int, target: dict[str, Any], chunk_dir: str, openalex_abstract: str | None) -> dict[str, Any]:
    article_id = str(target["article_id"]); path = Path(str(target["paper_markdown_path"]))
    result: dict[str, Any] = {"article_id": article_id, "paper_markdown_path": str(path), "status": "failed", "reason": "", "detail": ""}
    if openalex_abstract:
        sentences = sentence_spans(openalex_abstract, 0)
        if len(sentences) >= 2:
            result = build_complete_result(target, openalex_abstract, "openalex_api", sentences)
            chunk = Path(chunk_dir) / f"{index:06d}.json"
            chunk.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
            return result
    if result["status"] != "complete":
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            result.update(reason="markdown_unreadable", detail=str(error))
        else:
            located = locate_abstract(raw, str(target["title"]))
            if located is None:
                result.update(reason="abstract_not_found", detail="OpenAlex 无摘要，且未找到可信 Markdown 摘要")
            else:
                start, end, method = located; masked = cleaned_preserving_offsets(raw[start:end]); abstract = normalized(masked)
                sentences = sentence_spans(masked, start)
                if len(sentences) < 2:
                    result.update(reason="too_few_sentences", detail="摘要切句后少于 2 句")
                else:
                    result = build_complete_result(target, abstract, method, sentences)
    chunk = Path(chunk_dir) / f"{index:06d}.json"
    chunk.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def connect_state(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE IF NOT EXISTS abstract_state (article_id TEXT PRIMARY KEY, status TEXT, updated_at TEXT)")
    return connection


def iter_chunks(directory: Path) -> Iterable[dict[str, Any]]:
    for path in sorted(directory.glob("*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


def write_parquet(rows: Iterable[dict[str, Any]], schema: pa.Schema, path: Path) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp"); writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    batch: list[dict[str, Any]] = []; count = 0
    try:
        for row in rows:
            batch.append(row)
            if len(batch) == 10_000:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema)); count += len(batch); batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema)); count += len(batch)
    finally:
        writer.close()
    temporary.replace(path); return count


def attach_canonical_work_ids(targets: list[dict[str, Any]], canonical_path: Path) -> list[dict[str, Any]]:
    """Attach the one canonical OpenAlex Work ID selected by Phase 2 to each Nature target."""
    if not canonical_path.is_file():
        raise FileNotFoundError(f"Phase 2 的 canonical_target_works.parquet 不存在：{canonical_path}")
    canonical = pq.read_table(canonical_path).to_pylist()
    work_ids = {
        str(row["nature_article_id"]): str(row["work_id"])
        for row in canonical
        if row.get("is_nature_target") and row.get("nature_article_id") and row.get("work_id")
    }
    missing = [str(target["article_id"]) for target in targets if str(target["article_id"]) not in work_ids]
    if missing:
        raise ValueError(f"{len(missing)} 篇目标论文缺少 canonical OpenAlex Work ID，例如：{missing[:3]}")
    return [{**target, "work_id": work_ids[str(target["article_id"])]} for target in targets]


def prune_excluded_chunks(chunk_dir: Path, state: sqlite3.Connection, valid_article_ids: set[str]) -> int:
    """Remove Phase 3 records no longer present in the authoritative target table."""
    removed = 0
    for path in chunk_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("article_id") or "") not in valid_article_ids:
            path.unlink(missing_ok=True)
            removed += 1
    state_article_ids = {str(row[0]) for row in state.execute("SELECT article_id FROM abstract_state")}
    removed_state_ids = state_article_ids - valid_article_ids
    if removed_state_ids:
        state.executemany(
            "DELETE FROM abstract_state WHERE article_id = ?",
            ((article_id,) for article_id in removed_state_ids),
        )
        state.commit()
    return removed


def run(args: argparse.Namespace) -> None:
    if not args.targets.is_file(): raise FileNotFoundError(f"目标论文表不存在：{args.targets}")
    targets = pq.read_table(args.targets).to_pylist()
    if args.limit: targets = targets[: args.limit]
    targets = attach_canonical_work_ids(targets, args.canonical_targets)
    root = args.output_root; root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(root / "logs" / "phase3_extract_abstracts.log", args.verbose)
    state_path = root / "build_state.sqlite"; chunk_dir = root / "chunks" / "abstracts"
    if args.restart:
        shutil.rmtree(chunk_dir, ignore_errors=True)
        for path in (root / "abstracts.parquet", root / "abstract_sentences.parquet", root / "abstract_failures.csv"): path.unlink(missing_ok=True)
        if state_path.exists():
            reset_state = connect_state(state_path)
            reset_state.execute("DELETE FROM abstract_state")
            reset_state.commit()
            reset_state.close()
    state = connect_state(state_path)
    valid_article_ids = {str(target["article_id"]) for target in targets}
    if args.prune_excluded:
        removed = prune_excluded_chunks(chunk_dir, state, valid_article_ids)
        logger.info("[步骤 0/5] 已删除不再属于目标集的摘要 chunk：%d", removed)
    done = {row[0] for row in state.execute("SELECT article_id FROM abstract_state WHERE status='complete'")} if args.resume else set()
    if not args.resume and not args.restart and done: raise FileExistsError("已有摘要断点；继续请传 --resume，重建请传 --restart")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pending = [(index, target) for index, target in enumerate(targets, 1) if str(target["article_id"]) not in done]
    if args.workers < 0 or args.openalex_workers < 0 or args.openalex_batch_size < 1 or args.openalex_batch_size > 50:
        raise ValueError("--workers/--openalex-workers 不能为负数，--openalex-batch-size 必须在 1--50")
    workers = args.workers or max(1, os.cpu_count() or 1)
    logger.info("[步骤 1/5] 目标=%d，已完成=%d，待处理=%d，本地兜底并发=%d", len(targets), len(done), len(pending), workers)
    openalex_abstracts = fetch_openalex_abstracts([target for _, target in pending], args, logger) if pending else {}
    started = time.monotonic(); complete_count = len(done); failure_count = 0
    logger.info("[步骤 3/5] OpenAlex 摘要优先切句；仅未命中项读取 Markdown 兜底")
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="abstract-extract") as pool:
        futures = {
            pool.submit(extract_one, index, target, str(chunk_dir), openalex_abstracts.get(str(target["work_id"]))): target["article_id"]
            for index, target in pending
        }
        for future in as_completed(futures):
            result = future.result(); complete_count += result["status"] == "complete"; failure_count += result["status"] != "complete"
            state.execute("INSERT OR REPLACE INTO abstract_state VALUES (?, ?, ?)", (result["article_id"], result["status"], datetime.now(UTC).isoformat(timespec="seconds"))); state.commit()
            total_done = complete_count + failure_count
            if total_done % args.progress_every == 0 or total_done == len(targets):
                elapsed = max(time.monotonic() - started, 0.001)
                logger.info("[步骤 3/5] 进度：完成=%d/%d，成功=%d，失败=%d，速度=%.2f 篇/秒", total_done, len(targets), complete_count, failure_count, total_done / elapsed)
            if result["status"] != "complete": logger.debug("论文 %s 失败：%s | %s", result["article_id"], result["reason"], result["detail"])
    logger.info("[步骤 4/5] 合并摘要与句子 Parquet")
    chunks = [row for row in iter_chunks(chunk_dir) if str(row["article_id"]) in valid_article_ids]
    abstracts = write_parquet((row["abstract"] for row in chunks if row["status"] == "complete"), ABSTRACT_SCHEMA, root / "abstracts.parquet")
    sentences = write_parquet((sentence for row in chunks if row["status"] == "complete" for sentence in row["sentences"]), SENTENCE_SCHEMA, root / "abstract_sentences.parquet")
    logger.info("[步骤 5/5] 写入失败 CSV")
    with (root / "abstract_failures.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_COLUMNS); writer.writeheader()
        for row in chunks:
            if row["status"] != "complete": writer.writerow({"article_id": row["article_id"], "paper_markdown_path": row["paper_markdown_path"], "reason": row["reason"], "detail": row["detail"]})
    state.close(); logger.info("Phase 3 完成：摘要=%d，句子=%d，失败=%d", abstracts, sentences, sum(row["status"] != "complete" for row in chunks))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--canonical-targets", type=Path, default=DEFAULT_CANONICAL_TARGETS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--openalex-base-url", default=DEFAULT_OPENALEX_BASE_URL)
    parser.add_argument("--openalex-batch-size", type=int, default=50)
    parser.add_argument("--openalex-api-keys-env", default="OPENALEX_API_KEYS")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--openalex-workers", type=int, default=0, help="0=自动使用 16--64 个 OpenAlex 请求线程")
    parser.add_argument("--openalex-timeout-seconds", type=int, default=60)
    parser.add_argument("--openalex-retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0, help="0=使用全部逻辑 CPU"); parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--limit", type=int); parser.add_argument("--resume", action="store_true"); parser.add_argument("--restart", action="store_true"); parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--prune-excluded", action="store_true", help="删除已不在目标论文表中的旧 Phase 3 records")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.progress_every <= 0: print("Phase 3 构建失败：--progress-every 必须为正数", file=sys.stderr); return 1
    try: run(args)
    except (OSError, ValueError, RuntimeError, sqlite3.Error, pa.ArrowException) as error:
        print(f"Phase 3 构建失败：{error}", file=sys.stderr); return 1
    return 0


if __name__ == "__main__": raise SystemExit(main())
