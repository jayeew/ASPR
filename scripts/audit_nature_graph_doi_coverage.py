#!/usr/bin/env python3
"""审计本地 Nature 全文是否被 41 万论文引用图按 DOI 覆盖。

脚本默认无需参数即可运行。审计结果、断点状态、逐篇匹配关系和汇总统计
均写入 data/doi_coverage_audit/coverage_audit.sqlite3。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sqlite3
import sys
import uuid
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_DIR = (
    PROJECT_ROOT / "data/knowledge_corpus/nature_multihorizon_v6_1_uncapped_v2"
)
DEFAULT_GRAPH_PAPERS = DEFAULT_GRAPH_DIR / "papers_primary_articles.parquet"
DEFAULT_GRAPH_METADATA_MANIFEST = (
    DEFAULT_GRAPH_DIR / "target_openalex_metadata_reference_seed.manifest.json"
)
DEFAULT_NATURE_MANIFEST = Path("/mnt/d/aspr_nature_markdown/manifest.jsonl")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/doi_coverage_audit"
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
OPENALEX_PREFIX = "https://openalex.org/"
SCHEMA_VERSION = "nature-graph-doi-audit-v2"


@dataclass(frozen=True)
class NatureInput:
    """一条 Nature manifest 输入。"""

    article_id: str
    markdown_path: Path
    year: int | None
    journal_id: str
    pair_key: str


@dataclass(frozen=True)
class NatureResult:
    """一篇 Nature Markdown 的 DOI 提取结果。"""

    article_id: str
    markdown_path: str
    year: int | None
    journal_id: str
    pair_key: str
    doi_raw: str | None
    doi_normalized: str | None
    extraction_source: str
    status: str
    error: str | None
    file_size: int | None
    file_mtime_ns: int | None


def utc_now() -> str:
    """返回可排序的 UTC 时间。"""
    return datetime.now(UTC).isoformat(timespec="seconds")


def normalize_openalex_id(value: Any) -> str:
    """将 OpenAlex ID 统一为完整 URL。"""
    text = str(value or "").strip()
    if not text:
        return ""
    short = text.rsplit("/", 1)[-1].upper()
    if not short.startswith("W") or not short[1:].isdigit():
        return text
    return f"{OPENALEX_PREFIX}{short}"


def normalize_doi(value: Any) -> str | None:
    """把 DOI URL、doi: 前缀和常见尾部标点归一化。"""
    text = unquote(str(value or "")).strip().lower()
    if not text:
        return None
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi\s*:\s*", "", text)
    match = DOI_PATTERN.search(text)
    if match is None:
        return None
    doi = match.group(0).rstrip(".,;:)]}>\\'")
    return doi if DOI_PATTERN.fullmatch(doi) else None


def infer_nature_doi_year(doi: str | None) -> int | None:
    """从现代 Nature DOI 的 sXXXXX-YY/YYY- 片段推断年份。"""
    match = re.match(r"^10\.1038/s\d+-(\d{2,3})-", str(doi or ""), re.IGNORECASE)
    if match is None:
        return None
    return 2000 + (int(match.group(1)) % 100)


def path_fingerprint(*paths: Path) -> str:
    """构建足以判断本地输入是否变化的轻量指纹。"""
    parts: list[str] = [SCHEMA_VERSION]
    for path in paths:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        parts.append(f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}")
    import hashlib

    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def setup_logging(log_path: Path, verbose: bool) -> logging.Logger:
    """同时输出中文控制台日志和可追加的文件日志。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nature_graph_doi_audit")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def connect_database(path: Path) -> sqlite3.Connection:
    """打开审计数据库并启用适合断点写入的 WAL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=60000")
    connection.execute("PRAGMA foreign_keys=ON")
    create_schema(connection)
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    """创建原始数据、断点、匹配和汇总表。"""
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS audit_runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL,
            config_json TEXT NOT NULL,
            summary_json TEXT,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS pipeline_state (
            stage TEXT PRIMARY KEY,
            fingerprint TEXT NOT NULL,
            status TEXT NOT NULL,
            checkpoint INTEGER NOT NULL DEFAULT 0,
            total INTEGER,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS graph_papers (
            paper_id TEXT PRIMARY KEY,
            doi_raw TEXT,
            doi_normalized TEXT,
            publication_year INTEGER,
            domain12 TEXT,
            venue_family TEXT,
            status TEXT NOT NULL,
            source_table TEXT NOT NULL,
            input_fingerprint TEXT NOT NULL,
            processed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_graph_doi
            ON graph_papers(doi_normalized);
        CREATE TABLE IF NOT EXISTS nature_papers (
            article_id TEXT PRIMARY KEY,
            doi_raw TEXT,
            doi_normalized TEXT,
            extraction_source TEXT NOT NULL,
            markdown_path TEXT NOT NULL,
            year INTEGER,
            journal_id TEXT,
            pair_key TEXT,
            status TEXT NOT NULL,
            error TEXT,
            file_size INTEGER,
            file_mtime_ns INTEGER,
            processed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_nature_doi
            ON nature_papers(doi_normalized);
        CREATE TABLE IF NOT EXISTS doi_match_pairs (
            article_id TEXT NOT NULL,
            paper_id TEXT NOT NULL,
            doi_normalized TEXT NOT NULL,
            matched_at TEXT NOT NULL,
            PRIMARY KEY(article_id, paper_id),
            FOREIGN KEY(article_id) REFERENCES nature_papers(article_id),
            FOREIGN KEY(paper_id) REFERENCES graph_papers(paper_id)
        );
        CREATE TABLE IF NOT EXISTS doi_match_conflicts (
            article_id TEXT NOT NULL,
            paper_id TEXT NOT NULL,
            doi_normalized TEXT NOT NULL,
            nature_year INTEGER,
            graph_year INTEGER,
            reason TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            PRIMARY KEY(article_id, paper_id),
            FOREIGN KEY(article_id) REFERENCES nature_papers(article_id),
            FOREIGN KEY(paper_id) REFERENCES graph_papers(paper_id)
        );
        CREATE TABLE IF NOT EXISTS nature_coverage (
            article_id TEXT PRIMARY KEY,
            doi_normalized TEXT,
            coverage_status TEXT NOT NULL,
            match_count INTEGER NOT NULL,
            matched_paper_ids TEXT,
            audited_at TEXT NOT NULL,
            FOREIGN KEY(article_id) REFERENCES nature_papers(article_id)
        );
        CREATE TABLE IF NOT EXISTS coverage_breakdown (
            year INTEGER,
            journal_id TEXT,
            total INTEGER NOT NULL,
            valid_doi INTEGER NOT NULL,
            matched INTEGER NOT NULL,
            not_in_graph INTEGER NOT NULL,
            missing_doi INTEGER NOT NULL,
            coverage_rate_valid_doi REAL,
            audited_at TEXT NOT NULL,
            PRIMARY KEY(year, journal_id)
        );
        """)
    connection.commit()


def discover_graph_doi_table(explicit: Path | None) -> Path:
    """优先使用显式路径，否则从现有构建 manifest 找原始 DOI 表。"""
    if explicit is not None:
        return explicit
    payload = json.loads(DEFAULT_GRAPH_METADATA_MANIFEST.read_text(encoding="utf-8"))
    candidate = Path(str(payload.get("target_works", "")))
    if not candidate.is_file():
        raise FileNotFoundError(
            "无法找到含 DOI 的 target works 表；请通过 --graph-doi-table 指定"
        )
    return candidate


def read_nature_manifest(path: Path, limit: int | None) -> list[NatureInput]:
    """读取、校验并按 article_id 去重 Nature manifest。"""
    rows: dict[str, NatureInput] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"manifest 第 {line_number} 行不是合法 JSON") from exc
            article_id = str(item.get("article_id", "")).strip()
            markdown = str(
                item.get("paper_markdown_path")
                or item.get("project_paper_markdown_link")
                or ""
            ).strip()
            if not article_id or not markdown:
                raise ValueError(
                    f"manifest 第 {line_number} 行缺少 article_id/Markdown"
                )
            year_value = item.get("year")
            rows[article_id] = NatureInput(
                article_id=article_id,
                markdown_path=Path(markdown),
                year=int(year_value) if year_value is not None else None,
                journal_id=str(item.get("journal_id", "")),
                pair_key=str(item.get("pair_key", "")),
            )
            if limit is not None and len(rows) >= limit:
                break
    return list(rows.values())


def extract_nature_doi(item: NatureInput, max_bytes: int) -> NatureResult:
    """从 Markdown 头部抽取 DOI；缺失时使用 Nature article_id 可追溯回退。"""
    try:
        stat = item.markdown_path.stat()
        with item.markdown_path.open("rb") as handle:
            text = handle.read(max_bytes).decode("utf-8", errors="replace")
        matches = [normalize_doi(value) for value in DOI_PATTERN.findall(text)]
        valid = [value for value in matches if value]
        preferred = next(
            (value for value in valid if value.startswith("10.1038/")), None
        )
        if preferred:
            return nature_result(item, preferred, preferred, "markdown", "ok", stat)
        fallback = normalize_doi(f"10.1038/{item.article_id}")
        if fallback:
            return nature_result(
                item, fallback, fallback, "article_id_fallback", "ok_fallback", stat
            )
        return nature_result(item, None, None, "none", "doi_missing", stat)
    except OSError as exc:
        return NatureResult(
            item.article_id,
            str(item.markdown_path),
            item.year,
            item.journal_id,
            item.pair_key,
            None,
            None,
            "none",
            "read_error",
            f"{type(exc).__name__}: {exc}",
            None,
            None,
        )


def nature_result(
    item: NatureInput,
    raw: str | None,
    normalized: str | None,
    source: str,
    status: str,
    stat: os.stat_result,
) -> NatureResult:
    """构造成功读取文件后的 NatureResult。"""
    return NatureResult(
        item.article_id,
        str(item.markdown_path),
        item.year,
        item.journal_id,
        item.pair_key,
        raw,
        normalized,
        source,
        status,
        None,
        stat.st_size,
        stat.st_mtime_ns,
    )


def existing_nature_signatures(
    connection: sqlite3.Connection,
) -> dict[str, tuple[str, int | None, int | None]]:
    """读取已完成记录，用于逐文件断点续跑。"""
    rows = connection.execute(
        "SELECT article_id, markdown_path, file_size, file_mtime_ns FROM nature_papers"
    )
    return {
        str(row["article_id"]): (
            str(row["markdown_path"]),
            row["file_size"],
            row["file_mtime_ns"],
        )
        for row in rows
    }


def nature_needs_processing(
    item: NatureInput, existing: dict[str, tuple[str, int | None, int | None]]
) -> bool:
    """判断文件是否新增或自上次审计后发生变化。"""
    prior = existing.get(item.article_id)
    if prior is None:
        return True
    try:
        stat = item.markdown_path.stat()
    except OSError:
        return prior[1] is not None or prior[2] is not None
    return prior != (str(item.markdown_path), stat.st_size, stat.st_mtime_ns)


def upsert_nature_batch(
    connection: sqlite3.Connection, rows: Sequence[NatureResult]
) -> None:
    """原子写入一批 Nature DOI 结果。"""
    now = utc_now()
    connection.executemany(
        """
        INSERT INTO nature_papers(
            article_id, doi_raw, doi_normalized, extraction_source, markdown_path,
            year, journal_id, pair_key, status, error, file_size, file_mtime_ns,
            processed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(article_id) DO UPDATE SET
            doi_raw=excluded.doi_raw,
            doi_normalized=excluded.doi_normalized,
            extraction_source=excluded.extraction_source,
            markdown_path=excluded.markdown_path,
            year=excluded.year,
            journal_id=excluded.journal_id,
            pair_key=excluded.pair_key,
            status=excluded.status,
            error=excluded.error,
            file_size=excluded.file_size,
            file_mtime_ns=excluded.file_mtime_ns,
            processed_at=excluded.processed_at
        """,
        [
            (
                row.article_id,
                row.doi_raw,
                row.doi_normalized,
                row.extraction_source,
                row.markdown_path,
                row.year,
                row.journal_id,
                row.pair_key,
                row.status,
                row.error,
                row.file_size,
                row.file_mtime_ns,
                now,
            )
            for row in rows
        ],
    )
    connection.commit()


def ingest_nature(
    connection: sqlite3.Connection,
    inputs: Sequence[NatureInput],
    workers: int,
    batch_size: int,
    max_bytes: int,
    logger: logging.Logger,
) -> None:
    """并发抽取 Nature DOI，并在每批完成后保存断点。"""
    existing = existing_nature_signatures(connection)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        decisions = executor.map(
            nature_needs_processing,
            inputs,
            [existing] * len(inputs),
        )
        pending = [
            item for item, needed in zip(inputs, decisions, strict=True) if needed
        ]
    logger.info(
        "Nature DOI阶段：总计 %d，断点复用 %d，待处理 %d，并发数 %d",
        len(inputs),
        len(inputs) - len(pending),
        len(pending),
        workers,
    )
    completed = 0
    buffer: list[NatureResult] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: list[Future[NatureResult]] = [
            executor.submit(extract_nature_doi, item, max_bytes) for item in pending
        ]
        for future in as_completed(futures):
            result = future.result()
            buffer.append(result)
            completed += 1
            if len(buffer) >= batch_size:
                upsert_nature_batch(connection, buffer)
                buffer.clear()
                logger.info(
                    "Nature DOI进度：%d/%d（%.1f%%），本批已落库",
                    completed,
                    len(pending),
                    100.0 * completed / max(1, len(pending)),
                )
    if buffer:
        upsert_nature_batch(connection, buffer)
    retain_only_nature_inputs(connection, inputs)
    errors = connection.execute(
        "SELECT COUNT(*) FROM nature_papers WHERE status='read_error'"
    ).fetchone()[0]
    fallbacks = connection.execute(
        "SELECT COUNT(*) FROM nature_papers WHERE extraction_source='article_id_fallback'"
    ).fetchone()[0]
    logger.info(
        "Nature DOI阶段完成：读取错误 %d，article_id 回退 %d", errors, fallbacks
    )


def retain_only_nature_inputs(
    connection: sqlite3.Connection, inputs: Sequence[NatureInput]
) -> None:
    """成功读取 manifest 后移除不再属于当前清单的旧审计行。"""
    connection.execute(
        "CREATE TEMP TABLE IF NOT EXISTS current_nature_ids(id TEXT PRIMARY KEY)"
    )
    connection.execute("DELETE FROM current_nature_ids")
    connection.executemany(
        "INSERT INTO current_nature_ids(id) VALUES (?)",
        [(item.article_id,) for item in inputs],
    )
    connection.execute(
        "DELETE FROM nature_papers WHERE article_id NOT IN (SELECT id FROM current_nature_ids)"
    )
    connection.commit()


def parquet_columns(path: Path) -> set[str]:
    """返回 Parquet 字段集合。"""
    return set(pq.read_schema(path).names)


def load_doi_map(path: Path, logger: logging.Logger) -> dict[str, str | None]:
    """从现成 target works 表加载 OpenAlex ID→DOI 映射。"""
    columns = parquet_columns(path)
    id_column = "paper_id" if "paper_id" in columns else "id"
    if id_column not in columns or "doi" not in columns:
        raise ValueError(f"图 DOI 表缺少 ID 或 doi 字段：{path}")
    mapping: dict[str, str | None] = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=100_000, columns=[id_column, "doi"]):
        ids = batch.column(0).to_pylist()
        dois = batch.column(1).to_pylist()
        for paper_id, doi in zip(ids, dois, strict=True):
            normalized_id = normalize_openalex_id(paper_id)
            if normalized_id:
                mapping[normalized_id] = normalize_doi(doi)
    logger.info("已加载图 DOI 映射 %d 条：%s", len(mapping), path)
    return mapping


def graph_stage_state(
    connection: sqlite3.Connection, fingerprint: str
) -> tuple[str, int]:
    """读取图侧阶段状态；输入变化时安全地开启新一代写入。"""
    row = connection.execute(
        "SELECT fingerprint, status, checkpoint FROM pipeline_state WHERE stage='graph'"
    ).fetchone()
    if row is None or row["fingerprint"] != fingerprint:
        connection.execute(
            """
            INSERT INTO pipeline_state(stage, fingerprint, status, checkpoint, updated_at)
            VALUES ('graph', ?, 'running', 0, ?)
            ON CONFLICT(stage) DO UPDATE SET
                fingerprint=excluded.fingerprint,
                status='running', checkpoint=0, total=NULL,
                updated_at=excluded.updated_at
            """,
            (fingerprint, utc_now()),
        )
        connection.commit()
        return "running", 0
    return str(row["status"]), int(row["checkpoint"])


def ingest_graph(
    connection: sqlite3.Connection,
    graph_path: Path,
    doi_table: Path,
    fingerprint: str,
    batch_size: int,
    limit: int | None,
    logger: logging.Logger,
) -> None:
    """分批写入41万图节点，按批号实现真正的断点续跑。"""
    status, checkpoint = graph_stage_state(connection, fingerprint)
    expected = min(pq.ParquetFile(graph_path).metadata.num_rows, limit or 10**18)
    stored = connection.execute(
        "SELECT COUNT(*) FROM graph_papers WHERE input_fingerprint=?", (fingerprint,)
    ).fetchone()[0]
    if status == "complete" and stored == expected:
        logger.info("图 DOI阶段：输入未变化，直接复用已完成的 %d 条记录", stored)
        return
    doi_map = load_doi_map(doi_table, logger)
    columns = parquet_columns(graph_path)
    selected = ["paper_id"]
    selected.extend(
        name
        for name in ("publication_year", "domain12", "venue_family")
        if name in columns
    )
    parquet = pq.ParquetFile(graph_path)
    processed = 0
    for batch_index, batch in enumerate(
        parquet.iter_batches(batch_size=batch_size, columns=selected), start=1
    ):
        if batch_index <= checkpoint:
            processed += batch.num_rows
            continue
        rows = batch.to_pylist()
        if limit is not None:
            remaining = limit - processed
            rows = rows[: max(0, remaining)]
        if not rows:
            break
        upsert_graph_batch(connection, rows, doi_map, doi_table, fingerprint)
        processed += len(rows)
        connection.execute(
            """
            UPDATE pipeline_state SET checkpoint=?, total=?, updated_at=?
            WHERE stage='graph'
            """,
            (batch_index, expected, utc_now()),
        )
        connection.commit()
        logger.info(
            "图 DOI进度：%d/%d（%.1f%%），批次 %d 已落库",
            processed,
            expected,
            100.0 * processed / max(1, expected),
            batch_index,
        )
        if processed >= expected:
            break
    finish_graph_stage(connection, fingerprint, expected)
    missing = connection.execute(
        "SELECT COUNT(*) FROM graph_papers WHERE doi_normalized IS NULL"
    ).fetchone()[0]
    conflicts = connection.execute(
        "SELECT COUNT(*) FROM graph_papers WHERE status='doi_year_conflict'"
    ).fetchone()[0]
    logger.info(
        "图 DOI阶段完成：图节点 %d，缺少合法 DOI %d，DOI内嵌年份冲突 %d",
        expected,
        missing,
        conflicts,
    )


def upsert_graph_batch(
    connection: sqlite3.Connection,
    rows: Sequence[dict[str, Any]],
    doi_map: dict[str, str | None],
    doi_table: Path,
    fingerprint: str,
) -> None:
    """原子写入一批图节点 DOI。"""
    now = utc_now()
    values = []
    for row in rows:
        paper_id = normalize_openalex_id(row.get("paper_id"))
        doi = doi_map.get(paper_id)
        publication_year = row.get("publication_year")
        doi_year = infer_nature_doi_year(doi)
        year_conflict = (
            doi_year is not None
            and publication_year is not None
            and abs(doi_year - int(publication_year)) > 1
        )
        status = "doi_missing" if doi is None else "ok"
        if year_conflict:
            status = "doi_year_conflict"
        values.append(
            (
                paper_id,
                doi,
                doi,
                publication_year,
                row.get("domain12"),
                row.get("venue_family"),
                status,
                str(doi_table),
                fingerprint,
                now,
            )
        )
    connection.executemany(
        """
        INSERT INTO graph_papers(
            paper_id, doi_raw, doi_normalized, publication_year, domain12,
            venue_family, status, source_table, input_fingerprint, processed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            doi_raw=excluded.doi_raw,
            doi_normalized=excluded.doi_normalized,
            publication_year=excluded.publication_year,
            domain12=excluded.domain12,
            venue_family=excluded.venue_family,
            status=excluded.status,
            source_table=excluded.source_table,
            input_fingerprint=excluded.input_fingerprint,
            processed_at=excluded.processed_at
        """,
        values,
    )
    connection.commit()


def finish_graph_stage(
    connection: sqlite3.Connection, fingerprint: str, expected: int
) -> None:
    """清除旧输入代际并标记图侧阶段完成。"""
    actual = connection.execute(
        "SELECT COUNT(*) FROM graph_papers WHERE input_fingerprint=?", (fingerprint,)
    ).fetchone()[0]
    if actual != expected:
        raise RuntimeError(f"图侧落库行数不完整：期望 {expected}，实际 {actual}")
    connection.execute(
        "DELETE FROM graph_papers WHERE input_fingerprint<>?", (fingerprint,)
    )
    connection.execute(
        """
        UPDATE pipeline_state SET status='complete', total=?, updated_at=?
        WHERE stage='graph'
        """,
        (expected, utc_now()),
    )
    connection.commit()


def rebuild_matches(connection: sqlite3.Connection) -> None:
    """按归一化 DOI 精确重建匹配对、逐篇状态和分层统计。"""
    now = utc_now()
    connection.execute("DELETE FROM doi_match_pairs")
    connection.execute("DELETE FROM doi_match_conflicts")
    connection.execute("DELETE FROM nature_coverage")
    connection.execute("DELETE FROM coverage_breakdown")
    connection.execute(
        """
        INSERT INTO doi_match_pairs(article_id, paper_id, doi_normalized, matched_at)
        SELECT n.article_id, g.paper_id, n.doi_normalized, ?
        FROM nature_papers n
        JOIN graph_papers g ON g.doi_normalized=n.doi_normalized
        WHERE n.doi_normalized IS NOT NULL
          AND g.status='ok'
          AND (n.year IS NULL OR g.publication_year IS NULL
               OR ABS(n.year-g.publication_year)<=1)
        """,
        (now,),
    )
    connection.execute(
        """
        INSERT INTO doi_match_conflicts(
            article_id, paper_id, doi_normalized, nature_year, graph_year,
            reason, detected_at
        )
        SELECT n.article_id, g.paper_id, n.doi_normalized, n.year,
               g.publication_year, 'publication_year_conflict', ?
        FROM nature_papers n
        JOIN graph_papers g ON g.doi_normalized=n.doi_normalized
        WHERE n.doi_normalized IS NOT NULL
          AND (g.status='doi_year_conflict'
               OR (n.year IS NOT NULL AND g.publication_year IS NOT NULL
                   AND ABS(n.year-g.publication_year)>1))
        """,
        (now,),
    )
    connection.execute(
        """
        INSERT INTO nature_coverage(
            article_id, doi_normalized, coverage_status, match_count,
            matched_paper_ids, audited_at
        )
        SELECT n.article_id, n.doi_normalized,
               CASE
                 WHEN n.doi_normalized IS NULL THEN 'nature_doi_missing'
                 WHEN COUNT(m.paper_id)=0 THEN 'not_in_graph'
                 WHEN COUNT(m.paper_id)=1 THEN 'matched_unique'
                 ELSE 'matched_multiple_graph_nodes'
               END,
               COUNT(m.paper_id), GROUP_CONCAT(m.paper_id, '|'), ?
        FROM nature_papers n
        LEFT JOIN doi_match_pairs m ON m.article_id=n.article_id
        GROUP BY n.article_id, n.doi_normalized
        """,
        (now,),
    )
    connection.execute("""
        UPDATE nature_coverage
        SET coverage_status='doi_match_year_conflict'
        WHERE match_count=0
          AND EXISTS (
              SELECT 1 FROM doi_match_conflicts x
              WHERE x.article_id=nature_coverage.article_id
          )
        """)
    connection.execute(
        """
        INSERT INTO coverage_breakdown(
            year, journal_id, total, valid_doi, matched, not_in_graph,
            missing_doi, coverage_rate_valid_doi, audited_at
        )
        SELECT n.year, n.journal_id, COUNT(*),
               SUM(CASE WHEN c.coverage_status<>'nature_doi_missing' THEN 1 ELSE 0 END),
               SUM(CASE WHEN c.match_count>0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN c.coverage_status IN
                   ('not_in_graph', 'doi_match_year_conflict') THEN 1 ELSE 0 END),
               SUM(CASE WHEN c.coverage_status='nature_doi_missing' THEN 1 ELSE 0 END),
               CASE WHEN SUM(CASE WHEN c.coverage_status<>'nature_doi_missing' THEN 1 ELSE 0 END)=0
                    THEN NULL
                    ELSE 1.0 * SUM(CASE WHEN c.match_count>0 THEN 1 ELSE 0 END)
                         / SUM(CASE WHEN c.coverage_status<>'nature_doi_missing' THEN 1 ELSE 0 END)
               END, ?
        FROM nature_papers n JOIN nature_coverage c USING(article_id)
        GROUP BY n.year, n.journal_id
        """,
        (now,),
    )
    connection.commit()


def scalar(connection: sqlite3.Connection, query: str) -> int:
    """执行返回单个整数的查询。"""
    return int(connection.execute(query).fetchone()[0])


def year_range(
    connection: sqlite3.Connection, table: str, column: str
) -> list[int | None]:
    """返回数据集年份上下界。"""
    row = connection.execute(
        f"SELECT MIN({column}), MAX({column}) FROM {table}"
    ).fetchone()
    return [row[0], row[1]]


def build_summary(connection: sqlite3.Connection) -> dict[str, Any]:
    """生成可直接解读的覆盖率汇总。"""
    nature_total = scalar(connection, "SELECT COUNT(*) FROM nature_papers")
    valid = scalar(
        connection,
        "SELECT COUNT(*) FROM nature_papers WHERE doi_normalized IS NOT NULL",
    )
    matched = scalar(
        connection, "SELECT COUNT(*) FROM nature_coverage WHERE match_count>0"
    )
    graph_total = scalar(connection, "SELECT COUNT(*) FROM graph_papers")
    graph_with_doi = scalar(
        connection, "SELECT COUNT(*) FROM graph_papers WHERE doi_normalized IS NOT NULL"
    )
    graph_usable_doi = scalar(
        connection, "SELECT COUNT(*) FROM graph_papers WHERE status='ok'"
    )
    graph_year_conflicts = scalar(
        connection,
        "SELECT COUNT(*) FROM graph_papers WHERE status='doi_year_conflict'",
    )
    unique_graph_dois = scalar(
        connection,
        "SELECT COUNT(DISTINCT doi_normalized) FROM graph_papers WHERE doi_normalized IS NOT NULL",
    )
    statuses = {
        str(row[0]): int(row[1])
        for row in connection.execute(
            "SELECT coverage_status, COUNT(*) FROM nature_coverage GROUP BY coverage_status"
        )
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "completed_at": utc_now(),
        "graph": {
            "paper_count": graph_total,
            "publication_year_range": year_range(
                connection, "graph_papers", "publication_year"
            ),
            "with_syntactically_valid_doi": graph_with_doi,
            "usable_doi": graph_usable_doi,
            "missing_doi": graph_total - graph_with_doi,
            "doi_year_conflict": graph_year_conflicts,
            "unique_doi_count": unique_graph_dois,
            "duplicate_doi_node_count": graph_with_doi - unique_graph_dois,
        },
        "nature": {
            "paper_count": nature_total,
            "publication_year_range": year_range(connection, "nature_papers", "year"),
            "with_valid_doi": valid,
            "missing_doi": nature_total - valid,
            "matched_paper_count": matched,
            "coverage_rate_all": matched / nature_total if nature_total else None,
            "coverage_rate_valid_doi": matched / valid if valid else None,
            "coverage_status_counts": statuses,
            "year_conflict_candidate_count": scalar(
                connection,
                "SELECT COUNT(DISTINCT article_id) FROM doi_match_conflicts",
            ),
        },
    }


def export_query(connection: sqlite3.Connection, query: str, output: Path) -> None:
    """将数据库视图导出为便于人工查看的 UTF-8 CSV。"""
    cursor = connection.execute(query)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([column[0] for column in cursor.description])
        writer.writerows(cursor)


def export_artifacts(
    connection: sqlite3.Connection, output_dir: Path, summary: dict[str, Any]
) -> None:
    """在 SQLite 主结果之外输出 JSON 摘要和两张 CSV。"""
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    export_query(
        connection,
        """
        SELECT n.article_id, n.doi_normalized, n.year, n.journal_id,
               n.markdown_path, n.extraction_source, n.status AS doi_status,
               c.coverage_status, c.match_count, c.matched_paper_ids,
               (SELECT COUNT(*) FROM doi_match_conflicts x
                WHERE x.article_id=n.article_id) AS year_conflict_count
        FROM nature_papers n JOIN nature_coverage c USING(article_id)
        ORDER BY n.year, n.journal_id, n.article_id
        """,
        output_dir / "nature_doi_coverage.csv",
    )
    export_query(
        connection,
        "SELECT * FROM coverage_breakdown ORDER BY year, journal_id",
        output_dir / "coverage_by_year_journal.csv",
    )
    export_query(
        connection,
        """
        SELECT article_id, paper_id, doi_normalized, nature_year, graph_year,
               reason, detected_at
        FROM doi_match_conflicts
        ORDER BY nature_year, article_id, paper_id
        """,
        output_dir / "doi_year_conflicts.csv",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析命令行参数；默认值适配当前本地仓库和挂载目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-papers", type=Path, default=DEFAULT_GRAPH_PAPERS)
    parser.add_argument("--graph-doi-table", type=Path)
    parser.add_argument("--nature-manifest", type=Path, default=DEFAULT_NATURE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--database", type=Path)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Markdown 并发数；0=自动使用全部逻辑 CPU",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--markdown-max-bytes", type=int, default=262_144)
    parser.add_argument("--limit-nature", type=int, help="仅用于小规模验证")
    parser.add_argument("--limit-graph", type=int, help="仅用于小规模验证")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace, doi_table: Path) -> None:
    """在创建任务前完成路径和数值校验。"""
    for path in (args.graph_papers, doi_table, args.nature_manifest):
        if not Path(path).is_file():
            raise FileNotFoundError(f"输入文件不存在：{path}")
    if args.workers < 0:
        raise ValueError("--workers 不能为负数")
    if args.batch_size <= 0 or args.markdown_max_bytes <= 0:
        raise ValueError("--batch-size 和 --markdown-max-bytes 必须为正数")
    if args.limit_nature is not None and args.limit_nature <= 0:
        raise ValueError("--limit-nature 必须为正数")
    if args.limit_graph is not None and args.limit_graph <= 0:
        raise ValueError("--limit-graph 必须为正数")


def clear_derived_tables(connection: sqlite3.Connection) -> None:
    """显式强制重建时仅清空本脚本生成的派生表。"""
    connection.executescript("""
        DELETE FROM doi_match_pairs;
        DELETE FROM doi_match_conflicts;
        DELETE FROM nature_coverage;
        DELETE FROM coverage_breakdown;
        DELETE FROM graph_papers;
        DELETE FROM nature_papers;
        DELETE FROM pipeline_state;
        """)
    connection.commit()


def clear_match_tables(connection: sqlite3.Connection) -> None:
    """开始新一轮输入同步前清空可重建的匹配派生表。"""
    connection.executescript("""
        DELETE FROM doi_match_pairs;
        DELETE FROM doi_match_conflicts;
        DELETE FROM nature_coverage;
        DELETE FROM coverage_breakdown;
        """)
    connection.commit()


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    """执行完整审计并返回汇总。"""
    output_dir = Path(args.output_dir)
    database = (
        Path(args.database) if args.database else output_dir / "coverage_audit.sqlite3"
    )
    logger = setup_logging(output_dir / "audit.log", args.verbose)
    doi_table = discover_graph_doi_table(args.graph_doi_table)
    validate_args(args, doi_table)
    workers = args.workers or max(1, os.cpu_count() or 1)
    run_id = str(uuid.uuid4())
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {"resolved_graph_doi_table": str(doi_table), "resolved_workers": workers}
    )
    connection = connect_database(database)
    connection.execute(
        """
        UPDATE audit_runs
        SET finished_at=?, status='interrupted',
            error=COALESCE(error, '上一次进程未正常结束；本次运行已接管断点')
        WHERE status='running'
        """,
        (utc_now(),),
    )
    connection.execute(
        "INSERT INTO audit_runs(run_id, started_at, status, config_json) VALUES (?, ?, 'running', ?)",
        (run_id, utc_now(), json.dumps(config, ensure_ascii=False, sort_keys=True)),
    )
    connection.commit()
    try:
        if args.force_rebuild:
            logger.warning("收到 --force-rebuild：正在清空本审计库的旧派生结果")
            clear_derived_tables(connection)
        else:
            clear_match_tables(connection)
        logger.info(
            "开始 DOI 覆盖审计；本机逻辑 CPU=%s，并发=%d", os.cpu_count(), workers
        )
        logger.info("图节点表：%s", args.graph_papers)
        logger.info("图 DOI映射表：%s", doi_table)
        logger.info("Nature manifest：%s", args.nature_manifest)
        logger.info("SQLite结果库：%s", database)
        graph_fingerprint = path_fingerprint(args.graph_papers, doi_table)
        graph_fingerprint += f":batch={args.batch_size}"
        if args.limit_graph is not None:
            graph_fingerprint += f":limit={args.limit_graph}"
        ingest_graph(
            connection,
            args.graph_papers,
            doi_table,
            graph_fingerprint,
            args.batch_size,
            args.limit_graph,
            logger,
        )
        nature_inputs = read_nature_manifest(args.nature_manifest, args.limit_nature)
        logger.info("Nature manifest 已读取：%d 篇唯一论文", len(nature_inputs))
        ingest_nature(
            connection,
            nature_inputs,
            workers,
            args.batch_size,
            args.markdown_max_bytes,
            logger,
        )
        logger.info("正在按规范化 DOI 重建精确匹配关系和分层统计")
        rebuild_matches(connection)
        summary = build_summary(connection)
        export_artifacts(connection, output_dir, summary)
        connection.execute(
            """
            UPDATE audit_runs SET finished_at=?, status='complete', summary_json=?
            WHERE run_id=?
            """,
            (
                utc_now(),
                json.dumps(summary, ensure_ascii=False, sort_keys=True),
                run_id,
            ),
        )
        connection.commit()
        nature = summary["nature"]
        logger.info(
            "审计完成：Nature %d 篇，匹配 %d 篇；全量覆盖率 %.4f%%，有效 DOI 覆盖率 %.4f%%",
            nature["paper_count"],
            nature["matched_paper_count"],
            100.0 * (nature["coverage_rate_all"] or 0.0),
            100.0 * (nature["coverage_rate_valid_doi"] or 0.0),
        )
        logger.info("结果已落库：%s；摘要：%s", database, output_dir / "summary.json")
        return summary
    except (Exception, KeyboardInterrupt) as exc:
        connection.execute(
            "UPDATE audit_runs SET finished_at=?, status='failed', error=? WHERE run_id=?",
            (utc_now(), f"{type(exc).__name__}: {exc}", run_id),
        )
        connection.commit()
        logger.exception("审计失败；已保留断点，下次使用相同命令可继续")
        raise
    finally:
        connection.close()


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。"""
    args = parse_args(argv)
    summary = run_audit(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
