#!/usr/bin/env python3
"""构建 Claim Graph Phase 2 的两跳 OpenAlex Paper Citation Graph。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import multiprocessing
import os
import re
import shutil
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import unquote

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = Path("/mnt/d/FabCitationData/openalex-snapshot")
DEFAULT_ID_INDEX = Path("/home/jayee/workspace/FabCitation/openalex_snapshot_reference_check_results/analysis_state.db")
DEFAULT_TARGETS = PROJECT_ROOT / "data" / "claim_graph" / "nature_targets.parquet"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "claim_graph"
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)

NODE_SCHEMA = pa.schema([
    pa.field("work_id", pa.string(), False), pa.field("doi", pa.string()),
    pa.field("title", pa.string()), pa.field("publication_date", pa.string()),
    pa.field("publication_year", pa.int32()), pa.field("work_type", pa.string()),
    pa.field("source_id", pa.string()), pa.field("source_name", pa.string()),
    pa.field("primary_topic_id", pa.string()), pa.field("primary_topic_name", pa.string()),
    pa.field("field_id", pa.string()), pa.field("field_name", pa.string()),
    pa.field("referenced_works_count", pa.int32(), False),
    pa.field("is_nature_target", pa.bool_(), False), pa.field("nature_article_id", pa.string()),
    pa.field("hop_min", pa.int8(), False),
])
EDGE_SCHEMA = pa.schema([
    pa.field("citing_work_id", pa.string(), False), pa.field("cited_work_id", pa.string(), False),
    pa.field("citing_hop_min", pa.int8(), False), pa.field("cited_hop_min", pa.int8(), False),
])

_TARGET_DOIS: dict[str, str] = {}
_TARGET_IDS: set[str] = set()
_R1_IDS: set[str] = set()
_WORKER_PHASE = ""
_WORKER_CHUNK_DIR = Path()


def short_work_id(value: Any) -> str:
    text = str(value or "").strip().rstrip("/")
    suffix = text.rsplit("/", 1)[-1].upper()
    return suffix if suffix.startswith("W") and suffix[1:].isdigit() else ""


def normalize_doi(value: Any) -> str | None:
    text = unquote(str(value or "")).strip().lower()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    match = DOI_PATTERN.search(text)
    return match.group(0).rstrip(".,;:)]}>\\'") if match else None


def setup_logging(path: Path, verbose: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("claim_graph.phase2")
    logger.handlers.clear(); logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler, level in ((logging.StreamHandler(sys.stdout), logging.DEBUG if verbose else logging.INFO), (logging.FileHandler(path, encoding="utf-8", mode="a"), logging.DEBUG)):
        handler.setLevel(level); handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def init_worker(phase: str, chunk_dir: str) -> None:
    global _WORKER_PHASE, _WORKER_CHUNK_DIR
    _WORKER_PHASE, _WORKER_CHUNK_DIR = phase, Path(chunk_dir)


def project_node(record: dict[str, Any], hop: int, article_id: str | None) -> dict[str, Any]:
    source = ((record.get("primary_location") or {}).get("source") or {})
    topic = record.get("primary_topic") or {}
    field = topic.get("field") or {}
    return {
        "work_id": short_work_id(record.get("id")), "doi": normalize_doi(record.get("doi")),
        "title": str(record.get("title") or record.get("display_name") or "") or None,
        "publication_date": str(record.get("publication_date") or "") or None,
        "publication_year": record.get("publication_year"), "work_type": str(record.get("type") or "") or None,
        "source_id": short_work_id(source.get("id")) or None, "source_name": str(source.get("display_name") or "") or None,
        "primary_topic_id": short_work_id(topic.get("id")) or None, "primary_topic_name": str(topic.get("display_name") or "") or None,
        "field_id": short_work_id(field.get("id")) or None, "field_name": str(field.get("display_name") or "") or None,
        "referenced_works_count": len(record.get("referenced_works") or []),
        "is_nature_target": hop == 0, "nature_article_id": article_id, "hop_min": hop,
    }


def scan_shard(task: tuple[int, str]) -> dict[str, Any]:
    """Scan one shard and write its own JSONL chunk; globals are inherited by fork."""
    index, shard_text = task
    shard = Path(shard_text)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen = 0
    with gzip.open(shard, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            seen += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            work_id = short_work_id(record.get("id"))
            if not work_id:
                continue
            if _WORKER_PHASE == "pass1":
                article_id = _TARGET_DOIS.get(normalize_doi(record.get("doi")) or "")
                hop, matched = 0, article_id is not None
            elif _WORKER_PHASE == "pass2":
                article_id, hop, matched = None, 1, work_id in _R1_IDS
            else:
                article_id, hop, matched = None, 2, work_id in _TARGET_IDS
            if not matched:
                continue
            nodes.append(project_node(record, hop, article_id))
            if _WORKER_PHASE == "pass3":
                continue
            references = {short_work_id(value) for value in record.get("referenced_works") or []}
            references.discard("")
            for reference_id in references:
                if _WORKER_PHASE == "pass1":
                    cited_hop = 0 if reference_id in _TARGET_IDS else 1
                else:
                    cited_hop = 0 if reference_id in _TARGET_IDS else (1 if reference_id in _R1_IDS else 2)
                edges.append({"citing_work_id": work_id, "cited_work_id": reference_id, "citing_hop_min": hop, "cited_hop_min": cited_hop})
    chunk = _WORKER_CHUNK_DIR / f"{index:05d}.jsonl"
    with chunk.open("w", encoding="utf-8") as handle:
        for row in nodes:
            handle.write(json.dumps({"kind": "node", "row": row}, ensure_ascii=False) + "\n")
        for row in edges:
            handle.write(json.dumps({"kind": "edge", "row": row}, ensure_ascii=False) + "\n")
    return {"index": index, "shard": shard.name, "seen": seen, "nodes": len(nodes), "edges": len(edges)}


def connect_state(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE IF NOT EXISTS paper_graph_state (pass_id TEXT, shard_index INTEGER, status TEXT, records_seen INTEGER, matches_found INTEGER, finished_at TEXT, PRIMARY KEY(pass_id, shard_index))")
    return connection


def existing_ids(index_path: Path, ids: set[str], logger: logging.Logger) -> set[str]:
    """Use the supplied SQLite index only for membership filtering before scans 2/3."""
    if not index_path.is_file():
        raise FileNotFoundError(f"OpenAlex ID SQLite 不存在：{index_path}")
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    found: set[str] = set()
    ordered = sorted(ids)
    for start in range(0, len(ordered), 900):
        batch = ordered[start : start + 900]
        placeholders = ",".join("?" for _ in batch)
        found.update(row[0] for row in connection.execute(f"SELECT short_id FROM works_index WHERE short_id IN ({placeholders})", batch))
    connection.close()
    logger.info("SQLite membership：输入 %d，快照存在 %d，不存在 %d", len(ids), len(found), len(ids) - len(found))
    return found


def iter_chunk_rows(chunk_dir: Path, kind: str) -> Iterable[dict[str, Any]]:
    for path in sorted(chunk_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if payload["kind"] == kind:
                    yield payload["row"]


def scan_pass(pass_id: str, shards: list[Path], state: sqlite3.Connection, output_root: Path, workers: int, logger: logging.Logger) -> None:
    chunk_dir = output_root / "chunks" / "paper_graph" / pass_id
    chunk_dir.mkdir(parents=True, exist_ok=True)
    complete = {row[0] for row in state.execute("SELECT shard_index FROM paper_graph_state WHERE pass_id=? AND status='complete'", (pass_id,))}
    pending = [(idx, str(path)) for idx, path in enumerate(shards, 1) if idx not in complete]
    logger.info("[%s] 分片总数=%d，已完成=%d，待扫描=%d，并发=%d", pass_id, len(shards), len(complete), len(pending), workers)
    if not pending:
        return
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=init_worker, initargs=(pass_id, str(chunk_dir))) as pool:
        futures = {pool.submit(scan_shard, task): task for task in pending}
        done = len(complete); started = time.monotonic()
        for future in as_completed(futures):
            result = future.result(); done += 1
            state.execute("INSERT OR REPLACE INTO paper_graph_state VALUES (?, ?, 'complete', ?, ?, ?)", (pass_id, result["index"], result["seen"], result["nodes"], datetime.now(UTC).isoformat(timespec="seconds")))
            state.commit()
            logger.info("[%s][%d/%d] 完成 %s：读取=%d，命中=%d，边=%d，累计耗时=%.1fs", pass_id, done, len(shards), result["shard"], result["seen"], result["nodes"], result["edges"], time.monotonic() - started)


def write_parquet(rows: Iterable[dict[str, Any]], schema: pa.Schema, path: Path) -> int:
    writer = pq.ParquetWriter(path.with_suffix(path.suffix + ".tmp"), schema, compression="zstd")
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
    path.with_suffix(path.suffix + ".tmp").replace(path)
    return count


def metadata_completeness(row: dict[str, Any]) -> int:
    """Score only fields useful to distinguish duplicate OpenAlex Work records."""
    fields = (
        "title", "publication_date", "publication_year", "work_type", "source_id",
        "source_name", "primary_topic_id", "primary_topic_name", "field_id", "field_name",
    )
    return sum(row.get(field) not in (None, "") for field in fields)


def publication_date_key(row: dict[str, Any]) -> str:
    value = str(row.get("publication_date") or "")
    return value if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) else "0000-00-00"


def select_canonical_targets(output_root: Path, target_dois: set[str]) -> tuple[list[dict[str, Any]], int, int]:
    """Choose exactly one target Work per DOI after Pass 1 has collected all candidates."""
    candidates: dict[str, list[dict[str, Any]]] = {}
    for row in iter_chunk_rows(output_root / "chunks" / "paper_graph" / "pass1", "node"):
        doi = row.get("doi")
        if doi in target_dois:
            candidates.setdefault(doi, []).append(row)
    selected: list[dict[str, Any]] = []
    repeated_dois = 0
    discarded = 0
    for doi in sorted(candidates):
        rows = candidates[doi]
        if len(rows) > 1:
            repeated_dois += 1
            discarded += len(rows) - 1
        best_rank = max((publication_date_key(row), metadata_completeness(row), int(row.get("referenced_works_count") or 0)) for row in rows)
        tied = [row for row in rows if (publication_date_key(row), metadata_completeness(row), int(row.get("referenced_works_count") or 0)) == best_rank]
        selected.append(min(tied, key=lambda row: str(row["work_id"])))
    return selected, repeated_dois, discarded


def normalized_node_rows(output_root: Path, p_ids: set[str], r1_ids: set[str], r2_ids: set[str]) -> Iterable[dict[str, Any]]:
    """Emit only canonical P nodes and their two-hop neighborhood."""
    allowed = (("pass1", p_ids), ("pass2", r1_ids), ("pass3", r2_ids))
    for phase, permitted_ids in allowed:
        for row in iter_chunk_rows(output_root / "chunks" / "paper_graph" / phase, "node"):
            if row["work_id"] in permitted_ids:
                yield row


def normalized_edge_rows(output_root: Path, p_ids: set[str], r1_ids: set[str], r2_ids: set[str]) -> Iterable[dict[str, Any]]:
    """Normalize hop labels after Pass 1 knows all target Work IDs."""
    valid_ids = p_ids | r1_ids | r2_ids
    for phase in ("pass1", "pass2"):
        for row in iter_chunk_rows(output_root / "chunks" / "paper_graph" / phase, "edge"):
            citing_id, cited_id = row["citing_work_id"], row["cited_work_id"]
            if phase == "pass1" and citing_id not in p_ids:
                continue
            if phase == "pass2" and citing_id not in r1_ids:
                continue
            if citing_id not in valid_ids or cited_id not in valid_ids:
                continue
            yield {
                "citing_work_id": citing_id,
                "cited_work_id": cited_id,
                "citing_hop_min": 0 if citing_id in p_ids else 1,
                "cited_hop_min": 0 if cited_id in p_ids else (1 if cited_id in r1_ids else 2),
            }


def build_graph(args: argparse.Namespace) -> None:
    if not args.targets.is_file(): raise FileNotFoundError(f"目标表不存在：{args.targets}")
    work_root = args.snapshot / "data" / "works"
    shards = sorted(work_root.glob("updated_date=*/part_*.gz"))
    if not shards: raise FileNotFoundError(f"未找到 Works gzip 分片：{work_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_root / "logs" / "paper_graph.log", args.verbose)
    targets = pq.read_table(args.targets).to_pylist()
    doi_to_article = {str(row["doi"]): str(row["article_id"]) for row in targets}
    if len(doi_to_article) != len(targets): raise ValueError("nature_targets.parquet 存在重复 DOI，无法唯一匹配 OpenAlex Work")
    state_path = args.output_root / "build_state.sqlite"
    if state_path.exists() and not args.resume and not args.restart: raise FileExistsError("已存在 build_state.sqlite；继续请传 --resume，重建请传 --restart")
    if args.restart:
        shutil.rmtree(args.output_root / "chunks" / "paper_graph", ignore_errors=True)
        if state_path.exists():
            reset_state = connect_state(state_path)
            reset_state.execute("DELETE FROM paper_graph_state")
            reset_state.commit()
            reset_state.close()
        for path in (args.output_root / "canonical_target_works.parquet", args.output_root / "paper_nodes.parquet", args.output_root / "paper_edges.parquet", args.output_root / "paper_graph_stats.json"):
            path.unlink(missing_ok=True)
    state = connect_state(state_path)
    if args.workers < 0:
        raise ValueError("--workers 不能为负数")
    workers = args.workers or max(1, os.cpu_count() or 1)
    global _TARGET_DOIS, _TARGET_IDS, _R1_IDS
    _TARGET_DOIS = doi_to_article
    logger.info("[准备] 目标 DOI=%d，Works 分片=%d，workers=%d", len(targets), len(shards), workers)
    if args.pass_id in ("pass1", "all"):
        scan_pass("pass1", shards, state, args.output_root, workers, logger)
    canonical_targets, repeated_dois, discarded_targets = select_canonical_targets(args.output_root, set(doi_to_article))
    p_ids = {row["work_id"] for row in canonical_targets}
    if not p_ids: raise RuntimeError("Pass 1 未产生 Nature Work 节点，无法继续")
    write_parquet(iter(canonical_targets), NODE_SCHEMA, args.output_root / "canonical_target_works.parquet")
    logger.info("[去重] Pass 1 候选 Work=%d，唯一 DOI=%d，重复 DOI=%d，淘汰重复 Work=%d，保留规范 P=%d", len(canonical_targets) + discarded_targets, len(canonical_targets), repeated_dois, discarded_targets, len(p_ids))
    _TARGET_IDS = p_ids
    r1_raw = {row["cited_work_id"] for row in iter_chunk_rows(args.output_root / "chunks" / "paper_graph" / "pass1", "edge") if row["citing_work_id"] in p_ids and row["cited_work_id"] not in p_ids}
    _R1_IDS = existing_ids(args.id_index, r1_raw, logger)
    if args.pass_id in ("pass2", "all"):
        scan_pass("pass2", shards, state, args.output_root, workers, logger)
    r2_raw = {row["cited_work_id"] for row in iter_chunk_rows(args.output_root / "chunks" / "paper_graph" / "pass2", "edge") if row["citing_work_id"] in _R1_IDS and row["cited_work_id"] not in p_ids and row["cited_work_id"] not in _R1_IDS}
    r2_ids = existing_ids(args.id_index, r2_raw, logger) - p_ids - _R1_IDS
    _TARGET_IDS = r2_ids
    if args.pass_id in ("pass3", "all"):
        scan_pass("pass3", shards, state, args.output_root, workers, logger)
    logger.info("[合并] 开始生成 paper_nodes.parquet 与 paper_edges.parquet")
    node_count = write_parquet(normalized_node_rows(args.output_root, p_ids, _R1_IDS, r2_ids), NODE_SCHEMA, args.output_root / "paper_nodes.parquet")
    edge_count = write_parquet(normalized_edge_rows(args.output_root, p_ids, _R1_IDS, r2_ids), EDGE_SCHEMA, args.output_root / "paper_edges.parquet")
    stats = {"created_at": datetime.now(UTC).isoformat(timespec="seconds"), "target_dois": len(targets), "canonical_nature_works": len(p_ids), "unmatched_target_dois": len(targets) - len(canonical_targets), "duplicate_target_dois": repeated_dois, "discarded_duplicate_target_works": discarded_targets, "r1_existing": len(_R1_IDS), "r2_existing": len(r2_ids), "paper_nodes": node_count, "paper_edges": edge_count}
    (args.output_root / "paper_graph_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state.close(); logger.info("[完成] 节点=%d，边=%d；统计=%s", node_count, edge_count, args.output_root / "paper_graph_stats.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS); parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--id-index", type=Path, default=DEFAULT_ID_INDEX); parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=0, help="0=使用全部逻辑 CPU"); parser.add_argument("--pass", dest="pass_id", choices=("pass1", "pass2", "pass3", "all"), default="all")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--restart", action="store_true"); parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try: build_graph(args)
    except (OSError, RuntimeError, ValueError, sqlite3.Error, pa.ArrowException) as error:
        print(f"Phase 2 构建失败：{error}", file=sys.stderr); return 1
    return 0


if __name__ == "__main__": raise SystemExit(main())
