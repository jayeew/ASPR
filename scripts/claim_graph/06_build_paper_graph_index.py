#!/usr/bin/env python3
"""将静态 Paper Citation Graph 转为可局部更新、可快速查询的 SQLite 邻接索引。"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "claim_graph"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, default=DEFAULT_ROOT / "paper_nodes.parquet")
    parser.add_argument("--edges", type=Path, default=DEFAULT_ROOT / "paper_edges.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "paper_graph_index.sqlite")
    parser.add_argument("--batch-size", type=int, default=50_000, help="每次 SQLite 事务导入的行数")
    parser.add_argument("--resume", action="store_true", help="从已完成的 Parquet batch 继续")
    parser.add_argument("--restart", action="store_true", help="删除本脚本生成的 SQLite 后重建")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logging(path: Path, verbose: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("claim_graph.paper_index")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-524288")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS paper_nodes ("
        "work_id TEXT PRIMARY KEY, doi TEXT, title TEXT, publication_date TEXT, "
        "publication_year INTEGER, referenced_works_count INTEGER NOT NULL, "
        "is_nature_target INTEGER NOT NULL, nature_article_id TEXT UNIQUE, hop_min INTEGER NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS paper_edges ("
        "citing_work_id TEXT NOT NULL, cited_work_id TEXT NOT NULL, "
        "PRIMARY KEY(citing_work_id, cited_work_id)) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS import_state ("
        "source_name TEXT PRIMARY KEY, completed_batch INTEGER NOT NULL, "
        "completed_rows INTEGER NOT NULL, finished INTEGER NOT NULL DEFAULT 0)"
    )
    connection.commit()
    return connection


def state(connection: sqlite3.Connection, source: str) -> tuple[int, int, bool]:
    row = connection.execute(
        "SELECT completed_batch, completed_rows, finished FROM import_state WHERE source_name=?", (source,)
    ).fetchone()
    return (-1, 0, False) if row is None else (int(row[0]), int(row[1]), bool(row[2]))


def save_state(connection: sqlite3.Connection, source: str, batch: int, rows: int, finished: bool) -> None:
    connection.execute(
        "INSERT INTO import_state(source_name, completed_batch, completed_rows, finished) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(source_name) DO UPDATE SET completed_batch=excluded.completed_batch, "
        "completed_rows=excluded.completed_rows, finished=excluded.finished",
        (source, batch, rows, int(finished)),
    )


def import_nodes(connection: sqlite3.Connection, path: Path, batch_size: int, resume: bool, logger: logging.Logger) -> None:
    source = "paper_nodes"
    last_batch, completed_rows, finished = state(connection, source)
    if finished:
        logger.info("[节点] 已完成，跳过")
        return
    if not resume and (last_batch >= 0 or completed_rows > 0):
        raise FileExistsError("节点索引已有断点；请使用 --resume 或 --restart")
    parquet = pq.ParquetFile(path)
    started = time.monotonic()
    for batch_index, batch in enumerate(parquet.iter_batches(batch_size=batch_size)):
        if batch_index <= last_batch:
            continue
        rows = batch.to_pylist()
        values = [
            (
                str(row["work_id"]), row.get("doi"), row.get("title"), row.get("publication_date"),
                row.get("publication_year"), int(row.get("referenced_works_count") or 0),
                int(bool(row.get("is_nature_target"))), row.get("nature_article_id"), int(row.get("hop_min") or 0),
            )
            for row in rows
        ]
        with connection:
            connection.executemany(
                "INSERT INTO paper_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(work_id) DO UPDATE SET doi=excluded.doi, title=excluded.title, "
                "publication_date=excluded.publication_date, publication_year=excluded.publication_year, "
                "referenced_works_count=excluded.referenced_works_count, "
                "is_nature_target=excluded.is_nature_target, nature_article_id=excluded.nature_article_id, "
                "hop_min=excluded.hop_min",
                values,
            )
            completed_rows += len(values)
            save_state(connection, source, batch_index, completed_rows, False)
        logger.info("[节点] batch=%d，累计=%d，速度=%.0f 行/秒", batch_index + 1, completed_rows, completed_rows / max(time.monotonic() - started, 0.001))
    with connection:
        save_state(connection, source, last_batch if last_batch >= 0 else 0, completed_rows, True)


def import_edges(connection: sqlite3.Connection, path: Path, batch_size: int, resume: bool, logger: logging.Logger) -> None:
    source = "paper_edges"
    last_batch, completed_rows, finished = state(connection, source)
    if finished:
        logger.info("[边] 已完成，跳过")
        return
    if not resume and (last_batch >= 0 or completed_rows > 0):
        raise FileExistsError("边索引已有断点；请使用 --resume 或 --restart")
    parquet = pq.ParquetFile(path)
    started = time.monotonic()
    for batch_index, batch in enumerate(parquet.iter_batches(columns=["citing_work_id", "cited_work_id"], batch_size=batch_size)):
        if batch_index <= last_batch:
            continue
        values = [(str(row["citing_work_id"]), str(row["cited_work_id"])) for row in batch.to_pylist()]
        with connection:
            connection.executemany("INSERT OR IGNORE INTO paper_edges VALUES (?, ?)", values)
            completed_rows += len(values)
            save_state(connection, source, batch_index, completed_rows, False)
        logger.info("[边] batch=%d，累计=%d，速度=%.0f 行/秒", batch_index + 1, completed_rows, completed_rows / max(time.monotonic() - started, 0.001))
    with connection:
        save_state(connection, source, last_batch if last_batch >= 0 else 0, completed_rows, True)


def build_indexes(connection: sqlite3.Connection, logger: logging.Logger) -> None:
    logger.info("[索引] 建立反向引用和 Nature 映射索引")
    connection.execute("CREATE INDEX IF NOT EXISTS paper_edges_by_cited ON paper_edges(cited_work_id, citing_work_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS paper_nodes_by_article ON paper_nodes(nature_article_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS paper_nodes_by_target_date ON paper_nodes(is_nature_target, publication_date)")
    connection.commit()


def main() -> int:
    args = parse_args()
    if args.resume and args.restart:
        raise ValueError("--resume 与 --restart 不能同时使用")
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须为正数")
    for path in (args.nodes, args.edges):
        if not path.is_file():
            raise FileNotFoundError(f"输入不存在：{path}")
    output = args.output.resolve()
    logger = setup_logging(output.parent / "logs" / "phase6_paper_graph_index.log", args.verbose)
    if args.restart and output.exists():
        logger.info("[步骤 0/4] 删除旧 SQLite 索引：%s", output)
        output.unlink()
        for suffix in ("-wal", "-shm"):
            output.with_name(output.name + suffix).unlink(missing_ok=True)
    if output.exists() and not args.resume and not args.restart:
        raise FileExistsError("索引已存在；继续请传 --resume，重建请传 --restart")
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = connect(output)
    try:
        logger.info("[步骤 1/4] 导入 Paper 节点")
        import_nodes(connection, args.nodes, args.batch_size, args.resume, logger)
        logger.info("[步骤 2/4] 导入 Paper 引用边")
        import_edges(connection, args.edges, args.batch_size, args.resume, logger)
        logger.info("[步骤 3/4] 建立查询索引")
        build_indexes(connection, logger)
        node_count = connection.execute("SELECT COUNT(*) FROM paper_nodes").fetchone()[0]
        edge_count = connection.execute("SELECT COUNT(*) FROM paper_edges").fetchone()[0]
        target_count = connection.execute("SELECT COUNT(*) FROM paper_nodes WHERE is_nature_target=1").fetchone()[0]
        logger.info("[步骤 4/4] 完成：节点=%d，边=%d，Nature目标=%d，索引=%s", node_count, edge_count, target_count, output)
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, OSError, sqlite3.Error, ValueError) as error:
        print(f"Paper Graph 索引构建失败：{error}", file=sys.stderr)
        raise SystemExit(1) from error
