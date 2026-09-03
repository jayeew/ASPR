#!/usr/bin/env python3
"""从 Phase 1 目标论文表精确移除指定 article ID。"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGETS = PROJECT_ROOT / "data" / "claim_graph" / "nature_targets.parquet"


def setup_logging() -> logging.Logger:
    """Create concise Chinese console logging for this one-time table maintenance step."""
    logger = logging.getLogger("claim_graph.remove_targets")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger


def remove_targets(targets_path: Path, article_ids: set[str]) -> int:
    """Atomically replace the target Parquet after removing only resolved article IDs."""
    if not targets_path.is_file():
        raise FileNotFoundError(f"目标论文表不存在：{targets_path}")
    if not article_ids:
        raise ValueError("至少提供一个 --article-id")
    table = pq.read_table(targets_path)
    rows = table.to_pylist()
    existing = {str(row["article_id"]) for row in rows}
    missing = article_ids - existing
    if missing:
        raise ValueError(f"目标论文表中不存在 article ID：{sorted(missing)}")
    kept = [row for row in rows if str(row["article_id"]) not in article_ids]
    temporary = targets_path.with_suffix(targets_path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(kept, schema=table.schema), temporary, compression="zstd")
    temporary.replace(targets_path)
    return len(kept)


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--article-id", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Remove requested records and report the final target count."""
    args = build_parser().parse_args(argv)
    logger = setup_logging()
    article_ids = {value.strip() for value in args.article_id if value.strip()}
    try:
        kept = remove_targets(args.targets, article_ids)
    except (OSError, ValueError, pa.ArrowException) as error:
        print(f"目标表移除失败：{error}", file=sys.stderr)
        return 1
    logger.info("已移除 %d 篇目标论文；当前目标数=%d", len(article_ids), kept)
    logger.info("目标论文表：%s", args.targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
