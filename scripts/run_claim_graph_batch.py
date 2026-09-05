#!/usr/bin/env python3
"""Sequentially evaluate papers with one shared Claim Graph runtime."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.artifacts import read_model
from gear.claim_attribution import (
    AbstractClaimExtractor,
    ClaimGraphRuntime,
    run_graph_branch_shared_runtime,
)
from gear.config import load_config
from gear.review_contracts import InnovationPaperInput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-contracts", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, default=Path("data/claim_graph"))
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=Path("data/models/Qwen3-Embedding-4B"),
    )
    parser.add_argument("--config", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    config = load_config(args.config)
    runtime = ClaimGraphRuntime(args.graph_root, args.embedding_model)
    extractor = AbstractClaimExtractor(config)
    total = len(args.input_contracts)
    logging.info("加载共享 Claim Graph Runtime；模型和 FAISS 在全部论文间复用")
    try:
        for index, path in enumerate(args.input_contracts, 1):
            item = read_model(path, InnovationPaperInput)
            output_dir = args.output_root / item.paper_id
            logging.info("[%d/%d] 开始处理 %s", index, total, item.paper_id)
            result = run_graph_branch_shared_runtime(
                item,
                output_dir,
                runtime=runtime,
                extractor=extractor,
            )
            logging.info(
                "[%d/%d] 完成 %s：状态=%s，Claim=%d，FactCard=%d",
                index,
                total,
                item.paper_id,
                result.status.value,
                len(result.claims),
                len(result.fact_cards),
            )
    finally:
        runtime.close()
    logging.info("全部完成；共享模型与图索引已释放")


if __name__ == "__main__":
    main()
