#!/usr/bin/env python3
"""Run the GEAR evidence branch from the shared claims."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.innovation_200.common import (
    configure_limits,
    experiment_config,
    read_jsonl,
    run_stage,
    setup_stage_logging,
)
from experiments.innovation_200.extract_claims import paper_input
from gear.artifacts import read_model
from gear.contracts import PaperIR
from gear.innovation.contracts import ClaimSet
from gear.innovation.pipeline import run_branch
from gear.innovation.usage import progress_logging


def run_one(
    raw: dict,
    study: Path,
    graph_root: Path,
    embedding_model: Path,
    overwrite: bool,
    logger: logging.Logger,
) -> dict[str, object]:
    item = paper_input(raw)
    with progress_logging(logger, f"[GEAR] paper={item.paper_id}"):
        root = study / "papers" / item.paper_id
        target = root / "gear/analysis.json"
        if target.exists() and not overwrite:
            return {"skipped": True, "output": str(target)}
        if overwrite and (root / "gear").exists():
            shutil.rmtree(root / "gear")
        paper = read_model(root / "shared/paper_ir.json", PaperIR)
        claims = read_model(root / "shared/claims.json", ClaimSet)
        logger.info(
            "[GEAR] paper=%s [论文开始] claims=%d", item.paper_id, len(claims.claims)
        )
        result = run_branch(
            item,
            root,
            experiment_config(),
            paper,
            claims,
            "gear",
            graph_root,
            embedding_model,
            root / "gear_preparation",
        )
        logger.info(
            "[GEAR] paper=%s [论文完成] assessments=%d，status=%s",
            item.paper_id,
            len(result.assessments),
            result.status.value,
        )
        return {
            "skipped": False,
            "status": result.status.value,
            "assessments": len(result.assessments),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, default=ROOT / "data/claim_graph")
    parser.add_argument(
        "--embedding-model", type=Path, default=Path("/home/jayee/models/bge-m3")
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cli-limit", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_stage_logging(args.study, "run_gear", args.verbose)
    logger.info(
        "[配置] study=%s，workers=%d，cli_limit=%d，graph_root=%s，embedding=%s，overwrite=%s",
        args.study,
        args.workers,
        args.cli_limit,
        args.graph_root,
        args.embedding_model,
        args.overwrite,
    )
    configure_limits(args.cli_limit)
    logger.info(
        "[历史文献模式] PDF下载=%s（GEAR_HISTORICAL_PDF_ENABLED）；已有结果按断点复用",
        experiment_config().retrieval.openalex_pdf_enabled,
    )
    run_stage(
        read_jsonl(args.study / "papers.jsonl"),
        lambda row: run_one(
            row,
            args.study,
            args.graph_root,
            args.embedding_model,
            args.overwrite,
            logger,
        ),
        workers=args.workers,
        status_path=args.study / "status/run_gear.json",
        usage_dir=args.study / "status/usage/run_gear",
        logger=logger,
    )


if __name__ == "__main__":
    main()
