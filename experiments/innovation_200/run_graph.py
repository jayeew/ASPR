#!/usr/bin/env python3
"""Run single-claim and whole-paper joint Graph analysis."""

from __future__ import annotations

import argparse
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
    write_json,
)
from experiments.innovation_200.extract_claims import paper_input
from gear.innovation.graph_batch import (
    analyze_task,
    finalize_graph,
    model_tasks,
    prepare_graph,
)
from gear.innovation.locking import stage_lock
from gear.innovation.usage import progress_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, default=ROOT / "data/claim_graph")
    parser.add_argument(
        "--embedding-model", type=Path, default=ROOT / "data/models/Qwen3-Embedding-4B"
    )
    parser.add_argument(
        "--workers", type=int, default=32, help="Global language-model task concurrency"
    )
    parser.add_argument("--cli-limit", type=int, default=32)
    parser.add_argument("--stage", choices=("all", "prepare", "analyze"), default="all")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_stage_logging(args.study, "run_graph", args.verbose)
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
    if args.workers < 1 or args.embedding_batch_size < 1:
        parser.error("workers and embedding-batch-size must be positive")
    if args.overwrite and args.stage == "analyze":
        parser.error("--overwrite requires --stage prepare or all")
    rows = read_jsonl(args.study / "papers.jsonl")
    papers = [
        (paper_input(row), args.study / "papers" / str(row["paper_id"])) for row in rows
    ]
    config = experiment_config()
    with stage_lock(args.study / ".locks/run_graph_batch"):
        if args.stage in ("all", "prepare"):
            if args.overwrite:
                for _, root in papers:
                    if (root / "graph").exists():
                        archive = root / "graph_attempts"
                        from uuid import uuid4

                        archive.mkdir(parents=True, exist_ok=True)
                        (root / "graph").rename(archive / uuid4().hex)
            logger.info(
                "[Graph事实准备] papers=%d，embedding_batch_size=%d",
                len(papers),
                args.embedding_batch_size,
            )
            with progress_logging(logger, "[Graph准备]"):
                failures = prepare_graph(
                    papers,
                    config,
                    args.graph_root,
                    args.embedding_model,
                    args.embedding_batch_size,
                )
            write_json(
                args.study / "status/prepare_graph.json",
                {"status": "failed" if failures else "complete", "failures": failures},
            )
            if failures:
                raise RuntimeError(
                    "Graph preparation failed; inspect status/prepare_graph.json and rerun"
                )
        if args.stage == "prepare":
            return
        logger.info(
            "[Graph模型批处理] GPU准备已结束；单claim和整篇联合分析共享并发队列"
        )
        records = run_stage(
            model_tasks(papers),
            lambda task: analyze_task(task, config),
            workers=args.workers,
            status_path=args.study / "status/run_graph_models.json",
            usage_dir=args.study / "status/usage/run_graph",
            logger=logger,
        )
        final_records = run_stage(
            rows,
            lambda row: {
                "status": finalize_graph(
                    paper_input(row),
                    args.study / "papers" / str(row["paper_id"]),
                    config,
                ).status.value
            },
            workers=1,
            status_path=args.study / "status/run_graph.json",
            logger=logger,
        )
        if any(record["status"] == "failed" for record in records + final_records):
            raise RuntimeError(
                "Graph model tasks failed; rerun --stage analyze to resume"
            )


if __name__ == "__main__":
    main()
