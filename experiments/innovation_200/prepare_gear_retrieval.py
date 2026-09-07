#!/usr/bin/env python3
"""Prepare every Claim's Luna-generated search frame without network retrieval."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.innovation_200.common import (
    configure_limits,
    experiment_config,
    generate_with_retries,
    read_jsonl,
    run_stage,
    setup_stage_logging,
    write_json,
)
from gear.artifacts import read_model, write_model
from gear.contracts import PaperIR
from gear.evidence_supervisor import EvidenceSupervisor
from gear.innovation.contracts import ClaimSet
from gear.innovation.usage import progress_logging
from gear.trace import EvidenceStore


def claim_tasks(study: Path) -> list[dict[str, str]]:
    tasks: list[dict[str, str]] = []
    for row in read_jsonl(study / "papers.jsonl"):
        paper_id = str(row["paper_id"])
        claims = read_model(
            study / "papers" / paper_id / "shared" / "claims.json",
            ClaimSet,
        )
        tasks.extend(
            {
                "task_id": claim.claim_id,
                "paper_id": paper_id,
                "claim_id": claim.claim_id,
            }
            for claim in claims.claims
        )
    return tasks


def prepare_one(
    task: dict[str, str],
    study: Path,
    overwrite: bool,
    logger: logging.Logger,
) -> dict[str, object]:
    paper_id = task["paper_id"]
    claim_id = task["claim_id"]
    short_id = claim_id.rsplit("::", 1)[-1]
    paper_root = study / "papers" / paper_id
    output_dir = paper_root / "gear_preparation" / short_id
    frame_path = output_dir / "search_frame.json"
    plan_path = output_dir / "retrieval_plan.json"
    with progress_logging(logger, f"[GEAR准备] paper={paper_id} claim={claim_id}"):
        if frame_path.is_file() and plan_path.is_file() and not overwrite:
            return {"skipped": True, "output": str(frame_path)}
        paper = read_model(paper_root / "shared" / "paper_ir.json", PaperIR)
        claims = read_model(paper_root / "shared" / "claims.json", ClaimSet)
        claim = next(item for item in claims.claims if item.claim_id == claim_id)
        supervisor = EvidenceSupervisor(
            experiment_config(),
            EvidenceStore(output_dir),
        )
        target_span, paper_claim = supervisor._adapters(claim, paper)
        logger.info(
            "[GEAR准备] paper=%s claim=%s [检索框架生成开始]",
            paper_id,
            claim_id,
        )
        planner = supervisor.prior_art.query_planner
        frame = generate_with_retries(
            lambda: planner.build_frame(paper_claim, target_span, paper),
            retries=2,
        )
        normal_queries = planner.plan(paper_claim, frame)
        contrastive_query = None
        contrastive_limitation = None
        try:
            contrastive_query = planner.contrastive(paper_claim, frame)
        except ValueError as exc:
            contrastive_limitation = str(exc)
        write_model(frame_path, frame)
        write_json(
            plan_path,
            {
                "paper_id": paper_id,
                "claim_id": claim_id,
                "target_span_id": target_span.span_id,
                "normal_queries": [
                    query.model_dump(mode="json") for query in normal_queries
                ],
                "contrastive_query": (
                    contrastive_query.model_dump(mode="json")
                    if contrastive_query is not None
                    else None
                ),
                "contrastive_limitation": contrastive_limitation,
            },
        )
        logger.info(
            "[GEAR准备] paper=%s claim=%s [检索框架生成完成] normal_queries=%d",
            paper_id,
            claim_id,
            len(normal_queries),
        )
        return {
            "skipped": False,
            "normal_queries": len(normal_queries),
            "output": str(frame_path),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--cli-limit", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_stage_logging(args.study, "prepare_gear_retrieval", args.verbose)
    tasks = claim_tasks(args.study)
    logger.info(
        "[配置] study=%s，claims=%d，workers=%d，cli_limit=%d，overwrite=%s",
        args.study,
        len(tasks),
        args.workers,
        args.cli_limit,
        args.overwrite,
    )
    configure_limits(args.cli_limit)
    run_stage(
        tasks,
        lambda task: prepare_one(task, args.study, args.overwrite, logger),
        workers=args.workers,
        status_path=args.study / "status" / "prepare_gear_retrieval.json",
        usage_dir=args.study / "status" / "usage" / "prepare_gear_retrieval",
        logger=logger,
    )


if __name__ == "__main__":
    main()
