#!/usr/bin/env python3
"""Run and explicitly repair the resumable GEAR branch from shared claims."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime, timezone
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
from experiments.innovation_200.recovery import (
    clean_limited,
    execution_summary,
    repair_coverage,
    retry_failed,
    study_gear_lock,
)
from experiments.innovation_200.resource_guard import wait_for_memory
from gear.contracts import PaperIR
from gear.innovation.contracts import AnalysisResult, ClaimSet
from gear.innovation.pipeline import run_branch
from gear.innovation.usage import progress_logging
from gear.local_ranking import LocalScientificRanker


def run_one(
    raw: dict,
    study: Path,
    graph_root: Path,
    embedding_model: Path,
    overwrite: bool,
    logger: logging.Logger,
    shared_local_ranker: LocalScientificRanker | None = None,
) -> dict[str, object]:
    item = paper_input(raw)
    with progress_logging(logger, f"[GEAR] paper={item.paper_id}"):
        root = study / "papers" / item.paper_id
        target = root / "gear/analysis.json"
        claims = ClaimSet.model_validate_json(
            (root / "shared/claims.json").read_text(encoding="utf-8")
        )
        if target.exists() and not overwrite:
            saved = AnalysisResult.model_validate_json(
                target.read_text(encoding="utf-8")
            )
            return {
                "skipped": True,
                "output": str(target),
                "assessments": len(saved.assessments),
                **execution_summary(root, saved, len(claims.claims)),
            }
        if overwrite and (root / "gear").exists():
            archive = (
                root
                / "gear_attempts"
                / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(root / "gear"), str(archive))
            logger.info("[GEAR重跑归档] %s", archive)
        wait_for_memory(logger)
        paper = PaperIR.model_validate_json(
            (root / "shared/paper_ir.json").read_text(encoding="utf-8")
        )
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
            shared_local_ranker=shared_local_ranker,
        )
        summary = execution_summary(root, result, len(claims.claims))
        logger.info(
            "[GEAR] paper=%s [论文完成] assessments=%d，execution_status=%s，coverage_status=%s，scientific_limited=%s，branch_status=%s",
            item.paper_id,
            len(result.assessments),
            summary["execution_status"],
            summary["coverage_status"],
            summary["scientific_limited"],
            result.status.value,
        )
        return {
            "skipped": False,
            "status": result.status.value,
            "assessments": len(result.assessments),
            **summary,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, default=ROOT / "data/claim_graph")
    parser.add_argument(
        "--embedding-model", type=Path, default=Path("/home/jayee/models/bge-m3")
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cli-limit", type=int, default=8)
    parser.add_argument(
        "--roster",
        type=Path,
        help="Explicit GEAR paper subset; defaults to study/gear_papers.jsonl when present",
    )
    parser.add_argument(
        "--paper-id", action="append", help="Run only this roster paper; repeatable"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--repair-coverage",
        action="store_true",
        help="Supplement missing contrastive searches using archived healthy evidence; retry technical failures",
    )
    parser.add_argument(
        "--clean-limited",
        action="store_true",
        help="Once: archive limited/invalid claims, preserve valid successful claims",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry technical failures only; preserve scientific limitations and healthy evidence cards",
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="With --clean-limited or --retry-failed: archive and exit without model calls",
    )
    parser.add_argument("--retry-limited", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _selected_rows(
    study: Path, selected: list[str] | None, roster: Path | None = None
) -> list[dict]:
    rows = read_jsonl(study / "papers.jsonl")
    pinned = roster or study / "gear_papers.jsonl"
    if roster is not None or pinned.exists():
        scoped_ids = [str(row["paper_id"]) for row in read_jsonl(pinned)]
        if not scoped_ids or len(scoped_ids) != len(set(scoped_ids)):
            raise ValueError("GEAR roster must contain nonempty, unique paper IDs")
        missing = set(scoped_ids) - {str(row["paper_id"]) for row in rows}
        if missing:
            raise ValueError(f"GEAR roster IDs absent from study: {sorted(missing)}")
        rows = [row for row in rows if str(row["paper_id"]) in set(scoped_ids)]
    if not selected:
        return rows
    missing = set(selected) - {str(row["paper_id"]) for row in rows}
    if missing:
        raise ValueError(f"Paper IDs absent from study roster: {sorted(missing)}")
    return [row for row in rows if str(row["paper_id"]) in set(selected)]


def _run(args: argparse.Namespace, rows: list[dict]) -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    stage = f"run_gear_selected_{stamp}" if args.paper_id else "run_gear"
    logger = setup_stage_logging(args.study, stage, args.verbose)
    logger.info(
        "[配置] study=%s，papers=%d，workers=%d，cli_limit=%d，overwrite=%s",
        args.study,
        len(rows),
        args.workers,
        args.cli_limit,
        args.overwrite,
    )
    if args.clean_limited or args.retry_failed or args.repair_coverage:
        recover = (
            repair_coverage
            if args.repair_coverage
            else (retry_failed if args.retry_failed else clean_limited)
        )
        manifest = recover(args.study, [str(row["paper_id"]) for row in rows])
        logger.info(
            "[清理完成] archive=%s，moved=%d",
            manifest["archive"],
            len(manifest["moved"]),
        )
        if args.cleanup_only:
            return 0
    configure_limits(args.cli_limit)
    config = experiment_config()
    logger.info(
        "[历史文献模式] OpenAlex PDF下载=%s；外部全文=%s；已有结果按断点复用",
        config.retrieval.openalex_pdf_enabled,
        getattr(config.retrieval, "external_fulltext_enabled", False),
    )
    shared_local_ranker = (
        LocalScientificRanker(
            config.retrieval.recall_model_path, config.retrieval.reranker_model_path
        )
        if config.retrieval.local_recall_enabled
        and config.retrieval.local_reranker_enabled
        else None
    )
    try:
        records = run_stage(
            rows,
            lambda row: run_one(
                row,
                args.study,
                args.graph_root,
                args.embedding_model,
                args.overwrite,
                logger,
                shared_local_ranker,
            ),
            workers=args.workers,
            status_path=args.study / "status" / f"{stage}.json",
            usage_dir=args.study / "status/usage" / stage,
            logger=logger,
        )
    finally:
        if shared_local_ranker is not None:
            shared_local_ranker.close()
    return int(
        any(
            row["status"] == "failed"
            or row.get("result", {}).get("execution_status") == "completed_with_errors"
            for row in records
        )
    )


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.retry_limited:
        parser.error(
            "--retry-limited is retired: use --clean-limited --cleanup-only once, then resume without cleanup flags"
        )
    if args.cleanup_only and not (
        args.clean_limited or args.retry_failed or args.repair_coverage
    ):
        parser.error("--cleanup-only requires a recovery flag")
    if (
        sum(
            (
                args.overwrite,
                args.clean_limited,
                args.retry_failed,
                args.repair_coverage,
            )
        )
        > 1
    ):
        parser.error("Choose only one recovery/overwrite flag")
    if args.overwrite and args.clean_limited:
        parser.error("--overwrite cannot be combined with --clean-limited")
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.study = args.study.resolve()
    try:
        rows = _selected_rows(args.study, args.paper_id, args.roster)
        with study_gear_lock(args.study):
            exit_code = _run(args, rows)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"GEAR: {exc}\n")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
