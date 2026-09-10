#!/usr/bin/env python3
"""Generate supported whole-paper conditions; block confounded ablations."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.innovation_200.ablation_protocol import (
    SUPPORTED_REPORT_SYSTEMS,
    require_supported_generation,
)
from experiments.innovation_200.common import (
    configure_limits,
    read_jsonl,
    run_stage,
    setup_stage_logging,
    wait_for_inputs,
    write_json,
)
from experiments.innovation_200.contracts import SYSTEMS
from experiments.innovation_200.reporting import generate_report, report_markdown
from experiments.innovation_200.resource_guard import wait_for_memory


def generate(
    raw: dict,
    study: Path,
    overwrite: bool,
    wait_for_upstream: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, object]:
    paper_id = str(raw["paper_id"])
    systems = (str(raw["system"]),) if "system" in raw else SUPPORTED_REPORT_SYSTEMS
    for system in systems:
        require_supported_generation(system)
    generated = 0
    for system in systems:
        if system not in SYSTEMS:
            raise ValueError(f"Unknown system: {system}")
        target = study / "reports" / system / f"{paper_id}.json"
        if target.exists() and not overwrite:
            continue
        if wait_for_upstream:
            root = study / "papers" / paper_id
            paths = [root / "shared/paper_ir.json", root / "shared/claims.json"]
            producers = []
            if system not in ("direct_llm", "graph"):
                paths.append(root / "gear/analysis.json")
                producers.append(study / "status/run_gear.json")
            if system not in ("direct_llm", "gear"):
                paths.extend(
                    [root / "graph/analysis.json", root / "graph/joint/status.json"]
                )
                producers.append(study / "status/run_graph.json")
            wait_for_inputs(
                paths, paper_id=paper_id, producer_statuses=producers, logger=logger
            )
        wait_for_memory(logger)
        report = generate_report(paper_id, system, study / "papers" / paper_id)
        temporary = target.with_suffix(".json.tmp")
        write_json(temporary, report)
        temporary.replace(target)
        markdown = target.with_suffix(".md")
        markdown.write_text(report_markdown(report), encoding="utf-8")
        generated += 1
    return {"generated": generated, "total": len(systems)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cli-limit", type=int, default=16)
    parser.add_argument(
        "--systems", nargs="+", choices=SYSTEMS, default=list(SUPPORTED_REPORT_SYSTEMS)
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wait-for-inputs", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_stage_logging(args.study, "generate_reports", args.verbose)
    logger.info(
        "[配置] study=%s，workers=%d，cli_limit=%d，wait=%s，overwrite=%s，实验组=%d",
        args.study,
        args.workers,
        args.cli_limit,
        args.wait_for_inputs,
        args.overwrite,
        len(set(args.systems)),
    )
    configure_limits(args.cli_limit)
    run_stage(
        [
            dict(row, system=system, task_id=f"{row['paper_id']}__{system}")
            for row in read_jsonl(args.study / "papers.jsonl")
            for system in dict.fromkeys(args.systems)
        ],
        lambda row: generate(
            row, args.study, args.overwrite, args.wait_for_inputs, logger
        ),
        workers=args.workers,
        status_path=args.study / "status/generate_reports.json",
        usage_dir=args.study / "status/usage/generate_reports",
        logger=logger,
    )


if __name__ == "__main__":
    main()
