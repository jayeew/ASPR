#!/usr/bin/env python3
"""Evaluate innovation Claims for one system or compare all agreed systems."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.config import load_config
from gear.review_contracts import DiscussionResolvedReference, ReviewerView
from gear.reviewers.agent import run_direct_baseline
from gear.review_verifier import (
    InnovationEvaluator,
    aggregate_summaries,
    evaluate_human_agreement,
    load_predictions,
)
from gear.artifacts import read_jsonl, read_model


SYSTEMS = (
    "direct_fulltext_llm",
    "graph_only",
    "gear_only",
    "passive_fusion",
    "active_graph_gear",
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    direct = commands.add_parser("direct-baseline")
    direct.add_argument("--run-dir", type=Path, required=True)
    direct.add_argument("--config", type=Path)
    single = commands.add_parser("single")
    single.add_argument("--system", choices=SYSTEMS, required=True)
    single.add_argument("--prediction", type=Path, required=True)
    single.add_argument("--reference", type=Path, required=True)
    _evaluation_arguments(single)
    compare = commands.add_parser("compare")
    compare.add_argument("--cases", type=Path, required=True)
    _evaluation_arguments(compare)
    human = commands.add_parser("human-agreement")
    human.add_argument("--reviewer-views", type=Path, required=True)
    _evaluation_arguments(human)
    return root


def _evaluation_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument(
        "--embedding-model",
        type=Path,
        default=Path("data/models/Qwen3-Embedding-4B"),
    )
    command.add_argument("--config", type=Path)


def evaluator(args: argparse.Namespace) -> InnovationEvaluator:
    return InnovationEvaluator(load_config(args.config), args.embedding_model)


def main() -> None:
    args = parser().parse_args()
    if args.command == "direct-baseline":
        print(run_direct_baseline(args.run_dir, load_config(args.config)))
        return
    active = evaluator(args)
    if args.command == "single":
        reference = read_model(args.reference, DiscussionResolvedReference)
        predictions = load_predictions(args.system, args.prediction)
        result = active.evaluate(
            args.system, predictions, reference, args.output_dir
        )
        print(result.model_dump_json(indent=2))
        return
    if args.command == "human-agreement":
        views = [
            ReviewerView.model_validate(row)
            for row in read_jsonl(args.reviewer_views)
        ]
        summaries = evaluate_human_agreement(active, views, args.output_dir)
        aggregate_summaries(summaries, args.output_dir / "human_agreement.json")
        print(args.output_dir / "human_agreement.json")
        return
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    summaries = []
    for case in cases:
        reference = read_model(Path(case["reference"]), DiscussionResolvedReference)
        for system in SYSTEMS:
            predictions = load_predictions(system, Path(case[system]))
            summaries.append(
                active.evaluate(
                    system,
                    predictions,
                    reference,
                    args.output_dir / str(case["paper_id"]),
                )
            )
    aggregate_summaries(summaries, args.output_dir / "system_comparison.json")
    print(args.output_dir / "system_comparison.json")


if __name__ == "__main__":
    main()
