#!/usr/bin/env python3
"""Unified CLI for GEAR core, reliability, and Graph-ablation evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(PROJECT_ROOT := Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.gear.evaluation.runner import STAGES, EvaluationRunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    EvaluationRunner(
        manifest_path=args.manifest,
        judge_config_path=args.judge_config,
        output_dir=args.output_dir,
        resume=args.resume,
        runtime_config_path=args.runtime_config,
        workers=args.workers,
    ).run(args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
