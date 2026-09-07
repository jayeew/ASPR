#!/usr/bin/env python3
"""Lightweight orchestration; each stage remains independently runnable."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments/innovation_200"
sys.path.insert(0, str(ROOT))

from experiments.innovation_200.common import setup_stage_logging


def command(
    name: str,
    study: Path,
    workers: int,
    cli_limit: int,
    *,
    wait_for_inputs: bool = False,
    verbose: bool = False,
) -> list[str]:
    output = [
        sys.executable,
        str(SCRIPT / name),
        "--study",
        str(study),
        "--workers",
        str(workers),
        "--cli-limit",
        str(cli_limit),
    ]
    if wait_for_inputs:
        output.append("--wait-for-inputs")
    if verbose:
        output.append("--verbose")
    return output


def checked(args: list[str], logger: logging.Logger, name: str) -> None:
    started = time.monotonic()
    logger.info("[启动] %s：%s", name, " ".join(args))
    completed = subprocess.run(args, check=False)
    if completed.returncode:
        logger.error("[失败] %s：退出码=%d", name, completed.returncode)
        raise RuntimeError(
            f"Stage failed with exit code {completed.returncode}: {args[1]}"
        )
    logger.info("[完成] %s：耗时=%.1f秒", name, time.monotonic() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--cli-limit", type=int, choices=(16, 32, 48, 64), default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_stage_logging(args.study, "run_all", args.verbose)
    logger.info(
        "[总任务] study=%s，workers=%d，cli_limit=%d，skip_sampling=%s",
        args.study,
        args.workers,
        args.cli_limit,
        args.skip_sampling,
    )
    if not args.skip_sampling:
        sample_command = [
            sys.executable,
            str(SCRIPT / "sample_papers.py"),
            "--output",
            str(args.study),
            "--workers",
            str(args.workers),
            "--cli-limit",
            str(args.cli_limit),
        ]
        if args.verbose:
            sample_command.append("--verbose")
        checked(sample_command, logger, "sample_papers")
    logger.info("[并行阶段] 启动人工评审重建；同时执行共享claims抽取")
    reconstruct = subprocess.Popen(
        command(
            "reconstruct_reviews.py",
            args.study,
            args.workers,
            args.cli_limit,
            verbose=args.verbose,
        )
    )
    checked(
        command(
            "extract_claims.py",
            args.study,
            args.workers,
            args.cli_limit,
            verbose=args.verbose,
        ),
        logger,
        "extract_claims",
    )
    logger.info("[流式阶段] 启动GEAR、Graph、报告、人工测评和匿名比较")
    processes = [
        (
            subprocess.Popen(
                command(
                    "run_gear.py",
                    args.study,
                    2,
                    args.cli_limit,
                    verbose=args.verbose,
                )
            ),
            "GEAR",
        ),
        (
            subprocess.Popen(
                command(
                    "run_graph.py",
                    args.study,
                    args.workers,
                    args.cli_limit,
                    verbose=args.verbose,
                )
            ),
            "Graph",
        ),
        (
            subprocess.Popen(
                command(
                    "generate_reports.py",
                    args.study,
                    args.workers,
                    args.cli_limit,
                    wait_for_inputs=True,
                    verbose=args.verbose,
                )
            ),
            "report generation",
        ),
        (
            subprocess.Popen(
                command(
                    "evaluate_human.py",
                    args.study,
                    args.workers,
                    args.cli_limit,
                    wait_for_inputs=True,
                    verbose=args.verbose,
                )
            ),
            "human evaluation",
        ),
        (
            subprocess.Popen(
                command(
                    "compare_reports.py",
                    args.study,
                    args.workers,
                    args.cli_limit,
                    wait_for_inputs=True,
                    verbose=args.verbose,
                )
            ),
            "pairwise evaluation",
        ),
        (reconstruct, "human reconstruction"),
    ]
    logger.info(
        "[子进程] %s",
        "；".join(f"{name}=PID {process.pid}" for process, name in processes),
    )
    for process, name in processes:
        return_code = process.wait()
        if return_code != 0:
            logger.error("[失败] %s：退出码=%d", name, return_code)
            raise RuntimeError(f"{name} stage failed")
        logger.info("[完成] %s", name)
    checked(
        [
            sys.executable,
            str(SCRIPT / "summarize_results.py"),
            "--study",
            str(args.study),
            *(["--verbose"] if args.verbose else []),
        ],
        logger,
        "summarize_results",
    )
    logger.info("[总任务完成] 所有阶段结束：%s", args.study)


if __name__ == "__main__":
    main()
