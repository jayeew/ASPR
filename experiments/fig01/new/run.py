"""Run the canonical Fig.1 selected-case descriptive study."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.fig01.new.descriptive import run_descriptive_figure


DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def run_figure1(
    config_path: Path,
    stage: str = "all",
) -> Mapping[str, Any]:
    """Execute one resumable stage of the canonical Fig.1 pipeline.

    Args:
        config_path: Frozen selected-case configuration.
        stage: One of prepare, run, plot, audit, or all.

    Returns:
        Hash-addressed run manifest.
    """
    return run_descriptive_figure(config_path.resolve(), stage=stage)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        choices=["prepare", "run", "plot", "audit", "all"],
        default="all",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        os.execvpe(
            sys.executable,
            [
                sys.executable,
                "-m",
                "experiments.fig01.new.run",
                *sys.argv[1:],
            ],
            environment,
        )
    args = parse_args(argv)
    manifest = run_figure1(args.config, stage=args.stage)
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "status": manifest.get("status"),
                "artifact_id": manifest.get("artifact_id"),
            },
            indent=2,
        )
    )
    if args.stage in {"audit", "all"}:
        return 0 if manifest["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
