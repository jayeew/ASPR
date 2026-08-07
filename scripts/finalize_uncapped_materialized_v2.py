#!/usr/bin/env python3
"""Attach the uncapped v2 contract and audit materialized publication views."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.nature_multihorizon.expanded_dataset import (
    audit_materialized_expanded_dataset,
    write_json,
)

DATASET_VERSION = "nature-multihorizon-uncapped-v2"


def finalize(dataset_dir: Path, future_dir: Path) -> dict[str, object]:
    """Freeze the future contract into v6 and run the full materialized audit."""
    source_contract = json.loads(
        (future_dir / "expanded_dataset_contract.json").read_text(encoding="utf-8")
    )
    if source_contract.get("dataset_version") != DATASET_VERSION:
        raise ValueError("Future-layer contract is not uncapped v2")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_json(dataset_dir / "expanded_dataset_contract.json", source_contract)
    report = audit_materialized_expanded_dataset(
        dataset_dir,
        dataset_version=DATASET_VERSION,
        horizon_year_max={3: 2022, 5: 2020, 8: 2017},
    )
    if not report["overall_pass"]:
        raise RuntimeError(f"Uncapped v2 materialized audit failed: {report}")
    return report


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--future-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run finalization and print its quality report."""
    args = build_parser().parse_args(argv)
    report = finalize(args.dataset_dir, args.future_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
