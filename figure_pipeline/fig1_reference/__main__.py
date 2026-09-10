"""Run with python -m figure_pipeline.fig1_reference."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import ROOT, prepare
from .export import assemble, handoff, validate
from .render import render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["prepare", "render", "assemble", "validate", "all"]
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/fig1_reference"
    )
    parser.add_argument(
        "--only", help="Panel key or component ID to export independently"
    )
    parser.add_argument(
        "--preview-only", action="store_true", help="Skip 6x and 10x final PNG exports"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.stage in ["prepare", "all"]:
        prepare(out)
    if args.stage in ["render", "all"]:
        render(out, args.only)
    if args.stage in ["assemble", "all"]:
        assemble(out, not args.preview_only, args.only)
    passed = True
    if args.stage in ["validate", "all"]:
        passed = validate(out, not args.preview_only)["all_passed"]
    if args.stage in ["assemble", "validate", "all"]:
        handoff(out)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
