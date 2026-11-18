"""CLI: python -m figure_pipeline.fig2_reference all [--only c]."""

from __future__ import annotations

import argparse
from pathlib import Path

from .contrasts import add_contrasts
from .data import DEFAULT_OUT, prepare
from .export import assemble, handoff, package, validate
from .render import render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["prepare", "render", "assemble", "validate", "all"]
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", help="Panel letter or component ID")
    parser.add_argument(
        "--no-high-res",
        action="store_true",
        help="Preview only; remove stale 6x/10x PNGs",
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.stage in ["prepare", "all"]:
        prepare(out)
        add_contrasts(out)
    if args.stage in ["render", "all"]:
        render(out, args.only)
    if args.stage in ["assemble", "all"]:
        assemble(out, not args.no_high_res, args.only)
    if args.stage in ["validate", "assemble", "all"]:
        validate(out)
        handoff(out)
        print(f"Delivery: {package(out)}", flush=True)


if __name__ == "__main__":
    main()
