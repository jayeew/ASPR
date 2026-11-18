"""Independent country extension; never regenerate the accepted Fig7 clouds."""

from __future__ import annotations

import argparse
from pathlib import Path

from .country_data import DEFAULT_OUT, aggregate, audit, fetch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "fetch", "aggregate", "render"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "audit":
        audit(args.output_dir)
    elif args.stage == "fetch":
        fetch(args.output_dir, workers=args.workers)
    elif args.stage == "aggregate":
        aggregate(args.output_dir)
    else:
        from .country_render import render

        render(args.output_dir)


if __name__ == "__main__":
    main()
