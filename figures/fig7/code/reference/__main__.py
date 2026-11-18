"""Source-backed Fig7: prepare, predict, render, or all."""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'predict', 'render', 'all'])
    args = parser.parse_args()
    if args.stage in ('prepare', 'all'):
        from .data import prepare
        prepare()
    if args.stage in ('predict', 'all'):
        from .predict import generate
        generate()
    if args.stage in ('render', 'all'):
        from .unified import render
        render()


if __name__ == '__main__':
    main()
