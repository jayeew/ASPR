"""Reproduce the source-backed Figure 6 without model calls."""
from __future__ import annotations

import argparse

from .data import prepare
from .export import export, package
from .render import render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['all', 'prepare', 'render', 'package'], default='all', nargs='?')
    stage = parser.parse_args().stage
    if stage in ('prepare', 'all'):
        prepare()
    if stage in ('render', 'all'):
        render()
        export()
    if stage in ('package', 'all'):
        package()


if __name__ == '__main__':
    main()
