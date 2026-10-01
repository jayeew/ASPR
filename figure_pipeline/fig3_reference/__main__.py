"""Overwrite the reference figure using the prepared 100-paper study."""
from __future__ import annotations

import argparse

from .render import render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('render', 'assemble', 'all'), nargs='?', default='all')
    parser.add_argument('--panel', choices=tuple('abcdefgh'))
    args = parser.parse_args()
    render(args.panel)


if __name__ == '__main__':
    main()
