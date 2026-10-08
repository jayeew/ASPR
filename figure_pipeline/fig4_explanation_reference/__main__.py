from __future__ import annotations

import argparse
from .data import prepare
from .render import render
from .export import package


def main() -> None:
    parser=argparse.ArgumentParser(description='Render accepted Fig4 data in reference style; no model calls')
    parser.add_argument('stage',nargs='?',default='all',choices=['all','prepare','render','package'])
    args=parser.parse_args()
    if args.stage in ['all','prepare']:prepare()
    if args.stage in ['all','render']:render()
    if args.stage in ['all','package']:package()


if __name__=='__main__':
    main()
