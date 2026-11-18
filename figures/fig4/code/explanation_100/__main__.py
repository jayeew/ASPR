from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter

from figure_pipeline.fig3_revision.storage import read

from .materials import prepare
from .models import CONDITIONS, Config
from .pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description='Fig4 uniform hundred-paper expansion with pilot review')
    parser.add_argument('command', choices=['prepare', 'run', 'aggregate', 'render', 'package', 'status'])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pilot', action='store_true')
    group.add_argument('--remaining', action='store_true')
    parser.add_argument('--paper-id', action='append')
    parser.add_argument('--condition', action='append', choices=CONDITIONS)
    parser.add_argument('--stage', choices=['all', 'reference', 'reports', 'evaluate'], default='all')
    args = parser.parse_args()
    config = Config()
    if args.command == 'prepare':
        prepare(config)
    elif args.command == 'run':
        if not args.pilot and not args.remaining:
            parser.error('Choose --pilot or --remaining explicitly')
        asyncio.run(run(config, args.pilot, args.paper_id, args.stage, args.condition))
    elif args.command == 'status':
        records = [read(p) for p in (config.output / 'logs/calls').glob('*/record.json')]
        statuses = [read(p) for p in (config.output / 'task_status').rglob('*.json')]
        print(json.dumps({'calls': len(records), 'call_states': dict(Counter(r['state'] for r in records)),
                          'tasks': dict(Counter(r['state'] for r in statuses)),
                          'unresolved': [r for r in statuses if r['state'] in ('unresolved', 'blocked')]}, ensure_ascii=False, indent=2))
    elif args.command == 'aggregate':
        from .aggregate import aggregate
        aggregate(config)
    elif args.command == 'render':
        from .render import render
        render(config)
    else:
        from .handoff import package
        print(package(config))


if __name__ == '__main__':
    main()
