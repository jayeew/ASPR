from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser(description='Five-dimensional final reports using existing evidence only')
    parser.add_argument('command', choices=['prepare', 'pilot', 'run', 'aggregate', 'render', 'status', 'rerun-full'])
    parser.add_argument('--phase', choices=['prepare', 'pilot', 'run', 'check'], default='prepare')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--initial', type=int)
    parser.add_argument('--paper-id', action='append')
    parser.add_argument('--cohort', choices=['all', 'damage'], default='all')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='Render observed results with explicit missing denominators')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == 'rerun-full':
        if args.paper_id or args.cohort != 'all' or args.workers or args.initial or output != OUTPUT.resolve():
            parser.error('rerun-full fixes the existing output, 100 main papers, Full method and concurrency')
        from .full_rerun import run_full
        result = run_full(output, args.phase, args.timeout)
    elif args.command == 'prepare':
        from .materials import prepare_all
        result = prepare_all(output)
        result = {key: result[key] for key in ('main_papers', 'damage_papers', 'generation_tasks', 'reference_tasks')}
    elif args.command in {'pilot', 'run'}:
        from .runner import run
        pilot = args.command == 'pilot'
        result = run(output, 'pilot' if pilot else args.cohort,
                     args.workers or (8 if pilot else 64), args.initial or (8 if pilot else 16),
                     args.paper_id, args.timeout)
        print(json.dumps({key: value for key, value in result.items() if key != 'tasks'}, ensure_ascii=False))
        if result['failed_tasks']:
            raise SystemExit(1)
        return
    elif args.command == 'aggregate':
        from .aggregate import aggregate
        from .usage import summarise_usage
        result = aggregate(output)
        result['model_usage'] = summarise_usage(output)
    elif args.command == 'render':
        from .render import render
        result = render(output, allow_incomplete=args.allow_incomplete)
    else:
        from .io import read
        result = read(output / 'status/progress.json') if (output / 'status/progress.json').exists() else {'state': 'not_started'}
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
