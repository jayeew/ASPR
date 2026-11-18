"""Four explicit groups; fresh all-100 generation by default, explicit resume."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_CONFIG, load_config
from .progress import log


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Fig.3 fresh generation and bounded parallel evaluation')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    subs = parser.add_subparsers(dest='command', required=True)
    from .stages import STAGES
    for name in ('workflow', 'run', 'prepare', 'aggregate', 'render', 'assemble', 'status'):
        sub = subs.add_parser(name)
        sub.add_argument('--output-dir', type=Path)
        sub.add_argument('--paper-id', action='append')
        sub.add_argument('--cohort', choices=['all', 'pilot', 'remaining', 'recheck'], default='all')
        sub.add_argument('--workers', type=int)
        sub.add_argument('--resume', action='store_true')
        sub.add_argument('--overwrite', action='store_true')
        sub.add_argument('--reuse-and-pack', action='store_true')
        sub.add_argument('--estimate-calls', action='store_true')
        sub.add_argument('--method', '--variant', dest='method')
        if name == 'workflow':
            sub.add_argument('--group', choices=['all', 'generate', 'evaluate', 'recheck', 'export'], default='all')
        if name == 'run':
            sub.add_argument('--stage', choices=['prepare', *STAGES], required=True)
        if name == 'render':
            sub.add_argument('--panel', choices=list('abcdefgh'))
            sub.add_argument('--component')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.output_dir:
        config.output = args.output_dir.resolve()
    if args.reuse_and_pack or args.estimate_calls:
        if args.command != 'workflow' or args.group != 'evaluate' or args.overwrite:
            parser.error('--reuse-and-pack/--estimate-calls require workflow --group evaluate without --overwrite')
        from .packing import activate, estimate
        if args.estimate_calls:
            print(json.dumps(estimate(config), ensure_ascii=False, indent=2))
            return 0
        result = activate(config)
        log(config, '继承整理完成', json.dumps(result['counts'], ensure_ascii=False))
    from .runner import fresh, run_many, selected_ids, sequence, status
    ids = selected_ids(config, args.cohort, args.paper_id)
    workers = args.workers if args.workers is not None else config.workers
    if workers < 1:
        parser.error('--workers must be positive')
    if args.command == 'status':
        print(json.dumps(status(config), ensure_ascii=False, indent=2))
        return 0
    from .aggregate import aggregate
    from .render import assemble, render
    if args.command == 'workflow':
        if args.group in {'all', 'generate'} and not args.resume:
            fresh(config)
        groups = ('generate', 'evaluate', 'recheck', 'export') if args.group == 'all' else (args.group,)
        for group in groups:
            log(config, '脚本开始', f'任务组={group}，论文={len(ids)}，CLI最大并发={min(workers, config.cli_limit)}。')
            if group == 'export':
                aggregate(config, paper_ids=ids)
                render(config)
                assemble(config)
                continue
            result = sequence(config, group, workers, args.cohort, args.overwrite, args.paper_id, args.method)
            print(json.dumps(result, ensure_ascii=False))
            if any(result.get(key, 0) for key in ('failed', 'blocked', 'cancelled')):
                log(config, '脚本结束', '存在未完成任务，停止顺序入口；修复后使用--resume续跑。')
                return 1
    elif args.command in {'run', 'prepare'}:
        result = run_many(config, ('prepare' if args.command == 'prepare' else args.stage,), workers,
                          args.overwrite, ids, args.method)
        return int(any(result.get(key, 0) for key in ('failed', 'blocked', 'cancelled')))
    elif args.command == 'aggregate':
        aggregate(config, paper_ids=ids)
    elif args.command == 'render':
        render(config, args.panel, args.component)
    else:
        assemble(config)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('已停止。子任务结果保留；使用 --resume 继续本轮实验。', flush=True)
        raise SystemExit(130) from None
