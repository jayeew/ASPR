from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import signal
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from figure_pipeline.fig3_revision.client import terminate_active

from .materials import prepare
from .models import Config
from .pipeline import run


async def execute(config: Config, args: argparse.Namespace) -> None:
    loop, task = asyncio.get_running_loop(), asyncio.current_task()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=config.cli_limit))
    def stop() -> None:
        terminate_active()
        if task:
            task.cancel()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop)
    await run(config, args.stage, args.paper_id, args.reference_view)


def main() -> None:
    parser = argparse.ArgumentParser(description='Revised Fig5 reliability data, no plotting')
    parser.add_argument('command', choices=['prepare', 'run', 'aggregate', 'snapshot', 'status'])
    parser.add_argument('--stage', choices=['reference', 'generate', 'evaluate', 'all'], default='all')
    parser.add_argument('--paper-id', action='append')
    parser.add_argument('--reference-view', action='append', help='Reference-only pilot views; fixed cohort unchanged')
    parser.add_argument('--cli-limit', type=int, default=32)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.reference_view and args.stage != 'reference':
        parser.error('--reference-view requires --stage reference')
    config = Config(cli_limit=args.cli_limit, **({'output': args.output.resolve()} if args.output else {}))
    if args.command == 'status':
        from .aggregate import status
        print(json.dumps(status(config), ensure_ascii=False, indent=2))
        return
    config.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'snapshot':
        from .aggregate import aggregate
        # Derived tables only; never mutate model inputs, checkpoints or task definitions.
        with (config.output / 'tables.lock').open('a') as table_lock:
            fcntl.flock(table_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            from .reference_reuse import sync_order_references
            sync_order_references(config)
            aggregate(config)
        return
    with (config.output / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'prepare':
            prepare(config)
        elif args.command == 'run':
            asyncio.run(execute(config, args))
        else:
            from .aggregate import aggregate
            aggregate(config)


if __name__ == '__main__':
    main()
