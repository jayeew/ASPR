from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import signal

from figure_pipeline.fig3_revision.client import terminate_active

from .aggregate import aggregate, status
from .config import Config
from .materials import prepare
from .run import run


async def execute(config: Config) -> None:
    loop, task = asyncio.get_running_loop(), asyncio.current_task()

    def stop() -> None:
        terminate_active()
        if task is not None:
            task.cancel()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop)
    await run(config)


def main() -> None:
    parser = argparse.ArgumentParser(description='Fig4 original five-paper capability pilot; 70 calls maximum')
    parser.add_argument('stage', choices=['prepare', 'run', 'aggregate', 'status'])
    args = parser.parse_args()
    config = Config()
    config.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'status':
        print(json.dumps(status(config), ensure_ascii=False, indent=2))
        return
    with (config.output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'prepare':
            prepare(config)
        elif args.stage == 'aggregate':
            print(json.dumps(aggregate(config), ensure_ascii=False, indent=2))
        else:
            try:
                asyncio.run(execute(config))
            finally:
                aggregate(config)


if __name__ == '__main__':
    main()
