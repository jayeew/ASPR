from __future__ import annotations

import argparse
import asyncio
import json
import signal
from collections import Counter

from figure_pipeline.fig3_revision.client import terminate_active
from figure_pipeline.fig3_revision.storage import read
from figure_pipeline.fig4_rerun.config import CONDITIONS

from .aggregate import aggregate
from .evaluation import evaluate
from .generation import generate_analyses, generate_reports
from .models import Config
from .reference import prepare_references, split_reference
from .readout import readout


async def execute(config: Config, stage: str, paper: list[str] | None) -> None:
    loop, task = asyncio.get_running_loop(), asyncio.current_task()

    def stop() -> None:
        terminate_active()
        if task is not None:
            task.cancel()

    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop)
    if stage == 'reference':
        await prepare_references(config, paper)
    elif stage == 'reference-split':
        if not paper or len(paper) != 1:
            raise ValueError('Explicit split recovery requires exactly one --paper-id')
        await split_reference(config, paper[0])
    elif stage == 'analysis':
        await generate_analyses(config)
    elif stage == 'reports':
        await generate_reports(config)
    elif stage.startswith('evaluate-'):
        await evaluate(config, stage.split('-')[1])


def main() -> None:
    parser = argparse.ArgumentParser(description='Focused original-five-paper ablation explanation study')
    parser.add_argument('stage', choices=['reference', 'reference-split', 'analysis', 'reports', 'evaluate-existing', 'evaluate-new', 'aggregate', 'status'])
    parser.add_argument('--paper-id', action='append')
    parser.add_argument('--condition', action='append', choices=CONDITIONS)
    parser.add_argument('--retry-failed', action='store_true', help='Explicit retry after inspecting saved failed responses')
    args = parser.parse_args()
    if args.stage == 'status':
        records = [read(p) for p in (Config().output / 'logs/calls').glob('*/record.json')]
        print(json.dumps({'calls': len(records), 'by_stage_state': dict(Counter(
            r['stage'] + ':' + r['state'] for r in records)), 'failures': [
                {k: r.get(k) for k in ('paper_id', 'stage', 'method', 'model', 'state', 'error')}
                for r in records if r['state'] not in ('running', 'completed')]}, ensure_ascii=False, indent=2))
        return
    config = Config(paper_ids=args.paper_id, conditions=args.condition, retry_failed=args.retry_failed)
    if args.stage == 'aggregate':
        aggregate(config)
        readout(config)
        return
    asyncio.run(execute(config, args.stage, args.paper_id))


if __name__ == '__main__':
    main()
