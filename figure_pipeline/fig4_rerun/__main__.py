from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import signal
import shutil

from figure_pipeline.fig3_revision.client import terminate_active
from figure_pipeline.fig3_revision.storage import read, write

from .aggregate import aggregate, status
from .config import CONDITIONS, Config
from .engine import adopted_entries, ledger
from .materials import prepare
from .run import run


async def execute(config: Config, ids: list[str], conditions: list[str], ceiling: int) -> None:
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    def stop() -> None:
        terminate_active()
        task.cancel()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop)
    await run(config, ids, conditions, ceiling)


def main() -> None:
    parser = argparse.ArgumentParser(description='Fig4: one direct report and one fixed-core evaluation per condition')
    parser.add_argument('stage', choices=['prepare', 'run', 'aggregate', 'status', 'render'])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pilot', action='store_true')
    group.add_argument('--remaining', action='store_true')
    parser.add_argument('--paper-id', action='append')
    parser.add_argument('--condition', choices=CONDITIONS, action='append')
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--reevaluate-pilot', action='store_true',
                        help='Authorized second pilot evaluation: reuse reports, add at most 35 calls')
    args = parser.parse_args()
    config = Config(workers=min(max(args.workers, 1), 64), cli_limit=min(max(args.workers, 1), 64))
    config.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'status':
        print(json.dumps(status(config), ensure_ascii=False, indent=2))
        return
    if args.stage == 'render':
        from .render import render
        render(config)
        return
    with (config.output / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another rerun process holds the output lock')
        if args.stage == 'prepare':
            prepare(config)
        elif args.stage == 'aggregate':
            print(json.dumps(aggregate(config), ensure_ascii=False, indent=2))
        else:
            if args.reevaluate_pilot and not args.pilot:
                parser.error('--reevaluate-pilot requires --pilot')
            if not args.pilot and not args.remaining:
                parser.error('Choose --pilot or explicitly authorized --remaining; no automatic cohort expansion')
            pilot = read(config.output / 'pilot.json')['paper_ids']
            all_ids = [p['paper_id'] for p in read(config.output / 'papers.json')]
            ids = pilot if args.pilot else [p for p in all_ids if p not in pilot]
            if args.remaining:
                current = status(config)
                acceptance = config.output / 'pilot_acceptance.json'
                accepted = acceptance.exists() and read(acceptance)['evaluation_round'] == config.evaluation_round
                if current['evaluate']['pilot_complete'] != 35 and not accepted:
                    parser.error('Pilot is incomplete; remaining cohort cannot start')
                attempted = {(r['paper_id'], r['condition'], r['stage'])
                             for r in adopted_entries(ledger(config.output), config)}
                needed = sum(not (config.output / directory / c / f'{p}.json').exists()
                             and (p, c, stage) not in attempted
                             for p in ids for c in CONDITIONS
                             for stage, directory in [('generate', 'reports'), ('evaluate', 'evaluation')])
                if needed > current['calls_remaining']:
                    parser.error(f'Remaining cohort needs {needed} calls but only {current["calls_remaining"]} '
                                 'remain under 1400; discuss budget before starting')
            if args.paper_id:
                if set(args.paper_id) - set(ids):
                    parser.error('Requested paper is outside the selected cohort')
                ids = [p for p in ids if p in args.paper_id]
            revision = config.output / 'evaluation_revision.json'
            if args.reevaluate_pilot and not revision.exists():
                archive = config.output / 'previous_evaluation/v1'
                archive.mkdir(parents=True, exist_ok=True)
                for name in ['evaluation', 'figure', 'pilot_review.md', 'pilot_issues_for_review.md',
                             'unresolved_evaluations.json', 'condition_summary.csv', 'condition_summary_common.csv',
                             'paper_metrics.csv', 'full_lower_examples.json', 'summary.json', 'Fig4_pilot_delivery.zip']:
                    source = config.output / name
                    if source.exists():
                        if name == 'evaluation':
                            shutil.move(str(source), str(archive / name))
                        elif source.is_dir():
                            shutil.copytree(source, archive / name)
                        else:
                            shutil.copy2(source, archive / name)
                from .run import EVALUATOR
                write(revision, {'evaluation_round': config.evaluation_round, 'additional_calls_authorized': 35,
                                 'pilot_cumulative_ceiling': 105, 'reports_reused': 35, 'instruction': EVALUATOR,
                                 'total_call_limit': 1400, 'remaining_cohort_requires_budget_discussion': True})
            try:
                pilot_limit = 105 if revision.exists() else 70
                asyncio.run(execute(config, ids, args.condition or list(CONDITIONS), pilot_limit if args.pilot else 1330))
            finally:
                aggregate(config)


if __name__ == '__main__':
    main()
