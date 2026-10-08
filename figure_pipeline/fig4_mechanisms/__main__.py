"""Explicit preparation/generation/evaluation/aggregation with ordinary resumable files."""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import subprocess
from pathlib import Path
from typing import Any

from figure_pipeline.fig3_revision.client import terminate_active
from figure_pipeline.fig3_revision.storage import Store, roster

from .aggregate import aggregate, status
from .config import CONDITIONS, Config
from .evaluation import Evaluation
from .execution import Engine
from .fusion import evaluate_fusion
from .generation import Generation
from .io import artifact, fusion_tracking_enabled, read, record, recover_calls, write
from .prepare import prepare


async def model_stage(config: Config, papers: list[str], conditions: list[str], stage: str) -> None:
    track_fusion = fusion_tracking_enabled(config.output)
    if stage == 'fusion' and not track_fusion:
        raise ValueError('Fusion tracking was stopped by the user; see analysis_scope.json')
    engine = Engine(config, config.workers)
    loop = asyncio.get_running_loop()
    def stop() -> None:
        engine.stop = True
        engine.ready.set()
        terminate_active()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop)
    errors = (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, subprocess.SubprocessError)
    async def one(paper: str) -> None:
        async def generate() -> None:
            generation = Generation(config, paper, engine)
            async def job(condition: str) -> None:
                try:
                    await generation.generate(condition)
                except errors as exc:
                    record(config.output, 'generate', paper, condition, 'failed', error=str(exc))
                    print(f'FAILED generate {paper} {condition}: {exc}', flush=True)
            await asyncio.gather(*(job(c) for c in conditions))
        async def basic() -> None:
            async def job(condition: str) -> None:
                try:
                    if not (config.output/'reports'/condition/'papers'/f'{paper}.json').exists():
                        raise ValueError('Condition report missing')
                    await Evaluation(config, paper, engine).evaluate_condition(condition)
                    record(config.output, 'evaluate', paper, condition, 'completed')
                except errors as exc:
                    record(config.output, 'evaluate', paper, condition, 'failed', error=str(exc))
                    print(f'FAILED evaluate {paper} {condition}: {exc}', flush=True)
            await asyncio.gather(*(job(c) for c in conditions))
        async def information() -> None:
            try:
                evaluation = Evaluation(config, paper, engine)
                clusters = await evaluation.cluster_reports()
                if clusters['missing_conditions']:
                    raise ValueError('Seven-condition basic evaluation not complete')
                await evaluation.cross_relations(clusters)
                record(config.output, 'information', paper, '', 'completed')
            except errors as exc:
                record(config.output, 'information', paper, '', 'failed', error=str(exc))
                print(f'FAILED information {paper}: {exc}', flush=True)
        async def fusion() -> None:
            try:
                evaluation = Evaluation(config, paper, engine)
                if not all(evaluation.store.path('support', paper, c).exists() for c in ('F', 'E', 'G')):
                    raise ValueError('F/E/G support evaluation not complete')
                await evaluate_fusion(evaluation)
                record(config.output, 'fusion_events', paper, 'F', 'completed')
            except errors as exc:
                record(config.output, 'fusion_events', paper, 'F', 'failed', error=str(exc))
                print(f'FAILED fusion_events {paper}: {exc}', flush=True)
        if stage in {'generate', 'complete'}:
            await generate()
        if stage in {'basic', 'evaluate', 'complete'}:
            await basic()
        lanes = []
        if stage in {'information', 'evaluate', 'complete'}:
            lanes.append(information())
        if track_fusion and stage in {'fusion', 'evaluate', 'complete'}:
            lanes.append(fusion())
        await asyncio.gather(*lanes)
    try:
        await asyncio.gather(*(one(p) for p in papers))
    finally:
        if engine.stop:
            terminate_active()
        await engine.close()
    if engine.stop:
        raise InterruptedError('Stopped model stage; completed artifacts retained for resume')


def missing_artifacts(config: Config, papers: list[str]) -> list[str]:
    missing = []
    store = Store(config)
    for paper in papers:
        paths = [config.output/'reports'/c/'papers'/f'{paper}.json' for c in CONDITIONS]
        paths += [store.path(s, paper, c) for s in ('extract', 'support', 'novelty') for c in CONDITIONS]
        stages = ['information_clusters', 'cross_relations']
        if fusion_tracking_enabled(config.output):
            stages += ['fusion_transitions', 'fusion_risks']
        paths += [artifact(config.output, s, paper) for s in stages]
        missing.extend(str(p) for p in paths if not p.exists())
    return missing


async def full_run(config: Config, papers: list[str], conditions: list[str], pilot_only: bool) -> None:
    pilot = [p for p in read(config.output/'pilot.json')['paper_ids'] if p in papers]
    remaining = [p for p in papers if p not in pilot]
    for group in [pilot] + ([] if pilot_only else [remaining]):
        if not group:
            continue
        if group == remaining and pilot:
            completion = read(config.output/'pilot5_completion.json')
            write(config.output/'pilot5_completion.json', {**completion, 'remaining_cohort_started': True})
        await model_stage(config, group, conditions, 'complete')
        aggregate(config)
        missing = missing_artifacts(config, group)
        if group == pilot:
            write(config.output/'pilot5_completion.json', {'paper_ids': group,
                'expected_reports': len(group)*len(CONDITIONS), 'complete': not missing,
                'missing_artifacts': missing, 'scientific_unresolved_states_retained': True,
                'remaining_cohort_started': False})
        record(config.output, 'cohort_completion', '', '', 'failed' if missing else 'completed',
               paper_ids=group, missing_artifacts=missing)
        if missing:
            raise RuntimeError(f'{len(missing)} scientific artifacts missing; retained all completed work. '
                               f'First: {missing[0]}. Remaining cohort not started.')

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'generate', 'evaluate', 'aggregate', 'status', 'all', 'basic', 'information', 'fusion'])
    parser.add_argument('--source', type=Path, default=Config.model_fields['source'].default)
    parser.add_argument('--output', type=Path, default=Config.model_fields['output'].default)
    parser.add_argument('--dataset', type=Path, default=Config.model_fields['dataset'].default)
    parser.add_argument('--paper-id', action='append')
    parser.add_argument('--condition', choices=CONDITIONS, action='append')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument('--http', action='store_true', help='Use HTTP transport with existing OpenAI sign-in')
    args = parser.parse_args()
    config = Config(source=args.source.resolve(), output=args.output.resolve(), dataset=args.dataset.resolve(),
                    workers=args.workers, cli_limit=args.workers, cli_initial=args.workers)
    if args.http:
        config = config.model_copy(update={'executable': str(Path(__file__).with_name('codex_http'))})
    if config.output == config.source:
        parser.error('Fig4 output must be separate from Fig3 source')
    if args.stage == 'fusion' and not fusion_tracking_enabled(config.output):
        parser.error('Fusion tracking was stopped by the user; see analysis_scope.json')
    papers = [r['paper_id'] for r in roster(config)]
    if args.paper_id:
        unknown = set(args.paper_id)-set(papers)
        if unknown:
            parser.error('Unknown paper IDs: '+', '.join(sorted(unknown)))
        papers = [p for p in papers if p in args.paper_id]
    conditions = args.condition or list(CONDITIONS)
    if args.stage == 'status':
        print(json.dumps(status(config), ensure_ascii=False, indent=2))
        return
    if args.stage in {'prepare', 'all'}:
        if args.stage == 'prepare' or not (config.output/'conditions.json').exists():
            prepare(config)
    if args.stage in {'generate', 'evaluate', 'all', 'basic', 'information', 'fusion'}:
        recovered = recover_calls(config.output)
        if recovered:
            print(f'Recovered {recovered} completed call checkpoints.', flush=True)
    if args.stage == 'aggregate':
        print(json.dumps(aggregate(config), ensure_ascii=False, indent=2))
    elif args.stage == 'all':
        asyncio.run(full_run(config, papers, conditions, args.pilot))
    elif args.stage in {'generate', 'evaluate', 'basic', 'information', 'fusion'}:
        if args.pilot:
            selected = set(read(config.output/'pilot.json')['paper_ids'])
            papers = [p for p in papers if p in selected]
        asyncio.run(model_stage(config, papers, conditions, args.stage))
        print(json.dumps(aggregate(config), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
