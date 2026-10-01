"""Explicit dependencies and one async scheduler for all papers and child calls."""
from __future__ import annotations

import asyncio
import shutil
import signal
import subprocess
import time
from collections import Counter
from typing import Any

from .client import terminate_active
from .config import COMPARATORS, METHODS, Config
from .data import prepare_paper
from .execution import Engine
from .native_graph import NativeGraph
from .progress import log
from .retrieval import retrieval_needs_retry
from .stages import METHOD_STAGES, STAGES, Stages, recheck_ids
from .storage import Store, read, roster, write

GROUPS = {
    'generate': ('prepare', 'claims', 'manuscript_notes', 'review_sections', 'checklist', 'tone',
                 'review_dynamics', 'direct_a', 'gear', 'graph', 'full', 'eacl', 'reviewgrounder', 'evidence_pool'),
    'evaluate': ('core', 'reference', 'extract', 'support', 'novelty', 'quality_checklist', 'quality',
                 'concerns', 'clusters', 'fusion', 'preference'),
    'recheck': ('recheck',),
}


def selected_ids(config: Config, cohort: str = 'all', paper_ids: list[str] | None = None) -> set[str]:
    ids = {row['paper_id'] for row in roster(config)}
    fixed = recheck_ids(config)
    if cohort == 'pilot':
        ids = set(fixed[:5])
    elif cohort == 'remaining':
        ids -= set(fixed[:5])
    elif cohort == 'recheck':
        ids = set(fixed)
    return ids & set(paper_ids) if paper_ids else ids


def tasks(config: Config, stage: str) -> list[tuple[str, str]]:
    ids = recheck_ids(config) if stage == 'recheck' else [r['paper_id'] for r in roster(config)]
    variants = list(METHODS) if stage in METHOD_STAGES else ['']
    if stage == 'preference':
        variants = [method+'_'+order for method in COMPARATORS for order in ('AB', 'BA')]
    return [(ident, variant) for ident in ids for variant in variants]


def report_stage(method: str) -> str:
    return 'full' if method == 'fusion' else method


def dependencies(stage: str, method: str) -> list[tuple[str, str]]:
    reports = [(report_stage(m), '') for m in ('gear', 'graph', 'eacl', 'reviewgrounder')]
    simple = {
        'prepare': [], 'claims': [('prepare', '')], 'manuscript_notes': [('prepare', '')],
        'review_sections': [('prepare', '')], 'checklist': [('review_sections', '')],
        'tone': [('review_sections', '')], 'review_dynamics': [('checklist', ''), ('review_sections', '')],
        'direct_a': [('prepare', '')], 'eacl': [('prepare', '')], 'reviewgrounder': [('prepare', '')],
        'gear': [('claims', '')], 'graph': [('claims', '')], 'full': [('gear', ''), ('graph', '')],
        'evidence_pool': [('manuscript_notes', ''), ('checklist', ''), *reports],
        'core': [('claims', ''), ('manuscript_notes', '')],
        'reference': [('core', ''), ('evidence_pool', '')],
        'quality_checklist': [('reference', ''), ('manuscript_notes', ''), ('evidence_pool', '')],
        'clusters': [(s, m) for m in ('gear', 'graph', 'fusion') for s in ('extract', 'support')],
        'fusion': [('clusters', '')]+[(s, m) for m in ('gear', 'graph', 'fusion') for s in ('extract', 'support')],
        'recheck': [('core', ''), ('reference', ''), ('checklist', ''), ('evidence_pool', '')]+
                   [(s, m) for m in METHODS for s in ('extract', 'support', 'novelty', 'concerns')],
    }
    if stage in simple:
        return simple[stage]
    if stage == 'preference':
        opponent = method.rsplit('_', 1)[0]
        return [('reference', ''), ('evidence_pool', ''), ('full', ''), (report_stage(opponent), '')]
    return {
        'extract': [('core', ''), (report_stage(method), '')],
        'support': [('extract', method), ('reference', ''), ('evidence_pool', '')],
        'novelty': [('extract', method), ('reference', '')],
        'quality': [('quality_checklist', ''), (report_stage(method), '')],
        'concerns': [('checklist', ''), ('reference', ''), ('evidence_pool', ''), (report_stage(method), '')],
    }[stage]


def fresh(config: Config) -> None:
    # Explicitly requested experiment output areas only. No archive, version or fingerprint.
    for name in ('inputs', 'reports', 'evidence', 'annotations', 'derived', 'figures', 'logs'):
        path = config.output/name
        if path.exists():
            shutil.rmtree(path)


def reset(config: Config, stage: str, paper: str, method: str) -> None:
    Store(config).path(stage, paper, method).unlink(missing_ok=True)
    for area in ('annotations/checkpoints', 'inputs/tasks'):
        path = config.output/area/stage/(method or 'papers')/paper
        if path.exists():
            shutil.rmtree(path)
    if stage in {'gear', 'eacl', 'reviewgrounder', 'evidence_pool'}:
        directory = config.output/'evidence/retrieval'/paper
        prefixes = {'gear': 'gear_', 'eacl': 'eacl', 'reviewgrounder': 'reviewgrounder', 'evidence_pool': 'history'}
        for path in directory.glob(prefixes[stage]+'*'):
            if path.is_dir():
                shutil.rmtree(path)
    if stage == 'graph':
        (config.output/'evidence/native_graph'/f'{paper}.json').unlink(missing_ok=True)


def reset_scientific_report(config: Config, stage: str, paper: str) -> None:
    """Replace affected judgments/writing while retaining original successful source work."""
    if stage not in {'gear', 'graph', 'full'}:
        raise ValueError('Scientific report repair only supports GEAR, Graph and Full')
    Store(config).path(stage, paper).unlink(missing_ok=True)
    checkpoint = config.output/'annotations/checkpoints'/stage/'papers'/paper
    task_dir = config.output/'inputs/tasks'/stage/'papers'/paper
    affected = []
    if stage == 'gear':
        for path in (config.output/'evidence/retrieval'/paper).glob('gear_*/result.json'):
            if retrieval_needs_retry(read(path)):
                affected.append(path.parent.name)
    for directory in (checkpoint, task_dir):
        if not directory.exists():
            continue
        for path in directory.glob('*.json'):
            name = path.stem
            remove = stage == 'full' or name.startswith(('writer', '贡献报告原文'))
            remove = remove or stage == 'gear' and ('_assessment' in name or name == 'GEAR贡献')
            remove = remove or any(name.startswith(prefix) and any(
                word in name for word in ('relations', 'antecedent', '历史关系', '先例独立')) for prefix in affected)
            if remove:
                path.unlink()


async def execute(config: Config, stages: tuple[str, ...], workers: int, ids: set[str],
                  overwrite: bool = False, method: str | None = None, transport: Any = None,
                  stage_factory: Any = Stages) -> dict[str, int]:
    engine, native, store = Engine(config, workers, transport), NativeGraph(config), Store(config)
    rows = {r['paper_id']: r for r in roster(config)}
    selected = [(s, p, m) for s in stages for p, m in tasks(config, s)
                if p in ids and (method is None or m == method or m.startswith(method+'_') or s == method)]
    if overwrite:
        for s, p, m in selected:
            reset(config, s, p, m)
    counts: Counter[str] = Counter()
    jobs: dict[tuple[str, str, str], asyncio.Task[str]] = {}
    interrupted = False
    loop = asyncio.get_running_loop()
    stopper: asyncio.Task[Any] | None = None

    async def stop_after_grace() -> None:
        await asyncio.sleep(30)
        terminate_active()
        for job in jobs.values():
            job.cancel()

    def stop() -> None:
        nonlocal interrupted, stopper
        if interrupted:
            terminate_active()
            for job in jobs.values():
                job.cancel()
        else:
            interrupted = True
            engine.stop = True
            log(config, '停止请求', '停止新增任务；最多等待30秒保存，再次Ctrl+C立即结束CLI。')
            stopper = asyncio.create_task(stop_after_grace())

    loop.add_signal_handler(signal.SIGINT, stop)

    async def one(stage: str, paper: str, variant: str) -> str:
        started = time.monotonic()
        state, message = 'failed', ''
        try:
            from .packing import answers, complete, enabled
            packed = stage == 'concerns' and enabled(config)
            if packed and complete(config, paper, variant):
                retained = answers(config, paper, variant)
                original = store.get('checklist', paper)['concerns']
                store.put(stage, paper, {'matches': [retained[c['concern_id']] for c in original]}, variant)
                state = 'reused'
            elif not packed and store.path(stage, paper, variant).exists():
                state = 'reused'
            else:
                unavailable = []
                for upstream, condition in dependencies(stage, variant):
                    parent = jobs.get((upstream, paper, condition))
                    ready = await parent in {'completed', 'reused'} if parent else store.path(upstream, paper, condition).exists()
                    if not ready:
                        unavailable.append(upstream+'/'+condition)
                if unavailable:
                    state, message = 'blocked', '缺少上游：'+', '.join(unavailable)
                elif engine.stop:
                    state = 'cancelled'
                else:
                    result = await engine.io(lambda: prepare_paper(config, rows[paper])) if stage == 'prepare' else \
                        await stage_factory(config, paper, engine, native).run(stage, variant)
                    store.put(stage, paper, result, variant)
                    state = 'completed'
        except asyncio.CancelledError:
            state, message = 'cancelled', '用户停止'
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, subprocess.SubprocessError) as exc:
            state, message = ('cancelled' if engine.stop else 'failed'), f'{type(exc).__name__}: {exc}'
        finally:
            counts[state] += 1
            write(config.output/'logs/progress'/stage/(variant or 'papers')/f'{paper}.json',
                  {'stage': stage, 'paper_id': paper, 'method': variant, 'state': state,
                   'message': message, 'seconds': time.monotonic()-started})
            log(config, '任务进度', f'[{sum(counts.values())}/{len(selected)}] 成功{counts["completed"]}，'
                f'复用{counts["reused"]}，失败{counts["failed"]}，阻塞{counts["blocked"]}，取消{counts["cancelled"]}；'
                f'{state} {message}', stage=stage, paper_id=paper, method=variant)
        return state

    try:
        for stage, paper, variant in selected:
            jobs[(stage, paper, variant)] = asyncio.create_task(one(stage, paper, variant))
        await asyncio.gather(*jobs.values())
    finally:
        if stopper:
            stopper.cancel()
        loop.remove_signal_handler(signal.SIGINT)
        await engine.io(lambda: native.close(), gpu=True) if not engine.stop else None
        await engine.close()
    if interrupted:
        raise KeyboardInterrupt
    return dict(counts)


def run_many(config: Config, stages: tuple[str, ...], workers: int, overwrite: bool = False,
             paper_ids: set[str] | None = None, method: str | None = None, **_: Any) -> dict[str, int]:
    return asyncio.run(execute(config, stages, workers, paper_ids if paper_ids is not None else selected_ids(config), overwrite, method))


def sequence(config: Config, group: str, workers: int, cohort: str = 'all', overwrite: bool = False,
             paper_ids: list[str] | None = None, method: str | None = None) -> dict[str, int]:
    return run_many(config, GROUPS[group], workers, overwrite, selected_ids(config, cohort, paper_ids), method)


def status(config: Config) -> dict[str, Any]:
    result = {}
    for stage in ('prepare', *STAGES):
        expected = tasks(config, stage)
        states: Counter[str] = Counter()
        for p, m in expected:
            path = config.output/'logs/progress'/stage/(m or 'papers')/f'{p}.json'
            if path.exists():
                states[read(path)['state']] += 1
        result[stage] = {'files_present': sum(Store(config).path(stage, p, m).exists() for p, m in expected),
                         'expected': len(expected), 'last_states': dict(states)}
    return result
