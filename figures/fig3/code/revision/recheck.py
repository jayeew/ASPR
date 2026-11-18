"""Blind repeats with bounded batches and per-object recovery; no main-result edits."""
from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import Field, create_model

from . import models as m
from .client import BASE
from .config import METHODS
from .materials import encoded, fits
from .prompts import PROMPTS
from .storage import read, write

SPECS = {
    'support': ('units', 'unit_id', m.Support, m.SupportItem),
    'novelty': ('items', 'core_id', m.Novelty, m.NoveltyItem),
    'concerns': ('matches', 'concern_id', m.ConcernMatches, m.ConcernMatch),
}
COMPARE = {
    'support': ('support', 'scope_correct'),
    'novelty': ('difference_correct', 'scope_correct', 'effective_increment', 'material_overclaim'),
    'concerns': ('scope', 'reason_coverage', 'disagreement'),
}


def sample(stages: Any) -> list[dict[str, Any]]:
    """Keep the original seed, selection order and scientific sample unchanged."""
    rng = random.Random(stages.config.seed)
    concerns = sorted(stages.get('checklist')['concerns'], key=lambda c: c['concern_id'])
    rng.shuffle(concerns)
    work = []
    for method in METHODS:
        units = sorted([u for u in stages.get('extract', method)['units'] if u['needs_verification']],
                       key=lambda u: u['unit_id'])
        rng.shuffle(units)
        for kind, objects in [('support', units[:2]), ('novelty', stages.get('core')['items'][:2]),
                              ('concerns', concerns[:2])]:
            work.extend({'method': method, 'kind': kind, 'object': obj} for obj in objects)
    return work


def object_path(stages: Any, job: dict[str, Any]) -> Path:
    key = SPECS[job['kind']][1]
    return stages.config.output/'annotations/recheck_details'/stages.ident/job['method']/job['kind']/f'{job["object"][key]}.json'


def input_payload(stages: Any, job: dict[str, Any]) -> dict[str, Any]:
    method, kind, obj = job['method'], job['kind'], job['object']
    if kind == 'support':
        return stages.support_payload(method, [obj])
    if kind == 'novelty':
        return stages.novelty_payload(method, obj)
    from .packing import directory, enabled, payload
    card_path = directory(stages.config)/stages.ident/'neutral_cards.json'
    mode_path = stages.config.output/'annotations/concern_object_inputs'/method/stages.ident/f'{obj["concern_id"]}.json'
    if enabled(stages.config) and mode_path.exists() and read(mode_path)['material'] == 'neutral_card':
        cards = {c['concern_id']: c for c in read(card_path)['cards']}
        return payload(stages, method, [cards[obj['concern_id']]])
    return stages.concern_payload(method, [obj])


def merge_inputs(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Union object lists/evidence while keeping one complete common report."""
    result: dict[str, Any] = {}
    for part in parts:
        for key, value in part.items():
            if key == 'coverage' and isinstance(value, dict):
                value = [value]
            if key not in result:
                result[key] = value
            elif result[key] == value:
                continue
            elif isinstance(value, list) and isinstance(result[key], list):
                existing = {encoded(item) for item in result[key]}
                result[key] = result[key]+[item for item in value if encoded(item) not in existing]
            elif isinstance(value, dict) and isinstance(result[key], dict):
                result[key] = merge_inputs([result[key], value])
            else:
                raise ValueError(f'Cannot batch unequal shared field: {key}')
    return result


def fits_batch(stages: Any, kind: str, payload: dict[str, Any]) -> bool:
    from gear.codex_cli import _strict_response_schema
    schema = _strict_response_schema(SPECS[kind][2].model_json_schema())
    return fits(stages.config, payload, BASE+'\n'+PROMPTS[kind]+'\nINPUT:\n', schema)


def plan(stages: Any, work: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for job in work:
        grouped[(job['method'], job['kind'])].append(job)
    result = []
    for (_, kind), jobs in grouped.items():
        pending: list[Any] = []
        for job in jobs:
            candidate = pending+[job]
            material = merge_inputs([input_payload(stages, row) for row in candidate])
            if pending and (len(candidate) > 4 or not fits_batch(stages, kind, material)):
                result.append({'jobs': pending})
                pending = [job]
            else:
                pending = candidate
        if pending:
            result.append({'jobs': pending})
    return result


def unique_rows(rows: list[dict[str, Any]], key: str, expected: set[str]) -> dict[str, Any]:
    found, conflicts = {}, set()
    for row in rows:
        ident = row[key]
        if ident not in expected:
            continue
        if ident in found and found[ident] != row:
            conflicts.add(ident)
        found[ident] = row
    return {ident: row for ident, row in found.items() if ident not in conflicts}


def missing_schema(kind: str, ident: str) -> Any:
    field, key, schema, item = SPECS[kind]
    exact = create_model('RequestedRecheckObject', __base__=item,
                         **{key: (str, Field(json_schema_extra={'enum': [ident]}))})
    return create_model('MissingRecheckObject', __base__=schema,
                        **{field: (list[exact], Field(min_length=1, max_length=1))})


async def finish(stages: Any, job: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    kind, method, obj = job['kind'], job['method'], job['object']
    field, key, _, _ = SPECS[kind]
    ident = obj[key]
    if kind == 'novelty':
        from .scientific_tasks import effective_increment
        reference = next(r for r in stages.get('reference')['items'] if r['core_id'] == ident)
        prediction = next(r for r in stages.get('extract', method)['predictions'] if r['core_id'] == ident)
        second['effective_increment'] = effective_increment(reference, prediction, second)
    first = next((r for r in stages.get(kind, method)[field] if r[key] == ident), None)
    different = None if first is None else any(first[k] != second[k] for k in COMPARE[kind])
    explanation = None
    if different:
        # Reuse the actual task materials, including report/reference, not a new generic search.
        explanation = await stages.ask('recheck', {'object': obj, 'first': first, 'second': second,
            'original_task_materials': input_payload(stages, job)}, m.Analysis,
            f'explain_{method}_{kind}_{ident}', PROMPTS['adjudicate'], synthesis=True)
    result = {'task': kind, 'method': method, 'object_id': ident, 'first': first, 'second': second,
              'different': different, 'explanation': explanation,
              'comparison_status': 'first_missing' if first is None else 'compared'}
    write(object_path(stages, job), result)
    return result


async def repeat_batch(stages: Any, batch: dict[str, Any], index: int) -> list[Any]:
    jobs = [j for j in batch['jobs'] if not object_path(stages, j).exists()]
    if not jobs:
        return [read(object_path(stages, j)) for j in batch['jobs']]
    kind, method = jobs[0]['kind'], jobs[0]['method']
    field, key, schema, _ = SPECS[kind]
    # A fixed batch checkpoint is always interpreted against the original frozen batch.
    payload = lambda: merge_inputs([input_payload(stages, j) for j in batch['jobs']])
    answer = await stages.ask('recheck', payload, schema, f'repeat_batch_{index:05d}', PROMPTS[kind])
    accepted = unique_rows(answer[field], key, {j['object'][key] for j in batch['jobs']})

    async def one(job: dict[str, Any], _: int) -> Any:
        ident = job['object'][key]
        row = accepted.get(ident)
        if row is None:
            answer = await stages.ask('recheck', lambda: input_payload(stages, job), missing_schema(kind, ident),
                                      f'missing_{method}_{kind}_{ident}', PROMPTS[kind])
            row = unique_rows(answer[field], key, {ident}).get(ident)
            if row is None:
                raise ValueError(f'复核缺少对象: {method}/{kind}/{ident}')
        return await finish(stages, job, row)
    # Independent explanations/missing objects share the global CLI scheduler.
    values = await asyncio.gather(*(one(job, i) for i, job in enumerate(jobs)), return_exceptions=True)
    errors = [value for value in values if isinstance(value, Exception)]
    if errors:
        raise RuntimeError(f'复核批次存在{len(errors)}个未完成对象: {errors[0]}')
    return [read(object_path(stages, j)) for j in batch['jobs']]


def control_candidates(stages: Any) -> dict[str, Any]:
    """Bounded, method-balanced real quotations; no first-round evaluator labels."""
    queues = []
    for method in METHODS:
        supported = {u['unit_id']: u for u in stages.get('support', method)['units']
                     if u['support'] == 'supported' and u['scope_correct'] is True and not u['errors']}
        rows = sorted(stages.get('extract', method)['units'],
                      key=lambda u: (u['kind'] not in {'scope', 'increment', 'historical_comparison'}, u['unit_id']))
        queues.append([{'unit_key': method+'/'+u['unit_id'], 'quote': u['quote'], 'kind': u['kind'],
                        'original_locator': u['original_locator'],
                        'source_ids': list(dict.fromkeys(e['source_id'] for e in supported[u['unit_id']]['evidence']))}
                       for u in rows if u['unit_id'] in supported])
    selected, seen = [], set()
    for i in range(max((len(q) for q in queues), default=0)):
        for queue in queues:
            if i >= len(queue):
                continue
            row = queue[i]
            identity = encoded([row['quote'], row['source_ids']])
            if identity in seen or len(encoded({'units': selected+[row]})) > 7000:
                continue
            selected.append(row); seen.add(identity)
            if len(selected) == 12:
                return {'units': selected}
    return {'units': selected}


async def controls(stages: Any) -> list[Any]:
    path = stages.config.output/'inputs/tasks/recheck/papers'/stages.ident/'control_candidates.json'
    if path.exists():
        material = read(path)
    else:
        material = control_candidates(stages)
        write(path, material)
    constructed = await stages.ask('recheck', material, m.Controls, 'construct_controls', PROMPTS['controls']+
        ' Use the supplied small candidate set only. source_ids are report-level citations, not proof of quotation support. '
        'Omit a category when no suitable candidate exists; omission says nothing about other report units.')
    selected = {}
    for pair in constructed['pairs']:
        selected.setdefault(pair['kind'], pair)
    work = [{'kind': kind, 'side': side, 'statement': pair[side], 'source_ids': pair[side+'_source_ids']}
            for kind, pair in selected.items() for side in ('original', 'altered')]

    async def judge(job: Any, index: int) -> Any:
        material = stages.material({'statement': job['statement'], 'source_ids': job['source_ids']})
        if job['kind'] == 'source_mismatch':
            material['evidence_blocks'] = [b for b in material['evidence_blocks'] if any(
                p['source_id'] in job['source_ids'] for p in b['provenance'])]
        return await stages.ask('recheck', {'statement': job['statement'],
            'cited_source_ids': job['source_ids'], **material},
            m.ControlVerdict, f'control_{index:05d}', PROMPTS['recheck']+
            ' For supplied citations judge whether THESE sources support the statement; another uncited source '
            'does not repair a citation mismatch.')
    verdicts = await stages.map('控制评价', work, judge)
    for job, verdict in zip(work, verdicts):
        selected[job['kind']].setdefault('judgments', {})[job['side']] = verdict
    return list(selected.values())


async def run(stages: Any) -> dict[str, Any]:
    path = stages.config.output/'inputs/tasks/recheck/papers'/stages.ident/'repeat_batches.json'
    if path.exists():
        batches = read(path)['tasks']
    else:
        work = sample(stages)
        # Preserve any already completed object results; never rerun them.
        batches = plan(stages, work)
        write(path, {'tasks': batches})
    outcomes = await asyncio.gather(
        stages.map('批量抽样复核', batches, lambda batch, i: repeat_batch(stages, batch, i)),
        controls(stages), return_exceptions=True)
    errors = [value for value in outcomes if isinstance(value, Exception)]
    if errors:
        raise RuntimeError(f'复核存在未完成任务: {errors[0]}')
    checks, pairs = outcomes
    kinds = {pair['kind'] for pair in pairs}
    return {'checks': [row for group in checks for row in group], 'controls': pairs,
            'control_missing_categories': sorted({'source_mismatch', 'firstness_overreach', 'scope_deletion'}-kinds),
            'interpretation': 'Same-model blind repeat and constructed controls from bounded candidates, not new human validation.'}
