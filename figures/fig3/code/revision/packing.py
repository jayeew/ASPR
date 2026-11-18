"""Object-level concern recovery and bounded, report-independent preparation."""
from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import Field, create_model

from .client import BASE
from .config import METHODS, Config
from .materials import batches, encoded, fits
from .models import ConcernBrief, ConcernBriefs, ConcernMatch, ConcernMatches
from .progress import log
from .prompts import PROMPTS
from .storage import MissingInput, Store, read, roster, write

INSTRUCTION = ('Prepare neutral reviewer-concern records. Preserve every concern_id, reviewer, round, '
    'scientific object, scope, applicability, all distinct reasons and necessary qualifications. '
    'Retain short exact review quotes, distinguish reviewer opinion from manuscript facts, and retain '
    'uncertainty. Do not judge any method or report. Aim at 600 characters per brief, never exceeding 1000. '
    'Return one record per supplied concern_id. Do not combine different concerns.')


def one_object_schema(ident: str, brief: bool = False) -> Any:
    """Constrain a missing-object response to its existing task ID, not a new judgment label."""
    item = create_model('RequestedConcern', __base__=ConcernBrief if brief else ConcernMatch,
                        concern_id=(str, Field(json_schema_extra={'enum': [ident]})))
    field = 'concerns' if brief else 'matches'
    return create_model('MissingConcernBrief' if brief else 'MissingConcernMatch',
                        __base__=ConcernBriefs if brief else ConcernMatches,
                        **{field: (list[item], Field(min_length=1, max_length=1))})


def unambiguous(rows: list[dict[str, Any]], expected: set[str]) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        if row['concern_id'] in expected:
            grouped.setdefault(row['concern_id'], []).append(row)
    return {ident: values[0] for ident, values in grouped.items()
            if all(value == values[0] for value in values)}


def directory(config: Config) -> Path:
    return config.output/'inputs/concern_reuse'


def enabled(config: Config) -> bool:
    return (directory(config)/'manifest.json').exists()


def recover_paper(config: Config, paper: str) -> dict[str, Any]:
    store = Store(config)
    concerns = store.get('checklist', paper)['concerns']
    expected = {c['concern_id'] for c in concerns}
    accepted, conflicts, old = {}, {}, {}
    counters: Counter[str] = Counter()
    for method in METHODS:
        candidates: dict[str, list[dict[str, Any]]] = {}
        outside_rows: set[str] = set()
        manifest = config.output/'inputs/tasks/concerns'/method/paper/'真人问题匹配.json'
        groups = read(manifest)['tasks'] if manifest.exists() else []
        for i, group in enumerate(groups):
            step = f'concerns_{i:05d}'
            path = config.output/'annotations/checkpoints/concerns'/method/paper/f'{step}.json'
            if not path.exists():
                continue
            counters['old_answer_batches'] += 1
            scope = {c['concern_id'] for c in group}
            returned = set()
            for row in read(path).get('matches', []):
                if row.get('concern_id') in scope & expected:
                    candidates.setdefault(row['concern_id'], []).append(row)
                    returned.add(row['concern_id'])
                else:
                    counters['outside_batch'] += 1
                    outside_rows.add(encoded(row))
            if returned != scope:
                counters['incomplete_old_answers'] += 1
        # Completed outputs may contain answers whose original checkpoint was interrupted.
        final = store.path('concerns', paper, method)
        if final.exists():
            for row in read(final).get('matches', []):
                if row.get('concern_id') in expected and encoded(row) not in outside_rows:
                    candidates.setdefault(row['concern_id'], []).append(row)
                else:
                    counters['outside_final'] += 1
        accepted[method], conflicts[method] = {}, []
        for ident, rows in candidates.items():
            if all(row == rows[0] for row in rows):
                accepted[method][ident] = rows[0]
            else:
                conflicts[method].append(ident)
        pending = expected - accepted[method].keys()
        old[method] = []
        checkpoints = config.output/'annotations/checkpoints/concerns'/method/paper
        for i, group in enumerate(groups):
            ids = {c['concern_id'] for c in group}
            step = f'concerns_{i:05d}'
            if (ids <= pending and not ids.intersection(conflicts[method])
                    and not (checkpoints/f'{step}.json').exists()
                    and any(checkpoints.glob(step+'_read_*.json'))):
                old[method].append({'step': step, 'objects': group})
                pending -= ids
        counters['planned_old_batches'] += len(groups)
        counters['retained'] += len(accepted[method])
    return {'paper_id': paper, 'accepted': accepted, 'conflicts': conflicts,
            'continue_reading': old, 'counts': dict(counters), 'object_count': len(concerns)*len(METHODS)}


def recover_calls(config: Config) -> int:
    """Recover a completed call at its own logical path, never overwrite a result."""
    stages = {'core', 'reference', 'extract', 'support', 'novelty', 'quality_checklist',
              'quality', 'concerns', 'clusters', 'fusion', 'preference'}
    restored = 0
    errors = []
    for folder in sorted((config.output/'logs/calls').glob('*')):
        record_path = folder/'record.json'
        if not record_path.exists():
            continue
        try:
            record = read(record_path)
            if not isinstance(record, dict):
                raise TypeError('Call record must be a JSON object')
        except (MissingInput, TypeError) as exc:
            errors.append({'path': str(record_path), 'error': str(exc)})
            continue
        if record.get('state') != 'completed' or record.get('stage') not in stages:
            continue
        if not all(isinstance(record.get(key), str) and record[key] for key in ('paper_id', 'step')):
            errors.append({'path': str(record_path), 'error': 'Completed call lacks paper_id or step'})
            continue
        response = folder/'response.json'
        target = config.output/'annotations/checkpoints'/record['stage']/(record.get('method') or 'papers')/record['paper_id']/(record['step']+'.json')
        if response.exists() and not target.exists():
            try:
                result = read(response)
                if not isinstance(result, dict):
                    raise TypeError('Call response must be a JSON object')
            except (MissingInput, TypeError) as exc:
                errors.append({'path': str(response), 'error': str(exc)})
                continue
            write(target, result); restored += 1
    write(config.output/'logs/recovery_errors.json', {'errors': errors, 'restored_calls': restored})
    if errors:
        log(config, '恢复日志提示', f'跳过{len(errors)}个损坏调用记录或回答；原文件保留，详情见logs/recovery_errors.json。已恢复{restored}个断点。')
    return restored


def activate(config: Config) -> dict[str, Any]:
    if enabled(config):
        return read(directory(config)/'manifest.json')
    restored = recover_calls(config)
    totals: Counter[str] = Counter()
    for item in roster(config):
        paper = item['paper_id']
        path = directory(config)/f'{paper}.json'
        recovered = read(path) if path.exists() else recover_paper(config, paper)
        write(path, recovered)
        totals.update(recovered['counts'])
        totals['objects'] += recovered['object_count']
        totals['conflicts'] += sum(len(v) for v in recovered['conflicts'].values())
    # Publish only after every paper has been imported. Existing files are left in place.
    result = {'papers': [r['paper_id'] for r in roster(config)], 'counts': dict(totals),
              'restored_calls': restored, 'max_batch': 4}
    write(directory(config)/'manifest.json', result)
    return result


def object_path(config: Config, paper: str, method: str, ident: str) -> Path:
    return config.output/'annotations/concern_objects'/method/paper/f'{ident}.json'


def answers(config: Config, paper: str, method: str) -> dict[str, Any]:
    result = dict(read(directory(config)/f'{paper}.json')['accepted'][method])
    for path in (config.output/'annotations/concern_objects'/method/paper).glob('*.json'):
        result[path.stem] = read(path)
    return result


def complete(config: Config, paper: str, method: str) -> bool:
    original = Store(config).get('checklist', paper)['concerns']
    return {c['concern_id'] for c in original} <= answers(config, paper, method).keys()


def payload(stages: Any, method: str, cards: list[dict[str, Any]]) -> dict[str, Any]:
    original = {c['concern_id']: c for c in stages.get('checklist')['concerns']}
    return {'checklist': {'concerns': cards}, 'report': stages.report(method)['body'],
            **stages.object_material([original[c['concern_id']] for c in cards], 3000)}


def pack(stages: Any, method: str, cards: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Stable greedy packing; prefer the same selected evidence, max four concerns."""
    def source_keys(card: dict[str, Any]) -> tuple[Any, ...]:
        raw = next(c for c in stages.get('checklist')['concerns'] if c['concern_id'] == card['concern_id'])
        return tuple(b['block_id'] for b in stages.material(raw, 3000)['evidence_blocks'])+(card['concern_id'],)
    result, group = [], []
    for card in sorted(cards, key=source_keys):
        proposed = group+[card]
        material = payload(stages, method, proposed)
        if group and (len(proposed) > 4 or len(encoded(material)) > 64000):
            result.append(group); group = [card]
        else:
            group = proposed
    if group:
        result.append(group)
    return result


async def prepare_cards(stages: Any, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use original concern records; do not replace reviewer wording with a digest."""
    return raw


async def evaluate(stages: Any, method: str) -> dict[str, Any]:
    config, paper = stages.config, stages.ident
    raw = stages.get('checklist')['concerns']
    retained = answers(config, paper, method)
    inherited = len(retained)
    if {c['concern_id'] for c in raw} <= retained.keys():
        return {'matches': [retained[c['concern_id']] for c in raw]}
    recovered = read(directory(config)/f'{paper}.json')
    async def save(group: list[dict[str, Any]], step: str, material: Any) -> None:
        pending = [c for c in group if c['concern_id'] not in retained]
        if not pending:
            return
        prompt = (PROMPTS['concerns'] + '\nReturn one match per supplied concern_id; '
                  'never substitute claim IDs, titles or newly invented identifiers.') if step.startswith('packed_') else None
        value = await stages.ask('concerns', material, ConcernMatches, step, prompt)
        allowed = {c['concern_id'] for c in pending}
        collected: dict[str, list[Any]] = {}
        for row in value['matches']:
            if row['concern_id'] in allowed:
                collected.setdefault(row['concern_id'], []).append(row)
        for ident, rows in collected.items():
            if all(row == rows[0] for row in rows):
                write(config.output/'annotations/concern_object_inputs'/method/paper/f'{ident}.json',
                      {'material': 'original_concern'})
                write(object_path(config, paper, method, ident), rows[0]); retained[ident] = rows[0]
        log(config, '问题对象进度', f'[{len(retained)}/{len(raw)}] 继承{inherited}，本次新增{len(retained)-inherited}',
            stage='concerns', paper_id=paper, method=method)
        missing = [c for c in pending if c['concern_id'] not in retained]
        if missing:
            async def supplement(obj: dict[str, Any]) -> None:
                ident = obj['concern_id']
                # Reuse the same neutral scope, with the complete report and only the missing object.
                packed = step.startswith('packed_')
                inputs = payload(stages, method, [obj]) if packed else stages.concern_payload(method, [obj])
                result = await stages.ask('concerns', inputs, one_object_schema(ident),
                    f'missing_{ident}', PROMPTS['concerns']+f' Return exactly concern_id {ident}; no other IDs.')
                found = unambiguous(result['matches'], {ident})
                if ident not in found:
                    raise ValueError(f'Missing-object response still omitted {ident}')
                write(config.output/'annotations/concern_object_inputs'/method/paper/f'{ident}.json',
                      {'material': 'original_concern'})
                write(object_path(config, paper, method, ident), found[ident]); retained[ident] = found[ident]
            await asyncio.gather(*(supplement(obj) for obj in missing))
    # Finish old partial reading chains with precisely their original inputs and steps.
    await stages.engine.map(paper, 'concern_recovery', method, '继承未完归并', recovered['continue_reading'][method],
        lambda job, i: save(job['objects'], job['step'], lambda: stages.concern_payload(method, job['objects'])))
    cards = await prepare_cards(stages, raw)
    pending = [c for c in cards if c['concern_id'] not in retained]
    manifest = directory(config)/paper/method/'batches.json'
    if manifest.exists():
        groups = read(manifest)['groups']
    else:
        groups = pack(stages, method, pending)
        write(manifest, {'groups': groups})
    from gear.codex_cli import _strict_response_schema
    schema = _strict_response_schema(ConcernMatches.model_json_schema())
    async def judge(group: Any, i: int) -> None:
        remaining = [c for c in group if c['concern_id'] not in retained]
        if not remaining:
            return
        material = payload(stages, method, remaining)
        if not fits(config.model_copy(update={'material_max_chars': 64000, 'request_max_chars': 96000, 'request_max_bytes': 384000}), material, BASE+'\n'+PROMPTS['concerns']+'\nINPUT:\n', schema):
            raise ValueError('Single packed concern exceeds budget; report was not reduced')
        # Object IDs make a partial-result retry distinct from its previous batch.
        step = 'packed_'+'_'.join(c['concern_id'] for c in remaining)
        await save(remaining, step, material)
    await stages.engine.map(paper, 'concern_packed', method, '批量真人问题', groups, judge)
    missing = [c['concern_id'] for c in raw if c['concern_id'] not in retained]
    if missing:
        raise ValueError(f'Unfinished concerns: {missing}')
    return {'matches': [retained[c['concern_id']] for c in raw]}


def estimate(config: Config) -> dict[str, Any]:
    """Read-only forecast; placeholder brief lengths are not scientific outputs."""
    from .execution import Engine
    from .stages import Stages
    engine = Engine(config, 1)
    totals: Counter[str] = Counter()
    try:
        for item in roster(config):
            paper = item['paper_id']
            recovered = recover_paper(config, paper)
            totals.update(recovered['counts'])
            totals['objects'] += recovered['object_count']
            stages = Stages(config, paper, engine)
            raw = stages.get('checklist')['concerns']
            wanted = {c['concern_id'] for c in raw if any(c['concern_id'] not in recovered['accepted'][m] for m in METHODS)}
            large = [c for c in raw if c['concern_id'] in wanted and len(encoded(c)) > 1200]
            preparation = batches(large, 4, 7500)
            totals['neutral_preparation_batches'] += len(preparation)
            totals['oversize_neutral_batches'] += sum(len(encoded({'concerns': g})) > config.material_max_chars for g in preparation)
            for method in METHODS:
                existing = recovered['accepted'][method]
                carried = {c['concern_id'] for job in recovered['continue_reading'][method] for c in job['objects']}
                totals['continue_old_reading_batches'] += len(recovered['continue_reading'][method])
                for job in recovered['continue_reading'][method]:
                    for manifest in (config.output/'inputs/tasks/concerns'/method/paper).glob(job['step']+'_read_L*.json'):
                        totals['known_old_read_calls_remaining'] += sum(
                            not (config.output/'annotations/checkpoints/concerns'/method/paper/(manifest.stem+f'_{i:05d}.json')).exists()
                            for i in range(len(read(manifest)['tasks'])))
                for length, label in ((600, 'short_briefs'), (1000, 'max_briefs')):
                    cards = [c if len(encoded(c)) <= 1200 else {'concern_id': c['concern_id'], 'brief': 'x'*length}
                             for c in raw if c['concern_id'] not in existing and c['concern_id'] not in carried]
                    groups = pack(stages, method, cards)
                    totals['new_evaluation_'+label] += len(groups)
            engine.material_cache.clear()
            engine.artifacts.clear()
    finally:
        for pool in (engine.pool, engine.io_pool, engine.gpu_pool):
            pool.shutdown(wait=True)
    values = dict(totals)
    values['old_unfinished_answer_batches_lower_bound'] = totals['planned_old_batches']-totals['old_answer_batches']
    for label in ('short_briefs', 'max_briefs'):
        values['estimated_remaining_'+label] = (totals['new_evaluation_'+label]+totals['neutral_preparation_batches']
            +totals['continue_old_reading_batches']+totals['known_old_read_calls_remaining'])
    values['forecast_notes'] = ('New evaluation forecast uses 600/1000-character neutral briefs, actual reports and '
        'actual evidence packets. Add neutral preparation and continued old reading. Future recursive reductions, '
        'format repairs, failures and other evaluation stages are not included; this is not an exact total for script02.')
    return values
