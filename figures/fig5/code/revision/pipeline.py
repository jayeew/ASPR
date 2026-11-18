from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_100.models import ASPECTS, Analysis, Report
from figure_pipeline.fig4_explanation_study.evaluation import segments
from figure_pipeline.fig5_robustness.materials import original_blocks
from figure_pipeline.fig5_robustness.pipeline import (
    align_evaluation,
    expected_parts,
)
from figure_pipeline.fig5_robustness.runtime import now

from . import prompts
from .materials import digest, packet, retain, save_packet
from .models import Config, Evaluation, Reference, Supports
from .runtime import Runner, protected


def fixed(config: Config, paper: str) -> dict[str, Any]:
    return read(config.output / 'baseline/reference/papers' / f'{paper}.json')


def public(config: Config, paper: str) -> Any:
    return read(config.output / 'baseline/public_tasks' / f'{paper}.json')


def visible(data: dict[str, Any]) -> dict[str, Any]:
    return {'manuscript': data['manuscript'], 'graph': data['graph'], 'original_evidence': original_blocks(data),
            'graph_metric_definitions': prompts.DEFINITIONS}


def visible_aliases(config: Config, paper: str, data: dict[str, Any]) -> dict[str, str]:
    aliases = read(config.output / 'identities' / f'{paper}.json')
    return {p['source_id']: aliases[p['source_id']] for b in original_blocks(data) for p in b['provenance']}


def validate_reference(value: dict[str, Any], reference: dict[str, Any], data: dict[str, Any]) -> None:
    expected = {(q, p) for q, parts in expected_parts(reference).items() for p in parts}
    actual = [(p['question_id'], p['part_id']) for p in value['parts']]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('Conditional reference changed fixed identities')
    source_ids = {p['source_id'] for b in original_blocks(data) for p in b['provenance']}
    block_ids = {b['block_id'] for b in original_blocks(data)}
    for part in value['parts']:
        if part['answerability'] == 'answerable' and not part['evidence']:
            raise ValueError('Answerable without evidence')
        for evidence in part['evidence']:
            if evidence['kind'] == 'history' and evidence['source_id'] not in source_ids | block_ids:
                raise ValueError(f"Reference cites invisible historical source: {evidence['source_id']}")


async def reference(config: Config, runner: Runner, paper: str, view: str) -> dict[str, Any]:
    data = packet(config, paper, view)
    if view == 'ORDER':
        from .reference_reuse import sync_order_references
        await reference(config, runner, paper, 'F')
        sync_order_references(config)
    validated = config.output / 'validated_reference' / view / f'{paper}.json'
    if validated.exists():
        value = read(validated)
        validate_reference(value, fixed(config, paper), data)
        return value
    material = {'visible_packet': visible(data), 'independent_work_aliases': visible_aliases(config, paper, data), 'public_tasks': public(config, paper), 'fixed_reference': fixed(config, paper)}
    write(config.output / 'reference_inputs' / view / f'{paper}.json', material)
    value = await runner.call(paper, 'reference_final', view, prompts.REFERENCE, material, Reference)
    validate_reference(value, fixed(config, paper), data)
    write(config.output / 'validated_reference' / view / f'{paper}.json', value)
    return value


def validate_support(value: dict[str, Any], aliases: dict[str, str], data: dict[str, Any], ref: dict[str, Any]) -> None:
    expected = {(q, p) for q, parts in expected_parts(ref).items() for p in parts}
    works = set(aliases.values())
    blocks = {b['block_id'] for b in original_blocks(data)}
    if len(value['targets']) > 1:
        raise ValueError('At most one prespecified target per paper')
    for target in value['targets']:
        critical, control = set(target['critical_remove']), set(target['noncritical_remove'])
        groups = [set(g) for g in target['support_groups']]
        if (target['question_id'], target['part_id']) not in expected:
            raise ValueError('Unknown support target')
        if not critical or len(critical) != len(control) or critical & control or not critical | control <= works:
            raise ValueError('Invalid matched deletion identities')
        if not groups or any(not g or not g <= works or not g & critical for g in groups):
            raise ValueError('Critical deletion does not hit every support group')
        if not any(not g & control for g in groups):
            raise ValueError('Control destroys every support group')
        if not target['support_block_ids'] or not set(target['support_block_ids']) <= blocks:
            raise ValueError('Missing or invalid support block IDs')


async def support_design(config: Config, runner: Runner, paper: str) -> list[dict[str, Any]]:
    data, aliases = packet(config, paper, 'F'), read(config.output / 'identities' / f'{paper}.json')
    material = {'visible_packet': visible(data), 'source_aliases': aliases, 'fixed_reference': fixed(config, paper), 'public_tasks': public(config, paper)}
    write(config.output / 'support_inputs' / f'{paper}.json', material)
    previous = config.output / 'supports/full' / f'{paper}.json'
    if previous.exists() and read(previous)['targets']:
        value = read(previous)
        write(config.output / 'support_adoption' / f'{paper}.json', {'source': str(previous), 'reason': 'existing proposed target retained; same independent conditional-reference gate still required'})
    else:
        value = await runner.call(paper, 'supports_v2', 'full', prompts.SUPPORT, material, Supports)
    validate_support(value, aliases, data, fixed(config, paper))
    write(config.output / 'support_results' / f'{paper}.json', value)
    if not value['targets']:
        write(config.output / 'eligibility' / f'{paper}.json', {'eligible': False, 'reason': value['ineligible_reason']})
        return []
    target = value['targets'][0]
    for name, key in [('CRITICAL', 'critical_remove'), ('NONCRITICAL', 'noncritical_remove')]:
        kept = set(aliases.values()) - set(target[key])
        changed = retain(data, aliases, kept)
        save_packet(config, paper, name, changed, retained=sorted(kept), removed=target[key], target_question=target['question_id'], target_part=target['part_id'],
                    removed_characters=sum(len(b['text']) for b in original_blocks(data)) - sum(len(b['text']) for b in original_blocks(changed)))
    refs = await asyncio.gather(*(reference(config, runner, paper, v) for v in ('F', 'CRITICAL', 'NONCRITICAL')))
    states = [next(p['answerability'] for p in r['parts'] if (p['question_id'], p['part_id']) == (target['question_id'], target['part_id'])) for r in refs]
    eligible = states == ['answerable', 'unanswerable', 'answerable']
    write(config.output / 'eligibility' / f'{paper}.json', {'eligible': eligible, 'states': states, 'target': target,
          'reason': 'pre-generation conditional references validate target transitions' if eligible else 'conditional reference does not verify required transition'})
    if not eligible:
        return []
    cohort = read(config.output / 'cohort.json')
    split = 'exploration' if paper in cohort['exploration'] else 'confirmation'
    return [{'paper_id': paper, 'condition': c, 'view': c, 'split': split, 'kind': 'multistage'} for c in ('CRITICAL', 'NONCRITICAL')]


def generation_packet(config: Config, task: dict[str, Any]) -> dict[str, Any]:
    return packet(config, task['paper_id'], task['view'])


async def generate(config: Config, runner: Runner, task: dict[str, Any]) -> None:
    paper, condition = task['paper_id'], task['condition']
    target = config.output / 'reports' / condition / f'{paper}.json'
    if target.exists():
        Report.model_validate(read(target))
        return
    start = time.monotonic()
    write(config.output / 'timing' / condition / f'{paper}.json', {'started_at': now(), 'state': 'running'})
    data = generation_packet(config, task)
    model = 'gpt-5.6-luna' if condition in ('LUNA', 'DW_LUNA') else config.model
    effort = {'LOW': 'low', 'HIGH': 'high'}.get(condition, 'medium')
    material = {**visible(data), 'public_tasks': public(config, paper)}
    if not condition.startswith('DW'):
        async def branch(name: str) -> dict[str, Any]:
            if name == 'gear' and condition in ('K3', 'K5', 'G50', 'G25'):
                return read(config.output / 'baseline/analysis/gear' / f'{paper}.json')
            content = {'manuscript': data['manuscript'], 'public_tasks': public(config, paper)}
            if name == 'gear':
                content.update(neutral_claims=[c['claim'] for c in data['graph']['cards']], original_historical_passages=data['gear_evidence']['evidence_blocks'])
            else:
                content.update(native_graph=data['graph'], graph_metric_definitions=prompts.DEFINITIONS, original_historical_passages=data['graph_evidence']['evidence_blocks'])
            write(config.output / 'analysis_inputs' / f'{condition}_{name}' / f'{paper}.json', content)
            return await runner.call(paper, 'analysis', f'{condition}_{name}', prompts.GEAR if name == 'gear' else prompts.GRAPH, content, Analysis, model=model, effort=effort)
        gear, graph = await asyncio.gather(branch('gear'), branch('graph'))
        material.update(scientific_evidence_analysis=gear, knowledge_graph_analysis=graph)
    write(config.output / 'report_inputs' / condition / f'{paper}.json', material)
    value = await runner.call(paper, 'reports', condition, prompts.DW if condition.startswith('DW') else prompts.WRITER, material, Report, model=model, effort=effort)
    target.with_suffix('.md').write_text(value['body'] + '\n')
    write(config.output / 'timing' / condition / f'{paper}.json', {'state': 'completed', 'wall_seconds': time.monotonic() - start, 'ended_at': now(), 'includes_queue_wait': True})


async def evaluate(config: Config, runner: Runner, paper: str, view: str, tasks: list[dict[str, Any]], aspect_only: str | None = None) -> None:
    data, ref = packet(config, paper, view), read(config.output / 'validated_reference' / view / f'{paper}.json')
    available = [t for t in tasks if (config.output / 'reports' / t['condition'] / f'{paper}.json').exists()]
    random.Random(f'{config.seed}:{paper}:{view}:blind').shuffle(available)
    for aspect in ([aspect_only] if aspect_only else ASPECTS):
        questions = [q for q in fixed(config, paper)['questions'] if q['aspect'] == aspect]
        for index in range(0, len(available), 4):
            batch = available[index:index + 4]
            mapping = {f'R{i + 1:02d}': t['condition'] for i, t in enumerate(batch)}
            candidates = [{'report_id': r, 'report_segments': segments(read(config.output / 'reports' / c / f'{paper}.json')['body']),
                           'conditional_reference': [p for p in ref['parts'] if p['question_id'] in {q['question_id'] for q in questions}]} for r, c in mapping.items()]
            material = {'visible_packet': visible(data), 'independent_work_aliases': visible_aliases(config, paper, data), 'reference_questions': questions, 'public_tasks': public(config, paper), 'candidates': candidates}
            if len(batch) > 1 and not runner.fits(prompts.EVALUATE, material, Evaluation):
                # Split complete candidates, never truncate originals or report text.
                middle = len(batch) // 2
                await evaluate(config, runner, paper, view, batch[:middle], aspect)
                await evaluate(config, runner, paper, view, batch[middle:], aspect)
                continue
            key = f'{view}_{aspect}_b{index // 4}'
            # Mapping fixes batch composition on resume; new candidates get a distinct hash key.
            key += '_' + digest(mapping)[:8]
            write(config.output / 'evaluation_mapping' / key / f'{paper}.json', mapping)
            write(config.output / 'evaluation_inputs' / key / f'{paper}.json', material)
            if any(q['applicable'] for q in questions):
                value = await runner.call(paper, 'evaluation', key, prompts.EVALUATE, material, Evaluation)
            else:
                value = {'candidates': [{'report_id': r, 'questions': [{'question_id': q['question_id'], 'parts': []} for q in questions]} for r in mapping]}
            aligned = align_evaluation(value, material)
            for candidate in aligned['candidates']:
                for q in candidate['questions']:
                    for p in q['parts']:
                        consistent = (p['response_type'] == 'target_abstention' and p['explicitly_abstains'] is True or
                                      p['response_type'] in ('substantive_answer', 'omission') and p['explicitly_abstains'] is False or
                                      p['response_type'] == 'unresolved' and p['explicitly_abstains'] is None)
                        if not consistent:
                            p['technical_state'] = 'invalid_response_taxonomy'
                write(config.output / 'aligned' / mapping[candidate['report_id']] / aspect / f'{paper}.json', candidate)


async def run(config: Config, stage: str, paper_ids: list[str] | None = None, reference_views: list[str] | None = None) -> None:
    runner = Runner(config)
    cohort = read(config.output / 'cohort.json')
    tasks = read(config.output / 'tasks.json')
    papers = paper_ids or [*cohort['exploration'], *cohort['confirmation'], *[p for p in cohort['paper_ids'] if p not in cohort['exploration'] + cohort['confirmation'] + cohort['development']]]
    async def one(paper: str) -> None:
        current = [t for t in tasks if t['paper_id'] == paper]
        saved = config.output / 'paper_tasks' / f'{paper}.json'
        if saved.exists():
            current += [t for t in read(saved) if t['condition'] not in {x['condition'] for x in current}]
        # Gate every view on a private reference before generating its reports.
        valid_views = set()
        for view in sorted({t['view'] for t in current}):
            if reference_views and view not in reference_views:
                continue
            ref = await protected(runner, paper, 'validated_reference', view, reference(config, runner, paper, view))
            if ref is not None:
                valid_views.add(view)
        from .reference_audit import audit_references
        if reference_views is None and stage in ('all', 'reference'):
            audit = await protected(runner, paper, 'reference_consistency', 'nested_originals',
                                    audit_references(config, runner, paper))
            # A technical audit failure cannot silently pass into generation/evaluation.
            if audit is not True:
                return
            runner.status(paper, 'reference_consistency', 'nested_originals', 'completed')
        if paper in cohort['exploration'] + cohort['confirmation'] and stage in ('all', 'reference') and reference_views is None:
            additions = await protected(runner, paper, 'support_design', 'full', support_design(config, runner, paper))
            if additions:
                current += [t for t in additions if t['condition'] not in {x['condition'] for x in current}]
                valid_views.update(t['view'] for t in additions)
                if paper in cohort['small'] and not any(t['condition'] == 'DW_CRITICAL' for t in current):
                    current.append({'paper_id': paper, 'condition': 'DW_CRITICAL', 'view': 'CRITICAL', 'split': 'exploration', 'kind': 'direct'})
                # Recovery eligibility is chosen by preselected paper identity, never candidate outcomes.
                if paper in cohort['small'][:5] and not any(t['condition'] == 'RESTORE' for t in current):
                    current.append({'paper_id': paper, 'condition': 'RESTORE', 'view': 'F', 'split': 'exploration', 'kind': 'multistage'})
        write(config.output / 'paper_tasks' / f'{paper}.json', current)
        if stage == 'reference':
            return
        for task in current:
            if task['view'] in valid_views and stage in ('all', 'generate'):
                await protected(runner, paper, 'generation', task['condition'], generate(config, runner, task))
        if stage in ('all', 'evaluate'):
            for view in sorted(valid_views):
                await protected(runner, paper, 'grading', view, evaluate(config, runner, paper, view, [t for t in current if t['view'] == view]))
        if stage in ('all', 'evaluate') and paper in cohort['small'] and 'F' in valid_views:
            await protected(runner, paper, 'independent_review', 'F', independent_review(config, runner, paper))
    # Bounded paper fan-out; request-level semaphore independently caps CLI calls.
    semaphore = asyncio.Semaphore(config.cli_limit)
    async def bounded(paper: str) -> None:
        async with semaphore:
            await one(paper)
    await asyncio.gather(*(bounded(p) for p in papers))


async def independent_review(config: Config, runner: Runner, paper: str) -> None:
    """Second model, blind order, fixed small cohort; never overwrite primary judgments."""
    methods = ['F', 'LUNA', 'DW_F', 'DW_LUNA']
    available = [c for c in methods if (config.output / 'reports' / c / f'{paper}.json').exists()]
    random.Random(f'{config.seed}:{paper}:independent').shuffle(available)
    reference_data = read(config.output / 'validated_reference/F' / f'{paper}.json')
    for aspect in ASPECTS:
        questions = [q for q in fixed(config, paper)['questions'] if q['aspect'] == aspect]
        mapping = {f'R{i + 1:02d}': c for i, c in enumerate(available)}
        candidates = [{'report_id': r, 'report_segments': segments(read(config.output / 'reports' / c / f'{paper}.json')['body']),
                       'conditional_reference': [p for p in reference_data['parts'] if p['question_id'] in {q['question_id'] for q in questions}]} for r, c in mapping.items()]
        material = {'visible_packet': visible(packet(config, paper, 'F')), 'independent_work_aliases': visible_aliases(config, paper, packet(config, paper, 'F')), 'reference_questions': questions,
                    'public_tasks': public(config, paper), 'candidates': candidates}
        key = f'{aspect}_{digest(mapping)[:8]}'
        write(config.output / 'independent_inputs' / key / f'{paper}.json', material)
        write(config.output / 'independent_mapping' / key / f'{paper}.json', mapping)
        if not candidates or not any(q['applicable'] for q in questions):
            continue
        value = await runner.call(paper, 'independent_evaluation', key, prompts.EVALUATE, material, Evaluation,
                                  model='gpt-6-astra', effort='high')
        aligned = align_evaluation(value, material)
        write(config.output / 'independent_aligned' / key / f'{paper}.json', aligned)
