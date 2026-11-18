from __future__ import annotations

import asyncio
import copy
import os
import random
import re
from collections import Counter
from typing import Any

from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_100.models import ASPECTS, Analysis, Report
from figure_pipeline.fig4_explanation_study.evaluation import segments

from . import prompts
from .materials import original_blocks, sync_baselines, view
from .models import CONDITIONS, VIEWS, ConditionalReference, Config, Evaluation
from .runtime import Runner, Unavailable, now, upstream_processes


def expected_parts(reference: dict[str, Any], aspect: str | None = None) -> dict[str, set[str]]:
    return {q['question_id']: {p['part_id'] for p in q['answer_parts']} if q['applicable'] else set()
            for q in reference['questions'] if aspect is None or q['aspect'] == aspect}


def material_catalog(config: Config, paper: str) -> dict[str, Any]:
    base = view(config, paper, 'F')
    materials = {}
    for name in VIEWS:
        data = view(config, paper, name)
        blocks = original_blocks(data)
        materials[name] = {'blocks': [{'block_id': b['block_id'], 'source_ids': [p['source_id'] for p in b['provenance']]} for b in blocks],
                               'graph_key': 'K5' if name == 'K5' else 'F'}
    return {'manuscript': base['manuscript'], 'historical_originals': original_blocks(base),
                'native_graphs': {'F': base['graph'], 'K5': view(config, paper, 'K5')['graph']},
                'visible_materials': materials, 'graph_metric_definitions': prompts.GRAPH_DEFINITIONS}


def validate_reference(result: dict[str, Any], reference: dict[str, Any]) -> None:
    expected = {(q, p) for q, parts in expected_parts(reference).items() for p in parts}
    if sorted(v['view_id'] for v in result['views']) != sorted(VIEWS):
        raise ValueError('Conditional reference views incomplete or duplicated')
    for packet in result['views']:
        observed = [(p['question_id'], p['part_id']) for p in packet['parts']]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError('Conditional reference changed fixed part identities')
        if any(p['answerability'] == 'answerable' and not p['evidence'] for p in packet['parts']):
            raise ValueError('Answerable reference lacks evidence')


async def reference(config: Config, runner: Runner, paper: str) -> dict[str, Any]:
    fixed = read(config.output / 'baseline/reference/papers' / f'{paper}.json')
    material = material_catalog(config, paper)
    material['fixed_reference'] = fixed
    write(config.output / 'reference_inputs' / f'{paper}.json', material)
    result = await runner.call(paper, 'reference', 'papers', prompts.REFERENCE, material, ConditionalReference)
    validate_reference(result, fixed)
    write(config.output / 'validated_reference' / f'{paper}.json', result)
    return result


def conditions_for(config: Config, paper: str, requested: list[str] | None = None) -> list[str]:
    repeat = paper in read(config.output / 'cohort.json')['repeated']
    return [c for c in CONDITIONS if (c != 'F_REPEAT' or repeat) and (requested is None or c in requested)]


async def branch(config: Config, runner: Runner, paper: str, condition: str, name: str) -> dict[str, Any]:
    if condition == 'F' or (condition == 'K5' and name == 'gear'):
        source = 'gear' if name == 'gear' else 'graph_F'
        return read(config.output / 'baseline/analysis' / source / f'{paper}.json')
    data = view(config, paper, condition)
    material = {'manuscript': data['manuscript'], 'public_tasks': read(config.output / 'baseline/public_tasks' / f'{paper}.json')}
    if name == 'gear':
        material.update(neutral_claims=[c['claim'] for c in data['graph']['cards']],
                        original_historical_passages=data['gear_evidence']['evidence_blocks'])
    else:
        material.update(native_graph=data['graph'], graph_metric_definitions=prompts.GRAPH_DEFINITIONS,
                        original_historical_passages=data['graph_evidence']['evidence_blocks'])
    key = condition + '_' + name
    write(config.output / 'analysis_inputs' / key / f'{paper}.json', material)
    return await runner.call(paper, 'analysis', key, prompts.GEAR if name == 'gear' else prompts.GRAPH,
                             material, Analysis, model='gpt-5.6-luna' if condition == 'LUNA' else config.model, effort='medium')


async def generate(config: Config, runner: Runner, paper: str, condition: str) -> None:
    target = config.output / 'reports' / condition / f'{paper}.json'
    if condition == 'F':
        value = read(config.output / 'baseline/reports/F' / f'{paper}.json')
        Report.model_validate(value)
        write(target, value)
        runner.status(paper, 'reports', condition, 'reused')
        return
    gear, graph = await asyncio.gather(branch(config, runner, paper, condition, 'gear'),
                                       branch(config, runner, paper, condition, 'graph'))
    data = view(config, paper, condition)
    material = {'manuscript': data['manuscript'], 'graph': data['graph'], 'original_evidence': original_blocks(data),
                    'public_tasks': read(config.output / 'baseline/public_tasks' / f'{paper}.json'),
                    'scientific_evidence_analysis': gear, 'knowledge_graph_analysis': graph,
                    'graph_metric_definitions': prompts.GRAPH_DEFINITIONS}
    write(config.output / 'report_inputs' / condition / f'{paper}.json', material)
    result = await runner.call(paper, 'reports', condition, prompts.WRITER, material, Report,
                              model='gpt-5.6-luna' if condition == 'LUNA' else config.model, effort='medium')
    target.with_suffix('.md').write_text(result['body'] + '\n')


def blind_reference(parts: list[dict[str, Any]], packets: dict[str, str]) -> list[dict[str, Any]]:
    """Replace authored condition labels, preserving source quotations and identities."""
    result = copy.deepcopy(parts)
    pattern = re.compile(r'(?<![A-Za-z0-9_])(?:E50|K5|F)(?![A-Za-z0-9_])')
    def replace(text: str) -> str:
        return pattern.sub(lambda match: packets[match.group()], text)
    for part in result:
        for key in ('expected_content', 'acceptable_variants', 'reason'):
            part[key] = replace(part[key])
        for evidence in part['evidence']:
            evidence['location'] = replace(evidence['location'])
    return result


def evaluation_material(config: Config, paper: str, aspect: str) -> tuple[dict[str, Any], dict[str, str]]:
    fixed = read(config.output / 'baseline/reference/papers' / f'{paper}.json')
    reference_views = {v['view_id']: v['parts'] for v in read(config.output / 'validated_reference' / f'{paper}.json')['views']}
    order = conditions_for(config, paper)
    random.Random(f'{config.seed}:{paper}:blind').shuffle(order)
    mapping, candidates = {}, []
    catalog = material_catalog(config, paper)
    # Random packet identifiers expose availability, never condition or model names.
    packet_ids = {v: f'V{i:02d}' for i, v in enumerate(random.Random(f'{config.seed}:{paper}:views').sample(list(VIEWS), 3), 1)}
    for i, condition in enumerate(order, 1):
        path = config.output / 'reports' / condition / f'{paper}.json'
        if not path.exists():
            continue
        ident = f'R{i:02d}'
        mapping[ident] = condition
        name = condition if condition in VIEWS else 'F'
        candidates.append({'report_id': ident, 'report_segments': segments(read(path)['body']),
                               'material_packet_id': packet_ids[name], 'conditional_reference': blind_reference(
                                   [p for p in reference_views[name] if p['question_id'] in expected_parts(fixed, aspect)], packet_ids)})
    catalog['native_graphs'] = {packet_ids[v]: view(config, paper, v)['graph'] for v in ('F', 'K5')}
    catalog['visible_materials'] = {packet_ids[v]: {**m, 'graph_key': packet_ids[m['graph_key']]}
                                    for v, m in catalog['visible_materials'].items()}
    catalog.update(reference_questions=[q for q in fixed['questions'] if q['aspect'] == aspect], candidates=candidates)
    return catalog, mapping


def align_evaluation(result: dict[str, Any], material: dict[str, Any]) -> dict[str, Any]:
    expected = {q['question_id']: {p['part_id'] for p in q['answer_parts']} if q['applicable'] else set()
                for q in material['reference_questions']}
    candidates = {c['report_id']: c for c in material['candidates']}
    observed = [c['report_id'] for c in result['candidates']]
    if set(observed) != set(candidates) or len(observed) != len(set(observed)):
        raise ValueError('Evaluation report identities incomplete/duplicated')
    aligned = copy.deepcopy(result)
    for candidate in aligned['candidates']:
        questions = candidate['questions']
        if len(questions) != len(expected) or {q['question_id'] for q in questions} != set(expected):
            raise ValueError('Evaluation changed question identities')
        quotes = {s['segment_id']: s['text'] for s in candidates[candidate['report_id']]['report_segments']}
        for q in questions:
            if len(q['parts']) != len(expected[q['question_id']]) or {p['part_id'] for p in q['parts']} != expected[q['question_id']]:
                raise ValueError('Evaluation changed part identities')
            for part in q['parts']:
                ids = part['report_segment_ids'] + part['abstention_segment_ids'] + part['unsupported_segment_ids']
                invalid = any(i not in quotes for i in ids)
                missing = ((part['status'] in ('correct', 'incorrect') and not part['report_segment_ids']) or
                           (part['explicitly_abstains'] is True and not part['abstention_segment_ids']) or
                           (part['unsupported_definitive'] is True and not part['unsupported_segment_ids']))
                inconsistent = ((part['explicitly_abstains'] is False and part['abstention_appropriateness'] != 'not_applicable') or
                                (part['explicitly_abstains'] is True and part['abstention_appropriateness'] == 'not_applicable'))
                part['technical_state'] = 'invalid_evidence_alignment' if invalid or missing or inconsistent else 'completed'
                for name, key in [('report_quotes', 'report_segment_ids'), ('abstention_quotes', 'abstention_segment_ids'), ('unsupported_quotes', 'unsupported_segment_ids')]:
                    part[name] = [quotes[i] for i in part[key] if i in quotes]
    return aligned


async def evaluate(config: Config, runner: Runner, paper: str, aspect: str) -> None:
    material, mapping = evaluation_material(config, paper, aspect)
    write(config.output / 'evaluation_mapping' / aspect / f'{paper}.json', mapping)
    if not material['candidates']:
        raise Unavailable('No candidate reports')
    questions = material['reference_questions']
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for candidate in material['candidates']:
        if current and not runner.fits(prompts.EVALUATE, {**material, 'candidates': current + [candidate]}, Evaluation):
            batches.append(current)
            current = []
        current.append(candidate)
    if current:
        batches.append(current)
    all_candidates = []
    for i, batch in enumerate(batches):
        packet = {**material, 'candidates': batch}
        key = aspect if len(batches) == 1 else f'{aspect}_batch{i + 1}'
        write(config.output / 'evaluation_inputs' / key / f'{paper}.json', packet)
        if not any(q['applicable'] for q in questions):
            result = {'candidates': [{'report_id': c['report_id'], 'questions': [{'question_id': q['question_id'], 'parts': []} for q in questions]} for c in batch]}
        else:
            result = await runner.call(paper, 'evaluation', key, prompts.EVALUATE, packet, Evaluation, additional=i > 0)
        all_candidates.extend(align_evaluation(result, packet)['candidates'])
        write(config.output / 'aligned_evaluation' / aspect / f'{paper}.json', {'candidates': all_candidates})
    runner.status(paper, 'evaluation', aspect, 'completed')


async def protected(runner: Runner, paper: str, stage: str, condition: str, job: Any) -> Any:
    try:
        return await job
    except (OSError, ValueError, RuntimeError) as exc:
        runner.status(paper, stage, condition, 'unresolved', reason=str(exc))
        print(f'UNRESOLVED {paper} {stage}/{condition}: {str(exc)[:250]}', flush=True)
        return None


async def paper_run(config: Config, runner: Runner, paper: str, stage: str, requested: list[str] | None) -> None:
    if read(config.output / 'baseline_status.json')[paper] != 'ready':
        runner.status(paper, 'pipeline', 'all', 'upstream_unavailable', reason='Fig4 baseline incomplete; sample not replaced')
        return
    ref = await protected(runner, paper, 'reference', 'papers', reference(config, runner, paper))
    if ref is None or stage == 'reference':
        return
    if stage in ('all', 'generate'):
        await asyncio.gather(*(protected(runner, paper, 'reports', c, generate(config, runner, paper, c))
                               for c in conditions_for(config, paper, requested)))
    if stage in ('all', 'evaluate'):
        await asyncio.gather(*(protected(runner, paper, 'evaluation', a, evaluate(config, runner, paper, a)) for a in ASPECTS))


def check_first_two(config: Config, papers: list[str]) -> dict[str, Any]:
    checked = []
    for paper in papers:
        report_counts = 0
        for condition in conditions_for(config, paper):
            path = config.output / 'report_inputs' / condition / f'{paper}.json'
            if condition == 'F' or not path.exists():
                continue
            actual, expected = read(path), view(config, paper, condition)
            if (actual['manuscript'] != expected['manuscript'] or actual['graph'] != expected['graph']
                    or actual['original_evidence'] != original_blocks(expected)):
                raise ValueError(f'Actual generation inputs differ from prepared condition: {paper}/{condition}')
            allowed = {'manuscript', 'graph', 'original_evidence', 'public_tasks',
                       'scientific_evidence_analysis', 'knowledge_graph_analysis', 'graph_metric_definitions'}
            if set(actual) != allowed:
                raise ValueError('Writer contains unexpected/private fields')
            report_counts += (config.output / 'reports' / condition / f'{paper}.json').exists()
        technical = Counter()
        for path in (config.output / 'aligned_evaluation').glob(f'*/{paper}.json'):
            technical.update(p['technical_state'] for c in read(path)['candidates'] for q in c['questions'] for p in q['parts'])
        checked.append({'paper_id': paper, 'new_reports_present': report_counts, 'part_alignment_states': dict(technical)})
    return {'papers': checked, 'time': now(), 'checks': 'Actual writer material and private-field boundary; exact quote alignment',
            'scientific_validation': 'model evaluation, not independent expert confirmation', 'winner_gate': False}


async def run(config: Config, stage: str, papers: list[str] | None, conditions: list[str] | None, wait: bool) -> None:
    cohort = read(config.output / 'cohort.json')
    ids = papers or cohort['selected']
    if not set(ids) <= set(cohort['selected']):
        raise ValueError('Paper outside fixed Fig5 cohort')
    while upstream_processes():
        write(config.output / 'run_state.json', {'state': 'waiting_fig4', 'time': now(), 'pids': upstream_processes()})
        if not wait:
            raise Unavailable('Fig4 still running; use --wait-for-fig4 for the approved sequence')
        await asyncio.sleep(30)
    sync_baselines(config)
    runner = Runner(config)
    write(config.output / 'run_state.json', {'state': 'running', 'stage': stage, 'time': now(), 'pid': os.getpid(),
                                           'cli_limit': config.cli_limit})
    # All private references precede new reports, including the two integration papers.
    if stage != 'evaluate':
        await asyncio.gather(*(protected(runner, p, 'reference', 'papers', reference(config, runner, p))
                               for p in ids if read(config.output / 'baseline_status.json')[p] == 'ready'))
    # The first two are ordinary study papers, never a performance-based selection gate.
    first = sorted(ids)[:2]
    await asyncio.gather(*(paper_run(config, runner, p, stage, conditions) for p in first))
    write(config.output / 'first_two_status.json', check_first_two(config, first))
    await asyncio.gather(*(paper_run(config, runner, p, stage, conditions) for p in ids if p not in first))
    write(config.output / 'run_state.json', {'state': 'finished_with_recorded_statuses', 'stage': stage, 'time': now()})
