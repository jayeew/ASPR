from __future__ import annotations

import asyncio
import random
import re
import signal
from typing import Any

from figure_pipeline.fig3_revision.client import terminate_active
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_study.evaluation import segments

from . import prompts
from .materials import base, public_tasks, view
from .models import ASPECTS, CONDITIONS, NAMES, QUESTION_ASPECTS, QUESTION_IDS, Analysis, Config, Judgments, Reference, Report
from .runtime import Runner, TaskUnavailable


def selected(config: Config, pilot: bool, only: list[str] | None) -> list[str]:
    cohort = read(config.output / 'cohort.json')
    ids = cohort['pilot_ids'] if pilot else [p for p in cohort['paper_ids'] if p not in cohort['pilot_ids']]
    if not pilot:
        acceptance = config.output / 'pilot_acceptance.json'
        if not acceptance.exists() or not read(acceptance).get('accepted'):
            raise ValueError('Remaining 95 require the user acceptance record; pilot results are not automatic authorization')
    if only:
        if not set(only) <= set(ids):
            raise ValueError('Selected paper is outside the authorized cohort')
        ids = [p for p in ids if p in only]
    return ids


async def reference(config: Config, runner: Runner, paper: str) -> dict[str, Any]:
    material = read(config.output / 'inputs/reference' / f'{paper}.json')
    if config.focused_refinement and not config.expanded_protocol:
        from .refinement import refined_reference
        return await refined_reference(config, runner, paper)
    prompt = prompts.REFERENCE
    if config.expanded_protocol:
        material.update(factual_opportunities=read(config.output / 'factual_opportunities' / f'{paper}.json'),
                        graph_metric_definitions=prompts.GRAPH_DEFINITIONS)
        prompt = prompts.EXPANDED_REFERENCE
    result = await runner.call(paper, 'reference', 'papers', prompt, material, Reference, config.evaluation_effort)
    questions = result['questions']
    if [q['question_id'] for q in questions] != list(QUESTION_IDS):
        raise ValueError('Reference question identities/order incomplete; raw response retained')
    for q in questions:
        if q['aspect'] != QUESTION_ASPECTS[q['question_id']]:
            raise ValueError('Question aspect mismatch')
        if q['applicable'] and not 1 <= len(q['answer_parts']) <= 2:
            raise ValueError('Applicable question must have one or two answer parts')
        if config.expanded_protocol and q['applicable'] and q['question_id'][0] in 'JS' and len(q['answer_parts']) != 2:
            raise ValueError('Accepted joint/structural task requires two scientific parts')
        if not q['applicable'] and q['answer_parts']:
            raise ValueError('Inapplicable question has answer parts')
        if re.search(r'::|\b(?:GEAR_|HISTORY_|T\d{6}|10\.1038/)', q['question']):
            raise ValueError('Public task contains a private graph/source identifier')
    public = {'questions': [{k: q[k] for k in ('question_id', 'aspect', 'question')} for q in questions]}
    write(config.output / 'public_tasks' / f'{paper}.json', public)
    write(config.output / 'screening' / f'{paper}.json', {
        'graph_useful': result['graph_useful'], 'reason': result['eligibility_reason'], 'examples': result['graph_examples']})
    return result


async def analysis(config: Config, runner: Runner, paper: str, branch: str) -> dict[str, Any]:
    data = base(config, paper)
    material: dict[str, Any] = {'manuscript': data['manuscript'], 'public_tasks': public_tasks(config, paper)}
    if branch == 'gear':
        material.update(neutral_claims=[c['claim'] for c in data['graph']['cards']],
                        original_historical_passages=data['gear_evidence']['evidence_blocks'])
        prompt = prompts.GEAR_PROMPT
    else:
        condition = branch.removeprefix('graph_')
        material.update(native_graph=view(config, paper, condition)['graph'],
                        graph_metric_definitions=prompts.GRAPH_DEFINITIONS,
                        original_historical_passages=data['graph_evidence']['evidence_blocks'])
        prompt = prompts.GRAPH_PROMPT
    write(config.output / 'analysis_inputs' / branch / f'{paper}.json', material)
    extension = prompts.REFINED_ANALYSIS if config.focused_refinement else ''
    return await runner.call(paper, 'analysis', branch, prompt + prompts.TASK_ANALYSIS + extension,
                             material, Analysis, config.generation_effort)


async def report(config: Config, runner: Runner, paper: str, condition: str,
                 branches: dict[str, asyncio.Task[Any]]) -> None:
    material = view(config, paper, condition)
    material['public_tasks'] = public_tasks(config, paper)
    if condition not in ('T', 'G'):
        material['scientific_evidence_analysis'] = await branches['gear']
    if condition not in ('T', 'E'):
        key = 'graph_F' if condition in ('G', 'F') else 'graph_' + condition
        material['knowledge_graph_analysis'] = await branches[key]
        material['graph_metric_definitions'] = prompts.GRAPH_DEFINITIONS
    write(config.output / 'report_inputs' / condition / f'{paper}.json', material)
    extension = prompts.REFINED_WRITER if config.focused_refinement else ''
    result = await runner.call(paper, 'reports', condition, prompts.WRITER + extension, material, Report, config.generation_effort)
    (config.output / 'reports' / condition / f'{paper}.md').write_text(result['body'] + '\n', encoding='utf-8')


def availability(config: Config, paper: str, condition: str) -> dict[str, Any]:
    material = view(config, paper, condition)
    return {'manuscript': True,
            'original_history_source_ids': sorted({p['source_id'] for b in material.get('original_evidence', []) for p in b['provenance']}),
            'original_history_block_ids': [b['block_id'] for b in material.get('original_evidence', [])],
            'single_contribution_graph': condition not in ('T', 'E'),
            'joint_graph': condition not in ('T', 'E', 'F_noJ'),
            'non_path_structural_numbers': condition not in ('T', 'E', 'F_noM'),
            'citation_annotations': condition not in ('T', 'E', 'F_noP'),
            'raw_nodes_edges_communities': condition not in ('T', 'E'), 'manuscript_bibliography_retained': True}


def candidates(config: Config, paper: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    order = list(CONDITIONS)
    random.Random(paper + ':expansion_v1').shuffle(order)
    values, mapping = [], {}
    for i, condition in enumerate(order, 1):
        path = config.output / 'reports' / condition / f'{paper}.json'
        if path.exists():
            ident = f'R{i:02d}'
            mapping[ident] = condition
            values.append({'report_id': ident, 'report_segments': segments(read(path)['body']),
                           'material_availability': availability(config, paper, condition)})
    return values, mapping


async def evaluate(config: Config, runner: Runner, paper: str, aspect: str) -> None:
    ref = read(config.output / 'reference/papers' / f'{paper}.json')
    questions = [q for q in ref['questions'] if q['aspect'] == aspect]
    values, mapping = candidates(config, paper)
    write(config.output / 'evaluation_mapping' / f'{paper}.json', mapping)
    if not values:
        raise TaskUnavailable('No reports available for evaluation')
    if not any(q['applicable'] for q in questions):
        result = {'candidates': [{'report_id': v['report_id'], 'questions': [
            {'question_id': q['question_id'], 'parts': []} for q in questions]} for v in values]}
    elif (config.output / 'evaluation_review' / aspect / f'{paper}.json').exists() and not config.review_joint:
        result = Judgments.model_validate(read(config.output / 'evaluation_review' / aspect / f'{paper}.json')).model_dump()
    else:
        actual = read(config.output / 'inputs/reference' / f'{paper}.json')
        material = {k: actual[k] for k in ('manuscript', 'native_graph', 'evidence_blocks')}
        material.update(reference_questions=questions, candidates=values,
                        graph_metric_definitions=prompts.GRAPH_DEFINITIONS)
        extension = prompts.REFINED_EVALUATION if config.focused_refinement else ''
        if config.review_joint or (config.expanded_protocol and aspect == 'joint_contribution'):
            extension += prompts.JOINT_REVIEW
        result = await runner.call(paper, 'evaluation_review' if config.review_joint else 'evaluation', aspect, prompts.EVALUATION + extension,
                                   material, Judgments, config.evaluation_effort)
    returned = [c['report_id'] for c in result['candidates']]
    if len(returned) != len(set(returned)) or set(returned) != set(mapping):
        raise ValueError('Incomplete or duplicate evaluated report IDs')
    for candidate in result['candidates']:
        expected = {q['question_id']: {p['part_id'] for p in q['answer_parts']} for q in questions}
        observed = {q['question_id']: {p['part_id'] for p in q['parts']} for q in candidate['questions']}
        if observed != expected or len(candidate['questions']) != len(expected):
            raise ValueError('Evaluation omitted or changed fixed question/part identities')
        source = next(v for v in values if v['report_id'] == candidate['report_id'])
        index = {s['segment_id']: s['text'] for s in source['report_segments']}
        for q in candidate['questions']:
            for part in q['parts']:
                ids = part['report_segment_ids']
                repairs = []
                for i, ident in enumerate(ids):
                    canonical = f'P{int(ident[1:]):03d}' if re.fullmatch(r'P\d+', ident) else ident
                    if ident not in index and canonical in index:
                        repairs.append({'original': ident, 'resolved': canonical, 'reason': 'zero_padding_only'})
                        ids[i] = canonical
                if repairs:
                    part['segment_id_repairs'] = repairs
                part['technical_state'] = ('invalid_segment_id' if any(i not in index for i in ids) else
                                           'missing_report_evidence' if not ids and part['status'] in ('correct', 'incorrect') else 'completed')
                part['report_quotes'] = [index[i] for i in ids if i in index]
    write(config.output / 'aligned_evaluation' / aspect / f'{paper}.json', result)


async def protected(runner: Runner, paper: str, stage: str, condition: str, work: Any) -> Any:
    try:
        return await work
    except (OSError, ValueError, RuntimeError) as exc:
        previous = runner.config.output / 'task_status' / stage / condition / f'{paper}.json'
        category = read(previous).get('category') if previous.exists() else None
        runner.status(paper, stage, condition, 'unresolved', reason=str(exc), category=category)
        print(f'UNRESOLVED {paper} {stage}/{condition}: {str(exc)[:350]}', flush=True)
        return None


async def paper_run(config: Config, runner: Runner, paper: str, stage: str, conditions: list[str] | None) -> None:
    ref = await protected(runner, paper, 'reference', 'papers', reference(config, runner, paper))
    if ref is None or not ref['graph_useful'] or stage == 'reference':
        return
    if stage != 'evaluate':
        requested = conditions or CONDITIONS
        keys = set()
        for condition in requested:
            if condition not in ('T', 'G'):
                keys.add('gear')
            if condition not in ('T', 'E'):
                keys.add('graph_F' if condition in ('G', 'F') else 'graph_' + condition)
        branches = {key: asyncio.create_task(analysis(config, runner, paper, key)) for key in keys}
        jobs = [protected(runner, paper, 'reports', condition, report(config, runner, paper, condition, branches))
                for condition in conditions or CONDITIONS]
        await asyncio.gather(*jobs)
        # Retrieve failures even for an unrequested branch so they cannot disappear as unhandled futures.
        await asyncio.gather(*branches.values(), return_exceptions=True)
    if stage == 'reports':
        return
    await asyncio.gather(*(protected(runner, paper, 'evaluation', aspect, evaluate(config, runner, paper, aspect)) for aspect in ASPECTS))


async def run(config: Config, pilot: bool, only: list[str] | None, stage: str = 'all',
              conditions: list[str] | None = None) -> None:
    ids = selected(config, pilot, only)
    runner = Runner(config)
    loop, task = asyncio.get_running_loop(), asyncio.current_task()

    def stop() -> None:
        terminate_active()
        if task is not None:
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop)
    try:
        await asyncio.gather(*(paper_run(config, runner, paper, stage, conditions) for paper in ids))
    finally:
        await runner.close()
