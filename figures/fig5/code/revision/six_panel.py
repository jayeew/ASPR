"""Minimal six-panel data extension; nine evaluation calls, no generation."""
from __future__ import annotations

import asyncio
import shutil
from typing import Any

from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_study.evaluation import segments
from figure_pipeline.fig5_robustness.pipeline import align_evaluation

from . import prompts
from .materials import packet
from .models import Config, Evaluation
from .pipeline import fixed, public, validate_reference, visible, visible_aliases
from .runtime import Runner

ASPECTS = ('historical_verification', 'joint_contribution')


def prepare(config: Config) -> list[dict[str, Any]]:
    source = Config().output
    cohort = read(source / 'cohort.json')
    papers = [p for p in cohort['exploration'] if all((source / 'reports' / v / f'{p}.json').exists() for v in ('F', 'K5', 'LUNA'))][:3]
    if len(papers) != 3:
        raise ValueError('Three complete report triples required')
    tasks, manifest = [], []
    for paper in papers:
        paths = [f'{folder}/{paper}.json' for folder in ('baseline/reference/papers', 'baseline/public_tasks', 'identities')]
        for condition in ('F', 'K5', 'LUNA'):
            view = 'F' if condition == 'LUNA' else condition
            tasks.append({'paper_id': paper, 'condition': condition, 'view': view, 'split': 'exploration_initial', 'kind': 'reused_multistage'})
            paths += [f'reports/{condition}/{paper}.json', f'inputs/{view}/{paper}.json', f'validated_reference/{view}/{paper}.json']
        for rel in sorted(set(paths)):
            target = config.output / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source / rel, target)
            manifest.append({'path': rel, 'source': str(source / rel)})
    write(config.output / 'reuse_manifest.json', manifest)
    write(config.output / 'tasks.json', tasks)
    write(config.output / 'scope.json', {'papers': papers, 'aspects': ASPECTS, 'conditions': ['F', 'K5', 'LUNA'], 'new_call_cap_including_failures': 9, 'new_reports': 0, 'selection': 'First three original exploration roster entries with all three reports, independent of outcomes'})
    return tasks


async def evaluate(config: Config, runner: Runner, task: dict[str, Any]) -> None:
    paper, condition, view = task['paper_id'], task['condition'], task['view']
    questions = [q for q in fixed(config, paper)['questions'] if q['aspect'] in ASPECTS]
    if {q['aspect'] for q in questions} != set(ASPECTS):
        raise ValueError('Missing canonical aspect questions')
    data = packet(config, paper, view)
    ref = read(config.output / 'validated_reference' / view / f'{paper}.json')
    validate_reference(ref, fixed(config, paper), data)
    ids = {q['question_id'] for q in questions}
    material = {'visible_packet': visible(data), 'independent_work_aliases': visible_aliases(config, paper, data),
                'reference_questions': questions, 'public_tasks': public(config, paper),
                'candidates': [{'report_id': 'R01', 'report_segments': segments(read(config.output / 'reports' / condition / f'{paper}.json')['body']),
                                'conditional_reference': [p for p in ref['parts'] if p['question_id'] in ids]}]}
    key = condition + '_HJ'
    write(config.output / 'evaluation_inputs' / key / f'{paper}.json', material)
    prompt = prompts.EVALUATE + '\nEvaluate ONLY question IDs in reference_questions. Do not add empty containers for other public tasks.'
    try:
        value = await runner.call(paper, 'evaluation', key, prompt, material, Evaluation)
        extras = [q for c in value['candidates'] for q in c['questions'] if q['question_id'] not in ids]
        known = {q['question_id'] for q in fixed(config, paper)['questions']}
        if extras and all(not q['parts'] and q['question_id'] in known for q in extras):
            write(config.output / 'normalization' / key / f'{paper}.json', {'removed_empty_containers': extras})
            value = {**value, 'candidates': [{**c, 'questions': [q for q in c['questions'] if q['question_id'] in ids]} for c in value['candidates']]}
        aligned = align_evaluation(value, material)['candidates'][0]
        for q in aligned['questions']:
            for p in q['parts']:
                consistent = (p['response_type'] == 'target_abstention' and p['explicitly_abstains'] is True or
                              p['response_type'] in ('substantive_answer', 'omission') and p['explicitly_abstains'] is False or
                              p['response_type'] == 'unresolved' and p['explicitly_abstains'] is None)
                if not consistent:
                    p['technical_state'] = 'invalid_response_taxonomy'
        for aspect in ASPECTS:
            selected = {q['question_id'] for q in questions if q['aspect'] == aspect}
            write(config.output / 'aligned' / condition / aspect / f'{paper}.json',
                  {**aligned, 'questions': [q for q in aligned['questions'] if q['question_id'] in selected]})
    except (OSError, ValueError, RuntimeError) as exc:
        runner.status(paper, 'evaluation', key, 'unresolved', reason=str(exc))
        print(f'UNRESOLVED {paper} {key}: {exc}', flush=True)


async def main() -> None:
    config = Config(output=Config().output.parent / 'fig5_six_panel/panel_e', ordinary_limit=9, additional_limit=0, call_limit=9, cli_limit=9)
    tasks = prepare(config)
    # Validate all inputs before any paid dispatch.
    for task in tasks:
        paper, view = task['paper_id'], task['view']
        validate_reference(read(config.output / 'validated_reference' / view / f'{paper}.json'), fixed(config, paper), packet(config, paper, view))
    runner = Runner(config)
    await asyncio.gather(*(evaluate(config, runner, t) for t in tasks))


if __name__ == '__main__':
    asyncio.run(main())
