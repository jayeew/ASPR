"""Bounded initial Fig5 evaluation: reuse reports, at most 20 new calls."""
from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_study.evaluation import segments
from figure_pipeline.fig5_robustness.materials import original_blocks
from figure_pipeline.fig5_robustness.pipeline import align_evaluation

from . import prompts
from .aggregate import paired, parts_and_metrics
from .materials import packet
from .models import Config, Evaluation
from .pipeline import fixed, public, visible, visible_aliases
from .runtime import Runner


def complete(path: Path) -> bool:
    return path.exists() and all(p['technical_state'] == 'completed' for q in read(path)['questions'] for p in q['parts'])


async def evaluate_initial(config: Config, runner: Runner, task: dict[str, Any]) -> None:
    paper, view = task['paper_id'], task['view']
    aspects = [a for a in ('historical_verification', 'knowledge_position') if not complete(config.output / 'aligned' / view / a / f'{paper}.json')]
    if not aspects:
        return
    questions = [q for q in fixed(config, paper)['questions'] if q['aspect'] in aspects]
    if not questions or {q['aspect'] for q in questions} != set(aspects):
        raise ValueError('Missing fixed questions; refusing evaluation call')
    data = packet(config, paper, view)
    ref = read(config.output / 'validated_reference' / view / f'{paper}.json')
    material = {'visible_packet': visible(data), 'independent_work_aliases': visible_aliases(config, paper, data),
                'reference_questions': questions, 'public_tasks': public(config, paper),
                'candidates': [{'report_id': 'R01', 'report_segments': segments(read(config.output / 'reports' / view / f'{paper}.json')['body']),
                                'conditional_reference': [p for p in ref['parts'] if p['question_id'] in {q['question_id'] for q in questions}]}]}
    key = view + '_' + ''.join(aspects)
    write(config.output / 'evaluation_inputs' / key / f'{paper}.json', material)
    try:
        value = await runner.call(paper, 'evaluation', key, prompts.EVALUATE, material, Evaluation)
        expected_ids = {q['question_id'] for q in questions}
        known_ids = {q['question_id'] for q in fixed(config, paper)['questions']}
        extras = [q for c in value['candidates'] for q in c['questions'] if q['question_id'] not in expected_ids]
        if extras and all(not q['parts'] and q['question_id'] in known_ids for q in extras):
            write(config.output / 'normalization' / key / f'{paper}.json',
                  {'action': 'drop_empty_unrequested_question_containers', 'removed': extras,
                   'raw_response_preserved': True, 'judgments_changed': False})
            value = {**value, 'candidates': [{**c, 'questions': [q for q in c['questions'] if q['question_id'] in expected_ids]} for c in value['candidates']]}
        aligned = align_evaluation(value, material)['candidates'][0]
        for question in aligned['questions']:
            for part in question['parts']:
                consistent = (part['response_type'] == 'target_abstention' and part['explicitly_abstains'] is True or
                              part['response_type'] in ('substantive_answer', 'omission') and part['explicitly_abstains'] is False or
                              part['response_type'] == 'unresolved' and part['explicitly_abstains'] is None)
                if not consistent:
                    part['technical_state'] = 'invalid_response_taxonomy'
        for aspect in aspects:
            ids = {q['question_id'] for q in questions if q['aspect'] == aspect}
            write(config.output / 'aligned' / view / aspect / f'{paper}.json',
                  {**aligned, 'questions': [q for q in aligned['questions'] if q['question_id'] in ids]})
    except (OSError, ValueError, RuntimeError) as exc:
        runner.status(paper, 'evaluation', key, 'unresolved', reason=str(exc))
        print(f'UNRESOLVED {paper} {key}: {exc}', flush=True)


def export(config: Config, tasks: list[dict[str, Any]]) -> None:
    rows, metrics = parts_and_metrics(config, tasks)
    rows = [r for r in rows if r['aspect'] in ('historical_verification', 'knowledge_position')]
    metrics = [r for r in metrics if r['aspect'] in ('historical_verification', 'knowledge_position')]
    write_csv(config.output / 'answer_parts.csv', rows)
    write_csv(config.output / 'paper_metrics.csv', metrics)
    summary = []
    for condition in ('F', 'E50', 'CRITICAL', 'NONCRITICAL'):
        for aspect in ('historical_verification', 'knowledge_position'):
            group = [r for r in metrics if r['condition'] == condition and r['aspect'] == aspect]
            item = {'condition': condition, 'aspect': aspect, 'planned_papers': len(group)}
            for name in ('q_fixed', 'answer_coverage', 'unsupported_burden', 'answered_risk', 'reasonable_abstention_sensitivity', 'unnecessary_abstention_rate'):
                values = [r[name] for r in group if r[name] is not None]
                item[name + '_valid_papers'] = len(values)
                item[name + '_paper_mean'] = sum(values) / len(values) if values else None
            summary.append(item)
    write_csv(config.output / 'condition_summary.csv', summary)
    write_csv(config.output / 'paired_answerable.csv', paired(rows))
    targets = []
    for paper in sorted({t['paper_id'] for t in tasks}):
        eligibility = read(config.output / 'eligibility' / f'{paper}.json')
        target = eligibility['target']
        targets += [{**r, 'eligible': eligibility['eligible']} for r in rows
                    if r['paper_id'] == paper and (r['question_id'], r['part_id']) == (target['question_id'], target['part_id'])]
    write_csv(config.output / 'critical_target_parts.csv', targets)
    reference_rows, conditions = [], []
    for task in tasks:
        paper, view = task['paper_id'], task['view']
        data = packet(config, paper, view)
        base = packet(config, paper, 'F')
        aliases = read(config.output / 'identities' / f'{paper}.json')
        blocks, base_blocks = original_blocks(data), original_blocks(base)
        works = {aliases[p['source_id']] for b in blocks for p in b['provenance']}
        base_works = {aliases[p['source_id']] for b in base_blocks for p in b['provenance']}
        conditions.append({**task, 'independent_works': len(works), 'full_works': len(base_works),
                           'retained_fraction': len(works) / len(base_works) if base_works else None,
                           'removed_works': len(base_works - works), 'original_characters': sum(len(b['text']) for b in blocks),
                           'graph_unchanged': data['graph'] == base['graph'], 'manuscript_unchanged': data['manuscript'] == base['manuscript']})
        questions = {q['question_id'] for q in fixed(config, paper)['questions'] if q['aspect'] in ('historical_verification', 'knowledge_position')}
        reference_rows += [{**task, **p} for p in read(config.output / 'validated_reference' / view / f'{paper}.json')['parts'] if p['question_id'] in questions]
    write_csv(config.output / 'conditions.csv', conditions)
    write_csv(config.output / 'conditional_reference.csv', reference_rows)
    matrix = Counter((r['condition'], r['aspect'], r['answerability'], r.get('response_type', 'technical_missing')) for r in rows)
    write_csv(config.output / 'response_matrix.csv', [{'condition': k[0], 'aspect': k[1], 'answerability': k[2], 'response_type': k[3], 'parts': v} for k, v in sorted(matrix.items())])
    records = [read(p) for p in (config.output / 'logs/calls').glob('*/record.json')]
    write_csv(config.output / 'new_call_costs.csv', [{k: r.get(k) for k in ('paper_id', 'stage', 'method', 'state', 'model', 'seconds', 'usage', 'attempt_id')} for r in records])
    ledger = config.output / 'call_ledger.jsonl'
    calls = len(ledger.read_text().splitlines()) if ledger.exists() else 0
    write(config.output / 'completion.json', {'reports_reused': len(tasks), 'report_aspects': len(metrics),
          'metric_states': dict(Counter(r['state'] for r in metrics)), 'part_states': dict(Counter(r['technical_state'] for r in rows)),
          'new_calls': calls, 'hard_cap': 20, 'new_reports': 0})
    print(read(config.output / 'completion.json'), flush=True)


async def main() -> None:
    config = Config(output=Config().output.parent / 'fig5_initial', ordinary_limit=19, additional_limit=1, call_limit=20, cli_limit=8)
    tasks = read(config.output / 'tasks.json')
    runner = Runner(config)
    await asyncio.gather(*(evaluate_initial(config, runner, t) for t in tasks))
    export(config, tasks)


if __name__ == '__main__':
    asyncio.run(main())
