"""One focused five-paper development revision, charged to the shared extra-call budget."""
from __future__ import annotations

import argparse
import asyncio
import shutil
from typing import Any

from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_study.input_information import components
from figure_pipeline.fig3_revision.models import Record

from . import prompts
from .models import Config, PublicQuestion
from .runtime import Runner


class RevisedQuestions(Record):
    questions: list[PublicQuestion]
    measurement_notes: list[str]


def configuration() -> Config:
    original = Config().output
    return Config(output=original / 'pilot_refinement_v2', ledger_root=original,
                  supplemental_round=True, focused_refinement=True)


def topology(graph: dict[str, Any]) -> dict[str, Any]:
    cards = graph['cards']
    nodes = {n['claim_id'] for c in cards for n in c['neighbors']}
    targets = {c['claim']['claim_id'] for c in cards}
    local = {tuple(sorted(e)) for c in cards for e in c['neighbor_edges']}
    union = {tuple(sorted(e)) for e in graph['joint']['historical_edges']}
    insertions = {tuple(sorted((c['claim']['claim_id'], n['claim_id']))) for c in cards for n in c['neighbors']}
    def groups(edges: set[tuple[str, str]]) -> list[list[str]]:
        return sorted([sorted(g) for g in components(nodes | targets, edges | insertions)])
    neighbors = {n['claim_id']: n for c in cards for n in c['neighbors']}
    edge_examples = [{'edge': list(e), 'endpoints': [neighbors[n] for n in e],
                      'attached_target_claims': [[c['claim']['claim_id'] for c in cards
                                                 if any(n['claim_id'] == node for n in c['neighbors'])] for node in e]}
                     for e in sorted(union-local)]
    metrics = []
    for c in cards:
        values = {m['name']: m['value'] for m in c['metrics']}
        metrics.append({'claim': c['claim'], 'metrics': values,
                        'distinct_known_communities': len({n['community_id'] for n in c['neighbors'] if n['community_id'] is not None})})
    return {'joint_only_historical_edges': edge_examples, 'components_from_single_cards_and_targets': groups(local),
            'components_from_full_union_and_targets': groups(union), 'structural_profiles': metrics,
            'note': 'Derived from existing facts before new reports; not a winner-based sample selection.'}


def prepare(config: Config) -> None:
    original = config.ledger_root
    assert original is not None
    for name in ('cohort.json', 'papers.json', 'conditions.json', 'input_information.csv',
                 'input_information_details.json', 'input_inventory.csv', 'reuse_inventory.csv'):
        config.output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / name, config.output / name)
    for paper in read(original / 'cohort.json')['pilot_ids']:
        for directory in ('inputs/papers', 'inputs/reference'):
            write(config.output / directory / f'{paper}.json', read(original / directory / f'{paper}.json'))
        data = read(config.output / 'inputs/papers' / f'{paper}.json')
        write(config.output / 'factual_opportunities' / f'{paper}.json', topology(data['graph']))
    restrictions = read(original / 'known_restrictions.json')
    for path in (original / 'logs/calls').glob('*/record.json'):
        record = read(path)
        if record['state'] == 'failed' and 'limited access' in record.get('cli_error', ''):
            restrictions.append(dict(paper_id=record['paper_id'], stage=record['stage'], condition=record['method'],
                                     category='provider_content_restriction', detail='Previously restricted task; do not retry by changing task wording.'))
    write(config.output / 'known_restrictions.json', restrictions)
    write(config.output / 'protocol.json', dict(version='focused_J_S_v2', scope='original_five_only',
          unchanged_aspects=['historical_verification','knowledge_position','citation_contact'],
          changed_aspects=['joint_contribution','structural_resolution'], input_masks_unchanged=True,
          reconstruction_allowed=True, generation_model=config.model, generation_effort=config.generation_effort,
          evaluation_effort=config.evaluation_effort, ordinary_expected_calls=83,
          all_requests_charged_to_additional_budget=True, shared_total_limit=1980, additional_limit=180,
          remaining_95_requires_review=True, development_not_confirmatory=True))
    print('Prepared focused five-paper inputs without model calls.', flush=True)


async def refined_reference(config: Config, runner: Runner, paper: str) -> dict[str, Any]:
    original = config.ledger_root
    assert original is not None
    previous = read(original / 'reference/papers' / f'{paper}.json')
    existing = config.output / 'reference/papers' / f'{paper}.json'
    if existing.exists():
        result = read(existing)
        publish_reference(config, paper, result)
        return result
    material = read(config.output / 'inputs/reference' / f'{paper}.json')
    material.update(factual_opportunities=read(config.output / 'factual_opportunities' / f'{paper}.json'),
                    graph_metric_definitions=prompts.GRAPH_DEFINITIONS)
    revised = await runner.call(paper, 'reference_revision', 'papers', prompts.REFINED_REFERENCE,
                                material, RevisedQuestions, config.evaluation_effort)
    if [q['question_id'] for q in revised['questions']] != ['J1','J2','S1','S2']:
        raise ValueError('Revised reference must contain exactly the four authorized question slots')
    for q in revised['questions']:
        expected_aspect = 'joint_contribution' if q['question_id'].startswith('J') else 'structural_resolution'
        if q['aspect'] != expected_aspect or (q['applicable'] and len(q['answer_parts']) != 2):
            raise ValueError('Revised scientific part/aspect identity mismatch')
        if not q['applicable'] and q['answer_parts']:
            raise ValueError('Inapplicable task contains a scored part')
    replacements = {q['question_id']: q for q in revised['questions']}
    result = dict(previous, questions=[replacements.get(q['question_id'], q) for q in previous['questions']])
    write(existing, result)
    publish_reference(config, paper, result)
    return result


def publish_reference(config: Config, paper: str, result: dict[str, Any]) -> None:
    write(config.output / 'public_tasks' / f'{paper}.json', {'questions': [
        {k:q[k] for k in ('question_id','aspect','question')} for q in result['questions']]})
    write(config.output / 'screening' / f'{paper}.json', dict(graph_useful=result['graph_useful'], reason=result['eligibility_reason']))


async def review_joint(config: Config) -> None:
    from .pipeline import evaluate, protected
    config = config.model_copy(update={'review_joint': True})
    ids = read(config.output / 'cohort.json')['pilot_ids']
    for paper in ids:
        source = config.output / 'aligned_evaluation/joint_contribution' / f'{paper}.json'
        backup = config.output / 'evaluation_initial/joint_contribution' / f'{paper}.json'
        if not backup.exists():
            write(backup, read(source))
    runner = Runner(config)
    try:
        await asyncio.gather(*(protected(runner, paper, 'evaluation_review', 'joint_contribution',
                                        evaluate(config, runner, paper, 'joint_contribution')) for paper in ids))
    finally:
        await runner.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','reference','run','review-joint','aggregate','render','package','status'])
    args = parser.parse_args()
    config = configuration()
    prompts.GRAPH_DEFINITIONS += '\n' + prompts.STRUCTURE_DEFINITIONS
    if args.command == 'prepare':
        prepare(config)
    elif args.command == 'review-joint':
        asyncio.run(review_joint(config))
    elif args.command in ('reference','run'):
        from .pipeline import run
        asyncio.run(run(config, True, None, 'reference' if args.command == 'reference' else 'all'))
    elif args.command == 'aggregate':
        from .aggregate import aggregate
        aggregate(config)
    elif args.command == 'render':
        from .render import render
        render(config)
    elif args.command == 'package':
        from .handoff import package
        print(package(config))
    else:
        from collections import Counter
        records = [read(p) for p in (config.output/'logs/calls').glob('*/record.json')]
        print(Counter((r['stage'],r['state']) for r in records))


if __name__ == '__main__':
    main()
