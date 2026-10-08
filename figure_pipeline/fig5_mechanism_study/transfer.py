from __future__ import annotations

import argparse
import asyncio
import copy
from typing import Any

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_100.models import Analysis, Report
from figure_pipeline.fig4_explanation_study.evaluation import segments
from figure_pipeline.fig5_four_panel.aggregate import classify
from figure_pipeline.fig5_revision import prompts as grading
from figure_pipeline.fig5_revision.models import Evaluation
from figure_pipeline.fig5_revision.runtime import Runner, protected
from figure_pipeline.fig5_robustness import prompts
from figure_pipeline.fig5_robustness.pipeline import align_evaluation

from .diagnostic import OLD, ROOT, config, digest, raw_material

OUT = ROOT / 'outputs/fig5_mechanism_study/transfer_validation'
CONDITIONS = ('original_analysis_original_writer', 'replacement_analysis_original_writer')


def prepare() -> None:
    cohort = read(ROOT / 'outputs/fig5_revision/cohort.json')
    papers = cohort['paper_ids']
    write(OUT / 'protocol.json', {'papers': papers, 'n': len(papers), 'conditions': CONDITIONS,
          'selection': 'All 100 existing papers as one cohort; no candidate-score selection or exclusion for previous exploratory use.',
          'adaptation': 'Replace both branch-analysis models with GPT-5.6 Luna; retain GPT-6.1 Sol writer and original prompts.',
          'basis': '20-paper factorial diagnostic motivated this configuration; assess it across the unified100 cohort without claiming an untouched test set.',
          'primary_endpoint': 'Paired paper-level fixed-denominator grounded correct coverage.',
          'secondary_endpoints': ['confirmed unsupported assertions', 'reasonable abstention', 'resource consumption'],
          'hypothesis_direction': 'Coverage may be close to the original configuration; no required outcome.',
          'noninferiority_margin': None, 'call_cap': 400, 'ordinary_limit': 350, 'additional_limit': 50,
          'writer_prompt_hash': digest(prompts.WRITER), 'generation_effort': 'medium',
          'sample_limit': 'Unified existing cohort, not an untouched independent test set.'})
    for paper in papers:
        original_path = OLD / 'baseline/report_inputs/F' / f'{paper}.json'
        report_path = OLD / 'baseline/reports/F' / f'{paper}.json'
        if not original_path.exists():
            original_path = OUT.parent / 'adapter_development/inputs' / f'{paper}.json'
            report_path = OUT.parent / 'adapter_development/reports/original_model_free_report' / f'{paper}.json'
        original = read(original_path)
        write(OUT / 'original_inputs' / f'{paper}.json', original)
        write(OUT / 'reports' / CONDITIONS[0] / f'{paper}.json', read(report_path))
    from .cohort100 import extend_models
    extend_models(cohort)


async def generate(runner: Runner, paper: str) -> None:
    if (OUT / 'reports' / CONDITIONS[1] / f'{paper}.json').exists():
        return
    original = read(OUT / 'original_inputs' / f'{paper}.json')
    async def branch(name: str) -> dict[str, Any]:
        old_name = 'gear' if name == 'gear' else 'graph_F'
        local = OUT / 'analysis_inputs' / name / f'{paper}.json'
        data = read(local if local.exists() else OLD / 'baseline/analysis_inputs' / old_name / f'{paper}.json')
        write(OUT / 'analysis_inputs' / name / f'{paper}.json', data)
        return await runner.call(paper, 'analysis', name, prompts.GEAR if name == 'gear' else prompts.GRAPH,
                                  data, Analysis, model='gpt-5.6-luna', effort='medium')
    gear, graph = await asyncio.gather(branch('gear'), branch('graph'))
    material = {**copy.deepcopy(original), 'scientific_evidence_analysis': gear, 'knowledge_graph_analysis': graph}
    if raw_material(material) != raw_material(original):
        raise ValueError('Transfer changed raw evidence/public tasks/native graph')
    write(OUT / 'report_inputs' / CONDITIONS[1] / f'{paper}.json', material)
    await runner.call(paper, 'reports', CONDITIONS[1], prompts.WRITER, material, Report,
                       model='gpt-6.1-sol', effort='medium')


async def evaluate(runner: Runner, paper: str) -> None:
    if (OUT / 'aligned' / f'{paper}.json').exists():
        return
    import random
    original = read(OUT / 'original_inputs' / f'{paper}.json')
    fixed = read(OLD / 'baseline/reference/papers' / f'{paper}.json')
    order = list(CONDITIONS)
    random.Random(f'20261004:{paper}:transfer').shuffle(order)
    mapping = {f'R{i + 1:02d}': c for i, c in enumerate(order)}
    material = {'visible_packet': {k: v for k, v in raw_material(original).items() if k != 'public_tasks'},
                'public_tasks': original['public_tasks'], 'reference_questions': fixed['questions'],
                'candidates': [{'report_id': key, 'report_segments': segments(read(OUT / 'reports' / c / f'{paper}.json')['body'])}
                               for key, c in mapping.items()]}
    prompt = grading.EVALUATE + '\nBoth candidates share the full visible_packet. reference_questions provides the fallible full-material reference; no reduced-condition reference is needed. Judge independently of report length or incidental wording.\n'
    write(OUT / 'evaluation_inputs' / f'{paper}.json', material)
    write(OUT / 'mapping' / f'{paper}.json', mapping)
    value = await runner.call(paper, 'evaluation', 'paired', prompt, material, Evaluation)
    aligned = align_evaluation(value, material)
    for candidate in aligned['candidates']:
        for question in candidate['questions']:
            for part in question['parts']:
                expected = {'substantive_answer': False, 'omission': False,
                            'target_abstention': True, 'unresolved': None}[part['response_type']]
                if part['explicitly_abstains'] is not expected:
                    part['technical_state'] = 'invalid_response_taxonomy'
    write(OUT / 'aligned' / f'{paper}.json', aligned)


async def run(selected_papers: list[str] | None = None) -> None:
    cfg = config().model_copy(update={'output': OUT, 'cli_limit': 16, 'ordinary_limit': 350,
                                     'additional_limit': 50, 'call_limit': 400})
    runner = Runner(cfg)
    async def one(paper: str) -> None:
        restrictions = OUT / 'nonretryable_failures.json'
        if restrictions.exists() and paper in {r['paper_id'] for r in read(restrictions)}:
            runner.status(paper, 'study', 'all', 'unresolved', reason='Provider content restriction; no further calls for this paper. Retained in the 100-paper denominator.')
            return
        await protected(runner, paper, 'generation', 'transfer', generate(runner, paper))
        if (OUT / 'reports' / CONDITIONS[1] / f'{paper}.json').exists():
            await protected(runner, paper, 'grading', 'paired', evaluate(runner, paper))
    papers = selected_papers if selected_papers is not None else read(OUT / 'protocol.json')['papers']
    await asyncio.gather(*(one(p) for p in papers))
    aggregate()


def aggregate() -> None:
    rows, metrics = [], []
    for paper in read(OUT / 'protocol.json')['papers']:
        path = OUT / 'aligned' / f'{paper}.json'
        fixed = read(OLD / 'baseline/reference/papers' / path.name)
        aspects = {q['question_id']: q['aspect'] for q in fixed['questions']}
        observed = {}
        if path.exists():
            mapping = read(OUT / 'mapping' / path.name)
            observed = {(mapping[c['report_id']], q['question_id'], p['part_id']): p
                        for c in read(path)['candidates'] for q in c['questions'] for p in q['parts']}
        for condition in CONDITIONS:
            current = []
            for question in fixed['questions']:
                if not question['applicable']:
                    continue
                for expected in question['answer_parts']:
                    key = condition, question['question_id'], expected['part_id']
                    part = observed.get(key, {'technical_state': 'missing'})
                    current.append({**part, **classify(part, 'answerable'), 'paper_id': paper,
                                    'condition': condition, 'question_id': key[1], 'part_id': key[2],
                                    'aspect': question['aspect']})
            rows.extend(current)
            for aspect in ['all', *sorted(set(aspects.values()))]:
                selected = [r for r in current if aspect == 'all' or r['aspect'] == aspect]
                complete = bool(selected) and all(r['technical_state'] == 'completed' for r in selected)
                item = {'paper_id': paper, 'condition': condition, 'aspect': aspect,
                        'fixed_denominator': len(selected), 'complete': complete}
                for name in ('grounded_answer', 'confirmed_unsupported', 'reasonable_abstention'):
                    item[name] = sum(r[name] == 1 for r in selected) / len(selected) if complete else None
                metrics.append(item)
    for name, data in [('answer_parts', rows), ('paper_metrics', metrics)]:
        write(OUT / f'{name}.json', data)
        write_csv(OUT / f'{name}.csv', data)
    planned = len(read(OUT / 'protocol.json')['papers'])
    print(f'Model comparison: {len(list((OUT / "aligned").glob("*.json")))}/{planned} paired papers.', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'aggregate'])
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'run':
        asyncio.run(run())
    else:
        aggregate()


if __name__ == '__main__':
    main()
