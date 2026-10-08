from __future__ import annotations

import argparse
import asyncio
import copy
import fcntl
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_100.models import Report
from figure_pipeline.fig4_explanation_study.evaluation import segments
from figure_pipeline.fig5_four_panel.aggregate import classify, interval
from figure_pipeline.fig5_four_panel.evaluate import PROMPT, batches
from figure_pipeline.fig5_revision.models import Config, Evaluation
from figure_pipeline.fig5_revision.runtime import Runner, protected
from figure_pipeline.fig5_robustness import prompts
from figure_pipeline.fig5_robustness.pipeline import align_evaluation

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'outputs/fig5_robustness'
CORRECTED = ROOT / 'outputs/fig5_four_panel'
OUT = ROOT / 'outputs/fig5_mechanism_study/diagnostic'
MODELS = {'original': 'gpt-6.1-sol', 'replacement': 'gpt-5.6-luna'}
CONDITIONS = {
    'original_analysis_original_writer': ('original', 'original'),
    'original_analysis_replacement_writer': ('original', 'replacement'),
    'replacement_analysis_original_writer': ('replacement', 'original'),
    'replacement_analysis_replacement_writer': ('replacement', 'replacement'),
}


def config() -> Config:
    return Config(output=OUT, cli_limit=12, ordinary_limit=160,
                  additional_limit=40, call_limit=200, seed=20261004)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def writer_material(paper: str, analysis: str) -> dict[str, Any]:
    folder = OLD / ('baseline/report_inputs/F' if analysis == 'original' else 'report_inputs/LUNA')
    return read(folder / f'{paper}.json')


def raw_material(data: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in data.items()
            if k not in ('scientific_evidence_analysis', 'knowledge_graph_analysis')}


def prepare() -> None:
    papers = read(OLD / 'cohort.json')['selected']
    audit = []
    for paper in papers:
        original, replacement = (writer_material(paper, m) for m in MODELS)
        if raw_material(original) != raw_material(replacement):
            raise ValueError(f'Underlying writer materials differ: {paper}')
        audit.append({'paper_id': paper, 'raw_material_hash': digest(raw_material(original)),
                      'original_analysis_hash': digest(original),
                      'replacement_analysis_hash': digest(replacement), 'raw_material_equal': True})
        for condition, (analysis, writer) in CONDITIONS.items():
            write(OUT / 'report_inputs' / condition / f'{paper}.json', writer_material(paper, analysis))
            if analysis == writer:
                source = OLD / 'reports' / ('F' if writer == 'original' else 'LUNA') / f'{paper}.json'
                write(OUT / 'reports' / condition / f'{paper}.json', Report.model_validate(read(source)).model_dump())
    protocol = {'version': 1, 'papers': papers, 'conditions': CONDITIONS, 'models': MODELS,
                'prompt_sha256': digest(prompts.WRITER), 'generation_effort': 'medium',
                'new_generation_calls': 40, 'hard_call_cap': 200,
                'purpose': 'Stage localization; same raw evidence and same original writer prompt.',
                'sample_role': 'Previously examined diagnostic papers, not an untouched test set.',
                'evaluation': 'All four configurations jointly regraded anonymously; original results retained.',
                'inference': 'Descriptive paired estimates; no post-hoc noninferiority declaration.'}
    established = OUT / 'protocol.json'
    if established.exists() and read(established) != protocol:
        raise ValueError('Established diagnostic protocol changed')
    write(established, protocol)
    write(OUT / 'input_audit.json', audit)
    write_csv(OUT / 'input_audit.csv', audit)


def evaluation_packet(paper: str) -> tuple[dict[str, Any], dict[str, str]]:
    data = copy.deepcopy(read(CORRECTED / 'evaluation_inputs' / f'{paper}.json'))
    old_mapping = read(CORRECTED / 'mapping' / f'{paper}.json')
    full_id = next(c['material_packet_id'] for c in data['candidates'] if old_mapping[c['report_id']] == 'F')
    order = list(CONDITIONS)
    random.Random(f'20261004:{paper}:diagnostic').shuffle(order)
    mapping = {f'R{i + 1:02d}': name for i, name in enumerate(order)}
    data['candidates'] = [{'report_id': key, 'material_packet_id': full_id,
                           'report_segments': segments(read(OUT / 'reports' / value / f'{paper}.json')['body'])}
                          for key, value in mapping.items()]
    graph_id = data['visible_materials'][full_id]['graph_key']
    data['visible_materials'] = {full_id: data['visible_materials'][full_id]}
    data['conditional_references'] = {full_id: data['conditional_references'][full_id]}
    data['native_graphs'] = {graph_id: data['native_graphs'][graph_id]}
    return data, mapping


async def generate(runner: Runner, paper: str, condition: str) -> None:
    _, writer = CONDITIONS[condition]
    data = read(OUT / 'report_inputs' / condition / f'{paper}.json')
    value = await runner.call(paper, 'reports', condition, prompts.WRITER, data, Report,
                              model=MODELS[writer], effort='medium')
    (OUT / 'reports' / condition / f'{paper}.md').write_text(value['body'])


async def evaluate(runner: Runner, paper: str) -> None:
    data, mapping = evaluation_packet(paper)
    write(OUT / 'mapping' / f'{paper}.json', mapping)
    packets = batches(runner, data)
    combined = []
    for i, packet in enumerate(packets):
        key = f'all_batch_{i + 1}'
        write(OUT / 'evaluation_inputs' / key / f'{paper}.json', packet)
        result = await runner.call(paper, 'evaluation', key, PROMPT, packet, Evaluation)
        aligned = align_evaluation(result, packet)
        for candidate in aligned['candidates']:
            for question in candidate['questions']:
                for part in question['parts']:
                    expected = {'substantive_answer': False, 'omission': False,
                                'target_abstention': True, 'unresolved': None}[part['response_type']]
                    if part['explicitly_abstains'] is not expected:
                        part['technical_state'] = 'invalid_response_taxonomy'
        combined.extend(aligned['candidates'])
    write(OUT / 'aligned' / f'{paper}.json', {'candidates': combined})


async def run(stage: str) -> None:
    runner = Runner(config())
    async def one(paper: str) -> None:
        if stage in ('all', 'generate'):
            await asyncio.gather(*(protected(runner, paper, 'generation', condition,
                                            generate(runner, paper, condition))
                                   for condition in CONDITIONS))
        if stage in ('all', 'evaluate') and all((OUT / 'reports' / c / f'{paper}.json').exists() for c in CONDITIONS):
            await protected(runner, paper, 'grading', 'all', evaluate(runner, paper))
    await asyncio.gather(*(one(p) for p in read(OUT / 'protocol.json')['papers']))
    aggregate()


def aggregate() -> None:
    parts, metrics = [], []
    for paper in read(OUT / 'protocol.json')['papers']:
        path = OUT / 'aligned' / f'{paper}.json'
        if not path.exists():
            continue
        mapping = read(OUT / 'mapping' / f'{paper}.json')
        reference = read(OLD / 'baseline/reference/papers' / f'{paper}.json')
        aspects = {q['question_id']: q['aspect'] for q in reference['questions']}
        for candidate in read(path)['candidates']:
            condition = mapping[candidate['report_id']]
            current = []
            for question in candidate['questions']:
                for part in question['parts']:
                    row = {**part, **classify(part, 'answerable'), 'paper_id': paper,
                           'condition': condition, 'question_id': question['question_id'],
                           'aspect': aspects[question['question_id']]}
                    current.append(row)
            parts.extend(current)
            for aspect in ['all', *sorted(set(aspects.values()))]:
                selected = [r for r in current if aspect == 'all' or r['aspect'] == aspect]
                complete = bool(selected) and all(r['technical_state'] == 'completed' for r in selected)
                row = {'paper_id': paper, 'condition': condition, 'aspect': aspect,
                       'fixed_denominator': len(selected), 'complete': complete}
                for name in ('grounded_answer', 'confirmed_unsupported', 'reasonable_abstention'):
                    row[name] = sum(r[name] == 1 for r in selected) / len(selected) if complete else None
                row['incomplete'] = sum(r['behavior'] == 'incomplete' for r in selected) / len(selected) if complete else None
                metrics.append(row)
    summaries = []
    for condition in CONDITIONS:
        for aspect in sorted({r['aspect'] for r in metrics}):
            for name in ('grounded_answer', 'confirmed_unsupported', 'reasonable_abstention', 'incomplete'):
                values = [r[name] for r in metrics if r['condition'] == condition and r['aspect'] == aspect and r['complete']]
                summaries.append({'condition': condition, 'aspect': aspect, 'metric': name, **interval(values)})
    for name, rows in [('answer_parts', parts), ('paper_metrics', metrics), ('summary', summaries)]:
        write(OUT / f'{name}.json', rows)
        write_csv(OUT / f'{name}.csv', rows)
    print(json.dumps({'evaluated_papers': len({r['paper_id'] for r in parts}),
                      'available_reports': len(list((OUT / 'reports').glob('*/*.json')))}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'aggregate'])
    parser.add_argument('--stage', choices=['all', 'generate', 'evaluate'], default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'writer.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'prepare':
            prepare()
        elif args.command == 'run':
            asyncio.run(run(args.stage))
        else:
            aggregate()


if __name__ == '__main__':
    main()
