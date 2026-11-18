from __future__ import annotations

import argparse
import asyncio
from typing import Any

from figure_pipeline.fig3_revision.models import Record
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_100.models import Analysis, Report
from figure_pipeline.fig4_explanation_study.evaluation import segments
from figure_pipeline.fig5_revision import prompts as evaluation_prompts
from figure_pipeline.fig5_revision.models import Evaluation
from figure_pipeline.fig5_revision.runtime import Runner, protected
from figure_pipeline.fig5_robustness import prompts
from figure_pipeline.fig5_robustness.materials import original_blocks
from figure_pipeline.fig5_robustness.pipeline import align_evaluation

from .diagnostic import MODELS, OLD, ROOT, config, digest

OUT = ROOT / 'outputs/fig5_mechanism_study/adapter_development'


class TaskAnswer(Record):
    question_id: str
    concrete_observation: str
    supporting_evidence_and_location: str
    scientific_interpretation: str
    limitations_or_missing_evidence: str


class TaskReport(Record):
    answers: list[TaskAnswer]
    cited_source_ids: list[str]


INSTRUCTION = prompts.WRITER + '''
OUTPUT ORGANIZATION: Instead of a free-form body, return one structured answer for EVERY public
question_id, exactly once. Use only the public task to determine its scientific requirements; no private
reference is available. Each answer separates the concrete observation, its actual evidence/location,
scientific interpretation, and relevant limitations or missing evidence. Answer the actual requested
relation/construct with concrete entities and comparisons, not a different easier proxy. A generic
caution does not replace a supported observation. Do not invent evidence to fill a field: explicitly
state which requested judgment cannot be supported and why when necessary. Preserve source and native
graph identifiers exactly. There is no word target or preferred scientific conclusion. This is one
generation call, not a second review or access to hidden answers. The report is assembled from these
fields without another model or factual editing. Return the supplied schema, not a body field.
'''


def compose(value: dict[str, Any], public_tasks: dict[str, Any]) -> dict[str, Any]:
    expected = [q['question_id'] for q in public_tasks['questions']]
    actual = [a['question_id'] for a in value['answers']]
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError('Structured writer omitted or duplicated a public task')
    lookup = {a['question_id']: a for a in value['answers']}
    names = [('concrete_observation', '具体观察'), ('supporting_evidence_and_location', '证据与位置'),
             ('scientific_interpretation', '科学解释'), ('limitations_or_missing_evidence', '适用边界与证据缺口')]
    text = []
    for question in public_tasks['questions']:
        key = question['question_id']
        text.append(f"## {key}：{question['question']}")
        text.extend(f'**{label}**\n\n{lookup[key][field]}' for field, label in names)
    # Match the existing report schema, preserving any required non-body fields.
    return {'body': '\n\n'.join(text), 'cited_source_ids': value['cited_source_ids']}


def prepare() -> None:
    cohort = read(ROOT / 'outputs/fig5_revision/cohort.json')
    papers = cohort['development']
    write(OUT / 'protocol.json', {'papers': papers, 'role': 'Previously declared development papers only',
          'controlled_stage': 'Writer only; both models receive identical original-model branch analyses.',
          'conditions': ['original_model_free_report', 'replacement_model_free_report',
                         'original_model_structured_report', 'replacement_model_structured_report'],
          'structured_prompt_hash': digest(INSTRUCTION), 'effort': 'medium',
          'selection_rule': 'Assess coverage, factual errors and missing task IDs; retain all outcomes.',
          'hard_call_cap': 40, 'private_reference_visible_to_writer': False})
    for paper in papers:
        source = OLD / 'baseline/report_inputs/F' / f'{paper}.json'
        if not source.exists():
            write(OUT / 'missing_prerequisites' / f'{paper}.json',
                  {'reason': 'Original development graph analysis/report absent; preserve paper and complete prerequisites separately.'})
            continue
        material = read(source)
        write(OUT / 'inputs' / f'{paper}.json', material)
        write(OUT / 'reports/original_model_free_report' / f'{paper}.json', read(OLD / 'baseline/reports/F' / f'{paper}.json'))


async def prepare_missing() -> None:
    cfg = config().model_copy(update={'output': OUT / 'prerequisite_calls', 'cli_limit': 1,
                                     'ordinary_limit': 2, 'additional_limit': 2, 'call_limit': 4})
    runner = Runner(cfg)
    for path in (OUT / 'missing_prerequisites').glob('*.json'):
        paper = path.stem
        if (OUT / 'inputs' / path.name).exists():
            continue
        data = read(OLD / 'inputs/F' / path.name)
        public = read(OLD / 'baseline/public_tasks' / path.name)
        graph_input = {'manuscript': data['manuscript'], 'public_tasks': public,
                       'native_graph': data['graph'], 'graph_metric_definitions': prompts.GRAPH_DEFINITIONS,
                       'original_historical_passages': data['graph_evidence']['evidence_blocks']}
        graph = await runner.call(paper, 'analysis', 'graph', prompts.GRAPH, graph_input,
                                   Analysis, model=MODELS['original'], effort='medium')
        material = {'manuscript': data['manuscript'], 'graph': data['graph'],
                    'original_evidence': original_blocks(data), 'public_tasks': public,
                    'scientific_evidence_analysis': read(OLD / 'baseline/analysis/gear' / path.name),
                    'knowledge_graph_analysis': graph, 'graph_metric_definitions': prompts.GRAPH_DEFINITIONS}
        value = await runner.call(paper, 'reports', 'original_model_free_report', prompts.WRITER,
                                  material, Report, model=MODELS['original'], effort='medium')
        write(OUT / 'inputs' / path.name, material)
        write(OUT / 'reports/original_model_free_report' / path.name, value)


async def generate(runner: Runner, paper: str, model: str, structured: bool) -> None:
    condition = model + '_model_' + ('structured_report' if structured else 'free_report')
    target = OUT / 'reports' / condition / f'{paper}.json'
    if target.exists():
        return
    material = read(OUT / 'inputs' / f'{paper}.json')
    value = await runner.call(paper, 'structured' if structured else 'reports', condition,
                              INSTRUCTION if structured else prompts.WRITER, material,
                              TaskReport if structured else Report, model=MODELS[model], effort='medium')
    report = compose(value, material['public_tasks']) if structured else value
    write(target, Report.model_validate(report).model_dump())


async def evaluate(runner: Runner, paper: str) -> None:
    import random
    material = read(OUT / 'inputs' / f'{paper}.json')
    fixed = read(OLD / 'baseline/reference/papers' / f'{paper}.json')
    names = read(OUT / 'protocol.json')['conditions']
    random.Random(f'20261004:{paper}:adapter').shuffle(names)
    mapping = {f'R{i + 1:02d}': c for i, c in enumerate(names)}
    data = {'visible_packet': {k: v for k, v in material.items() if k not in
                               ('scientific_evidence_analysis', 'knowledge_graph_analysis', 'public_tasks')},
            'public_tasks': material['public_tasks'], 'reference_questions': fixed['questions'],
            'candidates': [{'report_id': key, 'report_segments': segments(read(OUT / 'reports' / c / f'{paper}.json')['body'])}
                           for key, c in mapping.items()]}
    write(OUT / 'evaluation_inputs' / f'{paper}.json', data)
    write(OUT / 'mapping' / f'{paper}.json', mapping)
    prompt = evaluation_prompts.EVALUATE + '\nAll candidates share the full visible_packet. Use reference_questions as the fallible full-material reference. No separate reduced-condition reference is needed.\n'
    value = await runner.call(paper, 'evaluation', 'all', prompt, data, Evaluation)
    aligned = align_evaluation(value, data)
    write(OUT / 'aligned' / f'{paper}.json', aligned)


async def run() -> None:
    cfg = config().model_copy(update={'output': OUT, 'cli_limit': 5, 'ordinary_limit': 30,
                                     'additional_limit': 6, 'call_limit': 36})
    runner = Runner(cfg)
    async def one(paper: str) -> None:
        await asyncio.gather(*(protected(runner, paper, 'generation', f'{model}_{structured}',
                                         generate(runner, paper, model, structured))
                               for model in MODELS for structured in (False, True)))
        if all((OUT / 'reports' / c / f'{paper}.json').exists() for c in read(OUT / 'protocol.json')['conditions']):
            await protected(runner, paper, 'grading', 'all', evaluate(runner, paper))
    await asyncio.gather(*(one(p) for p in read(OUT / 'protocol.json')['papers']))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'prerequisites', 'run'])
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'prerequisites':
        asyncio.run(prepare_missing())
    else:
        asyncio.run(run())


if __name__ == '__main__':
    main()
