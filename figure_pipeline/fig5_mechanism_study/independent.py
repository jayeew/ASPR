from __future__ import annotations

import asyncio
from typing import Literal

from figure_pipeline.fig3_revision.models import Record
from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig5_four_panel.evaluate import PROMPT, batches
from figure_pipeline.fig5_revision.models import Part
from figure_pipeline.fig5_revision.runtime import Runner, protected
from figure_pipeline.fig5_robustness.pipeline import align_evaluation

from .diagnostic import CONDITIONS, OLD, OUT, config, evaluation_packet


class DiagnosticPart(Part):
    answer_completeness: Literal['complete', 'partial', 'omitted', 'withheld', 'unresolved']
    error_categories: list[Literal['missing_required_content', 'incomplete_relation_or_construct',
                                   'contradicted_by_evidence', 'unsupported_assertion',
                                   'wrong_scope', 'unresolved']]
    error_evidence: str


class Question(Record):
    question_id: str
    parts: list[DiagnosticPart]


class Candidate(Record):
    report_id: str
    questions: list[Question]


class Evaluation(Record):
    candidates: list[Candidate]


INSTRUCTION = PROMPT + '''
INDEPENDENT DIAGNOSTIC: Add answer_completeness and error_categories for each part.
Empty error_categories means no established substantive error or incompleteness.
Distinguish missing_required_content and incomplete_relation_or_construct from contradicted_by_evidence.
A partial answer is not automatically factually false. unsupported_assertion requires an actual definite
claim without sufficient evidence; wrong_scope requires an actual scope error, not a sensible limitation.
An otherwise correct answer can contain multiple errors; retain all applicable categories. Withheld
judgments are evaluated against visible evidence and are not automatically errors. error_evidence must
give a concise specific report assertion or missing required component, with existing segment IDs for
an actual assertion. Do not infer factual errors from length or from failed coverage alone. Apply the
same criteria to all anonymous candidates, accept scientific equivalence, and do not assume any ranking.
'''


async def one(runner: Runner, paper: str) -> None:
    while not all((OUT / 'reports' / condition / f'{paper}.json').exists() for condition in CONDITIONS):
        await asyncio.sleep(10)
    data, mapping = evaluation_packet(paper)
    packets = batches(runner, data)
    result = []
    for i, packet in enumerate(packets):
        if not runner.fits(INSTRUCTION, packet, Evaluation):
            raise ValueError('Independent diagnostic exceeds context; needs complete-candidate splitting')
        key = f'error_attribution_{i + 1}'
        write(OUT / 'independent_inputs' / key / f'{paper}.json', packet)
        value = await runner.call(paper, 'independent_evaluation', key, INSTRUCTION, packet,
                                  Evaluation, model='gpt-6-astra', effort='high')
        result.extend(align_evaluation(value, packet)['candidates'])
    write(OUT / 'independent_aligned' / f'{paper}.json', {'mapping': mapping, 'candidates': result})


async def main() -> None:
    papers = read(OLD / 'cohort.json')['repeated']
    # Separate ledger avoids concurrent writers sharing a request budget counter.
    cfg = config().model_copy(update={'output': OUT / 'independent_calls', 'cli_limit': 5,
                                     'ordinary_limit': 15, 'additional_limit': 5, 'call_limit': 20})
    runner = Runner(cfg)
    write(OUT / 'independent_protocol.json', {'papers': papers, 'selection': 'Original prespecified repeat cohort',
          'judge': 'gpt-6-astra', 'hard_cap': 20, 'primary_evaluations_overwritten': False})
    await asyncio.gather(*(protected(runner, paper, 'independent_review', 'all', one(runner, paper)) for paper in papers))


if __name__ == '__main__':
    asyncio.run(main())
