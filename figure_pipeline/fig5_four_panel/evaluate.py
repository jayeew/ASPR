from __future__ import annotations

import asyncio
from typing import Any

from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_explanation_100 import prompts as fig4
from figure_pipeline.fig5_revision.models import Evaluation
from figure_pipeline.fig5_robustness import prompts as previous
from figure_pipeline.fig5_robustness.models import Config
from figure_pipeline.fig5_robustness.pipeline import align_evaluation
from figure_pipeline.fig5_robustness.runtime import Runner

from .prepare import OUT

PROMPT = previous.EVALUATE + fig4.REFINED_EVALUATION + '''
ONE CORRECTIVE EVALUATION; shared conditional_references are keyed by each candidate's material_packet_id.
Apply every public question and original applicable part to every supplied candidate. Return empty parts
for originally inapplicable questions. Never alter fixed applicability. Conditions/models are anonymous.
The independent_work_aliases map merges duplicate versions of the same original; do not double count them.
CORRECTED TAXONOMY takes precedence: response_type is substantive_answer, target_abstention, omission,
or unresolved. A completed factual answer with a scope/causality caveat remains substantive_answer, not
abstention. Explicit withholding of this requested judgment is target_abstention. Silence is omission.
For substantive_answer and omission explicitly_abstains=false; for target_abstention true; for unresolved null.
For omission status=not_written, scope_correct=null is permitted and DOES NOT mean evaluator uncertainty.
For a pure abstention status=not_written; assess appropriateness from the candidate's visible packet.
An insufficient partial answer is substantive_answer with status=incorrect, not evaluator uncertainty.
conflicting_material_error is true only when another assertion concerning THIS part materially contradicts
its correct answer; support this in reason and actual report segments. Missing discussion alone is not error.
Grounding must use only visible originals/native facts; generated claim text is not historical original proof.
unsupported_definitive requires an actual definite ungrounded assertion, never infer it from omission.
A true abstention/error needs exact existing segment IDs. Scientific unresolved means a genuine conflict or
unresolvable evaluation, not missing discussion, absent evidence, or a null scope field for silence.
The private reference is fallible; do not force a match when originals conflict; explain in reason.
Keep reason and support_reason concise (one short sentence each), IDs exact, and return labels only.
'''


def config() -> Config:
    return Config(output=OUT, ordinary_limit=20, additional_limit=10, call_limit=30, cli_limit=20)


def subset(data: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    keys = {c['material_packet_id'] for c in candidates}
    graphs = {data['visible_materials'][k]['graph_key'] for k in keys}
    return {**data, 'candidates': candidates,
            'conditional_references': {k: v for k, v in data['conditional_references'].items() if k in keys},
            'visible_materials': {k: v for k, v in data['visible_materials'].items() if k in keys},
            'native_graphs': {k: v for k, v in data['native_graphs'].items() if k in graphs}}


def batches(runner: Runner, data: dict[str, Any]) -> list[dict[str, Any]]:
    if runner.fits(PROMPT, data, Evaluation):
        return [data]
    result, current = [], []
    for candidate in data['candidates']:
        packet = subset(data, current + [candidate])
        if current and not runner.fits(PROMPT, packet, Evaluation):
            result.append(subset(data, current))
            current = []
        current.append(candidate)
    result.append(subset(data, current))
    if any(not runner.fits(PROMPT, x, Evaluation) for x in result):
        raise ValueError('Single candidate exceeds context: do not truncate original material')
    return result


async def evaluate(runner: Runner, paper: str, packets: list[dict[str, Any]]) -> None:
    aligned = []
    for index, data in enumerate(packets):
        key = 'all' if len(packets) == 1 else f'batch{index + 1}'
        write(OUT / 'dispatched_inputs' / key / f'{paper}.json', data)
        try:
            response = await runner.call(paper, 'evaluation', key, PROMPT, data, Evaluation, additional=index > 0)
            current = align_evaluation(response, data)
            for candidate in current['candidates']:
                for question in candidate['questions']:
                    for part in question['parts']:
                        expected = {'substantive_answer': False, 'omission': False,
                                    'target_abstention': True, 'unresolved': None}[part['response_type']]
                        if part['explicitly_abstains'] is not expected:
                            part['technical_state'] = 'invalid_response_taxonomy'
            aligned.extend(current['candidates'])
            write(OUT / 'aligned' / f'{paper}.json', {'candidates': aligned})
        except (OSError, ValueError, RuntimeError) as exc:
            runner.status(paper, 'evaluation', key, 'unresolved', reason=str(exc))
            print(f'UNRESOLVED {paper}/{key}: {str(exc)[:300]}', flush=True)


async def main() -> None:
    runner = Runner(config())
    plan = {p: batches(runner, read(OUT / 'evaluation_inputs' / f'{p}.json'))
            for p in read(OUT / 'protocol.json')['selected']}
    total = sum(map(len, plan.values()))
    write(OUT / 'capacity_preflight.json', {'planned_calls': total, 'papers': {p: len(v) for p, v in plan.items()}})
    if total > 30:
        raise ValueError(f'{total} planned calls exceed hard cap; no dispatch')
    print(f'PREFLIGHT: {total} calls; hard cap 30 including failures', flush=True)
    await asyncio.gather(*(evaluate(runner, paper, packets) for paper, packets in plan.items()))


if __name__ == '__main__':
    asyncio.run(main())
