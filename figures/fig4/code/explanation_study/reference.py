from __future__ import annotations

import asyncio
from typing import Any

from figure_pipeline.fig3_revision.storage import read, write

from .models import ASPECTS, Config, Reference
from .runtime import Runner

PROMPT = '''Act as a rigorous scientific reference curator for a small exploratory ablation study.
You receive a manuscript, original historical passages, fixed core contribution references and native
graph observations, but NO candidate reports or condition results. Work in Chinese.
First assess whether the graph has any concrete, scientifically relevant information useful beyond
repeating the manuscript. A large neighbor count alone is insufficient. Keep a paper if actual neighbor
content plus sources permits at least one checkable knowledge relationship for a principal contribution.
Exclude it if all graph neighborhoods are generic, irrelevant or unusable. Missing citation paths alone
does not make a graph useless when semantic neighbors or joint structure are informative. List specific
claim/source examples and limitations. This screening must never depend on whether a full report wins.

Then create a fixed reference of exactly TEN focused questions, TWO for each aspect:
historical_verification: scientifically important prior-work overlap, residual difference or correction
of an apparent antecedent, requiring a concrete independent historical source comparison where available.
knowledge_position: relevant scientific knowledge neighborhoods and distinctions between types of
relationship, e.g. same object versus parallel mechanism versus merely similar method. Do not conflate
historical evidence correctness with graph proximity.
joint_contribution: specific shared historical bases or combined connectivity involving >=2 target
contributions; explain what the joint view reveals that a single contribution view misses. A generic
whole-paper mechanism summary is insufficient. Avoid summing single-insertion effects.
structural_resolution: a scientifically useful quantitative distinction between contribution neighborhoods,
such as concentration/diversity versus connectivity, where numbers refine a tempting coarse impression.
Do not ask to merely copy a metric name/value. State the interpretation and its restricted scope.
citation_contact: specific named historical contacts distinguishing recorded two-hop/shared-reference/
direct-citation/semantic-only types and what interpretation the difference permits. If the snapshot has
no positive contact, mark both questions not applicable rather than invent a positive opportunity.

Use question IDs H1,H2,K1,K2,J1,J2,S1,S2,P1,P2 in that order. Each applicable question has exactly TWO
answer_parts, IDs <question_id>a and <question_id>b. Each part is a short independently checkable
scientific proposition, with exact supporting original source quote and location or graph field/value,
acceptable equivalent wording and what would be insufficient. For not-applicable questions use no parts.
Prefer stable concrete scientific questions useful to a scientist interpreting innovation, not an audit
of the system's internal field spelling. Do not reward mentioning numbers without correct interpretation.
Record whether the manuscript already answers the question; do not pretend an external source is unique
if the same scientific content is in the paper. Independent source verification can still be a distinct
evidentiary benefit, to be measured separately from answer correctness.
Reference judgments are not automatically true: verify against the original passages, preserving
abstract/full-text limits and contradictions. Generated checklist/manuscript summaries are not evidence.
Avoid facts not established by supplied material. Do not infer disciplinary identity from community IDs,
support/derivation from paths, firstness from missing prior art, or global impact from local connectivity.
For structural questions acknowledge that raw nodes/edges/community identities can permit reconstructing
some removed numerical summaries; such correct reconstruction should count, not be forbidden to create
an artificial ablation drop. Only existing graph facts, no new neighbors or graph construction.
These are exploratory case references, not newly obtained human expert labels or an independent test set.'''


async def prepare_references(config: Config, only: list[str] | None = None) -> None:
    ids = read(config.source / 'pilot.json')['paper_ids']
    write(config.output / 'pilot.json', {'paper_ids': ids})
    write(config.output / 'reference_protocol.json', {'prompt': PROMPT, 'aspects': ASPECTS,
          'screening_before_new_condition_reports': True, 'sample_expansion': False,
          'basis': 'manuscript_original_history_native_graph_not_condition_outputs'})
    runner = Runner(config)

    async def one(paper: str) -> None:
        material = read(config.source / 'inputs/evaluation' / f'{paper}.json')
        material.pop('fig3_criteria', None)
        result = await runner.call(paper, 'reference', 'papers', PROMPT, material, Reference)
        print(f'GRAPH ELIGIBLE {paper}: {result["graph_useful"]}; {result["eligibility_reason"]}', flush=True)

    await asyncio.gather(*(one(p) for p in ids if only is None or p in only))


async def split_reference(config: Config, paper: str) -> None:
    """Explicit recovery for a long response that disconnected; the same ten questions are retained."""
    material = read(config.source / 'inputs/evaluation' / f'{paper}.json')
    material.pop('fig3_criteria', None)
    runner = Runner(config)
    groups = {'history_knowledge': ['H1', 'H2', 'K1', 'K2'],
              'joint_structure': ['J1', 'J2', 'S1', 'S2'], 'paths': ['P1', 'P2']}

    async def one(name: str, ids: list[str]) -> dict[str, Any]:
        prompt = PROMPT + '\nFor this response ONLY, return the following question IDs: ' + ','.join(ids)
        prompt += '. This overrides the ten-question output count, not the scientific criteria. Do not add other questions.'
        return await runner.call(paper, 'reference_parts', name, prompt, material, Reference)

    parts = await asyncio.gather(*(one(name, ids) for name, ids in groups.items()))
    result = {**parts[0], 'questions': [q for p in parts for q in p['questions']],
              'limitations': list(dict.fromkeys(v for p in parts for v in p['limitations']))}
    write(config.output / 'reference/papers' / f'{paper}.json', result)
    write(config.output / 'reference_recovery' / f'{paper}.json', {
        'reason': 'Two original responses disconnected after about 965–971 seconds; neither had a final answer.',
        'split_groups': groups, 'scope_changed': False, 'combined_without_model_call': True})
    print(f'COMPLETE split reference {paper}: graph_useful={result["graph_useful"]}', flush=True)


async def repair_available_reference(config: Config, paper: str, question_ids: list[str]) -> None:
    from .generation import view
    original = read(config.output / 'reference/papers' / f'{paper}.json')
    archive = config.output / 'reference_initial/papers' / f'{paper}.json'
    if not archive.exists():
        write(archive, original)
    for path in (config.output / 'aligned_evaluation/existing').glob(f'*/{paper}.json'):
        archived = config.output / 'aligned_evaluation_initial/existing' / path.parent.name / path.name
        if not archived.exists():
            write(archived, read(path))
    actual = view(config, paper, 'F')
    material = {'manuscript': actual['manuscript'], 'native_graph': actual['graph'],
                'evidence_blocks': actual['original_evidence']}
    prompt = PROMPT + '\nFor this response ONLY, return these question IDs: ' + ','.join(question_ids)
    prompt += '''. This overrides the ten-question output count. Select scientifically useful comparisons
that can actually be resolved from the original passages and graph supplied HERE. The broad archive held
other papers, but they were not delivered to any writer. Do not assume or require those absent sources.
No candidate report is provided. Do not optimize for any condition ranking. Keep the same aspect mapping,
two answer parts per applicable question, and original scientific standards. This is an input-availability
correction for an exploratory fixed-input study, not a change in scientific truth or a new retrieval task.'''
    result = await Runner(config).call(paper, 'reference_repair', 'available_input', prompt, material, Reference)
    replacement = {q['question_id']: q for q in result['questions']}
    if set(replacement) != set(question_ids):
        raise ValueError(f'Reference repair IDs do not match requested IDs for {paper}')
    original['questions'] = [replacement.get(q['question_id'], q) for q in original['questions']]
    write(config.output / 'reference/papers' / f'{paper}.json', original)
    write(config.output / 'reference_repair_manifest' / f'{paper}.json', {
        'question_ids': question_ids,
        'aspects': sorted({q['aspect'] for q in result['questions']}),
        'reason': 'Initial questions required historical sources absent from all actual writer inputs.',
        'candidate_reports_provided': False, 'report_regeneration_required': False})
