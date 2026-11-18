from __future__ import annotations

import asyncio
import random
from typing import Any

from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig4_rerun.config import CONDITIONS

from .generation import eligible, view
from .models import ASPECTS, Config, Judgments
from .runtime import Runner

PROMPT = '''Evaluate anonymous scientific innovation reports against the TWO specific reference questions
provided for this aspect. Work in Chinese. This is evidence-based classification, not a holistic 0–3 grade.
Return one candidate entry per report_id, one question entry per supplied question_id, and exactly the
provided part IDs for every applicable question. For not-applicable questions return an empty parts list.

For each reference answer part classify what the REPORT ACTUALLY SAYS:
correct: explicit scientifically equivalent content, correct object/relation and necessary scope, supported
by the reference and original evidence. A bare metric value without the part's scientific meaning, a
generic caution, a source list, or a vaguely similar narrative does not meet a concrete answer part.
incorrect: the report makes a relevant substantive assertion that conflicts with evidence or materially
overstates scope. A correct sentence cannot cancel a conflicting overclaim elsewhere in the report.
not_written: the required content is absent, too generic, or explicitly left unknown. Correct abstention
earns no coverage for the absent answer but is NOT an error. Do not infer an absent answer from the reference.
unresolved: the report makes the relevant assertion but original evidence cannot resolve its correctness.
The fixed reference is a source-bound benchmark, not infallible truth. If it conflicts with originals,
mark unresolved and describe the discrepancy rather than forcing the candidate to follow it.

Select exact report segment IDs as evidence; never regenerate or paraphrase quotations. IDs only.
For correct/incorrect substantive content at least one relevant report segment is required. An abstention
can be located with its segment ID but remains not_written. Separate genuine scientific relation synthesis
from native shared-neighbor/union-connectivity observations. Multi-part generic summaries are not joint
graph observations. Structure parts require the specific contrast, resolution and bounded interpretation;
correct derivations from visible nodes/edges/community identities count even if no precomputed metric
was available. Do not artificially penalize reconstruction to make an ablation look worse.
Path parts require the stated concrete recorded contact type for the specified scientific sources.
Correctly saying 'the type is unknown' is not the same as identifying an observed two-hop/shared-reference
contact. Manuscript references do not automatically answer questions about a different native snapshot edge.

Also classify the actual evidentiary basis USED in the cited report passages:
independent_history: a concrete historical comparison is traced to an available independent original
historical source, with its content accurately used. Target-paper references or its description of a
previous work alone do not constitute independent historical verification.
native_graph: the report actually uses the relevant observed native fact with traceable identity/scope.
derived_visible_graph: the correct graph observation is explicitly derived from the visible raw facts.
manuscript_only: supported by target text/bibliography, without independent-source or graph verification.
unsupported: a stated claim lacks an adequate basis or has a wrong source attribution.
unclear: no claim or the actual grounding cannot be established.
Record source IDs actually cited/identifiably used by the report; do not supply the gold answer's citations
on its behalf. Availability of evidence in the input does not mean a report used it.

Each anonymous report has a material_availability description of WHAT ITS WRITER RECEIVED. This differs
from the common full evidence available to YOU. A report may truthfully say graph/path evidence was not
provided to it even though your reference includes it. Do not call that a scientific contradiction.
Judge quality independently of condition guesses, model identity, verbosity and assumed system ranking.
Do not reward the number of citations or penalize accurate limitation language. The outputs may all tie.
No overall preference, no total score, no claim that these are human expert annotations.'''


def segments(body: str) -> list[dict[str, str]]:
    return [{'segment_id': f'P{i:03d}', 'text': text} for i, text in enumerate(
        (p.strip() for p in body.split('\n\n') if p.strip()), 1)]


def availability(config: Config, paper: str, condition: str) -> dict[str, Any]:
    material = view(config, paper, condition)
    history = sorted({p['source_id'] for b in material.get('original_evidence', []) for p in b['provenance']})
    blocks = [b['block_id'] for b in material.get('original_evidence', [])]
    return {
        'manuscript': True, 'original_history_source_ids': history, 'original_history_block_ids': blocks,
        'single_contribution_graph': condition not in ('T', 'E'),
        'joint_graph': condition not in ('T', 'E', 'F_noJ'),
        'non_path_structural_numbers': condition not in ('T', 'E', 'F_noM'),
        'citation_annotations': condition not in ('T', 'E', 'F_noP'),
        'raw_nodes_edges_communities': condition not in ('T', 'E'),
        'manuscript_bibliography_retained': True,
    }


def candidates(config: Config, paper: str, report_set: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    order = list(CONDITIONS)
    random.Random(paper + ':' + report_set).shuffle(order)
    root = config.source if report_set == 'existing' else config.output
    values, mapping = [], {}
    for i, condition in enumerate(order, 1):
        report_id = f'R{i:02d}'
        report = read(root / 'reports' / condition / f'{paper}.json')
        mapping[report_id] = condition
        values.append({'report_id': report_id, 'report_segments': segments(report['body']),
                       'material_availability': availability(config, paper, condition)})
    return values, mapping


async def evaluate(config: Config, report_set: str) -> None:
    runner = Runner(config)

    async def one(paper: str, aspect: str) -> None:
        original = read(config.source / 'inputs/evaluation' / f'{paper}.json')
        reference = read(config.output / 'reference/papers' / f'{paper}.json')
        values, mapping = candidates(config, paper, report_set)
        material = {key: original[key] for key in ('manuscript', 'evidence_blocks', 'native_graph')}
        material.update(reference_questions=[q for q in reference['questions'] if q['aspect'] == aspect],
                        candidates=values)
        write(config.output / 'evaluation_mapping' / report_set / f'{paper}.json', mapping)
        if not any(q['applicable'] for q in material['reference_questions']):
            result = {'candidates': [{'report_id': value['report_id'], 'questions': [
                {'question_id': q['question_id'], 'parts': []} for q in material['reference_questions']]}
                for value in values]}
        else:
            repair_path = config.output / 'reference_repair_manifest' / f'{paper}.json'
            repaired = repair_path.exists() and aspect in read(repair_path)['aspects']
            stage = 'evaluation_' + report_set + ('_repaired' if repaired else '')
            result = await runner.call(paper, stage, aspect, PROMPT, material, Judgments)
        ids = [c['report_id'] for c in result['candidates']]
        if len(ids) != len(set(ids)) or set(ids) != set(mapping):
            raise ValueError(f'Incomplete/duplicate report IDs in {paper}/{aspect}; raw judgments retained')
        # Source passages are copied locally by ID. A model never rewrites the quoted report evidence.
        for candidate in result['candidates']:
            source = next(c for c in values if c['report_id'] == candidate['report_id'])
            index = {s['segment_id']: s['text'] for s in source['report_segments']}
            for question in candidate['questions']:
                for part in question['parts']:
                    if any(s not in index for s in part['report_segment_ids']):
                        part['technical_state'] = 'invalid_segment_id'
                    elif part['status'] in ('correct', 'incorrect') and not part['report_segment_ids']:
                        part['technical_state'] = 'missing_report_evidence'
                    else:
                        part['technical_state'] = 'completed'
                    part['report_quotes'] = [index[s] for s in part['report_segment_ids'] if s in index]
        write(config.output / 'aligned_evaluation' / report_set / aspect / f'{paper}.json', result)

    papers = eligible(config)
    if report_set == 'new':
        ready = [p for p in papers if all((config.output / 'reports' / c / f'{p}.json').exists() for c in CONDITIONS)]
        for paper in set(papers) - set(ready):
            print(f'UNRESOLVED new seven-report comparison: {paper}; missing reports remain NA', flush=True)
        papers = ready
    await asyncio.gather(*(one(p, aspect) for p in papers for aspect in ASPECTS))
