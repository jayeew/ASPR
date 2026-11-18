"""One constrained pointer-only repair of evaluator quotations; never rewrite judgments."""
from __future__ import annotations

import copy
import re
from typing import Any, Literal

from pydantic import model_validator

from .client import Scheduler
from .evaluation import evidence_audit, report_fragments, report_quote_audit, source_index
from .io import fingerprint, write
from .models import Checklist, Evaluation, Record


class Pointer(Record):
    target_id: str
    present: bool
    first_clause_id: str | None
    last_clause_id: str | None
    same_assertion_and_scope: bool
    confidence: Literal['high', 'low']
    reason: str

    @model_validator(mode='after')
    def consistent_pointer(self) -> Pointer:
        if self.present != (self.first_clause_id is not None and self.last_clause_id is not None):
            raise ValueError('A present match needs two clause IDs; an absent match needs neither')
        if not self.present and (self.first_clause_id is not None or self.last_clause_id is not None):
            raise ValueError('An absent match must not include clause IDs')
        return self


class Locations(Record):
    locations: list[Pointer]


INSTRUCTION = '''Locate failed evaluator quotations in ONLY the supplied source clauses. Do not call tools.
This is pointer-only transcription repair, not a scientific reevaluation. For each target return present,
first_clause_id and last_clause_id identifying ONE continuous original span from ONE permitted fragment.
Require the SAME COMPLETE scientific assertion and scope as the failed quotation, preserving negation,
numbers, conditions, attribution, causal direction and achieved versus proposed claims. Minor transcription,
PDF typography and wording paraphrase are eligible only if the same whole assertion is explicitly present.
A related topic, weaker claim, differently qualified result, translated report summary, or another source is
NOT a match. Do not combine distant passages, cross fragments, add omitted assumptions, or replace a false
claim with a correct but different claim. Select the shortest consecutive clause range containing the entire
assertion. If no high-confidence same-assertion/same-scope span exists, return present=false with null IDs.
Bibliographic title fields only locate title quotations; they do not verify scientific assertions in the body.
Never combine title and body fields, or use a title as a substitute for an absent detailed scientific assertion.
For report targets search only permitted report prose fields, never embedded evidence quotations. Do not
change judgments, support labels, dimension scores or applicability. Return exactly one entry per target ID.'''


def clause_spans(text: str, maximum: int = 600) -> list[tuple[int, int]]:
    """Partition each fragment exactly; consecutive IDs always reconstruct original text."""
    spans, start = [], 0
    boundaries = [m.end() for m in re.finditer(r'[.!?。！？;；](?:\s+|$)|\n\s*\n', text)]
    while start < len(text):
        limit = min(len(text), start + maximum)
        eligible = [end for end in boundaries if start < end <= limit]
        if eligible:
            end = eligible[0]
        elif limit < len(text):
            space = text.rfind(' ', start + maximum // 2, limit)
            end = space + 1 if space >= 0 else limit
        else:
            end = limit
        spans.append((start, end))
        start = end
    return spans


def collect_targets(value: dict[str, Any], sources: list[dict[str, Any]],
                    report: dict[str, Any] | None) -> list[dict[str, Any]]:
    (Evaluation if report is not None else Checklist).model_validate(value)
    index, targets = source_index(sources), []
    fragments = report_fragments(report) if report is not None else []
    for i, dimension in enumerate(value['dimensions']):
        prefix = ['dimensions', i]
        evidence_lists = [(prefix + ['evidence'], dimension['evidence'])] if report is None else [
            (prefix + ['checks', j, 'evidence'], check['evidence'])
            for j, check in enumerate(dimension['checks'])]
        for path, evidence in evidence_lists:
            for j, item in enumerate(evidence):
                if not evidence_audit(item, index)['located']:
                    targets.append({'path': path + [j, 'quote'], 'kind': 'source',
                                    'source_id': item['source_id'], 'quote': item['quote']})
        if report is not None:
            quotations = [(prefix + ['report_quotes', j], quote)
                          for j, quote in enumerate(dimension['report_quotes'])]
            quotations += [(prefix + ['checks', j, 'report_quote'], check['report_quote'])
                           for j, check in enumerate(dimension['checks'])]
            for path, quote in quotations:
                if not report_quote_audit(quote, fragments)['located']:
                    targets.append({'path': path, 'kind': 'report', 'source_id': None, 'quote': quote})
    for i, target in enumerate(targets):
        target['target_id'] = f'T{i:04d}'
    return targets


def build_candidates(targets: list[dict[str, Any]], sources: list[dict[str, Any]],
                     report: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested = {t['source_id'] for t in targets if t['kind'] == 'source'}
    fragments = [dict(s, kind='source', field='text') for s in sources if s['source_id'] in requested]
    titles_seen: set[tuple[str, str]] = set()
    for source in sources:
        identity = (source['source_id'], source.get('title', ''))
        if identity[0] in requested and identity[1] and identity not in titles_seen:
            fragments.append({'kind': 'source', 'field': 'title', 'source_id': identity[0], 'text': identity[1]})
            titles_seen.add(identity)
    if any(t['kind'] == 'report' for t in targets) and report is not None:
        fragments.extend({'kind': 'report', 'field': field, 'text': text, 'source_id': None}
                         for field, text in report_fragments(report))
    candidates, index = [], {}
    for i, fragment in enumerate(fragments):
        clauses = []
        for j, (start, end) in enumerate(clause_spans(fragment['text'])):
            identifier = f'F{i:04d}C{j:04d}'
            clauses.append({'clause_id': identifier, 'text': fragment['text'][start:end]})
            index[identifier] = {'fragment': fragment, 'fragment_id': i, 'position': j,
                                 'start': start, 'end': end}
        candidates.append({'fragment_id': i, 'kind': fragment['kind'],
                           'source_id': fragment['source_id'], 'field': fragment.get('field'),
                           'clauses': clauses})
    return candidates, index


def selected_span(target: dict[str, Any], pointer: dict[str, Any],
                  index: dict[str, Any]) -> dict[str, Any] | None:
    if not (pointer['present'] and pointer['same_assertion_and_scope'] and pointer['confidence'] == 'high'):
        return None
    first = index.get(pointer['first_clause_id'])
    last = index.get(pointer['last_clause_id'])
    if first is None or last is None:
        raise ValueError('Citation repair points to an unknown clause ID')
    if first['fragment_id'] != last['fragment_id'] or first['position'] > last['position']:
        raise ValueError('Citation repair cannot cross fragments or reverse a span')
    fragment = first['fragment']
    if fragment['kind'] != target['kind'] or fragment['source_id'] != target['source_id']:
        raise ValueError('Citation repair cannot change source identity or citation kind')
    start, end = first['start'], last['end']
    text = fragment['text'][start:end]
    return {'text': text, 'start': start, 'end': end,
            'source_start': int(fragment.get('start', 0)) + start if fragment.get('field') == 'text' else None,
            'source_end': int(fragment.get('start', 0)) + end if fragment.get('field') == 'text' else None,
            'source_id': fragment['source_id'], 'field': fragment.get('field'),
            'title_only': fragment.get('field') == 'title',
            'fragment_id': first['fragment_id']}


def apply_locations(value: dict[str, Any], targets: list[dict[str, Any]],
                    locations: dict[str, Any], index: dict[str, Any]) -> dict[str, Any]:
    checked = Locations.model_validate(locations).model_dump()
    pointers = {row['target_id']: row for row in checked['locations']}
    if len(pointers) != len(checked['locations']) or set(pointers) != {t['target_id'] for t in targets}:
        raise ValueError('Citation repair must answer each failed quotation exactly once')
    result, audit = copy.deepcopy(value), []
    for target in targets:
        pointer = pointers[target['target_id']]
        span = selected_span(target, pointer, index)
        if span is not None:
            container = result
            for part in target['path'][:-1]:
                container = container[part]
            container[target['path'][-1]] = span['text']
        audit.append(dict(target, pointer=pointer, repaired=span is not None, span=span))
    return {'value': result, 'audit': {'targets': audit, 'repaired_count': sum(r['repaired'] for r in audit),
                                     'scientific_judgments_and_scores_changed': False}}


async def repair_citations(scheduler: Scheduler, key: str, value: dict[str, Any],
                           sources: list[dict[str, Any]],
                           report: dict[str, Any] | None = None) -> dict[str, Any]:
    targets = collect_targets(value, sources, report)
    if not targets:
        return {'value': copy.deepcopy(value), 'audit': {'targets': [], 'repaired_count': 0,
                                                       'scientific_judgments_and_scores_changed': False}}
    candidates, index = build_candidates(targets, sources, report)
    payload = {'targets': targets, 'candidate_fragments': candidates}
    response = await scheduler.ask(key + '/citation_location', INSTRUCTION, payload, Locations, 'xhigh')
    result = apply_locations(value, targets, response, index)
    result['audit']['request_fingerprint'] = fingerprint(payload)
    result['audit']['meaning'] = ('Model-assisted same-assertion location, not independent scientific validation. '
                                'Unmatched quotations remain failures; original evaluator response is retained.')
    write(scheduler.output / 'audits/citation_repair' / key / (fingerprint(payload) + '.json'), result['audit'])
    return result
