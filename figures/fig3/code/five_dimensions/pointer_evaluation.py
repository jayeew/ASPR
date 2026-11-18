"""Evaluate with pointers to actual report prose and continuous original-source clauses."""
from __future__ import annotations

import copy
import re
from typing import Any, Literal

from pydantic import Field, model_validator

from .citation_repair import clause_spans
from .client import Scheduler
from .evaluation import dimension_prose
from .io import encoded, fingerprint, write
from .models import Dimension, Evaluation, Record, Report, assert_dimensions

EVALUATION_PROTOCOL = 'source_and_report_clause_pointers_v2'


class ReportPointer(Record):
    first_clause_id: str
    last_clause_id: str


class SourcePointer(ReportPointer):
    source_id: str
    source_field: Literal['text', 'title']
    explanation: str


class PointerAssertionCheck(Record):
    report_quote: ReportPointer
    support: Literal['supported', 'partly_supported', 'not_verifiable', 'contradicted']
    scope_correct: bool | None
    evidence: list[SourcePointer]
    reason: str


class PointerDimensionEvaluation(Record):
    dimension: Dimension
    status: Literal['assessed', 'not_applicable', 'unresolved']
    score: int | None = Field(ge=0, le=3)
    checks: list[PointerAssertionCheck]
    report_quotes: list[ReportPointer]
    reason: str

    @model_validator(mode='after')
    def valid_score(self) -> PointerDimensionEvaluation:
        if (self.status == 'assessed') != (self.score is not None):
            raise ValueError('Only assessed dimensions have a numeric score')
        return self


class PointerEvaluation(Record):
    dimensions: list[PointerDimensionEvaluation]

    @model_validator(mode='after')
    def all_dimensions(self) -> PointerEvaluation:
        assert_dimensions([row.dimension for row in self.dimensions])
        return self


POINTER_INSTRUCTION = '''
QUOTATION OUTPUT CONTRACT: The scientific evaluation rubric above is unchanged. Output quotations ONLY as
pointers under the supplied schema, never by retyping, paraphrasing, translating or summarising quotations.
The report's contribution_overview, synthesis, and each dimension's judgment/rationale/scope are now annotated
fields containing original text clauses with stable clause_id values. Read these clauses in order as ordinary
report prose. The report's own evidence and status remain visible for inspection but are NOT verified truth.
For each check, choose the actual substantive assertion expressed in the report, not an assertion you wish
it had made. report_quote and report_quotes must use first_clause_id/last_clause_id from ONE report prose
field. Do not quote your own summary, source evidence, the checklist, or text embedded in report.evidence
as a report assertion. If an applicable issue is absent, explain the omission in reason and score accordingly;
do not fabricate a quote for an omitted statement. Report-related checks can draw on overview or synthesis.
Source evidence pointers require source_id, source_field (text or title), first_clause_id, last_clause_id and
explanation. Use one continuous range in ONE original source fragment and ONE field. Preserve qualifiers,
negation, conditions, numerical values and surrounding context needed for interpretation. Never stitch
nonadjacent fragments, skip intervening clauses, or combine a title with body text. Choose the shortest
consecutive range containing the assertion and its necessary conditions. There is no need to reproduce the
text: software copies the selected original span exactly, including PDF artefacts and line numbers.
A title field is visible bibliographic information, not full-text verification of an experimental result.
Graph observations can establish recorded topology, not automatically scientific causality or priority.
A pointer proves only location, not relevance or scientific support. Make the support and scope judgments
from the selected source and the actual report assertion. When material does not support the report, keep
that failure in support/reason/score; do not select unrelated text merely to populate evidence. Evidence may
be empty for not_verifiable, and report_quotes may be empty for a genuinely absent answer scored 0.
Return all five dimensions using exactly the pre-existing applicability and unchanged 0–3 analysis-quality
rubric. This protocol does not change what is scored, does not ask for a novelty total, and does not permit
a more favourable score because pointers are easier to locate. Do not call tools or introduce outside facts.
'''


def add_fragment(index: dict[str, Any], text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    fragment_id = f"{index['namespace']}F{len(index['fragments']):05d}"
    fragment = {**metadata, 'fragment_id': fragment_id, 'text': text}
    index['fragments'][fragment_id] = fragment
    clauses = []
    for number, (start, end) in enumerate(clause_spans(text)):
        clause_id = f'{fragment_id}C{number:04d}'
        clauses.append({'clause_id': clause_id, 'text': text[start:end]})
        index['clauses'][clause_id] = {'fragment_id': fragment_id, 'start': start, 'end': end,
                                     'ordinal': number}
    return {'fragment_id': fragment_id, 'clauses': clauses}


def annotate_report(report: dict[str, Any], index: dict[str, Any]) -> dict[str, Any]:
    Report.model_validate(report)
    result = copy.deepcopy(report)
    for key in ('contribution_overview', 'synthesis'):
        result[key] = add_fragment(index, report[key], {'kind': 'report', 'field': key})
    for old, new in zip(report['dimensions'], result['dimensions']):
        for key in ('judgment', 'rationale', 'scope'):
            field = f"{old['dimension']}.{key}"
            new[key] = add_fragment(index, old[key], {'kind': 'report', 'field': field})
    return result


def annotate_sources(sources: list[dict[str, Any]], index: dict[str, Any]) -> list[dict[str, Any]]:
    result, titles = [], set()
    for source in sources:
        metadata = {key: value for key, value in source.items() if key not in {'text', 'title'}}
        item = add_fragment(index, source['text'], {'kind': 'source', 'source_field': 'text',
                                                   **metadata})
        result.append({**metadata, 'source_field': 'text', **item})
        title = source.get('title', '')
        identity = (source['source_id'], title)
        if title and identity not in titles:
            title_metadata = {key: value for key, value in metadata.items() if key not in {'start', 'end'}}
            item = add_fragment(index, title, {'kind': 'source', 'source_field': 'title', **title_metadata})
            result.append({**title_metadata, 'source_field': 'title', **item,
                           'title_only_notice': 'Bibliographic title only; not verification of full scientific claims.'})
            titles.add(identity)
    return result


def prepare_evaluation_payload(anonymous: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    namespace, original = 0, encoded(anonymous)
    while f'QP_{namespace}_' in original:
        namespace += 1
    index: dict[str, Any] = {'namespace': f'QP_{namespace}_', 'fragments': {}, 'clauses': {}}
    payload = copy.deepcopy(anonymous)
    payload['report'] = annotate_report(anonymous['report'], index)
    payload['source_passages'] = annotate_sources(anonymous['source_passages'], index)
    payload['quotation_protocol'] = EVALUATION_PROTOCOL
    return payload, index


def resolve_span(pointer: dict[str, Any], index: dict[str, Any], kind: str) -> dict[str, Any]:
    first = index['clauses'].get(pointer['first_clause_id'])
    last = index['clauses'].get(pointer['last_clause_id'])
    if first is None or last is None:
        raise ValueError('Unknown quotation clause ID')
    if kind == 'report' and first['fragment_id'] != last['fragment_id']:
        return resolve_report_prose(pointer, index, first, last)
    if first['fragment_id'] != last['fragment_id'] or first['ordinal'] > last['ordinal']:
        raise ValueError('Quotation range must be ordered and remain within one continuous fragment')
    fragment = index['fragments'][first['fragment_id']]
    if fragment['kind'] != kind:
        raise ValueError('Report prose and source evidence cannot substitute for one another')
    if kind == 'source' and (pointer['source_id'] != fragment['source_id']
                            or pointer['source_field'] != fragment['source_field']):
        raise ValueError('Quotation pointer does not match source identity and field')
    start, end = first['start'], last['end']
    source_text = kind == 'source' and fragment['source_field'] == 'text'
    return {'kind': kind, 'fragment_id': first['fragment_id'], 'first_clause_id': pointer['first_clause_id'],
            'last_clause_id': pointer['last_clause_id'], 'text': fragment['text'][start:end],
            'start': start, 'end': end, 'source_id': fragment.get('source_id'),
            'source_field': fragment.get('source_field'), 'report_field': fragment.get('field'),
            'source_type': fragment.get('source_type'),
            'title_only': fragment.get('source_field') == 'title',
            'source_start': int(fragment.get('start', 0)) + start if source_text else None,
            'source_end': int(fragment.get('start', 0)) + end if source_text else None}


def resolve_report_prose(pointer: dict[str, Any], index: dict[str, Any],
                         first: dict[str, Any], last: dict[str, Any]) -> dict[str, Any]:
    left, right = (index['fragments'][item['fragment_id']] for item in (first, last))
    fields = ('judgment', 'rationale', 'scope')
    left_name, _, left_field = left.get('field', '').rpartition('.')
    right_name, _, right_field = right.get('field', '').rpartition('.')
    if (left['kind'] != 'report' or right['kind'] != 'report' or not left_name
            or left_name != right_name or left_field not in fields or right_field not in fields
            or fields.index(left_field) >= fields.index(right_field)):
        raise ValueError('Report range must remain ordered within one continuous dimension prose block')
    fragments = {row['field'].rsplit('.', 1)[-1]: row['text'] for row in index['fragments'].values()
                 if row['kind'] == 'report' and row.get('field', '').startswith(left_name + '.')}
    prose, offsets = dimension_prose(fragments)
    start, end = offsets[left_field] + first['start'], offsets[right_field] + last['end']
    return {'kind': 'report', 'fragment_id': None,
            'first_clause_id': pointer['first_clause_id'], 'last_clause_id': pointer['last_clause_id'],
            'text': prose[start:end], 'start': start, 'end': end, 'source_id': None,
            'source_field': None, 'report_field': left_name + '.rendered_prose', 'source_type': None,
            'title_only': False, 'source_start': None, 'source_end': None,
            'rendering': 'Actual continuous judgment/rationale/scope with original fixed rendering labels.'}


def resolve_evaluation(raw: dict[str, Any], index: dict[str, Any]) -> dict[str, Any]:
    value = PointerEvaluation.model_validate(raw).model_dump(mode='json')
    audit = []
    for dimension in value['dimensions']:
        quotes = []
        for number, pointer in enumerate(dimension['report_quotes']):
            span = resolve_span(pointer, index, 'report')
            quotes.append(span['text'])
            audit.append({'dimension': dimension['dimension'], 'path': f'report_quotes[{number}]', **span})
        dimension['report_quotes'] = quotes
        for number, check in enumerate(dimension['checks']):
            span = resolve_span(check['report_quote'], index, 'report')
            check['report_quote'] = span['text']
            audit.append({'dimension': dimension['dimension'], 'path': f'checks[{number}].report_quote', **span})
            evidence = []
            for item_number, pointer in enumerate(check['evidence']):
                span = resolve_span(pointer, index, 'source')
                evidence.append({'source_id': pointer['source_id'], 'quote': span['text'],
                                 'explanation': pointer['explanation']})
                audit.append({'dimension': dimension['dimension'],
                              'path': f'checks[{number}].evidence[{item_number}]', **span})
            check['evidence'] = evidence
    result = Evaluation.model_validate(value).model_dump(mode='json')
    return {'value': result, 'audit': {'quotations': audit, 'quotation_count': len(audit),
                                     'scientific_judgments_and_scores_changed': False,
                                     'meaning': 'Exact pointer-to-text resolution; scientific relevance and support remain model judgments.'}}


class LocationCorrection(Record):
    target_id: str
    present: bool
    first_clause_id: str | None
    last_clause_id: str | None
    same_assertion_and_scope: bool
    confidence: Literal['high', 'low']
    reason: str

    @model_validator(mode='after')
    def paired_endpoints(self) -> LocationCorrection:
        supplied = self.first_clause_id is not None and self.last_clause_id is not None
        if self.present != supplied or (not self.present and
                (self.first_clause_id is not None or self.last_clause_id is not None)):
            raise ValueError('Present locations require both IDs; absent locations require neither')
        return self


class LocationCorrections(Record):
    corrections: list[LocationCorrection]


LOCATION_INSTRUCTION = '''Correct invalid quotation endpoint IDs ONLY. Use only the supplied original fragment.
You are NOT reevaluating science or changing any score, support label, reason or explanation. Each task has
an original pointer with at least one nonexistent clause ID, its fixed valid endpoint(s), the complete actual
clause list of the same original fragment, and the original assertion/interpretation already selected by the
judge. Return valid first/last IDs preserving every fixed endpoint. Correct ONLY invalid endpoint IDs.
Choose a range for the SAME assertion, support interpretation and scope, with all necessary qualifiers.
Do not broaden, weaken, change a claim, use a related topic, choose a more favourable excerpt, change source,
change field, or move to another fragment. Do not merely clamp an out-of-range number to the final clause.
If the intended same-assertion range cannot be located at high confidence, return present=false and null IDs.
For source evidence, use the supplied report assertion and unchanged support reason/explanation to locate
what the original pointer intended, not a new argument. For report pointers, identify actual report prose,
not your paraphrase. Source titles cannot replace body evidence. Do not call tools. Return every target once.'''


def pointer_rows(value: dict[str, Any], index: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for i, dimension in enumerate(value['dimensions']):
        path = ['dimensions', i]
        for j, pointer in enumerate(dimension['report_quotes']):
            rows.append({'path': path + ['report_quotes', j], 'pointer': pointer, 'kind': 'report',
                         'context': {'dimension': dimension['dimension'], 'reason': dimension['reason']}})
        for j, check in enumerate(dimension['checks']):
            check_path = path + ['checks', j]
            context = {'dimension': dimension['dimension'], 'support': check['support'],
                       'scope_correct': check['scope_correct'], 'reason': check['reason']}
            rows.append({'path': check_path + ['report_quote'], 'pointer': check['report_quote'],
                         'kind': 'report', 'context': context})
            try:
                assertion = resolve_span(check['report_quote'], index, 'report')['text']
            except ValueError:
                # Do not repair a source range without an actually locatable original report assertion.
                continue
            for k, pointer in enumerate(check['evidence']):
                rows.append({'path': check_path + ['evidence', k], 'pointer': pointer, 'kind': 'source',
                             'context': {**context, 'report_assertion': assertion,
                                         'explanation': pointer['explanation']}})
    return rows


def identifiable_fragment(pointer: dict[str, Any], index: dict[str, Any]) -> tuple[str, dict[str, str]] | None:
    endpoint_ids = [pointer['first_clause_id'], pointer['last_clause_id']]
    valid = {key: pointer[key] for key in ('first_clause_id', 'last_clause_id')
             if pointer[key] in index['clauses']}
    if len(valid) == 2:
        return None  # Valid but reversed/cross-fragment pointers are not transcription repair targets.
    fragments = []
    for identifier in endpoint_ids:
        if identifier in index['clauses']:
            fragments.append(index['clauses'][identifier]['fragment_id'])
        else:
            match = re.fullmatch(r'(.+F\d{5})C\d+', identifier)
            if match is None or match[1] not in index['fragments']:
                return None
            fragments.append(match[1])
    if fragments[0] != fragments[1]:
        return None
    return fragments[0], valid


def location_targets(value: dict[str, Any], index: dict[str, Any]) -> list[dict[str, Any]]:
    targets = []
    for row in pointer_rows(value, index):
        identity = identifiable_fragment(row['pointer'], index)
        if identity is None:
            continue
        fragment_id, fixed = identity
        fragment = index['fragments'][fragment_id]
        if fragment['kind'] != row['kind']:
            continue
        if row['kind'] == 'source' and (row['pointer']['source_id'] != fragment['source_id']
                or row['pointer']['source_field'] != fragment['source_field']):
            continue
        clauses = [{'clause_id': identifier, 'text': fragment['text'][entry['start']:entry['end']]}
                   for identifier, entry in sorted(index['clauses'].items(), key=lambda item: item[1]['ordinal'])
                   if entry['fragment_id'] == fragment_id]
        targets.append({**row, 'target_id': f"L{len(targets):04d}", 'fixed_endpoints': fixed,
                        'fragment_id': fragment_id, 'fragment': {
                            key: value for key, value in fragment.items() if key != 'text'}, 'clauses': clauses})
    return targets


def apply_pointer_corrections(value: dict[str, Any], targets: list[dict[str, Any]],
                              response: dict[str, Any], index: dict[str, Any]) -> dict[str, Any]:
    corrections = LocationCorrections.model_validate(response).model_dump()['corrections']
    by_id = {row['target_id']: row for row in corrections}
    if len(by_id) != len(corrections) or set(by_id) != {row['target_id'] for row in targets}:
        raise ValueError('Endpoint correction must answer every target exactly once')
    fixed_value, audit = copy.deepcopy(value), []
    for target in targets:
        correction = by_id[target['target_id']]
        accepted = correction['present'] and correction['same_assertion_and_scope'] and correction['confidence'] == 'high'
        span = None
        if accepted:
            updated = {**target['pointer'], **{k: correction[k] for k in ('first_clause_id', 'last_clause_id')}}
            if any(updated[key] != fixed for key, fixed in target['fixed_endpoints'].items()):
                raise ValueError('Endpoint correction changed an already valid fixed endpoint')
            if any(index['clauses'].get(updated[key], {}).get('fragment_id') != target['fragment_id']
                   for key in ('first_clause_id', 'last_clause_id')):
                raise ValueError('Endpoint correction must stay in the original fragment')
            span = resolve_span(updated, index, target['kind'])
            container = fixed_value
            for part in target['path'][:-1]:
                container = container[part]
            container[target['path'][-1]] = updated
        audit.append({'target_id': target['target_id'], 'path': target['path'],
                      'original_pointer': target['pointer'], 'correction': correction,
                      'fixed_endpoints': target['fixed_endpoints'], 'accepted': accepted, 'span': span})
    return {'value': fixed_value, 'audit': {'targets': audit,
             'repaired_count': sum(row['accepted'] for row in audit),
             'scientific_judgments_and_scores_changed': False}}


async def repair_pointer_locations(scheduler: Scheduler, key: str, raw: dict[str, Any],
                                    index: dict[str, Any]) -> dict[str, Any]:
    value = PointerEvaluation.model_validate(raw).model_dump(mode='json')
    targets = location_targets(value, index)
    if not targets:
        return {'value': value, 'audit': {'targets': [], 'repaired_count': 0,
                                        'scientific_judgments_and_scores_changed': False}}
    payload = {'targets': targets}
    digest = fingerprint(payload)
    destination = scheduler.output / 'audits/pointer_location' / key / f'{digest}.json'
    write(destination, {'state': 'requested', 'original_raw': raw, 'request': payload})
    response = await scheduler.ask(key + '/pointer_location', LOCATION_INSTRUCTION, payload,
                                   LocationCorrections, 'xhigh')
    # Record the raw repair even if it fails the fixed-source or fixed-endpoint checks.
    write(destination, {'state': 'returned', 'original_raw': raw, 'request': payload, 'response': response})
    result = apply_pointer_corrections(value, targets, response, index)
    result['audit']['request_fingerprint'] = digest
    write(destination, {'state': 'resolved', 'original_raw': raw, 'request': payload,
                        'response': response, **result})
    return result
