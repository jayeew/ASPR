"""Schema-enumerated report fields and discrete source clauses, with unchanged scientific scoring."""
from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field, create_model, model_validator

from .models import DimensionEvaluation, Evaluation, Record
from .pointer_evaluation import prepare_evaluation_payload, resolve_span

EVALUATION_PROTOCOL = 'source_clause_and_report_field_selectors_v3'

SELECTOR_INSTRUCTION = '''
QUOTATION OUTPUT CONTRACT: Keep the scientific rubric above unchanged. Select quotations using ONLY the
legal identifiers enumerated by the response schema. Do not write quotations, source IDs or ranges yourself.
report_quote is ONE short report field ID (R01 etc.), chosen from the actual contribution_overview, synthesis or
five dimensions' judgment/rationale/scope fields. report_quotes is a list of these report field IDs.
The program copies each selected field exactly. Report evidence quotations are visible for scrutiny but
are not selectable as report prose. A field can contain several assertions: explain in each check.reason
which actual assertion you are checking. You may check different assertions in the same field separately;
do not assign one favourable support label to a field that also contains an unsupported assertion.
Source evidence entries contain source_quote_ids and explanation. Each item in source_quote_ids is an object
with fragment_id (S001 etc.) and clause_index (a zero-based integer within that fragment's actual listed range).
Select one or more such objects; every allowed fragment and its index bound are enforced by the schema.
Each selected clause becomes its own exact original quotation with its actual source and text/title field
recorded by software. Multiple clauses, including clauses from different sources, may be selected where
needed, but they are NEVER joined into a purported continuous quotation. Select all clauses needed for the
actual support argument and its qualifications, including adjacent clauses where a statement spans them.
Do not invent endpoints, source identities, indices or quotation strings. There is no need to reproduce PDF typography.
A check labelled supported, partly_supported or contradicted MUST select at least one nonempty source
quotation. If no supplied source can establish support or contradiction for the assertion, use not_verifiable
with an empty evidence list when appropriate. Do not choose unrelated text to satisfy the schema. Absence
of a source is not evidence of contradiction or lack of novelty. A visible bibliographic title is not full-text
verification of a scientific result; graph topology is not automatically scientific causality or priority.
Location and scientific support are distinct. Preserve accurate negative, mixed or uncertain judgments.
The rubric gives no reward for positive language, many quotations, graph terms or report length. Assess
accuracy, completeness and grounding of the report, not the novelty magnitude of its target paper.
An absent answer to an applicable question is scored 0 and may have empty report_quotes/checks; explain
its omission rather than selecting an unrelated field. Other assessed scores require actual report evidence.
Use the common checklist's applicability. Do not turn an unsupported report into an unresolved dimension.
Return all five dimensions. Do not make a total score, editorial decision or claim of human validation.
Do not call tools or use outside material. This selector contract replaces earlier free-quotation/range
instructions only; scientific meanings of score, support, scope_correct, explanation and reason are unchanged.
'''


def prepare_selector_payload(anonymous: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, index = prepare_evaluation_payload(anonymous)
    index['selector_reports'], index['selector_sources'] = {}, {}
    report = payload['report']
    fields = [report[key] for key in ('contribution_overview', 'synthesis')]
    fields.extend(row[key] for row in report['dimensions'] for key in ('judgment', 'rationale', 'scope'))
    for number, field in enumerate(fields, 1):
        alias = f'R{number:02d}'
        index['selector_reports'][alias] = field['fragment_id']
        field['fragment_id'] = alias
        field['text'] = ''.join(clause['text'] for clause in field.pop('clauses'))
    for number, source in enumerate(payload['source_passages'], 1):
        alias = f'S{number:03d}'
        index['selector_sources'][alias] = {'fragment_id': source['fragment_id'],
                                           'clause_ids': [clause['clause_id'] for clause in source['clauses']]}
        source['fragment_id'] = alias
        source['clauses'] = [{'clause_index': i, 'text': clause['text']}
                             for i, clause in enumerate(source['clauses'])]
    payload['quotation_protocol'] = EVALUATION_PROTOCOL
    return payload, index


def selector_schema(index: dict[str, Any], allowed_dimensions: tuple[str, ...] | None = None) -> type[BaseModel]:
    report_ids = tuple(index['selector_reports'])
    choices = []
    for alias, source in index['selector_sources'].items():
        if source['clause_ids']:
            choices.append(create_model(f'SourceClause{alias}', __base__=Record,
                                        fragment_id=(Literal[alias], ...),
                                        clause_index=(int, Field(ge=0, le=len(source['clause_ids']) - 1))))
    if not report_ids or not choices:
        raise ValueError('Selector evaluation requires actual report fields and source clauses')
    report_id = Literal[report_ids]
    source_id = Union[tuple(choices)]
    evidence = create_model('SelectorEvidence', __base__=Record,
                            source_quote_ids=(list[source_id], Field(min_length=1)),
                            explanation=(str, ...))
    common = {'report_quote': (report_id, ...), 'scope_correct': (bool | None, ...), 'reason': (str, ...)}
    grounded = create_model('SourceBackedCheck', __base__=Record, **common,
                            support=(Literal['supported', 'partly_supported', 'contradicted'], ...),
                            evidence=(list[evidence], Field(min_length=1)))
    unverifiable = create_model('UnverifiableCheck', __base__=Record, **common,
                                support=(Literal['not_verifiable'], ...), evidence=(list[evidence], ...))
    check = Union[grounded, unverifiable]
    dimension = create_model('SelectorDimensionEvaluation', __base__=DimensionEvaluation,
                             checks=(list[check], ...), report_quotes=(list[report_id], ...))
    if allowed_dimensions is not None:
        if not allowed_dimensions or len(set(allowed_dimensions)) != len(allowed_dimensions):
            raise ValueError('Partial selector requires a nonempty unique dimension subset')
        partial_dimension = create_model('PartialSelectorDimension', __base__=dimension,
                                         dimension=(Literal[allowed_dimensions], ...))
        def exact_subset(self: BaseModel) -> BaseModel:
            actual = [row.dimension for row in self.dimensions]
            if len(actual) != len(allowed_dimensions) or set(actual) != set(allowed_dimensions):
                raise ValueError('Exactly the independently available reference dimensions are required')
            return self
        return create_model('PartialSelectorEvaluation', __base__=Record,
                            __validators__={'exact_subset': model_validator(mode='after')(exact_subset)},
                            dimensions=(list[partial_dimension], Field(min_length=len(allowed_dimensions),
                                                                      max_length=len(allowed_dimensions))))
    return create_model('SelectorEvaluation', __base__=Evaluation,
                        dimensions=(list[dimension], Field(min_length=5, max_length=5)))


def report_field(identifier: str, index: dict[str, Any]) -> dict[str, Any]:
    original_id = index['selector_reports'].get(identifier)
    fragment = index['fragments'].get(original_id)
    if fragment is None or fragment['kind'] != 'report':
        raise ValueError('Report selector must name an actual report prose field')
    return {'kind': 'report', 'selector_id': identifier, 'fragment_id': original_id, 'text': fragment['text'],
            'report_field': fragment['field'], 'start': 0, 'end': len(fragment['text']),
            'source_id': None, 'source_field': None, 'source_start': None, 'source_end': None,
            'title_only': False}


def source_clause(selected: dict[str, Any], index: dict[str, Any]) -> dict[str, Any]:
    source = index['selector_sources'].get(selected['fragment_id'])
    position = selected['clause_index']
    if source is None or not isinstance(position, int) or not 0 <= position < len(source['clause_ids']):
        raise ValueError('Source selector must name an actual fragment and in-range clause')
    identifier = source['clause_ids'][position]
    clause = index['clauses'][identifier]
    fragment = index['fragments'][clause['fragment_id']]
    if fragment['kind'] != 'source':
        raise ValueError('Report prose cannot substitute for original-source evidence')
    span = resolve_span({'first_clause_id': identifier, 'last_clause_id': identifier,
                         'source_id': fragment['source_id'], 'source_field': fragment['source_field']},
                        index, 'source')
    return {**span, 'selector_fragment_id': selected['fragment_id'], 'selector_clause_index': position}


def resolve_selector_evaluation(raw: dict[str, Any], index: dict[str, Any],
                                technical_failures: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    from .config import DIMENSIONS
    failed = {row['dimension']: row for row in technical_failures or []}
    allowed = tuple(d for d in DIMENSIONS if d not in failed) if technical_failures else None
    if allowed == ():
        if raw != {'dimensions': []}:
            raise ValueError('No scientific evaluation is permitted when all reference dimensions failed')
        value = {'dimensions': []}
    else:
        value = selector_schema(index, allowed).model_validate(raw).model_dump(mode='json')
    audit = []
    for dimension in value['dimensions']:
        quotes = []
        for number, identifier in enumerate(dimension['report_quotes']):
            span = report_field(identifier, index)
            quotes.append(span['text'])
            audit.append({'dimension': dimension['dimension'], 'path': f'report_quotes[{number}]', **span})
        dimension['report_quotes'] = quotes
        for number, check in enumerate(dimension['checks']):
            span = report_field(check['report_quote'], index)
            check['report_quote'] = span['text']
            audit.append({'dimension': dimension['dimension'], 'path': f'checks[{number}].report_quote', **span})
            evidence = []
            for group_number, selected in enumerate(check['evidence']):
                for item_number, identifier in enumerate(selected['source_quote_ids']):
                    span = source_clause(identifier, index)
                    evidence.append({'source_id': span['source_id'], 'quote': span['text'],
                                     'explanation': selected['explanation']})
                    audit.append({'dimension': dimension['dimension'],
                                  'path': f'checks[{number}].evidence[{len(evidence)-1}]',
                                  'selection_group': group_number, 'selection_index': item_number,
                                  **span})
            check['evidence'] = evidence
    for failure in technical_failures or []:
        value['dimensions'].append({'dimension': failure['dimension'], 'status': 'technical_failure', 'score': None,
                                    'checks': [], 'report_quotes': [], 'reason': failure['reason'], 'origin': failure['origin']})
    if technical_failures:
        value['dimensions'].sort(key=lambda row: list(DIMENSIONS).index(row['dimension']))
    result = Evaluation.model_validate(value).model_dump(mode='json')
    return {'value': result, 'audit': {'quotation_protocol': EVALUATION_PROTOCOL, 'quotations': audit,
                                     'quotation_count': len(audit), 'scientific_judgments_and_scores_changed': False,
                                     'meaning': 'Exact selector-to-text resolution; scientific relevance and support remain model judgments.'}}
