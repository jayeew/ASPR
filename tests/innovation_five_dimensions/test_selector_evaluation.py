from __future__ import annotations

import copy
from typing import Any

import jsonschema
import pytest
from pydantic import ValidationError

from gear.codex_cli import _strict_response_schema
from figure_pipeline.innovation_five_dimensions.config import DIMENSIONS
from figure_pipeline.innovation_five_dimensions.evaluation import validate_evaluation
from figure_pipeline.innovation_five_dimensions.selector_evaluation import (
    EVALUATION_PROTOCOL,
    prepare_selector_payload,
    resolve_selector_evaluation,
    selector_schema,
)


def anonymous() -> dict[str, Any]:
    evidence = {'source_id': 'SOURCE_A', 'quote': 'A was observed under B.', 'explanation': 'An observation.'}
    return {'paper': {'paper_id': 'P1'},
            'report': {'contribution_overview': 'A was observed.', 'synthesis': 'Only under B.',
                       'dimensions': [{'dimension': dim, 'judgment': 'A was observed.',
                                       'rationale': 'A is measured.', 'scope': 'Only under B.',
                                       'evidence': [evidence], 'status': 'mixed'} for dim in DIMENSIONS]},
            'checklist': {'dimensions': [{'dimension': dim, 'applicability': 'applicable',
                                         'criteria': ['Discuss A.'], 'scope': 'Only under B.',
                                         'reason': 'Relevant.', 'evidence': [evidence]} for dim in DIMENSIONS]},
            'source_passages': [{'source_id': 'SOURCE_A', 'source_type': 'manuscript', 'start': 100,
                                 'title': 'A measured phenomenon.', 'text': 'A was observed under B. A second observation.'},
                                {'source_id': 'SOURCE_A', 'source_type': 'manuscript', 'start': 1000,
                                 'text': 'A distant observation.'},
                                {'source_id': 'SOURCE_B', 'source_type': 'abstract',
                                 'text': 'A historical finding.'}],
            'citation_audit': {'eligible_for_scientific_evaluation': True}}


def raw_evaluation() -> dict[str, Any]:
    return {'dimensions': [{'dimension': dim, 'status': 'assessed', 'score': 2,
                            'report_quotes': ['R01'], 'reason': 'Specific, with omissions.',
                            'checks': [{'report_quote': 'R01', 'support': 'partly_supported',
                                        'scope_correct': False, 'reason': 'The condition is missing.',
                                        'evidence': [{'source_quote_ids': [{'fragment_id': 'S001', 'clause_index': 0}],
                                                      'explanation': 'The measured condition.'}]}]} for dim in DIMENSIONS]}


def schema_fixture() -> tuple[dict, dict, dict]:
    payload, index = prepare_selector_payload(anonymous())
    return payload, index, _strict_response_schema(selector_schema(index).model_json_schema())


def test_preparation_preserves_science_and_uses_short_schema_selector_ids() -> None:
    original = anonymous()
    unchanged = copy.deepcopy(original)
    payload, index, _ = schema_fixture()
    assert prepare_selector_payload(original) == (payload, index)
    assert original == unchanged
    assert payload['quotation_protocol'] == EVALUATION_PROTOCOL
    assert payload['report']['contribution_overview'] == {'fragment_id': 'R01', 'text': 'A was observed.'}
    assert list(index['selector_reports']) == [f'R{i:02d}' for i in range(1, 18)]
    for old, new in zip(original['report']['dimensions'], payload['report']['dimensions']):
        for key in ('judgment', 'rationale', 'scope'):
            assert new[key]['text'] == old[key]
        assert new['evidence'] == old['evidence'] and new['status'] == old['status']
    bodies = [s for s in payload['source_passages'] if s['source_field'] == 'text']
    for old, new in zip(original['source_passages'], bodies):
        assert ''.join(c['text'] for c in new['clauses']) == old['text']
        assert 'text' not in new
        assert [c['clause_index'] for c in new['clauses']] == list(range(len(new['clauses'])))
    assert payload['checklist'] == original['checklist']


def test_resolution_keeps_all_scores_labels_and_explanations() -> None:
    original = anonymous()
    _, index, shape = schema_fixture()
    raw = raw_evaluation()
    unchanged = copy.deepcopy(raw)
    jsonschema.validate(raw, shape)
    result = resolve_selector_evaluation(raw, index)
    assert raw == unchanged
    value = result['value']
    validate_evaluation(value, original['report'], original['source_passages'], original['checklist'])
    for before, after in zip(raw['dimensions'], value['dimensions']):
        for key in ('dimension', 'status', 'score', 'reason'):
            assert before[key] == after[key]
        for key in ('support', 'scope_correct', 'reason'):
            assert before['checks'][0][key] == after['checks'][0][key]
        assert after['checks'][0]['evidence'][0]['explanation'] == before['checks'][0]['evidence'][0]['explanation']
    assert result['audit']['scientific_judgments_and_scores_changed'] is False
    evidence = value['dimensions'][0]['checks'][0]['evidence'][0]
    assert evidence == {'source_id': 'SOURCE_A', 'quote': 'A was observed under B. ', 'explanation': 'The measured condition.'}


@pytest.mark.parametrize('case', ['unknown_report', 'report_clause_as_field', 'unknown_source', 'negative', 'out_of_range', 'source_id_added'])
def test_invalid_ids_and_source_specific_bounds_are_rejected_by_actual_json_schema(case: str) -> None:
    _, index, shape = schema_fixture()
    raw = raw_evaluation()
    check = raw['dimensions'][0]['checks'][0]
    source = check['evidence'][0]['source_quote_ids'][0]
    if case == 'unknown_report': check['report_quote'] = 'R99'
    elif case == 'report_clause_as_field': check['report_quote'] = 'QP_0_F00000C0000'
    elif case == 'unknown_source': source['fragment_id'] = 'S999'
    elif case == 'negative': source['clause_index'] = -1
    elif case == 'out_of_range': source['clause_index'] = 2
    else: source['source_id'] = 'SOURCE_B'
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw, shape)
    with pytest.raises(ValidationError):
        resolve_selector_evaluation(raw, index)


@pytest.mark.parametrize('support', ['supported', 'partly_supported', 'contradicted'])
@pytest.mark.parametrize('empty', ['evidence', 'source_quote_ids'])
def test_source_backed_labels_require_nonempty_evidence_in_json_schema(support: str, empty: str) -> None:
    _, index, shape = schema_fixture()
    raw = raw_evaluation()
    check = raw['dimensions'][0]['checks'][0]
    check['support'] = support
    if empty == 'evidence': check['evidence'] = []
    else: check['evidence'][0]['source_quote_ids'] = []
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw, shape)
    with pytest.raises(ValidationError):
        resolve_selector_evaluation(raw, index)


def test_unverifiable_can_have_no_evidence_without_changing_label() -> None:
    _, index, shape = schema_fixture()
    raw = raw_evaluation()
    raw['dimensions'][0]['checks'][0].update(support='not_verifiable', evidence=[])
    jsonschema.validate(raw, shape)
    value = resolve_selector_evaluation(raw, index)['value']
    assert value['dimensions'][0]['checks'][0]['support'] == 'not_verifiable'
    assert value['dimensions'][0]['checks'][0]['evidence'] == []


def test_noncontinuous_and_different_source_clauses_are_separate_evidence_not_joined() -> None:
    _, index, _ = schema_fixture()
    raw = raw_evaluation()
    raw['dimensions'][0]['checks'][0]['evidence'][0]['source_quote_ids'] = [
        {'fragment_id': 'S001', 'clause_index': 0}, {'fragment_id': 'S003', 'clause_index': 0},
        {'fragment_id': 'S004', 'clause_index': 0}]
    value = resolve_selector_evaluation(raw, index)['value']
    evidence = value['dimensions'][0]['checks'][0]['evidence']
    assert len(evidence) == 3
    assert [e['quote'] for e in evidence] == ['A was observed under B. ', 'A distant observation.', 'A historical finding.']
    assert [e['source_id'] for e in evidence] == ['SOURCE_A', 'SOURCE_A', 'SOURCE_B']


def test_title_is_an_independent_source_selection_with_no_body_offset() -> None:
    _, index, _ = schema_fixture()
    raw = raw_evaluation()
    raw['dimensions'][0]['checks'][0]['evidence'][0]['source_quote_ids'] = [{'fragment_id': 'S002', 'clause_index': 0}]
    result = resolve_selector_evaluation(raw, index)
    title = next(a for a in result['audit']['quotations'] if a.get('selector_fragment_id') == 'S002')
    assert title['source_field'] == 'title' and title['title_only'] is True
    assert title['source_start'] is None and title['source_end'] is None
    assert result['value']['dimensions'][0]['checks'][0]['evidence'][0]['quote'] == 'A measured phenomenon.'


def test_zero_missing_answer_and_shared_unresolved_retain_distinct_meanings() -> None:
    original = anonymous()
    _, index = prepare_selector_payload(original)
    raw = raw_evaluation()
    raw['dimensions'][0].update(score=0, report_quotes=[], checks=[], reason='No answer to the applicable issue.')
    raw['dimensions'][1].update(status='unresolved', score=None, report_quotes=[], checks=[])
    original['checklist']['dimensions'][1]['applicability'] = 'unresolved'
    value = resolve_selector_evaluation(raw, index)['value']
    validate_evaluation(value, original['report'], original['source_passages'], original['checklist'])
    assert value['dimensions'][0]['score'] == 0 and value['dimensions'][1]['score'] is None


def test_over_1000_source_clauses_use_bounded_branches_without_large_enum() -> None:
    original = anonymous()
    original['source_passages'][0]['text'] = ''.join(f'Sentence {i}. ' for i in range(1201))
    payload, index = prepare_selector_payload(original)
    assert len(payload['source_passages'][0]['clauses']) == 1201
    shape = _strict_response_schema(selector_schema(index).model_json_schema())
    def enum_count(value: Any) -> int:
        if isinstance(value, dict):
            return len(value.get('enum', [])) + sum(enum_count(v) for k, v in value.items() if k != 'enum')
        if isinstance(value, list): return sum(enum_count(v) for v in value)
        return 0
    assert enum_count(shape) < 100
    raw = raw_evaluation()
    raw['dimensions'][0]['checks'][0]['evidence'][0]['source_quote_ids'][0]['clause_index'] = 1200
    jsonschema.validate(raw, shape)
    value = resolve_selector_evaluation(raw, index)['value']
    assert value['dimensions'][0]['checks'][0]['evidence'][0]['quote'] == 'Sentence 1200. '
    raw['dimensions'][0]['checks'][0]['evidence'][0]['source_quote_ids'][0]['clause_index'] = 1201
    with pytest.raises(jsonschema.ValidationError): jsonschema.validate(raw, shape)
