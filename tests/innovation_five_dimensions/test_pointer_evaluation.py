from __future__ import annotations

import copy
import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from figure_pipeline.innovation_five_dimensions.config import DIMENSIONS
from figure_pipeline.innovation_five_dimensions.evaluation import validate_evaluation
from figure_pipeline.innovation_five_dimensions.pointer_evaluation import (
    EVALUATION_PROTOCOL,
    PointerEvaluation,
    apply_pointer_corrections,
    location_targets,
    prepare_evaluation_payload,
    repair_pointer_locations,
    resolve_evaluation,
    resolve_span,
)


def anonymous() -> dict[str, Any]:
    evidence = {'source_id': 'S1', 'quote': 'A was observed under B.', 'explanation': 'Direct experiment.'}
    report = {'contribution_overview': 'A was observed.', 'synthesis': 'Only under B.',
              'dimensions': [{'dimension': dim, 'judgment': 'A was observed.',
                              'rationale': 'This advances understanding of A.', 'scope': 'Only under B.',
                              'evidence': [evidence], 'status': 'supported'} for dim in DIMENSIONS]}
    checklist = {'dimensions': [{'dimension': dim, 'applicability': 'applicable',
                                'criteria': ['Discuss A.'], 'scope': 'Only under B.', 'reason': 'Main issue.',
                                'evidence': [evidence]} for dim in DIMENSIONS]}
    return {'paper': {'paper_id': 'P1'}, 'report': report, 'checklist': checklist,
            'source_passages': [{'source_id': 'S1', 'source_type': 'manuscript',
                                 'title': 'A new measurement.', 'start': 100,
                                 'text': 'A was observed under B. It was measured twice.'},
                                {'source_id': 'S1', 'source_type': 'manuscript',
                                 'title': 'A new measurement.', 'start': 1000,
                                 'text': 'Another disjoint passage.'},
                                {'source_id': 'S2', 'source_type': 'abstract',
                                 'text': 'A has also been observed elsewhere.'}],
            'citation_audit': {'eligible_for_scientific_evaluation': True}}


def pointer(field: dict[str, Any], first: int = 0, last: int = 0) -> dict[str, str]:
    return {'first_clause_id': field['clauses'][first]['clause_id'],
            'last_clause_id': field['clauses'][last]['clause_id']}


def raw_evaluation(payload: dict[str, Any]) -> dict[str, Any]:
    report_pointer = pointer(payload['report']['contribution_overview'])
    source = payload['source_passages'][0]
    source_pointer = {**pointer(source), 'source_id': 'S1', 'source_field': 'text', 'explanation': 'Direct experiment.'}
    return {'dimensions': [{'dimension': dim, 'status': 'assessed', 'score': 2,
                            'report_quotes': [copy.deepcopy(report_pointer)], 'reason': 'Specific but incomplete.',
                            'checks': [{'report_quote': copy.deepcopy(report_pointer), 'support': 'supported',
                                        'scope_correct': True, 'evidence': [copy.deepcopy(source_pointer)],
                                        'reason': 'Source reports the actual measurement.'}]} for dim in DIMENSIONS]}


def test_prepare_preserves_all_scientific_text_without_duplicating_sources() -> None:
    original = anonymous()
    unchanged = copy.deepcopy(original)
    payload, index = prepare_evaluation_payload(original)
    assert original == unchanged
    assert payload['quotation_protocol'] == EVALUATION_PROTOCOL
    assert payload['checklist'] == original['checklist']
    assert payload['citation_audit'] == original['citation_audit']
    for old, new in zip(original['report']['dimensions'], payload['report']['dimensions']):
        assert old['evidence'] == new['evidence'] and old['status'] == new['status']
        for field in ('judgment', 'rationale', 'scope'):
            assert ''.join(c['text'] for c in new[field]['clauses']) == old[field]
    text_fields = [f for f in payload['source_passages'] if f['source_field'] == 'text']
    title_fields = [f for f in payload['source_passages'] if f['source_field'] == 'title']
    assert len(title_fields) == 1  # Shared visible title is represented once, not once per source fragment.
    for old, new in zip(original['source_passages'], text_fields):
        assert 'text' not in new
        assert ''.join(c['text'] for c in new['clauses']) == old['text']
        assert old['source_type'] == new['source_type']
    assert not any('evidence' in f.get('field', '') for f in index['fragments'].values())
    assert prepare_evaluation_payload(original) == (payload, index)


def test_resolution_changes_only_quote_representation_and_passes_existing_validation() -> None:
    original = anonymous()
    payload, index = prepare_evaluation_payload(original)
    raw = raw_evaluation(payload)
    unchanged = copy.deepcopy(raw)
    result = resolve_evaluation(raw, index)
    assert raw == unchanged
    value = result['value']
    validate_evaluation(value, original['report'], original['source_passages'], original['checklist'])
    for before, after in zip(raw['dimensions'], value['dimensions']):
        for key in ('dimension', 'status', 'score', 'reason'):
            assert before[key] == after[key]
        for key in ('support', 'scope_correct', 'reason'):
            assert before['checks'][0][key] == after['checks'][0][key]
        assert after['report_quotes'] == ['A was observed.']
        assert after['checks'][0]['evidence'][0]['quote'] == 'A was observed under B. '
    assert result['audit']['scientific_judgments_and_scores_changed'] is False
    source_audit = next(a for a in result['audit']['quotations'] if a['kind'] == 'source')
    assert source_audit['source_start'] == 100


def test_source_range_copies_every_intermediate_clause_and_preserves_pdf_text() -> None:
    original = anonymous()
    text = 'A ﬁbre was trans-\nmitted. 109\nAnother sentence. Final finding.'
    original['source_passages'][0]['text'] = text
    payload, index = prepare_evaluation_payload(original)
    source = payload['source_passages'][0]
    selected = {**pointer(source, 0, len(source['clauses']) - 1), 'source_id': 'S1', 'source_field': 'text'}
    span = resolve_span(selected, index, 'source')
    assert span['text'] == text
    assert span['source_end'] == 100 + len(text)


@pytest.mark.parametrize('case', ['different_fragment', 'different_source', 'different_field',
                                 'unknown', 'reverse', 'wrong_source_id', 'wrong_source_field'])
def test_invalid_source_ranges_are_technical_failures(case: str) -> None:
    payload, index = prepare_evaluation_payload(anonymous())
    source = payload['source_passages'][0]
    selected = {**pointer(source), 'source_id': 'S1', 'source_field': 'text'}
    fragments = payload['source_passages']
    if case == 'different_fragment':
        selected['last_clause_id'] = pointer(fragments[2])['last_clause_id']
    elif case == 'different_source':
        selected['last_clause_id'] = pointer(fragments[3])['last_clause_id']
    elif case == 'different_field':
        selected['last_clause_id'] = pointer(fragments[1])['last_clause_id']
    elif case == 'unknown':
        selected['first_clause_id'] = 'INVENTED'
    elif case == 'reverse':
        selected.update(pointer(source, 1, 0))
    elif case == 'wrong_source_id':
        selected['source_id'] = 'S2'
    else:
        selected['source_field'] = 'title'
    with pytest.raises(ValueError):
        resolve_span(selected, index, 'source')


def test_report_cannot_quote_source_text_or_cross_report_fields() -> None:
    payload, index = prepare_evaluation_payload(anonymous())
    with pytest.raises(ValueError, match='cannot substitute'):
        resolve_span(pointer(payload['source_passages'][0]), index, 'report')
    bad = pointer(payload['report']['contribution_overview'])
    bad['last_clause_id'] = pointer(payload['report']['synthesis'])['last_clause_id']
    with pytest.raises(ValueError, match='one continuous dimension'):
        resolve_span(bad, index, 'report')


def test_title_pointer_remains_explicit_and_does_not_claim_body_offsets() -> None:
    payload, index = prepare_evaluation_payload(anonymous())
    title = next(f for f in payload['source_passages'] if f['source_field'] == 'title')
    selected = {**pointer(title), 'source_id': 'S1', 'source_field': 'title'}
    span = resolve_span(selected, index, 'source')
    assert span['text'] == 'A new measurement.'
    assert span['title_only'] is True and span['source_type'] == 'manuscript'
    assert span['source_start'] is None and span['source_end'] is None


def test_missing_answer_stays_zero_and_common_unresolved_stays_null() -> None:
    original = anonymous()
    payload, index = prepare_evaluation_payload(original)
    raw = raw_evaluation(payload)
    raw['dimensions'][0].update(score=0, report_quotes=[], checks=[], reason='The applicable issue is absent.')
    raw['dimensions'][1].update(score=None, status='unresolved', report_quotes=[], checks=[])
    original['checklist']['dimensions'][1]['applicability'] = 'unresolved'
    value = resolve_evaluation(raw, index)['value']
    validate_evaluation(value, original['report'], original['source_passages'], original['checklist'])
    assert value['dimensions'][0]['score'] == 0
    assert value['dimensions'][1]['score'] is None


def test_schema_rejects_summarised_report_quote_and_missing_dimension() -> None:
    payload, _ = prepare_evaluation_payload(anonymous())
    raw = raw_evaluation(payload)
    raw['dimensions'][0]['checks'][0]['report_quote'] = 'A summary the report never said.'
    with pytest.raises(ValidationError):
        PointerEvaluation.model_validate(raw)
    with pytest.raises(ValidationError, match='Exactly one'):
        PointerEvaluation.model_validate({'dimensions': []})


def test_namespace_collision_is_resolved_without_changing_report_science() -> None:
    original = anonymous()
    original['report']['synthesis'] = 'The code QP_0_ was observed.'
    payload, index = prepare_evaluation_payload(original)
    assert index['namespace'] == 'QP_1_'
    assert ''.join(c['text'] for c in payload['report']['synthesis']['clauses']) == original['report']['synthesis']


def test_same_dimension_range_is_real_rendered_prose_with_fixed_labels() -> None:
    from figure_pipeline.innovation_five_dimensions.evaluation import report_quote_audit, report_fragments
    from figure_pipeline.innovation_five_dimensions.reporting import render_report
    original = anonymous()
    payload, index = prepare_evaluation_payload(original)
    row = payload['report']['dimensions'][0]
    selected = pointer(row['judgment'])
    selected['last_clause_id'] = pointer(row['scope'])['last_clause_id']
    span = resolve_span(selected, index, 'report')
    assert span['text'] == ('A was observed.\n\nThis advances understanding of A.\n\n'
                            '**成立范围与判断状态：** Only under B.')
    assert span['text'] in render_report(original['report'])
    assert report_quote_audit(span['text'], report_fragments(original['report']))['located']
    assert 'rendered_prose' in span['report_field']


@pytest.mark.parametrize('case', ['reverse_same_dimension', 'cross_dimension'])
def test_rendered_prose_does_not_permit_reverse_or_cross_dimension(case: str) -> None:
    payload, index = prepare_evaluation_payload(anonymous())
    left, right = payload['report']['dimensions'][:2]
    selected = pointer(left['scope'] if case.startswith('reverse') else left['judgment'])
    selected['last_clause_id'] = pointer(left['judgment'] if case.startswith('reverse') else right['scope'])['last_clause_id']
    with pytest.raises(ValueError, match='one continuous dimension'):
        resolve_span(selected, index, 'report')


def invalid_endpoint_fixture() -> tuple[dict, dict, dict, list[dict]]:
    payload, index = prepare_evaluation_payload(anonymous())
    value = raw_evaluation(payload)
    selected = value['dimensions'][0]['checks'][0]['evidence'][0]
    selected['last_clause_id'] = selected['last_clause_id'].rsplit('C', 1)[0] + 'C9999'
    targets = location_targets(value, index)
    return payload, index, value, targets


def correction(target: dict, endpoint: str | None, present: bool = True) -> dict:
    return {'target_id': target['target_id'], 'present': present,
            'first_clause_id': target['pointer']['first_clause_id'] if present else None,
            'last_clause_id': endpoint if present else None,
            'same_assertion_and_scope': present, 'confidence': 'high', 'reason': 'Original intended assertion.'}


def test_endpoint_repair_preserves_source_known_endpoint_and_all_scientific_values() -> None:
    payload, index, value, targets = invalid_endpoint_fixture()
    assert len(targets) == 1
    assert targets[0]['context']['report_assertion'] == 'A was observed.'
    assert targets[0]['fixed_endpoints'] == {'first_clause_id': targets[0]['pointer']['first_clause_id']}
    endpoint = pointer(payload['source_passages'][0])['last_clause_id']
    response = {'corrections': [correction(targets[0], endpoint)]}
    result = apply_pointer_corrections(value, targets, response, index)
    expected = copy.deepcopy(value)
    expected['dimensions'][0]['checks'][0]['evidence'][0]['last_clause_id'] = endpoint
    assert result['value'] == expected
    assert result['audit']['repaired_count'] == 1
    resolved = resolve_evaluation(result['value'], index)
    assert resolved['value']['dimensions'][0]['score'] == value['dimensions'][0]['score']
    assert resolved['value']['dimensions'][0]['checks'][0]['support'] == 'supported'


@pytest.mark.parametrize('case', ['fixed_endpoint', 'other_fragment', 'unlocatable'])
def test_endpoint_repair_cannot_change_anchor_or_choose_another_fragment(case: str) -> None:
    payload, index, value, targets = invalid_endpoint_fixture()
    endpoint = pointer(payload['source_passages'][0])['last_clause_id']
    response = correction(targets[0], endpoint)
    if case == 'fixed_endpoint':
        response['first_clause_id'] = pointer(payload['source_passages'][0], 1, 1)['first_clause_id']
    elif case == 'other_fragment':
        response['last_clause_id'] = pointer(payload['source_passages'][2])['last_clause_id']
    else:
        response = correction(targets[0], None, False)
    if case == 'unlocatable':
        result = apply_pointer_corrections(value, targets, {'corrections': [response]}, index)
        assert result['value'] == value and result['audit']['repaired_count'] == 0
        with pytest.raises(ValueError, match='Unknown'):
            resolve_evaluation(result['value'], index)
    else:
        with pytest.raises(ValueError):
            apply_pointer_corrections(value, targets, {'corrections': [response]}, index)


def test_endpoint_repair_does_not_fix_real_cross_fragment_or_empty_evidence_errors() -> None:
    payload, index = prepare_evaluation_payload(anonymous())
    value = raw_evaluation(payload)
    value['dimensions'][0]['checks'][0]['evidence'] = []
    item = value['dimensions'][1]['checks'][0]['evidence'][0]
    item['last_clause_id'] = pointer(payload['source_passages'][2])['last_clause_id']
    assert location_targets(value, index) == []
    item['last_clause_id'] = 'MADE_UP_FRAGMENT_C9999'
    assert location_targets(value, index) == []


def test_endpoint_location_uses_one_limited_call_and_persists_original_raw(tmp_path: Path) -> None:
    payload, index, value, targets = invalid_endpoint_fixture()

    class FakeScheduler:
        output = tmp_path
        calls = 0

        async def ask(self, key: str, instruction: str, request: dict, schema: Any, effort: str) -> dict:
            self.calls += 1
            assert key == 'evaluation/p1/pointer_location'
            assert effort == 'xhigh'
            assert len(request['targets']) == 1
            sent = request['targets'][0]
            assert sent['fragment_id'] == targets[0]['fragment_id']
            assert all(c['clause_id'].startswith(sent['fragment_id']) for c in sent['clauses'])
            return {'corrections': [correction(sent, pointer(payload['source_passages'][0])['last_clause_id'])]}

    scheduler = FakeScheduler()
    result = asyncio.run(repair_pointer_locations(scheduler, 'evaluation/p1', value, index))
    assert scheduler.calls == 1 and result['audit']['repaired_count'] == 1
    import json
    audit_path = next(tmp_path.glob('audits/pointer_location/evaluation/p1/*.json'))
    saved = json.loads(audit_path.read_text())
    assert saved['original_raw'] == value and saved['state'] == 'resolved'
    assert saved['value'] == result['value']
    assert saved['response']['corrections'][0]['present']
    no_work = asyncio.run(repair_pointer_locations(scheduler, 'evaluation/p1', result['value'], index))
    assert scheduler.calls == 1 and no_work['audit']['repaired_count'] == 0
