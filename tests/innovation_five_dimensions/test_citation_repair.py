from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from figure_pipeline.innovation_five_dimensions.citation_repair import (
    apply_locations,
    build_candidates,
    clause_spans,
    collect_targets,
    repair_citations,
)
from figure_pipeline.innovation_five_dimensions.config import DIMENSIONS
from figure_pipeline.innovation_five_dimensions.evaluation import locate_quote, validate_checklist


def checklist() -> dict[str, Any]:
    return {'dimensions': [{'dimension': dimension, 'applicability': 'applicable',
                            'criteria': ['Discuss A.'], 'evidence': [], 'scope': 'Only B.',
                            'reason': 'A is a main finding.'} for dimension in DIMENSIONS]}


def source_evidence() -> dict[str, str]:
    return {'source_id': 'S1', 'quote': 'A was found under B.', 'explanation': 'Supports A under B.'}


def answer(target: str, first: str | None, last: str | None, present: bool = True) -> dict[str, Any]:
    return {'target_id': target, 'present': present, 'first_clause_id': first,
            'last_clause_id': last, 'same_assertion_and_scope': present,
            'confidence': 'high', 'reason': 'Same complete assertion.'}


def test_clauses_partition_exactly_including_long_pdf_lines() -> None:
    text = 'Intro. A was observed; under B.\n\n' + 'x' * 1300 + ' Last.'
    spans = clause_spans(text)
    assert ''.join(text[a:b] for a, b in spans) == text
    assert all(b - a <= 600 for a, b in spans)
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))


def test_source_repair_is_same_source_contiguous_original_and_preserves_judgments() -> None:
    value = checklist()
    value['dimensions'][0]['evidence'] = [source_evidence()]
    sources = [{'source_id': 'S1', 'text': 'A was observed under B. Other finding.', 'start': 100},
               {'source_id': 'PRIVATE_OTHER', 'text': 'Never send me.'}]
    targets = collect_targets(value, sources, None)
    candidates, index = build_candidates(targets, sources, None)
    assert len(candidates) == 1
    result = apply_locations(value, targets, {'locations': [answer('T0000', 'F0000C0000', 'F0000C0000')]}, index)
    assert result['value']['dimensions'][0]['evidence'][0]['quote'] == 'A was observed under B. '
    assert value['dimensions'][0]['evidence'][0]['quote'] == 'A was found under B.'
    result['value']['dimensions'][0]['evidence'][0]['quote'] = value['dimensions'][0]['evidence'][0]['quote']
    assert result['value'] == value
    assert result['audit']['targets'][0]['span']['source_start'] == 100


@pytest.mark.parametrize('kind', ['cross_fragment', 'cross_source', 'reverse', 'unknown'])
def test_invalid_pointer_cannot_join_fragments_or_switch_sources(kind: str) -> None:
    value = checklist()
    value['dimensions'][0]['evidence'] = [source_evidence()]
    value['dimensions'][1]['evidence'] = [dict(source_evidence(), source_id='S2')]
    sources = [{'source_id': 'S1', 'text': 'First. Second.'},
               {'source_id': 'S1', 'text': 'Third.', 'start': 400},
               {'source_id': 'S2', 'text': 'Another paper.'}]
    targets = collect_targets(value, sources, None)
    _, index = build_candidates(targets, sources, None)
    first, last = {'cross_fragment': ('F0000C0000', 'F0001C0000'),
                   'cross_source': ('F0002C0000', 'F0002C0000'),
                   'reverse': ('F0000C0001', 'F0000C0000'),
                   'unknown': ('F9999C0000', 'F9999C0000')}[kind]
    with pytest.raises(ValueError):
        apply_locations(value, targets, {'locations': [answer('T0000', first, last),
                                                     answer('T0001', None, None, False)]}, index)


def test_unmatched_low_confidence_or_different_assertion_remain_failures() -> None:
    value = checklist()
    value['dimensions'][0]['evidence'] = [source_evidence()]
    sources = [{'source_id': 'S1', 'text': 'A was observed under B.'}]
    targets = collect_targets(value, sources, None)
    _, index = build_candidates(targets, sources, None)
    for pointer in [answer('T0000', None, None, False),
                    dict(answer('T0000', 'F0000C0000', 'F0000C0000'), confidence='low'),
                    dict(answer('T0000', 'F0000C0000', 'F0000C0000'), same_assertion_and_scope=False)]:
        result = apply_locations(value, targets, {'locations': [pointer]}, index)
        assert result['value'] == value
        with pytest.raises(ValueError, match='citation failures'):
            validate_checklist(result['value'], sources)


def test_report_candidates_exclude_embedded_evidence_and_keep_score() -> None:
    report = {'contribution_overview': 'A was observed.', 'synthesis': 'Only B.', 'dimensions': [
        {'dimension': dim, 'judgment': 'A was observed.', 'rationale': 'This advances A.',
         'scope': 'Only B.', 'status': 'supported', 'evidence': [source_evidence()]} for dim in DIMENSIONS]}
    value = {'dimensions': [{'dimension': dim, 'status': 'assessed', 'score': 2, 'checks': [],
                             'report_quotes': [], 'reason': 'Accurate.'} for dim in DIMENSIONS]}
    value['dimensions'][0]['report_quotes'] = ['A was found.']
    targets = collect_targets(value, [], report)
    candidates, index = build_candidates(targets, [], report)
    assert all('A was found under B.' not in c['text'] for f in candidates for c in f['clauses'])
    result = apply_locations(value, targets, {'locations': [answer('T0000', 'F0000C0000', 'F0000C0000')]}, index)
    assert result['value']['dimensions'][0]['score'] == 2
    assert result['value']['dimensions'][0]['report_quotes'] == ['A was observed.']


def test_one_location_call_and_no_call_for_already_locatable_quotes(tmp_path: Path) -> None:
    class FakeScheduler:
        output = tmp_path
        calls = 0

        async def ask(self, key: str, instruction: str, payload: dict, schema: Any, effort: str) -> dict:
            self.calls += 1
            assert key == 'reference/p1/citation_location'
            assert effort == 'xhigh'
            assert set(payload) == {'targets', 'candidate_fragments'}
            return {'locations': [answer('T0000', 'F0000C0000', 'F0000C0000')]}

    scheduler = FakeScheduler()
    value = checklist()
    value['dimensions'][0]['evidence'] = [source_evidence()]
    sources = [{'source_id': 'S1', 'text': 'A was observed under B.'}]
    result = asyncio.run(repair_citations(scheduler, 'reference/p1', value, sources))
    assert scheduler.calls == 1
    validate_checklist(result['value'], sources)
    assert list(tmp_path.glob('audits/citation_repair/reference/p1/*.json'))
    repeated = asyncio.run(repair_citations(scheduler, 'reference/p1', result['value'], sources))
    assert scheduler.calls == 1
    assert repeated['audit']['repaired_count'] == 0


def test_literal_json_whitespace_is_auditable_without_decoding_other_escapes() -> None:
    span = locate_quote(r'The signal was trans-\nmitted through a fibre.', 'The signal was trans-\nmitted through a fibre.')
    assert span is not None
    assert span['text'] == 'The signal was trans-\nmitted through a fibre.'
    assert 'literal_JSON_whitespace_escape' in span['normalisation']
    assert locate_quote(r'Fe has \u0031 atom.', 'Fe has 1 atom.') is None


def test_visible_metadata_title_is_located_as_title_not_body() -> None:
    from figure_pipeline.innovation_five_dimensions.evaluation import evidence_audit, source_index
    sources = [{'source_id': 'S1', 'source_type': 'fulltext', 'title': 'A novel device.',
                'text': 'Different body material.', 'start': 800}]
    audit = evidence_audit({'source_id': 'S1', 'quote': 'A novel device.'}, source_index(sources))
    assert audit['located'] is True
    assert audit['source_type'] == 'fulltext'
    assert audit['source_field'] == audit['quote_field'] == 'title'
    assert audit['title_only'] is True
    assert audit['span']['source_start'] is None
    assert audit['span']['source_end'] is None
    assert audit['span']['start'] == 0
    assert 'does not verify' in audit['location_meaning']


def test_title_and_body_cannot_be_joined_or_use_unseen_metadata() -> None:
    from figure_pipeline.innovation_five_dimensions.evaluation import evidence_audit, source_index
    sources = [{'source_id': 'S1', 'source_type': 'fulltext', 'title': 'A novel device.',
                'text': 'Different body material.'}]
    assert not evidence_audit({'source_id': 'S1', 'quote': 'A novel device. Different body material.'},
                              source_index(sources))['located']
    assert not evidence_audit({'source_id': 'S1', 'quote': 'Unknown unpublished title.'},
                              source_index(sources))['located']
    value = checklist()
    value['dimensions'][0]['evidence'] = [source_evidence()]
    targets = collect_targets(value, sources, None)
    candidates, index = build_candidates(targets, sources, None)
    assert [c['field'] for c in candidates] == ['text', 'title']
    with pytest.raises(ValueError, match='cross fragments'):
        apply_locations(value, targets, {'locations': [answer('T0000', 'F0000C0000', 'F0001C0000')]}, index)
