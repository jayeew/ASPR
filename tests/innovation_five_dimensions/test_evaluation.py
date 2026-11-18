from __future__ import annotations

import copy
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_aggregate_provenance import save_provenance

from figure_pipeline.innovation_five_dimensions.aggregate import aggregate
from figure_pipeline.innovation_five_dimensions.config import DIMENSIONS
from figure_pipeline.innovation_five_dimensions.evaluation import (
    audit_report,
    locate_quote,
    validate_checklist,
    validate_evaluation,
)
from figure_pipeline.innovation_five_dimensions.io import write
from figure_pipeline.innovation_five_dimensions.models import Report


def report() -> dict:
    return {'contribution_overview': 'The experiment measures A.', 'synthesis': 'A is a conditional finding.',
            'dimensions': [{'dimension': dimension, 'judgment': 'A was observed.',
                            'rationale': 'The claim concerns A.', 'scope': 'Only under B.',
                            'status': 'supported', 'evidence': [evidence()]}
                           for dimension in DIMENSIONS]}


def evidence() -> dict:
    return {'source_id': 'S1', 'quote': 'A was observed under B.', 'explanation': 'Direct observation.'}


def sources() -> list[dict]:
    return [{'source_id': 'S1', 'source_type': 'manuscript', 'text': 'A was observed under B.'}]


def checklist() -> dict:
    return {'dimensions': [{'dimension': dimension, 'applicability': 'applicable',
                            'criteria': ['Discuss A.'], 'evidence': [evidence()],
                            'scope': 'Only under B.', 'reason': 'A is the selected contribution.'}
                           for dimension in DIMENSIONS]}


def evaluation(score: int = 2) -> dict:
    return {'dimensions': [{'dimension': dimension, 'status': 'assessed', 'score': score,
                            'checks': [{'report_quote': 'A was observed.', 'support': 'supported',
                                        'scope_correct': True, 'evidence': [evidence()],
                                        'reason': 'Matches the manuscript.'}],
                            'report_quotes': ['A was observed.'], 'reason': 'Specific and grounded.'}
                           for dimension in DIMENSIONS]}


def test_whitespace_location_preserves_source_span() -> None:
    text = 'Prefix. A\n  was\tobserved under B. Suffix.'
    span = locate_quote('A was observed under B.', text)
    assert span is not None
    assert text[span['start']:span['end']] == 'A\n  was\tobserved under B.'
    assert span['normalisation'] == 'whitespace_only'
    assert locate_quote('A was not observed under B.', text) is None
    assert locate_quote('', text) is None


@pytest.mark.parametrize(('quote', 'source', 'expected'), [
    ('The signal was transmitted.', 'Intro. The signal was trans-\nmitted. End.', 'The signal was trans-\nmitted.'),
    ('A fibre was measured.', 'A ﬁbre was measured.', 'A ﬁbre was measured.'),
    ('The fully fibre device.', 'The fullyﬁbre device.', 'The fullyﬁbre device.'),
    ('We measured A.', 'In this work, we measured A.', 'we measured A.'),
])
def test_pdf_typography_normalisation_returns_original_text(quote: str, source: str, expected: str) -> None:
    span = locate_quote(quote, source)
    assert span is not None
    assert span['text'] == expected
    assert source[span['start']:span['end']] == expected
    assert span['normalisation'].startswith('typography:')


def test_typography_matching_preserves_chemical_case_and_does_not_skip_text() -> None:
    assert locate_quote('Fe binds A.', 'FE binds A.') is None
    assert locate_quote('The Fe complex.', 'the FE complex.') is None
    assert locate_quote('A ... B.', 'A is much larger than B.') is None
    assert locate_quote('A therefore causes B.', 'A is associated with B.') is None
    assert locate_quote('The quartzsilicon interface.', 'The quartz-silicon interface.') is None
    mixed = 'ﬁbre light trans-\nmitted through the quartz-\nsilicon interface.'
    span = locate_quote('fibre light transmitted through the quartz-silicon interface.', mixed)
    assert span is not None and span['text'] == mixed


def test_fabricated_author_citation_is_not_a_technical_exclusion() -> None:
    value = report()
    value['dimensions'][0]['evidence'][0]['quote'] = 'A proves universal firstness.'
    audit = audit_report(value, sources())
    assert audit['unlocated_count'] == 1
    assert audit['eligible_for_scientific_evaluation'] is True
    assert audit['citations'][0]['problem'] == 'quotation_not_in_source'


def test_evaluator_cannot_quote_checklist_as_report() -> None:
    value = evaluation()
    value['dimensions'][0]['report_quotes'] = ['Discuss A.']
    with pytest.raises(ValueError, match='absent from report'):
        validate_evaluation(value, report(), sources())


def test_source_quote_in_report_is_not_an_author_assertion() -> None:
    value = evaluation()
    value['dimensions'][0]['report_quotes'] = [evidence()['quote']]
    with pytest.raises(ValueError, match='absent from report'):
        validate_evaluation(value, report(), sources())


def test_evaluator_citation_must_exist_in_visible_source() -> None:
    value = evaluation()
    value['dimensions'][0]['checks'][0]['evidence'][0]['quote'] = 'A holds universally.'
    with pytest.raises(ValueError, match='quotation_not_in_source'):
        validate_evaluation(value, report(), sources())
    with pytest.raises(ValueError, match='unknown_source_id'):
        validate_checklist(checklist(), [])


def test_multiple_source_passages_cannot_be_stitched() -> None:
    passages = [{'source_id': 'S1', 'source_type': 'manuscript', 'start': 20, 'text': 'A was observed'},
                {'source_id': 'S1', 'source_type': 'manuscript', 'start': 200, 'text': 'under B.'}]
    assert audit_report(report(), passages)['unlocated_count'] == 5
    passages.append({'source_id': 'S1', 'source_type': 'manuscript', 'start': 300,
                     'text': 'A was observed under B.'})
    audit = audit_report(report(), passages)
    assert audit['located_count'] == 5
    assert audit['citations'][0]['span']['source_start'] == 300


def test_all_five_dimensions_required_without_duplicate_names() -> None:
    value = report()
    value['dimensions'][1]['dimension'] = value['dimensions'][0]['dimension']
    with pytest.raises(ValidationError, match='Exactly one entry'):
        Report.model_validate(value)


def test_applicability_is_common_not_a_reward_for_abstention() -> None:
    neutral = checklist()
    neutral['dimensions'][0]['applicability'] = 'unresolved'
    with pytest.raises(ValueError, match='unresolved common reference'):
        validate_evaluation(evaluation(), report(), sources(), checklist=neutral)
    value = evaluation()
    value['dimensions'][0].update(status='unresolved', score=None)
    validate_evaluation(value, report(), sources(), checklist=neutral)
    neutral['dimensions'][0]['applicability'] = 'not_applicable'
    with pytest.raises(ValueError, match='applicability differs'):
        validate_evaluation(value, report(), sources(), checklist=neutral)


def save_result(root: Path, paper_id: str, method: str, score: int, unresolved: bool = False) -> None:
    value = evaluation(score)
    if unresolved:
        value['dimensions'][0].update(status='unresolved', score=None)
    write(root / 'reports' / method / f'{paper_id}.json', report())
    write(root / 'references' / f'{paper_id}.json', checklist())
    write(root / 'evaluations' / method / f'{paper_id}.json', value)
    save_provenance(root, paper_id, method)


def test_paired_denominators_keep_zero_exclude_missing_and_separate_damage(tmp_path: Path) -> None:
    roster = [{'paper_id': f'P{i}', 'cohort': 'main', 'methods': ['full', 'gear']} for i in range(1, 5)]
    roster.append({'paper_id': 'DAMAGE', 'cohort': 'damage', 'methods': ['full']})
    write(tmp_path / 'inputs/roster.json', roster)
    save_result(tmp_path, 'P1', 'full', 3)
    save_result(tmp_path, 'P1', 'gear', 1)
    save_result(tmp_path, 'P2', 'full', 1)
    save_result(tmp_path, 'P2', 'gear', 0, unresolved=True)
    save_result(tmp_path, 'P3', 'full', 0)
    save_result(tmp_path, 'P3', 'gear', 3)
    save_result(tmp_path, 'P4', 'gear', 2)
    save_result(tmp_path, 'DAMAGE', 'full', 3)
    result = aggregate(tmp_path)
    pair = next(r for r in result['paired_effects']
                if r['comparator'] == 'gear' and r['dimension'] == 'historical_originality')
    assert pair['n'] == 2 and pair['expected_pairs'] == 4
    assert pair['estimate'] == -.5
    summary = next(r for r in result['summary'] if r['cohort'] == 'main' and r['method'] == 'full'
                   and r['dimension'] == 'historical_originality')
    assert summary['n'] == 3 and summary['n_missing'] == 1
    assert summary['estimate'] == pytest.approx(4 / 3)
    assert result['completion']['main_completed_evaluations'] == 6
    assert result['completion']['damage_completed_evaluations'] == 1
    assert len(result['missing']) == 2
    assert any('requires an assessed report' in row['reason'] for row in result['missing'])
    assert (tmp_path / 'tables/paired_effects.csv').is_file()
    assert not any('total_score' in row for row in result['summary'])


def test_aggregate_rejects_duplicate_roster_papers(tmp_path: Path) -> None:
    row = {'paper_id': 'P1', 'cohort': 'main', 'methods': ['full']}
    write(tmp_path / 'inputs/roster.json', [row, copy.deepcopy(row)])
    with pytest.raises(ValueError, match='Duplicate paper'):
        aggregate(tmp_path)


def test_applicable_common_reference_cannot_hide_report_failure_as_unresolved() -> None:
    value = evaluation()
    value['dimensions'][0].update(status='unresolved', score=None)
    with pytest.raises(ValueError, match='requires an assessed report'):
        validate_evaluation(value, report(), sources(), checklist())
    value['dimensions'][0].update(status='assessed', score=0, checks=[], report_quotes=[])
    validate_evaluation(value, report(), sources(), checklist())
