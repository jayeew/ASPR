from __future__ import annotations

from figure_pipeline.fig5_four_panel.aggregate import classify
from figure_pipeline.fig5_four_panel.prepare import canonical_sources


def part(**updates: object) -> dict:
    return dict(technical_state='completed', response_type='omission', status='not_written',
                scope_correct=None, grounding='unsupported', unsupported_definitive=False,
                conflicting_material_error=False, abstention_appropriateness='not_applicable', **updates)


def test_omission_with_null_scope_is_not_scientific_uncertainty() -> None:
    result = classify(part(), 'answerable')
    assert result['behavior'] == 'incomplete'
    assert result['scientific_unresolved'] == 0
    assert result['confirmed_unsupported'] == 0


def test_correct_answer_with_scope_caution_is_not_abstention() -> None:
    value = part()
    value.update(response_type='substantive_answer', status='correct', scope_correct=True, grounding='native_graph')
    assert classify(value, 'answerable')['grounded_answer'] == 1
    value['conflicting_material_error'] = True
    assert classify(value, 'answerable')['grounded_answer'] == 0


def test_unsupported_assertion_takes_precedence_over_abstention() -> None:
    value = part()
    value.update(response_type='target_abstention', abstention_appropriateness='reasonable', unsupported_definitive=True)
    result = classify(value, 'unanswerable')
    assert result['behavior'] == 'unsupported'
    assert result['reasonable_abstention'] == 0


def test_duplicate_dois_on_one_original_are_one_work() -> None:
    block = {'block_id': 'B1', 'text': 'One shared original passage', 'provenance': [
        {'source_id': 'S1', 'doi': '10.test/one'}, {'source_id': 'S2', 'doi': '10.test/two'}]}
    data = {'gear_evidence': {'evidence_blocks': [block]}, 'graph_evidence': {'evidence_blocks': []}}
    aliases, merges = canonical_sources(data)
    assert aliases['S1'] == aliases['S2']
    assert len(merges) == 1
