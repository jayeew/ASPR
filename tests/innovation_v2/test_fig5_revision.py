from __future__ import annotations

import asyncio
import copy
from pathlib import Path

import pytest

from figure_pipeline.fig5_revision.materials import identities, retain
from figure_pipeline.fig5_revision.pipeline import validate_support


def data() -> dict:
    return {'gear_evidence': {'evidence_blocks': [
        {'block_id': 'B1', 'text': 'x', 'provenance': [{'source_id': 'A', 'title': 'One', 'doi': 'https://doi.org/10.1/X', 'work_id': 'W1'}]},
        {'block_id': 'B2', 'text': 'y', 'provenance': [{'source_id': 'B', 'title': 'Different title', 'work_id': 'https://openalex.org/W1'}]},
        {'block_id': 'B3', 'text': 'z', 'provenance': [{'source_id': 'C', 'title': 'Two', 'doi': '10.2/Y'}]},
    ]}, 'graph_evidence': {'evidence_blocks': [
        {'block_id': 'B4', 'text': 'w', 'provenance': [{'source_id': 'D', 'title': 'ONE'}]},
    ]}, 'graph': {'unchanged': True}}


def test_alias_deletion_removes_cross_branch_alternates() -> None:
    original = data()
    before = copy.deepcopy(original)
    aliases = identities(original)
    assert aliases['A'] == aliases['B'] == aliases['D'] == 'doi:10.1/x'
    reduced = retain(original, aliases, {'doi:10.2/y'})
    assert [b['block_id'] for b in reduced['gear_evidence']['evidence_blocks']] == ['B3']
    assert not reduced['graph_evidence']['evidence_blocks']
    assert reduced['graph'] == original['graph'] and original == before


def test_conflicting_explicit_identities_fail() -> None:
    original = data()
    original['gear_evidence']['evidence_blocks'][1]['provenance'][0]['doi'] = '10.2/Y'
    with pytest.raises(ValueError, match='Conflicting'):
        identities(original)


def test_critical_mask_must_remove_alternative_support() -> None:
    original = data()
    aliases = identities(original)
    reference = {'questions': [{'question_id': 'H1', 'applicable': True, 'answer_parts': [{'part_id': 'p'}]}]}
    value = {'targets': [{'question_id': 'H1', 'part_id': 'p', 'support_groups': [['doi:10.1/x'], ['doi:10.2/y']],
                         'critical_remove': ['doi:10.1/x'], 'noncritical_remove': ['doi:10.2/y'], 'support_block_ids': ['B1']}]}
    with pytest.raises(ValueError, match='hit every'):
        validate_support(value, aliases, original, reference)


def test_metrics_separate_boundary_caution_omission_and_abstention() -> None:
    from figure_pipeline.fig5_revision.aggregate import metric, scored
    base = {'technical_state': 'completed', 'status': 'correct', 'scope_correct': True, 'grounding': 'native_graph',
            'unsupported_definitive': False, 'conflicting_material_error': False, 'response_type': 'substantive_answer'}
    abstain = {**base, 'status': 'not_written', 'response_type': 'target_abstention'}
    omit = {**base, 'status': 'not_written', 'response_type': 'omission'}
    rows = [{**p, **scored(p), 'answerability': a} for p, a in [(base, 'answerable'), (abstain, 'unanswerable'), (omit, 'answerable')]]
    result = metric(rows)
    assert result['q_fixed'] == 1 / 3
    assert result['reasonable_abstention_sensitivity'] == 1
    assert result['unnecessary_abstention_rate'] == 0
    assert result['omissions'] == 1
    assert result['answered_risk_denominator'] == 1


def test_missing_technical_result_is_not_zero_risk() -> None:
    from figure_pipeline.fig5_revision.aggregate import metric, scored
    row = {'technical_state': 'missing', 'answerability': 'answerable'}
    result = metric([{**row, **scored(row)}])
    assert result['q_fixed'] is None
    assert result['unsupported_burden'] is None
    assert result['answered_risk'] is None


def test_unresolved_answered_risk_is_an_interval_not_zero() -> None:
    from figure_pipeline.fig5_revision.aggregate import metric, scored
    part = {'technical_state': 'completed', 'status': 'unresolved', 'scope_correct': None, 'grounding': 'unclear',
            'unsupported_definitive': None, 'conflicting_material_error': None, 'response_type': 'substantive_answer',
            'answerability': 'unresolved'}
    result = metric([{**part, **scored(part)}])
    assert result['answered_risk'] is None and result['answered_risk_unresolved'] == 1
    assert result['answered_risk_lower_bound'] == 0 and result['answered_risk_upper_bound'] == 1


def test_correct_excerpt_cannot_cancel_material_error() -> None:
    from figure_pipeline.fig5_revision.aggregate import scored
    row = {'technical_state': 'completed', 'status': 'correct', 'scope_correct': True, 'grounding': 'native_graph',
           'unsupported_definitive': False, 'conflicting_material_error': True, 'response_type': 'substantive_answer'}
    result = scored(row)
    assert result['correct'] == 0 and result['risk'] == 1


def test_reuse_allows_roundoff_but_not_changed_material() -> None:
    from figure_pipeline.fig5_revision.materials import equivalent
    assert equivalent({'value': 2.1625430743125347}, {'value': 2.1625430743125342})
    assert not equivalent({'value': 0.51}, {'value': 0.52})
    assert not equivalent({'sources': ['A', 'B']}, {'sources': ['A']})


def test_vetted_preprint_and_publication_share_deletion_identity() -> None:
    original = data()
    original['gear_evidence']['evidence_blocks'][0]['provenance'][0] = {'source_id': 'A', 'title': 'Published title', 'doi': '10.1371/journal.pgen.1009943'}
    original['gear_evidence']['evidence_blocks'][1]['provenance'][0] = {'source_id': 'B', 'title': 'Preprint title', 'doi': '10.1101/2021.11.15.468586'}
    aliases = identities(original)
    assert aliases['A'] == aliases['B']
    reduced = retain(original, aliases, set(aliases.values()) - {aliases['A']})
    assert not {'A', 'B'} & {p['source_id'] for b in reduced['gear_evidence']['evidence_blocks'] for p in b['provenance']}


def test_harmless_relabel_preserves_answerability_and_source_quotes() -> None:
    from figure_pipeline.fig5_revision.reference_reuse import relabel_reference
    original = {'parts': [{'answerability': 'answerable', 'expected_content': 'T00001 supports this',
                          'acceptable_variants': 'T00001', 'reason': 'unchanged',
                          'evidence': [{'source_id': 'GEAR_01', 'location': 'T00001:1', 'quote_or_value': 'literal T00001'}]}]}
    result = relabel_reference(original, {'T00001': 'B00001'})
    assert result['parts'][0]['answerability'] == 'answerable'
    assert result['parts'][0]['evidence'][0]['location'] == 'B00001:1'
    assert result['parts'][0]['evidence'][0]['quote_or_value'] == 'literal T00001'
    assert original['parts'][0]['expected_content'].startswith('T00001')


def test_common_answerable_denominator_distinguishes_missing_from_empty() -> None:
    from figure_pipeline.fig5_revision.aggregate import paired
    base = {'paper_id': 'paper', 'question_id': 'H1', 'part_id': 'H1a', 'aspect': 'H', 'split': 'exploration',
            'answerability': 'answerable', 'condition': 'F', 'technical_state': 'completed', 'correct': 1}
    missing = {**base, 'condition': 'E25', 'answerability': 'missing', 'technical_state': 'missing', 'correct': None}
    row = paired([base, missing])[0]
    assert row['common_answerable_denominator'] is None and row['state'] == 'reference_missing'
    insufficient = {**missing, 'answerability': 'unanswerable', 'technical_state': 'completed', 'correct': 0}
    row = paired([base, insufficient])[0]
    assert row['common_answerable_denominator'] == 0 and row['delta_q_answerable'] is None
    assert row['state'] == 'no_common_answerable_parts'


def test_supervised_dispatch_pause_does_not_spend_request(tmp_path: Path) -> None:
    from figure_pipeline.fig5_revision.models import Config
    from figure_pipeline.fig5_revision.runtime import Runner
    from figure_pipeline.fig5_robustness.runtime import Unavailable
    runner = Runner(Config(output=tmp_path))
    (tmp_path / 'pause_dispatch').touch()
    with pytest.raises(Unavailable, match='no new request'):
        runner.reserve('paper', 'analysis', 'E50_gear', 'gpt-6.1-sol', 'medium', False)
    assert not runner.entries
    assert not (tmp_path / 'call_ledger.jsonl').exists()


def test_reference_inconsistency_audit_does_not_relabel_negative_results() -> None:
    from figure_pipeline.fig5_revision.reference_audit import inconsistencies
    def reference(label: str) -> dict:
        return {'parts': [{'question_id': 'H1', 'part_id': 'p', 'answerability': label}]}
    normal = {'F': reference('answerable'), 'E50': reference('unanswerable'), 'E25': reference('unanswerable')}
    assert inconsistencies(normal) == []
    reversed_pair = {'F': reference('unanswerable'), 'E50': reference('answerable')}
    before = copy.deepcopy(reversed_pair)
    assert inconsistencies(reversed_pair) == [{'lower': 'E50', 'upper': 'F', 'question_id': 'H1', 'part_id': 'p'}]
    assert reversed_pair == before


def test_successful_resume_clears_wrapper_failure_state(tmp_path: Path) -> None:
    from figure_pipeline.fig3_revision.storage import read
    from figure_pipeline.fig5_revision.models import Config
    from figure_pipeline.fig5_revision.runtime import Runner, protected
    runner = Runner(Config(output=tmp_path))
    runner.status('paper', 'generation', 'E50', 'unresolved', reason='paused')
    async def completed() -> str:
        return 'restored'
    assert asyncio.run(protected(runner, 'paper', 'generation', 'E50', completed())) == 'restored'
    assert read(tmp_path / 'task_status/generation/E50/paper.json')['state'] == 'completed'


def test_capacity_retry_preserves_model_and_uses_additional_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import BaseModel

    from figure_pipeline.fig3_revision.storage import read
    from figure_pipeline.fig5_revision.models import Config
    from figure_pipeline.fig5_revision.runtime import (
        PreviousRunner,
        Runner,
        Unavailable,
    )
    runner = Runner(Config(output=tmp_path))
    calls = []
    async def request(self: Runner, paper: str, stage: str, condition: str, prompt: str, material: dict,
                      schema: type[BaseModel], model: str, effort: str, additional: bool) -> dict:
        calls.append((condition, model, effort, additional))
        if condition == 'critical_H':
            raise Unavailable('capacity')
        return {'result': 'complete'}
    async def no_wait(seconds: int) -> None:
        pass
    monkeypatch.setattr(PreviousRunner, 'call', request)
    monkeypatch.setattr(asyncio, 'sleep', no_wait)
    monkeypatch.setattr(runner, 'records', lambda paper, stage, condition: [(tmp_path, {'cli_error': 'Selected model is at capacity'})] if condition == 'critical_H' else [])
    value = asyncio.run(runner.call('paper', 'evaluation', 'critical_H', 'prompt', {}, BaseModel, 'fixed-model', 'high'))
    assert value == {'result': 'complete'}
    assert calls == [('critical_H', 'fixed-model', 'high', False), ('critical_H__capacity_retry', 'fixed-model', 'high', True)]
    assert read(tmp_path / 'evaluation/critical_H/paper.json') == value


def test_resume_wall_time_is_not_reported_as_complete_inference() -> None:
    from figure_pipeline.fig5_revision.diagnostics import latency_scope
    timing = {'state': 'completed', 'ended_at': '2026-10-03T12:01:00+00:00', 'wall_seconds': 60, 'includes_queue_wait': True}
    calls = [{'reused': False, 'started_at': '2026-10-03T11:50:00+00:00'},
             {'reused': False, 'started_at': '2026-10-03T12:00:01+00:00'}]
    resumed = latency_scope(timing, calls)
    assert resumed['wall_seconds'] is None and resumed['final_invocation_wall_seconds'] == 60
    assert resumed['wall_time_scope'] == 'resumed_partial_invocation'
    assert latency_scope(timing, calls[1:])['wall_seconds'] == 60
