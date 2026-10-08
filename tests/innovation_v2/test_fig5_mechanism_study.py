from __future__ import annotations

from pathlib import Path

import pytest

from figure_pipeline.fig3_revision.storage import read, write
from figure_pipeline.fig5_mechanism_study import transfer
from figure_pipeline.fig5_mechanism_study.adapter import compose
from figure_pipeline.fig5_mechanism_study.diagnostic import CONDITIONS, raw_material


def test_factorial_separates_analysis_and_writer() -> None:
    assert set(CONDITIONS.values()) == {
        ('original', 'original'), ('original', 'replacement'),
        ('replacement', 'original'), ('replacement', 'replacement'),
    }


def test_raw_material_keeps_public_tasks_and_actual_evidence() -> None:
    data = {'public_tasks': ['question'], 'original_evidence': ['source'],
            'graph': {'edge': 1}, 'scientific_evidence_analysis': 'interpretation',
            'knowledge_graph_analysis': 'interpretation'}
    result = raw_material(data)
    assert result == {'public_tasks': ['question'], 'original_evidence': ['source'], 'graph': {'edge': 1}}
    assert 'scientific_evidence_analysis' in data


def test_structured_report_cannot_silently_drop_a_task() -> None:
    with pytest.raises(ValueError, match='omitted or duplicated'):
        compose({'answers': [], 'cited_source_ids': []},
                {'questions': [{'question_id': 'question_1', 'question': 'Compare evidence.'}]})


def test_structured_assembly_preserves_qualifications_and_sources() -> None:
    value = {'answers': [{'question_id': 'question_1', 'concrete_observation': 'Observed within one assay.',
                          'supporting_evidence_and_location': 'source_1, original passage',
                          'scientific_interpretation': 'Association only.',
                          'limitations_or_missing_evidence': 'Does not establish causality.'}],
             'cited_source_ids': ['source_1']}
    result = compose(value, {'questions': [{'question_id': 'question_1', 'question': 'Compare evidence.'}]})
    assert 'Does not establish causality.' in result['body']
    assert 'Observed within one assay.' in result['body']
    assert result['cited_source_ids'] == ['source_1']


def test_unavailable_paper_retains_planned_denominator_without_zero_score(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transfer, 'OUT', tmp_path / 'study')
    monkeypatch.setattr(transfer, 'OLD', tmp_path / 'source')
    write(transfer.OUT / 'protocol.json', {'papers': ['paper_unavailable']})
    write(transfer.OLD / 'baseline/reference/papers/paper_unavailable.json', {
        'questions': [{'question_id': 'question', 'aspect': 'historical_verification', 'applicable': True,
                       'answer_parts': [{'part_id': 'required_comparison'}, {'part_id': 'required_limit'}]}]})
    transfer.aggregate()
    rows = [r for r in read(transfer.OUT / 'paper_metrics.json') if r['aspect'] == 'all']
    assert len(rows) == 2
    assert all(r['fixed_denominator'] == 2 and not r['complete'] and r['grounded_answer'] is None for r in rows)
    parts = read(transfer.OUT / 'answer_parts.json')
    assert len(parts) == 4
    assert all(r['behavior'] == 'technical_missing' for r in parts)
