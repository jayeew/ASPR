from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from figure_pipeline.fig5_robustness.aggregate import metric_row, part_values
from figure_pipeline.fig5_robustness.materials import (
    reduce_sources,
    reduced_graph,
    select,
)
from figure_pipeline.fig5_robustness.models import ConditionalReference, Config
from figure_pipeline.fig5_robustness.pipeline import (
    align_evaluation,
    validate_reference,
)
from figure_pipeline.fig5_robustness.runtime import Runner, Unavailable


def block(ident: str, alias: str, doi: str) -> dict[str, Any]:
    return {'block_id': ident, 'text': 'Original ' + doi,
                'provenance': [{'source_id': alias, 'doi': doi, 'title': doi, 'source_type': 'fulltext'}]}


def test_work_deletion_removes_both_aliases_preserves_target() -> None:
    data = {'manuscript': 'Target with bibliography', 'graph': {'cards': []},
                'gear_evidence': {'evidence_blocks': [block('a', 'G1', 'https://doi.org/10.x/ONE'), block('b', 'G2', '10.x/two')]},
                'graph_evidence': {'evidence_blocks': [block('a', 'H1', '10.x/one'), block('b', 'H2', '10.x/two')]}}
    result, mask = reduce_sources(data, 'paper', 20261002)
    assert len(mask['retained']) == len(mask['removed']) == 1
    for group in ('gear_evidence', 'graph_evidence'):
        assert len(result[group]['evidence_blocks']) == 1
        for b in result[group]['evidence_blocks']:
            assert all(mask['source_aliases'][p['source_id']] in mask['retained'] for p in b['provenance'])
    assert result['manuscript'] == data['manuscript']
    assert result['graph'] == data['graph']
    assert len(data['gear_evidence']['evidence_blocks']) == 2
    assert reduce_sources(data, 'paper', 20261002) == (result, mask)


def test_sampling_never_depends_on_score_or_input_order() -> None:
    rows = [{'paper_id': f'p{i}', 'group': 'g', 'sparse': i < 5, 'development': i == 0} for i in range(10)]
    a = select(rows, {'g': (2, 2)}, '20261002')
    assert a == select(list(reversed(rows)), {'g': (2, 2)}, '20261002')
    assert 'p0' not in a and len(a) == 4


def judgment(**changes: Any) -> dict[str, Any]:
    return dict(part_id='H1a', status='not_written', scope_correct=True, grounding='unclear',
                report_segment_ids=[], evidence_source_ids=[], reason='Limited evidence',
                explicitly_abstains=True, abstention_segment_ids=['P001'],
                abstention_appropriateness='reasonable', unsupported_definitive=False,
                unsupported_segment_ids=[], support_reason='Visible material insufficient', **changes)


def test_abstention_unknown_and_technical_missing_keep_denominator() -> None:
    p = judgment()
    p['technical_state'] = 'completed'
    row = {**p, **part_values(p)}
    metrics = metric_row('p', 'E50', 'historical_verification', [row])
    assert metrics['grounded_correct'] == 0 and metrics['explicit_abstention'] == 1
    assert metrics['unsupported_definitive'] == 0 and metrics['abstention_reasonable'] == 1
    p['unsupported_definitive'] = None
    metrics = metric_row('p', 'E50', 'h', [{**p, **part_values(p)}])
    assert metrics['fixed_denominator'] == 1
    assert metrics['unsupported_definitive_unresolved'] == 1
    assert metrics['scientific_unresolved'] == 1
    p['technical_state'] = 'missing'
    assert metric_row('p', 'E50', 'h', [{**p, **part_values(p)}])['grounded_correct'] is None


def test_quotes_are_copied_and_bad_locators_are_missing() -> None:
    material = {'reference_questions': [{'question_id': 'H1', 'applicable': True, 'answer_parts': [{'part_id': 'H1a'}]}],
                    'candidates': [{'report_id': 'R1', 'report_segments': [{'segment_id': 'P001', 'text': 'No independent source available.'}]}]}
    result = {'candidates': [{'report_id': 'R1', 'questions': [{'question_id': 'H1', 'parts': [judgment()]}]}]}
    aligned = align_evaluation(result, material)['candidates'][0]['questions'][0]['parts'][0]
    assert aligned['abstention_quotes'] == ['No independent source available.']
    result['candidates'][0]['questions'][0]['parts'][0]['abstention_segment_ids'] = ['P999']
    assert align_evaluation(result, material)['candidates'][0]['questions'][0]['parts'][0]['technical_state'] != 'completed'
    result['candidates'] *= 2
    with pytest.raises(ValueError):
        align_evaluation(result, material)


def test_reference_keeps_parts_even_when_unanswerable() -> None:
    fixed = {'questions': [{'question_id': 'H1', 'applicable': True, 'answer_parts': [{'part_id': 'H1a'}]}]}
    result = {'views': [{'view_id': v, 'parts': [{'question_id': 'H1', 'part_id': 'H1a', 'answerability': 'unanswerable', 'evidence': []}]} for v in ('F', 'E50', 'K5')]}
    validate_reference(result, fixed)
    result['views'][1]['parts'] = []
    with pytest.raises(ValueError):
        validate_reference(result, fixed)


def test_budget_and_recovery_do_not_reissue_success(tmp_path: Path) -> None:
    from figure_pipeline.fig3_revision.storage import write
    c = Config(output=tmp_path, ordinary_limit=1, additional_limit=1, call_limit=2)
    r = Runner(c)
    r.reserve('p', 'reference', 'papers', 'gpt-6.1-sol', 'high', False)
    with pytest.raises(Unavailable):
        r.reserve('q', 'reference', 'papers', 'gpt-6.1-sol', 'high', False)
    r.reserve('p', 'reference', 'papers', 'gpt-6.1-sol', 'high', True)
    with pytest.raises(Unavailable):
        r.reserve('q', 'reference', 'papers', 'gpt-6.1-sol', 'high', True)
    write(tmp_path / 'reference/papers/p.json', {'views': []})
    result = asyncio.run(r.call('p', 'reference', 'papers', 'unused', {}, ConditionalReference))
    assert result == {'views': []} and len(r.entries) == 2


def test_graph_recomputation_handles_zero_small_and_truncated() -> None:
    from figure_pipeline.fig3_revision.storage import read
    from gear.review_contracts import GraphFactCard
    root = Config().output
    if not (root / 'cohort.json').exists():
        pytest.skip('Prepared local dataset unavailable')
    examples = []
    for paper in read(root / 'cohort.json')['paper_ids']:
        data = read(root / 'inputs/F' / f'{paper}.json')
        examples.extend(data['graph']['cards'])
    cards = [next(c for c in examples if len(c['neighbors']) == n) for n in (0, 3, 10)]
    class Runtime:
        def _connections(self) -> None:
            pass

        def _metrics(self, neighbors: list[Any], claim_type: Any) -> list[Any]:
            from gear.review_contracts import MetricFact
            return [MetricFact(name='nearest_prior_similarity', value=len(neighbors) if neighbors else None)]
    graph = reduced_graph({'cards': cards, 'joint': {'historical_edges': []}}, Runtime())
    assert [len(c['neighbors']) for c in graph['cards']] == [0, 3, 5]
    assert graph['cards'][0]['metrics'][0]['value'] is None
    assert graph['cards'][2]['metrics'][0]['value'] == 5
    for c in graph['cards']:
        GraphFactCard.model_validate(c)
        assert all(set(e) <= {n['claim_id'] for n in c['neighbors']} for e in c['neighbor_edges'])
