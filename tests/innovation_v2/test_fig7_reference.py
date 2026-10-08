"""Source, count and display boundaries for graph-grounded direction clouds."""
from __future__ import annotations

import copy

import pytest

from figure_pipeline.fig7_reference.predict import check_merge, frequencies, valid_candidates
from figure_pipeline.fig7_reference.render import cloud_layout, intersect


def test_phrase_frequency_counts_distinct_directions_with_shared_font_scale() -> None:
    label = {'category': 'Method', 'phrase': 'Spatial molecular profiling'}
    rows = frequencies([
        {'candidate_id': 'D1', 'labels': [label, label]},
        {'candidate_id': 'D2', 'labels': [label, {'category': 'Problem', 'phrase': 'Tumor niche dynamics'}]},
        {'candidate_id': 'D3', 'labels': [{'category': 'Problem', 'phrase': 'Tumor niche dynamics'}]},
    ])
    assert [r['frequency'] for r in rows] == [2, 2]
    assert [r['font_size_pt'] for r in rows] == [18, 18]
    assert rows[0]['direction_ids'] == ['D1', 'D2']


def test_top_twenty_preserves_full_table_and_equal_sizes() -> None:
    rows = frequencies([{'candidate_id': f'D{i}', 'labels': [{'category': 'Method', 'phrase': f'Phrase {i:02d}'}]}
                        for i in range(24)])
    assert len(rows) == 24
    assert sum(r['displayed'] for r in rows) == 20
    assert {r['font_size_pt'] for r in rows} == {18}


def sample_packet() -> dict:
    return {'packet_id': 'P01', 'group_columns': ['id', 'edge_examples'],
            'edge_columns': ['id', 'earlier', 'later'], 'groups': [['C1', [['E1', 'N1', 'N2']]]],
            'evidence_columns': ['id', 'paper_id', 'claim', 'group'],
            'evidence': [['N1', 'paper1', 'Observed single-cell states.', 'C1'],
                         ['N2', 'paper2', 'Observed spatial neighborhoods.', 'C2']]}


def sample_direction() -> dict:
    return {'direction': 'Proposed future extension', 'graph_refs': ['C1', 'E1'],
            'citations': [{'source_id': 'N1', 'quote': 'single-cell states'},
                          {'source_id': 'N2', 'quote': 'spatial neighborhoods'}]}


def test_quotes_paper_independence_and_edge_endpoints_are_required() -> None:
    packet, direction = sample_packet(), sample_direction()
    good, bad = valid_candidates(packet, {'directions': [direction]})
    assert len(good) == 1 and not bad
    altered = copy.deepcopy(direction)
    altered['citations'][1]['quote'] = 'unobserved causal mechanism'
    assert valid_candidates(packet, {'directions': [altered]})[1][0]['errors'] == ['non_exact_quote']
    packet['evidence'][1][1] = 'paper1'
    assert 'fewer_than_two_papers' in valid_candidates(packet, {'directions': [direction]})[1][0]['errors']
    altered = copy.deepcopy(direction)
    altered['citations'] = [altered['citations'][0]] * 2
    assert 'edge_endpoints_not_cited' in valid_candidates(sample_packet(), {'directions': [altered]})[1][0]['errors']


def test_merge_cannot_invent_omit_or_double_count_candidates() -> None:
    candidates = [{'candidate_id': 'D1'}, {'candidate_id': 'D2'}]
    check_merge(candidates, {'retained': [{'candidate_id': 'D1'}], 'excluded': [{'candidate_id': 'D2'}]})
    with pytest.raises(ValueError):
        check_merge(candidates, {'retained': [{'candidate_id': 'D1'}, {'candidate_id': 'D1'}], 'excluded': []})
    with pytest.raises(ValueError):
        check_merge(candidates, {'retained': [{'candidate_id': 'D3'}], 'excluded': [{'candidate_id': 'D2'}]})


def test_cloud_layout_keeps_size_and_does_not_overlap() -> None:
    rows = [{'phrase': phrase, 'frequency': 1, 'font_size_pt': 18} for phrase in
            ['Spatial molecular profiling', 'Tumor niche dynamics', 'Mechanism-guided catalyst design']]
    first = cloud_layout(rows, 288, 390)
    assert first == cloud_layout(rows, 288, 390)
    assert {r['font_size_pt'] for r in first} == {18}
    for i, row in enumerate(first):
        for other in first[i+1:]:
            assert not intersect(tuple(row['box']), tuple(other['box']))


def test_supplied_cross_packet_neighbor_ids_are_valid_references() -> None:
    packet = sample_packet()
    packet['neighbor_labels'] = {'C2': ['Spatial biology']}
    direction = sample_direction()
    direction['graph_refs'].append('C2')
    accepted, rejected = valid_candidates(packet, {'directions': [direction]})
    assert len(accepted) == 1 and not rejected
    direction['graph_refs'].append('C999')
    assert 'unknown_graph_reference' in valid_candidates(packet, {'directions': [direction]})[1][0]['errors']


def test_edge_endpoint_group_remains_valid_outside_top_neighbor_labels() -> None:
    packet = sample_packet()
    direction = sample_direction()
    direction['graph_refs'].append('C2')
    accepted, rejected = valid_candidates(packet, {'directions': [direction]})
    assert len(accepted) == 1 and not rejected
