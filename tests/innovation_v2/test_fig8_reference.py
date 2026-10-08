"""Numerical boundary checks for the retrospective figure's valid denominators."""
from __future__ import annotations

from collections import defaultdict

import pytest

from figure_pipeline.fig8_reference.data import paper_metrics
from figure_pipeline.fig8_reference.helpers import effective_communities, neighborhood_overlap


def metadata() -> dict:
    return dict(nature_article_id='paper', title='Example', source_name='Nature Communications',
                publication_year=2025, publication_date='2025-01-02', field_name='Example field',
                primary_topic_name='Example topic', work_type='article')


def test_unassigned_nodes_are_not_an_additional_community() -> None:
    assert effective_communities([1, 1, None, None]) == 1
    assert effective_communities([None]) is None
    assert effective_communities([1, 2, None]) == pytest.approx(2)


def test_empty_neighborhood_is_missing_but_disjoint_pair_is_zero() -> None:
    assert neighborhood_overlap([set(), {'a'}]) == (None, 0)
    assert neighborhood_overlap([{'a'}, {'b'}, set()]) == (0, 1)
    value, pairs = neighborhood_overlap([{'a', 'b'}, {'b', 'c'}, set()])
    assert value == pytest.approx(1 / 3)
    assert pairs == 1


def test_union_is_deduplicated_and_shared_fraction_has_its_own_denominator() -> None:
    claims = [dict(claim_id='c1', claim_type='METHOD'), dict(claim_id='c2', claim_type='FINDING')]
    row = paper_metrics(metadata(), claims, {'c1': {'a', 'b'}, 'c2': {'b', 'c'}},
                        {'a': 1, 'b': 1, 'c': None})
    assert row['n_historical_neighbors'] == 3
    assert row['community_coverage'] == pytest.approx(2 / 3)
    assert row['effective_communities'] == 1
    assert row['shared_neighbor_fraction'] == pytest.approx(1 / 3)
    assert row['mean_jaccard'] == pytest.approx(1 / 3)
    assert row['type_METHOD'] == .5
    assert row['scatter_eligible']


def test_no_community_can_keep_distribution_but_not_scatter() -> None:
    claims = [dict(claim_id='c1', claim_type='METHOD'), dict(claim_id='c2', claim_type='METHOD')]
    row = paper_metrics(metadata(), claims, {'c1': {'a'}, 'c2': {'a'}}, {'a': None})
    assert row['mean_jaccard'] == row['shared_neighbor_fraction'] == 1
    assert row['effective_communities'] is None
    assert not row['scatter_eligible']


def test_claimless_paper_is_kept_with_missing_metrics() -> None:
    row = paper_metrics(metadata(), [], defaultdict(set), {})
    assert row['n_claims'] == 0
    assert row['type_METHOD'] is None
    assert row['shared_neighbor_fraction'] is None
    assert row['scatter_missing_reasons'] == ['no_claim_records']
