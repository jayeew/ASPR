from __future__ import annotations

import pytest

from experiments.fig8_extension.fig8_preview import (
    TYPES,
    checked_claims,
    group_composition,
    structural_metrics,
)


def test_structural_metrics_count_unique_neighbors_and_ignore_empty_claims() -> None:
    result = structural_metrics([{'a', 'b'}, {'b', 'c'}, set()], {'a': 'x', 'b': 'x', 'c': 'y'})
    assert result['D'] == pytest.approx(1.8898815748)
    assert result['J'] == pytest.approx(1 / 3)
    assert result['S'] == pytest.approx(1 / 3)


def test_missing_neighborhoods_and_communities_are_not_zero_scores() -> None:
    assert structural_metrics([set()], {}) == {'D': None, 'J': None, 'S': None}
    assert structural_metrics([{'a'}, set()], {}) == {'D': None, 'J': None, 'S': None}


def test_claimless_paper_stays_in_eligible_denominator() -> None:
    rows = [{'field': 'x', 'claim_n': 1, **{t: float(t == 'METHOD') for t in TYPES}},
            {'field': 'x', 'claim_n': 0, **{t: None for t in TYPES}}]
    group = group_composition(rows, 'field')[0]
    assert group['eligible_n'] == 2
    assert group['paper_n'] == 1
    assert group['METHOD'] == 1
    assert group_composition(rows[1:], 'field')[0]['METHOD'] is None


def test_claim_quote_must_exist_in_source() -> None:
    paper = {'pilot_id': 'P01', 'sentences': [{'id': 'S01', 'text': 'Invented text.'}], 'abstract': 'Real text.'}
    with pytest.raises(ValueError, match='Support text'):
        checked_claims(paper, {'P01': [{'type': 'FINDING', 'sentences': ['S01'], 'claim': 'A finding.'}]})
