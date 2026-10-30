from __future__ import annotations

from collections import Counter

import pytest

from experiments.fig8_extension.fig8_redraw import TYPES, brand_status, compose, summarize_entities


def test_mixed_year_brand_does_not_become_control() -> None:
    assert brand_status(Counter({'False': 10, '': 1})) == 'Mixed / unconfirmed'
    assert brand_status(Counter({'True': 5, 'False': 4})) == 'Mixed / unconfirmed'
    assert brand_status(Counter({'True': 3})) == 'Nature Portfolio'


def test_journal_mean_is_paper_weighted_not_claim_pooled() -> None:
    rows = [dict(journal='x', claim_n=1, **{t: float(t == 'METHOD') for t in TYPES}),
            dict(journal='x', claim_n=3, **{t: float(t == 'FINDING') for t in TYPES}),
            dict(journal='x', claim_n=0, **{t: None for t in TYPES})]
    row = compose(rows, ['journal'])[0]
    assert row['eligible_n'] == 3
    assert row['metric_n'] == 2
    assert row['METHOD'] == row['FINDING'] == .5


def test_country_metric_denominator_excludes_missing_claims_but_keeps_output_mass() -> None:
    metrics = [dict(pilot_id='1', claim_n=1, **{t: float(t == 'FINDING') for t in TYPES}),
               dict(pilot_id='2', claim_n=0, **{t: None for t in TYPES})]
    weights = [dict(pilot_id='1', scenario='main', dimension='country', entity='US', weight=.4),
               dict(pilot_id='2', scenario='main', dimension='country', entity='US', weight=.6)]
    row = summarize_entities(weights, metrics, {})[0]
    assert row['fractional_papers'] == 1
    assert row['metric_fractional_papers'] == .4
    assert row['FINDING'] == pytest.approx(1)
    assert row['effective_n'] == pytest.approx(1)


def test_absent_metrics_remain_missing_not_zero() -> None:
    row = dict(journal='x', claim_n=0, **{t: None for t in TYPES})
    assert compose([row], ['journal'])[0]['METHOD'] is None
