"""Tests for the fixed-medium v6.1 temporal OOF protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gear.nature_multihorizon.modeling_v6_1 import (
    _rank_group_ids,
    _weighted_rank_correlation,
    explicit_temporal_splits,
)


FOLDS = (
    {
        "fold_id": 1,
        "train_year_max": 1985,
        "test_year_min": 1986,
        "test_year_max": 1999,
    },
    {
        "fold_id": 2,
        "train_year_max": 1999,
        "test_year_min": 2000,
        "test_year_max": 2004,
    },
    {
        "fold_id": 3,
        "train_year_max": 2004,
        "test_year_min": 2005,
        "test_year_max": 2009,
    },
    {
        "fold_id": 4,
        "train_year_max": 2009,
        "test_year_min": 2010,
        "test_year_max": 2012,
    },
    {
        "fold_id": 5,
        "train_year_max": 2012,
        "test_year_min": 2013,
        "test_year_max": 2013,
    },
    {
        "fold_id": 6,
        "train_year_max": 2013,
        "test_year_min": 2014,
        "test_year_max": 2017,
    },
)


def test_explicit_temporal_splits_cover_every_test_year_once() -> None:
    frame = pd.DataFrame(
        {
            "paper_id": [f"W{year}" for year in range(1980, 2018)],
            "publication_year": list(range(1980, 2018)),
        }
    )
    folds = explicit_temporal_splits(frame, FOLDS)
    test_positions = np.concatenate([item["test_index"] for item in folds])
    observed = frame.iloc[test_positions]["publication_year"].tolist()
    assert observed == list(range(1986, 2018))
    assert all(
        item["train_year_max"] < item["test_year_min"] for item in folds
    )


def test_weighted_bootstrap_spearman_matches_expanded_sample() -> None:
    left = np.asarray([0.0, 0.0, 1.0, 2.0, 3.0])
    right = np.asarray([3.0, 1.0, 1.0, 2.0, 0.0])
    counts = np.asarray([2.0, 0.0, 1.0, 3.0, 1.0])
    weighted = _weighted_rank_correlation(
        _rank_group_ids(left),
        _rank_group_ids(right),
        counts,
    )
    positions = np.repeat(np.arange(len(left)), counts.astype(int))
    expanded = float(spearmanr(left[positions], right[positions]).statistic)
    assert np.isclose(weighted, expanded)
