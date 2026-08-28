from __future__ import annotations

from datetime import date

from experiments.gear.evaluation.acquire_claim_adoption_contexts import (
    _within_horizon,
)


def test_claim_context_window_uses_exact_five_year_boundary() -> None:
    cutoff = date(2014, 7, 16)

    assert _within_horizon(date(2019, 7, 16), 2019, cutoff=cutoff, horizon_years=5)
    assert not _within_horizon(date(2019, 7, 17), 2019, cutoff=cutoff, horizon_years=5)
    assert not _within_horizon(date(2014, 7, 16), 2014, cutoff=cutoff, horizon_years=5)


def test_claim_context_without_date_excludes_ambiguous_boundary_years() -> None:
    cutoff = date(2014, 7, 16)

    assert _within_horizon(None, 2016, cutoff=cutoff, horizon_years=5)
    assert not _within_horizon(None, 2014, cutoff=cutoff, horizon_years=5)
    assert not _within_horizon(None, 2019, cutoff=cutoff, horizon_years=5)


def test_claim_context_window_handles_leap_day_cutoff() -> None:
    cutoff = date(2016, 2, 29)

    assert _within_horizon(date(2021, 2, 28), 2021, cutoff=cutoff, horizon_years=5)
    assert not _within_horizon(date(2021, 3, 1), 2021, cutoff=cutoff, horizon_years=5)
