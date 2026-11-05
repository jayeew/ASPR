"""Meaningful cutoff boundary tests for uncertain bibliographic dates."""

from __future__ import annotations

from .package import date_bounds


def test_month_precision_crosses_cutoff() -> None:
    lo, hi, precision, _ = date_bounds([{"parts": [2025, 10], "field": "published"}])
    assert lo < "2025-10-07" <= hi
    assert precision == "month"


def test_day_of_announcement_is_excluded() -> None:
    _, hi, _, _ = date_bounds([{"parts": [2025, 10, 7], "field": "published"}])
    assert not hi < "2025-10-07"


def test_digitization_does_not_replace_print_date() -> None:
    lo, _, _, basis = date_bounds(
        [
            {"parts": [2002, 5, 1], "field": "published-online"},
            {"parts": [1989, 7, 1], "field": "published-print"},
        ]
    )
    assert lo == "1989-07-01" and basis == "published-print"


def test_unknown_date_stays_unknown() -> None:
    assert date_bounds([]) == (None, None, "unknown", "unavailable")
