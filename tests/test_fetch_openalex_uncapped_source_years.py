"""Tests for uncapped source-by-year OpenAlex acquisition."""

from __future__ import annotations

from typing import Any

import pandas as pd

from scripts.fetch_openalex_uncapped_source_years import (
    LEGACY_CAP,
    audit_annual_continuity,
    fetch_complete_partition,
)


class FakeOpenAlex:
    """Return two deterministic cursor pages."""

    def get_json(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        assert endpoint == "/works"
        cursor = str((params or {}).get("cursor"))
        if cursor == "*":
            return {
                "meta": {"count": 3, "next_cursor": "page-2"},
                "results": [{"id": "W1"}, {"id": "W2"}],
            }
        return {
            "meta": {"count": 3, "next_cursor": None},
            "results": [{"id": "W3"}],
        }


def test_complete_partition_exhausts_cursor_without_cap() -> None:
    """The fetch must stop only at cursor exhaustion and match meta.count."""
    rows, expected, pages = fetch_complete_partition(
        FakeOpenAlex(), filters=["type:article"], per_page=200  # type: ignore[arg-type]
    )
    assert expected == len(rows) == 3
    assert pages == 2


def test_continuity_audit_rejects_legacy_cap() -> None:
    """A source total of exactly 25,000 must remain a hard failure."""
    partitions = pd.DataFrame(
        [
            {
                "source_display_name": "Nature",
                "year": 2020,
                "expected_count": LEGACY_CAP,
                "fetched_count": LEGACY_CAP,
                "unique_count": LEGACY_CAP,
            }
        ]
    )
    report = audit_annual_continuity(partitions)
    assert report["checks"]["no_source_equals_legacy_cap"] is False
    assert report["overall_pass"] is False
