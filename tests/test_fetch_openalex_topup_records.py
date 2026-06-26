from __future__ import annotations

import json
from pathlib import Path

from scripts import fetch_openalex_topup_records as mod


def test_unique_query_records_deduplicates_and_filters_years(monkeypatch) -> None:
    def fake_fetch(query: str, max_records: int, timeout_seconds: int):
        assert max_records == 3
        assert timeout_seconds == 7
        return [
            {"id": "https://openalex.org/W1", "publication_year": 1995, "display_name": f"{query} old"},
            {"id": "https://openalex.org/W2", "publication_year": 2020, "display_name": f"{query} new"},
            {"id": "https://openalex.org/W1", "publication_year": 1995, "display_name": "duplicate"},
        ]

    monkeypatch.setattr(mod, "fetch_openalex_works_for_query", fake_fetch)

    rows = mod.unique_query_records(
        domain="Magnetic Properties of Thin Films",
        queries=["giant magnetoresistance", "spin valve"],
        max_records_per_query=3,
        timeout_seconds=7,
        year_min=1990,
        year_max=2010,
    )

    assert len(rows) == 1
    assert rows[0]["domain"] == "magnetic_properties_of_thin_films"
    assert rows[0]["query"] == "giant magnetoresistance"
    assert rows[0]["work"]["id"] == "https://openalex.org/W1"


def test_write_jsonl_round_trips_records(tmp_path: Path) -> None:
    out = tmp_path / "records.jsonl"
    n_rows = mod.write_jsonl(out, [{"domain": "crispr", "work": {"id": "W1"}}])

    assert n_rows == 1
    assert json.loads(out.read_text(encoding="utf-8")) == {"domain": "crispr", "work": {"id": "W1"}}
