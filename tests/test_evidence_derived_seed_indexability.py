"""Fail-closed tests for seed DOI indexability resolution."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from innovation_impact_feature_selection.evidence_derived.core import (
    EvidenceProtocol,
    ProtocolError,
    canonical_json,
    sha256_text,
)
from innovation_impact_feature_selection.evidence_derived.providers import (
    ProviderRecord,
)
from innovation_impact_feature_selection.evidence_derived.resolve_seed_indexability import (
    resolve_seed_indexability,
)


class FakeCrossref:
    def __init__(self, result: dict[str, Any] | Exception) -> None:
        self.result = result
        self.calls = 0

    def validate_doi(self, doi: str) -> dict[str, Any]:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeOpenAlex:
    def __init__(
        self, result: ProviderRecord | Exception, configured: bool = True
    ) -> None:
        self.result = result
        self.calls = 0
        self._slots = ["A"] if configured else []

    @property
    def configured_slots(self) -> list[str]:
        return self._slots

    def fetch_doi(self, doi: str) -> ProviderRecord:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _database(tmp_path: Path) -> Path:
    path = tmp_path / "evidence.sqlite3"
    with EvidenceProtocol(path, tmp_path) as engine:
        engine.initialize()
        engine.connection.execute(
            "INSERT INTO seed_inputs VALUES(?,?,?,?,?,?,?,?,?)",
            (
                "S1",
                "development",
                "10.1/example",
                "citation",
                2020,
                "en",
                "seed.csv",
                "a" * 64,
                "seed_only",
            ),
        )
        engine.connection.execute(
            "INSERT INTO seed_recall VALUES(?,?,?,?,?,?,?)",
            ("S1", "development", "", "unchecked", "unchecked", "", "[]"),
        )
        engine.connection.commit()
    return path


def _recall(path: Path) -> sqlite3.Row:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    row = connection.execute("SELECT * FROM seed_recall WHERE seed_id='S1'").fetchone()
    connection.close()
    assert row is not None
    return row


def test_exact_cache_hit_avoids_providers_and_preserves_unchecked_recall(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    raw = {"id": "https://openalex.org/W1", "display_name": "Cached title"}
    raw_json = canonical_json(raw)
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO provider_cache_records VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "OpenAlex",
            "doi:10.1/example",
            "https://openalex.org/W1",
            "10.1/example",
            "Cached title",
            "",
            "en",
            2020,
            "article",
            "",
            "[]",
            raw_json,
            "fixture",
            sha256_text(raw_json),
            "b" * 64,
            "raw_cache_only_no_decisions",
        ),
    )
    connection.commit()
    connection.close()
    crossref = FakeCrossref({"status": "not_found"})
    openalex = FakeOpenAlex(ProviderRecord("not_found", {}, "A"))

    result = resolve_seed_indexability(database, tmp_path, crossref, openalex)

    row = _recall(database)
    assert (row["indexability"], row["recall_status"], row["reason_code"]) == (
        "indexable",
        "unchecked",
        "CACHE_DOI_MATCH",
    )
    assert row["work_id"]
    assert crossref.calls == openalex.calls == 0
    assert result["indexable_count"] == 1
    assert (
        len((tmp_path / "seed_indexability_manifest.sha256").read_text().strip()) == 64
    )


def test_exact_cache_hit_uses_raw_publication_date_in_cutoff_year(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE seed_inputs SET publication_year=2026 WHERE seed_id='S1'"
    )
    raw = {
        "id": "https://openalex.org/W2026",
        "display_name": "Before exact cutoff",
        "publication_date": "2026-05-29",
    }
    raw_json = canonical_json(raw)
    connection.execute(
        "INSERT INTO provider_cache_records VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "OpenAlex",
            "doi:10.1/example",
            "https://openalex.org/W2026",
            "10.1/example",
            "Before exact cutoff",
            "",
            "en",
            2026,
            "article",
            "",
            "[]",
            raw_json,
            "fixture",
            sha256_text(raw_json),
            "b" * 64,
            "raw_cache_only_no_decisions",
        ),
    )
    connection.commit()
    connection.close()
    crossref = FakeCrossref({"status": "not_found"})
    openalex = FakeOpenAlex(ProviderRecord("not_found", {}, "A"), configured=False)

    result = resolve_seed_indexability(database, tmp_path, crossref, openalex)

    row = _recall(database)
    assert row["indexability"] == "indexable"
    assert row["reason_code"] == "CACHE_DOI_MATCH"
    assert crossref.calls == openalex.calls == 0
    assert result["indexable_count"] == 1


def test_both_endpoints_not_found_is_not_indexed(tmp_path: Path) -> None:
    database = _database(tmp_path)
    crossref = FakeCrossref({"doi": "10.1/example", "status": "not_found"})
    openalex = FakeOpenAlex(ProviderRecord("not_found", {}, "A"))

    resolve_seed_indexability(database, tmp_path, crossref, openalex)

    row = _recall(database)
    assert row["indexability"] == "not_indexed"
    assert row["reason_code"] == "OPENALEX_DOI_NOT_INDEXED"
    assert row["recall_status"] == "unchecked"
    assert not row["work_id"]


def test_provider_errors_fail_closed(tmp_path: Path) -> None:
    database = _database(tmp_path)
    crossref = FakeCrossref(ProtocolError("Crossref unavailable"))
    openalex = FakeOpenAlex(ProtocolError("OpenAlex unavailable"))

    resolve_seed_indexability(database, tmp_path, crossref, openalex)

    row = _recall(database)
    assert row["indexability"] == "unchecked"
    assert row["reason_code"] == "OPENALEX_PROVIDER_ERROR_CROSSREF_ERROR"
    assert row["recall_status"] == "unchecked"
    assert not row["work_id"]


def test_crossref_only_does_not_claim_openalex_indexability(tmp_path: Path) -> None:
    database = _database(tmp_path)
    crossref = FakeCrossref(
        {
            "doi": "10.1/example",
            "status": "validated",
            "title": "Crossref title",
            "year": 2020,
            "type": "journal-article",
            "raw": {"DOI": "10.1/example", "title": ["Crossref title"]},
        }
    )
    openalex = FakeOpenAlex(ProviderRecord("found", {}, "A"), configured=False)

    result = resolve_seed_indexability(database, tmp_path, crossref, openalex)

    row = _recall(database)
    assert row["indexability"] == "unchecked"
    assert row["reason_code"] == "OPENALEX_KEY_UNAVAILABLE_CROSSREF_VALIDATED"
    assert row["work_id"]
    assert result["openalex_key_slots"] == []

    # A persisted Crossref fallback must not become an OpenAlex cache hit later.
    resolve_seed_indexability(database, tmp_path, crossref, openalex)
    rerun = _recall(database)
    assert rerun["indexability"] == "unchecked"
    assert rerun["reason_code"] == "OPENALEX_KEY_UNAVAILABLE_CROSSREF_VALIDATED"
