#!/usr/bin/env python3
"""Resolve seed DOI indexability without importing legacy recall decisions."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

try:
    from .core import (
        EvidenceProtocol,
        ProtocolError,
        canonical_json,
        file_hash,
        sha256_text,
        utc_now,
    )
    from .providers import CrossrefClient, OpenAlexClient, ProviderRecord
except ImportError:
    from core import (  # type: ignore[no-redef]
        EvidenceProtocol,
        ProtocolError,
        canonical_json,
        file_hash,
        sha256_text,
        utc_now,
    )
    from providers import (  # type: ignore[no-redef]
        CrossrefClient,
        OpenAlexClient,
        ProviderRecord,
    )


class CrossrefLookup(Protocol):
    def validate_doi(self, doi: str) -> dict[str, Any]: ...


class OpenAlexLookup(Protocol):
    @property
    def configured_slots(self) -> list[str]: ...

    def fetch_doi(self, doi: str) -> ProviderRecord: ...


@dataclass(frozen=True)
class Resolution:
    seed_id: str
    work: dict[str, Any] | None
    indexability: str
    reason_code: str
    cache_record: tuple[Any, ...] | None = None


def normalize_doi(value: str) -> str:
    normalized = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        normalized = normalized.removeprefix(prefix)
    return normalized.strip()


def _abstract(record: Mapping[str, Any]) -> str:
    inverted = record.get("abstract_inverted_index")
    if not isinstance(inverted, Mapping):
        return ""
    positioned = [
        (int(position), str(token))
        for token, positions in inverted.items()
        if isinstance(positions, list)
        for position in positions
    ]
    return " ".join(token for _, token in sorted(positioned))


def _openalex_work(record: Mapping[str, Any], doi: str) -> dict[str, Any]:
    return {
        "doi": normalize_doi(str(record.get("doi") or doi)),
        "openalex_id": str(record.get("id") or ""),
        "title": str(record.get("display_name") or record.get("title") or ""),
        "publication_year": record.get("publication_year"),
        "language": str(record.get("language") or ""),
        "work_type": str(record.get("type") or ""),
        "abstract": _abstract(record),
        "source_route": "seed_indexability_openalex_doi",
    }


def _crossref_work(record: Mapping[str, Any], doi: str) -> dict[str, Any]:
    return {
        "doi": doi,
        "openalex_id": "",
        "title": str(record.get("title") or ""),
        "publication_year": record.get("year"),
        "publication_date": str(record.get("publication_date") or ""),
        "language": "",
        "work_type": str(record.get("type") or ""),
        "abstract": "",
        "source_route": "seed_indexability_crossref_doi",
    }


def _within_cutoff(work: Mapping[str, Any], cutoff: date) -> bool:
    publication_date = str(work.get("publication_date") or "")
    if publication_date:
        try:
            return date.fromisoformat(publication_date) <= cutoff
        except ValueError:
            return False
    year = work.get("publication_year")
    if year is None:
        return True
    numeric_year = int(year)
    if numeric_year == cutoff.year:
        # A year-only record cannot prove it predates an intra-year cutoff.
        return False
    return numeric_year < cutoff.year


def _cache_tuple(
    provider: str,
    doi: str,
    work: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> tuple[Any, ...]:
    raw_json = canonical_json(raw)
    payload_hash = sha256_text(raw_json)
    provider_id = str(work.get("openalex_id") or doi)
    referenced = raw.get("referenced_works") or []
    return (
        provider,
        f"doi:{doi}",
        provider_id,
        doi,
        str(work.get("title") or ""),
        str(work.get("abstract") or ""),
        str(work.get("language") or ""),
        work.get("publication_year"),
        str(work.get("work_type") or ""),
        str(raw.get("id") or raw.get("URL") or ""),
        canonical_json(referenced),
        raw_json,
        "seed_indexability_doi_endpoint",
        payload_hash,
        payload_hash,
        "raw_cache_only_no_decisions",
    )


def _cached_resolution(
    row: sqlite3.Row, cutoff: date
) -> tuple[dict[str, Any] | None, str]:
    try:
        raw = json.loads(str(row["raw_json"]) or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        raw = {}
    publication_date = ""
    if isinstance(raw, Mapping):
        publication_date = str(raw.get("publication_date") or "")
    work = {
        "doi": normalize_doi(str(row["doi"])),
        "openalex_id": str(row["provider_id"]) if row["provider"] == "OpenAlex" else "",
        "title": str(row["title"]),
        "publication_year": row["publication_year"],
        "publication_date": publication_date,
        "language": str(row["language"]),
        "work_type": str(row["work_type"]),
        "abstract": str(row["abstract"]),
        "source_route": "seed_indexability_cache_doi",
    }
    if not work["title"] or not _within_cutoff(work, cutoff):
        return None, "CACHE_RECORD_INVALID_OR_AFTER_CUTOFF"
    return work, "CACHE_DOI_MATCH"


def _resolve_remote(
    seed: sqlite3.Row,
    cutoff: date,
    crossref: CrossrefLookup,
    openalex: OpenAlexLookup,
) -> Resolution:
    doi = normalize_doi(str(seed["doi"]))
    crossref_result: dict[str, Any] | None = None
    crossref_error = ""
    try:
        crossref_result = crossref.validate_doi(doi)
    except ProtocolError as error:
        crossref_error = str(error).split(":", 1)[0]

    crossref_work: dict[str, Any] | None = None
    crossref_cache: tuple[Any, ...] | None = None
    if crossref_result and crossref_result.get("status") == "validated":
        candidate = _crossref_work(crossref_result, doi)
        if candidate["title"] and _within_cutoff(candidate, cutoff):
            crossref_work = candidate
            raw = crossref_result.get("raw") or crossref_result
            crossref_cache = _cache_tuple("Crossref", doi, candidate, raw)

    if not openalex.configured_slots:
        reason = "OPENALEX_KEY_UNAVAILABLE"
        if crossref_work:
            reason += "_CROSSREF_VALIDATED"
        elif crossref_error:
            reason += "_CROSSREF_ERROR"
        return Resolution(
            seed["seed_id"], crossref_work, "unchecked", reason, crossref_cache
        )

    try:
        result = openalex.fetch_doi(doi)
    except ProtocolError:
        reason = "OPENALEX_PROVIDER_ERROR"
        if crossref_error:
            reason += "_CROSSREF_ERROR"
        return Resolution(
            seed["seed_id"], crossref_work, "unchecked", reason, crossref_cache
        )
    if result.status == "not_found":
        reason = "OPENALEX_DOI_NOT_INDEXED"
        if crossref_error:
            reason += "_CROSSREF_ERROR"
        return Resolution(
            seed["seed_id"], crossref_work, "not_indexed", reason, crossref_cache
        )

    work = _openalex_work(result.record, doi)
    work["publication_date"] = result.record.get("publication_date")
    if not work["title"] or not _within_cutoff(work, cutoff):
        return Resolution(
            seed["seed_id"],
            None,
            "not_indexed",
            "OPENALEX_RECORD_INVALID_OR_AFTER_CUTOFF",
        )
    cache = _cache_tuple("OpenAlex", doi, work, result.record)
    return Resolution(
        seed["seed_id"],
        work,
        "indexable",
        f"OPENALEX_DOI_MATCH_SLOT_{result.key_slot}",
        cache,
    )


def resolve_seed_indexability(
    database: Path,
    output_dir: Path | None = None,
    crossref: CrossrefLookup | None = None,
    openalex: OpenAlexLookup | None = None,
) -> dict[str, Any]:
    """Resolve all seeds; provider failures remain explicitly unchecked."""
    engine = EvidenceProtocol(database, output_dir)
    engine.initialize()
    destination = (output_dir or database.parent).resolve()
    protocol = json.loads(engine.protocol_path.read_text(encoding="utf-8"))
    cutoff = date.fromisoformat(str(protocol["cutoff_date"]))
    crossref_client = crossref or CrossrefClient()
    openalex_client = openalex or OpenAlexClient()
    seeds = list(
        engine.connection.execute("SELECT * FROM seed_inputs ORDER BY seed_id")
    )
    cache_rows = list(
        engine.connection.execute(
            "SELECT * FROM provider_cache_records ORDER BY provider,record_key"
        )
    )
    cache_by_doi: dict[str, sqlite3.Row] = {}
    for row in cache_rows:
        # Only an OpenAlex cache hit proves indexability in the search provider.
        # Crossref cache rows are bibliographic fallback evidence only.
        if row["provider"] != "OpenAlex":
            continue
        doi = normalize_doi(str(row["doi"]))
        if doi and doi not in cache_by_doi:
            cache_by_doi[doi] = row

    input_hash = sha256_text(canonical_json([dict(row) for row in seeds]))
    resolutions: list[Resolution] = []
    for seed in seeds:
        doi = normalize_doi(str(seed["doi"]))
        if seed["publication_year"] and int(seed["publication_year"]) > cutoff.year:
            resolutions.append(
                Resolution(seed["seed_id"], None, "not_indexed", "SEED_AFTER_CUTOFF")
            )
            continue
        cached = cache_by_doi.get(doi)
        if cached:
            work, reason = _cached_resolution(cached, cutoff)
            if work:
                resolutions.append(
                    Resolution(seed["seed_id"], work, "indexable", reason)
                )
                continue
        resolutions.append(
            _resolve_remote(seed, cutoff, crossref_client, openalex_client)
        )

    for resolution in resolutions:
        if resolution.cache_record:
            engine.connection.execute(
                "INSERT OR IGNORE INTO provider_cache_records VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                resolution.cache_record,
            )
        work_id = ""
        if resolution.work:
            work_id, _ = engine.ingest_work(resolution.work)
            publication_date = str(resolution.work.get("publication_date") or "")
            if publication_date:
                engine.connection.execute(
                    "INSERT INTO work_publication_dates VALUES(?,?) "
                    "ON CONFLICT(work_id) DO UPDATE SET publication_date=excluded.publication_date",
                    (work_id, publication_date),
                )
        engine.connection.execute(
            "UPDATE seed_recall SET work_id=?,indexability=?,reason_code=?,"
            "recall_status='unchecked',matched_query_ids_json='[]' WHERE seed_id=?",
            (
                work_id,
                resolution.indexability,
                resolution.reason_code,
                resolution.seed_id,
            ),
        )
    engine.connection.commit()

    recall_rows = [
        dict(row)
        for row in engine.connection.execute(
            "SELECT * FROM seed_recall ORDER BY seed_id"
        )
    ]
    if any(row["recall_status"] != "unchecked" for row in recall_rows):
        raise ProtocolError("Seed indexability resolver must not set recall decisions")
    counts = dict(sorted(Counter(row["indexability"] for row in recall_rows).items()))
    manifest = {
        "artifact": "seed_indexability_provenance",
        "generated_at": utc_now(),
        "protocol_sha256": file_hash(engine.protocol_path),
        "cutoff_date": cutoff.isoformat(),
        "resolution_order": [
            "provider_cache_exact_doi",
            "crossref_doi",
            "openalex_doi",
            "doi_openalex_title_year_dedup",
        ],
        "openalex_key_slots": list(openalex_client.configured_slots),
        "secret_material_persisted": False,
        "seed_inputs_sha256": input_hash,
        "seed_recall_sha256": sha256_text(canonical_json(recall_rows)),
        "seed_count": len(recall_rows),
        "indexability_counts": counts,
        "unchecked_count": counts.get("unchecked", 0),
        "not_indexed_count": counts.get("not_indexed", 0),
        "indexable_count": counts.get("indexable", 0),
    }
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "seed_indexability_manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    manifest_hash = file_hash(manifest_path)
    (destination / "seed_indexability_manifest.sha256").write_text(
        manifest_hash + "\n", encoding="utf-8"
    )
    engine.close()
    return {**manifest, "manifest_sha256": manifest_hash}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(canonical_json(resolve_seed_indexability(args.database, args.output_dir)))


if __name__ == "__main__":
    main()
