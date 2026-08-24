"""Retrieve exact OpenAlex source leads cited by H2-reviewed discovery texts."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

import requests
from common import (
    DATABASE_PATH,
    normalize_doi,
    normalize_text,
    sha256_file,
    write_csv,
    write_json,
)
from database import initialize, log_event
from providers import (
    fetch_json,
    insert_openalex_record,
    openalex_api_keys,
    openalex_record,
    openalex_url,
    safe_provider_error,
)

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "contextual_review_citation_leads_H2_v4.csv"
OUTPUT = ROOT / "outputs" / "contextual_review_citation_lead_matches_v4.csv"
SUMMARY = ROOT / "outputs" / "contextual_review_citation_lead_matches_v4.json"
FIELDS = (
    "lead_id",
    "review_record_key",
    "cited_title",
    "cited_authors_year",
    "cited_doi_or_url",
    "proposed_indicator_or_construct",
    "match_status",
    "matched_record_key",
    "matched_doi",
    "matched_title",
    "matched_year",
    "match_basis",
    "error",
)


def _read(path: Path) -> list[dict[str, str]]:
    """Read the frozen H2 citation-lead ledger."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _score(title: str, result: dict[str, Any]) -> tuple[int, int]:
    """Rank candidates by exact normalized title then token overlap."""
    expected = normalize_text(title)
    actual = normalize_text(
        str(result.get("display_name") or result.get("title") or "")
    )
    if expected and expected == actual:
        return (1, len(expected))
    tokens = set(expected.split())
    return (0, len(tokens & set(actual.split())))


def _fetch(lead: dict[str, str], key: str) -> tuple[dict[str, Any] | None, str, str]:
    """Find a high-confidence OpenAlex match without revealing the API key."""
    doi = normalize_doi(lead.get("cited_doi_or_url", ""))
    title = str(lead.get("cited_title", "")).strip()
    try:
        if doi:
            payload = fetch_json(
                openalex_url(filter_expression=f"doi:{doi}", per_page=5, api_key=key)
            )
            results = payload.get("results", [])
            if isinstance(results, list) and results:
                return dict(results[0]), "doi_exact", ""
        payload = fetch_json(openalex_url(expression=title, per_page=5, api_key=key))
        results = payload.get("results", [])
        if not isinstance(results, list) or not results:
            return None, "", "NO_OPENALEX_MATCH"
        candidates = [item for item in results if isinstance(item, dict)]
        if not candidates:
            return None, "", "NO_OPENALEX_MATCH"
        best = max(candidates, key=lambda item: _score(title, item))
        exact, overlap = _score(title, best)
        if exact:
            return best, "title_exact", ""
        if overlap >= 3:
            return best, "title_token_overlap", ""
        return None, "", "LOW_CONFIDENCE_TITLE_MATCH"
    except (
        OSError,
        RuntimeError,
        ValueError,
        requests.RequestException,
    ) as error:
        return None, "", safe_provider_error(error)


def retrieve(
    connection: sqlite3.Connection, input_path: Path, output: Path, summary: Path
) -> dict[str, Any]:
    """Retrieve, store, and export non-authorizing original-source matches."""
    leads = _read(input_path)
    keys = openalex_api_keys() or [""]
    rows: list[dict[str, str]] = []
    matched = 0
    for index, lead in enumerate(leads, start=1):
        item, basis, error = _fetch(lead, keys[(index - 1) % len(keys)])
        row = {
            "lead_id": f"RCL{index:03d}",
            "review_record_key": str(lead.get("review_record_key", "")),
            "cited_title": str(lead.get("cited_title", "")),
            "cited_authors_year": str(lead.get("cited_authors_year", "")),
            "cited_doi_or_url": str(lead.get("cited_doi_or_url", "")),
            "proposed_indicator_or_construct": str(
                lead.get("proposed_indicator_or_construct", "")
            ),
            "match_status": "unmatched",
            "matched_record_key": "",
            "matched_doi": "",
            "matched_title": "",
            "matched_year": "",
            "match_basis": basis,
            "error": error,
        }
        if item is not None:
            record = openalex_record(item, "h2_review_citation_lead")
            insert_openalex_record(connection, record)
            row.update(
                {
                    "match_status": "matched",
                    "matched_record_key": str(record["record_key"]),
                    "matched_doi": str(record["doi"]),
                    "matched_title": str(record["title"]),
                    "matched_year": str(record["publication_year"] or ""),
                }
            )
            matched += 1
        rows.append(row)
    connection.commit()
    write_csv(output, rows, FIELDS)
    result = {
        "schema_version": "contextual_review_citation_lead_retrieval_v4",
        "lead_count": len(leads),
        "matched_count": matched,
        "unmatched_count": len(leads) - matched,
        "input_sha256": sha256_file(input_path),
        "output_path": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "formal_k_q_p_changed": False,
        "selection_authorization": False,
    }
    write_json(summary, result)
    log_event(
        connection,
        "review_citation_lead_retrieval",
        "contextual_review",
        "batch_001",
        result,
    )
    connection.commit()
    return result


def main() -> None:
    """Run the read-to-match citation-lead recovery routine."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        result = retrieve(
            connection,
            args.input.resolve(),
            args.output.resolve(),
            args.summary.resolve(),
        )
    finally:
        connection.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
