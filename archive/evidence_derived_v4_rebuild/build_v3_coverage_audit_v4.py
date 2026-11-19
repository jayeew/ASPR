"""Build a non-authorizing v3-to-v4 indicator-family coverage audit.

The recovered v3 library is a discovery-coverage benchmark, never a source of
formula approval or feature selection.  Each historical family must therefore
receive a transparent v4 disposition before v4 can claim saturated indicator
discovery.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from common import DATABASE_PATH, normalize_term, sha256_file, write_csv, write_json


ROOT = Path(__file__).resolve().parent
ANCHOR = ROOT / "outputs" / "v3_coverage_anchor" / "complete_indicator_library_v3.csv"
DEFAULT_CSV = ROOT / "outputs" / "v3_to_v4_indicator_coverage_audit.csv"
DEFAULT_JSON = ROOT / "outputs" / "v3_to_v4_indicator_coverage_summary.json"
STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with",
    "paper", "article", "publication", "research", "scientific", "academic", "citation", "impact", "potential",
    "indicator", "index", "metric", "measure", "score", "predictor", "feature", "count", "number", "level",
}
FIELDS = (
    "v3_feature_id", "v3_canonical_name_en", "v3_scope_role", "v3_t0", "v3_english_fulltext_verified",
    "v3_dimension_id", "coverage_disposition", "matched_v4_feature_ids_json", "matched_v4_terms_json",
    "candidate_v4_record_keys_json", "candidate_v4_record_count", "required_next_action", "audit_note",
)


def _tokens(value: str) -> set[str]:
    return {
        item for item in re.findall(r"[a-z0-9]+", normalize_term(value))
        if len(item) >= 3 and item not in STOPWORDS
    }


def _read_anchor(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 432:
        raise ValueError(f"v3 coverage anchor must contain 432 families, found {len(rows)}")
    return rows


def _record_candidates(
    records: Iterable[tuple[str, set[str]]], tokens: set[str]
) -> list[str]:
    """Return a conservative, non-authorizing lexical discovery queue."""
    if len(tokens) < 2:
        return []
    candidates: list[str] = []
    for record_key, haystack in records:
        overlap = len(tokens & haystack)
        # Two independent construct tokens, or all tokens for short labels.
        if overlap >= min(2, len(tokens)):
            candidates.append(str(record_key))
            if len(candidates) >= 25:
                break
    return candidates


def build(connection: sqlite3.Connection, anchor: Path, output: Path, summary: Path) -> dict[str, Any]:
    """Write every historical family with a non-authorizing v4 coverage status."""
    v3 = _read_anchor(anchor)
    v4_families = list(connection.execute(
        "SELECT feature_id, canonical_name_en FROM indicator_families ORDER BY feature_id"
    ))
    v4_terms = [str(row[0]) for row in connection.execute(
        "SELECT canonical_term FROM canonical_terms WHERE status='active' ORDER BY canonical_term_id"
    )]
    family_by_name: dict[str, list[str]] = {}
    for feature_id, name in v4_families:
        family_by_name.setdefault(normalize_term(str(name)), []).append(str(feature_id))
    term_by_name = {normalize_term(term): term for term in v4_terms}
    record_tokens = [
        (str(record_key), _tokens(f"{title or ''} {abstract or ''}"))
        for record_key, title, abstract in connection.execute(
            "SELECT record_key, title, abstract FROM records ORDER BY record_key"
        )
    ]
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for item in v3:
        name = str(item["canonical_name_en"])
        normalized = normalize_term(name)
        family_matches = sorted(family_by_name.get(normalized, []))
        term_matches = [term_by_name[normalized]] if normalized in term_by_name else []
        candidates = _record_candidates(record_tokens, _tokens(name))
        if family_matches:
            disposition = "recalled_as_v4_canonical_family"
            action = "Reconcile v3/v4 definitions and evidence; do not auto-approve."
        elif term_matches:
            disposition = "recalled_as_v4_term_only"
            action = "Route to source-level extraction and independent coding."
        elif candidates:
            disposition = "candidate_v4_record_queue"
            action = "Screen listed English records, then extract if eligible."
        else:
            disposition = "unrecalled_coverage_gap"
            action = "Run source/DOI recovery and term-expansion search for this v3 family."
        counts[disposition] += 1
        rows.append({
            "v3_feature_id": item["feature_id"], "v3_canonical_name_en": name,
            "v3_scope_role": item["scope_role"], "v3_t0": item["maximum_information_time"],
            "v3_english_fulltext_verified": item["english_fulltext_verified"],
            "v3_dimension_id": item["dimension_id"], "coverage_disposition": disposition,
            "matched_v4_feature_ids_json": json.dumps(family_matches, ensure_ascii=False),
            "matched_v4_terms_json": json.dumps(term_matches, ensure_ascii=False),
            "candidate_v4_record_keys_json": json.dumps(candidates, ensure_ascii=False),
            "candidate_v4_record_count": len(candidates), "required_next_action": action,
            "audit_note": "v3 is a coverage benchmark only; it does not authorize v4 inclusion, formula, mapping, or selection.",
        })
    write_csv(output, rows, FIELDS)
    recalled = counts["recalled_as_v4_canonical_family"]
    result = {
        "schema_version": "v3_to_v4_indicator_coverage_audit_v4",
        "anchor_path": str(anchor.resolve()), "anchor_sha256": sha256_file(anchor),
        "v3_family_count": len(v3), "v4_canonical_family_count": len(v4_families),
        "counts_by_disposition": dict(sorted(counts.items())),
        "canonical_family_recall": recalled / len(v3),
        "coverage_gate": {
            "all_historical_families_require_auditable_disposition": True,
            "passed": False,
            "interpretation": "The recovered 432 labels are a coverage benchmark, not a quota. Each must ultimately be reconciled as a v4 family, a documented merge, a documented exclusion, or an unresolved evidence gap before final v4 dimensions, feature selection, matrix freeze, or OOF training.",
        },
        "output_csv": str(output.resolve()), "output_csv_sha256": sha256_file(output),
    }
    write_json(summary, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--anchor", type=Path, default=ANCHOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--summary", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()
    connection = sqlite3.connect(args.database.resolve())
    try:
        print(json.dumps(build(connection, args.anchor.resolve(), args.output.resolve(), args.summary.resolve()), ensure_ascii=False, indent=2))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
