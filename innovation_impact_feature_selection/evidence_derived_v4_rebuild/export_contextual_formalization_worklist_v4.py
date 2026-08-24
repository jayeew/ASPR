"""Export H2-promoted candidates for formula and local-data formalization.

Promotion here is intentionally narrow: it authorizes a three-role definition
and data-correspondence review, not feature selection or a claim that the
formula can already be materialized in the local corpus.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from common import DATABASE_PATH, ROOT, sha256_file, write_csv, write_json
from database import initialize

OUTPUT = ROOT / "outputs" / "contextual_formalization_input_v4.csv"
MANIFEST = ROOT / "outputs" / "contextual_formalization_input_manifest_v4.json"
INVENTORY = ROOT / "outputs" / "local_t0_input_inventory_v4.json"


def build_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return only H2-promoted candidates with immutable evidence provenance."""
    rows = connection.execute("""
        SELECT c.candidate_id, c.record_key, r.doi, r.title, r.publication_year,
               c.raw_name_en, c.canonical_name_en AS extracted_name_en,
               f.family_name_en, f.merge_or_split_reason, c.source_role,
               c.formula_location, c.evidence_span, c.formula, c.parameters,
               c.required_data, c.maximum_information_time,
               f.scope_role, c.extraction_notes, a.local_path AS fulltext_local_path,
               a.sha256 AS fulltext_sha256, f.missing_rule_status,
               f.rationale AS canonicalization_h2_rationale
        FROM contextual_candidate_canonicalization_final f
        JOIN contextual_fulltext_indicator_candidates c USING(candidate_id)
        JOIN records r USING(record_key)
        JOIN fulltext_acquisitions a USING(record_key)
        LEFT JOIN contextual_formalization_final z USING(candidate_id)
        WHERE f.promotion_decision = 'promote_for_formalization'
          AND z.candidate_id IS NULL
        ORDER BY f.family_name_en, c.candidate_id
        """).fetchall()
    return [dict(row) for row in rows]


def main() -> None:
    """Write the scoped formalization worklist and frozen inventory reference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--inventory", type=Path, default=INVENTORY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if not args.inventory.is_file():
        raise FileNotFoundError(args.inventory)
    connection = initialize(args.database.resolve())
    try:
        rows = build_rows(connection)
    finally:
        connection.close()
    if not rows:
        raise ValueError("No H2-promoted contextual candidates are available.")
    write_csv(args.output.resolve(), rows, list(rows[0]))
    result = {
        "schema_version": "contextual_formalization_worklist_v4",
        "row_count": len(rows),
        "input_path": str(args.output.resolve()),
        "input_sha256": sha256_file(args.output.resolve()),
        "inventory_path": str(args.inventory.resolve()),
        "inventory_sha256": sha256_file(args.inventory.resolve()),
        "scope": "formula and local-data formalization only; no final selection",
        "required_reviewer_fields": [
            "{ROLE}_canonical_name_en",
            "{ROLE}_label_zh",
            "{ROLE}_formula",
            "{ROLE}_units",
            "{ROLE}_parameters",
            "{ROLE}_direction",
            "{ROLE}_missing_rule",
            "{ROLE}_required_data_json",
            "{ROLE}_research_group",
            "{ROLE}_research_group_evidence",
            "{ROLE}_data_match_decision",
            "{ROLE}_local_source_ids_json",
            "{ROLE}_local_columns_json",
            "{ROLE}_derivation_description",
            "{ROLE}_formalization_decision",
            "{ROLE}_rationale",
        ],
    }
    write_json(args.manifest.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
