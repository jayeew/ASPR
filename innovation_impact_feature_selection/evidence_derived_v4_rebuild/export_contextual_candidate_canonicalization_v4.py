"""Export H2-retained full-text candidates for blind family canonicalization.

This is deliberately a *pre-promotion* step.  A retained extraction candidate is
not yet an indicator-family record and cannot enter feature selection until the
two blind reviewers and H2 have resolved synonymy, scope and evidence sufficiency.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from common import DATABASE_PATH, ROOT, sha256_file, write_csv, write_json
from database import initialize

OUTPUT = ROOT / "outputs" / "contextual_candidate_canonicalization_input_v4.csv"
MANIFEST = ROOT / "outputs" / "contextual_candidate_canonicalization_manifest_v4.json"


def export_candidates(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return retained candidates plus immutable source and full-text provenance."""
    rows = connection.execute("""
        SELECT c.*, r.doi, r.title, r.publication_year, a.local_path,
               a.sha256 AS fulltext_sha256, a.access_statement
        FROM contextual_fulltext_indicator_candidates c
        JOIN records r USING(record_key)
        JOIN fulltext_acquisitions a USING(record_key)
        LEFT JOIN contextual_candidate_canonicalization_final f USING(candidate_id)
        WHERE c.h2_decision = 'retain_as_candidate' AND a.status = 'downloaded'
          AND f.candidate_id IS NULL
        ORDER BY c.canonical_name_en, c.record_key
        """).fetchall()
    return [dict(row) for row in rows]


def main() -> None:
    """Write the immutable input sheet for independent reviewer coding."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        rows = export_candidates(connection)
    finally:
        connection.close()
    if not rows:
        raise ValueError(
            "No H2-retained candidates with a lawful downloaded full text."
        )
    fields = list(rows[0])
    write_csv(args.output.resolve(), rows, fields)
    result = {
        "schema_version": "contextual_candidate_canonicalization_export_v4",
        "row_count": len(rows),
        "output_path": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output.resolve()),
        "scope": "pre-promotion blind canonicalization; no feature-selection authority",
        "reviewer_editable_fields": [
            "{ROLE}_family_name_en",
            "{ROLE}_merge_or_split_reason",
            "{ROLE}_formula_reproducible",
            "{ROLE}_t0_computable",
            "{ROLE}_scope_role",
            "{ROLE}_missing_rule_status",
            "{ROLE}_promotion_decision",
            "{ROLE}_rationale",
        ],
    }
    write_json(args.manifest.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
