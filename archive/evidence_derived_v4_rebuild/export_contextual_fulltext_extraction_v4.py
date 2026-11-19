"""Export acquired contextual full texts for independent formula extraction."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from common import DATABASE_PATH, read_json, sha256_file, write_csv, write_json

ROOT = Path(__file__).resolve().parent
TEXT_REPORT = ROOT / "outputs" / "contextual_fulltext_text_extraction_v4.json"
OUTPUT = ROOT / "outputs" / "contextual_fulltext_extraction_input_v4.csv"
MANIFEST = ROOT / "outputs" / "contextual_fulltext_extraction_input_v4.manifest.json"
BRIEF = ROOT / "CONTEXTUAL_FULLTEXT_EXTRACTION_BRIEF_V4.md"
FIELDS = (
    "record_key",
    "doi",
    "title",
    "text_path",
    "text_sha256",
    "pdf_path",
    "pdf_sha256",
    "linked_v3_feature_ids_json",
    "linked_v3_labels_json",
    "source_disposition",
    "source_notes",
)


def build(
    connection: sqlite3.Connection,
    text_report: Path,
    output: Path,
    manifest: Path,
    brief: Path,
    include_reviewed: bool,
) -> dict[str, Any]:
    """Bind each text snapshot to the historical labels that led to it."""
    report = read_json(text_report)
    documents = report.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("No contextual full-text documents are available")
    rows: list[dict[str, Any]] = []
    for document in documents:
        key = str(document["record_key"])
        previously_reviewed = connection.execute(
            "SELECT 1 FROM contextual_fulltext_source_final WHERE record_key = ?",
            (key,),
        ).fetchone()
        if previously_reviewed is not None and not include_reviewed:
            continue
        linked = connection.execute(
            """
            SELECT h.v3_feature_id, r.v3_canonical_name_en
            FROM v3_contextual_recovery_hits h
            JOIN v3_coverage_reconciliation r USING(v3_feature_id)
            WHERE h.record_key = ? ORDER BY h.v3_feature_id
            """,
            (key,),
        ).fetchall()
        rows.append(
            {
                "record_key": key,
                "doi": document["doi"],
                "title": document["title"],
                "text_path": document["text_path"],
                "text_sha256": document["text_sha256"],
                "pdf_path": document["pdf_path"],
                "pdf_sha256": document["pdf_sha256"],
                "linked_v3_feature_ids_json": json.dumps(
                    [str(x[0]) for x in linked], ensure_ascii=False
                ),
                "linked_v3_labels_json": json.dumps(
                    [str(x[1]) for x in linked], ensure_ascii=False
                ),
                "source_disposition": "",
                "source_notes": "",
            }
        )
    write_csv(output, rows, FIELDS)
    brief.write_text(
        """# Independent contextual full-text indicator extraction (v4)

Read each local English text file identified in the input.  The linked v3
labels are discovery leads only, not approved features.

For every source write a source review with exactly one disposition:

- `formula_or_application`: it explicitly defines or applies a paper-level,
  T0-computable innovation/potential-impact/opportunity/control indicator;
- `review_discovery_only`: useful terminology or cited original sources but no
  source-authorized formula/application; or
- `no_relevant_indicator`.

For every candidate formula/application, extract a separate mention row with:
`record_key, raw_name_en, canonical_name_en, source_role,
formula_location, evidence_span, formula, parameters, required_data,
maximum_information_time, scope_role, requires_future, extraction_notes`.

`source_role` must be one of `original_definition`, `original_application`,
`validation`, `review_discovery`, or `mathematical_foundation`.  Do not infer a
formula from prose, do not use a review as sole formula authority, and do not
make a final feature/dimension decision.  Quote a compact, exact English span
and a page/section/equation location whenever a formula is reported.
""",
        encoding="utf-8",
    )
    result = {
        "schema_version": "contextual_fulltext_extraction_input_v4",
        "row_count": len(rows),
        "text_report_sha256": sha256_file(text_report),
        "input_path": str(output.resolve()),
        "input_sha256": sha256_file(output),
        "brief_path": str(brief.resolve()),
        "brief_sha256": sha256_file(brief),
    }
    write_json(manifest, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--text-report", type=Path, default=TEXT_REPORT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--brief", type=Path, default=BRIEF)
    parser.add_argument("--include-reviewed", action="store_true")
    args = parser.parse_args()
    connection = sqlite3.connect(args.database.resolve())
    connection.row_factory = sqlite3.Row
    try:
        print(
            json.dumps(
                build(
                    connection,
                    args.text_report.resolve(),
                    args.output.resolve(),
                    args.manifest.resolve(),
                    args.brief.resolve(),
                    args.include_reviewed,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
