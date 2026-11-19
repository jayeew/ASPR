"""Export unique contextual probe hits for independent source-recovery screening."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from common import DATABASE_PATH, sha256_file, write_csv, write_json

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "contextual_source_screening_input_v4.csv"
MANIFEST = ROOT / "outputs" / "contextual_source_screening_input_v4.manifest.json"
FIELDS = (
    "record_key",
    "doi",
    "title",
    "abstract",
    "openalex_language",
    "publication_year",
    "work_type",
    "source_url",
    "linked_v3_feature_ids_json",
    "linked_v3_labels_json",
    "screen_decision",
    "evidence_span",
    "rationale",
)


def export(
    connection: sqlite3.Connection,
    output: Path,
    manifest: Path,
    unreviewed_only: bool,
    maximum: int | None,
) -> dict[str, Any]:
    """Export each contextual hit once, with linked H2-routed labels visible."""
    rows: list[dict[str, Any]] = []
    for record in connection.execute(
        """
        SELECT r.record_key, r.doi, r.title, r.abstract, r.language,
               r.publication_year, r.work_type, r.source_url,
               GROUP_CONCAT(h.v3_feature_id, char(31)) AS feature_ids,
               GROUP_CONCAT(x.v3_canonical_name_en, char(31)) AS labels
        FROM v3_contextual_recovery_hits h
        JOIN records r USING(record_key)
        JOIN v3_coverage_reconciliation x USING(v3_feature_id)
        LEFT JOIN contextual_source_final screened USING(record_key)
        WHERE (? = 0 OR screened.record_key IS NULL)
        GROUP BY r.record_key
        ORDER BY COUNT(DISTINCT h.v3_feature_id) DESC, r.record_key
        """,
        (int(unreviewed_only),),
    ):
        ids = (
            []
            if not record["feature_ids"]
            else sorted(str(record["feature_ids"]).split(chr(31)))
        )
        labels = (
            [] if not record["labels"] else sorted(str(record["labels"]).split(chr(31)))
        )
        if maximum is not None and len(rows) >= maximum:
            break
        rows.append(
            {
                "record_key": record["record_key"],
                "doi": record["doi"],
                "title": record["title"],
                "abstract": record["abstract"],
                "openalex_language": record["language"],
                "publication_year": record["publication_year"],
                "work_type": record["work_type"],
                "source_url": record["source_url"],
                "linked_v3_feature_ids_json": json.dumps(ids, ensure_ascii=False),
                "linked_v3_labels_json": json.dumps(labels, ensure_ascii=False),
                "screen_decision": "",
                "evidence_span": "",
                "rationale": "",
            }
        )
    write_csv(output, rows, FIELDS)
    result = {
        "schema_version": "contextual_source_screening_export_v4",
        "row_count": len(rows),
        "output_path": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "decision_vocabulary": [
            "include_definition_or_review",
            "exclude_not_relevant",
            "uncertain",
        ],
        "scope": "bounded_h2_contextual_probe_hits_not_formal_literature_screening",
        "unreviewed_only": unreviewed_only,
        "maximum": maximum,
    }
    write_json(manifest, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--unreviewed-only", action="store_true")
    parser.add_argument("--maximum", type=int)
    args = parser.parse_args()
    connection = sqlite3.connect(args.database.resolve())
    connection.row_factory = sqlite3.Row
    try:
        print(
            json.dumps(
                export(
                    connection,
                    args.output.resolve(),
                    args.manifest.resolve(),
                    args.unreviewed_only,
                    args.maximum,
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
