"""Export H2-cited original-source matches for independent title/abstract screening."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import DATABASE_PATH, ROOT, sha256_file, write_csv, write_json

INPUT = ROOT / "outputs" / "contextual_review_citation_lead_matches_v4.csv"
OUTPUT = ROOT / "outputs" / "review_citation_lead_screening_input_v4.csv"
MANIFEST = ROOT / "outputs" / "review_citation_lead_screening_input_v4.manifest.json"
FIELDS = (
    "record_key",
    "doi",
    "title",
    "abstract",
    "openalex_language",
    "publication_year",
    "work_type",
    "source_url",
    "citation_lead_ids_json",
    "proposed_constructs_json",
    "screen_decision",
    "evidence_span",
    "rationale",
)


def _matches(path: Path) -> list[dict[str, str]]:
    """Read only successful OpenAlex original-source matches."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("match_status") == "matched"
        ]


def export(
    connection: sqlite3.Connection, input_path: Path, output: Path, manifest: Path
) -> dict[str, Any]:
    """Create a deduplicated, non-authorizing screening input."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _matches(input_path):
        grouped[str(row["matched_record_key"])].append(row)
    rows: list[dict[str, Any]] = []
    for record_key, leads in sorted(grouped.items()):
        record = connection.execute(
            """SELECT record_key, doi, title, abstract, language, publication_year,
                      work_type, source_url FROM records WHERE record_key = ?""",
            (record_key,),
        ).fetchone()
        if record is None:
            raise ValueError(f"Citation lead record is absent: {record_key}")
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
                "citation_lead_ids_json": json.dumps(
                    sorted(str(lead["lead_id"]) for lead in leads), ensure_ascii=False
                ),
                "proposed_constructs_json": json.dumps(
                    sorted(
                        str(lead["proposed_indicator_or_construct"]) for lead in leads
                    ),
                    ensure_ascii=False,
                ),
                "screen_decision": "",
                "evidence_span": "",
                "rationale": "",
            }
        )
    write_csv(output, rows, FIELDS)
    result = {
        "schema_version": "review_citation_lead_screening_input_v4",
        "row_count": len(rows),
        "input_sha256": sha256_file(input_path),
        "output_path": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "selection_authorization": False,
    }
    write_json(manifest, result)
    return result


def main() -> None:
    """Write the citation-source screening input and manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    connection = sqlite3.connect(args.database.resolve())
    connection.row_factory = sqlite3.Row
    try:
        result = export(
            connection,
            args.input.resolve(),
            args.output.resolve(),
            args.manifest.resolve(),
        )
    finally:
        connection.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
