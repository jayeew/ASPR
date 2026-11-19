"""Import H2-adjudicated contextual source and formula-candidate evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common import (
    DATABASE_PATH,
    json_hash,
    parse_bool,
    sha256_file,
    utc_now,
    write_json,
)
from database import initialize, log_event

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "outputs" / "contextual_fulltext_source_review_H2_completed_v4.csv"
MENTIONS = (
    ROOT / "outputs" / "contextual_fulltext_indicator_mentions_H2_completed_v4.csv"
)
SUMMARY = ROOT / "outputs" / "contextual_fulltext_h2_import_summary_v4.json"
SOURCE_DISPOSITIONS = {
    "formula_or_application",
    "review_discovery_only",
    "no_relevant_indicator",
}
MENTION_DECISIONS = {"retain_as_candidate", "reject"}
SCOPE_ROLES = {
    "direct_innovation",
    "t0_substantive",
    "t0_opportunity",
    "context_control",
    "out_of_scope",
}


def read(path: Path) -> list[dict[str, str]]:
    """Read a completed H2 CSV."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def import_evidence(
    connection: Any, sources: Path, mentions: Path, summary: Path
) -> dict[str, Any]:
    """Store H2 source dispositions and retained formula candidates."""
    source_rows = read(sources)
    if len(source_rows) != 11:
        raise ValueError(f"Expected 11 H2 source reviews, found {len(source_rows)}")
    source_digest, mention_digest = sha256_file(sources), sha256_file(mentions)
    formula_sources: set[str] = set()
    for row in source_rows:
        key = str(row.get("record_key", "")).strip()
        disposition = str(row.get("h2_final_source_disposition", "")).strip()
        notes = str(row.get("h2_final_source_notes", "")).strip()
        if not key or disposition not in SOURCE_DISPOSITIONS or not notes:
            raise ValueError(f"Invalid H2 source disposition for {key!r}")
        if (
            connection.execute(
                "SELECT 1 FROM records WHERE record_key = ?", (key,)
            ).fetchone()
            is None
        ):
            raise ValueError(f"Unknown source record: {key}")
        formula_sources.add(key) if disposition == "formula_or_application" else None
        connection.execute(
            """
            INSERT INTO contextual_fulltext_source_final(
                record_key, final_disposition, notes, artifact_path,
                artifact_sha256, finalized_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_key) DO UPDATE SET
                final_disposition=excluded.final_disposition, notes=excluded.notes,
                artifact_path=excluded.artifact_path, artifact_sha256=excluded.artifact_sha256,
                finalized_at=excluded.finalized_at
            """,
            (key, disposition, notes, str(sources.resolve()), source_digest, utc_now()),
        )
    retained = 0
    for row in read(mentions):
        decision = str(row.get("h2_decision", "")).strip()
        if decision not in MENTION_DECISIONS:
            raise ValueError("Invalid H2 candidate decision")
        if decision == "reject":
            continue
        required = (
            "record_key",
            "raw_name_en",
            "canonical_name_en",
            "source_role",
            "formula_location",
            "evidence_span",
            "formula",
            "parameters",
            "required_data",
            "maximum_information_time",
            "scope_role",
            "requires_future",
            "extraction_notes",
        )
        if any(not str(row.get(field, "")).strip() for field in required):
            raise ValueError("Retained H2 candidate has incomplete evidence fields")
        key = str(row["record_key"])
        if key not in formula_sources:
            raise ValueError(
                "Retained candidate lacks H2 formula/application source disposition"
            )
        scope = str(row["scope_role"])
        if scope not in SCOPE_ROLES:
            raise ValueError("Retained candidate has invalid scope role")
        candidate_id = (
            "CFT_"
            + json_hash(
                {"record_key": key, "canonical_name_en": row["canonical_name_en"]}
            )[:16]
        )
        evidence_fields = (
            "raw_name_en",
            "canonical_name_en",
            "source_role",
            "formula_location",
            "evidence_span",
            "formula",
            "parameters",
            "required_data",
            "maximum_information_time",
            "scope_role",
        )
        connection.execute(
            """
            INSERT INTO contextual_fulltext_indicator_candidates(
                candidate_id, record_key, raw_name_en, canonical_name_en,
                source_role, formula_location, evidence_span, formula,
                parameters, required_data, maximum_information_time, scope_role,
                requires_future, extraction_notes, h2_decision, artifact_path,
                artifact_sha256, imported_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(record_key, canonical_name_en) DO UPDATE SET
                raw_name_en=excluded.raw_name_en, source_role=excluded.source_role,
                formula_location=excluded.formula_location, evidence_span=excluded.evidence_span,
                formula=excluded.formula, parameters=excluded.parameters,
                required_data=excluded.required_data, maximum_information_time=excluded.maximum_information_time,
                scope_role=excluded.scope_role, requires_future=excluded.requires_future,
                extraction_notes=excluded.extraction_notes, h2_decision=excluded.h2_decision,
                artifact_path=excluded.artifact_path, artifact_sha256=excluded.artifact_sha256,
                imported_at=excluded.imported_at
            """,
            (
                candidate_id,
                key,
                *[str(row[field]) for field in evidence_fields],
                int(parse_bool(row["requires_future"], "requires_future")),
                str(row["extraction_notes"]),
                decision,
                str(mentions.resolve()),
                mention_digest,
                utc_now(),
            ),
        )
        retained += 1
    result = {
        "schema_version": "contextual_fulltext_h2_import_v4",
        "source_rows": len(source_rows),
        "formula_or_application_sources": len(formula_sources),
        "retained_candidates": retained,
        "source_sha256": source_digest,
        "mention_sha256": mention_digest,
        "selection_authorization": False,
    }
    log_event(
        connection,
        "contextual_fulltext_h2_import",
        "contextual_fulltext_batch",
        "batch_001",
        result,
    )
    connection.commit()
    write_json(summary, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--sources", type=Path, default=SOURCES)
    parser.add_argument("--mentions", type=Path, default=MENTIONS)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        print(
            json.dumps(
                import_evidence(
                    connection,
                    args.sources.resolve(),
                    args.mentions.resolve(),
                    args.summary.resolve(),
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
