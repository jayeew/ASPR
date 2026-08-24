"""Validate and import H2-adjudicated contextual full-text batch-two evidence."""

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
OUTPUT_DIR = ROOT / "outputs"
SOURCE_INPUT = OUTPUT_DIR / "contextual_fulltext_source_review_H2_batch2_v4.csv"
MENTION_INPUT = OUTPUT_DIR / "contextual_fulltext_indicator_mentions_H2_batch2_v4.csv"
SOURCES = OUTPUT_DIR / "contextual_fulltext_source_review_H2_batch2_completed_v4.csv"
MENTIONS = (
    OUTPUT_DIR / "contextual_fulltext_indicator_mentions_H2_batch2_completed_v4.csv"
)
SUMMARY = OUTPUT_DIR / "contextual_fulltext_h2_batch2_import_summary_v4.json"
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
SOURCE_H2_FIELDS = {"h2_final_source_disposition", "h2_final_source_notes"}
MENTION_H2_FIELDS = {
    "h2_decision",
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
}


def _read(path: Path) -> list[dict[str, str]]:
    """Read one UTF-8 CSV with a mandatory header."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing header: {path}")
        return list(reader)


def _index(
    rows: list[dict[str, str]], keys: tuple[str, ...]
) -> dict[tuple[str, ...], dict[str, str]]:
    """Index rows by stable identity and reject duplicate or blank primary keys."""
    result: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        identity = tuple(str(row.get(field, "")).strip() for field in keys)
        if not identity[0] or identity in result:
            raise ValueError(f"Duplicate or blank row identity: {identity}")
        result[identity] = row
    return result


def _assert_only_h2_fields_changed(
    input_rows: list[dict[str, str]],
    completed_rows: list[dict[str, str]],
    keys: tuple[str, ...],
    allowed_changes: set[str],
) -> None:
    """Ensure the adjudicator did not modify AI/H1 or frozen evidence columns."""
    original, completed = _index(input_rows, keys), _index(completed_rows, keys)
    if set(original) != set(completed):
        raise ValueError("H2 output no longer has the frozen input row identities.")
    for identity, before in original.items():
        after = completed[identity]
        changed = {
            field
            for field in set(before) | set(after)
            if str(before.get(field, "")) != str(after.get(field, ""))
        }
        if not changed <= allowed_changes:
            raise ValueError(
                f"H2 changed protected fields for {identity}: {sorted(changed)}"
            )


def _candidate_id(record_key: str, canonical_name: str) -> str:
    """Create a deterministic ID from its source and normalized candidate name."""
    return (
        "CFT_"
        + json_hash({"record_key": record_key, "canonical_name_en": canonical_name})[
            :16
        ]
    )


def import_evidence(
    connection: Any,
    source_input: Path,
    mention_input: Path,
    sources: Path,
    mentions: Path,
    summary: Path,
    batch_id: str,
) -> dict[str, Any]:
    """Import only H2-approved source definitions and candidate extractions."""
    source_rows, final_source_rows = _read(source_input), _read(sources)
    mention_rows, final_mention_rows = _read(mention_input), _read(mentions)
    _assert_only_h2_fields_changed(
        source_rows, final_source_rows, ("record_key",), SOURCE_H2_FIELDS
    )
    _assert_only_h2_fields_changed(
        mention_rows,
        final_mention_rows,
        ("record_key", "ai_canonical_name_en", "h1_canonical_name_en"),
        MENTION_H2_FIELDS,
    )
    source_sha, mention_sha = sha256_file(sources), sha256_file(mentions)
    formula_sources: set[str] = set()
    for row in final_source_rows:
        key = str(row["record_key"]).strip()
        disposition = str(row.get("h2_final_source_disposition", "")).strip()
        notes = str(row.get("h2_final_source_notes", "")).strip()
        if disposition not in SOURCE_DISPOSITIONS or not notes:
            raise ValueError(f"Invalid H2 source disposition: {key}")
        if (
            connection.execute(
                "SELECT 1 FROM records WHERE record_key = ?", (key,)
            ).fetchone()
            is None
        ):
            raise ValueError(f"Unknown source record: {key}")
        if disposition == "formula_or_application":
            formula_sources.add(key)
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
            (key, disposition, notes, str(sources.resolve()), source_sha, utc_now()),
        )
    retained = 0
    required = MENTION_H2_FIELDS - {"h2_decision"}
    for row in final_mention_rows:
        decision = str(row.get("h2_decision", "")).strip()
        if decision not in MENTION_DECISIONS:
            raise ValueError("Invalid or missing H2 candidate decision")
        if decision == "reject":
            continue
        if any(not str(row.get(field, "")).strip() for field in required):
            raise ValueError("Retained H2 candidate lacks required evidence fields")
        record_key = str(row["record_key"]).strip()
        if record_key not in formula_sources:
            raise ValueError("Retained candidate lacks H2 formula/application source")
        scope = str(row["scope_role"])
        if scope not in SCOPE_ROLES or parse_bool(
            row["requires_future"], "requires_future"
        ):
            raise ValueError(
                "Retained candidate violates scope or future-information rule"
            )
        fields = (
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
                _candidate_id(record_key, str(row["canonical_name_en"])),
                record_key,
                *[str(row[field]) for field in fields],
                int(parse_bool(row["requires_future"], "requires_future")),
                str(row["extraction_notes"]),
                decision,
                str(mentions.resolve()),
                mention_sha,
                utc_now(),
            ),
        )
        retained += 1
    result = {
        "schema_version": "contextual_fulltext_h2_batch2_import_v4",
        "source_rows": len(final_source_rows),
        "formula_or_application_sources": len(formula_sources),
        "retained_candidates": retained,
        "source_sha256": source_sha,
        "mention_sha256": mention_sha,
        "selection_authorization": False,
    }
    log_event(
        connection,
        "contextual_fulltext_h2_import",
        "contextual_fulltext_batch",
        batch_id,
        result,
    )
    connection.commit()
    write_json(summary, result)
    return result


def main() -> None:
    """Run the batch-two H2 evidence importer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--source-input", type=Path, default=SOURCE_INPUT)
    parser.add_argument("--mention-input", type=Path, default=MENTION_INPUT)
    parser.add_argument("--sources", type=Path, default=SOURCES)
    parser.add_argument("--mentions", type=Path, default=MENTIONS)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--batch-id", default="batch_002")
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        result = import_evidence(
            connection,
            args.source_input.resolve(),
            args.mention_input.resolve(),
            args.sources.resolve(),
            args.mentions.resolve(),
            args.summary.resolve(),
            str(args.batch_id),
        )
    finally:
        connection.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
