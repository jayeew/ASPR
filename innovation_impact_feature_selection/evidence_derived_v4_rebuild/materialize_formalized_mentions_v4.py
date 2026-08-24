"""Materialize only H2-formalized evidence into canonical indicator families."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from common import DATABASE_PATH, json_hash, write_json
from database import initialize, log_event
from indicators import build_indicator_families

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "outputs" / "formalized_mention_materialization_summary_v4.json"


def _mention_id(candidate_id: str) -> str:
    """Derive a deterministic ID for a formalized evidence mention."""
    return "V4FORM_" + json_hash({"candidate_id": candidate_id})[:16].upper()


def _rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """Read exactly the H2-promoted formula/data records with full provenance."""
    return connection.execute("""
        SELECT f.*, c.record_key, c.raw_name_en, c.source_role,
               c.formula_location, c.evidence_span, c.maximum_information_time,
               c.scope_role, c.requires_future, c.extraction_notes,
               k.formula_reproducible, k.t0_computable,
               r.doi, a.final_url, a.local_path, a.sha256, a.access_statement
        FROM contextual_formalization_final f
        JOIN contextual_candidate_canonicalization_final k USING(candidate_id)
        JOIN contextual_fulltext_indicator_candidates c USING(candidate_id)
        JOIN records r USING(record_key)
        JOIN fulltext_acquisitions a USING(record_key)
        WHERE f.formalization_decision = 'promote_for_formalization'
          AND k.promotion_decision = 'promote_for_formalization'
          AND a.status = 'downloaded'
        ORDER BY f.canonical_name_en, f.candidate_id
        """).fetchall()


def _clear_derived(connection: sqlite3.Connection) -> None:
    """Clear the superseded v4 family layer, never source/candidate evidence."""
    for table in (
        "dimension_decisions",
        "feature_decisions",
        "candidate_dimensions",
        "dimension_coding",
        "feature_data_audit",
        "feature_data_correspondence_reviews",
        "feature_operationalization_reviews",
        "indicator_families",
        "indicator_mention_reviews",
        "indicator_mentions",
    ):
        connection.execute(f"DELETE FROM {table}")


def materialize(connection: sqlite3.Connection) -> dict[str, Any]:
    """Rebuild canonical families solely from formalized H2-approved evidence."""
    rows = _rows(connection)
    _clear_derived(connection)
    for row in rows:
        candidate_id = str(row["candidate_id"])
        research_group = str(row["research_group"])
        group_id = "rg_" + json_hash({"research_group": research_group})[:16]
        connection.execute(
            """
            INSERT INTO indicator_mentions(
                mention_id, record_key, raw_name_en, canonical_name_en, label_zh,
                source_id, research_group, research_group_id, research_group_evidence,
                source_role, formula_location, evidence_span, formula, units, parameters,
                direction, missing_rule, required_data_json, maximum_information_time,
                scope_role, validation_summary, evidence_direction, negative_evidence,
                fulltext_source_url, fulltext_local_path, fulltext_sha256, fulltext_license,
                english_fulltext_verified, article_level, primary_or_foundational_evidence,
                formula_reproducible, t0_computable, requires_future, data_status,
                bias_policy, fatal_validity_concern, uses_outcome_for_selection,
                quality_audit_status, nonconstant, h2_approved, evidence_strength,
                stability_score, stability_basis, selection_priority, redundancy_family,
                extracted_by, verified_by, verification_notes, adjudication_notes, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?)
            """,
            (
                _mention_id(candidate_id),
                row["record_key"],
                row["raw_name_en"],
                row["canonical_name_en"],
                row["label_zh"],
                row["doi"],
                research_group,
                group_id,
                row["research_group_evidence"],
                row["source_role"],
                row["formula_location"],
                row["evidence_span"],
                row["formula"],
                row["units"],
                row["parameters"],
                row["direction"],
                row["missing_rule"],
                row["required_data_json"],
                row["maximum_information_time"],
                row["scope_role"],
                "Formula/data correspondence formally adjudicated; predictive validation is not assumed.",
                "definition_or_application",
                row["rationale"],
                row["final_url"],
                row["local_path"],
                row["sha256"],
                row["access_statement"],
                1,
                1,
                int(
                    row["source_role"]
                    in {"original_definition", "original_application"}
                ),
                int(row["formula_reproducible"]),
                int(row["t0_computable"]),
                int(row["requires_future"]),
                "derivable_from_audited_inputs",
                "T0-only; operational-equivalence transform and missing rule are audited.",
                0,
                0,
                "not_audited",
                0,
                1,
                "original_application",
                0.0,
                "Stability requires downstream data-quality audit.",
                2,
                row["canonical_name_en"],
                "AI|H1",
                "AI|H1|H2",
                "Three-role formalization with H2 final decision.",
                row["rationale"],
                "candidate",
            ),
        )
    family_count = build_indicator_families(connection)
    result = {
        "schema_version": "formalized_mention_materialization_v4",
        "materialized_mentions": len(rows),
        "canonical_families": family_count,
        "selection_authorization": False,
    }
    log_event(
        connection,
        "formalized_mention_materialization",
        "indicator_library",
        "h2_formalized_only",
        result,
    )
    connection.commit()
    return result


def main() -> None:
    """Run deterministic family materialization."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()
    connection = initialize(args.database.resolve())
    try:
        result = materialize(connection)
    finally:
        connection.close()
    write_json(args.summary.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
