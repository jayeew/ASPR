from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pyarrow.parquet as parquet

from common import (
    sha256_file,
    utc_now,
    write_csv,
    write_json,
)
from database import (
    invalidate_stages,
    require_complete,
    set_stage,
    snapshot_import_file,
)


ROOT = Path(__file__).resolve().parent
LOCAL_DATA_ROOT = (
    ROOT.parent.parent
    / "data"
    / "knowledge_corpus"
    / "nature_multihorizon_v6_1_local"
)
DEFAULT_INVENTORY_PATH = (
    ROOT / "outputs" / "local_t0_input_inventory_v3.json"
)
ALLOWED_SOURCE_SPECS = {
    "papers_common": {
        "file": "papers_common_all.parquet",
        "role": "paper_identity_publication_context",
        "maximum_information_time": "T0",
    },
    "paper_references": {
        "file": "paper_references.parquet",
        "role": "focal_backward_citation_edges",
        "maximum_information_time": "T0",
    },
    "reference_metadata": {
        "file": "reference_metadata.parquet",
        "role": "referenced_work_prepublication_metadata",
        "maximum_information_time": "T0",
    },
    "historical_paper_references": {
        "file": "historical_paper_references.parquet",
        "role": "strictly_prior_reference_history",
        "maximum_information_time": "T0-1",
    },
    "historical_paper_sources": {
        "file": "historical_paper_sources.parquet",
        "role": "strictly_prior_cited_source_history",
        "maximum_information_time": "T0-1",
    },
    "field_citation_events": {
        "file": "field_citation_events_aggregated.parquet",
        "role": "strictly_prior_field_distance_inputs",
        "maximum_information_time": "T0-1",
    },
    "source_field_citation_events": {
        "file": "source_field_citation_events.parquet",
        "role": "strictly_prior_source_field_inputs",
        "maximum_information_time": "T0-1",
    },
    "target_openalex_metadata": {
        "file": "target_openalex_metadata.parquet",
        "role": "publication_time_authorship_affiliation_counts",
        "maximum_information_time": "T0",
    },
    "innovation_candidate_features": {
        "file": "innovation_candidate_features.parquet",
        "role": "outcome_blind_t0_innovation_candidates",
        "maximum_information_time": "T0",
    },
    "opportunity_features": {
        "file": "opportunity_features.parquet",
        "role": "outcome_blind_prepublication_opportunity_features",
        "maximum_information_time": "T0-1",
    },
    "control_features": {
        "file": "control_features_v6_1.parquet",
        "role": "publication_time_context_controls",
        "maximum_information_time": "T0",
    },
}
PROHIBITED_SOURCE_FILES = (
    "targets_zero_inclusive.parquet",
    "cohort_membership.parquet",
)
DECISIONS = {
    "exact_materialized",
    "exact_derivable",
    "candidate_formula_completion",
    "no_match",
    "construct_mismatch",
    "future_only",
    "insufficient_evidence",
}
EVIDENCE_FIELDS = (
    "feature_id",
    "canonical_name_en",
    "reviewer_role",
    "alias_names_evidence",
    "formula_evidence",
    "required_data_evidence",
    "maximum_information_time_evidence",
    "scope_role_evidence",
    "article_level_evidence",
    "t0_computable_evidence",
    "requires_future_evidence",
    "research_groups_evidence",
    "mention_ids_evidence",
    "local_input_inventory_path",
    "local_input_inventory_sha256",
    "match_decision",
    "local_source_ids_json",
    "local_columns_json",
    "derivation_description",
    "construct_equivalence_notes",
    "reason",
)
H2_FIELDS = (
    *EVIDENCE_FIELDS,
    "ai_payload_json",
    "h1_payload_json",
)


def _source_inventory(
    source_id: str,
    spec: Mapping[str, str],
    data_root: Path,
) -> Dict[str, Any]:
    """Hash and describe one allowed local T0 Parquet source."""
    path = (data_root / spec["file"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    schema = parquet.read_schema(path)
    metadata = parquet.ParquetFile(path).metadata
    return {
        "source_id": source_id,
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": metadata.num_rows,
        "role": spec["role"],
        "maximum_information_time": spec["maximum_information_time"],
        "columns": [
            {"name": field.name, "type": str(field.type)}
            for field in schema
        ],
    }


def build_local_t0_input_inventory(
    output_path: Path = DEFAULT_INVENTORY_PATH,
    data_root: Path = LOCAL_DATA_ROOT,
) -> Dict[str, Any]:
    """Build a deterministic inventory that excludes every outcome table."""
    sources = {
        source_id: _source_inventory(source_id, spec, data_root)
        for source_id, spec in ALLOWED_SOURCE_SPECS.items()
    }
    payload = {
        "schema_version": "local_t0_input_inventory_v3",
        "scope": (
            "Outcome-blind fields available no later than publication time; "
            "inclusion in this inventory does not establish construct "
            "equivalence for any indicator."
        ),
        "sources": sources,
        "prohibited_source_files": list(PROHIBITED_SOURCE_FILES),
        "outcome_columns_used": False,
        "target_count_influence": False,
    }
    write_json(output_path, payload)
    return {
        "path": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "source_count": len(sources),
        "column_count": sum(
            len(source["columns"]) for source in sources.values()
        ),
    }


def _evidence_row(
    family: sqlite3.Row,
    role: str,
    inventory_path: Path,
) -> Dict[str, Any]:
    """Build one protected family-to-input correspondence row."""
    return {
        "feature_id": family["feature_id"],
        "canonical_name_en": family["canonical_name_en"],
        "reviewer_role": role,
        "alias_names_evidence": family["alias_names_json"],
        "formula_evidence": family["formula"],
        "required_data_evidence": family["required_data_json"],
        "maximum_information_time_evidence": family[
            "maximum_information_time"
        ],
        "scope_role_evidence": family["scope_role"],
        "article_level_evidence": str(bool(family["article_level"])).lower(),
        "t0_computable_evidence": str(
            bool(family["t0_computable"])
        ).lower(),
        "requires_future_evidence": str(
            bool(family["requires_future"])
        ).lower(),
        "research_groups_evidence": family["research_groups_json"],
        "mention_ids_evidence": family["mention_ids_json"],
        "local_input_inventory_path": str(inventory_path.resolve()),
        "local_input_inventory_sha256": sha256_file(inventory_path),
        "match_decision": "",
        "local_source_ids_json": "",
        "local_columns_json": "",
        "derivation_description": "",
        "construct_equivalence_notes": "",
        "reason": "",
    }


def export_data_correspondence(
    connection: sqlite3.Connection,
    output_path: Path,
    reviewer_role: str,
    inventory_path: Path = DEFAULT_INVENTORY_PATH,
) -> int:
    """Export all canonical families for independent local-data matching."""
    require_complete(connection, ["indicators_extracted"])
    role = reviewer_role.strip().upper()
    if role not in {"AI", "H1", "H2"}:
        raise ValueError("Correspondence reviewer must be AI, H1, or H2")
    if not inventory_path.is_file():
        raise FileNotFoundError(inventory_path)
    rows: List[Dict[str, Any]] = []
    for family in connection.execute(
        """
        SELECT * FROM indicator_families
        WHERE status = 'candidate'
        ORDER BY feature_id
        """
    ):
        row = _evidence_row(family, role, inventory_path)
        if role == "H2":
            reviews = {
                str(review["reviewer_role"]): review
                for review in connection.execute(
                    """
                    SELECT * FROM feature_data_correspondence_reviews
                    WHERE feature_id = ?
                      AND reviewer_role IN ('AI', 'H1')
                    """,
                    (family["feature_id"],),
                )
            }
            if not {"AI", "H1"}.issubset(reviews):
                raise RuntimeError(
                    "H2 correspondence export requires AI and H1: "
                    f"{family['feature_id']}"
                )
            row["ai_payload_json"] = reviews["AI"]["payload_json"]
            row["h1_payload_json"] = reviews["H1"]["payload_json"]
        rows.append(row)
    write_csv(output_path, rows, H2_FIELDS if role == "H2" else EVIDENCE_FIELDS)
    return len(rows)


def _json_list(value: Any, field: str, feature_id: str) -> List[str]:
    """Parse a nonempty, deduplicated JSON string list."""
    try:
        parsed = json.loads(str(value or ""))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{field} is not JSON for {feature_id}"
        ) from error
    if not isinstance(parsed, list):
        raise ValueError(f"{field} must be a list for {feature_id}")
    values = sorted(
        {str(item).strip() for item in parsed if str(item).strip()}
    )
    if not values:
        raise ValueError(f"{field} is empty for {feature_id}")
    return values


def _assert_evidence(
    row: Mapping[str, Any],
    family: sqlite3.Row,
    inventory_path: Path,
) -> None:
    """Prevent reviewers from changing family or inventory evidence."""
    expected = _evidence_row(
        family,
        str(row.get("reviewer_role") or "").strip().upper(),
        inventory_path,
    )
    protected = EVIDENCE_FIELDS[:15]
    for field in protected:
        if str(row.get(field) or "") != str(expected[field] or ""):
            raise ValueError(
                f"Correspondence review changed {field}: "
                f"{family['feature_id']}"
            )


def import_data_correspondence(
    connection: sqlite3.Connection,
    input_path: Path,
) -> Dict[str, Any]:
    """Import one complete AI/H1/H2 family-to-data review artifact."""
    require_complete(connection, ["indicators_extracted"])
    snapshot_path = snapshot_import_file(
        connection,
        input_path,
        "feature_data_correspondence",
    )
    with snapshot_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError("Data correspondence artifact has no rows")
    roles = {
        str(row.get("reviewer_role") or "").strip().upper()
        for row in rows
    }
    if len(roles) != 1 or not roles.issubset({"AI", "H1", "H2"}):
        raise ValueError("One correspondence artifact requires one role")
    role = next(iter(roles))
    if role == "H1" and any(
        str(field).casefold().startswith(("ai_", "h2_"))
        for field in rows[0]
    ):
        raise ValueError("Blind H1 correspondence cannot see AI/H2 columns")
    family_rows = {
        str(row["feature_id"]): row
        for row in connection.execute(
            """
            SELECT * FROM indicator_families
            WHERE status = 'candidate'
            ORDER BY feature_id
            """
        )
    }
    observed_ids = {
        str(row.get("feature_id") or "").strip() for row in rows
    }
    if observed_ids != set(family_rows) or len(rows) != len(family_rows):
        raise ValueError(
            "Correspondence review must contain every candidate family "
            "exactly once"
        )
    inventory_paths = {
        str(row.get("local_input_inventory_path") or "").strip()
        for row in rows
    }
    if len(inventory_paths) != 1:
        raise ValueError("Correspondence rows mix local inventories")
    inventory_path = Path(next(iter(inventory_paths))).resolve()
    if not inventory_path.is_file():
        raise FileNotFoundError(inventory_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    sources = set(inventory["sources"])
    imported = 0
    positive = 0
    for row in rows:
        feature_id = str(row["feature_id"]).strip()
        family = family_rows[feature_id]
        _assert_evidence(row, family, inventory_path)
        decision = str(row.get("match_decision") or "").strip().casefold()
        if decision not in DECISIONS:
            raise ValueError(
                f"Invalid correspondence decision for {feature_id}: "
                f"{decision}"
            )
        reason = str(row.get("reason") or "").strip()
        if not reason:
            raise ValueError(
                f"Correspondence review lacks reason: {feature_id}"
            )
        prior_roles = {
            str(item[0])
            for item in connection.execute(
                """
                SELECT reviewer_role
                FROM feature_data_correspondence_reviews
                WHERE feature_id = ?
                """,
                (feature_id,),
            )
        }
        if role == "H2" and not {"AI", "H1"}.issubset(prior_roles):
            raise ValueError(
                f"H2 correspondence lacks AI/H1: {feature_id}"
            )
        if role in {"AI", "H1"} and "H2" in prior_roles:
            raise ValueError(
                f"{role} correspondence is frozen after H2: {feature_id}"
            )
        payload = {
            "match_decision": decision,
            "local_source_ids_json": "",
            "local_columns_json": "",
            "derivation_description": str(
                row.get("derivation_description") or ""
            ).strip(),
            "construct_equivalence_notes": str(
                row.get("construct_equivalence_notes") or ""
            ).strip(),
            "reason": reason,
            "inventory_sha256": sha256_file(inventory_path),
        }
        if decision in {
            "exact_materialized",
            "exact_derivable",
            "candidate_formula_completion",
        }:
            source_ids = _json_list(
                row.get("local_source_ids_json"),
                "local_source_ids_json",
                feature_id,
            )
            unknown = sorted(set(source_ids) - sources)
            if unknown:
                raise ValueError(
                    f"Unknown local sources for {feature_id}: {unknown}"
                )
            columns = _json_list(
                row.get("local_columns_json"),
                "local_columns_json",
                feature_id,
            )
            valid_columns = {
                f"{source_id}:{field['name']}"
                for source_id, source in inventory["sources"].items()
                for field in source["columns"]
            }
            unknown_columns = sorted(set(columns) - valid_columns)
            if unknown_columns:
                raise ValueError(
                    f"Unknown local columns for {feature_id}: "
                    f"{unknown_columns}"
                )
            if not payload["derivation_description"]:
                raise ValueError(
                    f"Positive match lacks derivation: {feature_id}"
                )
            if not payload["construct_equivalence_notes"]:
                raise ValueError(
                    f"Positive match lacks equivalence notes: {feature_id}"
                )
            payload["local_source_ids_json"] = json.dumps(source_ids)
            payload["local_columns_json"] = json.dumps(columns)
            positive += 1
        connection.execute(
            """
            INSERT INTO feature_data_correspondence_reviews(
                feature_id, reviewer_role, decision, payload_json,
                reason, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(feature_id, reviewer_role) DO UPDATE SET
                decision = excluded.decision,
                payload_json = excluded.payload_json,
                reason = excluded.reason,
                reviewed_at = excluded.reviewed_at
            """,
            (
                feature_id,
                role,
                decision,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                reason,
                utc_now(),
            ),
        )
        imported += 1
    if role == "H2":
        set_stage(
            connection,
            "data_correspondence_reviewed",
            "complete",
            {
                "families_reviewed": imported,
                "h2_possible_correspondence": positive,
                "inventory_path": str(inventory_path),
                "inventory_sha256": sha256_file(inventory_path),
            },
        )
    invalidate_stages(
        connection,
        (
            "operationalizations_reviewed",
            "dimensions_derived",
            "features_selected",
            "audit_complete",
        ),
        "local data correspondence review changed",
    )
    connection.commit()
    return {
        "reviewer_role": role,
        "imported": imported,
        "positive": positive,
    }
