#!/usr/bin/env python3
"""Validate and atomically import adjudicated indicator census artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from .core import (
        EvidenceProtocol,
        ProtocolError,
        canonical_json,
        file_hash,
        stable_id,
        utc_now,
    )
except ImportError:  # Direct execution from this directory.
    from core import (  # type: ignore[no-redef]
        EvidenceProtocol,
        ProtocolError,
        canonical_json,
        file_hash,
        stable_id,
        utc_now,
    )

ROOT = Path(__file__).resolve().parent
DEFAULT_REVIEWS = ROOT / "outputs" / "reviews"

ARTIFACTS = {
    "indicator_families": "indicator_census_adjudicated.csv",
    "indicator_mentions": "indicator_mentions_adjudicated.csv",
    "indicator_evidence": "indicator_evidence_adjudicated.csv",
    "indicator_data_mapping": "indicator_field_mapping_adjudicated.csv",
    "hard_gate_decisions": "hard_gate_decisions_adjudicated.csv",
    "evidence_tiers": "evidence_tiers_adjudicated.csv",
}

DATABASE_COLUMNS = {
    "indicator_families": (
        "indicator_id", "canonical_name", "aliases_json", "dimension_ids_json",
        "mention_ids_json", "definition", "formula", "definition_source_ids_json",
        "independent_teams_json", "role", "maximum_information_time", "missing_rule",
        "zero_denominator_rule", "empty_set_rule", "coverage_rule", "fallback_rule",
        "status",
    ),
    "indicator_mentions": (
        "mention_id", "work_id", "dimension_id", "raw_name",
        "definition_evidence", "source_role",
    ),
    "indicator_evidence": (
        "evidence_id", "indicator_id", "work_id", "evidence_role", "quote",
        "locator", "source_hash", "peer_reviewed", "team_id",
    ),
    "indicator_data_mapping": (
        "indicator_id", "mapping_type", "fields_json", "derivation",
        "source_snapshot_hash", "coverage", "missing_rate", "unique_count",
        "near_constant", "audit_status",
    ),
    "hard_gate_decisions": (
        "indicator_id", "h1_scope", "h2_t0", "h3_reproducibility",
        "h4_computability", "h5_validity_ethics", "h6_data_integrity",
        "primary_reason", "independent_reason", "deterministic_evidence_json",
        "all_pass",
    ),
    "evidence_tiers": ("indicator_id", "tier", "reason", "independent_approved"),
}

INTEGER_COLUMNS = {
    "peer_reviewed", "near_constant", "unique_count", "h1_scope", "h2_t0",
    "h3_reproducibility", "h4_computability", "h5_validity_ethics",
    "h6_data_integrity", "all_pass", "independent_approved",
}
FLOAT_COLUMNS = {"coverage", "missing_rate"}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json_list(row: dict[str, str], field: str) -> list[Any]:
    try:
        value = json.loads(row[field])
    except (KeyError, json.JSONDecodeError) as error:
        raise ProtocolError(f"Invalid {field}") from error
    if not isinstance(value, list):
        raise ProtocolError(f"{field} must be a JSON list")
    return value


def _validate(
    engine: EvidenceProtocol,
    rows: dict[str, list[dict[str, str]]],
    reviews: Path,
) -> None:
    families = rows["indicator_families"]
    family_ids = {row["indicator_id"] for row in families}
    if len(family_ids) != len(families):
        raise ProtocolError("Duplicate adjudicated indicator IDs")
    if len({row["canonical_name"].casefold() for row in families}) != len(families):
        raise ProtocolError("Duplicate adjudicated canonical indicator names")
    inventory = _read(reviews / "current_indicator_source_inventory.csv")
    expected_raw = {row["family_id"] for row in inventory}
    observed_raw = [
        item for row in families for item in _json_list(row, "raw_family_ids_json")
    ]
    if len(observed_raw) != len(set(observed_raw)) or set(observed_raw) != expected_raw:
        raise ProtocolError("Raw indicator families are not covered exactly once")
    dimensions = {
        row[0] for row in engine.connection.execute("SELECT dimension_id FROM candidate_dimensions")
    }
    covered_dimensions = {
        item for row in families for item in _json_list(row, "dimension_ids_json")
    }
    if covered_dimensions != dimensions:
        raise ProtocolError("Final indicator families do not cover every candidate dimension")
    included = {
        row[0]
        for row in engine.connection.execute(
            "SELECT work_id FROM screening_final WHERE decision='include'"
        )
    }
    mentions = rows["indicator_mentions"]
    expected_mentions = {
        row[0] for row in engine.connection.execute("SELECT mention_id FROM construct_mentions")
    }
    observed_mentions = {row["mention_id"] for row in mentions}
    if observed_mentions != expected_mentions or len(observed_mentions) != len(mentions):
        raise ProtocolError("Adjudicated indicator mentions do not close over construct mentions")
    if any(
        row["work_id"] not in included or row["dimension_id"] not in dimensions
        for row in mentions
    ):
        raise ProtocolError("Indicator mention has an ineligible work or dimension")
    for row in families:
        if not set(_json_list(row, "dimension_ids_json")).issubset(dimensions):
            raise ProtocolError(f"Invalid dimension for {row['indicator_id']}")
        if not set(_json_list(row, "mention_ids_json")).issubset(observed_mentions):
            raise ProtocolError(f"Invalid mention for {row['indicator_id']}")
        if not _json_list(row, "definition_source_ids_json"):
            raise ProtocolError(f"Missing definition source for {row['indicator_id']}")
    for table in (
        "indicator_evidence", "indicator_data_mapping", "hard_gate_decisions",
        "evidence_tiers",
    ):
        ids = [row["indicator_id"] for row in rows[table]]
        if len(ids) != len(set(ids)) or set(ids) != family_ids:
            raise ProtocolError(f"{table} does not have exactly one row per family")
    field_inventory = {row["matrix_field"]: row for row in _read(
        reviews / "available_matrix_field_inventory.csv"
    )}
    for row in rows["indicator_data_mapping"]:
        fields = _json_list(row, "fields_json")
        if row["mapping_type"] == "direct":
            if len(fields) != 1 or fields[0] not in field_inventory:
                raise ProtocolError(f"Invalid direct mapping for {row['indicator_id']}")
            expected = field_inventory[fields[0]]
            if int(float(row["unique_count"])) != int(expected["unique_count"]):
                raise ProtocolError(f"Mapping QA mismatch for {row['indicator_id']}")
        elif row["mapping_type"] == "unavailable":
            if fields:
                raise ProtocolError(f"Unavailable mapping has fields for {row['indicator_id']}")
        else:
            raise ProtocolError(f"Unsupported final mapping type for {row['indicator_id']}")
    future_pass = sum(
        int(row["h2_t0"]) and row["indicator_id"] in {
            family["indicator_id"]
            for family in families
            if family["maximum_information_time"] != "T0"
        }
        for row in rows["hard_gate_decisions"]
    )
    if future_pass:
        raise ProtocolError("Future-only indicators passed H2")


def _value(field: str, raw: str) -> Any:
    if raw == "":
        return None
    if field in INTEGER_COLUMNS:
        return int(float(raw))
    if field in FLOAT_COLUMNS:
        return float(raw)
    return raw


def _session(
    role: str,
    manifest_path: Path,
    output_hash: str,
    object_ids: list[str],
) -> tuple[str, ...]:
    manifest_hash = file_hash(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = str(payload.get("run_id") or stable_id("RUN", role, manifest_hash))
    return (
        stable_id("RS", role, manifest_hash), run_id, role, manifest_hash, output_hash,
        str(payload.get("model_label") or f"codex-gpt-5-{role.casefold().replace(' ', '-') }"),
        canonical_json({
            "stage": "census-indicators", "object_ids": object_ids,
            "manifest": str(manifest_path.resolve()), "manifest_sha256": manifest_hash,
        }),
        "Validated indicator census, mapping, gate, and tier review artifact",
        str(payload.get("generated_at") or utc_now()),
    )


def import_artifacts(database: Path, output_dir: Path, reviews: Path) -> dict[str, int]:
    rows = {table: _read(reviews / name) for table, name in ARTIFACTS.items()}
    with EvidenceProtocol(database, output_dir) as engine:
        engine.initialize()
        _validate(engine, rows, reviews)
        with engine.connection:
            for table in (
                "final_features", "final_dimensions", "evidence_tiers",
                "hard_gate_decisions", "indicator_data_mapping", "indicator_evidence",
                "indicator_mentions", "indicator_families",
            ):
                engine.connection.execute(f"DELETE FROM {table}")
            for table, columns in DATABASE_COLUMNS.items():
                placeholders = ",".join("?" for _ in columns)
                engine.connection.executemany(
                    f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
                    [tuple(_value(field, row[field]) for field in columns) for row in rows[table]],
                )
            manifests = (
                ("Primary AI", reviews / "indicator_census_primary.manifest.json"),
                ("Independent Reviewer AI", reviews / "indicator_census_independent.manifest.json"),
                ("Adjudicator AI", reviews / "indicator_census_adjudicated.manifest.json"),
            )
            object_ids = list(ARTIFACTS)
            output_hash = file_hash(reviews / ARTIFACTS["indicator_families"])
            for role, manifest in manifests:
                engine.connection.execute(
                    "INSERT OR REPLACE INTO review_sessions VALUES(?,?,?,?,?,?,?,?,?)",
                    _session(role, manifest, output_hash, object_ids),
                )
        count = engine.validate_indicator_census()
    return {
        "F_all": count,
        "direct": sum(row["mapping_type"] == "direct" for row in rows["indicator_data_mapping"]),
        "unavailable": sum(row["mapping_type"] == "unavailable" for row in rows["indicator_data_mapping"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT / "outputs" / "evidence_derived.sqlite3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    args = parser.parse_args()
    print(json.dumps(import_artifacts(args.database, args.output_dir, args.reviews), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
