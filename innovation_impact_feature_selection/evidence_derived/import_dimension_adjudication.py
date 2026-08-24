#!/usr/bin/env python3
"""Validate and atomically import the adjudicated dimension artifacts."""

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
        utc_now,
    )
except ImportError:  # Direct execution from this directory.
    from core import (  # type: ignore[no-redef]
        EvidenceProtocol,
        ProtocolError,
        canonical_json,
        file_hash,
        utc_now,
    )

ROOT = Path(__file__).resolve().parent
DEFAULT_REVIEWS = ROOT / "outputs" / "reviews"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json_list(row: dict[str, str], field: str) -> list[Any]:
    try:
        value = json.loads(row[field])
    except (KeyError, json.JSONDecodeError) as error:
        raise ProtocolError(f"Invalid {field} in adjudicated dimension artifact") from error
    if not isinstance(value, list):
        raise ProtocolError(f"{field} must be a JSON list")
    return value


def _validate(
    engine: EvidenceProtocol,
    dimensions: list[dict[str, str]],
    mentions: list[dict[str, str]],
) -> None:
    included = {
        row[0]
        for row in engine.connection.execute(
            "SELECT work_id FROM screening_final WHERE decision='include'"
        )
    }
    dimension_ids = {row["dimension_id"] for row in dimensions}
    mention_ids = {row["mention_id"] for row in mentions}
    if len(dimension_ids) != len(dimensions) or len(mention_ids) != len(mentions):
        raise ProtocolError("Duplicate dimension or mention identifiers")
    covered = {row["work_id"] for row in mentions}
    if covered != included:
        raise ProtocolError(
            f"Adjudicated mention coverage differs from included works: "
            f"missing={len(included-covered)}, extra={len(covered-included)}"
        )
    for row in dimensions:
        sources = set(_json_list(row, "source_work_ids_json"))
        teams = _json_list(row, "independent_teams_json")
        merge_log = json.loads(row["merge_split_log_json"])
        if not sources or not sources.issubset(included) or not teams:
            raise ProtocolError(f"Invalid sources/teams for {row['dimension_id']}")
        if not merge_log or not merge_log.get("indicator_examples"):
            raise ProtocolError(f"Missing adjudication/indicator evidence for {row['dimension_id']}")
        if any(row.get(field, "").strip() != "1" for field in (
            "primary_approved",
            "independent_approved",
            "independent_non_alias_confirmed",
        )):
            raise ProtocolError(f"Missing approvals for {row['dimension_id']}")
    for row in mentions:
        linked = set(_json_list(row, "dimension_ids_json"))
        indicators = _json_list(row, "indicator_mentions_json")
        if not linked or not linked.issubset(dimension_ids) or not indicators:
            raise ProtocolError(f"Invalid mention linkage for {row['mention_id']}")
        required = (
            "construct",
            "role",
            "information_source",
            "T0_boundary",
            "bias_risk",
            "discipline_scope",
            "independent_team",
            "evidence_quote",
        )
        if any(not row.get(field, "").strip() for field in required):
            raise ProtocolError(f"Incomplete mention {row['mention_id']}")


def _review_session(manifest_path: Path, role: str, output_hash: str) -> tuple[str, ...]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = file_hash(manifest_path)
    run_id = str(payload.get("run_id") or f"dimension-{role.casefold().replace(' ', '-')}")
    return (
        f"SESSION_{manifest_hash[:16]}",
        run_id,
        role,
        str(
            payload.get("input", {}).get("sha256")
            or payload.get("input_sha256")
            or manifest_hash
        ),
        output_hash,
        str(payload.get("model_label") or "codex-gpt-5"),
        canonical_json(
            {
                "stage": "derive-dimensions",
                "object_ids": ["construct_mentions", "candidate_dimensions"],
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": manifest_hash,
            }
        ),
        "Validated formal dimension coding/adjudication artifact",
        str(payload.get("generated_at") or utc_now()),
    )


def import_artifacts(database: Path, output_dir: Path, reviews: Path) -> dict[str, int]:
    dimension_path = reviews / "candidate_dimensions_adjudicated.csv"
    mention_path = reviews / "construct_mentions_adjudicated.csv"
    dimensions = _rows(dimension_path)
    mentions = _rows(mention_path)
    with EvidenceProtocol(database, output_dir) as engine:
        engine.initialize()
        _validate(engine, dimensions, mentions)
        with engine.connection:
            engine.connection.execute("DELETE FROM construct_mentions")
            engine.connection.execute("DELETE FROM candidate_dimensions")
            engine.connection.executemany(
                "INSERT INTO construct_mentions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        row["mention_id"],
                        row["work_id"],
                        row["construct"],
                        row["role"],
                        row["information_source"],
                        row["T0_boundary"],
                        row["bias_risk"],
                        row["discipline_scope"],
                        row["indicator_mentions_json"],
                        row["independent_team"],
                        row["evidence_quote"],
                    )
                    for row in mentions
                ],
            )
            engine.connection.executemany(
                "INSERT INTO candidate_dimensions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        row["dimension_id"],
                        row["label"],
                        row["definition"],
                        row["role"],
                        row["T0_boundary"],
                        row["source_work_ids_json"],
                        row["independent_teams_json"],
                        row["merge_split_log_json"],
                        int(row["primary_approved"]),
                        int(row["independent_approved"]),
                        int(row["independent_non_alias_confirmed"]),
                    )
                    for row in dimensions
                ],
            )
            manifests = (
                (reviews / "formal_dimension_coding_primary.manifest.json", "Primary AI"),
                (reviews / "construct_dimensions_independent.manifest.json", "Independent Reviewer AI"),
                (reviews / "dimension_adjudication.manifest.json", "Adjudicator AI"),
            )
            for manifest, role in manifests:
                session = _review_session(manifest, role, file_hash(dimension_path))
                engine.connection.execute(
                    "INSERT OR REPLACE INTO review_sessions VALUES(?,?,?,?,?,?,?,?,?)",
                    session,
                )
        count = engine.validate_dimensions()
    return {"M": count, "construct_mentions": len(mentions)}


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
