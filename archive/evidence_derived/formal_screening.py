#!/usr/bin/env python3
"""Validate dual screening, prepare adjudication, and import final dispositions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
except ImportError:
    from core import (  # type: ignore[no-redef]
        EvidenceProtocol,
        ProtocolError,
        canonical_json,
        file_hash,
        stable_id,
        utc_now,
    )

csv.field_size_limit(sys.maxsize)

EXCLUSION_CODES = {
    "E_LANGUAGE_NON_ENGLISH",
    "E_NOT_PAPER_LEVEL",
    "E_NOT_INNOVATION_OR_T0_IMPACT",
    "E_FUTURE_OUTCOME_ONLY",
    "E_NOT_METRIC_PREDICTOR_VALIDATION",
    "E_DUPLICATE",
    "E_INSUFFICIENT_METADATA",
    "E_WRONG_DOCUMENT_TYPE",
}
CODE_PRECEDENCE = [
    "E_LANGUAGE_NON_ENGLISH",
    "E_WRONG_DOCUMENT_TYPE",
    "E_DUPLICATE",
    "E_INSUFFICIENT_METADATA",
    "E_NOT_PAPER_LEVEL",
    "E_FUTURE_OUTCOME_ONLY",
    "E_NOT_INNOVATION_OR_T0_IMPACT",
    "E_NOT_METRIC_PREDICTOR_VALIDATION",
]


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate(rows: list[dict[str, str]], expected: set[str], role: str) -> None:
    ids = [row["work_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ProtocolError(f"{role} screening does not exactly cover formal pool")
    for row in rows:
        if row.get("reviewer_role") != role:
            raise ProtocolError(f"Reviewer-role mismatch for {row['work_id']}")
        if row.get("decision") not in {"include", "exclude", "uncertain"}:
            raise ProtocolError(f"Invalid screening decision for {row['work_id']}")
        code = row.get("exclusion_code", "")
        if row["decision"] == "exclude" and code not in EXCLUSION_CODES:
            raise ProtocolError(f"Invalid exclusion code for {row['work_id']}: {code}")
        if row["decision"] != "exclude" and code:
            raise ProtocolError(f"Non-exclusion has exclusion code: {row['work_id']}")
        if not all(
            row.get(field, "").strip()
            for field in ("language_evidence", "eligibility_evidence", "reason")
        ):
            raise ProtocolError(f"Incomplete screening evidence for {row['work_id']}")


def _decision_tuple(
    primary: dict[str, str], independent: dict[str, str]
) -> tuple[str, str, str] | None:
    if primary["decision"] == independent["decision"] == "include":
        return "include", "", "DUAL_AGREEMENT_INCLUDE"
    if primary["decision"] == independent["decision"] == "exclude":
        codes = {primary["exclusion_code"], independent["exclusion_code"]}
        code = next(item for item in CODE_PRECEDENCE if item in codes)
        return "exclude", code, "DUAL_AGREEMENT_EXCLUDE_DETERMINISTIC_CODE"
    return None


def prepare(
    database: Path, primary_path: Path, independent_path: Path, output_dir: Path
) -> dict[str, Any]:
    primary_rows = _rows(primary_path)
    independent_rows = _rows(independent_path)
    with EvidenceProtocol(database, output_dir) as engine:
        engine.initialize()
        expected = {
            row[0]
            for row in engine.connection.execute(
                "SELECT work_id FROM formal_pool_records"
            )
        }
        _validate(primary_rows, expected, "Primary AI")
        _validate(independent_rows, expected, "Independent Reviewer AI")
        engine.connection.execute("DELETE FROM screening_decisions")
        by_role = {
            "Primary AI": (primary_rows, primary_path),
            "Independent Reviewer AI": (independent_rows, independent_path),
        }
        for role, (rows, path) in by_role.items():
            artifact_hash = file_hash(path)
            for row in rows:
                engine.connection.execute(
                    "INSERT INTO screening_decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        row["work_id"],
                        role,
                        row["decision"],
                        row.get("exclusion_code", ""),
                        row["language_evidence"],
                        row["eligibility_evidence"],
                        row.get("role", ""),
                        row.get("t0_judgment", ""),
                        row.get("run_id") or stable_id("RUN", role, artifact_hash),
                        row.get("input_hash") or "",
                        row.get("output_hash") or artifact_hash,
                        row.get("model_label") or "Codex",
                        row["reason"],
                    ),
                )
            session_id = stable_id("RS", "formal-screen", role, artifact_hash)
            engine.connection.execute(
                "INSERT OR REPLACE INTO review_sessions VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    session_id,
                    stable_id("RUN", session_id),
                    role,
                    file_hash(output_dir / "formal_screening_blind.csv"),
                    artifact_hash,
                    "Codex separate review session",
                    str(path.resolve()),
                    "Blind formal literature screening",
                    utc_now(),
                ),
            )
        primary_by_id = {row["work_id"]: row for row in primary_rows}
        independent_by_id = {row["work_id"]: row for row in independent_rows}
        auto: dict[str, tuple[str, str, str]] = {}
        contested: list[dict[str, str]] = []
        metadata = {
            row["work_id"]: dict(row)
            for row in engine.connection.execute(
                "SELECT p.work_id,w.doi,w.openalex_id,w.title,w.abstract,w.publication_year,"
                "w.language,w.work_type,p.routes_json,p.query_ids_json FROM formal_pool_records p "
                "JOIN works w USING(work_id)"
            )
        }
        for work_id in sorted(expected):
            primary = primary_by_id[work_id]
            independent = independent_by_id[work_id]
            decision = _decision_tuple(primary, independent)
            if decision:
                auto[work_id] = decision
                continue
            contested.append(
                {
                    **metadata[work_id],
                    "primary_decision": primary["decision"],
                    "primary_exclusion_code": primary.get("exclusion_code", ""),
                    "primary_reason": primary["reason"],
                    "independent_decision": independent["decision"],
                    "independent_exclusion_code": independent.get("exclusion_code", ""),
                    "independent_reason": independent["reason"],
                }
            )
        adjudication_path = output_dir / "formal_screening_adjudication_input.csv"
        fields = list(contested[0]) if contested else ["work_id"]
        with adjudication_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(contested)
        auto_path = output_dir / "formal_screening_auto_agreements.json"
        auto_path.write_text(canonical_json(auto) + "\n", encoding="utf-8")
        engine.set_metadata(
            "formal_screening_prepared",
            {
                "pool_count": len(expected),
                "auto_count": len(auto),
                "contested_count": len(contested),
                "primary_sha256": file_hash(primary_path),
                "independent_sha256": file_hash(independent_path),
                "adjudication_input_sha256": file_hash(adjudication_path),
            },
        )
        engine.set_metadata("stage", "formal-screening-adjudication")
        engine.connection.commit()
        return {
            "pool_count": len(expected),
            "auto_count": len(auto),
            "contested_count": len(contested),
            "adjudication_input_sha256": file_hash(adjudication_path),
        }


def finalize(
    database: Path, adjudication_path: Path, output_dir: Path
) -> dict[str, int]:
    adjudication_rows = _rows(adjudication_path)
    with EvidenceProtocol(database, output_dir) as engine:
        engine.initialize()
        prepared = engine.get_metadata("formal_screening_prepared", {})
        if not prepared:
            raise ProtocolError("Dual screening must be prepared before adjudication")
        auto = json.loads(
            (output_dir / "formal_screening_auto_agreements.json").read_text(
                encoding="utf-8"
            )
        )
        contested_ids = (
            {
                row[0]
                for row in engine.connection.execute(
                    "SELECT work_id FROM formal_pool_records WHERE work_id NOT IN ("
                    + ",".join("?" for _ in auto)
                    + ")",
                    tuple(auto),
                )
            }
            if auto
            else {
                row[0]
                for row in engine.connection.execute(
                    "SELECT work_id FROM formal_pool_records"
                )
            }
        )
        if {row["work_id"] for row in adjudication_rows} != contested_ids:
            raise ProtocolError(
                "Adjudication output does not exactly cover contested records"
            )
        final = {work_id: tuple(values) for work_id, values in auto.items()}
        for row in adjudication_rows:
            decision = row.get("final_decision", "")
            code = row.get("final_exclusion_code", "")
            if decision not in {"include", "exclude"}:
                raise ProtocolError(f"Invalid adjudication: {row['work_id']}")
            if decision == "exclude" and code not in EXCLUSION_CODES:
                raise ProtocolError(f"Invalid adjudication code: {row['work_id']}")
            if decision == "include" and code:
                raise ProtocolError(
                    f"Included adjudication has exclusion code: {row['work_id']}"
                )
            if not row.get("adjudication_reason", "").strip():
                raise ProtocolError(f"Adjudication lacks reason: {row['work_id']}")
            final[row["work_id"]] = (
                decision,
                code,
                row["adjudication_reason"],
            )
        if len(final) != int(prepared["pool_count"]):
            raise ProtocolError("Final screening closure failed")
        engine.connection.execute("DELETE FROM screening_final")
        run_id = stable_id("RUN", "formal-adjudication", file_hash(adjudication_path))
        for work_id, values in sorted(final.items()):
            engine.connection.execute(
                "INSERT INTO screening_final VALUES(?,?,?,?,?)",
                (work_id, values[0], values[1], values[2], run_id),
            )
        session_id = stable_id("RS", run_id)
        engine.connection.execute(
            "INSERT OR REPLACE INTO review_sessions VALUES(?,?,?,?,?,?,?,?,?)",
            (
                session_id,
                run_id,
                "Adjudicator AI",
                str(prepared["adjudication_input_sha256"]),
                file_hash(adjudication_path),
                "Codex adjudication session",
                str(adjudication_path.resolve()),
                "Resolve all dual-screening disagreements and uncertainties",
                utc_now(),
            ),
        )
        engine.connection.commit()
        engine.finalize_screening()
        engine.set_metadata("stage", "screen")
        engine.connection.commit()
        counts = {
            row[0]: row[1]
            for row in engine.connection.execute(
                "SELECT decision,COUNT(*) FROM screening_final GROUP BY decision"
            )
        }
        return {"total": len(final), **counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--primary", type=Path, required=True)
    prepare_parser.add_argument("--independent", type=Path, required=True)
    final_parser = commands.add_parser("finalize")
    final_parser.add_argument("--adjudication", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.database, args.primary, args.independent, args.output_dir)
    else:
        result = finalize(args.database, args.adjudication, args.output_dir)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
