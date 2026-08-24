#!/usr/bin/env python3
"""Validate and import one dual-reviewed, adjudicated discovery round."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    from .core import (
        EvidenceProtocol,
        ProtocolError,
        canonical_json,
        file_hash,
        normalize_text,
        sha256_text,
        stable_id,
        utc_now,
    )
except ImportError:
    from core import (  # type: ignore[no-redef]
        EvidenceProtocol,
        ProtocolError,
        canonical_json,
        file_hash,
        normalize_text,
        sha256_text,
        stable_id,
        utc_now,
    )

ALLOWED_EXCLUSION_CODES = {
    "E_LANGUAGE_NON_ENGLISH",
    "E_NOT_PAPER_LEVEL",
    "E_NOT_INNOVATION_OR_T0_IMPACT",
    "E_FUTURE_OUTCOME_ONLY",
    "E_NOT_METRIC_PREDICTOR_VALIDATION",
    "E_DUPLICATE",
    "E_INSUFFICIENT_METADATA",
    "E_WRONG_DOCUMENT_TYPE",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ProtocolError(f"Missing manifest: {manifest_path}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_hash = value.get("output_sha256") or value.get("output", {}).get("sha256")
    if output_hash != file_hash(path):
        raise ProtocolError(f"Output hash mismatch: {path}")
    if int(value.get("row_count", -1)) < 1:
        raise ProtocolError(f"Invalid row count in manifest: {manifest_path}")
    value["output_sha256"] = output_hash
    if "input_sha256" not in value:
        inputs = {
            key: nested.get("sha256", "")
            for key, nested in value.items()
            if key.endswith("_input") and isinstance(nested, dict)
        }
        value["input_sha256"] = sha256_text(canonical_json(inputs))
    return value


def indexed(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        work_id = row.get("work_id", "").strip()
        if not work_id or work_id in result:
            raise ProtocolError(f"Missing or duplicate work_id: {work_id!r}")
        result[work_id] = row
    return result


def mentions(value: str, kind: str, fallback_evidence: str) -> list[dict[str, str]]:
    parsed = json.loads(value or "[]")
    if not isinstance(parsed, list):
        raise ProtocolError("Mention JSON must be a list")
    result: list[dict[str, str]] = []
    key = "term" if kind == "term" else "indicator"
    for item in parsed:
        if isinstance(item, str):
            label, evidence = item.strip(), fallback_evidence
        elif isinstance(item, dict):
            label = str(item.get(key) or item.get("label") or "").strip()
            evidence = str(item.get("evidence") or fallback_evidence).strip()
        else:
            raise ProtocolError("Mention entries must be strings or objects")
        if label and normalize_text(label):
            result.append({"label": label, "evidence": evidence})
    return result


def merged_mentions(
    final: dict[str, str], primary: dict[str, str], independent: dict[str, str], kind: str
) -> list[dict[str, str]]:
    column = f"{kind}_mentions_json"
    sources = [final[column]] if final.get(column, "").strip() else [
        primary.get(column, "[]"), independent.get(column, "[]")
    ]
    merged: dict[str, dict[str, str]] = {}
    for value in sources:
        for item in mentions(value, kind, final.get("adjudication_reason", "")):
            merged.setdefault(normalize_text(item["label"]), item)
    return list(merged.values())


def upsert_family(
    engine: EvidenceProtocol,
    table: str,
    prefix: str,
    round_no: int,
    work_id: str,
    item: dict[str, str],
) -> bool:
    normalized = normalize_text(item["label"])
    row = engine.connection.execute(
        f"SELECT * FROM {table} WHERE normalized_label=?", (normalized,)
    ).fetchone()
    if row is None:
        engine.connection.execute(
            f"INSERT INTO {table} VALUES(?,?,?,?,?,?,?,?)",
            (
                stable_id(prefix, normalized), item["label"], normalized,
                canonical_json([item["label"]]), canonical_json([work_id]),
                canonical_json([item["evidence"]]), round_no, 1,
            ),
        )
        return True
    aliases = sorted(set(json.loads(row["aliases_json"])) | {item["label"]})
    works = sorted(set(json.loads(row["source_work_ids_json"])) | {work_id})
    evidence = list(json.loads(row["evidence_json"]))
    if item["evidence"] and item["evidence"] not in evidence:
        evidence.append(item["evidence"])
    engine.connection.execute(
        f"UPDATE {table} SET aliases_json=?,source_work_ids_json=?,evidence_json=? "
        "WHERE normalized_label=?",
        (canonical_json(aliases), canonical_json(works), canonical_json(evidence), normalized),
    )
    return False


def validate_decision(row: dict[str, str], column: str) -> None:
    decision = row.get(column, "")
    if decision not in {"include", "exclude", "uncertain"}:
        raise ProtocolError(f"Invalid {column}: {decision!r}")
    if decision == "exclude" and not row.get("exclusion_code", ""):
        raise ProtocolError(f"Excluded row lacks code: {row.get('work_id')}")
    if row.get("exclusion_code", "") and row["exclusion_code"] not in ALLOWED_EXCLUSION_CODES:
        raise ProtocolError(
            f"Unregistered exclusion code for {row.get('work_id')}: "
            f"{row['exclusion_code']}"
        )
    if not row.get("evidence", "").strip() or not row.get("reason", "").strip():
        raise ProtocolError(f"Decision lacks evidence/reason: {row.get('work_id')}")


def import_round(
    database: Path, round_no: int, primary_path: Path,
    independent_path: Path, adjudicated_path: Path,
) -> dict[str, Any]:
    primary_manifest = read_manifest(primary_path)
    independent_manifest = read_manifest(independent_path)
    adjudicated_manifest = read_manifest(adjudicated_path)
    primary = indexed(read_rows(primary_path))
    independent = indexed(read_rows(independent_path))
    adjudicated = indexed(read_rows(adjudicated_path))
    if set(primary) != set(independent) or set(primary) != set(adjudicated):
        raise ProtocolError("Reviewer/adjudicator work-id sets differ")
    if any(int(row.get("round_no", 0)) != round_no for row in adjudicated.values()):
        raise ProtocolError("Adjudicated rows have the wrong round number")
    engine = EvidenceProtocol(database)
    engine.initialize()
    assigned = {
        row[0] for row in engine.connection.execute(
            "SELECT work_id FROM discovery_round_records WHERE round_no=?", (round_no,)
        )
    }
    if assigned != set(primary):
        raise ProtocolError("Reviewed work IDs do not equal the frozen round assignment")
    prior = engine.connection.execute(
        "SELECT COUNT(*) FROM saturation_rounds WHERE round_no=?", (round_no,)
    ).fetchone()[0]
    if prior:
        raise ProtocolError(f"Round {round_no} is already imported")
    new_terms = 0
    new_indicators = 0
    for work_id in sorted(primary):
        p_row, i_row, a_row = primary[work_id], independent[work_id], adjudicated[work_id]
        validate_decision(p_row, "primary_decision")
        validate_decision(i_row, "independent_decision")
        final_decision = a_row.get("final_decision", "")
        if final_decision not in {"include", "exclude", "uncertain"}:
            raise ProtocolError(f"Invalid final decision: {work_id}")
        if not a_row.get("adjudication_reason", "").strip():
            raise ProtocolError(f"Missing adjudication reason: {work_id}")
        for role, row, column, manifest in (
            ("Primary AI", p_row, "primary_decision", primary_manifest),
            ("Independent Reviewer AI", i_row, "independent_decision", independent_manifest),
        ):
            engine.connection.execute(
                "INSERT INTO discovery_decisions VALUES(?,?,?,?,?,?,?,?)",
                (round_no, work_id, role, row[column], row.get("exclusion_code", ""),
                 row["evidence"], row["reason"], manifest["run_id"]),
            )
        engine.connection.execute(
            "INSERT INTO discovery_final VALUES(?,?,?,?,?,?)",
            (round_no, work_id, final_decision,
             a_row.get("final_exclusion_code", a_row.get("exclusion_code", "")),
             a_row["adjudication_reason"], adjudicated_manifest["run_id"]),
        )
        if final_decision != "include":
            continue
        term_items = merged_mentions(a_row, p_row, i_row, "term")
        indicator_items = merged_mentions(a_row, p_row, i_row, "indicator")
        engine.connection.execute(
            "INSERT INTO discovery_extractions VALUES(?,?,?,?,?,?,?)",
            (round_no, work_id, canonical_json(term_items), canonical_json(indicator_items),
             canonical_json({"adjudication_reason": a_row["adjudication_reason"]}),
             "Adjudicator AI", adjudicated_manifest["run_id"]),
        )
        for item in term_items:
            new_terms += int(upsert_family(engine, "discovery_term_families", "DTF", round_no, work_id, item))
        for item in indicator_items:
            new_indicators += int(upsert_family(engine, "discovery_indicator_families", "DIF", round_no, work_id, item))
    for role, manifest, path in (
        ("Primary AI", primary_manifest, primary_path),
        ("Independent Reviewer AI", independent_manifest, independent_path),
        ("Adjudicator AI", adjudicated_manifest, adjudicated_path),
    ):
        engine.connection.execute(
            "INSERT INTO review_sessions VALUES(?,?,?,?,?,?,?,?,?)",
            (stable_id("RS", manifest["run_id"]), manifest["run_id"], role,
             manifest.get("input_sha256") or manifest.get("primary_input_sha256", ""),
             manifest["output_sha256"], manifest["model_label"],
             canonical_json({"stage": "saturation", "object_ids": sorted(primary)}),
             f"Validated round {round_no} artifact {path.name}", utc_now()),
        )
    evidence_hash = file_hash(adjudicated_path)
    stop_basis = engine.record_saturation_round(
        round_no, new_terms, new_indicators, True, evidence_hash
    )
    protocol_stop = stop_basis in {"strict_zero_zero", "maximum_round_15"}
    engine.set_metadata(
        "stage", "saturation_stopped" if protocol_stop else "saturation"
    )
    engine.connection.commit()
    counts = {
        "round_no": round_no, "rows": len(primary),
        "final_includes": sum(r.get("final_decision") == "include" for r in adjudicated.values()),
        "new_term_families": new_terms, "new_indicator_families": new_indicators,
        "stop": protocol_stop,
        "stop_basis": stop_basis,
    }
    engine.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--adjudicated", type=Path, required=True)
    args = parser.parse_args()
    result = import_round(
        args.database, args.round, args.primary, args.independent, args.adjudicated
    )
    print(canonical_json(result))


if __name__ == "__main__":
    main()
