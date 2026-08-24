#!/usr/bin/env python3
"""Adjudicate two locked discovery reviews with explicit disagreement overrides."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from .core import canonical_json, file_hash, normalize_text, sha256_text
except ImportError:
    from core import (  # type: ignore[no-redef]
        canonical_json,
        file_hash,
        normalize_text,
        sha256_text,
    )


CODE_PRIORITY = {
    "E_DUPLICATE": 80,
    "E_LANGUAGE_NON_ENGLISH": 70,
    "E_WRONG_DOCUMENT_TYPE": 60,
    "E_NOT_PAPER_LEVEL": 50,
    "E_FUTURE_OUTCOME_ONLY": 40,
    "E_NOT_INNOVATION_OR_T0_IMPACT": 30,
    "E_NOT_METRIC_PREDICTOR_VALIDATION": 20,
    "E_INSUFFICIENT_METADATA": 10,
    "": 0,
}
ALLOWED_EXCLUSION_CODES = set(CODE_PRIORITY) - {""}


def load(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return fields, {row["work_id"]: row for row in rows}


def items(value: str, kind: str, source: str, fallback: str) -> list[dict[str, Any]]:
    key = "term" if kind == "term" else "indicator"
    result: list[dict[str, Any]] = []
    for item in json.loads(value or "[]"):
        if isinstance(item, str):
            label, evidence = item, fallback
        else:
            label = str(item.get(key) or item.get("label") or "")
            evidence = str(item.get("evidence") or fallback)
        if normalize_text(label):
            result.append({key: label, "evidence": evidence, "sources": [source]})
    return result


def merge_items(
    primary: dict[str, str], independent: dict[str, str], kind: str
) -> str:
    key = "term" if kind == "term" else "indicator"
    column = f"{kind}_mentions_json"
    merged: dict[str, dict[str, Any]] = {}
    for row, role in ((primary, "Primary AI"), (independent, "Independent Reviewer AI")):
        for item in items(row[column], kind, role, row["evidence"]):
            normalized = normalize_text(item[key])
            if normalized not in merged:
                merged[normalized] = item
            else:
                current = merged[normalized]
                current["sources"] = sorted(set(current["sources"] + item["sources"]))
                if item["evidence"] not in current["evidence"]:
                    current["evidence"] += " || " + item["evidence"]
    return canonical_json(list(merged.values()))


def adjudicate(
    primary_path: Path, independent_path: Path, overrides_path: Path,
    output: Path, round_no: int,
) -> dict[str, Any]:
    primary_fields, primary = load(primary_path)
    _, independent = load(independent_path)
    overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    if set(primary) != set(independent):
        raise RuntimeError("Review work-id sets differ")
    disagreements = {
        key for key in primary
        if primary[key]["primary_decision"] != independent[key]["independent_decision"]
    }
    uncertain_agreements = {
        key for key in primary
        if primary[key]["primary_decision"] == independent[key]["independent_decision"] == "uncertain"
    }
    if not disagreements.issubset(overrides) or not uncertain_agreements.issubset(overrides):
        raise RuntimeError("Every disagreement and uncertain agreement needs an override")
    invalid_codes = {
        row["exclusion_code"]
        for rows in (primary, independent)
        for row in rows.values()
        if row["exclusion_code"] and row["exclusion_code"] not in ALLOWED_EXCLUSION_CODES
    }
    if invalid_codes:
        raise RuntimeError(f"Unregistered exclusion codes: {sorted(invalid_codes)}")
    metadata_fields = primary_fields[: primary_fields.index("primary_decision")]
    final_fields = list(metadata_fields)
    final_fields += [
        "primary_decision", "primary_exclusion_code", "primary_evidence", "primary_reason",
        "independent_decision", "independent_exclusion_code", "independent_evidence",
        "independent_reason", "final_decision", "final_exclusion_code",
        "adjudication_reason", "evidence", "term_mentions_json", "indicator_mentions_json",
    ]
    final_rows: list[dict[str, str]] = []
    for work_id, p_row in primary.items():
        i_row = independent[work_id]
        p_decision, i_decision = p_row["primary_decision"], i_row["independent_decision"]
        if work_id in overrides:
            decision, code, reason = overrides[work_id]
            reason = f"Manual adjudication: {reason}"
        else:
            decision = p_decision
            codes = [p_row["exclusion_code"], i_row["exclusion_code"]]
            code = max(codes, key=lambda value: CODE_PRIORITY[value]) if decision == "exclude" else ""
            if codes[0] != codes[1]:
                reason = (
                    "Manual code adjudication after concordant exclusion: selected the more "
                    "specific governing exclusion code under the preregistered hierarchy."
                )
            else:
                reason = "Deterministic carry-forward of concordant blind-review decision and code."
        base = {field: p_row.get(field, "") for field in metadata_fields}
        base.update(
            {
                "primary_decision": p_decision,
                "primary_exclusion_code": p_row["exclusion_code"],
                "primary_evidence": p_row["evidence"],
                "primary_reason": p_row["reason"],
                "independent_decision": i_decision,
                "independent_exclusion_code": i_row["exclusion_code"],
                "independent_evidence": i_row["evidence"],
                "independent_reason": i_row["reason"],
                "final_decision": decision,
                "final_exclusion_code": code,
                "adjudication_reason": reason,
                "evidence": f"Primary AI: {p_row['evidence']} || Independent Reviewer AI: {i_row['evidence']}",
                "term_mentions_json": merge_items(p_row, i_row, "term"),
                "indicator_mentions_json": merge_items(p_row, i_row, "indicator"),
            }
        )
        final_rows.append(base)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fields)
        writer.writeheader()
        writer.writerows(final_rows)
    counts = {decision: sum(row["final_decision"] == decision for row in final_rows) for decision in ("include", "exclude", "uncertain")}
    manifest = {
        "schema_version": "discovery_adjudication_manifest_1",
        "round_no": round_no,
        "row_count": len(final_rows),
        "decision_counts": counts,
        "decision_disagreements_reviewed": len(disagreements),
        "code_disagreements_reviewed": sum(
            primary[key]["exclusion_code"] != independent[key]["exclusion_code"]
            for key in primary if primary[key]["primary_decision"] == independent[key]["independent_decision"]
        ),
        "input_sha256": sha256_text(canonical_json({"primary": file_hash(primary_path), "independent": file_hash(independent_path)})),
        "primary_input": {"path": str(primary_path), "sha256": file_hash(primary_path)},
        "independent_input": {"path": str(independent_path), "sha256": file_hash(independent_path)},
        "overrides_sha256": file_hash(overrides_path),
        "output_sha256": file_hash(output),
        "run_id": f"adjudication-round-{round_no:02d}-{file_hash(output)[:16]}",
        "model_label": "Codex Adjudicator AI (GPT-5)",
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()
    print(canonical_json(adjudicate(args.primary, args.independent, args.overrides, args.output, args.round)))


if __name__ == "__main__":
    main()
