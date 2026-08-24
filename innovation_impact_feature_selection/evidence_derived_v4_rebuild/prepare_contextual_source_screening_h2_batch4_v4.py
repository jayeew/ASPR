"""Create a protected H2 adjudication sheet for contextual source-screening batch four."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from common import ROOT, sha256_bytes, sha256_file, write_csv, write_json

OUTPUT_DIR = ROOT / "outputs"
INPUT = OUTPUT_DIR / "contextual_source_screening_input_batch4_v4.csv"
AI = OUTPUT_DIR / "contextual_source_screening_AI_batch4_completed_v4.csv"
H1 = OUTPUT_DIR / "contextual_source_screening_H1_batch4_completed_v4.csv"
OUTPUT = OUTPUT_DIR / "contextual_source_screening_H2_batch4_v4.csv"
MANIFEST = OUTPUT_DIR / "contextual_source_screening_H2_batch4_input_manifest_v4.json"
DECISIONS = {"include_definition_or_review", "exclude_not_relevant", "uncertain"}


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a UTF-8 review CSV."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing header: {path}")
        return list(reader.fieldnames), list(reader)


def _index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Index unique source records."""
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = str(row.get("record_key", "")).strip()
        if not key or key in result:
            raise ValueError(f"Blank/duplicate record key: {key!r}")
        result[key] = row
    return result


def _check_review(
    frozen: dict[str, dict[str, str]], review: dict[str, dict[str, str]], role: str
) -> None:
    """Verify blind reviewers changed only their assigned decision fields."""
    if set(frozen) != set(review):
        raise ValueError(f"{role} record set differs from frozen input")
    editable = {"screen_decision", "evidence_span", "rationale"}
    for key, base in frozen.items():
        changed = {
            field
            for field in set(base) | set(review[key])
            if str(base.get(field, "")) != str(review[key].get(field, ""))
        }
        if not changed <= editable:
            raise ValueError(f"{role} changed protected fields: {key}")
        if str(review[key].get("screen_decision", "")) not in DECISIONS:
            raise ValueError(f"{role} invalid decision: {key}")


def _h2_required(key: str, ai: str, h1: str) -> bool:
    """Review disagreements, non-exclusions, and a deterministic exclusion audit."""
    if ai != h1 or ai != "exclude_not_relevant":
        return True
    return int(sha256_bytes(key.encode("utf-8"))[:8], 16) % 10 == 0


def main() -> None:
    """Join the independent sheets without deciding final H2 values."""
    fields, frozen_rows = _read(INPUT)
    _, ai_rows = _read(AI)
    _, h1_rows = _read(H1)
    frozen, ai, h1 = _index(frozen_rows), _index(ai_rows), _index(h1_rows)
    _check_review(frozen, ai, "AI")
    _check_review(frozen, h1, "H1")
    rows: list[dict[str, str]] = []
    for base in frozen_rows:
        key = str(base["record_key"])
        row = dict(base)
        for prefix, review in (("ai", ai[key]), ("h1", h1[key])):
            for field in ("screen_decision", "evidence_span", "rationale"):
                row[f"{prefix}_{field}"] = str(review[field])
        row["h2_review_required"] = str(
            int(_h2_required(key, row["ai_screen_decision"], row["h1_screen_decision"]))
        )
        row["h2_final_screen_decision"] = ""
        row["h2_final_evidence_span"] = ""
        row["h2_final_rationale"] = ""
        rows.append(row)
    output_fields = fields + [
        "ai_screen_decision",
        "ai_evidence_span",
        "ai_rationale",
        "h1_screen_decision",
        "h1_evidence_span",
        "h1_rationale",
        "h2_review_required",
        "h2_final_screen_decision",
        "h2_final_evidence_span",
        "h2_final_rationale",
    ]
    write_csv(OUTPUT, rows, output_fields)
    result: dict[str, Any] = {
        "schema_version": "contextual_source_screening_h2_batch4_input_v4",
        "record_count": len(rows),
        "h2_review_required_count": sum(
            row["h2_review_required"] == "1" for row in rows
        ),
        "inputs": {str(path.resolve()): sha256_file(path) for path in (INPUT, AI, H1)},
        "output": {str(OUTPUT.resolve()): sha256_file(OUTPUT)},
        "h2_editable_fields": [
            "h2_final_screen_decision",
            "h2_final_evidence_span",
            "h2_final_rationale",
        ],
    }
    write_json(MANIFEST, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
