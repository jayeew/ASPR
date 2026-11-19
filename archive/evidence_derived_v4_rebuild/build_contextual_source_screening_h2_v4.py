"""Create protected H2 source-screening sheets from two blind reviews."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common import sha256_bytes, sha256_file, write_csv, write_json

DECISIONS = {"include_definition_or_review", "exclude_not_relevant", "uncertain"}


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read one UTF-8 CSV with a header."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), list(reader)


def _index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Index a frozen batch by its stable record identifier."""
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        key = str(row.get("record_key", "")).strip()
        if not key or key in indexed:
            raise ValueError(f"Blank/duplicate record key: {key!r}")
        indexed[key] = row
    return indexed


def _review_columns(role: str) -> tuple[str, str, str]:
    """Return the explicit blind-review fields for one reviewer role."""
    return f"{role}_decision", f"{role}_rationale", f"{role}_evidence_span"


def _normalized_decision(value: str) -> str:
    """Map the compact blind-sheet labels to the frozen database vocabulary."""
    aliases = {
        "include": "include_definition_or_review",
        "exclude": "exclude_not_relevant",
    }
    return aliases.get(value.strip(), value.strip())


def _validate_blind(
    frozen: dict[str, dict[str, str]], review: dict[str, dict[str, str]], role: str
) -> None:
    """Require reviewers to preserve every non-review field."""
    if set(frozen) != set(review):
        raise ValueError(f"{role} record set differs from frozen input")
    decision_column, rationale_column, evidence_column = _review_columns(role)
    editable = {decision_column, rationale_column, evidence_column}
    for key, base in frozen.items():
        changed = {
            field
            for field in set(base) | set(review[key])
            if str(base.get(field, "")) != str(review[key].get(field, ""))
        }
        if not changed <= editable:
            raise ValueError(f"{role} changed protected fields: {key}")
        if (
            _normalized_decision(str(review[key].get(decision_column, "")))
            not in DECISIONS
        ):
            raise ValueError(f"{role} invalid decision: {key}")


def _h2_required(key: str, ai_decision: str, h1_decision: str) -> bool:
    """Adjudicate non-exclusions, disagreement, plus a deterministic audit sample."""
    return (
        ai_decision != h1_decision
        or ai_decision != "exclude_not_relevant"
        or int(sha256_bytes(key.encode("utf-8"))[:8], 16) % 10 == 0
    )


def build(
    input_path: Path, ai_path: Path, h1_path: Path, output: Path, manifest: Path
) -> dict[str, Any]:
    """Join two blind coding sheets, retaining all frozen and blind fields."""
    fields, frozen_rows = _read(input_path)
    _, ai_rows = _read(ai_path)
    _, h1_rows = _read(h1_path)
    frozen, ai, h1 = _index(frozen_rows), _index(ai_rows), _index(h1_rows)
    _validate_blind(frozen, ai, "AI")
    _validate_blind(frozen, h1, "H1")
    rows: list[dict[str, str]] = []
    for base in frozen_rows:
        key = str(base["record_key"])
        row = dict(base)
        for prefix, review in (("ai", ai[key]), ("h1", h1[key])):
            decision_column, rationale_column, evidence_column = _review_columns(
                prefix.upper()
            )
            row[f"{prefix}_screen_decision"] = _normalized_decision(
                str(review[decision_column])
            )
            row[f"{prefix}_evidence_span"] = str(review[evidence_column])
            row[f"{prefix}_rationale"] = str(review[rationale_column])
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
    write_csv(output, rows, output_fields)
    result = {
        "schema_version": "contextual_source_screening_h2_input_v4",
        "record_count": len(rows),
        "h2_review_required_count": sum(
            row["h2_review_required"] == "1" for row in rows
        ),
        "inputs": {
            str(path.resolve()): sha256_file(path)
            for path in (input_path, ai_path, h1_path)
        },
        "output": {str(output.resolve()): sha256_file(output)},
        "h2_editable_fields": [
            "h2_final_screen_decision",
            "h2_final_evidence_span",
            "h2_final_rationale",
        ],
    }
    write_json(manifest, result)
    return result


def main() -> None:
    """Parse paths and create an H2 input sheet."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ai", type=Path, required=True)
    parser.add_argument("--h1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                *(
                    path.resolve()
                    for path in (
                        args.input,
                        args.ai,
                        args.h1,
                        args.output,
                        args.manifest,
                    )
                )
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
