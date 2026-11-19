"""Join blind pre-promotion candidate-family reviews for H2 adjudication."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common import sha256_file, write_csv, write_json

PROPOSAL_FIELDS = (
    "family_name_en",
    "merge_or_split_reason",
    "formula_reproducible",
    "t0_computable",
    "scope_role",
    "missing_rule_status",
    "promotion_decision",
    "rationale",
)


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a nonempty CSV sheet."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing header: {path}")
        return list(reader.fieldnames), list(reader)


def _index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Index an input sheet by the immutable candidate key."""
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        key = str(row.get("candidate_id", "")).strip()
        if not key or key in result:
            raise ValueError(f"Blank/duplicate candidate ID: {key!r}")
        result[key] = row
    return result


def _validate(
    frozen: dict[str, dict[str, str]], review: dict[str, dict[str, str]], role: str
) -> None:
    """Ensure a blind reviewer changed only its assigned proposal fields."""
    if set(frozen) != set(review):
        raise ValueError(f"{role} candidate set differs from frozen input")
    allowed = {f"{role}_{field}" for field in PROPOSAL_FIELDS}
    for key, base in frozen.items():
        changed = {
            field
            for field in set(base) | set(review[key])
            if str(base.get(field, "")) != str(review[key].get(field, ""))
        }
        if not changed <= allowed:
            raise ValueError(f"{role} changed protected candidate fields: {key}")
        if not str(review[key].get(f"{role}_promotion_decision", "")).strip():
            raise ValueError(f"{role} has blank promotion decision: {key}")


def build(
    input_path: Path, ai_path: Path, h1_path: Path, output: Path, manifest: Path
) -> dict[str, Any]:
    """Create a protected H2 sheet while preserving both blind proposals."""
    fields, base_rows = _read(input_path)
    _, ai_rows = _read(ai_path)
    _, h1_rows = _read(h1_path)
    base, ai, h1 = _index(base_rows), _index(ai_rows), _index(h1_rows)
    _validate(base, ai, "AI")
    _validate(base, h1, "H1")
    rows: list[dict[str, str]] = []
    for frozen in base_rows:
        key = str(frozen["candidate_id"])
        row = dict(frozen)
        for role, review in (("AI", ai[key]), ("H1", h1[key])):
            for field in PROPOSAL_FIELDS:
                row[f"{role.lower()}_{field}"] = str(review[f"{role}_{field}"])
        for field in PROPOSAL_FIELDS:
            row[f"h2_{field}"] = ""
        rows.append(row)
    output_fields = (
        fields
        + [f"{role}_{field}" for role in ("ai", "h1") for field in PROPOSAL_FIELDS]
        + [f"h2_{field}" for field in PROPOSAL_FIELDS]
    )
    write_csv(output, rows, output_fields)
    result = {
        "schema_version": "contextual_candidate_canonicalization_h2_input_v4",
        "candidate_count": len(rows),
        "inputs": {
            str(path.resolve()): sha256_file(path)
            for path in (input_path, ai_path, h1_path)
        },
        "output": {str(output.resolve()): sha256_file(output)},
        "h2_editable_fields": [f"h2_{field}" for field in PROPOSAL_FIELDS],
    }
    write_json(manifest, result)
    return result


def main() -> None:
    """Parse paths and construct an H2 canonicalization workbook."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ai", type=Path, required=True)
    parser.add_argument("--h1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.input.resolve(),
        args.ai.resolve(),
        args.h1.resolve(),
        args.output.resolve(),
        args.manifest.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
