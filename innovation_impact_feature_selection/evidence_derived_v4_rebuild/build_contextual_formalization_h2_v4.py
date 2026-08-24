"""Join two blind formula/data formalization reviews for H2 adjudication."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common import sha256_file, write_csv, write_json

FIELDS = (
    "canonical_name_en",
    "label_zh",
    "formula",
    "units",
    "parameters",
    "direction",
    "missing_rule",
    "required_data_json",
    "research_group",
    "research_group_evidence",
    "data_match_decision",
    "local_source_ids_json",
    "local_columns_json",
    "derivation_description",
    "formalization_decision",
    "rationale",
)


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a header-bearing UTF-8 CSV."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing header: {path}")
        headers = list(reader.fieldnames)
        rows = list(reader)
    # Early worklists used ``h2_rationale`` for the already-frozen
    # canonicalization rationale.  H2 needs its own formalization rationale,
    # so normalize that legacy source column before composing protected fields.
    if "h2_rationale" in headers:
        headers[headers.index("h2_rationale")] = "canonicalization_h2_rationale"
        for row in rows:
            row["canonicalization_h2_rationale"] = row.pop("h2_rationale", "")
    return headers, rows


def _index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Index a sheet by immutable candidate ID."""
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        key = str(row.get("candidate_id", "")).strip()
        if not key or key in output:
            raise ValueError(f"Blank/duplicate candidate ID: {key!r}")
        output[key] = row
    return output


def _validate(
    frozen: dict[str, dict[str, str]], review: dict[str, dict[str, str]], role: str
) -> None:
    """Prevent a reviewer from changing evidence or the other review's scope."""
    if set(frozen) != set(review):
        raise ValueError(f"{role} candidate set differs from frozen worklist")
    allowed = {f"{role}_{field}" for field in FIELDS}
    for key, base in frozen.items():
        changed = {
            field
            for field in set(base) | set(review[key])
            if str(base.get(field, "")) != str(review[key].get(field, ""))
        }
        if not changed <= allowed:
            raise ValueError(f"{role} changed protected fields for {key}")
        if not str(review[key].get(f"{role}_formalization_decision", "")).strip():
            raise ValueError(f"{role} has no formalization decision for {key}")


def build(
    input_path: Path, ai_path: Path, h1_path: Path, output: Path, manifest: Path
) -> dict[str, Any]:
    """Create H2-editable formalization rows with both blind payloads frozen."""
    headers, frozen_rows = _read(input_path)
    _, ai_rows = _read(ai_path)
    _, h1_rows = _read(h1_path)
    frozen, ai, h1 = _index(frozen_rows), _index(ai_rows), _index(h1_rows)
    _validate(frozen, ai, "AI")
    _validate(frozen, h1, "H1")
    rows: list[dict[str, str]] = []
    for base in frozen_rows:
        key = str(base["candidate_id"])
        row = dict(base)
        for role, review in (("AI", ai[key]), ("H1", h1[key])):
            for field in FIELDS:
                row[f"{role.lower()}_{field}"] = str(review[f"{role}_{field}"])
        for field in FIELDS:
            row[f"h2_{field}"] = ""
        rows.append(row)
    output_fields = (
        headers
        + [f"{role}_{field}" for role in ("ai", "h1") for field in FIELDS]
        + [f"h2_{field}" for field in FIELDS]
    )
    write_csv(output, rows, output_fields)
    result = {
        "schema_version": "contextual_formalization_h2_input_v4",
        "candidate_count": len(rows),
        "inputs": {
            str(p.resolve()): sha256_file(p) for p in (input_path, ai_path, h1_path)
        },
        "output": {str(output.resolve()): sha256_file(output)},
        "h2_editable_fields": [f"h2_{field}" for field in FIELDS],
    }
    write_json(manifest, result)
    return result


def main() -> None:
    """Build the H2 workbook from command-line paths."""
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
