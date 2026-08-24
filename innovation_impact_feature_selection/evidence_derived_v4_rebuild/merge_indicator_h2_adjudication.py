"""Merge independently reviewed H2 indicator sub-batches deterministically."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common import sha256_file, utc_now


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DEFAULT_FORMULA = OUTPUTS / "indicator_adjudication_H2_formula_sources_completed_v4.csv"
DEFAULT_REMAINING = OUTPUTS / "indicator_adjudication_H2_remaining_sources_completed_v4.csv"
DEFAULT_OUTPUT = OUTPUTS / "indicator_adjudication_H2_completed_v4.csv"
DEFAULT_MANIFEST = OUTPUTS / "indicator_adjudication_H2_completed_v4.manifest.json"


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def merge(
    formula_path: Path,
    remaining_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Join disjoint H2-review rows without changing reviewer decisions."""
    formula_fields, formula_rows = _read(formula_path)
    remaining_fields, remaining_rows = _read(remaining_path)
    if formula_fields != remaining_fields:
        raise ValueError("H2 sub-batch CSV headers differ")
    rows = [*formula_rows, *remaining_rows]
    identities = [
        "|".join(
            (
                row.get("record_key", ""),
                row.get("raw_name_en", ""),
                row.get("formula_location", ""),
            )
        )
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("H2 sub-batches contain duplicate indicator rows")
    source_keys = {row.get("record_key", "") for row in rows}
    if len(rows) != 99 or len(source_keys) != 77:
        raise ValueError(
            "Expected 99 rows covering 77 source records after H2 merge"
        )
    if any(row.get("disposition_decided_by") != "H1|H2" for row in rows):
        raise ValueError("Every merged row must carry H1|H2 disposition")
    rows.sort(
        key=lambda row: (
            row.get("record_key", ""),
            row.get("raw_name_en", ""),
            row.get("formula_location", ""),
        )
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=formula_fields)
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema_version": "indicator_h2_subbatch_merge_v4",
        "formula_subbatch": str(formula_path.resolve()),
        "formula_subbatch_sha256": sha256_file(formula_path),
        "remaining_subbatch": str(remaining_path.resolve()),
        "remaining_subbatch_sha256": sha256_file(remaining_path),
        "output": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "rows": len(rows),
        "sources": len(source_keys),
        "completed_at": utc_now(),
    }
    manifest_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    """Run deterministic H2 indicator-sub-batch merging."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--formula", type=Path, default=DEFAULT_FORMULA)
    parser.add_argument("--remaining", type=Path, default=DEFAULT_REMAINING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(
        json.dumps(
            merge(
                args.formula.resolve(),
                args.remaining.resolve(),
                args.output.resolve(),
                args.manifest.resolve(),
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
