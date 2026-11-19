"""Validate frozen evidence-v3 definitions before matrix reconstruction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


EXPECTED_SET_COUNTS = {
    "strict_7": (7, 4),
    "fulltext_16": (16, 10),
    "source_154": (154, 48),
    "ultrarelaxed_221": (221, 55),
}
SAFE_T0_GATES = (
    "G01_IN_SCOPE_ROLE",
    "G02_ARTICLE_LEVEL",
    "G05_PUBLICATION_TIME",
    "G06_NO_FUTURE_INFORMATION",
    "G08_BIAS_GUARDRAIL",
    "G09_NO_FATAL_VALIDITY_CONCERN",
    "G10_OUTCOME_BLIND_SELECTION",
)
REQUIRED_FILES = (
    "complete_indicator_library_v3.csv",
    "feature_gate_decisions_v3.csv",
    "candidate_dimensions_v3.csv",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def read_rows(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    """Read a CSV and require its minimum schema."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required_columns - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path.name} missing columns: {', '.join(missing)}")
        return [dict(row) for row in reader]


def load_bundle(definition_dir: Path) -> Mapping[str, Any]:
    """Load and cross-check the three non-reconstructible frozen exports."""
    paths = {name: definition_dir / name for name in REQUIRED_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen definition files: " + ", ".join(missing))
    library_rows = read_rows(paths["complete_indicator_library_v3.csv"], {"feature_id"})
    gate_rows = read_rows(paths["feature_gate_decisions_v3.csv"], {"feature_id", "gate_checks_json"})
    dimension_rows = read_rows(paths["candidate_dimensions_v3.csv"], {"dimension_id", "feature_ids_json", "construct_role"})
    library = {row["feature_id"]: row for row in library_rows}
    gates = {row["feature_id"]: json.loads(row["gate_checks_json"]) for row in gate_rows}
    if len(library) != len(library_rows) or len(gates) != len(gate_rows):
        raise ValueError("Indicator library or gate decisions contain duplicate feature IDs")
    if set(library) != set(gates):
        raise ValueError("Indicator library and gate decisions do not cover the same features")
    feature_to_dimension: dict[str, str] = {}
    for row in dimension_rows:
        for feature_id in json.loads(row["feature_ids_json"]):
            if feature_id in feature_to_dimension:
                raise ValueError(f"Feature maps to multiple dimensions: {feature_id}")
            feature_to_dimension[feature_id] = row["dimension_id"]
    if set(feature_to_dimension) != set(library):
        raise ValueError("Dimension file does not map exactly the library feature IDs")

    def passes(feature_id: str, extras: tuple[str, ...]) -> bool:
        return all(bool(gates[feature_id][gate]) for gate in (*SAFE_T0_GATES, *extras))

    sets = {
        "strict_7": sorted(fid for fid, values in gates.items() if all(values.values())),
        "fulltext_16": sorted(fid for fid in library if passes(fid, ("G03_PRIMARY_OR_FOUNDATIONAL_EVIDENCE", "G13_ENGLISH_FULLTEXT_FORMULA_EVIDENCE"))),
        "source_154": sorted(fid for fid in library if passes(fid, ("G03_PRIMARY_OR_FOUNDATIONAL_EVIDENCE",))),
        "ultrarelaxed_221": sorted(fid for fid in library if passes(fid, ())),
    }
    previous: set[str] = set()
    summary: dict[str, Mapping[str, Any]] = {}
    for name, feature_ids in sets.items():
        current = set(feature_ids)
        if previous and not previous.issubset(current):
            raise ValueError(f"Feature sets are not nested at {name}")
        actual = (len(current), len({feature_to_dimension[fid] for fid in current}))
        if actual != EXPECTED_SET_COUNTS[name]:
            raise ValueError(f"{name}: expected {EXPECTED_SET_COUNTS[name]}, found {actual}")
        summary[name] = {"feature_count": actual[0], "dimension_count": actual[1], "feature_ids": feature_ids}
        previous = current
    return {
        "passed": True,
        "definition_dir": str(definition_dir.resolve()),
        "files": {name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size} for name, path in paths.items()},
        "sets": summary,
    }


def main() -> None:
    """Validate a restored bundle and write an auditable JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--definition-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = load_bundle(args.definition_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"validated frozen definition bundle: {args.definition_dir}")


if __name__ == "__main__":
    main()
