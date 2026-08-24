#!/usr/bin/env python3
"""Recover development seed citations as inputs, never legacy decisions."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from .core import canonical_json, sha256_bytes, stable_id
except ImportError:
    from core import canonical_json, sha256_bytes, stable_id  # type: ignore[no-redef]


SOURCE_REVISION = "8acaeaa"
SOURCE_PATHS = (
    "innovation_impact_feature_selection/literature_evidence.json",
    "innovation_impact_feature_selection/expanded_review_v2/additional_literature_evidence.json",
)


def git_blob(repository: Path, revision: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    return result.stdout


def recover(repository: Path, output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    for path in SOURCE_PATHS:
        payload = git_blob(repository, SOURCE_REVISION, path)
        sources[f"{SOURCE_REVISION}:{path}"] = sha256_bytes(payload)
        value = json.loads(payload)
        for record in value["records"]:
            doi = str(record.get("doi") or "").strip().lower()
            if not doi:
                raise RuntimeError(f"Development seed lacks DOI: {record}")
            rows.append(
                {
                    "seed_id": stable_id("DEV", doi),
                    "cohort": "development",
                    "doi": doi,
                    "citation": str(record.get("citation") or ""),
                    "publication_year": record.get("year"),
                    "language": "en",
                    "source_revision": SOURCE_REVISION,
                    "source_path": path,
                }
            )
    if len(rows) != 53 or len({row["doi"] for row in rows}) != 53:
        raise RuntimeError("Expected exactly 53 unique DOI development seeds")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["seed_id"]))
    manifest = {
        "schema_version": "development_seed_recovery_1",
        "legacy_use": "seed_inputs_only_no_decisions",
        "source_hashes": sources,
        "seed_count": len(rows),
        "unique_doi_count": len({row["doi"] for row in rows}),
        "output_sha256": sha256_bytes(output.read_bytes()),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_json(recover(args.repository.resolve(), args.output.resolve())))


if __name__ == "__main__":
    main()
