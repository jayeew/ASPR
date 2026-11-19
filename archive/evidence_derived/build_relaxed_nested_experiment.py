#!/usr/bin/env python3
"""Build the isolated 7/16/153/219 relaxed nested HGB experiment."""

from __future__ import annotations

import argparse
import json
import shutil
from itertools import pairwise
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .core import canonical_json, file_hash, sha256_text

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
V3_OUTPUT = (
    ROOT.parent
    / "evidence_derived_v3"
    / "experiments"
    / "oof_feature_set_comparison_v3"
    / "outputs"
    / "uncapped_v2"
)
DEFAULT_OUTPUT = ROOT / "experiments" / "relaxed_7_16_153_219_20260820"
EXCLUDED_CONSTANTS = {"EF0118", "EF0304"}


class RelaxedBuildError(RuntimeError):
    """Raised when the relaxed experiment cannot be frozen safely."""


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_sets() -> dict[str, list[str]]:
    payload = json.loads((V3_OUTPUT / "feature_sets.json").read_text(encoding="utf-8"))
    source = payload["sets"]
    sets = {
        "strict_training": list(source["strict_7"]["feature_ids"]),
        "primary": list(source["fulltext_16"]["feature_ids"]),
        "expanded": [
            item
            for item in source["source_154"]["feature_ids"]
            if item not in EXCLUDED_CONSTANTS
        ],
        "broad_t0": [
            item
            for item in source["ultrarelaxed_221"]["feature_ids"]
            if item not in EXCLUDED_CONSTANTS
        ],
    }
    expected = {
        "strict_training": 7,
        "primary": 16,
        "expanded": 153,
        "broad_t0": 219,
    }
    if {name: len(values) for name, values in sets.items()} != expected:
        raise RelaxedBuildError("Unexpected relaxed feature counts")
    order = ("strict_training", "primary", "expanded", "broad_t0")
    for left, right in pairwise(order):
        if not set(sets[left]).issubset(sets[right]):
            raise RelaxedBuildError(f"Feature sets are not nested: {left}/{right}")
    return sets


def _validate_fields(source: Path, sets: dict[str, list[str]]) -> pd.DataFrame:
    inventory = pd.read_csv(
        ROOT / "outputs" / "reviews" / "available_matrix_field_inventory.csv"
    ).set_index("matrix_field")
    schema = set(pq.ParquetFile(source).schema_arrow.names)
    broad = sets["broad_t0"]
    missing = sorted(set(broad) - schema)
    if missing:
        raise RelaxedBuildError(f"Source matrix lacks fields: {missing}")
    if any(int(inventory.loc[field, "unique_count"]) <= 1 for field in broad):
        raise RelaxedBuildError("Broad set contains a constant field")
    if set(EXCLUDED_CONSTANTS) - schema:
        raise RelaxedBuildError("Expected constant fields are absent from source")
    if any(int(inventory.loc[field, "unique_count"]) != 1 for field in EXCLUDED_CONSTANTS):
        raise RelaxedBuildError("Excluded constant-field contract changed")
    return inventory


def build(output: Path) -> dict[str, Any]:
    """Build frozen matrices and manifests in a new, isolated directory."""
    output = output.resolve()
    if output.exists():
        raise RelaxedBuildError(f"Experiment directory already exists: {output}")
    output.mkdir(parents=True)
    try:
        sets = _load_sets()
        source = V3_OUTPUT / "indicator_matrix_221.parquet"
        inventory = _validate_fields(source, sets)
        protocol = {
            "contract": "evidence_derived_relaxed_nested_protocol_v1",
            "feature_sets": ["strict", "primary", "expanded", "broad_t0"],
            "counts": {name: len(values) for name, values in sets.items()},
            "excluded_constant_fields": sorted(EXCLUDED_CONSTANTS),
            "outcome_columns_used": False,
            "historical_performance_informed": True,
            "interpretation": "exploratory_model_development_not_new_confirmatory_selection",
        }
        protocol_hash = sha256_text(canonical_json(protocol))
        _write_json(output / "relaxed_protocol.json", protocol)
        full_sets = {
            "all": sets["broad_t0"],
            "model": sets["broad_t0"],
            "strict": sets["strict_training"],
            **sets,
        }
        freeze_hash = sha256_text(canonical_json(full_sets))
        frozen = {
            "protocol_hash": protocol_hash,
            "freeze_hash": freeze_hash,
            "frozen_before_model_training": True,
            "outcome_columns_used": False,
            "historical_performance_informed": True,
            "sets": sets,
            "canonical_sets": {
                "F_all": full_sets["all"],
                "F_model": full_sets["model"],
                "F_strict": full_sets["strict"],
            },
        }
        _write_json(output / "final_feature_sets.json", frozen)
        paper_ids = pq.read_table(source, columns=["paper_id"]).to_pandas()
        if paper_ids["paper_id"].isna().any() or paper_ids["paper_id"].duplicated().any():
            raise RelaxedBuildError("Source paper_id grain is invalid")
        matrix_sets: dict[str, Any] = {}
        names = {
            "strict": "strict_training",
            "primary": "primary",
            "expanded": "expanded",
            "broad_t0": "broad_t0",
        }
        for output_name, source_set in names.items():
            features = sets[source_set]
            path = output / f"final_training_features_{output_name}.parquet"
            table = pq.read_table(source, columns=["paper_id", *features])
            pq.write_table(table, path, compression="zstd", version="2.6")
            matrix_sets[output_name] = {
                "source_set": source_set,
                "indicator_ids": features,
                "feature_names": features,
                "path": str(path),
                "sha256": file_hash(path),
                "row_count": table.num_rows,
            }
        matrix_manifest = {
            "contract": "evidence_derived_training_matrices_v1",
            "protocol_hash": protocol_hash,
            "feature_set_freeze_hash": freeze_hash,
            "frozen_before_model_training": True,
            "outcome_columns_used": False,
            "id_column": "paper_id",
            "training_source": str(source.resolve()),
            "training_source_sha256": file_hash(source),
            "sets": matrix_sets,
        }
        _write_json(output / "training_matrix_manifest.json", matrix_manifest)
        lineage = inventory.loc[sets["broad_t0"]].reset_index()
        lineage.to_csv(output / "feature_lineage_219.csv", index=False)
        (output / "audit_report.md").write_text(
            "# Relaxed nested experiment audit\n\n"
            "Status: **COMPLETE**\n\n"
            "- Sets are nested: 7 / 16 / 153 / 219.\n"
            "- EF0118 and EF0304 were excluded because each is constant.\n"
            "- All retained fields are T0 and nonconstant.\n"
            "- No outcome columns were used.\n"
            "- This is a historical-performance-informed exploratory experiment.\n",
            encoding="utf-8",
        )
        shutil.copy2(
            PROJECT_ROOT / "configs" / "nature_multihorizon" / "hgb_uncapped_v2.json",
            output / "hgb_uncapped_v2.json",
        )
        result = {
            "contract": "relaxed_7_16_153_219_build_v1",
            "output": str(output),
            "counts": {name: len(values) for name, values in sets.items()},
            "freeze_hash": freeze_hash,
            "protocol_hash": protocol_hash,
            "row_count": len(paper_ids),
            "source_sha256": file_hash(source),
            "artifacts": {
                path.name: file_hash(path)
                for path in sorted(output.iterdir())
                if path.is_file()
            },
        }
        _write_json(output / "build_manifest.json", result)
        return result
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
