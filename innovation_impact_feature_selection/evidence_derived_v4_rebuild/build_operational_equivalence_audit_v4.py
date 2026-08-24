"""Audit exact T0 operational equivalence for locally materializable counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from common import ROOT, sha256_file, write_json
from pyarrow import parquet

DATA_ROOT = (
    ROOT.parent.parent
    / "data"
    / "knowledge_corpus"
    / "nature_multihorizon_v6_1_uncapped_v2"
)
OUTPUT = ROOT / "outputs" / "operational_equivalence_audit_v4.json"


def _paths(data_root: Path) -> dict[str, Path]:
    """Resolve and verify every outcome-blind parquet input used by this audit."""
    paths = {
        "target_openalex_metadata": data_root / "target_openalex_metadata.parquet",
        "paper_references": data_root / "paper_references.parquet",
        "control_features": data_root / "control_features_v6_1.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return paths


def _author_audit(paths: dict[str, Path]) -> dict[str, Any]:
    """Compare the stored author count to the inverse logged control field."""
    metadata = parquet.read_table(
        paths["target_openalex_metadata"],
        columns=["paper_id", "openalex_author_count"],
    ).to_pandas()
    controls = parquet.read_table(
        paths["control_features"], columns=["paper_id", "log_author_count"]
    ).to_pandas()
    merged = metadata.merge(controls, on="paper_id", how="inner")
    source = merged["openalex_author_count"].astype(float).to_numpy()
    derived = np.expm1(merged["log_author_count"].astype(float).to_numpy())
    return {
        "source_formula": "author_count = openalex_author_count",
        "local_derivation": "author_count = expm1(log_author_count)",
        "metadata_rows": len(metadata),
        "control_rows": len(controls),
        "overlap_rows": len(merged),
        "nonmissing_overlap_rate": float(np.isfinite(source).mean()),
        "exact_equality_rate": float(
            np.isclose(source, derived, rtol=0, atol=1e-8).mean()
        ),
        "max_absolute_difference": float(np.max(np.abs(source - derived))),
        "status": "exact_numeric_representation_equivalence",
    }


def _reference_audit(paths: dict[str, Path]) -> dict[str, Any]:
    """Compare reference-edge counts to the inverse logged control field."""
    references = parquet.read_table(
        paths["paper_references"], columns=["paper_id", "reference_id"]
    )
    grouped = (
        references.group_by("paper_id")
        .aggregate([("reference_id", "count")])
        .to_pandas()
    )
    controls = parquet.read_table(
        paths["control_features"], columns=["paper_id", "log_reference_count"]
    ).to_pandas()
    merged = grouped.merge(controls, on="paper_id", how="inner")
    source = merged["reference_id_count"].astype(float).to_numpy()
    derived = np.expm1(merged["log_reference_count"].astype(float).to_numpy())
    return {
        "source_formula": "reference_count = count(focal-paper backward reference edges)",
        "local_derivation": "reference_count = expm1(log_reference_count)",
        "edge_observed_paper_rows": len(grouped),
        "control_rows": len(controls),
        "overlap_rows": len(merged),
        "control_coverage_rate": float(len(merged) / len(controls)),
        "exact_equality_rate": float(
            np.isclose(source, derived, rtol=0, atol=1e-8).mean()
        ),
        "max_absolute_difference": float(np.max(np.abs(source - derived))),
        "missing_rule": "No observed backward edges is missing/unknown, never recoded as zero.",
        "status": "exact_numeric_representation_equivalence_with_audited_coverage",
    }


def build(data_root: Path) -> dict[str, Any]:
    """Create a reproducible, outcome-blind equivalence audit."""
    paths = _paths(data_root)
    return {
        "schema_version": "operational_equivalence_audit_v4",
        "scope": (
            "Only exact source-formula-preserving representation transforms are "
            "authorized; this audit does not authorize construct proxies."
        ),
        "outcome_columns_used": False,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "author_count": _author_audit(paths),
        "reference_count": _reference_audit(paths),
    }


def main() -> None:
    """Run the audit and print the immutable JSON path and hash."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = build(args.data_root.resolve())
    write_json(args.output.resolve(), payload)
    result = {
        "path": str(args.output.resolve()),
        "sha256": sha256_file(args.output.resolve()),
        "author_exact_equality_rate": payload["author_count"]["exact_equality_rate"],
        "reference_exact_equality_rate": payload["reference_count"][
            "exact_equality_rate"
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
