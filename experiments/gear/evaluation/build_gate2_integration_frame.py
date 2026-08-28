"""Aggregate claim-level Gate-1 records into paper-level Gate-2 endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_integration_frame(
    gate1_path: Path,
    output_dir: Path,
    *,
    paper_manifest_path: Path | None = None,
    epsilon: float = 0.1,
    integration_splits: list[str] | None = None,
) -> dict[str, Any]:
    """Create one immutable integration observation per held-out paper."""
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")
    claims = pd.read_parquet(gate1_path)
    required = {
        "paper_id",
        "domain12",
        "publication_year",
        "outer_fold_id",
        "structural_innovation_score",
        "shuffled_structural_score",
        "structural_score_at_zero",
        "future_structural_outcome",
    }
    missing = sorted(required - set(claims))
    if missing:
        raise ValueError(f"Gate-2 integration columns are missing: {missing}")
    if paper_manifest_path is not None:
        allowed = _manifest_ids(paper_manifest_path)
        claims = claims[claims["paper_id"].astype(str).isin(allowed)].copy()
    if integration_splits is not None:
        if "integration_split" not in claims:
            raise ValueError(
                "integration_split is required when split filters are used"
            )
        claims = claims[claims["integration_split"].isin(integration_splits)].copy()
    rows: list[dict[str, Any]] = []
    for paper_id, group in claims.groupby("paper_id", observed=True):
        rows.append(
            {
                "paper_id": str(paper_id),
                "integration_split": (
                    str(group["integration_split"].iloc[0])
                    if "integration_split" in group
                    else "unspecified"
                ),
                "domain12": str(group["domain12"].iloc[0]),
                "publication_year": int(group["publication_year"].iloc[0]),
                "outer_fold_id": group["outer_fold_id"].iloc[0],
                "gear_evidence_score": _noisy_or(
                    group["structural_score_at_zero"] / epsilon
                ),
                "joint_structural_score": _noisy_or(
                    group["structural_innovation_score"]
                ),
                "shuffled_structural_score": _noisy_or(
                    group["shuffled_structural_score"]
                ),
                "future_structural_outcome": float(
                    group["future_structural_outcome"].iloc[0]
                ),
                "claims": len(group),
            }
        )
    output = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gate2_integration_frame.parquet"
    output.to_parquet(output_path, index=False)
    report = {
        "contract": "gear_gate2_integration_frame_v1",
        "papers": len(output),
        "domains": int(output["domain12"].nunique()) if not output.empty else 0,
        "outer_folds": (
            int(output["outer_fold_id"].nunique()) if not output.empty else 0
        ),
        "paper_manifest_applied": paper_manifest_path is not None,
        "integration_splits": integration_splits or [],
        "claim_aggregation": "noisy_or",
        "output_sha256": "sha256:"
        + hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    (output_dir / "gate2_integration_frame_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _manifest_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", []) if isinstance(payload, dict) else payload
    return {str(case["paper_id"]) for case in cases}


def _noisy_or(values: pd.Series) -> float:
    bounded = pd.to_numeric(values).clip(0.0, 1.0).to_numpy(float)
    return float(np.clip(1.0 - np.prod(1.0 - bounded), 0.0, 1.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate1", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-manifest", type=Path)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--split", action="append")
    args = parser.parse_args()
    result = build_integration_frame(
        args.gate1,
        args.output_dir,
        paper_manifest_path=args.paper_manifest,
        epsilon=args.epsilon,
        integration_splits=args.split,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_integration_frame"]
