"""Collect graph-blind, evidence-gated variables from completed GEAR runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from gear.claim_attribution import (
    CLAIM_TYPES,
    FEATURE_SCHEMA_VERSION,
    PATHWAYS,
    pathway_hypothesis,
)
from gear.review_contracts import ReviewBundle
from gear.structural_innovation import _evidence_variables
from gear.trace import EvidenceStore


def collect_evidence(
    runs_dir: Path,
    output_dir: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Emit claim- and paper-level variables with failed runs excluded."""
    target_ids = _manifest_case_ids(manifest_path) if manifest_path else None
    claim_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for bundle_path in sorted(runs_dir.glob("*/review_bundle.json")):
        if target_ids is not None and bundle_path.parent.name not in target_ids:
            continue
        try:
            claim_rows.extend(_collect_run(bundle_path))
        except (OSError, TypeError, ValueError) as exc:
            failures.append(
                {"run_path": str(bundle_path.parent), "reason": type(exc).__name__}
            )
    claims = pd.DataFrame(claim_rows)
    papers = _aggregate_papers(claims)
    output_dir.mkdir(parents=True, exist_ok=True)
    claims.to_parquet(output_dir / "stage_a_claim_evidence.parquet", index=False)
    papers.to_parquet(output_dir / "stage_a_paper_evidence.parquet", index=False)
    summary = {
        "contract": "gear_stage_a_evidence_collection_v1",
        "claim_rows": len(claims),
        "paper_rows": len(papers),
        "claim_t0_schema_version": FEATURE_SCHEMA_VERSION,
        "claim_t0_complete_rows": (
            int(claims["claim_t0_schema_version"].eq(FEATURE_SCHEMA_VERSION).sum())
            if "claim_t0_schema_version" in claims
            else 0
        ),
        "failed_runs": failures,
        "target_manifest": str(manifest_path.resolve()) if manifest_path else None,
        "target_papers": len(target_ids) if target_ids is not None else None,
        "manifest_filtered": target_ids is not None,
        "blinded_to_future_outcome": True,
        "aggregation": "centrality_weighted_top3_claims",
    }
    (output_dir / "evidence_collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _manifest_case_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    case_ids = {str(case.get("case_id") or case["paper_id"]) for case in cases}
    if not case_ids or len(case_ids) != len(cases):
        raise ValueError("manifest must contain unique case_id values")
    return case_ids


def _collect_run(bundle_path: Path) -> list[dict[str, Any]]:
    bundle = ReviewBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    if bundle.state is None:
        raise ValueError("review bundle has no state")
    store = EvidenceStore(bundle_path.parent)
    output: list[dict[str, Any]] = []
    for item in bundle.state.claim_inventory:
        variables = _evidence_variables(bundle.state, item, store, None)
        pathway = pathway_hypothesis(item)
        output.append(
            {
                "paper_id": bundle.state.paper_id,
                "claim_id": item.claim_id,
                "claim_text": item.text,
                "claim_centrality": item.centrality,
                # This categorical value comes only from the validated manuscript
                # inventory; downstream code must never infer it from claim text.
                "claim_type": item.claim_type,
                "pathway_hypothesis": pathway,
                "claim_t0_schema_version": FEATURE_SCHEMA_VERSION,
                **{
                    f"claim_type__{value}": float(item.claim_type == value)
                    for value in CLAIM_TYPES
                },
                **{f"pathway__{value}": float(pathway == value) for value in PATHWAYS},
                "manuscript_validity": variables["validity"],
                "evidence_coverage": variables["coverage"],
                "antecedent_risk": variables["antecedent"],
                "residual_novelty": variables["residual"],
                "mechanism_validity": variables["mechanism"],
                "gear_run_path": str(bundle_path.parent.resolve()),
                "review_status": bundle.status.value,
                "verification_passed": bundle.verification.passed,
                "blinded_to_future_outcome": True,
            }
        )
    return output


def _aggregate_papers(claims: pd.DataFrame) -> pd.DataFrame:
    if claims.empty:
        return pd.DataFrame()
    rows = [_aggregate_group(group) for _, group in claims.groupby("paper_id")]
    return pd.DataFrame(rows).sort_values("paper_id").reset_index(drop=True)


def _aggregate_group(group: pd.DataFrame) -> dict[str, Any]:
    selected = group.sort_values(
        ["claim_centrality", "claim_id"], ascending=[False, True]
    ).head(3)
    weights = pd.to_numeric(selected["claim_centrality"]).clip(lower=0.0)
    if float(weights.sum()) <= 0.0:
        weights = pd.Series(1.0, index=selected.index)
    weights = weights / weights.sum()

    def weighted(column: str) -> float:
        return float((pd.to_numeric(selected[column]) * weights).sum())

    base = (
        pd.to_numeric(selected["manuscript_validity"])
        * pd.to_numeric(selected["evidence_coverage"])
        * pd.to_numeric(selected["residual_novelty"])
    )
    return {
        "paper_id": str(selected.iloc[0]["paper_id"]),
        "gear_evidence_score": float((base * weights).sum()),
        "mechanism_validity": weighted("mechanism_validity"),
        "antecedent_risk": float(pd.to_numeric(selected["antecedent_risk"]).max()),
        "evidence_coverage": weighted("evidence_coverage"),
        "gear_run_path": str(selected.iloc[0]["gear_run_path"]),
        "review_status": str(selected.iloc[0]["review_status"]),
        "verification_passed": bool(selected.iloc[0]["verification_passed"]),
        "blinded_to_future_outcome": True,
        "claims_used": len(selected),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            collect_evidence(args.runs_dir, args.output_dir, args.manifest),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["collect_evidence"]
