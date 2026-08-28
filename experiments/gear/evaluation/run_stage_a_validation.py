"""Run the real-data Stage-A validation and emit auditable artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .range_restriction_audit import audit_score_range
from .stage_a_dataset import (
    build_score_stratified_cohort,
    cohort_quality,
    load_evidence_inventory,
    load_stage_a_population,
)
from .stage_a_gate0 import evaluate_gate0
from .stage_a_three_arm import run_three_arm_experiment

LEGACY_REVIEW_PATH = Path("data/review_innovation_opinions_v1/papers.csv")
NATURE_MANIFEST_PATH = Path("/mnt/d/aspr_nature_markdown/manifest.jsonl")


def run_validation(
    output_dir: Path,
    *,
    per_decile: int = 20,
    gear_evidence_path: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    population = load_stage_a_population()
    if gear_evidence_path is not None:
        if not gear_evidence_path.is_file():
            raise FileNotFoundError(gear_evidence_path)
        evidence = pd.read_parquet(gear_evidence_path)
        cohort = _frozen_gear_target_cohort(population, evidence)
    else:
        evidence = load_evidence_inventory()
        cohort = build_score_stratified_cohort(
            population,
            per_decile=per_decile,
            evidence_inventory=evidence,
        )
    if gear_evidence_path is not None:
        cohort = _attach_gear_evidence(cohort, evidence)
    graph_validity, domain_metrics, fold_metrics, decile_metrics = _graph_validity(
        population
    )
    data_quality = _data_quality(population, cohort, evidence)
    gate0 = evaluate_gate0(population)
    integration_frame = _integration_frame(cohort)
    arms, integration = run_three_arm_experiment(integration_frame)
    legacy_range = _legacy_range_audit()
    conclusion = _conclusion(graph_validity, gate0, integration)
    cohort.to_csv(output_dir / "stage_a_cohort_200.csv", index=False)
    _evidence_template(cohort).to_csv(
        output_dir / "stage_a_gear_evidence_template.csv", index=False
    )
    domain_metrics.to_csv(output_dir / "graph_validity_by_domain.csv", index=False)
    fold_metrics.to_csv(output_dir / "graph_validity_by_fold.csv", index=False)
    decile_metrics.to_csv(output_dir / "graph_validity_by_decile.csv", index=False)
    if not arms.empty:
        arms.to_csv(output_dir / "stage_a_three_arm_scores.csv", index=False)
    result = {
        "contract": "gear_stage_a_real_data_validation_v1",
        "conclusion": conclusion,
        "graph_predictive_validity": graph_validity,
        "gate0": gate0,
        "integration": integration,
        "data_quality": data_quality,
        "cohort_quality": cohort_quality(cohort),
        "legacy_review_range_audit": legacy_range,
        "sources": _sources(),
    }
    _write_json(output_dir / "stage_a_validation.json", result)
    _write_json(output_dir / "data_quality_audit.json", data_quality)
    _write_json(output_dir / "gate0_report.json", gate0)
    result["artifact_sha256"] = _artifact_hashes(output_dir)
    _write_json(output_dir / "stage_a_validation_manifest.json", result)
    return result


def _frozen_gear_target_cohort(
    population: pd.DataFrame, evidence: pd.DataFrame
) -> pd.DataFrame:
    """Bind the outcome-blind GEAR target manifest without resampling it."""
    if evidence["paper_id"].astype(str).duplicated().any():
        raise ValueError("GEAR evidence target papers must be unique")
    target_ids = set(evidence["paper_id"].astype(str))
    cohort = population[population["paper_id"].astype(str).isin(target_ids)].copy()
    if set(cohort["paper_id"].astype(str)) != target_ids:
        raise ValueError(
            "GEAR evidence target contains papers outside the OOF population"
        )
    cohort["manuscript_path"] = pd.NA
    cohort["review_history_path"] = pd.NA
    cohort["gear_evidence_available"] = False
    cohort["integration_eligible"] = False
    cohort["evidence_status"] = "gear_evidence_pending_attachment"
    return cohort.sort_values(["score_decile", "domain12", "stable_key"]).reset_index(
        drop=True
    )


def _graph_validity(
    population: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    finite = population.dropna(
        subset=["expected_diffusion_score", "realized_diffusion_target"]
    ).copy()
    rho = float(
        finite["expected_diffusion_score"].corr(
            finite["realized_diffusion_target"], method="spearman"
        )
    )
    domain = _group_spearman(finite, "domain12")
    folds = _group_spearman(finite, "outer_fold_id")
    finite["score_decile"] = np.clip(
        np.floor(finite["prospective_5y_diffusion_percentile"] / 10.0), 0, 9
    ).astype(int)
    decile = (
        finite.groupby("score_decile", observed=True)
        .agg(
            papers=("paper_id", "size"),
            mean_expected_diffusion=("expected_diffusion_score", "mean"),
            mean_realized_diffusion=("realized_diffusion_target", "mean"),
            uptake_rate=("future_uptake", "mean"),
        )
        .reset_index()
    )
    top = finite[finite["score_decile"].eq(9)]["realized_diffusion_target"].mean()
    overall = finite["realized_diffusion_target"].mean()
    brier = float(
        np.mean(
            (
                finite["uptake_probability"].to_numpy(float)
                - finite["future_uptake"].to_numpy(float)
            )
            ** 2
        )
    )
    return (
        {
            "status": "supported" if rho >= 0.6 else "not_supported",
            "papers": len(finite),
            "spearman": rho,
            "top_decile_lift": float(top / overall) if overall else None,
            "uptake_brier": brier,
            "worst_domain_spearman": float(domain["spearman"].min()),
            "worst_fold_spearman": float(folds["spearman"].min()),
            "claim_scope": "Graph predictive validity only; not integration utility",
        },
        domain,
        folds,
        decile,
    )


def _group_spearman(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(column, observed=True):
        rows.append(
            {
                column: value,
                "papers": len(group),
                "spearman": float(
                    group["expected_diffusion_score"].corr(
                        group["realized_diffusion_target"], method="spearman"
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman").reset_index(drop=True)


def _data_quality(
    population: pd.DataFrame, cohort: pd.DataFrame, evidence: pd.DataFrame
) -> dict[str, Any]:
    nature = _nature_manifest_summary()
    evidence_ids = set(evidence.get("paper_id", pd.Series(dtype=str)).astype(str))
    oof_ids = set(population["paper_id"].astype(str))
    overlap = len(evidence_ids & oof_ids)
    eligible = int(cohort["integration_eligible"].sum())
    blocking: list[dict[str, str]] = []
    if overlap == 0:
        blocking.append(
            {
                "severity": "critical",
                "code": "no_oof_gear_evidence_overlap",
                "evidence": "overlap=0",
                "impact": "The three-arm integration estimand is not identifiable.",
                "remediation": (
                    "Acquire submission/publication-time manuscripts for the frozen "
                    "OOF cohort and run GEAR without exposing future outcomes."
                ),
            }
        )
    elif eligible < 100:
        blocking.append(
            {
                "severity": "critical",
                "code": "insufficient_oof_gear_evidence_overlap",
                "evidence": f"integration_eligible={eligible}<100",
                "impact": "The registered three-arm minimum is not met.",
                "remediation": "Complete additional blinded GEAR runs.",
            }
        )
    return {
        "oof_rows": len(population),
        "oof_unique_papers": int(population["paper_id"].nunique()),
        "oof_publication_year_min": int(population["publication_year"].min()),
        "oof_publication_year_max": int(population["publication_year"].max()),
        "oof_realized_outcome_finite_rate": float(
            population["realized_diffusion_target"].notna().mean()
        ),
        "score_table_join_rate": float(
            population["prospective_5y_diffusion_percentile"].notna().mean()
        ),
        "available_evidence_inventory_rows": len(evidence),
        "evidence_oof_overlap": overlap,
        "selected_cohort_integration_eligible": eligible,
        "nature_fulltext_inventory": nature,
        "blocking_issues": blocking,
    }


def _nature_manifest_summary() -> dict[str, Any]:
    if not NATURE_MANIFEST_PATH.is_file():
        return {"status": "missing"}
    rows = [
        json.loads(line)
        for line in NATURE_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    years = [int(row["year"]) for row in rows]
    return {
        "status": "available",
        "rows": len(rows),
        "publication_year_min": min(years),
        "publication_year_max": max(years),
    }


def _integration_frame(cohort: pd.DataFrame) -> pd.DataFrame:
    eligible = cohort[cohort["integration_eligible"]].copy()
    if eligible.empty:
        return pd.DataFrame(columns=sorted({"paper_id"}))
    return eligible.rename(
        columns={
            "expected_diffusion_score": "graph_expected_diffusion",
            "realized_diffusion_target": "future_structural_outcome",
        }
    )


def _attach_gear_evidence(cohort: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    required = {
        "paper_id",
        "gear_evidence_score",
        "mechanism_validity",
        "antecedent_risk",
        "evidence_coverage",
        "gear_run_path",
        "blinded_to_future_outcome",
    }
    missing = sorted(required - set(evidence))
    if missing:
        raise ValueError(f"GEAR evidence is missing columns: {missing}")
    fields = sorted(required)
    output = cohort.drop(
        columns=["gear_evidence_available", "integration_eligible", "evidence_status"]
    ).merge(evidence[fields], on="paper_id", how="left", validate="one_to_one")
    output["gear_evidence_available"] = output["gear_evidence_score"].notna()
    output["integration_eligible"] = (
        output["gear_evidence_available"]
        & output["realized_diffusion_target"].notna()
        & output["blinded_to_future_outcome"].fillna(False)
    )
    output["evidence_status"] = np.where(
        output["integration_eligible"], "ready", "gear_evidence_required"
    )
    return output


def _evidence_template(cohort: pd.DataFrame) -> pd.DataFrame:
    base = [
        "paper_id",
        "score_decile",
        "domain12",
        "publication_year",
        "manuscript_path",
        "review_history_path",
    ]
    evidence_fields = [
        "gear_evidence_score",
        "mechanism_validity",
        "antecedent_risk",
        "evidence_coverage",
        "gear_run_path",
        "blinded_to_future_outcome",
    ]
    template = cohort[
        base + [field for field in evidence_fields if field in cohort]
    ].copy()
    for column in (
        "gear_evidence_score",
        "mechanism_validity",
        "antecedent_risk",
        "evidence_coverage",
    ):
        if column not in template:
            template[column] = pd.NA
    if "gear_run_path" not in template:
        template["gear_run_path"] = pd.NA
    if "blinded_to_future_outcome" not in template:
        template["blinded_to_future_outcome"] = True
    return template


def _legacy_range_audit() -> dict[str, Any]:
    if not LEGACY_REVIEW_PATH.is_file():
        return {"status": "missing"}
    frame = pd.read_csv(LEGACY_REVIEW_PATH).rename(
        columns={"fig3_sw_percentile": "graph_percentile"}
    )
    frame["graph_percentile"] *= 100.0
    return audit_score_range(frame)


def _conclusion(
    graph: dict[str, Any], gate0: dict[str, Any], integration: dict[str, Any]
) -> dict[str, Any]:
    established = (
        graph["status"] == "supported"
        and gate0["status"] == "passed"
        and integration["status"] == "estimated"
        and integration["real_hgb"]["integration_value"] > 0.0
        and integration["real_minus_shuffled_value"] > 0.0
    )
    return {
        "stage_a_established": established,
        "verdict": "supported" if established else "not_yet_identifiable",
        "graph_predictive_validity": graph["status"],
        "deterministic_correctness": gate0["status"],
        "integration_utility": integration["status"],
        "claim_allowed": established,
    }


def _sources() -> list[dict[str, str]]:
    return [
        {
            "id": "oof",
            "path": str(
                Path(
                    "data/calibration/releases/gear-d5-primary16-current/"
                    "oof_predictions.parquet"
                ).resolve()
            ),
        },
        {
            "id": "score_table",
            "path": str(
                Path(
                    "data/calibration/releases/gear-d5-primary16-current/"
                    "score_table.parquet"
                ).resolve()
            ),
        },
        {"id": "nature_manifest", "path": str(NATURE_MANIFEST_PATH)},
    ]


def _artifact_hashes(output_dir: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for path in sorted(output_dir.iterdir()):
        if not path.is_file() or path.name == "stage_a_validation_manifest.json":
            continue
        output[path.name] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return output


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-decile", type=int, default=20)
    parser.add_argument("--gear-evidence", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_validation(
        args.output_dir,
        per_decile=args.per_decile,
        gear_evidence_path=args.gear_evidence,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_validation"]
