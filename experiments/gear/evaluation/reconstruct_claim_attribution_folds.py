"""Reconstruct strict held-out Claim Attribution folds from frozen Gate-1 inputs.

This is deliberately a local replay of the registered Ridge fitting procedure.
It never invokes a model service and it refuses to use any non-development
labels.  The produced rows are therefore suitable for ranking metrics that
cannot be inferred from the aggregate Gate-1 reports alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .train_claim_attribution_head import (
    FORMAL_SPLIT_COUNTS,
    _conditional_attribution_target,
    _domain_predictions,
    _load_candidate_input,
    _temporal_predictions,
    _validate_candidate_inputs,
)


def reconstruct_strict_claim_attribution_folds(
    temporal_path: Path,
    domain_path: Path,
    release_dir: Path,
    output_dir: Path,
    *,
    seed: int = 20260828,
    expected_split_counts: dict[str, int] = FORMAL_SPLIT_COUNTS,
) -> pd.DataFrame:
    """Write and validate strict fold rows plus frozen top-claim summaries."""
    temporal = _load_candidate_input(temporal_path)
    domain = _load_candidate_input(domain_path)
    _validate_candidate_inputs(temporal, domain, expected_split_counts)
    _validate_release_sources(temporal_path, domain_path, release_dir)
    development = _conditional_attribution_target(
        temporal.loc[temporal["integration_split"].eq("development")].copy()
    )
    temporal_rows = _annotate_temporal(
        _temporal_predictions(development, seed), development
    )
    domain_rows = _annotate_domain(_domain_predictions(development, seed), development)
    rows = pd.concat([temporal_rows, domain_rows], ignore_index=True)
    _validate_strict_rows(rows, development)
    _validate_against_frozen_gate_reports(rows, release_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_dir / "claim_attribution_strict_fold_rows.csv", index=False)
    summary, metrics, selection = _claim_retrieval_summary(rows)
    summary.to_csv(output_dir / "panel_e_top_claim.csv", index=False)
    metrics.to_csv(output_dir / "panel_e_claim_retrieval_metrics.csv", index=False)
    _write_audit(
        rows,
        summary,
        metrics,
        selection,
        temporal_path,
        domain_path,
        release_dir,
        output_dir,
    )
    return summary


def _annotate_temporal(
    predictions: pd.DataFrame, development: pd.DataFrame
) -> pd.DataFrame:
    output = predictions.copy()
    year_by_paper = development.groupby("paper_id")["publication_year"].first()
    output["axis"] = "temporal"
    output["protocol"] = "strict_forward_publication_year"
    output["fold_id"] = output["paper_id"].map(year_by_paper).astype(int).astype(str)
    output["heldout_publication_year"] = output["fold_id"].astype(int)
    output["heldout_domain12"] = pd.NA
    return _attach_heuristic(output, development)


def _annotate_domain(
    predictions: pd.DataFrame, development: pd.DataFrame
) -> pd.DataFrame:
    output = predictions.copy()
    domain_by_paper = development.groupby("paper_id")["domain12"].first()
    output["axis"] = "domain"
    output["protocol"] = "leave_one_domain12_out"
    output["fold_id"] = output["paper_id"].map(domain_by_paper).astype(str)
    output["heldout_publication_year"] = pd.NA
    output["heldout_domain12"] = output["fold_id"]
    return _attach_heuristic(output, development)


def _attach_heuristic(
    predictions: pd.DataFrame, development: pd.DataFrame
) -> pd.DataFrame:
    """Attach the registered T0-only centrality heuristic without refitting it."""
    return predictions.merge(
        development[["paper_id", "claim_id", "claim_centrality"]],
        on=["paper_id", "claim_id"],
        how="left",
        validate="one_to_one",
    )


def _validate_release_sources(temporal: Path, domain: Path, release_dir: Path) -> None:
    expected = [
        "sha256:" + hashlib.sha256(temporal.read_bytes()).hexdigest(),
        "sha256:" + hashlib.sha256(domain.read_bytes()).hexdigest(),
    ]
    for axis in ("temporal", "domain"):
        report = _json(release_dir / f"gate1_{axis}.json")
        if report["claim_attribution_runtime_candidate"]["source_sha256"] != expected:
            raise ValueError("Gate-1 source digests do not match the frozen release")


def _validate_strict_rows(rows: pd.DataFrame, development: pd.DataFrame) -> None:
    if rows.duplicated(["axis", "paper_id", "claim_id"]).any():
        raise ValueError("strict fold reconstruction has duplicate claim predictions")
    if not rows["prediction"].between(0.0, 1.0).all():
        raise ValueError("strict fold predictions are outside the normalized range")
    if not rows["future_claim_adoption_share"].between(0.0, 1.0).all():
        raise ValueError("strict fold labels are outside the normalized range")
    for axis, group in rows.groupby("axis", observed=True):
        sums = group.groupby("paper_id")["prediction"].sum()
        if not np.allclose(sums.to_numpy(), 1.0):
            raise ValueError(f"{axis} fold predictions are not per-paper normalized")
        if axis == "temporal":
            year = development.groupby("paper_id")["publication_year"].first()
            for fold, papers in group.groupby("fold_id")["paper_id"]:
                if not year.loc[papers.unique()].eq(int(fold)).all():
                    raise ValueError(
                        "temporal rows are not assigned to their held-out year"
                    )
                if not development.loc[
                    development["publication_year"].lt(int(fold)), "paper_id"
                ].nunique():
                    raise ValueError(
                        "temporal fold has no strictly earlier training papers"
                    )
        else:
            domain = development.groupby("paper_id")["domain12"].first()
            for fold, papers in group.groupby("fold_id")["paper_id"]:
                if not domain.loc[papers.unique()].eq(fold).all():
                    raise ValueError(
                        "domain rows are not assigned to their held-out domain"
                    )
                if (
                    development.loc[
                        development["domain12"].ne(fold), "paper_id"
                    ].nunique()
                    < 10
                ):
                    raise ValueError(
                        "domain fold lacks the registered training minimum"
                    )


def _validate_against_frozen_gate_reports(
    rows: pd.DataFrame, release_dir: Path
) -> None:
    for axis in ("temporal", "domain"):
        report = _json(release_dir / f"gate1_{axis}.json")["metrics"]
        group = rows.loc[rows["axis"].eq(axis)]
        rho = group["future_claim_adoption_share"].corr(
            group["prediction"], method="spearman"
        )
        null_rho = group["future_claim_adoption_share"].corr(
            group["permutation_prediction"], method="spearman"
        )
        actual = (len(group), group["paper_id"].nunique(), float(rho), float(null_rho))
        expected = (
            report["rows"],
            report["papers"],
            report["spearman_rho"],
            report["within_paper_permutation_rho"],
        )
        if actual[:2] != expected[:2] or not np.allclose(actual[2:], expected[2:]):
            raise ValueError(
                f"{axis} reconstruction does not reproduce the frozen Gate-1 report"
            )


PRIMARY_K = 3
CANDIDATE_KS = (1, 2, 3, 5)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260829


def _claim_retrieval_summary(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate the frozen top-3 retrieval endpoint on strict outer folds."""
    selection = _development_selection_record()
    summary: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for index, (axis, group) in enumerate(rows.groupby("axis", sort=True)):
        eligible = _eligible_retrieval_rows(group)
        learned = _paper_metrics(eligible, "prediction", PRIMARY_K)
        heuristic = _paper_metrics(eligible, "claim_centrality", PRIMARY_K)
        uniform = _uniform_metrics(eligible, PRIMARY_K)
        seed = BOOTSTRAP_SEED + index
        learned_ci = _mean_ci(learned, seed)
        uniform_ci = _mean_ci(uniform, seed + 100)
        heuristic_ci = _mean_ci(heuristic, seed + 200)
        uniform_difference = learned - uniform
        heuristic_difference = learned - heuristic
        uniform_difference_ci = _mean_ci(uniform_difference, seed + 300)
        heuristic_difference_ci = _mean_ci(heuristic_difference, seed + 400)
        metrics = {
            "recall_at_3": (learned, uniform, heuristic),
            "precision_at_3": (_paper_precision(eligible, "prediction", PRIMARY_K),),
            "ndcg_at_3": (_paper_ndcg(eligible, "prediction", PRIMARY_K),),
            "mrr": (_paper_mrr(eligible, "prediction"),),
        }
        for name, values in metrics.items():
            if name == "recall_at_3":
                for method, value, ci in (
                    ("learned", values[0], learned_ci),
                    ("uniform_random_expectation", values[1], uniform_ci),
                    ("claim_centrality", values[2], heuristic_ci),
                ):
                    metric_rows.append(
                        _metric_row(axis, name, method, value, ci, eligible)
                    )
            else:
                value = values[0]
                metric_rows.append(
                    _metric_row(
                        axis,
                        name,
                        "learned_secondary",
                        value,
                        _mean_ci(value, seed + len(metric_rows)),
                        eligible,
                    )
                )
        summary.append(
            {
                "axis": axis,
                "k": PRIMARY_K,
                "primary_metric": "paper_level_recall_at_3_any_important_claim",
                "learned_recall_at_3": float(learned.mean()),
                "uniform_random_recall_at_3": float(uniform.mean()),
                "claim_centrality_recall_at_3": float(heuristic.mean()),
                "advantage_over_uniform": float(uniform_difference.mean()),
                "advantage_over_uniform_ci95_low": uniform_difference_ci[0],
                "advantage_over_uniform_ci95_high": uniform_difference_ci[1],
                "advantage_over_centrality": float(heuristic_difference.mean()),
                "advantage_over_centrality_ci95_low": heuristic_difference_ci[0],
                "advantage_over_centrality_ci95_high": heuristic_difference_ci[1],
                "precision_at_3": float(metrics["precision_at_3"][0].mean()),
                "ndcg_at_3": float(metrics["ndcg_at_3"][0].mean()),
                "mrr": float(metrics["mrr"][0].mean()),
                "eligible_papers": int(eligible["paper_id"].nunique()),
                "eligible_claims": len(eligible),
                "protocol": str(eligible["protocol"].iloc[0]),
                "selection_protocol": selection["selection_protocol"],
                "multiplicity_treatment": selection["multiplicity_treatment"],
            }
        )
    return (
        pd.DataFrame(summary).sort_values("axis").reset_index(drop=True),
        pd.DataFrame(metric_rows),
        selection,
    )


def _development_selection_record() -> dict[str, Any]:
    """Record the frozen development-only k protocol without touching test rows."""
    return {
        "candidate_k": list(CANDIDATE_KS),
        "selected_k": PRIMARY_K,
        "selection_protocol": (
            "predeclared Recall@3 primary endpoint; candidate k values {1,2,3,5} "
            "are development-only planning values and cannot change the frozen k=3 test endpoint"
        ),
        "multiplicity_treatment": (
            "No held-out selection or winner search. Candidate-k exploration is confined "
            "to development planning; one predeclared primary endpoint per axis is reported."
        ),
    }


def _eligible_retrieval_rows(frame: pd.DataFrame) -> pd.DataFrame:
    relevant = frame.groupby("paper_id")["future_claim_adoption_share"].transform(
        lambda values: values.gt(0.0).any()
    )
    return frame.loc[relevant].copy()


def _paper_metrics(frame: pd.DataFrame, score: str, k: int) -> pd.Series:
    values: dict[str, float] = {}
    for paper, group in frame.groupby("paper_id", sort=True):
        ordered = group.sort_values([score, "claim_id"], ascending=[False, True])
        relevant = ordered["future_claim_adoption_share"].gt(0.0).to_numpy()
        values[str(paper)] = float(relevant[:k].any())
    return pd.Series(values, dtype=float)


def _uniform_metrics(frame: pd.DataFrame, k: int) -> pd.Series:
    values: dict[str, float] = {}
    for paper, group in frame.groupby("paper_id", sort=True):
        n_claims = len(group)
        relevant = int(group["future_claim_adoption_share"].gt(0.0).sum())
        values[str(paper)] = float(
            1.0 - _comb(n_claims - relevant, k) / _comb(n_claims, k)
        )
    return pd.Series(values, dtype=float)


def _paper_precision(frame: pd.DataFrame, score: str, k: int) -> pd.Series:
    return _paper_statistic(frame, score, k, lambda relevant, _: relevant[:k].mean())


def _paper_ndcg(frame: pd.DataFrame, score: str, k: int) -> pd.Series:
    def ndcg(relevant: np.ndarray, relevant_total: int) -> float:
        discounts = 1.0 / np.log2(np.arange(2, min(k, len(relevant)) + 2))
        dcg = float((relevant[:k] * discounts).sum())
        ideal = float(discounts[: min(k, relevant_total)].sum())
        return dcg / ideal if ideal else 0.0

    return _paper_statistic(frame, score, k, ndcg)


def _paper_mrr(frame: pd.DataFrame, score: str) -> pd.Series:
    def mrr(relevant: np.ndarray, _: int) -> float:
        indexes = np.flatnonzero(relevant)
        return float(1.0 / (int(indexes[0]) + 1)) if len(indexes) else 0.0

    return _paper_statistic(frame, score, len(frame), mrr)


def _paper_statistic(
    frame: pd.DataFrame, score: str, k: int, statistic: Any
) -> pd.Series:
    values: dict[str, float] = {}
    for paper, group in frame.groupby("paper_id", sort=True):
        ordered = group.sort_values([score, "claim_id"], ascending=[False, True])
        relevant = ordered["future_claim_adoption_share"].gt(0.0).to_numpy(dtype=float)
        values[str(paper)] = float(statistic(relevant, int(relevant.sum())))
    return pd.Series(values, dtype=float)


def _comb(n: int, k: int) -> float:
    if k > n:
        return 0.0
    return float(math.comb(n, k))


def _mean_ci(values: pd.Series, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = [
        float(rng.choice(values.to_numpy(), len(values), replace=True).mean())
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _metric_row(
    axis: str,
    metric: str,
    method: str,
    values: pd.Series,
    ci: tuple[float, float],
    eligible: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "axis": axis,
        "metric": metric,
        "method": method,
        "k": PRIMARY_K if metric != "mrr" else pd.NA,
        "estimate": float(values.mean()),
        "ci95_low": ci[0],
        "ci95_high": ci[1],
        "eligible_papers": int(eligible["paper_id"].nunique()),
        "eligible_claims": len(eligible),
    }


def _write_audit(
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    selection: dict[str, Any],
    temporal: Path,
    domain: Path,
    release_dir: Path,
    output_dir: Path,
) -> None:
    audit = {
        "contract": "fig4_panel_e_strict_top3_retrieval_v2",
        "model_calls": False,
        "labeling_calls": False,
        "source_paths": [str(temporal), str(domain)],
        "release_dir": str(release_dir),
        "only_development_labels_used": True,
        "in_sample_rows_in_endpoint": False,
        "rows_by_axis": {axis: len(group) for axis, group in rows.groupby("axis")},
        "papers_by_axis": {
            axis: int(group["paper_id"].nunique())
            for axis, group in rows.groupby("axis")
        },
        "primary_endpoint": {
            "metric": "paper-level Recall@3: at least one independently labeled important claim in the model top 3",
            "important_claim_definition": "future_claim_adoption_share > 0 from frozen adoption labels",
            "ranking": "frozen claim-attribution prediction; ties by claim_id",
            "baseline": "within-paper uniform-random top-3 expectation",
            "deterministic_heuristic": "frozen T0 claim_centrality; ties by claim_id",
            "bootstrap": {
                "unit": "paper",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "interval": "percentile 95%",
            },
        },
        "development_selection": selection,
        "summary": summary.to_dict(orient="records"),
        "all_metrics": metrics.to_dict(orient="records"),
    }
    (output_dir / "panel_e_top_claim_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--domain", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reconstruct_strict_claim_attribution_folds(
        args.temporal, args.domain, args.release_dir, args.output_dir
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
