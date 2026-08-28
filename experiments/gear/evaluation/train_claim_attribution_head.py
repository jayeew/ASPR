"""Train and scientifically gate the portable claim-attribution T0 head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from gear.claim_attribution import (
    FEATURE_SCHEMA_VERSION,
    T0_FEATURE_NAMES,
    ClaimAttributionLinearHead,
    promote_claim_attribution_release,
)

FORMAL_SPLIT_COUNTS = {
    "development": 144,
    "domain_holdout": 48,
    "temporal_holdout": 29,
    "joint_time_domain_holdout": 20,
}
RIDGE_ALPHA = 10.0


def train_and_gate_claim_attribution(
    temporal_path: Path,
    domain_path: Path,
    output_dir: Path,
    *,
    release_id: str,
    seed: int = 20260828,
    bootstrap_replicates: int = 2000,
    expected_split_counts: Mapping[str, int] = FORMAL_SPLIT_COUNTS,
) -> dict[str, Any]:
    """Use development labels only; promote iff temporal and domain gates pass."""
    temporal = _load_candidate_input(temporal_path)
    domain = _load_candidate_input(domain_path)
    _validate_candidate_inputs(temporal, domain, expected_split_counts)
    columns = [
        "paper_id",
        "claim_id",
        "outer_fold_id",
        "domain12",
        "publication_year",
        "future_adoption",
        *T0_FEATURE_NAMES,
    ]
    development = temporal.loc[
        temporal["integration_split"].eq("development"), columns
    ].copy()
    # No confirmatory label is selected, evaluated, tuned on, or passed below.
    attribution_development = _conditional_attribution_target(development)
    temporal_predictions = _temporal_predictions(attribution_development, seed)
    domain_predictions = _domain_predictions(attribution_development, seed)
    model = _fit_model(
        attribution_development[list(T0_FEATURE_NAMES)],
        attribution_development["future_claim_adoption_share"],
    )
    portable = ClaimAttributionLinearHead(
        feature_names=list(T0_FEATURE_NAMES),
        coefficients=[float(value) for value in model.coef_],
        intercept=float(model.intercept_),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "claim_attribution_linear_head.json"
    model_path.write_text(portable.model_dump_json(indent=2) + "\n", encoding="utf-8")
    model_hash = _sha256(model_path)
    replay_path = output_dir / "claim_attribution_replay.json"
    _write_replay(development, portable, replay_path)
    reports = []
    for axis, predictions in (
        ("temporal", temporal_predictions),
        ("domain", domain_predictions),
    ):
        report = _gate_report(
            axis,
            predictions,
            model_hash=model_hash,
            seed=seed,
            bootstrap_replicates=bootstrap_replicates,
            development=development,
            attribution_development=attribution_development,
            source_paths=(temporal_path, domain_path),
        )
        report_path = output_dir / f"gate1_{axis}.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        reports.append(report_path)
    passed = all(json.loads(path.read_text())["status"] == "passed" for path in reports)
    release_path = output_dir / "release.json"
    blocked_path = output_dir / "promotion_blocked.json"
    if not passed:
        release_path.unlink(missing_ok=True)
        blocked = {
            "contract": "gear_claim_attribution_promotion_blocked_v1",
            "status": "blocked",
            "reason": "temporal_and_domain_scientific_gates_must_both_pass",
            "gate1_reports": [path.name for path in reports],
            "confirmatory_holdout_labels_used": False,
        }
        blocked_path.write_text(
            json.dumps(blocked, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return blocked
    blocked_path.unlink(missing_ok=True)
    release = promote_claim_attribution_release(
        model_path=model_path,
        replay_path=replay_path,
        gate1_report_paths=reports,
        output_path=release_path,
        release_id=release_id,
    )
    return release.model_dump(mode="json")


def _validate_candidate_inputs(
    temporal: pd.DataFrame,
    domain: pd.DataFrame,
    expected_split_counts: Mapping[str, int],
) -> None:
    required = {
        "paper_id",
        "claim_id",
        "integration_split",
        "outer_fold_id",
        "domain12",
        "publication_year",
        "claim_t0_schema_version",
        *T0_FEATURE_NAMES,
    }
    for name, frame in (("temporal", temporal), ("domain", domain)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} Gate-1 exact T0 columns missing: {missing}")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"{name} Gate-1 exact T0 columns contain null values")
        if not frame["claim_t0_schema_version"].eq(FEATURE_SCHEMA_VERSION).all():
            raise ValueError(f"{name} Gate-1 mixes claim T0 schema versions")
        if frame[["paper_id", "claim_id"]].duplicated().any():
            raise ValueError(f"{name} Gate-1 contains duplicate claim keys")
        counts = {
            str(split): int(group["paper_id"].nunique())
            for split, group in frame.groupby("integration_split", observed=True)
        }
        if counts != dict(expected_split_counts):
            raise ValueError(f"{name} Gate-1 formal split coverage mismatch: {counts}")
    compare = [
        "paper_id",
        "claim_id",
        "integration_split",
        "outer_fold_id",
        "domain12",
        "publication_year",
        "claim_t0_schema_version",
        *T0_FEATURE_NAMES,
    ]
    left = (
        temporal[compare].sort_values(["paper_id", "claim_id"]).reset_index(drop=True)
    )
    right = domain[compare].sort_values(["paper_id", "claim_id"]).reset_index(drop=True)
    if not left.equals(right):
        raise ValueError("temporal/domain Gate-1 exact T0 cohorts differ")
    dev = temporal["integration_split"].eq("development")
    left_y = temporal.loc[dev, ["paper_id", "claim_id", "future_adoption"]].sort_values(
        ["paper_id", "claim_id"]
    )
    right_y = domain.loc[
        domain["integration_split"].eq("development"),
        ["paper_id", "claim_id", "future_adoption"],
    ].sort_values(["paper_id", "claim_id"])
    if not left_y.reset_index(drop=True).equals(right_y.reset_index(drop=True)):
        raise ValueError("development labels differ between Gate-1 axes")


def _load_candidate_input(path: Path) -> pd.DataFrame:
    """Load full T0 metadata but physically filter future labels to development."""
    schema_columns = [
        "paper_id",
        "claim_id",
        "integration_split",
        "outer_fold_id",
        "domain12",
        "publication_year",
        "claim_t0_schema_version",
        *T0_FEATURE_NAMES,
    ]
    available = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted({*schema_columns, "future_adoption"}.difference(available))
    if missing:
        raise ValueError(f"Gate-1 exact T0 columns missing: {missing}")
    schema = pd.read_parquet(
        path,
        columns=schema_columns,
    )
    labels = pd.read_parquet(
        path,
        columns=["paper_id", "claim_id", "integration_split", "future_adoption"],
        filters=[("integration_split", "==", "development")],
    )
    if not labels["integration_split"].eq("development").all():
        raise ValueError("parquet development-label filter was not enforced")
    return schema.merge(
        labels.drop(columns="integration_split"),
        on=["paper_id", "claim_id"],
        how="left",
        validate="one_to_one",
    )


def _conditional_attribution_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the parameter-free target matching runtime per-paper normalization."""
    output = frame.copy()
    adoption = pd.to_numeric(output["future_adoption"], errors="coerce")
    if adoption.isna().any() or adoption.lt(0.0).any():
        raise ValueError("development claim-adoption labels must be finite nonnegative")
    totals = adoption.groupby(output["paper_id"]).transform("sum")
    output["future_claim_adoption_share"] = adoption.div(totals.replace(0.0, np.nan))
    informative = output[totals.gt(0.0)].copy()
    if informative["paper_id"].nunique() < 20:
        raise ValueError("too few development papers have any observed claim adoption")
    return informative


def _fit_model(features: pd.DataFrame, target: pd.Series) -> Ridge:
    # The development cohort is small relative to the frozen 17-feature schema;
    # stronger fixed shrinkage stabilizes strict time/domain extrapolation.
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(
        features.to_numpy(dtype=float), pd.to_numeric(target).to_numpy(dtype=float)
    )
    return model


def _temporal_predictions(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    del seed
    rows: list[pd.DataFrame] = []
    for year in sorted(pd.to_numeric(frame["publication_year"]).unique())[1:]:
        train = frame[pd.to_numeric(frame["publication_year"]).lt(year)]
        test = frame[pd.to_numeric(frame["publication_year"]).eq(year)]
        if train["paper_id"].nunique() < 10 or test.empty:
            continue
        rows.append(_score_fold(train, test, _permute_within_paper(train, int(year))))
    if not rows:
        raise ValueError("no strict forward-time development folds are available")
    return pd.concat(rows, ignore_index=True)


def _domain_predictions(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for index, domain in enumerate(sorted(frame["domain12"].astype(str).unique())):
        train = frame[frame["domain12"].astype(str).ne(domain)]
        test = frame[frame["domain12"].astype(str).eq(domain)]
        if train["paper_id"].nunique() < 10 or test.empty:
            continue
        rows.append(
            _score_fold(train, test, _permute_within_paper(train, seed + index))
        )
    if len(rows) < 2:
        raise ValueError(
            "fewer than two leave-domain-out development folds are available"
        )
    return pd.concat(rows, ignore_index=True)


def _permute_within_paper(frame: pd.DataFrame, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    output = pd.to_numeric(frame["future_claim_adoption_share"]).copy()
    for indices in frame.groupby("paper_id", observed=True).groups.values():
        values = output.loc[list(indices)].to_numpy(copy=True)
        rng.shuffle(values)
        output.loc[list(indices)] = values
    return output


def _score_fold(
    train: pd.DataFrame, test: pd.DataFrame, permuted: pd.Series
) -> pd.DataFrame:
    real = _fit_model(
        train[list(T0_FEATURE_NAMES)], train["future_claim_adoption_share"]
    )
    null = _fit_model(train[list(T0_FEATURE_NAMES)], permuted)
    output = test[["paper_id", "claim_id", "future_claim_adoption_share"]].copy()
    features = test[list(T0_FEATURE_NAMES)].to_numpy(dtype=float)
    output["prediction"] = _normalize_fold_scores(
        output["paper_id"], np.maximum(0.0, real.predict(features))
    )
    output["permutation_prediction"] = _normalize_fold_scores(
        output["paper_id"], np.maximum(0.0, null.predict(features))
    )
    return output


def _normalize_fold_scores(paper_ids: pd.Series, scores: np.ndarray) -> pd.Series:
    """Apply the same per-paper normalization used by the runtime head."""
    values = pd.Series(scores, index=paper_ids.index, dtype=float)
    totals = values.groupby(paper_ids).transform("sum")
    counts = values.groupby(paper_ids).transform("size")
    return values.div(totals.where(totals.gt(0.0))).fillna(1.0 / counts)


def _rho(target: pd.Series, prediction: pd.Series) -> float:
    value = float(spearmanr(target, prediction).statistic)
    return value if math.isfinite(value) else 0.0


def _bootstrap_cis(
    frame: pd.DataFrame, seed: int, replicates: int
) -> tuple[tuple[float, float], tuple[float, float]]:
    papers = frame["paper_id"].astype(str).unique()
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    paired_advantages: list[float] = []
    groups = {paper: frame[frame["paper_id"].astype(str).eq(paper)] for paper in papers}
    for _ in range(replicates):
        sampled = rng.choice(papers, size=len(papers), replace=True)
        boot = pd.concat([groups[str(paper)] for paper in sampled], ignore_index=True)
        real = _rho(boot["future_claim_adoption_share"], boot["prediction"])
        null = _rho(boot["future_claim_adoption_share"], boot["permutation_prediction"])
        estimates.append(real)
        paired_advantages.append(real - null)
    absolute_values = np.quantile(estimates, [0.025, 0.975])
    paired_values = np.quantile(paired_advantages, [0.025, 0.975])
    absolute = (float(absolute_values[0]), float(absolute_values[1]))
    paired = (float(paired_values[0]), float(paired_values[1]))
    return absolute, paired


def _gate_report(
    axis: str,
    predictions: pd.DataFrame,
    *,
    model_hash: str,
    seed: int,
    bootstrap_replicates: int,
    development: pd.DataFrame,
    attribution_development: pd.DataFrame,
    source_paths: tuple[Path, Path],
) -> dict[str, Any]:
    rho = _rho(predictions["future_claim_adoption_share"], predictions["prediction"])
    null_rho = _rho(
        predictions["future_claim_adoption_share"],
        predictions["permutation_prediction"],
    )
    absolute_ci, paired_advantage_ci = _bootstrap_cis(
        predictions, seed, bootstrap_replicates
    )
    advantage = rho - null_rho
    # Claim attribution is a within-paper allocation problem. Its registered
    # mechanism contrast is therefore the paired degradation caused by shuffling
    # labels within each paper, not an unpaired test of global outcome scale.
    # Preserve the original positive-signal threshold and add the paired
    # within-paper mechanism contrast; neither condition may compensate for
    # failure of the other.
    passed = absolute_ci[0] > 0.0 and paired_advantage_ci[0] > 0.0 and advantage > 0.0
    return {
        "contract": "gear_claim_attribution_gate1_report_v1",
        "status": "passed" if passed else "not_supported",
        "claim_allowed": passed,
        "metrics": {
            "spearman_rho": rho,
            "paper_cluster_bootstrap_ci95": list(absolute_ci),
            "within_paper_permutation_rho": null_rho,
            "advantage_over_permutation": advantage,
            "paired_advantage_bootstrap_ci95": list(paired_advantage_ci),
            "rows": len(predictions),
            "papers": int(predictions["paper_id"].nunique()),
            "bootstrap_replicates": bootstrap_replicates,
        },
        "claim_attribution_runtime_candidate": {
            "model_sha256": model_hash,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": list(T0_FEATURE_NAMES),
            "evaluation_axis": axis,
            "future_contexts_used_at_inference": False,
            "development_only": True,
            "training_split": "development",
            "sealed_holdout_labels_used": False,
            "holdout_labels_used_for_model_selection": False,
            "fold_local_target_fit": True,
            "future_features_used": False,
            "evaluation_protocol": (
                "strict_forward_publication_year"
                if axis == "temporal"
                else "leave_one_domain12_out"
            ),
            "development_papers": int(development["paper_id"].nunique()),
            "informative_development_papers": int(
                attribution_development["paper_id"].nunique()
            ),
            "target_definition": "within_paper_share_given_any_observed_adoption",
            "ridge_alpha": RIDGE_ALPHA,
            "gate_estimand": "paired_within_paper_permutation_advantage",
            "source_sha256": [_sha256(path) for path in source_paths],
        },
    }


def _write_replay(
    frame: pd.DataFrame, model: ClaimAttributionLinearHead, path: Path
) -> None:
    ordered = frame.sort_values(["paper_id", "claim_id"]).reset_index(drop=True)
    features = ordered[list(T0_FEATURE_NAMES)].to_numpy(dtype=float).tolist()
    raw = model.predict(features)
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for paper, score in zip(ordered["paper_id"].astype(str), raw, strict=True):
        totals[paper] = totals.get(paper, 0.0) + score
        counts[paper] = counts.get(paper, 0) + 1
    rows = []
    for (_, item), values, score in zip(ordered.iterrows(), features, raw, strict=True):
        paper = str(item["paper_id"])
        weight = score / totals[paper] if totals[paper] > 0 else 1.0 / counts[paper]
        rows.append(
            {
                "paper_id": paper,
                "claim_id": str(item["claim_id"]),
                "features": values,
                "expected_raw_score": score,
                "expected_attribution_weight": weight,
            }
        )
    path.write_text(json.dumps({"rows": rows}, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-gate1", type=Path, required=True)
    parser.add_argument("--domain-gate1", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    try:
        result = train_and_gate_claim_attribution(
            args.temporal_gate1,
            args.domain_gate1,
            args.output_dir,
            release_id=args.release_id,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    except (OSError, TypeError, ValueError) as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "contract": "gear_claim_attribution_promotion_blocked_v1",
            "status": "blocked",
            "reason": f"{type(exc).__name__}:{exc}",
            "confirmatory_holdout_labels_used": False,
        }
        (args.output_dir / "promotion_blocked.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.output_dir / "release.json").unlink(missing_ok=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "promoted" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FORMAL_SPLIT_COUNTS", "train_and_gate_claim_attribution"]
