"""Leakage-safe nested OOF evaluation for the Nature multi-horizon models."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .contracts import FeatureSpec, SplitSpec
from .models import (
    MODEL_COMPLEXITY,
    DomainYearCalibrator,
    FittedRankBlendModel,
    RankBlendEstimator,
    SimplexPairwiseRanker,
    TargetResidualizer,
    empirical_cdf,
    fit_candidate_model,
    percentile_rank,
    project_simplex,
)
from .splits import FoldIndices, NestedSplitPlan, make_nested_folds, split_sealed_holdout
from .targets import DIFFUSION_TARGET_COMPONENTS, FoldLocalDiffusionTarget


DEFAULT_CANDIDATES: Tuple[str, ...] = (
    "domain_year_only",
    "bibliographic_aux10_ridge",
    "mechanism5_equal_weight",
    "mechanism5_simplex",
    "gam18",
    "hgb18",
    "rank_blend",
)


@dataclass
class NestedOOFResult:
    """The four normalized tables consumed by releases and figure views."""

    oof_predictions: pd.DataFrame
    evaluation_metrics: pd.DataFrame
    model_ledger: pd.DataFrame
    holdout_predictions: pd.DataFrame
    summary: Dict[str, Any]
    split_plan: NestedSplitPlan


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return float("nan")
    return float(frame["left"].corr(frame["right"], method="spearman"))


def _year_bins(frame: pd.DataFrame, width: int) -> pd.Series:
    years = pd.to_numeric(frame.get("publication_year", pd.Series(np.nan, index=frame.index)), errors="coerce")
    return (np.floor(years / width) * width).astype("Int64").astype("string").fillna("missing")


def conditional_spearman(
    frame: pd.DataFrame,
    score_col: str,
    target_col: str,
    *,
    domain_col: str = "domain12",
    year_bin_width: int = 5,
    min_cell_size: int = 30,
) -> tuple[float, int, int]:
    """Spearman after ranking predictions and targets within domain-period."""

    work = frame.copy()
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work["_target"] = pd.to_numeric(work[target_col], errors="coerce")
    work["_domain"] = work.get(domain_col, pd.Series("unknown", index=work.index)).astype("string")
    work["_period"] = _year_bins(work, year_bin_width)
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["_score", "_target"])
    counts = work.groupby(["_domain", "_period"], observed=True)["_score"].transform("size")
    work = work.loc[counts >= int(min_cell_size)].copy()
    if work.empty:
        return float("nan"), 0, 0
    groups = [work["_domain"], work["_period"]]
    work["_score_rank"] = work["_score"].groupby(groups, observed=True).rank(method="average", pct=True)
    work["_target_rank"] = work["_target"].groupby(groups, observed=True).rank(method="average", pct=True)
    return (
        safe_spearman(work["_score_rank"], work["_target_rank"]),
        int(len(work)),
        int(work.groupby(["_domain", "_period"], observed=True).ngroups),
    )


def domain_metrics(
    frame: pd.DataFrame,
    score_col: str,
    target_col: str,
    *,
    domain_col: str = "domain12",
    min_domain_size: int = 50,
) -> tuple[pd.DataFrame, float, float]:
    rows: List[Dict[str, Any]] = []
    for domain, group in frame.groupby(domain_col, dropna=False, observed=True):
        valid = group[[score_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < int(min_domain_size):
            continue
        rho = safe_spearman(valid[score_col], valid[target_col])
        rows.append({"domain12": str(domain), "n": int(len(valid)), "spearman": rho})
    table = pd.DataFrame(rows)
    finite = table["spearman"].replace([np.inf, -np.inf], np.nan).dropna() if not table.empty else pd.Series(dtype=float)
    macro = float(finite.mean()) if len(finite) else float("nan")
    positive = float((finite > 0).mean()) if len(finite) else float("nan")
    return table, macro, positive


def top_decile_enrichment(score: Sequence[float], target: Sequence[float]) -> float:
    """Smoothed prevalence ratio for true top-decile papers in predicted top decile."""

    frame = pd.DataFrame({"score": score, "target": target}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 20:
        return float("nan")
    predicted_top = frame["score"].rank(method="average", pct=True) >= 0.90
    true_top = frame["target"].rank(method="average", pct=True) >= 0.90
    # Jeffreys smoothing avoids infinite headline values in small samples.
    observed = (float((predicted_top & true_top).sum()) + 0.5) / (float(predicted_top.sum()) + 1.0)
    prevalence = (float(true_top.sum()) + 0.5) / (float(len(frame)) + 1.0)
    return float(observed / prevalence) if prevalence > 0 else float("nan")


def evaluate_prediction_frame(
    frame: pd.DataFrame,
    *,
    raw_score_col: str = "prediction_uncalibrated",
    calibrated_score_col: str = "prediction_calibrated",
    target_col: str = "target_adjusted_oof",
    split_spec: Optional[SplitSpec] = None,
) -> Dict[str, float]:
    """Compute the fixed global, macro-domain, and conditional OOF metrics."""

    spec = split_spec or SplitSpec()
    if raw_score_col not in frame.columns and "prediction_raw" in frame.columns:
        raw_score_col = "prediction_raw"
    finite = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[raw_score_col, calibrated_score_col, target_col]
    )
    _, macro, positive = domain_metrics(
        finite,
        raw_score_col,
        target_col,
        min_domain_size=spec.min_domain_oof_size,
    )
    conditional, conditional_n, conditional_cells = conditional_spearman(
        finite,
        raw_score_col,
        target_col,
        year_bin_width=spec.year_bin_width,
        min_cell_size=spec.min_conditional_cell_size,
    )
    return {
        "rho_global_calibrated": safe_spearman(finite[calibrated_score_col], finite[target_col]),
        "rho_global_uncalibrated": safe_spearman(finite[raw_score_col], finite[target_col]),
        "rho_domain_macro": macro,
        "rho_conditional": conditional,
        "positive_domain_ratio": positive,
        "top_decile_enrichment": top_decile_enrichment(finite[calibrated_score_col], finite[target_col]),
        "n_finite_oof": float(len(finite)),
        "n_conditional_rows": float(conditional_n),
        "n_conditional_cells": float(conditional_cells),
    }


def _cluster_bootstrap_intervals(
    frame: pd.DataFrame,
    *,
    split_spec: SplitSpec,
    iterations: int,
    seed: int,
) -> Dict[str, Tuple[float, float]]:
    work = frame.copy().reset_index(drop=True)
    domains = work.get("domain12", pd.Series("unknown", index=work.index)).astype("string")
    clusters = (domains + "|" + _year_bins(work, split_spec.year_bin_width)).to_numpy()
    unique = np.unique(clusters)
    if len(unique) < 2 or iterations <= 0:
        return {}
    group_indices = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    samples: Dict[str, List[float]] = {}
    for _ in range(int(iterations)):
        selected = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([group_indices[value] for value in selected])
        metrics = evaluate_prediction_frame(work.iloc[indices], split_spec=split_spec)
        for key, value in metrics.items():
            if np.isfinite(value):
                samples.setdefault(key, []).append(float(value))
    return {
        key: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for key, values in samples.items()
        if len(values) >= max(20, iterations // 10)
    }


def paired_cluster_bootstrap_delta(
    selected: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    split_spec: SplitSpec,
    iterations: int,
    seed: int,
) -> Dict[str, float]:
    """Paired cluster-bootstrap CI for selected-minus-baseline Spearman."""

    keys = [column for column in ("paper_id", "_source_row") if column in selected and column in baseline]
    if not keys:
        raise ValueError("Paired comparison requires paper_id or _source_row")
    selected_keep = keys + [
        "prediction_calibrated",
        "target_adjusted_oof",
        "domain12",
        "publication_year",
    ]
    baseline_keep = keys + ["prediction_calibrated"]
    merged = selected[selected_keep].merge(
        baseline[baseline_keep],
        on=keys,
        how="inner",
        suffixes=("_selected", "_baseline"),
        validate="one_to_one",
    )
    observed = safe_spearman(merged["prediction_calibrated_selected"], merged["target_adjusted_oof"]) - safe_spearman(
        merged["prediction_calibrated_baseline"], merged["target_adjusted_oof"]
    )
    cluster = (
        merged["domain12"].astype("string").fillna("unknown")
        + "|"
        + _year_bins(merged, split_spec.year_bin_width)
    ).to_numpy()
    unique = np.unique(cluster)
    rng = np.random.default_rng(seed)
    groups = {value: np.flatnonzero(cluster == value) for value in unique}
    deltas: List[float] = []
    if len(unique) >= 2:
        for _ in range(int(iterations)):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            indices = np.concatenate([groups[value] for value in sampled])
            sample = merged.iloc[indices]
            left = safe_spearman(sample["prediction_calibrated_selected"], sample["target_adjusted_oof"])
            right = safe_spearman(sample["prediction_calibrated_baseline"], sample["target_adjusted_oof"])
            if np.isfinite(left) and np.isfinite(right):
                deltas.append(left - right)
    low, high = (float("nan"), float("nan"))
    if len(deltas) >= max(20, iterations // 10):
        low, high = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    return {
        "delta_global_calibrated": float(observed),
        "ci_low": low,
        "ci_high": high,
        "n_paired": int(len(merged)),
        "iterations": int(iterations),
    }


def _adjust_targets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    *,
    adjust: bool,
    future_citers_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if set(DIFFUSION_TARGET_COMPONENTS).issubset(train.columns) and set(
        DIFFUSION_TARGET_COMPONENTS
    ).issubset(test.columns):
        composer = FoldLocalDiffusionTarget().fit(train)
        train_y = composer.transform(train)
        test_y = composer.transform(test)
    else:
        # Synthetic/unit-test targets and registered alternative targets retain
        # their explicit scalar column. Production RGPM-D always takes the
        # fold-local component path above.
        train_y = pd.to_numeric(train[target_col], errors="coerce").to_numpy(float)
        test_y = pd.to_numeric(test[target_col], errors="coerce").to_numpy(float)
    if not adjust or future_citers_col not in train.columns:
        return percentile_rank(train_y), empirical_cdf(test_y, train_y), train_y, test_y
    residualizer = TargetResidualizer(future_citers_col=future_citers_col).fit(train, train_y)
    return (
        residualizer.transform(train, train_y),
        residualizer.transform(test, test_y),
        train_y,
        test_y,
    )


def _candidate_inner_predictions(
    development: pd.DataFrame,
    outer_fold: Any,
    candidate_ids: Sequence[str],
    target_col: str,
    *,
    feature_spec: FeatureSpec,
    seed: int,
    adjust_target: bool,
    future_citers_col: str,
) -> tuple[Dict[str, pd.DataFrame], Optional[np.ndarray]]:
    base_ids = tuple(model for model in candidate_ids if model != "rank_blend")
    rows: Dict[str, List[pd.DataFrame]] = {model: [] for model in base_ids}
    for inner in outer_fold.inner_folds:
        train = development.iloc[inner.train_idx].reset_index(drop=True)
        valid = development.iloc[inner.test_idx].reset_index(drop=True)
        train_target, valid_target, _, _ = _adjust_targets(
            train,
            valid,
            target_col,
            adjust=adjust_target,
            future_citers_col=future_citers_col,
        )
        for model_offset, model_id in enumerate(base_ids):
            model = fit_candidate_model(
                model_id,
                train,
                train_target,
                feature_spec=feature_spec,
                seed=seed + outer_fold.fold_id * 10_007 + inner.fold_id * 101 + model_offset,
            )
            raw_train = model.predict(train)
            raw_valid = model.predict(valid)
            calibrator = DomainYearCalibrator().fit(train, raw_train, train_target)
            calibrated = calibrator.predict(valid, raw_valid)
            part = valid[[column for column in ("paper_id", "domain12", "publication_year") if column in valid]].copy()
            part["target"] = valid_target
            part["raw_native"] = raw_valid
            part["raw"] = empirical_cdf(raw_valid, raw_train)
            part["calibrated"] = calibrated
            rows[model_id].append(part)
    predictions = {
        model: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for model, parts in rows.items()
    }
    blend_weights: Optional[np.ndarray] = None
    if "rank_blend" in candidate_ids:
        required = ("mechanism5_simplex", "gam18", "hgb18")
        if all(name in predictions and len(predictions[name]) for name in required):
            base = predictions[required[0]].copy()
            matrix = np.column_stack(
                [percentile_rank(predictions[name]["raw"].to_numpy(float)) for name in required]
            )
            optimizer = SimplexPairwiseRanker(seed=seed + outer_fold.fold_id * 97).fit(
                matrix,
                base["target"].to_numpy(float),
            )
            blend_weights = project_simplex(optimizer.weights_)
            base["raw"] = matrix @ blend_weights
            calibrated_matrix = np.column_stack(
                [percentile_rank(predictions[name]["calibrated"].to_numpy(float)) for name in required]
            )
            base["calibrated"] = calibrated_matrix @ blend_weights
            predictions["rank_blend"] = base
    return predictions, blend_weights


def _select_candidate(
    metrics: Mapping[str, Mapping[str, float]],
    candidate_ids: Sequence[str],
) -> str:
    reference = metrics.get("mechanism5_simplex", {})
    eligible: List[str] = []
    selector_whitelist = {"gam18", "hgb18", "rank_blend"}
    for model_id in candidate_ids:
        if model_id not in selector_whitelist:
            continue
        values = metrics.get(model_id, {})
        global_rho = values.get("rho_global_calibrated", float("nan"))
        if not np.isfinite(global_rho):
            continue
        conditional = values.get("rho_conditional", float("nan"))
        macro = values.get("rho_domain_macro", float("nan"))
        positive = values.get("positive_domain_ratio", float("nan"))
        ref_conditional = reference.get("rho_conditional", float("nan"))
        ref_macro = reference.get("rho_domain_macro", float("nan"))
        if np.isfinite(ref_conditional) and np.isfinite(conditional) and conditional < ref_conditional - 0.02:
            continue
        if np.isfinite(ref_macro) and np.isfinite(macro) and macro < ref_macro - 0.02:
            continue
        if np.isfinite(positive) and positive < 0.75:
            continue
        eligible.append(model_id)
    if not eligible:
        eligible = [
            model
            for model in candidate_ids
            if model in selector_whitelist
            and np.isfinite(metrics.get(model, {}).get("rho_global_calibrated", np.nan))
        ]
    if not eligible:
        raise RuntimeError("No candidate produced finite inner-fold predictions")
    best = eligible[0]
    for model in eligible[1:]:
        current = metrics[model]["rho_global_calibrated"]
        incumbent = metrics[best]["rho_global_calibrated"]
        if current > incumbent + 0.005:
            best = model
        elif abs(current - incumbent) <= 0.005 and MODEL_COMPLEXITY.get(model, 99) < MODEL_COMPLEXITY.get(best, 99):
            best = model
    return best


def _select_on_full_development(
    development: pd.DataFrame,
    candidate_ids: Sequence[str],
    target_col: str,
    *,
    feature_spec: FeatureSpec,
    split_spec: SplitSpec,
    adjust_target: bool,
    future_citers_col: str,
) -> tuple[str, Dict[str, Dict[str, float]], Optional[np.ndarray]]:
    """Run the locked inner selection once on the complete development set."""

    selection_plan = make_nested_folds(
        development,
        n_outer=split_spec.inner_folds,
        n_inner=2,
        seed=split_spec.seed + 404_003,
        year_bin_width=split_spec.year_bin_width,
    )

    class SelectionFold:
        fold_id = 0
        inner_folds = tuple(
            FoldIndices(
                fold_id=fold.fold_id,
                train_idx=fold.train_idx,
                test_idx=fold.test_idx,
            )
            for fold in selection_plan.outer_folds
        )

    predictions, blend_weights = _candidate_inner_predictions(
        development,
        SelectionFold(),
        candidate_ids,
        target_col,
        feature_spec=feature_spec,
        seed=split_spec.seed + 505_007,
        adjust_target=adjust_target,
        future_citers_col=future_citers_col,
    )
    metrics: Dict[str, Dict[str, float]] = {}
    for model_id, prediction in predictions.items():
        if prediction.empty:
            continue
        renamed = prediction.rename(
            columns={
                "raw": "prediction_uncalibrated",
                "raw_native": "prediction_raw",
                "calibrated": "prediction_calibrated",
                "target": "target_adjusted_oof",
            }
        )
        metrics[model_id] = evaluate_prediction_frame(
            renamed, split_spec=split_spec
        )
    return _select_candidate(metrics, candidate_ids), metrics, blend_weights


def _metric_rows(
    metrics: Mapping[str, float],
    *,
    horizon: int,
    model_id: str,
    scope: str,
    intervals: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> List[Dict[str, Any]]:
    intervals = intervals or {}
    rows = []
    for metric, value in metrics.items():
        low, high = intervals.get(metric, (float("nan"), float("nan")))
        rows.append(
            {
                "horizon": int(horizon),
                "model_id": model_id,
                "scope": scope,
                "metric": metric,
                "value": float(value),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return rows


def run_nested_oof(
    frame: pd.DataFrame,
    *,
    horizon: int,
    target_col: str,
    feature_spec: Optional[FeatureSpec] = None,
    split_spec: Optional[SplitSpec] = None,
    candidate_ids: Sequence[str] = DEFAULT_CANDIDATES,
    future_citers_col: str = "n_future_citers",
    adjust_target: bool = True,
    run_holdout: bool = True,
    bootstrap_iterations: Optional[int] = None,
    outer_folds: Optional[int] = None,
    inner_folds: Optional[int] = None,
    seed: Optional[int] = None,
) -> NestedOOFResult:
    """Run nested OOF selection and an optional predeclared temporal holdout."""

    features = feature_spec or FeatureSpec()
    splits = split_spec or SplitSpec()
    updates: Dict[str, Any] = {}
    if outer_folds is not None:
        updates["outer_folds"] = int(outer_folds)
    if inner_folds is not None:
        updates["inner_folds"] = int(inner_folds)
    if seed is not None:
        updates["seed"] = int(seed)
    if updates:
        splits = splits.model_copy(update=updates)
    required = {"paper_id", target_col, "domain12", "publication_year", "venue_family"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Nested OOF input is missing required columns: {missing}")
    data = frame.copy().reset_index(drop=True)
    data["_source_row"] = np.arange(len(data), dtype=int)
    holdout_split = split_sealed_holdout(
        data,
        horizon,
        holdout_years=splits.sealed_holdout_years,
    )
    development = data.iloc[holdout_split.development_idx].reset_index(drop=True)
    if len(development) < splits.outer_folds * splits.inner_folds:
        raise ValueError("Development cohort is too small for requested nested folds")
    plan = make_nested_folds(
        development,
        n_outer=splits.outer_folds,
        n_inner=splits.inner_folds,
        seed=splits.seed,
        year_bin_width=splits.year_bin_width,
    )
    oof_parts: List[pd.DataFrame] = []
    ledger_rows: List[Dict[str, Any]] = []
    selected_models: List[str] = []

    for outer in plan.outer_folds:
        inner_predictions, blend_weights = _candidate_inner_predictions(
            development,
            outer,
            candidate_ids,
            target_col,
            feature_spec=features,
            seed=splits.seed,
            adjust_target=adjust_target,
            future_citers_col=future_citers_col,
        )
        inner_metrics: Dict[str, Dict[str, float]] = {}
        for model_id, prediction in inner_predictions.items():
            if prediction.empty:
                continue
            renamed = prediction.rename(
                columns={
                    "raw": "prediction_raw",
                    "calibrated": "prediction_calibrated",
                    "target": "target_adjusted_oof",
                }
            )
            values = evaluate_prediction_frame(renamed, split_spec=splits)
            inner_metrics[model_id] = values
        selected = _select_candidate(inner_metrics, candidate_ids)
        selected_models.append(selected)
        for model_id, values in inner_metrics.items():
            ledger_rows.append(
                {
                    "horizon": int(horizon),
                    "outer_fold": int(outer.fold_id),
                    "candidate_id": model_id,
                    "selected": bool(model_id == selected),
                    "inner_fold_count": int(len(outer.inner_folds)),
                    "selection_seed": int(splits.seed),
                    **values,
                }
            )

        train = development.iloc[outer.train_idx].reset_index(drop=True)
        test = development.iloc[outer.test_idx].reset_index(drop=True)
        train_target, test_target, _, test_target_raw = _adjust_targets(
            train,
            test,
            target_col,
            adjust=adjust_target,
            future_citers_col=future_citers_col,
        )
        mechanism_model = fit_candidate_model(
            "mechanism5_simplex",
            train,
            train_target,
            feature_spec=features,
            seed=splits.seed + outer.fold_id * 1009,
        )
        channels = mechanism_model.mechanism_channels(test)
        mechanism_score = mechanism_model.predict(test)
        outer_models: Dict[str, Any] = {}
        ordered_candidate_ids = sorted(candidate_ids, key=lambda value: value == "rank_blend")
        for model_offset, model_id in enumerate(ordered_candidate_ids):
            if model_id == "rank_blend" and blend_weights is None:
                continue
            if model_id == "mechanism5_simplex":
                fitted = mechanism_model
                fit_seed = splits.seed + outer.fold_id * 1009
            elif model_id == "rank_blend":
                base_names = ("mechanism5_simplex", "gam18", "hgb18")
                if not all(name in outer_models for name in base_names):
                    raise RuntimeError("Rank blend requires fitted Simplex, GAM, and HGB bases")
                base_models = {name: outer_models[name] for name in base_names}
                references = {name: model.predict(train) for name, model in base_models.items()}
                fitted = FittedRankBlendModel(
                    "rank_blend",
                    None,
                    RankBlendEstimator(base_models, blend_weights, references),
                    "blend",
                    features,
                )
                fit_seed = splits.seed + outer.fold_id * 2003 + model_offset
            else:
                fit_seed = splits.seed + outer.fold_id * 2003 + model_offset
                fitted = fit_candidate_model(
                    model_id,
                    train,
                    train_target,
                    feature_spec=features,
                    seed=fit_seed,
                )
            outer_models[model_id] = fitted
            weights = getattr(fitted.estimator, "weights_", None)
            for ledger_row in reversed(ledger_rows):
                if (
                    ledger_row["outer_fold"] == int(outer.fold_id)
                    and ledger_row["candidate_id"] == model_id
                ):
                    ledger_row["outer_fit_seed"] = int(fit_seed)
                    ledger_row["outer_train_n"] = int(len(train))
                    ledger_row["outer_test_n"] = int(len(test))
                    ledger_row["outer_weights_json"] = (
                        json.dumps([float(value) for value in np.asarray(weights)])
                        if weights is not None
                        else ""
                    )
                    ledger_row["selection_reason"] = (
                        "highest_inner_global_after_guardrails"
                        if model_id == selected
                        else "not_selected"
                    )
                    break
            raw_train = fitted.predict(train)
            raw_test = fitted.predict(test)
            calibrator = DomainYearCalibrator().fit(train, raw_train, train_target)
            for ledger_row in reversed(ledger_rows):
                if (
                    ledger_row["outer_fold"] == int(outer.fold_id)
                    and ledger_row["candidate_id"] == model_id
                ):
                    ledger_row["calibration_domain_count"] = int(
                        len(calibrator.domain_corrections_)
                    )
                    ledger_row["calibration_reference_n"] = int(
                        len(calibrator.calibrated_reference_)
                    )
                    break
            calibrated_test = calibrator.predict(test, raw_test)
            part = test[
                [
                    column
                    for column in (
                        "paper_id",
                        "domain12",
                        "publication_year",
                        "venue_family",
                        "n_future_citers",
                        "common_cohort_member",
                        "high_quality_cohort_member",
                        "uncapped_cohort_member",
                        "cap_hit",
                        "_source_row",
                    )
                    if column in test
                ]
            ].copy()
            part["horizon"] = int(horizon)
            part["outer_fold"] = int(outer.fold_id)
            part["model_id"] = model_id
            part["is_selected"] = bool(model_id == selected)
            part["target_raw"] = test_target_raw
            part["target_adjusted_oof"] = test_target
            part["score_mechanism"] = mechanism_score
            part["prediction_raw"] = raw_test
            part["prediction_uncalibrated"] = empirical_cdf(raw_test, raw_train)
            part["prediction_calibrated"] = calibrated_test
            part["prediction_percentile"] = empirical_cdf(
                calibrated_test,
                calibrator.calibrated_reference_,
            )
            for channel in channels:
                part[f"mechanism__{channel}"] = channels[channel].to_numpy(float)
            oof_parts.append(part)

    oof = pd.concat(oof_parts, ignore_index=True).sort_values(["_source_row", "model_id"]).reset_index(drop=True)
    selected_oof = oof.loc[oof["is_selected"]].copy()
    if selected_oof["paper_id"].duplicated().any():
        raise RuntimeError("Selector emitted multiple OOF predictions for a paper")
    point_metrics = evaluate_prediction_frame(selected_oof, split_spec=splits)
    iterations = splits.bootstrap_iterations if bootstrap_iterations is None else int(bootstrap_iterations)
    intervals = _cluster_bootstrap_intervals(
        selected_oof,
        split_spec=splits,
        iterations=iterations,
        seed=splits.seed + int(horizon) * 7919,
    )
    metric_rows = _metric_rows(
        point_metrics,
        horizon=horizon,
        model_id="nested_selector",
        scope="development_oof",
        intervals=intervals,
    )
    aggregate_model_metrics: Dict[str, Dict[str, float]] = {}
    for model_id, model_oof in oof.groupby("model_id", observed=True):
        values = evaluate_prediction_frame(model_oof, split_spec=splits)
        aggregate_model_metrics[str(model_id)] = values
        metric_rows.extend(
            _metric_rows(
                values,
                horizon=horizon,
                model_id=str(model_id),
                scope="development_oof_all_models",
            )
        )

    final_model_id, full_selection_metrics, final_blend_weights = (
        _select_on_full_development(
            development,
            candidate_ids,
            target_col,
            feature_spec=features,
            split_spec=splits,
            adjust_target=adjust_target,
            future_citers_col=future_citers_col,
        )
    )
    for model_id, values in full_selection_metrics.items():
        ledger_rows.append(
            {
                "horizon": int(horizon),
                "outer_fold": 0,
                "candidate_id": model_id,
                "selected": bool(model_id == final_model_id),
                "inner_fold_count": int(splits.inner_folds),
                "selection_seed": int(splits.seed + 404_003),
                "selection_reason": (
                    "locked_full_development_inner_selection"
                    if model_id == final_model_id
                    else "not_selected_full_development"
                ),
                **values,
            }
        )

    baseline_ids = (
        "domain_year_only",
        "bibliographic_aux10_ridge",
        "mechanism5_equal_weight",
        "mechanism5_simplex",
    )
    available_baselines = [
        model_id
        for model_id in baseline_ids
        if np.isfinite(aggregate_model_metrics.get(model_id, {}).get("rho_global_calibrated", np.nan))
    ]
    upgrade_gate: Dict[str, Any] = {
        "pass": False,
        "reason": "no_fixed_baseline_available",
    }
    if available_baselines:
        strongest_baseline = max(
            available_baselines,
            key=lambda model_id: aggregate_model_metrics[model_id]["rho_global_calibrated"],
        )
        comparison = paired_cluster_bootstrap_delta(
            selected_oof,
            oof.loc[oof["model_id"].eq(strongest_baseline)],
            split_spec=splits,
            iterations=iterations,
            seed=splits.seed + int(horizon) * 6151,
        )
        simplex_metrics = aggregate_model_metrics.get("mechanism5_simplex", {})
        conditional_ok = bool(
            not np.isfinite(simplex_metrics.get("rho_conditional", np.nan))
            or point_metrics.get("rho_conditional", -np.inf)
            >= simplex_metrics["rho_conditional"] - 0.02
        )
        macro_ok = bool(
            not np.isfinite(simplex_metrics.get("rho_domain_macro", np.nan))
            or point_metrics.get("rho_domain_macro", -np.inf)
            >= simplex_metrics["rho_domain_macro"] - 0.02
        )
        positive_domain_ok = bool(
            np.isfinite(point_metrics.get("positive_domain_ratio", np.nan))
            and point_metrics["positive_domain_ratio"] >= 0.75
        )
        upgrade_gate = {
            "strongest_baseline": strongest_baseline,
            **comparison,
            "required_delta": 0.03,
            "pass": bool(
                comparison["delta_global_calibrated"] >= 0.03
                and np.isfinite(comparison["ci_low"])
                and comparison["ci_low"] > 0.0
                and conditional_ok
                and macro_ok
                and positive_domain_ok
            ),
            "conditional_noninferiority": conditional_ok,
            "domain_macro_noninferiority": macro_ok,
            "positive_domain_ratio_gate": positive_domain_ok,
            "reason": "paired_cluster_bootstrap",
        }
        metric_rows.extend(
            [
                {
                    "horizon": int(horizon),
                    "model_id": "nested_selector",
                    "scope": "upgrade_gate",
                    "metric": "delta_vs_strongest_new_baseline",
                    "value": float(comparison["delta_global_calibrated"]),
                    "ci_low": float(comparison["ci_low"]),
                    "ci_high": float(comparison["ci_high"]),
                },
                {
                    "horizon": int(horizon),
                    "model_id": "nested_selector",
                    "scope": "upgrade_gate",
                    "metric": "pass_delta_0_03_ci_low_positive",
                    "value": float(upgrade_gate["pass"]),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                },
                {
                    "horizon": int(horizon),
                    "model_id": "nested_selector",
                    "scope": "upgrade_gate",
                    "metric": "conditional_noninferiority_vs_simplex",
                    "value": float(conditional_ok),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                },
                {
                    "horizon": int(horizon),
                    "model_id": "nested_selector",
                    "scope": "upgrade_gate",
                    "metric": "domain_macro_noninferiority_vs_simplex",
                    "value": float(macro_ok),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                },
                {
                    "horizon": int(horizon),
                    "model_id": "nested_selector",
                    "scope": "upgrade_gate",
                    "metric": "positive_domain_ratio_at_least_075",
                    "value": float(positive_domain_ok),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                },
            ]
        )

    holdout_predictions = pd.DataFrame()
    if run_holdout and len(holdout_split.holdout_idx):
        holdout = data.iloc[holdout_split.holdout_idx].reset_index(drop=True)
        full_target, holdout_target, _, holdout_target_raw = _adjust_targets(
            development,
            holdout,
            target_col,
            adjust=adjust_target,
            future_citers_col=future_citers_col,
        )
        final_model = fit_candidate_model(
            final_model_id,
            development,
            full_target,
            feature_spec=features,
            seed=splits.seed + 77_777,
            blend_weights=(
                final_blend_weights
                if final_model_id == "rank_blend"
                else None
            ),
        )
        final_mechanism = fit_candidate_model(
            "mechanism5_simplex",
            development,
            full_target,
            feature_spec=features,
            seed=splits.seed + 88_888,
        )
        raw_development = final_model.predict(development)
        raw_holdout = final_model.predict(holdout)
        calibrator = DomainYearCalibrator().fit(development, raw_development, full_target)
        holdout_predictions = holdout[
            [
                column
                for column in (
                    "paper_id",
                    "domain12",
                    "publication_year",
                    "venue_family",
                    "n_future_citers",
                    "common_cohort_member",
                    "high_quality_cohort_member",
                    "uncapped_cohort_member",
                    "cap_hit",
                    "_source_row",
                )
                if column in holdout
            ]
        ].copy()
        holdout_predictions["horizon"] = int(horizon)
        holdout_predictions["target_raw"] = holdout_target_raw
        holdout_predictions["target_adjusted_oof"] = holdout_target
        holdout_predictions["score_mechanism"] = final_mechanism.predict(holdout)
        holdout_predictions["prediction_raw"] = raw_holdout
        holdout_predictions["prediction_uncalibrated"] = empirical_cdf(
            raw_holdout, raw_development
        )
        holdout_predictions["prediction_calibrated"] = calibrator.predict(holdout, raw_holdout)
        holdout_predictions["prediction_percentile"] = empirical_cdf(
            holdout_predictions["prediction_calibrated"],
            calibrator.calibrated_reference_,
        )
        holdout_predictions["model_id"] = final_model_id
        holdout_predictions["is_selected"] = True
        channels = final_mechanism.mechanism_channels(holdout)
        for channel in channels:
            holdout_predictions[f"mechanism__{channel}"] = channels[channel].to_numpy(float)
        holdout_metrics = evaluate_prediction_frame(holdout_predictions, split_spec=splits)
        holdout_intervals = _cluster_bootstrap_intervals(
            holdout_predictions,
            split_spec=splits,
            iterations=iterations,
            seed=splits.seed + int(horizon) * 104_729,
        )
        metric_rows.extend(
            _metric_rows(
                holdout_metrics,
                horizon=horizon,
                model_id=final_model_id,
                scope="sealed_temporal_holdout",
                intervals=holdout_intervals,
            )
        )

    summary = {
        "horizon": int(horizon),
        "target_col": target_col,
        "n_input": int(len(data)),
        "n_development": int(len(development)),
        "n_oof_finite": int(point_metrics["n_finite_oof"]),
        "n_holdout": int(len(holdout_predictions)),
        "selected_model_counts": {str(k): int(v) for k, v in pd.Series(selected_models).value_counts().items()},
        "final_model_id": final_model_id,
        "final_selection_protocol": "locked_full_development_inner_cv",
        "metrics": point_metrics,
        "all_model_metrics": aggregate_model_metrics,
        "upgrade_gate": upgrade_gate,
        "split_audit": dict(plan.audit),
    }
    return NestedOOFResult(
        oof_predictions=oof,
        evaluation_metrics=pd.DataFrame(metric_rows),
        model_ledger=pd.DataFrame(ledger_rows),
        holdout_predictions=holdout_predictions,
        summary=summary,
        split_plan=plan,
    )


def run_multi_horizon_oof(
    frame: pd.DataFrame,
    target_columns: Mapping[int, str],
    **kwargs: Any,
) -> Dict[int, NestedOOFResult]:
    """Run independent nested models for every configured horizon."""

    horizon_col = kwargs.pop("horizon_col", "horizon")
    results: Dict[int, NestedOOFResult] = {}
    for horizon, target_col in sorted(target_columns.items()):
        subset = frame.loc[frame[horizon_col].eq(horizon)].copy() if horizon_col in frame else frame.copy()
        results[int(horizon)] = run_nested_oof(
            subset,
            horizon=int(horizon),
            target_col=target_col,
            **kwargs,
        )
    return results


def evaluate_oof_predictions(
    oof_predictions: pd.DataFrame,
    *,
    bootstrap_iterations: int = 2_000,
    seed: int = 20260710,
    split_spec: Optional[SplitSpec] = None,
) -> pd.DataFrame:
    """Recompute release metrics from the normalized long OOF table."""

    spec = split_spec or SplitSpec()
    spec = spec.model_copy(update={"seed": int(seed)})
    rows: List[Dict[str, Any]] = []
    for horizon, horizon_frame in oof_predictions.groupby("horizon", observed=True):
        aggregate: Dict[str, Dict[str, float]] = {}
        for model_id, model_frame in horizon_frame.groupby("model_id", observed=True):
            values = evaluate_prediction_frame(model_frame, split_spec=spec)
            aggregate[str(model_id)] = values
            rows.extend(
                _metric_rows(
                    values,
                    horizon=int(horizon),
                    model_id=str(model_id),
                    scope="development_oof_all_models",
                )
            )
        if "is_selected" in horizon_frame:
            selected = horizon_frame.loc[horizon_frame["is_selected"].fillna(False).astype(bool)]
            values = evaluate_prediction_frame(selected, split_spec=spec)
            intervals = _cluster_bootstrap_intervals(
                selected,
                split_spec=spec,
                iterations=int(bootstrap_iterations),
                seed=int(seed) + int(horizon) * 7919,
            )
            rows.extend(
                _metric_rows(
                    values,
                    horizon=int(horizon),
                    model_id="nested_selector",
                    scope="development_oof",
                    intervals=intervals,
                )
            )
            if "n_future_citers" in selected:
                count_rho = safe_spearman(
                    selected["target_adjusted_oof"],
                    np.log1p(
                        pd.to_numeric(
                            selected["n_future_citers"], errors="coerce"
                        )
                    ),
                )
                rows.append(
                    {
                        "horizon": int(horizon),
                        "model_id": "target_adjustment",
                        "scope": "target_adjustment_diagnostic",
                        "metric": "rho_adjusted_target_vs_log_future_citers",
                        "value": float(count_rho),
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                        "sensitivity": "count_only_spline",
                    }
                )
            if "n_future_citers" in selected:
                for threshold in (10, 20, 50):
                    sensitivity = selected[
                        pd.to_numeric(selected["n_future_citers"], errors="coerce")
                        >= threshold
                    ]
                    if len(sensitivity) < 3:
                        continue
                    sensitivity_rows = _metric_rows(
                        evaluate_prediction_frame(sensitivity, split_spec=spec),
                        horizon=int(horizon),
                        model_id="nested_selector",
                        scope="sensitivity_citation_threshold",
                    )
                    for row in sensitivity_rows:
                        row["sensitivity"] = f"future_citers_ge_{threshold}"
                    rows.extend(sensitivity_rows)
            for flag, scope in (
                ("common_cohort_member", "sensitivity_common_cohort"),
                (
                    "high_quality_cohort_member",
                    "sensitivity_high_quality_reference",
                ),
                (
                    "uncapped_cohort_member",
                    "sensitivity_uncapped_future_citers",
                ),
            ):
                if flag not in selected:
                    continue
                sensitivity = selected[
                    selected[flag].fillna(False).astype(bool)
                ]
                if len(sensitivity) < 3:
                    continue
                sensitivity_rows = _metric_rows(
                    evaluate_prediction_frame(sensitivity, split_spec=spec),
                    horizon=int(horizon),
                    model_id="nested_selector",
                    scope=scope,
                )
                for row in sensitivity_rows:
                    row["sensitivity"] = flag
                rows.extend(sensitivity_rows)
            if "outer_fold" in selected:
                for fold_id, fold_frame in selected.groupby("outer_fold", observed=True):
                    fold_rows = _metric_rows(
                        evaluate_prediction_frame(fold_frame, split_spec=spec),
                        horizon=int(horizon),
                        model_id="nested_selector",
                        scope="fold_stability",
                    )
                    for row in fold_rows:
                        row["sensitivity"] = f"outer_fold_{int(fold_id)}"
                    rows.extend(fold_rows)
            baseline_ids = (
                "domain_year_only",
                "bibliographic_aux10_ridge",
                "mechanism5_equal_weight",
                "mechanism5_simplex",
            )
            available = [
                model_id
                for model_id in baseline_ids
                if np.isfinite(aggregate.get(model_id, {}).get("rho_global_calibrated", np.nan))
            ]
            if available:
                strongest = max(
                    available,
                    key=lambda model_id: aggregate[model_id]["rho_global_calibrated"],
                )
                comparison = paired_cluster_bootstrap_delta(
                    selected,
                    horizon_frame.loc[horizon_frame["model_id"].eq(strongest)],
                    split_spec=spec,
                    iterations=int(bootstrap_iterations),
                    seed=int(seed) + int(horizon) * 6151,
                )
                simplex = aggregate.get("mechanism5_simplex", {})
                conditional_ok = bool(
                    not np.isfinite(simplex.get("rho_conditional", np.nan))
                    or values.get("rho_conditional", -np.inf)
                    >= simplex["rho_conditional"] - 0.02
                )
                macro_ok = bool(
                    not np.isfinite(simplex.get("rho_domain_macro", np.nan))
                    or values.get("rho_domain_macro", -np.inf)
                    >= simplex["rho_domain_macro"] - 0.02
                )
                positive_ok = bool(
                    np.isfinite(values.get("positive_domain_ratio", np.nan))
                    and values["positive_domain_ratio"] >= 0.75
                )
                passed = bool(
                    comparison["delta_global_calibrated"] >= 0.03
                    and np.isfinite(comparison["ci_low"])
                    and comparison["ci_low"] > 0
                    and conditional_ok
                    and macro_ok
                    and positive_ok
                )
                rows.extend(
                    [
                        {
                            "horizon": int(horizon),
                            "model_id": "nested_selector",
                            "scope": "upgrade_gate",
                            "metric": "delta_vs_strongest_new_baseline",
                            "value": comparison["delta_global_calibrated"],
                            "ci_low": comparison["ci_low"],
                            "ci_high": comparison["ci_high"],
                        },
                        {
                            "horizon": int(horizon),
                            "model_id": "nested_selector",
                            "scope": "upgrade_gate",
                            "metric": "pass_delta_0_03_ci_low_positive",
                            "value": float(passed),
                            "ci_low": float("nan"),
                            "ci_high": float("nan"),
                        },
                        {
                            "horizon": int(horizon),
                            "model_id": "nested_selector",
                            "scope": "upgrade_gate",
                            "metric": "conditional_noninferiority_vs_simplex",
                            "value": float(conditional_ok),
                            "ci_low": float("nan"),
                            "ci_high": float("nan"),
                        },
                        {
                            "horizon": int(horizon),
                            "model_id": "nested_selector",
                            "scope": "upgrade_gate",
                            "metric": "domain_macro_noninferiority_vs_simplex",
                            "value": float(macro_ok),
                            "ci_low": float("nan"),
                            "ci_high": float("nan"),
                        },
                        {
                            "horizon": int(horizon),
                            "model_id": "nested_selector",
                            "scope": "upgrade_gate",
                            "metric": "positive_domain_ratio_at_least_075",
                            "value": float(positive_ok),
                            "ci_low": float("nan"),
                            "ci_high": float("nan"),
                        },
                    ]
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        if "sensitivity" not in result:
            result["sensitivity"] = "main"
        else:
            result["sensitivity"] = result["sensitivity"].fillna("main").astype(str)
    return result


# Public naming aliases used by release builders.
evaluate_oof = evaluate_prediction_frame
run_nested_cv = run_nested_oof
