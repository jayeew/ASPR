"""Build and promote the real Primary16 D-excess/P structural-head sidecar.

The promotion is deliberately fail closed: candidate artifacts and a validation
report are always written, but ``manifest.json`` is written only when both the
latest temporal block and leave-one-domain-out scientific gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from gear.diffusion_forecast import (
    ForecastRelease,
    RuntimeFeatureRelease,
    sha256_file,
)
from gear.nature_multihorizon.perturbation_targets import (
    build_perturbation_components,
)
from gear.nature_multihorizon.targets_v6 import (
    PERTURBATION_COMPONENTS,
    FoldLocalExcessDiffusionTransformer,
    FoldLocalPerturbationTargetTransformer,
)
from gear.nature_multihorizon.taxonomy import assign_domain12

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARENT_MANIFEST = PROJECT_ROOT / (
    "data/calibration/releases/gear-d5-primary16-current/release_manifest.json"
)
RAW_INPUTS = PROJECT_ROOT / (
    "outputs/gear/stage_b_targeted_expansion_20260828/real_perturbation_481/"
    "real_perturbation_inputs.parquet"
)
COHORT_PREDICTIONS = PROJECT_ROOT / (
    "outputs/gear/stage_b_targeted_expansion_20260828/hgb_p_validation_241/"
    "hgb_p_oof_predictions.parquet"
)
OUTPUT_DIR = PROJECT_ROOT / (
    "data/calibration/graph_calibration/gear_structural_head_release_v1"
)
RUNTIME_MANIFEST = PROJECT_ROOT / (
    "data/calibration/runtime_features/gear-d5-primary16-dev10-v1/"
    "runtime_manifest.json"
)
STAGE_B_MANIFEST = PROJECT_ROOT / (
    "outputs/gear/stage_b_targeted_expansion_20260828/"
    "complete_graph_benchmark_manifest.json"
)
STAGE_C_MANIFEST = PROJECT_ROOT / (
    "outputs/gear/stage_c_randomized_actions_20260828/randomized_manifest_150.json"
)
SEED = 20260828


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _load_frame(
    parent: ForecastRelease, raw_path: Path, cohort_path: Path
) -> tuple[pd.DataFrame, list[str], list[str]]:
    raw = pd.read_parquet(raw_path)
    cohort = pd.read_parquet(cohort_path, columns=["paper_id"])
    cohort_ids = set(cohort["paper_id"].astype(str))
    raw = raw.loc[raw["paper_id"].astype(str).isin(cohort_ids)].reset_index(drop=True)
    components = build_perturbation_components(raw)
    base = pd.concat([raw, components], axis=1)
    base = base.loc[:, ~base.columns.duplicated()]
    # The parent OOF release is authoritative for folds, domains, and years.
    base = base.drop(
        columns=["outer_fold_id", "domain12", "publication_year"], errors="ignore"
    )
    oof = pd.read_parquet(
        parent.path("oof_predictions"),
        columns=[
            "paper_id",
            "outer_fold_id",
            "domain12",
            "publication_year",
            "realized_diffusion_target",
        ],
    )
    features = pd.read_parquet(parent.path("training_snapshot"))
    registry = json.loads(parent.path("feature_registry").read_text(encoding="utf-8"))
    feature_names = [str(value) for value in registry["feature_names"]]
    categorical = [str(value) for value in registry["categorical_feature_names"]]
    frame = (
        base.merge(oof, on="paper_id", how="inner", validate="one_to_one")
        .merge(
            features[["paper_id", *feature_names]],
            on="paper_id",
            how="inner",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )
    frame["rgpm_d_fold"] = pd.to_numeric(
        frame["realized_diffusion_target"], errors="coerce"
    )
    frame["n_future_citers"] = pd.to_numeric(
        frame["future_citer_count"], errors="coerce"
    )
    frame["opportunity_score"] = 0.0
    required = {"paper_id", *feature_names, *PERTURBATION_COMPONENTS}
    if len(frame) != 241 or frame["paper_id"].nunique() != 241:
        raise ValueError("the structural-head cohort must resolve exactly 241 papers")
    if set(frame["paper_id"].astype(str)) != cohort_ids:
        raise ValueError("raw inputs, Primary16 OOF metadata, and cohort do not align")
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"structural-head training frame lacks columns: {missing}")
    return frame, feature_names, categorical


def _head_pipeline(
    feature_names: list[str], categorical: list[str], seed: int
) -> Pipeline:
    numeric = [value for value in feature_names if value not in categorical]
    design = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("design", design),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    learning_rate=0.06,
                    max_iter=240,
                    max_leaf_nodes=15,
                    l2_regularization=0.5,
                    random_state=seed,
                ),
            ),
        ]
    )


def _field_year_pipeline(seed: int) -> Pipeline:
    design = ColumnTransformer(
        [
            (
                "year",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                ["publication_year"],
            ),
            (
                "domain",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["domain12"],
            ),
        ]
    )
    return Pipeline(
        [
            ("design", design),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=160,
                    max_leaf_nodes=10,
                    l2_regularization=2.0,
                    random_state=seed,
                ),
            ),
        ]
    )


def _fit_head(
    train: pd.DataFrame,
    target: pd.Series,
    feature_names: list[str],
    categorical: list[str],
    seed: int,
) -> Pipeline:
    values = pd.to_numeric(target, errors="coerce")
    valid = values.notna()
    if int(valid.sum()) < 20:
        raise ValueError("each structural head requires 20 finite training labels")
    model = _head_pipeline(feature_names, categorical, seed)
    model.fit(train.loc[valid, feature_names], values.loc[valid])
    return model


def _support(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: list[str],
    categorical: list[str],
) -> np.ndarray:
    numeric = [value for value in feature_names if value not in categorical]
    train_numeric = train[numeric].apply(pd.to_numeric, errors="coerce")
    test_numeric = test[numeric].apply(pd.to_numeric, errors="coerce")
    lower = train_numeric.quantile(0.01)
    upper = train_numeric.quantile(0.99)
    numeric_support = (
        test_numeric.ge(lower, axis=1) & test_numeric.le(upper, axis=1)
    ).mean(axis=1)
    category_support = pd.Series(1.0, index=test.index)
    for column in categorical:
        seen = set(train[column].dropna().astype(str))
        category_support *= test[column].astype(str).isin(seen).astype(float)
    coverage = test[feature_names].notna().mean(axis=1)
    return np.clip(
        coverage.to_numpy(float)
        * (
            0.8 * numeric_support.to_numpy(float)
            + 0.2 * category_support.to_numpy(float)
        ),
        0.0,
        1.0,
    )


def _calibration_width(
    train: pd.DataFrame,
    d_target: pd.Series,
    p_target: pd.Series,
    feature_names: list[str],
    categorical: list[str],
    seed: int,
) -> tuple[float, int, float]:
    hashes = (
        train["paper_id"]
        .astype(str)
        .map(lambda value: int(hashlib.sha256(value.encode()).hexdigest()[:12], 16))
    )
    calibration = hashes.mod(5).eq(0)
    if int(calibration.sum()) < 20 or int((~calibration).sum()) < 40:
        order = hashes.sort_values(kind="stable").index
        calibration = pd.Series(False, index=train.index)
        calibration.loc[order[: max(20, len(train) // 5)]] = True
    fit = ~calibration
    d_model = _fit_head(
        train.loc[fit], d_target.loc[fit], feature_names, categorical, seed + 71
    )
    p_model = _fit_head(
        train.loc[fit], p_target.loc[fit], feature_names, categorical, seed + 72
    )
    valid = calibration & d_target.notna() & p_target.notna()
    d_error = np.abs(
        np.clip(d_model.predict(train.loc[valid, feature_names]), 0.0, 1.0)
        - d_target.loc[valid].to_numpy(float)
    )
    p_error = np.abs(
        np.clip(p_model.predict(train.loc[valid, feature_names]), 0.0, 1.0)
        - p_target.loc[valid].to_numpy(float)
    )
    residual = np.maximum(d_error, p_error)
    quantile = min(1.0, np.ceil(0.90 * (len(residual) + 1)) / len(residual))
    half_width = float(np.quantile(residual, quantile, method="higher"))
    covered = int(np.count_nonzero(residual <= half_width))
    calibration_reliability = _wilson_lower_bound(covered, len(residual))
    return min(1.0, 2.0 * half_width), len(residual), calibration_reliability


def _wilson_lower_bound(successes: int, trials: int, z: float = 1.645) -> float:
    if trials <= 0:
        return 0.0
    rate = successes / trials
    denominator = 1.0 + z**2 / trials
    center = rate + z**2 / (2.0 * trials)
    radius = z * np.sqrt(rate * (1.0 - rate) / trials + z**2 / (4.0 * trials**2))
    return float(np.clip((center - radius) / denominator, 0.0, 1.0))


def _targets(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    excess = FoldLocalExcessDiffusionTransformer().fit(train)
    train_targets = excess.transform(train)
    test_targets = excess.transform(test)
    perturbation = FoldLocalPerturbationTargetTransformer().fit(train)
    train_targets = train_targets.join(perturbation.transform(train))
    test_targets = test_targets.join(perturbation.transform(test))
    provenance = {
        "excess_target_fit_scope": "outer_training_fold_only",
        "perturbation_target_fit_scope": "outer_training_fold_only",
        "excess_design_columns": list(excess.design_columns_),
        "excess_transform_sha256": _sha256_payload(
            {
                "beta": excess.beta_.tolist(),
                "residual_reference": excess.residual_reference_.tolist(),
                "design_columns": excess.design_columns_,
            }
        ),
        "perturbation_transform_sha256": _sha256_payload(
            {key: value.tolist() for key, value in perturbation.references_.items()}
        ),
    }
    return train_targets, test_targets, provenance


def _fit_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: list[str],
    categorical: list[str],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_targets, test_targets, provenance = _targets(train, test)
    d_model = _fit_head(
        train,
        train_targets["excess_diffusion_fold"],
        feature_names,
        categorical,
        seed,
    )
    p_model = _fit_head(
        train,
        train_targets["perturbation_fold"],
        feature_names,
        categorical,
        seed + 1,
    )
    component_models = {
        component: _fit_head(
            train,
            train_targets[f"{component}_fold"],
            feature_names,
            categorical,
            seed + 10 + index,
        )
        for index, component in enumerate(PERTURBATION_COMPONENTS)
    }
    field_year_model = _field_year_pipeline(seed + 30)
    field_year_model.fit(train[["domain12", "publication_year"]], train["rgpm_d_fold"])
    width, calibration_rows, calibration_reliability = _calibration_width(
        train,
        train_targets["excess_diffusion_fold"],
        train_targets["perturbation_fold"],
        feature_names,
        categorical,
        seed,
    )
    result = pd.DataFrame(
        {
            "paper_id": test["paper_id"].astype(str).to_numpy(),
            "domain12": test["domain12"].astype(str).to_numpy(),
            "publication_year": test["publication_year"].astype(int).to_numpy(),
            "excess_diffusion_target_fold": test_targets[
                "excess_diffusion_fold"
            ].to_numpy(float),
            "perturbation_target_fold": test_targets["perturbation_fold"].to_numpy(
                float
            ),
            "excess_diffusion_head_d": np.clip(
                d_model.predict(test[feature_names]), 0.0, 1.0
            ),
            "field_year_base": np.clip(
                field_year_model.predict(test[["domain12", "publication_year"]]),
                0.0,
                1.0,
            ),
            "perturbation_head_p": np.clip(
                p_model.predict(test[feature_names]), 0.0, 1.0
            ),
            "prediction_interval_width": width,
            "ood_reliability": _support(train, test, feature_names, categorical),
            "calibration_reliability": calibration_reliability,
        },
        index=test.index,
    )
    for component, model in component_models.items():
        result[f"perturbation_component_{component}"] = np.clip(
            model.predict(test[feature_names]), 0.0, 1.0
        )
        result[f"{component}_target_fold"] = test_targets[f"{component}_fold"].to_numpy(
            float
        )
    report = {
        **provenance,
        "field_year_base_fit_scope": "outer_training_fold_only",
        "train_papers": int(train["paper_id"].nunique()),
        "test_papers": int(test["paper_id"].nunique()),
        "paper_overlap": len(
            set(train["paper_id"].astype(str)) & set(test["paper_id"].astype(str))
        ),
        "calibration_rows": calibration_rows,
        "calibration_reliability": calibration_reliability,
        "prediction_interval_width": width,
    }
    return result, report


def _cross_fit(
    frame: pd.DataFrame,
    split_column: str,
    feature_names: list[str],
    categorical: list[str],
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    predictions: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    for index, split in enumerate(
        sorted(frame[split_column].dropna().unique(), key=str)
    ):
        train = frame.loc[frame[split_column].ne(split)].copy()
        test = frame.loc[frame[split_column].eq(split)].copy()
        predicted, report = _fit_fold(
            train, test, feature_names, categorical, seed + 100 * index
        )
        predicted[f"heldout_{split_column}"] = str(split)
        predictions.append(predicted)
        reports.append({f"heldout_{split_column}": str(split), **report})
    result = pd.concat(predictions, ignore_index=True)
    if len(result) != len(frame) or result["paper_id"].duplicated().any():
        raise ValueError("cross-fitting did not produce one paper-independent row")
    return result, reports


def _permuted_predictions(
    frame: pd.DataFrame,
    split_column: str,
    target_column: str,
    feature_names: list[str],
    categorical: list[str],
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    generator = np.random.default_rng(seed)
    for index, split in enumerate(
        sorted(frame[split_column].dropna().unique(), key=str)
    ):
        train = frame.loc[frame[split_column].ne(split)].copy()
        test = frame.loc[frame[split_column].eq(split)].copy()
        train_targets, test_targets, _ = _targets(train, test)
        target = train_targets[target_column].copy()
        groups = pd.DataFrame(
            {
                "domain12": train["domain12"].astype(str),
                "year_bin": (train["publication_year"].astype(int) // 5) * 5,
            },
            index=train.index,
        )
        for indexes in groups.groupby(["domain12", "year_bin"]).groups.values():
            indexes = list(indexes)
            target.loc[indexes] = generator.permutation(target.loc[indexes])
        if target.equals(train_targets[target_column]):
            target.loc[:] = generator.permutation(target.to_numpy())
        model = _fit_head(
            train,
            target,
            feature_names,
            categorical,
            seed + index,
        )
        rows.append(
            pd.DataFrame(
                {
                    "paper_id": test["paper_id"].astype(str),
                    "target": test_targets[target_column].to_numpy(float),
                    "permuted_prediction": np.clip(
                        model.predict(test[feature_names]), 0.0, 1.0
                    ),
                    "heldout_split": str(split),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _metrics(
    frame: pd.DataFrame,
    target: str,
    prediction: str,
    permuted: pd.DataFrame,
    seed: int,
) -> dict[str, float | int]:
    values = frame[["paper_id", target, prediction]].merge(
        permuted[["paper_id", "permuted_prediction"]],
        on="paper_id",
        how="inner",
        validate="one_to_one",
    )
    values = values.dropna()
    rho = float(values[target].corr(values[prediction], method="spearman"))
    shuffled = float(
        values[target].corr(values["permuted_prediction"], method="spearman")
    )
    generator = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(2000):
        sample = values.iloc[generator.integers(0, len(values), len(values))]
        estimate = sample[target].corr(sample[prediction], method="spearman")
        if np.isfinite(estimate):
            bootstrap.append(float(estimate))
    quantiles: np.ndarray = np.asarray(
        np.quantile(bootstrap, [0.025, 0.975]), dtype=float
    )
    low = float(quantiles[0])
    high = float(quantiles[1])
    return {
        "papers": len(values),
        "spearman": rho,
        "permuted_spearman": shuffled,
        "real_minus_permuted": rho - shuffled,
        "spearman_ci95_low": low,
        "spearman_ci95_high": high,
        "mae": float(mean_absolute_error(values[target], values[prediction])),
    }


def _scientific_validation(
    frame: pd.DataFrame,
    temporal: pd.DataFrame,
    domain: pd.DataFrame,
    feature_names: list[str],
    categorical: list[str],
) -> dict[str, Any]:
    temporal_d_shuffle = _permuted_predictions(
        frame,
        "outer_fold_id",
        "excess_diffusion_fold",
        feature_names,
        categorical,
        SEED + 1000,
    )
    temporal_p_shuffle = _permuted_predictions(
        frame,
        "outer_fold_id",
        "perturbation_fold",
        feature_names,
        categorical,
        SEED + 2000,
    )
    domain_d_shuffle = _permuted_predictions(
        frame,
        "domain12",
        "excess_diffusion_fold",
        feature_names,
        categorical,
        SEED + 3000,
    )
    domain_p_shuffle = _permuted_predictions(
        frame,
        "domain12",
        "perturbation_fold",
        feature_names,
        categorical,
        SEED + 4000,
    )
    latest = str(
        frame.groupby("outer_fold_id", observed=True)["publication_year"]
        .mean()
        .idxmax()
    )
    temporal_latest = temporal.loc[temporal["heldout_outer_fold_id"].eq(latest)]
    latest_d_shuffle = temporal_d_shuffle.loc[
        temporal_d_shuffle["heldout_split"].eq(latest)
    ]
    latest_p_shuffle = temporal_p_shuffle.loc[
        temporal_p_shuffle["heldout_split"].eq(latest)
    ]
    metrics: dict[str, dict[str, Any]] = {
        "forward_temporal_latest": {
            "heldout_outer_fold_id": latest,
            "d_excess": _metrics(
                temporal_latest,
                "excess_diffusion_target_fold",
                "excess_diffusion_head_d",
                latest_d_shuffle,
                SEED + 11,
            ),
            "perturbation": _metrics(
                temporal_latest,
                "perturbation_target_fold",
                "perturbation_head_p",
                latest_p_shuffle,
                SEED + 12,
            ),
        },
        "leave_one_domain_out": {
            "d_excess": _metrics(
                domain,
                "excess_diffusion_target_fold",
                "excess_diffusion_head_d",
                domain_d_shuffle,
                SEED + 21,
            ),
            "perturbation": _metrics(
                domain,
                "perturbation_target_fold",
                "perturbation_head_p",
                domain_p_shuffle,
                SEED + 22,
            ),
        },
    }
    gates: dict[str, bool] = {}
    for scope in ("forward_temporal_latest", "leave_one_domain_out"):
        heads = metrics[scope]
        for head in ("d_excess", "perturbation"):
            result = heads[head]
            gates[f"{scope}:{head}:positive_ci"] = (
                float(result["spearman_ci95_low"]) > 0.0
            )
            gates[f"{scope}:{head}:beats_permutation"] = (
                float(result["real_minus_permuted"]) > 0.0
            )
    gates["cohort_exactly_241_unique_papers"] = (
        len(frame) == 241 and frame["paper_id"].nunique() == 241
    )
    gates["latest_temporal_holdout_at_least_100"] = len(temporal_latest) >= 100
    gates["primary16_only"] = len(feature_names) == 16
    return {"metrics": metrics, "promotion_gates": gates}


def _fit_full_bundle(
    frame: pd.DataFrame,
    feature_names: list[str],
    categorical: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    full_targets, _, provenance = _targets(frame, frame)
    heads = {
        "excess_diffusion_head_d": _fit_head(
            frame,
            full_targets["excess_diffusion_fold"],
            feature_names,
            categorical,
            SEED,
        ),
        "perturbation_head_p": _fit_head(
            frame,
            full_targets["perturbation_fold"],
            feature_names,
            categorical,
            SEED + 1,
        ),
    }
    for index, component in enumerate(PERTURBATION_COMPONENTS):
        heads[f"perturbation_component_{component}"] = _fit_head(
            frame,
            full_targets[f"{component}_fold"],
            feature_names,
            categorical,
            SEED + 10 + index,
        )
    field_year_model = _field_year_pipeline(SEED + 30)
    field_year_model.fit(frame[["domain12", "publication_year"]], frame["rgpm_d_fold"])
    width, calibration_rows, calibration_reliability = _calibration_width(
        frame,
        full_targets["excess_diffusion_fold"],
        full_targets["perturbation_fold"],
        feature_names,
        categorical,
        SEED,
    )
    numeric = [value for value in feature_names if value not in categorical]
    numeric_frame = frame[numeric].apply(pd.to_numeric, errors="coerce")
    bundle = {
        "contract": "gear_frozen_structural_head_bundle_v1",
        "feature_names": feature_names,
        "categorical_feature_names": categorical,
        "uses_future_features": False,
        "inference_columns": [*feature_names, "domain12", "publication_year"],
        "heads": heads,
        "field_year_model": field_year_model,
        "prediction_interval_width": width,
        "calibration_reliability": calibration_reliability,
        "numeric_support_lower": numeric_frame.quantile(0.01).to_dict(),
        "numeric_support_upper": numeric_frame.quantile(0.99).to_dict(),
        "categorical_support": {
            column: sorted(frame[column].dropna().astype(str).unique())
            for column in categorical
        },
        "target_provenance": provenance,
        "calibration_rows": calibration_rows,
    }
    reference = frame[
        [
            "paper_id",
            "outer_fold_id",
            "domain12",
            "publication_year",
            *feature_names,
            "rgpm_d_fold",
            "n_future_citers",
            *PERTURBATION_COMPONENTS,
        ]
    ].copy()
    for column in full_targets:
        reference[column] = full_targets[column].to_numpy()
    reference["feature_time_basis"] = "T0_only"
    reference["future_columns_role"] = "label_construction_only_not_inference"
    return bundle, reference


def _predict_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    feature_names = list(bundle["feature_names"])
    output = pd.DataFrame(index=frame.index)
    for name, model in bundle["heads"].items():
        output[name] = np.clip(model.predict(frame[feature_names]), 0.0, 1.0)
    output["field_year_base"] = np.clip(
        bundle["field_year_model"].predict(frame[["domain12", "publication_year"]]),
        0.0,
        1.0,
    )
    numeric = [
        value
        for value in feature_names
        if value not in bundle["categorical_feature_names"]
    ]
    values = frame[numeric].apply(pd.to_numeric, errors="coerce")
    lower = pd.Series(bundle["numeric_support_lower"])
    upper = pd.Series(bundle["numeric_support_upper"])
    support = (values.ge(lower, axis=1) & values.le(upper, axis=1)).mean(axis=1)
    category_support = pd.Series(1.0, index=frame.index)
    for column, allowed in bundle["categorical_support"].items():
        category_support *= frame[column].astype(str).isin(set(allowed)).astype(float)
    coverage = frame[feature_names].notna().mean(axis=1)
    output["ood_reliability"] = np.clip(
        coverage * (0.8 * support + 0.2 * category_support), 0.0, 1.0
    )
    output["prediction_interval_width"] = bundle["prediction_interval_width"]
    output["calibration_reliability"] = bundle["calibration_reliability"]
    return output


def _runtime_prediction_table(
    bundle: dict[str, Any], runtime: RuntimeFeatureRelease
) -> pd.DataFrame:
    features = pd.read_parquet(runtime.path("runtime_feature_table"))
    scores = pd.read_parquet(runtime.path("runtime_score_table"))
    anatomy = pd.read_parquet(runtime.path("runtime_anatomy_table"))
    if list(runtime.manifest.feature_names) != list(bundle["feature_names"]):
        raise ValueError("runtime structural features differ from frozen Primary16")
    frame = (
        features.merge(
            scores[["paper_id", "as_of_date", "source_max_year"]],
            on="paper_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            anatomy[["paper_id", "target_field"]],
            on="paper_id",
            how="inner",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )
    if len(frame) != runtime.manifest.target_count:
        raise ValueError("runtime structural metadata coverage is incomplete")
    frame["domain12"] = frame["target_field"].map(
        lambda value: assign_domain12({"primary_field": str(value)})[0]
    )
    if frame["domain12"].isin({"unmapped", "out_of_scope_nonnatural"}).any():
        raise ValueError("runtime structural field-year base is unmapped")
    as_of = pd.to_datetime(frame["as_of_date"], errors="coerce")
    if as_of.isna().any():
        raise ValueError("runtime structural cutoffs are invalid")
    frame["publication_year"] = as_of.dt.year.astype(int)
    predicted = _predict_bundle(bundle, frame)
    output = pd.DataFrame(
        {
            "paper_id": frame["paper_id"].astype(str),
            "prediction_protocol": "frozen_t0_runtime",
            "as_of_date": as_of.dt.strftime("%Y-%m-%d"),
            "target_publication_date": as_of.dt.strftime("%Y-%m-%d"),
            "feature_source_max_date": frame["source_max_year"]
            .astype(int)
            .astype(str)
            .add("-12-31"),
            "outer_fold_id": pd.Series([None] * len(frame), dtype="object"),
            "feature_source_sha256": runtime.manifest.assets[
                "runtime_feature_table"
            ].sha256,
        }
    )
    for column in predicted:
        output[column] = predicted[column].to_numpy()
    source_max = pd.to_datetime(output["feature_source_max_date"], errors="coerce")
    if (source_max > as_of).any():
        raise ValueError("runtime structural features extend past the cutoff")
    return output.sort_values("paper_id", kind="stable").reset_index(drop=True)


def _parent_oof_inference_table(
    bundle: dict[str, Any],
    parent: ForecastRelease,
    training_ids: set[str],
) -> pd.DataFrame:
    feature_names = list(bundle["feature_names"])
    oof = pd.read_parquet(
        parent.path("oof_predictions"),
        columns=["paper_id", "domain12", "publication_year"],
    )
    features = pd.read_parquet(
        parent.path("training_snapshot"), columns=["paper_id", *feature_names]
    )
    frame = oof.loc[~oof["paper_id"].astype(str).isin(training_ids)].merge(
        features, on="paper_id", how="inner", validate="one_to_one"
    )
    expected_rows = len(oof) - len(training_ids)
    if len(frame) != expected_rows or frame["paper_id"].duplicated().any():
        raise ValueError("parent OOF Primary16 inference coverage is incomplete")
    predicted = _predict_bundle(bundle, frame)
    dates = frame["publication_year"].astype(int).astype(str).add("-01-01")
    output = pd.DataFrame(
        {
            "paper_id": frame["paper_id"].astype(str),
            "prediction_protocol": "frozen_t0_out_of_training",
            "as_of_date": dates,
            "target_publication_date": dates,
            "feature_source_max_date": dates,
            "outer_fold_id": pd.Series([None] * len(frame), dtype="object"),
            "feature_source_sha256": parent.manifest.assets["training_snapshot"].sha256,
        }
    )
    for column in predicted:
        output[column] = predicted[column].to_numpy()
    return output.sort_values("paper_id", kind="stable").reset_index(drop=True)


def _prediction_table(
    temporal: pd.DataFrame,
    runtime_predictions: pd.DataFrame,
    parent_inference: pd.DataFrame,
    parent_feature_sha256: str,
) -> pd.DataFrame:
    output = temporal.copy()
    output["prediction_protocol"] = "strict_oof"
    output["outer_fold_id"] = output.pop("heldout_outer_fold_id")
    dates = output["publication_year"].astype(int).astype(str) + "-01-01"
    output["as_of_date"] = dates
    output["target_publication_date"] = dates
    output["feature_source_max_date"] = dates
    output["feature_source_sha256"] = parent_feature_sha256
    keep = [
        "paper_id",
        "prediction_protocol",
        "as_of_date",
        "target_publication_date",
        "feature_source_max_date",
        "feature_source_sha256",
        "outer_fold_id",
        "excess_diffusion_head_d",
        "field_year_base",
        "perturbation_head_p",
        "prediction_interval_width",
        "ood_reliability",
        "calibration_reliability",
        *[
            f"perturbation_component_{component}"
            for component in PERTURBATION_COMPONENTS
        ],
    ]
    historical = output[keep]
    runtime = runtime_predictions[keep]
    inference = parent_inference[keep]
    return (
        pd.concat([historical, runtime, inference], ignore_index=True)
        .sort_values("paper_id", kind="stable")
        .reset_index(drop=True)
    )


def _asset(path: Path) -> dict[str, Any]:
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _coverage_audit(
    predictions: pd.DataFrame,
    training_ids: set[str],
    runtime_ids: set[str],
) -> dict[str, Any]:
    indexed = predictions.set_index("paper_id")
    if not indexed.index.is_unique:
        raise ValueError("structural-head coverage table has duplicate papers")

    def cohort(path: Path, expected: int) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        ids = [str(row["paper_id"]) for row in payload["cases"]]
        available = [value for value in ids if value in indexed.index]
        protocols = (
            indexed.loc[available, "prediction_protocol"].value_counts().to_dict()
        )
        valid_protocol = all(
            str(indexed.at[value, "prediction_protocol"])
            == ("strict_oof" if value in training_ids else "frozen_t0_out_of_training")
            for value in available
        )
        return {
            "manifest": str(path.relative_to(PROJECT_ROOT)),
            "manifest_sha256": sha256_file(path),
            "expected_papers": expected,
            "available_papers": len(available),
            "protocol_counts": {
                str(key): int(value) for key, value in protocols.items()
            },
            "protocols_correct_by_training_identity": valid_protocol,
            "passed": len(ids) == expected
            and len(set(ids)) == expected
            and len(available) == expected
            and valid_protocol,
        }

    runtime_available = [value for value in runtime_ids if value in indexed.index]
    runtime_protocols = {
        str(indexed.at[value, "prediction_protocol"]) for value in runtime_available
    }
    result: dict[str, Any] = {
        "contract": "gear_structural_head_coverage_audit_v1",
        "stage_b_241": cohort(STAGE_B_MANIFEST, 241),
        "stage_c_150": cohort(STAGE_C_MANIFEST, 150),
        "runtime_10": {
            "expected_papers": 10,
            "available_papers": len(runtime_available),
            "protocols": sorted(runtime_protocols),
            "passed": len(runtime_ids) == 10
            and len(runtime_available) == 10
            and runtime_protocols == {"frozen_t0_runtime"},
        },
    }
    result["passed"] = all(
        bool(result[key]["passed"])
        for key in ("stage_b_241", "stage_c_150", "runtime_10")
    )
    return result


def build_release(
    output_dir: Path = OUTPUT_DIR,
    raw_path: Path = RAW_INPUTS,
    cohort_path: Path = COHORT_PREDICTIONS,
    runtime_manifest_path: Path = RUNTIME_MANIFEST,
) -> dict[str, Any]:
    parent = ForecastRelease(PARENT_MANIFEST)
    parent.verify()
    runtime_release = RuntimeFeatureRelease(runtime_manifest_path)
    runtime_release.verify(parent)
    frame, feature_names, categorical = _load_frame(parent, raw_path, cohort_path)
    temporal, temporal_folds = _cross_fit(
        frame, "outer_fold_id", feature_names, categorical, SEED
    )
    domain, domain_folds = _cross_fit(
        frame, "domain12", feature_names, categorical, SEED + 10000
    )
    validation = _scientific_validation(
        frame, temporal, domain, feature_names, categorical
    )
    gates = validation["promotion_gates"]
    promoted = all(bool(value) for value in gates.values())
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.iterdir():
        if old.is_file():
            old.unlink()
        elif old.is_dir():
            shutil.rmtree(old)
    bundle, training_reference = _fit_full_bundle(frame, feature_names, categorical)
    model_path = output_dir / "model.joblib"
    joblib.dump(bundle, model_path, compress=3)
    training_path = output_dir / "training_reference.parquet"
    training_reference.to_parquet(training_path, index=False)
    training_ids = set(training_reference["paper_id"].astype(str))
    runtime_predictions = _runtime_prediction_table(bundle, runtime_release)
    parent_inference = _parent_oof_inference_table(bundle, parent, training_ids)
    prediction_path = output_dir / "prediction_table.parquet"
    prediction_table = _prediction_table(
        temporal,
        runtime_predictions,
        parent_inference,
        parent.manifest.assets["training_snapshot"].sha256,
    )
    prediction_table.to_parquet(prediction_path, index=False)
    coverage = _coverage_audit(
        prediction_table,
        training_ids,
        set(runtime_predictions["paper_id"].astype(str)),
    )
    gates["stage_b_stage_c_runtime_exact_coverage"] = bool(coverage["passed"])
    promoted = all(bool(value) for value in gates.values())
    coverage_path = output_dir / "coverage_audit.json"
    coverage_path.write_text(
        json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporal_audit_path = output_dir / "temporal_oof_audit.parquet"
    temporal.sort_values("paper_id", kind="stable").to_parquet(
        temporal_audit_path, index=False
    )
    domain_audit_path = output_dir / "domain_oof_audit.parquet"
    domain.sort_values("paper_id", kind="stable").to_parquet(
        domain_audit_path, index=False
    )
    replay_source = frame.sort_values("paper_id", kind="stable").iloc[::8].copy()
    replay_prediction = _predict_bundle(bundle, replay_source)
    replay = replay_source[
        ["paper_id", *feature_names, "domain12", "publication_year"]
    ].reset_index(drop=True)
    replay = pd.concat(
        [replay, replay_prediction.reset_index(drop=True).add_prefix("expected_")],
        axis=1,
    )
    replay_path = output_dir / "runtime_replay.parquet"
    replay.to_parquet(replay_path, index=False)
    parent_registry_hash = parent.manifest.assets["feature_registry"].sha256
    registry = {
        "contract": "gear_structural_head_feature_registry_v1",
        "feature_names": feature_names,
        "categorical_feature_names": categorical,
        "feature_time_basis": "T0_only",
        "uses_future_features": False,
        "parent_feature_registry_sha256": parent_registry_hash,
        "inference_columns": [*feature_names, "domain12", "publication_year"],
        "label_only_future_columns": [
            "realized_diffusion_target",
            "future_citer_count",
            *PERTURBATION_COMPONENTS,
        ],
        "label_only_columns_excluded_from_models": True,
        "excess_target_fit_scope": "outer_training_fold_only",
        "perturbation_target_fit_scope": "outer_training_fold_only",
        "field_year_base_fit_scope": "outer_training_fold_only",
        "historical_prediction_protocol": "strict_paper_independent_oof",
    }
    registry_path = output_dir / "feature_registry.json"
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "contract": "gear_structural_head_validation_v1",
        "status": "supported" if promoted else "not_supported",
        "promotion_passed": promoted,
        "cohort_papers": len(frame),
        "prediction_rows": len(temporal),
        "runtime_prediction_rows": len(runtime_predictions),
        "parent_oof_inference_rows": len(parent_inference),
        "feature_names": feature_names,
        "uses_future_features": False,
        "prediction_protocol": "strict_paper_independent_oof",
        "target_fit_scope": "outer_training_fold_only",
        "scientific_threshold": (
            "for both D-excess and P: latest-temporal and leave-one-domain-out "
            "Spearman 95% bootstrap lower bound > 0 and real > within-field/year "
            "permutation; exact 241-paper coverage"
        ),
        **validation,
        "temporal_fold_reports": temporal_folds,
        "domain_fold_reports": domain_folds,
        "source_assets": {
            "parent_manifest": sha256_file(PARENT_MANIFEST),
            "raw_perturbation_inputs": sha256_file(raw_path),
            "cohort_predictions": sha256_file(cohort_path),
            "runtime_manifest": sha256_file(runtime_manifest_path),
        },
    }
    report_path = output_dir / "validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifact_paths = {
        "model": model_path,
        "feature_registry": registry_path,
        "training_reference": training_path,
        "prediction_table": prediction_path,
        "validation_report": report_path,
        "runtime_replay": replay_path,
        "temporal_oof_audit": temporal_audit_path,
        "domain_oof_audit": domain_audit_path,
        "coverage_audit": coverage_path,
    }
    if not promoted:
        blocked = {
            "contract": "gear_structural_head_promotion_block_v1",
            "status": "blocked_by_scientific_gate",
            "failed_gates": sorted(key for key, value in gates.items() if not value),
            "candidate_assets": {
                name: _asset(path) for name, path in artifact_paths.items()
            },
        }
        (output_dir / "promotion_blocked.json").write_text(
            json.dumps(blocked, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return blocked
    assets = {name: _asset(path) for name, path in artifact_paths.items()}
    release_fingerprint = _sha256_payload(
        {name: value["sha256"] for name, value in sorted(assets.items())}
    ).removeprefix("sha256:")[:12]
    manifest = {
        "contract": "gear_structural_head_release_v1",
        "release_id": f"gear-structural-head-primary16-{release_fingerprint}",
        "parent_forecast_release_id": parent.manifest.release_id,
        "parent_feature_registry_sha256": parent_registry_hash,
        "feature_protocol_version": parent.manifest.protocol_version,
        "horizon_years": 5,
        "status": "promoted",
        "feature_time_basis": "T0_only",
        "uses_future_features": False,
        "historical_prediction_policy": "strict_oof_only",
        "runtime_prediction_policy": "frozen_model_t0_only",
        "excess_target_fit_scope": "outer_training_fold_only",
        "perturbation_target_fit_scope": "outer_training_fold_only",
        "feature_names": feature_names,
        "training_row_count": len(frame),
        "runtime_feature_release_id": runtime_release.manifest.release_id,
        "runtime_feature_table_sha256": runtime_release.manifest.assets[
            "runtime_feature_table"
        ].sha256,
        "runtime_score_table_sha256": runtime_release.manifest.assets[
            "runtime_score_table"
        ].sha256,
        "runtime_anatomy_table_sha256": runtime_release.manifest.assets[
            "runtime_anatomy_table"
        ].sha256,
        "runtime_prediction_row_count": len(runtime_predictions),
        "parent_oof_predictions_sha256": parent.manifest.assets[
            "oof_predictions"
        ].sha256,
        "parent_training_snapshot_sha256": parent.manifest.assets[
            "training_snapshot"
        ].sha256,
        "parent_oof_inference_row_count": len(parent_inference),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "assets": assets,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--raw-inputs", type=Path, default=RAW_INPUTS)
    parser.add_argument("--cohort-predictions", type=Path, default=COHORT_PREDICTIONS)
    parser.add_argument("--runtime-manifest", type=Path, default=RUNTIME_MANIFEST)
    args = parser.parse_args()
    result = build_release(
        args.output_dir.resolve(),
        args.raw_inputs.resolve(),
        args.cohort_predictions.resolve(),
        args.runtime_manifest.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "promoted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
