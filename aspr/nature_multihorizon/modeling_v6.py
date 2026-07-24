"""Nested expanding-year OOF models for v6 two-part influence forecasts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .evidence_registry import EvidenceRegistry
from .prediction_registry_v6 import PredictionRegistry
from .source_audit_v6 import sha256_file
from .targets_v6 import FoldLocalDiffusionTargetTransformer


MODEL_PROTOCOL_VERSION = "aspr-two-part-nested-oof-v6-1"
MODEL_PARAMETER_GRID: Tuple[Mapping[str, Any], ...] = (
    {
        "parameter_id": "compact",
        "max_leaf_nodes": 15,
        "max_depth": 3,
        "min_samples_leaf": 50,
        "learning_rate": 0.05,
        "max_iter": 150,
        "l2_regularization": 10.0,
    },
    {
        "parameter_id": "medium",
        "max_leaf_nodes": 31,
        "max_depth": 4,
        "min_samples_leaf": 50,
        "learning_rate": 0.05,
        "max_iter": 200,
        "l2_regularization": 10.0,
    },
)


@dataclass(frozen=True)
class TemporalFold:
    """One expanding-year train/test split."""

    fold_id: int
    train_year_min: int
    train_year_max: int
    test_year_min: int
    test_year_max: int
    train_index: np.ndarray
    test_index: np.ndarray


class PlattCalibrator:
    """Logistic calibration of held-out raw probabilities."""

    def __init__(self) -> None:
        self.model_: Optional[LogisticRegression] = None

    @staticmethod
    def _logit(probabilities: Sequence[float]) -> np.ndarray:
        values = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(values / (1.0 - values)).reshape(-1, 1)

    def fit(
        self, probabilities: Sequence[float], outcomes: Sequence[float]
    ) -> "PlattCalibrator":
        probabilities_array = np.asarray(probabilities, dtype=float)
        outcomes_array = np.asarray(outcomes, dtype=float)
        valid = np.isfinite(probabilities_array) & np.isfinite(outcomes_array)
        if valid.sum() >= 100 and len(np.unique(outcomes_array[valid])) == 2:
            model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            model.fit(
                self._logit(probabilities_array[valid]),
                outcomes_array[valid].astype(int),
            )
            self.model_ = model
        return self

    def predict(self, probabilities: Sequence[float]) -> np.ndarray:
        values = np.asarray(probabilities, dtype=float)
        if self.model_ is None:
            return np.clip(values, 0.0, 1.0)
        return self.model_.predict_proba(self._logit(values))[:, 1]


class ConditionalCalibrator:
    """Monotone calibration of conditional diffusion predictions."""

    def __init__(self) -> None:
        self.model_: Optional[IsotonicRegression] = None

    def fit(
        self, predictions: Sequence[float], outcomes: Sequence[float]
    ) -> "ConditionalCalibrator":
        predictions_array = np.asarray(predictions, dtype=float)
        outcomes_array = np.asarray(outcomes, dtype=float)
        valid = np.isfinite(predictions_array) & np.isfinite(outcomes_array)
        if valid.sum() >= 200 and len(np.unique(predictions_array[valid])) >= 10:
            model = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            model.fit(predictions_array[valid], outcomes_array[valid])
            self.model_ = model
        return self

    def predict(self, predictions: Sequence[float]) -> np.ndarray:
        values = np.asarray(predictions, dtype=float)
        if self.model_ is None:
            return np.clip(values, 0.0, 1.0)
        return np.asarray(self.model_.predict(values), dtype=float)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def make_expanding_year_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    year_column: str = "publication_year",
    initial_fraction: float = 0.20,
) -> Tuple[TemporalFold, ...]:
    """Create balanced contiguous tests with strictly earlier training years."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    years = pd.to_numeric(frame[year_column], errors="coerce")
    if years.isna().any():
        raise ValueError("temporal folds require complete publication years")
    counts = years.astype(int).value_counts().sort_index()
    if len(counts) <= n_splits:
        raise ValueError("not enough distinct years for expanding folds")
    initial_target = max(1, int(math.ceil(len(frame) * initial_fraction)))
    initial_year = int(counts.cumsum().ge(initial_target).idxmax())
    test_counts = counts[counts.index > initial_year]
    if len(test_counts) < n_splits:
        initial_year = int(counts.index[-n_splits - 1])
        test_counts = counts[counts.index > initial_year]
    cumulative = test_counts.cumsum().to_numpy(dtype=float)
    total = float(cumulative[-1])
    boundaries = [0]
    for split_index in range(1, n_splits):
        target = total * split_index / n_splits
        proposed = int(np.searchsorted(cumulative, target, side="left") + 1)
        minimum = boundaries[-1] + 1
        maximum = len(test_counts) - (n_splits - split_index)
        boundaries.append(min(max(proposed, minimum), maximum))
    boundaries.append(len(test_counts))
    test_years = test_counts.index.to_numpy(dtype=int)
    folds: List[TemporalFold] = []
    for fold_id, (left, right) in enumerate(
        zip(boundaries[:-1], boundaries[1:]), start=1
    ):
        selected_years = test_years[left:right]
        if not len(selected_years):
            continue
        test_min = int(selected_years.min())
        test_max = int(selected_years.max())
        train_positions = np.flatnonzero(years.to_numpy(dtype=int) < test_min)
        test_positions = np.flatnonzero(
            years.to_numpy(dtype=int) >= test_min
        )
        test_positions = test_positions[
            years.to_numpy(dtype=int)[test_positions] <= test_max
        ]
        if not len(train_positions) or not len(test_positions):
            continue
        folds.append(
            TemporalFold(
                fold_id=fold_id,
                train_year_min=int(years.iloc[train_positions].min()),
                train_year_max=int(years.iloc[train_positions].max()),
                test_year_min=test_min,
                test_year_max=test_max,
                train_index=train_positions,
                test_index=test_positions,
            )
        )
    if len(folds) != n_splits:
        raise ValueError("could not create the requested expanding folds")
    for fold in folds:
        if fold.train_year_max >= fold.test_year_min:
            raise ValueError("temporal fold leakage detected")
    return tuple(folds)


def _prepare_feature_frame(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    categorical_names: Sequence[str],
) -> pd.DataFrame:
    output = frame[list(feature_names)].copy()
    categorical = set(categorical_names)
    for column in output:
        if column in categorical:
            output[column] = output[column].astype("string").fillna("missing")
        else:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _build_preprocessor(
    feature_names: Sequence[str],
    categorical_names: Sequence[str],
) -> ColumnTransformer:
    categorical = [name for name in feature_names if name in categorical_names]
    numeric = [name for name in feature_names if name not in categorical_names]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="missing",
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                min_frequency=10,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )


@dataclass
class FittedTwoPartModel:
    """Fitted fold-local two-part model and target reference."""

    feature_names: Tuple[str, ...]
    categorical_names: Tuple[str, ...]
    preprocessor: ColumnTransformer
    classifier: HistGradientBoostingClassifier
    regressor: HistGradientBoostingRegressor
    target_transformer: FoldLocalDiffusionTargetTransformer

    def predict_raw(self, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        features = _prepare_feature_frame(
            frame, self.feature_names, self.categorical_names
        )
        transformed = self.preprocessor.transform(features)
        uptake = self.classifier.predict_proba(transformed)[:, 1]
        conditional = np.clip(self.regressor.predict(transformed), 0.0, 1.0)
        return uptake, conditional


@dataclass
class CalibratedTwoPartBundle:
    """Final development-fitted estimator with OOF-only calibration."""

    fitted_model: FittedTwoPartModel
    uptake_calibrator: PlattCalibrator
    conditional_calibrator: ConditionalCalibrator
    conditional_residual_quantile: float
    realized_residual_quantile: float

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict calibrated two-part scores without reading outcomes."""
        uptake_raw, conditional_raw = self.fitted_model.predict_raw(frame)
        uptake = self.uptake_calibrator.predict(uptake_raw)
        conditional = self.conditional_calibrator.predict(conditional_raw)
        expected = uptake * conditional
        return pd.DataFrame(
            {
                "uptake_probability_raw": uptake_raw,
                "uptake_probability_calibrated": uptake,
                "conditional_diffusion_raw": conditional_raw,
                "conditional_diffusion_calibrated": conditional,
                "expected_diffusion_score": expected,
                "conditional_interval_low": np.clip(
                    conditional - self.conditional_residual_quantile,
                    0.0,
                    1.0,
                ),
                "conditional_interval_high": np.clip(
                    conditional + self.conditional_residual_quantile,
                    0.0,
                    1.0,
                ),
                "realized_interval_low": np.clip(
                    expected - self.realized_residual_quantile,
                    0.0,
                    1.0,
                ),
                "realized_interval_high": np.clip(
                    expected + self.realized_residual_quantile,
                    0.0,
                    1.0,
                ),
            },
            index=frame.index,
        )


def _fit_two_part(
    training: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    categorical_names: Sequence[str],
    parameters: Mapping[str, Any],
    seed: int,
) -> FittedTwoPartModel:
    preprocessor = _build_preprocessor(feature_names, categorical_names)
    training_features = _prepare_feature_frame(
        training, feature_names, categorical_names
    )
    transformed = preprocessor.fit_transform(training_features)
    common_parameters = {
        "max_leaf_nodes": int(parameters["max_leaf_nodes"]),
        "max_depth": int(parameters["max_depth"]),
        "min_samples_leaf": int(parameters["min_samples_leaf"]),
        "learning_rate": float(parameters["learning_rate"]),
        "max_iter": int(parameters["max_iter"]),
        "l2_regularization": float(parameters["l2_regularization"]),
        "early_stopping": False,
        "random_state": int(seed),
    }
    uptake = pd.to_numeric(training["future_uptake"], errors="coerce")
    if uptake.isna().any() or uptake.nunique() != 2:
        raise ValueError("uptake training outcome must be complete and binary")
    classifier = HistGradientBoostingClassifier(**common_parameters)
    classifier.fit(transformed, uptake.astype(int).to_numpy())

    conditional_mask = training["conditional_diffusion_member"].eq(1)
    conditional_training = training.loc[conditional_mask]
    target_transformer = FoldLocalDiffusionTargetTransformer().fit(
        conditional_training
    )
    target = target_transformer.transform(conditional_training)[
        "rgpm_d_fold"
    ].to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise ValueError("conditional training target contains missing values")
    regressor = HistGradientBoostingRegressor(
        loss="squared_error", **common_parameters
    )
    regressor.fit(transformed[conditional_mask.to_numpy()], target)
    return FittedTwoPartModel(
        feature_names=tuple(feature_names),
        categorical_names=tuple(categorical_names),
        preprocessor=preprocessor,
        classifier=classifier,
        regressor=regressor,
        target_transformer=target_transformer,
    )


def _realized_diffusion(
    frame: pd.DataFrame,
    transformer: FoldLocalDiffusionTargetTransformer,
) -> Tuple[np.ndarray, np.ndarray]:
    conditional = transformer.transform(frame)["rgpm_d_fold"].to_numpy(
        dtype=float, copy=True
    )
    uptake = pd.to_numeric(frame["future_uptake"], errors="coerce").to_numpy(
        dtype=float
    )
    if "conditional_diffusion_member" in frame:
        eligible = frame["conditional_diffusion_member"].eq(1).to_numpy()
        conditional[~eligible] = np.nan
    else:
        eligible = uptake == 1
    realized = np.full(len(frame), np.nan, dtype=float)
    zero = uptake == 0
    positive = (uptake == 1) & eligible & np.isfinite(conditional)
    realized[zero] = 0.0
    realized[positive] = conditional[positive]
    return conditional, realized


def realized_diffusion_target(
    frame: pd.DataFrame,
    transformer: FoldLocalDiffusionTargetTransformer,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a frozen development target reference to evaluation outcomes."""
    return _realized_diffusion(frame, transformer)


def safe_spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Return Spearman rho or NaN for insufficient/constant inputs."""
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    valid = np.isfinite(left_array) & np.isfinite(right_array)
    if valid.sum() < 3:
        return float("nan")
    if len(np.unique(left_array[valid])) < 2 or len(np.unique(right_array[valid])) < 2:
        return float("nan")
    return float(spearmanr(left_array[valid], right_array[valid]).statistic)


def _inner_oof_for_parameters(
    training: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    categorical_names: Sequence[str],
    parameters: Mapping[str, Any],
    n_inner: int,
    seed: int,
) -> pd.DataFrame:
    folds = make_expanding_year_folds(
        training, n_splits=n_inner, initial_fraction=0.25
    )
    rows: List[pd.DataFrame] = []
    for fold in folds:
        inner_train = training.iloc[fold.train_index]
        inner_test = training.iloc[fold.test_index]
        model = _fit_two_part(
            inner_train,
            feature_names=feature_names,
            categorical_names=categorical_names,
            parameters=parameters,
            seed=seed + fold.fold_id,
        )
        uptake_raw, conditional_raw = model.predict_raw(inner_test)
        conditional_target, realized = _realized_diffusion(
            inner_test, model.target_transformer
        )
        rows.append(
            pd.DataFrame(
                {
                    "paper_id": inner_test["paper_id"].astype(str).to_numpy(),
                    "future_uptake": inner_test["future_uptake"].to_numpy(
                        dtype=float
                    ),
                    "conditional_diffusion_target": conditional_target,
                    "realized_diffusion_target": realized,
                    "uptake_probability_raw": uptake_raw,
                    "conditional_diffusion_raw": conditional_raw,
                    "inner_fold_id": fold.fold_id,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _select_parameters(
    training: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    categorical_names: Sequence[str],
    parameter_grid: Sequence[Mapping[str, Any]],
    n_inner: int,
    seed: int,
) -> Tuple[Mapping[str, Any], pd.DataFrame, pd.DataFrame]:
    ledger_rows = []
    predictions: Dict[str, pd.DataFrame] = {}
    for parameters in parameter_grid:
        parameter_id = str(parameters["parameter_id"])
        inner = _inner_oof_for_parameters(
            training,
            feature_names=feature_names,
            categorical_names=categorical_names,
            parameters=parameters,
            n_inner=n_inner,
            seed=seed,
        )
        expected = (
            inner["uptake_probability_raw"].to_numpy(dtype=float)
            * inner["conditional_diffusion_raw"].to_numpy(dtype=float)
        )
        rho = safe_spearman(expected, inner["realized_diffusion_target"])
        uptake = inner["future_uptake"].to_numpy(dtype=float)
        probability = inner["uptake_probability_raw"].to_numpy(dtype=float)
        valid = np.isfinite(uptake) & np.isfinite(probability)
        brier = float(np.mean((probability[valid] - uptake[valid]) ** 2))
        objective = rho - 0.10 * brier if np.isfinite(rho) else -np.inf
        ledger_rows.append(
            {
                "parameter_id": parameter_id,
                "inner_expected_spearman": rho,
                "inner_uptake_brier": brier,
                "selection_objective": objective,
                "complexity": int(parameters["max_leaf_nodes"]),
            }
        )
        predictions[parameter_id] = inner
    ledger = pd.DataFrame(ledger_rows).sort_values(
        ["selection_objective", "complexity"],
        ascending=[False, True],
        kind="stable",
    )
    selected_id = str(ledger.iloc[0]["parameter_id"])
    selected = next(
        item for item in parameter_grid if item["parameter_id"] == selected_id
    )
    return selected, predictions[selected_id], ledger


def _fit_calibrators(
    inner: pd.DataFrame,
) -> Tuple[
    PlattCalibrator,
    ConditionalCalibrator,
    float,
    float,
]:
    uptake_calibrator = PlattCalibrator().fit(
        inner["uptake_probability_raw"], inner["future_uptake"]
    )
    positive = (
        inner["future_uptake"].eq(1)
        & inner["conditional_diffusion_target"].notna()
    )
    conditional_calibrator = ConditionalCalibrator().fit(
        inner.loc[positive, "conditional_diffusion_raw"],
        inner.loc[positive, "conditional_diffusion_target"],
    )
    calibrated = conditional_calibrator.predict(
        inner.loc[positive, "conditional_diffusion_raw"]
    )
    conditional_residuals = np.abs(
        calibrated
        - inner.loc[positive, "conditional_diffusion_target"].to_numpy(
            dtype=float
        )
    )
    conditional_residual_quantile = _conformal_absolute_quantile(
        conditional_residuals, coverage=0.90
    )
    uptake_calibrated = uptake_calibrator.predict(
        inner["uptake_probability_raw"]
    )
    conditional_calibrated = conditional_calibrator.predict(
        inner["conditional_diffusion_raw"]
    )
    expected = uptake_calibrated * conditional_calibrated
    realized = inner["realized_diffusion_target"].to_numpy(dtype=float)
    valid = np.isfinite(expected) & np.isfinite(realized)
    realized_residual_quantile = _conformal_absolute_quantile(
        np.abs(expected[valid] - realized[valid]),
        coverage=0.90,
    )
    return (
        uptake_calibrator,
        conditional_calibrator,
        conditional_residual_quantile,
        realized_residual_quantile,
    )


def select_final_parameters(
    model_ledger: pd.DataFrame,
    *,
    model_id: str,
    parameter_grid: Sequence[Mapping[str, Any]] = MODEL_PARAMETER_GRID,
) -> Mapping[str, Any]:
    """Apply the frozen majority rule to outer-fold selections."""
    selected = model_ledger[
        model_ledger["model_id"].eq(model_id)
        & model_ledger["selected"].eq(True)
    ]
    if selected.empty:
        raise ValueError(f"no selected development parameters for {model_id}")
    counts = selected["parameter_id"].astype(str).value_counts()
    candidates = set(counts[counts.eq(counts.max())].index)
    available = {
        str(parameters["parameter_id"]): parameters
        for parameters in parameter_grid
    }
    missing = sorted(candidates - set(available))
    if missing:
        raise ValueError(f"selected parameter ids are unregistered: {missing}")
    return min(
        (available[parameter_id] for parameter_id in candidates),
        key=lambda parameters: (
            int(parameters["max_leaf_nodes"]),
            int(parameters["max_depth"]),
            str(parameters["parameter_id"]),
        ),
    )


def fit_calibrated_final_model(
    training: pd.DataFrame,
    development_oof: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    categorical_names: Sequence[str] = ("domain12", "venue_family"),
    parameters: Mapping[str, Any],
    horizon: int,
    seed: int,
) -> CalibratedTwoPartBundle:
    """Fit calibration from OOF only, then refit the estimator on development."""
    training = training.copy()
    if "horizon" not in training:
        training["horizon"] = int(horizon)
    (
        uptake_calibrator,
        conditional_calibrator,
        conditional_residual_quantile,
        realized_residual_quantile,
    ) = _fit_calibrators(development_oof)
    fitted = _fit_two_part(
        training,
        feature_names=feature_names,
        categorical_names=categorical_names,
        parameters=parameters,
        seed=int(seed),
    )
    return CalibratedTwoPartBundle(
        fitted_model=fitted,
        uptake_calibrator=uptake_calibrator,
        conditional_calibrator=conditional_calibrator,
        conditional_residual_quantile=conditional_residual_quantile,
        realized_residual_quantile=realized_residual_quantile,
    )


def _conformal_absolute_quantile(
    residuals: Sequence[float],
    *,
    coverage: float,
) -> float:
    """Return the finite-sample split-conformal absolute-residual quantile."""
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan")
    if not 0.0 < float(coverage) < 1.0:
        raise ValueError("coverage must be strictly between zero and one")
    quantile_level = min(
        1.0,
        math.ceil((len(values) + 1) * float(coverage)) / len(values),
    )
    return float(np.quantile(values, quantile_level, method="higher"))


def build_feature_sets(
    innovation_registry: EvidenceRegistry,
    prediction_registry: PredictionRegistry,
) -> Dict[str, Tuple[str, ...]]:
    """Return preregistered main comparisons in deterministic order."""
    controls = prediction_registry.strong_control_names
    innovation = innovation_registry.primary_feature_names
    opportunity = prediction_registry.opportunity_feature_names
    output: Dict[str, Tuple[str, ...]] = {
        "controls_only": tuple(controls),
        "innovation_plus_controls": tuple((*controls, *innovation)),
        "opportunity_only_plus_controls": tuple((*controls, *opportunity)),
        "innovation_plus_opportunity_plus_controls": tuple(
            (*controls, *innovation, *opportunity)
        ),
    }
    dimension_members: Dict[str, Tuple[str, ...]] = {}
    for dimension_id in innovation_registry.dimensions:
        members = tuple(
            metric.code_name
            for metric in innovation_registry.metrics.values()
            if metric.dimension_id == dimension_id
            and metric.code_name in innovation
        )
        if not members:
            continue
        dimension_members[dimension_id] = members
        output[f"{dimension_id.lower()}_plus_controls"] = tuple(
            (*controls, *members)
        )
    for dimension_id, members in dimension_members.items():
        remaining = tuple(name for name in innovation if name not in members)
        candidate = tuple((*controls, *remaining))
        if candidate not in output.values():
            output[f"leave_out_{dimension_id.lower()}"] = candidate
    return output


def assemble_development_frame(
    dataset_dir: Path,
    *,
    horizon: int,
    development_end_year: int,
) -> pd.DataFrame:
    """Read only development-period labels and merge publication-time views."""
    root = Path(dataset_dir)
    targets = pd.read_parquet(
        root / "targets_zero_inclusive.parquet",
        filters=[
            ("horizon", "=", int(horizon)),
            ("publication_year", "<=", int(development_end_year)),
        ],
    )
    membership = pd.read_parquet(
        root / "cohort_membership.parquet",
        filters=[
            ("horizon", "=", int(horizon)),
            ("publication_year", "<=", int(development_end_year)),
        ],
    )
    membership = membership[membership["cohort_member"].eq(1)].copy()
    target_columns = [
        "paper_id",
        "future_uptake",
        "future_field_reach",
        "future_subfield_reach",
        "future_topic_reach",
        "future_field_simpson",
        "future_topic_simpson",
    ]
    membership_columns = [
        "paper_id",
        "publication_year",
        "domain12",
        "venue_family",
        "conditional_diffusion_member",
        "reference_evidence_eligible",
        "cap_hit",
    ]
    frame = membership[membership_columns].merge(
        targets[target_columns],
        on="paper_id",
        how="inner",
        validate="one_to_one",
    )
    for path in (
        "innovation_features.parquet",
        "control_features.parquet",
        "opportunity_features.parquet",
    ):
        view = pd.read_parquet(root / path)
        duplicate = (
            set(view.columns)
            & set(frame.columns)
            - {"paper_id"}
        )
        view = view.drop(columns=sorted(duplicate), errors="ignore")
        frame = frame.merge(
            view, on="paper_id", how="left", validate="one_to_one"
        )
    if frame["publication_year"].gt(development_end_year).any():
        raise ValueError("sealed-period label entered development frame")
    if frame["future_uptake"].isna().any():
        raise ValueError("development uptake outcome is incomplete")
    return frame.sort_values(["publication_year", "paper_id"]).reset_index(
        drop=True
    )


def run_nested_development_oof(
    frame: pd.DataFrame,
    *,
    feature_sets: Mapping[str, Sequence[str]],
    horizon: int,
    n_outer: int = 5,
    n_inner: int = 4,
    parameter_grid: Sequence[Mapping[str, Any]] = MODEL_PARAMETER_GRID,
    seed: int = 20260723,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run nested expanding-year OOF without touching sealed labels."""
    if int(horizon) not in {3, 5, 8}:
        raise ValueError("v6 horizons are D3, D5, and D8")
    frame = frame.copy()
    if "horizon" in frame:
        observed_horizons = (
            pd.to_numeric(frame["horizon"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        if len(observed_horizons) != 1 or int(observed_horizons[0]) != int(
            horizon
        ):
            raise ValueError("input frame horizon differs from requested horizon")
    else:
        frame["horizon"] = int(horizon)
    categorical_names = ("domain12", "venue_family")
    outer_folds = make_expanding_year_folds(
        frame, n_splits=n_outer, initial_fraction=0.20
    )
    prediction_rows: List[pd.DataFrame] = []
    ledger_rows: List[pd.DataFrame] = []
    fold_rows: List[Dict[str, Any]] = []
    for fold in outer_folds:
        training = frame.iloc[fold.train_index].copy()
        testing = frame.iloc[fold.test_index].copy()
        training_uptake_prevalence = float(
            pd.to_numeric(
                training["future_uptake"], errors="coerce"
            ).mean()
        )
        fold_rows.append(
            {
                "outer_fold_id": fold.fold_id,
                "train_year_min": fold.train_year_min,
                "train_year_max": fold.train_year_max,
                "test_year_min": fold.test_year_min,
                "test_year_max": fold.test_year_max,
                "n_train": len(training),
                "n_test": len(testing),
            }
        )
        for model_index, (model_id, feature_names) in enumerate(
            feature_sets.items()
        ):
            missing = sorted(set(feature_names) - set(frame.columns))
            if missing:
                raise ValueError(f"{model_id} is missing features: {missing}")
            selected, inner, ledger = _select_parameters(
                training,
                feature_names=feature_names,
                categorical_names=categorical_names,
                parameter_grid=parameter_grid,
                n_inner=n_inner,
                seed=seed + fold.fold_id * 100 + model_index * 10,
            )
            (
                uptake_calibrator,
                conditional_calibrator,
                conditional_residual_quantile,
                realized_residual_quantile,
            ) = _fit_calibrators(inner)
            model = _fit_two_part(
                training,
                feature_names=feature_names,
                categorical_names=categorical_names,
                parameters=selected,
                seed=seed + fold.fold_id * 1000 + model_index,
            )
            uptake_raw, conditional_raw = model.predict_raw(testing)
            uptake_calibrated = uptake_calibrator.predict(uptake_raw)
            conditional_calibrated = conditional_calibrator.predict(
                conditional_raw
            )
            conditional_target, realized = _realized_diffusion(
                testing, model.target_transformer
            )
            expected = uptake_calibrated * conditional_calibrated
            conditional_interval_low = np.clip(
                conditional_calibrated - conditional_residual_quantile,
                0.0,
                1.0,
            )
            conditional_interval_high = np.clip(
                conditional_calibrated + conditional_residual_quantile,
                0.0,
                1.0,
            )
            realized_interval_low = np.clip(
                expected - realized_residual_quantile,
                0.0,
                1.0,
            )
            realized_interval_high = np.clip(
                expected + realized_residual_quantile,
                0.0,
                1.0,
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "paper_id": testing["paper_id"].astype(str).to_numpy(),
                        "publication_year": testing[
                            "publication_year"
                        ].to_numpy(dtype=int),
                        "domain12": testing["domain12"].astype(str).to_numpy(),
                        "horizon": int(horizon),
                        "model_id": model_id,
                        "outer_fold_id": fold.fold_id,
                        "future_uptake": testing["future_uptake"].to_numpy(
                            dtype=float
                        ),
                        "training_uptake_prevalence": (
                            training_uptake_prevalence
                        ),
                        "conditional_diffusion_member": testing[
                            "conditional_diffusion_member"
                        ].to_numpy(dtype=int),
                        "conditional_diffusion_target": conditional_target,
                        "realized_diffusion_target": realized,
                        "uptake_probability_raw": uptake_raw,
                        "uptake_probability_calibrated": uptake_calibrated,
                        "conditional_diffusion_raw": conditional_raw,
                        "conditional_diffusion_calibrated": conditional_calibrated,
                        "expected_diffusion_score": expected,
                        "conditional_interval_low": conditional_interval_low,
                        "conditional_interval_high": conditional_interval_high,
                        "realized_interval_low": realized_interval_low,
                        "realized_interval_high": realized_interval_high,
                        "selected_parameter_id": selected["parameter_id"],
                        "scope": "development_nested_temporal_oof",
                    }
                )
            )
            ledger = ledger.copy()
            ledger["outer_fold_id"] = fold.fold_id
            ledger["model_id"] = model_id
            ledger["selected"] = ledger["parameter_id"].eq(
                selected["parameter_id"]
            )
            ledger_rows.append(ledger)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    if predictions.duplicated(["paper_id", "model_id"]).any():
        raise ValueError("nested OOF produced duplicate paper/model predictions")
    ledger = pd.concat(ledger_rows, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    return predictions, ledger, folds


def bootstrap_spearman_interval(
    truth: Sequence[float],
    prediction: Sequence[float],
    *,
    iterations: int,
    seed: int,
) -> Tuple[float, float]:
    """Paper-cluster bootstrap CI for one OOF correlation."""
    truth_array = np.asarray(truth, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    valid = np.isfinite(truth_array) & np.isfinite(prediction_array)
    truth_array = truth_array[valid]
    prediction_array = prediction_array[valid]
    if len(truth_array) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(iterations), dtype=float)
    for index in range(int(iterations)):
        sample = rng.integers(0, len(truth_array), size=len(truth_array))
        estimates[index] = safe_spearman(
            truth_array[sample], prediction_array[sample]
        )
    finite = estimates[np.isfinite(estimates)]
    if not len(finite):
        return float("nan"), float("nan")
    return (
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    )


def bootstrap_paired_spearman_gain_interval(
    truth: Sequence[float],
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    iterations: int,
    seed: int,
) -> Tuple[float, float]:
    """Return a paired paper-bootstrap CI for candidate-minus-baseline rho."""
    truth_array = np.asarray(truth, dtype=float)
    candidate_array = np.asarray(candidate, dtype=float)
    baseline_array = np.asarray(baseline, dtype=float)
    valid = (
        np.isfinite(truth_array)
        & np.isfinite(candidate_array)
        & np.isfinite(baseline_array)
    )
    truth_array = truth_array[valid]
    candidate_array = candidate_array[valid]
    baseline_array = baseline_array[valid]
    if len(truth_array) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(iterations), dtype=float)
    for index in range(int(iterations)):
        sample = rng.integers(0, len(truth_array), size=len(truth_array))
        estimates[index] = safe_spearman(
            truth_array[sample], candidate_array[sample]
        ) - safe_spearman(truth_array[sample], baseline_array[sample])
    finite = estimates[np.isfinite(estimates)]
    if not len(finite):
        return float("nan"), float("nan")
    return (
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    )


def expected_calibration_error(
    outcomes: Sequence[float],
    probabilities: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    """Return equal-width binary expected calibration error."""
    outcomes_array = np.asarray(outcomes, dtype=float)
    probabilities_array = np.asarray(probabilities, dtype=float)
    valid = np.isfinite(outcomes_array) & np.isfinite(probabilities_array)
    outcomes_array = outcomes_array[valid]
    probabilities_array = np.clip(probabilities_array[valid], 0.0, 1.0)
    if not len(outcomes_array):
        return float("nan")
    boundaries = np.linspace(0.0, 1.0, int(n_bins) + 1)
    assignments = np.minimum(
        np.digitize(probabilities_array, boundaries[1:-1], right=False),
        int(n_bins) - 1,
    )
    error = 0.0
    for bin_id in range(int(n_bins)):
        member = assignments == bin_id
        if member.any():
            error += float(member.mean()) * abs(
                float(outcomes_array[member].mean())
                - float(probabilities_array[member].mean())
            )
    return float(error)


def evaluate_development_oof(
    predictions: pd.DataFrame,
    *,
    bootstrap_iterations: int = 500,
    min_domain_rows: int = 200,
    seed: int = 20260723,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate rank, uptake calibration, domain macro, and interval coverage."""
    metric_rows: List[Dict[str, Any]] = []
    domain_rows: List[Dict[str, Any]] = []
    for model_index, (model_id, group) in enumerate(
        predictions.groupby("model_id", sort=False)
    ):
        truth = group["realized_diffusion_target"].to_numpy(dtype=float)
        expected = group["expected_diffusion_score"].to_numpy(dtype=float)
        rho = safe_spearman(truth, expected)
        ci_low, ci_high = bootstrap_spearman_interval(
            truth,
            expected,
            iterations=bootstrap_iterations,
            seed=seed + model_index,
        )
        uptake = group["future_uptake"].to_numpy(dtype=float)
        probability = group["uptake_probability_calibrated"].to_numpy(
            dtype=float
        )
        uptake_valid = np.isfinite(uptake) & np.isfinite(probability)
        brier = float(
            np.mean((probability[uptake_valid] - uptake[uptake_valid]) ** 2)
        )
        training_prevalence = group[
            "training_uptake_prevalence"
        ].to_numpy(dtype=float)
        climatology_valid = uptake_valid & np.isfinite(training_prevalence)
        climatology_brier = float(
            np.mean(
                (
                    training_prevalence[climatology_valid]
                    - uptake[climatology_valid]
                )
                ** 2
            )
        )
        brier_skill = (
            float(1.0 - brier / climatology_brier)
            if climatology_brier > 0
            else float("nan")
        )
        uptake_ece = expected_calibration_error(
            uptake[uptake_valid],
            probability[uptake_valid],
            n_bins=10,
        )
        conditional = group["conditional_diffusion_member"].eq(1)
        conditional_rho = safe_spearman(
            group.loc[conditional, "conditional_diffusion_target"],
            group.loc[conditional, "conditional_diffusion_calibrated"],
        )
        interval_valid = (
            conditional
            & group["conditional_diffusion_target"].notna()
            & group["conditional_interval_low"].notna()
            & group["conditional_interval_high"].notna()
        )
        coverage = float(
            (
                group.loc[interval_valid, "conditional_diffusion_target"]
                .ge(group.loc[interval_valid, "conditional_interval_low"])
                & group.loc[interval_valid, "conditional_diffusion_target"].le(
                    group.loc[interval_valid, "conditional_interval_high"]
                )
            ).mean()
        )
        width = float(
            (
                group.loc[interval_valid, "conditional_interval_high"]
                - group.loc[interval_valid, "conditional_interval_low"]
            ).mean()
        )
        realized_interval_valid = (
            group["realized_diffusion_target"].notna()
            & group["realized_interval_low"].notna()
            & group["realized_interval_high"].notna()
        )
        realized_coverage = float(
            (
                group.loc[
                    realized_interval_valid, "realized_diffusion_target"
                ].ge(
                    group.loc[
                        realized_interval_valid, "realized_interval_low"
                    ]
                )
                & group.loc[
                    realized_interval_valid, "realized_diffusion_target"
                ].le(
                    group.loc[
                        realized_interval_valid, "realized_interval_high"
                    ]
                )
            ).mean()
        )
        realized_width = float(
            (
                group.loc[
                    realized_interval_valid, "realized_interval_high"
                ]
                - group.loc[
                    realized_interval_valid, "realized_interval_low"
                ]
            ).mean()
        )
        per_domain = []
        for domain, domain_group in group.groupby("domain12", sort=True):
            domain_rho = safe_spearman(
                domain_group["realized_diffusion_target"],
                domain_group["expected_diffusion_score"],
            )
            reportable = len(domain_group) >= int(min_domain_rows)
            domain_rows.append(
                {
                    "model_id": model_id,
                    "domain12": domain,
                    "n_rows": len(domain_group),
                    "spearman_expected": domain_rho,
                    "reportable": int(reportable),
                }
            )
            if reportable and np.isfinite(domain_rho):
                per_domain.append(domain_rho)
        metric_rows.append(
            {
                "model_id": model_id,
                "n_oof": len(group),
                "n_realized_finite": int(np.isfinite(truth).sum()),
                "spearman_expected": rho,
                "spearman_ci_low": ci_low,
                "spearman_ci_high": ci_high,
                "spearman_conditional": conditional_rho,
                "domain_macro_spearman": (
                    float(np.mean(per_domain)) if per_domain else np.nan
                ),
                "n_reportable_domains": len(per_domain),
                "uptake_brier": brier,
                "uptake_climatology_brier": climatology_brier,
                "uptake_brier_skill_score": brier_skill,
                "uptake_ece_10": uptake_ece,
                "conditional_interval_coverage_90": coverage,
                "conditional_interval_mean_width": width,
                "realized_interval_coverage_90": realized_coverage,
                "realized_interval_mean_width": realized_width,
                "scope": "development_nested_temporal_oof",
            }
        )
    metrics = pd.DataFrame(metric_rows)
    controls = metrics.loc[
        metrics["model_id"].eq("controls_only"), "spearman_expected"
    ]
    baseline = float(controls.iloc[0]) if len(controls) else np.nan
    metrics["gain_over_controls"] = metrics["spearman_expected"] - baseline
    metrics["gain_over_controls_ci_low"] = np.nan
    metrics["gain_over_controls_ci_high"] = np.nan
    if len(controls):
        controls_predictions = predictions.loc[
            predictions["model_id"].eq("controls_only"),
            [
                "paper_id",
                "outer_fold_id",
                "realized_diffusion_target",
                "expected_diffusion_score",
            ],
        ].rename(
            columns={
                "realized_diffusion_target": "baseline_truth",
                "expected_diffusion_score": "baseline_prediction",
            }
        )
        for model_index, model_id in enumerate(metrics["model_id"]):
            if model_id == "controls_only":
                metrics.loc[
                    metrics["model_id"].eq(model_id),
                    [
                        "gain_over_controls_ci_low",
                        "gain_over_controls_ci_high",
                    ],
                ] = 0.0
                continue
            candidate = predictions.loc[
                predictions["model_id"].eq(model_id),
                [
                    "paper_id",
                    "outer_fold_id",
                    "realized_diffusion_target",
                    "expected_diffusion_score",
                ],
            ].merge(
                controls_predictions,
                on=["paper_id", "outer_fold_id"],
                how="inner",
                validate="one_to_one",
            )
            both_truth = (
                candidate["realized_diffusion_target"].notna()
                & candidate["baseline_truth"].notna()
            )
            if not np.allclose(
                candidate.loc[
                    both_truth, "realized_diffusion_target"
                ].to_numpy(dtype=float),
                candidate.loc[both_truth, "baseline_truth"].to_numpy(
                    dtype=float
                ),
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError("paired model targets disagree")
            gain_low, gain_high = (
                bootstrap_paired_spearman_gain_interval(
                    candidate["realized_diffusion_target"],
                    candidate["expected_diffusion_score"],
                    candidate["baseline_prediction"],
                    iterations=bootstrap_iterations,
                    seed=seed + 10_000 + model_index,
                )
            )
            metrics.loc[
                metrics["model_id"].eq(model_id),
                [
                    "gain_over_controls_ci_low",
                    "gain_over_controls_ci_high",
                ],
            ] = (gain_low, gain_high)
    return metrics, pd.DataFrame(domain_rows)


def evaluate_temporal_folds(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report every outer test block and identify the latest development block."""
    rows: List[Dict[str, Any]] = []
    for (model_id, fold_id), group in predictions.groupby(
        ["model_id", "outer_fold_id"], sort=True
    ):
        truth = group["realized_diffusion_target"].to_numpy(dtype=float)
        expected = group["expected_diffusion_score"].to_numpy(dtype=float)
        uptake = group["future_uptake"].to_numpy(dtype=float)
        probability = group["uptake_probability_calibrated"].to_numpy(
            dtype=float
        )
        conditional = group["conditional_diffusion_member"].eq(1)
        rows.append(
            {
                "model_id": model_id,
                "outer_fold_id": int(fold_id),
                "test_year_min": int(group["publication_year"].min()),
                "test_year_max": int(group["publication_year"].max()),
                "n_rows": len(group),
                "n_realized_finite": int(np.isfinite(truth).sum()),
                "spearman_expected": safe_spearman(truth, expected),
                "spearman_conditional": safe_spearman(
                    group.loc[
                        conditional, "conditional_diffusion_target"
                    ],
                    group.loc[
                        conditional, "conditional_diffusion_calibrated"
                    ],
                ),
                "uptake_brier": float(
                    np.mean((probability - uptake) ** 2)
                ),
            }
        )
    output = pd.DataFrame(rows)
    if len(output):
        output["latest_development_fold"] = (
            output["test_year_max"].eq(output["test_year_max"].max()).astype(int)
        )
    return output


def write_development_run(
    output_dir: Path,
    *,
    predictions: pd.DataFrame,
    model_ledger: pd.DataFrame,
    folds: pd.DataFrame,
    metrics: pd.DataFrame,
    domain_metrics: pd.DataFrame,
    temporal_fold_metrics: pd.DataFrame,
    acceptance_gates: pd.DataFrame,
    lineage: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Write one immutable-style development run with content hashes."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "oof_predictions": root / "development_oof_predictions.parquet",
        "model_ledger": root / "development_model_ledger.parquet",
        "folds": root / "development_temporal_folds.parquet",
        "metrics": root / "development_metrics.csv",
        "domain_metrics": root / "development_domain_metrics.csv",
        "temporal_fold_metrics": root
        / "development_temporal_fold_metrics.csv",
        "acceptance_gates": root / "development_acceptance_gates.csv",
    }
    predictions.to_parquet(paths["oof_predictions"], index=False)
    model_ledger.to_parquet(paths["model_ledger"], index=False)
    folds.to_parquet(paths["folds"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    domain_metrics.to_csv(paths["domain_metrics"], index=False)
    temporal_fold_metrics.to_csv(
        paths["temporal_fold_metrics"], index=False
    )
    acceptance_gates.to_csv(paths["acceptance_gates"], index=False)
    manifest = {
        "artifact_kind": "aspr_v6_development_nested_oof",
        "model_protocol_version": MODEL_PROTOCOL_VERSION,
        "sealed_holdout_accessed": False,
        "lineage": dict(lineage),
        "outputs": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    manifest["artifact_id"] = _canonical_hash(manifest)
    (root / "development_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "MODEL_PARAMETER_GRID",
    "MODEL_PROTOCOL_VERSION",
    "CalibratedTwoPartBundle",
    "TemporalFold",
    "assemble_development_frame",
    "bootstrap_paired_spearman_gain_interval",
    "bootstrap_spearman_interval",
    "build_feature_sets",
    "evaluate_development_oof",
    "evaluate_temporal_folds",
    "expected_calibration_error",
    "fit_calibrated_final_model",
    "make_expanding_year_folds",
    "run_nested_development_oof",
    "realized_diffusion_target",
    "safe_spearman",
    "select_final_parameters",
    "write_development_run",
]
