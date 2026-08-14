"""Fold-local feature transforms and rank models for Nature multi-horizon V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import SplineTransformer

from .contracts import AUXILIARY_FEATURES, CORE_FEATURES, FeatureSpec, MECHANISM_FEATURES
from .splits import make_stratification_labels


MODEL_COMPLEXITY: Dict[str, int] = {
    "domain_year_only": 0,
    "bibliographic_aux10_ridge": 1,
    "mechanism5_equal_weight": 1,
    "mechanism5_simplex": 2,
    "gam18": 3,
    "hgb18": 4,
    "rank_blend": 5,
}
BASELINE_MODEL_IDS: Tuple[str, ...] = (
    "domain_year_only",
    "bibliographic_aux10_ridge",
    "mechanism5_equal_weight",
    "mechanism5_simplex",
)
PERFORMANCE_MODEL_IDS: Tuple[str, ...] = (
    "mechanism5_simplex",
    "gam18",
    "hgb18",
    "rank_blend",
)


def safe_numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Read one numeric column without silently substituting future fields."""

    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)


def percentile_rank(values: Sequence[float]) -> np.ndarray:
    """Rank finite values to [0, 1], retaining missing values."""

    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(method="average", pct=True, na_option="keep").to_numpy(float)


def empirical_cdf(values: Sequence[float], reference: Sequence[float]) -> np.ndarray:
    """Map values to the empirical CDF fitted on a reference sample."""

    values_array = np.asarray(values, dtype=float)
    ref = np.sort(np.asarray(reference, dtype=float)[np.isfinite(reference)])
    result = np.full(values_array.shape, np.nan, dtype=float)
    valid = np.isfinite(values_array)
    if not len(ref):
        return result
    result[valid] = np.searchsorted(ref, values_array[valid], side="right") / float(len(ref))
    return result


def project_simplex(values: Sequence[float]) -> np.ndarray:
    """Euclidean projection onto the non-negative unit simplex."""

    vector = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if not len(vector):
        return vector
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered)
    candidates = ordered * np.arange(1, len(vector) + 1) > cumulative - 1.0
    if not candidates.any():
        return np.ones(len(vector), dtype=float) / len(vector)
    rho = int(np.flatnonzero(candidates)[-1])
    threshold = (cumulative[rho] - 1.0) / float(rho + 1)
    result = np.maximum(vector - threshold, 0.0)
    return result / result.sum() if result.sum() else np.ones(len(vector)) / len(vector)


@dataclass(frozen=True)
class TransformedFeatures:
    """All fold-local model matrices generated from one fitted transformer."""

    core8: np.ndarray
    mechanism5: np.ndarray
    auxiliary10: np.ndarray
    full18: np.ndarray
    mechanism_names: Tuple[str, ...]


class FoldLocalFeatureTransformer:
    """Training-fold imputation, core percentiles, and auxiliary scaling."""

    def __init__(self, feature_spec: Optional[FeatureSpec] = None) -> None:
        self.feature_spec = feature_spec or FeatureSpec()
        self.core_medians_: Dict[str, float] = {}
        self.core_references_: Dict[str, np.ndarray] = {}
        self.aux_medians_: Dict[str, float] = {}
        self.aux_scales_: Dict[str, float] = {}
        self.is_fitted_ = False

    def fit(self, frame: pd.DataFrame) -> "FoldLocalFeatureTransformer":
        required = set(self.feature_spec.prediction_features)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing registered prediction features: {missing}")
        for column in self.feature_spec.core_features:
            values = safe_numeric(frame, column)
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if len(finite) else 0.0
            filled = np.where(np.isfinite(values), values, median)
            self.core_medians_[column] = median
            self.core_references_[column] = np.sort(filled)
        for column in self.feature_spec.auxiliary_features:
            values = safe_numeric(frame, column)
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if len(finite) else 0.0
            q25, q75 = np.percentile(finite, [25, 75]) if len(finite) else (0.0, 0.0)
            scale = float(q75 - q25)
            if not np.isfinite(scale) or scale <= 1e-9:
                scale = float(np.std(finite)) if len(finite) else 1.0
            self.aux_medians_[column] = median
            self.aux_scales_[column] = scale if np.isfinite(scale) and scale > 1e-9 else 1.0
        self.is_fitted_ = True
        return self

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError("FoldLocalFeatureTransformer must be fitted first")

    def transform(self, frame: pd.DataFrame) -> TransformedFeatures:
        self._check_fitted()
        missing = sorted(
            set(self.feature_spec.prediction_features) - set(frame.columns)
        )
        if missing:
            raise ValueError(
                f"Missing registered prediction features at transform: {missing}"
            )
        core_columns = []
        for column in self.feature_spec.core_features:
            values = safe_numeric(frame, column)
            values = np.where(np.isfinite(values), values, self.core_medians_[column])
            core_columns.append(empirical_cdf(values, self.core_references_[column]))
        core = np.column_stack(core_columns) if core_columns else np.empty((len(frame), 0))
        core_lookup = {name: core[:, idx] for idx, name in enumerate(self.feature_spec.core_features)}
        mechanisms = np.column_stack(
            [
                np.mean(np.column_stack([core_lookup[item] for item in members]), axis=1)
                for members in self.feature_spec.mechanisms.values()
            ]
        )
        aux_columns = []
        for column in self.feature_spec.auxiliary_features:
            values = safe_numeric(frame, column)
            values = np.where(np.isfinite(values), values, self.aux_medians_[column])
            aux_columns.append((values - self.aux_medians_[column]) / self.aux_scales_[column])
        auxiliary = np.column_stack(aux_columns) if aux_columns else np.empty((len(frame), 0))
        return TransformedFeatures(
            core8=np.nan_to_num(core),
            mechanism5=np.nan_to_num(mechanisms),
            auxiliary10=np.nan_to_num(auxiliary),
            full18=np.nan_to_num(np.column_stack([core, auxiliary])),
            mechanism_names=tuple(self.feature_spec.mechanisms),
        )

    def fit_transform(self, frame: pd.DataFrame) -> TransformedFeatures:
        return self.fit(frame).transform(frame)


def _pair_differences(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    max_pairs: int,
    groups: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    valid = np.flatnonzero(np.isfinite(y))
    if len(valid) < 2:
        return np.empty((0, x.shape[1]))
    if groups is None:
        group_values = np.full(len(y), "all", dtype=object)
    else:
        group_values = pd.Series(groups).astype("string").fillna("missing").to_numpy()
    pools = [indices for _, indices in pd.Series(valid).groupby(group_values[valid]).groups.items() if len(indices) >= 2]
    # pandas group indices above are positions within ``valid``; map them back.
    pools = [valid[np.asarray(pool, dtype=int)] for pool in pools]
    if not pools:
        pools = [valid]
    allocations = np.maximum(1, np.floor(max_pairs * np.array([len(p) for p in pools]) / sum(len(p) for p in pools)).astype(int))
    rows = []
    for pool, count in zip(pools, allocations):
        first = rng.choice(pool, size=count, replace=True)
        second = rng.choice(pool, size=count, replace=True)
        keep = (first != second) & np.isfinite(y[first]) & np.isfinite(y[second]) & (y[first] != y[second])
        first, second = first[keep], second[keep]
        swap = y[first] < y[second]
        high = np.where(swap, second, first)
        low = np.where(swap, first, second)
        rows.append(x[high] - x[low])
    return np.vstack(rows) if rows else np.empty((0, x.shape[1]))


class SimplexPairwiseRanker:
    """Non-negative pairwise logistic ranker whose weights sum to one."""

    def __init__(
        self,
        *,
        l2: float = 0.01,
        epochs: int = 250,
        learning_rate: float = 0.05,
        max_pairs: int = 50_000,
        seed: int = 20260710,
    ) -> None:
        self.l2 = float(l2)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.max_pairs = int(max_pairs)
        self.seed = int(seed)
        self.weights_: Optional[np.ndarray] = None

    def fit(
        self,
        x: np.ndarray,
        y: Sequence[float],
        *,
        groups: Optional[Sequence[Any]] = None,
    ) -> "SimplexPairwiseRanker":
        x = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)
        differences = _pair_differences(
            x,
            y_array,
            seed=self.seed,
            max_pairs=self.max_pairs,
            groups=groups,
        )
        weights = np.ones(x.shape[1], dtype=float) / max(1, x.shape[1])
        if len(differences):
            for _ in range(self.epochs):
                margins = np.clip(differences @ weights, -40.0, 40.0)
                coefficient = -1.0 / (1.0 + np.exp(margins))
                gradient = (differences * coefficient[:, None]).mean(axis=0) + self.l2 * weights
                weights = project_simplex(weights - self.learning_rate * gradient)
        self.weights_ = weights
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("SimplexPairwiseRanker must be fitted first")
        return np.asarray(x, dtype=float) @ self.weights_


class EqualWeightRanker:
    """Fixed equal-weight mechanism baseline."""

    def fit(self, x: np.ndarray, y: Sequence[float], **_: Any) -> "EqualWeightRanker":
        self.n_features_ = int(np.asarray(x).shape[1])
        return self

    @property
    def weights_(self) -> np.ndarray:
        return np.ones(self.n_features_) / max(1, self.n_features_)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float).mean(axis=1)


class RidgeRanker:
    """Ridge regression on target ranks."""

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = float(alpha)
        self.model_ = Ridge(alpha=self.alpha)

    def fit(self, x: np.ndarray, y: Sequence[float], **_: Any) -> "RidgeRanker":
        self.model_.fit(np.asarray(x), percentile_rank(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model_.predict(np.asarray(x))


class GAMRanker:
    """Spline-additive ridge model (a compact, deterministic GAM)."""

    def __init__(self, *, alpha: float = 10.0, n_knots: int = 4, degree: int = 2) -> None:
        self.spline_ = SplineTransformer(n_knots=int(n_knots), degree=int(degree), include_bias=False)
        self.ridge_ = Ridge(alpha=float(alpha))

    def fit(self, x: np.ndarray, y: Sequence[float], **_: Any) -> "GAMRanker":
        expanded = self.spline_.fit_transform(np.asarray(x, dtype=float))
        self.ridge_.fit(expanded, percentile_rank(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.ridge_.predict(self.spline_.transform(np.asarray(x, dtype=float)))


class HGBRanker:
    """Shallow histogram gradient boosting fitted to target ranks."""

    def __init__(
        self,
        *,
        max_depth: int = 3,
        max_leaf_nodes: int = 15,
        learning_rate: float = 0.05,
        l2_regularization: float = 10.0,
        max_iter: int = 150,
        seed: int = 20260710,
    ) -> None:
        self.model_ = HistGradientBoostingRegressor(
            max_depth=int(max_depth),
            max_leaf_nodes=int(max_leaf_nodes),
            learning_rate=float(learning_rate),
            l2_regularization=float(l2_regularization),
            max_iter=int(max_iter),
            early_stopping=False,
            random_state=int(seed),
        )

    def fit(self, x: np.ndarray, y: Sequence[float], **_: Any) -> "HGBRanker":
        self.model_.fit(np.asarray(x, dtype=float), percentile_rank(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model_.predict(np.asarray(x, dtype=float))


class DomainYearOnlyRanker:
    """Publication-time calibration-only baseline."""

    def fit(self, frame: pd.DataFrame, y: Sequence[float]) -> "DomainYearOnlyRanker":
        target = percentile_rank(y)
        domains = frame.get("domain12", pd.Series("unknown", index=frame.index)).astype("string")
        self.global_ = float(np.nanmean(target))
        self.domain_ = pd.Series(target).groupby(domains.reset_index(drop=True)).mean().to_dict()
        years = pd.to_numeric(frame.get("publication_year", pd.Series(np.nan, index=frame.index)), errors="coerce")
        self.year_center_ = float(years.median()) if years.notna().any() else 0.0
        centered = years.fillna(self.year_center_).to_numpy(float) - self.year_center_
        residual = target - np.array([self.domain_.get(str(value), self.global_) for value in domains])
        design = np.column_stack([centered, centered**2])
        self.year_model_ = Ridge(alpha=10.0).fit(design, residual)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        domains = frame.get("domain12", pd.Series("unknown", index=frame.index)).astype("string")
        base = np.array([self.domain_.get(str(value), self.global_) for value in domains], dtype=float)
        years = pd.to_numeric(frame.get("publication_year", pd.Series(np.nan, index=frame.index)), errors="coerce")
        centered = years.fillna(self.year_center_).to_numpy(float) - self.year_center_
        return base + self.year_model_.predict(np.column_stack([centered, centered**2]))


@dataclass
class FittedCandidateModel:
    """Unified prediction interface for all candidate families."""

    model_id: str
    transformer: Optional[FoldLocalFeatureTransformer]
    estimator: Any
    feature_view: str
    feature_spec: FeatureSpec

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        if self.transformer is None:
            raise RuntimeError("This candidate consumes a frame, not a feature matrix")
        transformed = self.transformer.transform(frame)
        return getattr(transformed, self.feature_view)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model_id == "domain_year_only":
            return self.estimator.predict(frame)
        return np.asarray(self.estimator.predict(self._matrix(frame)), dtype=float)

    def mechanism_channels(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.transformer is None:
            transformer = FoldLocalFeatureTransformer(self.feature_spec).fit(frame)
        else:
            transformer = self.transformer
        transformed = transformer.transform(frame)
        return pd.DataFrame(transformed.mechanism5, columns=transformed.mechanism_names, index=frame.index)


class RankBlendEstimator:
    """Non-negative blend of independently fitted base rank models."""

    def __init__(
        self,
        base_models: Mapping[str, FittedCandidateModel],
        weights: Sequence[float],
        prediction_references: Mapping[str, Sequence[float]],
    ) -> None:
        self.base_models = dict(base_models)
        self.names = tuple(base_models)
        self.weights_ = project_simplex(weights)
        self.prediction_references = {
            name: np.asarray(prediction_references[name], dtype=float) for name in self.names
        }

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        columns = [
            empirical_cdf(self.base_models[name].predict(frame), self.prediction_references[name])
            for name in self.names
        ]
        return np.nan_to_num(np.column_stack(columns)) @ self.weights_


@dataclass
class FittedRankBlendModel(FittedCandidateModel):
    estimator: RankBlendEstimator

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_frame(frame)

    def mechanism_channels(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.estimator.base_models["mechanism5_simplex"].mechanism_channels(frame)


def pair_groups(frame: pd.DataFrame, width: int = 5) -> np.ndarray:
    domains = frame.get("domain12", pd.Series("unknown", index=frame.index)).astype("string").fillna("unknown")
    years = pd.to_numeric(frame.get("publication_year", pd.Series(np.nan, index=frame.index)), errors="coerce")
    bins = (np.floor(years / width) * width).astype("Int64").astype("string").fillna("missing")
    return (domains + "|" + bins).to_numpy()


def _fit_non_blend_model(
    model_id: str,
    frame: pd.DataFrame,
    y: np.ndarray,
    *,
    feature_spec: FeatureSpec,
    seed: int,
) -> FittedCandidateModel:
    if model_id == "domain_year_only":
        estimator = DomainYearOnlyRanker().fit(frame, y)
        return FittedCandidateModel(model_id, None, estimator, "frame", feature_spec)
    transformer = FoldLocalFeatureTransformer(feature_spec)
    transformed = transformer.fit_transform(frame)
    if model_id == "bibliographic_aux10_ridge":
        estimator, view = RidgeRanker(alpha=10.0), "auxiliary10"
    elif model_id == "mechanism5_equal_weight":
        estimator, view = EqualWeightRanker(), "mechanism5"
    elif model_id == "mechanism5_simplex":
        estimator, view = SimplexPairwiseRanker(seed=seed), "mechanism5"
    elif model_id == "gam18":
        estimator, view = GAMRanker(alpha=10.0), "full18"
    elif model_id == "hgb18":
        estimator, view = HGBRanker(seed=seed), "full18"
    else:
        raise ValueError(f"Unknown model_id: {model_id}")
    matrix = getattr(transformed, view)
    estimator.fit(matrix, y, groups=pair_groups(frame))
    return FittedCandidateModel(model_id, transformer, estimator, view, feature_spec)


def fit_candidate_model(
    model_id: str,
    frame: pd.DataFrame,
    target: Sequence[float] | str,
    *,
    feature_spec: Optional[FeatureSpec] = None,
    seed: int = 20260710,
    blend_folds: int = 4,
    blend_weights: Optional[Sequence[float]] = None,
) -> FittedCandidateModel:
    """Fit one named model using transformations learned only from ``frame``."""

    spec = feature_spec or FeatureSpec()
    y = safe_numeric(frame, target) if isinstance(target, str) else np.asarray(target, dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < 5:
        raise ValueError("At least five finite training targets are required")
    train = frame.loc[valid].reset_index(drop=True)
    y_train = y[valid]
    if model_id != "rank_blend":
        return _fit_non_blend_model(model_id, train, y_train, feature_spec=spec, seed=seed)

    base_names = ("mechanism5_simplex", "gam18", "hgb18")
    weights = np.asarray(blend_weights, dtype=float) if blend_weights is not None else None
    if weights is None:
        n_folds = max(2, min(int(blend_folds), len(train)))
        oof = np.full((len(train), len(base_names)), np.nan)
        labels, _, _ = make_stratification_labels(train, n_splits=n_folds)
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold_id, (fit_idx, valid_idx) in enumerate(
            splitter.split(np.zeros(len(train), dtype=np.int8), labels),
            start=1,
        ):
            for col, name in enumerate(base_names):
                fitted = _fit_non_blend_model(
                    name,
                    train.iloc[fit_idx].reset_index(drop=True),
                    y_train[fit_idx],
                    feature_spec=spec,
                    seed=seed + fold_id * 101 + col,
                )
                oof[valid_idx, col] = fitted.predict(train.iloc[valid_idx])
        ranked = np.column_stack([percentile_rank(oof[:, idx]) for idx in range(oof.shape[1])])
        optimizer = SimplexPairwiseRanker(seed=seed + 9901, max_pairs=20_000).fit(ranked, y_train)
        weights = optimizer.weights_
    base_models = {
        name: _fit_non_blend_model(name, train, y_train, feature_spec=spec, seed=seed + idx * 1009)
        for idx, name in enumerate(base_names)
    }
    references = {name: model.predict(train) for name, model in base_models.items()}
    estimator = RankBlendEstimator(base_models, weights, references)
    return FittedRankBlendModel("rank_blend", None, estimator, "blend", spec)


class DomainYearCalibrator:
    """Fold-local additive domain/year calibration of a scientific score."""

    def __init__(self, *, alpha: float = 10.0) -> None:
        self.alpha = float(alpha)
        self.is_fitted_ = False

    def fit(
        self,
        frame: pd.DataFrame,
        raw_score: Sequence[float],
        target: Sequence[float],
    ) -> "DomainYearCalibrator":
        raw = np.asarray(raw_score, dtype=float)
        target_rank = percentile_rank(target)
        valid = np.isfinite(raw) & np.isfinite(target_rank)
        train = frame.loc[valid].reset_index(drop=True)
        raw = raw[valid]
        target_rank = target_rank[valid]
        self.score_reference_ = np.sort(raw)
        score_pct = empirical_cdf(raw, self.score_reference_)
        residual = target_rank - score_pct
        domains = train.get("domain12", pd.Series("unknown", index=train.index)).astype("string").fillna("unknown")
        domain_series = pd.Series(residual).groupby(domains).mean()
        self.domain_corrections_ = {str(k): float(v) for k, v in domain_series.items()}
        domain_adjusted = residual - np.array([self.domain_corrections_.get(str(value), 0.0) for value in domains])
        years = pd.to_numeric(train.get("publication_year", pd.Series(np.nan, index=train.index)), errors="coerce")
        self.year_median_ = float(years.median()) if years.notna().any() else 0.0
        year_values = years.fillna(self.year_median_).to_numpy(float).reshape(-1, 1)
        unique_years = np.unique(year_values)
        self.year_spline_: Optional[SplineTransformer]
        self.year_model_: Optional[Ridge]
        if len(unique_years) >= 2:
            knots = max(2, min(4, len(unique_years)))
            self.year_spline_ = SplineTransformer(n_knots=knots, degree=min(2, knots - 1), include_bias=False)
            design = self.year_spline_.fit_transform(year_values)
            self.year_model_ = Ridge(alpha=self.alpha).fit(design, domain_adjusted)
        else:
            self.year_spline_ = None
            self.year_model_ = None
        self.is_fitted_ = True
        self.calibrated_reference_ = np.sort(self.predict(train, raw))
        return self

    def predict(self, frame: pd.DataFrame, raw_score: Sequence[float]) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("DomainYearCalibrator must be fitted first")
        raw = np.asarray(raw_score, dtype=float)
        calibrated = empirical_cdf(raw, self.score_reference_)
        domains = frame.get("domain12", pd.Series("unknown", index=frame.index)).astype("string").fillna("unknown")
        calibrated += np.array([self.domain_corrections_.get(str(value), 0.0) for value in domains])
        if self.year_spline_ is not None and self.year_model_ is not None:
            years = pd.to_numeric(frame.get("publication_year", pd.Series(np.nan, index=frame.index)), errors="coerce")
            values = years.fillna(self.year_median_).to_numpy(float).reshape(-1, 1)
            calibrated += self.year_model_.predict(self.year_spline_.transform(values))
        return calibrated


class TargetResidualizer:
    """Label-side future-citer adjustment fitted on a train fold.

    Domain/year adjustment is deliberately opt-in and exists only for a
    registered sensitivity analysis.  The headline target adjusts solely for
    ``log1p(n_future_citers)``.
    """

    def __init__(
        self,
        *,
        future_citers_col: str = "n_future_citers",
        adjust_domain_year: bool = False,
    ) -> None:
        self.future_citers_col = future_citers_col
        self.adjust_domain_year = bool(adjust_domain_year)
        self.is_fitted_ = False

    def _design(self, frame: pd.DataFrame, *, fit: bool) -> np.ndarray:
        citations = np.log1p(np.maximum(safe_numeric(frame, self.future_citers_col), 0.0))
        years = safe_numeric(frame, "publication_year")
        if fit:
            self.citation_median_ = float(np.nanmedian(citations)) if np.isfinite(citations).any() else 0.0
            self.year_median_ = float(np.nanmedian(years)) if np.isfinite(years).any() else 0.0
            domains = frame.get("domain12", pd.Series("unknown", index=frame.index)).astype("string").fillna("unknown")
            self.domains_ = tuple(sorted(str(value) for value in domains.unique())) if self.adjust_domain_year else ()
        citations = np.where(np.isfinite(citations), citations, self.citation_median_)
        citation_values = citations.reshape(-1, 1)
        if fit:
            unique_citations = np.unique(citation_values)
            if len(unique_citations) >= 3:
                knots = max(3, min(5, len(unique_citations)))
                self.citation_spline_ = SplineTransformer(
                    n_knots=knots,
                    degree=min(2, knots - 1),
                    include_bias=False,
                )
                citation_design = self.citation_spline_.fit_transform(
                    citation_values
                )
            else:
                self.citation_spline_ = None
                citation_design = citation_values
        elif self.citation_spline_ is not None:
            citation_design = self.citation_spline_.transform(citation_values)
        else:
            citation_design = citation_values
        if not self.adjust_domain_year:
            return citation_design
        years = np.where(np.isfinite(years), years, self.year_median_) - self.year_median_
        domains = frame.get("domain12", pd.Series("unknown", index=frame.index)).astype("string").fillna("unknown")
        one_hot = np.column_stack([(domains == value).to_numpy(float) for value in self.domains_])
        return np.column_stack([citation_design, years, years**2, one_hot])

    def fit(self, frame: pd.DataFrame, target: Sequence[float]) -> "TargetResidualizer":
        y = np.asarray(target, dtype=float)
        valid = np.isfinite(y)
        train = frame.loc[valid].reset_index(drop=True)
        y = y[valid]
        design = self._design(train, fit=True)
        self.model_ = Ridge(alpha=10.0).fit(design, y)
        residual = y - self.model_.predict(design)
        self.residual_reference_ = np.sort(residual[np.isfinite(residual)])
        self.is_fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame, target: Sequence[float]) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("TargetResidualizer must be fitted first")
        y = np.asarray(target, dtype=float)
        residual = y - self.model_.predict(self._design(frame, fit=False))
        return empirical_cdf(residual, self.residual_reference_)


# Concise aliases used by configuration and figure labels.
Mechanism5Simplex = SimplexPairwiseRanker
GAM18 = GAMRanker
HGB18 = HGBRanker
