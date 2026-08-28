"""Fold-local v6 two-part outcome construction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from gear.graph_prior_contracts import TargetHeadProvenance

BREADTH_COMPONENTS: tuple[str, ...] = (
    "future_field_reach",
    "future_subfield_reach",
    "future_topic_reach",
)
EVENNESS_COMPONENTS: tuple[str, ...] = (
    "future_field_simpson",
    "future_topic_simpson",
)
DIFFUSION_COMPONENTS: tuple[str, ...] = (
    *BREADTH_COMPONENTS,
    *EVENNESS_COMPONENTS,
)
PERTURBATION_COMPONENTS: tuple[str, ...] = (
    "boundary_expansion",
    "community_mixing_change",
    "dependency_displacement",
    "path_shortening",
)


def _mid_distribution_percentile(
    values: Sequence[float] | np.ndarray,
    reference: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Map values by 0.5*(F_train(x-)+F_train(x))."""
    array = np.asarray(values, dtype=float)
    ordered = np.sort(
        np.asarray(reference, dtype=float)[
            np.isfinite(np.asarray(reference, dtype=float))
        ]
    )
    output = np.full(array.shape, np.nan, dtype=float)
    valid = np.isfinite(array)
    if not len(ordered):
        return output
    left = np.searchsorted(ordered, array[valid], side="left")
    right = np.searchsorted(ordered, array[valid], side="right")
    output[valid] = (left + right) / (2.0 * len(ordered))
    return output


class FoldLocalDiffusionTargetTransformer:
    """Fit D3/D5/D8 component references on positive training uptake only."""

    def __init__(
        self,
        *,
        breadth_weight: float = 0.5,
        evenness_weight: float = 0.5,
    ) -> None:
        if breadth_weight < 0 or evenness_weight < 0:
            raise ValueError("diffusion block weights cannot be negative")
        total = float(breadth_weight + evenness_weight)
        if not np.isclose(total, 1.0):
            raise ValueError("diffusion block weights must sum to one")
        self.breadth_weight = float(breadth_weight)
        self.evenness_weight = float(evenness_weight)
        self.references_: dict[str, np.ndarray] = {}
        self.horizon_: int | None = None
        self.is_fitted_ = False

    @staticmethod
    def _check_columns(frame: pd.DataFrame) -> None:
        required = {"horizon", "future_uptake", *DIFFUSION_COMPONENTS}
        missing = sorted(required - set(frame))
        if missing:
            raise ValueError(f"diffusion components are missing columns: {missing}")

    def fit(self, training_frame: pd.DataFrame) -> FoldLocalDiffusionTargetTransformer:
        """Fit references without inspecting validation or test outcomes."""
        self._check_columns(training_frame)
        horizons = (
            pd.to_numeric(training_frame["horizon"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
        )
        if len(horizons) != 1 or int(horizons[0]) not in {3, 5, 8}:
            raise ValueError("fit requires exactly one registered horizon")
        positive = training_frame[
            pd.to_numeric(training_frame["future_uptake"], errors="coerce").eq(1)
        ]
        if positive.empty:
            raise ValueError("positive-uptake training rows are required")
        references: dict[str, np.ndarray] = {}
        for column in DIFFUSION_COMPONENTS:
            values = pd.to_numeric(positive[column], errors="coerce").to_numpy(
                dtype=float
            )
            if column in BREADTH_COMPONENTS:
                values = np.log1p(values)
            finite = values[np.isfinite(values)]
            if not len(finite):
                raise ValueError(f"training component has no finite values: {column}")
            references[column] = np.sort(finite)
        self.references_ = references
        self.horizon_ = int(horizons[0])
        self.is_fitted_ = True
        return self

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        conditional_only: bool = True,
    ) -> pd.DataFrame:
        """Apply only training references and preserve component missingness."""
        if not self.is_fitted_ or self.horizon_ is None:
            raise RuntimeError("target transformer must be fitted first")
        self._check_columns(frame)
        horizons = pd.to_numeric(frame["horizon"], errors="coerce")
        if not horizons.dropna().astype(int).eq(self.horizon_).all():
            raise ValueError("transform horizon differs from fitted horizon")
        ranked: dict[str, np.ndarray] = {}
        for column in DIFFUSION_COMPONENTS:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            if column in BREADTH_COMPONENTS:
                values = np.log1p(values)
            ranked[column] = _mid_distribution_percentile(
                values, self.references_[column]
            )
        breadth_matrix = np.column_stack(
            [ranked[column] for column in BREADTH_COMPONENTS]
        )
        evenness_matrix = np.column_stack(
            [ranked[column] for column in EVENNESS_COMPONENTS]
        )
        all_finite = np.isfinite(
            np.column_stack([breadth_matrix, evenness_matrix])
        ).all(axis=1)
        breadth = np.full(len(frame), np.nan, dtype=float)
        evenness = np.full(len(frame), np.nan, dtype=float)
        breadth[all_finite] = breadth_matrix[all_finite].mean(axis=1)
        evenness[all_finite] = evenness_matrix[all_finite].mean(axis=1)
        diffusion = self.breadth_weight * breadth + self.evenness_weight * evenness
        if conditional_only:
            positive = pd.to_numeric(frame["future_uptake"], errors="coerce").eq(1)
            diffusion[~positive.to_numpy()] = np.nan
            breadth[~positive.to_numpy()] = np.nan
            evenness[~positive.to_numpy()] = np.nan
        return pd.DataFrame(
            {
                "rgpm_d_breadth_fold": breadth,
                "rgpm_d_evenness_fold": evenness,
                "rgpm_d_fold": diffusion,
                "target_reference_horizon": self.horizon_,
                "target_transform_scope": "training_fold_only",
            },
            index=frame.index,
        )

    def fit_transform(self, training_frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(training_frame).transform(training_frame)


class FoldLocalExcessDiffusionTransformer:
    """Residualize diffusion against fold-local exposure and opportunity."""

    REQUIRED: ClassVar[set[str]] = {
        "rgpm_d_fold",
        "n_future_citers",
        "domain12",
        "publication_year",
    }

    def fit(self, training_frame: pd.DataFrame) -> FoldLocalExcessDiffusionTransformer:
        missing = sorted(self.REQUIRED - set(training_frame))
        if missing:
            raise ValueError(f"excess diffusion inputs are missing: {missing}")
        target = pd.to_numeric(training_frame["rgpm_d_fold"], errors="coerce")
        design = self._design(training_frame, fitting=True)
        valid = target.notna() & np.isfinite(design.to_numpy(dtype=float)).all(axis=1)
        if int(valid.sum()) < 2:
            raise ValueError("insufficient finite training rows for null model")
        matrix = design.loc[valid].to_numpy(dtype=float)
        outcome = target.loc[valid].to_numpy(dtype=float)
        penalty = 1e-6 * np.eye(matrix.shape[1])
        penalty[0, 0] = 0.0
        self.beta_ = np.linalg.pinv(matrix.T @ matrix + penalty) @ matrix.T @ outcome
        residual = outcome - matrix @ self.beta_
        self.residual_reference_ = np.sort(residual[np.isfinite(residual)])
        self.is_fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("excess diffusion transformer must be fitted first")
        missing = sorted(self.REQUIRED - set(frame))
        if missing:
            raise ValueError(f"excess diffusion inputs are missing: {missing}")
        design = self._design(frame, fitting=False).to_numpy(dtype=float)
        observed = pd.to_numeric(frame["rgpm_d_fold"], errors="coerce").to_numpy(float)
        expected = design @ self.beta_
        residual = observed - expected
        invalid = ~np.isfinite(observed) | ~np.isfinite(design).all(axis=1)
        expected[invalid] = np.nan
        residual[invalid] = np.nan
        percentile = _mid_distribution_percentile(residual, self.residual_reference_)
        return pd.DataFrame(
            {
                "expected_diffusion_null_fold": expected,
                "excess_diffusion_raw_fold": residual,
                "excess_diffusion_fold": percentile,
                "excess_null_fit_scope": "outer_training_fold_only",
            },
            index=frame.index,
        )

    def fit_transform(self, training_frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(training_frame).transform(training_frame)

    def _design(self, frame: pd.DataFrame, *, fitting: bool) -> pd.DataFrame:
        numeric = pd.DataFrame(
            {
                "intercept": 1.0,
                "log_future_citers": np.log1p(
                    pd.to_numeric(frame["n_future_citers"], errors="coerce")
                ),
                "opportunity_score": pd.to_numeric(
                    frame.get("opportunity_score", 0.0), errors="coerce"
                ),
            },
            index=frame.index,
        )
        categories = pd.get_dummies(
            pd.DataFrame(
                {
                    "domain": frame["domain12"].fillna("<missing>").astype(str),
                    "year": frame["publication_year"].fillna("<missing>").astype(str),
                },
                index=frame.index,
            ),
            prefix=("domain", "year"),
            dtype=float,
        )
        design = pd.concat([numeric, categories], axis=1)
        if fitting:
            self.design_columns_ = list(design.columns)
        return design.reindex(columns=self.design_columns_, fill_value=0.0)


class FoldLocalPerturbationTargetTransformer:
    """Rank four perturbation dimensions using outer-training references only."""

    def fit(
        self, training_frame: pd.DataFrame
    ) -> FoldLocalPerturbationTargetTransformer:
        missing = sorted(set(PERTURBATION_COMPONENTS) - set(training_frame))
        if missing:
            raise ValueError(f"perturbation components are missing: {missing}")
        self.references_ = {}
        for column in PERTURBATION_COMPONENTS:
            values = pd.to_numeric(training_frame[column], errors="coerce").to_numpy(
                float
            )
            finite = values[np.isfinite(values)]
            if not len(finite):
                raise ValueError(f"training perturbation component is empty: {column}")
            self.references_[column] = np.sort(finite)
        self.is_fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("perturbation transformer must be fitted first")
        ranked = [
            _mid_distribution_percentile(
                pd.to_numeric(frame[column], errors="coerce").to_numpy(float),
                self.references_[column],
            )
            for column in PERTURBATION_COMPONENTS
        ]
        matrix = np.column_stack(ranked)
        score = np.where(np.isfinite(matrix).all(axis=1), matrix.mean(axis=1), np.nan)
        output: dict[str, object] = {
            f"{column}_fold": values
            for column, values in zip(PERTURBATION_COMPONENTS, ranked, strict=True)
        }
        output["perturbation_fold"] = score
        output["perturbation_fit_scope"] = "outer_training_fold_only"
        return pd.DataFrame(output, index=frame.index)

    def fit_transform(self, training_frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(training_frame).transform(training_frame)


def build_fold_local_structural_targets(
    training_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    *,
    outer_fold_id: str,
    horizon_years: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[TargetHeadProvenance]]:
    """Fit D/excess/P transforms on outer-train rows and apply them unchanged."""
    train = training_frame.copy()
    evaluation = evaluation_frame.copy()
    diffusion = FoldLocalDiffusionTargetTransformer().fit(train)
    train = train.join(diffusion.transform(train))
    evaluation = evaluation.join(diffusion.transform(evaluation))
    excess = FoldLocalExcessDiffusionTransformer().fit(train)
    train = train.join(excess.transform(train))
    evaluation = evaluation.join(excess.transform(evaluation))
    perturbation = FoldLocalPerturbationTargetTransformer().fit(train)
    train = train.join(perturbation.transform(train))
    evaluation = evaluation.join(perturbation.transform(evaluation))
    provenance = [
        _provenance(
            head="diffusion",
            outer_fold_id=outer_fold_id,
            horizon_years=horizon_years,
            training_rows=len(training_frame),
            target_columns=list(DIFFUSION_COMPONENTS),
            references=diffusion.references_,
        ),
        _provenance(
            head="excess_diffusion",
            outer_fold_id=outer_fold_id,
            horizon_years=horizon_years,
            training_rows=len(training_frame),
            target_columns=sorted(FoldLocalExcessDiffusionTransformer.REQUIRED),
            references={
                "beta": excess.beta_,
                "residual": excess.residual_reference_,
                "design": excess.design_columns_,
            },
        ),
        _provenance(
            head="perturbation",
            outer_fold_id=outer_fold_id,
            horizon_years=horizon_years,
            training_rows=len(training_frame),
            target_columns=list(PERTURBATION_COMPONENTS),
            references=perturbation.references_,
        ),
    ]
    return train, evaluation, provenance


def _provenance(
    *,
    head: str,
    outer_fold_id: str,
    horizon_years: int,
    training_rows: int,
    target_columns: list[str],
    references: dict[str, Any],
) -> TargetHeadProvenance:
    payload = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in references.items()
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return TargetHeadProvenance(
        head=head,  # type: ignore[arg-type]
        outer_fold_id=outer_fold_id,
        horizon_years=horizon_years,
        training_rows=training_rows,
        target_columns=target_columns,
        training_reference_sha256=f"sha256:{digest}",
    )


__all__ = [
    "BREADTH_COMPONENTS",
    "DIFFUSION_COMPONENTS",
    "EVENNESS_COMPONENTS",
    "PERTURBATION_COMPONENTS",
    "FoldLocalDiffusionTargetTransformer",
    "FoldLocalExcessDiffusionTransformer",
    "FoldLocalPerturbationTargetTransformer",
    "build_fold_local_structural_targets",
]
