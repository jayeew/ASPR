"""Fold-local v6 two-part outcome construction."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd


BREADTH_COMPONENTS: Tuple[str, ...] = (
    "future_field_reach",
    "future_subfield_reach",
    "future_topic_reach",
)
EVENNESS_COMPONENTS: Tuple[str, ...] = (
    "future_field_simpson",
    "future_topic_simpson",
)
DIFFUSION_COMPONENTS: Tuple[str, ...] = (
    *BREADTH_COMPONENTS,
    *EVENNESS_COMPONENTS,
)


def _mid_distribution_percentile(
    values: Sequence[float],
    reference: Sequence[float],
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
        self.references_: Dict[str, np.ndarray] = {}
        self.horizon_: int | None = None
        self.is_fitted_ = False

    @staticmethod
    def _check_columns(frame: pd.DataFrame) -> None:
        required = {"horizon", "future_uptake", *DIFFUSION_COMPONENTS}
        missing = sorted(required - set(frame))
        if missing:
            raise ValueError(f"diffusion components are missing columns: {missing}")

    def fit(self, training_frame: pd.DataFrame) -> "FoldLocalDiffusionTargetTransformer":
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
            pd.to_numeric(
                training_frame["future_uptake"], errors="coerce"
            ).eq(1)
        ]
        if positive.empty:
            raise ValueError("positive-uptake training rows are required")
        references: Dict[str, np.ndarray] = {}
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
        ranked: Dict[str, np.ndarray] = {}
        for column in DIFFUSION_COMPONENTS:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                dtype=float
            )
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
        diffusion = (
            self.breadth_weight * breadth
            + self.evenness_weight * evenness
        )
        if conditional_only:
            positive = pd.to_numeric(
                frame["future_uptake"], errors="coerce"
            ).eq(1)
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

    def fit_transform(
        self, training_frame: pd.DataFrame
    ) -> pd.DataFrame:
        return self.fit(training_frame).transform(training_frame)


__all__ = [
    "BREADTH_COMPONENTS",
    "DIFFUSION_COMPONENTS",
    "EVENNESS_COMPONENTS",
    "FoldLocalDiffusionTargetTransformer",
]
