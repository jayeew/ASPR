"""Deterministic split construction for Nature multi-horizon models.

The module deliberately contains no target-aware logic.  It only balances the
publication-time strata requested by the analysis contract and records every
fallback used for sparse strata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


DEFAULT_HOLDOUT_YEARS: Dict[int, Tuple[int, int]] = {
    3: (2019, 2022),
    5: (2017, 2020),
    8: (2014, 2017),
}


@dataclass(frozen=True)
class FoldIndices:
    """Positional row indices for one train/test fold."""

    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class OuterFold:
    """One outer fold and its inner folds (relative to the full input frame)."""

    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    inner_folds: Tuple[FoldIndices, ...]


@dataclass(frozen=True)
class NestedSplitPlan:
    """Complete nested-CV plan plus a machine-readable fallback audit."""

    outer_folds: Tuple[OuterFold, ...]
    assignments: pd.DataFrame
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class HoldoutSplit:
    """Development and sealed temporal holdout indices."""

    horizon: int
    holdout_start_year: int
    holdout_end_year: int
    development_idx: np.ndarray
    holdout_idx: np.ndarray
    excluded_idx: np.ndarray


def publication_year_bin(values: pd.Series, width: int = 5) -> pd.Series:
    """Return stable integer publication-year bins."""

    if int(width) <= 0:
        raise ValueError("year-bin width must be positive")
    years = pd.to_numeric(values, errors="coerce")
    result = (np.floor(years / int(width)) * int(width)).astype("Int64")
    return result.astype("string").fillna("missing")


def _resolve_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Missing required split column; expected one of {list(candidates)}")


def _minimum_class_size(labels: pd.Series) -> int:
    counts = labels.value_counts(dropna=False)
    return int(counts.min()) if len(counts) else 0


def make_stratification_labels(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    domain_col: Optional[str] = None,
    year_col: Optional[str] = None,
    venue_col: Optional[str] = None,
    year_bin_width: int = 5,
) -> tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """Build hierarchical strata with deterministic sparse-cell fallbacks.

    Rows first use ``domain x 5-year-bin x venue family``.  Cells too small for
    the requested number of folds fall back to ``domain x year``, then
    ``domain``, then one global pool.  The returned level series makes this
    behaviour auditable per row.
    """

    if int(n_splits) < 2:
        raise ValueError("n_splits must be at least 2")
    if len(frame) < int(n_splits):
        raise ValueError(f"Need at least {n_splits} rows, received {len(frame)}")

    domain_col = domain_col or _resolve_column(frame, ("domain12", "domain"))
    year_col = year_col or _resolve_column(frame, ("publication_year", "year"))
    venue_col = venue_col or _resolve_column(frame, ("venue_family", "source_family", "source_id"))

    domain = frame[domain_col].astype("string").fillna("unmapped")
    year_bin = publication_year_bin(frame[year_col], year_bin_width)
    venue = frame[venue_col].astype("string").fillna("unknown")

    fine = "fine|" + domain + "|" + year_bin + "|" + venue
    domain_year = "domain_year|" + domain + "|" + year_bin
    domain_only = "domain|" + domain
    labels = pd.Series("global", index=frame.index, dtype="string")
    levels = pd.Series("global", index=frame.index, dtype="string")

    fine_counts = fine.value_counts()
    use_fine = fine.map(fine_counts).fillna(0).ge(n_splits)
    labels.loc[use_fine] = fine.loc[use_fine]
    levels.loc[use_fine] = "domain_year_venue"

    unresolved = ~use_fine
    dy_counts = domain_year.loc[unresolved].value_counts()
    use_dy = unresolved & domain_year.map(dy_counts).fillna(0).ge(n_splits)
    labels.loc[use_dy] = domain_year.loc[use_dy]
    levels.loc[use_dy] = "domain_year"

    unresolved &= ~use_dy
    domain_counts = domain_only.loc[unresolved].value_counts()
    use_domain = unresolved & domain_only.map(domain_counts).fillna(0).ge(n_splits)
    labels.loc[use_domain] = domain_only.loc[use_domain]
    levels.loc[use_domain] = "domain"

    # A global pool smaller than n_splits cannot be stratified.  Attach those
    # few rows to the largest valid stratum while retaining their fallback
    # level in the audit column.
    global_mask = levels.eq("global")
    if 0 < int(global_mask.sum()) < n_splits:
        valid_counts = labels.loc[~global_mask].value_counts()
        if len(valid_counts):
            labels.loc[global_mask] = str(valid_counts.index[0])
            levels.loc[global_mask] = "global_merged"
        else:
            labels.loc[:] = "global"
    if _minimum_class_size(labels) < n_splits:
        # Defensive final collapse.  This path is expected only for highly
        # pathological collections of missing grouping metadata.
        sparse = labels.map(labels.value_counts()).lt(n_splits)
        labels.loc[sparse] = "global"
        if 0 < int(labels.eq("global").sum()) < n_splits:
            labels.loc[:] = "global"
            levels.loc[:] = "global"

    audit = {
        "n_rows": int(len(frame)),
        "n_splits": int(n_splits),
        "domain_col": domain_col,
        "year_col": year_col,
        "venue_col": venue_col,
        "year_bin_width": int(year_bin_width),
        "fallback_counts": {str(k): int(v) for k, v in levels.value_counts().sort_index().items()},
        "n_final_strata": int(labels.nunique()),
        "minimum_final_stratum_size": _minimum_class_size(labels),
    }
    return labels, levels, audit


def _stratified_folds(
    positions: np.ndarray,
    labels: pd.Series,
    *,
    n_splits: int,
    seed: int,
) -> List[FoldIndices]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    dummy = np.zeros(len(positions), dtype=np.int8)
    result: List[FoldIndices] = []
    for fold_id, (train_local, test_local) in enumerate(splitter.split(dummy, labels.to_numpy()), start=1):
        result.append(
            FoldIndices(
                fold_id=fold_id,
                train_idx=positions[np.asarray(train_local, dtype=int)],
                test_idx=positions[np.asarray(test_local, dtype=int)],
            )
        )
    return result


def make_nested_folds(
    frame: pd.DataFrame,
    *,
    n_outer: int = 5,
    n_inner: int = 4,
    seed: int = 20260710,
    domain_col: Optional[str] = None,
    year_col: Optional[str] = None,
    venue_col: Optional[str] = None,
    year_bin_width: int = 5,
) -> NestedSplitPlan:
    """Create deterministic 5-outer/4-inner stratified folds.

    All indices are positional indices into ``frame.reset_index(drop=True)``.
    Inner folds are mapped back to those full-frame positions so consumers
    cannot accidentally mix local and global indices.
    """

    data = frame.reset_index(drop=True)
    positions = np.arange(len(data), dtype=int)
    outer_labels, outer_levels, outer_audit = make_stratification_labels(
        data,
        n_splits=n_outer,
        domain_col=domain_col,
        year_col=year_col,
        venue_col=venue_col,
        year_bin_width=year_bin_width,
    )
    outer_raw = _stratified_folds(positions, outer_labels, n_splits=n_outer, seed=seed)
    outer_folds: List[OuterFold] = []
    inner_audits: List[Dict[str, Any]] = []
    test_assignment = np.full(len(data), -1, dtype=int)

    for outer in outer_raw:
        outer_train = data.iloc[outer.train_idx].reset_index(drop=True)
        inner_labels, _, inner_audit = make_stratification_labels(
            outer_train,
            n_splits=n_inner,
            domain_col=domain_col,
            year_col=year_col,
            venue_col=venue_col,
            year_bin_width=year_bin_width,
        )
        inner_positions = np.arange(len(outer.train_idx), dtype=int)
        inner_local = _stratified_folds(
            inner_positions,
            inner_labels,
            n_splits=n_inner,
            seed=seed + outer.fold_id * 1009,
        )
        inner_global = tuple(
            FoldIndices(
                fold_id=fold.fold_id,
                train_idx=outer.train_idx[fold.train_idx],
                test_idx=outer.train_idx[fold.test_idx],
            )
            for fold in inner_local
        )
        test_assignment[outer.test_idx] = outer.fold_id
        outer_folds.append(
            OuterFold(
                fold_id=outer.fold_id,
                train_idx=outer.train_idx,
                test_idx=outer.test_idx,
                inner_folds=inner_global,
            )
        )
        inner_audit.update({"outer_fold": int(outer.fold_id), "n_outer_train": int(len(outer.train_idx))})
        inner_audits.append(inner_audit)

    assignments = pd.DataFrame(
        {
            "row_position": positions,
            "outer_fold": test_assignment,
            "stratification_level": outer_levels.to_numpy(),
            "stratification_label": outer_labels.to_numpy(),
        }
    )
    audit: Dict[str, Any] = {
        "method": "nested_stratified_cv",
        "outer": outer_audit,
        "inner": inner_audits,
        "seed": int(seed),
        "outer_folds": int(n_outer),
        "inner_folds": int(n_inner),
        "all_rows_assigned_once": bool(np.all(test_assignment > 0)),
    }
    return NestedSplitPlan(tuple(outer_folds), assignments, audit)


def split_sealed_holdout(
    frame: pd.DataFrame,
    horizon: int,
    *,
    year_col: Optional[str] = None,
    holdout_years: Optional[Mapping[int, Tuple[int, int]]] = None,
) -> HoldoutSplit:
    """Split development rows from the predeclared latest temporal holdout."""

    ranges = dict(holdout_years or DEFAULT_HOLDOUT_YEARS)
    if int(horizon) not in ranges:
        raise ValueError(f"No sealed holdout range configured for horizon={horizon}")
    year_col = year_col or _resolve_column(frame, ("publication_year", "year"))
    years = pd.to_numeric(frame[year_col], errors="coerce").to_numpy(dtype=float)
    start, end = ranges[int(horizon)]
    development = np.flatnonzero(np.isfinite(years) & (years < start))
    holdout = np.flatnonzero(np.isfinite(years) & (years >= start) & (years <= end))
    included = np.zeros(len(frame), dtype=bool)
    included[development] = True
    included[holdout] = True
    return HoldoutSplit(
        horizon=int(horizon),
        holdout_start_year=int(start),
        holdout_end_year=int(end),
        development_idx=development,
        holdout_idx=holdout,
        excluded_idx=np.flatnonzero(~included),
    )


def split_strict_label_availability(
    frame: pd.DataFrame,
    horizon: int,
    *,
    year_col: Optional[str] = None,
    holdout_years: Optional[Mapping[int, Tuple[int, int]]] = None,
) -> HoldoutSplit:
    """Temporal holdout whose training labels existed before test started.

    A publication from year ``y`` has a complete ``tau``-year label only after
    ``y + tau``.  The strict development set therefore requires
    ``publication_year + horizon < holdout_start_year``.
    """

    ordinary = split_sealed_holdout(
        frame,
        horizon,
        year_col=year_col,
        holdout_years=holdout_years,
    )
    year_col = year_col or _resolve_column(frame, ("publication_year", "year"))
    years = pd.to_numeric(frame[year_col], errors="coerce").to_numpy(dtype=float)
    development = np.flatnonzero(
        np.isfinite(years)
        & ((years + int(horizon)) < ordinary.holdout_start_year)
    )
    included = np.zeros(len(frame), dtype=bool)
    included[development] = True
    included[ordinary.holdout_idx] = True
    return HoldoutSplit(
        horizon=int(horizon),
        holdout_start_year=ordinary.holdout_start_year,
        holdout_end_year=ordinary.holdout_end_year,
        development_idx=development,
        holdout_idx=ordinary.holdout_idx,
        excluded_idx=np.flatnonzero(~included),
    )


# Public aliases used by the CLI and retained for discoverability.
build_nested_folds = make_nested_folds
make_outer_inner_splits = make_nested_folds
