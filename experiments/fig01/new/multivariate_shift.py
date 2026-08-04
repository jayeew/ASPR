"""Stage-aligned multivariate feature-space displacement for Fig. 1."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .event_data import write_json
from .feature_materialization import (
    DISPLAY_EXCLUDED_SOURCES,
    PREFERRED_SOURCE_REPRESENTATIVES,
    TIER_PRIORITY,
    materialize_values,
    selected_cases,
)


CORE_TIERS = {
    "source_formula_existing",
    "source_formula_local_surrogate",
}
CORE_SCOPE_ROLES = {"direct_innovation", "t0_substantive"}
CORE_DIMENSION_ROLES = {"substantive_innovation", "t0_potential"}
STAGES: Tuple[Tuple[int, str, str, Tuple[int, ...]], ...] = (
    (0, "pre", "Pre", (-6, -5, -4, -3, -2, -1)),
    (1, "landmark", "Landmark", (0, 1, 2)),
    (2, "early", "Early", (3, 4, 5)),
    (3, "late", "Late", (6, 7, 8)),
)


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _feature_pool(
    materialization: pd.DataFrame,
    library: Mapping[str, Mapping[str, str]],
    dimensions: Mapping[str, Mapping[str, str]],
) -> pd.DataFrame:
    frame = materialization.copy()
    frame["scope_role"] = frame["feature_id"].map(
        lambda value: library[str(value)]["scope_role"]
    )
    frame["construct_role"] = frame["dimension_id"].map(
        lambda value: dimensions[str(value)]["construct_role"]
    )
    frame = frame.loc[
        frame["materialization_status"].eq("materialized")
        & frame["tier"].isin(CORE_TIERS)
        & frame["scope_role"].isin(CORE_SCOPE_ROLES)
        & frame["construct_role"].isin(CORE_DIMENSION_ROLES)
        & ~frame["registered_source_column"].isin(
            DISPLAY_EXCLUDED_SOURCES
        )
    ].copy()
    winners: List[pd.Series] = []
    for source, rows in frame.groupby(
        "registered_source_column", sort=True
    ):
        preferred = PREFERRED_SOURCE_REPRESENTATIVES.get(str(source))
        if preferred and preferred in set(rows["feature_id"]):
            winner = rows.loc[rows["feature_id"].eq(preferred)].iloc[0]
        else:
            winner = (
                rows.assign(
                    _tier=rows["tier"].map(TIER_PRIORITY).fillna(99)
                )
                .sort_values(["_tier", "feature_id"], kind="stable")
                .iloc[0]
            )
        winners.append(winner)
    result = pd.DataFrame(winners).drop(columns=["_tier"], errors="ignore")
    result = result.sort_values(
        ["dimension_id", "feature_id"], kind="stable"
    ).reset_index(drop=True)
    result["feature_weight_within_dimension"] = 1.0 / result.groupby(
        "dimension_id"
    )["feature_id"].transform("count")
    result["dimension_weight"] = 1.0 / result["dimension_id"].nunique()
    return result


def _percentile_frame(
    frame: pd.DataFrame,
    values: Mapping[str, np.ndarray],
    feature_ids: Sequence[str],
) -> pd.DataFrame:
    result = frame[
        ["paper_id", "domain", "publication_year"]
    ].copy()
    years = result["publication_year"].astype(int)
    for feature_id in feature_ids:
        raw = pd.Series(values[feature_id], index=result.index, dtype=float)
        result[feature_id] = raw.groupby(years).rank(
            method="average", pct=True
        )
    return result


def _dimension_groups(
    pool: pd.DataFrame,
) -> Mapping[str, List[str]]:
    return {
        str(dimension): rows["feature_id"].astype(str).tolist()
        for dimension, rows in pool.groupby("dimension_id", sort=True)
    }


def _distance(
    delta: pd.Series | np.ndarray,
    feature_ids: Sequence[str],
    dimension_groups: Mapping[str, Sequence[str]],
) -> Tuple[float, Dict[str, float]]:
    values = pd.Series(np.asarray(delta, dtype=float), index=feature_ids)
    dimension_mse: Dict[str, float] = {}
    for dimension, members in dimension_groups.items():
        dimension_values = values[list(members)].to_numpy(dtype=float)
        if not np.isfinite(dimension_values).any():
            raise ValueError("A displacement dimension has no finite values")
        dimension_mse[dimension] = float(
            np.nanmean(np.square(dimension_values))
        )
    displacement = 100.0 * float(
        np.sqrt(np.nanmean(list(dimension_mse.values())))
    )
    return displacement, dimension_mse


def _year_medians(
    percentile: pd.DataFrame,
    domain: str,
    feature_ids: Sequence[str],
) -> pd.DataFrame:
    return (
        percentile.loc[percentile["domain"].eq(domain)]
        .groupby("publication_year")[list(feature_ids)]
        .median()
        .sort_index()
    )


def _stage_result(
    medians: pd.DataFrame,
    start_year: int,
    offsets: Sequence[int],
    feature_ids: Sequence[str],
    dimension_groups: Mapping[str, Sequence[str]],
) -> Tuple[float, Dict[str, float]]:
    pre_years = [start_year + value for value in STAGES[0][3]]
    stage_years = [start_year + value for value in offsets]
    if not set(pre_years + stage_years).issubset(medians.index):
        raise ValueError("Incomplete stage years for multivariate displacement")
    baseline = medians.loc[pre_years, list(feature_ids)].mean(axis=0)
    stage = medians.loc[stage_years, list(feature_ids)].mean(axis=0)
    return _distance(
        stage - baseline,
        feature_ids,
        dimension_groups,
    )


def _bootstrap_year_medians(
    percentile: pd.DataFrame,
    domain: str,
    years: Sequence[int],
    feature_ids: Sequence[str],
    draws: int,
    seed: int,
) -> Mapping[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    output: Dict[int, np.ndarray] = {}
    domain_rows = percentile.loc[percentile["domain"].eq(domain)]
    for year in years:
        matrix = domain_rows.loc[
            domain_rows["publication_year"].eq(year), list(feature_ids)
        ].to_numpy(dtype=float)
        if len(matrix) == 0:
            raise ValueError(f"No bootstrap rows for {domain} in {year}")
        positions = rng.integers(
            0, len(matrix), size=(draws, len(matrix))
        )
        with np.errstate(all="ignore"):
            output[year] = np.nanmedian(matrix[positions], axis=1)
    return output


def _bootstrap_distances(
    year_bootstrap: Mapping[int, np.ndarray],
    start_year: int,
    offsets: Sequence[int],
    feature_ids: Sequence[str],
    dimension_groups: Mapping[str, Sequence[str]],
) -> np.ndarray:
    pre = np.mean(
        [year_bootstrap[start_year + value] for value in STAGES[0][3]],
        axis=0,
    )
    stage = np.mean(
        [year_bootstrap[start_year + value] for value in offsets],
        axis=0,
    )
    delta = stage - pre
    feature_positions = {
        feature_id: index
        for index, feature_id in enumerate(feature_ids)
    }
    dimension_mse = []
    for members in dimension_groups.values():
        positions = [feature_positions[value] for value in members]
        dimension_mse.append(
            np.nanmean(np.square(delta[:, positions]), axis=1)
        )
    return 100.0 * np.sqrt(np.nanmean(dimension_mse, axis=0))


def _known_landmarks(cases: pd.DataFrame) -> Mapping[str, List[int]]:
    return {
        str(domain): sorted(
            rows["landmark_start_year"].astype(int).unique().tolist()
        )
        for domain, rows in cases.groupby("domain", sort=True)
    }


def _placebo_events(
    percentile: pd.DataFrame,
    domain: str,
    start_year: int,
    known_landmarks: Mapping[str, Sequence[int]],
) -> List[Tuple[str, int, str]]:
    minimum = int(percentile["publication_year"].min()) + 6
    maximum = int(percentile["publication_year"].max()) - 8
    events = [
        (domain, year, "same_domain_historical")
        for year in range(minimum, maximum + 1)
        if abs(year - start_year) > 8
    ]
    for control in sorted(set(percentile["domain"].astype(str)) - {domain}):
        known = known_landmarks.get(control, ())
        if any(start_year - 6 <= value <= start_year + 8 for value in known):
            continue
        events.append((control, start_year, "contemporaneous_control"))
    return events


def _placebos(
    percentile: pd.DataFrame,
    domain: str,
    start_year: int,
    feature_ids: Sequence[str],
    dimension_groups: Mapping[str, Sequence[str]],
    known_landmarks: Mapping[str, Sequence[int]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    medians = {
        value: _year_medians(percentile, value, feature_ids)
        for value in set(percentile["domain"].astype(str))
    }
    for placebo_domain, placebo_start, kind in _placebo_events(
        percentile, domain, start_year, known_landmarks
    ):
        for stage_index, stage, label, offsets in STAGES[1:]:
            try:
                displacement, _ = _stage_result(
                    medians[placebo_domain],
                    placebo_start,
                    offsets,
                    feature_ids,
                    dimension_groups,
                )
            except ValueError:
                continue
            rows.append(
                {
                    "focal_domain": domain,
                    "placebo_domain": placebo_domain,
                    "placebo_start_year": placebo_start,
                    "placebo_kind": kind,
                    "stage_index": stage_index,
                    "stage": stage,
                    "stage_label": label,
                    "displacement_pp": displacement,
                }
            )
    return pd.DataFrame(rows)


def build_multivariate_shift(
    output_dir: Path,
    *,
    bootstrap_draws: int = 2_000,
) -> Mapping[str, Any]:
    """Build stage displacement, placebo, and dimension-contribution tables."""
    panel_data = output_dir / "panel_data"
    cases = selected_cases(panel_data)
    all_cases = pd.read_csv(panel_data / "domain_selection.csv")
    (
        frame,
        values,
        materialization,
        library,
        _feature_to_dimension,
        dimensions,
    ) = materialize_values()
    pool = _feature_pool(materialization, library, dimensions)
    feature_ids = pool["feature_id"].astype(str).tolist()
    dimension_groups = _dimension_groups(pool)
    percentile = _percentile_frame(frame, values, feature_ids)
    known_landmarks = _known_landmarks(all_cases)
    displacement_rows: List[Dict[str, Any]] = []
    contribution_rows: List[Dict[str, Any]] = []
    bootstrap_rows: List[Dict[str, Any]] = []
    placebo_frames: List[pd.DataFrame] = []
    for case in cases.itertuples(index=False):
        domain = str(case.domain)
        start_year = int(case.landmark_start_year)
        medians = _year_medians(percentile, domain, feature_ids)
        years = list(range(start_year - 6, start_year + 9))
        year_bootstrap = _bootstrap_year_medians(
            percentile,
            domain,
            years,
            feature_ids,
            bootstrap_draws,
            _stable_seed("fig1_multivariate_shift", domain),
        )
        placebo = _placebos(
            percentile,
            domain,
            start_year,
            feature_ids,
            dimension_groups,
            known_landmarks,
        )
        placebo_frames.append(placebo)
        for stage_index, stage, label, offsets in STAGES:
            if stage_index == 0:
                displacement = 0.0
                dimension_mse = {
                    value: 0.0 for value in dimension_groups
                }
                samples = np.zeros(bootstrap_draws, dtype=np.float32)
            else:
                displacement, dimension_mse = _stage_result(
                    medians,
                    start_year,
                    offsets,
                    feature_ids,
                    dimension_groups,
                )
                samples = _bootstrap_distances(
                    year_bootstrap,
                    start_year,
                    offsets,
                    feature_ids,
                    dimension_groups,
                ).astype(np.float32)
            low, high = np.quantile(samples, [0.025, 0.975])
            stage_placebo = placebo.loc[
                placebo["stage_index"].eq(stage_index),
                "displacement_pp",
            ]
            if stage_index == 0:
                placebo_low = placebo_median = placebo_high = 0.0
                placebo_n = 0
                empirical_p = np.nan
            else:
                placebo_low, placebo_median, placebo_high = (
                    stage_placebo.quantile([0.05, 0.50, 0.95]).tolist()
                )
                placebo_n = int(len(stage_placebo))
                empirical_p = float(
                    (1 + stage_placebo.ge(displacement).sum())
                    / (1 + placebo_n)
                )
            displacement_rows.append(
                {
                    "episode_id": str(case.episode_id),
                    "domain": domain,
                    "stage_index": stage_index,
                    "stage": stage,
                    "stage_label": label,
                    "displacement_pp": displacement,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "placebo_low": placebo_low,
                    "placebo_median": placebo_median,
                    "placebo_high": placebo_high,
                    "placebo_n": placebo_n,
                    "empirical_p": empirical_p,
                    "bootstrap_draws": bootstrap_draws,
                }
            )
            total = float(sum(dimension_mse.values()))
            for dimension, mse in dimension_mse.items():
                contribution_rows.append(
                    {
                        "episode_id": str(case.episode_id),
                        "domain": domain,
                        "stage_index": stage_index,
                        "stage": stage,
                        "stage_label": label,
                        "dimension_id": dimension,
                        "dimension_label": dimensions[dimension]["label"],
                        "dimension_rms_pp": 100.0 * float(np.sqrt(mse)),
                        "contribution_share": (
                            float(mse / total) if total > 0 else 0.0
                        ),
                    }
                )
            bootstrap_rows.extend(
                {
                    "episode_id": str(case.episode_id),
                    "domain": domain,
                    "stage_index": stage_index,
                    "draw": draw,
                    "displacement_pp": float(value),
                }
                for draw, value in enumerate(samples)
            )
    displacement = pd.DataFrame(displacement_rows)
    contributions = pd.DataFrame(contribution_rows)
    contributions["contribution_rank"] = contributions.groupby(
        ["domain", "stage_index"]
    )["contribution_share"].rank(
        method="first", ascending=False
    ).astype(int)
    placebos = pd.concat(placebo_frames, ignore_index=True)
    paths = {
        "feature_pool": panel_data / "multivariate_feature_pool.csv",
        "displacement": panel_data
        / "multivariate_stage_displacement.csv",
        "contributions": panel_data
        / "multivariate_dimension_contributions.csv",
        "placebos": panel_data / "multivariate_placebos.csv",
        "bootstrap": panel_data / "multivariate_shift_bootstrap.parquet",
    }
    pool.to_csv(paths["feature_pool"], index=False)
    displacement.to_csv(paths["displacement"], index=False)
    contributions.to_csv(paths["contributions"], index=False)
    placebos.to_csv(paths["placebos"], index=False)
    pd.DataFrame(bootstrap_rows).to_parquet(
        paths["bootstrap"], index=False
    )
    manifest = {
        "artifact_kind": "fig1_multivariate_feature_space_displacement",
        "design_version": "fig1-multivariate-shift-v8.3",
        "source_registry_feature_count": 154,
        "materialized_feature_count": int(
            materialization["materialization_status"]
            .eq("materialized")
            .sum()
        ),
        "core_feature_count": len(pool),
        "core_dimension_count": len(dimension_groups),
        "core_feature_ids": feature_ids,
        "exact_source_formula_feature_count": int(
            pool["tier"].eq("source_formula_existing").sum()
        ),
        "local_operationalization_feature_count": int(
            pool["tier"].eq("source_formula_local_surrogate").sum()
        ),
        "selection_rule": (
            "materialized publication-time substantive features in eligible "
            "source-backed tiers; one deterministic representative per local "
            "source column; no effect-amplitude or outcome-based selection"
        ),
        "local_operationalization_disclosure": (
            "EF0017 and EF0240 are transparent local operationalizations, "
            "not claims of source-equivalent formula reproduction"
        ),
        "weighting": "equal within dimension; equal across dimensions",
        "year_normalization": "within-publication-year percentile rank",
        "stage_windows": {
            stage: list(offsets)
            for _, stage, _, offsets in STAGES
        },
        "bootstrap_draws": bootstrap_draws,
        "placebo_definition": (
            "same-domain non-overlapping historical pseudo-events plus "
            "contemporaneous control domains without a known landmark in "
            "the focal 15-year window"
        ),
        "future_information_used": False,
        "outcome_used_for_selection": False,
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
    }
    write_json(output_dir / "analysis_manifest_multivariate.json", manifest)
    return manifest


__all__ = [
    "CORE_DIMENSION_ROLES",
    "CORE_SCOPE_ROLES",
    "CORE_TIERS",
    "STAGES",
    "build_multivariate_shift",
]
