"""Locked multi-horizon cohort rules and structural-validation sampling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .contracts import CORE_FEATURES, CohortSpec


def build_cohort_membership(
    papers: pd.DataFrame,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    spec: CohortSpec | None = None,
    required_feature_names: Sequence[str] | None = None,
    complete_end_year: int = 2025,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Apply horizon-specific observation, quality, and adoption rules.

    A failed or unknown future request remains a fetch failure.  Only an
    explicit successful request may contribute a numeric future-citer count.
    """

    cohort_spec = spec or CohortSpec()
    registered_features = tuple(required_feature_names or CORE_FEATURES)
    if not registered_features or len(registered_features) != len(
        set(registered_features)
    ):
        raise ValueError("required_feature_names must be non-empty and unique")
    paper_columns = ["paper_id", "publication_year"]
    for optional in ("domain12", "work_type", "document_type", "venue_family"):
        if optional in papers:
            paper_columns.append(optional)
    paper_frame = papers[paper_columns].drop_duplicates("paper_id")
    feature_columns = [
        "paper_id",
        "valid_reference_count",
        "reference_metadata_coverage",
        *registered_features,
    ]
    missing_features = set(feature_columns) - set(features)
    if missing_features:
        raise ValueError(f"features is missing cohort fields: {sorted(missing_features)}")
    target_columns = [
        "paper_id",
        "horizon",
        "fetch_status",
        "target_valid",
        "cap_hit",
        "n_future_citers",
        "future_uptake",
        "rgpm_d_raw",
    ]
    missing_targets = set(target_columns) - set(targets)
    if missing_targets:
        raise ValueError(f"targets is missing cohort fields: {sorted(missing_targets)}")

    joined = (
        targets[target_columns]
        .merge(paper_frame, on="paper_id", how="left", validate="many_to_one")
        .merge(features[feature_columns], on="paper_id", how="left", validate="many_to_one")
    )
    joined = joined[joined["horizon"].isin(cohort_spec.horizons)].copy()
    joined["publication_year"] = pd.to_numeric(joined["publication_year"], errors="coerce")
    joined["n_future_citers"] = pd.to_numeric(joined["n_future_citers"], errors="coerce")
    joined["complete_window"] = (
        joined["publication_year"] + joined["horizon"] <= int(complete_end_year)
    ).astype(int)
    joined["future_fetch_success"] = joined["fetch_status"].isin(
        ["success", "zero_success", "fetched", "checkpoint"]
    ).astype(int)
    joined["future_citer_gate"] = (
        joined["n_future_citers"] >= int(cohort_spec.min_future_citers)
    ).astype(int)
    joined["target_quality_gate"] = (
        joined["target_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(joined["rgpm_d_raw"], errors="coerce"))
    ).astype(int)
    joined["reference_count_gate"] = (
        pd.to_numeric(joined["valid_reference_count"], errors="coerce")
        >= int(cohort_spec.min_valid_references)
    ).astype(int)
    joined["reference_coverage_gate"] = (
        pd.to_numeric(joined["reference_metadata_coverage"], errors="coerce")
        >= float(cohort_spec.min_reference_metadata_coverage)
    ).astype(int)
    joined["high_quality_reference_coverage"] = (
        pd.to_numeric(joined["reference_metadata_coverage"], errors="coerce")
        >= float(cohort_spec.high_quality_reference_coverage)
    ).astype(int)
    joined["publication_feature_finite"] = np.isfinite(
        joined[list(registered_features)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    ).all(axis=1).astype(int)
    # Compatibility alias for v1 figures; v6 gates use the semantically named
    # field above and record the actual feature registry in the definition.
    joined["core8_finite"] = joined["publication_feature_finite"]
    joined["natural_science_scope"] = joined.get("domain12", "unmapped").isin(
        [
            "life_molecular",
            "ecology_evolution_microbiology",
            "neuroscience",
            "clinical_health",
            "chemistry",
            "physics",
            "astronomy_space",
            "earth_climate_environment",
            "materials_nanoscience",
            "engineering_energy",
            "mathematics_statistics",
            "computer_science_ai",
        ]
    ).astype(int)
    work_type = joined.get("work_type", joined.get("document_type", ""))
    joined["work_type_gate"] = work_type.fillna("").astype(str).isin(
        cohort_spec.allowed_work_types
    ).astype(int)
    gates = [
        "complete_window",
        "future_fetch_success",
        "future_citer_gate",
        "natural_science_scope",
        "work_type_gate",
    ]
    if cohort_spec.require_target_quality_for_cohort:
        gates.append("target_quality_gate")
    if cohort_spec.require_reference_quality_for_cohort:
        gates.extend(["reference_count_gate", "reference_coverage_gate"])
    if cohort_spec.require_all_features_finite:
        gates.append("publication_feature_finite")
    joined["cohort_member"] = joined[gates].all(axis=1).astype(int)
    joined["reference_evidence_eligible"] = (
        joined["cohort_member"].eq(1)
        & joined["reference_count_gate"].eq(1)
        & joined["reference_coverage_gate"].eq(1)
    ).astype(int)
    joined["conditional_diffusion_member"] = (
        joined["cohort_member"].eq(1)
        & pd.to_numeric(joined["future_uptake"], errors="coerce").eq(1)
        & joined["target_quality_gate"].eq(1)
    ).astype(int)
    joined["high_quality_cohort_member"] = (
        joined["reference_evidence_eligible"].eq(1)
        & joined["high_quality_reference_coverage"].eq(1)
    ).astype(int)
    joined["uncapped_cohort_member"] = (
        joined["cohort_member"].eq(1)
        & pd.to_numeric(joined["cap_hit"], errors="coerce").fillna(0).eq(0)
    ).astype(int)
    common_counts = (
        joined[joined["cohort_member"].eq(1)]
        .groupby("paper_id", observed=True)["horizon"]
        .nunique()
    )
    common_ids = set(
        common_counts[common_counts.eq(len(cohort_spec.horizons))].index.astype(str)
    )
    joined["common_cohort_member"] = joined["paper_id"].astype(str).isin(common_ids).astype(int)
    joined["observed_zero_future_citers"] = (
        joined["future_fetch_success"].eq(1)
        & joined["n_future_citers"].eq(0)
    ).astype(int)

    def exclusion_reasons(row: pd.Series) -> str:
        reasons = [column for column in gates if int(row[column]) == 0]
        return json.dumps(reasons, ensure_ascii=False)

    joined["exclusion_reasons"] = joined.apply(exclusion_reasons, axis=1)
    joined["cohort_definition"] = (
        f"future_citers>={cohort_spec.min_future_citers}; complete horizon; "
        f"work_type in {list(cohort_spec.allowed_work_types)}; "
        f"require_all_features_finite={cohort_spec.require_all_features_finite}; "
        f"require_reference_quality_for_cohort="
        f"{cohort_spec.require_reference_quality_for_cohort}; "
        f"require_target_quality_for_cohort="
        f"{cohort_spec.require_target_quality_for_cohort}; "
        f"publication_feature_registry={list(registered_features)}; "
        "failed fetches remain missing"
    )
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        joined.to_parquet(output_path, index=False)
    return joined


def cohort_quality_summary(membership: pd.DataFrame) -> Dict[str, Any]:
    """Summarize GO gates independently for each horizon."""

    by_horizon: Dict[str, Any] = {}
    for horizon, group in membership.groupby("horizon"):
        members = group[group["cohort_member"] == 1]
        domain_counts = members.groupby("domain12")["paper_id"].nunique()
        by_horizon[str(int(horizon))] = {
            "n_rows": int(len(group)),
            "n_members": int(len(members)),
            "n_domains_200": int((domain_counts >= 200).sum()),
            "cohort_5000_gate": bool(len(members) >= 5_000),
            "eight_domains_200_gate": bool((domain_counts >= 200).sum() >= 8),
            "fetch_success_rate": float(group["future_fetch_success"].mean()),
            "cap_hit_rate_in_cohort": float(
                pd.to_numeric(members.get("cap_hit"), errors="coerce")
                .fillna(0)
                .astype(bool)
                .mean()
            )
            if len(members)
            else 0.0,
        }
    return {"by_horizon": by_horizon}


def select_structural_subset(
    membership: pd.DataFrame,
    *,
    max_papers: int = 5_000,
    max_per_domain: int = 500,
    min_future_reference_coverage: float = 0.80,
    seed: int = 20260710,
) -> pd.DataFrame:
    """Pre-lock a stratified structural subset without using model scores."""

    frame = membership.copy()
    if "future_citer_reference_coverage" not in frame:
        frame["future_citer_reference_coverage"] = np.nan
    eligible = frame[
        (frame["cohort_member"] == 1)
        & (frame["cap_hit"] == 0)
        & (frame["n_future_citers"].between(0, 999, inclusive="both"))
        & (
            pd.to_numeric(frame["future_citer_reference_coverage"], errors="coerce")
            >= float(min_future_reference_coverage)
        )
    ].copy()
    eligible["publication_year_bin"] = (
        pd.to_numeric(eligible["publication_year"], errors="coerce").astype(int) // 5 * 5
    )
    selected: List[pd.DataFrame] = []
    for (_, domain, _), group in eligible.groupby(
        ["horizon", "domain12", "publication_year_bin"], sort=True
    ):
        domain_total = len(
            eligible[
                (eligible["horizon"] == group["horizon"].iloc[0])
                & (eligible["domain12"] == domain)
            ]
        )
        fraction = min(1.0, max_per_domain / max(1, domain_total))
        n = max(1, int(round(len(group) * fraction)))
        selected.append(group.sample(n=min(n, len(group)), random_state=seed))
    output = pd.concat(selected, ignore_index=True) if selected else eligible.head(0)
    horizon_parts: List[pd.DataFrame] = []
    for horizon, horizon_group in output.groupby("horizon", sort=True):
        if len(horizon_group) > max_papers:
            horizon_group = horizon_group.sample(
                n=max_papers, random_state=seed + int(horizon)
            )
        horizon_parts.append(horizon_group)
    output = (
        pd.concat(horizon_parts, ignore_index=True)
        if horizon_parts
        else output.head(0)
    )
    output = output.sort_values(["horizon", "domain12", "publication_year", "paper_id"])
    output["structural_subset_member"] = 1
    horizon_counts = output.groupby("horizon", observed=True)["paper_id"].transform(
        "size"
    )
    horizon_domains = output.groupby("horizon", observed=True)["domain12"].transform(
        "nunique"
    )
    output["structural_subset_exploratory"] = (
        horizon_counts.lt(2_000) | horizon_domains.lt(6)
    ).astype(int)
    return output.reset_index(drop=True)
