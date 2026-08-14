"""Adapters that preserve the old scientific route on current v6.1 artifacts."""

from __future__ import annotations

import copy
import itertools
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from gear.nature_multihorizon.screening_v6_1 import (
    reference_subsampling_stability,
)
from experiments.common.new.base.builders_1_5 import (
    build_fig1,
    build_fig2,
    build_fig3,
    build_fig4,
    build_fig5,
)
from experiments.common.new.base.builders_6_10 import (
    build_fig6,
    build_fig7,
    build_fig9,
    build_fig10,
)
from experiments.common.new.base.common import (
    ANGLE_LABELS,
    FEATURE_LABELS,
    FigureBundle,
    SuitePaths,
    bootstrap_mean_interval,
    grouped_percentile,
    safe_spearman,
    stable_seed,
)

from experiments.common.new.adapters.contracts import (
    ANGLE_FEATURES,
    FEATURE_DIRECTION,
    PRIMARY_FEATURES,
    STATUS_BLOCKED_COMPARABILITY,
    STATUS_DESCRIPTIVE,
    STATUS_DRAFT_LABELS,
)
from experiments.common.new.adapters.io import sha256_file, stable_hash
from experiments.common.new.adapters.fig2_evidence import (
    build_fig2_evidence_map,
)


BASE_BUILDERS = {
    1: build_fig1,
    2: build_fig2,
    3: build_fig3,
    4: build_fig4,
    5: build_fig5,
    6: build_fig6,
    7: build_fig7,
    9: build_fig9,
    10: build_fig10,
}


def _dataset(paths: SuitePaths, filename: str) -> Path:
    return paths["v6_1_dataset"] / filename


def _analysis(paths: SuitePaths, relative: str) -> Path:
    return paths["v6_1_analysis"] / relative


def _titles(paths: SuitePaths) -> pd.DataFrame:
    columns = ["id", "doi", "title", "year", "source_display_name"]
    return pd.read_csv(paths["target_works"], usecols=columns).rename(
        columns={"id": "paper_id"}
    )


def _measurement_scene(
    paths: SuitePaths,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build one outcome-blind G-/G0/G+5 publication-time measurement scene."""
    features = pd.read_parquet(
        _dataset(paths, "innovation_candidate_features.parquet"),
        columns=[
            "paper_id",
            "publication_year",
            "domain12",
            "valid_reference_count",
            *PRIMARY_FEATURES,
        ],
    )
    eligible = features.loc[
        features["publication_year"].between(2010, 2012)
        & features["valid_reference_count"].ge(20)
        & features[list(PRIMARY_FEATURES)].notna().all(axis=1)
    ].copy()
    eligible["selection_hash"] = eligible["paper_id"].map(
        lambda value: stable_hash(str(value), seed)
    )
    focal = eligible.sort_values("selection_hash", kind="stable").iloc[0]
    paper_id = str(focal["paper_id"])
    year = int(focal["publication_year"])
    paper_references = pd.read_parquet(
        _dataset(paths, "paper_references.parquet")
    )
    focal_references = paper_references.loc[
        paper_references["paper_id"].astype(str).eq(paper_id),
        "reference_id",
    ].astype(str)
    reference_metadata = pd.read_parquet(
        _dataset(paths, "reference_metadata.parquet")
    )
    metadata = reference_metadata.loc[
        reference_metadata["reference_id"].astype(str).isin(focal_references)
    ].copy()
    sources = (
        metadata.loc[metadata["source_id"].fillna("").astype(str).ne("")]
        .groupby("source_id", as_index=False)
        .agg(reference_count=("reference_id", "nunique"))
        .sort_values(["reference_count", "source_id"], ascending=[False, True])
        .head(12)
    )
    source_ids = sources["source_id"].astype(str).tolist()
    history_sources = pd.read_parquet(
        _dataset(paths, "historical_paper_sources.parquet")
    )
    prior = history_sources.loc[
        history_sources["publication_year"].lt(year)
    ]
    pair_counts: Dict[Tuple[str, str], int] = {}
    for values in prior["cited_source_ids"]:
        occupied = sorted(set(str(value) for value in values) & set(source_ids))
        for left, right in itertools.combinations(occupied, 2):
            pair_counts[(left, right)] = pair_counts.get((left, right), 0) + 1
    top_pairs = sorted(
        pair_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:30]
    history_references = pd.read_parquet(
        _dataset(paths, "historical_paper_references.parquet")
    )
    future = history_references.loc[
        history_references["publication_year"].between(year + 1, year + 5)
    ]
    future = future.loc[
        future["reference_ids"].map(
            lambda values: paper_id in {str(value) for value in values}
        )
    ].sort_values(["publication_year", "work_id"], kind="stable").head(12)
    title_row = _titles(paths)
    title_row = title_row.loc[title_row["paper_id"].astype(str).eq(paper_id)]
    title = (
        str(title_row.iloc[0]["title"])
        if not title_row.empty
        else paper_id
    )
    node_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    stages = ("G−", "G0", "G+5")
    angles = np.linspace(0, 2 * np.pi, max(len(source_ids), 1), endpoint=False)
    source_positions = {
        source_id: (math.cos(angle), math.sin(angle))
        for source_id, angle in zip(source_ids, angles)
    }
    for stage in stages:
        for source_id in source_ids:
            x_value, y_value = source_positions[source_id]
            node_rows.append(
                {
                    "stage": stage,
                    "node_id": source_id,
                    "node_type": "reference_source",
                    "label": source_id.rsplit("/", 1)[-1],
                    "x": x_value,
                    "y": y_value,
                }
            )
        for (left, right), weight in top_pairs:
            edge_rows.append(
                {
                    "stage": stage,
                    "source": left,
                    "target": right,
                    "edge_type": "strictly_prior_cocitation",
                    "weight": weight,
                }
            )
        if stage in {"G0", "G+5"}:
            node_rows.append(
                {
                    "stage": stage,
                    "node_id": paper_id,
                    "node_type": "focal_paper",
                    "label": "Focal paper",
                    "x": 0.0,
                    "y": 0.0,
                }
            )
            for source_id in source_ids:
                edge_rows.append(
                    {
                        "stage": stage,
                        "source": paper_id,
                        "target": source_id,
                        "edge_type": "focal_reference",
                        "weight": 1,
                    }
                )
    for index, row in enumerate(future.itertuples(index=False)):
        angle = 2 * np.pi * index / max(len(future), 1)
        citer_id = str(row.work_id)
        node_rows.append(
            {
                "stage": "G+5",
                "node_id": citer_id,
                "node_type": "future_citer",
                "label": f"+{int(row.publication_year) - year}y",
                "x": 1.45 * math.cos(angle),
                "y": 1.45 * math.sin(angle),
            }
        )
        edge_rows.append(
            {
                "stage": "G+5",
                "source": citer_id,
                "target": paper_id,
                "edge_type": "future_citation",
                "weight": 1,
            }
        )
    manifest = pd.DataFrame(
        [
            {
                "paper_id": paper_id,
                "title": title,
                "publication_year": year,
                "domain12": focal["domain12"],
                "valid_reference_count": int(focal["valid_reference_count"]),
                "selection_rule": (
                    "minimum stable hash among 2010–2012 complete-core8 "
                    "papers with >=20 valid references; no outcome used"
                ),
                "selection_hash": focal["selection_hash"],
                "future_citers_visible_in_local_history": int(len(future)),
            }
        ]
    )
    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows), manifest


def _cluster_rank_interval(
    frame: pd.DataFrame,
    left: str,
    right: str,
    *,
    iterations: int,
    seed: int,
) -> Tuple[float, float]:
    """Fixed-rank cluster bootstrap over domain-year cells."""
    work = frame[[left, right, "domain12", "publication_year"]].dropna().copy()
    if len(work) < 20:
        return float("nan"), float("nan")
    work["x"] = work[left].rank(method="average", pct=True)
    work["y"] = work[right].rank(method="average", pct=True)
    grouped = (
        work.groupby(["domain12", "publication_year"], as_index=False)
        .agg(
            n=("x", "size"),
            sx=("x", "sum"),
            sy=("y", "sum"),
            sxx=("x", lambda values: float(np.square(values).sum())),
            syy=("y", lambda values: float(np.square(values).sum())),
            sxy=("x", lambda values: 0.0),
        )
    )
    # pandas named aggregation cannot access both columns; calculate cross-products separately.
    cross = (
        work.assign(xy=work["x"] * work["y"])
        .groupby(["domain12", "publication_year"], as_index=False)["xy"]
        .sum()
    )
    grouped = grouped.drop(columns="sxy").merge(
        cross,
        on=["domain12", "publication_year"],
        how="left",
    )
    arrays = grouped[["n", "sx", "sy", "sxx", "syy", "xy"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sample = arrays[rng.integers(0, len(arrays), len(arrays))].sum(axis=0)
        n, sx, sy, sxx, syy, sxy = sample
        covariance = sxy - sx * sy / n
        denominator = math.sqrt(
            max(sxx - sx * sx / n, 0.0) * max(syy - sy * sy / n, 0.0)
        )
        estimates[index] = covariance / denominator if denominator else np.nan
    estimates = estimates[np.isfinite(estimates)]
    if not len(estimates):
        return float("nan"), float("nan")
    return float(np.quantile(estimates, 0.025)), float(
        np.quantile(estimates, 0.975)
    )


def _future_component_correlations(
    paths: SuitePaths,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Relate publication-time indicators to frozen five-year graph outcomes."""
    features = pd.read_parquet(
        _dataset(paths, "innovation_candidate_features.parquet"),
        columns=[
            "paper_id",
            "publication_year",
            "domain12",
            *PRIMARY_FEATURES,
        ],
    )
    targets = pd.read_parquet(
        _dataset(paths, "targets_zero_inclusive.parquet"),
        columns=[
            "paper_id",
            "horizon",
            "target_valid",
            "future_field_reach",
            "future_subfield_reach",
            "future_topic_reach",
            "future_field_simpson",
            "future_topic_simpson",
            "rgpm_d_raw",
        ],
    )
    targets = targets.loc[
        targets["horizon"].eq(5) & targets["target_valid"].eq(1)
    ].drop_duplicates("paper_id")
    frame = features.merge(targets, on="paper_id", how="inner")
    components = {
        "future_field_reach": "Field reach",
        "future_subfield_reach": "Subfield reach",
        "future_topic_reach": "Topic reach",
        "future_field_simpson": "Field evenness",
        "future_topic_simpson": "Topic evenness",
        "rgpm_d_raw": "D5 composite",
    }
    for column in (
        "future_field_reach",
        "future_subfield_reach",
        "future_topic_reach",
    ):
        frame[column] = np.log1p(pd.to_numeric(frame[column], errors="coerce"))
    for column in (*PRIMARY_FEATURES, *components):
        frame[f"{column}__fy"] = grouped_percentile(
            frame,
            column,
            ["domain12", "publication_year"],
            id_column="paper_id",
        )
    rows: List[Dict[str, Any]] = []
    for feature_index, feature in enumerate(PRIMARY_FEATURES):
        for component_index, (component, component_label) in enumerate(
            components.items()
        ):
            left = f"{feature}__fy"
            right = f"{component}__fy"
            paired = frame[[left, right]].dropna()
            low, high = _cluster_rank_interval(
                frame,
                left,
                right,
                iterations=iterations,
                seed=seed + 101 * feature_index + component_index,
            )
            rows.append(
                {
                    "code_name": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "future_component": component,
                    "future_component_label": component_label,
                    "n": int(len(paired)),
                    "spearman": safe_spearman(paired[left], paired[right]),
                    "ci_low": low,
                    "ci_high": high,
                    "normalization": "domain-year percentile",
                    "interval_method": "fixed-rank domain-year cluster bootstrap",
                }
            )
    return pd.DataFrame(rows)


def _fig2_selection_stages(decisions: pd.DataFrame) -> pd.DataFrame:
    """Summarize the registered screening route without hard-coded counts."""
    local = pd.to_numeric(
        decisions["raw_overall_coverage"],
        errors="coerce",
    ).fillna(0).gt(0)
    runtime = pd.to_numeric(
        decisions["eligible_all_runtime_gates"],
        errors="coerce",
    ).fillna(0).eq(1)
    nonredundant = runtime & ~decisions["proposed_final_role"].eq("excluded")
    primary = decisions["proposed_final_role"].eq("primary")
    rows = [
        {
            "stage_order": 1,
            "stage": "Literature candidates",
            "count": int(len(decisions)),
            "criterion": "Multi-source evidence map",
        },
        {
            "stage_order": 2,
            "stage": "Locally computable",
            "count": int(local.sum()),
            "criterion": "Frozen Nature/OpenAlex tables",
        },
        {
            "stage_order": 3,
            "stage": "Runtime-gate pass",
            "count": int(runtime.sum()),
            "criterion": "Coverage · stability · fidelity",
        },
        {
            "stage_order": 4,
            "stage": "Non-redundant eligible",
            "count": int(nonredundant.sum()),
            "criterion": "Exact duplicates removed",
        },
        {
            "stage_order": 5,
            "stage": "Primary indicators",
            "count": int(primary.sum()),
            "criterion": "One frozen family representative",
        },
    ]
    output = pd.DataFrame(rows)
    output["removed_since_previous"] = (
        output["count"].shift(1) - output["count"]
    ).fillna(0).astype(int)
    return output


def _fig2_indicator_basis(bundle: FigureBundle) -> pd.DataFrame:
    """Create one ordered, direction-aware five-angle/eight-indicator table."""
    primary = bundle.tables["primary_indicator_map"].copy()
    quality = bundle.tables["primary_quality_gates"].copy()
    primary = primary.merge(
        quality[
            [
                "code_name",
                "coverage_pass",
                "stability_pass",
                "approximation_pass",
            ]
        ],
        on="code_name",
        how="left",
    )
    order = {feature: index + 1 for index, feature in enumerate(PRIMARY_FEATURES)}
    primary["display_order"] = primary["code_name"].map(order).astype(int)
    primary["direction"] = primary["code_name"].map(FEATURE_DIRECTION).astype(int)
    primary["direction_label"] = np.where(
        primary["direction"].eq(1),
        "higher = stronger signal",
        "lower = stronger signal",
    )
    primary["evidence_badge"] = primary.apply(
        lambda row: (
            f"F{int(row['original_source_count'])}"
            f"·P{int(row['application_source_count'])}"
            f"·V{int(row['validation_source_count'])}"
        ),
        axis=1,
    )
    primary["all_primary_gates_pass"] = (
        primary[
            ["coverage_pass", "stability_pass", "approximation_pass"]
        ]
        .fillna(0)
        .eq(1)
        .all(axis=1)
    )
    return primary.sort_values("display_order", kind="stable").reset_index(
        drop=True
    )


def _fig2_indicator_relations(
    correlations: pd.DataFrame,
    basis: pd.DataFrame,
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build a sparse, direction-aware relation network for the eight signals."""
    nodes = basis[
        [
            "code_name",
            "feature_label",
            "angle_id",
            "angle_label",
            "display_order",
            "direction",
        ]
    ].copy()
    lookup = nodes.set_index("code_name")
    order = lookup["display_order"].to_dict()
    rows: List[Dict[str, Any]] = []
    for row in correlations.itertuples(index=False):
        left = str(row.feature_x)
        right = str(row.feature_y)
        if left not in order or right not in order or order[left] >= order[right]:
            continue
        raw = float(row.spearman)
        oriented = (
            raw
            * int(lookup.loc[left, "direction"])
            * int(lookup.loc[right, "direction"])
        )
        if abs(oriented) < float(threshold):
            continue
        rows.append(
            {
                "source": left,
                "target": right,
                "source_label": lookup.loc[left, "feature_label"],
                "target_label": lookup.loc[right, "feature_label"],
                "source_angle_id": lookup.loc[left, "angle_id"],
                "target_angle_id": lookup.loc[right, "angle_id"],
                "raw_spearman": raw,
                "oriented_spearman": oriented,
                "absolute_spearman": abs(oriented),
                "cross_angle": bool(
                    lookup.loc[left, "angle_id"]
                    != lookup.loc[right, "angle_id"]
                ),
                "threshold": float(threshold),
            }
        )
    return nodes, pd.DataFrame(rows)


def _fig2_oriented_future_correlations(
    future: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the a-priori signal direction to prospective associations."""
    output = future.copy()
    output["direction"] = output["code_name"].map(FEATURE_DIRECTION).astype(int)
    output["oriented_spearman"] = output["spearman"] * output["direction"]
    positive = output["direction"].eq(1)
    output["oriented_ci_low"] = np.where(
        positive,
        output["ci_low"],
        -output["ci_high"],
    )
    output["oriented_ci_high"] = np.where(
        positive,
        output["ci_high"],
        -output["ci_low"],
    )
    feature_order = {
        feature: index + 1 for index, feature in enumerate(PRIMARY_FEATURES)
    }
    component_order = {
        label: index + 1
        for index, label in enumerate(
            [
                "Field reach",
                "Subfield reach",
                "Topic reach",
                "Field evenness",
                "Topic evenness",
                "D5 composite",
            ]
        )
    }
    output["feature_order"] = output["code_name"].map(feature_order).astype(int)
    output["component_order"] = output["future_component_label"].map(
        component_order
    ).astype(int)
    output["ci_excludes_zero"] = (
        output["oriented_ci_low"].gt(0)
        | output["oriented_ci_high"].lt(0)
    )
    return output.sort_values(
        ["feature_order", "component_order"],
        kind="stable",
    ).reset_index(drop=True)


def _fig2_known_group_profiles(
    paths: SuitePaths,
    membership: pd.DataFrame,
    effects: pd.DataFrame,
    *,
    sample_per_group: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create oriented matched-control profiles for the old Fig.2 visual route."""
    features = pd.read_parquet(
        _dataset(paths, "innovation_candidate_features.parquet"),
        columns=[
            "paper_id",
            "publication_year",
            "domain12",
            *PRIMARY_FEATURES,
        ],
    )
    angle_lookup = {
        feature: angle_id
        for angle_id, angle_features in ANGLE_FEATURES.items()
        for feature in angle_features
    }
    long_rows: List[pd.DataFrame] = []
    for feature in PRIMARY_FEATURES:
        values = features[
            ["paper_id", "publication_year", "domain12", feature]
        ].copy()
        values["raw_percentile"] = grouped_percentile(
            values,
            feature,
            ["domain12", "publication_year"],
            id_column="paper_id",
        )
        direction = int(FEATURE_DIRECTION[feature])
        values["oriented_percentile"] = np.where(
            direction == 1,
            values["raw_percentile"],
            1.0 - values["raw_percentile"],
        )
        subset = membership[
            ["pair_id", "paper_id", "group"]
        ].merge(
            values[
                [
                    "paper_id",
                    "publication_year",
                    "domain12",
                    "raw_percentile",
                    "oriented_percentile",
                ]
            ],
            on="paper_id",
            how="left",
        )
        subset["code_name"] = feature
        subset["feature_label"] = FEATURE_LABELS[feature]
        subset["angle_id"] = angle_lookup[feature]
        subset["direction"] = direction
        long_rows.append(subset)
    long = pd.concat(long_rows, ignore_index=True).dropna(
        subset=["oriented_percentile"]
    )
    summary = (
        long.groupby(
            ["code_name", "feature_label", "angle_id", "group"],
            as_index=False,
        )
        .agg(
            n=("paper_id", "size"),
            mean=("oriented_percentile", "mean"),
            q25=("oriented_percentile", lambda values: values.quantile(0.25)),
            median=("oriented_percentile", "median"),
            q75=("oriented_percentile", lambda values: values.quantile(0.75)),
        )
    )
    samples: List[pd.DataFrame] = []
    for (feature, group), subset in long.groupby(
        ["code_name", "group"],
        sort=True,
    ):
        n = min(int(sample_per_group), len(subset))
        samples.append(
            subset.sample(
                n=n,
                random_state=stable_seed(f"fig2-profile:{feature}:{group}", seed),
            )
        )
    sample = pd.concat(samples, ignore_index=True)
    oriented_effects = effects.copy()
    oriented_effects["direction"] = oriented_effects["code_name"].map(
        FEATURE_DIRECTION
    ).astype(int)
    negative = oriented_effects["direction"].eq(-1)
    original_low = oriented_effects["ci_low"].copy()
    original_high = oriented_effects["ci_high"].copy()
    oriented_effects["oriented_difference"] = (
        oriented_effects["mean_percentile_difference"]
        * oriented_effects["direction"]
    )
    oriented_effects["oriented_ci_low"] = np.where(
        negative,
        -original_high,
        original_low,
    )
    oriented_effects["oriented_ci_high"] = np.where(
        negative,
        -original_low,
        original_high,
    )
    oriented_effects["angle_id"] = oriented_effects["code_name"].map(
        angle_lookup
    )
    oriented_effects["display_order"] = oriented_effects["code_name"].map(
        {feature: index + 1 for index, feature in enumerate(PRIMARY_FEATURES)}
    )
    return (
        sample,
        summary,
        oriented_effects.sort_values("display_order", kind="stable"),
    )


def _enhance_fig2(
    bundle: FigureBundle,
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    nodes, edges, scene = _measurement_scene(
        paths,
        int(config["fig2"]["seed"]),
    )
    future = _future_component_correlations(
        paths,
        int(config["fig2"].get("future_bootstrap_iterations", 300)),
        int(config["fig2"]["seed"]),
    )
    selection = _fig2_selection_stages(bundle.tables["candidate_decisions"])
    basis = _fig2_indicator_basis(bundle)
    relation_nodes, relation_edges = _fig2_indicator_relations(
        bundle.tables["indicator_correlations"],
        basis,
        float(config["fig2"].get("relation_abs_spearman_min", 0.40)),
    )
    oriented_future = _fig2_oriented_future_correlations(future)
    profile_sample, profile_summary, oriented_effects = (
        _fig2_known_group_profiles(
            paths,
            bundle.tables["known_group_membership"],
            bundle.tables["known_group_effects"],
            sample_per_group=int(
                config["fig2"].get("profile_sample_per_group", 220)
            ),
            seed=int(config["fig2"]["seed"]),
        )
    )
    bundle.tables.update(
        {
            "measurement_scene_nodes": nodes,
            "measurement_scene_edges": edges,
            "measurement_scene_manifest": scene,
            "future_component_correlations": future,
            "fig2_selection_stages": selection,
            "fig2_indicator_basis": basis,
            "fig2_relation_nodes": relation_nodes,
            "fig2_relation_edges": relation_edges,
            "fig2_oriented_future_correlations": oriented_future,
            "fig2_known_group_profile_sample": profile_sample,
            "fig2_known_group_profile_summary": profile_summary,
            "fig2_known_group_oriented_effects": oriented_effects,
        }
    )
    bundle.panel_text = {
        "a": {
            **scene.iloc[0].to_dict(),
            "message": (
                "All eight signals are measured at G0 from the focal "
                "references and the strictly prior graph; G+5 is validation only."
            ),
        },
        "b": {
            "selection_stages": selection.to_dict("records"),
            "message": (
                "The frozen primary basis is chosen by evidence, publication-time "
                "feasibility, runtime gates, and within-family non-redundancy."
            ),
        },
        "c": {
            "relation_threshold": float(
                config["fig2"].get("relation_abs_spearman_min", 0.40)
            ),
            "relation_edge_count": int(len(relation_edges)),
            "future_validation_n_max": int(oriented_future["n"].max()),
            "message": (
                "Fixed-direction correlations describe complementarity; D5 "
                "associations validate interpretation and never select indicators."
            ),
        },
        "d": {
            "matched_pairs_max": int(oriented_effects["n_pairs"].max()),
            "message": (
                "High-D5 papers are a known-group construct check, not a complete "
                "ground truth for innovation."
            ),
        },
    }
    bundle.chart_contract = {
        "figure_id": 2,
        "scientific_route": (
            "measurement boundary -> evidence-governed basis -> "
            "mechanism relations and future signatures -> known-group audit"
        ),
        "panels": {
            "a": {
                "mark": "fixed-layout G-/G0/G+5 network triptych",
                "data": [
                    "measurement_scene_nodes",
                    "measurement_scene_edges",
                    "measurement_scene_manifest",
                ],
            },
            "b": {
                "mark": "screening funnel plus five-angle indicator basis",
                "data": [
                    "fig2_selection_stages",
                    "fig2_indicator_basis",
                    "observation_angles",
                ],
            },
            "c": {
                "mark": (
                    "sparse indicator-relation network plus prospective "
                    "D5 dot matrix"
                ),
                "data": [
                    "fig2_relation_nodes",
                    "fig2_relation_edges",
                    "fig2_oriented_future_correlations",
                ],
            },
            "d": {
                "mark": "matched-control percentile raincloud and paired effects",
                "data": [
                    "fig2_known_group_profile_sample",
                    "fig2_known_group_profile_summary",
                    "fig2_known_group_oriented_effects",
                ],
            },
        },
        "traditional_heatmap_count": 0,
        "outcome_used_for_indicator_selection": False,
        "indicator_direction_source": "frozen FEATURE_DIRECTION contract",
    }
    retained_tables = (
        "candidate_decisions",
        "observation_angles",
        "source_map",
        "primary_indicator_map",
        "primary_quality_gates",
        "measurement_scene_nodes",
        "measurement_scene_edges",
        "measurement_scene_manifest",
        "fig2_selection_stages",
        "fig2_indicator_basis",
        "fig2_relation_nodes",
        "fig2_relation_edges",
        "fig2_oriented_future_correlations",
        "fig2_known_group_profile_sample",
        "fig2_known_group_profile_summary",
        "fig2_known_group_oriented_effects",
    )
    bundle.tables = {
        name: bundle.tables[name]
        for name in retained_tables
    }
    bundle.source_paths.extend(
        [
            _dataset(paths, "paper_references.parquet"),
            _dataset(paths, "reference_metadata.parquet"),
            _dataset(paths, "historical_paper_sources.parquet"),
            _dataset(paths, "historical_paper_references.parquet"),
            _dataset(paths, "targets_zero_inclusive.parquet"),
            paths["target_works"],
        ]
    )
    bundle.title = (
        "Publication-time reference signals organize observable graph change"
    )
    return bundle


def _enhance_fig3(
    bundle: FigureBundle,
    paths: SuitePaths,
) -> FigureBundle:
    targets = pd.read_parquet(
        _dataset(paths, "targets_zero_inclusive.parquet"),
        columns=[
            "paper_id",
            "horizon",
            "target_valid",
            "future_uptake",
            "rgpm_d_breadth",
            "rgpm_d_evenness",
            "rgpm_d_raw",
            "definition_version",
        ],
    )
    targets = targets.loc[targets["horizon"].eq(5)].copy()
    summary = pd.DataFrame(
        [
            {
                "step": 1,
                "component": "Future uptake",
                "definition": "whether any eligible future adopter is observed",
                "weight": np.nan,
                "valid_n": int(targets["future_uptake"].notna().sum()),
            },
            {
                "step": 2,
                "component": "Breadth",
                "definition": "mean train-fold percentiles of field/subfield/topic reach",
                "weight": 0.5,
                "valid_n": int(targets["rgpm_d_breadth"].notna().sum()),
            },
            {
                "step": 3,
                "component": "Evenness",
                "definition": "mean train-fold percentiles of field/topic Simpson",
                "weight": 0.5,
                "valid_n": int(targets["rgpm_d_evenness"].notna().sum()),
            },
            {
                "step": 4,
                "component": "Realized D5",
                "definition": "zero for no uptake; 0.5 breadth + 0.5 evenness otherwise",
                "weight": 1.0,
                "valid_n": int(targets["rgpm_d_raw"].notna().sum()),
            },
        ]
    )
    counts = pd.DataFrame(
        [
            {
                "total_rows": int(len(targets)),
                "target_valid": int(targets["target_valid"].eq(1).sum()),
                "positive_uptake": int(targets["future_uptake"].eq(1).sum()),
                "zero_uptake": int(targets["future_uptake"].eq(0).sum()),
                "definition_version": "|".join(
                    sorted(targets["definition_version"].dropna().astype(str).unique())
                ),
            }
        ]
    )
    fold_path = (
        paths["v6_1_figure_baseline"]
        / "experiment_09/angle_ablation_fold_metrics.csv"
    )
    folds = pd.read_csv(fold_path)
    folds["diagnostic"] = np.where(
        folds["model_id"].str.startswith("k1_plus"),
        "add to K1",
        "delete from full",
    )
    folds["angle_number"] = (
        folds["model_id"].str.extract(r"a([1-5])", expand=False).astype(int)
    )
    folds["angle_id"] = folds["angle_number"].map(
        {index + 1: angle for index, angle in enumerate(ANGLE_FEATURES)}
    )
    folds["angle_label"] = folds["angle_id"].map(ANGLE_LABELS)
    bundle.tables.update(
        {
            "d5_target_construction": summary,
            "d5_target_counts": counts,
            "angle_fold_stability": folds,
        }
    )
    bundle.panel_text["target_construction"] = counts.iloc[0].to_dict()
    bundle.chart_contract["panels"] = {
        "a": {
            "mark": "D5 target-construction flow",
            "data": ["d5_target_construction", "d5_target_counts"],
        },
        "b": {
            "mark": "two-part model and six expanding temporal folds",
            "data": ["temporal_folds"],
        },
        "c": {
            "mark": "model-performance estimation ladder",
            "data": ["model_ladder", "paired_model_gains"],
        },
        "d": {
            "mark": "OOF prediction-realization joint density",
            "data": ["oof_joint_density"],
        },
        "e": {
            "mark": "realized D5 raincloud by OOF prediction decile",
            "data": ["prediction_decile_sample"],
        },
        "f": {
            "mark": "five-angle add-delete effects and temporal stability",
            "data": ["angle_add_delete", "angle_fold_stability"],
        },
    }
    bundle.title = (
        "Publication-time signals rank future D5 diffusion under temporal OOF"
    )
    bundle.source_paths.extend(
        [_dataset(paths, "targets_zero_inclusive.parquet"), fold_path]
    )
    return bundle


def _round_robin_sample(
    frame: pd.DataFrame,
    count: int,
) -> pd.DataFrame:
    """Select a deterministic score-tier sample with broad domain coverage."""
    work = frame.sort_values(
        ["domain12", "selection_hash", "paper_id"],
        kind="stable",
    )
    grouped = {
        str(domain): group.reset_index(drop=True)
        for domain, group in work.groupby("domain12", sort=True)
    }
    selected: List[pd.Series] = []
    offset = 0
    while len(selected) < count:
        changed = False
        for domain in sorted(grouped):
            group = grouped[domain]
            if offset < len(group):
                selected.append(group.iloc[offset])
                changed = True
                if len(selected) >= count:
                    break
        if not changed:
            break
        offset += 1
    return pd.DataFrame(selected)


def _current_blind_pack(
    paths: SuitePaths,
    seed: int,
    sample_per_stratum: int,
    labeler_count: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a current-score 30-paper answer key and blinded task templates."""
    oof = pd.read_parquet(
        _analysis(paths, "oof_d361264b867c/oof_predictions.parquet"),
        columns=[
            "paper_id",
            "publication_year",
            "domain12",
            "horizon",
            "model_id",
            "expected_diffusion_score",
            "realized_diffusion_target",
        ],
    )
    frame = oof.loc[
        oof["horizon"].eq(5)
        & oof["model_id"].eq("final_innovation_plus_k1")
    ].dropna(subset=["expected_diffusion_score"]).drop_duplicates("paper_id")
    frame["score_percentile"] = frame["expected_diffusion_score"].rank(
        method="first",
        pct=True,
    )
    frame["selection_hash"] = frame["paper_id"].map(
        lambda value: stable_hash(str(value), seed)
    )
    titles = _titles(paths).rename(
        columns={"source_display_name": "target_source_display_name"}
    )
    papers = pd.read_parquet(
        _dataset(paths, "papers_primary_articles.parquet"),
        columns=[
            "paper_id",
            "source_display_name",
            "display_topic_label",
            "openalex_primary_subfield",
        ],
    )
    frame = frame.merge(titles, on="paper_id", how="left").merge(
        papers,
        on="paper_id",
        how="left",
    )
    frame["source_display_name"] = frame["source_display_name"].fillna(
        frame["target_source_display_name"]
    )
    definitions = {
        "low": frame["score_percentile"].le(0.10),
        "middle": frame["score_percentile"].between(0.45, 0.55),
        "high": frame["score_percentile"].ge(0.90),
    }
    selected = []
    for tier, mask in definitions.items():
        tier_sample = _round_robin_sample(
            frame.loc[mask],
            sample_per_stratum,
        )
        tier_sample["global_fig3_tier"] = tier
        selected.append(tier_sample)
    answer = pd.concat(selected, ignore_index=True)
    answer["blinded_case_id"] = [
        f"F4V61-{index:03d}" for index in range(1, len(answer) + 1)
    ]
    answer["assignment_role"] = "primary_validation_labeling_sample"
    answer["validation_score"] = answer["expected_diffusion_score"]
    answer["text_available"] = False
    answer["evidence_packet_status"] = (
        "metadata_only_blocked_until_local_abstract_or_full_text_is_resolved"
    )
    packet_columns = [
        "blinded_case_id",
        "assignment_role",
        "title",
        "doi",
        "publication_year",
        "domain12",
        "openalex_primary_subfield",
        "display_topic_label",
        "source_display_name",
        "text_available",
        "evidence_packet_status",
    ]
    packet = answer[packet_columns].copy()
    templates = []
    for labeler in range(1, labeler_count + 1):
        assigned = packet.copy()
        assigned["labeler_id"] = f"labeler_{labeler}"
        assigned["label_novelty_1_5"] = np.nan
        assigned["label_significance_1_5"] = np.nan
        assigned["label_prior_art_1_5"] = np.nan
        assigned["label_confidence_1_5"] = np.nan
        assigned["label_notes"] = ""
        templates.append(assigned)
    labels = pd.concat(templates, ignore_index=True)
    audit = pd.DataFrame(
        [
            {
                "required_paper_labeler_rows": (
                    3 * sample_per_stratum * labeler_count
                ),
                "completed_paper_labeler_rows": 0,
                "score_source": "current v6.1 D5 temporal OOF",
                "paper_count": int(3 * sample_per_stratum),
                "labeler_count": labeler_count,
                "text_ready_papers": int(packet["text_available"].sum()),
                "publication_claim_ready": False,
                "blocking_reason": (
                    "human labels are empty and the frozen 1980–2017 table "
                    "does not contain abstracts/full text"
                ),
            }
        ]
    )
    return answer, packet, labels, audit


def _enhance_fig4(
    bundle: FigureBundle,
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    answer, packet, labels, audit = _current_blind_pack(
        paths,
        int(config["fig4"]["seed"]),
        int(config["fig4"]["blind_sample_per_stratum"]),
        int(config["fig4"]["labeler_count"]),
    )
    peer_review = pd.read_csv(
        paths["fig4_root"] / "fig4_metrics_summary.csv"
    )
    peer_columns = [
        column
        for column in (
            "paper_id",
            "title",
            "doi",
            "year",
            "has_peer_review_text",
            "agent_success",
            "novelty_alignment",
            "significance_alignment",
            "prior_art_alignment",
            "evidence_rigor_alignment",
            "limitations_alignment",
            "future_work_alignment",
            "claim_evidence_coverage",
        )
        if column in peer_review
    ]
    aspect_columns = [
        ("novelty_alignment", "Novelty"),
        ("significance_alignment", "Significance"),
        ("prior_art_alignment", "Prior-art difference"),
        ("evidence_rigor_alignment", "Evidence rigor"),
        ("limitations_alignment", "Limitations"),
        ("future_work_alignment", "Future work"),
        ("claim_evidence_coverage", "Claim-evidence coverage"),
    ]
    aspect_rows: List[Dict[str, Any]] = []
    for aspect_order, (column, label) in enumerate(aspect_columns, start=1):
        values = pd.to_numeric(peer_review[column], errors="coerce").dropna()
        estimate, low, high = bootstrap_mean_interval(
            values.to_numpy(float),
            iterations=2000,
            seed=int(config["fig4"]["seed"]) + aspect_order,
        )
        aspect_rows.append(
            {
                "aspect_order": aspect_order,
                "aspect": column,
                "aspect_label": label,
                "n_valid": int(len(values)),
                "mean_alignment": estimate,
                "ci_low": low,
                "ci_high": high,
                "cohort_role": (
                    "range-restricted transparent-peer-review diagnostic"
                ),
            }
        )
    aspect_summary = pd.DataFrame(aspect_rows)
    completion = labels[
        ["blinded_case_id", "labeler_id"]
    ].copy()
    completion["complete"] = labels[
        [
            "label_novelty_1_5",
            "label_significance_1_5",
            "label_prior_art_1_5",
        ]
    ].notna().all(axis=1).astype(int)
    bundle.tables.update(
        {
            "validation_sample_coverage": answer,
            "v6_1_blinded_answer_key": answer,
            "v6_1_blinded_packet": packet,
            "v6_1_blinded_label_templates": labels,
            "v6_1_blinded_completion_audit": audit,
            "transparent_peer_review_cohort": peer_review[peer_columns],
            "transparent_review_aspect_summary": aspect_summary,
            "blinded_label_completion_matrix": completion,
        }
    )
    retained_tables = (
        "validation_sample_coverage",
        "v6_1_blinded_answer_key",
        "v6_1_blinded_packet",
        "v6_1_blinded_label_templates",
        "v6_1_blinded_completion_audit",
        "transparent_peer_review_cohort",
        "transparent_review_aspect_summary",
        "blinded_label_completion_matrix",
    )
    bundle.tables = {
        name: bundle.tables[name]
        for name in retained_tables
    }
    bundle.status = STATUS_DRAFT_LABELS
    bundle.panel_text["a"] = {
        "sampling": (
            "Current v6.1 OOF low/middle/high score strata; 10 papers each."
        ),
        "required_judgements": 90,
        "completed_judgements": 0,
        "text_ready_papers": 0,
    }
    bundle.panel_text["cohort_b"] = {
        "paper_count": int(peer_review["paper_id"].nunique()),
        "status": "diagnostic_existing_transparent_peer_review_audit",
    }
    bundle.chart_contract["current_score_sample"] = True
    bundle.chart_contract["legacy_30_paper_pack_used"] = False
    bundle.chart_contract["human_labels_invented"] = False
    bundle.chart_contract["panels"] = {
        "a": {
            "mark": "two-cohort validation bridge",
            "data": [
                "v6_1_blinded_completion_audit",
                "transparent_peer_review_cohort",
            ],
        },
        "b": {
            "mark": "current-score validation-frame dot distribution",
            "data": ["validation_sample_coverage"],
        },
        "c": {
            "mark": "transparent-peer-review aspect interval plot",
            "data": ["transparent_review_aspect_summary"],
        },
        "d": {
            "mark": "30-paper by three-labeler completion matrix",
            "data": [
                "blinded_label_completion_matrix",
                "v6_1_blinded_completion_audit",
            ],
        },
        "e": {
            "mark": "blocked inferential endpoints",
            "data": ["v6_1_blinded_completion_audit"],
        },
    }
    bundle.title = (
        "Blinded human and transparent-peer-review validation remains evidence-gated"
    )
    bundle.source_paths = [
        _analysis(paths, "oof_d361264b867c/oof_predictions.parquet"),
        _dataset(paths, "papers_primary_articles.parquet"),
        paths["target_works"],
        paths["fig4_root"] / "fig4_metrics_summary.csv",
    ]
    return bundle


def _enhance_fig5(
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    local = copy.deepcopy(config)
    local["fig5"]["cutoffs"] = [2002, 2007, 2012]
    bundle = build_fig5(local, paths)
    windows = bundle.tables["historical_windows"].copy()
    # The base builder labels ``cutoff - 1`` as the training end even though
    # the scores themselves are frozen temporal-OOF predictions. Replace that
    # display field with the actual registered fold boundary for every scored
    # seed window.
    windows["training_end"] = [1999, 2004, 2009]
    windows["prediction_start"] = [2000, 2005, 2010]
    windows["prediction_end"] = [2002, 2007, 2012]
    windows["validation_start"] = [2003, 2008, 2013]
    windows["validation_end"] = [2006, 2011, 2016]
    windows["training_origin"] = windows["training_end"]
    windows["registered_seed_window"] = [
        "2000–2002",
        "2005–2007",
        "2010–2012",
    ]
    windows["registered_validation_window"] = [
        "2003–2006",
        "2008–2011",
        "2013–2016",
    ]
    fold_path = _analysis(
        paths,
        "oof_d361264b867c/temporal_folds.csv",
    )
    registered_folds = pd.read_csv(fold_path)
    registered_folds = registered_folds.loc[
        registered_folds["horizon"].eq(5)
    ]
    source_fold_ids = []
    source_test_windows = []
    score_window_pass = []
    for row in windows.itertuples(index=False):
        match = registered_folds.loc[
            registered_folds["train_year_max"].eq(int(row.training_end))
            & registered_folds["test_year_min"].le(
                int(row.prediction_start)
            )
            & registered_folds["test_year_max"].ge(
                int(row.prediction_end)
            )
        ]
        source_fold_ids.append(
            int(match.iloc[0]["fold_id"]) if len(match) == 1 else np.nan
        )
        source_test_windows.append(
            (
                f"{int(match.iloc[0]['test_year_min'])}–"
                f"{int(match.iloc[0]['test_year_max'])}"
            )
            if len(match) == 1
            else ""
        )
        score_window_pass.append(len(match) == 1)
    windows["source_oof_fold_id"] = source_fold_ids
    windows["source_oof_test_window"] = source_test_windows
    windows["score_window_within_registered_oof_fold"] = score_window_pass
    windows["temporal_contract_pass"] = (
        windows["training_end"].astype(int).to_numpy()
        < windows["prediction_start"].astype(int).to_numpy()
    ) & windows["score_window_within_registered_oof_fold"].to_numpy(bool)
    windows["d5_label_maturity_embargo_pass"] = False
    windows["evaluation_scope"] = (
        "retrospective publication-year OOF; D5 label maturity not embargoed"
    )
    bundle.tables["historical_windows"] = windows
    bundle.panel_text["a"] = {
        "forecast_origins": [1999, 2004, 2009],
        "all_temporal_contracts_pass": bool(
            windows["temporal_contract_pass"].all()
        ),
        "d5_label_maturity_embargo_pass": False,
        "claim_boundary": (
            "The registered OOF artifacts separate papers by publication "
            "year but do not enforce a five-year target-maturity embargo."
        ),
    }
    bundle.title = (
        "Retrospective historical windows test subsequent frontier ranking"
    )
    bundle.source_paths.append(fold_path)
    return bundle


def _audit_sample(paths: SuitePaths, per_domain: int) -> pd.DataFrame:
    """Select an outcome-blind domain × era × reference-volume audit sample."""
    source = pd.read_parquet(
        _analysis(paths, "screening_ceec00f0809b/stability_sample.parquet")
    )
    source = source.copy()
    source["reference_volume_bin"] = (
        source.groupby("domain12")["valid_reference_count"]
        .transform(
            lambda values: pd.qcut(
                values.rank(method="first"),
                4,
                labels=False,
                duplicates="drop",
            )
        )
        .astype("Int64")
    )
    selected: List[pd.Series] = []
    for _, domain in source.groupby("domain12", sort=True):
        strata = {
            (int(era), int(reference_bin)): group.sort_values(
                ["selection_hash", "paper_id"],
                kind="stable",
            ).reset_index(drop=True)
            for (era, reference_bin), group in domain.groupby(
                ["publication_era_5y", "reference_volume_bin"],
                sort=True,
            )
        }
        offset = 0
        domain_selected = 0
        while domain_selected < per_domain:
            changed = False
            for key in sorted(strata):
                group = strata[key]
                if offset >= len(group):
                    continue
                selected.append(group.iloc[offset])
                domain_selected += 1
                changed = True
                if domain_selected >= per_domain:
                    break
            if not changed:
                break
            offset += 1
    return pd.DataFrame(selected).reset_index(drop=True)


def _reference_doses(
    paths: SuitePaths,
    cache_path: Path,
    *,
    per_domain: int,
    repetitions: int,
    execute: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run or load exact deletion-dose recomputation on frozen local inputs."""
    sample = _audit_sample(paths, per_domain)
    sample_sha256 = stable_hash(
        "\n".join(sorted(sample["paper_id"].astype(str))),
        0,
    )
    input_paths = [
        _dataset(paths, "innovation_candidate_features.parquet"),
        _dataset(paths, "paper_references.parquet"),
        _dataset(paths, "reference_metadata.parquet"),
        _dataset(paths, "field_citation_events_aggregated.parquet"),
        _dataset(paths, "historical_paper_sources.parquet"),
        _dataset(paths, "historical_paper_references.parquet"),
        _dataset(paths, "source_field_citation_events.parquet"),
    ]
    input_signature = stable_hash(
        "\n".join(
            f"{path.resolve()}:{sha256_file(path)}"
            for path in input_paths
        ),
        0,
    )
    computation_version = "fig6-stratified-exact-deletion-v3"
    sample_summary = (
        sample.groupby("domain12", as_index=False)
        .agg(
            n_papers=("paper_id", "size"),
            era_count=("publication_era_5y", "nunique"),
            reference_volume_bin_count=(
                "reference_volume_bin",
                "nunique",
            ),
        )
    )
    sample_summary["requested_per_domain"] = per_domain
    sample_summary["shortfall"] = (
        sample_summary["n_papers"] < sample_summary["requested_per_domain"]
    )
    if cache_path.is_file():
        cached = pd.read_parquet(cache_path)
        cache_columns = {
            "audit_sample_sha256",
            "input_signature_sha256",
            "dose_repetitions",
            "audit_papers_per_domain",
            "computation_version",
        }
        cache_valid = (
            not cached.empty
            and cache_columns.issubset(cached.columns)
            and cached["audit_sample_sha256"].eq(sample_sha256).all()
            and cached["input_signature_sha256"].eq(input_signature).all()
            and cached["dose_repetitions"].eq(repetitions).all()
            and cached["audit_papers_per_domain"].eq(per_domain).all()
            and cached["computation_version"].eq(computation_version).all()
        )
        if cache_valid:
            return cached, sample_summary, sample
    if not execute:
        raise RuntimeError(
            "Fig.6 exact-dose cache is absent or stale and execution is disabled"
        )
    baseline = pd.DataFrame({"code_name": list(PRIMARY_FEATURES)})
    baseline["repetition"] = 0
    baseline["n_paired"] = len(sample)
    baseline["spearman"] = 1.0
    baseline["median_relative_error"] = 0.0
    baseline["relative_error_scale_floor"] = np.nan
    baseline["relative_error_denominator_policy"] = "full_reference_baseline"
    baseline["reference_retention"] = 1.0
    outputs = [baseline]
    full_features = pd.read_parquet(
        _dataset(paths, "innovation_candidate_features.parquet")
    )
    paper_references = pd.read_parquet(
        _dataset(paths, "paper_references.parquet")
    )
    reference_metadata = pd.read_parquet(
        _dataset(paths, "reference_metadata.parquet")
    )
    field_events = pd.read_parquet(
        _dataset(paths, "field_citation_events_aggregated.parquet")
    )
    historical_sources = pd.read_parquet(
        _dataset(paths, "historical_paper_sources.parquet")
    )
    historical_references = pd.read_parquet(
        _dataset(paths, "historical_paper_references.parquet")
    )
    source_field_events = pd.read_parquet(
        _dataset(paths, "source_field_citation_events.parquet")
    )
    for fraction in (0.75, 0.50, 0.25, 0.10):
        repetitions_frame, _ = reference_subsampling_stability(
            sample,
            full_features,
            paper_references,
            reference_metadata,
            field_events,
            historical_sources,
            historical_references,
            source_field_events,
            fraction=fraction,
            repetitions=repetitions,
            salt=f"experiments-new-fig6-{fraction}",
            field_profile_window_years=5,
            relative_error_denominator_policy="median_absolute_floor",
        )
        repetitions_frame = repetitions_frame.loc[
            repetitions_frame["code_name"].isin(PRIMARY_FEATURES)
        ].copy()
        repetitions_frame["reference_retention"] = fraction
        outputs.append(repetitions_frame)
    result = pd.concat(outputs, ignore_index=True)
    result["audit_sample_sha256"] = sample_sha256
    result["input_signature_sha256"] = input_signature
    result["dose_repetitions"] = repetitions
    result["audit_papers_per_domain"] = per_domain
    result["computation_version"] = computation_version
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path, index=False, compression="zstd")
    return result, sample_summary, sample


def _enhance_fig6(
    bundle: FigureBundle,
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    cache = paths.project_root / config["fig6"]["dose_cache"]
    doses, sample_summary, audit_sample = _reference_doses(
        paths,
        cache,
        per_domain=int(config["fig6"]["audit_papers_per_domain"]),
        repetitions=int(config["fig6"]["dose_repetitions"]),
        execute=bool(config["fig6"]["execute_reference_doses"]),
    )
    bundle.tables["reference_dose_stability"] = doses
    bundle.tables["audit_sample"] = audit_sample
    bundle.tables["audit_sample_by_domain"] = sample_summary
    domain_path = (
        paths["v6_1_figure_baseline"]
        / "experiment_07/data/domain_metrics.csv"
    )
    horizon_path = (
        paths["v6_1_figure_baseline"]
        / "experiment_05/data/horizon_metrics.csv"
    )
    fold_path = (
        paths["v6_1_figure_baseline"]
        / "experiment_06/data/fold_metrics.csv"
    )
    bundle.tables["registered_domain_metrics"] = pd.read_csv(domain_path)
    bundle.tables["registered_horizon_metrics"] = pd.read_csv(horizon_path)
    bundle.tables["registered_fold_metrics"] = pd.read_csv(fold_path)
    levels = sorted(
        float(value)
        for value in doses["reference_retention"].dropna().unique()
    )
    bundle.panel_text["reference_doses"] = {
        "available_retention_levels": levels,
        "audit_sample_n": int(sample_summary["n_papers"].sum()),
        "requested_sample_n": int(
            12 * config["fig6"]["audit_papers_per_domain"]
        ),
        "domain_shortfalls": sample_summary.loc[
            sample_summary["shortfall"], "domain12"
        ].tolist(),
        "contamination_mapping_status": (
            "registered exact implementation absent; no proxy values emitted"
        ),
    }
    bundle.chart_contract["panels"] = {
        "a": {
            "mark": "paired 12-domain D5 performance plot",
            "data": ["registered_domain_metrics"],
        },
        "b": {
            "mark": "exact multi-dose reference-deletion small multiples",
            "data": [
                "reference_dose_stability",
                "audit_sample",
                "audit_sample_by_domain",
            ],
        },
        "c": {
            "mark": "registered horizon and temporal-fold stability",
            "data": [
                "registered_horizon_metrics",
                "registered_fold_metrics",
            ],
        },
        "d": {
            "mark": "current-model specification curve",
            "data": ["specification_curve", "specification_flags"],
        },
        "e": {
            "mark": "reference-count and metadata-coverage reliability boundary",
            "data": ["reliability_units", "reliability_boundary"],
        },
        "f": {
            "mark": "heuristic failure cases and safeguards",
            "data": ["failure_modes", "failure_cases"],
        },
    }
    complete_levels = {0.1, 0.25, 0.5, 0.75, 1.0}.issubset(
        set(levels)
    )
    if not complete_levels:
        bundle.status = "DRAFT_REFERENCE_DOSES_INCOMPLETE"
    bundle.source_paths.extend(
        [
            cache,
            domain_path,
            horizon_path,
            fold_path,
            _analysis(paths, "screening_ceec00f0809b/stability_sample.parquet"),
            _dataset(paths, "innovation_candidate_features.parquet"),
            _dataset(paths, "paper_references.parquet"),
            _dataset(paths, "reference_metadata.parquet"),
            _dataset(paths, "field_citation_events_aggregated.parquet"),
            _dataset(paths, "historical_paper_sources.parquet"),
            _dataset(paths, "historical_paper_references.parquet"),
            _dataset(paths, "source_field_citation_events.parquet"),
        ]
    )
    return bundle


def _enhance_fig7(
    bundle: FigureBundle,
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    pure = pd.read_parquet(
        _analysis(
            paths,
            "supplement_innovation_only_3b387272d53d/"
            "innovation_only_oof_predictions.parquet",
        ),
        columns=[
            "paper_id",
            "publication_year",
            "domain12",
            "expected_diffusion_score",
            "realized_diffusion_target",
        ],
    )
    papers = pd.read_parquet(
        _dataset(paths, "papers_primary_articles.parquet"),
        columns=["paper_id", "source_display_name"],
    )
    features = pd.read_parquet(
        _dataset(paths, "innovation_candidate_features.parquet"),
        columns=["paper_id", "valid_reference_count"],
    )
    controls = pd.read_parquet(
        _dataset(paths, "control_features_v6_1.parquet"),
        columns=[
            "paper_id",
            "log_author_count",
            "log_institution_count",
            "log_country_count",
        ],
    )
    frame = pure.merge(papers, on="paper_id", how="left")
    frame = frame.merge(features, on="paper_id", how="left").merge(
        controls,
        on="paper_id",
        how="left",
    )
    names = frame["source_display_name"].fillna("").astype(str)
    frame["venue_family_audit"] = "Other Nature Portfolio"
    frame.loc[names.eq("Nature"), "venue_family_audit"] = "Nature flagship"
    frame.loc[
        names.eq("Nature Communications"), "venue_family_audit"
    ] = "Nature Communications"
    frame.loc[
        names.eq("Scientific Reports"), "venue_family_audit"
    ] = "Scientific Reports"
    frame.loc[
        names.str.startswith("Nature ")
        & ~names.eq("Nature Communications"),
        "venue_family_audit",
    ] = "Nature specialist journals"
    frame.loc[
        names.str.startswith("npj "),
        "venue_family_audit",
    ] = "npj series"
    frame.loc[
        names.str.startswith("Communications "),
        "venue_family_audit",
    ] = "Communications series"
    frame["innovation_percentile"] = grouped_percentile(
        frame,
        "expected_diffusion_score",
        ["domain12", "publication_year"],
        id_column="paper_id",
    )
    frame["impact_percentile"] = grouped_percentile(
        frame,
        "realized_diffusion_target",
        ["domain12", "publication_year"],
        id_column="paper_id",
    )
    allowed = set(bundle.tables["venue_portfolio"]["analysis_venue_family"])
    frame = frame.loc[frame["venue_family_audit"].isin(allowed)]
    association_rows = []
    for family_index, (family, group) in enumerate(
        frame.groupby("venue_family_audit", sort=True)
    ):
        paired = group[
            [
                "innovation_percentile",
                "impact_percentile",
                "domain12",
                "publication_year",
            ]
        ].dropna()
        low, high = _cluster_rank_interval(
            group,
            "innovation_percentile",
            "impact_percentile",
            iterations=int(config["fig7"]["bootstrap_iterations"]),
            seed=int(config["fig7"]["seed"]) + family_index,
        )
        association_rows.append(
            {
                "analysis_venue_family": family,
                "n_papers": int(len(paired)),
                "spearman": safe_spearman(
                    paired["innovation_percentile"],
                    paired["impact_percentile"],
                ),
                "ci_low": low,
                "ci_high": high,
                "normalization": "domain-year percentile",
                "interval_method": (
                    "fixed-rank domain-year cluster bootstrap"
                ),
            }
        )
    associations = pd.DataFrame(association_rows)
    balance = (
        frame.groupby("venue_family_audit", as_index=False)
        .agg(
            n_papers=("paper_id", "size"),
            year_mean=("publication_year", "mean"),
            reference_median=("valid_reference_count", "median"),
            log_author_mean=("log_author_count", "mean"),
            log_institution_mean=("log_institution_count", "mean"),
            log_country_mean=("log_country_count", "mean"),
            domain_count=("domain12", "nunique"),
        )
        .rename(columns={"venue_family_audit": "analysis_venue_family"})
    )
    bundle.tables["venue_within_association"] = associations
    bundle.tables["venue_common_support_audit"] = balance
    institution_count_integrity_pass = not np.allclose(
        balance["log_author_mean"].to_numpy(float),
        balance["log_institution_mean"].to_numpy(float),
        equal_nan=True,
    )
    bundle.chart_contract["panels"] = {
        "a": {
            "mark": "venue innovation-D5 portfolio map",
            "data": ["venue_portfolio"],
        },
        "b": {
            "mark": "paper-bootstrap venue rank distributions",
            "data": ["venue_bootstrap_ranks"],
        },
        "c": {
            "mark": "top-1/5-percent enrichment interval plot",
            "data": ["venue_enrichment"],
        },
        "d": {
            "mark": "five-angle venue profile small multiples",
            "data": ["venue_angle_profiles"],
        },
        "e": {
            "mark": "within-venue innovation-D5 interval plot",
            "data": ["venue_within_association"],
        },
        "f": {
            "mark": "common-support audit dot table",
            "data": ["venue_common_support_audit"],
        },
    }
    bundle.panel_text["f"] = (
        "Within-venue association between publication-time innovation-only "
        "score and realized D5, both normalized within domain-year."
    )
    bundle.panel_text["g"] = (
        "Association-only audit of field, year, reference-volume and team-size support."
    )
    bundle.panel_text["support_audit"] = {
        "common_support_established": False,
        "institution_count_integrity_pass": (
            institution_count_integrity_pass
        ),
        "warning": (
            ""
            if institution_count_integrity_pass
            else (
                "Frozen author and institution counts are identical; "
                "institution count is withheld from interpretation."
            )
        ),
    }
    bundle.chart_contract["common_support_established"] = False
    bundle.chart_contract[
        "institution_count_integrity_pass"
    ] = institution_count_integrity_pass
    bundle.source_paths.append(
        _dataset(paths, "control_features_v6_1.parquet")
    )
    return bundle


def _enhance_fig9(
    bundle: FigureBundle,
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    case = bundle.tables["case_manifest"].iloc[0]
    manuscript = Path(str(case["manuscript_markdown"]))
    checkpoint = Path(str(config["fig9"]["checkpoint_path"]))
    if not checkpoint.is_absolute():
        checkpoint = paths.project_root / checkpoint
    boundary = pd.DataFrame(
        [
            {
                "case_id": case["case_id"],
                "publication_year": int(case["year"]),
                "main_oof_cohort_max_year": 2017,
                "manuscript_available": manuscript.is_file(),
                "checkpoint_available": checkpoint.exists(),
                "current_eight_indicator_fingerprint_available": False,
                "reason": (
                    "case DOI is not resolved to a frozen local OpenAlex work "
                    "with reference IDs; no indicator imputation is permitted"
                ),
            }
        ]
    )
    bundle.tables["case_measurement_boundary"] = boundary
    bundle.chart_contract["case_indicator_imputation"] = False
    bundle.chart_contract["checkpoint_path"] = str(checkpoint.resolve())
    bundle.panel_text["case_materialization"] = boundary.iloc[0].to_dict()
    if not checkpoint.exists():
        bundle.status = "BLOCKED_MISSING_MODEL_USING_HASHED_PRIOR_OUTPUT"
    return bundle


def _enhance_fig10(bundle: FigureBundle, paths: SuitePaths) -> FigureBundle:
    reruns = bundle.tables["true_module_reruns"].copy()
    rows = []
    full = reruns.loc[reruns["variant"].eq("full ASPR")]
    for variant, group in reruns.loc[
        ~reruns["variant"].eq("full ASPR")
    ].groupby("variant"):
        rows.append(
            {
                "variant": variant,
                "case_count": int(group["case_id"].nunique()),
                "full_source": "|".join(sorted(full["source"].astype(str).unique())),
                "variant_source": "|".join(
                    sorted(group["source"].astype(str).unique())
                ),
                "same_generation_path": bool(
                    set(full["source"].astype(str))
                    == set(group["source"].astype(str))
                ),
                "one_switch_contract_verified": False,
            }
        )
    audit = pd.DataFrame(rows)
    required_variants = pd.DataFrame(
        [
            {
                "variant": variant,
                "required_case_count": 50,
                "same_model": False,
                "same_prompt": False,
                "same_retrieval_cache": False,
                "same_scorer": False,
                "same_decoding": False,
                "one_switch_only": False,
                "ready_for_main_ablation": False,
            }
            for variant in (
                "Full ASPR",
                "no graph evidence",
                "no ASPR-Qwen",
                "no prior-art retrieval",
                "no evidence trace",
                "no fusion",
                "no verifier",
                "generic LLM baseline",
            )
        ]
    )
    bundle.tables["ablation_comparability_audit"] = audit
    bundle.tables["required_same_path_variants"] = required_variants
    bundle.status = STATUS_BLOCKED_COMPARABILITY
    bundle.chart_contract["same_path_gate"] = {
        "passed": bool(
            audit["same_generation_path"].all()
            and audit["one_switch_contract_verified"].all()
        ),
        "legacy_400_rows_main_evidence": False,
    }
    bundle.panel_text["comparability"] = {
        "same_path_variants": int(audit["same_generation_path"].sum()),
        "variant_count": int(len(audit)),
        "claim": "blocked until a unified one-switch runner is executed",
    }
    bundle.panel_text["d"] = {
        "blocked": True,
        "completed_judgements": 0,
        "required_judgements": 750,
    }
    bundle.panel_text["e"] = {
        "blocked": True,
        "reason": "quality and runtime must be measured on same-path reruns",
    }
    bundle.panel_text["f"] = (
        "Representative degradation cases are withheld until the same-path "
        "one-switch rerun is complete."
    )
    bundle.chart_contract["panels"] = {
        "a": {
            "mark": "module switch inventory",
            "data": ["module_inventory"],
        },
        "b": {
            "mark": "comparability gate",
            "data": ["ablation_comparability_audit"],
        },
        "c": {
            "mark": "required same-path variant contract",
            "data": ["required_same_path_variants"],
        },
        "d": {
            "mark": "blocked human-preference panel",
            "data": ["preference_completion_audit"],
        },
        "e": {
            "mark": "blocked quality-cost panel",
            "data": [],
        },
        "f": {
            "mark": "blocked degradation-case panel",
            "data": [],
        },
    }
    bundle.chart_contract["mismatched_numeric_deltas_rendered"] = False
    bundle.chart_contract["projected_quality_cost_rendered"] = False
    bundle.chart_contract["representative_ablation_claims_rendered"] = False
    bundle.notes.append(
        "The blocked main figure renders the comparability gate only; legacy "
        "protocol-mismatched numeric tables remain machine-auditable but are "
        "not plotted as ablation evidence."
    )
    return bundle


def build_new_bundle(
    figure_id: int,
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    """Build one new figure bundle using current artifacts and strict adapters."""
    if figure_id == 2:
        return build_fig2_evidence_map(config, paths)
    if figure_id == 5:
        return _enhance_fig5(config, paths)
    bundle = BASE_BUILDERS[figure_id](config, paths)
    if figure_id == 1:
        bundle.status = STATUS_DESCRIPTIVE
    elif figure_id == 3:
        bundle = _enhance_fig3(bundle, paths)
    elif figure_id == 4:
        bundle = _enhance_fig4(bundle, config, paths)
    elif figure_id == 6:
        bundle = _enhance_fig6(bundle, config, paths)
    elif figure_id == 7:
        bundle = _enhance_fig7(bundle, config, paths)
    elif figure_id == 9:
        bundle = _enhance_fig9(bundle, config, paths)
    elif figure_id == 10:
        bundle = _enhance_fig10(bundle, paths)
    return bundle
