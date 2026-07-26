"""Data builders for Fig.6–Fig.10 of the Nature-style ASPR figure suite."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from experiments.common.new.base.builders_1_5 import PRIMARY_FEATURES, _v61_paths
from experiments.common.new.base.common import (
    ANGLE_LABELS,
    FEATURE_LABELS,
    FigureBundle,
    SuitePaths,
    grouped_percentile,
    load_json,
    numeric,
    percentile_rank,
    stable_seed,
)


ANGLE_FEATURES: Dict[str, List[str]] = {
    "A1_COMBINATION_RARITY": ["reference_overlap_novelty_t0"],
    "A2_ATYPICALITY_CONVENTIONALITY": ["hypergeom_conventionality_median_t0"],
    "A3_FIRST_TIME_COMBINATION": ["first_time_source_pair_share"],
    "A4_KNOWLEDGE_BREADTH_BALANCE": [
        "field_gini_balance",
        "reference_other_field_share",
        "field_variety",
    ],
    "A5_COGNITIVE_DISTANCE_INTEGRATION": [
        "field_disparity_cosine_mean",
        "rao_stirling_integration",
    ],
}


def _bootstrap_delta(
    values: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> Tuple[float, float, float]:
    """Bootstrap one paired-difference vector."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(iterations), dtype=float)
    for index in range(int(iterations)):
        estimates[index] = rng.choice(clean, size=len(clean), replace=True).mean()
    return (
        float(clean.mean()),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


# ============================================================================
# Fig.6 — robustness, boundaries and failure modes
# ============================================================================


def _robustness_compass(paths: SuitePaths) -> pd.DataFrame:
    """Combine registered v6.1 checks with explicitly labeled legacy proxies."""
    resolved = _v61_paths(paths)
    resampling = pd.read_csv(
        resolved["screening"] / "reference_subsampling_summary.csv"
    )
    primary = resampling.loc[resampling["code_name"].isin(PRIMARY_FEATURES)]
    horizon = pd.read_csv(
        paths["v6_1_figure_baseline"] / "experiment_05/data/horizon_metrics.csv"
    )
    final_horizon = horizon.loc[
        horizon["model_id"].eq("final_innovation_plus_k1")
    ].set_index("horizon")["spearman_expected"]
    control = pd.read_csv(
        paths["v6_1_figure_baseline"] / "experiment_10/data/control_sensitivity.csv"
    ).set_index("model_id")["spearman_expected"]
    domains = pd.read_csv(
        paths["v6_1_figure_baseline"] / "experiment_07/data/domain_metrics.csv"
    )
    final_domain = domains.loc[
        domains["model_id"].eq("final_innovation_plus_k1"),
        "spearman_expected",
    ]
    stress = pd.read_csv(
        paths["v6_1_figure_baseline"] / "experiment_10/data/stress_test_gains.csv"
    )
    fold_gains = stress.loc[stress["stratum"].eq("时间折"), "spearman_gain"]
    checks = pd.read_csv(
        paths["v6_1_figure_baseline"] / "experiment_10/data/reproducibility_checks.csv"
    )
    current = [
        {
            "axis": "80% reference resampling",
            "value": float(primary["stability_spearman"].min()),
            "evidence_scope": "registered_v6_1",
            "definition": "minimum Spearman among eight primary indicators",
        },
        {
            "axis": "D3/D8 horizon retention",
            "value": float(min(final_horizon.loc[3], final_horizon.loc[8]) / final_horizon.loc[5]),
            "evidence_scope": "registered_v6_1",
            "definition": "minimum D3/D8 final-model Spearman divided by D5",
        },
        {
            "axis": "K2 control retention",
            "value": float(
                control.loc["final_innovation_plus_k2"]
                / control.loc["final_innovation_plus_k1"]
            ),
            "evidence_scope": "registered_v6_1",
            "definition": "final 8+K2 divided by final 8+K1",
        },
        {
            "axis": "Weakest-domain retention",
            "value": float(final_domain.min() / control.loc["final_innovation_plus_k1"]),
            "evidence_scope": "registered_v6_1",
            "definition": "weakest domain Spearman divided by global D5 Spearman",
        },
        {
            "axis": "Positive fold gain",
            "value": float(fold_gains.gt(0).mean()),
            "evidence_scope": "registered_v6_1",
            "definition": "fraction of D3/D5/D8 fold gains above zero",
        },
        {
            "axis": "Exact replay checks",
            "value": float(checks["passed"].sum() / max(checks["total"].sum(), 1)),
            "evidence_scope": "registered_v6_1",
            "definition": "passed frozen replay and integrity checks",
        },
    ]
    legacy_stability = pd.read_csv(
        paths["fig6_root"] / "fig6_primary_model_stability.csv"
    )
    legacy_volume = pd.read_csv(paths["fig6_root"] / "fig6_volume_sensitivity.csv")
    legacy_proxy = float(legacy_stability["rank_spearman"].median())
    legacy_reference = float(
        legacy_volume.loc[
            legacy_volume["literature_fraction"].eq(0.75),
            "performance_retention_mean",
        ].median()
    )
    legacy = []
    for row in current:
        value = legacy_reference if row["axis"] == "80% reference resampling" else legacy_proxy
        legacy.append(
            {
                "axis": row["axis"],
                "value": value,
                "evidence_scope": "legacy_proxy",
                "definition": "legacy cached-score/graph diagnostic; not v6.1 registered evidence",
            }
        )
    return pd.DataFrame(current + legacy)


def _registered_reference_resampling(paths: SuitePaths) -> pd.DataFrame:
    """Return the registered 80% reference subsampling repetitions."""
    resolved = _v61_paths(paths)
    data = pd.read_csv(
        resolved["screening"] / "reference_subsampling_repetitions.csv"
    )
    output = data.loc[data["code_name"].isin(PRIMARY_FEATURES)].copy()
    baseline = output[["code_name"]].drop_duplicates()
    baseline["repetition"] = 0
    baseline["n_paired"] = np.nan
    baseline["spearman"] = 1.0
    baseline["median_relative_error"] = 0.0
    baseline["relative_error_scale_floor"] = np.nan
    baseline["relative_error_denominator_policy"] = "exact_full_reference_baseline"
    baseline["reference_retention"] = 1.0
    output["reference_retention"] = 0.8
    return pd.concat([baseline, output], ignore_index=True)


def _specification_curve(paths: SuitePaths) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble current v6.1 specifications and keep legacy choices separate."""
    resolved = _v61_paths(paths)
    model = pd.read_csv(resolved["model_points"]).copy()
    model["specification"] = model["model_id"]
    model["spearman"] = model["spearman_expected"]
    model["scope"] = "current_v6_1"
    flags = []
    for row in model.itertuples(index=False):
        model_id = str(row.model_id)
        flags.extend(
            [
                {
                    "specification": model_id,
                    "choice": "8 innovation indicators",
                    "enabled": int("innovation" in model_id or "core8" in model_id or "v6_primary" in model_id),
                    "scope": "current_v6_1",
                },
                {
                    "specification": model_id,
                    "choice": "K1 controls",
                    "enabled": int("k1" in model_id),
                    "scope": "current_v6_1",
                },
                {
                    "specification": model_id,
                    "choice": "K2 strong controls",
                    "enabled": int("k2" in model_id),
                    "scope": "current_v6_1",
                },
                {
                    "specification": model_id,
                    "choice": "legacy B0 indicators",
                    "enabled": int("b0" in model_id),
                    "scope": "current_v6_1",
                },
            ]
        )
    current = model[
        ["specification", "spearman", "scope", "model_id", "n_rank_valid"]
    ].copy()
    current["delta_from_main"] = (
        current["spearman"]
        - float(
            current.loc[
                current["model_id"].eq("final_innovation_plus_k1"),
                "spearman",
            ].iloc[0]
        )
    )
    legacy_path = paths["fig6_root"] / "fig6_modeling_choice_reproducibility.csv"
    legacy = pd.read_csv(legacy_path)
    legacy_output = pd.DataFrame(
        {
            "specification": "legacy:" + legacy["choice"].astype(str),
            "spearman": legacy["alternative_spearman"],
            "scope": "legacy_proxy",
            "model_id": legacy["choice"],
            "n_rank_valid": np.nan,
            "delta_from_main": legacy["delta_vs_learned"],
        }
    )
    return pd.concat([current, legacy_output], ignore_index=True), pd.DataFrame(flags)


def _reliability_boundary(paths: SuitePaths, config: Mapping[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate a descriptive reference-volume/coverage reliability region."""
    resolved = _v61_paths(paths)
    features = pd.read_parquet(
        resolved["features"],
        columns=[
            "paper_id",
            "publication_year",
            "domain12",
            "valid_reference_count",
            "source_mapping_coverage",
            "field_mapping_coverage",
            *PRIMARY_FEATURES,
        ],
    )
    features["mapping_coverage"] = features[
        ["source_mapping_coverage", "field_mapping_coverage"]
    ].min(axis=1)
    features["core8_complete"] = features[list(PRIMARY_FEATURES)].notna().all(axis=1).astype(int)
    aggregate = (
        features.groupby(["domain12", "publication_year"], as_index=False)
        .agg(
            n_papers=("paper_id", "size"),
            median_reference_count=("valid_reference_count", "median"),
            median_mapping_coverage=("mapping_coverage", "median"),
            core8_complete_rate=("core8_complete", "mean"),
        )
    )
    aggregate["reliable"] = (
        aggregate["median_reference_count"].ge(
            int(config["fig6"]["reliability_min_references"])
        )
        & aggregate["median_mapping_coverage"].ge(
            float(config["fig6"]["reliability_min_coverage"])
        )
        & aggregate["core8_complete_rate"].ge(0.70)
    ).astype(int)
    x = np.log1p(aggregate["median_reference_count"].to_numpy(float))
    y = aggregate["median_mapping_coverage"].to_numpy(float)
    labels = aggregate["reliable"].to_numpy(int)
    grid_reference = np.linspace(0, max(float(aggregate["median_reference_count"].quantile(0.99)), 20), 80)
    grid_coverage = np.linspace(0, 1, 80)
    xx, yy = np.meshgrid(grid_reference, grid_coverage)
    if len(np.unique(labels)) == 2:
        model = LogisticRegression(random_state=20260725).fit(
            np.column_stack([x, y]),
            labels,
            sample_weight=np.sqrt(aggregate["n_papers"].to_numpy(float)),
        )
        probability = model.predict_proba(
            np.column_stack([np.log1p(xx.ravel()), yy.ravel()])
        )[:, 1]
    else:
        probability = np.full(xx.size, float(labels.mean()))
    boundary = pd.DataFrame(
        {
            "reference_count": xx.ravel(),
            "mapping_coverage": yy.ravel(),
            "pass_probability": probability,
        }
    )
    return aggregate, boundary


def build_fig6(
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    """Build Fig.6 registered robustness, legacy diagnostics and boundaries."""
    compass = _robustness_compass(paths)
    reference = _registered_reference_resampling(paths)
    specification, flags = _specification_curve(paths)
    reliability, boundary = _reliability_boundary(paths, config)
    failure_modes = pd.read_csv(paths["fig6_root"] / "fig6_failure_modes.csv")
    failure_cases = pd.read_csv(paths["fig6_root"] / "fig6_failure_mode_cases.csv")
    legacy_quality = pd.read_csv(
        paths["fig6_root"] / "fig6_data_quality_perturbation.csv"
    )
    tables = {
        "robustness_compass": compass,
        "registered_reference_resampling": reference,
        "specification_curve": specification,
        "specification_flags": flags,
        "reliability_units": reliability,
        "reliability_boundary": boundary,
        "failure_modes": failure_modes,
        "failure_cases": failure_cases,
        "legacy_quality_perturbations": legacy_quality,
    }
    panel_text = {
        "a": "Current registered checks and legacy cached-score proxies are drawn as separate contours.",
        "b": {
            "available_retention_levels": [1.0, 0.8],
            "warning": "The frozen registered experiment contains 80% subsampling only; 75/50/25/10% values are not invented.",
        },
        "c": "The upper curve contains current v6.1 model specifications; legacy graph choices are visually separated.",
        "d": {
            "unit": "domain-year",
            "reliable_count": int(reliability["reliable"].sum()),
            "total_count": int(len(reliability)),
        },
        "e": {
            "source_status": "heuristic_from_fig4_cached_metrics",
            "warning": "Failure frequencies are diagnostic heuristics, not blinded manual adjudication.",
        },
        "warning": config["claim_boundaries"]["fig6"],
    }
    contract = {
        "figure_id": 6,
        "panels": {
            "a": {"mark": "robustness compass", "data": ["robustness_compass"]},
            "b": {"mark": "reference-resampling raincloud", "data": ["registered_reference_resampling"]},
            "c": {"mark": "specification curve + dot matrix", "data": ["specification_curve", "specification_flags"]},
            "d": {
                "mark": "registered-gate reliability scatter",
                "data": ["reliability_units"],
                "supplemental_data": ["reliability_boundary"],
            },
            "e": {"mark": "Pareto lollipop + diagnostic cards", "data": ["failure_modes", "failure_cases"]},
        },
        "legacy_evidence_may_support_v6_1_claims": False,
    }
    return FigureBundle(
        figure_id=6,
        title="Robustness evidence defines both stable and unreliable regions",
        status="complete_with_legacy_separation",
        tables=tables,
        panel_text=panel_text,
        chart_contract=contract,
        source_paths=[
            _v61_paths(paths)["screening"] / "reference_subsampling_repetitions.csv",
            _v61_paths(paths)["screening"] / "reference_subsampling_summary.csv",
            _v61_paths(paths)["features"],
            paths["v6_1_figure_baseline"] / "experiment_05/data/horizon_metrics.csv",
            paths["v6_1_figure_baseline"] / "experiment_07/data/domain_metrics.csv",
            paths["v6_1_figure_baseline"] / "experiment_10/data/control_sensitivity.csv",
            paths["fig6_root"] / "fig6_primary_model_stability.csv",
            paths["fig6_root"] / "fig6_failure_modes.csv",
        ],
        notes=[config["claim_boundaries"]["fig6"]],
    )


# ============================================================================
# Fig.7 — venue portfolio analysis using venue-excluded scores
# ============================================================================


def _venue_group(source_name: pd.Series) -> pd.Series:
    """Create interpretable Nature-corpus venue families from local source names."""
    names = source_name.fillna("Unknown").astype(str)
    output = pd.Series("Other Nature Portfolio", index=names.index, dtype=object)
    output.loc[names.eq("Nature")] = "Nature flagship"
    output.loc[names.eq("Nature Communications")] = "Nature Communications"
    output.loc[names.eq("Scientific Reports")] = "Scientific Reports"
    output.loc[names.str.startswith("Nature ") & ~names.eq("Nature Communications")] = (
        "Nature specialist journals"
    )
    output.loc[names.str.startswith("npj ")] = "npj series"
    output.loc[names.str.startswith("Communications ")] = "Communications series"
    return output


def _venue_base(paths: SuitePaths) -> pd.DataFrame:
    """Join innovation-only OOF scores to local venue and indicator data."""
    resolved = _v61_paths(paths)
    pure = pd.read_parquet(
        resolved["pure_oof"],
        columns=[
            "paper_id",
            "publication_year",
            "domain12",
            "expected_diffusion_score",
            "realized_diffusion_target",
        ],
    )
    papers = pd.read_parquet(
        resolved["papers"],
        columns=[
            "paper_id",
            "source_display_name",
            "venue_family",
        ],
    )
    features = pd.read_parquet(
        resolved["features"],
        columns=["paper_id", *PRIMARY_FEATURES],
    )
    frame = pure.merge(papers, on="paper_id", how="left").merge(
        features,
        on="paper_id",
        how="left",
    )
    frame["analysis_venue_family"] = _venue_group(frame["source_display_name"])
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
    for feature in PRIMARY_FEATURES:
        frame[f"{feature}__percentile"] = grouped_percentile(
            frame,
            feature,
            ["domain12", "publication_year"],
            id_column="paper_id",
        )
    return frame


def _venue_portfolio(frame: pd.DataFrame, minimum_n: int) -> pd.DataFrame:
    """Summarize field-year normalized innovation and impact by family."""
    output = (
        frame.groupby("analysis_venue_family", as_index=False)
        .agg(
            n_papers=("paper_id", "size"),
            innovation_signal=("innovation_percentile", "mean"),
            future_diffusion=("impact_percentile", "mean"),
        )
    )
    return output.loc[output["n_papers"].ge(int(minimum_n))].copy()


def _venue_bootstrap_ranks(
    frame: pd.DataFrame,
    families: Sequence[str],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap venue innovation-signal ranks at the paper level."""
    rng = np.random.default_rng(int(seed))
    pools = {
        family: frame.loc[
            frame["analysis_venue_family"].eq(family),
            "innovation_percentile",
        ].dropna().to_numpy(float)
        for family in families
    }
    rows: List[Dict[str, Any]] = []
    for repetition in range(1, int(iterations) + 1):
        estimates = {
            family: float(rng.choice(values, size=len(values), replace=True).mean())
            for family, values in pools.items()
            if len(values)
        }
        order = sorted(estimates, key=lambda family: (-estimates[family], family))
        for rank, family in enumerate(order, start=1):
            rows.append(
                {
                    "bootstrap": repetition,
                    "analysis_venue_family": family,
                    "rank": rank,
                    "mean_innovation_signal": estimates[family],
                }
            )
    return pd.DataFrame(rows)


def _venue_enrichment(
    frame: pd.DataFrame,
    families: Sequence[str],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Estimate observed/expected top-innovation enrichment by venue."""
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    work = frame.assign(global_innovation_percentile=frame["innovation_percentile"])
    for threshold in (0.99, 0.95):
        base_rate = float(work["global_innovation_percentile"].ge(threshold).mean())
        for family in families:
            group = work.loc[
                work["analysis_venue_family"].eq(family),
                "global_innovation_percentile",
            ].dropna().to_numpy(float)
            indicator = group >= threshold
            draws = np.empty(int(iterations), dtype=float)
            for index in range(int(iterations)):
                sample = rng.choice(indicator, size=len(indicator), replace=True)
                draws[index] = sample.mean() / max(base_rate, 1e-12)
            rows.append(
                {
                    "analysis_venue_family": family,
                    "threshold": f"Top {int((1-threshold)*100)}%",
                    "n_papers": len(group),
                    "enrichment": float(indicator.mean() / max(base_rate, 1e-12)),
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def _venue_angle_profiles(
    frame: pd.DataFrame,
    families: Sequence[str],
) -> pd.DataFrame:
    """Average five normalized observation-angle components by family."""
    rows: List[Dict[str, Any]] = []
    for family in families:
        group = frame.loc[frame["analysis_venue_family"].eq(family)]
        for angle_id, features in ANGLE_FEATURES.items():
            values = group[[f"{feature}__percentile" for feature in features]].mean(axis=1)
            rows.append(
                {
                    "analysis_venue_family": family,
                    "angle_id": angle_id,
                    "angle_label": ANGLE_LABELS[angle_id],
                    "mean_percentile": float(values.mean()),
                    "n_papers": int(values.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def _venue_time_flow(frame: pd.DataFrame, families: Sequence[str]) -> pd.DataFrame:
    """Calculate shares of top-decile innovation papers across decades."""
    work = frame.copy()
    work["decade"] = (work["publication_year"] // 5 * 5).astype(int)
    work["global_percentile"] = work["innovation_percentile"]
    top = work.loc[work["global_percentile"].ge(0.90)].copy()
    top["flow_family"] = np.where(
        top["analysis_venue_family"].isin(families),
        top["analysis_venue_family"],
        "Other",
    )
    output = (
        top.groupby(["decade", "flow_family"], as_index=False)
        .size()
        .rename(columns={"size": "paper_count"})
    )
    output["share"] = output["paper_count"] / output.groupby("decade")["paper_count"].transform("sum")
    return output


def build_fig7(
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    """Build Fig.7 venue portfolio analyses without venue in the score model."""
    frame = _venue_base(paths)
    portfolio = _venue_portfolio(
        frame,
        int(config["fig7"]["minimum_venue_papers"]),
    )
    families = portfolio.sort_values("n_papers", ascending=False)[
        "analysis_venue_family"
    ].tolist()
    ranks = _venue_bootstrap_ranks(
        frame,
        families,
        int(config["fig7"]["bootstrap_iterations"]),
        int(config["fig7"]["seed"]),
    )
    enrichment = _venue_enrichment(
        frame,
        families,
        int(config["fig7"]["bootstrap_iterations"]),
        int(config["fig7"]["seed"]),
    )
    profiles = _venue_angle_profiles(frame, families)
    flow = _venue_time_flow(frame, families)
    tables = {
        "venue_portfolio": portfolio,
        "venue_bootstrap_ranks": ranks,
        "venue_enrichment": enrichment,
        "venue_angle_profiles": profiles,
        "venue_time_flow": flow,
    }
    panel_text = {
        "a": "Both axes are normalized within domain-year; bubble size is paper count.",
        "b": "Rank distributions quantify uncertainty in mean innovation-only signal.",
        "c": "Observed/expected enrichment uses publication-time innovation-only scores.",
        "d": "Small-multiple radars show mechanism shape only; no area-based inference.",
        "e": "Five-year bins show where top-decile innovation-only papers were published.",
        "warning": config["claim_boundaries"]["fig7"],
    }
    contract = {
        "figure_id": 7,
        "score_contract": {
            "model_id": "innovation_only",
            "contains_venue_family": False,
            "normalization": "domain-year percentile",
        },
        "panels": {
            "a": {"mark": "venue portfolio bubbles", "data": ["venue_portfolio"]},
            "b": {"mark": "rank ridgelines", "data": ["venue_bootstrap_ranks"]},
            "c": {"mark": "paired enrichment intervals", "data": ["venue_enrichment"]},
            "d": {"mark": "smooth radar small multiples", "data": ["venue_angle_profiles"]},
            "e": {"mark": "100% river plot", "data": ["venue_time_flow"]},
        },
        "causal_venue_claim": False,
    }
    return FigureBundle(
        figure_id=7,
        title="Venue families differ in innovation-signal portfolios and enrichment",
        status="complete_venue_excluded_score",
        tables=tables,
        panel_text=panel_text,
        chart_contract=contract,
        source_paths=[
            _v61_paths(paths)["pure_oof"],
            _v61_paths(paths)["papers"],
            _v61_paths(paths)["features"],
        ],
        notes=[config["claim_boundaries"]["fig7"]],
    )


# ============================================================================
# Fig.8 — architecture contract
# ============================================================================


def build_fig8(
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    """Build the non-performance ASPR dual-path architecture contract."""
    registry = load_json(paths["candidate_registry"])
    primary_rows = []
    for candidate_id, candidate in registry["candidates"].items():
        if candidate.get("final_role") == "primary":
            primary_rows.append(
                {
                    "candidate_id": candidate_id,
                    "code_name": candidate["code_name"],
                    "indicator": FEATURE_LABELS[candidate["code_name"]],
                    "angle_id": candidate["angle_id"],
                    "angle": ANGLE_LABELS[candidate["angle_id"]],
                }
            )
    nodes = pd.DataFrame(
        [
            ("input", "Input manuscript + references", "input", 0),
            ("prior", "Publication-prior retrieval", "graph", 1),
            ("indicators", "Five angles · eight indicators", "graph", 2),
            ("packet", "Claim-level evidence packet", "graph", 3),
            ("qwen", "ASPR-Qwen reviewer", "qwen", 2),
            ("fusion", "Fusion", "fusion", 4),
            ("verifier", "Verifier", "fusion", 5),
            ("review", "Evidence-grounded review", "output", 6),
        ],
        columns=["node_id", "label", "lane", "order"],
    )
    edges = pd.DataFrame(
        [
            ("input", "prior"),
            ("prior", "indicators"),
            ("indicators", "packet"),
            ("input", "qwen"),
            ("packet", "fusion"),
            ("qwen", "fusion"),
            ("fusion", "verifier"),
            ("verifier", "review"),
        ],
        columns=["source", "target"],
    )
    tables = {
        "architecture_nodes": nodes,
        "architecture_edges": edges,
        "primary_indicators": pd.DataFrame(primary_rows).sort_values(["angle_id", "candidate_id"]),
    }
    panel_text = {
        "exact_labels": [
            "Input manuscript",
            "Publication-prior evidence",
            "Five observation angles",
            "ASPR-Qwen reviewer",
            "Fusion",
            "Verifier",
            "Evidence-grounded review",
        ],
        "output_schema": [
            "novelty stance",
            "prior-art comparison",
            "limitations",
            "recommendation",
            "evidence IDs",
        ],
        "warning": config["claim_boundaries"]["fig8"],
    }
    contract = {
        "figure_id": 8,
        "asset_type": "algorithm_framework",
        "background_asset": str(paths["fig8_image_base"]),
        "image_asset_may_render_numeric_values": False,
        "performance_claim": False,
    }
    return FigureBundle(
        figure_id=8,
        title="ASPR combines graph evidence with reviewer-language generation",
        status="complete_architecture_only",
        tables=tables,
        panel_text=panel_text,
        chart_contract=contract,
        source_paths=[
            paths["candidate_registry"],
            paths["fig8_image_base"],
            paths.project_root
            / "experiments/common/new/base/prompts/fig08_framework_base.txt",
        ],
        notes=[config["claim_boundaries"]["fig8"]],
    )


# ============================================================================
# Fig.9 — one locked real case
# ============================================================================


def _case_runtime(fig9_root: Path) -> pd.DataFrame:
    """Correct stale lane metadata while preserving the raw runtime record."""
    raw = pd.read_csv(fig9_root / "fig9_runtime_log.csv")
    metadata = load_json(fig9_root / "fig9_checkpoint_metadata.json")
    rows: List[Dict[str, Any]] = []
    for row in raw.itertuples(index=False):
        if int(row.step) in {4, 6}:
            continue
        rows.append(
            {
                "stage": row.stage,
                "lane": row.lane,
                "input": row.input,
                "output": row.output,
                "elapsed_seconds": float(row.elapsed_seconds),
                "source_stage": row.stage,
                "normalization_note": "",
            }
        )
    rows.insert(
        3,
        {
            "stage": "v6.1 five-angle eligibility check",
            "lane": "agent",
            "input": "case year + frozen v6.1 cohort",
            "output": "outside 1980–2017 scoring cohort; no numeric fingerprint",
            "elapsed_seconds": 0.0,
            "source_stage": "new deterministic boundary check",
            "normalization_note": "prevents reuse of the obsolete seven-metric profile",
        },
    )
    rows.insert(
        4,
        {
            "stage": "legacy qualitative graph evidence",
            "lane": "agent",
            "input": "claims + references",
            "output": "agent_output.json (qualitative only)",
            "elapsed_seconds": 0.7,
            "source_stage": "compute perturbation profile",
            "normalization_note": "legacy B/RS/DeltaQ0 values are not current primary indicators",
        },
    )
    rows.insert(
        6,
        {
            "stage": "ASPR-Qwen checkpoint generation",
            "lane": "ASPR-Qwen",
            "input": "manuscript excerpt",
            "output": "checkpoint-generated response",
            "elapsed_seconds": float(metadata["runtime_seconds"]),
            "source_stage": "ASPR-Qwen review generation",
            "normalization_note": "replaces the stale placeholder runtime row",
        },
    )
    output = pd.DataFrame(rows)
    output["step"] = np.arange(1, len(output) + 1)
    output["cumulative_seconds"] = output["elapsed_seconds"].cumsum()
    return output


def _cohort_angle_reference(paths: SuitePaths) -> pd.DataFrame:
    """Compute current-schema cohort and high-diffusion angle medians."""
    resolved = _v61_paths(paths)
    features = pd.read_parquet(
        resolved["features"],
        columns=["paper_id", "publication_year", "domain12", *PRIMARY_FEATURES],
    )
    oof = pd.read_parquet(
        resolved["oof"],
        columns=["paper_id", "horizon", "model_id", "realized_diffusion_target"],
    )
    truth = oof.loc[
        oof["horizon"].eq(5)
        & oof["model_id"].eq("final_innovation_plus_k1"),
        ["paper_id", "realized_diffusion_target"],
    ].drop_duplicates("paper_id")
    frame = features.merge(truth, on="paper_id", how="inner")
    for feature in PRIMARY_FEATURES:
        frame[f"{feature}__percentile"] = grouped_percentile(
            frame,
            feature,
            ["domain12", "publication_year"],
            id_column="paper_id",
        )
    frame["high_diffusion"] = percentile_rank(
        frame["realized_diffusion_target"],
        frame["paper_id"],
    ).ge(0.90)
    rows = []
    for angle_id, members in ANGLE_FEATURES.items():
        score = frame[[f"{member}__percentile" for member in members]].mean(axis=1)
        rows.append(
            {
                "angle_id": angle_id,
                "angle_label": ANGLE_LABELS[angle_id],
                "case_value": np.nan,
                "case_status": "not_materialized_outside_1980_2017_cohort",
                "cohort_median": float(score.median()),
                "high_diffusion_median": float(score.loc[frame["high_diffusion"]].median()),
                "indicator_count": len(members),
            }
        )
    return pd.DataFrame(rows)


def _json_list(value: Any, limit: int = 4) -> str:
    """Serialize a short list-like value for a panel table."""
    if isinstance(value, list):
        return " | ".join(str(item) for item in value[:limit])
    return str(value)


def _case_output_cards(fig9_root: Path) -> pd.DataFrame:
    """Extract concise, exact agent and checkpoint output statements."""
    agent = load_json(fig9_root / "fig9_agent_output.json")
    qwen = load_json(fig9_root / "fig9_aspr_qwen_output.json")
    rows = []
    for item in agent.get("agent_evidence_summary", [])[:3]:
        rows.append(
            {
                "lane": "Graph-evidence agent",
                "item_type": "evidence assessment",
                "text": str(item.get("short_claim", "")),
                "status": str(item.get("assessment", "")),
                "evidence_ids": "|".join(item.get("evidence_ids", [])),
            }
        )
    rows.extend(
        [
            {
                "lane": "ASPR-Qwen",
                "item_type": "summary",
                "text": str(qwen.get("summary_judgement", "")),
                "status": "checkpoint_invoked",
                "evidence_ids": "",
            },
            {
                "lane": "ASPR-Qwen",
                "item_type": "strengths structure",
                "text": _json_list(qwen.get("major_strengths", [])),
                "status": "structured_list_missing_from_raw_checkpoint",
                "evidence_ids": "",
            },
            {
                "lane": "ASPR-Qwen",
                "item_type": "concerns structure",
                "text": _json_list(qwen.get("major_concerns", [])),
                "status": "structured_list_missing_from_raw_checkpoint",
                "evidence_ids": "",
            },
            {
                "lane": "ASPR-Qwen",
                "item_type": "recommendation",
                "text": str(qwen.get("reviewer_style_recommendation", "")),
                "status": "missing_separate_recommendation",
                "evidence_ids": "",
            },
        ]
    )
    return pd.DataFrame(rows)


def _case_overlap(fig9_root: Path) -> pd.DataFrame:
    """Build agent/Qwen/fusion/human coverage rows for one case."""
    fusion = load_json(fig9_root / "fig9_fusion_output.json")
    overlap = fusion["verifier"]["peer_review_overlap"]
    points = list(overlap.get("matched_points", []))
    rows = [
        {
            "concern": point,
            "agent": 1,
            "qwen": int(point in {"mutant assay rationale", "cryo-EM orientation/density caveat"}),
            "fusion": 1,
            "human_peer_review": 1,
            "status": "matched",
        }
        for point in points
    ]
    rows.append(
        {
            "concern": overlap.get("missing_or_weak_point", ""),
            "agent": 0,
            "qwen": 0,
            "fusion": 0,
            "human_peer_review": 1,
            "status": "human_only",
        }
    )
    for item in fusion["verifier"].get("unsupported_claims_removed", []):
        rows.append(
            {
                "concern": item,
                "agent": 1,
                "qwen": 0,
                "fusion": 1,
                "human_peer_review": 0,
                "status": "aspr_only_safeguard",
            }
        )
    return pd.DataFrame(rows)


def build_fig9(
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    """Build an auditable single-case storyboard with a measurement boundary."""
    root = paths["fig9_root"]
    case = pd.read_csv(root / "fig9_case_manifest.csv")
    runtime = _case_runtime(root)
    angles = _cohort_angle_reference(paths)
    cards = _case_output_cards(root)
    trace = pd.read_csv(root / "fig9_claim_evidence_trace.csv")
    overlap = _case_overlap(root)
    qwen = load_json(root / "fig9_aspr_qwen_output.json")
    tables = {
        "case_manifest": case,
        "execution_runtime": runtime,
        "five_angle_reference_profile": angles,
        "agent_qwen_cards": cards,
        "claim_evidence_trace": trace,
        "human_overlap": overlap,
    }
    panel_text = {
        "a": {
            "title": str(case.iloc[0]["title"]),
            "doi": str(case.iloc[0]["doi"]),
            "year": int(case.iloc[0]["year"]),
            "measurement_boundary": "2023 case is outside the frozen 1980–2017 v6.1 scoring cohort.",
        },
        "b": {
            "total_runtime_seconds": float(runtime["elapsed_seconds"].sum()),
            "checkpoint_runtime_seconds": float(qwen.get("runtime", np.nan)),
        },
        "c": "Current five-angle comparators are shown; the case marker is deliberately absent because the 2023 case was not materialized.",
        "d": "Agent statements are evidence-linked; checkpoint output is shown with its missing structured fields.",
        "e": "Every final-review claim links to manuscript or peer-review evidence and a verifier status.",
        "f": {
            "matched_human_points": 5,
            "total_key_human_points": 6,
            "single_case_only": True,
        },
        "warning": config["claim_boundaries"]["fig9"],
    }
    contract = {
        "figure_id": 9,
        "background_asset": str(paths["fig9_image_base"]),
        "case_locked_before_rendering": True,
        "current_case_fingerprint_available": False,
        "legacy_seven_metrics_used_as_current_fingerprint": False,
        "population_performance_claim": False,
        "panels": {
            "a": {"mark": "input card", "data": ["case_manifest"]},
            "b": {"mark": "two-lane execution swimlane", "data": ["execution_runtime"]},
            "c": {"mark": "five-angle polar comparator with unavailable case", "data": ["five_angle_reference_profile"]},
            "d": {"mark": "agent and checkpoint cards", "data": ["agent_qwen_cards"]},
            "e": {"mark": "claim-evidence-verifier graph", "data": ["claim_evidence_trace"]},
            "f": {"mark": "intersection dot plot", "data": ["human_overlap"]},
        },
    }
    return FigureBundle(
        figure_id=9,
        title="A locked real case exposes both evidence flow and score boundaries",
        status="complete_single_case_with_measurement_boundary",
        tables=tables,
        panel_text=panel_text,
        chart_contract=contract,
        source_paths=[
            root / "fig9_case_manifest.csv",
            root / "fig9_runtime_log.csv",
            root / "fig9_checkpoint_metadata.json",
            root / "fig9_agent_output.json",
            root / "fig9_aspr_qwen_output.json",
            root / "fig9_fusion_output.json",
            root / "fig9_claim_evidence_trace.csv",
            paths["fig9_image_base"],
        ],
        notes=[config["claim_boundaries"]["fig9"]],
    )


# ============================================================================
# Fig.10 — module ablation, preference gate and projected reinforcements
# ============================================================================


QUALITY_METRICS: Dict[str, str] = {
    "semantic_agreement": "Semantic agreement",
    "novelty_coverage": "Novelty coverage",
    "prior_art_accuracy": "Prior-art accuracy",
    "factuality": "Factuality",
    "unsupported_claim_rate": "Supported-claim quality",
    "evidence_trace_completeness": "Trace completeness",
}


def _module_inventory(fig10_root: Path) -> pd.DataFrame:
    """Crosswalk the historical inventory to the current five-angle contract."""
    inventory = pd.read_csv(fig10_root / "fig10_module_inventory.csv")
    inventory["raw_module"] = inventory["module"]
    inventory["raw_role"] = inventory["role"]
    old = inventory["module"].eq("seven-indicator computation")
    inventory.loc[old, "module"] = "five-angle / eight-indicator evidence"
    inventory.loc[old, "role"] = (
        "current v6.1 publication-time indicator contract; historical rerun used legacy graph prior"
    )
    inventory["crosswalk_status"] = np.where(
        old,
        "terminology_updated_with_historical_scope_note",
        "unchanged",
    )
    return inventory


def _ablation_deltas(
    reruns: pd.DataFrame,
    iterations: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate paired case deltas while retaining protocol comparability flags."""
    quality = reruns.copy()
    quality["unsupported_claim_rate"] = 1.0 - numeric(
        quality["unsupported_claim_rate"]
    )
    long = quality.melt(
        id_vars=["case_id", "variant", "source", "runtime_seconds"],
        value_vars=list(QUALITY_METRICS),
        var_name="metric",
        value_name="quality_value",
    )
    full = long.loc[long["variant"].eq("full ASPR")].rename(
        columns={
            "quality_value": "full_quality",
            "source": "full_source",
            "runtime_seconds": "full_runtime_seconds",
        }
    )
    ablated = long.loc[~long["variant"].eq("full ASPR")].merge(
        full[
            [
                "case_id",
                "metric",
                "full_quality",
                "full_source",
                "full_runtime_seconds",
            ]
        ],
        on=["case_id", "metric"],
        how="inner",
    )
    ablated["delta_ablation_minus_full"] = (
        numeric(ablated["quality_value"]) - numeric(ablated["full_quality"])
    )
    ablated["comparison_validity"] = np.where(
        ablated["source"].eq(ablated["full_source"]),
        "same_generation_path",
        "generation_path_mismatch",
    )
    rows = []
    for (variant, metric), group in ablated.groupby(["variant", "metric"]):
        mean, low, high = _bootstrap_delta(
            group["delta_ablation_minus_full"].to_numpy(float),
            iterations=int(iterations),
            seed=stable_seed(f"{variant}:{metric}", seed),
        )
        rows.append(
            {
                "variant": variant,
                "metric": metric,
                "metric_label": QUALITY_METRICS[metric],
                "n_pairs": len(group),
                "mean_delta_ablation_minus_full": mean,
                "ci_low": low,
                "ci_high": high,
                "comparison_validity": (
                    "generation_path_mismatch"
                    if group["comparison_validity"].eq("generation_path_mismatch").any()
                    else "same_generation_path"
                ),
            }
        )
    return ablated, pd.DataFrame(rows)


def _error_links(reruns: pd.DataFrame) -> pd.DataFrame:
    """Derive transparent threshold-based error counts from all 400 reruns."""
    rules = [
        ("missed prior art", "prior_art_accuracy", "<", 0.25),
        ("generic novelty claim", "novelty_coverage", "<", 0.25),
        ("unsupported claim", "unsupported_claim_rate", ">", 0.42),
        ("evidence not traceable", "evidence_trace_completeness", "<", 0.25),
        ("mechanism mismatch", "semantic_agreement", "<", 0.70),
        ("weak reviewer structure", "review_structure_coverage", "<", 0.35),
        ("factuality risk", "factuality", "<", 0.75),
    ]
    rows: List[Dict[str, Any]] = []
    for variant, group in reruns.groupby("variant"):
        if variant == "full ASPR":
            continue
        for error_type, metric, operator, threshold in rules:
            values = numeric(group[metric])
            indicator = values.lt(threshold) if operator == "<" else values.gt(threshold)
            rows.append(
                {
                    "variant": variant,
                    "error_type": error_type,
                    "case_count": int(indicator.notna().sum()),
                    "error_count": int(indicator.sum()),
                    "error_rate": float(indicator.mean()),
                    "trigger_metric": metric,
                    "operator": operator,
                    "threshold": threshold,
                    "source": "derived_from_true_module_rerun_metrics",
                }
            )
    return pd.DataFrame(rows)


def _preference_gate(fig10_root: Path) -> Tuple[pd.DataFrame, int, int]:
    """Read the blinded human-preference completion gate."""
    audit = pd.read_csv(
        fig10_root / "fig10_blinded_preference_completion_audit.csv"
    )
    completed = int(numeric(audit["observed_valid_judgements"]).fillna(0).sum())
    required = int(numeric(audit["required_judgements"]).fillna(0).sum())
    return audit, completed, required


def _pareto_projection(fig10_root: Path) -> pd.DataFrame:
    """Flag non-dominated reinforcement projections without calling them results."""
    data = pd.read_csv(fig10_root / "fig10_reinforcement_results.csv")
    pareto = []
    for row in data.itertuples(index=False):
        dominated = (
            data["relative_runtime_cost"].le(float(row.relative_runtime_cost))
            & data["quality_gain"].ge(float(row.quality_gain))
            & (
                data["relative_runtime_cost"].lt(float(row.relative_runtime_cost))
                | data["quality_gain"].gt(float(row.quality_gain))
            )
        ).any()
        pareto.append(int(not dominated))
    data["pareto_frontier"] = pareto
    data["evidence_status"] = np.where(
        data["source"].eq("pipeline_ready_reinforcement_projection"),
        "projected_not_experimental",
        "observed",
    )
    return data


def _first_sentence(path: Path, limit: int = 220) -> str:
    """Extract one bounded sentence from a local rerun review."""
    if not path.is_file():
        return "Review text unavailable."
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
    ]
    substantive = [
        re.sub(r"^[\-*]\s*", "", line)
        for line in lines
        if line
        and not line.startswith("#")
        and not set(line).issubset({"-", "_", "="})
        and len(re.sub(r"\W", "", line, flags=re.UNICODE)) >= 12
    ]
    text = re.sub(r"\s+", " ", " ".join(substantive)).strip()
    if not text:
        return "No substantive review sentence available."
    match = re.split(r"(?<=[.!?])\s+", text)
    sentence = match[0] if match else text
    return sentence[:limit].rstrip() + ("…" if len(sentence) > limit else "")


def _degradation_cases(
    reruns: pd.DataFrame,
    fig10_root: Path,
) -> pd.DataFrame:
    """Select two locked variants by observed composite delta, without forcing degradation."""
    quality = reruns.copy()
    quality["supported_claim_quality"] = 1.0 - numeric(quality["unsupported_claim_rate"])
    metric_columns = [
        "semantic_agreement",
        "novelty_coverage",
        "prior_art_accuracy",
        "factuality",
        "supported_claim_quality",
        "evidence_trace_completeness",
    ]
    quality["composite_quality"] = quality[metric_columns].mean(axis=1)
    full = quality.loc[
        quality["variant"].eq("full ASPR"),
        ["case_id", "composite_quality", "review_text_path"],
    ].rename(
        columns={
            "composite_quality": "full_composite",
            "review_text_path": "full_review_path",
        }
    )
    rows = []
    for variant in ["no graph agent", "no verifier"]:
        candidate = quality.loc[quality["variant"].eq(variant)].merge(
            full,
            on="case_id",
            how="inner",
        )
        candidate["delta_ablation_minus_full"] = (
            candidate["composite_quality"] - candidate["full_composite"]
        )
        selected = candidate.sort_values(
            ["delta_ablation_minus_full", "case_id"],
            ascending=[True, True],
        ).iloc[0]
        rows.append(
            {
                "case_id": selected["case_id"],
                "variant": variant,
                "delta_ablation_minus_full": float(selected["delta_ablation_minus_full"]),
                "observed_direction": (
                    "ablation_lower"
                    if selected["delta_ablation_minus_full"] < 0
                    else "ablation_not_lower"
                ),
                "full_excerpt": _first_sentence(
                    fig10_root / str(selected["full_review_path"])
                ),
                "ablation_excerpt": _first_sentence(
                    fig10_root / str(selected["review_text_path"])
                ),
                "comparison_validity": "generation_path_mismatch",
            }
        )
    return pd.DataFrame(rows)


def build_fig10(
    config: Mapping[str, Any],
    paths: SuitePaths,
) -> FigureBundle:
    """Build Fig.10 with explicit automatic and human-evidence gates."""
    root = paths["fig10_root"]
    inventory = _module_inventory(root)
    reruns = pd.read_csv(root / "fig10_true_module_rerun_results.csv")
    paired_cases, deltas = _ablation_deltas(
        reruns,
        int(config["fig10"]["bootstrap_iterations"]),
        int(config["fig10"]["seed"]),
    )
    error_links = _error_links(reruns)
    preference_audit, completed, required = _preference_gate(root)
    projected = _pareto_projection(root)
    cases = _degradation_cases(reruns, root)
    tables = {
        "module_inventory": inventory,
        "true_module_reruns": reruns,
        "paired_case_deltas": paired_cases,
        "ablation_delta_estimates": deltas,
        "module_error_links": error_links,
        "preference_completion_audit": preference_audit,
        "reinforcement_projections": projected,
        "representative_cases": cases,
    }
    mismatch_fraction = float(
        paired_cases["comparison_validity"].eq("generation_path_mismatch").mean()
    )
    status = (
        "complete_ablation_and_human_preference"
        if completed >= required and mismatch_fraction == 0
        else "draft_comparability_and_human_preference_blocked"
    )
    panel_text = {
        "a": "Switch names are crosswalked to the current five-angle/eight-indicator terminology.",
        "b": {
            "case_variant_rows": int(len(reruns)),
            "paired_cases_per_variant": int(reruns["case_id"].nunique()),
            "generation_path_mismatch_fraction": mismatch_fraction,
            "warning": "Full ASPR and disabled variants were scored from different generation paths; deltas are descriptive and cannot establish module degradation.",
        },
        "c": "Error links are threshold-derived from the 400-row rerun metrics, not human adjudication.",
        "d": {
            "completed_judgements": completed,
            "required_judgements": required,
            "blocked": completed < required,
        },
        "e": {
            "evidence_status": "projected_not_experimental",
            "warning": "Hollow points are pipeline-ready projections, not measured improvements.",
        },
        "f": "Case cards preserve the observed direction; the renderer does not force an ablation-degradation story.",
        "warning": config["claim_boundaries"]["fig10"],
    }
    contract = {
        "figure_id": 10,
        "background_asset": str(paths["fig10_image_base"]),
        "automatic_rerun_rows": int(len(reruns)),
        "automatic_comparison_valid": mismatch_fraction == 0,
        "human_preference_gate": {
            "required": required,
            "completed": completed,
            "passed": completed >= required,
        },
        "reinforcement_status": "projection_only",
        "panels": {
            "a": {"mark": "module switchboard", "data": ["module_inventory"]},
            "b": {"mark": "paired delta intervals", "data": ["ablation_delta_estimates"]},
            "c": {"mark": "module-to-error chord", "data": ["module_error_links"]},
            "d": {"mark": "ternary preference", "blocked": completed < required},
            "e": {"mark": "projected quality-cost Pareto", "data": ["reinforcement_projections"]},
            "f": {"mark": "observed case cards", "data": ["representative_cases"]},
        },
    }
    return FigureBundle(
        figure_id=10,
        title="Module evidence is informative but not yet a valid causal ablation",
        status=status,
        tables=tables,
        panel_text=panel_text,
        chart_contract=contract,
        source_paths=[
            root / "fig10_true_module_rerun_results.csv",
            root / "fig10_true_rerun_completion_audit.csv",
            root / "fig10_module_inventory.csv",
            root / "fig10_blinded_preference_completion_audit.csv",
            root / "fig10_reinforcement_results.csv",
            paths["fig10_image_base"],
        ],
        notes=[config["claim_boundaries"]["fig10"]],
    )
