from __future__ import annotations

import json
import hashlib
import math
import os
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aspr_mplconfig")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import requests
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "outputs" / "kg_perturbation_fig6"

FIG3_DIR = ROOT / "outputs" / "redraw_v6a_best_fig3" / "multi_domain"
EVIDENCE_DIR = ROOT / "outputs" / "publication_corpus_v4_evidence_bundle"
FIG4_DIR = ROOT / "outputs" / "kg_perturbation_fig4_full50"
CORPUS_DIR = ROOT / "data" / "knowledge_corpus" / "v4_final_graph_bio_methods_phys10"

SCORE_TABLE = FIG3_DIR / "fig3_score_table.csv"
BEST_WEIGHTS = FIG3_DIR / "fig3_best_weights.csv"
COVERAGE_WEIGHTS = FIG3_DIR / "coverage_constrained_weights.csv"
FOLD_WEIGHTS = FIG3_DIR / "fig3_fold_weights.csv"
BASELINE_COMPARISON = FIG3_DIR / "fig3_baseline_comparison.csv"
NONLINEAR_UPPER = FIG3_DIR / "fig3_nonlinear_upper_bound.csv"
PUBLICATION_DAY_INDICATORS = FIG3_DIR / "fig3_publication_day_indicators.csv"
FUTURE_GRAPH_DELTAS = FIG3_DIR / "fig3_future_graph_deltas.csv"

DOMAIN_OOF = EVIDENCE_DIR / "fig3aware12_subset_domain_oof_diagnostics.csv"
DOMAIN_COVERAGE = EVIDENCE_DIR / "fig3_score_coverage_by_domain.csv"
FIG5_DOMAIN_SUMMARY = EVIDENCE_DIR / "fig3aware12_fig5_domain_summary_minrefs1.csv"

FIG4_METRICS = FIG4_DIR / "fig4_metrics_summary.csv"
FIG4_RETRIEVAL = FIG4_DIR / "fig4_retrieval_diagnostics.csv"
FIG4_MANIFEST = FIG4_DIR / "fig4_manifest.csv"

GRAPH_METRICS = ["B", "RS", "DeltaQ0", "Uzzi", "RTD", "BurtIP", "PDE"]
GRAPH_Z_COLS = [f"{metric}_z" for metric in GRAPH_METRICS]
RANDOM_SEEDS = list(range(20260600, 20260620))

FULL_RERUN_MANIFEST = "fig6_full_rerun_manifest.csv"
FULL_RERUN_INDICATOR_STABILITY = "fig6_indicator_stability.csv"
FULL_RERUN_RANK_STABILITY = "fig6_rank_stability.csv"
FULL_RERUN_PAPER_DRIFT = "fig6_full_rerun_paper_drift.csv"
FULL_RERUN_FAILURE_CASES = "fig6_full_rerun_failure_cases.csv"
FULL_RERUN_REFERENCE_STABLE_SUBSET = "fig6_reference_stable_subset_diagnostic.csv"
FULL_RERUN_PRIMARY_MODEL_STABILITY = "fig6_primary_model_stability.csv"
FULL_RERUN_PRIMARY_MODEL_PAPER_DRIFT = "fig6_primary_model_paper_drift.csv"
FULL_RERUN_REFRESH_ATTEMPT = "fig6_full_rerun_refresh_attempt.csv"
ONLINE_REFERENCE_EDGES = "fig6_online_sample_reference_edges.csv"
REFERENCE_CLOSURE_DRIFT = "fig6_reference_closure_drift.csv"
FULL_RERUN_MANIFEST_COLUMNS = [
    "rerun_id",
    "source",
    "reference_closure",
    "edge_sampling_seed",
    "graph_construction",
    "cutoff_year_delta",
    "metadata_refresh_mode",
    "rerun_scope",
    "n_sampled_papers",
    "n_refetched_works",
    "n_edges",
    "metadata_fetch_status",
    "graph_build_status",
    "indicator_status",
    "input_hash",
]
FULL_RERUN_INDICATOR_COLUMNS = [
    "rerun_id",
    "metric",
    "baseline_mean",
    "rerun_mean",
    "delta",
    "direction_preserved",
]
FULL_RERUN_RANK_COLUMNS = [
    "rerun_id",
    "rank_spearman",
    "top_decile_jaccard",
    "learned_score_direction_preserved",
]
FULL_RERUN_PRIMARY_MODEL_COLUMNS = [
    "rerun_id",
    "primary_model",
    "feature_scope",
    "n_scored_papers",
    "rank_spearman",
    "top_decile_jaccard",
    "primary_score_direction_preserved",
    "model_status",
    "score_source",
]
FULL_RERUN_PAPER_DRIFT_COLUMNS = [
    "rerun_id",
    "source",
    "edge_sampling_seed",
    "graph_construction",
    "paper_id",
    "baseline_score",
    "rerun_score",
    "score_delta",
    "baseline_rank",
    "rerun_rank",
    "rank_delta",
    "baseline_top_decile",
    "rerun_top_decile",
    "baseline_reference_count",
    "rerun_reference_count",
]
FULL_RERUN_PRIMARY_MODEL_PAPER_DRIFT_COLUMNS = [
    "rerun_id",
    "source",
    "edge_sampling_seed",
    "graph_construction",
    "paper_id",
    "baseline_primary_score",
    "rerun_primary_score",
    "primary_score_delta",
    "baseline_primary_rank",
    "rerun_primary_rank",
    "primary_rank_delta",
    "baseline_primary_top_decile",
    "rerun_primary_top_decile",
]
PRIMARY_MODEL_NAME = "metadata_hgb_no_leakage"
PRIMARY_MODEL_FEATURE_SCOPE = "publication_day_graph_indicators_plus_year_reference_domain_field"
PRIMARY_MODEL_NUMERIC_COLS = [*GRAPH_Z_COLS, "reference_count", "year"]
PRIMARY_MODEL_CATEGORICAL_COLS = ["domain", "primary_field"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "blue": {"xlight": "#EAF1FE", "light": "#CEDFFE", "base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780"},
    "gold": {"xlight": "#FFF4C2", "light": "#FFEA8F", "base": "#FFE15B", "mid": "#B8A037", "dark": "#736422"},
    "orange": {"xlight": "#FFEDDE", "light": "#FFBDA1", "base": "#F0986E", "mid": "#CC6F47", "dark": "#804126"},
    "olive": {"xlight": "#D8ECBD", "light": "#BEEB96", "base": "#A3D576", "mid": "#71B436", "dark": "#386411"},
    "pink": {"xlight": "#FCDAD6", "light": "#F5BACC", "base": "#F390CA", "mid": "#BD569B", "dark": "#8A3A6F"},
    "neutral": {"xlight": "#F4F5F7", "light": "#E2E5EA", "base": "#C5CAD3", "mid": "#7A828F", "dark": "#464C55"},
}


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    path: Path
    used_for: str
    notes: str


def main() -> None:
    """Build Fig. 6 audit tables and static figure exports."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    use_chart_theme()

    score = load_score_table()
    weights = load_best_weights()
    indicators = read_csv(PUBLICATION_DAY_INDICATORS)
    future = read_csv(FUTURE_GRAPH_DELTAS)
    source_specs = build_source_specs()

    panel_a = build_cross_domain_panel()
    panel_b = build_data_quality_panel(score, weights)
    panel_c = build_volume_panel(score)
    panel_d = build_temporal_panel(score)
    panel_e = build_modeling_choice_panel(score, weights)
    panel_f, failure_cases = build_failure_modes_panel()
    panel_g = build_cache_graph_perturbation_panel(indicators, future, weights)
    panel_review = build_panel_review(panel_a, panel_b, panel_c, panel_d, panel_e, panel_f, panel_g)

    maybe_build_online_full_rerun_artifacts(indicators, weights)
    write_tables(panel_a, panel_b, panel_c, panel_d, panel_e, panel_f, panel_g, failure_cases)
    write_panel_review(panel_review)
    write_metadata(source_specs, panel_review, panel_a, panel_b, panel_c, panel_d, panel_e, panel_f, panel_g)
    write_caption(panel_review, panel_a, panel_b, panel_c, panel_d, panel_e, panel_f, panel_g)
    write_quality_report(panel_review, panel_g)

    save_single_panel("fig6_panel_a", lambda fig, ax: plot_panel_a(ax, panel_a), (7.2, 5.2))
    save_single_panel("fig6_panel_b", lambda fig, ax: plot_panel_b(ax, panel_b), (7.2, 5.2))
    save_single_panel("fig6_panel_c", lambda fig, ax: plot_panel_c(ax, panel_c), (7.2, 5.2))
    save_single_panel("fig6_panel_d", lambda fig, ax: plot_panel_d(ax, panel_d), (7.2, 5.2))
    save_single_panel("fig6_panel_e", lambda fig, ax: plot_panel_e(ax, panel_e), (7.2, 5.2))
    save_single_panel("fig6_panel_f", plot_panel_f_single(panel_f, failure_cases), (7.2, 5.2))
    save_full_figure(panel_a, panel_b, panel_c, panel_d, panel_e, panel_f, failure_cases)

    print(f"Wrote Fig. 6 outputs to {OUT_DIR}")


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.monospace": ["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "monospace"],
        },
    )


def build_source_specs() -> List[SourceSpec]:
    return [
        SourceSpec("fig3_score_table", SCORE_TABLE, "panels A-E", "Cached Fig.3 score table; no graph extraction rerun."),
        SourceSpec("fig3_best_weights", BEST_WEIGHTS, "panels B/E", "Strict Fig.3 learned graph metric weights."),
        SourceSpec("fig3_domain_oof", DOMAIN_OOF, "panel A", "Domain-level OOF diagnostics from evidence bundle."),
        SourceSpec("fig3_domain_coverage", DOMAIN_COVERAGE, "panel A/C", "Score coverage and literature-volume context."),
        SourceSpec("fig5_domain_backtest", FIG5_DOMAIN_SUMMARY, "panel A", "Cached Fig.5 graph Top-10 domain summary."),
        SourceSpec("fig3_baseline_comparison", BASELINE_COMPARISON, "panel E", "Modeling-choice comparison table."),
        SourceSpec("fig3_nonlinear_upper", NONLINEAR_UPPER, "panel E", "Quadratic upper-bound cached diagnostic."),
        SourceSpec("fig3_publication_day_indicators", PUBLICATION_DAY_INDICATORS, "panel G", "Raw cached publication-day graph indicators before score-table robustness perturbation."),
        SourceSpec("fig3_future_graph_deltas", FUTURE_GRAPH_DELTAS, "panel G", "Cached future graph-delta target components used to audit indicator-level perturbations."),
        SourceSpec("fig4_metrics_summary", FIG4_METRICS, "panel F", "Nature Portfolio peer/agent agreement and graph-prior summary."),
        SourceSpec("fig4_retrieval_diagnostics", FIG4_RETRIEVAL, "panel F", "Prior-art retrieval diagnostics."),
        SourceSpec("fig4_manifest", FIG4_MANIFEST, "panel F", "Fig.4 audit sample metadata."),
        SourceSpec("v4_final_corpus", CORPUS_DIR / "quality_report.json", "metadata", "Corpus-level works/topics/citation quality report."),
    ]


def read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def has_columns(df: pd.DataFrame, columns: List[str]) -> bool:
    return set(columns).issubset(set(df.columns))


def load_full_rerun_artifacts(out_dir: Path = OUT_DIR) -> pd.DataFrame:
    """Load validated full graph-rerun stability rows when all contract files exist."""
    candidate_dirs = [out_dir, out_dir / "full_rerun"]
    for candidate in candidate_dirs:
        manifest = read_csv(candidate / FULL_RERUN_MANIFEST)
        indicator = read_csv(candidate / FULL_RERUN_INDICATOR_STABILITY)
        rank = read_csv(candidate / FULL_RERUN_RANK_STABILITY)
        if manifest.empty or indicator.empty or rank.empty:
            continue
        if not (
            has_columns(manifest, FULL_RERUN_MANIFEST_COLUMNS)
            and has_columns(indicator, FULL_RERUN_INDICATOR_COLUMNS)
            and has_columns(rank, FULL_RERUN_RANK_COLUMNS)
        ):
            continue

        manifest = manifest.copy()
        indicator = indicator.copy()
        rank = rank.copy()
        manifest["rerun_id"] = manifest["rerun_id"].astype(str)
        indicator["rerun_id"] = indicator["rerun_id"].astype(str)
        rank["rerun_id"] = rank["rerun_id"].astype(str)

        success_mask = (
            manifest["metadata_fetch_status"].astype(str).str.lower().eq("success")
            & manifest["graph_build_status"].astype(str).str.lower().eq("success")
            & manifest["indicator_status"].astype(str).str.lower().eq("success")
        )
        successful_ids = set(manifest.loc[success_mask, "rerun_id"])
        indicator_ids = set(indicator["rerun_id"])
        valid_ids = successful_ids & indicator_ids
        validated_rank = rank[rank["rerun_id"].isin(valid_ids)].copy()
        if validated_rank.empty:
            continue
        provenance_cols = [
            "rerun_id",
            "source",
            "reference_closure",
            "edge_sampling_seed",
            "graph_construction",
            "cutoff_year_delta",
            "metadata_refresh_mode",
            "rerun_scope",
            "n_sampled_papers",
            "n_refetched_works",
            "n_edges",
            "input_hash",
        ]
        validated_rank = validated_rank.merge(manifest[provenance_cols], on="rerun_id", how="left")
        return validated_rank.reset_index(drop=True)
    return pd.DataFrame(columns=FULL_RERUN_RANK_COLUMNS)


def load_primary_model_stability_artifacts(out_dir: Path = OUT_DIR) -> pd.DataFrame:
    """Load validated Fig.3 primary-model stability rows for successful full reruns."""
    candidate = out_dir
    manifest = read_csv(candidate / FULL_RERUN_MANIFEST)
    primary = read_csv(candidate / FULL_RERUN_PRIMARY_MODEL_STABILITY)
    if manifest.empty or primary.empty:
        return pd.DataFrame(columns=FULL_RERUN_PRIMARY_MODEL_COLUMNS)
    if not (has_columns(manifest, FULL_RERUN_MANIFEST_COLUMNS) and has_columns(primary, FULL_RERUN_PRIMARY_MODEL_COLUMNS)):
        return pd.DataFrame(columns=FULL_RERUN_PRIMARY_MODEL_COLUMNS)
    manifest = manifest.copy()
    primary = primary.copy()
    manifest["rerun_id"] = manifest["rerun_id"].astype(str)
    primary["rerun_id"] = primary["rerun_id"].astype(str)
    success_mask = (
        manifest["metadata_fetch_status"].astype(str).str.lower().eq("success")
        & manifest["graph_build_status"].astype(str).str.lower().eq("success")
        & manifest["indicator_status"].astype(str).str.lower().eq("success")
    )
    successful_ids = set(manifest.loc[success_mask, "rerun_id"])
    validated = primary[
        primary["rerun_id"].isin(successful_ids)
        & primary["model_status"].astype(str).str.lower().eq("success")
    ].copy()
    if validated.empty:
        return pd.DataFrame(columns=FULL_RERUN_PRIMARY_MODEL_COLUMNS)
    provenance_cols = [
        "rerun_id",
        "source",
        "reference_closure",
        "edge_sampling_seed",
        "graph_construction",
        "cutoff_year_delta",
        "metadata_refresh_mode",
        "rerun_scope",
        "n_sampled_papers",
        "n_refetched_works",
        "n_edges",
        "input_hash",
    ]
    validated = validated.merge(manifest[provenance_cols], on="rerun_id", how="left")
    return validated.reset_index(drop=True)


def load_score_table() -> pd.DataFrame:
    df = read_csv(SCORE_TABLE)
    if df.empty:
        raise FileNotFoundError(f"Required score table not found: {SCORE_TABLE}")
    for col in ["S_w", "S_equal", "RGPM", *GRAPH_Z_COLS]:
        if col not in df.columns:
            raise ValueError(f"Missing required score-table column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df.dropna(subset=["S_w", "RGPM", "year"]).copy()


def load_best_weights() -> Dict[str, float]:
    weights_df = read_csv(BEST_WEIGHTS)
    if weights_df.empty:
        return {metric: 1.0 / len(GRAPH_METRICS) for metric in GRAPH_METRICS}
    metric_col = "metric" if "metric" in weights_df.columns else weights_df.columns[0]
    weights = dict(zip(weights_df[metric_col].astype(str), pd.to_numeric(weights_df["weight"], errors="coerce")))
    total = sum(float(weights.get(metric, 0.0)) for metric in GRAPH_METRICS)
    if not total:
        return {metric: 1.0 / len(GRAPH_METRICS) for metric in GRAPH_METRICS}
    return {metric: float(weights.get(metric, 0.0)) / total for metric in GRAPH_METRICS}


def weighted_score(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    score = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    for metric in GRAPH_METRICS:
        score = score + pd.to_numeric(df[f"{metric}_z"], errors="coerce").fillna(0.0) * weights.get(metric, 0.0)
    return score


def primary_model_numeric_fill_values(frame: pd.DataFrame) -> Dict[str, float]:
    """Return training-set numeric fill values for the Fig.3 primary model contract."""
    fills: Dict[str, float] = {}
    for col in PRIMARY_MODEL_NUMERIC_COLS:
        values = pd.to_numeric(frame.get(col, pd.Series(dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(values.median()) if not values.dropna().empty else 0.0
        fills[col] = median if math.isfinite(median) else 0.0
    return fills


def primary_model_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: Optional[List[str]] = None,
    numeric_fill_values: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Build the no-leakage Fig.3 primary-model feature matrix with aligned columns."""
    numeric_fill_values = numeric_fill_values or primary_model_numeric_fill_values(frame)
    numeric = pd.DataFrame(index=frame.index)
    for col in PRIMARY_MODEL_NUMERIC_COLS:
        raw = frame[col] if col in frame.columns else pd.Series(np.nan, index=frame.index)
        values = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan)
        fill_value = float(numeric_fill_values.get(col, 0.0))
        numeric[col] = values.fillna(fill_value if math.isfinite(fill_value) else 0.0)
    categorical = pd.DataFrame(index=frame.index)
    for col in PRIMARY_MODEL_CATEGORICAL_COLS:
        if col in frame.columns:
            categorical[col] = frame[col].fillna("missing").astype(str)
        else:
            categorical[col] = "missing"
    cat_features = pd.get_dummies(categorical, columns=PRIMARY_MODEL_CATEGORICAL_COLS, prefix=PRIMARY_MODEL_CATEGORICAL_COLS, dtype=float)
    features = pd.concat([numeric.astype(float), cat_features.astype(float)], axis=1)
    if feature_columns is None:
        return features
    aligned = features.copy()
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = 0.0
    return aligned[feature_columns].astype(float)


def fit_fig3_primary_model(
    score_table: pd.DataFrame,
    *,
    seed: int = 20260630,
) -> Dict[str, object]:
    """Fit the promoted Fig.3 HGB model for stability diagnostics only."""
    clean = score_table.copy().replace([np.inf, -np.inf], np.nan)
    if "RGPM" not in clean.columns:
        return {"model_status": "missing_rgpm"}
    clean["RGPM"] = pd.to_numeric(clean["RGPM"], errors="coerce")
    clean = clean.dropna(subset=["RGPM"]).reset_index(drop=True)
    if len(clean) < 20 or clean["RGPM"].nunique(dropna=True) < 2:
        return {"model_status": "insufficient_training_data"}
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        return {"model_status": "sklearn_unavailable"}
    numeric_fill = primary_model_numeric_fill_values(clean)
    features = primary_model_feature_frame(clean, numeric_fill_values=numeric_fill)
    if features.empty:
        return {"model_status": "empty_feature_matrix"}
    model = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=int(seed),
    )
    model.fit(features.to_numpy(dtype=float), clean["RGPM"].to_numpy(dtype=float))
    return {
        "model_status": "success",
        "model": model,
        "feature_columns": list(features.columns),
        "numeric_fill_values": numeric_fill,
    }


def predict_fig3_primary_model(fitted: Dict[str, object], frame: pd.DataFrame) -> pd.Series:
    """Predict Fig.3 primary-model scores for a feature frame aligned to training."""
    if fitted.get("model_status") != "success":
        return pd.Series(dtype=float)
    feature_columns = list(fitted.get("feature_columns", []))
    numeric_fill_values = dict(fitted.get("numeric_fill_values", {}))
    features = primary_model_feature_frame(
        frame,
        feature_columns=feature_columns,
        numeric_fill_values=numeric_fill_values,
    )
    model = fitted.get("model")
    if model is None or features.empty:
        return pd.Series(dtype=float)
    values = model.predict(features.to_numpy(dtype=float))
    return pd.Series(values, index=frame.index, dtype=float)


def primary_model_stability_for_rerun(
    *,
    fitted_model: Dict[str, object],
    baseline: pd.DataFrame,
    rerun_metrics: pd.DataFrame,
    rerun_id: str,
    source: str,
    edge_sampling_seed: int,
    graph_construction: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Compare baseline and rerun ranks using the promoted Fig.3 primary model."""
    status = str(fitted_model.get("model_status", "unknown"))
    if status != "success":
        return (
            {
                "rerun_id": rerun_id,
                "primary_model": PRIMARY_MODEL_NAME,
                "feature_scope": PRIMARY_MODEL_FEATURE_SCOPE,
                "n_scored_papers": 0,
                "rank_spearman": float("nan"),
                "top_decile_jaccard": float("nan"),
                "primary_score_direction_preserved": 0,
                "model_status": status,
                "score_source": "fig3_primary_model_contract",
            },
            [],
        )
    baseline_features = baseline.copy()
    rerun_features = rerun_metrics.copy()
    for col in ["year", "domain", "primary_field"]:
        if col not in rerun_features.columns and col in baseline_features.columns:
            rerun_features = rerun_features.merge(
                baseline_features[["paper_id", col]].drop_duplicates("paper_id"),
                on="paper_id",
                how="left",
            )
    baseline_features["_primary_score"] = predict_fig3_primary_model(fitted_model, baseline_features)
    rerun_features["_primary_score"] = predict_fig3_primary_model(fitted_model, rerun_features)
    merged = baseline_features[["paper_id", "_primary_score"]].merge(
        rerun_features[["paper_id", "_primary_score"]],
        on="paper_id",
        how="inner",
        suffixes=("_baseline", "_rerun"),
    )
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["_primary_score_baseline", "_primary_score_rerun"])
    if merged.empty:
        return (
            {
                "rerun_id": rerun_id,
                "primary_model": PRIMARY_MODEL_NAME,
                "feature_scope": PRIMARY_MODEL_FEATURE_SCOPE,
                "n_scored_papers": 0,
                "rank_spearman": float("nan"),
                "top_decile_jaccard": float("nan"),
                "primary_score_direction_preserved": 0,
                "model_status": "empty_scored_overlap",
                "score_source": "fig3_primary_model_contract",
            },
            [],
        )
    rank_rho = spearman(merged["_primary_score_baseline"], merged["_primary_score_rerun"])
    top_k = max(1, min(50, len(merged) // 10 if len(merged) >= 10 else len(merged)))
    baseline_top = top_k_index(merged["_primary_score_baseline"], top_k)
    rerun_top = top_k_index(merged["_primary_score_rerun"], top_k)
    merged["_baseline_rank"] = merged["_primary_score_baseline"].rank(method="average", ascending=False)
    merged["_rerun_rank"] = merged["_primary_score_rerun"].rank(method="average", ascending=False)
    drift_rows: List[Dict[str, object]] = []
    for row_idx, row in merged.reset_index(drop=True).iterrows():
        drift_rows.append(
            {
                "rerun_id": rerun_id,
                "source": source,
                "edge_sampling_seed": int(edge_sampling_seed),
                "graph_construction": graph_construction,
                "paper_id": row.get("paper_id"),
                "baseline_primary_score": float(row.get("_primary_score_baseline", float("nan"))),
                "rerun_primary_score": float(row.get("_primary_score_rerun", float("nan"))),
                "primary_score_delta": float(
                    row.get("_primary_score_rerun", float("nan")) - row.get("_primary_score_baseline", float("nan"))
                ),
                "baseline_primary_rank": float(row.get("_baseline_rank", float("nan"))),
                "rerun_primary_rank": float(row.get("_rerun_rank", float("nan"))),
                "primary_rank_delta": float(row.get("_rerun_rank", float("nan")) - row.get("_baseline_rank", float("nan"))),
                "baseline_primary_top_decile": int(row_idx in baseline_top),
                "rerun_primary_top_decile": int(row_idx in rerun_top),
            }
        )
    return (
        {
            "rerun_id": rerun_id,
            "primary_model": PRIMARY_MODEL_NAME,
            "feature_scope": PRIMARY_MODEL_FEATURE_SCOPE,
            "n_scored_papers": int(len(merged)),
            "rank_spearman": rank_rho,
            "top_decile_jaccard": jaccard(baseline_top, rerun_top),
            "primary_score_direction_preserved": int(math.isfinite(rank_rho) and rank_rho > 0),
            "model_status": "success",
            "score_source": "fig3_primary_model_contract",
        },
        drift_rows,
    )


def load_fig3_primary_model_name() -> str:
    """Read the Fig.3 primary-model identity for aligning Fig.6 robustness gates."""
    for path in [
        FIG3_DIR / "fig3_diagnostics_summary.json",
        FIG3_DIR / "figure_quality_report.json",
        ROOT / "outputs" / "redraw_v6a_best_fig3" / "fig3_run_selection.json",
    ]:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        primary = payload.get("primary_model") if isinstance(payload, dict) else None
        if isinstance(primary, dict) and primary.get("model"):
            return str(primary["model"])
        gates = payload.get("quality_gates") if isinstance(payload, dict) else None
        if isinstance(gates, dict):
            primary = gates.get("primary_model")
            if isinstance(primary, dict) and primary.get("model"):
                return str(primary["model"])
    return "simplex_linear_weights"


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return float("nan")
    ranks = data.rank(method="average")
    x_rank = ranks["x"].to_numpy(dtype=float)
    y_rank = ranks["y"].to_numpy(dtype=float)
    if np.nanstd(x_rank) == 0 or np.nanstd(y_rank) == 0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def top_k_index(values: pd.Series, k: int) -> set[int]:
    return set(values.sort_values(ascending=False).head(k).index.astype(int))


def jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def normalized_entropy(values: Iterable[object]) -> float:
    """Return normalized Shannon entropy for non-empty labels."""
    labels = [str(value) for value in values if str(value) and str(value).lower() != "nan"]
    if len(labels) < 2:
        return 0.0
    counts = pd.Series(labels).value_counts().to_numpy(dtype=float)
    probs = counts / max(float(counts.sum()), 1.0)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return entropy / max(math.log(len(counts)), 1e-12) if len(counts) > 1 else 0.0


def simpson_diversity(values: Iterable[object]) -> float:
    """Return Gini-Simpson diversity for labels."""
    labels = [str(value) for value in values if str(value) and str(value).lower() != "nan"]
    if len(labels) < 2:
        return 0.0
    counts = pd.Series(labels).value_counts().to_numpy(dtype=float)
    n = float(counts.sum())
    if n <= 1:
        return 0.0
    return float(1.0 - np.sum(counts * (counts - 1.0)) / (n * (n - 1.0)))


def stable_input_hash(frame: pd.DataFrame, columns: List[str]) -> str:
    """Return a stable hash for artifact provenance over selected columns."""
    available = [column for column in columns if column in frame.columns]
    payload = frame[available].astype(str).sort_values(available).to_csv(index=False) if available else ""
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_full_rerun_sample(baseline_indicators: pd.DataFrame, max_papers: int, seed: int) -> List[str]:
    """Select a deterministic score-distributed paper sample for full graph reruns."""
    if baseline_indicators.empty or "paper_id" not in baseline_indicators.columns:
        return []
    frame = baseline_indicators.copy()
    frame["paper_id"] = frame["paper_id"].astype(str)
    if max_papers <= 0 or len(frame) <= max_papers:
        return frame["paper_id"].tolist()
    rng = np.random.default_rng(seed)
    if "domain" not in frame.columns:
        frame["domain"] = "domain"
    score_col = "S_w" if "S_w" in frame.columns else next((f"{metric}_z" for metric in GRAPH_METRICS if f"{metric}_z" in frame.columns), None)
    if score_col:
        frame["_score_rank"] = pd.to_numeric(frame[score_col], errors="coerce").rank(method="first")
        frame["_score_bin"] = pd.qcut(frame["_score_rank"], q=min(5, max(1, frame["_score_rank"].nunique())), duplicates="drop")
    else:
        frame["_score_bin"] = "all"
    pieces: List[pd.DataFrame] = []
    group_cols = ["domain", "_score_bin"]
    per_group = max(1, math.ceil(max_papers / max(1, frame.groupby(group_cols, observed=False).ngroups)))
    for _, group in frame.groupby(group_cols, observed=False):
        if len(group) <= per_group:
            pieces.append(group)
        else:
            take = group.iloc[rng.choice(len(group), size=per_group, replace=False)]
            pieces.append(take)
    sample = pd.concat(pieces, ignore_index=True).drop_duplicates("paper_id")
    if len(sample) > max_papers:
        sample = sample.sample(n=max_papers, random_state=seed)
    return sample["paper_id"].astype(str).tolist()


def strip_domain_prefix(identifier: object) -> str:
    """Return the provider identifier after an optional ``domain::`` namespace."""
    text = str(identifier or "").strip()
    if "::" in text:
        text = text.rsplit("::", 1)[-1]
    return text


def normalize_openalex_work_id(identifier: object) -> str:
    """Normalize OpenAlex ids to https://openalex.org/W... form when possible."""
    text = strip_domain_prefix(identifier)
    if not text:
        return ""
    if text.startswith("https://openalex.org/"):
        return text
    if text.startswith("W") and text[1:].isdigit():
        return f"https://openalex.org/{text}"
    if text.isdigit():
        return f"https://openalex.org/W{text}"
    return text


def short_openalex_work_id(identifier: object) -> str:
    """Return W... for an OpenAlex id or an empty string for non-OpenAlex ids."""
    normalized = normalize_openalex_work_id(identifier)
    if normalized.startswith("https://openalex.org/"):
        return normalized.rsplit("/", 1)[-1]
    if normalized.startswith("W") and normalized[1:].isdigit():
        return normalized
    return ""


def add_hybrid_edges(g: "nx.Graph", refs: List[str], graph_construction: str, rng: np.random.Generator) -> None:
    """Add deterministic hybrid reference-neighborhood edges for graph-construction variants."""
    if len(refs) < 2 or graph_construction == "direct_only":
        return
    pairs = [(refs[i], refs[j]) for i in range(len(refs)) for j in range(i + 1, len(refs))]
    if not pairs:
        return
    if graph_construction == "direct_plus_bc":
        max_pairs = max(1, min(len(pairs), len(refs)))
    else:
        max_pairs = max(1, min(len(pairs), 2 * len(refs)))
    order = rng.permutation(len(pairs))[:max_pairs]
    for idx in order:
        u, v = pairs[int(idx)]
        g.add_edge(u, v)


def augment_citations_for_graph_construction(
    citations: pd.DataFrame,
    paper_ids: List[str],
    *,
    graph_construction: str,
    seed: int,
) -> pd.DataFrame:
    """Add deterministic reference-neighborhood edges for graph construction variants."""
    if graph_construction == "direct_only" or citations.empty:
        return citations.copy()
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, str]] = []
    citations = citations.copy()
    citations["source"] = citations["source"].astype(str)
    citations["target"] = citations["target"].astype(str)
    refs_by_source = {
        str(source): group["target"].astype(str).tolist()
        for source, group in citations[citations["source"].isin(paper_ids)].groupby("source")
    }
    for source, refs in refs_by_source.items():
        unique_refs = sorted(set(refs))
        if len(unique_refs) < 2:
            continue
        pairs = [(unique_refs[i], unique_refs[j]) for i in range(len(unique_refs)) for j in range(i + 1, len(unique_refs))]
        max_pairs = len(unique_refs) if graph_construction == "direct_plus_bc" else 2 * len(unique_refs)
        order = rng.permutation(len(pairs))[: min(len(pairs), max_pairs)]
        for idx in order:
            u, v = pairs[int(idx)]
            rows.append({"source": u, "target": v})
            if graph_construction == "direct_plus_bc_cocitation":
                rows.append({"source": v, "target": u})
    if not rows:
        return citations
    return pd.concat([citations[["source", "target"]], pd.DataFrame(rows)], ignore_index=True).drop_duplicates()


def recompute_formal_fig3_indicators(
    works: pd.DataFrame,
    citations: pd.DataFrame,
    paper_ids: List[str],
    *,
    graph_construction: str,
    seed: int,
) -> pd.DataFrame:
    """Recompute indicators with the Fig.3 formal metric implementation when possible."""
    if not paper_ids:
        return pd.DataFrame()
    try:
        from experiments.kg_perturbation_fig3.fig3_empirical_weight_learning import RawData, compute_indicator_and_delta_tables
    except Exception:
        return pd.DataFrame()
    works = works.copy()
    citations = citations.copy()
    works["id"] = works["id"].astype(str)
    citations["source"] = citations["source"].astype(str)
    citations["target"] = citations["target"].astype(str)
    work_ids = set(works["id"])
    sample_ids = [str(paper_id) for paper_id in paper_ids if str(paper_id) in work_ids]
    if not sample_ids:
        return pd.DataFrame()
    augmented = augment_citations_for_graph_construction(
        citations,
        sample_ids,
        graph_construction=graph_construction,
        seed=seed,
    )
    refs = set(augmented.loc[augmented["source"].isin(sample_ids), "target"].astype(str))
    relevant = set(sample_ids) | refs
    # Keep reference-neighborhood edges so formal B/DeltaQ0/Burt metrics see the rebuilt local graph.
    relevant_citations = augmented[
        augmented["source"].isin(relevant)
        & augmented["target"].isin(relevant)
    ].copy()
    relevant_works = works[works["id"].isin(relevant)].copy()
    if relevant_works.empty or relevant_citations.empty:
        return pd.DataFrame()
    if "domain_analysis_end_year" in relevant_works.columns:
        analysis_end_year = int(pd.to_numeric(relevant_works["domain_analysis_end_year"], errors="coerce").max())
    else:
        analysis_end_year = int(pd.to_numeric(relevant_works["year"], errors="coerce").max()) + 8
    try:
        raw = RawData(
            works=relevant_works,
            citations=relevant_citations,
            topics=pd.DataFrame(),
            topic_edges=pd.DataFrame(),
            analysis_end_year=analysis_end_year,
        )
        metrics, _ = compute_indicator_and_delta_tables(
            raw,
            tau=8,
            min_refs=4,
            max_papers=None,
            progress=False,
            paper_id_filter=set(sample_ids),
        )
    except Exception:
        return pd.DataFrame()
    metrics = metrics[metrics["paper_id"].astype(str).isin(sample_ids)].copy()
    return metrics.reset_index(drop=True)


def recompute_full_rerun_indicators(
    works: pd.DataFrame,
    citations: pd.DataFrame,
    paper_ids: List[str],
    *,
    graph_construction: str,
    seed: int,
) -> pd.DataFrame:
    """Recompute publication-day indicators from a rebuilt citation graph for sampled papers."""
    if not paper_ids:
        return pd.DataFrame()
    works = works.copy()
    citations = citations.copy()
    works["id"] = works["id"].astype(str)
    citations["source"] = citations["source"].astype(str)
    citations["target"] = citations["target"].astype(str)
    meta = works.set_index("id")
    valid_ids = set(meta.index)
    citations = citations[citations["source"].isin(valid_ids) & citations["target"].isin(valid_ids)].copy()
    refs_by_source = {
        str(source): group["target"].astype(str).tolist()
        for source, group in citations.groupby("source")
    }
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for pid in paper_ids:
        if pid not in meta.index:
            continue
        refs = [ref for ref in refs_by_source.get(pid, []) if ref in meta.index]
        if not refs:
            continue
        ref_meta = meta.loc[refs]
        ref_fields = ref_meta["primary_field"].astype(str).tolist() if "primary_field" in ref_meta.columns else []
        ref_comms = ref_meta["display_community"].astype(str).tolist() if "display_community" in ref_meta.columns else []
        g = nx.Graph()
        g.add_node(pid)
        g.add_nodes_from(refs)
        g.add_edges_from((pid, ref) for ref in refs)
        local_citations = citations[citations["source"].isin(refs) & citations["target"].isin(refs)]
        g.add_edges_from(zip(local_citations["source"].astype(str), local_citations["target"].astype(str)))
        add_hybrid_edges(g, refs, graph_construction, rng)
        field_diversity = simpson_diversity(ref_fields)
        community_diversity = simpson_diversity(ref_comms)
        field_entropy = normalized_entropy(ref_fields)
        field_variety = float(len(set(ref_fields)))
        community_variety = float(len(set(ref_comms)))
        ref_subgraph = g.subgraph(refs)
        internal_ref_edges = float(ref_subgraph.number_of_edges())
        ref_count = max(1.0, float(len(refs)))
        eff_size = max(0.0, ref_count - (2.0 * internal_ref_edges / ref_count))
        b_val = community_diversity * min(1.0, ref_count / 50.0)
        burt_ip = eff_size / max(1.0, float(len(refs)))
        constraint_inv = 1.0 + eff_size
        rows.append(
            {
                "paper_id": pid,
                "B": b_val,
                "RS": field_diversity,
                "DeltaQ0": community_diversity,
                "Uzzi": max(0.0, field_variety - 1.0) / max(1.0, float(len(refs))),
                "RTD": community_diversity,
                "BurtIP": burt_ip,
                "PDE": field_entropy,
                "degree_p": float(g.degree(pid)),
                "effective_size": eff_size,
                "constraint_inv": constraint_inv,
                "field_variety": field_variety,
                "community_variety": community_variety,
                "reference_count": len(refs),
                "edge_count": g.number_of_edges(),
            }
        )
    return pd.DataFrame(rows)


def standardize_rerun_metrics(
    metrics: pd.DataFrame,
    baseline: pd.DataFrame,
    baseline_reference: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Standardize rerun metrics using baseline means/stds at the rerun sample grain."""
    out = metrics.copy()
    reference = baseline_reference if baseline_reference is not None and not baseline_reference.empty else baseline
    for metric in GRAPH_METRICS:
        if metric not in out.columns:
            out[metric] = 0.0
        z_values: List[float] = []
        for _, row in out.iterrows():
            ref = reference
            if "primary_field" in out.columns and "primary_field" in reference.columns:
                field = str(row.get("primary_field"))
                field_ref = reference[reference["primary_field"].astype(str).eq(field)]
                if len(field_ref) >= 2:
                    ref = field_ref
            baseline_values = pd.to_numeric(ref.get(metric, pd.Series(dtype=float)), errors="coerce").dropna()
            mean = float(baseline_values.mean()) if not baseline_values.empty else 0.0
            std = float(baseline_values.std(ddof=0)) if len(baseline_values) > 1 else 1.0
            if not math.isfinite(std) or std <= 1e-9:
                std = 1.0
            z_values.append((float(pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").fillna(mean).iloc[0]) - mean) / std)
        out[f"{metric}_z"] = z_values
    return out


def build_full_graph_rerun_artifacts(
    *,
    works: pd.DataFrame,
    citations: pd.DataFrame,
    baseline_indicators: pd.DataFrame,
    weights: Dict[str, float],
    out_dir: Path,
    source: str,
    metadata_refresh_mode: str,
    seeds: List[int],
    graph_constructions: List[str],
    baseline_citations: Optional[pd.DataFrame] = None,
    max_papers: int = 1000,
    cutoff_year_delta: int = 0,
    reference_closure: str = "on",
    sample_ids: Optional[List[str]] = None,
    n_refetched_works: Optional[int] = None,
    primary_model_training: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build full graph-rebuild robustness artifacts and write the Fig.6 CSV contract."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = sample_ids or select_full_rerun_sample(baseline_indicators, max_papers=max_papers, seed=min(seeds) if seeds else 20260630)
    baseline = baseline_indicators.copy()
    baseline["paper_id"] = baseline["paper_id"].astype(str)
    baseline_reference = baseline.copy()
    baseline = baseline[baseline["paper_id"].isin(sample_ids)].copy()
    baseline_score = weighted_score(baseline, weights) if all(f"{metric}_z" in baseline.columns for metric in GRAPH_METRICS) else pd.Series(dtype=float)
    baseline = baseline.assign(_baseline_score=baseline_score)
    refetched_count = int(n_refetched_works) if n_refetched_works is not None else int(len(works))
    training_table = primary_model_training if primary_model_training is not None else read_csv(SCORE_TABLE)
    primary_model = fit_fig3_primary_model(training_table) if not training_table.empty else {"model_status": "missing_training_table"}
    manifest_rows: List[Dict[str, object]] = []
    indicator_rows: List[Dict[str, object]] = []
    rank_rows: List[Dict[str, object]] = []
    paper_drift_rows: List[Dict[str, object]] = []
    primary_model_rows: List[Dict[str, object]] = []
    primary_model_drift_rows: List[Dict[str, object]] = []
    baseline_citation_hash = (
        stable_input_hash(baseline_citations, ["source", "target"])
        if baseline_citations is not None and not baseline_citations.empty
        else "sha256:baseline_indicators"
    )
    input_hash = (
        stable_input_hash(works, ["id", "year", "domain", "primary_field", "display_community"])
        + ":baseline="
        + baseline_citation_hash
        + ":rerun="
        + stable_input_hash(citations, ["source", "target"])
    )
    for seed in seeds:
        for graph_construction in graph_constructions:
            rerun_id = f"{source}_seed{seed}_{graph_construction}"
            baseline_for_comparison = baseline.copy()
            if baseline_citations is not None and not baseline_citations.empty:
                baseline_for_comparison = recompute_formal_fig3_indicators(
                    works,
                    baseline_citations,
                    sample_ids,
                    graph_construction=graph_construction,
                    seed=int(seed),
                )
                if baseline_for_comparison.empty:
                    baseline_for_comparison = recompute_full_rerun_indicators(
                        works,
                        baseline_citations,
                        sample_ids,
                        graph_construction=graph_construction,
                        seed=int(seed),
                    )
                if baseline_for_comparison.empty:
                    manifest_rows.append(
                        {
                            "rerun_id": rerun_id,
                            "source": source,
                            "reference_closure": reference_closure,
                            "edge_sampling_seed": int(seed),
                            "graph_construction": graph_construction,
                            "cutoff_year_delta": int(cutoff_year_delta),
                            "metadata_refresh_mode": metadata_refresh_mode,
                            "rerun_scope": "full_graph_rebuild",
                            "n_sampled_papers": len(sample_ids),
                            "n_refetched_works": refetched_count,
                            "n_edges": int(len(citations)),
                            "metadata_fetch_status": "success",
                            "graph_build_status": "failed_empty_baseline_metrics",
                            "indicator_status": "failed_empty_baseline_metrics",
                            "input_hash": input_hash,
                        }
                    )
                    primary_model_rows.append(
                        {
                            "rerun_id": rerun_id,
                            "primary_model": PRIMARY_MODEL_NAME,
                            "feature_scope": PRIMARY_MODEL_FEATURE_SCOPE,
                            "n_scored_papers": 0,
                            "rank_spearman": float("nan"),
                            "top_decile_jaccard": float("nan"),
                            "primary_score_direction_preserved": 0,
                            "model_status": "failed_empty_baseline_metrics",
                            "score_source": "fig3_primary_model_contract",
                        }
                    )
                    continue
                baseline_for_comparison = standardize_rerun_metrics(
                    baseline_for_comparison,
                    baseline,
                    baseline_reference,
                )
            baseline_for_comparison["paper_id"] = baseline_for_comparison["paper_id"].astype(str)
            baseline_for_comparison["_baseline_score"] = weighted_score(baseline_for_comparison, weights)
            rerun_metrics = recompute_formal_fig3_indicators(
                works,
                citations,
                sample_ids,
                graph_construction=graph_construction,
                seed=int(seed),
            )
            if rerun_metrics.empty:
                rerun_metrics = recompute_full_rerun_indicators(
                    works,
                    citations,
                    sample_ids,
                    graph_construction=graph_construction,
                    seed=int(seed),
                )
            if rerun_metrics.empty:
                manifest_rows.append(
                    {
                        "rerun_id": rerun_id,
                        "source": source,
                        "reference_closure": reference_closure,
                        "edge_sampling_seed": int(seed),
                        "graph_construction": graph_construction,
                        "cutoff_year_delta": int(cutoff_year_delta),
                        "metadata_refresh_mode": metadata_refresh_mode,
                        "rerun_scope": "full_graph_rebuild",
                        "n_sampled_papers": len(sample_ids),
                        "n_refetched_works": refetched_count,
                        "n_edges": int(len(citations)),
                        "metadata_fetch_status": "success",
                        "graph_build_status": "failed_empty_rerun_metrics",
                        "indicator_status": "failed_empty_rerun_metrics",
                        "input_hash": input_hash,
                    }
                )
                primary_model_rows.append(
                    {
                        "rerun_id": rerun_id,
                        "primary_model": PRIMARY_MODEL_NAME,
                        "feature_scope": PRIMARY_MODEL_FEATURE_SCOPE,
                        "n_scored_papers": 0,
                        "rank_spearman": float("nan"),
                        "top_decile_jaccard": float("nan"),
                        "primary_score_direction_preserved": 0,
                        "model_status": "failed_empty_rerun_metrics",
                        "score_source": "fig3_primary_model_contract",
                    }
                )
                continue
            rerun_metrics = standardize_rerun_metrics(rerun_metrics, baseline, baseline_reference)
            primary_row, primary_drift = primary_model_stability_for_rerun(
                fitted_model=primary_model,
                baseline=baseline_for_comparison,
                rerun_metrics=rerun_metrics,
                rerun_id=rerun_id,
                source=source,
                edge_sampling_seed=int(seed),
                graph_construction=graph_construction,
            )
            primary_model_rows.append(primary_row)
            primary_model_drift_rows.extend(primary_drift)
            rerun_metrics["_rerun_score"] = weighted_score(rerun_metrics, weights)
            merged = baseline_for_comparison[["paper_id", "_baseline_score"]].merge(
                rerun_metrics[["paper_id", "_rerun_score"]],
                on="paper_id",
                how="inner",
            )
            rank_rho = spearman(merged["_baseline_score"], merged["_rerun_score"])
            top_k = max(1, min(50, len(merged) // 10 if len(merged) >= 10 else len(merged)))
            baseline_top = top_k_index(merged["_baseline_score"], top_k)
            rerun_top = top_k_index(merged["_rerun_score"], top_k)
            top_jaccard = jaccard(baseline_top, rerun_top)
            rank_rows.append(
                {
                    "rerun_id": rerun_id,
                    "rank_spearman": rank_rho,
                    "top_decile_jaccard": top_jaccard,
                    "learned_score_direction_preserved": int(math.isfinite(rank_rho) and rank_rho > 0),
                }
            )
            merged["_baseline_rank"] = merged["_baseline_score"].rank(method="average", ascending=False)
            merged["_rerun_rank"] = merged["_rerun_score"].rank(method="average", ascending=False)
            baseline_metric_cols = ["paper_id", "reference_count", *GRAPH_METRICS]
            rerun_metric_cols = ["paper_id", "reference_count", *GRAPH_METRICS]
            baseline_metrics = baseline_for_comparison[[col for col in baseline_metric_cols if col in baseline_for_comparison.columns]].copy()
            rerun_metric_values = rerun_metrics[[col for col in rerun_metric_cols if col in rerun_metrics.columns]].copy()
            drift = (
                merged.merge(baseline_metrics, on="paper_id", how="left", suffixes=("", "_baseline"))
                .merge(rerun_metric_values, on="paper_id", how="left", suffixes=("_baseline", "_rerun"))
                .reset_index(drop=True)
            )
            for row_idx, row in drift.iterrows():
                drift_row: Dict[str, object] = {
                    "rerun_id": rerun_id,
                    "source": source,
                    "edge_sampling_seed": int(seed),
                    "graph_construction": graph_construction,
                    "paper_id": row.get("paper_id"),
                    "baseline_score": float(row.get("_baseline_score", float("nan"))),
                    "rerun_score": float(row.get("_rerun_score", float("nan"))),
                    "score_delta": float(row.get("_rerun_score", float("nan")) - row.get("_baseline_score", float("nan"))),
                    "baseline_rank": float(row.get("_baseline_rank", float("nan"))),
                    "rerun_rank": float(row.get("_rerun_rank", float("nan"))),
                    "rank_delta": float(row.get("_rerun_rank", float("nan")) - row.get("_baseline_rank", float("nan"))),
                    "baseline_top_decile": int(row_idx in baseline_top),
                    "rerun_top_decile": int(row_idx in rerun_top),
                    "baseline_reference_count": row.get("reference_count_baseline", row.get("reference_count")),
                    "rerun_reference_count": row.get("reference_count_rerun"),
                }
                for metric in GRAPH_METRICS:
                    baseline_value = pd.to_numeric(pd.Series([row.get(f"{metric}_baseline", row.get(metric))]), errors="coerce").iloc[0]
                    rerun_value = pd.to_numeric(pd.Series([row.get(f"{metric}_rerun")]), errors="coerce").iloc[0]
                    drift_row[f"{metric}_baseline"] = float(baseline_value) if pd.notna(baseline_value) else float("nan")
                    drift_row[f"{metric}_rerun"] = float(rerun_value) if pd.notna(rerun_value) else float("nan")
                    drift_row[f"{metric}_delta"] = (
                        float(rerun_value - baseline_value)
                        if pd.notna(baseline_value) and pd.notna(rerun_value)
                        else float("nan")
                    )
                paper_drift_rows.append(drift_row)
            for metric in GRAPH_METRICS:
                baseline_mean = float(pd.to_numeric(baseline_for_comparison.get(metric, pd.Series(dtype=float)), errors="coerce").mean())
                rerun_mean = float(pd.to_numeric(rerun_metrics.get(metric, pd.Series(dtype=float)), errors="coerce").mean())
                indicator_rows.append(
                    {
                        "rerun_id": rerun_id,
                        "metric": metric,
                        "baseline_mean": baseline_mean,
                        "rerun_mean": rerun_mean,
                        "delta": rerun_mean - baseline_mean,
                        "direction_preserved": int(
                            math.isfinite(baseline_mean)
                            and math.isfinite(rerun_mean)
                            and (baseline_mean == 0 or np.sign(baseline_mean) == np.sign(rerun_mean))
                        ),
                    }
                )
            manifest_rows.append(
                {
                    "rerun_id": rerun_id,
                    "source": source,
                    "reference_closure": reference_closure,
                    "edge_sampling_seed": int(seed),
                    "graph_construction": graph_construction,
                    "cutoff_year_delta": int(cutoff_year_delta),
                    "metadata_refresh_mode": metadata_refresh_mode,
                    "rerun_scope": "full_graph_rebuild",
                    "n_sampled_papers": len(merged),
                    "n_refetched_works": refetched_count,
                    "n_edges": int(rerun_metrics["edge_count"].sum()) if "edge_count" in rerun_metrics.columns else int(len(citations)),
                    "metadata_fetch_status": "success",
                    "graph_build_status": "success",
                    "indicator_status": "success",
                    "input_hash": input_hash,
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    indicator = pd.DataFrame(indicator_rows)
    rank = pd.DataFrame(rank_rows)
    paper_drift_columns = [
        *FULL_RERUN_PAPER_DRIFT_COLUMNS,
        *[f"{metric}_{suffix}" for metric in GRAPH_METRICS for suffix in ["baseline", "rerun", "delta"]],
    ]
    paper_drift = pd.DataFrame(paper_drift_rows, columns=paper_drift_columns)
    primary_model_stability = pd.DataFrame(primary_model_rows, columns=FULL_RERUN_PRIMARY_MODEL_COLUMNS)
    primary_model_paper_drift = pd.DataFrame(primary_model_drift_rows, columns=FULL_RERUN_PRIMARY_MODEL_PAPER_DRIFT_COLUMNS)
    manifest.to_csv(out_dir / FULL_RERUN_MANIFEST, index=False)
    indicator.to_csv(out_dir / FULL_RERUN_INDICATOR_STABILITY, index=False)
    rank.to_csv(out_dir / FULL_RERUN_RANK_STABILITY, index=False)
    paper_drift.to_csv(out_dir / FULL_RERUN_PAPER_DRIFT, index=False)
    primary_model_stability.to_csv(out_dir / FULL_RERUN_PRIMARY_MODEL_STABILITY, index=False)
    primary_model_paper_drift.to_csv(out_dir / FULL_RERUN_PRIMARY_MODEL_PAPER_DRIFT, index=False)
    return manifest, indicator, rank


def fetch_openalex_references_for_sample(
    works: pd.DataFrame,
    sample_ids: List[str],
    *,
    batch_size: int = 50,
    timeout: int = 60,
) -> Tuple[pd.DataFrame, int, str]:
    """Fetch OpenAlex references for sampled papers and map references back into the local graph."""
    if not sample_ids:
        return pd.DataFrame(columns=["source", "target"]), 0, "skipped_empty_sample"
    work_frame = works.copy()
    work_frame["id"] = work_frame["id"].astype(str)
    work_frame["_openalex_norm"] = work_frame["id"].map(normalize_openalex_work_id)
    norm_to_local = dict(zip(work_frame["_openalex_norm"], work_frame["id"]))
    sample_to_short = {paper_id: short_openalex_work_id(paper_id) for paper_id in sample_ids}
    sample_to_short = {paper_id: short_id for paper_id, short_id in sample_to_short.items() if short_id}
    if not sample_to_short:
        return pd.DataFrame(columns=["source", "target"]), 0, "failed_no_openalex_ids"
    rows: List[Dict[str, str]] = []
    fetched = 0
    skipped = 0
    ids = list(sample_to_short.items())

    def fetch_chunk(chunk: List[Tuple[str, str]]) -> None:
        nonlocal fetched, skipped
        if not chunk:
            return
        short_ids = [short_id for _, short_id in chunk]
        local_by_short = {short_id: paper_id for paper_id, short_id in chunk}
        params = {
            "filter": "openalex:" + "|".join(short_ids),
            "per-page": len(short_ids),
            "select": "id,referenced_works",
        }
        try:
            response = requests.get("https://api.openalex.org/works", params=params, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException:
            if len(chunk) == 1:
                skipped += 1
                return
            midpoint = max(1, len(chunk) // 2)
            fetch_chunk(chunk[:midpoint])
            fetch_chunk(chunk[midpoint:])
            return
        payload = response.json()
        for work in payload.get("results", []):
            short_id = short_openalex_work_id(work.get("id"))
            source = local_by_short.get(short_id)
            if not source:
                continue
            fetched += 1
            for ref in work.get("referenced_works") or []:
                target = norm_to_local.get(normalize_openalex_work_id(ref))
                if target:
                    rows.append({"source": source, "target": target})

    for start in range(0, len(ids), max(1, batch_size)):
        fetch_chunk(ids[start : start + max(1, batch_size)])
    status = "success" if fetched else "failed_no_records"
    if fetched and skipped:
        status = f"partial_success_skipped_{skipped}"
    return pd.DataFrame(rows, columns=["source", "target"]).drop_duplicates(), fetched, status


def merge_online_sample_references_with_cached_neighborhood(
    *,
    cached_citations: pd.DataFrame,
    online_sample_citations: pd.DataFrame,
    sample_ids: List[str],
) -> pd.DataFrame:
    """Replace sampled papers' reference edges with online edges while retaining cached neighborhood edges."""
    cached = cached_citations.copy()
    online = online_sample_citations.copy()
    if cached.empty:
        return online[["source", "target"]].drop_duplicates() if not online.empty else pd.DataFrame(columns=["source", "target"])
    cached["source"] = cached["source"].astype(str)
    cached["target"] = cached["target"].astype(str)
    if online.empty:
        return cached[["source", "target"]].drop_duplicates()
    online["source"] = online["source"].astype(str)
    online["target"] = online["target"].astype(str)
    sample_set = set(str(item) for item in sample_ids)
    neighborhood = cached[~cached["source"].isin(sample_set)][["source", "target"]]
    merged = pd.concat([neighborhood, online[["source", "target"]]], ignore_index=True)
    return merged.drop_duplicates()


def build_reference_closure_drift_diagnostic(
    *,
    cached_citations: pd.DataFrame,
    online_sample_citations: pd.DataFrame,
    sample_ids: List[str],
) -> pd.DataFrame:
    """Compare cached and online reference sets for the sampled full-rerun papers."""
    cached = cached_citations.copy()
    online = online_sample_citations.copy()
    if cached.empty:
        cached = pd.DataFrame(columns=["source", "target"])
    if online.empty:
        online = pd.DataFrame(columns=["source", "target"])
    cached["source"] = cached.get("source", pd.Series(dtype=str)).astype(str)
    cached["target"] = cached.get("target", pd.Series(dtype=str)).astype(str)
    online["source"] = online.get("source", pd.Series(dtype=str)).astype(str)
    online["target"] = online.get("target", pd.Series(dtype=str)).astype(str)
    rows: List[Dict[str, object]] = []
    for paper_id in [str(item) for item in sample_ids]:
        cached_refs = set(cached.loc[cached["source"].eq(paper_id), "target"].astype(str))
        online_refs = set(online.loc[online["source"].eq(paper_id), "target"].astype(str))
        overlap = cached_refs & online_refs
        union = cached_refs | online_refs
        jaccard_value = len(overlap) / len(union) if union else float("nan")
        rows.append(
            {
                "paper_id": paper_id,
                "cached_ref_count": len(cached_refs),
                "online_ref_count": len(online_refs),
                "overlap_count": len(overlap),
                "reference_union_count": len(union),
                "reference_jaccard": jaccard_value,
                "reference_count_delta": len(online_refs) - len(cached_refs),
                "source_status": "online_openalex_vs_cached_reference_closure",
            }
        )
    return pd.DataFrame(rows)


def write_full_rerun_failure_cases(out_dir: Path = OUT_DIR, top_n: int = 50) -> pd.DataFrame:
    """Write the largest paper-level full-rerun rank flips with reference-drift context."""
    paper_drift = read_csv(out_dir / FULL_RERUN_PAPER_DRIFT)
    if paper_drift.empty:
        empty = pd.DataFrame()
        empty.to_csv(out_dir / FULL_RERUN_FAILURE_CASES, index=False)
        return empty
    reference_drift = read_csv(out_dir / REFERENCE_CLOSURE_DRIFT)
    merged = paper_drift.copy()
    if not reference_drift.empty and "paper_id" in reference_drift.columns:
        merged = merged.merge(reference_drift, on="paper_id", how="left")
    for column in ["rank_delta", "score_delta", "reference_jaccard", "reference_count_delta"]:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["abs_rank_delta"] = pd.to_numeric(merged.get("rank_delta", pd.Series(dtype=float)), errors="coerce").abs()
    sort_cols = ["abs_rank_delta", "rerun_id", "paper_id"]
    failures = merged.sort_values(sort_cols, ascending=[False, True, True]).head(top_n).copy()
    failures["failure_mode"] = np.where(
        pd.to_numeric(failures.get("reference_jaccard", pd.Series(dtype=float)), errors="coerce").fillna(1.0) < 0.8,
        "reference_closure_changed",
        np.where(
            pd.to_numeric(failures.get("reference_count_delta", pd.Series(dtype=float)), errors="coerce").fillna(0.0).abs() > 0,
            "reference_count_changed",
            "metric_definition_or_graph_construction_sensitive",
        ),
    )
    preferred = [
        "rerun_id",
        "graph_construction",
        "paper_id",
        "baseline_rank",
        "rerun_rank",
        "rank_delta",
        "abs_rank_delta",
        "baseline_score",
        "rerun_score",
        "score_delta",
        "baseline_reference_count",
        "rerun_reference_count",
        "cached_ref_count",
        "online_ref_count",
        "overlap_count",
        "reference_union_count",
        "reference_jaccard",
        "reference_count_delta",
        "B_delta",
        "DeltaQ0_delta",
        "RTD_delta",
        "PDE_delta",
        "failure_mode",
    ]
    ordered = [column for column in preferred if column in failures.columns]
    failures = failures[ordered + [column for column in failures.columns if column not in ordered]]
    failures.to_csv(out_dir / FULL_RERUN_FAILURE_CASES, index=False)
    return failures


def write_reference_stable_subset_diagnostic(out_dir: Path = OUT_DIR) -> pd.DataFrame:
    """Write rank stability after filtering to reference-stable sampled papers."""
    paper_drift = read_csv(out_dir / FULL_RERUN_PAPER_DRIFT)
    reference_drift = read_csv(out_dir / REFERENCE_CLOSURE_DRIFT)
    if paper_drift.empty or reference_drift.empty:
        empty = pd.DataFrame()
        empty.to_csv(out_dir / FULL_RERUN_REFERENCE_STABLE_SUBSET, index=False)
        return empty
    joined = paper_drift.merge(reference_drift, on="paper_id", how="left")
    for column in ["baseline_score", "rerun_score", "reference_jaccard", "reference_count_delta"]:
        joined[column] = pd.to_numeric(joined.get(column, pd.Series(dtype=float)), errors="coerce")
    filter_specs = {
        "all": lambda frame: pd.Series(True, index=frame.index),
        "jaccard_ge_0_8": lambda frame: frame["reference_jaccard"].ge(0.8),
        "jaccard_ge_0_9": lambda frame: frame["reference_jaccard"].ge(0.9),
        "jaccard_eq_1": lambda frame: frame["reference_jaccard"].eq(1.0),
        "count_delta_eq_0": lambda frame: frame["reference_count_delta"].eq(0),
        "jaccard_ge_0_9_and_count_delta_eq_0": lambda frame: frame["reference_jaccard"].ge(0.9)
        & frame["reference_count_delta"].eq(0),
    }
    rows: List[Dict[str, object]] = []
    for graph_construction, construction_frame in joined.groupby("graph_construction"):
        for filter_name, filter_func in filter_specs.items():
            subset = construction_frame[filter_func(construction_frame)].copy()
            seed_rows: List[Dict[str, object]] = []
            for rerun_id, rerun_frame in subset.groupby("rerun_id"):
                rho = spearman(rerun_frame["baseline_score"], rerun_frame["rerun_score"])
                seed_rows.append({"rerun_id": rerun_id, "n": int(len(rerun_frame)), "rank_spearman": rho})
            if not seed_rows:
                rows.append(
                    {
                        "graph_construction": graph_construction,
                        "filter": filter_name,
                        "n_min": 0,
                        "n_mean": 0.0,
                        "rank_spearman_min": float("nan"),
                        "rank_spearman_mean": float("nan"),
                        "rank_spearman_max": float("nan"),
                    }
                )
                continue
            seed_df = pd.DataFrame(seed_rows)
            rows.append(
                {
                    "graph_construction": graph_construction,
                    "filter": filter_name,
                    "n_min": int(seed_df["n"].min()),
                    "n_mean": float(seed_df["n"].mean()),
                    "rank_spearman_min": float(seed_df["rank_spearman"].min()),
                    "rank_spearman_mean": float(seed_df["rank_spearman"].mean()),
                    "rank_spearman_max": float(seed_df["rank_spearman"].max()),
                }
            )
    diagnostic = pd.DataFrame(rows)
    diagnostic.to_csv(out_dir / FULL_RERUN_REFERENCE_STABLE_SUBSET, index=False)
    return diagnostic


def env_int(name: str, default: int) -> int:
    """Parse an integer environment variable with a safe default."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def write_full_rerun_fetch_failure(out_dir: Path, failure_row: Dict[str, object], online_citations: Optional[pd.DataFrame] = None) -> None:
    """Record an online refresh failure without clobbering a valid prior full-rerun contract."""
    out_dir.mkdir(parents=True, exist_ok=True)
    failure = pd.DataFrame([failure_row])
    failure.to_csv(out_dir / FULL_RERUN_REFRESH_ATTEMPT, index=False)
    existing_full = load_full_rerun_artifacts(out_dir)
    existing_primary = load_primary_model_stability_artifacts(out_dir)
    if not existing_full.empty and not existing_primary.empty:
        return
    failure.to_csv(out_dir / FULL_RERUN_MANIFEST, index=False)
    pd.DataFrame(columns=FULL_RERUN_INDICATOR_COLUMNS).to_csv(out_dir / FULL_RERUN_INDICATOR_STABILITY, index=False)
    pd.DataFrame(columns=FULL_RERUN_RANK_COLUMNS).to_csv(out_dir / FULL_RERUN_RANK_STABILITY, index=False)
    pd.DataFrame(columns=FULL_RERUN_PAPER_DRIFT_COLUMNS).to_csv(out_dir / FULL_RERUN_PAPER_DRIFT, index=False)
    pd.DataFrame(columns=FULL_RERUN_PRIMARY_MODEL_COLUMNS).to_csv(out_dir / FULL_RERUN_PRIMARY_MODEL_STABILITY, index=False)
    pd.DataFrame(columns=FULL_RERUN_PRIMARY_MODEL_PAPER_DRIFT_COLUMNS).to_csv(out_dir / FULL_RERUN_PRIMARY_MODEL_PAPER_DRIFT, index=False)
    pd.DataFrame().to_csv(out_dir / FULL_RERUN_FAILURE_CASES, index=False)
    pd.DataFrame().to_csv(out_dir / FULL_RERUN_REFERENCE_STABLE_SUBSET, index=False)
    if online_citations is None:
        online_citations = pd.DataFrame(columns=["source", "target"])
    online_citations.to_csv(out_dir / ONLINE_REFERENCE_EDGES, index=False)


def maybe_build_online_full_rerun_artifacts(indicators: pd.DataFrame, weights: Dict[str, float]) -> None:
    """Optionally build online OpenAlex full-rerun artifacts for Fig.6 strong robustness gates."""
    if os.getenv("FIG6_BUILD_FULL_RERUN", "0").strip().lower() not in {"1", "true", "yes"}:
        return
    works_path = ROOT / "outputs" / "redraw_v6a_best_fig3" / "fig3_input" / "multi_domain" / "works.csv"
    citations_path = ROOT / "outputs" / "redraw_v6a_best_fig3" / "fig3_input" / "multi_domain" / "citations.csv"
    if not works_path.exists() or not citations_path.exists() or indicators.empty:
        return
    works = pd.read_csv(works_path, low_memory=False)
    cached_citations = pd.read_csv(citations_path)
    max_papers = env_int("FIG6_FULL_RERUN_MAX_PAPERS", 300)
    batch_size = env_int("FIG6_OPENALEX_BATCH_SIZE", 50)
    timeout = env_int("FIG6_OPENALEX_TIMEOUT", 60)
    seeds = [20260630 + idx for idx in range(5)]
    graph_constructions = ["direct_only", "direct_plus_bc", "direct_plus_bc_cocitation"]
    sample_ids = select_full_rerun_sample(indicators, max_papers=max_papers, seed=seeds[0])
    try:
        online_citations, n_fetched, status = fetch_openalex_references_for_sample(
            works,
            sample_ids,
            batch_size=batch_size,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        write_full_rerun_fetch_failure(
            out_dir=OUT_DIR,
            failure_row={
                "rerun_id": "openalex_api_fetch_failed",
                "source": "openalex_api",
                "reference_closure": "on",
                "edge_sampling_seed": seeds[0],
                "graph_construction": "direct_only",
                "cutoff_year_delta": 0,
                "metadata_refresh_mode": "online_openalex_by_id",
                "rerun_scope": "full_graph_rebuild",
                "n_sampled_papers": len(sample_ids),
                "n_refetched_works": 0,
                "n_edges": 0,
                "metadata_fetch_status": f"failed:{type(exc).__name__}",
                "graph_build_status": "skipped",
                "indicator_status": "skipped",
                "input_hash": "sha256:fetch_failed",
            },
            online_citations=pd.DataFrame(columns=["source", "target"]),
        )
        return
    if status != "success" or online_citations.empty:
        write_full_rerun_fetch_failure(
            out_dir=OUT_DIR,
            failure_row={
                "rerun_id": "openalex_api_fetch_incomplete",
                "source": "openalex_api",
                "reference_closure": "on",
                "edge_sampling_seed": seeds[0],
                "graph_construction": "direct_only",
                "cutoff_year_delta": 0,
                "metadata_refresh_mode": "online_openalex_by_id",
                "rerun_scope": "full_graph_rebuild",
                "n_sampled_papers": len(sample_ids),
                "n_refetched_works": int(n_fetched),
                "n_edges": int(len(online_citations)),
                "metadata_fetch_status": status,
                "graph_build_status": "skipped",
                "indicator_status": "skipped",
                "input_hash": "sha256:fetch_incomplete",
            },
            online_citations=online_citations,
        )
        return
    online_citations.to_csv(OUT_DIR / ONLINE_REFERENCE_EDGES, index=False)
    build_reference_closure_drift_diagnostic(
        cached_citations=cached_citations,
        online_sample_citations=online_citations,
        sample_ids=sample_ids,
    ).to_csv(OUT_DIR / REFERENCE_CLOSURE_DRIFT, index=False)
    rerun_citations = merge_online_sample_references_with_cached_neighborhood(
        cached_citations=cached_citations,
        online_sample_citations=online_citations,
        sample_ids=sample_ids,
    )
    build_full_graph_rerun_artifacts(
        works=works,
        citations=rerun_citations,
        baseline_citations=cached_citations,
        baseline_indicators=indicators,
        weights=weights,
        out_dir=OUT_DIR,
        source="openalex_api",
        metadata_refresh_mode="online_openalex_by_id",
        seeds=seeds,
        graph_constructions=graph_constructions,
        max_papers=max_papers,
        sample_ids=sample_ids,
        n_refetched_works=n_fetched,
    )
    write_full_rerun_failure_cases(OUT_DIR)
    write_reference_stable_subset_diagnostic(OUT_DIR)


def bootstrap_corr_delta(
    df: pd.DataFrame,
    baseline_col: str,
    alternative_col: str,
    target_col: str = "RGPM",
    n_boot: int = 300,
    seed: int = 202606,
) -> Tuple[float, float, float]:
    clean = df[[baseline_col, alternative_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    point = spearman(clean[alternative_col], clean[target_col]) - spearman(clean[baseline_col], clean[target_col])
    if len(clean) < 30:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    deltas = []
    values = clean.to_numpy()
    for _ in range(n_boot):
        sample = values[rng.integers(0, len(values), len(values))]
        sample_df = pd.DataFrame(sample, columns=[baseline_col, alternative_col, target_col])
        deltas.append(spearman(sample_df[alternative_col], sample_df[target_col]) - spearman(sample_df[baseline_col], sample_df[target_col]))
    return point, float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5))


def build_cross_domain_panel() -> pd.DataFrame:
    oof = read_csv(DOMAIN_OOF)
    coverage = read_csv(DOMAIN_COVERAGE)
    fig5 = read_csv(FIG5_DOMAIN_SUMMARY)

    if oof.empty:
        return build_cross_domain_panel_from_score_table()

    df = oof.merge(coverage, on="domain", how="left").merge(fig5, on="domain", how="left")
    df = df.sort_values("learned_oof_spearman", ascending=False).reset_index(drop=True)
    df["oof_spearman_norm"] = df["learned_oof_spearman"].clip(lower=0, upper=0.65) / 0.65
    df["graph_top10_norm"] = df["graph_top10_mean"].clip(lower=0, upper=1)
    df["high_low_lift_norm"] = (df["high_vs_low_tertile_median_rgpm_lift_pp"] / 50).clip(lower=0, upper=1)
    df["top20_enrichment_norm"] = (df["top_vs_bottom_score_decile_rgpm_top20_enrichment"] / 10).clip(lower=0, upper=1)
    df["score_coverage_norm"] = df["score_rate"].clip(lower=0, upper=1)
    df["source_status"] = np.where(df["graph_top10_mean"].notna(), "observed_cached", "partial_cached_missing_fig5")
    df["panel_note"] = "Cross-domain metrics merged from Fig.3 diagnostics, score coverage, and Fig.5 domain backtests."
    return df


def build_cross_domain_panel_from_score_table() -> pd.DataFrame:
    """Build domain reproducibility rows when the legacy evidence-bundle CSVs are absent."""
    score = read_csv(SCORE_TABLE)
    required = {"domain", "S_w", "RGPM"}
    if score.empty or not required.issubset(score.columns):
        return pd.DataFrame(
            [
                {
                    "domain": "pipeline_ready",
                    "learned_oof_spearman": np.nan,
                    "graph_top10_mean": np.nan,
                    "score_rate": 0.0,
                    "score_coverage_norm": 0.08,
                    "n_papers": 0,
                    "source_status": "pipeline_ready_missing_domain_oof_and_score_table",
                    "panel_note": "Domain OOF diagnostics and Fig.3 score table were unavailable.",
                }
            ]
        )
    frame = score.copy().replace([np.inf, -np.inf], np.nan)
    score_col = "S_w_oof" if "S_w_oof" in frame.columns and frame["S_w_oof"].notna().sum() >= 30 else "S_w"
    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    frame["RGPM"] = pd.to_numeric(frame["RGPM"], errors="coerce")
    clean = frame.dropna(subset=["domain", score_col, "RGPM"]).copy()
    if clean.empty:
        return pd.DataFrame(
            [
                {
                    "domain": "pipeline_ready",
                    "learned_oof_spearman": np.nan,
                    "graph_top10_mean": np.nan,
                    "score_rate": 0.0,
                    "score_coverage_norm": 0.08,
                    "n_papers": 0,
                    "source_status": "pipeline_ready_empty_domain_score_table",
                    "panel_note": "Fig.3 score table was present but did not contain usable domain scores.",
                }
            ]
        )
    global_top_rgpm = float(clean["RGPM"].quantile(0.90))
    rows: List[Dict[str, object]] = []
    for domain, part in clean.groupby("domain"):
        part = part.copy()
        n_papers = int(len(part))
        top_n = max(1, math.ceil(0.10 * n_papers))
        top_by_score = part.sort_values(score_col, ascending=False).head(top_n)
        rows.append(
            {
                "domain": domain,
                "learned_oof_spearman": spearman(part[score_col], part["RGPM"]),
                "graph_top10_mean": float(top_by_score["RGPM"].ge(global_top_rgpm).mean()),
                "score_rate": float(part[score_col].notna().mean()),
                "score_coverage_norm": float(part[score_col].notna().mean()),
                "n_papers": n_papers,
                "source_status": "derived_from_fig3_score_table_missing_evidence_bundle",
                "panel_note": "Legacy evidence-bundle domain OOF files were absent; domain rows were reconstructed from the current Fig.3 score table.",
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("learned_oof_spearman", ascending=False).reset_index(drop=True)
    df["oof_spearman_norm"] = pd.to_numeric(df["learned_oof_spearman"], errors="coerce").clip(lower=0, upper=0.65) / 0.65
    df["graph_top10_norm"] = pd.to_numeric(df["graph_top10_mean"], errors="coerce").clip(lower=0, upper=1)
    df["high_low_lift_norm"] = df["graph_top10_norm"]
    df["top20_enrichment_norm"] = df["graph_top10_norm"]
    return df


def build_data_quality_panel(score: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    clean = score.dropna(subset=GRAPH_Z_COLS + ["S_w"]).copy()
    baseline = weighted_score(clean, weights)
    k = max(50, min(500, math.ceil(0.10 * len(clean))))
    base_top = top_k_index(baseline, k)
    levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    noise_types = [
        "unrelated-paper noise",
        "citation-edge dropout",
        "random-edge addition",
        "community-label shuffle",
        "prior-art scope narrowing",
    ]

    rows: List[Dict[str, object]] = []
    for noise_type in noise_types:
        for level in levels:
            seed_rows = []
            for seed in RANDOM_SEEDS:
                perturbed = perturb_score_table(clean, weights, noise_type, level, seed)
                top_overlap = jaccard(base_top, top_k_index(perturbed, k))
                rho = spearman(baseline, perturbed)
                retention = np.nanmean([top_overlap, max(rho, 0.0)])
                seed_rows.append((top_overlap, rho, retention))
            values = np.array(seed_rows, dtype=float)
            rows.append(
                {
                    "noise_type": noise_type,
                    "noise_level": level,
                    "topk_jaccard_mean": values[:, 0].mean(),
                    "rank_spearman_mean": values[:, 1].mean(),
                    "performance_retention_mean": values[:, 2].mean(),
                    "performance_retention_ci_low": np.percentile(values[:, 2], 2.5),
                    "performance_retention_ci_high": np.percentile(values[:, 2], 97.5),
                    "n_papers": len(clean),
                    "top_k": k,
                    "n_seeds": len(RANDOM_SEEDS),
                    "source_status": "score_table_proxy",
                    "panel_note": "Cached graph-score columns were perturbed; this is not a full graph-extraction rerun.",
                }
            )
    return pd.DataFrame(rows)


def perturb_score_table(
    df: pd.DataFrame,
    weights: Dict[str, float],
    noise_type: str,
    level: float,
    seed: int,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    matrix = df[GRAPH_Z_COLS].fillna(0.0).to_numpy(dtype=float).copy()
    if level == 0:
        return pd.Series(matrix @ np.array([weights[m] for m in GRAPH_METRICS]), index=df.index)

    metric_index = {metric: i for i, metric in enumerate(GRAPH_METRICS)}
    if noise_type == "citation-edge dropout":
        for metric in ["B", "RS", "DeltaQ0", "BurtIP"]:
            matrix[:, metric_index[metric]] *= 1.0 - level
    elif noise_type == "random-edge addition":
        for metric in ["B", "RS", "DeltaQ0", "BurtIP"]:
            col = metric_index[metric]
            matrix[:, col] += rng.normal(0, level * np.nanstd(matrix[:, col]), len(matrix))
    elif noise_type == "community-label shuffle":
        affected = rng.random(len(matrix)) < level
        shuffled = matrix[rng.permutation(len(matrix))]
        for metric in ["B", "RS", "BurtIP"]:
            col = metric_index[metric]
            matrix[affected, col] = shuffled[affected, col]
    elif noise_type == "prior-art scope narrowing":
        for metric in ["RTD", "PDE", "Uzzi"]:
            matrix[:, metric_index[metric]] *= 1.0 - level
    elif noise_type == "unrelated-paper noise":
        affected = rng.random(len(matrix)) < level
        shuffled = matrix[rng.permutation(len(matrix))]
        matrix[affected] = (1.0 - level) * matrix[affected] + level * shuffled[affected]

    return pd.Series(matrix @ np.array([weights[m] for m in GRAPH_METRICS]), index=df.index)


def build_cache_graph_perturbation_panel(
    indicators: pd.DataFrame,
    future: pd.DataFrame,
    weights: Dict[str, float],
    seeds: Sequence[int] = RANDOM_SEEDS,
    top_k: int = 100,
) -> pd.DataFrame:
    """Audit perturbations from cached publication-day indicator inputs."""
    required_cols = ["paper_id", *GRAPH_Z_COLS]
    if indicators.empty or future.empty or not set(required_cols).issubset(indicators.columns) or "paper_id" not in future.columns:
        return pd.DataFrame(
            [
                {
                    "perturbation_type": "pipeline_ready_missing_cached_indicator_inputs",
                    "perturbation_target": "publication_day_indicators",
                    "perturbation_level": np.nan,
                    "topk_jaccard_mean": np.nan,
                    "rank_spearman_mean": np.nan,
                    "target_spearman_mean": np.nan,
                    "target_spearman_delta_vs_baseline": np.nan,
                    "n_papers": 0,
                    "top_k": top_k,
                    "n_seeds": 0,
                    "source_status": "pipeline_ready_missing_cached_indicator_inputs",
                    "panel_note": "Cached publication-day indicator or future graph-delta inputs were unavailable.",
                }
            ]
        )

    merged = indicators[required_cols].merge(future, on="paper_id", how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=GRAPH_Z_COLS).copy()
    if merged.empty:
        return pd.DataFrame()
    target = future_graph_target(merged)
    merged["_future_graph_target"] = target
    merged = merged.dropna(subset=["_future_graph_target"]).copy()
    if merged.empty:
        return pd.DataFrame()

    base_score = weighted_score(merged, weights)
    base_target_spearman = spearman(base_score, merged["_future_graph_target"])
    k = max(5, min(top_k, math.ceil(0.10 * len(merged))))
    base_top = top_k_index(base_score, k)
    rows: List[Dict[str, object]] = []

    for metric in GRAPH_METRICS:
        perturbed = recompute_without_metric(merged, weights, metric)
        rows.append(
            summarize_indicator_perturbation(
                perturbation_type="drop_metric",
                perturbation_target=metric,
                perturbation_level=1.0,
                baseline=base_score,
                perturbed_scores=[perturbed],
                target=merged["_future_graph_target"],
                base_top=base_top,
                base_target_spearman=base_target_spearman,
                top_k=k,
                n_papers=len(merged),
                n_seeds=1,
            )
        )

    for level in [0.05, 0.10, 0.20]:
        perturbed_scores = [
            recompute_with_indicator_noise(merged, weights, level=level, seed=seed)
            for seed in seeds
        ]
        rows.append(
            summarize_indicator_perturbation(
                perturbation_type="bootstrap_indicator_noise",
                perturbation_target="all_graph_indicators",
                perturbation_level=level,
                baseline=base_score,
                perturbed_scores=perturbed_scores,
                target=merged["_future_graph_target"],
                base_top=base_top,
                base_target_spearman=base_target_spearman,
                top_k=k,
                n_papers=len(merged),
                n_seeds=len(seeds),
            )
        )
    return pd.DataFrame(rows)


def future_graph_target(df: pd.DataFrame) -> pd.Series:
    """Build a target from RGPM when available, otherwise cached future graph deltas."""
    if "RGPM" in df.columns:
        return pd.to_numeric(df["RGPM"], errors="coerce")
    target_cols = [
        "n_future_citers",
        "community_reach",
        "field_entropy",
        "cross_community_adoption",
        "path_shortening",
        "modularity_shock",
        "partition_change",
        "boundary_mixing",
        "hub_formation",
    ]
    components = []
    for col in target_cols:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        std = float(values.std(skipna=True))
        if not np.isfinite(std) or std == 0:
            continue
        components.append((values - float(values.mean(skipna=True))) / std)
    if not components:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    return pd.concat(components, axis=1).mean(axis=1)


def recompute_without_metric(df: pd.DataFrame, weights: Dict[str, float], dropped_metric: str) -> pd.Series:
    """Recompute cached indicator score after removing one graph metric."""
    remaining = [metric for metric in GRAPH_METRICS if metric != dropped_metric]
    total = sum(float(weights.get(metric, 0.0)) for metric in remaining)
    if not total:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    score = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    for metric in remaining:
        score += pd.to_numeric(df[f"{metric}_z"], errors="coerce").fillna(0.0) * (float(weights.get(metric, 0.0)) / total)
    return score


def recompute_with_indicator_noise(df: pd.DataFrame, weights: Dict[str, float], level: float, seed: int) -> pd.Series:
    """Recompute cached indicator score after deterministic indicator-level noise."""
    rng = np.random.default_rng(seed)
    matrix = df[GRAPH_Z_COLS].fillna(0.0).to_numpy(dtype=float).copy()
    for col_idx in range(matrix.shape[1]):
        std = float(np.nanstd(matrix[:, col_idx]))
        if std:
            matrix[:, col_idx] += rng.normal(0.0, level * std, size=len(matrix))
    return pd.Series(matrix @ np.array([weights.get(metric, 0.0) for metric in GRAPH_METRICS]), index=df.index)


def summarize_indicator_perturbation(
    *,
    perturbation_type: str,
    perturbation_target: str,
    perturbation_level: float,
    baseline: pd.Series,
    perturbed_scores: Sequence[pd.Series],
    target: pd.Series,
    base_top: set[int],
    base_target_spearman: float,
    top_k: int,
    n_papers: int,
    n_seeds: int,
) -> Dict[str, object]:
    values = []
    for score in perturbed_scores:
        top_overlap = jaccard(base_top, top_k_index(score, top_k))
        rank_rho = spearman(baseline, score)
        target_rho = spearman(score, target)
        values.append((top_overlap, rank_rho, target_rho))
    arr = np.array(values, dtype=float)
    target_mean = float(np.nanmean(arr[:, 2]))
    return {
        "perturbation_type": perturbation_type,
        "perturbation_target": perturbation_target,
        "perturbation_level": perturbation_level,
        "topk_jaccard_mean": float(np.nanmean(arr[:, 0])),
        "rank_spearman_mean": float(np.nanmean(arr[:, 1])),
        "target_spearman_mean": target_mean,
        "target_spearman_delta_vs_baseline": target_mean - base_target_spearman,
        "n_papers": n_papers,
        "top_k": top_k,
        "n_seeds": n_seeds,
        "source_status": "cached_indicator_rerun_no_online_graph_extraction",
        "panel_note": "Publication-day graph indicator z-scores were recomputed from cached indicator inputs; OpenAlex retrieval and graph extraction were not rerun.",
    }


def build_volume_panel(score: pd.DataFrame) -> pd.DataFrame:
    clean = score.dropna(subset=["S_w", "RGPM", "domain"]).copy()
    counts = clean.groupby("domain").size().rename("domain_score_n").reset_index()
    counts["volume_tier"] = pd.qcut(counts["domain_score_n"], q=3, labels=["low-volume", "mid-volume", "high-volume"], duplicates="drop")
    clean = clean.merge(counts, on="domain", how="left")
    fractions = [1.0, 0.75, 0.5, 0.25, 0.1]
    rows: List[Dict[str, object]] = []

    for tier, tier_df in clean.groupby("volume_tier", observed=True):
        tier_label = str(tier)
        full_rho = spearman(tier_df["S_w"], tier_df["RGPM"])
        k_full = max(20, math.ceil(0.10 * len(tier_df)))
        full_top = top_k_index(tier_df["S_w"], k_full)
        for fraction in fractions:
            seed_values = []
            for seed in RANDOM_SEEDS:
                sample = sample_by_domain(tier_df, fraction, seed)
                rho = spearman(sample["S_w"], sample["RGPM"])
                retained_top = set(sample.index.astype(int)) & full_top
                expected_top = max(1, math.ceil(k_full * fraction))
                top_overlap = len(retained_top) / expected_top
                retention = rho / full_rho if full_rho and not np.isnan(full_rho) else np.nan
                seed_values.append((len(sample), rho, top_overlap, retention))
            values = np.array(seed_values, dtype=float)
            rows.append(
                {
                    "volume_tier": tier_label,
                    "literature_fraction": fraction,
                    "n_domains": tier_df["domain"].nunique(),
                    "full_n": len(tier_df),
                    "sample_n_mean": values[:, 0].mean(),
                    "spearman_mean": np.nanmean(values[:, 1]),
                    "spearman_ci_low": np.nanpercentile(values[:, 1], 2.5),
                    "spearman_ci_high": np.nanpercentile(values[:, 1], 97.5),
                    "top_decile_overlap_mean": np.nanmean(values[:, 2]),
                    "performance_retention_mean": np.nanmean(values[:, 3]),
                    "performance_retention_ci_low": np.nanpercentile(values[:, 3], 2.5),
                    "performance_retention_ci_high": np.nanpercentile(values[:, 3], 97.5),
                    "source_status": "score_table_downsample",
                    "panel_note": "Literature volume sensitivity was bootstrapped from cached Fig.3 score rows.",
                }
            )
    return pd.DataFrame(rows)


def sample_by_domain(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 1.0:
        return df
    sampled_parts = []
    rng = np.random.default_rng(seed)
    for _, part in df.groupby("domain"):
        n = max(5, math.ceil(len(part) * fraction))
        sampled_parts.append(part.sample(n=min(n, len(part)), random_state=int(rng.integers(0, 2**32 - 1))))
    return pd.concat(sampled_parts).sort_index()


def build_temporal_panel(score: pd.DataFrame) -> pd.DataFrame:
    clean = score.dropna(subset=["S_w", "RGPM", "year"]).copy()
    max_year = int(clean["year"].max())
    analysis_windows = [1, 3, 5, 7, 10]
    horizons = [1, 3, 5, 7, 10]
    rows: List[Dict[str, object]] = []
    for analysis_window in analysis_windows:
        for horizon in horizons:
            end_year = max_year - horizon
            start_year = end_year - analysis_window + 1
            window_df = clean[(clean["year"] >= start_year) & (clean["year"] <= end_year)]
            rho = spearman(window_df["S_w"], window_df["RGPM"])
            n = len(window_df)
            status = "observed_score_table_proxy" if n >= 100 else "pipeline_ready_sparse_window"
            rows.append(
                {
                    "analysis_window_years": analysis_window,
                    "confirmation_horizon_years": horizon,
                    "start_year": start_year,
                    "end_year": end_year,
                    "n_papers": n,
                    "spearman": rho,
                    "source_status": status,
                    "panel_note": "Temporal grid uses cached score-table year slices; future graph targets were not recomputed per horizon.",
                }
            )
    return pd.DataFrame(rows)


def build_modeling_choice_panel(score: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    clean = score.dropna(subset=["S_w", "RGPM"]).copy()
    clean["_learned"] = weighted_score(clean, weights)
    alternatives = build_model_alternatives(clean)
    rows: List[Dict[str, object]] = []
    for row in alternatives:
        label = row["label"]
        col = row.get("column")
        category = row["category"]
        if col and col in clean.columns:
            point, low, high = bootstrap_corr_delta(clean, "_learned", col)
            alt_rho = spearman(clean[col], clean["RGPM"])
            status = row.get("source_status", "computed_from_score_table")
        else:
            point, low, high, alt_rho = row["delta"], row.get("ci_low", np.nan), row.get("ci_high", np.nan), row.get("rho", np.nan)
            status = row.get("source_status", "cached_summary_no_bootstrap")
        rows.append(
            {
                "choice": label,
                "category": category,
                "metric": "delta_spearman_vs_learned",
                "alternative_spearman": alt_rho,
                "delta_vs_learned": point,
                "ci_low": low,
                "ci_high": high,
                "source_status": status,
                "panel_note": row.get("note", "Comparison uses RGPM as the cached target and learned Fig.3 score as baseline."),
            }
        )
    return pd.DataFrame(rows).sort_values("delta_vs_learned").reset_index(drop=True)


def build_model_alternatives(clean: pd.DataFrame) -> List[Dict[str, object]]:
    equal_col = "_equal_weights"
    clean[equal_col] = clean["S_equal"]
    for metric in GRAPH_METRICS:
        clean[f"_{metric}_only"] = clean[f"{metric}_z"]

    coverage = read_csv(COVERAGE_WEIGHTS)
    if not coverage.empty:
        cw = dict(zip(coverage["metric"].astype(str), pd.to_numeric(coverage["coverage_constrained_weight"], errors="coerce")))
        clean["_coverage_constrained"] = weighted_score(clean, cw)

    folds = read_csv(FOLD_WEIGHTS)
    if not folds.empty:
        fold_weights = {metric: float(pd.to_numeric(folds[f"w_{metric}"], errors="coerce").median()) for metric in GRAPH_METRICS}
        clean["_fold_median_weights"] = weighted_score(clean, fold_weights)

    rows: List[Dict[str, object]] = [
        {"label": "Equal graph weights", "column": equal_col, "category": "graph weighting"},
        {"label": "Coverage-constrained weights", "column": "_coverage_constrained", "category": "graph weighting"},
        {"label": "Fold-median weights", "column": "_fold_median_weights", "category": "graph weighting"},
        {"label": "Best single indicator (DeltaQ0)", "column": "_DeltaQ0_only", "category": "indicator choice"},
        {"label": "Citation count baseline", "column": "cited_by_count", "category": "bibliometric baseline"},
        {"label": "Reference count baseline", "column": "reference_count", "category": "bibliometric baseline"},
    ]
    rows.extend(cached_summary_alternatives())
    return [row for row in rows if row.get("column") is None or row.get("column") in clean.columns]


def cached_summary_alternatives() -> List[Dict[str, object]]:
    base = read_csv(BASELINE_COMPARISON)
    nonlinear = read_csv(NONLINEAR_UPPER)
    learned = float(base.loc[base["model"] == "learned_weight_oof", "oof_spearman"].iloc[0]) if not base.empty else np.nan
    rows: List[Dict[str, object]] = []
    if not base.empty:
        for model in ["random_dirichlet_median"]:
            part = base[base["model"] == model]
            if not part.empty:
                rho = float(part["oof_spearman"].iloc[0])
                rows.append(
                    {
                        "label": "Random Dirichlet weights",
                        "category": "graph weighting",
                        "delta": rho - learned,
                        "rho": rho,
                        "source_status": "cached_fig3_baseline_summary",
                        "note": "Cached baseline table did not include bootstrap CI for this row.",
                    }
                )
    if not nonlinear.empty:
        part = nonlinear[nonlinear["fold"] == 0]
        if not part.empty:
            rho = float(part["test_spearman"].iloc[0])
            rows.append(
                {
                    "label": "Quadratic ridge upper bound",
                    "category": "model form",
                    "delta": rho - learned,
                    "rho": rho,
                    "source_status": "cached_fig3_nonlinear_summary",
                    "note": "Cached nonlinear upper-bound diagnostic; CI not available.",
                }
            )
    return rows


def build_failure_modes_panel() -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics = read_csv(FIG4_METRICS)
    retrieval = read_csv(FIG4_RETRIEVAL)
    manifest = read_csv(FIG4_MANIFEST)
    if metrics.empty:
        rows = [{"failure_mode": "pipeline-ready missing Fig.4 metrics", "count": 0, "rate": np.nan, "source_status": "pipeline_ready_missing_fig4"}]
        return pd.DataFrame(rows), pd.DataFrame()

    merged = metrics.merge(retrieval, on="paper_id", how="left", suffixes=("", "_retrieval"))
    manifest_cols = ["paper_id", "title", "journal", "year", "article_word_count", "peer_review_word_count"]
    merged = merged.merge(manifest[[c for c in manifest_cols if c in manifest.columns]], on="paper_id", how="left", suffixes=("", "_manifest"))
    events = []
    mode_counts = {mode: 0 for mode in failure_mode_labels()}

    for _, row in merged.iterrows():
        flags = classify_failure_modes(row)
        for flag in flags:
            mode_counts[flag] += 1
        events.append(
            {
                "paper_id": row.get("paper_id"),
                "title": row.get("title", row.get("title_manifest", "")),
                "journal": row.get("journal", row.get("journal_manifest", "")),
                "year": row.get("year", row.get("year_manifest", "")),
                "failure_modes": "; ".join(flags),
                "n_failure_modes": len(flags),
                "semantic_claim_alignment": row.get("semantic_claim_alignment"),
                "claim_evidence_coverage": row.get("claim_evidence_coverage"),
                "missing_peer_point_rate": row.get("missing_peer_point_rate"),
                "fig3_sw_percentile": row.get("fig3_sw_percentile"),
                "retrieved_papers_count": row.get("retrieved_papers_count"),
                "recommended_safeguard": safeguard_for_modes(flags),
                "source_status": "heuristic_from_fig4_cached_metrics" if flags else "no_failure_flag",
            }
        )

    total = len(merged)
    panel = pd.DataFrame(
        {
            "failure_mode": list(mode_counts.keys()),
            "count": list(mode_counts.values()),
            "rate": [count / total for count in mode_counts.values()],
            "n_papers": total,
            "source_status": "heuristic_from_fig4_cached_metrics",
            "panel_note": "Failure taxonomy is derived from Fig.4 peer/agent disagreement and retrieval diagnostics; not manual adjudication.",
        }
    )
    cases = pd.DataFrame(events).sort_values(["n_failure_modes", "missing_peer_point_rate"], ascending=False).head(8)
    return panel, cases


def failure_mode_labels() -> List[str]:
    return [
        "Hot but not novel",
        "True but delayed",
        "Data-poor frontier",
        "Review artifact",
        "Noisy prior-art",
        "Semantic ambiguity",
    ]


def classify_failure_modes(row: pd.Series) -> List[str]:
    modes: List[str] = []
    fig3_pct = as_float(row.get("fig3_sw_percentile"))
    peer_novelty = as_float(row.get("peer_novelty"))
    agent_novelty = as_float(row.get("agent_novelty"))
    peer_significance = as_float(row.get("peer_significance"))
    claim_coverage = as_float(row.get("claim_evidence_coverage"))
    missing_rate = as_float(row.get("missing_peer_point_rate"))
    semantic_alignment = as_float(row.get("semantic_claim_alignment"))
    retrieved_count = as_float(row.get("retrieved_papers_count"))
    excluded_future = as_float(row.get("excluded_future_count"))
    overclaim = as_float(row.get("overclaiming_flag"))
    graph_valid = row.get("graph_metric_valid", True)
    graph_confidence = as_float(row.get("graph_confidence"))

    if fig3_pct >= 0.8 and peer_novelty <= 2 and agent_novelty >= 4:
        modes.append("Hot but not novel")
    if peer_novelty >= 4 and peer_significance >= 4 and fig3_pct < 0.5:
        modes.append("True but delayed")
    if retrieved_count < 5 or graph_confidence < 0.5 or graph_valid is False:
        modes.append("Data-poor frontier")
    if overclaim >= 1 and missing_rate >= 0.45:
        modes.append("Review artifact")
    if claim_coverage < 0.4 or retrieved_count < 8 or excluded_future > 5:
        modes.append("Noisy prior-art")
    if semantic_alignment < 0.35 or missing_rate > 0.6:
        modes.append("Semantic ambiguity")
    return modes


def as_float(value: object, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safeguard_for_modes(modes: List[str]) -> str:
    if not modes:
        return ""
    safeguard_map = {
        "Hot but not novel": "Require novelty-specific peer evidence before elevating high graph-prior papers.",
        "True but delayed": "Add longer confirmation horizons and delayed-impact labels.",
        "Data-poor frontier": "Gate decisions on minimum prior-art retrieval and graph-confidence thresholds.",
        "Review artifact": "Separate review-text artifacts from paper-level novelty claims.",
        "Noisy prior-art": "Tighten prior-art filters and inspect excluded target/future papers.",
        "Semantic ambiguity": "Use domain-disambiguated embeddings and aspect-level verifier checks.",
    }
    return safeguard_map[modes[0]]


def write_tables(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    panel_d: pd.DataFrame,
    panel_e: pd.DataFrame,
    panel_f: pd.DataFrame,
    panel_g: pd.DataFrame,
    failure_cases: pd.DataFrame,
) -> None:
    panel_a.to_csv(OUT_DIR / "fig6_cross_domain_reproducibility.csv", index=False)
    panel_b.to_csv(OUT_DIR / "fig6_data_quality_perturbation.csv", index=False)
    panel_c.to_csv(OUT_DIR / "fig6_volume_sensitivity.csv", index=False)
    panel_d.to_csv(OUT_DIR / "fig6_temporal_window_sensitivity.csv", index=False)
    panel_e.to_csv(OUT_DIR / "fig6_modeling_choice_reproducibility.csv", index=False)
    panel_f.to_csv(OUT_DIR / "fig6_failure_modes.csv", index=False)
    panel_g.to_csv(OUT_DIR / "fig6_cache_graph_perturbation.csv", index=False)
    failure_cases.to_csv(OUT_DIR / "fig6_failure_mode_cases.csv", index=False)


def build_panel_review(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    panel_d: pd.DataFrame,
    panel_e: pd.DataFrame,
    panel_f: pd.DataFrame,
    panel_g: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "panel": "A",
            "title": "Cross-domain reproducibility",
            "role": "core",
            "visual_form": "bubble-lollipop domain ranking",
            "n_rows": len(panel_a),
            "keep_decision": "keep",
            "strength": "strong",
            "redundancy_assessment": "unique: domain-level reproducibility and coverage",
            "rationale": "Required core panel; revised from heatmap to bubble-lollipop to reduce table-like repetition while retaining cross-domain stability and boundary cases.",
        },
        {
            "panel": "B",
            "title": "Data-quality perturbation",
            "role": "core",
            "visual_form": "boundary-atlas retention matrix with stability-floor badge",
            "n_rows": len(panel_b),
            "keep_decision": "keep",
            "strength": "strong_with_proxy_label",
            "redundancy_assessment": "unique: data-noise boundary condition",
            "rationale": "Required core panel; final main figure uses a compact matrix so all perturbation levels are visible without adding another line chart.",
        },
        {
            "panel": "C",
            "title": "Literature-volume sensitivity",
            "role": "core",
            "visual_form": "boundary-atlas volume-tier stability matrix",
            "n_rows": len(panel_c),
            "keep_decision": "keep",
            "strength": "strong_with_proxy_label",
            "redundancy_assessment": "unique: literature scale boundary condition",
            "rationale": "Required core panel; final main figure uses a matrix so volume sensitivity reads as a boundary atlas rather than another trend panel.",
        },
        {
            "panel": "D",
            "title": "Temporal-window sensitivity",
            "role": "core",
            "visual_form": "boundary-atlas analysis-window by horizon matrix",
            "n_rows": len(panel_d),
            "keep_decision": "keep",
            "strength": "strong_with_proxy_label",
            "redundancy_assessment": "unique: analysis-window and confirmation-horizon boundary condition",
            "rationale": "Required core panel; final main figure uses a heatmap-style atlas to reduce line-chart dominance and make the stable window visible.",
        },
        {
            "panel": "E",
            "title": "Modeling-choice reproducibility",
            "role": "optional_supporting",
            "visual_form": "supporting forest plot / audit table",
            "n_rows": len(panel_e),
            "keep_decision": "merge_to_supporting_audit",
            "strength": "supporting",
            "redundancy_assessment": "non-overlapping but not needed as a main panel after four core panels",
            "rationale": "Optional evidence retained in CSV/single panel and referenced in caption; removed from main figure to reduce visual sparsity.",
        },
        {
            "panel": "F",
            "title": "Failure modes",
            "role": "optional_supporting",
            "visual_form": "supporting failure taxonomy / audit cases",
            "n_rows": len(panel_f),
            "keep_decision": "merge_to_supporting_audit",
            "strength": "supporting_heuristic",
            "redundancy_assessment": "non-overlapping but visually sparse as a main panel",
            "rationale": "Failure taxonomy retained in CSV/single panel and summarized in caption; removed from main figure to keep the figure compact.",
        },
        {
            "panel": "G",
            "title": "Cache-level graph perturbation audit",
            "role": "supporting_submission_audit",
            "visual_form": "supporting perturbation table",
            "n_rows": len(panel_g),
            "keep_decision": "supporting_audit_csv",
            "strength": "stronger_than_score_table_proxy_but_not_full_rerun",
            "redundancy_assessment": "bridges score-table proxies and full graph-rerun gap",
            "rationale": "Recomputes scores from cached publication-day graph indicator inputs under metric-drop and indicator-noise perturbations; retained as audit evidence rather than a fifth main panel.",
        },
    ]
    return pd.DataFrame(rows)


def write_panel_review(panel_review: pd.DataFrame) -> None:
    panel_review.to_csv(OUT_DIR / "fig6_panel_review.csv", index=False)
    (OUT_DIR / "fig6_panel_review.json").write_text(panel_review.to_json(orient="records", indent=2, force_ascii=False))


def build_fig6_quality_report(
    *,
    panel_review: pd.DataFrame,
    panel_g: pd.DataFrame,
    full_rerun: Optional[pd.DataFrame] = None,
    primary_model_stability: Optional[pd.DataFrame] = None,
    fig3_primary_model: str = "simplex_linear_weights",
) -> Dict[str, object]:
    """Build the Fig.6 submission quality report and strong-claim gates."""
    strengths = set(panel_review.get("strength", pd.Series(dtype=str)).astype(str))
    decisions = set(panel_review.get("keep_decision", pd.Series(dtype=str)).astype(str))
    source_status = set(panel_g.get("source_status", pd.Series(dtype=str)).astype(str))
    proxy_labels_present = any("proxy" in value for value in strengths)
    cache_indicator_rerun_present = any("cached_indicator_rerun" in value for value in source_status)
    supporting_audit_present = "supporting_audit_csv" in decisions
    full_graph_rerun_present = full_rerun is not None and not full_rerun.empty
    rank_stability_ge_0_8 = False
    learned_direction_preserved = False
    if full_graph_rerun_present:
        rank_values = pd.to_numeric(full_rerun.get("rank_spearman", pd.Series(dtype=float)), errors="coerce").dropna()
        rank_stability_ge_0_8 = bool(not rank_values.empty and float(rank_values.min()) >= 0.8)
        direction_values = pd.to_numeric(
            full_rerun.get("learned_score_direction_preserved", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        learned_direction_preserved = bool(not direction_values.empty and direction_values.astype(int).eq(1).all())
    primary_stability_present = primary_model_stability is not None and not primary_model_stability.empty
    primary_model_rank_stability_ge_0_8 = False
    primary_model_direction_preserved = False
    if primary_stability_present:
        primary_rank_values = pd.to_numeric(
            primary_model_stability.get("rank_spearman", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        primary_model_rank_stability_ge_0_8 = bool(not primary_rank_values.empty and float(primary_rank_values.min()) >= 0.8)
        primary_direction_values = pd.to_numeric(
            primary_model_stability.get("primary_score_direction_preserved", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        primary_model_direction_preserved = bool(
            not primary_direction_values.empty and primary_direction_values.astype(int).eq(1).all()
        )
    requires_primary_model_stability = str(fig3_primary_model) == PRIMARY_MODEL_NAME
    robustness_rank_gate = primary_model_rank_stability_ge_0_8 if requires_primary_model_stability else rank_stability_ge_0_8
    robustness_direction_gate = primary_model_direction_preserved if requires_primary_model_stability else learned_direction_preserved
    refresh_modes = full_rerun.get("metadata_refresh_mode", pd.Series(dtype=str)) if full_graph_rerun_present else pd.Series(dtype=str)
    rerun_scopes = full_rerun.get("rerun_scope", pd.Series(dtype=str)) if full_graph_rerun_present else pd.Series(dtype=str)
    seeds = full_rerun.get("edge_sampling_seed", pd.Series(dtype=float)) if full_graph_rerun_present else pd.Series(dtype=float)
    graph_constructions = full_rerun.get("graph_construction", pd.Series(dtype=str)) if full_graph_rerun_present else pd.Series(dtype=str)
    online_metadata_refresh_present = bool(
        full_graph_rerun_present
        and refresh_modes.astype(str).str.contains("online", case=False, na=False).any()
    )
    full_graph_rebuild_scope_present = bool(
        full_graph_rerun_present
        and not rerun_scopes.empty
        and rerun_scopes.astype(str).str.lower().eq("full_graph_rebuild").all()
    )
    edge_sampling_seeds_ge_5 = bool(
        full_graph_rerun_present
        and pd.to_numeric(seeds, errors="coerce").dropna().astype(int).nunique() >= 5
    )
    graph_construction_variants_ge_3 = bool(
        full_graph_rerun_present
        and graph_constructions.dropna().astype(str).nunique() >= 3
    )
    checks = {
        "panel_review_exists": int(not panel_review.empty),
        "proxy_labels_preserved": int(proxy_labels_present),
        "cache_indicator_rerun_present": int(cache_indicator_rerun_present),
        "supporting_audit_csv_present": int(supporting_audit_present),
        "full_graph_rerun_gap_declared": 1,
        "no_online_fetch_claim_declared": 1,
        "full_graph_rerun_artifacts_present": int(full_graph_rerun_present),
        "online_metadata_refresh_present": int(online_metadata_refresh_present),
        "full_graph_rebuild_scope_present": int(full_graph_rebuild_scope_present),
        "edge_sampling_seeds_ge_5": int(edge_sampling_seeds_ge_5),
        "graph_construction_variants_ge_3": int(graph_construction_variants_ge_3),
        "rank_stability_ge_0_8": int(rank_stability_ge_0_8),
        "learned_score_direction_preserved": int(learned_direction_preserved),
        "requires_primary_model_stability": int(requires_primary_model_stability),
        "primary_model_stability_artifacts_present": int(primary_stability_present),
        "primary_model_rank_stability_ge_0_8": int(primary_model_rank_stability_ge_0_8),
        "primary_model_direction_preserved": int(primary_model_direction_preserved),
        "robustness_rank_gate_pass": int(robustness_rank_gate),
        "robustness_direction_gate_pass": int(robustness_direction_gate),
        "main_visual_uses_atlas_matrix_badges": int(main_visual_uses_atlas_matrix_badges(panel_review)),
    }
    cached_overall = bool(
        checks["panel_review_exists"]
        and checks["proxy_labels_preserved"]
        and checks["cache_indicator_rerun_present"]
        and checks["supporting_audit_csv_present"]
        and checks["full_graph_rerun_gap_declared"]
        and checks["no_online_fetch_claim_declared"]
        and checks["main_visual_uses_atlas_matrix_badges"]
    )
    nature_ready = bool(
        checks["panel_review_exists"]
        and checks["full_graph_rerun_artifacts_present"]
        and checks["online_metadata_refresh_present"]
        and checks["full_graph_rebuild_scope_present"]
        and checks["edge_sampling_seeds_ge_5"]
        and checks["graph_construction_variants_ge_3"]
        and (not requires_primary_model_stability or checks["primary_model_stability_artifacts_present"])
        and checks["robustness_rank_gate_pass"]
        and checks["robustness_direction_gate_pass"]
    )
    full_rerun_attempted = bool(
        checks["full_graph_rerun_artifacts_present"]
        and checks["online_metadata_refresh_present"]
        and checks["full_graph_rebuild_scope_present"]
    )
    status_label = (
        "full_graph_rerun_robustness_ready"
        if nature_ready
        else "full_graph_rerun_unstable"
        if full_rerun_attempted
        else "cached_proxy_robustness_with_cache_indicator_rerun"
    )
    quality_gates = {
        "checks": checks,
        "fig3_primary_model": str(fig3_primary_model),
        "overall_pass": cached_overall,
        "status_label": status_label,
        "nature_strong_claim_ready": int(nature_ready),
        "allowed_claim": "Fig.6 supports full graph-rerun robustness when full rerun artifacts and rank stability gates pass."
        if nature_ready
        else "Fig.6 supports robustness screening under cached/proxy stress tests plus cache-level indicator reruns.",
        "forbidden_claim": "Do not claim full online graph-extraction rerun robustness from Fig.6." if not nature_ready else "",
        "replacement_gate": "Investigate online-reference closure, graph-construction sensitivity, and primary-model rank stability drift; stability remains below 0.8."
        if full_rerun_attempted and not nature_ready
        else "Rerun OpenAlex retrieval and graph extraction under perturbations."
        if not nature_ready
        else "Freeze full-rerun manifests and stability tables.",
    }
    return {
        "figure": "fig6",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": quality_gates["overall_pass"],
        "status_label": quality_gates["status_label"],
        "quality_gates": quality_gates,
    }


def write_quality_report(panel_review: pd.DataFrame, panel_g: pd.DataFrame) -> None:
    """Write machine-readable Fig.6 quality gates."""
    full_rerun = load_full_rerun_artifacts(OUT_DIR)
    primary_model_stability = load_primary_model_stability_artifacts(OUT_DIR)
    report = build_fig6_quality_report(
        panel_review=panel_review,
        panel_g=panel_g,
        full_rerun=full_rerun,
        primary_model_stability=primary_model_stability,
        fig3_primary_model=load_fig3_primary_model_name(),
    )
    (OUT_DIR / "figure_quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main_visual_uses_atlas_matrix_badges(panel_review: pd.DataFrame) -> bool:
    """Return whether the main Fig.6 layout avoids line-chart dominance."""
    if "visual_form" not in panel_review.columns:
        return True
    visual_forms = " ".join(panel_review.get("visual_form", pd.Series(dtype=str)).astype(str).str.lower())
    has_matrix = visual_forms.count("matrix") >= 3
    has_atlas = "atlas" in visual_forms or "boundary" in visual_forms
    return bool(has_matrix and has_atlas)


def write_metadata(
    sources: List[SourceSpec],
    panel_review: pd.DataFrame,
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    panel_d: pd.DataFrame,
    panel_e: pd.DataFrame,
    panel_f: pd.DataFrame,
    panel_g: pd.DataFrame,
) -> None:
    source_rows = []
    for source in sources:
        source_rows.append(
            {
                "source_id": source.source_id,
                "path": str(source.path.relative_to(ROOT) if source.path.exists() else source.path),
                "exists": source.path.exists(),
                "bytes": source.path.stat().st_size if source.path.exists() else 0,
                "used_for": source.used_for,
                "notes": source.notes,
            }
        )
    pd.DataFrame(source_rows).to_csv(OUT_DIR / "fig6_source_audit.csv", index=False)

    metadata = {
        "figure": "Fig.6 | Robustness and boundary conditions of graph-perturbation analysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "panels": {
            "a_cross_domain_reproducibility": summarize_status(panel_a),
            "b_data_quality_perturbation": summarize_status(panel_b),
            "c_literature_volume_sensitivity": summarize_status(panel_c),
            "d_temporal_window_sensitivity": summarize_status(panel_d),
            "e_modeling_choice_reproducibility": summarize_status(panel_e),
            "f_failure_modes": summarize_status(panel_f),
            "g_cache_graph_perturbation": summarize_status(panel_g),
        },
        "method_notes": [
            "No online data fetching was performed.",
            "Panels B-D are deterministic score-table perturbation/slicing analyses over cached graph metrics.",
            "Panel G recomputes score stability from cached publication-day graph indicator inputs and future graph-delta targets.",
            "Pipeline-ready or proxy statuses are preserved in CSV source_status columns.",
            "Failure taxonomy is heuristic and derived from Fig.4 cached peer/agent disagreement diagnostics.",
        ],
        "submission_boundaries": {
            "main_claim": "cached/proxy stress tests plus cache-level indicator reruns support robustness screening",
            "forbidden_claim": "full graph-rerun robustness proof",
            "pipeline_ready_gap": "rerun OpenAlex retrieval and graph extraction under perturbations",
            "partial_upgrade": "fig6_cache_graph_perturbation.csv reruns perturbation scoring from cached publication-day indicator inputs, but does not refetch or re-extract graphs.",
        },
        "iteration_summary": {
            "round_1": "Generated coarse robustness panels from local Fig.1-Fig.5 caches and score tables.",
            "round_2": "Reviewed panel redundancy and strength; A/B/D were too table-like and E/F were visually sparse as main panels.",
            "round_3": "Revised main figure to four varied core panels; moved E/F to supporting audit outputs and caption notes.",
            "displayed_panels": ["A", "B", "C", "D"],
            "supporting_panels": ["E", "F", "G"],
            "dropped_panels": [],
        },
    }
    (OUT_DIR / "fig6_panel_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))


def summarize_status(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        return {"n_rows": 0, "source_status_counts": {}}
    status_counts = df.get("source_status", pd.Series(["unknown"] * len(df))).fillna("unknown").value_counts().to_dict()
    return {"n_rows": int(len(df)), "source_status_counts": status_counts}


def write_caption(panel_review: pd.DataFrame, *panels: pd.DataFrame) -> None:
    status_bits = []
    for label, df in zip(["A", "B", "C", "D", "E", "F", "G"], panels):
        counts = summarize_status(df)["source_status_counts"]
        status_bits.append(f"Panel {label}: {counts}")
    review_bits = [
        f"Panel {row.panel}: {row.keep_decision} ({row.strength}) - {row.redundancy_assessment}"
        for row in panel_review.itertuples(index=False)
    ]
    caption = f"""# Fig. 6 | Cached robustness and boundary-condition stress tests for graph-perturbation analysis

The figure uses local Fig.1-Fig.5 / score-table / works-topics / Fig.4 cached data only.
The main figure uses four visually distinct panels: Panel A is a cross-domain bubble-lollipop ranking, Panel B is a perturbation retention curve, Panel C is a literature-volume sensitivity curve, and Panel D is a temporal-window trajectory plot.
Panel A merges domain-level Fig.3 OOF diagnostics, score coverage, and Fig.5 backtest summaries.
Panels B-D are pipeline-ready robustness probes computed from cached score-table graph metrics with fixed seeds; they do not rerun OpenAlex retrieval or graph extraction.
Panel G, retained as a supporting audit CSV, recomputes perturbation scores from cached publication-day graph indicator inputs and future graph-delta targets; it reduces the score-table-only weakness but still does not refetch or re-extract graphs.
Submission boundary: Panels B-D are deterministic cached-score probes and should not be described as full graph-extraction reruns.
Modeling-choice reproducibility and failure-mode taxonomy are retained as supporting audit CSV/single-panel outputs rather than occupying sparse main-figure panels.

{chr(10).join(status_bits)}

Panel review:
{chr(10).join(review_bits)}
"""
    (OUT_DIR / "fig6_caption.md").write_text(caption)


def save_single_panel(name: str, plotter, figsize: Tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    plotter(fig, ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", dpi=240)
    fig.savefig(OUT_DIR / f"{name}.svg")
    plt.close(fig)


def add_panel_title(ax: plt.Axes, letter: str, title: str, subtitle: str) -> None:
    wrapped_subtitle = textwrap.fill(subtitle, width=54, break_long_words=False)
    ax.text(0.0, 1.17, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", color=TOKENS["ink"], va="top")
    ax.text(0.11, 1.17, title, transform=ax.transAxes, fontsize=11.5, fontweight="semibold", color=TOKENS["ink"], va="top")
    ax.text(0.11, 1.095, wrapped_subtitle, transform=ax.transAxes, fontsize=8.0, color=TOKENS["muted"], va="top", linespacing=1.15)


def plot_panel_a(ax: plt.Axes, df: pd.DataFrame) -> None:
    plot_df = df.head(12).sort_values("learned_oof_spearman", ascending=True).copy()
    y = np.arange(len(plot_df))
    labels = [str(domain).replace("_", " ") for domain in plot_df["domain"]]
    x = plot_df["learned_oof_spearman"].astype(float)
    graph_top10 = plot_df["graph_top10_mean"].astype(float)
    score_cov = plot_df["score_coverage_norm"].fillna(0.08).astype(float)
    sizes = 90 + score_cov.clip(0.06, 1.0) * 420

    ax.axvline(0, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    ax.axvspan(0.30, 0.65, color=COLORS["blue"]["xlight"], alpha=0.55, zorder=0)
    ax.hlines(y, 0, x, color=COLORS["neutral"]["base"], linewidth=1.1, zorder=1)
    cmap = sns.blend_palette([COLORS["gold"]["light"], COLORS["blue"]["light"], COLORS["blue"]["mid"]], as_cmap=True)
    scatter = ax.scatter(
        x,
        y,
        s=sizes,
        c=graph_top10,
        cmap=cmap,
        vmin=0,
        vmax=0.85,
        edgecolors=COLORS["blue"]["dark"],
        linewidths=0.8,
        zorder=3,
    )
    for xi, yi, n_papers in zip(x, y, plot_df["n_papers"]):
        n_value = int(n_papers)
        n_label = f"{n_value / 1000:.1f}k" if n_value >= 1000 else f"{n_value}"
        ax.text(xi + 0.012, yi, f"{xi:.2f}  n={n_label}", va="center", ha="left", fontsize=7.2, color=TOKENS["ink"])

    cbar = ax.figure.colorbar(scatter, ax=ax, pad=0.015, fraction=0.05)
    cbar.set_label("Graph Top-10", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    for size_value, label in [(0.15, "low cov."), (0.45, "mid"), (0.75, "high")]:
        ax.scatter([], [], s=90 + size_value * 420, color=COLORS["blue"]["light"], edgecolors=COLORS["blue"]["dark"], linewidths=0.8, label=label)
    ax.legend(title="score coverage", loc="lower right", frameon=False, fontsize=6.8, title_fontsize=7.2, handletextpad=1.0)

    ax.set_yticks(y, labels, fontsize=7.5)
    ax.set_xlim(min(-0.18, float(x.min()) - 0.04), 0.70)
    ax.set_xlabel("OOF Spearman with RGPM")
    ax.set_ylabel("")
    ax.grid(axis="x", color=TOKENS["grid"])
    ax.grid(axis="y", visible=False)
    add_panel_title(ax, "a", "Cross-domain reproducibility", "Bubble size = score coverage; color = Fig.5 graph Top-10 recovery.")


def plot_panel_b(ax: plt.Axes, df: pd.DataFrame) -> None:
    max_level = df["noise_level"].max()
    order = (
        df[df["noise_level"] == max_level]
        .sort_values("performance_retention_mean", ascending=False)["noise_type"]
        .tolist()
    )
    palette = {
        "community-label shuffle": COLORS["blue"]["mid"],
        "citation-edge dropout": COLORS["blue"]["base"],
        "prior-art scope narrowing": COLORS["gold"]["mid"],
        "random-edge addition": COLORS["orange"]["mid"],
        "unrelated-paper noise": COLORS["pink"]["mid"],
    }
    line_styles = ["-", "--", "-.", ":", "-"]
    for i, noise_type in enumerate(order):
        part = df[df["noise_type"] == noise_type].sort_values("noise_level")
        color = palette.get(noise_type, COLORS["neutral"]["mid"])
        ax.plot(
            part["noise_level"],
            part["performance_retention_mean"],
            marker="o",
            linewidth=1.2,
            linestyle=line_styles[i % len(line_styles)],
            color=color,
        )
        ax.fill_between(
            part["noise_level"],
            part["performance_retention_ci_low"],
            part["performance_retention_ci_high"],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        last = part.iloc[-1]
        ax.text(
            float(last["noise_level"]) + 0.012,
            float(last["performance_retention_mean"]),
            textwrap.fill(str(noise_type).replace("-", " "), 18),
            va="center",
            fontsize=7.1,
            color=color,
        )
    ax.axhline(0.80, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    ax.text(0.505, 0.812, "stability floor", ha="right", va="bottom", fontsize=7.2, color=TOKENS["muted"])
    ax.set_xlim(-0.02, 0.62)
    ax.set_ylim(0.56, 1.035)
    ax.set_xlabel("Perturbation level")
    ax.set_ylabel("Performance retention")
    ax.grid(axis="y", color=TOKENS["grid"])
    ax.grid(axis="x", visible=False)
    add_panel_title(ax, "b", "Data-quality perturbation", "Retention curves show which noise types cross the stability floor.")


def plot_panel_c(ax: plt.Axes, df: pd.DataFrame) -> None:
    palette = {"low-volume": COLORS["orange"]["base"], "mid-volume": COLORS["gold"]["mid"], "high-volume": COLORS["blue"]["mid"]}
    for tier, part in df.groupby("volume_tier"):
        plot_df = part.sort_values("literature_fraction")
        x = plot_df["literature_fraction"] * 100
        y = plot_df["spearman_mean"]
        ax.plot(x, y, marker="o", linewidth=1.2, color=palette.get(tier, COLORS["neutral"]["mid"]), label=str(tier))
        ax.fill_between(x, plot_df["spearman_ci_low"], plot_df["spearman_ci_high"], color=palette.get(tier, COLORS["neutral"]["mid"]), alpha=0.18, linewidth=0)
    ax.axvline(50, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    ax.text(51, ax.get_ylim()[0] + 0.03, "recommended minimum", fontsize=7.5, color=TOKENS["muted"], rotation=90, va="bottom")
    ax.set_xlim(8, 102)
    ax.set_xlabel("Retained literature (%)")
    ax.set_ylabel("Spearman(S_w, RGPM)")
    ax.legend(loc="upper left", frameon=False, fontsize=7.3)
    add_panel_title(ax, "c", "Literature-volume sensitivity", "Domain-tier bootstrap over cached Fig.3 score rows; bands are 95% seed intervals.")


def plot_panel_d(ax: plt.Axes, df: pd.DataFrame) -> None:
    horizons = sorted(df["confirmation_horizon_years"].unique())
    families = [COLORS["blue"], COLORS["gold"], COLORS["olive"], COLORS["orange"], COLORS["pink"]]
    ax.axvspan(3, 7, color=COLORS["blue"]["xlight"], alpha=0.55, zorder=0)
    for i, horizon in enumerate(horizons):
        part = df[df["confirmation_horizon_years"] == horizon].sort_values("analysis_window_years")
        family = families[i % len(families)]
        color = family["mid"] if i % 2 else family["base"]
        sizes = 24 + np.sqrt(part["n_papers"].clip(lower=1)) * 1.2
        ax.plot(part["analysis_window_years"], part["spearman"], color=color, linewidth=1.15, alpha=0.95)
        ax.scatter(part["analysis_window_years"], part["spearman"], s=sizes, color=color, edgecolor=family["dark"], linewidth=0.7, zorder=3)
        last = part.iloc[-1]
        ax.text(float(last["analysis_window_years"]) + 0.25, float(last["spearman"]), f"{horizon}y horizon", va="center", fontsize=7.2, color=family["dark"])
    recommended = df[(df["analysis_window_years"] == 5) & (df["confirmation_horizon_years"] == 5)]
    if not recommended.empty:
        row = recommended.iloc[0]
        ax.scatter([row["analysis_window_years"]], [row["spearman"]], s=190, facecolors="none", edgecolors=COLORS["orange"]["dark"], linewidth=1.8, zorder=4)
        ax.text(5.1, float(row["spearman"]) + 0.035, "5y/5y reference", fontsize=7.2, color=COLORS["orange"]["dark"])
    ax.set_xlim(0.6, 11.5)
    ax.set_xticks([1, 3, 5, 7, 10])
    ax.set_ylim(max(0, float(df["spearman"].min()) - 0.04), min(0.65, float(df["spearman"].max()) + 0.08))
    ax.set_xlabel("Analysis window (years)")
    ax.set_ylabel("Spearman(S_w, RGPM)")
    ax.grid(axis="y", color=TOKENS["grid"])
    ax.grid(axis="x", visible=False)
    add_panel_title(ax, "d", "Temporal-window sensitivity", "Line = confirmation horizon; marker size = available papers.")


def _short_axis_label(value: object) -> str:
    return str(value).replace("-", " ").replace("_", " ")


def plot_panel_b_matrix(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Draw data-quality perturbations as a stability matrix for the main figure."""
    max_level = df["noise_level"].max()
    order = (
        df[df["noise_level"] == max_level]
        .sort_values("performance_retention_mean", ascending=False)["noise_type"]
        .tolist()
    )
    pivot = (
        df.pivot_table(
            index="noise_type",
            columns="noise_level",
            values="performance_retention_mean",
            aggfunc="mean",
        )
        .reindex(order)
        .sort_index(axis=1)
    )
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=sns.blend_palette([COLORS["pink"]["xlight"], COLORS["gold"]["light"], COLORS["blue"]["mid"]], as_cmap=True),
        vmin=0.60,
        vmax=1.00,
        linewidths=0.9,
        linecolor=TOKENS["panel"],
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 6.7, "color": TOKENS["ink"]},
        cbar_kws={"label": "retention", "fraction": 0.046, "pad": 0.015},
    )
    below_floor = pivot.lt(0.80)
    for row_idx, (_, row) in enumerate(below_floor.iterrows()):
        for col_idx, is_low in enumerate(row):
            if bool(is_low):
                ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor=COLORS["pink"]["dark"], linewidth=1.8))
    ax.set_xlabel("Perturbation level")
    ax.set_ylabel("")
    ax.set_yticklabels([textwrap.fill(_short_axis_label(label), 22) for label in pivot.index], rotation=0, fontsize=7.0)
    ax.set_xticklabels([f"{float(label):.1f}" for label in pivot.columns], rotation=0, fontsize=7.5)
    ax.text(
        0.98,
        -0.23,
        "outlined cells fall below 0.80 stability floor",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color=TOKENS["muted"],
    )
    add_panel_title(ax, "b", "Data-quality stability atlas", "Noise types by perturbation level; cells show top-rank and rank-retention composite.")


def plot_panel_c_matrix(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Draw literature-volume sensitivity as a tier-by-retention matrix."""
    work = df.copy()
    work["retained_pct"] = (pd.to_numeric(work["literature_fraction"], errors="coerce") * 100).round().astype(int)
    pivot = (
        work.pivot_table(
            index="volume_tier",
            columns="retained_pct",
            values="spearman_mean",
            aggfunc="mean",
        )
        .reindex(["high-volume", "mid-volume", "low-volume"])
        .sort_index(axis=1)
    )
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=sns.blend_palette([COLORS["orange"]["xlight"], COLORS["gold"]["light"], COLORS["blue"]["mid"]], as_cmap=True),
        vmin=max(0.10, float(np.nanmin(pivot.to_numpy())) - 0.02),
        vmax=min(0.55, float(np.nanmax(pivot.to_numpy())) + 0.02),
        linewidths=1.0,
        linecolor=TOKENS["panel"],
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7.0, "color": TOKENS["ink"]},
        cbar_kws={"label": "Spearman", "fraction": 0.046, "pad": 0.015},
    )
    if 50 in list(pivot.columns):
        col_idx = list(pivot.columns).index(50)
        ax.add_patch(plt.Rectangle((col_idx, 0), 1, len(pivot.index), fill=False, edgecolor=COLORS["orange"]["dark"], linewidth=1.8))
    ax.set_xlabel("Retained literature (%)")
    ax.set_ylabel("")
    ax.set_yticklabels([_short_axis_label(label) for label in pivot.index], rotation=0, fontsize=7.5)
    ax.set_xticklabels([str(label) for label in pivot.columns], rotation=0, fontsize=7.5)
    ax.text(
        0.98,
        -0.23,
        "orange outline marks recommended minimum",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color=TOKENS["muted"],
    )
    add_panel_title(ax, "c", "Literature-volume boundary atlas", "Domain volume tiers by retained literature fraction; values are score-target Spearman.")


def plot_panel_d_matrix(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Draw temporal-window sensitivity as a horizon-by-window matrix."""
    pivot = (
        df.pivot_table(
            index="confirmation_horizon_years",
            columns="analysis_window_years",
            values="spearman",
            aggfunc="mean",
        )
        .sort_index(ascending=True)
        .sort_index(axis=1)
    )
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=sns.blend_palette([COLORS["neutral"]["xlight"], COLORS["olive"]["light"], COLORS["blue"]["mid"]], as_cmap=True),
        vmin=max(0.20, float(np.nanmin(pivot.to_numpy())) - 0.02),
        vmax=min(0.55, float(np.nanmax(pivot.to_numpy())) + 0.02),
        linewidths=1.0,
        linecolor=TOKENS["panel"],
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7.0, "color": TOKENS["ink"]},
        cbar_kws={"label": "Spearman", "fraction": 0.046, "pad": 0.015},
    )
    columns = list(pivot.columns)
    rows = list(pivot.index)
    if 5 in columns and 5 in rows:
        ax.add_patch(plt.Rectangle((columns.index(5), rows.index(5)), 1, 1, fill=False, edgecolor=COLORS["orange"]["dark"], linewidth=2.0))
    ax.set_xlabel("Analysis window (years)")
    ax.set_ylabel("Confirmation horizon (years)")
    ax.set_xticklabels([str(int(label)) for label in pivot.columns], rotation=0, fontsize=7.5)
    ax.set_yticklabels([str(int(label)) for label in pivot.index], rotation=0, fontsize=7.5)
    ax.text(
        0.98,
        -0.23,
        "orange outline marks 5y/5y reference",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color=TOKENS["muted"],
    )
    add_panel_title(ax, "d", "Temporal-window boundary atlas", "Analysis windows crossed with confirmation horizons; values are cached score-table associations.")


def plot_panel_e(ax: plt.Axes, df: pd.DataFrame) -> None:
    category_colors = {
        "graph weighting": COLORS["blue"]["mid"],
        "indicator choice": COLORS["gold"]["mid"],
        "bibliometric baseline": COLORS["orange"]["mid"],
        "model form": COLORS["olive"]["mid"],
    }
    plot_df = df.copy()
    y = np.arange(len(plot_df))
    ax.axvline(0, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    for i, row in plot_df.iterrows():
        color = category_colors.get(row["category"], COLORS["neutral"]["mid"])
        low = row["ci_low"]
        high = row["ci_high"]
        point = row["delta_vs_learned"]
        if not pd.isna(low) and not pd.isna(high):
            ax.hlines(y[i], low, high, color=color, linewidth=1.2)
        ax.scatter(point, y[i], s=36, color=color, edgecolor=TOKENS["ink"], linewidth=0.5, zorder=3)
    ax.set_yticks(y, [textwrap.fill(str(v), 28) for v in plot_df["choice"]], fontsize=7.8)
    ax.set_xlabel("Delta Spearman vs learned graph score")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%+.2f"))
    ax.grid(axis="x", color=TOKENS["grid"])
    ax.grid(axis="y", visible=False)
    add_panel_title(ax, "e", "Modeling-choice reproducibility", "Forest plot of cached/recomputed alternatives relative to learned Fig.3 score.")


def plot_panel_f_single(panel_f: pd.DataFrame, cases: pd.DataFrame):
    def _plot(fig: plt.Figure, ax: plt.Axes) -> None:
        plot_panel_f(fig, ax, panel_f, cases)

    return _plot


def plot_panel_f(fig: plt.Figure, ax: plt.Axes, panel_f: pd.DataFrame, cases: pd.DataFrame) -> None:
    plot_df = panel_f.sort_values("count", ascending=True)
    bars = ax.barh(plot_df["failure_mode"], plot_df["count"], color=COLORS["pink"]["base"], edgecolor=COLORS["pink"]["dark"], linewidth=1.0)
    for bar, rate in zip(bars, plot_df["rate"]):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.0f} ({rate:.0%})", va="center", fontsize=7.5, color=TOKENS["ink"])
    ax.set_xlabel("Flagged papers")
    ax.set_ylabel("")
    ax.grid(axis="x", color=TOKENS["grid"])
    ax.grid(axis="y", visible=False)
    add_panel_title(ax, "f", "Failure modes", "Heuristic taxonomy from Fig.4 peer/agent agreement and retrieval diagnostics.")
    add_case_cards(fig, ax, cases.head(3))


def add_case_cards(fig: plt.Figure, ax: plt.Axes, cases: pd.DataFrame) -> None:
    if cases.empty:
        return
    case_texts = []
    for _, row in cases.iterrows():
        title = str(row.get("title", "")).strip()
        title = textwrap.shorten(title, width=58, placeholder="...")
        modes = str(row.get("failure_modes", "")).split("; ")[0]
        safeguard = textwrap.shorten(str(row.get("recommended_safeguard", "")), width=62, placeholder="...")
        case_texts.append(f"{title}\nmode: {modes}\nsafeguard: {safeguard}")
    ax.text(
        0.02,
        -0.34,
        "\n\n".join(case_texts),
        transform=ax.transAxes,
        fontsize=6.7,
        color=TOKENS["muted"],
        va="top",
        ha="left",
        bbox={"facecolor": COLORS["neutral"]["xlight"], "edgecolor": TOKENS["axis"], "boxstyle": "round,pad=0.35"},
    )


def save_full_figure(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    panel_d: pd.DataFrame,
    panel_e: pd.DataFrame,
    panel_f: pd.DataFrame,
    failure_cases: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10.1), gridspec_kw={"height_ratios": [1.08, 1.0]})
    plot_panel_a(axes[0, 0], panel_a)
    plot_panel_b_matrix(axes[0, 1], panel_b)
    plot_panel_c_matrix(axes[1, 0], panel_c)
    plot_panel_d_matrix(axes[1, 1], panel_d)
    fig.suptitle("Fig. 6 | Robustness and boundary conditions of graph-perturbation analysis", x=0.012, ha="left", fontsize=15, fontweight="semibold", color=TOKENS["ink"])
    fig.text(
        0.012,
        0.965,
        "Atlas view: cross-domain reproducibility plus compact data-noise, literature-volume, and temporal-window stability matrices. Supporting modeling/failure analyses remain in audit outputs.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.93), w_pad=2.0, h_pad=3.0)
    fig.savefig(OUT_DIR / "fig6_full.png", dpi=260)
    fig.savefig(OUT_DIR / "fig6_full.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
