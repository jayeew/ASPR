from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
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

DOMAIN_OOF = EVIDENCE_DIR / "fig3aware12_subset_domain_oof_diagnostics.csv"
DOMAIN_COVERAGE = EVIDENCE_DIR / "fig3_score_coverage_by_domain.csv"
FIG5_DOMAIN_SUMMARY = EVIDENCE_DIR / "fig3aware12_fig5_domain_summary_minrefs1.csv"

FIG4_METRICS = FIG4_DIR / "fig4_metrics_summary.csv"
FIG4_RETRIEVAL = FIG4_DIR / "fig4_retrieval_diagnostics.csv"
FIG4_MANIFEST = FIG4_DIR / "fig4_manifest.csv"

GRAPH_METRICS = ["B", "RS", "DeltaQ0", "Uzzi", "RTD", "BurtIP", "PDE"]
GRAPH_Z_COLS = [f"{metric}_z" for metric in GRAPH_METRICS]
RANDOM_SEEDS = list(range(20260600, 20260620))

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
    source_specs = build_source_specs()

    panel_a = build_cross_domain_panel()
    panel_b = build_data_quality_panel(score, weights)
    panel_c = build_volume_panel(score)
    panel_d = build_temporal_panel(score)
    panel_e = build_modeling_choice_panel(score, weights)
    panel_f, failure_cases = build_failure_modes_panel()
    panel_review = build_panel_review(panel_a, panel_b, panel_c, panel_d, panel_e, panel_f)

    write_tables(panel_a, panel_b, panel_c, panel_d, panel_e, panel_f, failure_cases)
    write_panel_review(panel_review)
    write_metadata(source_specs, panel_review, panel_a, panel_b, panel_c, panel_d, panel_e, panel_f)
    write_caption(panel_review, panel_a, panel_b, panel_c, panel_d, panel_e, panel_f)

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
        return pd.DataFrame(
            [{
                "domain": "pipeline_ready",
                "source_status": "pipeline_ready_missing_domain_oof",
                "panel_note": "Domain OOF diagnostics were unavailable.",
            }]
        )

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
    failure_cases: pd.DataFrame,
) -> None:
    panel_a.to_csv(OUT_DIR / "fig6_cross_domain_reproducibility.csv", index=False)
    panel_b.to_csv(OUT_DIR / "fig6_data_quality_perturbation.csv", index=False)
    panel_c.to_csv(OUT_DIR / "fig6_volume_sensitivity.csv", index=False)
    panel_d.to_csv(OUT_DIR / "fig6_temporal_window_sensitivity.csv", index=False)
    panel_e.to_csv(OUT_DIR / "fig6_modeling_choice_reproducibility.csv", index=False)
    panel_f.to_csv(OUT_DIR / "fig6_failure_modes.csv", index=False)
    failure_cases.to_csv(OUT_DIR / "fig6_failure_mode_cases.csv", index=False)


def build_panel_review(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    panel_d: pd.DataFrame,
    panel_e: pd.DataFrame,
    panel_f: pd.DataFrame,
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
            "visual_form": "multi-line perturbation retention curve",
            "n_rows": len(panel_b),
            "keep_decision": "keep",
            "strength": "strong_with_proxy_label",
            "redundancy_assessment": "unique: data-noise boundary condition",
            "rationale": "Required core panel; revised from heatmap to stability curves so degradation shape and failure thresholds are visible.",
        },
        {
            "panel": "C",
            "title": "Literature-volume sensitivity",
            "role": "core",
            "visual_form": "sensitivity curve",
            "n_rows": len(panel_c),
            "keep_decision": "keep",
            "strength": "strong_with_proxy_label",
            "redundancy_assessment": "unique: literature scale boundary condition",
            "rationale": "Required core panel; downsampled score rows expose low-volume instability separately from data-noise effects.",
        },
        {
            "panel": "D",
            "title": "Temporal-window sensitivity",
            "role": "core",
            "visual_form": "horizon trajectories over analysis-window length",
            "n_rows": len(panel_d),
            "keep_decision": "keep",
            "strength": "strong_with_proxy_label",
            "redundancy_assessment": "unique: analysis-window and confirmation-horizon boundary condition",
            "rationale": "Required core panel; revised from heatmap to horizon trajectories to avoid another table-like matrix.",
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
    ]
    return pd.DataFrame(rows)


def write_panel_review(panel_review: pd.DataFrame) -> None:
    panel_review.to_csv(OUT_DIR / "fig6_panel_review.csv", index=False)
    (OUT_DIR / "fig6_panel_review.json").write_text(panel_review.to_json(orient="records", indent=2, force_ascii=False))


def write_metadata(
    sources: List[SourceSpec],
    panel_review: pd.DataFrame,
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    panel_d: pd.DataFrame,
    panel_e: pd.DataFrame,
    panel_f: pd.DataFrame,
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
        },
        "method_notes": [
            "No online data fetching was performed.",
            "Panels B-D are deterministic score-table perturbation/slicing analyses over cached graph metrics.",
            "Pipeline-ready or proxy statuses are preserved in CSV source_status columns.",
            "Failure taxonomy is heuristic and derived from Fig.4 cached peer/agent disagreement diagnostics.",
        ],
        "iteration_summary": {
            "round_1": "Generated coarse robustness panels from local Fig.1-Fig.5 caches and score tables.",
            "round_2": "Reviewed panel redundancy and strength; A/B/D were too table-like and E/F were visually sparse as main panels.",
            "round_3": "Revised main figure to four varied core panels; moved E/F to supporting audit outputs and caption notes.",
            "displayed_panels": ["A", "B", "C", "D"],
            "supporting_panels": ["E", "F"],
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
    for label, df in zip(["A", "B", "C", "D", "E", "F"], panels):
        counts = summarize_status(df)["source_status_counts"]
        status_bits.append(f"Panel {label}: {counts}")
    review_bits = [
        f"Panel {row.panel}: {row.keep_decision} ({row.strength}) - {row.redundancy_assessment}"
        for row in panel_review.itertuples(index=False)
    ]
    caption = f"""# Fig. 6 | Robustness and boundary conditions of graph-perturbation analysis

The figure uses local Fig.1-Fig.5 / score-table / works-topics / Fig.4 cached data only.
The main figure uses four visually distinct panels: Panel A is a cross-domain bubble-lollipop ranking, Panel B is a perturbation retention curve, Panel C is a literature-volume sensitivity curve, and Panel D is a temporal-window trajectory plot.
Panel A merges domain-level Fig.3 OOF diagnostics, score coverage, and Fig.5 backtest summaries.
Panels B-D are pipeline-ready robustness probes computed from cached score-table graph metrics with fixed seeds; they do not rerun OpenAlex retrieval or graph extraction.
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
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 11.2))
    plot_panel_a(axes[0, 0], panel_a)
    plot_panel_b(axes[0, 1], panel_b)
    plot_panel_c(axes[1, 0], panel_c)
    plot_panel_d(axes[1, 1], panel_d)
    fig.suptitle("Fig. 6 | Robustness and boundary conditions of graph-perturbation analysis", x=0.012, ha="left", fontsize=15, fontweight="semibold", color=TOKENS["ink"])
    fig.text(
        0.012,
        0.965,
        "Four-panel closure view: cross-domain reproducibility, data-noise stability, literature-volume sensitivity, and temporal-window boundaries. Supporting modeling/failure analyses remain in audit outputs.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.94), w_pad=2.2, h_pad=3.0)
    fig.savefig(OUT_DIR / "fig6_full.png", dpi=260)
    fig.savefig(OUT_DIR / "fig6_full.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
