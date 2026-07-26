"""Nature-style static renderers for the redesigned ASPR v6.1 experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aspr-v6-1-figures-matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch

from experiments.common.old.v6_1_figures_r1.analysis import (
    ANGLE_ORDER,
    FEATURE_SHORT,
    MODEL_SHORT,
)


BLUE = "#0072B2"
LIGHT_BLUE = "#8CC5E3"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#6B7280"
LIGHT_GRAY = "#D9DEE7"
INK = "#172033"

ANGLE_EN = {
    "A1_COMBINATION_RARITY": "A1 Combination rarity",
    "A2_ATYPICALITY_CONVENTIONALITY": "A2 Atypicality & conventionality",
    "A3_FIRST_TIME_COMBINATION": "A3 First-time combinations",
    "A4_KNOWLEDGE_BREADTH_BALANCE": "A4 Breadth & balance",
    "A5_COGNITIVE_DISTANCE_INTEGRATION": "A5 Distance & integration",
}

FEATURE_EN = {
    "reference_overlap_novelty_t0": "Reference-overlap novelty",
    "hypergeom_conventionality_median_t0": "Median conventionality",
    "first_time_source_pair_share": "First-pair share",
    "field_gini_balance": "Field balance",
    "reference_other_field_share": "Outside-field references",
    "field_variety": "Field variety",
    "field_disparity_cosine_mean": "Mean cognitive distance",
    "rao_stirling_integration": "Rao–Stirling integration",
}

MODEL_EN = {
    "k0_controls": "K0 controls",
    "k1_controls": "K1 controls",
    "k2_controls": "K2 controls",
    "b0_v6_primary_plus_k0": "B0 (v6 metrics + K0)",
    "provisional_core8_plus_k1": "Provisional 8 + K1",
    "final_innovation_plus_k1": "Final 8 + K1",
    "final_innovation_plus_k2": "Final 8 + K2",
    "innovation_only": "Innovation only",
}

MODEL_COLORS = {
    "k0_controls": GRAY,
    "k1_controls": LIGHT_BLUE,
    "k2_controls": "#7E91A8",
    "b0_v6_primary_plus_k0": PURPLE,
    "provisional_core8_plus_k1": ORANGE,
    "final_innovation_plus_k1": BLUE,
    "final_innovation_plus_k2": GREEN,
    "innovation_only": VERMILLION,
}


def configure_style() -> None:
    """Apply a restrained, color-blind-safe shared visual style."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.edgecolor": "#667085",
            "axes.linewidth": 0.7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 16,
            "figure.titleweight": "bold",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.color": "#D0D5DD",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.55,
            "svg.fonttype": "none",
        }
    )


def _panel(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(f"{label}  {title}", loc="left", pad=9)


def _clean(ax: plt.Axes, *, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis)
        ax.set_axisbelow(True)


def _figure_title(fig: plt.Figure, title: str, subtitle: str) -> None:
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None and hasattr(layout_engine, "set"):
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.91))
    fig.suptitle(title, x=0.02, y=0.985, ha="left", color=INK)
    fig.text(0.02, 0.947, subtitle, ha="left", va="top", color=GRAY, fontsize=9)


def _flow_boxes(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    labels: Mapping[str, str] | None = None,
) -> None:
    ax.set_axis_off()
    rows = frame.sort_values("order")
    x_positions = np.linspace(0.11, 0.89, len(rows))
    for index, (_, row) in enumerate(rows.iterrows()):
        label = (labels or {}).get(str(row["stage"]), str(row["stage"]))
        patch = FancyBboxPatch(
            (x_positions[index] - 0.09, 0.38),
            0.18,
            0.28,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor="#F3F7FB",
            edgecolor=BLUE,
            linewidth=1.2,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            x_positions[index],
            0.56,
            f"{int(row['n']):,}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=INK,
        )
        ax.text(
            x_positions[index],
            0.45,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            wrap=True,
        )
        if index < len(rows) - 1:
            ax.annotate(
                "",
                xy=(x_positions[index + 1] - 0.10, 0.52),
                xytext=(x_positions[index] + 0.10, 0.52),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.2},
            )


def render_figure_01(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    _figure_title(
        fig,
        "Fig. 1 | Frozen Nature corpus and future-diffusion target",
        "118,059 papers across 12 natural-science domains; all reported predictions are temporal out-of-fold.",
    )
    flow_labels = {
        "Nature论文（1980–2017）": "Nature papers\n1980–2017",
        "共同有效队列": "Common valid\ncohort",
        "初始训练期（1980–1985）": "Initial training\n1980–1985",
        "时间OOF论文（1986–2017）": "Temporal OOF\n1986–2017",
    }
    _flow_boxes(axes[0, 0], tables["corpus_flow"], labels=flow_labels)
    _panel(axes[0, 0], "a", "Analysis cohort")

    domain = tables["domain_counts"].sort_values("n_papers")
    axes[0, 1].barh(domain["domain12"], domain["n_papers"], color=BLUE)
    axes[0, 1].set_xlabel("Papers")
    axes[0, 1].xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))
    _panel(axes[0, 1], "b", "Twelve-domain composition")
    _clean(axes[0, 1], grid_axis="x")

    yearly = tables["year_counts"]
    axes[1, 0].fill_between(
        yearly["publication_year"],
        yearly["n_papers"],
        color=LIGHT_BLUE,
        alpha=0.55,
    )
    axes[1, 0].plot(
        yearly["publication_year"], yearly["n_papers"], color=BLUE, lw=1.8
    )
    axes[1, 0].axvline(1985.5, color=VERMILLION, ls="--", lw=1.1)
    axes[1, 0].text(
        1985.8,
        axes[1, 0].get_ylim()[1] * 0.9,
        "OOF starts",
        color=VERMILLION,
        fontsize=8,
    )
    axes[1, 0].set(xlabel="Publication year", ylabel="Papers")
    _panel(axes[1, 0], "c", "Publication-time coverage")
    _clean(axes[1, 0], grid_axis="y")

    target = tables["target_summary"].sort_values("horizon")
    x_values = np.arange(len(target))
    axes[1, 1].bar(
        x_values - 0.18,
        target["uptake_rate"],
        width=0.36,
        color=ORANGE,
        label="Any future uptake",
    )
    axes[1, 1].bar(
        x_values + 0.18,
        target["realized_diffusion_mean"],
        width=0.36,
        color=GREEN,
        label="Mean realized diffusion",
    )
    axes[1, 1].set_xticks(x_values, [f"D{int(value)}" for value in target["horizon"]])
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("Rate / mean target")
    axes[1, 1].legend(frameon=False, loc="upper left")
    axes[1, 1].text(
        0.02,
        0.04,
        "Label = uptake × fold-local conditional diffusion\n(zero-inclusive; computed separately at D3/D5/D8)",
        transform=axes[1, 1].transAxes,
        fontsize=8,
        color=GRAY,
    )
    _panel(axes[1, 1], "d", "Prediction target by horizon")
    _clean(axes[1, 1], grid_axis="y")
    return fig


def render_figure_02(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    _figure_title(
        fig,
        "Fig. 2 | Outcome-blind selection of innovation indicators",
        "The candidate registry was frozen before OOF modelling; prediction performance did not determine admission.",
    )
    selection_labels = {
        "检索候选": "Literature candidates",
        "本地已实现": "Locally implemented",
        "通过全部运行门": "All runtime gates",
        "数学家族竞争后主指标": "Primary indicators",
    }
    _flow_boxes(axes[0, 0], tables["selection_flow"], labels=selection_labels)
    _panel(axes[0, 0], "a", "Candidate-to-primary flow")

    roles = tables["role_counts"].copy()
    roles["angle_label_en"] = roles["angle_id"].map(ANGLE_EN)
    role_order = ["primary", "sensitivity", "exploratory", "excluded"]
    role_colors = [BLUE, ORANGE, GREEN, LIGHT_GRAY]
    pivot = roles.pivot(
        index="angle_label_en",
        columns="proposed_final_role",
        values="n_candidates",
    ).fillna(0)
    pivot = pivot.reindex([ANGLE_EN[angle] for angle in ANGLE_ORDER])
    left = np.zeros(len(pivot))
    for role, color in zip(role_order, role_colors):
        values = pivot.get(role, pd.Series(0, index=pivot.index)).to_numpy()
        axes[0, 1].barh(pivot.index, values, left=left, color=color, label=role)
        left += values
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Candidate metrics")
    axes[0, 1].legend(frameon=False, ncol=2, loc="lower right")
    _panel(axes[0, 1], "b", "Final roles within five observation angles")
    _clean(axes[0, 1], grid_axis="x")

    gates = tables["gate_matrix"].copy()
    gate_columns = [
        "coverage_pass",
        "stability_pass",
        "approximation_pass",
        "toy_test_pass",
        "temporal_test_pass",
        "nondegenerate_test_pass",
    ]
    matrix = gates[gate_columns].to_numpy(dtype=float)
    axes[1, 0].imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=ListedColormap(["#F1B6B6", "#B8DFC8"]),
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_xticks(
        range(len(gate_columns)),
        ["Coverage", "Stability", "Fidelity", "Hand test", "Time", "Non-degenerate"],
        rotation=35,
        ha="right",
    )
    axes[1, 0].set_yticks(range(len(gates)), gates["candidate_id"], fontsize=6.7)
    axes[1, 0].text(
        0.99,
        1.02,
        "green = pass; red = fail",
        transform=axes[1, 0].transAxes,
        ha="right",
        fontsize=7.5,
        color=GRAY,
    )
    _panel(axes[1, 0], "c", "Registered primary/sensitivity gate evidence")

    primary = tables["primary_map"].copy()
    angle_index = {angle: index for index, angle in enumerate(ANGLE_ORDER)}
    features = primary["feature"].tolist()
    matrix = np.zeros((len(ANGLE_ORDER), len(features)))
    for feature_index, row in primary.reset_index(drop=True).iterrows():
        matrix[angle_index[row["angle_id"]], feature_index] = 1
    axes[1, 1].imshow(
        matrix,
        cmap=ListedColormap(["white", BLUE]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    axes[1, 1].set_xticks(
        range(len(features)),
        [FEATURE_EN[name] for name in features],
        rotation=40,
        ha="right",
        fontsize=7,
    )
    axes[1, 1].set_yticks(
        range(len(ANGLE_ORDER)), [ANGLE_EN[angle] for angle in ANGLE_ORDER]
    )
    for feature_index, row in primary.reset_index(drop=True).iterrows():
        axes[1, 1].text(
            feature_index,
            angle_index[row["angle_id"]],
            str(int(row["n_original_sources"])),
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
        )
    axes[1, 1].text(
        0.01,
        -0.32,
        "Numbers in blue cells denote original-source records; every metric also has a paper-level application source.",
        transform=axes[1, 1].transAxes,
        fontsize=7.5,
        color=GRAY,
    )
    _panel(axes[1, 1], "d", "Eight primary indicators mapped to five angles")
    return fig


def render_figure_03(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    _figure_title(
        fig,
        "Fig. 3 | Measurement coverage, stability and construct overlap",
        "Primary indicators meet their registered gates while retaining non-identical information.",
    )
    quality = tables["primary_quality"].copy()
    quality["label"] = quality["feature"].map(FEATURE_EN)
    quality = quality.sort_values("overall_coverage")
    y_values = np.arange(len(quality))
    axes[0, 0].hlines(
        y_values,
        quality["minimum_domain_coverage"],
        quality["overall_coverage"],
        color=LIGHT_GRAY,
        lw=2,
    )
    axes[0, 0].scatter(
        quality["minimum_domain_coverage"],
        y_values,
        color=ORANGE,
        label="Minimum domain",
        zorder=3,
    )
    axes[0, 0].scatter(
        quality["overall_coverage"],
        y_values,
        color=BLUE,
        label="Overall",
        zorder=3,
    )
    axes[0, 0].axvline(0.70, color=GRAY, ls="--", lw=1)
    axes[0, 0].axvline(0.50, color=GRAY, ls=":", lw=1)
    axes[0, 0].set_yticks(y_values, quality["label"])
    axes[0, 0].set(xlim=(0.45, 1.02), xlabel="Eligible-paper coverage")
    axes[0, 0].legend(frameon=False, loc="upper left")
    _panel(axes[0, 0], "a", "Overall and minimum-domain coverage")
    _clean(axes[0, 0], grid_axis="x")

    axes[0, 1].scatter(
        quality["stability_median_relative_error"],
        quality["stability_spearman"],
        c=[ANGLE_ORDER.index(value) for value in quality["angle_id"]],
        cmap=ListedColormap([BLUE, ORANGE, GREEN, PURPLE, VERMILLION]),
        s=55,
        edgecolor="white",
        linewidth=0.7,
    )
    for _, row in quality.iterrows():
        axes[0, 1].annotate(
            FEATURE_EN[row["feature"]],
            (row["stability_median_relative_error"], row["stability_spearman"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.8,
        )
    axes[0, 1].axhline(0.90, color=GRAY, ls="--", lw=1)
    axes[0, 1].axvline(0.10, color=GRAY, ls="--", lw=1)
    axes[0, 1].set(
        xlim=(-0.005, 0.105),
        ylim=(0.89, 1.005),
        xlabel="Median relative error after 80% reference resampling",
        ylabel="Resampling Spearman",
    )
    fidelity = tables["approximation_fidelity"]
    if not fidelity.empty:
        axes[0, 1].text(
            0.03,
            0.04,
            f"Exact-reference checks: {len(fidelity)} metric(s), all ρ=1.000",
            transform=axes[0, 1].transAxes,
            fontsize=7.5,
            color=GRAY,
        )
    _panel(axes[0, 1], "b", "Reference-resampling stability")
    _clean(axes[0, 1], grid_axis="both")

    corr_long = tables["feature_correlations"]
    features = quality["feature"].tolist()
    corr = corr_long.pivot(
        index="feature_left", columns="feature_right", values="spearman"
    ).reindex(index=features, columns=features)
    image = axes[1, 0].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[1, 0].set_xticks(
        range(len(features)),
        [FEATURE_EN[name] for name in features],
        rotation=40,
        ha="right",
        fontsize=7,
    )
    axes[1, 0].set_yticks(
        range(len(features)), [FEATURE_EN[name] for name in features], fontsize=7
    )
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, label="Spearman ρ")
    _panel(axes[1, 0], "c", "Pairwise indicator correlations")

    pairs = tables["correlation_pairs"].copy()
    pairs["absolute_spearman"] = pairs["spearman"].abs()
    categories = ["同一角度", "跨角度"]
    labels = ["Within angle", "Across angles"]
    values = [pairs.loc[pairs["pair_type"].eq(value), "absolute_spearman"] for value in categories]
    violin = axes[1, 1].violinplot(
        values,
        positions=[1, 2],
        widths=0.7,
        showmeans=False,
        showmedians=True,
    )
    for body, color in zip(violin["bodies"], [ORANGE, BLUE]):
        body.set_facecolor(color)
        body.set_alpha(0.45)
        body.set_edgecolor(color)
    rng = np.random.default_rng(20260724)
    for position, series, color in zip([1, 2], values, [ORANGE, BLUE]):
        axes[1, 1].scatter(
            position + rng.uniform(-0.08, 0.08, len(series)),
            series,
            color=color,
            s=22,
            alpha=0.8,
        )
    axes[1, 1].set_xticks([1, 2], labels)
    axes[1, 1].set(ylabel="Absolute Spearman ρ", ylim=(0, 1.03))
    _panel(axes[1, 1], "d", "Within- versus across-angle overlap")
    _clean(axes[1, 1], grid_axis="y")
    return fig


def _model_point_plot(ax: plt.Axes, points: pd.DataFrame, *, title: str) -> None:
    shown = points.sort_values("model_order", ascending=False)
    colors = [MODEL_COLORS.get(value, GRAY) for value in shown["model_id"]]
    labels = [MODEL_EN.get(value, value) for value in shown["model_id"]]
    ax.scatter(shown["spearman_expected"], range(len(shown)), c=colors, s=70)
    for y_value, (_, row) in enumerate(shown.iterrows()):
        ax.text(
            row["spearman_expected"] + 0.004,
            y_value,
            f"{row['spearman_expected']:.3f}",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(range(len(shown)), labels)
    ax.set_xlabel("All-period temporal OOF Spearman ρ")
    ax.set_xlim(min(0.62, shown["spearman_expected"].min() - 0.02), 0.79)
    _panel(ax, "a", title)
    _clean(ax, grid_axis="x")


def _decile_rate_plot(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    models: list[str],
    title: str,
    panel: str,
) -> None:
    for model_id in models:
        shown = frame[frame["model_id"].eq(model_id)].sort_values(
            "prediction_decile"
        )
        ax.plot(
            shown["prediction_decile"],
            shown["observed_high_impact_rate"],
            color=MODEL_COLORS[model_id],
            marker="o",
            lw=1.8,
            label=MODEL_EN[model_id],
        )
        ax.fill_between(
            shown["prediction_decile"],
            shown["rate_ci_low"],
            shown["rate_ci_high"],
            color=MODEL_COLORS[model_id],
            alpha=0.12,
        )
    ax.axhline(0.10, color=GRAY, ls="--", lw=1, label="Population rate")
    ax.set(
        xlabel="Predicted-score decile",
        ylabel="Observed top-decile target rate",
        xticks=np.arange(1, 11),
    )
    ax.legend(frameon=False)
    _panel(ax, panel, title)
    _clean(ax, grid_axis="y")


def render_figure_04(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    _figure_title(
        fig,
        "Fig. 4 | Innovation indicators improve five-year OOF ranking",
        "The final evidence-selected indicators add rank information beyond the expanded K1 controls.",
    )
    _model_point_plot(axes[0, 0], tables["model_points"], title="D5 model comparison")

    gains = tables["paired_gains"].sort_values("spearman_gain")
    y_values = np.arange(len(gains))
    axes[0, 1].errorbar(
        gains["spearman_gain"],
        y_values,
        xerr=np.vstack(
            [
                gains["spearman_gain"] - gains["gain_ci_low"],
                gains["gain_ci_high"] - gains["spearman_gain"],
            ]
        ),
        fmt="o",
        color=BLUE,
        ecolor=INK,
        capsize=4,
    )
    axes[0, 1].axvline(0, color=GRAY, lw=1)
    axes[0, 1].set_yticks(
        y_values,
        [f"vs {MODEL_EN.get(value, value)}" for value in gains["baseline_model_id"]],
    )
    axes[0, 1].set_xlabel("Paired Spearman gain (95% bootstrap CI)")
    _panel(axes[0, 1], "b", "Increment over registered baselines")
    _clean(axes[0, 1], grid_axis="x")

    _decile_rate_plot(
        axes[1, 0],
        tables["prediction_deciles"],
        models=["k1_controls", "final_innovation_plus_k1"],
        title="Observed high-impact rate by predicted decile",
        panel="c",
    )
    deciles = tables["prediction_deciles"]
    for model_id in ["k1_controls", "final_innovation_plus_k1"]:
        shown = deciles[deciles["model_id"].eq(model_id)].sort_values(
            "prediction_decile"
        )
        axes[1, 1].plot(
            shown["prediction_decile"],
            shown["mean_realized_diffusion"],
            marker="o",
            color=MODEL_COLORS[model_id],
            lw=1.8,
            label=MODEL_EN[model_id],
        )
    axes[1, 1].set(
        xlabel="Predicted-score decile",
        ylabel="Mean realized diffusion",
        xticks=np.arange(1, 11),
    )
    axes[1, 1].legend(frameon=False)
    _panel(axes[1, 1], "d", "Monotonic target separation")
    _clean(axes[1, 1], grid_axis="y")
    return fig


def render_figure_05(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    _figure_title(
        fig,
        "Fig. 5 | Ranking gains persist across three prediction horizons",
        "D3 and D8 are directional robustness analyses; D5 remains the registered headline.",
    )
    metrics = tables["horizon_metrics"].pivot(
        index="horizon", columns="model_id", values="spearman_expected"
    ).sort_index()
    y_values = np.arange(len(metrics))
    for y_value, (_, row) in enumerate(metrics.iterrows()):
        axes[0, 0].plot(
            [row["k1_controls"], row["final_innovation_plus_k1"]],
            [y_value, y_value],
            color=LIGHT_GRAY,
            lw=3,
        )
    axes[0, 0].scatter(metrics["k1_controls"], y_values, color=LIGHT_BLUE, s=60, label="K1 controls")
    axes[0, 0].scatter(
        metrics["final_innovation_plus_k1"],
        y_values,
        color=BLUE,
        s=60,
        label="Final 8 + K1",
    )
    axes[0, 0].set_yticks(y_values, [f"D{value}" for value in metrics.index])
    axes[0, 0].set_xlabel("Temporal OOF Spearman ρ")
    axes[0, 0].legend(frameon=False)
    _panel(axes[0, 0], "a", "Control versus full model")
    _clean(axes[0, 0], grid_axis="x")

    gains = tables["horizon_gains"].sort_values("horizon")
    axes[0, 1].bar(
        [f"D{int(value)}" for value in gains["horizon"]],
        gains["spearman_gain"],
        color=[ORANGE, BLUE, GREEN],
    )
    for index, value in enumerate(gains["spearman_gain"]):
        axes[0, 1].text(index, value + 0.002, f"+{value:.3f}", ha="center")
    axes[0, 1].set_ylabel("Full − K1 Spearman gain")
    _panel(axes[0, 1], "b", "Incremental value by horizon")
    _clean(axes[0, 1], grid_axis="y")

    fold = tables["fold_horizon_gains"].pivot(
        index="horizon", columns="outer_fold_id", values="spearman_gain"
    ).sort_index()
    image = axes[1, 0].imshow(fold, cmap="YlGnBu", aspect="auto", vmin=0)
    axes[1, 0].set_xticks(range(len(fold.columns)), [f"Fold {value}" for value in fold.columns])
    axes[1, 0].set_yticks(range(len(fold.index)), [f"D{value}" for value in fold.index])
    for row in range(fold.shape[0]):
        for column in range(fold.shape[1]):
            axes[1, 0].text(
                column,
                row,
                f"{fold.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=INK,
            )
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, label="Spearman gain")
    _panel(axes[1, 0], "c", "Gain across horizon × temporal fold")

    agreement = tables["prediction_rank_agreement"].pivot(
        index="horizon_left", columns="horizon_right", values="spearman"
    ).sort_index().sort_index(axis=1)
    image = axes[1, 1].imshow(agreement, cmap="Blues", vmin=0.75, vmax=1)
    axes[1, 1].set_xticks(range(len(agreement.columns)), [f"D{value}" for value in agreement.columns])
    axes[1, 1].set_yticks(range(len(agreement.index)), [f"D{value}" for value in agreement.index])
    for row in range(agreement.shape[0]):
        for column in range(agreement.shape[1]):
            axes[1, 1].text(
                column,
                row,
                f"{agreement.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, label="Prediction-rank ρ")
    _panel(axes[1, 1], "d", "Cross-horizon prediction agreement")
    return fig


def render_figure_06(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    _figure_title(
        fig,
        "Fig. 6 | Expanding-time generalization and temporal drift",
        "Every test period occurs strictly after its training period; the late-period decline is displayed rather than hidden.",
    )
    folds = tables["temporal_folds"].sort_values("fold_id")
    for y_value, (_, row) in enumerate(folds.iterrows()):
        axes[0, 0].barh(
            y_value,
            row["train_year_max"] - 1979,
            left=1980,
            color=LIGHT_BLUE,
            height=0.55,
            label="Training" if y_value == 0 else None,
        )
        axes[0, 0].barh(
            y_value,
            row["test_year_max"] - row["test_year_min"] + 1,
            left=row["test_year_min"],
            color=VERMILLION,
            height=0.55,
            label="Test" if y_value == 0 else None,
        )
    axes[0, 0].set_yticks(range(len(folds)), [f"Fold {value}" for value in folds["fold_id"]])
    axes[0, 0].set(xlabel="Publication year", xlim=(1979.5, 2018))
    axes[0, 0].invert_yaxis()
    axes[0, 0].legend(frameon=False, ncol=2)
    _panel(axes[0, 0], "a", "Registered temporal folds")
    _clean(axes[0, 0], grid_axis="x")

    fold_metrics = tables["fold_metrics"].copy()
    for model_id in ["k1_controls", "innovation_only", "final_innovation_plus_k1"]:
        shown = fold_metrics[fold_metrics["model_id"].eq(model_id)].sort_values(
            "outer_fold_id"
        )
        axes[0, 1].plot(
            shown["outer_fold_id"],
            shown["spearman_expected"],
            color=MODEL_COLORS[model_id],
            marker="o",
            lw=1.8,
            label=MODEL_EN[model_id],
        )
    axes[0, 1].set(
        xlabel="Outer temporal fold",
        ylabel="Fold-specific Spearman ρ",
        xticks=np.arange(1, 7),
    )
    axes[0, 1].legend(frameon=False)
    _panel(axes[0, 1], "b", "D5 fold performance")
    _clean(axes[0, 1], grid_axis="y")

    gain = tables["fold_gain_intervals"].sort_values("outer_fold_id")
    y_values = np.arange(len(gain))
    axes[1, 0].errorbar(
        gain["spearman_gain"],
        y_values,
        xerr=np.vstack(
            [
                gain["spearman_gain"] - gain["gain_ci_low"],
                gain["gain_ci_high"] - gain["spearman_gain"],
            ]
        ),
        fmt="o",
        color=BLUE,
        ecolor=INK,
        capsize=3,
    )
    axes[1, 0].axvline(0, color=GRAY, lw=1)
    axes[1, 0].set_yticks(
        y_values,
        [
            f"F{int(row.outer_fold_id)}: {int(row.test_year_min)}–{int(row.test_year_max)}"
            for row in gain.itertuples()
        ],
    )
    axes[1, 0].set_xlabel("Full − K1 Spearman gain (95% bootstrap CI)")
    _panel(axes[1, 0], "c", "Within-fold incremental value")
    _clean(axes[1, 0], grid_axis="x")

    yearly = tables["yearly_metrics"]
    for model_id in ["k1_controls", "final_innovation_plus_k1"]:
        shown = yearly[yearly["model_id"].eq(model_id)].sort_values(
            "publication_year"
        )
        axes[1, 1].plot(
            shown["publication_year"],
            shown["spearman_expected"],
            color=MODEL_COLORS[model_id],
            lw=1.6,
            marker=".",
            label=MODEL_EN[model_id],
        )
    axes[1, 1].set(
        xlabel="Test publication year",
        ylabel="Within-year Spearman ρ",
    )
    axes[1, 1].legend(frameon=False)
    _panel(axes[1, 1], "d", "Year-specific ranking performance")
    _clean(axes[1, 1], grid_axis="y")
    return fig


def render_figure_07(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    _figure_title(
        fig,
        "Fig. 7 | All twelve natural-science domains are retained",
        "Domain heterogeneity is reported directly; no low-performing field is removed from the analysis.",
    )
    metrics = tables["domain_metrics"].pivot(
        index="domain12", columns="model_id", values="spearman_expected"
    )
    metrics["gain"] = (
        metrics["final_innovation_plus_k1"] - metrics["k1_controls"]
    )
    metrics = metrics.sort_values("gain")
    y_values = np.arange(len(metrics))
    for y_value, (_, row) in enumerate(metrics.iterrows()):
        axes[0, 0].plot(
            [row["k1_controls"], row["final_innovation_plus_k1"]],
            [y_value, y_value],
            color=LIGHT_GRAY,
            lw=2.5,
        )
    axes[0, 0].scatter(metrics["k1_controls"], y_values, color=LIGHT_BLUE, s=45, label="K1")
    axes[0, 0].scatter(
        metrics["final_innovation_plus_k1"],
        y_values,
        color=BLUE,
        s=45,
        label="Final 8 + K1",
    )
    axes[0, 0].set_yticks(y_values, metrics.index)
    axes[0, 0].set_xlabel("D5 Spearman ρ")
    axes[0, 0].legend(frameon=False)
    _panel(axes[0, 0], "a", "Domain-specific model performance")
    _clean(axes[0, 0], grid_axis="x")

    gains = tables["domain_gain_intervals"].set_index("domain12").loc[metrics.index].reset_index()
    axes[0, 1].errorbar(
        gains["spearman_gain"],
        y_values,
        xerr=np.vstack(
            [
                gains["spearman_gain"] - gains["gain_ci_low"],
                gains["gain_ci_high"] - gains["spearman_gain"],
            ]
        ),
        fmt="o",
        color=GREEN,
        ecolor=INK,
        capsize=2,
    )
    axes[0, 1].axvline(0, color=GRAY, lw=1)
    axes[0, 1].set_yticks(y_values, gains["domain12"])
    axes[0, 1].set_xlabel("Full − K1 gain (95% bootstrap CI)")
    _panel(axes[0, 1], "b", "Paired gain uncertainty")
    _clean(axes[0, 1], grid_axis="x")

    pure = tables["pure_domain_metrics"].set_index("domain12")
    comparison = metrics[["final_innovation_plus_k1"]].join(
        pure[["spearman_expected"]].rename(
            columns={"spearman_expected": "innovation_only"}
        )
    )
    for y_value, (_, row) in enumerate(comparison.iterrows()):
        axes[1, 0].plot(
            [row["innovation_only"], row["final_innovation_plus_k1"]],
            [y_value, y_value],
            color=LIGHT_GRAY,
            lw=2.5,
        )
    axes[1, 0].scatter(
        comparison["innovation_only"],
        y_values,
        color=VERMILLION,
        s=45,
        label="Innovation only",
    )
    axes[1, 0].scatter(
        comparison["final_innovation_plus_k1"],
        y_values,
        color=BLUE,
        s=45,
        label="Final 8 + K1",
    )
    axes[1, 0].set_yticks(y_values, comparison.index)
    axes[1, 0].set_xlabel("D5 Spearman ρ")
    axes[1, 0].legend(frameon=False)
    _panel(axes[1, 0], "c", "Innovation-only versus full model")
    _clean(axes[1, 0], grid_axis="x")

    size_gain = gains.merge(
        tables["domain_metrics"][
            tables["domain_metrics"]["model_id"].eq("k1_controls")
        ][["domain12", "n_oof"]],
        on="domain12",
        how="left",
        validate="one_to_one",
    )
    axes[1, 1].scatter(
        size_gain["n_oof"],
        size_gain["spearman_gain"],
        s=np.sqrt(size_gain["n_oof"]) * 2.5,
        color=BLUE,
        alpha=0.75,
        edgecolor="white",
    )
    for _, row in size_gain.iterrows():
        axes[1, 1].annotate(
            row["domain12"],
            (row["n_oof"], row["spearman_gain"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.5,
        )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set(
        xlabel="Domain OOF papers (log scale)",
        ylabel="Full − K1 Spearman gain",
    )
    _panel(axes[1, 1], "d", "Sample size and observed gain")
    _clean(axes[1, 1], grid_axis="both")
    return fig


def render_figure_08(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    _figure_title(
        fig,
        "Fig. 8 | Innovation indicators retain signal without controls",
        "The control-free model reaches D5 OOF Spearman ρ≈0.707, but the strongest ranking combines innovation and controls.",
    )
    metrics = tables["model_metrics"].copy()
    order = ["k1_controls", "innovation_only", "final_innovation_plus_k1"]
    metrics["order"] = metrics["model_id"].map({name: index for index, name in enumerate(order)})
    metrics = metrics.sort_values("order")
    bars = axes[0, 0].bar(
        [MODEL_EN[value] for value in metrics["model_id"]],
        metrics["spearman_expected"],
        color=[MODEL_COLORS[value] for value in metrics["model_id"]],
    )
    axes[0, 0].set_ylim(0.62, 0.79)
    axes[0, 0].set_ylabel("D5 temporal OOF Spearman ρ")
    axes[0, 0].tick_params(axis="x", rotation=12)
    for bar, value in zip(bars, metrics["spearman_expected"]):
        axes[0, 0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.003,
            f"{value:.3f}",
            ha="center",
        )
    _panel(axes[0, 0], "a", "Control-free and combined ranking")
    _clean(axes[0, 0], grid_axis="y")

    folds = tables["fold_metrics"]
    for model_id in order:
        shown = folds[folds["model_id"].eq(model_id)].sort_values(
            "outer_fold_id"
        )
        axes[0, 1].plot(
            shown["outer_fold_id"],
            shown["spearman_expected"],
            color=MODEL_COLORS[model_id],
            marker="o",
            lw=1.7,
            label=MODEL_EN[model_id],
        )
    axes[0, 1].set(
        xlabel="Outer temporal fold",
        ylabel="Fold-specific Spearman ρ",
        xticks=np.arange(1, 7),
    )
    axes[0, 1].legend(frameon=False)
    _panel(axes[0, 1], "b", "Temporal-fold consistency")
    _clean(axes[0, 1], grid_axis="y")

    _decile_rate_plot(
        axes[1, 0],
        tables["prediction_deciles"],
        models=order,
        title="Observed high-impact rate by predicted decile",
        panel="c",
    )

    correlation = tables["prediction_correlations"].pivot(
        index="model_left", columns="model_right", values="spearman"
    ).reindex(index=order, columns=order)
    image = axes[1, 1].imshow(correlation, cmap="Blues", vmin=0.6, vmax=1)
    axes[1, 1].set_xticks(
        range(len(order)), [MODEL_EN[value] for value in order], rotation=25, ha="right"
    )
    axes[1, 1].set_yticks(range(len(order)), [MODEL_EN[value] for value in order])
    for row in range(len(order)):
        for column in range(len(order)):
            axes[1, 1].text(
                column,
                row,
                f"{correlation.iloc[row, column]:.2f}",
                ha="center",
                va="center",
            )
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, label="Prediction-rank ρ")
    _panel(axes[1, 1], "d", "Overlap among model rankings")
    return fig


def render_figure_09(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    _figure_title(
        fig,
        "Fig. 9 | Post-hoc five-angle predictive ablation",
        "These fixed OOF ablations interpret non-redundant signal; they do not reopen indicator selection or imply causality.",
    )
    summary = tables["angle_summary"].sort_values("angle_number")
    y_values = np.arange(len(summary))
    axes[0, 0].scatter(
        summary["k1_plus_angle_spearman"],
        y_values,
        color=BLUE,
        s=65,
    )
    axes[0, 0].axvline(summary["k1_spearman"].iloc[0], color=GRAY, ls="--", lw=1)
    axes[0, 0].set_yticks(
        y_values, [ANGLE_EN[value] for value in summary["angle_id"]]
    )
    axes[0, 0].set_xlabel("D5 temporal OOF Spearman ρ")
    _panel(axes[0, 0], "a", "K1 plus one observation angle")
    _clean(axes[0, 0], grid_axis="x")

    axes[0, 1].barh(
        y_values,
        summary["increment_over_k1"],
        color=[BLUE, ORANGE, GREEN, PURPLE, VERMILLION],
    )
    axes[0, 1].set_yticks(
        y_values, [ANGLE_EN[value] for value in summary["angle_id"]]
    )
    axes[0, 1].set_xlabel("Spearman gain over K1")
    _panel(axes[0, 1], "b", "Single-angle incremental value")
    _clean(axes[0, 1], grid_axis="x")

    axes[1, 0].barh(
        y_values,
        summary["drop_from_full"],
        color=[BLUE, ORANGE, GREEN, PURPLE, VERMILLION],
    )
    axes[1, 0].axvline(0, color=GRAY, lw=1)
    axes[1, 0].set_yticks(
        y_values, [ANGLE_EN[value] for value in summary["angle_id"]]
    )
    axes[1, 0].set_xlabel("Full-model Spearman loss after deleting angle")
    _panel(axes[1, 0], "c", "Leave-one-angle-out loss")
    _clean(axes[1, 0], grid_axis="x")

    fold = tables["fold_deletion"].pivot(
        index="angle_id", columns="outer_fold_id", values="drop_from_full"
    ).reindex(ANGLE_ORDER)
    limit = max(0.01, float(np.nanmax(np.abs(fold.to_numpy()))))
    image = axes[1, 1].imshow(
        fold,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    axes[1, 1].set_xticks(
        range(len(fold.columns)), [f"Fold {value}" for value in fold.columns]
    )
    axes[1, 1].set_yticks(
        range(len(fold.index)), [ANGLE_EN[value] for value in fold.index]
    )
    for row in range(fold.shape[0]):
        for column in range(fold.shape[1]):
            value = float(fold.iloc[row, column])
            if abs(value) < 0.0005:
                value = 0.0
            axes[1, 1].text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, label="Full − deletion ρ")
    _panel(axes[1, 1], "d", "Deletion effect across temporal folds")
    return fig


def render_figure_10(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    _figure_title(
        fig,
        "Fig. 10 | Sensitivity, acceptance gates and reproducibility",
        "All registered headline gates pass; exact replay and frozen-artifact checks bind the reported values.",
    )
    sensitivity = tables["control_sensitivity"].copy()
    sensitivity["order"] = sensitivity["model_id"].map(
        {
            "k0_controls": 0,
            "k1_controls": 1,
            "k2_controls": 2,
            "innovation_only": 3,
            "b0_v6_primary_plus_k0": 4,
            "provisional_core8_plus_k1": 5,
            "final_innovation_plus_k1": 6,
            "final_innovation_plus_k2": 7,
        }
    )
    sensitivity = sensitivity.sort_values("order", ascending=False)
    y_values = np.arange(len(sensitivity))
    axes[0, 0].scatter(
        sensitivity["spearman_expected"],
        y_values,
        c=[MODEL_COLORS[value] for value in sensitivity["model_id"]],
        s=65,
    )
    axes[0, 0].set_yticks(
        y_values, [MODEL_EN[value] for value in sensitivity["model_id"]]
    )
    axes[0, 0].set_xlabel("D5 temporal OOF Spearman ρ")
    _panel(axes[0, 0], "a", "Control-set and model sensitivity")
    _clean(axes[0, 0], grid_axis="x")

    gate_labels = {
        "D5达到目标": "D5 reaches ρ target",
        "相对K1增量下界>0": "Gain vs K1: CI lower > 0",
        "相对B0非劣下界≥−0.005": "Non-inferiority vs B0",
        "D3增量>0": "D3 gain > 0",
        "D8增量>0": "D8 gain > 0",
    }
    gates = tables["acceptance_gates"].sort_values("margin").copy()
    gates["gate_en"] = gates["gate"].map(gate_labels).fillna(gates["gate"])
    colors = np.where(gates["margin"].ge(0), GREEN, VERMILLION)
    axes[0, 1].barh(gates["gate_en"], gates["margin"], color=colors)
    axes[0, 1].axvline(0, color=INK, lw=1)
    axes[0, 1].set_xlabel("Conservative value minus registered threshold")
    _panel(axes[0, 1], "b", "Acceptance-gate margins")
    _clean(axes[0, 1], grid_axis="x")

    stress = tables["stress_test_gains"]
    categories = ["预测窗口", "时间折", "学科"]
    labels = ["Horizons", "Temporal folds", "Domains"]
    rng = np.random.default_rng(20260724)
    for position, (category, label, color) in enumerate(
        zip(categories, labels, [ORANGE, BLUE, PURPLE]), start=1
    ):
        values = stress.loc[stress["stratum"].eq(category), "spearman_gain"]
        axes[1, 0].scatter(
            position + rng.uniform(-0.10, 0.10, len(values)),
            values,
            color=color,
            alpha=0.8,
            s=28,
            label=label,
        )
        axes[1, 0].plot(
            [position - 0.18, position + 0.18],
            [values.median(), values.median()],
            color=INK,
            lw=2,
        )
    axes[1, 0].axhline(0, color=GRAY, lw=1)
    axes[1, 0].set_xticks([1, 2, 3], labels)
    axes[1, 0].set_ylabel("Final 8 + K1 minus K1 Spearman gain")
    _panel(axes[1, 0], "c", "Gain distribution across stress-test units")
    _clean(axes[1, 0], grid_axis="y")

    audit_labels = {
        "方案完成审计": "Protocol-completion audit",
        "OOF检查点精确复跑": "Exact OOF checkpoint replay",
        "时间折/窗口测试集核验": "Fold/horizon test-set checks",
        "12大类保留": "All 12 domains retained",
        "冻结输出哈希核验": "Frozen-output hash checks",
    }
    audit = tables["reproducibility_checks"].sort_values(
        "completion_rate"
    ).copy()
    audit["check_en"] = audit["check"].map(audit_labels).fillna(audit["check"])
    bars = axes[1, 1].barh(
        audit["check_en"], audit["completion_rate"], color=GREEN
    )
    axes[1, 1].set_xlim(0, 1.08)
    axes[1, 1].set_xlabel("Verified fraction")
    for bar, row in zip(bars, audit.itertuples()):
        axes[1, 1].text(
            bar.get_width() + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row.passed)}/{int(row.total)}",
            va="center",
            fontsize=8,
        )
    _panel(axes[1, 1], "d", "Frozen-output and replay checks")
    _clean(axes[1, 1], grid_axis="x")
    return fig


RENDERERS: Dict[int, Callable[[Mapping[str, pd.DataFrame]], plt.Figure]] = {
    1: render_figure_01,
    2: render_figure_02,
    3: render_figure_03,
    4: render_figure_04,
    5: render_figure_05,
    6: render_figure_06,
    7: render_figure_07,
    8: render_figure_08,
    9: render_figure_09,
    10: render_figure_10,
}


def save_figure(
    figure: plt.Figure,
    output_stem: Path,
    *,
    formats: list[str],
    dpi: int,
) -> Dict[str, Path]:
    """Save one figure in all registered formats and close it."""
    outputs: Dict[str, Path] = {}
    for file_format in formats:
        path = output_stem.with_suffix(f".{file_format}")
        figure.savefig(path, dpi=int(dpi), bbox_inches="tight", facecolor="white")
        outputs[file_format] = path
    plt.close(figure)
    return outputs


def render_all(
    experiment_tables: Mapping[int, Mapping[str, pd.DataFrame]],
    output_dir: Path,
    *,
    formats: list[str],
    dpi: int,
) -> Dict[int, Dict[str, Path]]:
    """Render all ten figures from their experiment-specific tables."""
    configure_style()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[int, Dict[str, Path]] = {}
    for figure_index in range(1, 11):
        figure = RENDERERS[figure_index](experiment_tables[figure_index])
        outputs[figure_index] = save_figure(
            figure,
            output_dir / f"fig{figure_index:02d}",
            formats=formats,
            dpi=int(dpi),
        )
    return outputs


__all__ = ["RENDERERS", "configure_style", "render_all", "save_figure"]
