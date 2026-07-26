"""Deterministic renderers for Fig.6–Fig.10."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde

from experiments.common.new.base.builders_6_10 import QUALITY_METRICS
from experiments.common.new.base.common import (
    ANGLE_ORDER,
    ANGLE_SHORT,
    BLUE,
    FEATURE_LABELS,
    GRAY,
    INK,
    LIGHT_BLUE,
    LIGHT_GRAY,
    LIGHT_ORANGE,
    MID_GRAY,
    OLIVE,
    ORANGE,
    PALE_GRAY,
    PINK,
    PURPLE,
    VERMILLION,
    WHITE,
    FigureBundle,
    clean_axes,
    configure_style,
    draft_panel,
    figure_title,
    stable_seed,
)
from experiments.common.new.base.renderers_1_5 import (
    _arrow,
    _box,
    _finish_composite,
    _panel_outputs,
    _short,
)


def _add_background(fig: Figure, path: Path, alpha: float = 0.10) -> None:
    """Place one generated no-text asset behind deterministic overlays."""
    image = plt.imread(path)
    background = fig.add_axes([0, 0, 1, 1], zorder=-100)
    background.imshow(image, aspect="auto", alpha=alpha)
    background.set_axis_off()


def _polar_xy(
    values: np.ndarray,
    *,
    center: Tuple[float, float],
    radius: float,
    start: float = np.pi / 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a vector to closed polar coordinates in axis space."""
    theta = start + np.linspace(0, 2 * np.pi, len(values), endpoint=False)
    x = center[0] + radius * values * np.cos(theta)
    y = center[1] + radius * values * np.sin(theta)
    return np.r_[x, x[0]], np.r_[y, y[0]], theta


# ============================================================================
# Fig.6
# ============================================================================


def _draw_fig6a(ax: Axes, bundle: FigureBundle) -> None:
    """Draw current and legacy robustness contours without combining their evidence."""
    data = bundle.tables["robustness_compass"]
    panel_title = "a  Observed robustness retention compass"
    ax.set_title(panel_title, loc="left", pad=8, color=INK, fontweight="bold")
    ax.set_axis_off()
    current = data.loc[data["evidence_scope"].eq("registered_v6_1")].copy()
    legacy = data.loc[data["evidence_scope"].eq("legacy_proxy")].copy()
    axes = current["axis"].tolist()
    axis_labels = {
        "80% reference resampling": "80% reference\nresampling",
        "Exact replay checks": "Exact replay\nchecks",
        "Positive fold gain": "Positive\nfold gain",
        "Weakest-field retention": "Weakest-field\nretention",
        "K2 control retention": "K2-control\nretention",
        "D3/D8 horizon retention": "D3/D8-horizon\nretention",
    }
    center = (0.50, 0.48)
    for ring in (0.6, 0.8, 1.0):
        x, y, _ = _polar_xy(
            np.full(len(axes), ring),
            center=center,
            radius=0.37,
        )
        ax.plot(x, y, transform=ax.transAxes, color=LIGHT_GRAY, linewidth=0.7)
        ax.text(
            center[0],
            center[1] + 0.37 * ring,
            f"{ring:.1f}",
            transform=ax.transAxes,
            fontsize=5,
            color=GRAY,
            ha="center",
            va="bottom",
        )
    current_values = current.set_index("axis").loc[axes, "value"].to_numpy(float)
    legacy_values = legacy.set_index("axis").loc[axes, "value"].to_numpy(float)
    for index, axis_name in enumerate(axes):
        direction = np.array(
            [
                math.cos(np.pi / 2 + index * 2 * np.pi / len(axes)),
                math.sin(np.pi / 2 + index * 2 * np.pi / len(axes)),
            ]
        )
        ax.plot(
            [center[0], center[0] + 0.38 * direction[0]],
            [center[1], center[1] + 0.38 * direction[1]],
            transform=ax.transAxes,
            color=PALE_GRAY,
            linewidth=0.7,
        )
        label = center + 0.46 * direction
        ax.text(
            label[0],
            label[1],
            axis_labels.get(axis_name, textwrap.fill(axis_name, 15)),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.7,
            color=INK,
        )
    current_x, current_y, _ = _polar_xy(current_values, center=center, radius=0.37)
    legacy_x, legacy_y, _ = _polar_xy(legacy_values, center=center, radius=0.37)
    ax.plot(current_x, current_y, transform=ax.transAxes, color=BLUE, linewidth=2.0)
    ax.fill(current_x, current_y, transform=ax.transAxes, color=LIGHT_BLUE, alpha=0.22)
    ax.plot(
        legacy_x,
        legacy_y,
        transform=ax.transAxes,
        color=ORANGE,
        linewidth=1.2,
        linestyle="--",
    )
    handles = [
        Line2D([0], [0], color=BLUE, lw=2, label="Registered v6.1"),
        Line2D([0], [0], color=ORANGE, lw=1.2, ls="--", label="Legacy proxy"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center", ncol=2, fontsize=6)
    ax.text(
        0.5,
        0.01,
        "Radii are observed retention/check rates, not inferred maximum tolerances.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=VERMILLION,
    )


def _draw_fig6b(ax: Axes, bundle: FigureBundle) -> None:
    """Draw registered 80% reference-resampling distributions."""
    data = bundle.tables["registered_reference_resampling"].copy()
    data = data.loc[data["reference_retention"].eq(0.8)]
    order = (
        data.groupby("code_name")["spearman"]
        .median()
        .sort_values()
        .index.tolist()
    )
    ax.set_title("b  Registered 80% reference-resampling stability", loc="left", pad=8, color=INK, fontweight="bold")
    sns.violinplot(
        data=data,
        x="spearman",
        y="code_name",
        order=order,
        orient="h",
        inner=None,
        cut=0,
        density_norm="width",
        color=LIGHT_BLUE,
        linewidth=0.5,
        ax=ax,
    )
    sns.stripplot(
        data=data,
        x="spearman",
        y="code_name",
        order=order,
        orient="h",
        size=2.0,
        color=BLUE,
        alpha=0.45,
        jitter=0.16,
        ax=ax,
    )
    ax.axvline(0.90, color=VERMILLION, linestyle="--", linewidth=0.9)
    ax.set_yticks(
        range(len(order)),
        [textwrap.fill(FEATURE_LABELS.get(code, code.replace("_", " ")), 22) for code in order],
    )
    ax.set_xlabel("Spearman(full references, 80% references), 20 repetitions")
    ax.set_ylabel("")
    ax.set_xlim(min(0.88, float(data["spearman"].min()) - 0.01), 1.005)
    clean_axes(ax, grid_axis="x")
    ax.text(
        0.99,
        0.02,
        "Only the frozen 80% stress level exists; deeper deletion doses are not imputed.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.6,
        color=GRAY,
    )


def _draw_fig6c(ax: Axes, bundle: FigureBundle) -> None:
    """Draw a specification curve with a dot-matrix contract."""
    estimates = bundle.tables["specification_curve"].copy()
    flags = bundle.tables["specification_flags"].copy()
    ax.set_title("c  Modeling-choice specification curve", loc="left", pad=8, color=INK, fontweight="bold")
    ax.set_axis_off()
    upper = ax.inset_axes([0.05, 0.40, 0.92, 0.50])
    lower = ax.inset_axes([0.05, 0.04, 0.92, 0.29], sharex=upper)
    current = estimates.loc[estimates["scope"].eq("current_v6_1")].sort_values(
        "spearman",
        ascending=False,
    )
    legacy = estimates.loc[estimates["scope"].eq("legacy_proxy")].sort_values(
        "spearman",
        ascending=False,
    )
    ordered = pd.concat([current, legacy], ignore_index=True)
    ordered["x"] = np.arange(len(ordered))
    ordered["display_id"] = [
        f"C{index + 1}" if row.scope == "current_v6_1" else f"L{index + 1 - len(current)}"
        for index, row in ordered.iterrows()
    ]
    current_count = len(current)
    upper.plot(
        ordered.loc[: current_count - 1, "x"],
        ordered.loc[: current_count - 1, "spearman"],
        color=LIGHT_BLUE,
        linewidth=1.0,
    )
    for row in ordered.itertuples(index=False):
        is_main = row.specification == "final_innovation_plus_k1"
        upper.scatter(
            row.x,
            row.spearman,
            s=42 if is_main else 22,
            color=ORANGE if is_main else BLUE if row.scope == "current_v6_1" else MID_GRAY,
            marker="o" if row.scope == "current_v6_1" else "D",
            edgecolor=WHITE,
            linewidth=0.4,
            zorder=3,
        )
    upper.axvline(current_count - 0.5, color=LIGHT_GRAY, linestyle=":", linewidth=1)
    upper.text(
        current_count - 0.45,
        upper.get_ylim()[1],
        " legacy cached-score proxies →",
        ha="left",
        va="top",
        fontsize=5.3,
        color=GRAY,
    )
    upper.set_ylabel("Spearman")
    upper.set_xticks([])
    clean_axes(upper, grid_axis="y")
    main = ordered.loc[ordered["specification"].eq("final_innovation_plus_k1")]
    if not main.empty:
        upper.annotate(
            "C1 main",
            (float(main.iloc[0]["x"]), float(main.iloc[0]["spearman"])),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=5.2,
            color=ORANGE,
            fontweight="bold",
        )
    choices = flags["choice"].drop_duplicates().tolist()
    x_map = ordered.set_index("specification")["x"].to_dict()
    for row_index, choice in enumerate(choices):
        subset = flags.loc[flags["choice"].eq(choice)]
        for row in subset.itertuples(index=False):
            if row.specification not in x_map:
                continue
            lower.scatter(
                x_map[row.specification],
                row_index,
                s=18,
                facecolor=INK if int(row.enabled) else WHITE,
                edgecolor=INK,
                linewidth=0.6,
            )
    lower.set_yticks(range(len(choices)), [_short(choice, 24) for choice in choices])
    lower.set_xticks(
        ordered["x"],
        ordered["display_id"],
        rotation=0,
        ha="center",
        fontsize=5.2,
    )
    lower.set_xlim(-0.6, len(ordered) - 0.4)
    lower.invert_yaxis()
    for spine in lower.spines.values():
        spine.set_visible(False)
    lower.tick_params(length=0)
    ax.text(
        0.05,
        0.005,
        "C = current v6.1 specification; L = legacy cached-score proxy. Exact names are retained in panel_data.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.9,
        color=GRAY,
    )


def _draw_fig6d(ax: Axes, bundle: FigureBundle) -> None:
    """Draw the empirical reliability boundary across domain-year units."""
    data = bundle.tables["reliability_units"]
    ax.set_title("d  Reliable and failed domain-year regions", loc="left", pad=8, color=INK, fontweight="bold")
    failed = data.loc[data["reliable"].eq(0)]
    passed = data.loc[data["reliable"].eq(1)]
    ax.scatter(
        failed["median_reference_count"],
        failed["median_mapping_coverage"],
        s=10 + 20 * np.sqrt(failed["n_papers"] / max(failed["n_papers"].max(), 1)),
        marker="x",
        color=VERMILLION,
        alpha=0.55,
        linewidth=0.65,
        label=f"failed, n={len(failed)}",
    )
    ax.scatter(
        passed["median_reference_count"],
        passed["median_mapping_coverage"],
        s=13 + 24 * np.sqrt(passed["n_papers"] / max(passed["n_papers"].max(), 1)),
        facecolor=WHITE,
        edgecolor=BLUE,
        linewidth=0.9,
        label=f"passed, n={len(passed)}",
    )
    ax.axvline(10, color=ORANGE, linestyle="--", linewidth=0.9)
    ax.axhline(0.8, color=ORANGE, linestyle="--", linewidth=0.9)
    ax.set_xlabel("Median valid references")
    ax.set_ylabel("Median source/field mapping coverage")
    x_max = float(data["median_reference_count"].quantile(0.995)) + 2
    ax.set_xlim(-1, x_max)
    ax.set_ylim(-0.02, 1.02)
    ax.fill_between(
        [10, x_max],
        [0.8, 0.8],
        [1.02, 1.02],
        color=LIGHT_BLUE,
        alpha=0.10,
        zorder=-1,
    )
    ax.legend(frameon=False, fontsize=6)
    clean_axes(ax, grid_axis="both")
    ax.text(
        0.99,
        0.02,
        "Registered gate: ≥10 valid references and ≥0.80 mapping coverage",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.6,
        color=GRAY,
    )


def _draw_fig6e(ax: Axes, bundle: FigureBundle) -> None:
    """Draw diagnostic failure frequencies and three safeguard cards."""
    modes = bundle.tables["failure_modes"].sort_values("count", ascending=True)
    cases = bundle.tables["failure_cases"].head(3)
    ax.set_title("e  Heuristic failure taxonomy and safeguards", loc="left", pad=8, color=INK, fontweight="bold")
    ax.set_axis_off()
    left = ax.inset_axes([0.00, 0.08, 0.48, 0.82])
    y = np.arange(len(modes))
    left.hlines(y, 0, modes["rate"], color=LIGHT_BLUE, linewidth=2.0)
    left.scatter(modes["rate"], y, s=32, color=BLUE)
    left.set_yticks(y, [_short(value, 22) for value in modes["failure_mode"]])
    left.set_xlabel("Heuristic rate in 50 cached cases")
    left.set_xlim(0, 1.08)
    clean_axes(left, grid_axis="x")
    for index, row in enumerate(cases.itertuples(index=False)):
        bottom = 0.67 - index * 0.29
        ax.add_patch(
            FancyBboxPatch(
                (0.54, bottom),
                0.44,
                0.23,
                boxstyle="round,pad=0.012,rounding_size=0.014",
                transform=ax.transAxes,
                facecolor=WHITE,
                edgecolor=LIGHT_ORANGE,
                linewidth=0.9,
            )
        )
        ax.text(
            0.56,
            bottom + 0.18,
            _short(row.title, 52),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color=INK,
            fontweight="bold",
        )
        ax.text(
            0.56,
            bottom + 0.095,
            f"{row.failure_modes}\nSafeguard: {_short(row.recommended_safeguard, 52)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.2,
            color=GRAY,
        )
    ax.text(
        0.99,
        0.01,
        "Diagnostic heuristics; no blinded manual adjudication",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=VERMILLION,
        fontsize=5.8,
    )


def render_fig6(
    bundle: FigureBundle,
    figure_dir: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render Fig.6 and its standalone panels."""
    configure_style()
    drawers = {
        "a": _draw_fig6a,
        "b": _draw_fig6b,
        "c": _draw_fig6c,
        "d": _draw_fig6d,
        "e": _draw_fig6e,
    }
    outputs = _panel_outputs(bundle, drawers, figure_dir, dpi=dpi)
    fig = plt.figure(figsize=(16.5, 11.8))
    grid = fig.add_gridspec(2, 6, hspace=0.32, wspace=0.52)
    axes = {
        "a": fig.add_subplot(grid[0, :2]),
        "b": fig.add_subplot(grid[0, 2:4]),
        "c": fig.add_subplot(grid[0, 4:]),
        "d": fig.add_subplot(grid[1, :3]),
        "e": fig.add_subplot(grid[1, 3:]),
    }
    for key, ax in axes.items():
        drawers[key](ax, bundle)
    figure_title(
        fig,
        "Fig. 6 | Robustness, reliability boundaries and failure modes",
        "Registered v6.1 evidence is separated from legacy graph proxies; unrun stress levels are never interpolated.",
    )
    outputs.update(_finish_composite(fig, bundle, figure_dir, formats, dpi))
    return outputs


# ============================================================================
# Fig.7
# ============================================================================


VENUE_COLORS = {
    "Nature flagship": ORANGE,
    "Nature Communications": BLUE,
    "Scientific Reports": OLIVE,
    "Nature specialist journals": PURPLE,
    "npj series": PINK,
    "Communications series": VERMILLION,
    "Other": MID_GRAY,
}

VENUE_DISPLAY_LABELS = {
    "Nature Communications": "Nature\nCommunications",
    "Nature flagship": "Nature\nflagship",
    "Scientific Reports": "Scientific\nReports",
    "Nature specialist journals": "Nature specialist\njournals",
}


def _draw_fig7a(ax: Axes, bundle: FigureBundle) -> None:
    """Draw venue innovation/impact portfolios."""
    data = bundle.tables["venue_portfolio"]
    label_offsets = {
        "Nature Communications": (6, 7),
        "Nature flagship": (6, 5),
        "Scientific Reports": (6, 5),
        "Nature specialist journals": (6, 7),
    }
    ax.set_title("a  Venue portfolio map", loc="left", pad=8, color=INK, fontweight="bold")
    ax.axvline(0.5, color=LIGHT_GRAY, linewidth=0.7)
    ax.axhline(0.5, color=LIGHT_GRAY, linewidth=0.7)
    for row in data.itertuples(index=False):
        color = VENUE_COLORS.get(row.analysis_venue_family, MID_GRAY)
        size = 80 + 900 * math.sqrt(float(row.n_papers) / float(data["n_papers"].max()))
        ax.scatter(
            row.innovation_signal,
            row.future_diffusion,
            s=size,
            color=color,
            alpha=0.72,
            edgecolor=WHITE,
            linewidth=1.0,
        )
        ax.annotate(
            VENUE_DISPLAY_LABELS.get(
                row.analysis_venue_family,
                row.analysis_venue_family,
            ),
            (row.innovation_signal, row.future_diffusion),
            xytext=label_offsets.get(row.analysis_venue_family, (5, 4)),
            textcoords="offset points",
            fontsize=6.2,
            color=INK,
            linespacing=0.9,
        )
    ax.set_xlabel("Mean publication-time innovation-only percentile")
    ax.set_ylabel("Mean future D5 diffusion percentile")
    ax.set_xlim(min(0.42, float(data["innovation_signal"].min()) - 0.02), max(0.60, float(data["innovation_signal"].max()) + 0.025))
    ax.set_ylim(min(0.45, float(data["future_diffusion"].min()) - 0.02), max(0.61, float(data["future_diffusion"].max()) + 0.025))
    clean_axes(ax, grid_axis="both")
    ax.text(
        0.02,
        0.02,
        "Field-year normalized; bubble area ∝ paper volume",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color=GRAY,
    )


def _draw_fig7b(ax: Axes, bundle: FigureBundle) -> None:
    """Draw bootstrap venue rank distributions as ridgelines."""
    data = bundle.tables["venue_bootstrap_ranks"]
    ax.set_title("b  Bootstrap rank distributions", loc="left", pad=8, color=INK, fontweight="bold")
    families = (
        data.groupby("analysis_venue_family")["rank"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    max_rank = int(data["rank"].max())
    grid = np.linspace(0.6, max_rank + 0.4, 200)
    for index, family in enumerate(families):
        values = data.loc[data["analysis_venue_family"].eq(family), "rank"].to_numpy(float)
        jittered = values + np.random.default_rng(stable_seed(family, 20260725)).normal(0, 0.08, len(values))
        density = gaussian_kde(jittered, bw_method=0.22)(grid)
        density = density / max(float(density.max()), 1e-12) * 0.65
        ax.fill_between(
            grid,
            index,
            index + density,
            color=VENUE_COLORS.get(family, MID_GRAY),
            alpha=0.55,
        )
        ax.plot(grid, index + density, color=VENUE_COLORS.get(family, MID_GRAY), linewidth=1)
        ax.scatter(np.median(values), index, s=22, color=INK, zorder=3)
    ax.set_yticks(range(len(families)), [_short(value, 26) for value in families])
    ax.set_xticks(range(1, max_rank + 1))
    ax.set_xlabel("Bootstrap rank (1 = highest mean innovation-only signal)")
    ax.set_ylim(-0.2, len(families) - 0.05)
    clean_axes(ax, grid_axis="x")


def _draw_fig7c(ax: Axes, bundle: FigureBundle) -> None:
    """Draw top 1% and 5% enrichment intervals."""
    data = bundle.tables["venue_enrichment"].copy()
    ax.set_title("c  Enrichment of high innovation-signal papers", loc="left", pad=8, color=INK, fontweight="bold")
    families = data["analysis_venue_family"].drop_duplicates().tolist()
    offsets = {"Top 1%": -0.12, "Top 5%": 0.12}
    markers = {"Top 1%": "o", "Top 5%": "s"}
    for threshold, group in data.groupby("threshold"):
        y = np.array([families.index(value) for value in group["analysis_venue_family"]], dtype=float) + offsets[threshold]
        ax.errorbar(
            group["enrichment"],
            y,
            xerr=[
                group["enrichment"] - group["ci_low"],
                group["ci_high"] - group["enrichment"],
            ],
            fmt=markers[threshold],
            color=ORANGE if threshold == "Top 1%" else BLUE,
            capsize=2.5,
            markersize=4.5,
            linewidth=1.0,
            label=threshold,
        )
    ax.axvline(1.0, color=LIGHT_GRAY, linewidth=0.9)
    ax.set_yticks(range(len(families)), [_short(value, 28) for value in families])
    ax.set_xlabel("Observed / expected enrichment")
    ax.legend(frameon=False, fontsize=6)
    clean_axes(ax, grid_axis="x")
    ax.text(
        0.99,
        0.02,
        "Association only; no venue causal effect",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.7,
        color=VERMILLION,
    )


def _smooth_radar(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create a closed, periodic smooth radar curve."""
    count = len(values)
    theta = np.linspace(0, 2 * np.pi, count, endpoint=False)
    theta_closed = np.r_[theta, 2 * np.pi]
    values_closed = np.r_[values, values[0]]
    spline = CubicSpline(theta_closed, values_closed, bc_type="periodic")
    dense_theta = np.linspace(0, 2 * np.pi, 300)
    dense_values = np.clip(spline(dense_theta), 0, 1)
    return dense_theta, dense_values


def _draw_fig7d(ax: Axes, bundle: FigureBundle) -> None:
    """Draw one smooth five-angle radar per venue family."""
    data = bundle.tables["venue_angle_profiles"]
    ax.set_title("d  Five-angle mechanism profiles", loc="left", pad=8, color=INK, fontweight="bold")
    ax.set_axis_off()
    families = data["analysis_venue_family"].drop_duplicates().tolist()[:4]
    locations = [
        [0.00, 0.49, 0.48, 0.36],
        [0.52, 0.49, 0.48, 0.36],
        [0.00, 0.07, 0.48, 0.36],
        [0.52, 0.07, 0.48, 0.36],
    ]
    theta_axes = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    for family, location in zip(families, locations):
        polar = ax.inset_axes(location, projection="polar")
        group = data.loc[data["analysis_venue_family"].eq(family)].set_index("angle_id")
        values = group.reindex(ANGLE_ORDER)["mean_percentile"].to_numpy(float)
        theta, smooth = _smooth_radar(values)
        color = VENUE_COLORS.get(family, MID_GRAY)
        polar.plot(theta, smooth, color=color, linewidth=1.5)
        polar.fill(theta, smooth, color=color, alpha=0.18)
        polar.scatter(theta_axes, values, s=12, color=color, zorder=3)
        polar.set_ylim(0.35, 0.65)
        polar.set_xticks(theta_axes, [f"A{index + 1}" for index in range(5)], fontsize=5)
        polar.set_yticks([0.4, 0.5, 0.6], ["", "0.5", ""], fontsize=4)
        polar.grid(color=LIGHT_GRAY, linewidth=0.5)
        polar.spines["polar"].set_color(LIGHT_GRAY)
        polar.set_title(
            VENUE_DISPLAY_LABELS.get(family, family),
            fontsize=5.8,
            color=INK,
            pad=2,
            linespacing=0.85,
        )
    ax.text(
        0.5,
        0.0,
        "A1 rarity · A2 atypicality · A3 first-time · A4 breadth · A5 integration\n"
        "Same field-year percentile scale; shape is descriptive, not an area test.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.5,
        color=GRAY,
    )


def _draw_fig7e(ax: Axes, bundle: FigureBundle) -> None:
    """Draw a 100% river plot of top-decile innovation-paper shares."""
    data = bundle.tables["venue_time_flow"]
    ax.set_title("e  Where top-decile innovation papers were carried over time", loc="left", pad=8, color=INK, fontweight="bold")
    pivot = (
        data.pivot(index="decade", columns="flow_family", values="share")
        .fillna(0)
        .sort_index()
    )
    families = pivot.columns.tolist()
    ax.stackplot(
        pivot.index,
        [pivot[family].to_numpy(float) for family in families],
        labels=families,
        colors=[VENUE_COLORS.get(family, MID_GRAY) for family in families],
        alpha=0.76,
        linewidth=0.4,
        edgecolor=WHITE,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of top-decile papers")
    ax.set_xlabel("Five-year publication bin")
    ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="upper left")
    clean_axes(ax, grid_axis="y")


def render_fig7(
    bundle: FigureBundle,
    figure_dir: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render Fig.7 and its standalone panels."""
    configure_style()
    drawers = {
        "a": _draw_fig7a,
        "b": _draw_fig7b,
        "c": _draw_fig7c,
        "d": _draw_fig7d,
        "e": _draw_fig7e,
    }
    outputs = _panel_outputs(bundle, drawers, figure_dir, dpi=dpi)
    fig = plt.figure(figsize=(16.5, 11.5))
    grid = fig.add_gridspec(2, 6, hspace=0.32, wspace=0.52)
    axes = {
        "a": fig.add_subplot(grid[0, :2]),
        "b": fig.add_subplot(grid[0, 2:4]),
        "c": fig.add_subplot(grid[0, 4:]),
        "d": fig.add_subplot(grid[1, :3]),
        "e": fig.add_subplot(grid[1, 3:]),
    }
    for key, ax in axes.items():
        drawers[key](ax, bundle)
    figure_title(
        fig,
        "Fig. 7 | Venue portfolios under a venue-excluded innovation score",
        "All scores are innovation-only and field-year normalized; the local frozen corpus contains Nature Portfolio venues only.",
    )
    outputs.update(_finish_composite(fig, bundle, figure_dir, formats, dpi))
    return outputs


# ============================================================================
# Fig.8
# ============================================================================


def _draw_fig8a(ax: Axes, bundle: FigureBundle) -> None:
    """Draw the complete dual-lane ASPR architecture."""
    ax.set_axis_off()
    image_path = Path(bundle.chart_contract["background_asset"])
    ax.imshow(plt.imread(image_path), extent=[0, 1, 0, 1], aspect="auto", alpha=0.34)
    ax.text(
        0.02,
        0.96,
        "ASPR dual-path evidence-grounded review framework",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        color=INK,
        fontweight="bold",
    )
    _box(ax, (0.025, 0.41), 0.15, 0.19, "Input manuscript\n+ references", BLUE, fontsize=8)
    _box(ax, (0.235, 0.66), 0.17, 0.14, "Publication-prior\nevidence", BLUE, fontsize=7.5)
    _box(ax, (0.445, 0.66), 0.17, 0.14, "Five observation angles\n8 primary indicators", BLUE, fontsize=7.2)
    _box(ax, (0.655, 0.66), 0.15, 0.14, "Claim-level\nevidence packet", BLUE, fontsize=7.5)
    _box(ax, (0.34, 0.25), 0.22, 0.15, "ASPR-Qwen reviewer\nstrengths · weaknesses · questions", PURPLE, fontsize=7.2)
    _box(ax, (0.655, 0.33), 0.12, 0.14, "Fusion", ORANGE, fontsize=8)
    _box(ax, (0.805, 0.33), 0.12, 0.14, "Verifier", ORANGE, fontsize=8)
    _box(ax, (0.81, 0.62), 0.17, 0.24, "Evidence-grounded review\n\nnovelty stance\nprior-art comparison\nlimitations\nrecommendation\nevidence IDs", OLIVE, fontsize=7)
    _arrow(ax, (0.175, 0.53), (0.235, 0.73), BLUE)
    _arrow(ax, (0.405, 0.73), (0.445, 0.73), BLUE)
    _arrow(ax, (0.615, 0.73), (0.655, 0.73), BLUE)
    _arrow(ax, (0.175, 0.47), (0.34, 0.33), PURPLE)
    _arrow(ax, (0.56, 0.33), (0.655, 0.40), PURPLE)
    _arrow(ax, (0.73, 0.66), (0.705, 0.47), BLUE)
    _arrow(ax, (0.775, 0.40), (0.805, 0.40), ORANGE)
    _arrow(ax, (0.865, 0.47), (0.895, 0.62), ORANGE)
    indicators = bundle.tables["primary_indicators"]
    indicator_text = " · ".join(
        f"{index + 1} {row.indicator}"
        for index, row in enumerate(indicators.itertuples(index=False))
    )
    ax.text(
        0.50,
        0.08,
        textwrap.fill(indicator_text, 145),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.8,
        color=GRAY,
    )
    ax.text(
        0.98,
        0.02,
        "Architecture only — no numerical performance claim",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=VERMILLION,
        fontweight="bold",
    )


def render_fig8(
    bundle: FigureBundle,
    figure_dir: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render the single-canvas Fig.8 framework."""
    configure_style()
    drawers = {"a": _draw_fig8a}
    outputs = _panel_outputs(bundle, drawers, figure_dir, dpi=dpi)
    fig, ax = plt.subplots(figsize=(16, 8.8))
    _draw_fig8a(ax, bundle)
    figure_title(
        fig,
        "Fig. 8 | ASPR dual-path evidence review framework",
        "Graph evidence and ASPR-Qwen merge only after their separate artifacts are preserved.",
    )
    outputs.update(_finish_composite(fig, bundle, figure_dir, formats, dpi))
    return outputs


# ============================================================================
# Fig.9
# ============================================================================


def _draw_fig9a(ax: Axes, bundle: FigureBundle) -> None:
    """Draw the locked input case and run settings."""
    case = bundle.tables["case_manifest"].iloc[0]
    ax.set_title("a  Locked input paper", loc="left", pad=8, color=INK, fontweight="bold")
    ax.set_axis_off()
    ax.add_patch(
        FancyBboxPatch(
            (0.03, 0.08),
            0.94,
            0.80,
            boxstyle="round,pad=0.018,rounding_size=0.022",
            transform=ax.transAxes,
            facecolor=WHITE,
            edgecolor=LIGHT_BLUE,
            linewidth=1.1,
        )
    )
    ax.text(
        0.07,
        0.78,
        textwrap.fill(str(case["title"]), 44),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        0.07,
        0.48,
        f"{case['venue']} · {int(case['year'])}\nDOI {case['doi']}\n"
        f"transparent peer review: {case['transparent_peer_review']}\n"
        "checkpoint: Qwen3 review adapter",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        color=GRAY,
        linespacing=1.45,
    )
    ax.text(
        0.07,
        0.13,
        "Measurement boundary\n2023 is outside the frozen 1980–2017 v6.1 scoring cohort.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=VERMILLION,
        fontweight="bold",
    )


def _draw_fig9b(ax: Axes, bundle: FigureBundle) -> None:
    """Draw the corrected two-lane execution swimlane."""
    data = bundle.tables["execution_runtime"].sort_values("step")
    ax.set_title("b  Real execution lanes and recorded runtimes", loc="left", pad=8, color=INK, fontweight="bold")
    ax.set_axis_off()
    positions = {
        1: (0.06, 0.67),
        2: (0.17, 0.67),
        3: (0.30, 0.67),
        4: (0.42, 0.67),
        5: (0.54, 0.67),
        6: (0.66, 0.67),
        7: (0.55, 0.25),
        8: (0.76, 0.47),
        9: (0.86, 0.47),
        10: (0.95, 0.47),
    }
    short_stage = {
        1: "Parse",
        2: "Peer-review\nextract",
        3: "Prior-art\nretrieval",
        4: "v6.1\neligibility",
        5: "Legacy graph\nevidence",
        6: "Agent\nevaluation",
        7: "ASPR-Qwen",
        8: "Fusion",
        9: "Verify",
        10: "Export",
    }
    colors = {
        "input": MID_GRAY,
        "agent": BLUE,
        "ASPR-Qwen": PURPLE,
        "fusion": ORANGE,
        "verifier": ORANGE,
        "export": OLIVE,
    }
    ax.plot([0.04, 0.68], [0.67, 0.67], transform=ax.transAxes, color=PALE_GRAY, linewidth=8, zorder=0)
    ax.plot([0.17, 0.58], [0.25, 0.25], transform=ax.transAxes, color=PALE_GRAY, linewidth=8, zorder=0)
    ax.plot([0.74, 0.97], [0.47, 0.47], transform=ax.transAxes, color=PALE_GRAY, linewidth=8, zorder=0)
    ax.text(0.04, 0.58, "publication-prior graph lane", transform=ax.transAxes, fontsize=5.5, color=GRAY)
    ax.text(0.18, 0.16, "ASPR-Qwen lane", transform=ax.transAxes, fontsize=5.5, color=GRAY)
    for start, end in [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (2, 7),
        (6, 8),
        (7, 8),
        (8, 9),
        (9, 10),
    ]:
        ax.annotate(
            "",
            xy=positions[end],
            xytext=positions[start],
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "color": LIGHT_GRAY, "lw": 0.75},
        )
    for row in data.itertuples(index=False):
        step = int(row.step)
        x_value, y_value = positions[step]
        ax.add_patch(
            Circle(
                (x_value, y_value),
                0.028,
                transform=ax.transAxes,
                facecolor=colors.get(str(row.lane), MID_GRAY),
                edgecolor=WHITE,
                linewidth=0.8,
                zorder=3,
            )
        )
        if step <= 6:
            label_y = 0.80 if step % 2 else 0.91
            vertical_alignment = "bottom"
        elif step == 7:
            label_y = 0.10
            vertical_alignment = "top"
        else:
            label_y = 0.60 if step % 2 == 0 else 0.70
            vertical_alignment = "bottom"
        ax.text(
            x_value,
            label_y,
            f"{step} {short_stage[step]}\n{row.elapsed_seconds:.1f}s",
            transform=ax.transAxes,
            ha="center",
            va=vertical_alignment,
            fontsize=4.6,
            color=INK,
            linespacing=0.9,
        )
    ax.text(
        0.98,
        0.02,
        f"Recorded total {bundle.panel_text['b']['total_runtime_seconds']:.1f}s · "
        f"checkpoint {bundle.panel_text['b']['checkpoint_runtime_seconds']:.1f}s",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=GRAY,
        fontsize=6,
    )


def _draw_fig9c(ax: Axes, bundle: FigureBundle) -> None:
    """Draw five-angle cohort comparators and a deliberate case NA."""
    data = bundle.tables["five_angle_reference_profile"]
    ax.set_title("c  Current five-angle fingerprint boundary", loc="left", pad=8, color=INK, fontweight="bold")
    ax.set_axis_off()
    center = (0.50, 0.50)
    cohort = (
        data.set_index("angle_id")
        .loc[list(ANGLE_ORDER), "cohort_median"]
        .to_numpy(float)
    )
    high = (
        data.set_index("angle_id")
        .loc[list(ANGLE_ORDER), "high_diffusion_median"]
        .to_numpy(float)
    )
    for ring in [0.25, 0.5, 0.75, 1.0]:
        x, y, _ = _polar_xy(np.full(5, ring), center=center, radius=0.36)
        ax.plot(x, y, transform=ax.transAxes, color=PALE_GRAY, linewidth=0.6)
    cohort_x, cohort_y, theta = _polar_xy(cohort, center=center, radius=0.36)
    high_x, high_y, _ = _polar_xy(high, center=center, radius=0.36)
    ax.plot(cohort_x, cohort_y, transform=ax.transAxes, color=BLUE, linewidth=1.5, label="Cohort median")
    ax.plot(high_x, high_y, transform=ax.transAxes, color=ORANGE, linewidth=1.5, label="High-D5 median")
    ax.fill(high_x, high_y, transform=ax.transAxes, color=LIGHT_ORANGE, alpha=0.12)
    for index, angle_id in enumerate(ANGLE_ORDER):
        direction = np.array([math.cos(theta[index]), math.sin(theta[index])])
        label = np.array(center) + 0.46 * direction
        ax.text(
            label[0],
            label[1],
            ANGLE_SHORT[angle_id],
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.6,
            color=INK,
        )
    ax.add_patch(
        Circle(
            center,
            0.11,
            transform=ax.transAxes,
            facecolor=WHITE,
            edgecolor=VERMILLION,
            linestyle="--",
            linewidth=1.2,
        )
    )
    ax.text(
        *center,
        "CASE\nNOT SCORED",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=VERMILLION,
        fontsize=6.2,
        fontweight="bold",
    )
    ax.legend(frameon=False, loc="lower center", ncol=2, fontsize=5.7)


def _draw_fig9d(ax: Axes, bundle: FigureBundle) -> None:
    """Draw complementary agent and checkpoint output cards."""
    data = bundle.tables["agent_qwen_cards"]
    ax.set_title("d  Graph-evidence agent and ASPR-Qwen outputs", loc="left", pad=8, color=INK, fontweight="bold")
    ax.set_axis_off()
    for column, lane in enumerate(["Graph-evidence agent", "ASPR-Qwen"]):
        left = 0.02 + column * 0.50
        color = BLUE if column == 0 else PURPLE
        ax.add_patch(
            FancyBboxPatch(
                (left, 0.06),
                0.46,
                0.84,
                boxstyle="round,pad=0.015,rounding_size=0.020",
                transform=ax.transAxes,
                facecolor=WHITE,
                edgecolor=color,
                linewidth=1.1,
            )
        )
        ax.text(
            left + 0.03,
            0.84,
            lane,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            color=color,
            fontweight="bold",
        )
        rows = data.loc[data["lane"].eq(lane)].head(4)
        y_value = 0.73
        for row in rows.itertuples(index=False):
            ax.text(
                left + 0.03,
                y_value,
                "• " + textwrap.fill(str(row.text), 42),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5.7,
                color=INK,
            )
            ax.text(
                left + 0.03,
                y_value - 0.105,
                f"{row.status}" + (f" · {row.evidence_ids}" if row.evidence_ids else ""),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=4.8,
                color=GRAY if "missing" not in str(row.status) else VERMILLION,
            )
            y_value -= 0.18


def _draw_fig9e(ax: Axes, bundle: FigureBundle) -> None:
    """Draw a claim–evidence–verifier trace graph."""
    data = bundle.tables["claim_evidence_trace"].head(7)
    ax.set_title("e  Claim → evidence → verifier trace", loc="left", pad=8, color=INK, fontweight="bold")
    ax.set_axis_off()
    y_positions = np.linspace(0.84, 0.12, len(data))
    status_labels = {
        "supported": "supported",
        "supported_after_revision": "supported\npost-revision",
        "supported_with_caveat": "supported\nwith caveat",
        "supported_with_mechanistic_caveat": "mechanistic\ncaveat",
        "human_peer_review_overlap": "human-review\noverlap",
        "low_confidence_flag": "low-confidence\nflag",
    }
    for y_value, row in zip(y_positions, data.itertuples(index=False)):
        status_color = (
            BLUE
            if str(row.verifier_status) in {"supported", "supported_after_revision"}
            else ORANGE
            if "caveat" in str(row.verifier_status) or "uncertainty" in str(row.verifier_status)
            else PURPLE
        )
        ax.text(
            0.02,
            y_value,
            _short(row.claim, 47),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=4.9,
            color=INK,
        )
        ax.add_patch(
            Circle(
                (0.56, y_value),
                0.027,
                transform=ax.transAxes,
                facecolor=LIGHT_BLUE,
                edgecolor=BLUE,
                linewidth=0.8,
            )
        )
        ax.text(
            0.56,
            y_value,
            row.evidence_id,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.0,
            color=INK,
            fontweight="bold",
        )
        ax.text(
            0.98,
            y_value,
            status_labels.get(
                str(row.verifier_status),
                textwrap.fill(str(row.verifier_status).replace("_", " "), 16),
            ),
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=5.0,
            color=status_color,
        )
        ax.plot([0.42, 0.53], [y_value, y_value], transform=ax.transAxes, color=LIGHT_GRAY, linewidth=0.7)
        ax.plot([0.59, 0.78], [y_value, y_value], transform=ax.transAxes, color=status_color, linewidth=0.8)
    ax.text(0.02, 0.94, "final-review claim", transform=ax.transAxes, fontsize=5.2, color=GRAY)
    ax.text(0.56, 0.94, "evidence ID", transform=ax.transAxes, ha="center", fontsize=5.2, color=GRAY)
    ax.text(0.98, 0.94, "verifier", transform=ax.transAxes, ha="right", fontsize=5.2, color=GRAY)


def _draw_fig9f(ax: Axes, bundle: FigureBundle) -> None:
    """Draw who found each concern in the locked single case."""
    data = bundle.tables["human_overlap"]
    ax.set_title("f  ASPR–human overlap in one case", loc="left", pad=8, color=INK, fontweight="bold")
    columns = ["agent", "qwen", "fusion", "human_peer_review"]
    labels = ["Agent", "Qwen", "Fusion", "Human"]
    y = np.arange(len(data))[::-1]
    for y_value, row in zip(y, data.itertuples(index=False)):
        active = [index for index, column in enumerate(columns) if int(getattr(row, column))]
        if len(active) > 1:
            ax.plot(
                [min(active), max(active)],
                [y_value, y_value],
                color=LIGHT_GRAY,
                linewidth=1.1,
                zorder=0,
            )
        for index in range(len(columns)):
            enabled = index in active
            ax.scatter(
                index,
                y_value,
                s=27,
                facecolor=(
                    VERMILLION
                    if row.status == "human_only" and index == 3
                    else ORANGE
                    if row.status == "aspr_only_safeguard" and enabled
                    else INK
                    if enabled
                    else WHITE
                ),
                edgecolor=INK if enabled else LIGHT_GRAY,
                linewidth=0.7,
                zorder=2,
            )
    ax.set_xticks(range(len(columns)), labels)
    ax.set_yticks(y, [_short(value, 34) for value in data["concern"]])
    ax.set_xlim(-0.5, 3.5)
    clean_axes(ax, grid_axis="x")
    ax.text(
        0.99,
        0.02,
        "5/6 key human points matched; one case is not a population estimate",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.6,
        color=VERMILLION,
    )


def render_fig9(
    bundle: FigureBundle,
    figure_dir: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render Fig.9 and its standalone panels."""
    configure_style()
    drawers = {
        "a": _draw_fig9a,
        "b": _draw_fig9b,
        "c": _draw_fig9c,
        "d": _draw_fig9d,
        "e": _draw_fig9e,
        "f": _draw_fig9f,
    }
    outputs = _panel_outputs(bundle, drawers, figure_dir, dpi=dpi)
    fig = plt.figure(figsize=(17.0, 12.0))
    _add_background(fig, Path(bundle.chart_contract["background_asset"]), alpha=0.07)
    grid = fig.add_gridspec(2, 6, hspace=0.34, wspace=0.56)
    axes = {
        "a": fig.add_subplot(grid[0, :2]),
        "b": fig.add_subplot(grid[0, 2:4]),
        "c": fig.add_subplot(grid[0, 4:]),
        "d": fig.add_subplot(grid[1, :2]),
        "e": fig.add_subplot(grid[1, 2:4]),
        "f": fig.add_subplot(grid[1, 4:]),
    }
    for key, ax in axes.items():
        ax.set_facecolor((1, 1, 1, 0.90))
        drawers[key](ax, bundle)
    figure_title(
        fig,
        "Fig. 9 | Auditable end-to-end ASPR run on one locked paper",
        "The 2023 case is real and traceable; its current v6.1 numeric fingerprint is intentionally unavailable outside the frozen cohort.",
    )
    outputs.update(_finish_composite(fig, bundle, figure_dir, formats, dpi))
    return outputs


# ============================================================================
# Fig.10
# ============================================================================


def _draw_fig10a(ax: Axes, bundle: FigureBundle) -> None:
    """Draw the module switchboard and exact ablation names."""
    inventory = bundle.tables["module_inventory"]
    ax.set_title("a  Modules and one-to-one switches", loc="left", pad=8, color=INK, fontweight="bold")
    ax.set_axis_off()
    image_path = Path(bundle.chart_contract["background_asset"])
    ax.imshow(plt.imread(image_path), extent=[0, 1, 0, 1], aspect="auto", alpha=0.22)
    display = inventory.loc[~inventory["ablation_switch"].eq("full ASPR")].copy()
    y_positions = np.linspace(0.82, 0.16, len(display))
    family_colors = {
        "retrieval": BLUE,
        "graph agent": BLUE,
        "ASPR-Qwen": PURPLE,
        "fusion": ORANGE,
        "trace": OLIVE,
        "verifier": ORANGE,
    }
    for y_value, row in zip(y_positions, display.itertuples(index=False)):
        color = family_colors.get(str(row.family), MID_GRAY)
        ax.add_patch(
            FancyBboxPatch(
                (0.05, y_value - 0.035),
                0.62,
                0.07,
                boxstyle="round,pad=0.008,rounding_size=0.014",
                transform=ax.transAxes,
                facecolor=WHITE,
                edgecolor=color,
                linewidth=0.9,
            )
        )
        ax.text(
            0.08,
            y_value,
            _short(row.module, 40),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.8,
            color=INK,
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.72, y_value - 0.025),
                0.10,
                0.05,
                boxstyle="round,pad=0.002,rounding_size=0.025",
                transform=ax.transAxes,
                facecolor=LIGHT_BLUE,
                edgecolor=color,
                linewidth=0.8,
            )
        )
        ax.add_patch(
            Circle(
                (0.795, y_value),
                0.020,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=WHITE,
                linewidth=0.6,
            )
        )
        ax.text(
            0.86,
            y_value,
            _short(row.ablation_switch, 24),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.2,
            color=GRAY,
        )
    ax.text(
        0.04,
        0.03,
        "The historical “seven-indicator” inventory row is crosswalked, not silently relabeled as a v6.1 rerun.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.5,
        color=VERMILLION,
    )


def _draw_fig10b(ax: Axes, bundle: FigureBundle) -> None:
    """Draw paired metric deltas with a comparability warning."""
    data = bundle.tables["ablation_delta_estimates"].copy()
    ax.set_title("b  Automatic-score deltas (protocol-mismatched)", loc="left", pad=8, color=INK, fontweight="bold")
    variants = data["variant"].drop_duplicates().tolist()
    metrics = list(QUALITY_METRICS)
    metric_colors = {
        metric: color
        for metric, color in zip(
            metrics,
            [BLUE, ORANGE, OLIVE, PURPLE, PINK, VERMILLION],
        )
    }
    offsets = np.linspace(-0.24, 0.24, len(metrics))
    for metric_index, metric in enumerate(metrics):
        group = data.loc[data["metric"].eq(metric)].set_index("variant").reindex(variants)
        y = np.arange(len(variants)) + offsets[metric_index]
        ax.errorbar(
            group["mean_delta_ablation_minus_full"],
            y,
            xerr=[
                group["mean_delta_ablation_minus_full"] - group["ci_low"],
                group["ci_high"] - group["mean_delta_ablation_minus_full"],
            ],
            fmt="o",
            color=metric_colors[metric],
            capsize=1.8,
            markersize=3.2,
            linewidth=0.8,
            label=QUALITY_METRICS[metric],
        )
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(range(len(variants)), [_short(value, 26) for value in variants])
    ax.set_xlabel("Quality delta; positive means disabled path scored higher")
    clean_axes(ax, grid_axis="x")
    ax.legend(frameon=False, fontsize=4.8, ncol=2, loc="lower right")
    ax.text(
        0.99,
        0.98,
        "NOT MODULE-CAUSAL\nfull and disabled paths differ",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=VERMILLION,
        fontweight="bold",
    )


def _chord_curve(
    ax: Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    width: float,
    color: str,
) -> None:
    """Draw one compact Bezier chord."""
    vertices = [
        start,
        (0.5, start[1]),
        (0.5, end[1]),
        end,
    ]
    path = MplPath(vertices, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=width,
            alpha=0.23,
            capstyle="round",
            transform=ax.transAxes,
        )
    )


def _draw_fig10c(ax: Axes, bundle: FigureBundle) -> None:
    """Draw module-to-error links from threshold-derived case counts."""
    data = bundle.tables["module_error_links"]
    ax.set_title("c  Disabled path → error flags", loc="left", pad=8, color=INK, fontweight="bold")
    ax.set_axis_off()
    variants = data["variant"].drop_duplicates().tolist()
    errors = data["error_type"].drop_duplicates().tolist()
    left_y = np.linspace(0.88, 0.12, len(variants))
    right_y = np.linspace(0.88, 0.12, len(errors))
    left_map = dict(zip(variants, left_y))
    right_map = dict(zip(errors, right_y))
    max_count = max(int(data["error_count"].max()), 1)
    for row in data.itertuples(index=False):
        if int(row.error_count) <= 0:
            continue
        _chord_curve(
            ax,
            (0.30, left_map[row.variant]),
            (0.70, right_map[row.error_type]),
            0.3 + 4.0 * float(row.error_count) / max_count,
            VERMILLION if float(row.error_rate) > 0.5 else BLUE,
        )
    for variant, y_value in left_map.items():
        ax.text(
            0.02,
            y_value,
            _short(variant, 28),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.3,
            color=INK,
        )
        ax.add_patch(Circle((0.30, y_value), 0.012, transform=ax.transAxes, color=BLUE))
    for error, y_value in right_map.items():
        ax.add_patch(Circle((0.70, y_value), 0.012, transform=ax.transAxes, color=VERMILLION))
        ax.text(
            0.98,
            y_value,
            _short(error, 28),
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=5.3,
            color=INK,
        )
    ax.text(
        0.5,
        0.02,
        "Chord width = threshold-derived case count; not human error adjudication",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.4,
        color=GRAY,
    )


def _draw_fig10d(ax: Axes, bundle: FigureBundle) -> None:
    """Hard-block the unfinished human preference ternary panel."""
    completed = int(bundle.panel_text["d"]["completed_judgements"])
    required = int(bundle.panel_text["d"]["required_judgements"])
    draft_panel(
        ax,
        "d  Blinded human preference ternary",
        f"{completed}/{required} valid judgements completed.\n"
        "No full-win / tie / comparator-win points are rendered.",
    )


def _draw_fig10e(ax: Axes, bundle: FigureBundle) -> None:
    """Draw projected—not observed—quality/cost Pareto points."""
    data = bundle.tables["reinforcement_projections"].sort_values("relative_runtime_cost")
    ax.set_title("e  Projected quality–cost frontier", loc="left", pad=8, color=INK, fontweight="bold")
    frontier = data.loc[data["pareto_frontier"].eq(1)].sort_values("relative_runtime_cost")
    ax.plot(
        frontier["relative_runtime_cost"],
        frontier["quality_gain"],
        color=LIGHT_GRAY,
        linestyle="--",
        linewidth=0.9,
    )
    for row in data.itertuples(index=False):
        ax.scatter(
            row.relative_runtime_cost,
            row.quality_gain,
            s=65,
            facecolor=WHITE,
            edgecolor=ORANGE if int(row.pareto_frontier) else MID_GRAY,
            linewidth=1.3,
            marker="o",
        )
        ax.annotate(
            _short(str(row.reinforcement).replace("+ ", ""), 24),
            (row.relative_runtime_cost, row.quality_gain),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=5.4,
            color=INK,
        )
    ax.set_xlabel("Projected relative runtime cost")
    ax.set_ylabel("Projected composite quality gain")
    clean_axes(ax, grid_axis="both")
    ax.text(
        0.99,
        0.02,
        "PROJECTIONS ONLY — hollow markers",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=VERMILLION,
        fontsize=5.8,
        fontweight="bold",
    )


def _draw_fig10f(ax: Axes, bundle: FigureBundle) -> None:
    """Draw two observed case cards without forcing degradation."""
    data = bundle.tables["representative_cases"]
    ax.set_title("f  Representative protocol-mismatched case pairs", loc="left", pad=8, color=INK, fontweight="bold")
    ax.set_axis_off()
    for index, row in enumerate(data.itertuples(index=False)):
        bottom = 0.53 if index == 0 else 0.06
        color = BLUE if index == 0 else ORANGE
        ax.add_patch(
            FancyBboxPatch(
                (0.02, bottom),
                0.96,
                0.38,
                boxstyle="round,pad=0.014,rounding_size=0.020",
                transform=ax.transAxes,
                facecolor=WHITE,
                edgecolor=color,
                linewidth=1.0,
            )
        )
        ax.text(
            0.05,
            bottom + 0.32,
            f"{row.case_id} · {row.variant} · Δ={row.delta_ablation_minus_full:+.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.1,
            color=INK,
            fontweight="bold",
        )
        ax.text(
            0.05,
            bottom + 0.23,
            "FULL PATH: " + textwrap.fill(str(row.full_excerpt), 64),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.1,
            color=GRAY,
        )
        ax.text(
            0.05,
            bottom + 0.12,
            "DISABLED: " + textwrap.fill(str(row.ablation_excerpt), 72),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.1,
            color=color,
        )
        ax.text(
            0.95,
            bottom + 0.02,
            f"{row.observed_direction} · generation-path mismatch",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.0,
            color=VERMILLION,
        )


def render_fig10(
    bundle: FigureBundle,
    figure_dir: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render Fig.10 with visible comparability and human-evidence gates."""
    configure_style()
    drawers = {
        "a": _draw_fig10a,
        "b": _draw_fig10b,
        "c": _draw_fig10c,
        "d": _draw_fig10d,
        "e": _draw_fig10e,
        "f": _draw_fig10f,
    }
    outputs = _panel_outputs(bundle, drawers, figure_dir, dpi=dpi)
    fig = plt.figure(figsize=(17.0, 12.0))
    _add_background(fig, Path(bundle.chart_contract["background_asset"]), alpha=0.06)
    grid = fig.add_gridspec(2, 6, hspace=0.34, wspace=0.58)
    axes = {
        "a": fig.add_subplot(grid[0, :2]),
        "b": fig.add_subplot(grid[0, 2:4]),
        "c": fig.add_subplot(grid[0, 4:]),
        "d": fig.add_subplot(grid[1, :2]),
        "e": fig.add_subplot(grid[1, 2:4]),
        "f": fig.add_subplot(grid[1, 4:]),
    }
    for key, ax in axes.items():
        ax.set_facecolor((1, 1, 1, 0.92))
        drawers[key](ax, bundle)
    figure_title(
        fig,
        "Fig. 10 | Module evidence, preference gate and quality–cost projections",
        "All 400 automatic rows are retained, but generation-path mismatch blocks causal ablation claims and 0/750 human preferences keep the figure in DRAFT.",
        draft=True,
    )
    outputs.update(_finish_composite(fig, bundle, figure_dir, formats, dpi))
    return outputs
