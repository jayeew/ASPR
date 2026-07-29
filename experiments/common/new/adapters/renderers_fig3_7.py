"""Compact, vector-native renderers for the current Fig.3--Fig.7 route.

The old experiments define the scientific questions.  These renderers retain
that narrative while using only the current v6.1 tables and definitions.
They deliberately avoid the former PNG-on-PNG vertical extensions.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox
from experiments.common.new.base.common import (
    ANGLE_COLORS,
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
    PURPLE,
    VERMILLION,
    WHITE,
    FigureBundle,
    clean_axes,
    configure_style,
    export_figure,
    figure_title,
    panel_title,
)
from experiments.common.new.base.renderers_1_5 import (
    _draw_fig5a,
    _draw_fig5d,
    _draw_fig5e,
)
from experiments.common.new.base.renderers_6_10 import (
    VENUE_COLORS,
    VENUE_DISPLAY_LABELS,
    _draw_fig7a,
    _draw_fig7b,
    _draw_fig7c,
)


FIGURE_IDS = (3, 4, 5, 6, 7)


def _short(value: object, width: int) -> str:
    """Return one compact display label."""
    return textwrap.shorten(str(value), width=width, placeholder="…")


def _clean_generated_artifacts(figure_id: int, figure_dir: Path) -> None:
    """Remove only reproducible figure images from prior renderer versions."""
    stems = {
        3: ("fig03_full", "fig03_legacy_route_extension"),
        4: ("fig04_full",),
        5: ("fig05_full",),
        6: ("fig06_full", "fig06_reference_dose_extension"),
        7: ("fig07_full", "fig07_common_support_extension"),
    }[figure_id]
    for stem in (*stems, "figure_full"):
        for extension in ("png", "svg", "pdf"):
            (figure_dir / f"{stem}.{extension}").unlink(missing_ok=True)
    panel_dir = figure_dir / "panels"
    if panel_dir.exists():
        for path in panel_dir.glob(f"fig{figure_id:02d}_*"):
            if path.is_file():
                path.unlink()


def _axis_bbox(fig: plt.Figure, axes: Sequence[plt.Axes]) -> Bbox:
    """Return a padded tight bounding box for a panel and its inset axes."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [
        axis.get_tightbbox(renderer).transformed(
            fig.dpi_scale_trans.inverted()
        )
        for axis in axes
    ]
    return Bbox.union(boxes).expanded(1.07, 1.13)


def _export_compact(
    figure_id: int,
    fig: plt.Figure,
    groups: Mapping[str, Sequence[plt.Axes]],
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Export one true-vector composite plus independently reusable panels."""
    outputs: Dict[str, Path] = {}
    rendered = export_figure(
        fig,
        figure_dir / "figure_full",
        formats=formats,
        dpi=dpi,
    )
    outputs.update({f"figure_{key}": value for key, value in rendered.items()})
    panel_dir = figure_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for panel, axes in groups.items():
        extent = _axis_bbox(fig, axes)
        for extension in formats:
            path = panel_dir / f"fig{figure_id:02d}_{panel}.{extension}"
            kwargs: Dict[str, object] = {"bbox_inches": extent}
            if extension == "png":
                kwargs["dpi"] = dpi
            fig.savefig(path, **kwargs)
            outputs[f"panel_{panel}_{extension}"] = path
    plt.close(fig)
    return outputs


def _box(
    ax: plt.Axes,
    left: float,
    bottom: float,
    width: float,
    height: float,
    text: str,
    color: str,
    *,
    facecolor: str = WHITE,
    fontsize: float = 6.3,
) -> None:
    """Draw a restrained rounded process box in axes coordinates."""
    ax.add_patch(
        FancyBboxPatch(
            (left, bottom),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            transform=ax.transAxes,
            facecolor=facecolor,
            edgecolor=color,
            linewidth=1.0,
        )
    )
    ax.text(
        left + width / 2,
        bottom + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.1,
        fontweight="bold",
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = MID_GRAY,
) -> None:
    """Draw one process arrow in axes coordinates."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color=color,
        )
    )


# ============================================================================
# Fig.3
# ============================================================================


def _fig3_target(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Explain the realized D5 outcome before model performance is shown."""
    panel_title(ax, "a", "The D5 outcome measures later uptake and diffusion")
    ax.set_axis_off()
    data = bundle.tables["d5_target_construction"].sort_values("step")
    counts = bundle.tables["d5_target_counts"].iloc[0]
    x_positions = np.linspace(0.11, 0.89, len(data))
    for index, (x_value, row) in enumerate(
        zip(x_positions, data.itertuples(index=False))
    ):
        color = ORANGE if index == len(data) - 1 else BLUE
        facecolor = "#FFF8EE" if index == len(data) - 1 else "#F5FAFC"
        _box(
            ax,
            x_value - 0.095,
            0.42,
            0.19,
            0.28,
            f"{row.component}\n{textwrap.fill(str(row.definition), 24)}",
            color,
            facecolor=facecolor,
            fontsize=5.0,
        )
        if index < len(data) - 1:
            _arrow(
                ax,
                (x_value + 0.10, 0.56),
                (x_positions[index + 1] - 0.10, 0.56),
            )
    ax.text(
        0.5,
        0.31,
        "If no future uptake: D5 = 0    |    otherwise: "
        "D5 = 0.5 × breadth + 0.5 × evenness",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=INK,
        fontsize=6.5,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.14,
        f"{int(counts['target_valid']):,} valid papers · "
        f"{int(counts['positive_uptake']):,} with uptake · "
        f"{int(counts['zero_uptake']):,} zero-uptake papers",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=GRAY,
        fontsize=6.0,
    )
    ax.text(
        0.5,
        0.04,
        "Breadth/evenness percentile references are fitted inside each training fold.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=VERMILLION,
        fontsize=5.5,
    )
    return [ax]


def _fig3_model(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Show how the eight signals are combined under publication-year OOF."""
    panel_title(ax, "b", "Eight signals enter one two-part temporal-OOF model")
    ax.set_axis_off()
    _box(
        ax,
        0.02,
        0.55,
        0.22,
        0.24,
        "5 source-backed angles\n8 frozen indicators\n+ K1 controls",
        BLUE,
        facecolor="#F5FAFC",
    )
    _box(
        ax,
        0.37,
        0.67,
        0.24,
        0.17,
        "Part 1\nP(future uptake)",
        PURPLE,
        facecolor="#F9F6FC",
    )
    _box(
        ax,
        0.37,
        0.39,
        0.24,
        0.17,
        "Part 2\nD5 | future uptake",
        PURPLE,
        facecolor="#F9F6FC",
    )
    _box(
        ax,
        0.76,
        0.53,
        0.21,
        0.25,
        "Expected D5\nprobability ×\nconditional score",
        ORANGE,
        facecolor="#FFF8EE",
    )
    for start, end in (
        ((0.24, 0.67), (0.37, 0.75)),
        ((0.24, 0.67), (0.37, 0.48)),
        ((0.61, 0.75), (0.76, 0.67)),
        ((0.61, 0.48), (0.76, 0.62)),
    ):
        _arrow(ax, start, end)
    folds = bundle.tables["temporal_folds"].sort_values("fold_id")
    left, right, y_value = 0.055, 0.95, 0.19
    ax.plot(
        [left, right],
        [y_value, y_value],
        transform=ax.transAxes,
        color=LIGHT_GRAY,
        linewidth=2.0,
    )
    for index, row in enumerate(folds.itertuples(index=False)):
        cell_width = (right - left) / len(folds)
        x0 = left + index * cell_width + 0.005
        x1 = left + (index + 1) * cell_width - 0.005
        color = list(ANGLE_COLORS.values())[index % len(ANGLE_COLORS)]
        ax.plot(
            [x0, x1],
            [y_value, y_value],
            transform=ax.transAxes,
            color=color,
            linewidth=6,
            solid_capstyle="butt",
        )
        ax.text(
            (x0 + x1) / 2,
            y_value - 0.055,
            f"F{int(row.fold_id)}\n"
            f"{int(row.test_year_min)}–{int(row.test_year_max)}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=4.6,
            color=GRAY,
            linespacing=0.9,
        )
    ax.text(
        0.5,
        0.015,
        "Retrospective temporal OOF by publication year; this run does not impose a D5 label-maturity embargo.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.2,
        color=VERMILLION,
    )
    return [ax]


def _fig3_ladder(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Compare fixed feature sets on exactly the same D5 OOF rows."""
    panel_title(ax, "c", "The eight indicators add rank information beyond controls")
    data = bundle.tables["model_ladder"].sort_values(
        "spearman_expected",
        ascending=True,
    )
    y_values = np.arange(len(data))
    primary_ids = {
        "k1_controls",
        "innovation_only",
        "final_innovation_plus_k1",
    }
    for y_value, row in zip(y_values, data.itertuples(index=False)):
        primary = row.model_id in primary_ids
        color = (
            ORANGE
            if row.model_id == "final_innovation_plus_k1"
            else BLUE
            if primary
            else MID_GRAY
        )
        ax.plot(
            [row.fold_min, row.fold_max],
            [y_value, y_value],
            color=color,
            alpha=0.42,
            linewidth=1.6,
        )
        ax.scatter(
            row.spearman_expected,
            y_value,
            s=42 if primary else 24,
            color=color,
            edgecolor=WHITE,
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            float(row.spearman_expected) + 0.006,
            y_value,
            f"{float(row.spearman_expected):.3f}",
            ha="left",
            va="center",
            fontsize=5.6,
            color=INK,
        )
    ax.set_yticks(
        y_values,
        [_short(value, 25) for value in data["model_label_en"]],
        fontsize=6.0,
    )
    ax.set_xlim(min(0.42, float(data["fold_min"].min()) - 0.02), 0.86)
    ax.set_xlabel("D5 temporal-OOF Spearman · whisker = six-fold range")
    clean_axes(ax, grid_axis="x")
    paired = bundle.tables["paired_model_gains"]
    gain = paired.loc[paired["baseline_model_id"].eq("k1_controls")].iloc[0]
    ax.text(
        0.02,
        0.98,
        f"Final 8 + K1 vs K1: Δρ={gain.spearman_gain:+.4f} "
        f"[{gain.gain_ci_low:+.4f}, {gain.gain_ci_high:+.4f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color=BLUE,
        fontweight="bold",
    )
    return [ax]


def _fig3_density(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Show the full OOF prediction-realization relationship."""
    panel_title(ax, "d", "OOF predictions preserve the realized D5 ordering")
    ax.set_axis_off()
    data = bundle.tables["oof_joint_density"]
    x_values = data["expected_diffusion_score"].to_numpy(float)
    y_values = data["realized_diffusion_target"].to_numpy(float)
    joint = ax.inset_axes([0.12, 0.12, 0.70, 0.70])
    top = ax.inset_axes([0.12, 0.83, 0.70, 0.10], sharex=joint)
    right = ax.inset_axes([0.83, 0.12, 0.10, 0.70], sharey=joint)
    joint.hexbin(
        x_values,
        y_values,
        gridsize=48,
        bins="log",
        mincnt=1,
        cmap="Blues",
        linewidths=0,
    )
    top.hist(
        x_values,
        bins=48,
        density=True,
        color=LIGHT_BLUE,
        edgecolor="none",
    )
    right.hist(
        y_values,
        bins=48,
        density=True,
        orientation="horizontal",
        color=LIGHT_ORANGE,
        edgecolor="none",
    )
    top.set_axis_off()
    right.set_axis_off()
    joint.set_xlabel("Expected D5 diffusion")
    joint.set_ylabel("Realized D5 diffusion")
    clean_axes(joint)
    rho = float(
        data[
            ["expected_diffusion_score", "realized_diffusion_target"]
        ].corr(method="spearman").iloc[0, 1]
    )
    ax.text(
        0.94,
        0.91,
        f"ρ = {rho:.3f}\nN = {len(data):,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color=INK,
        fontweight="bold",
    )
    return [ax, joint, top, right]


def _fig3_deciles(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Show realized outcomes, not only one correlation coefficient."""
    panel_title(ax, "e", "Realized D5 rises across prediction deciles")
    data = bundle.tables["prediction_decile_sample"].copy()
    grouped = [
        data.loc[
            data["prediction_decile"].eq(decile),
            "realized_diffusion_target",
        ].dropna().to_numpy(float)
        for decile in range(1, 11)
    ]
    violin = ax.violinplot(
        grouped,
        positions=np.arange(1, 11),
        widths=0.78,
        showextrema=False,
    )
    for index, body in enumerate(violin["bodies"], start=1):
        body.set_facecolor(ORANGE if index == 10 else LIGHT_BLUE)
        body.set_edgecolor("none")
        body.set_alpha(0.66)
    rng = np.random.default_rng(20260725)
    for decile, values in enumerate(grouped, start=1):
        if not len(values):
            continue
        selected = rng.choice(values, size=min(45, len(values)), replace=False)
        ax.scatter(
            decile + rng.uniform(-0.16, 0.16, len(selected)),
            selected,
            s=4,
            color=INK,
            alpha=0.18,
            edgecolor="none",
            rasterized=True,
        )
        ax.scatter(
            decile,
            np.median(values),
            s=20,
            color=ORANGE if decile == 10 else BLUE,
            edgecolor=WHITE,
            linewidth=0.4,
            zorder=4,
        )
    enrichment = float(
        data.loc[data["prediction_decile"].eq(10), "enrichment_over_base"].iloc[0]
    )
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("OOF prediction decile")
    ax.set_ylabel("Realized D5 diffusion")
    ax.set_ylim(-0.02, 1.02)
    clean_axes(ax, grid_axis="y")
    ax.text(
        0.98,
        0.96,
        f"Top predicted decile\n{enrichment:.2f}× enriched for realized top-decile D5",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=ORANGE,
        fontweight="bold",
    )
    return [ax]


def _fig3_angles(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Replace old simplex weights with valid five-angle diagnostics."""
    panel_title(ax, "f", "Each angle contributes differently, with temporal drift")
    ax.set_axis_off()
    effects = bundle.tables["angle_add_delete"].sort_values("angle_number")
    folds = bundle.tables["angle_fold_stability"].copy()
    left = ax.inset_axes([0.04, 0.13, 0.43, 0.76])
    right = ax.inset_axes([0.57, 0.13, 0.40, 0.76])
    y_values = np.arange(len(effects))[::-1]
    for y_value, row in zip(y_values, effects.itertuples(index=False)):
        color = ANGLE_COLORS[str(row.angle_id)]
        left.plot(
            [0, row.increment_over_k1],
            [y_value - 0.09, y_value - 0.09],
            color=color,
            linewidth=1.3,
        )
        left.scatter(
            row.increment_over_k1,
            y_value - 0.09,
            s=28,
            color=color,
            edgecolor=WHITE,
            linewidth=0.5,
            label="Add to K1" if y_value == y_values[0] else None,
        )
        left.plot(
            [0, row.drop_from_full],
            [y_value + 0.09, y_value + 0.09],
            color=color,
            linewidth=1.0,
            alpha=0.75,
        )
        left.scatter(
            row.drop_from_full,
            y_value + 0.09,
            s=25,
            facecolor=WHITE,
            edgecolor=color,
            linewidth=1.0,
            label="Delete from full" if y_value == y_values[0] else None,
        )
    left.axvline(0, color=LIGHT_GRAY, linewidth=0.8)
    left.set_yticks(
        y_values,
        [ANGLE_SHORT[str(value)] for value in effects["angle_id"]],
        fontsize=5.5,
    )
    left.set_xlabel("Global Δ Spearman")
    left.set_title("Global add / delete effects", loc="left", fontsize=6.8)
    clean_axes(left, grid_axis="x")
    left.text(
        0.98,
        0.98,
        "post-hoc · filled = add · hollow = delete",
        transform=left.transAxes,
        ha="right",
        va="top",
        fontsize=4.6,
        color=GRAY,
    )

    summaries = (
        folds.groupby(["angle_number", "angle_id", "diagnostic"])[
            "spearman_expected"
        ]
        .agg(["median", "min", "max"])
        .reset_index()
    )
    offsets = {"add to K1": -0.09, "delete from full": 0.09}
    markers = {"add to K1": "o", "delete from full": "s"}
    for row in summaries.itertuples(index=False):
        x_value = int(row.angle_number) + offsets[str(row.diagnostic)]
        color = ANGLE_COLORS[str(row.angle_id)]
        right.plot(
            [x_value, x_value],
            [row.min, row.max],
            color=LIGHT_GRAY,
            linewidth=1.2,
            zorder=1,
        )
        right.scatter(
            x_value,
            row.median,
            s=27,
            marker=markers[str(row.diagnostic)],
            facecolor=color if row.diagnostic == "add to K1" else WHITE,
            edgecolor=color,
            linewidth=0.8,
            zorder=3,
        )
    right.set_xticks(
        range(1, 6),
        [f"A{index}" for index in range(1, 6)],
    )
    right.set_ylabel("Fold-specific Spearman")
    right.set_title("Median and range across six folds", loc="left", fontsize=6.8)
    clean_axes(right, grid_axis="y")
    return [ax, left, right]


def render_fig3_compact(
    bundle: FigureBundle,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render Fig.3 as one coherent six-panel vector figure."""
    configure_style()
    _clean_generated_artifacts(3, figure_dir)
    fig = plt.figure(figsize=(15.8, 9.4))
    grid = fig.add_gridspec(
        2,
        6,
        left=0.045,
        right=0.985,
        bottom=0.055,
        top=0.905,
        hspace=0.34,
        wspace=0.60,
    )
    axes = {
        "a": fig.add_subplot(grid[0, :2]),
        "b": fig.add_subplot(grid[0, 2:4]),
        "c": fig.add_subplot(grid[0, 4:]),
        "d": fig.add_subplot(grid[1, :2]),
        "e": fig.add_subplot(grid[1, 2:4]),
        "f": fig.add_subplot(grid[1, 4:]),
    }
    drawers = {
        "a": _fig3_target,
        "b": _fig3_model,
        "c": _fig3_ladder,
        "d": _fig3_density,
        "e": _fig3_deciles,
        "f": _fig3_angles,
    }
    groups = {
        panel: drawers[panel](axis, bundle)
        for panel, axis in axes.items()
    }
    figure_title(
        fig,
        "Fig. 3 | Publication-time signals rank later D5 diffusion",
        "One D5 definition, one paper set and six publication-year OOF folds; "
        "five-angle diagnostics explain the fitted system without recreating obsolete linear weights.",
    )
    return _export_compact(3, fig, groups, figure_dir, formats, dpi)


# ============================================================================
# Fig.4
# ============================================================================


def _fig4_bridge(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Make the two validation cohorts and their claim boundaries explicit."""
    panel_title(ax, "a", "Two evidence routes test different aspects of validity")
    ax.set_axis_off()
    _box(
        ax,
        0.02,
        0.37,
        0.18,
        0.31,
        "Five angles\nEight publication-time\nindicators",
        BLUE,
        facecolor="#F5FAFC",
    )
    _box(
        ax,
        0.29,
        0.53,
        0.26,
        0.25,
        "Cohort A · 30 current-score papers\nlow / middle / high × 10\n3 blinded labelers",
        PURPLE,
        facecolor="#F9F6FC",
        fontsize=5.8,
    )
    _box(
        ax,
        0.29,
        0.18,
        0.26,
        0.25,
        "Cohort B · 50 transparent reviews\nquote-grounded diagnostic\nrange-restricted",
        ORANGE,
        facecolor="#FFF8EE",
        fontsize=5.8,
    )
    _box(
        ax,
        0.67,
        0.53,
        0.29,
        0.25,
        "Planned construct validity\nnovelty · significance · prior art\nagreement + rating trend",
        PURPLE,
        facecolor=WHITE,
        fontsize=5.7,
    )
    _box(
        ax,
        0.67,
        0.18,
        0.29,
        0.25,
        "Available diagnostic evidence\nreview-aspect coverage\nand traceability gaps",
        ORANGE,
        facecolor=WHITE,
        fontsize=5.7,
    )
    for start, end in (
        ((0.20, 0.53), (0.29, 0.65)),
        ((0.20, 0.53), (0.29, 0.30)),
        ((0.55, 0.65), (0.67, 0.65)),
        ((0.55, 0.30), (0.67, 0.30)),
    ):
        _arrow(ax, start, end)
    audit = bundle.tables["v6_1_blinded_completion_audit"].iloc[0]
    ax.text(
        0.5,
        0.04,
        f"Cohort A gate: {int(audit.completed_paper_labeler_rows)}/"
        f"{int(audit.required_paper_labeler_rows)} completed; "
        "Cohort B cannot substitute for the missing blinded labels.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=VERMILLION,
        fontsize=5.6,
        fontweight="bold",
    )
    return [ax]


def _fig4_sample(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Show the locked current-score validation frame without outcomes."""
    panel_title(ax, "b", "The locked sample spans low, middle and high OOF scores")
    data = bundle.tables["validation_sample_coverage"].copy()
    tier_order = ["low", "middle", "high"]
    colors = {"low": OLIVE, "middle": PURPLE, "high": ORANGE}
    rng = np.random.default_rng(20260725)
    for index, tier in enumerate(tier_order):
        group = data.loc[data["global_fig3_tier"].eq(tier)]
        jitter = rng.uniform(-0.10, 0.10, len(group))
        ax.scatter(
            group["validation_score"],
            index + jitter,
            s=34,
            facecolor=WHITE,
            edgecolor=colors[tier],
            linewidth=1.0,
            label=f"{tier.title()} · n={len(group)}",
        )
        ax.plot(
            [
                float(group["validation_score"].min()),
                float(group["validation_score"].max()),
            ],
            [index, index],
            color=colors[tier],
            linewidth=1.2,
            alpha=0.65,
            zorder=0,
        )
    ax.set_yticks(range(3), ["Low", "Middle", "High"])
    ax.set_xlabel("Current v6.1 expected-D5 temporal-OOF score")
    clean_axes(ax, grid_axis="x")
    ax.text(
        0.99,
        0.03,
        "Scores and realized D5 targets are absent from the blinded packet.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.4,
        color=GRAY,
    )
    return [ax]


def _fig4_aspects(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Show the existing transparent-review audit with honest scope."""
    panel_title(ax, "c", "Transparent reviews expose aspect-specific evidence gaps")
    data = bundle.tables["transparent_review_aspect_summary"].sort_values(
        "aspect_order",
        ascending=False,
    )
    y_values = np.arange(len(data))
    for y_value, row in zip(y_values, data.itertuples(index=False)):
        color = (
            VERMILLION
            if row.aspect == "claim_evidence_coverage"
            else BLUE
        )
        ax.plot(
            [row.ci_low, row.ci_high],
            [y_value, y_value],
            color=LIGHT_ORANGE
            if row.aspect == "claim_evidence_coverage"
            else LIGHT_BLUE,
            linewidth=2.0,
        )
        ax.scatter(
            row.mean_alignment,
            y_value,
            s=34,
            color=color,
            edgecolor=WHITE,
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            min(float(row.ci_high) + 0.025, 1.02),
            y_value,
            f"{float(row.mean_alignment):.2f} · n={int(row.n_valid)}",
            ha="left",
            va="center",
            fontsize=5.4,
            color=INK,
        )
    ax.axvline(0.5, color=LIGHT_GRAY, linestyle=":", linewidth=0.8)
    ax.set_yticks(y_values, data["aspect_label"], fontsize=6.0)
    ax.set_xlim(-0.03, 1.08)
    ax.set_xlabel("Mean cached aspect alignment · paper bootstrap 95% interval")
    clean_axes(ax, grid_axis="x")
    ax.text(
        0.99,
        0.02,
        "50 accepted Nature Portfolio papers; diagnostic only, not validation of the current D5 score.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.2,
        color=VERMILLION,
    )
    return [ax]


def _fig4_completion(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Render the complete 30 × 3 evidence gate rather than empty result axes."""
    panel_title(ax, "d", "The blinded-evidence gate remains closed")
    data = bundle.tables["blinded_label_completion_matrix"].copy()
    case_order = sorted(data["blinded_case_id"].unique())
    labeler_order = sorted(data["labeler_id"].unique())
    matrix = (
        data.pivot(
            index="labeler_id",
            columns="blinded_case_id",
            values="complete",
        )
        .reindex(index=labeler_order, columns=case_order)
        .fillna(0)
    )
    ax.imshow(
        matrix.to_numpy(float),
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(labeler_order)), ["R1", "R2", "R3"])
    ax.set_xticks(
        [0, 9, 19, 29],
        ["Case 1", "10", "20", "30"],
    )
    for x_value in np.arange(-0.5, len(case_order), 1):
        ax.axvline(x_value, color=WHITE, linewidth=0.35)
    for y_value in np.arange(-0.5, len(labeler_order), 1):
        ax.axhline(y_value, color=WHITE, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    audit = bundle.tables["v6_1_blinded_completion_audit"].iloc[0]
    ax.text(
        0.99,
        1.02,
        f"{int(audit.completed_paper_labeler_rows)}/"
        f"{int(audit.required_paper_labeler_rows)} complete · "
        f"{int(audit.text_ready_papers)}/30 papers have reviewable text",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.1,
        color=VERMILLION,
        fontweight="bold",
    )
    ax.text(
        0.0,
        -0.18,
        "A filled cell requires novelty, significance and prior-art labels from that reviewer.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
        color=GRAY,
    )
    return [ax]


def _fig4_blocked(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """List the exact analyses withheld by the human-evidence gate."""
    panel_title(ax, "e", "Inferential endpoints are withheld, not imputed")
    ax.set_axis_off()
    endpoints = (
        ("Human novelty ↔ innovation-only score", "WITHHELD"),
        ("Human significance ↔ expected D5", "WITHHELD"),
        ("Ordinal agreement / weighted κ", "WITHHELD"),
        ("Agreement, miss and over-call quote cases", "WITHHELD"),
    )
    for index, (label, status) in enumerate(endpoints):
        y_value = 0.78 - index * 0.18
        ax.add_patch(
            FancyBboxPatch(
                (0.04, y_value - 0.055),
                0.92,
                0.115,
                boxstyle="round,pad=0.006,rounding_size=0.014",
                transform=ax.transAxes,
                facecolor="#FFF8F1",
                edgecolor=LIGHT_ORANGE,
                linewidth=0.8,
            )
        )
        ax.text(
            0.07,
            y_value,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.0,
            color=INK,
        )
        ax.text(
            0.93,
            y_value,
            status,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=5.6,
            color=VERMILLION,
            fontweight="bold",
        )
    ax.text(
        0.5,
        0.04,
        "Publication claim: not ready. The figure documents the protocol and available diagnostic evidence only.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.5,
        color=VERMILLION,
        fontweight="bold",
        wrap=True,
    )
    return [ax]


def render_fig4_compact(
    bundle: FigureBundle,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render a polished but explicitly evidence-gated Fig.4."""
    configure_style()
    _clean_generated_artifacts(4, figure_dir)
    fig = plt.figure(figsize=(15.8, 10.2))
    grid = fig.add_gridspec(
        3,
        6,
        left=0.045,
        right=0.985,
        bottom=0.055,
        top=0.905,
        height_ratios=[0.72, 1.0, 0.88],
        hspace=0.42,
        wspace=0.55,
    )
    axes = {
        "a": fig.add_subplot(grid[0, :]),
        "b": fig.add_subplot(grid[1, :3]),
        "c": fig.add_subplot(grid[1, 3:]),
        "d": fig.add_subplot(grid[2, :3]),
        "e": fig.add_subplot(grid[2, 3:]),
    }
    drawers = {
        "a": _fig4_bridge,
        "b": _fig4_sample,
        "c": _fig4_aspects,
        "d": _fig4_completion,
        "e": _fig4_blocked,
    }
    groups = {
        panel: drawers[panel](axis, bundle)
        for panel, axis in axes.items()
    }
    figure_title(
        fig,
        "Fig. 4 | Human construct validity remains evidence-gated",
        "The current-score 30-paper protocol is locked and blinded; a separate "
        "50-paper transparent-review cohort supplies diagnostics but cannot replace 90 missing judgments.",
        draft=True,
    )
    return _export_compact(4, fig, groups, figure_dir, formats, dpi)


# ============================================================================
# Fig.5
# ============================================================================


def _fig5_landscape(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Draw a sparse predicted-realized topic map with only five direct labels."""
    data = bundle.tables["topic_landscape"].copy()
    cutoff = int(data["cutoff"].iloc[0])
    panel_title(ax, "b", f"Predicted–realized topic landscape · cutoff {cutoff}")
    styles = {
        "background": (LIGHT_GRAY, "o", 0.22, 8),
        "hit": (ORANGE, "o", 0.95, 54),
        "false_positive": (BLUE, "s", 0.90, 40),
        "miss": (WHITE, "X", 1.0, 45),
    }
    for classification, group in data.groupby("classification"):
        color, marker, alpha, size = styles[str(classification)]
        ax.scatter(
            group["x"],
            group["y"],
            s=size,
            marker=marker,
            facecolor=color,
            edgecolor=VERMILLION if classification == "miss" else WHITE,
            linewidth=0.75,
            alpha=alpha,
            label=str(classification).replace("_", " ").title(),
        )
    selected = []
    for classification, count in (
        ("hit", 2),
        ("false_positive", 2),
        ("miss", 2),
    ):
        group = data.loc[data["classification"].eq(classification)]
        if classification == "miss":
            group = group.nsmallest(count, "realized_frontier_score_rank")
        else:
            group = group.nsmallest(count, "prediction_score_rank")
        selected.append(group)
    labels = pd.concat(selected, ignore_index=True).head(6)
    offsets = [(7, 7), (7, -14), (-7, 8), (-7, -15), (7, 8), (7, -14)]
    for offset, row in zip(offsets, labels.itertuples(index=False)):
        ax.annotate(
            textwrap.fill(_short(row.display_topic_label, 34), 18),
            (row.x, row.y),
            xytext=offset,
            textcoords="offset points",
            ha="left" if offset[0] > 0 else "right",
            va="bottom" if offset[1] > 0 else "top",
            fontsize=4.9,
            color=INK,
            linespacing=0.95,
            arrowprops={
                "arrowstyle": "-",
                "color": LIGHT_GRAY,
                "linewidth": 0.5,
            },
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(
        "Topic-label TF–IDF layout · coordinates have no metric interpretation"
    )
    ax.legend(frameon=False, fontsize=5.0, loc="upper right")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return [ax]


def _fig5_bump(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Track the ten predicted topics without the former union-of-ranks clutter."""
    data = bundle.tables["rank_bump_topics"].copy()
    cutoff = int(data["cutoff"].max())
    data = data.loc[
        data["cutoff"].eq(cutoff)
        & data["prediction_score_rank"].le(10)
    ].sort_values("prediction_score_rank")
    panel_title(ax, "c", f"Predicted top 10 versus realized rank · cutoff {cutoff}")
    hits: list[tuple[object, int, int]] = []
    for row in data.itertuples(index=False):
        predicted = int(row.prediction_score_rank)
        realized_raw = int(row.realized_frontier_score_rank)
        realized = min(realized_raw, 12)
        hit = realized_raw <= 10
        if hit:
            hits.append((row, predicted, realized))
        color = ORANGE if hit else MID_GRAY
        ax.plot(
            [0, 1],
            [predicted, realized],
            color=color,
            linewidth=1.5 if hit else 0.9,
            alpha=0.90,
        )
        ax.scatter(
            [0, 1],
            [predicted, realized],
            s=18,
            facecolor=color if hit else WHITE,
            edgecolor=color,
            linewidth=0.7,
            zorder=3,
        )
    for row, predicted, realized in hits:
        ax.annotate(
            _short(row.display_topic_label, 30),
            (0.50, (predicted + realized) / 2),
            xytext=(5, 5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=4.8,
            color=ORANGE,
            fontweight="bold",
        )
    ax.axhline(10.5, color=LIGHT_GRAY, linestyle=":", linewidth=0.8)
    ax.text(
        1.02,
        11.45,
        ">10",
        ha="left",
        va="center",
        fontsize=5.0,
        color=GRAY,
    )
    ax.set_xlim(-0.18, 1.12)
    ax.set_ylim(12.5, 0.5)
    ax.set_xticks([0, 1], ["Predicted", "Realized"])
    ax.set_yticks(range(1, 13))
    ax.set_ylabel("Rank")
    clean_axes(ax, grid_axis="y")
    ax.text(
        0.98,
        0.97,
        f"{len(hits)}/10 predicted topics remained in the realized top 10",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.0,
        color=VERMILLION,
    )
    return [ax]


def render_fig5_compact(
    bundle: FigureBundle,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Retain the old frontier-prediction story in one compact current figure."""
    configure_style()
    _clean_generated_artifacts(5, figure_dir)
    fig = plt.figure(figsize=(15.8, 9.5))
    grid = fig.add_gridspec(
        2,
        7,
        left=0.045,
        right=0.985,
        bottom=0.055,
        top=0.905,
        height_ratios=[0.88, 1.12],
        hspace=0.34,
        wspace=0.62,
    )
    axes = {
        "a": fig.add_subplot(grid[0, :2]),
        "b": fig.add_subplot(grid[0, 2:5]),
        "c": fig.add_subplot(grid[0, 5:]),
        "d": fig.add_subplot(grid[1, :4]),
        "e": fig.add_subplot(grid[1, 4:]),
    }
    drawers = {
        "a": _draw_fig5a,
        "b": _fig5_landscape,
        "c": _fig5_bump,
        "d": _draw_fig5d,
        "e": _draw_fig5e,
    }
    groups = {
        panel: drawers[panel](axis, bundle) or [axis]
        for panel, axis in axes.items()
    }
    figure_title(
        fig,
        "Fig. 5 | Retrospective historical-window tests of frontier ranking",
        "Three ordered publication windows preserve heterogeneous and negative "
        "results; the current OOF artifacts do not impose a D5 label-maturity embargo.",
    )
    return _export_compact(5, fig, groups, figure_dir, formats, dpi)


# ============================================================================
# Fig.6
# ============================================================================


def _domain_label(value: str) -> str:
    """Create compact labels for the 12-domain robustness panel."""
    replacements = {
        "astronomy_space": "Astronomy & space",
        "chemistry": "Chemistry",
        "clinical_health": "Clinical & health",
        "computer_science_ai": "Computer science & AI",
        "earth_climate_environment": "Earth, climate & environment",
        "ecology_evolution_microbiology": "Ecology, evolution & microbiology",
        "engineering_energy": "Engineering & energy",
        "life_molecular": "Life & molecular",
        "materials_nanoscience": "Materials & nanoscience",
        "mathematics_statistics": "Mathematics & statistics",
        "neuroscience": "Neuroscience",
        "physics": "Physics",
    }
    return replacements.get(str(value), str(value).replace("_", " ").title())


def _fig6_domains(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Show whether the innovation indicators help across all 12 domains."""
    panel_title(ax, "a", "Final 8 + K1 improves D5 ranking across all 12 domains")
    data = bundle.tables["registered_domain_metrics"].copy()
    data = data.loc[
        data["model_id"].isin(["k1_controls", "final_innovation_plus_k1"])
    ]
    pivot = data.pivot(
        index="domain12",
        columns="model_id",
        values="spearman_expected",
    ).dropna()
    pivot["gain"] = (
        pivot["final_innovation_plus_k1"] - pivot["k1_controls"]
    )
    pivot = pivot.sort_values("final_innovation_plus_k1")
    y_values = np.arange(len(pivot))
    for y_value, (domain, row) in enumerate(pivot.iterrows()):
        ax.plot(
            [row["k1_controls"], row["final_innovation_plus_k1"]],
            [y_value, y_value],
            color=LIGHT_BLUE,
            linewidth=2.0,
        )
        ax.scatter(
            row["k1_controls"],
            y_value,
            s=22,
            facecolor=WHITE,
            edgecolor=BLUE,
            linewidth=0.9,
            zorder=3,
        )
        ax.scatter(
            row["final_innovation_plus_k1"],
            y_value,
            s=28,
            color=ORANGE,
            edgecolor=WHITE,
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            max(row["k1_controls"], row["final_innovation_plus_k1"]) + 0.009,
            y_value,
            f"{row['gain']:+.3f}",
            ha="left",
            va="center",
            fontsize=4.9,
            color=INK,
        )
    ax.set_yticks(
        y_values,
        [_domain_label(value) for value in pivot.index],
        fontsize=5.1,
    )
    ax.set_xlabel("Within-domain D5 OOF Spearman")
    ax.set_xlim(
        min(0.44, float(data["spearman_expected"].min()) - 0.03),
        max(0.87, float(data["spearman_expected"].max()) + 0.08),
    )
    clean_axes(ax, grid_axis="x")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=WHITE,
            markeredgecolor=BLUE,
            label="K1 controls",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=ORANGE,
            markeredgecolor=WHITE,
            label="Final 8 + K1",
        ),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=5.5, loc="lower right")
    return [ax]


def _fig6_doses(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Put the exact deletion-dose experiment in the main figure."""
    panel_title(ax, "b", "Exact reference deletion reveals indicator-specific tolerance")
    ax.set_axis_off()
    data = bundle.tables["reference_dose_stability"].copy()
    feature_order = list(FEATURE_LABELS)
    child_axes: list[plt.Axes] = [ax]
    for index, feature in enumerate(feature_order):
        row, column = divmod(index, 2)
        inset = ax.inset_axes(
            [0.04 + column * 0.49, 0.73 - row * 0.22, 0.43, 0.17]
        )
        child_axes.append(inset)
        group = data.loc[data["code_name"].eq(feature)].copy()
        summary = (
            group.groupby("reference_retention")["spearman"]
            .agg(["median", "min", "max"])
            .reset_index()
            .sort_values("reference_retention")
        )
        inset.fill_between(
            summary["reference_retention"],
            summary["min"],
            summary["max"],
            color=LIGHT_BLUE,
            alpha=0.30,
            linewidth=0,
        )
        inset.plot(
            summary["reference_retention"],
            summary["median"],
            color=BLUE,
            linewidth=1.1,
        )
        inset.scatter(
            summary["reference_retention"],
            summary["median"],
            s=14,
            color=BLUE,
            edgecolor=WHITE,
            linewidth=0.35,
            zorder=3,
        )
        inset.axhline(
            0.90,
            color=VERMILLION,
            linestyle="--",
            linewidth=0.6,
        )
        inset.set_xlim(0.06, 1.04)
        inset.set_ylim(
            min(0.0, float(summary["min"].min()) - 0.05),
            1.02,
        )
        inset.set_xticks([0.1, 0.5, 1.0])
        inset.set_yticks([0.0, 0.5, 0.9, 1.0])
        inset.set_title(
            textwrap.fill(FEATURE_LABELS[feature], 25),
            loc="left",
            fontsize=5.3,
            pad=1.5,
            color=INK,
        )
        if row == 3:
            inset.set_xlabel("references retained", fontsize=4.7)
        else:
            inset.set_xticklabels([])
        if column == 0:
            inset.set_ylabel("ρ vs full", fontsize=4.7)
        else:
            inset.set_yticklabels([])
        inset.tick_params(labelsize=4.3, length=2)
        clean_axes(inset, grid_axis="y")
    sample_n = int(bundle.tables["audit_sample_by_domain"]["n_papers"].sum())
    ax.text(
        0.99,
        0.01,
        f"N={sample_n:,} stratified papers · 20 deterministic repetitions per non-full dose · "
        "band = observed min–max",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.9,
        color=GRAY,
    )
    return child_axes


def _fig6_time(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Separate horizon robustness from publication-period drift."""
    panel_title(ax, "c", "Horizon-stable, but later folds are harder")
    ax.set_axis_off()
    horizon_ax = ax.inset_axes([0.10, 0.57, 0.84, 0.30])
    fold_ax = ax.inset_axes([0.10, 0.12, 0.84, 0.32])
    horizons = bundle.tables["registered_horizon_metrics"].copy()
    for model_id, color, marker, label in (
        ("k1_controls", BLUE, "o", "K1"),
        ("final_innovation_plus_k1", ORANGE, "s", "Final 8 + K1"),
    ):
        group = horizons.loc[horizons["model_id"].eq(model_id)].sort_values(
            "horizon"
        )
        horizon_ax.plot(
            group["horizon"],
            group["spearman_expected"],
            color=color,
            linewidth=1.2,
            marker=marker,
            markersize=4,
            label=label,
        )
    horizon_ax.set_xticks([3, 5, 8], ["D3", "D5", "D8"])
    horizon_ax.set_ylabel("Global ρ")
    horizon_ax.set_ylim(0.64, 0.80)
    horizon_ax.legend(frameon=False, fontsize=5.2, ncol=2)
    clean_axes(horizon_ax, grid_axis="y")

    folds = bundle.tables["registered_fold_metrics"].copy()
    for model_id, color, marker in (
        ("k1_controls", BLUE, "o"),
        ("final_innovation_plus_k1", ORANGE, "s"),
    ):
        group = folds.loc[folds["model_id"].eq(model_id)].sort_values(
            "outer_fold_id"
        )
        fold_ax.plot(
            group["outer_fold_id"],
            group["spearman_expected"],
            color=color,
            linewidth=1.1,
            marker=marker,
            markersize=3.8,
        )
    fold_ax.set_xticks(range(1, 7), [f"F{index}" for index in range(1, 7)])
    fold_ax.set_xlabel("Publication-year OOF fold")
    fold_ax.set_ylabel("Fold ρ")
    fold_ax.set_ylim(0.45, 0.88)
    clean_axes(fold_ax, grid_axis="y")
    return [ax, horizon_ax, fold_ax]


def _fig6_specs(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Compare current v6.1 specifications without legacy proxy points."""
    panel_title(ax, "d", "Current specifications preserve the main ranking result")
    data = bundle.tables["specification_curve"].copy()
    data = data.loc[data["scope"].eq("current_v6_1")].sort_values(
        "spearman",
        ascending=True,
    )
    y_values = np.arange(len(data))
    for y_value, row in zip(y_values, data.itertuples(index=False)):
        primary = row.specification == "final_innovation_plus_k1"
        color = ORANGE if primary else BLUE
        ax.plot(
            [0.62, row.spearman],
            [y_value, y_value],
            color=LIGHT_ORANGE if primary else LIGHT_BLUE,
            linewidth=1.5,
        )
        ax.scatter(
            row.spearman,
            y_value,
            s=38 if primary else 24,
            color=color,
            edgecolor=WHITE,
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            float(row.spearman) + 0.004,
            y_value,
            f"{float(row.spearman):.3f}",
            ha="left",
            va="center",
            fontsize=5.2,
            color=INK,
        )
    ax.set_yticks(
        y_values,
        [_short(value.replace("_", " "), 28) for value in data["specification"]],
        fontsize=5.2,
    )
    ax.set_xlim(0.62, 0.79)
    ax.set_xlabel("D5 temporal-OOF Spearman")
    clean_axes(ax, grid_axis="x")
    ax.text(
        0.98,
        0.03,
        "Legacy cached-score proxies are excluded from this comparison.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=GRAY,
    )
    return [ax]


def _fig6_boundary(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Show data adequacy rather than calling threshold failures model failures."""
    panel_title(ax, "e", "Data adequacy requires references and mapping")
    data = bundle.tables["reliability_units"].copy()
    failed = data.loc[data["reliable"].eq(0)]
    passed = data.loc[data["reliable"].eq(1)]
    ax.scatter(
        failed["median_reference_count"],
        failed["median_mapping_coverage"],
        s=9 + 18 * np.sqrt(failed["n_papers"] / failed["n_papers"].max()),
        marker="x",
        color=VERMILLION,
        alpha=0.50,
        linewidth=0.6,
        label=f"below registered gate · {len(failed)} units",
    )
    ax.scatter(
        passed["median_reference_count"],
        passed["median_mapping_coverage"],
        s=11 + 20 * np.sqrt(passed["n_papers"] / passed["n_papers"].max()),
        facecolor=WHITE,
        edgecolor=BLUE,
        linewidth=0.8,
        label=f"meets registered gate · {len(passed)} units",
    )
    x_max = float(data["median_reference_count"].quantile(0.995)) + 2
    ax.fill_between(
        [10, x_max],
        [0.8, 0.8],
        [1.02, 1.02],
        color=LIGHT_BLUE,
        alpha=0.12,
        zorder=-1,
    )
    ax.axvline(10, color=ORANGE, linestyle="--", linewidth=0.8)
    ax.axhline(0.8, color=ORANGE, linestyle="--", linewidth=0.8)
    ax.set_xlim(-1, x_max)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Median valid references in domain-year unit")
    ax.set_ylabel("Median source/field mapping coverage")
    ax.legend(frameon=False, fontsize=5.0, loc="lower right")
    clean_axes(ax, grid_axis="both")
    ax.text(
        0.02,
        0.98,
        "Gate also requires ≥70% complete core-8 features",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.1,
        color=GRAY,
    )
    return [ax]


def _fig6_failures(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Expose the diagnostic failure taxonomy without promoting it to a rate claim."""
    panel_title(ax, "f", "Failure diagnostics point to safeguards")
    ax.set_axis_off()
    modes = bundle.tables["failure_modes"].sort_values(
        ["count", "failure_mode"],
        ascending=[False, True],
    )
    cases = bundle.tables["failure_cases"].head(2)
    left = ax.inset_axes([0.02, 0.12, 0.42, 0.76])
    y_values = np.arange(len(modes))[::-1]
    left.hlines(
        y_values,
        0,
        modes["count"],
        color=LIGHT_BLUE,
        linewidth=1.6,
    )
    left.scatter(modes["count"], y_values, s=27, color=BLUE)
    left.set_yticks(
        y_values,
        [_short(value, 21) for value in modes["failure_mode"]],
        fontsize=4.9,
    )
    left.set_xlabel("Heuristic flags in 50 cached cases")
    clean_axes(left, grid_axis="x")
    left.text(
        0.98,
        0.98,
        "diagnostic · no blind adjudication",
        transform=left.transAxes,
        ha="right",
        va="top",
        fontsize=4.5,
        color=VERMILLION,
    )
    for index, row in enumerate(cases.itertuples(index=False)):
        bottom = 0.54 - index * 0.39
        ax.add_patch(
            FancyBboxPatch(
                (0.49, bottom),
                0.49,
                0.31,
                boxstyle="round,pad=0.009,rounding_size=0.015",
                transform=ax.transAxes,
                facecolor="#FFF8F1",
                edgecolor=LIGHT_ORANGE,
                linewidth=0.8,
            )
        )
        ax.text(
            0.515,
            bottom + 0.255,
            _short(row.title, 54),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.3,
            color=INK,
            fontweight="bold",
        )
        ax.text(
            0.515,
            bottom + 0.155,
            f"{_short(row.failure_modes, 62)}\n"
            f"Safeguard: {_short(row.recommended_safeguard, 62)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=4.8,
            color=GRAY,
            linespacing=1.1,
        )
    return [ax, left]


def render_fig6_compact(
    bundle: FigureBundle,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render the registered robustness evidence in one six-panel figure."""
    configure_style()
    _clean_generated_artifacts(6, figure_dir)
    fig = plt.figure(figsize=(15.8, 10.1))
    grid = fig.add_gridspec(
        2,
        6,
        left=0.050,
        right=0.985,
        bottom=0.055,
        top=0.905,
        hspace=0.34,
        wspace=0.65,
    )
    axes = {
        "a": fig.add_subplot(grid[0, :2]),
        "b": fig.add_subplot(grid[0, 2:4]),
        "c": fig.add_subplot(grid[0, 4:]),
        "d": fig.add_subplot(grid[1, :2]),
        "e": fig.add_subplot(grid[1, 2:4]),
        "f": fig.add_subplot(grid[1, 4:]),
    }
    drawers = {
        "a": _fig6_domains,
        "b": _fig6_doses,
        "c": _fig6_time,
        "d": _fig6_specs,
        "e": _fig6_boundary,
        "f": _fig6_failures,
    }
    groups = {
        panel: drawers[panel](axis, bundle)
        for panel, axis in axes.items()
    }
    figure_title(
        fig,
        "Fig. 6 | Robustness evidence identifies stable and unreliable regions",
        "Registered v6.1 results are separated from legacy score proxies; the "
        "main perturbation panel recomputes all eight indicators after actual reference deletion.",
    )
    return _export_compact(6, fig, groups, figure_dir, formats, dpi)


# ============================================================================
# Fig.7
# ============================================================================


def _fig7_profiles(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Show four venue profiles on one common five-angle coordinate system."""
    panel_title(ax, "d", "Five-angle venue profiles")
    ax.set_axis_off()
    data = bundle.tables["venue_angle_profiles"].copy()
    families = data["analysis_venue_family"].drop_duplicates().tolist()
    child_axes: list[plt.Axes] = [ax]
    bottoms = np.linspace(0.72, 0.12, len(families))
    for index, (family, bottom) in enumerate(zip(families, bottoms)):
        inset = ax.inset_axes([0.27, bottom, 0.70, 0.145])
        child_axes.append(inset)
        group = (
            data.loc[data["analysis_venue_family"].eq(family)]
            .set_index("angle_id")
            .reindex(ANGLE_ORDER)
        )
        values = group["mean_percentile"].to_numpy(float)
        color = VENUE_COLORS.get(str(family), BLUE)
        inset.axhline(0.50, color=LIGHT_GRAY, linestyle=":", linewidth=0.7)
        inset.plot(
            range(1, 6),
            values,
            color=color,
            linewidth=1.2,
        )
        inset.scatter(
            range(1, 6),
            values,
            s=18,
            color=color,
            edgecolor=WHITE,
            linewidth=0.4,
            zorder=3,
        )
        inset.set_xlim(0.75, 5.25)
        inset.set_ylim(0.42, 0.59)
        inset.set_yticks([0.45, 0.50, 0.55])
        inset.tick_params(axis="y", labelsize=4.1, length=2)
        if index == len(families) - 1:
            inset.set_xticks(
                range(1, 6),
                ["A1", "A2", "A3", "A4", "A5"],
                fontsize=4.7,
            )
        else:
            inset.set_xticks([])
        clean_axes(inset, grid_axis="y")
        ax.text(
            0.24,
            bottom + 0.073,
            VENUE_DISPLAY_LABELS.get(family, family).replace("\n", " "),
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=5.0,
            color=INK,
        )
    ax.text(
        0.99,
        0.02,
        "A1 rarity · A2 atypicality · A3 first-time · A4 breadth · A5 integration · "
        "same field-year percentile scale",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.6,
        color=GRAY,
    )
    return child_axes


def _fig7_association(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Show publication-time/future association inside each venue family."""
    panel_title(ax, "e", "Innovation signals rank D5 within venues")
    data = bundle.tables["venue_within_association"].sort_values(
        "spearman",
        ascending=True,
    )
    y_values = np.arange(len(data))
    for y_value, row in zip(y_values, data.itertuples(index=False)):
        color = VENUE_COLORS.get(str(row.analysis_venue_family), BLUE)
        ax.plot(
            [row.ci_low, row.ci_high],
            [y_value, y_value],
            color=color,
            alpha=0.42,
            linewidth=2.0,
        )
        ax.scatter(
            row.spearman,
            y_value,
            s=37,
            color=color,
            edgecolor=WHITE,
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            float(row.ci_high) + 0.012,
            y_value,
            f"{float(row.spearman):.3f} · n={int(row.n_papers):,}",
            ha="left",
            va="center",
            fontsize=5.2,
            color=INK,
        )
    ax.axvline(0, color=LIGHT_GRAY, linewidth=0.8)
    ax.set_yticks(
        y_values,
        [
            VENUE_DISPLAY_LABELS.get(value, value).replace("\n", " ")
            for value in data["analysis_venue_family"]
        ],
        fontsize=5.5,
    )
    ax.set_xlim(0.35, 0.86)
    ax.set_xlabel("Within-venue Spearman · domain-year percentiles")
    clean_axes(ax, grid_axis="x")
    ax.text(
        0.99,
        0.02,
        "95% fixed-rank domain-year cluster-bootstrap interval",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=GRAY,
    )
    return [ax]


def _fig7_support(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Expose covariate imbalance rather than asserting common support."""
    panel_title(ax, "f", "Covariate imbalance limits venue comparisons")
    ax.set_axis_off()
    data = bundle.tables["venue_common_support_audit"].copy()
    families = data["analysis_venue_family"].tolist()
    measures = [
        ("year_mean", "Mean year", "{:.0f}"),
        ("reference_median", "Median refs", "{:.0f}"),
        ("log_author_mean", "Mean log authors", "{:.2f}"),
        ("log_country_mean", "Mean log countries", "{:.2f}"),
        ("domain_count", "Domains", "{:.0f}"),
    ]
    child_axes: list[plt.Axes] = [ax]
    for index, (column, label, number_format) in enumerate(measures):
        inset = ax.inset_axes([0.23 + index * 0.15, 0.18, 0.125, 0.68])
        child_axes.append(inset)
        values = pd.to_numeric(data[column], errors="coerce")
        y_values = np.arange(len(data))[::-1]
        color = BLUE if index % 2 == 0 else ORANGE
        inset.scatter(
            values,
            y_values,
            s=24,
            color=color,
            edgecolor=WHITE,
            linewidth=0.45,
        )
        for x_value, y_value in zip(values, y_values):
            inset.annotate(
                number_format.format(float(x_value)),
                (float(x_value), float(y_value)),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=4.2,
                color=INK,
            )
        span = float(values.max() - values.min())
        padding = max(span * 0.15, 0.05)
        inset.set_xlim(float(values.min()) - padding, float(values.max()) + padding)
        inset.set_ylim(-0.55, len(data) - 0.30)
        inset.set_yticks([])
        inset.set_xticks([])
        inset.set_xlabel(label, fontsize=4.7)
        clean_axes(inset, grid_axis="x")
    y_values = np.arange(len(data))[::-1]
    for y_value, family in zip(y_values, families):
        ax.text(
            0.20,
            0.18 + 0.68 * (y_value + 0.5) / len(data),
            VENUE_DISPLAY_LABELS.get(family, family).replace("\n", " "),
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=5.2,
            color=INK,
        )
    integrity_problem = bool(
        np.allclose(
            data["log_author_mean"].to_numpy(float),
            data["log_institution_mean"].to_numpy(float),
            equal_nan=True,
        )
    )
    warning = (
        "Frozen author and institution counts are identical; institution count is withheld."
        if integrity_problem
        else "Team and institution metadata pass the frozen consistency audit."
    )
    ax.text(
        0.99,
        0.03,
        warning + " These imbalances preclude a venue-causal interpretation.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.9,
        color=VERMILLION,
    )
    return child_axes


def render_fig7_compact(
    bundle: FigureBundle,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render venue portfolio selection patterns in one compact figure."""
    configure_style()
    _clean_generated_artifacts(7, figure_dir)
    fig = plt.figure(figsize=(15.8, 9.7))
    grid = fig.add_gridspec(
        2,
        6,
        left=0.050,
        right=0.985,
        bottom=0.055,
        top=0.905,
        hspace=0.36,
        wspace=0.66,
    )
    axes = {
        "a": fig.add_subplot(grid[0, :2]),
        "b": fig.add_subplot(grid[0, 2:4]),
        "c": fig.add_subplot(grid[0, 4:]),
        "d": fig.add_subplot(grid[1, :2]),
        "e": fig.add_subplot(grid[1, 2:4]),
        "f": fig.add_subplot(grid[1, 4:]),
    }
    _draw_fig7a(axes["a"], bundle)
    _draw_fig7b(axes["b"], bundle)
    _draw_fig7c(axes["c"], bundle)
    groups: Dict[str, Sequence[plt.Axes]] = {
        "a": [axes["a"]],
        "b": [axes["b"]],
        "c": [axes["c"]],
        "d": _fig7_profiles(axes["d"], bundle),
        "e": _fig7_association(axes["e"], bundle),
        "f": _fig7_support(axes["f"], bundle),
    }
    figure_title(
        fig,
        "Fig. 7 | Nature Portfolio venues carry different innovation profiles",
        "The score excludes venue and is normalized within field-year; portfolio "
        "differences and within-venue associations describe selection, not journal effects.",
    )
    return _export_compact(7, fig, groups, figure_dir, formats, dpi)


RENDERERS = {
    3: render_fig3_compact,
    4: render_fig4_compact,
    5: render_fig5_compact,
    6: render_fig6_compact,
    7: render_fig7_compact,
}


def render_fig3_to_fig7(
    figure_id: int,
    bundle: FigureBundle,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Dispatch one of the compact Fig.3--Fig.7 renderers."""
    if figure_id not in RENDERERS:
        raise ValueError(f"Unsupported compact renderer: Fig.{figure_id}")
    np.random.seed(20260725)
    return RENDERERS[figure_id](bundle, figure_dir, formats, dpi)
