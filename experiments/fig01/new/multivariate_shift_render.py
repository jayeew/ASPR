"""Render Fig. 1 with multivariate displacement and dimension contributions."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .descriptive_contract import STAGE_KEYS
from .descriptive_render import (
    GAINED,
    GRID_GREY,
    INK,
    LANDMARK,
    LOST,
    MID_GREY,
    NAVY,
    RETAINED,
    SKELETON,
    WHITE,
    _accessibility_previews,
    _draw_network,
    _figure_size,
    _row_domain_label,
    _save_bundle,
    _set_style,
    _stage_progression_arrows,
    _topic_colors,
)
from .event_data import canonical_hash, write_json


SHIFT = "#315F8C"
OBSERVED_BAR = "#9CB3C8"
PLACEBO = "#C8D0D5"
PLACEBO_DARK = "#7C8790"
DIMENSION_COLORS: Mapping[str, str] = {
    "CD029": "#C67722",
    "CD031": "#7B5AA6",
    "CD032": "#315F8C",
}
DIMENSION_LEGEND: Mapping[str, str] = {
    "CD029": "Interdisciplinary integration",
    "CD031": "Knowledge diversity",
    "CD032": "Concept emergence",
}


def _header(figure: Figure, outer: GridSpec) -> None:
    network = figure.add_subplot(outer[0, 0])
    network.set_axis_off()
    network.text(
        0.0, 0.80, "a", fontsize=8.0, fontweight="bold", va="center"
    )
    network.text(
        0.055,
        0.80,
        "Topic-coupling transitions",
        fontsize=6.2,
        fontweight="bold",
        color=NAVY,
        va="center",
    )
    for index, label in enumerate(
        ("t−6 to t−1", "through t+2", "through t+5", "through t+8")
    ):
        network.text(
            (0.62 + index + 0.5) / 4.62,
            0.16,
            label,
            fontsize=5.3,
            fontweight="bold" if index == 1 else "normal",
            color=GAINED if index == 1 else INK,
            ha="center",
            va="center",
        )
    integrated = figure.add_subplot(outer[0, 1:3])
    integrated.set_axis_off()
    integrated.text(
        0.0, 0.80, "b", fontsize=8.0, fontweight="bold", va="center"
    )
    integrated.text(
        0.075,
        0.80,
        "Displacement and dimension contribution",
        fontsize=5.8,
        fontweight="bold",
        color=NAVY,
        va="center",
    )


def _displacement_limit(displacement: pd.DataFrame) -> float:
    values = displacement[
        ["ci_high", "placebo_high", "displacement_pp"]
    ].to_numpy(dtype=float)
    maximum = float(np.nanmax(values))
    return max(20.0, math.ceil(maximum / 5.0) * 5.0)


def _integrated_panel(
    axis: Axes,
    displacement: pd.DataFrame,
    contributions: pd.DataFrame,
    *,
    x_limit: float,
    show_x: bool,
) -> None:
    rows = displacement.loc[
        displacement["stage_index"].gt(0)
    ].sort_values("stage_index", kind="stable")
    stage_y = {1: 2.55, 2: 1.55, 3: 0.55}
    axis.axhspan(2.13, 2.94, color="#F7E9CB", alpha=0.72, lw=0)
    axis.axvline(0, color=RETAINED, linewidth=0.55)
    for guide in np.arange(10, x_limit + 0.1, 10):
        axis.axvline(
            guide,
            color=GRID_GREY,
            linewidth=0.35,
            linestyle=(0, (1.5, 2.0)),
        )
    for row in rows.itertuples(index=False):
        stage_index = int(row.stage_index)
        y = stage_y[stage_index]
        observed = float(row.displacement_pp)
        axis.plot(
            [float(row.placebo_low), float(row.placebo_high)],
            [y + 0.17, y + 0.17],
            color=PLACEBO,
            linewidth=4.0,
            solid_capstyle="round",
            zorder=1,
        )
        axis.plot(
            [float(row.placebo_median), float(row.placebo_median)],
            [y + 0.09, y + 0.25],
            color=PLACEBO_DARK,
            linewidth=0.8,
            zorder=2,
        )
        if stage_index < 3:
            axis.barh(
                [y - 0.10],
                [observed],
                height=0.25,
                color=OBSERVED_BAR,
                edgecolor=WHITE,
                linewidth=0.45,
                zorder=2,
            )
        else:
            stage_contributions = contributions.loc[
                contributions["stage_index"].eq(stage_index)
            ].sort_values("dimension_id", kind="stable")
            left = 0.0
            for contribution in stage_contributions.itertuples(index=False):
                dimension = str(contribution.dimension_id)
                share = float(contribution.contribution_share)
                width = observed * share
                axis.barh(
                    [y - 0.10],
                    [width],
                    left=[left],
                    height=0.25,
                    color=DIMENSION_COLORS[dimension],
                    edgecolor=WHITE,
                    linewidth=0.45,
                    zorder=2,
                )
                if share >= 0.20 and width >= 2.5:
                    axis.text(
                        left + width / 2.0,
                        y - 0.10,
                        f"{100.0 * share:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=3.55,
                        color=WHITE,
                        fontweight="bold",
                        zorder=4,
                    )
                left += width
        axis.plot(
            [float(row.ci_low), float(row.ci_high)],
            [y - 0.10, y - 0.10],
            color=SHIFT,
            linewidth=1.1,
            solid_capstyle="round",
            zorder=3,
        )
        for cap in (float(row.ci_low), float(row.ci_high)):
            axis.plot(
                [cap, cap],
                [y - 0.17, y - 0.03],
                color=SHIFT,
                linewidth=0.7,
                zorder=3,
            )
        axis.scatter(
            [observed],
            [y - 0.10],
            s=13,
            marker="D",
            facecolor=WHITE,
            edgecolor=SHIFT,
            linewidth=0.7,
            zorder=4,
        )
        axis.text(
            observed + 0.45,
            y - 0.10,
            f"{observed:.1f}",
            fontsize=4.1,
            color=SHIFT,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=5,
        )
    axis.set_xlim(-0.60, x_limit + 1.55)
    axis.set_ylim(0.02, 3.05)
    axis.set_yticks([stage_y[index] - 0.10 for index in (1, 2, 3)])
    axis.set_yticklabels(["LM", "Early", "Late"])
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_visible(show_x)
    axis.tick_params(axis="y", length=0, pad=1.6)
    axis.tick_params(axis="x", length=1.8, width=0.45, pad=1.0)
    ticks = np.arange(0, x_limit + 0.1, 10)
    axis.set_xticks(ticks)
    if show_x:
        axis.set_xticklabels([str(int(value)) for value in ticks])
    else:
        axis.set_xticklabels([])


def _main_figure(
    config: Mapping[str, Any],
    selection: pd.DataFrame,
    nodes: pd.DataFrame,
    transitions: pd.DataFrame,
    representatives: pd.DataFrame,
    summaries: pd.DataFrame,
    displacement: pd.DataFrame,
    contributions: pd.DataFrame,
) -> Figure:
    selected = selection.loc[selection["selected"].astype(bool)].sort_values(
        "selection_rank", kind="stable"
    )
    figure = plt.figure(figsize=_figure_size(config, "main"))
    outer = GridSpec(
        5,
        3,
        figure=figure,
        width_ratios=[0.585, 0.250, 0.165],
        height_ratios=[0.16, 1, 1, 1, 1],
        left=0.018,
        right=0.992,
        top=0.992,
        bottom=0.073,
        hspace=0.105,
        wspace=0.075,
    )
    _header(figure, outer)
    x_limit = _displacement_limit(displacement)
    for row_index, row in enumerate(selected.to_dict("records"), start=1):
        domain = str(row["domain"])
        domain_nodes = nodes.loc[nodes["domain"].eq(domain)]
        domain_transitions = transitions.loc[
            transitions["domain"].eq(domain)
        ]
        domain_representatives = representatives.loc[
            representatives["domain"].eq(domain)
        ]
        domain_summaries = summaries.loc[summaries["domain"].eq(domain)]
        network_grid = GridSpecFromSubplotSpec(
            1,
            5,
            subplot_spec=outer[row_index, 0],
            width_ratios=[0.50, 1, 1, 1, 1],
            wspace=0.040,
        )
        label_axis = figure.add_subplot(network_grid[0, 0])
        _row_domain_label(label_axis, pd.Series(row))
        colors = _topic_colors(domain_nodes)
        network_axes: List[Axes] = []
        for stage_index in range(len(STAGE_KEYS)):
            axis = figure.add_subplot(network_grid[0, stage_index + 1])
            network_axes.append(axis)
            summary = domain_summaries.loc[
                domain_summaries["stage_index"].eq(stage_index)
            ].iloc[0]
            _draw_network(
                axis,
                nodes=domain_nodes,
                transitions=domain_transitions,
                representatives=domain_representatives,
                summary=summary,
                view="main",
                topic_colors=colors,
                graph_settings=config["graph"],
                compact=True,
            )
        if bool(config["graph"]["show_stage_progression_arrows"]):
            _stage_progression_arrows(figure, network_axes, compact=True)
        show_x = row_index == len(selected)
        integrated_axis = figure.add_subplot(outer[row_index, 1:3])
        domain_displacement = displacement.loc[
            displacement["domain"].eq(domain)
        ]
        _integrated_panel(
            integrated_axis,
            domain_displacement,
            contributions.loc[contributions["domain"].eq(domain)],
            x_limit=x_limit,
            show_x=show_x,
        )
    graph_legend = [
        Line2D([0], [0], color=RETAINED, lw=1.1, label="retained"),
        Line2D([0], [0], color=GAINED, lw=1.35, label="gained"),
        Line2D(
            [0],
            [0],
            color=LOST,
            lw=1.2,
            linestyle=(0, (2.5, 1.8)),
            label="lost",
        ),
        Line2D(
            [0],
            [0],
            color=SKELETON,
            lw=0.8,
            linestyle=(0, (1, 1.7)),
            label="union skeleton",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor=LANDMARK,
            markeredgecolor=WHITE,
            markersize=6,
            label="landmark topic",
        ),
    ]
    figure.legend(
        handles=graph_legend,
        loc="lower left",
        bbox_to_anchor=(0.018, 0.030),
        ncol=5,
        frameon=False,
        fontsize=4.6,
        handlelength=1.8,
        columnspacing=0.9,
        borderaxespad=0,
    )
    dimension_legend = [
        Patch(
            facecolor=DIMENSION_COLORS[dimension],
            edgecolor=WHITE,
            linewidth=0.4,
            label=label,
        )
        for dimension, label in DIMENSION_LEGEND.items()
    ]
    uncertainty_legend = [
        Line2D(
            [0],
            [0],
            color=SHIFT,
            lw=1.1,
            marker="D",
            markerfacecolor=WHITE,
            markeredgecolor=SHIFT,
            markeredgewidth=0.6,
            markersize=3.4,
            label="observed · 95% CI",
        ),
        Line2D(
            [0],
            [0],
            color=PLACEBO,
            lw=3.8,
            marker="|",
            markeredgecolor=PLACEBO_DARK,
            markeredgewidth=0.8,
            markersize=5.0,
            label="placebo 5–95%",
        ),
    ]
    figure.legend(
        handles=uncertainty_legend,
        loc="lower right",
        bbox_to_anchor=(0.992, 0.045),
        ncol=2,
        frameon=False,
        fontsize=3.65,
        handlelength=1.8,
        columnspacing=0.9,
        borderaxespad=0,
    )
    figure.legend(
        handles=dimension_legend,
        loc="lower right",
        bbox_to_anchor=(0.992, 0.029),
        ncol=3,
        frameon=False,
        fontsize=3.65,
        handlelength=0.9,
        columnspacing=0.7,
        borderaxespad=0,
    )
    return figure


def render_multivariate_shift_figure(
    config: Mapping[str, Any],
    output_dir: Path,
) -> Mapping[str, Any]:
    """Render the multivariate Fig. 1 candidate and accessibility previews."""
    _set_style(config)
    panel_data = output_dir / "panel_data"
    selection = pd.read_csv(panel_data / "domain_selection.csv")
    nodes = pd.read_parquet(panel_data / "snapshot_nodes.parquet")
    transitions = pd.read_parquet(panel_data / "transition_edges.parquet")
    representatives = pd.read_parquet(
        panel_data / "representative_papers.parquet"
    )
    summaries = pd.read_csv(panel_data / "snapshot_summary.csv")
    displacement = pd.read_csv(
        panel_data / "multivariate_stage_displacement.csv"
    )
    contributions = pd.read_csv(
        panel_data / "multivariate_dimension_contributions.csv"
    )
    figure = _main_figure(
        config,
        selection,
        nodes,
        transitions,
        representatives,
        summaries,
        displacement,
        contributions,
    )
    artifacts, layout_qa = _save_bundle(
        figure,
        output_dir / "figure_full_multivariate_shift",
        dpi=int(config["plot"]["dpi"]),
    )
    plt.close(figure)
    accessibility = _accessibility_previews(
        Path(artifacts["png"]["path"]),
        output_dir / "qa_multivariate_shift",
    )
    manifest: Dict[str, Any] = {
        "artifact_kind": "fig1_multivariate_shift_render",
        "design_version": "fig1-multivariate-shift-v8.3",
        "chart_type": "decomposed_bullet_forest",
        "graph_panels_changed": False,
        "displacement_encoding": (
            "stacked bar endpoint is the dimension-equal RMS shift in "
            "within-year percentile points; stages are relative to t-6:t-1"
        ),
        "uncertainty_encoding": (
            "year-stratified 95% bootstrap CI and 5-95% placebo interval"
        ),
        "contribution_encoding": (
            "Late bar colour fractions are dimension shares of squared "
            "displacement; LM and Early bars encode total displacement only"
        ),
        "bar_segment_semantics": (
            "Late segment widths equal observed total displacement multiplied "
            "by the squared-displacement share; they encode composition and "
            "are not additive dimension-specific percentile-point effects"
        ),
        "artifacts": artifacts,
        "layout_qa": layout_qa,
        "accessibility_previews": accessibility,
    }
    manifest["artifact_id"] = canonical_hash(manifest)
    write_json(output_dir / "render_manifest_multivariate.json", manifest)
    return manifest


__all__ = [
    "DIMENSION_COLORS",
    "DIMENSION_LEGEND",
    "render_multivariate_shift_figure",
]
