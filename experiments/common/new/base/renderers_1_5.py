"""Deterministic renderers for Fig.1–Fig.5."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Wedge
from matplotlib.path import Path as MplPath
from scipy.stats import gaussian_kde

from experiments.common.new.base.common import (
    ANGLE_COLORS,
    ANGLE_LABELS,
    ANGLE_ORDER,
    ANGLE_SHORT,
    BLUE,
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
    export_figure,
    figure_title,
    panel_title,
)


PanelDrawer = Callable[[Axes, FigureBundle], None]


def _short(value: Any, width: int = 26) -> str:
    """Wrap and truncate one figure label."""
    text = str(value)
    if len(text) > width:
        text = text[: max(width - 1, 1)].rstrip() + "…"
    return "\n".join(textwrap.wrap(text, width=max(width // 2, 8)))


def _panel_outputs(
    bundle: FigureBundle,
    drawers: Mapping[str, PanelDrawer],
    figure_dir: Path,
    *,
    dpi: int,
) -> Dict[str, Path]:
    """Render every panel as a standalone audit artifact."""
    panel_dir = figure_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}
    for panel, drawer in drawers.items():
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        drawer(ax, bundle)
        fig.tight_layout()
        paths = export_figure(
            fig,
            panel_dir / f"fig{bundle.figure_id:02d}_{panel}",
            formats=("png", "svg"),
            dpi=dpi,
        )
        plt.close(fig)
        outputs.update({f"panel_{panel}_{key}": value for key, value in paths.items()})
    return outputs


def _finish_composite(
    fig: Figure,
    bundle: FigureBundle,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Export a composed figure and return output paths."""
    fig.subplots_adjust(top=0.91)
    outputs = export_figure(
        fig,
        figure_dir / f"fig{bundle.figure_id:02d}_full",
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)
    return {f"full_{key}": value for key, value in outputs.items()}


def _cell_transform(
    x: pd.Series,
    y: pd.Series,
    left: float,
    bottom: float,
    width: float,
    height: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform normalized graph coordinates into an axis cell."""
    return (
        left + (np.asarray(x, dtype=float) + 1.0) * width / 2.0,
        bottom + (np.asarray(y, dtype=float) + 1.0) * height / 2.0,
    )


# ============================================================================
# Fig.1
# ============================================================================


def _draw_fig1a(ax: Axes, bundle: FigureBundle) -> None:
    """Draw twelve fixed-layout topic-network snapshots."""
    nodes = bundle.tables["snapshot_nodes"]
    edges = bundle.tables["snapshot_edges"]
    ax.set_axis_off()
    panel_title(ax, "a", "Four landmark fields across three fixed-layout stages")
    stages = ["pre", "landmark_window", "post"]
    stage_labels = ["Before", "Landmark window", "After"]
    domains = nodes[["domain", "domain_label"]].drop_duplicates().sort_values("domain_label")
    colors = sns.color_palette("husl", max(nodes["community"].nunique(), 8))
    community_order = sorted(nodes["community"].unique())
    color_map = {community: colors[index % len(colors)] for index, community in enumerate(community_order)}
    top_margin = 0.88
    row_height = 0.205
    cell_width = 0.285
    for column, label in enumerate(stage_labels):
        ax.text(
            0.17 + column * 0.30,
            0.90,
            label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color=GRAY,
            fontsize=7.5,
            fontweight="bold",
        )
    for row, domain in enumerate(domains.itertuples(index=False)):
        bottom = top_margin - (row + 1) * row_height
        ax.text(
            0.01,
            bottom + row_height * 0.45,
            str(domain.domain_label),
            transform=ax.transAxes,
            ha="left",
            va="center",
            color=INK,
            fontsize=7,
            fontweight="bold",
        )
        for column, stage in enumerate(stages):
            left = 0.085 + column * 0.30
            stage_nodes = nodes.loc[
                nodes["domain"].eq(domain.domain) & nodes["stage"].eq(stage)
            ].copy()
            stage_edges = edges.loc[
                edges["domain"].eq(domain.domain) & edges["stage"].eq(stage)
            ].copy()
            stage_edges["weight"] = pd.to_numeric(stage_edges["weight"], errors="coerce")
            stage_edges = stage_edges.dropna(subset=["weight"]).nlargest(20, "weight")
            if stage_nodes.empty:
                continue
            x_map: Dict[int, float] = {}
            y_map: Dict[int, float] = {}
            tx, ty = _cell_transform(
                stage_nodes["x"],
                stage_nodes["y"],
                left,
                bottom,
                cell_width,
                row_height * 0.78,
            )
            for community, x_value, y_value in zip(
                stage_nodes["community"].astype(int),
                tx,
                ty,
            ):
                x_map[int(community)] = float(x_value)
                y_map[int(community)] = float(y_value)
            for edge in stage_edges.itertuples(index=False):
                source = int(edge.source_community)
                target = int(edge.target_community)
                if source not in x_map or target not in x_map:
                    continue
                ax.plot(
                    [x_map[source], x_map[target]],
                    [y_map[source], y_map[target]],
                    transform=ax.transAxes,
                    color=ORANGE if stage == "landmark_window" else LIGHT_GRAY,
                    alpha=0.35 if stage == "landmark_window" else 0.26,
                    linewidth=0.4 + 1.1 * math.log1p(float(edge.weight)) / 8,
                    zorder=1,
                )
            sizes = 5 + 19 * np.sqrt(stage_nodes["paper_count_visible"] / max(stage_nodes["paper_count_visible"].max(), 1))
            for node, x_value, y_value, size in zip(
                stage_nodes.itertuples(index=False),
                tx,
                ty,
                sizes,
            ):
                marker = "*" if int(node.is_landmark_community) else "o"
                ax.scatter(
                    x_value,
                    y_value,
                    s=float(size) * (1.5 if marker == "*" else 1.0),
                    marker=marker,
                    color=color_map[node.community],
                    edgecolor=WHITE,
                    linewidth=0.35,
                    transform=ax.transAxes,
                    zorder=3,
                )
            ax.add_patch(
                FancyBboxPatch(
                    (left - 0.006, bottom - 0.006),
                    cell_width + 0.012,
                    row_height * 0.80 + 0.012,
                    boxstyle="round,pad=0.003,rounding_size=0.006",
                    transform=ax.transAxes,
                    facecolor="none",
                    edgecolor=PALE_GRAY,
                    linewidth=0.7,
                )
            )
    ax.text(
        0.99,
        0.01,
        "Star = landmark topic; amber = links visible in the landmark window",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=GRAY,
        fontsize=6.5,
    )


def _draw_fig1b(ax: Axes, bundle: FigureBundle) -> None:
    """Draw CRISPR structural differences on the frozen post layout."""
    nodes = bundle.tables["snapshot_nodes"]
    nodes = nodes.loc[nodes["domain"].eq("crispr") & nodes["stage"].eq("post")].copy()
    diff = bundle.tables["structural_differences"]
    diff = diff.loc[diff["domain"].eq("crispr")].copy()
    diff["delta_weight"] = pd.to_numeric(diff["delta_weight"], errors="coerce")
    diff = diff.dropna(subset=["delta_weight"]).nlargest(28, "delta_weight")
    panel_title(ax, "b", "CRISPR post-minus-pre structural difference")
    ax.set_axis_off()
    if nodes.empty:
        return
    x = dict(zip(nodes["community"].astype(int), nodes["x"].astype(float)))
    y = dict(zip(nodes["community"].astype(int), nodes["y"].astype(float)))
    for edge in diff.itertuples(index=False):
        source, target = int(edge.source_community), int(edge.target_community)
        if source not in x or target not in x:
            continue
        is_new = bool(edge.is_new_bridge)
        ax.plot(
            [x[source], x[target]],
            [y[source], y[target]],
            color=ORANGE if is_new else BLUE,
            alpha=0.65 if is_new else 0.28,
            linewidth=0.5 + 2.6 * math.log1p(max(float(edge.delta_weight), 0)) / 10,
            zorder=1,
        )
    sizes = 16 + 70 * np.sqrt(nodes["paper_count_visible"] / max(nodes["paper_count_visible"].max(), 1))
    ax.scatter(
        nodes["x"],
        nodes["y"],
        s=sizes,
        c=np.where(nodes["is_landmark_community"].eq(1), ORANGE, LIGHT_BLUE),
        edgecolor=WHITE,
        linewidth=0.6,
        zorder=3,
    )
    labels = nodes.nlargest(4, "paper_count_visible")
    topic_list = "\n".join(
        f"• {_short(row.label, 30).replace(chr(10), ' ')}"
        for row in labels.itertuples(index=False)
    )
    ax.text(
        0.98,
        0.98,
        "Largest visible post topics\n" + topic_list,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=INK,
        bbox={"facecolor": WHITE, "edgecolor": PALE_GRAY, "boxstyle": "round,pad=0.35", "alpha": 0.92},
    )
    ax.text(
        0.02,
        0.02,
        "Amber: newly visible bridge\nBlue: intensified pre-existing bridge",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=GRAY,
        fontsize=6.8,
    )


def _draw_fig1c(ax: Axes, bundle: FigureBundle) -> None:
    """Draw aligned distributions for five transparent graph diagnostics."""
    data = bundle.tables["event_aligned_metrics"].copy()
    panel_title(ax, "c", "Event-aligned structural diagnostics")
    metric_labels = {
        "community_coverage": "Topic coverage",
        "bridge_share": "Bridge share",
        "modularity": "Modularity",
        "diffusion_reach": "Diffusion reach",
        "diffusion_evenness": "Diffusion evenness",
    }
    metrics = list(metric_labels)
    rng = np.random.default_rng(20260725)
    for row, metric in enumerate(metrics):
        subset = data.loc[data["metric"].eq(metric)]
        baseline = len(metrics) - row - 1
        ax.axhline(baseline, color=PALE_GRAY, linewidth=0.6, zorder=0)
        for period, group in subset.groupby("relative_period"):
            values = group["z_value"].dropna().to_numpy(float)
            jitter = rng.uniform(-0.11, 0.11, len(values))
            ax.scatter(
                np.full(len(values), period) + jitter,
                baseline + values * 0.10,
                s=13,
                color=ANGLE_COLORS[ANGLE_ORDER[row]],
                alpha=0.55,
                edgecolor=WHITE,
                linewidth=0.25,
            )
            if len(values):
                ax.plot(
                    [period - 0.16, period + 0.16],
                    [baseline + np.median(values) * 0.10] * 2,
                    color=INK,
                    linewidth=1.5,
                )
    ax.axvspan(-0.42, 0.42, color=LIGHT_ORANGE, alpha=0.25, zorder=0)
    ax.set_xticks(range(-2, 3), [r"$t-2$", r"$t-1$", r"$t_0$", r"$t+1$", r"$t+2$"])
    ax.set_yticks(range(len(metrics)), [metric_labels[m] for m in metrics[::-1]])
    ax.set_xlabel("Landmark-relative two-year period")
    ax.set_ylabel("Within-field standardized distribution")
    clean_axes(ax, grid_axis="x")


def _draw_fig1d(ax: Axes, bundle: FigureBundle) -> None:
    """Draw landmark versus pseudo-event distributions and their difference."""
    pool = bundle.tables["pseudo_event_pool"]
    draws = bundle.tables["pseudo_event_draws"]
    panel_title(ax, "d", "Landmarks versus matched pseudo-events")
    real = pool.loc[pool["event_type"].eq("landmark"), "shock"].dropna().to_numpy(float)
    pseudo = pool.loc[pool["event_type"].eq("pseudo"), "shock"].dropna().to_numpy(float)
    positions = [0.0, 0.8]
    violin = ax.violinplot([real, pseudo], positions=positions, widths=0.55, showextrema=False)
    for index, body in enumerate(violin["bodies"]):
        body.set_facecolor(ORANGE if index == 0 else LIGHT_BLUE)
        body.set_edgecolor("none")
        body.set_alpha(0.72)
    rng = np.random.default_rng(20260725)
    ax.scatter(
        positions[0] + rng.uniform(-0.08, 0.08, len(real)),
        real,
        s=18,
        color=ORANGE,
        edgecolor=WHITE,
        linewidth=0.4,
        zorder=3,
    )
    ax.scatter(
        positions[1] + rng.uniform(-0.10, 0.10, len(pseudo)),
        pseudo,
        s=9,
        color=BLUE,
        alpha=0.45,
        edgecolor="none",
        zorder=2,
    )
    difference = draws["difference"].dropna().to_numpy(float)
    if len(difference):
        mean = float(np.mean(difference))
        low, high = np.quantile(difference, [0.025, 0.975])
        ax.errorbar(
            1.72,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            color=INK,
            capsize=4,
            markersize=5,
            linewidth=1.3,
        )
        ax.text(
            1.72,
            high + 0.05,
            f"Δ={mean:+.2f}\n95% [{low:+.2f}, {high:+.2f}]",
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK,
        )
    ax.axhline(0, color=LIGHT_GRAY, linewidth=0.8)
    ax.set_xticks([0.0, 0.8, 1.72], ["Landmarks\nn=4", "Pseudo-event pool", "Paired draw\nDifference"])
    ax.set_ylabel("Standardized structural shock")
    ax.set_xlim(-0.45, 2.15)
    clean_axes(ax, grid_axis="y")
    ax.text(
        0.02,
        0.01,
        "Descriptive matched control; not causal identification",
        transform=ax.transAxes,
        color=VERMILLION,
        fontsize=6.5,
        ha="left",
        va="bottom",
    )


def render_fig1(
    bundle: FigureBundle,
    figure_dir: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render Fig.1 and all standalone panels."""
    configure_style()
    drawers = {"a": _draw_fig1a, "b": _draw_fig1b, "c": _draw_fig1c, "d": _draw_fig1d}
    outputs = _panel_outputs(bundle, drawers, figure_dir, dpi=dpi)
    fig = plt.figure(figsize=(15.5, 10.6))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.28, wspace=0.28)
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[1, 2])
    for ax, panel in zip([ax_a, ax_b, ax_c, ax_d], "abcd"):
        drawers[panel](ax, bundle)
    figure_title(
        fig,
        "Fig. 1 | Landmark papers and knowledge-graph reconfiguration",
        "Four real fields; fixed topic layouts; event alignment and matched pseudo-events are descriptive.",
    )
    outputs.update(_finish_composite(fig, bundle, figure_dir, formats, dpi))
    return outputs


# ============================================================================
# Fig.2
# ============================================================================


def _sankey_node_layout(flow: pd.DataFrame) -> Dict[Tuple[int, str], Tuple[float, float, float]]:
    """Allocate deterministic node intervals for an alluvial diagram."""
    stages = sorted(flow["stage_order"].unique())
    stage_names: Dict[int, set[str]] = {}
    for stage in stages:
        subset = flow.loc[flow["stage_order"].eq(stage)]
        stage_names.setdefault(stage - 1, set()).update(subset["source"].astype(str))
        stage_names.setdefault(stage, set()).update(subset["target"].astype(str))
    layout: Dict[Tuple[int, str], Tuple[float, float, float]] = {}
    for stage, names in stage_names.items():
        counts: Dict[str, float] = {}
        if stage == 0:
            subset = flow.loc[flow["stage_order"].eq(1)]
            counts = subset.groupby("source")["candidate_count"].sum().to_dict()
        else:
            subset = flow.loc[flow["stage_order"].eq(stage)]
            counts = subset.groupby("target")["candidate_count"].sum().to_dict()
        ordered = sorted(names, key=lambda name: (-counts.get(name, 0), name))
        total = sum(counts.get(name, 0) for name in ordered)
        gap = 0.018
        available = 0.78 - gap * max(len(ordered) - 1, 0)
        cursor = 0.08
        for name in ordered:
            height = available * counts.get(name, 0) / max(total, 1)
            layout[(stage, name)] = (cursor, cursor + height, counts.get(name, 0))
            cursor += height + gap
    return layout


def _ribbon(
    ax: Axes,
    x0: float,
    x1: float,
    y0a: float,
    y0b: float,
    y1a: float,
    y1b: float,
    color: str,
    alpha: float,
) -> None:
    """Draw one cubic alluvial ribbon."""
    control = (x1 - x0) * 0.44
    vertices = [
        (x0, y0a),
        (x0 + control, y0a),
        (x1 - control, y1a),
        (x1, y1a),
        (x1, y1b),
        (x1 - control, y1b),
        (x0 + control, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(
        PathPatch(
            MplPath(vertices, codes),
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
        )
    )


def _draw_fig2a(ax: Axes, bundle: FigureBundle) -> None:
    """Draw the 50-candidate evidence-governance Sankey."""
    flow = bundle.tables["candidate_flow"].copy()
    panel_title(ax, "a", "From 50 literature candidates to frozen roles")
    ax.set_axis_off()
    layout = _sankey_node_layout(flow)
    x_positions = np.linspace(0.03, 0.97, 5)
    outgoing: Dict[Tuple[int, str], float] = {}
    incoming: Dict[Tuple[int, str], float] = {}
    total = 50.0
    palette = [BLUE, PURPLE, ORANGE, OLIVE, PINK]
    for row in flow.sort_values(["stage_order", "source", "target"]).itertuples(index=False):
        stage = int(row.stage_order)
        source_key = (stage - 1, str(row.source))
        target_key = (stage, str(row.target))
        source_low, source_high, source_count = layout[source_key]
        target_low, target_high, target_count = layout[target_key]
        source_cursor = outgoing.get(source_key, source_low)
        target_cursor = incoming.get(target_key, target_low)
        source_height = (source_high - source_low) * float(row.candidate_count) / max(source_count, 1)
        target_height = (target_high - target_low) * float(row.candidate_count) / max(target_count, 1)
        role_color = (
            VERMILLION
            if "Excluded" in str(row.target)
            or any(token in str(row.target) for token in ("leakage", "gate", "unavailable"))
            else palette[(stage - 1) % len(palette)]
        )
        _ribbon(
            ax,
            x_positions[stage - 1] + 0.012,
            x_positions[stage] - 0.012,
            source_cursor,
            source_cursor + source_height,
            target_cursor,
            target_cursor + target_height,
            role_color,
            0.20,
        )
        outgoing[source_key] = source_cursor + source_height
        incoming[target_key] = target_cursor + target_height
    for (stage, name), (low, high, count) in layout.items():
        color = (
            VERMILLION
            if "Excluded" in name
            or any(token in name for token in ("leakage", "gate", "unavailable"))
            else palette[stage % len(palette)]
        )
        ax.add_patch(
            FancyBboxPatch(
                (x_positions[stage] - 0.011, low),
                0.022,
                max(high - low, 0.003),
                boxstyle="round,pad=0.002,rounding_size=0.003",
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=WHITE,
                linewidth=0.45,
            )
        )
        align = "left" if stage < 4 else "right"
        x_text = x_positions[stage] + (0.018 if stage < 4 else -0.018)
        if high - low > 0.024 or count >= 4:
            ax.text(
                x_text,
                (low + high) / 2,
                f"{_short(name, 22)}  {int(count)}",
                transform=ax.transAxes,
                ha=align,
                va="center",
                fontsize=5.8,
                color=INK,
            )
    stage_labels = ["Discovery", "Observation angle", "Local feasibility", "Runtime gate", "Frozen role"]
    for x_value, label in zip(x_positions, stage_labels):
        ax.text(
            x_value,
            0.98,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.7,
            color=GRAY,
            fontweight="bold",
        )
    ax.text(
        0.99,
        0.015,
        "Ribbon width = candidate count; no OOF outcome used",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color=GRAY,
    )


def _draw_fig2b(ax: Axes, bundle: FigureBundle) -> None:
    """Draw the five-angle evidence wheel and eight indicators."""
    indicators = bundle.tables["primary_indicator_map"]
    panel_title(ax, "b", "Five source-backed observation angles")
    ax.set_aspect("equal")
    ax.set_axis_off()
    center = (0.5, 0.61)
    radius = 0.29
    angles = np.linspace(90, 450, 6)
    for index, angle_id in enumerate(ANGLE_ORDER):
        theta1, theta2 = angles[index], angles[index + 1]
        color = ANGLE_COLORS[angle_id]
        ax.add_patch(
            Wedge(
                center,
                radius,
                theta1,
                theta2,
                width=0.12,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=WHITE,
                linewidth=1.2,
                alpha=0.88,
            )
        )
        theta = math.radians((theta1 + theta2) / 2)
        ax.text(
            center[0] + 0.23 * math.cos(theta),
            center[1] + 0.23 * math.sin(theta),
            _short(ANGLE_LABELS[angle_id], 20),
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=WHITE,
            fontsize=5.6,
            fontweight="bold",
        )
        members = indicators.loc[indicators["angle_id"].eq(angle_id)]
        satellite_angles = np.linspace(theta1 + 8, theta2 - 8, max(len(members), 1))
        for member_index, (satellite, row) in enumerate(
            zip(satellite_angles, members.itertuples(index=False)),
            start=int(indicators.index.min()) + 1,
        ):
            rad = math.radians(satellite)
            x_value = center[0] + 0.37 * math.cos(rad)
            y_value = center[1] + 0.37 * math.sin(rad)
            indicator_number = int(
                indicators.reset_index(drop=True)
                .index[indicators.reset_index(drop=True)["code_name"].eq(row.code_name)][0]
            ) + 1
            ax.add_patch(
                Circle(
                    (x_value, y_value),
                    0.035,
                    transform=ax.transAxes,
                    facecolor=WHITE,
                    edgecolor=color,
                    linewidth=1.2,
                )
            )
            ax.text(
                x_value,
                y_value + 0.004,
                f"{indicator_number}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=6.0,
                fontweight="bold",
                color=INK,
            )
            ax.text(
                x_value,
                y_value - 0.043,
                f"S{int(row.original_source_count)}·A{int(row.application_source_count)}·✓",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=4.4,
                color=GRAY,
            )
    ax.add_patch(
        Circle(
            center,
            0.11,
            transform=ax.transAxes,
            facecolor=PALE_GRAY,
            edgecolor=LIGHT_GRAY,
            linewidth=0.8,
        )
    )
    ax.text(
        *center,
        "Publication-time\nreference evidence",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=INK,
        fontsize=6.2,
        fontweight="bold",
    )
    ordered = indicators.reset_index(drop=True)
    for index, row in ordered.iterrows():
        column = index % 2
        line = index // 2
        ax.text(
            0.02 + column * 0.50,
            0.18 - line * 0.043,
            f"{index + 1}  {_short(row['feature_label'], 30).replace(chr(10), ' ')}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=INK,
            fontsize=4.9,
        )


def _draw_fig2c(ax: Axes, bundle: FigureBundle) -> None:
    """Draw distributions plus the suite's sole traditional correlation matrix."""
    panel_title(ax, "c", "Distribution shape and non-redundancy diagnostics")
    ax.set_axis_off()
    violin_ax = ax.inset_axes([0.00, 0.02, 0.42, 0.90])
    matrix_ax = ax.inset_axes([0.48, 0.18, 0.50, 0.74])
    distributions = bundle.tables["indicator_distributions"]
    order = list(distributions["feature_label"].drop_duplicates())
    sns.violinplot(
        data=distributions,
        x="robust_value",
        y="feature_label",
        order=order,
        orient="h",
        inner=None,
        color=LIGHT_BLUE,
        linewidth=0.5,
        cut=0,
        ax=violin_ax,
    )
    sample = distributions.groupby("feature_label", group_keys=False).sample(
        n=min(90, int(distributions.groupby("feature_label").size().min())),
        random_state=20260725,
    )
    sns.stripplot(
        data=sample,
        x="robust_value",
        y="feature_label",
        order=order,
        orient="h",
        size=1.1,
        alpha=0.30,
        color=INK,
        jitter=0.16,
        ax=violin_ax,
    )
    violin_ax.set_xlabel("Robust-scaled raw value (median 0, IQR 1; clipped)")
    violin_ax.set_ylabel("")
    violin_ax.set_yticks(range(len(order)), [_short(value, 24) for value in order])
    clean_axes(violin_ax, grid_axis="x")
    corr = bundle.tables["indicator_correlations"]
    code_order = list(dict.fromkeys(corr["feature_x"]))
    matrix = corr.pivot(index="feature_x", columns="feature_y", values="spearman").loc[
        code_order,
        code_order,
    ]
    mask = np.tril(np.ones(matrix.shape, dtype=bool), k=0)
    visible = np.ma.masked_where(mask, matrix.to_numpy(float))
    image = matrix_ax.imshow(visible, cmap="vlag", vmin=-1, vmax=1)
    for row in range(len(code_order)):
        for column in range(len(code_order)):
            if row <= column:
                value = matrix.iloc[row, column]
                if row != column:
                    matrix_ax.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5.4,
                        color=INK if abs(value) < 0.55 else WHITE,
                    )
    for index, code_name in enumerate(code_order):
        values = distributions.loc[
            distributions["code_name"].eq(code_name),
            "robust_value",
        ].dropna().to_numpy(float)
        if len(values) >= 3 and np.std(values) > 0:
            density_x = np.linspace(-3, 3, 60)
            density = gaussian_kde(values)(density_x)
            density = density / max(float(density.max()), 1e-12)
            mini_x = index - 0.42 + 0.84 * (density_x + 3) / 6
            mini_y = index + 0.35 - 0.70 * density
            matrix_ax.plot(mini_x, mini_y, color=INK, linewidth=0.65)
    labels = [str(index + 1) for index in range(len(code_order))]
    matrix_ax.set_xticks(range(len(code_order)), labels, fontsize=6)
    matrix_ax.set_yticks(range(len(code_order)), labels, fontsize=6)
    matrix_ax.tick_params(length=0)
    matrix_ax.set_title("Spearman upper triangle", fontsize=7.5, color=GRAY)
    colorbar = ax.inset_axes([0.55, 0.07, 0.35, 0.025])
    plt.colorbar(image, cax=colorbar, orientation="horizontal")
    colorbar.tick_params(labelsize=5)
    legend = "\n".join(
        f"{index + 1} {FEATURE_LABEL}"
        for index, FEATURE_LABEL in enumerate(order)
    )
    ax.text(
        0.47,
        0.02,
        legend,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.7,
        color=GRAY,
        linespacing=1.1,
    )


def _draw_fig2d(ax: Axes, bundle: FigureBundle) -> None:
    """Draw coverage versus resampling stability with registered gates."""
    data = bundle.tables["primary_quality_gates"].copy()
    panel_title(ax, "d", "Every primary indicator passes fixed runtime gates")
    data["label"] = data["code_name"].map(
        bundle.tables["primary_indicator_map"].set_index("code_name")["feature_label"]
    )
    sizes = 45 + 190 * np.clip(data["minimum_domain_coverage"], 0, 1)
    for angle_id, group in data.groupby("angle_id"):
        ax.scatter(
            group["overall_coverage"],
            group["stability_spearman"],
            s=sizes.loc[group.index],
            color=ANGLE_COLORS[angle_id],
            edgecolor=np.where(group["approximation_pass"].eq(1), INK, VERMILLION),
            linewidth=0.8,
            alpha=0.88,
            label=ANGLE_SHORT[angle_id],
        )
        for row in group.itertuples(index=False):
            indicator_number = int(
                bundle.tables["primary_indicator_map"]
                .reset_index(drop=True)
                .index[
                    bundle.tables["primary_indicator_map"]
                    .reset_index(drop=True)["code_name"]
                    .eq(row.code_name)
                ][0]
            ) + 1
            ax.text(
                row.overall_coverage,
                row.stability_spearman,
                str(indicator_number),
                ha="center",
                va="center",
                fontsize=5.2,
                color=WHITE,
                fontweight="bold",
                zorder=4,
            )
    ax.axvline(0.70, color=VERMILLION, linestyle="--", linewidth=0.9)
    ax.axhline(0.90, color=VERMILLION, linestyle="--", linewidth=0.9)
    ax.fill_betweenx([0.90, 1.0], 0.70, 1.0, color=LIGHT_BLUE, alpha=0.10)
    ax.set_xlim(0.68, 1.015)
    ax.set_ylim(0.895, 1.005)
    ax.set_xlabel("Overall eligible-case coverage")
    ax.set_ylabel("80% reference-resampling Spearman")
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=5.5)
    clean_axes(ax, grid_axis="both")
    ax.text(
        0.71,
        0.902,
        "registered pass region",
        color=BLUE,
        fontsize=6.2,
        ha="left",
        va="bottom",
    )
    key = bundle.tables["primary_indicator_map"].reset_index(drop=True)
    ax.text(
        0.02,
        0.98,
        "  ".join(f"{index + 1} {row.feature_label}" for index, row in key.iterrows()),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.5,
        color=GRAY,
        wrap=True,
    )


def _draw_fig2e(ax: Axes, bundle: FigureBundle) -> None:
    """Draw known-group paired percentile effects."""
    data = bundle.tables["known_group_effects"].sort_values(
        "mean_percentile_difference"
    )
    panel_title(ax, "e", "Known-group construct plausibility")
    y = np.arange(len(data))
    ax.errorbar(
        data["mean_percentile_difference"],
        y,
        xerr=[
            data["mean_percentile_difference"] - data["ci_low"],
            data["ci_high"] - data["mean_percentile_difference"],
        ],
        fmt="o",
        color=BLUE,
        ecolor=LIGHT_BLUE,
        capsize=2.5,
        linewidth=1.2,
        markersize=4.5,
    )
    ax.axvline(0, color=LIGHT_GRAY, linewidth=0.9)
    ax.set_yticks(y, [_short(value, 24) for value in data["feature_label"]])
    ax.set_xlabel("Paired field-year percentile difference\nhigh D5 diffusion − matched control")
    clean_axes(ax, grid_axis="x")
    ax.text(
        0.99,
        0.02,
        "Known-group difference ≠ complete innovation truth",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=VERMILLION,
    )


def render_fig2(
    bundle: FigureBundle,
    figure_dir: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render Fig.2 and all standalone panels."""
    configure_style()
    drawers = {
        "a": _draw_fig2a,
        "b": _draw_fig2b,
        "c": _draw_fig2c,
        "d": _draw_fig2d,
        "e": _draw_fig2e,
    }
    outputs = _panel_outputs(bundle, drawers, figure_dir, dpi=dpi)
    fig = plt.figure(figsize=(16.5, 13.2))
    grid = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.05, 0.82], hspace=0.30, wspace=0.70)
    axes = {
        "a": fig.add_subplot(grid[0, :4]),
        "b": fig.add_subplot(grid[0, 4:]),
        "c": fig.add_subplot(grid[1, :]),
        "d": fig.add_subplot(grid[2, :3]),
        "e": fig.add_subplot(grid[2, 3:]),
    }
    for key, ax in axes.items():
        drawers[key](ax, bundle)
    figure_title(
        fig,
        "Fig. 2 | Evidence governance and construct diagnostics",
        "Fifty source-discovered candidates → five observation angles → eight frozen primary indicators; no outcome-based selection.",
    )
    outputs.update(_finish_composite(fig, bundle, figure_dir, formats, dpi))
    return outputs


# ============================================================================
# Fig.3
# ============================================================================


def _box(
    ax: Axes,
    xy: Tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
    *,
    fontsize: float = 7.0,
) -> None:
    """Draw one deterministic rounded schematic box."""
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            transform=ax.transAxes,
            facecolor=WHITE,
            edgecolor=color,
            linewidth=1.2,
        )
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize,
    )


def _arrow(ax: Axes, start: Tuple[float, float], end: Tuple[float, float], color: str = GRAY) -> None:
    """Draw one schematic arrow in axis coordinates."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.1},
    )


def _draw_fig3a(ax: Axes, bundle: FigureBundle) -> None:
    """Draw the temporal OOF two-part model contract."""
    panel_title(ax, "a", "Two-part medium model under six temporal folds")
    ax.set_axis_off()
    _box(ax, (0.02, 0.56), 0.23, 0.25, "5 angles · 8 indicators\n+ K1 controls", BLUE)
    _box(ax, (0.37, 0.68), 0.25, 0.18, "Part 1\nP(future uptake > 0)", PURPLE)
    _box(ax, (0.37, 0.40), 0.25, 0.18, "Part 2\nDiffusion | uptake", PURPLE)
    _box(ax, (0.73, 0.54), 0.24, 0.25, "Expected D5 diffusion\np × conditional score", ORANGE)
    _arrow(ax, (0.25, 0.685), (0.37, 0.77), BLUE)
    _arrow(ax, (0.25, 0.685), (0.37, 0.49), BLUE)
    _arrow(ax, (0.62, 0.77), (0.73, 0.67), PURPLE)
    _arrow(ax, (0.62, 0.49), (0.73, 0.63), PURPLE)
    folds = bundle.tables["temporal_folds"].sort_values("fold_id")
    left, right, y = 0.05, 0.95, 0.20
    years_min = int(folds["test_year_min"].min())
    years_max = int(folds["test_year_max"].max())
    ax.plot([left, right], [y, y], transform=ax.transAxes, color=LIGHT_GRAY, linewidth=2)
    for row in folds.itertuples(index=False):
        x0 = left + (int(row.test_year_min) - years_min) / max(years_max - years_min, 1) * (right - left)
        x1 = left + (int(row.test_year_max) - years_min) / max(years_max - years_min, 1) * (right - left)
        ax.plot(
            [x0, x1],
            [y, y],
            transform=ax.transAxes,
            color=ANGLE_COLORS[ANGLE_ORDER[(int(row.fold_id) - 1) % 5]],
            linewidth=7,
            solid_capstyle="butt",
        )
        ax.text(
            (x0 + x1) / 2,
            y - 0.055,
            f"F{int(row.fold_id)}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=5.8,
            color=GRAY,
        )
    ax.text(
        0.5,
        0.06,
        "Every transform, fit and calibration is learned inside the training portion of its fold",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=GRAY,
    )


def _draw_fig3b(ax: Axes, bundle: FigureBundle) -> None:
    """Draw global OOF model points with cross-fold ranges."""
    data = bundle.tables["model_ladder"].sort_values("spearman_expected")
    panel_title(ax, "b", "Global OOF performance across fixed feature sets")
    y = np.arange(len(data))
    for index, row in enumerate(data.itertuples(index=False)):
        primary = row.model_id in {"k1_controls", "final_innovation_plus_k1", "innovation_only"}
        color = ORANGE if row.model_id == "final_innovation_plus_k1" else BLUE if primary else MID_GRAY
        ax.plot([row.fold_min, row.fold_max], [index, index], color=color, alpha=0.45, linewidth=2.0)
        ax.scatter(row.spearman_expected, index, s=42 if primary else 27, color=color, zorder=3)
        ax.text(
            row.spearman_expected + 0.006,
            index,
            (
                f"{row.spearman_expected:.4f}  (Δ over K1 +0.0857)"
                if row.model_id == "final_innovation_plus_k1"
                else f"{row.spearman_expected:.4f}"
            ),
            va="center",
            ha="left",
            fontsize=6.3,
            color=INK,
        )
    ax.set_yticks(y, [_short(value, 25) for value in data["model_label_en"]])
    ax.set_xlabel("D5 temporal-OOF Spearman (whisker = six-fold range, not CI)")
    ax.set_xlim(min(data["fold_min"].min() - 0.02, 0.42), 0.86)
    clean_axes(ax, grid_axis="x")
    paired = bundle.tables["paired_model_gains"]
    gain = paired.loc[paired["baseline_model_id"].eq("k1_controls")].iloc[0]
    ax.text(
        0.01,
        0.985,
        f"Paired Δρ 95% CI [{gain.gain_ci_low:+.4f}, {gain.gain_ci_high:+.4f}]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        color=BLUE,
    )


def _draw_fig3c(ax: Axes, bundle: FigureBundle) -> None:
    """Draw OOF prediction versus realized D5 joint density."""
    panel_title(ax, "c", "OOF prediction and realized diffusion")
    data = bundle.tables["oof_joint_density"]
    x = data["expected_diffusion_score"].to_numpy(float)
    y = data["realized_diffusion_target"].to_numpy(float)
    hex_ax = ax.inset_axes([0.08, 0.10, 0.76, 0.73])
    top_ax = ax.inset_axes([0.08, 0.84, 0.76, 0.10], sharex=hex_ax)
    right_ax = ax.inset_axes([0.85, 0.10, 0.10, 0.73], sharey=hex_ax)
    image = hex_ax.hexbin(
        x,
        y,
        gridsize=55,
        bins="log",
        mincnt=1,
        cmap="Blues",
        linewidths=0,
    )
    top_ax.hist(x, bins=55, density=True, color=LIGHT_BLUE, edgecolor="none")
    right_ax.hist(y, bins=55, density=True, orientation="horizontal", color=LIGHT_ORANGE, edgecolor="none")
    top_ax.set_axis_off()
    right_ax.set_axis_off()
    clean_axes(hex_ax)
    hex_ax.set_xlabel("Expected D5 diffusion score")
    hex_ax.set_ylabel("Realized D5 diffusion")
    ax.set_axis_off()
    ax.text(
        0.95,
        0.93,
        f"ρ={bundle.panel_text['c']['spearman']:.4f}\nN={bundle.panel_text['c']['n']:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=INK,
        fontsize=7,
        fontweight="bold",
    )
    ax.text(
        0.08,
        0.02,
        "Darker hexagons contain more papers",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=GRAY,
        fontsize=5.8,
    )


def _draw_fig3d(ax: Axes, bundle: FigureBundle) -> None:
    """Draw realized diffusion distributions across OOF prediction deciles."""
    data = bundle.tables["prediction_decile_sample"].copy()
    panel_title(ax, "d", "Realized D5 diffusion across prediction deciles")
    sns.violinplot(
        data=data,
        x="prediction_decile",
        y="realized_diffusion_target",
        inner=None,
        cut=0,
        density_norm="width",
        color=LIGHT_BLUE,
        linewidth=0.4,
        ax=ax,
    )
    sample = data.groupby("prediction_decile", group_keys=False).sample(
        n=min(90, int(data.groupby("prediction_decile").size().min())),
        random_state=20260725,
    )
    sns.stripplot(
        data=sample,
        x="prediction_decile",
        y="realized_diffusion_target",
        size=1.3,
        jitter=0.18,
        color=INK,
        alpha=0.28,
        ax=ax,
    )
    ax.set_xlabel("OOF prediction decile")
    ax.set_ylabel("Realized D5 diffusion")
    clean_axes(ax, grid_axis="y")
    top = float(bundle.panel_text["d"]["highest_decile_enrichment"])
    ax.text(
        0.99,
        0.96,
        f"Top prediction decile:\n{top:.2f}× enrichment of realized top-decile papers",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color=ORANGE,
        fontweight="bold",
    )


def _draw_fig3e(ax: Axes, bundle: FigureBundle) -> None:
    """Draw the post-hoc angle add/delete dual-ring dot plot."""
    data = bundle.tables["angle_add_delete"].copy()
    panel_title(ax, "e", "What each observation angle adds and protects")
    ax.set_aspect("equal")
    ax.set_axis_off()
    center = np.array([0.5, 0.48])
    theta = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(data), endpoint=False)
    add_max = max(float(data["increment_over_k1"].max()), 1e-12)
    drop_max = max(float(data["drop_from_full"].max()), 1e-12)
    for radius, label in [(0.26, "Delete loss"), (0.42, "Add gain")]:
        ax.add_patch(Circle(center, radius, transform=ax.transAxes, fill=False, edgecolor=LIGHT_GRAY, linewidth=0.8))
        ax.text(
            center[0],
            center[1] + radius,
            label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color=GRAY,
            fontsize=5.5,
        )
    for angle_value, row in zip(theta, data.itertuples(index=False)):
        color = ANGLE_COLORS[row.angle_id]
        direction = np.array([math.cos(angle_value), math.sin(angle_value)])
        add_radius = 0.30 + 0.12 * float(row.increment_over_k1) / add_max
        delete_radius = 0.14 + 0.12 * float(row.drop_from_full) / drop_max
        ax.plot(
            [center[0], center[0] + 0.44 * direction[0]],
            [center[1], center[1] + 0.44 * direction[1]],
            transform=ax.transAxes,
            color=PALE_GRAY,
            linewidth=0.8,
        )
        ax.scatter(
            center[0] + add_radius * direction[0],
            center[1] + add_radius * direction[1],
            s=42,
            color=color,
            edgecolor=WHITE,
            linewidth=0.6,
            transform=ax.transAxes,
            zorder=3,
        )
        ax.scatter(
            center[0] + delete_radius * direction[0],
            center[1] + delete_radius * direction[1],
            s=31,
            facecolor=WHITE,
            edgecolor=color,
            linewidth=1.2,
            transform=ax.transAxes,
            zorder=3,
        )
        label_pos = center + 0.49 * direction
        ax.text(
            label_pos[0],
            label_pos[1],
            f"{ANGLE_SHORT[row.angle_id]}\n"
            f"+{row.increment_over_k1:.4f} / −{row.drop_from_full:.4f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.6,
            color=INK,
        )
    ax.text(
        0.5,
        0.48,
        "filled\nadd gain\n\nhollow\ndelete loss",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=GRAY,
        fontsize=5.2,
        linespacing=0.9,
    )
    ax.text(
        0.5,
        0.01,
        "Post-hoc interpretation only",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=VERMILLION,
        fontsize=5.6,
    )


def render_fig3(
    bundle: FigureBundle,
    figure_dir: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render Fig.3 and all standalone panels."""
    configure_style()
    drawers = {
        "a": _draw_fig3a,
        "b": _draw_fig3b,
        "c": _draw_fig3c,
        "d": _draw_fig3d,
        "e": _draw_fig3e,
    }
    outputs = _panel_outputs(bundle, drawers, figure_dir, dpi=dpi)
    fig = plt.figure(figsize=(16.8, 10.6))
    grid = fig.add_gridspec(2, 6, height_ratios=[0.92, 1.05], hspace=0.32, wspace=0.45)
    axes = {
        "a": fig.add_subplot(grid[0, :2]),
        "b": fig.add_subplot(grid[0, 2:4]),
        "c": fig.add_subplot(grid[0, 4:]),
        "d": fig.add_subplot(grid[1, :4]),
        "e": fig.add_subplot(grid[1, 4:]),
    }
    for key, ax in axes.items():
        drawers[key](ax, bundle)
    figure_title(
        fig,
        "Fig. 3 | Temporal out-of-fold predictive validity",
        "D5 is the single primary result; models share papers, labels and folds. Angle diagnostics are post-hoc.",
    )
    outputs.update(_finish_composite(fig, bundle, figure_dir, formats, dpi))
    return outputs


# ============================================================================
# Fig.4
# ============================================================================


def _draw_fig4a(ax: Axes, bundle: FigureBundle) -> None:
    """Draw the locked validation-frame score coverage."""
    data = bundle.tables["validation_sample_coverage"].copy()
    panel_title(ax, "a", "Locked low / middle / high validation frame")
    score = data["validation_score"].dropna().to_numpy(float)
    if len(score) >= 3 and np.std(score) > 0:
        grid = np.linspace(float(np.min(score)), float(np.max(score)), 300)
        density = gaussian_kde(score)(grid)
        ax.fill_between(grid, 0, density, color=LIGHT_BLUE, alpha=0.55)
        ax.plot(grid, density, color=BLUE, linewidth=1.2)
    colors = {"low": OLIVE, "middle": PURPLE, "high": ORANGE}
    roles = data.get("assignment_role", pd.Series("", index=data.index)).astype(str)
    primary = roles.str.contains("primary")
    for tier, group in data.loc[primary].groupby("global_fig3_tier"):
        ax.scatter(
            group["validation_score"],
            np.full(len(group), -0.015),
            marker="|",
            s=85,
            color=colors.get(str(tier), GRAY),
            label=f"{str(tier).title()} primary, n={len(group)}",
        )
    ax.set_xlabel("Legacy Fig.3 validation score used only to lock the blinded sample")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=6.2)
    clean_axes(ax)
    ax.text(
        0.99,
        0.95,
        "30 primary cases × 3 labelers = 90 required judgements",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color=GRAY,
    )


def _draw_fig4_blocked(ax: Axes, bundle: FigureBundle, panel: str, title: str) -> None:
    """Draw one explicit blinded-evidence gate."""
    completed = int(bundle.panel_text["a"]["completed_judgements"])
    required = int(bundle.panel_text["a"]["required_judgements"])
    draft_panel(
        ax,
        f"{panel}  {title}",
        f"{completed}/{required} blinded judgements completed.\n"
        "The renderer refuses to invent human labels or external-validity estimates.",
    )


def render_fig4(
    bundle: FigureBundle,
    figure_dir: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render Fig.4 with hard DRAFT blocks when blinded labels are absent."""
    configure_style()
    drawers: Dict[str, PanelDrawer] = {
        "a": _draw_fig4a,
        "b": lambda ax, item: _draw_fig4_blocked(ax, item, "b", "Human ↔ ASPR label connections"),
        "c": lambda ax, item: _draw_fig4_blocked(ax, item, "c", "Agreement effect estimates"),
        "d": lambda ax, item: _draw_fig4_blocked(ax, item, "d", "Score by human rating"),
        "e": lambda ax, item: _draw_fig4_blocked(ax, item, "e", "Quote-grounded cases"),
    }
    outputs = _panel_outputs(bundle, drawers, figure_dir, dpi=dpi)
    fig = plt.figure(figsize=(15.5, 10.0))
    grid = fig.add_gridspec(2, 6, hspace=0.32, wspace=0.35)
    axes = {
        "a": fig.add_subplot(grid[0, :3]),
        "b": fig.add_subplot(grid[0, 3:]),
        "c": fig.add_subplot(grid[1, :2]),
        "d": fig.add_subplot(grid[1, 2:4]),
        "e": fig.add_subplot(grid[1, 4:]),
    }
    for key, ax in axes.items():
        drawers[key](ax, bundle)
    figure_title(
        fig,
        "Fig. 4 | Blinded peer-review construct validity",
        "The sampling frame is locked, but the external-validity claim is deliberately blocked until all human labels are complete.",
        draft=bundle.status.startswith("draft"),
    )
    outputs.update(_finish_composite(fig, bundle, figure_dir, formats, dpi))
    return outputs


# ============================================================================
# Fig.5
# ============================================================================


def _draw_fig5a(ax: Axes, bundle: FigureBundle) -> None:
    """Draw strict historical prediction and validation windows."""
    data = bundle.tables["historical_windows"].sort_values("cutoff")
    panel_title(ax, "a", "Historical cutoffs and forward validation")
    y = np.arange(len(data))[::-1]
    for y_value, row in zip(y, data.itertuples(index=False)):
        ax.plot([1980, row.training_end], [y_value, y_value], color=LIGHT_GRAY, linewidth=8, solid_capstyle="butt")
        ax.plot([row.prediction_start, row.prediction_end], [y_value, y_value], color=BLUE, linewidth=8, solid_capstyle="butt")
        ax.plot([row.validation_start, row.validation_end], [y_value, y_value], color=ORANGE, linewidth=8, solid_capstyle="butt")
        ax.text(
            row.cutoff,
            y_value + 0.12,
            f"cutoff {int(row.cutoff)}",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=INK,
        )
    ax.set_yticks(y, [f"Backtest {index + 1}" for index in range(len(data))])
    ax.set_xlim(1979, int(data["validation_end"].max()) + 1)
    ax.set_xlabel("Calendar year")
    clean_axes(ax, grid_axis="x")
    ax.text(
        0.01,
        0.02,
        "gray training history   blue scored seed window   orange future validation",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=GRAY,
    )


def _draw_fig5b(ax: Axes, bundle: FigureBundle) -> None:
    """Draw the main-cutoff topic landscape."""
    data = bundle.tables["topic_landscape"]
    panel_title(ax, "b", f"Predicted–realized topic landscape at cutoff {int(data['cutoff'].iloc[0])}")
    style = {
        "background": (LIGHT_GRAY, "o", 0.28, 10),
        "hit": (ORANGE, "o", 0.95, 54),
        "false_positive": (BLUE, "s", 0.90, 42),
        "miss": (WHITE, "X", 1.0, 48),
    }
    for classification, group in data.groupby("classification"):
        color, marker, alpha, size = style[classification]
        ax.scatter(
            group["x"],
            group["y"],
            s=size,
            marker=marker,
            facecolor=color,
            edgecolor=VERMILLION if classification == "miss" else WHITE,
            linewidth=0.8,
            alpha=alpha,
            label=classification.replace("_", " ").title(),
        )
    labels = data.loc[data["classification"].ne("background")].nsmallest(
        10,
        "prediction_score_rank",
    )
    for row in labels.head(7).itertuples(index=False):
        ax.annotate(
            _short(row.display_topic_label, 25),
            (row.x, row.y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.5,
            color=INK,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Topic-label TF–IDF landscape; coordinates have no metric interpretation")
    ax.legend(frameon=False, fontsize=5.8, loc="best")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_fig5c(ax: Axes, bundle: FigureBundle) -> None:
    """Draw predicted versus realized rank movements."""
    data = bundle.tables["rank_bump_topics"].copy()
    main_cutoff = int(data["cutoff"].max())
    data = data.loc[data["cutoff"].eq(main_cutoff)].copy()
    panel_title(ax, "c", f"Top-10 rank movement at cutoff {main_cutoff}")
    for row in data.itertuples(index=False):
        pred = min(int(row.prediction_score_rank), 12)
        realized = min(int(row.realized_frontier_score_rank), 12)
        hit = pred <= 10 and realized <= 10
        ax.plot(
            [0, 1],
            [pred, realized],
            color=ORANGE if hit else MID_GRAY,
            alpha=0.8,
            linewidth=1.3 if hit else 0.75,
        )
        if pred <= 10:
            ax.text(-0.02, pred, _short(row.display_topic_label, 19), ha="right", va="center", fontsize=4.9)
        if realized <= 10:
            ax.text(1.02, realized, _short(row.display_topic_label, 19), ha="left", va="center", fontsize=4.9)
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(12.5, 0.5)
    ax.set_xticks([0, 1], ["Predicted rank", "Realized rank"])
    ax.set_ylabel("Top rank")
    clean_axes(ax, grid_axis="y")


def _draw_fig5d(ax: Axes, bundle: FigureBundle) -> None:
    """Draw four locked seed-paper evidence cards."""
    data = bundle.tables["seed_cards"]
    panel_title(ax, "d", "Representative seed papers selected by historical predictions")
    ax.set_axis_off()
    for index, row in enumerate(data.itertuples(index=False)):
        column = index % 2
        grid_row = index // 2
        left = 0.02 + column * 0.50
        bottom = 0.52 - grid_row * 0.45
        ax.add_patch(
            FancyBboxPatch(
                (left, bottom),
                0.46,
                0.38,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                transform=ax.transAxes,
                facecolor=WHITE,
                edgecolor=LIGHT_BLUE,
                linewidth=1.0,
            )
        )
        ax.text(
            left + 0.02,
            bottom + 0.32,
            _short(row.title, 54),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            color=INK,
            fontweight="bold",
        )
        ax.text(
            left + 0.02,
            bottom + 0.19,
            f"{int(row.publication_year)} · {_short(row.display_topic_label, 34)}\n"
            f"OOF prediction {row.expected_diffusion_score:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color=GRAY,
        )
        values = [float(getattr(row, angle)) for angle in ANGLE_ORDER]
        for angle_index, (angle, value) in enumerate(zip(ANGLE_ORDER, values)):
            x0 = left + 0.02 + angle_index * 0.082
            ax.add_patch(
                Circle(
                    (x0 + 0.025, bottom + 0.07),
                    0.020 + 0.018 * value,
                    transform=ax.transAxes,
                    facecolor=ANGLE_COLORS[angle],
                    edgecolor=WHITE,
                    linewidth=0.4,
                    alpha=0.75,
                )
            )
            ax.text(
                x0 + 0.025,
                bottom + 0.025,
                f"A{angle_index + 1}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=4.8,
                color=GRAY,
            )


def _draw_fig5e(ax: Axes, bundle: FigureBundle) -> None:
    """Draw paired historical backtest estimates without bars."""
    data = bundle.tables["backtest_metrics"]
    panel_title(ax, "e", "Multi-cutoff frontier backtest")
    methods = list(data["method"].drop_duplicates())
    colors = {
        "ASPR temporal OOF": ORANGE,
        "K1 control-only": BLUE,
        "Historical growth": OLIVE,
        "Publication-prior popularity": PURPLE,
        "Random": MID_GRAY,
    }
    metrics = [
        ("precision_at_10", "Precision@10"),
        ("ndcg_at_10", "NDCG@10"),
        ("frontier_coverage", "Hotspot coverage"),
    ]
    offsets = np.linspace(-0.24, 0.24, len(methods))
    rng = np.random.default_rng(20260725)
    for metric_index, (metric, label) in enumerate(metrics):
        baseline = len(metrics) - metric_index - 1
        ax.axhline(baseline, color=PALE_GRAY, linewidth=0.7, zorder=0)
        for method_index, method in enumerate(methods):
            values = data.loc[data["method"].eq(method), metric].to_numpy(float)
            y_value = baseline + offsets[method_index]
            ax.scatter(
                values,
                np.full(len(values), y_value)
                + rng.uniform(-0.018, 0.018, len(values)),
                s=18,
                facecolor=WHITE,
                edgecolor=colors[method],
                linewidth=0.85,
                zorder=3,
            )
            ax.scatter(
                np.mean(values),
                y_value,
                s=34,
                color=colors[method],
                edgecolor=WHITE,
                linewidth=0.5,
                zorder=4,
            )
    ax.set_yticks(range(len(metrics)), [label for _, label in metrics[::-1]])
    ax.set_xlabel("Metric value across three historical cutoffs")
    ax.set_xlim(-0.03, 1.03)
    clean_axes(ax, grid_axis="x")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[method],
            markeredgecolor=WHITE,
            label=method,
            markersize=5,
        )
        for method in methods
    ]
    ax.legend(handles=handles, frameon=False, fontsize=4.8, ncol=2, loc="lower right")
    aspr = data.loc[data["method"].eq("ASPR temporal OOF")]
    ax.text(
        0.99,
        0.96,
        f"ASPR mean P@10={aspr['precision_at_10'].mean():.2f}\n"
        f"mean NDCG@10={aspr['ndcg_at_10'].mean():.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=INK,
        fontsize=6.5,
    )


def render_fig5(
    bundle: FigureBundle,
    figure_dir: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render Fig.5 and all standalone panels."""
    configure_style()
    drawers = {
        "a": _draw_fig5a,
        "b": _draw_fig5b,
        "c": _draw_fig5c,
        "d": _draw_fig5d,
        "e": _draw_fig5e,
    }
    outputs = _panel_outputs(bundle, drawers, figure_dir, dpi=dpi)
    fig = plt.figure(figsize=(16.5, 11.5))
    grid = fig.add_gridspec(2, 6, height_ratios=[0.9, 1.1], hspace=0.33, wspace=0.45)
    axes = {
        "a": fig.add_subplot(grid[0, :2]),
        "b": fig.add_subplot(grid[0, 2:4]),
        "c": fig.add_subplot(grid[0, 4:]),
        "d": fig.add_subplot(grid[1, :4]),
        "e": fig.add_subplot(grid[1, 4:]),
    }
    for key, ax in axes.items():
        drawers[key](ax, bundle)
    figure_title(
        fig,
        "Fig. 5 | Strict historical frontier backtests",
        "Three publication-time cutoffs; future topic realization is evaluated only in the subsequent four-year window.",
    )
    outputs.update(_finish_composite(fig, bundle, figure_dir, formats, dpi))
    return outputs
