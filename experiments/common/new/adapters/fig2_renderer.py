"""Publication renderer for the redesigned evidence-governance Fig.2."""

from __future__ import annotations

import json
import math
import textwrap
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from colorspacious import cspace_convert
from matplotlib import colors as mcolors
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox
from PIL import Image
from pycirclize import Circos

from experiments.common.new.base.common import FigureBundle


INK = "#263238"
NAVY = "#18344C"
NEUTRAL = "#56636D"
MID_GREY = "#7C8790"
LIGHT_GREY = "#D7DEE2"
GRID_GREY = "#E8ECEF"
AMBER = "#D88918"
MAGENTA = "#B24F7A"
WHITE = "#FFFFFF"
PALE_AMBER = "#F7E9CB"

ANGLE_COLORS = {
    "A1_COMBINATION_RARITY": "#3C6E9E",
    "A2_ATYPICALITY_CONVENTIONALITY": "#D2772A",
    "A3_FIRST_TIME_COMBINATION": "#16847A",
    "A4_KNOWLEDGE_BREADTH_BALANCE": "#9A6FB0",
    "A5_COGNITIVE_DISTANCE_INTEGRATION": "#A85563",
}
ROLE_COLORS = {
    "primary": AMBER,
    "sensitivity": NAVY,
    "exploratory": "#16847A",
    "excluded": MAGENTA,
}
ROLE_LABELS = {
    "primary": "Primary",
    "sensitivity": "Sensitivity",
    "exploratory": "Exploratory",
    "excluded": "Excluded",
}


def _blend(color: str, white_fraction: float) -> str:
    """Blend one color toward white."""
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    mixed = rgb * (1.0 - white_fraction) + np.ones(3) * white_fraction
    return mcolors.to_hex(mixed)


def _renderer_config(bundle: FigureBundle) -> Dict[str, Any]:
    """Resolve renderer defaults plus optional figure configuration."""
    defaults: Dict[str, Any] = {
        "canvas_px": [6400, 5200],
        "width_ratios": [1.0, 1.06],
        "height_ratios": [1.0, 1.0],
        "min_font_pt": 5.5,
        "outer_sector_space_deg": 4.0,
        "indicator_arc_width": 26.0,
        "indicator_arc_gap": 4.0,
        "qa_preview_width": 1600,
    }
    defaults.update(bundle.chart_contract.get("render_config", {}))
    return defaults


def _verify_dependencies(bundle: FigureBundle) -> Dict[str, str]:
    """Require the exact plotting dependency versions in the contract."""
    observed: Dict[str, str] = {}
    expected = bundle.chart_contract["required_plot_packages"]
    for package, required in expected.items():
        try:
            observed[package] = version(package)
        except PackageNotFoundError as error:
            raise RuntimeError(
                f"Missing Fig.2 plotting dependency: {package}=={required}"
            ) from error
        if observed[package] != required:
            raise RuntimeError(
                f"Fig.2 requires {package}=={required}; "
                f"observed {observed[package]}"
            )
    return observed


def _rc_params() -> Dict[str, Any]:
    """Return the fixed publication typography and vector settings."""
    return {
        "font.family": "DejaVu Sans",
        "font.size": 5.5,
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "axes.edgecolor": LIGHT_GREY,
        "axes.linewidth": 0.55,
        "text.color": INK,
        "svg.fonttype": "none",
        "svg.hashsalt": "aspr-fig2-evidence-map-v1",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }


def _panel_header(
    ax: plt.Axes,
    panel: str,
    title: str,
    subtitle: str,
) -> None:
    """Draw one consistent panel label, title and subtitle."""
    ax.set_axis_off()
    ax.text(
        0.005,
        0.995,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.045,
        0.992,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color=INK,
    )
    if subtitle:
        ax.text(
            0.045,
            0.952,
            textwrap.fill(subtitle, width=66),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.5,
            color=NEUTRAL,
            linespacing=0.92,
            clip_on=True,
        )


def _flow_patch(
    ax: plt.Axes,
    x0: float,
    x1: float,
    source_low: float,
    source_high: float,
    target_low: float,
    target_high: float,
    color: str,
) -> None:
    """Draw one cubic, quantity-preserving alluvial ribbon."""
    curve = 0.48 * (x1 - x0)
    vertices = [
        (x0, source_low),
        (x0 + curve, source_low),
        (x1 - curve, target_low),
        (x1, target_low),
        (x1, target_high),
        (x1 - curve, target_high),
        (x0 + curve, source_high),
        (x0, source_high),
        (x0, source_low),
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
            transform=ax.transAxes,
            facecolor=color,
            edgecolor=WHITE,
            linewidth=0.28,
            alpha=0.32,
            zorder=1,
        )
    )


def _draw_stage_chain(ax: plt.Axes, stages: pd.DataFrame) -> None:
    """Draw the five-stage, outcome-blind contraction chain."""
    x_values = np.linspace(0.095, 0.905, len(stages))
    center_y = 0.805
    heights = 0.050 + 0.095 * stages["count"].to_numpy(float) / 50.0
    for index, (x_value, height, row) in enumerate(
        zip(x_values, heights, stages.itertuples(index=False))
    ):
        final = index == len(stages) - 1
        width = 0.115
        ax.add_patch(
            FancyBboxPatch(
                (x_value - width / 2, center_y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.003,rounding_size=0.006",
                transform=ax.transAxes,
                facecolor=WHITE if final else _blend(NAVY, 0.91),
                edgecolor=AMBER if final else _blend(NAVY, 0.50),
                linewidth=1.0 if final else 0.65,
                zorder=3,
            )
        )
        ax.text(
            x_value,
            center_y + 0.013,
            f"{int(row.count)}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color=AMBER if final else NAVY,
            zorder=4,
        )
        ax.text(
            x_value,
            center_y - height / 2 - 0.015,
            textwrap.fill(str(row.stage), width=16),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=5.5,
            color=INK,
            linespacing=0.92,
        )
        if index < len(stages) - 1:
            x_next = x_values[index + 1]
            next_height = heights[index + 1]
            _flow_patch(
                ax,
                x_value + width / 2,
                x_next - width / 2,
                center_y - height / 2,
                center_y + height / 2,
                center_y - next_height / 2,
                center_y + next_height / 2,
                LIGHT_GREY,
            )
            removed = int(stages.iloc[index + 1]["removed_since_previous"])
            ax.text(
                (x_value + x_next) / 2,
                center_y + max(height, next_height) / 2 + 0.016,
                f"−{removed}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=5.5,
                color=MAGENTA,
                zorder=5,
            )


def _node_intervals(
    counts: Sequence[int],
    *,
    bottom: float,
    top: float,
    gap: float,
) -> list[tuple[float, float]]:
    """Allocate top-to-bottom node intervals with a shared unit scale."""
    scale = (top - bottom - gap * (len(counts) - 1)) / float(sum(counts))
    output = []
    cursor = top
    for count in counts:
        high = cursor
        low = high - count * scale
        output.append((low, high))
        cursor = low - gap
    return output


def _draw_candidate_alluvial(
    ax: plt.Axes,
    flows: pd.DataFrame,
    families: pd.DataFrame,
) -> None:
    """Draw the complete 50-candidate angle-to-role alluvial."""
    angle_ids = families.sort_values("angle_order")["angle_id"].tolist()
    role_order = (
        flows.sort_values("role_order")["role"].drop_duplicates().tolist()
    )
    angle_counts = (
        flows.groupby("angle_id")["candidate_count"].sum().reindex(angle_ids)
    )
    role_counts = (
        flows.groupby("role")["candidate_count"].sum().reindex(role_order)
    )
    source_nodes = dict(
        zip(
            angle_ids,
            _node_intervals(
                angle_counts.astype(int).tolist(),
                bottom=0.405,
                top=0.645,
                gap=0.012,
            ),
        )
    )
    target_nodes = dict(
        zip(
            role_order,
            _node_intervals(
                role_counts.astype(int).tolist(),
                bottom=0.405,
                top=0.645,
                gap=0.016,
            ),
        )
    )
    unit = (
        0.645 - 0.405 - 0.012 * (len(angle_ids) - 1)
    ) / float(angle_counts.sum())
    source_cursor = {key: value[1] for key, value in source_nodes.items()}
    target_cursor = {key: value[1] for key, value in target_nodes.items()}
    for angle_id in angle_ids:
        for role in role_order:
            row = flows.loc[
                flows["angle_id"].eq(angle_id) & flows["role"].eq(role)
            ].iloc[0]
            count = int(row["candidate_count"])
            if count == 0:
                continue
            source_high = source_cursor[angle_id]
            source_low = source_high - unit * count
            target_high = target_cursor[role]
            target_low = target_high - unit * count
            _flow_patch(
                ax,
                0.285,
                0.755,
                source_low,
                source_high,
                target_low,
                target_high,
                ANGLE_COLORS[angle_id],
            )
            source_cursor[angle_id] = source_low
            target_cursor[role] = target_low
    family_lookup = families.set_index("angle_id")
    for angle_id, (low, high) in source_nodes.items():
        color = ANGLE_COLORS[angle_id]
        ax.add_patch(
            FancyBboxPatch(
                (0.267, low),
                0.018,
                high - low,
                boxstyle="round,pad=0.001,rounding_size=0.003",
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=WHITE,
                linewidth=0.45,
                zorder=4,
            )
        )
        row = family_lookup.loc[angle_id]
        ax.text(
            0.015,
            (low + high) / 2,
            f"{row.angle_code}  {row.ring_label}  · {int(row.candidate_count)}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.5,
            color=INK,
            fontweight="bold",
            clip_on=True,
        )
    for role, (low, high) in target_nodes.items():
        color = ROLE_COLORS[role]
        ax.add_patch(
            FancyBboxPatch(
                (0.755, low),
                0.018,
                high - low,
                boxstyle="round,pad=0.001,rounding_size=0.003",
                transform=ax.transAxes,
                facecolor=WHITE,
                edgecolor=color,
                linewidth=0.85,
                zorder=4,
            )
        )
        ax.text(
            0.785,
            (low + high) / 2,
            f"{ROLE_LABELS[role]}\n{int(role_counts[role])}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.5,
            fontweight="bold",
            color=color,
        )


def _draw_gate_cards(
    ax: plt.Axes,
    rules: pd.DataFrame,
    threshold_statement: str,
) -> None:
    """Draw six non-sequential rule cards and their numeric thresholds."""
    ax.text(
        0.015,
        0.335,
        textwrap.fill(threshold_statement, width=84),
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.5,
        color=INK,
        linespacing=0.90,
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": PALE_AMBER,
            "edgecolor": _blend(AMBER, 0.55),
            "linewidth": 0.5,
        },
        clip_on=True,
    )
    for index, row in enumerate(rules.itertuples(index=False)):
        column = index % 2
        row_index = index // 2
        x = 0.015 + column * 0.493
        y = 0.230 - row_index * 0.100
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                0.475,
                0.082,
                boxstyle="round,pad=0.004,rounding_size=0.006",
                transform=ax.transAxes,
                facecolor=_blend(NAVY, 0.965),
                edgecolor=LIGHT_GREY,
                linewidth=0.55,
            )
        )
        ax.text(
            x + 0.010,
            y + 0.059,
            f"{row.gate_id}  {row.label}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.5,
            fontweight="bold",
            color=NAVY,
        )
        ax.text(
            x + 0.010,
            y + 0.026,
            textwrap.fill(str(row.display_text), width=39),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.5,
            color=NEUTRAL,
            linespacing=0.88,
            clip_on=True,
        )


def _draw_panel_a(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Draw candidate scope, role allocation and screening rules."""
    text = bundle.panel_text["a"]
    _panel_header(ax, "a", text["title"], text["subtitle"])
    stages = bundle.tables["fig2_selection_stages"].sort_values("stage_order")
    flows = bundle.tables["fig2_candidate_role_flows"]
    families = bundle.tables["fig2_candidate_families"]
    rules = bundle.tables["fig2_selection_rules"]
    _draw_stage_chain(ax, stages)
    _draw_candidate_alluvial(ax, flows, families)
    _draw_gate_cards(ax, rules, text["threshold_statement"])
    return [ax]


def _indicator_intervals(
    nodes: pd.DataFrame,
    node_width: float,
    gap: float,
) -> Dict[str, tuple[str, float, float, float]]:
    """Allocate equal-width indicator arcs inside equal-width sectors."""
    output: Dict[str, tuple[str, float, float, float]] = {}
    for angle_code, group in nodes.groupby("angle_code", sort=False):
        group = group.sort_values("display_order")
        total = len(group) * node_width + max(len(group) - 1, 0) * gap
        cursor = 50.0 - total / 2.0
        for row in group.itertuples(index=False):
            start = cursor
            end = start + node_width
            output[str(row.code_name)] = (
                str(angle_code),
                start,
                end,
                (start + end) / 2.0,
            )
            cursor = end + gap
    return output


def _edge_band_width(value: float) -> float:
    """Map the registered absolute-correlation bins to ribbon widths."""
    if value >= 0.70:
        return 5.2
    if value >= 0.60:
        return 4.2
    if value >= 0.50:
        return 3.3
    return 2.5


def _allocate_edge_slots(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    intervals: Mapping[str, tuple[str, float, float, float]],
) -> Dict[tuple[int, str], tuple[float, float]]:
    """Allocate deterministic, non-overlapping link slots within each node."""
    order = nodes.set_index("code_name")["display_order"].to_dict()
    output: Dict[tuple[int, str], tuple[float, float]] = {}
    for code_name in nodes["code_name"]:
        incident = edges.loc[
            edges["source"].eq(code_name) | edges["target"].eq(code_name)
        ].copy()
        if incident.empty:
            continue
        incident["other_order"] = incident.apply(
            lambda row: order[
                row["target"] if row["source"] == code_name else row["source"]
            ],
            axis=1,
        )
        incident = incident.sort_values(["other_order", "edge_order"])
        widths = incident["absolute_spearman"].map(_edge_band_width).tolist()
        _, start, end, _ = intervals[str(code_name)]
        total = sum(widths) + 0.45 * max(len(widths) - 1, 0)
        cursor = (start + end - total) / 2.0
        for row, width in zip(incident.itertuples(index=False), widths):
            output[(int(row.edge_order), str(code_name))] = (
                cursor,
                cursor + width,
            )
            cursor += width + 0.45
    return output


def _circular_mean(left: float, right: float) -> float:
    """Return the angular mean of two radians, respecting wrap-around."""
    vector = np.exp(1j * left) + np.exp(1j * right)
    return float(np.angle(vector) % (2 * np.pi))


def _draw_circos(
    polar_ax: plt.Axes,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    render_config: Mapping[str, Any],
) -> None:
    """Draw equal sectors, equal indicator arcs and seven relation ribbons."""
    angle_rows = (
        nodes.sort_values("display_order")
        .drop_duplicates("angle_id")
        [["angle_id", "angle_code", "angle_label", "angle_ring_label"]]
    )
    angle_ids = angle_rows["angle_id"].tolist()
    code_to_id = dict(zip(angle_rows["angle_code"], angle_rows["angle_id"]))
    code_to_label = dict(
        zip(angle_rows["angle_code"], angle_rows["angle_ring_label"])
    )
    sectors = {code: 100.0 for code in angle_rows["angle_code"]}
    circos = Circos(
        sectors,
        space=float(render_config["outer_sector_space_deg"]),
    )
    intervals = _indicator_intervals(
        nodes,
        float(render_config["indicator_arc_width"]),
        float(render_config["indicator_arc_gap"]),
    )
    for sector in circos.sectors:
        angle_id = code_to_id[sector.name]
        color = ANGLE_COLORS[angle_id]
        outer = sector.add_track((91, 100))
        outer.axis(fc=color, ec=WHITE, lw=0.75)
        sector.text(
            f"{sector.name}  {code_to_label[sector.name]}",
            r=105,
            size=5.5,
            color=INK,
            fontweight="bold",
            orientation="horizontal",
        )
        inner = sector.add_track((75, 88))
        members = nodes.loc[nodes["angle_code"].eq(sector.name)]
        for row in members.itertuples(index=False):
            _, start, end, midpoint = intervals[str(row.code_name)]
            inner.rect(
                start,
                end,
                fc=WHITE,
                ec=color,
                lw=1.0,
                zorder=4,
            )
            direction = " ↓" if int(row.direction) == -1 else ""
            inner.text(
                f"{row.indicator_id}{direction}",
                x=midpoint,
                r=81.5,
                size=5.5,
                color=INK,
                fontweight="bold",
                orientation="horizontal",
                zorder=5,
            )
    slots = _allocate_edge_slots(nodes, edges, intervals)
    for row in edges.sort_values("edge_order").itertuples(index=False):
        source_sector = intervals[str(row.source)][0]
        target_sector = intervals[str(row.target)][0]
        source_slot = slots[(int(row.edge_order), str(row.source))]
        target_slot = slots[(int(row.edge_order), str(row.target))]
        if float(row.oriented_spearman) >= 0:
            circos.link(
                (source_sector, *source_slot),
                (target_sector, *target_slot),
                r1=74.8,
                r2=74.8,
                color=NAVY,
                alpha=0.31,
                height_ratio=0.55,
                allow_twist=False,
                ec=_blend(NAVY, 0.25),
                lw=0.30,
                zorder=1,
            )
        else:
            for side in (0, 1):
                fraction = 0.44
                source_width = source_slot[1] - source_slot[0]
                target_width = target_slot[1] - target_slot[0]
                if side == 0:
                    source_part = (
                        source_slot[0],
                        source_slot[0] + source_width * fraction,
                    )
                    target_part = (
                        target_slot[0],
                        target_slot[0] + target_width * fraction,
                    )
                else:
                    source_part = (
                        source_slot[1] - source_width * fraction,
                        source_slot[1],
                    )
                    target_part = (
                        target_slot[1] - target_width * fraction,
                        target_slot[1],
                    )
                circos.link(
                    (source_sector, *source_part),
                    (target_sector, *target_part),
                    r1=74.8,
                    r2=74.8,
                    color=MAGENTA,
                    alpha=0.38,
                    height_ratio=0.55,
                    allow_twist=False,
                    ec=_blend(MAGENTA, 0.30),
                    lw=0.25,
                    zorder=1,
                )
    circos.plotfig(ax=polar_ax)
    sector_lookup = {sector.name: sector for sector in circos.sectors}
    rho_texts = []
    for row in edges.sort_values("edge_order").itertuples(index=False):
        source_sector, _, _, source_mid = intervals[str(row.source)]
        target_sector, _, _, target_mid = intervals[str(row.target)]
        theta = _circular_mean(
            sector_lookup[source_sector].x_to_rad(source_mid),
            sector_lookup[target_sector].x_to_rad(target_mid),
        )
        radius = 34.0 + 5.0 * ((int(row.edge_order) - 1) % 3)
        rho_texts.append(
            polar_ax.text(
                theta,
                radius,
                f"{float(row.oriented_spearman):+.3f}",
                fontsize=5.5,
                ha="center",
                va="center",
                color=NAVY
                if float(row.oriented_spearman) >= 0
                else MAGENTA,
                bbox={
                    "boxstyle": "round,pad=0.13",
                    "facecolor": WHITE,
                    "edgecolor": LIGHT_GREY,
                    "linewidth": 0.35,
                    "alpha": 0.90,
                },
                zorder=10,
            )
        )
    adjust_text(
        rho_texts,
        ax=polar_ax,
        expand=(1.05, 1.10),
        force_text=(0.02, 0.03),
        max_move=(3, 3),
        ensure_inside_axes=True,
        prevent_crossings=True,
    )
    polar_ax.set_ylim(0, 108)
    polar_ax.set_axis_off()


def _draw_relation_list(
    ax: plt.Axes,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    text: Mapping[str, str],
) -> None:
    """Provide exact lookup values beside the circular relation overview."""
    ax.text(
        0.680,
        0.872,
        "Observed relations",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.0,
        fontweight="bold",
        color=INK,
    )
    for index, row in enumerate(
        edges.sort_values("edge_order").itertuples(index=False)
    ):
        y = 0.828 - 0.043 * index
        color = NAVY if float(row.oriented_spearman) >= 0 else MAGENTA
        ax.plot(
            [0.682, 0.710],
            [y, y],
            transform=ax.transAxes,
            color=color,
            linewidth=_edge_band_width(float(row.absolute_spearman)) * 0.36,
            alpha=0.65,
            solid_capstyle="round",
        )
        if float(row.oriented_spearman) < 0:
            ax.plot(
                [0.686, 0.706],
                [y, y],
                transform=ax.transAxes,
                color=WHITE,
                linewidth=0.5,
            )
        ax.text(
            0.720,
            y,
            f"{row.source_id}–{row.target_id}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.5,
            color=INK,
            fontweight="bold",
        )
        ax.text(
            0.985,
            y,
            f"{float(row.oriented_spearman):+.3f}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=5.5,
            color=color,
            fontweight="bold",
        )
    ax.text(
        0.680,
        0.510,
        textwrap.fill(text["relation_method"], width=30),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color=NEUTRAL,
        linespacing=0.92,
        clip_on=True,
    )
    ax.text(
        0.680,
        0.463,
        "Indicator key",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        fontweight="bold",
        color=INK,
    )
    for index, row in enumerate(
        nodes.sort_values("display_order").itertuples(index=False)
    ):
        y = 0.428 - index * 0.030
        color = ANGLE_COLORS[str(row.angle_id)]
        direction = " ↓" if int(row.direction) == -1 else ""
        ax.text(
            0.680,
            y,
            f"{row.indicator_id}{direction}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.5,
            fontweight="bold",
            color=color,
        )
        ax.text(
            0.720,
            y,
            str(row.short_label).replace("\\n", " ").replace("\n", " "),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.5,
            color=INK,
            clip_on=True,
        )
    ax.text(
        0.680,
        0.178,
        textwrap.fill(text["isolated_note"], width=30),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": _blend(LIGHT_GREY, 0.62),
            "edgecolor": LIGHT_GREY,
            "linewidth": 0.45,
        },
    )
    ax.text(
        0.680,
        0.105,
        textwrap.fill(text["relation_boundary"], width=30),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color=NEUTRAL,
        linespacing=0.96,
        clip_on=True,
    )


def _draw_panel_b(
    ax: plt.Axes,
    bundle: FigureBundle,
    render_config: Mapping[str, Any],
) -> list[plt.Axes]:
    """Draw the five-angle/eight-indicator circular evidence map."""
    text = bundle.panel_text["b"]
    _panel_header(ax, "b", text["title"], text["subtitle"])
    polar_ax = ax.inset_axes(
        [0.012, 0.105, 0.615, 0.755],
        projection="polar",
    )
    nodes = bundle.tables["fig2_relation_nodes"].sort_values("display_order")
    edges = bundle.tables["fig2_relation_edges"].sort_values("edge_order")
    _draw_circos(polar_ax, nodes, edges, render_config)
    _draw_relation_list(ax, nodes, edges, text)
    return [ax, polar_ax]


def _source_chip_label(value: str) -> str:
    """Reduce one frozen author-year source to a compact capsule label."""
    return value.replace(" et al., ", " ").replace(", ", " ")


def _draw_panel_c(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Draw meaning, inclusion/exclusion boundary and source strips."""
    text = bundle.panel_text["c"]
    _panel_header(ax, "c", text["title"], text["subtitle"])
    data = bundle.tables["fig2_dimension_provenance"].sort_values("angle_order")
    top = 0.875
    row_height = 0.160
    for index, row in enumerate(data.itertuples(index=False)):
        y_top = top - index * row_height
        y_bottom = y_top - row_height + 0.010
        color = ANGLE_COLORS[str(row.angle_id)]
        if index % 2 == 0:
            ax.add_patch(
                FancyBboxPatch(
                    (0.010, y_bottom),
                    0.980,
                    y_top - y_bottom,
                    boxstyle="round,pad=0.001,rounding_size=0.003",
                    transform=ax.transAxes,
                    facecolor=_blend(color, 0.973),
                    edgecolor="none",
                    zorder=0,
                )
            )
        ax.plot(
            [0.015, 0.985],
            [y_bottom, y_bottom],
            transform=ax.transAxes,
            color=LIGHT_GREY,
            linewidth=0.45,
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.018, y_top - 0.050),
                0.040,
                0.038,
                boxstyle="round,pad=0.003,rounding_size=0.012",
                transform=ax.transAxes,
                facecolor=WHITE,
                edgecolor=color,
                linewidth=1.0,
            )
        )
        ax.text(
            0.038,
            y_top - 0.031,
            row.angle_code,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.5,
            fontweight="bold",
            color=color,
        )
        ax.text(
            0.070,
            y_top - 0.010,
            row.angle_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            fontweight="bold",
            color=color,
        )
        ax.text(
            0.985,
            y_top - 0.010,
            f"registered sources n={int(row.registered_source_count)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.5,
            color=NEUTRAL,
        )
        columns = [
            (0.070, "MEANING", str(row.meaning_short), 23),
            (0.375, "INCLUDE", str(row.include_short), 23),
            (0.680, "EXCLUDE", str(row.exclude_short), 23),
        ]
        for x, heading, body, wrap_width in columns:
            ax.text(
                x,
                y_top - 0.041,
                heading,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5.5,
                fontweight="bold",
                color=NEUTRAL,
            )
            ax.text(
                x,
                y_top - 0.064,
                textwrap.fill(body, width=wrap_width),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5.5,
                color=INK,
                linespacing=0.90,
                clip_on=True,
            )
        sources = str(row.key_sources).split(" | ")
        ax.text(
            0.070,
            y_bottom + 0.015,
            "SOURCES",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.5,
            fontweight="bold",
            color=NEUTRAL,
        )
        x_start = 0.145
        y_chip = y_bottom + 0.004
        for source_index, source in enumerate(sources):
            label = _source_chip_label(source)
            chip_width = max(0.120, 0.040 + len(label) * 0.0070)
            x = x_start
            ax.add_patch(
                FancyBboxPatch(
                    (x, y_chip),
                    chip_width,
                    0.024,
                    boxstyle="round,pad=0.002,rounding_size=0.007",
                    transform=ax.transAxes,
                    facecolor=WHITE,
                    edgecolor=_blend(color, 0.50),
                    linewidth=0.45,
                )
            )
            ax.text(
                x + chip_width / 2,
                y_chip + 0.012,
                label,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=5.5,
                color=INK,
            )
            x_start += chip_width + 0.010
    ax.text(
        0.015,
        0.012,
        textwrap.fill(text["footer"], width=112),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.5,
        color=NEUTRAL,
        clip_on=True,
    )
    return [ax]


def _micro_axis(
    ax: plt.Axes,
    *,
    x0: float,
    x1: float,
    y: float,
    value: float,
    minimum: float,
    maximum: float,
    threshold: float,
    color: str,
    lower_is_better: bool = False,
) -> None:
    """Draw one independent mini-axis with threshold, point and exact value."""
    normalized = (value - minimum) / max(maximum - minimum, 1e-12)
    normalized = min(max(normalized, 0.0), 1.0)
    threshold_norm = (threshold - minimum) / max(maximum - minimum, 1e-12)
    threshold_norm = min(max(threshold_norm, 0.0), 1.0)
    x_value = x0 + normalized * (x1 - x0)
    x_threshold = x0 + threshold_norm * (x1 - x0)
    ax.plot(
        [x0, x1],
        [y, y],
        transform=ax.transAxes,
        color=LIGHT_GREY,
        linewidth=0.65,
        zorder=1,
    )
    ax.plot(
        [x_threshold, x_threshold],
        [y - 0.019, y + 0.019],
        transform=ax.transAxes,
        color=AMBER,
        linewidth=0.65,
        zorder=2,
    )
    ax.scatter(
        [x_value],
        [y],
        transform=ax.transAxes,
        s=13,
        facecolor=WHITE if lower_is_better else color,
        edgecolor=color,
        linewidth=0.75,
        zorder=4,
    )
    ax.text(
        (x0 + x1) / 2,
        y + 0.022,
        f"{value:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.5,
        color=color,
        zorder=5,
    )


def _draw_ledger_header(ax: plt.Axes) -> None:
    """Draw aligned headers for the eight-row evidence ledger."""
    headers = [
        (0.015, "Indicator · registered formula", "left"),
        (0.395, "Dir.", "center"),
        (0.445, "F/P/V", "center"),
        (0.525, "Overall\n≥.70", "center"),
        (0.605, "Min field\n≥.50", "center"),
        (0.685, "Resample ρ\n≥.90", "center"),
        (0.765, "MRE\n≤.10", "center"),
        (0.825, "Fidelity", "left"),
    ]
    for x, label, alignment in headers:
        ax.text(
            x,
            0.890,
            label,
            transform=ax.transAxes,
            ha=alignment,
            va="center",
            fontsize=5.5,
            fontweight="bold",
            color=INK,
            linespacing=0.90,
        )
    ax.plot(
        [0.012, 0.990],
        [0.854, 0.854],
        transform=ax.transAxes,
        color=NEUTRAL,
        linewidth=0.65,
    )


def _draw_ledger_rows(ax: plt.Axes, ledger: pd.DataFrame) -> None:
    """Draw formulas, directions, evidence and four independent scales."""
    y_centres = np.linspace(0.805, 0.190, len(ledger))
    for index, (y, row) in enumerate(
        zip(y_centres, ledger.sort_values("display_order").itertuples(index=False))
    ):
        color = ANGLE_COLORS[str(row.angle_id)]
        if index % 2 == 0:
            ax.add_patch(
                FancyBboxPatch(
                    (0.010, y - 0.039),
                    0.980,
                    0.078,
                    boxstyle="round,pad=0.001,rounding_size=0.003",
                    transform=ax.transAxes,
                    facecolor=_blend(color, 0.975),
                    edgecolor="none",
                )
            )
        ax.plot(
            [0.012, 0.990],
            [y - 0.044, y - 0.044],
            transform=ax.transAxes,
            color=GRID_GREY,
            linewidth=0.45,
        )
        ax.plot(
            [0.014, 0.014],
            [y - 0.032, y + 0.032],
            transform=ax.transAxes,
            color=color,
            linewidth=2.0,
        )
        ax.text(
            0.023,
            y + 0.019,
            f"{row.indicator_id}  {row.feature_label}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.5,
            fontweight="bold",
            color=color,
        )
        formula = str(row.display_formula)
        ax.text(
            0.023,
            y - 0.014,
            formula,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.5,
            color=INK,
            linespacing=0.86,
        )
        direction_symbol = "↓" if int(row.direction) == -1 else "↑"
        ax.text(
            0.395,
            y,
            direction_symbol,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.5,
            fontweight="bold",
            color=MAGENTA if int(row.direction) == -1 else NAVY,
            linespacing=0.86,
        )
        ax.text(
            0.445,
            y,
            str(row.evidence_badge).replace(" · ", "\n"),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.5,
            color=INK,
            linespacing=0.82,
        )
        _micro_axis(
            ax,
            x0=0.495,
            x1=0.555,
            y=y - 0.010,
            value=float(row.overall_coverage),
            minimum=0.70,
            maximum=1.00,
            threshold=0.70,
            color=color,
        )
        _micro_axis(
            ax,
            x0=0.575,
            x1=0.635,
            y=y - 0.010,
            value=float(row.minimum_domain_coverage),
            minimum=0.50,
            maximum=1.00,
            threshold=0.50,
            color=color,
        )
        _micro_axis(
            ax,
            x0=0.655,
            x1=0.715,
            y=y - 0.010,
            value=float(row.stability_spearman),
            minimum=0.90,
            maximum=1.00,
            threshold=0.90,
            color=color,
        )
        _micro_axis(
            ax,
            x0=0.735,
            x1=0.795,
            y=y - 0.010,
            value=float(row.stability_median_relative_error),
            minimum=0.00,
            maximum=0.10,
            threshold=0.10,
            color=color,
            lower_is_better=True,
        )
        ax.text(
            0.825,
            y,
            (
                f"ρ={float(row.approximation_spearman):.3f}\n"
                f"MRE={float(row.approximation_median_relative_error):.3f}"
                if bool(row.approximation_applicable)
                else "exact"
            ),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.5,
            color=INK,
            linespacing=0.86,
            clip_on=True,
        )


def _draw_panel_d(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Draw the formal definition and evidence-quality ledger."""
    text = bundle.panel_text["d"]
    _panel_header(ax, "d", text["title"], text["subtitle"])
    _draw_ledger_header(ax)
    _draw_ledger_rows(ax, bundle.tables["fig2_indicator_ledger"])
    ax.text(
        0.015,
        0.118,
        textwrap.fill(text["evidence_definition"], width=132),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color=NEUTRAL,
        linespacing=0.90,
        clip_on=True,
    )
    ax.text(
        0.015,
        0.072,
        textwrap.fill(text["test_statement"], width=132),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color=INK,
        linespacing=0.90,
        clip_on=True,
    )
    ax.text(
        0.015,
        0.030,
        textwrap.fill(text["selection_boundary"], width=132),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color=NEUTRAL,
        linespacing=0.90,
        clip_on=True,
    )
    return [ax]


def _save_figure(
    fig: plt.Figure,
    path: Path,
    *,
    dpi: int,
    bbox_inches: Bbox | None = None,
) -> None:
    """Save one deterministic raster or vector artifact."""
    suffix = path.suffix.lower()
    kwargs: Dict[str, Any] = {
        "facecolor": WHITE,
        "edgecolor": "none",
        "pad_inches": 0,
    }
    if bbox_inches is None:
        # Pin the export to the requested canvas rather than inheriting a
        # process-wide ``savefig.bbox='tight'`` setting.
        kwargs["bbox_inches"] = Bbox.from_bounds(
            0.0,
            0.0,
            float(fig.get_figwidth()),
            float(fig.get_figheight()),
        )
    else:
        kwargs["bbox_inches"] = bbox_inches
    if suffix == ".png":
        kwargs["dpi"] = int(dpi)
    elif suffix == ".svg":
        kwargs["metadata"] = {
            "Date": None,
            "Creator": "ASPR Fig.2 evidence-map renderer",
        }
    elif suffix == ".pdf":
        kwargs["metadata"] = {
            "CreationDate": None,
            "ModDate": None,
            "Creator": "ASPR Fig.2 evidence-map renderer",
        }
    fig.savefig(path, **kwargs)


def _export_panels(
    fig: plt.Figure,
    groups: Mapping[str, Sequence[plt.Axes]],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Export each full panel cell without raster recomposition."""
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outputs: Dict[str, Path] = {}
    for panel, axes in groups.items():
        extents = [axis.get_window_extent(renderer) for axis in axes]
        extent = Bbox.union(extents).transformed(
            fig.dpi_scale_trans.inverted()
        )
        for extension in formats:
            path = panel_dir / f"fig02_{panel}.{extension}"
            _save_figure(fig, path, dpi=dpi, bbox_inches=extent)
            outputs[f"panel_{panel}_{extension}"] = path
    return outputs


def _cvd_preview(
    rgb: np.ndarray,
    *,
    cvd_type: str,
) -> np.ndarray:
    """Simulate one complete color-vision-deficiency condition."""
    space = {
        "name": "sRGB1+CVD",
        "cvd_type": cvd_type,
        "severity": 100,
    }
    converted = cspace_convert(rgb, space, "sRGB1")
    return np.clip(converted, 0.0, 1.0)


def _palette_cvd_distance(cvd_type: str) -> float:
    """Return the minimum pairwise RGB distance among the five angle colors."""
    rgb = np.asarray(
        [mcolors.to_rgb(color) for color in ANGLE_COLORS.values()],
        dtype=float,
    )
    converted = _cvd_preview(rgb, cvd_type=cvd_type)
    distances = [
        float(np.linalg.norm(converted[left] - converted[right]))
        for left in range(len(converted))
        for right in range(left + 1, len(converted))
    ]
    return min(distances)


def _write_accessibility_qa(
    figure_png: Path,
    output_dir: Path,
    *,
    preview_width: int,
    package_versions: Mapping[str, str],
) -> Dict[str, Path]:
    """Write grayscale and CVD previews plus a compact QA record."""
    qa_dir = output_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(figure_png).convert("RGB")
    original_size = image.size
    height = max(1, round(original_size[1] * preview_width / original_size[0]))
    preview = image.resize(
        (int(preview_width), int(height)),
        Image.Resampling.LANCZOS,
    )
    outputs: Dict[str, Path] = {}
    grayscale = qa_dir / "figure_full_grayscale.png"
    preview.convert("L").save(grayscale)
    outputs["qa_grayscale"] = grayscale
    rgb = np.asarray(preview, dtype=float) / 255.0
    for name, cvd_type in (
        ("deuteranopia", "deuteranomaly"),
        ("protanopia", "protanomaly"),
    ):
        converted = (_cvd_preview(rgb, cvd_type=cvd_type) * 255).round().astype(
            np.uint8
        )
        path = qa_dir / f"figure_full_{name}.png"
        Image.fromarray(converted, mode="RGB").save(path)
        outputs[f"qa_{name}"] = path
    record = {
        "source_png": str(figure_png.resolve()),
        "source_size_px": list(original_size),
        "preview_size_px": list(preview.size),
        "angle_palette_min_rgb_distance": {
            "deuteranopia": _palette_cvd_distance("deuteranomaly"),
            "protanopia": _palette_cvd_distance("protanomaly"),
        },
        "non_color_encodings": [
            "direct labels",
            "negative-ribbon white channel",
            "outlined MRE points",
            "fixed role and angle order",
        ],
        "packages": dict(package_versions),
    }
    record_path = qa_dir / "visual_accessibility.json"
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["qa_record"] = record_path
    return outputs


def render_fig2_evidence_map(
    bundle: FigureBundle,
    output_dir: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render the fixed-canvas four-panel Fig.2 and QA previews."""
    package_versions = _verify_dependencies(bundle)
    render_config = _renderer_config(bundle)
    canvas_width, canvas_height = map(int, render_config["canvas_px"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(_rc_params()):
        fig = plt.figure(
            figsize=(canvas_width / dpi, canvas_height / dpi),
            dpi=dpi,
            facecolor=WHITE,
        )
        grid = fig.add_gridspec(
            2,
            2,
            width_ratios=render_config["width_ratios"],
            height_ratios=render_config["height_ratios"],
            left=0.018,
            right=0.988,
            top=0.985,
            bottom=0.018,
            wspace=0.035,
            hspace=0.055,
        )
        axes = {
            "a": fig.add_subplot(grid[0, 0]),
            "b": fig.add_subplot(grid[0, 1]),
            "c": fig.add_subplot(grid[1, 0]),
            "d": fig.add_subplot(grid[1, 1]),
        }
        groups = {
            "a": _draw_panel_a(axes["a"], bundle),
            "b": _draw_panel_b(axes["b"], bundle, render_config),
            "c": _draw_panel_c(axes["c"], bundle),
            "d": _draw_panel_d(axes["d"], bundle),
        }
        outputs: Dict[str, Path] = {}
        for extension in formats:
            path = output_dir / f"figure_full.{extension}"
            _save_figure(fig, path, dpi=dpi)
            outputs[f"figure_full_{extension}"] = path
        outputs.update(_export_panels(fig, groups, output_dir, formats, dpi))
        plt.close(fig)
    png_path = output_dir / "figure_full.png"
    if png_path.is_file():
        outputs.update(
            _write_accessibility_qa(
                png_path,
                output_dir,
                preview_width=int(render_config["qa_preview_width"]),
                package_versions=package_versions,
            )
        )
    return outputs
