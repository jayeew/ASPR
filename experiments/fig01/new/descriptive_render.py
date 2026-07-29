"""Render the high-density Nature redesign of Fig. 1 from frozen panel tables."""

from __future__ import annotations

import hashlib
import math
import shutil
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

from .descriptive_contract import (
    DOMAIN_LABELS,
    FEATURE_SHORT_LABELS,
    FEATURE_STYLES,
    STAGE_KEYS,
)
from .event_data import canonical_hash, sha256_file, write_json


INK = "#263238"
NAVY = "#18344C"
RETAINED = "#56636D"
GAINED = "#D88918"
LOST = "#B24F7A"
LANDMARK = "#C23B33"
MID_GREY = "#7C8790"
LIGHT_GREY = "#D7DEE2"
SKELETON = "#C8D0D5"
GRID_GREY = "#E8ECEF"
PALE_AMBER = "#F7E9CB"
WHITE = "#FFFFFF"
NODE_PALETTE = (
    "#3C6E9E",
    "#D2772A",
    "#16847A",
    "#9A6FB0",
    "#A85563",
    "#5C7F4F",
    "#6E8792",
    "#C49A3A",
    "#4B8C9D",
    "#B56B9A",
    "#8073AC",
    "#8C6D31",
)


# ============================================================================
# Shared style, saving, and QA
# ============================================================================


def _mm_to_inches(value: float) -> float:
    return float(value) / 25.4


def _set_style(config: Mapping[str, Any]) -> None:
    font = str(config["plot"]["font_family"])
    mpl.rcParams.update(
        {
            "font.family": font,
            "font.size": float(config["plot"]["body_font_pt"]),
            "axes.titlesize": 6.5,
            "axes.labelsize": 6.0,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "axes.linewidth": 0.55,
            "axes.edgecolor": MID_GREY,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.transparent": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "aspr-fig1-old-spacious-transition-v6-0",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "path.simplify": False,
        }
    )


def _figure_size(config: Mapping[str, Any], kind: str) -> Tuple[float, float]:
    prefix = "main" if kind == "main" else "domain"
    return (
        _mm_to_inches(float(config["plot"][f"{prefix}_width_mm"])),
        _mm_to_inches(float(config["plot"][f"{prefix}_height_mm"])),
    )


def _text_bounds_qa(figure: Figure) -> Mapping[str, Any]:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.canvas.get_width_height()
    outside: List[str] = []
    for artist in figure.findobj(match=mpl.text.Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        extent = artist.get_window_extent(renderer=renderer)
        if (
            extent.x0 < -2
            or extent.y0 < -2
            or extent.x1 > width + 2
            or extent.y1 > height + 2
        ):
            outside.append(artist.get_text()[:80])
    return {
        "canvas_width_px": int(width),
        "canvas_height_px": int(height),
        "out_of_canvas_text_count": int(len(outside)),
        "out_of_canvas_text": outside,
    }


def _save_bundle(
    figure: Figure,
    output_base: Path,
    *,
    dpi: int,
) -> Tuple[Mapping[str, Mapping[str, Any]], Mapping[str, Any]]:
    """Save exact-size deterministic PNG/SVG/PDF bundles."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    qa = _text_bounds_qa(figure)
    artifacts: Dict[str, Mapping[str, Any]] = {}
    for extension in ("png", "svg", "pdf"):
        path = output_base.with_suffix(f".{extension}")
        if extension == "png":
            metadata = {"Software": "ASPR deterministic Fig.1 renderer"}
        elif extension == "svg":
            metadata = {
                "Creator": "ASPR deterministic Fig.1 renderer",
                "Date": None,
            }
        else:
            metadata = {
                "Creator": "ASPR deterministic Fig.1 renderer",
                "Producer": "Matplotlib",
                "CreationDate": None,
                "ModDate": None,
            }
        figure.savefig(
            path,
            dpi=dpi if extension == "png" else None,
            bbox_inches=None,
            pad_inches=0,
            metadata=metadata,
        )
        artifacts[extension] = {
            "path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return artifacts, qa


def _accessibility_previews(
    source: Path,
    output_dir: Path,
) -> Mapping[str, Mapping[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(source).convert("RGB")
    grayscale_path = output_dir / "figure_full_grayscale.png"
    image.convert("L").convert("RGB").save(grayscale_path)
    pixels = np.asarray(image, dtype=float) / 255.0
    deuteranopia = np.asarray(
        [
            [0.367, 0.861, -0.228],
            [0.280, 0.673, 0.047],
            [-0.012, 0.043, 0.969],
        ],
        dtype=float,
    )
    simulated = np.clip(pixels @ deuteranopia.T, 0.0, 1.0)
    deuteranopia_path = output_dir / "figure_full_deuteranopia.png"
    Image.fromarray(
        np.round(simulated * 255.0).astype(np.uint8),
        mode="RGB",
    ).save(deuteranopia_path)
    return {
        "grayscale": {
            "path": str(grayscale_path.resolve()),
            "sha256": sha256_file(grayscale_path),
            "size_bytes": int(grayscale_path.stat().st_size),
        },
        "deuteranopia": {
            "path": str(deuteranopia_path.resolve()),
            "sha256": sha256_file(deuteranopia_path),
            "size_bytes": int(deuteranopia_path.stat().st_size),
        },
    }


# ============================================================================
# Network transition panel
# ============================================================================


def _topic_colors(nodes: pd.DataFrame) -> Mapping[str, str]:
    """Assign one persistent colour per displayed topic, as in Fig.1 old."""
    ranked = (
        nodes.groupby("node_id", as_index=False)
        .agg(
            landmark_topic=("landmark_topic", "max"),
            maximum_paper_count=("paper_count", "max"),
        )
        .sort_values(
            ["landmark_topic", "maximum_paper_count", "node_id"],
            ascending=[False, False, True],
            kind="stable",
        )
    )
    names = ranked["node_id"].astype(str).tolist()
    return {
        name: NODE_PALETTE[index]
        for index, name in enumerate(names[: len(NODE_PALETTE)])
    }


def _edge_curvature(source: str, target: str) -> float:
    digest = hashlib.sha256(
        f"{min(source, target)}|{max(source, target)}".encode("utf-8")
    ).digest()
    sign = -1.0 if digest[0] % 2 else 1.0
    magnitude = 0.075 + (digest[1] / 255.0) * 0.075
    return sign * magnitude


def _draw_curved_edge(
    axis: Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    source: str,
    target: str,
    color: str,
    linewidth: float,
    linestyle: str | Tuple[int, Tuple[int, ...]],
    alpha: float,
    zorder: float,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-",
        connectionstyle=f"arc3,rad={_edge_curvature(source, target):.4f}",
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    axis.add_patch(patch)


def _node_positions(
    nodes: pd.DataFrame,
    view: str,
) -> Mapping[str, Tuple[float, float]]:
    x_column = f"{view}_x"
    y_column = f"{view}_y"
    display_column = f"display_{view}"
    unique = nodes.loc[nodes[display_column].astype(bool)].drop_duplicates(
        "node_id"
    )
    return {
        str(row.node_id): (
            float(getattr(row, x_column)),
            float(getattr(row, y_column)),
        )
        for row in unique.itertuples(index=False)
    }


def _bead_offsets(
    count: int,
    radius: float,
    topic_id: str,
) -> List[Tuple[float, float]]:
    if count <= 0:
        return []
    digest = hashlib.sha256(topic_id.encode("utf-8")).digest()
    phase = 2.0 * math.pi * digest[0] / 255.0
    return [
        (
            radius * math.cos(phase + 2.0 * math.pi * index / count),
            radius * math.sin(phase + 2.0 * math.pi * index / count),
        )
        for index in range(count)
    ]


def _network_footer(
    summary: pd.Series,
    view: str,
    compact: bool,
) -> Tuple[str, str]:
    topics = int(summary[f"{view}_active_topic_count"])
    active_edges = int(summary[f"{view}_active_edge_count"])
    gained = int(summary[f"{view}_gained_edge_count"])
    lost = int(summary[f"{view}_lost_edge_count"])
    if int(summary["stage_index"]) == 0:
        transition = (
            f"baseline · E{active_edges}"
            if compact
            else f"baseline · {active_edges} edges"
        )
    else:
        transition = (
            f"Δn +{int(summary['increment_paper_count'])}\n"
            f"E +{gained}/−{lost}"
            if compact
            else f"Δn +{int(summary['increment_paper_count'])}\n"
            f"+{gained} / −{lost} edges"
        )
    return (
        f"n={int(summary['paper_count'])}\n{topics} topics",
        transition,
    )


def _draw_network(
    axis: Axes,
    *,
    nodes: pd.DataFrame,
    transitions: pd.DataFrame,
    representatives: pd.DataFrame,
    summary: pd.Series,
    view: str,
    topic_colors: Mapping[str, str],
    graph_settings: Mapping[str, Any],
    compact: bool,
) -> None:
    """Draw one fixed-layout graph with skeleton and signed transitions."""
    stage_index = int(summary["stage_index"])
    display_column = f"display_{view}"
    stage_nodes = nodes[
        nodes["stage_index"].eq(stage_index)
        & nodes[display_column].astype(bool)
    ].copy()
    suppressed_ids = set(
        stage_nodes.loc[
            stage_nodes["suppressed_pre_landmark"].astype(bool),
            "node_id",
        ].astype(str)
    )
    stage_edges = transitions[
        transitions["display_scope"].eq(view)
        & transitions["stage_index"].eq(stage_index)
    ].copy()
    if suppressed_ids:
        stage_edges = stage_edges[
            ~stage_edges["source"].astype(str).isin(suppressed_ids)
            & ~stage_edges["target"].astype(str).isin(suppressed_ids)
        ].copy()
    stage_papers = representatives[
        representatives["stage_index"].eq(stage_index)
        & representatives[display_column].astype(bool)
    ].copy()
    positions = _node_positions(nodes, view)
    axis.set_axis_off()
    axis.set_xlim(-1.14, 1.14)
    axis.set_ylim(-1.28, 1.14)
    axis.set_aspect("equal", adjustable="box")
    axis.add_patch(
        Rectangle(
            (0.008, 0.025),
            0.984,
            0.945,
            transform=axis.transAxes,
            facecolor="none",
            edgecolor=LIGHT_GREY,
            linewidth=0.36 if compact else 0.48,
            zorder=-1,
        )
    )
    year = (
        f"{int(summary['start_year'])}–"
        f"{str(int(summary['end_year']))[-2:]}"
    )
    axis.text(
        0.5,
        0.982,
        year,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=4.0 if compact else 5.0,
        color=MID_GREY,
        bbox={
            "boxstyle": "round,pad=0.05",
            "facecolor": mpl.colors.to_rgba(WHITE, 0.88),
            "edgecolor": "none",
        },
        clip_on=False,
        zorder=10,
    )
    maximum_edge = max(
        float(stage_edges["union_max_weight"].max())
        if len(stage_edges)
        else 0.0,
        1e-12,
    )
    active = stage_nodes[stage_nodes["active"].astype(bool)].copy()
    active_ids = set(active["node_id"].astype(str))
    # Keep the union layout for positional continuity, but render only topics
    # that are active in the current snapshot. This prevents edges from ending
    # at invisible, inactive union nodes.
    stage_edges = stage_edges[
        stage_edges["source"].astype(str).isin(active_ids)
        & stage_edges["target"].astype(str).isin(active_ids)
    ].copy()
    skeleton = stage_edges.drop_duplicates(["source", "target"])
    for row in skeleton.itertuples(index=False):
        _draw_curved_edge(
            axis,
            positions[str(row.source)],
            positions[str(row.target)],
            source=str(row.source),
            target=str(row.target),
            color=SKELETON,
            linewidth=(0.26 if compact else 0.34)
            + (0.42 if compact else 0.58)
            * math.sqrt(float(row.union_max_weight) / maximum_edge),
            linestyle=(0, (1.0, 1.7)),
            alpha=0.28,
            zorder=0.2,
        )
    status_style = {
        "baseline": (RETAINED, "-", 0.70),
        "retained": (RETAINED, "-", 0.78),
        "gained": (GAINED, "-", 0.92),
        "lost": (LOST, (0, (2.5, 1.8)), 0.90),
    }
    visible = stage_edges[stage_edges["status"].isin(status_style)]
    visible = visible.sort_values(
        ["status", "display_weight"],
        ascending=[True, True],
        kind="stable",
    )
    for row in visible.itertuples(index=False):
        color, linestyle, alpha = status_style[str(row.status)]
        _draw_curved_edge(
            axis,
            positions[str(row.source)],
            positions[str(row.target)],
            source=str(row.source),
            target=str(row.target),
            color=color,
            linewidth=(0.55 if compact else 0.72)
            + (1.15 if compact else 1.55)
            * math.sqrt(float(row.display_weight) / maximum_edge),
            linestyle=linestyle,
            alpha=alpha,
            zorder=1.0 if row.status != "gained" else 1.4,
        )
    maximum_count = max(
        int(nodes.loc[nodes[display_column].astype(bool), "paper_count"].max()),
        1,
    )
    for row in active.itertuples(index=False):
        x = float(getattr(row, f"{view}_x"))
        y = float(getattr(row, f"{view}_y"))
        color = topic_colors.get(str(row.node_id), "#6E8792")
        relative_count = float(row.paper_count) / maximum_count
        outer_area = (
            (62.0 if compact else 112.0)
            + (205.0 if compact else 350.0)
            * relative_count
        )
        inner_area = (
            (38.0 if compact else 68.0)
            + (138.0 if compact else 235.0)
            * float(row.paper_count)
            / maximum_count
        )
        axis.scatter(
            [x],
            [y],
            s=outer_area,
            facecolors=mpl.colors.to_rgba(color, 0.075),
            edgecolors="none",
            zorder=2.2,
        )
        axis.scatter(
            [x],
            [y],
            s=inner_area,
            facecolors=mpl.colors.to_rgba(color, 0.20),
            edgecolors=GAINED if bool(row.new_node) else color,
            linewidths=1.15 if bool(row.new_node) else 0.65,
            zorder=2.5,
        )
        axis.scatter(
            [x],
            [y],
            s=8.5 if compact else 13.5,
            facecolors=color,
            edgecolors=WHITE,
            linewidths=0.45,
            zorder=4,
        )
        papers = stage_papers[stage_papers["topic_id"].eq(str(row.node_id))]
        radius = 0.055 + 0.078 * math.sqrt(relative_count)
        offsets = _bead_offsets(len(papers), radius, str(row.node_id))
        if bool(graph_settings["show_internal_cluster_spokes"]):
            for dx, dy in offsets:
                axis.plot(
                    [x, x + dx],
                    [y, y + dy],
                    color=color,
                    linewidth=0.28 if compact else 0.38,
                    alpha=0.28,
                    zorder=3.1,
                )
        for paper, (dx, dy) in zip(
            papers.itertuples(index=False),
            offsets,
        ):
            axis.scatter(
                [x + dx],
                [y + dy],
                s=5.8 if compact else 9.2,
                facecolors=LANDMARK if bool(paper.is_landmark) else color,
                edgecolors=WHITE,
                linewidths=0.32,
                zorder=4.4,
            )
    if stage_index >= 1:
        landmarks = active[active["landmark_topic"].astype(bool)]
        if not landmarks.empty:
            axis.scatter(
                landmarks[f"{view}_x"],
                landmarks[f"{view}_y"],
                s=102 if compact else 162,
                facecolors="none",
                edgecolors=LANDMARK,
                linewidths=0.72 if compact else 0.92,
                alpha=0.72,
                zorder=6.7,
            )
            axis.scatter(
                landmarks[f"{view}_x"],
                landmarks[f"{view}_y"],
                s=72 if compact else 115,
                marker="*",
                color=LANDMARK,
                edgecolors=WHITE,
                linewidths=0.55,
                zorder=7,
            )
    label_count = int(
        graph_settings[
            "main_maximum_labels_per_snapshot"
            if compact
            else "detail_maximum_labels_per_snapshot"
        ]
    )
    if "community_relation" not in active:
        active["community_relation"] = "field_backbone_context"
    if "landmark_coupling_weight" not in active:
        active["landmark_coupling_weight"] = 0.0
    active["_direct_landmark_neighbor"] = active[
        "community_relation"
    ].eq("direct_landmark_neighbor")
    labels = active.sort_values(
        [
            "landmark_topic",
            "_direct_landmark_neighbor",
            "landmark_coupling_weight",
            "paper_count",
            "node_id",
        ],
        ascending=[False, False, False, False, True],
        kind="stable",
    ).head(label_count)
    label_specs: List[Tuple[Any, float, float, Any, str, str, str]] = []
    if compact:
        compact_labels = labels.assign(
            _label_side=np.where(
                labels[f"{view}_x"].astype(float).ge(0.0),
                "right",
                "left",
            )
        )
        for side in ("left", "right"):
            side_rows = compact_labels[
                compact_labels["_label_side"].eq(side)
            ].sort_values(f"{view}_y", ascending=False, kind="stable")
            slots = np.linspace(
                0.60 if side == "left" else 0.24,
                -0.10 if side == "left" else -0.46,
                max(1, len(side_rows)),
            )
            for slot, row in zip(
                slots,
                side_rows.itertuples(index=False),
            ):
                label_specs.append(
                    (
                        row,
                        float(getattr(row, f"{view}_x")),
                        float(getattr(row, f"{view}_y")),
                        (-1.03 if side == "left" else 1.03, float(slot)),
                        "data",
                        "left" if side == "left" else "right",
                        "center",
                    )
                )
    else:
        for row in labels.itertuples(index=False):
            node_x = float(getattr(row, f"{view}_x"))
            node_y = float(getattr(row, f"{view}_y"))
            offset = 6.2
            if node_x >= 0.0:
                dx = -offset
                horizontal = "right"
            else:
                dx = offset
                horizontal = "left"
            if node_y >= 0.0:
                dy = -0.72 * offset
                vertical = "top"
            else:
                dy = 0.72 * offset
                vertical = "bottom"
            label_specs.append(
                (
                    row,
                    node_x,
                    node_y,
                    (dx, dy),
                    "offset points",
                    horizontal,
                    vertical,
                )
            )
    for row, node_x, node_y, xytext, textcoords, horizontal, vertical in (
        label_specs
    ):
        axis.annotate(
            textwrap.shorten(
                str(row.topic_label),
                width=16 if compact else 25,
                placeholder="…",
            ),
            xy=(node_x, node_y),
            xytext=xytext,
            textcoords=textcoords,
            ha=horizontal,
            va=vertical,
            fontsize=4.05 if compact else 5.15,
            color=INK,
            arrowprops={
                "arrowstyle": "-",
                "color": MID_GREY,
                "linewidth": 0.28 if compact else 0.38,
                "alpha": 0.58,
                "shrinkA": 0.0,
                "shrinkB": 2.0,
            },
            bbox={
                "boxstyle": "round,pad=0.07",
                "facecolor": mpl.colors.to_rgba(WHITE, 0.82),
                "edgecolor": "none",
            },
            zorder=8,
        )
    footer_left, footer_right = _network_footer(summary, view, compact)
    axis.text(
        0.035,
        0.043,
        footer_left,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=3.85 if compact else 4.9,
        color=MID_GREY,
        zorder=9,
    )
    axis.text(
        0.965,
        0.043,
        footer_right,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=3.75 if compact else 4.75,
        color=GAINED if stage_index > 0 else MID_GREY,
        zorder=9,
    )


# ============================================================================
# Annual trajectory and descriptive forest panels
# ============================================================================


def _trajectory_strip(
    axis: Axes,
    rows: pd.DataFrame,
    *,
    y_limit: float,
    compact: bool,
    show_x: bool,
    show_y: bool,
) -> int:
    rows = rows.sort_values("event_time", kind="stable")
    feature = str(rows["feature"].iloc[0])
    color, marker, linestyle = FEATURE_STYLES[feature]
    x = rows["event_time"].to_numpy(dtype=float)
    median = rows["delta_median"].to_numpy(dtype=float)
    q25 = rows["delta_q25"].to_numpy(dtype=float)
    q75 = rows["delta_q75"].to_numpy(dtype=float)
    axis.axvspan(-0.5, 2.5, color=PALE_AMBER, alpha=0.72, lw=0, zorder=0)
    for guide in (-0.5 * y_limit, 0.0, 0.5 * y_limit):
        axis.axhline(
            guide,
            color=RETAINED if guide == 0 else GRID_GREY,
            linewidth=0.45 if guide == 0 else 0.35,
            linestyle="-" if guide == 0 else (0, (1.5, 2.0)),
            alpha=0.72,
            zorder=0.4,
        )
    axis.fill_between(
        x,
        q25,
        q75,
        color=color,
        alpha=0.13,
        linewidth=0,
        zorder=1,
    )
    axis.plot(
        x,
        median,
        color=color,
        linestyle=linestyle,
        linewidth=0.68 if compact else 0.88,
        alpha=0.68,
        zorder=1.8,
    )
    emphasized = x >= -1
    axis.plot(
        x[emphasized],
        median[emphasized],
        color=color,
        linestyle=linestyle,
        linewidth=1.15 if compact else 1.48,
        marker=marker,
        markersize=2.5 if compact else 3.2,
        markerfacecolor=WHITE,
        markeredgewidth=0.55,
        alpha=0.98,
        zorder=2,
    )
    pre_event = x < -1
    axis.scatter(
        x[pre_event],
        median[pre_event],
        marker=marker,
        s=4.8 if compact else 7.0,
        facecolors=WHITE,
        edgecolors=color,
        linewidths=0.45,
        alpha=0.72,
        zorder=2,
    )
    missing = rows[~rows["eligible"].astype(bool)]
    if not missing.empty:
        axis.scatter(
            missing["event_time"],
            np.zeros(len(missing)),
            marker="x",
            s=8 if compact else 13,
            color=MID_GREY,
            linewidths=0.55,
            zorder=3,
        )
    clipped = int(
        np.sum(np.isfinite(median) & (np.abs(median) > y_limit))
        + np.sum(np.isfinite(q25) & (np.abs(q25) > y_limit))
        + np.sum(np.isfinite(q75) & (np.abs(q75) > y_limit))
    )
    if clipped:
        high = rows[
            rows[["delta_median", "delta_q75"]]
            .max(axis=1, skipna=True)
            .gt(y_limit)
        ]
        low = rows[
            rows[["delta_median", "delta_q25"]]
            .min(axis=1, skipna=True)
            .lt(-y_limit)
        ]
        axis.scatter(
            high["event_time"],
            np.full(len(high), y_limit * 0.96),
            marker="^",
            s=7,
            facecolors=WHITE,
            edgecolors=color,
            linewidths=0.5,
            zorder=4,
        )
        axis.scatter(
            low["event_time"],
            np.full(len(low), -y_limit * 0.96),
            marker="v",
            s=7,
            facecolors=WHITE,
            edgecolors=color,
            linewidths=0.5,
            zorder=4,
        )
    axis.text(
        0.012,
        0.91,
        FEATURE_SHORT_LABELS[feature],
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=4.65 if compact else 5.6,
        color=color,
        fontweight="bold",
    )
    axis.set_xlim(-6.25, 8.25)
    axis.set_ylim(-y_limit, y_limit)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_visible(show_x)
    axis.tick_params(axis="both", length=1.8, width=0.45, pad=1.0)
    if show_x:
        axis.set_xticks([-6, -3, 0, 3, 6, 8])
        axis.set_xticklabels(["−6", "−3", "0", "+3", "+6", "+8"])
        axis.set_xlabel("Years from landmark start", labelpad=0.5)
    else:
        axis.set_xticks([])
    if show_y:
        limit_points = int(round(y_limit * 100))
        axis.set_yticks([-y_limit, 0.0, y_limit])
        axis.set_yticklabels(
            [f"−{limit_points}", "0", f"+{limit_points}"]
        )
    else:
        axis.set_yticks([])
    return clipped


def _forest_strip(
    axis: Axes,
    row: pd.Series,
    *,
    x_limit: float,
    compact: bool,
    show_x: bool,
) -> int:
    feature = str(row["feature"])
    color, marker, _ = FEATURE_STYLES[feature]
    axis.axvline(0, color=RETAINED, linewidth=0.55, zorder=0)
    effect = float(row["effect"])
    low = float(row["ci_low"])
    high = float(row["ci_high"])
    axis.plot(
        [low, high],
        [0.5, 0.5],
        color=color,
        linewidth=1.05 if compact else 1.4,
        solid_capstyle="round",
        zorder=2,
    )
    axis.scatter(
        [effect],
        [0.5],
        s=12 if compact else 20,
        marker=marker,
        facecolors=color,
        edgecolors=WHITE,
        linewidths=0.45,
        zorder=3,
    )
    clipped = int(low < -x_limit or high > x_limit or abs(effect) > x_limit)
    if low < -x_limit:
        axis.scatter(
            [-x_limit * 0.96],
            [0.5],
            marker="<",
            s=8,
            facecolors=WHITE,
            edgecolors=color,
            linewidths=0.5,
        )
    if high > x_limit:
        axis.scatter(
            [x_limit * 0.96],
            [0.5],
            marker=">",
            s=8,
            facecolors=WHITE,
            edgecolors=color,
            linewidths=0.5,
        )
    effect_points = int(round(effect * 100.0))
    axis.text(
        0.985,
        0.94,
        f"{effect_points:+d} pp",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=4.25 if compact else 5.1,
        color=color,
        fontweight="bold",
    )
    axis.set_xlim(-x_limit, x_limit)
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_visible(show_x)
    axis.tick_params(axis="x", length=1.8, width=0.45, pad=1.0)
    if show_x:
        axis.set_xticks([-0.5, 0, 0.5])
        axis.set_xticklabels(["−50", "0", "+50"])
        axis.set_xlabel("Late − pre (pp)", labelpad=0.5)
    else:
        axis.set_xticks([])
    return clipped


def _domain_tables(
    domain: str,
    *,
    nodes: pd.DataFrame,
    transitions: pd.DataFrame,
    representatives: pd.DataFrame,
    summaries: pd.DataFrame,
    annual: pd.DataFrame,
    trajectory_scales: pd.DataFrame,
    effects: pd.DataFrame,
) -> Tuple[pd.DataFrame, ...]:
    return tuple(
        frame.loc[frame["domain"].eq(domain)].copy()
        for frame in (
            nodes,
            transitions,
            representatives,
            summaries,
            annual,
            trajectory_scales,
            effects,
        )
    )


def _indicator_order(
    annual: pd.DataFrame,
) -> List[str]:
    return (
        annual[["feature", "display_rank"]]
        .drop_duplicates()
        .sort_values("display_rank", kind="stable")["feature"]
        .astype(str)
        .tolist()
    )


def _stage_progression_arrows(
    figure: Figure,
    axes: Sequence[Axes],
    *,
    compact: bool,
) -> None:
    """Add old-style temporal progression arrows between fixed snapshots."""
    for left_axis, right_axis in zip(axes[:-1], axes[1:]):
        left = left_axis.get_position()
        right = right_axis.get_position()
        start_x = left.x1 + 0.0007
        end_x = right.x0 - 0.0007
        if end_x <= start_x:
            continue
        center_y = 0.5 * (left.y0 + left.y1)
        figure.add_artist(
            FancyArrowPatch(
                (start_x, center_y),
                (end_x, center_y),
                transform=figure.transFigure,
                arrowstyle="-|>",
                mutation_scale=4.3 if compact else 5.8,
                linewidth=0.48 if compact else 0.65,
                color=MID_GREY,
                alpha=0.72,
                clip_on=False,
                zorder=20,
            )
        )


# ============================================================================
# Combined Nature double-column figure
# ============================================================================


def _header_row(
    figure: Figure,
    outer: GridSpec,
) -> None:
    network_axis = figure.add_subplot(outer[0, 0])
    network_axis.set_axis_off()
    network_axis.text(
        0.0,
        0.80,
        "a",
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="center",
    )
    network_axis.text(
        0.055,
        0.80,
        "Topic-coupling transitions",
        fontsize=6.2,
        fontweight="bold",
        color=NAVY,
        ha="left",
        va="center",
    )
    stage_labels = (
        "t−6 to t−1",
        "through t+2",
        "through t+5",
        "through t+8",
    )
    for index, label in enumerate(stage_labels):
        center = (0.62 + index + 0.5) / 4.62
        network_axis.text(
            center,
            0.16,
            label,
            fontsize=5.3,
            fontweight="bold" if index == 1 else "normal",
            color=GAINED if index == 1 else INK,
            ha="center",
            va="center",
        )
    trajectory_header = figure.add_subplot(outer[0, 1])
    trajectory_header.set_axis_off()
    trajectory_header.text(
        0.0,
        0.80,
        "b",
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="center",
    )
    trajectory_header.text(
        0.15,
        0.80,
        "Annual indicator trajectories",
        fontsize=5.9,
        fontweight="bold",
        color=NAVY,
        ha="left",
        va="center",
    )
    forest_header = figure.add_subplot(outer[0, 2])
    forest_header.set_axis_off()
    forest_header.text(
        0.0,
        0.80,
        "c",
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="center",
    )
    forest_header.text(
        0.22,
        0.80,
        "Late − pre",
        fontsize=5.7,
        fontweight="bold",
        color=NAVY,
        ha="left",
        va="center",
    )
    forest_header.text(
        0.22,
        0.28,
        "t+6:8 − t−3:−1\n95% CI · ±70 pp",
        fontsize=3.7,
        color=MID_GREY,
        ha="left",
        va="top",
    )


def _row_domain_label(
    axis: Axes,
    row: pd.Series,
) -> None:
    axis.set_axis_off()
    domain = str(row["domain"])
    label = DOMAIN_LABELS.get(domain, domain)
    label = {
        "CRISPR–Cas genome editing": "CRISPR–Cas",
        "Graphene and 2D materials": "Graphene / 2D",
        "Click chemistry (CuAAC)": "Click chemistry",
        "Electrospray mass spectrometry": "Electrospray MS",
    }.get(label, label)
    axis.text(
        0.20,
        0.54,
        label,
        fontsize=6.2,
        fontweight="bold",
        color=NAVY,
        ha="center",
        va="center",
        rotation=90,
    )
    start = int(row["landmark_start_year"])
    end = int(row["landmark_end_year"])
    year = str(start) if start == end else f"{start}–{str(end)[-2:]}"
    axis.text(
        0.72,
        0.61,
        f"LM\n{year}",
        fontsize=4.35,
        color=LANDMARK,
        ha="center",
        va="center",
    )


def _main_figure(
    config: Mapping[str, Any],
    domain_selection: pd.DataFrame,
    nodes: pd.DataFrame,
    transitions: pd.DataFrame,
    representatives: pd.DataFrame,
    summaries: pd.DataFrame,
    annual: pd.DataFrame,
    trajectory_scales: pd.DataFrame,
    effects: pd.DataFrame,
) -> Tuple[Figure, Mapping[str, Any]]:
    selected = domain_selection[
        domain_selection["selected"].astype(bool)
    ].sort_values("selection_rank", kind="stable")
    figure = plt.figure(figsize=_figure_size(config, "main"))
    outer = GridSpec(
        5,
        3,
        figure=figure,
        width_ratios=[0.585, 0.285, 0.130],
        height_ratios=[0.16, 1, 1, 1, 1],
        left=0.018,
        right=0.992,
        top=0.992,
        bottom=0.073,
        hspace=0.105,
        wspace=0.075,
    )
    _header_row(figure, outer)
    clipped_trajectory = 0
    clipped_effect = 0
    for row_index, row in enumerate(
        selected.to_dict("records"),
        start=1,
    ):
        row_series = pd.Series(row)
        domain = str(row["domain"])
        (
            domain_nodes,
            domain_transitions,
            domain_representatives,
            domain_summaries,
            domain_annual,
            domain_scales,
            domain_effects,
        ) = _domain_tables(
            domain,
            nodes=nodes,
            transitions=transitions,
            representatives=representatives,
            summaries=summaries,
            annual=annual,
            trajectory_scales=trajectory_scales,
            effects=effects,
        )
        network_grid = GridSpecFromSubplotSpec(
            1,
            5,
            subplot_spec=outer[row_index, 0],
            width_ratios=[0.50, 1, 1, 1, 1],
            wspace=0.040,
        )
        label_axis = figure.add_subplot(network_grid[0, 0])
        _row_domain_label(label_axis, row_series)
        colors = _topic_colors(domain_nodes)
        network_axes: List[Axes] = []
        for stage_index in range(len(STAGE_KEYS)):
            axis = figure.add_subplot(network_grid[0, stage_index + 1])
            network_axes.append(axis)
            summary = domain_summaries[
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
            _stage_progression_arrows(
                figure,
                network_axes,
                compact=True,
            )
        feature_order = _indicator_order(domain_annual)
        domain_y_limit = float(
            domain_scales["domain_shared_display_limit"].iloc[0]
        )
        trajectory_grid = GridSpecFromSubplotSpec(
            len(feature_order),
            1,
            subplot_spec=outer[row_index, 1],
            hspace=0.025,
        )
        forest_grid = GridSpecFromSubplotSpec(
            len(feature_order),
            1,
            subplot_spec=outer[row_index, 2],
            hspace=0.025,
        )
        for feature_index, feature in enumerate(feature_order):
            show_x = (
                row_index == len(selected)
                and feature_index == len(feature_order) - 1
            )
            trajectory_axis = figure.add_subplot(
                trajectory_grid[feature_index, 0]
            )
            clipped_trajectory += _trajectory_strip(
                trajectory_axis,
                domain_annual[domain_annual["feature"].eq(feature)],
                y_limit=domain_y_limit,
                compact=True,
                show_x=show_x,
                show_y=feature_index == 0,
            )
            forest_axis = figure.add_subplot(
                forest_grid[feature_index, 0]
            )
            effect_row = domain_effects[
                domain_effects["feature"].eq(feature)
            ].iloc[0]
            clipped_effect += _forest_strip(
                forest_axis,
                effect_row,
                x_limit=float(
                    config["indicators"]["trajectory_display_scale"][
                        "shared_effect_limit"
                    ]
                ),
                compact=True,
                show_x=show_x,
            )
    legend = [
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
            markersize=6.0,
            label="landmark topic",
        ),
    ]
    figure.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.018, 0.030),
        ncol=5,
        frameon=False,
        fontsize=4.6,
        handlelength=1.8,
        columnspacing=0.9,
        borderaxespad=0,
    )
    return figure, {
        "clipped_trajectory_elements": int(clipped_trajectory),
        "clipped_effect_intervals": int(clipped_effect),
    }


# ============================================================================
# Four larger domain-detail figures
# ============================================================================


def _domain_figure(
    config: Mapping[str, Any],
    selection_row: pd.Series,
    nodes: pd.DataFrame,
    transitions: pd.DataFrame,
    representatives: pd.DataFrame,
    summaries: pd.DataFrame,
    annual: pd.DataFrame,
    trajectory_scales: pd.DataFrame,
    effects: pd.DataFrame,
    landmarks: pd.DataFrame,
) -> Tuple[Figure, Mapping[str, Any]]:
    domain = str(selection_row["domain"])
    figure = plt.figure(figsize=_figure_size(config, "domain"))
    outer = GridSpec(
        3,
        2,
        figure=figure,
        width_ratios=[0.76, 0.24],
        height_ratios=[0.18, 1.55, 1.0],
        left=0.035,
        right=0.985,
        top=0.985,
        bottom=0.125,
        hspace=0.14,
        wspace=0.075,
    )
    header = figure.add_subplot(outer[0, :])
    header.set_axis_off()
    header.text(
        0.0,
        0.70,
        "a",
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="center",
    )
    header.text(
        0.035,
        0.70,
        DOMAIN_LABELS.get(domain, domain),
        fontsize=8.2,
        fontweight="bold",
        color=NAVY,
        ha="left",
        va="center",
    )
    header.text(
        0.99,
        0.70,
        str(selection_row["episode_label"]),
        fontsize=5.8,
        color=LANDMARK,
        ha="right",
        va="center",
    )
    network_grid = GridSpecFromSubplotSpec(
        1,
        4,
        subplot_spec=outer[1, :],
        wspace=0.035,
    )
    colors = _topic_colors(nodes)
    stage_titles = (
        "t−6 to t−1",
        "through t+2",
        "through t+5",
        "through t+8",
    )
    network_axes: List[Axes] = []
    for stage_index, stage_title in enumerate(stage_titles):
        axis = figure.add_subplot(network_grid[0, stage_index])
        network_axes.append(axis)
        _draw_network(
            axis,
            nodes=nodes,
            transitions=transitions,
            representatives=representatives,
            summary=summaries[
                summaries["stage_index"].eq(stage_index)
            ].iloc[0],
            view="detail",
            topic_colors=colors,
            graph_settings=config["graph"],
            compact=False,
        )
        axis.text(
            0.5,
            1.01,
            stage_title,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=5.8,
            color=GAINED if stage_index == 1 else INK,
            fontweight="bold" if stage_index == 1 else "normal",
        )
    if bool(config["graph"]["show_stage_progression_arrows"]):
        _stage_progression_arrows(
            figure,
            network_axes,
            compact=False,
        )
    feature_order = _indicator_order(annual)
    domain_y_limit = float(
        trajectory_scales["domain_shared_display_limit"].iloc[0]
    )
    trajectory_grid = GridSpecFromSubplotSpec(
        len(feature_order),
        1,
        subplot_spec=outer[2, 0],
        hspace=0.035,
    )
    forest_grid = GridSpecFromSubplotSpec(
        len(feature_order),
        1,
        subplot_spec=outer[2, 1],
        hspace=0.035,
    )
    clipped_trajectory = 0
    clipped_effect = 0
    for feature_index, feature in enumerate(feature_order):
        show_x = feature_index == len(feature_order) - 1
        trajectory_axis = figure.add_subplot(
            trajectory_grid[feature_index, 0]
        )
        clipped_trajectory += _trajectory_strip(
            trajectory_axis,
            annual[annual["feature"].eq(feature)],
            y_limit=domain_y_limit,
            compact=False,
            show_x=show_x,
            show_y=feature_index == 0,
        )
        if feature_index == 0:
            trajectory_axis.text(
                0.0,
                1.16,
                "b",
                transform=trajectory_axis.transAxes,
                fontsize=8.0,
                fontweight="bold",
                ha="left",
                va="center",
            )
            trajectory_axis.text(
                0.025,
                1.16,
                "Annual publication-time indicator change",
                transform=trajectory_axis.transAxes,
                fontsize=6.2,
                fontweight="bold",
                color=NAVY,
                ha="left",
                va="center",
            )
        forest_axis = figure.add_subplot(forest_grid[feature_index, 0])
        effect_row = effects[effects["feature"].eq(feature)].iloc[0]
        clipped_effect += _forest_strip(
            forest_axis,
            effect_row,
            x_limit=float(
                config["indicators"]["trajectory_display_scale"][
                    "shared_effect_limit"
                ]
            ),
            compact=False,
            show_x=show_x,
        )
        if feature_index == 0:
            forest_axis.text(
                0.0,
                1.16,
                "c",
                transform=forest_axis.transAxes,
                fontsize=8.0,
                fontweight="bold",
                ha="left",
                va="center",
            )
            forest_axis.text(
                0.12,
                1.16,
                "Late t+6:t+8 − pre t−3:t−1",
                transform=forest_axis.transAxes,
                fontsize=5.6,
                fontweight="bold",
                color=NAVY,
                ha="left",
                va="center",
            )
    landmark_text = " | ".join(
        f"{int(row.publication_year)}: {row.title}"
        for row in landmarks.itertuples(index=False)
    )
    figure.text(
        0.035,
        0.025,
        "Landmark: "
        + textwrap.shorten(landmark_text, width=105, placeholder="…"),
        ha="left",
        va="bottom",
        fontsize=4.8,
        color=LANDMARK,
    )
    return figure, {
        "clipped_trajectory_elements": int(clipped_trajectory),
        "clipped_effect_intervals": int(clipped_effect),
    }


# ============================================================================
# Public render entry
# ============================================================================


def render_descriptive_figure(
    config: Mapping[str, Any],
    output_dir: Path,
) -> Mapping[str, Any]:
    """Render the combined figure and four larger domain-detail figures."""
    _set_style(config)
    panel_data = output_dir / "panel_data"
    domain_selection = pd.read_csv(panel_data / "domain_selection.csv")
    selected = domain_selection[
        domain_selection["selected"].astype(bool)
    ].sort_values("selection_rank", kind="stable")
    selected_domains = set(selected["domain"].astype(str))
    domains_root = output_dir / "domains"
    if domains_root.is_dir():
        for child in domains_root.iterdir():
            if child.is_dir() and child.name not in selected_domains:
                shutil.rmtree(child)
    nodes = pd.read_parquet(panel_data / "snapshot_nodes.parquet")
    transitions = pd.read_parquet(panel_data / "transition_edges.parquet")
    representatives = pd.read_parquet(
        panel_data / "representative_papers.parquet"
    )
    summaries = pd.read_csv(panel_data / "snapshot_summary.csv")
    annual = pd.read_csv(panel_data / "annual_indicator_trajectories.csv")
    trajectory_scales = pd.read_csv(
        panel_data / "trajectory_display_scales.csv"
    )
    effects = pd.read_csv(panel_data / "indicator_effects.csv")
    indicator_display = pd.read_csv(
        panel_data / "indicator_display_filter.csv"
    )
    display_keys = indicator_display[
        indicator_display["display"].astype(bool)
    ][["episode_id", "domain", "feature"]].drop_duplicates()
    annual_display = annual.merge(
        display_keys,
        on=["episode_id", "domain", "feature"],
        how="inner",
        validate="many_to_one",
    )
    scale_display = trajectory_scales.merge(
        display_keys,
        on=["episode_id", "domain", "feature"],
        how="inner",
        validate="one_to_one",
    )
    effect_display = effects.merge(
        display_keys,
        on=["episode_id", "domain", "feature"],
        how="inner",
        validate="one_to_one",
    )
    landmarks = pd.read_csv(panel_data / "landmark_papers.csv")
    main, main_data_qa = _main_figure(
        config,
        domain_selection,
        nodes,
        transitions,
        representatives,
        summaries,
        annual_display,
        scale_display,
        effect_display,
    )
    main_artifacts, main_layout_qa = _save_bundle(
        main,
        output_dir / "figure_full",
        dpi=int(config["plot"]["dpi"]),
    )
    artifacts: Dict[str, Any] = {"figure_full": main_artifacts}
    layout_qa: Dict[str, Any] = {
        "figure_full": {**main_layout_qa, **main_data_qa}
    }
    plt.close(main)
    accessibility = _accessibility_previews(
        Path(main_artifacts["png"]["path"]),
        output_dir / "qa",
    )
    for row in selected.itertuples(index=False):
        domain = str(row.domain)
        (
            domain_nodes,
            domain_transitions,
            domain_representatives,
            domain_summaries,
            domain_annual,
            domain_scales,
            domain_effects,
        ) = _domain_tables(
            domain,
            nodes=nodes,
            transitions=transitions,
            representatives=representatives,
            summaries=summaries,
            annual=annual_display,
            trajectory_scales=scale_display,
            effects=effect_display,
        )
        domain_figure, domain_data_qa = _domain_figure(
            config,
            pd.Series(row._asdict()),
            domain_nodes,
            domain_transitions,
            domain_representatives,
            domain_summaries,
            domain_annual,
            domain_scales,
            domain_effects,
            landmarks[landmarks["domain"].eq(domain)],
        )
        output_base = (
            output_dir / "domains" / domain / f"figure_{domain}"
        )
        domain_artifacts, domain_layout_qa = _save_bundle(
            domain_figure,
            output_base,
            dpi=int(config["plot"]["dpi"]),
        )
        artifacts[f"domain_{domain}"] = domain_artifacts
        layout_qa[f"domain_{domain}"] = {
            **domain_layout_qa,
            **domain_data_qa,
        }
        plt.close(domain_figure)
    selected_records = selected[
        [
            "selection_rank",
            "episode_id",
            "domain",
            "episode_label",
            "graph_change_score",
        ]
    ].to_dict("records")
    panel_text = {
        "title": "Landmark-field knowledge-graph transitions",
        "selected_cases": selected_records,
        "displayed_indicator_count": int(len(display_keys)),
        "displayed_indicator_counts_by_domain": {
            str(key): int(value)
            for key, value in display_keys.groupby("domain").size().items()
        },
        "supported_reading": (
            "Within four deliberately selected high-change cases, fixed-layout "
            "cumulative topic-coupling states and a display-filtered subset of "
            "frozen publication-time innovation indicators change around the "
            "landmark window."
        ),
        "unsupported_reading": (
            "The figure does not establish that landmark papers caused the "
            "changes, that the cases are representative, or that the "
            "post-selection descriptive intervals are confirmatory tests."
        ),
    }
    contract = {
        "figure_id": 1,
        "design_version": config["design_version"],
        "analytical_question": (
            "How do topic-coupling structure and frozen publication-time "
            "innovation signals evolve in four deliberately selected "
            "high-change landmark fields?"
        ),
        "takeaway": panel_text["supported_reading"],
        "surface": "standalone_static_python",
        "main_layout": (
            "183 × 168 mm; four field rows; a, four fixed-layout transition "
            "networks accumulating three-year additions from t−6 in the "
            "visual language of Fig.1 old; b, 15-year "
            "indicator small multiples with one explicitly labelled "
            "field-shared symmetric zoom; c, aligned common-scale "
            "post-selection descriptive bootstrap forests"
        ),
        "data_sufficiency": {
            "candidate_domains": 10,
            "selected_domains": int(len(selected)),
            "graph_snapshots_per_case": 4,
            "graph_history_start_offset": int(
                config["graph"]["history_start_offset"]
            ),
            "graph_snapshot_mode": "cumulative",
            "years_per_graph_addition": int(
                config["windows"]["years_per_stage"]
            ),
            "annual_slots_per_indicator": 15,
            "registered_selected_indicators": int(len(indicator_display)),
            "displayed_indicators": int(len(display_keys)),
            "annual_minimum_valid_n": int(
                config["indicators"]["minimum_annual_valid"]
            ),
            "bootstrap_draws": int(config["bootstrap"]["draws"]),
        },
        "selection_disclosure": {
            "graph_change_used_for_domain_selection": True,
            "indicator_change_used_for_indicator_selection": True,
            "display_case_refresh_used_indicator_contrasts": True,
            "selection_frozen_before_current_rerender": True,
            "display_filter_uses_only_indicator_change_magnitude": True,
            "display_filter_changes_model_features": False,
            "future_impact_outcome_used": False,
            "selection_is_exploratory": True,
        },
        "visual_encoding": {
            "retained_edges": "dark-grey solid",
            "gained_edges": "amber solid",
            "lost_edges": "magenta dashed",
            "union_skeleton": "pale-grey dotted",
            "inactive_topic_policy": (
                "inactive union topics remain available for fixed-layout "
                "auditing but are not rendered; edges are limited to pairs "
                "of currently active topics"
            ),
            "new_topics": "amber outline",
            "landmark_topic": (
                "red star and outline absent before t0, introduced in the "
                "landmark snapshot, and retained in all later snapshots"
            ),
            "topic_halo_area": "all papers assigned to displayed topic",
            "topic_colour": (
                "one persistent categorical colour per topic, not one colour "
                "per subfield"
            ),
            "paper_beads": "up to five deterministic real papers",
            "internal_cluster_spokes": (
                "thin topic-coloured spokes visually bind representative "
                "paper beads to their topic halo"
            ),
            "stage_progression_arrows": (
                "grey arrows encode left-to-right publication-window order"
            ),
            "calendar_year_position": (
                "the cumulative calendar-year range is printed above each "
                "network snapshot"
            ),
            "community_text_labels": (
                "landmark topic first; direct landmark-coupling neighbors by "
                "coupling weight next; high-volume field-backbone context last"
            ),
            "trajectory_scale": (
                "all indicator strips within a field share one symmetric "
                "zero-centred limit: the maximum frozen feature limit in "
                "trajectory_display_scales.csv; the explicit ±pp range is "
                "shown once on the field's first strip"
            ),
            "effect_scale": "shared ±70 percentile-point x range",
            "boundary_triangles": (
                "open triangles mark annual medians or IQR bounds outside the "
                "explicitly labelled feature-specific display range"
            ),
        },
        "interval_definition": (
            "2,000 publication-year-stratified bootstrap draws; equal-weight "
            "mean of three year-specific medians in late t+6:t+8 minus pre "
            "t−3:t−1; percentile 95% interval"
        ),
        "indicator_display_rule": {
            "role": str(config["indicator_display"]["role"]),
            "minimum_absolute_late_pre_effect": float(
                config["indicator_display"][
                    "minimum_absolute_late_pre_effect"
                ]
            ),
            "minimum_annual_peak_to_peak": float(
                config["indicator_display"][
                    "minimum_annual_peak_to_peak"
                ]
            ),
            "minimum_per_domain": int(
                config["indicator_display"]["minimum_per_domain"]
            ),
            "maximum_per_domain": int(
                config["indicator_display"]["maximum_per_domain"]
            ),
            "complete_decisions": (
                "panel_data/indicator_display_filter.csv"
            ),
        },
        "palette_policy": (
            "colour-blind-safe restrained roots plus line style, open/filled "
            "markers, and direct labels; grayscale and deuteranopia previews "
            "are exported"
        ),
        "claim_boundary": panel_text["unsupported_reading"],
        "outputs": artifacts,
    }
    render_manifest = {
        "artifact_kind": "fig1_cumulative_transition_render",
        "design_version": config["design_version"],
        "status": "DESCRIPTIVE_SELECTED_CASES",
        "submission_size_mm": {
            "width": float(config["plot"]["main_width_mm"]),
            "height": float(config["plot"]["main_height_mm"]),
        },
        "dpi": int(config["plot"]["dpi"]),
        "artifacts": artifacts,
        "accessibility_previews": accessibility,
        "layout_qa": layout_qa,
    }
    render_manifest["artifact_id"] = canonical_hash(render_manifest)
    write_json(output_dir / "panel_text.json", panel_text)
    write_json(output_dir / "chart_contract.json", contract)
    write_json(output_dir / "render_manifest.json", render_manifest)
    return render_manifest


__all__ = ["render_descriptive_figure"]
