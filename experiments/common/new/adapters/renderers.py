"""Publication-oriented renderers for the new experiment adapters."""

from __future__ import annotations

import math
import shutil
import textwrap
from pathlib import Path
from typing import Dict, Mapping, Sequence

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
)
from matplotlib.transforms import Bbox

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
from experiments.common.new.base.renderers_1_5 import (
    render_fig1,
    render_fig3,
    render_fig5,
)
from experiments.common.new.base.renderers_6_10 import (
    render_fig6,
    render_fig7,
    render_fig9,
    render_fig10,
)
from experiments.common.new.adapters.renderers_fig3_7 import (
    render_fig3_to_fig7,
)


BASE_RENDERERS = {
    1: render_fig1,
    3: render_fig3,
    5: render_fig5,
    6: render_fig6,
    7: render_fig7,
    9: render_fig9,
    10: render_fig10,
}


def _export_axis_groups(
    fig: plt.Figure,
    groups: Mapping[str, Sequence[plt.Axes]],
    figure_dir: Path,
    *,
    prefix: str,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Export one or more axes as independently reusable panel artifacts."""
    panel_dir = figure_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outputs: Dict[str, Path] = {}
    for panel, axes in groups.items():
        extents = [
            axis.get_tightbbox(renderer).transformed(
                fig.dpi_scale_trans.inverted()
            )
            for axis in axes
        ]
        extent = Bbox.union(extents).expanded(1.06, 1.12)
        for extension in formats:
            path = panel_dir / f"{prefix}_{panel}.{extension}"
            kwargs: Dict[str, object] = {"bbox_inches": extent}
            if extension == "png":
                kwargs["dpi"] = dpi
            fig.savefig(path, **kwargs)
            outputs[f"panel_{panel}_{extension}"] = path
    return outputs


def _fig2_blend(color: str, amount: float = 0.88) -> str:
    """Blend one palette root with white for quiet publication fills."""
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    blended = rgb * (1.0 - amount) + np.ones(3, dtype=float) * amount
    return mcolors.to_hex(blended)


def _fig2_panel_frame(
    ax: plt.Axes,
    panel: str,
    title: str,
    subtitle: str,
) -> None:
    """Create one restrained, old-route panel container."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        FancyBboxPatch(
            (0.002, 0.002),
            0.996,
            0.996,
            boxstyle="round,pad=0.006,rounding_size=0.022",
            transform=ax.transAxes,
            facecolor=WHITE,
            edgecolor=LIGHT_GRAY,
            linewidth=0.8,
            clip_on=False,
            zorder=0,
        )
    )
    ax.text(
        0.022,
        0.965,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.080,
        0.960,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.6,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.080,
        0.913,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color=GRAY,
    )


def _fig2_short_feature(label: str) -> str:
    replacements = {
        "Reference-overlap novelty": "Overlap\nnovelty",
        "Median conventionality": "Median\nconventionality ↓",
        "First-pair share": "First-pair\nshare",
        "Field balance": "Field\nbalance",
        "Outside-field references": "Outside-field\nreferences",
        "Field variety": "Field\nvariety",
        "Mean cognitive distance": "Mean cognitive\ndistance",
        "Rao–Stirling integration": "Rao–Stirling\nintegration",
    }
    return replacements.get(str(label), textwrap.fill(str(label), 15))


def _draw_fig2_route_a(
    ax: plt.Axes,
    bundle: FigureBundle,
) -> Sequence[plt.Axes]:
    """Draw the publication-time measurement boundary as a compact triptych."""
    _fig2_panel_frame(
        ax,
        "a",
        "From observable graph change to a publication-time measurement scene",
        "Fig.1 motivates the change; the eight signals are measured only when the focal paper enters G0.",
    )
    nodes = bundle.tables["measurement_scene_nodes"]
    edges = bundle.tables["measurement_scene_edges"]
    manifest = bundle.tables["measurement_scene_manifest"].iloc[0]
    stages = ("G−", "G0", "G+5")
    titles = (
        ("G−", "strictly prior source graph"),
        ("G0", "focal paper + references"),
        ("G+5", "later uptake · validation only"),
    )
    child_axes: list[plt.Axes] = [ax]
    lefts = (0.045, 0.365, 0.685)
    for stage_index, (stage, (headline, caption)) in enumerate(
        zip(stages, titles)
    ):
        inset = ax.inset_axes([lefts[stage_index], 0.235, 0.270, 0.535])
        child_axes.append(inset)
        inset.set_axis_off()
        inset.set_facecolor("#FBFCFD")
        stage_nodes = nodes.loc[nodes["stage"].eq(stage)].copy()
        lookup = stage_nodes.set_index("node_id")[["x", "y"]].to_dict("index")
        stage_edges = edges.loc[edges["stage"].eq(stage)].copy()
        prior = (
            stage_edges.loc[
                stage_edges["edge_type"].eq("strictly_prior_cocitation")
            ]
            .sort_values("weight", ascending=False, kind="stable")
            .head(18)
        )
        other = stage_edges.loc[
            ~stage_edges["edge_type"].eq("strictly_prior_cocitation")
        ]
        stage_edges = pd.concat([prior, other], ignore_index=True)
        for row in stage_edges.itertuples(index=False):
            if row.source not in lookup or row.target not in lookup:
                continue
            source = lookup[row.source]
            target = lookup[row.target]
            if row.edge_type == "future_citation":
                color, alpha, width = ORANGE, 0.72, 0.75
            elif row.edge_type == "focal_reference":
                color, alpha, width = BLUE, 0.50, 0.65
            else:
                color, alpha = LIGHT_GRAY, 0.70
                width = 0.35 + 0.11 * math.log1p(float(row.weight))
            inset.plot(
                [source["x"], target["x"]],
                [source["y"], target["y"]],
                color=color,
                linewidth=width,
                alpha=alpha,
                zorder=1,
            )
        source_nodes = stage_nodes.loc[
            stage_nodes["node_type"].eq("reference_source")
        ]
        inset.scatter(
            source_nodes["x"],
            source_nodes["y"],
            s=22,
            facecolor=_fig2_blend(BLUE, 0.58),
            edgecolor=WHITE,
            linewidth=0.55,
            zorder=3,
        )
        focal = stage_nodes.loc[stage_nodes["node_type"].eq("focal_paper")]
        if not focal.empty:
            inset.scatter(
                focal["x"],
                focal["y"],
                s=105,
                marker="*",
                facecolor=ORANGE,
                edgecolor=WHITE,
                linewidth=0.75,
                zorder=5,
            )
        citers = stage_nodes.loc[stage_nodes["node_type"].eq("future_citer")]
        if not citers.empty:
            inset.scatter(
                citers["x"],
                citers["y"],
                s=27,
                marker="s",
                facecolor=WHITE,
                edgecolor=INK,
                linewidth=0.65,
                zorder=4,
            )
        inset.set_xlim(-1.65, 1.65)
        inset.set_ylim(-1.65, 1.65)
        inset.set_title(
            f"{headline}\n{caption}",
            fontsize=6.6,
            color=INK,
            pad=2,
            fontweight="bold" if stage == "G0" else "normal",
        )
        ax.add_patch(
            FancyBboxPatch(
                (lefts[stage_index] + 0.012, 0.125),
                0.246,
                0.070,
                boxstyle="round,pad=0.005,rounding_size=0.010",
                transform=ax.transAxes,
                facecolor=(
                    _fig2_blend(BLUE, 0.90)
                    if stage == "G0"
                    else PALE_GRAY
                ),
                edgecolor=(
                    LIGHT_BLUE if stage == "G0" else LIGHT_GRAY
                ),
                linewidth=0.65,
            )
        )
        stage_note = {
            "G−": "History is frozen before publication",
            "G0": "Compute five angles · eight signals",
            "G+5": "Observe D5 reach and evenness",
        }[stage]
        ax.text(
            lefts[stage_index] + 0.135,
            0.160,
            stage_note,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.9,
            color=BLUE if stage == "G0" else GRAY,
            fontweight="bold" if stage == "G0" else "normal",
        )
    for start, end in ((0.318, 0.355), (0.638, 0.675)):
        ax.add_patch(
            FancyArrowPatch(
                (start, 0.505),
                (end, 0.505),
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color=MID_GRAY,
            )
        )
    ax.text(
        0.045,
        0.050,
        textwrap.shorten(str(manifest["title"]), width=82)
        + f" · {int(manifest['publication_year'])} · "
        f"{int(manifest['valid_reference_count'])} valid references",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color=GRAY,
    )
    ax.text(
        0.955,
        0.050,
        "stable-hash illustration · no outcome used to select the paper",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.6,
        color=VERMILLION,
    )
    return child_axes


def _draw_fig2_route_b(
    ax: plt.Axes,
    bundle: FigureBundle,
) -> Sequence[plt.Axes]:
    """Draw the registered screen and the five-angle/eight-indicator basis."""
    _fig2_panel_frame(
        ax,
        "b",
        "Evidence governance yields five mechanisms and eight primary signals",
        "Counts, sources, directions and gate status come directly from the frozen v6.1 registry.",
    )
    stages = bundle.tables["fig2_selection_stages"].sort_values("stage_order")
    basis = bundle.tables["fig2_indicator_basis"].sort_values("display_order")
    angles = bundle.tables["observation_angles"].set_index("angle_id")
    centres = np.linspace(0.105, 0.895, len(stages))
    for index, (x_value, row) in enumerate(
        zip(centres, stages.itertuples(index=False))
    ):
        final = index == len(stages) - 1
        ax.add_patch(
            FancyBboxPatch(
                (x_value - 0.071, 0.725),
                0.142,
                0.126,
                boxstyle="round,pad=0.006,rounding_size=0.014",
                transform=ax.transAxes,
                facecolor=_fig2_blend(BLUE, 0.80 if final else 0.92),
                edgecolor=BLUE if final else LIGHT_BLUE,
                linewidth=1.0 if final else 0.7,
            )
        )
        ax.text(
            x_value,
            0.811,
            f"{int(row.count)}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=BLUE,
        )
        ax.text(
            x_value,
            0.754,
            textwrap.fill(str(row.stage), 16),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.7,
            color=INK,
            fontweight="bold",
            linespacing=0.95,
        )
        ax.text(
            x_value,
            0.700,
            textwrap.fill(str(row.criterion), 20),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=4.7,
            color=GRAY,
            linespacing=0.95,
        )
        if index < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x_value + 0.074, 0.787),
                    (centres[index + 1] - 0.074, 0.787),
                    transform=ax.transAxes,
                    arrowstyle="-|>",
                    mutation_scale=9,
                    linewidth=0.8,
                    color=MID_GRAY,
                )
            )
            removed = int(stages.iloc[index + 1]["removed_since_previous"])
            ax.text(
                (x_value + centres[index + 1]) / 2,
                0.812,
                f"−{removed}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=4.8,
                color=VERMILLION,
            )
    ax.text(
        0.035,
        0.630,
        "Peer-reviewed formula + paper use  ·  publication-time only  ·  "
        "local frozen data  ·  coverage/stability/fidelity  ·  OOF-blind family representative",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.6,
        color=GRAY,
    )

    angle_layout = {
        "A1_COMBINATION_RARITY": (0.035, 0.448, 0.445, 0.135),
        "A2_ATYPICALITY_CONVENTIONALITY": (0.035, 0.285, 0.445, 0.135),
        "A3_FIRST_TIME_COMBINATION": (0.035, 0.122, 0.445, 0.135),
        "A4_KNOWLEDGE_BREADTH_BALANCE": (0.515, 0.330, 0.450, 0.253),
        "A5_COGNITIVE_DISTANCE_INTEGRATION": (0.515, 0.122, 0.450, 0.178),
    }
    for angle_id in ANGLE_ORDER:
        x_value, y_value, width, height = angle_layout[angle_id]
        color = ANGLE_COLORS[angle_id]
        source_values = angles.loc[angle_id, "source_ids"]
        source_count = (
            len(source_values)
            if isinstance(source_values, (list, tuple, set))
            else len(str(source_values).split("|"))
        )
        ax.add_patch(
            FancyBboxPatch(
                (x_value, y_value),
                width,
                height,
                boxstyle="round,pad=0.007,rounding_size=0.014",
                transform=ax.transAxes,
                facecolor=_fig2_blend(color, 0.93),
                edgecolor=_fig2_blend(color, 0.35),
                linewidth=0.9,
            )
        )
        ax.text(
            x_value + 0.014,
            y_value + height - 0.026,
            ANGLE_LABELS[angle_id],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.3,
            color=color,
            fontweight="bold",
        )
        ax.text(
            x_value + width - 0.014,
            y_value + height - 0.026,
            f"{source_count} classification sources",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=4.7,
            color=GRAY,
        )
        members = basis.loc[basis["angle_id"].eq(angle_id)]
        available = height - 0.052
        row_height = available / max(len(members), 1)
        for member_index, row in enumerate(members.itertuples(index=False)):
            y_row = y_value + height - 0.053 - (member_index + 0.5) * row_height
            direction = "↓" if int(row.direction) == -1 else "↑"
            gate = "✓" if bool(row.all_primary_gates_pass) else "!"
            ax.text(
                x_value + 0.016,
                y_row,
                f"{int(row.display_order)}  {row.feature_label} {direction}",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=5.3,
                color=INK,
                fontweight="bold",
            )
            ax.text(
                x_value + width - 0.016,
                y_row,
                f"{row.evidence_badge} · {gate}",
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=4.8,
                color=color,
            )
    ax.text(
        0.965,
        0.042,
        "F/P/V = formula / paper-level application / validation sources · ✓ = all primary runtime gates passed",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=GRAY,
    )
    return [ax]


def _draw_fig2_route_c(
    ax: plt.Axes,
    bundle: FigureBundle,
) -> Sequence[plt.Axes]:
    """Draw sparse signal relations and prospective D5 mechanism signatures."""
    _fig2_panel_frame(
        ax,
        "c",
        "The mechanisms are related, but their prospective graph signatures differ",
        "Direction is frozen a priori; sparse relations describe complementarity, not outcome-based feature selection.",
    )
    relation_ax = ax.inset_axes([0.035, 0.125, 0.435, 0.740])
    matrix_ax = ax.inset_axes([0.525, 0.165, 0.440, 0.655])
    relation_ax.set_axis_off()
    nodes = bundle.tables["fig2_relation_nodes"].sort_values("display_order")
    edges = bundle.tables["fig2_relation_edges"]
    x_values = dict(zip(ANGLE_ORDER, np.linspace(0.09, 0.91, 5)))
    positions: Dict[str, tuple[float, float]] = {}
    for angle_id in ANGLE_ORDER:
        members = nodes.loc[nodes["angle_id"].eq(angle_id)]
        y_options = {
            1: [0.50],
            2: [0.64, 0.36],
            3: [0.73, 0.50, 0.27],
        }.get(len(members), np.linspace(0.75, 0.25, max(len(members), 1)))
        for y_value, row in zip(y_options, members.itertuples(index=False)):
            positions[str(row.code_name)] = (x_values[angle_id], float(y_value))
    for row in edges.sort_values(
        ["absolute_spearman", "source", "target"],
        ascending=[True, True, True],
    ).itertuples(index=False):
        source = positions[str(row.source)]
        target = positions[str(row.target)]
        positive = float(row.oriented_spearman) >= 0
        same_angle = row.source_angle_id == row.target_angle_id
        curvature = 0.28 if same_angle else 0.12
        if source[1] > target[1]:
            curvature *= -1
        relation_ax.add_patch(
            FancyArrowPatch(
                source,
                target,
                transform=relation_ax.transAxes,
                arrowstyle="-",
                connectionstyle=f"arc3,rad={curvature}",
                linewidth=0.75
                + 2.0 * max(float(row.absolute_spearman) - 0.40, 0),
                linestyle="-" if positive else "--",
                color=BLUE if positive else ORANGE,
                alpha=0.68,
                zorder=1,
            )
        )
        midpoint = (
            (source[0] + target[0]) / 2,
            (source[1] + target[1]) / 2 + (0.035 if positive else -0.035),
        )
        relation_ax.text(
            midpoint[0],
            midpoint[1],
            f"{float(row.oriented_spearman):+.2f}",
            transform=relation_ax.transAxes,
            ha="center",
            va="center",
            fontsize=4.5,
            color=BLUE if positive else ORANGE,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": WHITE,
                "edgecolor": "none",
                "alpha": 0.88,
            },
            zorder=2,
        )
    for angle_id in ANGLE_ORDER:
        x_value = x_values[angle_id]
        color = ANGLE_COLORS[angle_id]
        short_angle = {
            "A1_COMBINATION_RARITY": "Combination\nrarity",
            "A2_ATYPICALITY_CONVENTIONALITY": (
                "Atypicality &\nconventionality"
            ),
            "A3_FIRST_TIME_COMBINATION": "First-time\ncombinations",
            "A4_KNOWLEDGE_BREADTH_BALANCE": "Breadth &\nbalance",
            "A5_COGNITIVE_DISTANCE_INTEGRATION": (
                "Distance &\nintegration"
            ),
        }[angle_id]
        relation_ax.text(
            x_value,
            0.94,
            short_angle,
            transform=relation_ax.transAxes,
            ha="center",
            va="top",
            fontsize=5.2,
            color=color,
            fontweight="bold",
        )
    for row in nodes.itertuples(index=False):
        x_value, y_value = positions[str(row.code_name)]
        color = ANGLE_COLORS[str(row.angle_id)]
        relation_ax.add_patch(
            FancyBboxPatch(
                (x_value - 0.076, y_value - 0.052),
                0.152,
                0.104,
                boxstyle="round,pad=0.004,rounding_size=0.014",
                transform=relation_ax.transAxes,
                facecolor=WHITE,
                edgecolor=color,
                linewidth=0.9,
                zorder=3,
            )
        )
        relation_ax.text(
            x_value - 0.060,
            y_value,
            str(int(row.display_order)),
            transform=relation_ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.5,
            color=color,
            fontweight="bold",
            zorder=4,
        )
        relation_ax.text(
            x_value + 0.010,
            y_value,
            _fig2_short_feature(str(row.feature_label)),
            transform=relation_ax.transAxes,
            ha="center",
            va="center",
            fontsize=4.6,
            color=INK,
            linespacing=0.92,
            zorder=4,
        )
    threshold = float(edges["threshold"].iloc[0]) if not edges.empty else 0.40
    relation_ax.text(
        0.00,
        0.01,
        f"Edges shown only when |oriented Spearman| ≥ {threshold:.2f}  ·  "
        "solid = positive, dashed = negative",
        transform=relation_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.8,
        color=GRAY,
    )
    relation_ax.set_title(
        "Signal relation network",
        loc="left",
        fontsize=7.0,
        color=INK,
        pad=2,
        fontweight="bold",
    )

    future = bundle.tables["fig2_oriented_future_correlations"].copy()
    component_labels = (
        future.sort_values("component_order")
        ["future_component_label"]
        .drop_duplicates()
        .tolist()
    )
    feature_codes = (
        future.sort_values("feature_order")["code_name"].drop_duplicates().tolist()
    )
    feature_order = {feature: index for index, feature in enumerate(feature_codes)}
    component_order = {
        component: index for index, component in enumerate(component_labels)
    }
    for row in future.itertuples(index=False):
        x_value = component_order[str(row.future_component_label)]
        y_value = len(feature_codes) - 1 - feature_order[str(row.code_name)]
        value = float(row.oriented_spearman)
        color = BLUE if value >= 0 else ORANGE
        filled = bool(row.ci_excludes_zero)
        matrix_ax.scatter(
            [x_value],
            [y_value],
            s=24 + 210 * abs(value),
            facecolor=color if filled else WHITE,
            edgecolor=color,
            linewidth=0.9,
            alpha=0.88,
            zorder=3,
        )
        if abs(value) >= 0.30:
            matrix_ax.text(
                x_value,
                y_value,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=4.2,
                color=WHITE if filled else color,
                fontweight="bold",
                zorder=4,
            )
    matrix_ax.set_xticks(
        range(len(component_labels)),
        [textwrap.fill(value, 11) for value in component_labels],
        fontsize=5.2,
    )
    matrix_ax.set_yticks(
        range(len(feature_codes)),
        [str(index) for index in range(len(feature_codes), 0, -1)],
        fontsize=5.4,
    )
    matrix_ax.set_xlim(-0.55, len(component_labels) - 0.45)
    matrix_ax.set_ylim(-0.55, len(feature_codes) - 0.45)
    matrix_ax.grid(color=PALE_GRAY, linewidth=0.65)
    matrix_ax.set_axisbelow(True)
    matrix_ax.spines[:].set_visible(False)
    matrix_ax.tick_params(length=0)
    matrix_ax.set_title(
        "Publication-time signals versus five-year graph outcomes",
        loc="left",
        fontsize=7.0,
        color=INK,
        pad=6,
        fontweight="bold",
    )
    matrix_ax.text(
        0.0,
        -0.13,
        "Dot area = |ρ| · filled = 95% cluster-bootstrap interval excludes 0 · "
        f"n up to {int(future['n'].max()):,}\n"
        "D5 validates interpretation only; it never changes metric inclusion.",
        transform=matrix_ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.7,
        color=GRAY,
        linespacing=1.25,
    )
    return [ax, relation_ax, matrix_ax]


def _draw_fig2_route_d(
    ax: plt.Axes,
    bundle: FigureBundle,
) -> Sequence[plt.Axes]:
    """Draw matched-control percentile profiles with paired effects."""
    _fig2_panel_frame(
        ax,
        "d",
        "High-D5 papers exhibit stronger publication-time signal profiles",
        "Every row uses the same field-year percentile axis and matched domain/year/reference-volume controls.",
    )
    profile_ax = ax.inset_axes([0.205, 0.145, 0.615, 0.705])
    effect_ax = ax.inset_axes([0.835, 0.145, 0.145, 0.705])
    sample = bundle.tables["fig2_known_group_profile_sample"].copy()
    summary = bundle.tables["fig2_known_group_profile_summary"].copy()
    effects = bundle.tables["fig2_known_group_oriented_effects"].sort_values(
        "display_order"
    )
    feature_codes = effects["code_name"].tolist()
    y_lookup = {
        feature: len(feature_codes) - 1 - index
        for index, feature in enumerate(feature_codes)
    }
    rng = np.random.default_rng(20260725)
    for row in effects.itertuples(index=False):
        feature = str(row.code_name)
        y_value = y_lookup[feature]
        color = ANGLE_COLORS[str(row.angle_id)]
        for group, offset, point_color, marker in (
            ("Matched control", -0.105, MID_GRAY, "o"),
            ("High future diffusion", 0.105, color, "o"),
        ):
            points = sample.loc[
                sample["code_name"].eq(feature)
                & sample["group"].eq(group),
                "oriented_percentile",
            ].to_numpy(float)
            jitter = rng.uniform(-0.055, 0.055, len(points))
            profile_ax.scatter(
                points * 100,
                y_value + offset + jitter,
                s=4.0,
                marker=marker,
                facecolor=point_color,
                edgecolor="none",
                alpha=0.18 if group == "Matched control" else 0.22,
                rasterized=True,
                zorder=1,
            )
            stat = summary.loc[
                summary["code_name"].eq(feature)
                & summary["group"].eq(group)
            ]
            if stat.empty:
                continue
            stat_row = stat.iloc[0]
            profile_ax.plot(
                [float(stat_row["q25"]) * 100, float(stat_row["q75"]) * 100],
                [y_value + offset, y_value + offset],
                color=point_color,
                linewidth=2.8,
                solid_capstyle="round",
                zorder=3,
            )
            profile_ax.scatter(
                [float(stat_row["median"]) * 100],
                [y_value + offset],
                s=27,
                facecolor=WHITE if group == "Matched control" else point_color,
                edgecolor=point_color if group == "Matched control" else WHITE,
                linewidth=0.7,
                zorder=4,
            )
    profile_ax.axvline(50, color=LIGHT_GRAY, linewidth=0.9, zorder=0)
    profile_ax.set_xlim(0, 100)
    profile_ax.set_ylim(-0.65, len(feature_codes) - 0.35)
    profile_ax.set_xticks([0, 25, 50, 75, 100])
    profile_ax.set_yticks(
        [y_lookup[feature] for feature in feature_codes],
        [
            f"{int(row.display_order)}  {_fig2_short_feature(str(row.feature_label)).replace(chr(10), ' ')}"
            for row in effects.itertuples(index=False)
        ],
        fontsize=5.5,
    )
    for tick, row in zip(
        profile_ax.get_yticklabels(),
        effects.itertuples(index=False),
    ):
        tick.set_color(ANGLE_COLORS[str(row.angle_id)])
        tick.set_fontweight("bold")
    profile_ax.set_xlabel(
        "Field-year percentile in the a-priori innovation-oriented direction"
    )
    profile_ax.grid(axis="x", color=PALE_GRAY, linewidth=0.65)
    profile_ax.set_axisbelow(True)
    profile_ax.spines[["top", "right", "left"]].set_visible(False)
    profile_ax.tick_params(axis="y", length=0)
    profile_ax.scatter(
        [],
        [],
        s=20,
        color=MID_GRAY,
        alpha=0.45,
        label="Matched control",
    )
    profile_ax.scatter(
        [],
        [],
        s=20,
        color=BLUE,
        alpha=0.65,
        label="High D5 diffusion",
    )
    profile_ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.08),
        ncol=2,
        fontsize=5.5,
        handletextpad=0.4,
        columnspacing=1.1,
    )

    effect_ax.set_axis_off()
    effect_ax.text(
        0.02,
        1.02,
        "Paired Δ (95% CI)",
        transform=effect_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        color=INK,
        fontweight="bold",
    )
    for row in effects.itertuples(index=False):
        y_value = (y_lookup[str(row.code_name)] + 0.5) / len(feature_codes)
        effect_ax.text(
            0.02,
            y_value,
            (
                f"{float(row.oriented_difference) * 100:+.1f} pp\n"
                f"[{float(row.oriented_ci_low) * 100:+.1f}, "
                f"{float(row.oriented_ci_high) * 100:+.1f}]"
            ),
            transform=effect_ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.1,
            color=ANGLE_COLORS[str(row.angle_id)],
            fontweight="bold",
            linespacing=1.05,
        )
    ax.text(
        0.030,
        0.055,
        "Known-group plausibility ≠ complete innovation truth. "
        "The fixed direction reverses median conventionality only.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.1,
        color=VERMILLION,
    )
    return [ax, profile_ax, effect_ax]


def _clear_fig2_obsolete_artifacts(
    figure_dir: Path,
    bundle: FigureBundle,
) -> None:
    """Remove only obsolete, reproducible Fig.2 new renderer artifacts."""
    for stem in ("fig02_full", "fig02_legacy_route_extension"):
        for extension in ("png", "svg", "pdf"):
            (figure_dir / f"{stem}.{extension}").unlink(missing_ok=True)
    panel_dir = figure_dir / "panels"
    if panel_dir.exists():
        for path in panel_dir.glob("fig02_*"):
            if path.is_file():
                path.unlink()
    data_dir = figure_dir / "panel_data"
    if data_dir.exists():
        allowed = set(bundle.tables)
        for path in data_dir.iterdir():
            if (
                path.is_file()
                and path.suffix in {".csv", ".parquet"}
                and path.stem not in allowed
            ):
                path.unlink()


def _render_fig2_current_route(
    bundle: FigureBundle,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render one coherent, vector-native four-panel Fig.2."""
    configure_style()
    _clear_fig2_obsolete_artifacts(figure_dir, bundle)
    fig = plt.figure(figsize=(19.2, 13.0))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.024,
        right=0.988,
        bottom=0.035,
        top=0.905,
        wspace=0.090,
        hspace=0.135,
    )
    axes = {
        "a": fig.add_subplot(grid[0, 0]),
        "b": fig.add_subplot(grid[0, 1]),
        "c": fig.add_subplot(grid[1, 0]),
        "d": fig.add_subplot(grid[1, 1]),
    }
    drawers = {
        "a": _draw_fig2_route_a,
        "b": _draw_fig2_route_b,
        "c": _draw_fig2_route_c,
        "d": _draw_fig2_route_d,
    }
    groups: Dict[str, Sequence[plt.Axes]] = {}
    for panel, panel_ax in axes.items():
        groups[panel] = drawers[panel](panel_ax, bundle)
    figure_title(
        fig,
        "Fig. 2 | Publication-time reference signals organize observable graph change",
        "50 literature candidates → five source-backed mechanisms → eight frozen indicators; "
        "prospective D5 evidence validates interpretation but never selects features.",
    )
    outputs = {
        f"figure_{key}": value
        for key, value in export_figure(
            fig,
            figure_dir / "figure_full",
            formats=formats,
            dpi=dpi,
        ).items()
    }
    outputs.update(
        _export_axis_groups(
            fig,
            groups,
            figure_dir,
            prefix="fig02",
            formats=formats,
            dpi=dpi,
        )
    )
    plt.close(fig)
    return outputs


def _draw_target_flow(ax: plt.Axes, bundle: FigureBundle) -> None:
    data = bundle.tables["d5_target_construction"].sort_values("step")
    counts = bundle.tables["d5_target_counts"].iloc[0]
    ax.set_axis_off()
    panel_title(ax, "f", "D5 target construction")
    x_positions = np.linspace(0.08, 0.91, len(data))
    for index, (x_value, row) in enumerate(
        zip(x_positions, data.itertuples(index=False))
    ):
        ax.add_patch(
            plt.Rectangle(
                (x_value - 0.095, 0.38),
                0.19,
                0.30,
                transform=ax.transAxes,
                facecolor=WHITE,
                edgecolor=BLUE if index < 3 else ORANGE,
                linewidth=1.1,
            )
        )
        ax.text(
            x_value,
            0.61,
            row.component,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            color=INK,
            fontweight="bold",
        )
        ax.text(
            x_value,
            0.48,
            textwrap.fill(str(row.definition), 25),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.6,
            color=GRAY,
        )
        if index < len(data) - 1:
            ax.annotate(
                "",
                xy=(x_positions[index + 1] - 0.10, 0.53),
                xytext=(x_value + 0.10, 0.53),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "color": LIGHT_GRAY},
            )
    ax.text(
        0.02,
        0.15,
        f"Total D5 rows {int(counts['total_rows']):,} · "
        f"valid {int(counts['target_valid']):,} · "
        f"positive uptake {int(counts['positive_uptake']):,} · "
        f"zero uptake {int(counts['zero_uptake']):,}",
        transform=ax.transAxes,
        fontsize=6.4,
        color=INK,
    )
    ax.text(
        0.02,
        0.06,
        "All component percentile references are fitted within the training fold.",
        transform=ax.transAxes,
        fontsize=6,
        color=VERMILLION,
    )


def _draw_angle_folds(ax: plt.Axes, bundle: FigureBundle) -> None:
    data = bundle.tables["angle_fold_stability"].copy()
    panel_title(ax, "g", "Six-fold angle add/delete stability")
    offsets = {"add to K1": -0.08, "delete from full": 0.08}
    markers = {"add to K1": "o", "delete from full": "s"}
    for diagnostic, group in data.groupby("diagnostic"):
        for angle_number, angle_group in group.groupby("angle_number"):
            x = angle_number + offsets[diagnostic]
            ax.scatter(
                np.full(len(angle_group), x),
                angle_group["spearman_expected"],
                s=16,
                marker=markers[diagnostic],
                facecolor=(
                    ANGLE_COLORS[str(angle_group.iloc[0]["angle_id"])]
                    if diagnostic == "add to K1"
                    else WHITE
                ),
                edgecolor=ANGLE_COLORS[str(angle_group.iloc[0]["angle_id"])],
                linewidth=0.8,
                alpha=0.78,
                label=diagnostic if angle_number == 1 else None,
            )
            ax.plot(
                [x, x],
                [
                    angle_group["spearman_expected"].min(),
                    angle_group["spearman_expected"].max(),
                ],
                color=LIGHT_GRAY,
                linewidth=0.6,
                zorder=0,
            )
    ax.set_xticks(
        range(1, 6),
        [ANGLE_SHORT[angle] for angle in ANGLE_COLORS],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("Fold-specific D5 Spearman")
    ax.legend(frameon=False, fontsize=6.2, ncol=2)
    clean_axes(ax, grid_axis="y")
    ax.text(
        0.99,
        0.02,
        "Post-hoc interpretation only; folds are never used to reselect indicators.",
        transform=ax.transAxes,
        ha="right",
        fontsize=5.7,
        color=VERMILLION,
    )


def _render_fig3_extension(
    bundle: FigureBundle,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    configure_style()
    fig = plt.figure(figsize=(15.5, 6.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8], wspace=0.28)
    target_axis = fig.add_subplot(grid[0, 0])
    folds_axis = fig.add_subplot(grid[0, 1])
    _draw_target_flow(target_axis, bundle)
    _draw_angle_folds(folds_axis, bundle)
    figure_title(
        fig,
        "Fig. 3f–g | Target and temporal-stability extensions",
        "The target construction precedes model evaluation; angle diagnostics show every registered temporal fold.",
    )
    outputs = export_figure(
        fig,
        figure_dir / "fig03_legacy_route_extension",
        formats=formats,
        dpi=dpi,
    )
    outputs.update(
        _export_axis_groups(
            fig,
            {"f": [target_axis], "g": [folds_axis]},
            figure_dir,
            prefix="fig03",
            formats=formats,
            dpi=dpi,
        )
    )
    plt.close(fig)
    return {f"extension_{key}": value for key, value in outputs.items()}


def _render_fig4_current(
    bundle: FigureBundle,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    configure_style()
    data = bundle.tables["validation_sample_coverage"]
    fig = plt.figure(figsize=(15.5, 10.0))
    grid = fig.add_gridspec(2, 6, hspace=0.34, wspace=0.36)
    axes = {
        "a": fig.add_subplot(grid[0, :3]),
        "b": fig.add_subplot(grid[0, 3:]),
        "c": fig.add_subplot(grid[1, :2]),
        "d": fig.add_subplot(grid[1, 2:4]),
        "e": fig.add_subplot(grid[1, 4:]),
    }
    ax = axes["a"]
    panel_title(ax, "a", "Current v6.1 low / middle / high validation frame")
    for tier, color in (
        ("low", BLUE),
        ("middle", PURPLE),
        ("high", ORANGE),
    ):
        group = data.loc[data["global_fig3_tier"].eq(tier)]
        ax.scatter(
            group["validation_score"],
            np.full(len(group), {"low": 0, "middle": 1, "high": 2}[tier]),
            s=38,
            facecolor=WHITE,
            edgecolor=color,
            label=f"{tier.title()}, n={len(group)}",
        )
    ax.set_yticks([0, 1, 2], ["Low", "Middle", "High"])
    ax.set_xlabel("Current v6.1 D5 temporal-OOF expected diffusion score")
    ax.legend(frameon=False, fontsize=6.4, ncol=3)
    clean_axes(ax, grid_axis="x")
    messages = {
        "b": "Human ↔ ASPR label connections",
        "c": "Agreement estimates",
        "d": "Score by human rating",
        "e": "Quote-grounded cases",
    }
    for panel, title in messages.items():
        draft_panel(
            axes[panel],
            f"{panel}  {title}",
            "0/90 blinded judgements completed.\n"
            "The current frozen score corpus also lacks resolved manuscript text for these 30 cases.",
        )
    figure_title(
        fig,
        "Fig. 4 | Current-score blinded construct-validity frame",
        "The obsolete legacy-score pack has been replaced; inferential panels stay blocked until text and human labels are complete.",
        draft=True,
    )
    outputs = export_figure(
        fig,
        figure_dir / "figure_full",
        formats=formats,
        dpi=dpi,
    )
    panel_dir = figure_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for panel, axis in axes.items():
        extent = axis.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()
        ).expanded(1.08, 1.15)
        for extension in formats:
            path = panel_dir / f"fig04_{panel}.{extension}"
            kwargs: Dict[str, object] = {"bbox_inches": extent}
            if extension == "png":
                kwargs["dpi"] = dpi
            fig.savefig(path, **kwargs)
    plt.close(fig)
    return {f"full_{key}": value for key, value in outputs.items()}


def _render_fig6_extension(
    bundle: FigureBundle,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    configure_style()
    data = bundle.tables["reference_dose_stability"].copy()
    data = data.loc[data["reference_retention"].lt(1.0)]
    features = data["code_name"].drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.0), sharex=True, sharey=True)
    for axis, feature in zip(axes.ravel(), features):
        group = data.loc[data["code_name"].eq(feature)]
        summary = (
            group.groupby("reference_retention")["spearman"]
            .agg(["median", "min", "max"])
            .reset_index()
            .sort_values("reference_retention")
        )
        axis.vlines(
            summary["reference_retention"],
            summary["min"],
            summary["max"],
            color=LIGHT_BLUE,
            linewidth=1.1,
        )
        axis.scatter(
            summary["reference_retention"],
            summary["median"],
            s=24,
            color=BLUE,
            edgecolor=WHITE,
            linewidth=0.5,
        )
        axis.axhline(0.90, color=VERMILLION, linestyle="--", linewidth=0.7)
        axis.set_title(
            textwrap.fill(feature.replace("_", " "), 27),
            fontsize=6.2,
        )
        clean_axes(axis, grid_axis="y")
    for axis in axes[-1]:
        axis.set_xlabel("Reference retention")
    for axis in axes[:, 0]:
        axis.set_ylabel("Spearman vs full")
    figure_title(
        fig,
        "Fig. 6b extension | Exact reference-deletion dose response",
        "Each point is the median over deterministic recomputations; vertical strokes show the observed repetition range.",
    )
    outputs = export_figure(
        fig,
        figure_dir / "fig06_reference_dose_extension",
        formats=formats,
        dpi=dpi,
    )
    outputs.update(
        _export_axis_groups(
            fig,
            {"b0": list(axes.ravel())},
            figure_dir,
            prefix="fig06",
            formats=formats,
            dpi=dpi,
        )
    )
    plt.close(fig)
    return {f"extension_{key}": value for key, value in outputs.items()}


def _render_fig7_extension(
    bundle: FigureBundle,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    configure_style()
    associations = bundle.tables["venue_within_association"].copy()
    data = bundle.tables["venue_common_support_audit"].copy()
    measures = [
        ("year_mean", "Mean year"),
        ("reference_median", "Median references"),
        ("log_author_mean", "Mean log authors"),
        ("log_institution_mean", "Mean log institutions"),
        ("domain_count", "Domains represented"),
    ]
    fig = plt.figure(figsize=(16.0, 7.2))
    grid = fig.add_gridspec(2, len(measures), height_ratios=[0.9, 1.1])
    association_axis = fig.add_subplot(grid[0, :])
    families_association = associations["analysis_venue_family"].tolist()
    y_association = np.arange(len(families_association))[::-1]
    association_axis.errorbar(
        associations["spearman"],
        y_association,
        xerr=[
            associations["spearman"] - associations["ci_low"],
            associations["ci_high"] - associations["spearman"],
        ],
        fmt="o",
        color=BLUE,
        ecolor=LIGHT_BLUE,
        capsize=2.0,
        linewidth=1.1,
    )
    association_axis.axvline(0, color=GRAY, linewidth=0.8)
    association_axis.set_yticks(y_association, families_association)
    association_axis.set_xlabel(
        "Within-venue Spearman: innovation-only OOF score vs realized D5"
    )
    panel_title(
        association_axis,
        "f",
        "Within-venue predictive association",
    )
    association_axis.text(
        0.99,
        0.04,
        "Domain-year percentiles · cluster-bootstrap intervals",
        transform=association_axis.transAxes,
        ha="right",
        va="bottom",
        color=GRAY,
        fontsize=6.2,
    )
    clean_axes(association_axis, grid_axis="x")
    axes = [fig.add_subplot(grid[1, index]) for index in range(len(measures))]
    families = data["analysis_venue_family"].tolist()
    y = np.arange(len(families))[::-1]
    for index, (axis, (column, label)) in enumerate(zip(axes, measures)):
        axis.scatter(
            data[column],
            y,
            s=30,
            color=BLUE if index % 2 == 0 else ORANGE,
            edgecolor=WHITE,
        )
        axis.set_xlabel(label)
        axis.set_yticks(
            y,
            [textwrap.fill(value, 22) if index == 0 else "" for value in families],
        )
        clean_axes(axis, grid_axis="x")
    panel_title(axes[0], "g", "Common-support audit")
    figure_title(
        fig,
        "Fig. 7f–g | Within-venue validity and common-support audit",
        (
            "Venue-excluded associations and descriptive covariates expose "
            "support differences; neither identifies a venue causal effect."
        ),
    )
    outputs = export_figure(
        fig,
        figure_dir / "fig07_common_support_extension",
        formats=formats,
        dpi=dpi,
    )
    outputs.update(
        _export_axis_groups(
            fig,
            {"f": [association_axis], "g": axes},
            figure_dir,
            prefix="fig07",
            formats=formats,
            dpi=dpi,
        )
    )
    plt.close(fig)
    return {f"extension_{key}": value for key, value in outputs.items()}


def _compose_vertical(
    figure_id: int,
    figure_dir: Path,
    extension_path: Path,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    base_path = figure_dir / f"fig{figure_id:02d}_full.png"
    base = mpimg.imread(base_path)
    extension = mpimg.imread(extension_path)
    width = 16.5
    base_ratio = base.shape[0] / base.shape[1]
    extension_ratio = extension.shape[0] / extension.shape[1]
    fig = plt.figure(figsize=(width, width * (base_ratio + extension_ratio)))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[base_ratio, extension_ratio],
        hspace=0.015,
    )
    for axis, image in zip(
        [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])],
        [base, extension],
    ):
        axis.imshow(image)
        axis.set_axis_off()
    outputs = export_figure(
        fig,
        figure_dir / "figure_full",
        formats=formats,
        dpi=dpi,
    )
    plt.close(fig)
    return {f"full_{key}": value for key, value in outputs.items()}


def _copy_base_full(
    figure_id: int,
    figure_dir: Path,
    formats: Sequence[str],
) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    for extension in formats:
        source = figure_dir / f"fig{figure_id:02d}_full.{extension}"
        target = figure_dir / f"figure_full.{extension}"
        shutil.copy2(source, target)
        outputs[f"full_{extension}"] = target
    return outputs


def _render_fig10_blocked(
    bundle: FigureBundle,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render only the Fig.10 comparability gate, never mismatched deltas."""
    configure_style()
    fig, axes_array = plt.subplots(2, 3, figsize=(16.5, 10.5))
    axes = dict(zip(("a", "b", "c", "d", "e", "f"), axes_array.ravel()))

    inventory = bundle.tables["module_inventory"].sort_values("module_order")
    axis = axes["a"]
    axis.set_axis_off()
    panel_title(axis, "a", "Registered modules and switches")
    displayed = inventory.loc[
        ~inventory["ablation_switch"].eq("full ASPR")
    ].head(8)
    y_values = np.linspace(0.82, 0.14, len(displayed))
    for y_value, row in zip(y_values, displayed.itertuples(index=False)):
        axis.text(
            0.04,
            y_value,
            textwrap.shorten(str(row.module), width=43),
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=7.0,
            color=INK,
        )
        axis.text(
            0.96,
            y_value,
            textwrap.shorten(str(row.ablation_switch), width=26),
            transform=axis.transAxes,
            ha="right",
            va="center",
            fontsize=6.4,
            color=BLUE,
        )
        axis.plot(
            [0.50, 0.70],
            [y_value, y_value],
            transform=axis.transAxes,
            color=LIGHT_GRAY,
            linewidth=1.0,
        )

    audit = bundle.tables["ablation_comparability_audit"]
    axis = axes["b"]
    panel_title(axis, "b", "Existing reruns fail the same-path gate")
    y = np.arange(len(audit))[::-1]
    axis.scatter(
        np.zeros(len(audit)),
        y,
        marker="x",
        s=50,
        color=VERMILLION,
        linewidth=1.5,
    )
    axis.set_yticks(
        y,
        [textwrap.shorten(str(value), width=28) for value in audit["variant"]],
    )
    axis.set_xticks([0], ["different generation path"])
    axis.set_xlim(-0.45, 0.45)
    axis.text(
        0.98,
        0.03,
        "0 / 7 variants pass\none-switch comparability",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color=VERMILLION,
        fontsize=7.0,
        fontweight="bold",
    )
    clean_axes(axis, grid_axis="x")

    required = bundle.tables["required_same_path_variants"]
    axis = axes["c"]
    axis.set_axis_off()
    panel_title(axis, "c", "Required unified rerun contract")
    requirements = [
        "same model and checkpoint",
        "same prompt and decoding",
        "same retrieval cache and scorer",
        "50 identical cases per variant",
        "exactly one switch differs",
    ]
    for index, requirement in enumerate(requirements):
        y_value = 0.80 - index * 0.13
        axis.scatter(
            [0.08],
            [y_value],
            transform=axis.transAxes,
            marker="s",
            s=42,
            facecolor=WHITE,
            edgecolor=VERMILLION,
            linewidth=1.0,
        )
        axis.text(
            0.14,
            y_value,
            requirement,
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=7.2,
            color=INK,
        )
    axis.text(
        0.05,
        0.08,
        f"{len(required)} registered variants · none ready for main inference",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color=GRAY,
        fontsize=6.6,
    )

    preference = bundle.tables["preference_completion_audit"].iloc[0]
    draft_panel(
        axes["d"],
        "d  Blinded human preference",
        (
            f"{int(preference['observed_valid_judgements'])} / "
            f"{int(preference['required_judgements'])} required judgments "
            "completed. No ternary preference result is rendered."
        ),
    )
    draft_panel(
        axes["e"],
        "e  Measured quality–cost frontier",
        (
            "Blocked until quality and runtime are measured on the unified "
            "same-path rerun. Projected points are deliberately withheld."
        ),
    )
    draft_panel(
        axes["f"],
        "f  Representative degradation cases",
        (
            "Blocked until no-graph and no-verifier outputs satisfy the "
            "one-switch contract. No directional case story is inferred."
        ),
    )
    figure_title(
        fig,
        "Fig. 10 | ASPR module-ablation evidence gate",
        (
            "The requested same-path experiment is not yet available; this "
            "figure records what is missing without plotting incomparable "
            "numeric deltas."
        ),
    )
    fig.text(
        0.985,
        0.985,
        "BLOCKED — GENERATION PATHS NOT COMPARABLE",
        ha="right",
        va="top",
        color=VERMILLION,
        fontsize=9,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.94])
    outputs: Dict[str, Path] = {}
    for stem_name in ("fig10_full", "figure_full"):
        rendered = export_figure(
            fig,
            figure_dir / stem_name,
            formats=formats,
            dpi=dpi,
        )
        for extension, path in rendered.items():
            outputs[f"{stem_name}_{extension}"] = path
    panel_dir = figure_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for panel, axis in axes.items():
        extent = axis.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()
        ).expanded(1.08, 1.16)
        for extension in formats:
            path = panel_dir / f"fig10_{panel}.{extension}"
            kwargs: Dict[str, object] = {"bbox_inches": extent}
            if extension == "png":
                kwargs["dpi"] = dpi
            fig.savefig(path, **kwargs)
    plt.close(fig)
    return outputs


def render_new_figure(
    figure_id: int,
    bundle: FigureBundle,
    figure_dir: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render a figure and any route-restoring extension panels."""
    # Some inherited seaborn strip plots use NumPy's legacy global RNG for
    # jitter even when their row samples are fixed. Reset it at every render
    # so identical panel data produce pixel-identical PNGs.
    np.random.seed(20260725)
    if figure_id == 2:
        from experiments.common.new.adapters.fig2_renderer import (
            render_fig2_evidence_map,
        )

        return render_fig2_evidence_map(
            bundle,
            figure_dir,
            formats=formats,
            dpi=dpi,
        )
    if 3 <= figure_id <= 7:
        return render_fig3_to_fig7(
            figure_id,
            bundle,
            figure_dir,
            formats,
            dpi,
        )
    if figure_id == 10:
        return _render_fig10_blocked(bundle, figure_dir, formats, dpi)
    outputs = BASE_RENDERERS[figure_id](
        bundle,
        figure_dir,
        formats=formats,
        dpi=dpi,
    )
    if figure_id == 3:
        extra = _render_fig3_extension(bundle, figure_dir, formats, dpi)
        outputs.update(extra)
        outputs.update(
            _compose_vertical(
                figure_id,
                figure_dir,
                figure_dir / "fig03_legacy_route_extension.png",
                formats,
                dpi,
            )
        )
    elif figure_id == 6:
        extra = _render_fig6_extension(bundle, figure_dir, formats, dpi)
        outputs.update(extra)
        outputs.update(
            _compose_vertical(
                figure_id,
                figure_dir,
                figure_dir / "fig06_reference_dose_extension.png",
                formats,
                dpi,
            )
        )
    elif figure_id == 7:
        extra = _render_fig7_extension(bundle, figure_dir, formats, dpi)
        outputs.update(extra)
        outputs.update(
            _compose_vertical(
                figure_id,
                figure_dir,
                figure_dir / "fig07_common_support_extension.png",
                formats,
                dpi,
            )
        )
    else:
        outputs.update(_copy_base_full(figure_id, figure_dir, formats))
    return outputs
