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
from matplotlib.transforms import Bbox

from experiments.common.new.base.common import (
    ANGLE_COLORS,
    ANGLE_SHORT,
    BLUE,
    GRAY,
    INK,
    LIGHT_BLUE,
    LIGHT_GRAY,
    ORANGE,
    PALE_GRAY,
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
    render_fig2,
    render_fig3,
    render_fig5,
)
from experiments.common.new.base.renderers_6_10 import (
    render_fig6,
    render_fig7,
    render_fig9,
    render_fig10,
)


BASE_RENDERERS = {
    1: render_fig1,
    2: render_fig2,
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


def _draw_measurement_scene(ax: plt.Axes, bundle: FigureBundle) -> None:
    nodes = bundle.tables["measurement_scene_nodes"]
    edges = bundle.tables["measurement_scene_edges"]
    manifest = bundle.tables["measurement_scene_manifest"].iloc[0]
    ax.set_axis_off()
    panel_title(ax, "f", "Publication-time G− / G0 / G+5 measurement scene")
    stages = ["G−", "G0", "G+5"]
    stage_labels = {
        "G−": "Strictly prior graph",
        "G0": "Focal paper enters",
        "G+5": "Future spread (validation only)",
    }
    for stage_index, stage in enumerate(stages):
        left = 0.01 + stage_index * 0.33
        inset = ax.inset_axes([left, 0.13, 0.30, 0.72])
        inset.set_axis_off()
        stage_nodes = nodes.loc[nodes["stage"].eq(stage)].copy()
        lookup = stage_nodes.set_index("node_id")[["x", "y"]].to_dict("index")
        stage_edges = edges.loc[edges["stage"].eq(stage)]
        for row in stage_edges.itertuples(index=False):
            if row.source not in lookup or row.target not in lookup:
                continue
            source = lookup[row.source]
            target = lookup[row.target]
            color = (
                ORANGE
                if row.edge_type == "future_citation"
                else BLUE
                if row.edge_type == "focal_reference"
                else LIGHT_GRAY
            )
            width = 0.35 + 0.12 * math.log1p(float(row.weight))
            inset.plot(
                [source["x"], target["x"]],
                [source["y"], target["y"]],
                color=color,
                linewidth=width,
                alpha=0.65,
                zorder=1,
            )
        styles = {
            "reference_source": (LIGHT_BLUE, 24, "o"),
            "focal_paper": (ORANGE, 80, "*"),
            "future_citer": (WHITE, 34, "s"),
        }
        for node_type, group in stage_nodes.groupby("node_type"):
            color, size, marker = styles[node_type]
            inset.scatter(
                group["x"],
                group["y"],
                s=size,
                marker=marker,
                facecolor=color,
                edgecolor=INK if node_type == "future_citer" else WHITE,
                linewidth=0.6,
                zorder=3,
            )
        inset.set_xlim(-1.65, 1.65)
        inset.set_ylim(-1.65, 1.65)
        inset.set_title(
            f"{stage}\n{stage_labels[stage]}",
            fontsize=6.4,
            color=INK,
            pad=2,
        )
    ax.text(
        0.01,
        0.02,
        textwrap.shorten(str(manifest["title"]), width=95)
        + f" · {int(manifest['publication_year'])} · "
        f"{int(manifest['valid_reference_count'])} valid references\n"
        "Selection used publication-time eligibility and stable hash only.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=GRAY,
    )


def _draw_future_correlations(ax: plt.Axes, bundle: FigureBundle) -> None:
    data = bundle.tables["future_component_correlations"].copy()
    ax.set_axis_off()
    panel_title(ax, "g", "Indicators versus five-year graph outcomes")
    components = data["future_component_label"].drop_duplicates().tolist()
    features = data["feature_label"].drop_duplicates().tolist()
    for index, component in enumerate(components):
        inset = ax.inset_axes(
            [0.01 + index * 0.164, 0.10, 0.15, 0.78]
        )
        group = data.loc[
            data["future_component_label"].eq(component)
        ].set_index("feature_label").reindex(features)
        y = np.arange(len(group))[::-1]
        inset.axvline(0, color=LIGHT_GRAY, linewidth=0.7)
        inset.hlines(
            y,
            group["ci_low"],
            group["ci_high"],
            color=LIGHT_BLUE,
            linewidth=1.0,
        )
        inset.scatter(
            group["spearman"],
            y,
            s=16,
            color=BLUE,
            edgecolor=WHITE,
            linewidth=0.4,
        )
        inset.set_title(component, fontsize=5.8, color=INK)
        inset.set_xlim(-0.28, 0.58)
        inset.set_ylim(-0.7, len(features) - 0.3)
        inset.set_yticks(
            y,
            [
                textwrap.fill(value, 17) if index == 0 else ""
                for value in features
            ],
            fontsize=4.7,
        )
        inset.tick_params(axis="x", labelsize=4.7)
        clean_axes(inset, grid_axis="x")
    ax.text(
        0.99,
        0.01,
        "Field-year percentiles; whiskers are fixed-rank domain-year cluster bootstrap intervals.\n"
        "Outcome association never changes indicator inclusion.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.7,
        color=VERMILLION,
    )


def _render_fig2_extension(
    bundle: FigureBundle,
    figure_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    configure_style()
    fig = plt.figure(figsize=(16.5, 7.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.08], wspace=0.30)
    scene_axis = fig.add_subplot(grid[0, 0])
    future_axis = fig.add_subplot(grid[0, 1])
    _draw_measurement_scene(scene_axis, bundle)
    _draw_future_correlations(future_axis, bundle)
    figure_title(
        fig,
        "Fig. 2f–g | Legacy-route measurement extensions",
        "The left panel makes the publication-time measurement boundary explicit; the right panel checks prospective graph outcomes.",
    )
    outputs = export_figure(
        fig,
        figure_dir / "fig02_legacy_route_extension",
        formats=formats,
        dpi=dpi,
    )
    outputs.update(
        _export_axis_groups(
            fig,
            {"f": [scene_axis], "g": [future_axis]},
            figure_dir,
            prefix="fig02",
            formats=formats,
            dpi=dpi,
        )
    )
    plt.close(fig)
    return {f"extension_{key}": value for key, value in outputs.items()}


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
    if figure_id == 4:
        return _render_fig4_current(bundle, figure_dir, formats, dpi)
    if figure_id == 10:
        return _render_fig10_blocked(bundle, figure_dir, formats, dpi)
    outputs = BASE_RENDERERS[figure_id](
        bundle,
        figure_dir,
        formats=formats,
        dpi=dpi,
    )
    if figure_id == 2:
        extra = _render_fig2_extension(bundle, figure_dir, formats, dpi)
        outputs.update(extra)
        outputs.update(
            _compose_vertical(
                figure_id,
                figure_dir,
                figure_dir / "fig02_legacy_route_extension.png",
                formats,
                dpi,
            )
        )
    elif figure_id == 3:
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
