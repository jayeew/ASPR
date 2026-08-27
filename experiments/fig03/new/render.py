"""Nature-style frozen-data renderer for the ASPR Fig.3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
from PIL import Image
from scipy.interpolate import RectBivariateSpline

from .analysis import read_json, sha256_file, write_json

INK = "#263746"
MUTED = "#6E7F8D"
GRID = "#DCE2E7"
FRAME = "#E7EBEE"
BLUE_DARK = "#245A83"
BLUE = "#4F8DB8"
BLUE_LIGHT = "#DCEAF3"
AMBER = "#C6903F"
ORANGE = "#C96B3B"
PALE = "#F6F8F9"
INSUFFICIENT = "#F7F8F8"
DEGENERATE = "#F2ECE4"
NOT_MATURE = "#E8ECEF"
WHITE = "#FFFFFF"

DOMAIN_SHORT = {
    "clinical_health": "Clinical",
    "chemistry": "Chemistry",
    "mathematics_statistics": "Math/stat",
    "computer_science_ai": "CS/AI",
    "engineering_energy": "Eng/energy",
    "earth_climate_environment": "Earth/climate",
    "astronomy_space": "Astronomy",
    "life_molecular": "Life/molecular",
    "physics": "Physics",
    "ecology_evolution_microbiology": "Eco/evol/micro",
    "materials_nanoscience": "Materials",
    "neuroscience": "Neuroscience",
}


def configure_matplotlib() -> None:
    """Apply the compact print visual system."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 6.5,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.45,
            "axes.titlecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def performance_cmap() -> LinearSegmentedColormap:
    """Return the blue-low to red-high sequential performance palette."""
    return LinearSegmentedColormap.from_list(
        "aspr_performance",
        ["#274D9B", "#3F86B8", "#70B4B2", "#E4D58A", "#E49A5A", "#C94F47"],
    )


def display_domains(config: Mapping[str, Any]) -> list[str]:
    """Return the display-only domain order, falling back to the canonical order."""
    rows = config.get("domain_display_order", config["domain_order"])
    return [str(row["id"]) for row in rows]


def gain_cmap() -> LinearSegmentedColormap:
    """Return blue-decrease, white-zero, orange-increase colors."""
    return LinearSegmentedColormap.from_list(
        "aspr_gain",
        ["#376F9A", "#B9D3E2", "#FAFAF8", "#E9BEA7", "#C76637"],
    )


def panel_label(axis: Axes, letter: str, title: str, note: str = "") -> None:
    """Place a compact panel label and title."""
    axis.set_axis_off()
    axis.text(0.0, 0.92, letter, fontsize=10.8, weight="bold", color=INK, va="top")
    axis.text(0.055, 0.92, title, fontsize=8.8, weight="bold", color=INK, va="top")
    if note:
        axis.text(0.055, 0.12, note, fontsize=6.8, color=MUTED, va="bottom")


def add_panel_frame(figure: Figure, specs: Sequence[mpl.gridspec.SubplotSpec]) -> None:
    """Draw one very light boundary around a logical panel."""
    bounds = Bbox.union([spec.get_position(figure) for spec in specs])
    pad_x = 0.004
    pad_y = 0.004
    figure.add_artist(
        Rectangle(
            (bounds.x0 - pad_x, bounds.y0 - pad_y),
            bounds.width + 2 * pad_x,
            bounds.height + 2 * pad_y,
            transform=figure.transFigure,
            facecolor="none",
            edgecolor=FRAME,
            linewidth=0.45,
            zorder=20,
            clip_on=False,
        )
    )


def draw_panel_a(axis: Axes, score_summary: pd.DataFrame) -> None:
    """Render the five-node ASPR construction flow."""
    axis.set_axis_off()
    axis.text(0.0, 0.98, "a", fontsize=10.8, weight="bold", color=INK, va="top")
    axis.text(
        0.065,
        0.98,
        "ASPR construction",
        fontsize=8.8,
        weight="bold",
        color=INK,
        va="top",
    )
    nodes = [
        "T0\nindicators",
        "two-part\nHGB",
        "raw\nscore",
        "D5\nECDF",
        "ASPR\n0–100",
    ]
    starts = np.linspace(0.012, 0.812, 5)
    width = 0.176
    for index, (start, label) in enumerate(zip(starts, nodes)):
        focal = index == 4
        box = FancyBboxPatch(
            (start, 0.565),
            width,
            0.235,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            transform=axis.transAxes,
            facecolor=BLUE_LIGHT if focal else WHITE,
            edgecolor=BLUE_DARK if focal else GRID,
            linewidth=0.9 if focal else 0.5,
        )
        axis.add_patch(box)
        axis.text(
            start + width / 2,
            0.6825,
            label,
            ha="center",
            va="center",
            fontsize=6.5,
            weight="bold" if focal else "normal",
            color=BLUE_DARK if focal else INK,
            linespacing=0.95,
        )
        if index < 4:
            axis.add_patch(
                FancyArrowPatch(
                    (start + width + 0.004, 0.6825),
                    (starts[index + 1] - 0.004, 0.6825),
                    transform=axis.transAxes,
                    arrowstyle="-|>",
                    mutation_scale=5.5,
                    color=MUTED,
                    linewidth=0.55,
                )
            )
    axis.add_patch(
        FancyBboxPatch(
            (0.012, 0.27),
            0.976,
            0.235,
            boxstyle="round,pad=0.006,rounding_size=0.01",
            transform=axis.transAxes,
            facecolor="#F2F6F8",
            edgecolor=GRID,
            linewidth=0.45,
        )
    )
    axis.text(
        0.5,
        0.3875,
        "raw = P(uptake) × E(diffusion | uptake)\nASPR = 100 × ECDFD5(raw)",
        ha="center",
        va="center",
        fontsize=6.5,
        color=INK,
        linespacing=1.25,
    )
    row = score_summary.iloc[0]
    axis.text(
        0.012,
        0.205,
        f"{int(row['scored_papers']):,} papers · mature D5 ≤ {int(row['mature_d5_year_max'])}",
        fontsize=6.6,
        weight="bold",
        color=INK,
    )
    axis.add_patch(
        Rectangle(
            (0.012, 0.035),
            0.976,
            0.105,
            transform=axis.transAxes,
            facecolor="#F8F6F1",
            edgecolor="none",
        )
    )
    axis.text(
        0.035,
        0.0875,
        "Predictive signal—not causality or novelty",
        fontsize=6.5,
        color=INK,
        va="center",
    )


def draw_panel_b_header(axis: Axes) -> None:
    """Render the aligned Panel b heading."""
    axis.set_axis_off()
    axis.text(
        0.0,
        0.92,
        "b",
        transform=axis.transAxes,
        fontsize=10.8,
        weight="bold",
        color=INK,
        va="top",
    )
    axis.text(
        0.045,
        0.92,
        "OOF enrichment across horizons and feature sets",
        transform=axis.transAxes,
        fontsize=8.8,
        weight="bold",
        color=INK,
        va="top",
    )
    axis.text(
        0.045,
        0.12,
        "Each cell: ten fold-local deciles with 95% year-block CI · D10 share / lift · ★ selected D5 set",
        transform=axis.transAxes,
        fontsize=6.8,
        color=MUTED,
        va="bottom",
    )


def draw_panel_b(
    axis: Axes,
    deciles: pd.DataFrame,
    *,
    horizon: int,
    model_id: str,
    color: str,
    show_y_axis: bool,
    show_x_axis: bool,
    official: bool,
) -> None:
    """Render one horizon–feature-set OOF enrichment curve."""
    deciles = deciles.loc[
        deciles["horizon"].eq(horizon) & deciles["model_id"].eq(model_id)
    ].sort_values("prediction_decile")
    x = deciles["prediction_decile"].to_numpy(dtype=int)
    y = deciles["observed_top_share"].to_numpy(dtype=float)
    low = y - deciles["ci_low"].to_numpy(dtype=float)
    high = deciles["ci_high"].to_numpy(dtype=float) - y
    axis.plot(x, y, color=color, linewidth=1.25, zorder=2)
    axis.errorbar(
        x,
        y,
        yerr=np.vstack([low, high]),
        fmt="none",
        ecolor=MUTED,
        elinewidth=0.65,
        capsize=1.8,
        zorder=1,
    )
    axis.scatter(
        x[:-1],
        y[:-1],
        s=18,
        facecolor=WHITE,
        edgecolor=color,
        linewidth=0.75,
        zorder=3,
    )
    axis.scatter(
        [10],
        [y[-1]],
        s=32,
        facecolor=color,
        edgecolor=WHITE,
        linewidth=0.7,
        zorder=4,
    )
    baseline = float(deciles["baseline_top_share"].iloc[0])
    axis.axhline(baseline, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.7)
    axis.annotate(
        f"{100 * y[-1]:.2f}%\n{deciles['enrichment_over_baseline'].iloc[-1]:.2f}×",
        xy=(10, y[-1]),
        xytext=(9.15, 0.300),
        ha="right",
        va="center",
        fontsize=6.5,
        weight="bold",
        color=color,
        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.65},
    )
    axis.set_xlim(0.7, 10.3)
    axis.set_ylim(0.0, 0.45)
    axis.set_xticks([1, 5, 10], ["D1", "D5", "D10"])
    axis.set_yticks([0.0, 0.2, 0.4])
    axis.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    axis.grid(axis="y", color=GRID, linewidth=0.4)
    axis.spines[["top", "right"]].set_visible(False)
    if not show_y_axis:
        axis.tick_params(axis="y", left=False, labelleft=False)
        axis.spines["left"].set_visible(False)
    if not show_x_axis:
        axis.tick_params(axis="x", bottom=False, labelbottom=False)
        axis.spines["bottom"].set_visible(False)
    if official:
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color(AMBER)
            spine.set_linewidth(1.0)
        axis.text(
            0.03,
            0.92,
            "★",
            transform=axis.transAxes,
            fontsize=7.0,
            color=AMBER,
            weight="bold",
            va="top",
        )


def draw_panel_b_board(
    figure: Figure,
    spec: mpl.gridspec.SubplotSpec,
    deciles: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    """Render all twelve horizon–feature-set curves as small multiples."""
    horizons = [3, 5, 8]
    models = [str(row["id"]) for row in config["model_sets"]]
    labels = [str(row["label"]) for row in config["model_sets"]]
    colors = ["#315D9B", "#4F8DB8", "#C67A3F", "#B94A48"]
    board = spec.subgridspec(3, 4, hspace=0.15, wspace=0.10)
    for row, horizon in enumerate(horizons):
        for column, (model_id, label, color) in enumerate(
            zip(models, labels, colors, strict=True)
        ):
            axis = figure.add_subplot(board[row, column])
            draw_panel_b(
                axis,
                deciles,
                horizon=horizon,
                model_id=model_id,
                color=color,
                show_y_axis=column == 0,
                show_x_axis=row == 2,
                official=horizon == 5 and model_id == "primary",
            )
            if row == 0:
                axis.set_title(
                    label,
                    fontsize=6.8,
                    weight="bold",
                    color=INK,
                    pad=3,
                )
            if column == 0:
                axis.set_ylabel(
                    f"D{horizon}",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=8,
                    fontsize=7.0,
                    weight="bold",
                    color=INK,
                )


def matrix_for_group(
    table: pd.DataFrame,
    domains: Sequence[str],
    years: Sequence[int],
    value: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build aligned numeric and status matrices."""
    index = pd.MultiIndex.from_product(
        [domains, years], names=["domain12", "window_end"]
    )
    aligned = table.set_index(["domain12", "window_end"]).reindex(index)
    values = aligned[value].to_numpy(dtype=float).reshape(len(domains), len(years))
    statuses = (
        aligned["status"]
        .fillna("insufficient")
        .to_numpy()
        .reshape(len(domains), len(years))
    )
    return values, statuses


def draw_status_background(axis: Axes, statuses: np.ndarray) -> None:
    """Render sparse and outcome-not-mature cells as quiet neutral blocks."""
    for row, column in zip(*np.where(statuses != "reliable")):
        status = str(statuses[row, column])
        if status == "not_mature":
            color = NOT_MATURE
        elif status == "degenerate":
            color = DEGENERATE
        else:
            color = INSUFFICIENT
        axis.add_patch(
            Rectangle(
                (column, row),
                1,
                1,
                facecolor=color,
                edgecolor=color,
                linewidth=0.0,
                zorder=0,
            )
        )


def style_heatmap(
    axis: Axes,
    years: Sequence[int],
    labels: Sequence[str],
    *,
    show_y: bool,
    show_x: bool,
    show_right_year: bool = False,
    sparse_y: bool = False,
) -> None:
    """Apply shared heatmap coordinates and print-safe labels."""
    axis.set_xlim(0, len(years))
    axis.set_ylim(len(labels), 0)
    requested_ticks = [1990, 2005, 2020] if show_right_year else [1990, 2005]
    ticks = [year for year in requested_ticks if year in years]
    positions = [years.index(year) + 0.5 for year in ticks]
    axis.set_xticks(
        positions if show_x else [], [str(year) for year in ticks] if show_x else []
    )
    y_indices = list(range(len(labels)))
    if sparse_y:
        y_indices = list(range(0, len(labels), 2))
    axis.set_yticks(
        [index + 0.5 for index in y_indices] if show_y else [],
        [labels[index] for index in y_indices] if show_y else [],
    )
    axis.tick_params(length=0, pad=1.3, labelsize=6.5)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(GRID)
        spine.set_linewidth(0.35)


def draw_performance_board(
    figure: Figure,
    spec: mpl.gridspec.SubplotSpec,
    table: pd.DataFrame,
    config: Mapping[str, Any],
) -> list[Axes]:
    """Render the exact 3-by-4 performance board without a side summary."""
    models = [str(row["id"]) for row in config["model_sets"]]
    model_labels = [str(row["label"]) for row in config["model_sets"]]
    horizons = [int(value) for value in config["horizons"]]
    domains = display_domains(config)
    domain_labels = [DOMAIN_SHORT[domain] for domain in domains]
    years = list(
        range(int(config["landscape_year_min"]), int(config["landscape_year_max"]) + 1)
    )
    grid = spec.subgridspec(
        4,
        5,
        width_ratios=[0.78, 1, 1, 1, 1],
        height_ratios=[0.22, 1, 1, 1],
        wspace=0.035,
        hspace=0.15,
    )
    norm = Normalize(
        float(config["performance_color_min"]),
        float(config["performance_color_max"]),
        clip=False,
    )
    corner_axis = figure.add_subplot(grid[0, 0])
    corner_axis.set_axis_off()
    corner_axis.text(
        0.0,
        1.45,
        "Forward prediction\nwindow",
        fontsize=6.5,
        weight="bold",
        color=MUTED,
        va="center",
        linespacing=0.95,
    )
    feature_axis = figure.add_subplot(grid[0, 1:5])
    feature_axis.set_axis_off()
    feature_axis.text(
        0.5,
        1.90,
        "Frozen feature set",
        fontsize=7.0,
        weight="bold",
        color=MUTED,
        ha="center",
        va="top",
    )
    key_axis = figure.add_subplot(grid[1:, 0])
    draw_domain_key(key_axis, domains)
    axes: list[Axes] = [corner_axis, feature_axis, key_axis]
    for row, horizon in enumerate(horizons):
        for column, model_id in enumerate(models):
            axis = figure.add_subplot(grid[row + 1, column + 1])
            group = table[table["horizon"].eq(horizon) & table["model_id"].eq(model_id)]
            values, statuses = matrix_for_group(group, domains, years, "spearman")
            draw_status_background(axis, statuses)
            masked = np.ma.masked_where(statuses != "reliable", values)
            axis.pcolormesh(
                masked, cmap=performance_cmap(), norm=norm, shading="flat", zorder=1
            )
            start_2014 = years.index(2014)
            axis.add_patch(
                Rectangle(
                    (start_2014, 0),
                    len(years) - start_2014,
                    len(domains),
                    facecolor="#7D8790",
                    edgecolor="none",
                    alpha=0.075,
                    zorder=2,
                )
            )
            style_heatmap(
                axis,
                years,
                domain_labels,
                show_y=False,
                show_x=row == 2,
                show_right_year=column == len(models) - 1,
            )
            if row == 0:
                axis.set_title(
                    model_labels[column], fontsize=7.2, pad=2.6, weight="bold"
                )
            if column == 0:
                axis.text(
                    0.018,
                    0.5,
                    f"D{horizon}",
                    transform=axis.transAxes,
                    fontsize=7.4,
                    weight="bold",
                    color=INK,
                    ha="left",
                    va="center",
                    bbox={
                        "facecolor": WHITE,
                        "edgecolor": "none",
                        "alpha": 0.78,
                        "pad": 0.3,
                    },
                )
            if horizon == 5 and model_id == "primary":
                for spine in axis.spines.values():
                    spine.set_color(AMBER)
                    spine.set_linewidth(1.2)
                axis.text(
                    0.015,
                    0.98,
                    "★",
                    transform=axis.transAxes,
                    fontsize=7.5,
                    color=AMBER,
                    va="top",
                    zorder=5,
                )
                axis.text(
                    0.985,
                    0.78,
                    "Selected D5 Primary16",
                    transform=axis.transAxes,
                    fontsize=6.5,
                    weight="bold",
                    color=INK,
                    ha="right",
                    va="top",
                    bbox={
                        "facecolor": WHITE,
                        "edgecolor": "none",
                        "alpha": 0.78,
                        "pad": 0.3,
                    },
                    zorder=5,
                )
            axes.append(axis)
    return axes


def draw_domain_key(axis: Axes, domains: Sequence[str]) -> None:
    """Show the one shared top-to-bottom domain order for all heatmaps."""
    axis.set_axis_off()
    axis.text(
        0.0, 1.0, "domain order", fontsize=6.8, weight="bold", color=INK, va="top"
    )
    positions = np.linspace(0.88, 0.035, len(domains))
    for index, (domain, position) in enumerate(zip(domains, positions), start=1):
        axis.text(
            0.0,
            position,
            f"{index:>2}  {DOMAIN_SHORT[domain]}",
            fontsize=6.5,
            color=MUTED,
            va="center",
        )


def draw_panel_c_title(axis: Axes) -> None:
    """Render the Panel c title in its own header row."""
    panel_label(
        axis,
        "c",
        "Year- and domain-resolved OOF performance",
        "Year-resolved forward test blocks; grey shading marks post-2014 publication years",
    )


def draw_panel_c_toolbar(axis: Axes, config: Mapping[str, Any]) -> None:
    """Render the neutral-state key and absolute color scale."""
    axis.set_axis_off()
    legend = [
        Patch(facecolor=INSUFFICIENT, edgecolor=GRID, label="structural n < 30"),
        Patch(
            facecolor=DEGENERATE, edgecolor=GRID, label="constant ranks; ρ undefined"
        ),
        Patch(facecolor=NOT_MATURE, edgecolor=GRID, label="outcome not yet mature"),
    ]
    axis.legend(
        handles=legend,
        loc="center left",
        bbox_to_anchor=(0.04, 0.48),
        frameon=False,
        ncol=3,
        fontsize=6.8,
        handlelength=1.0,
        columnspacing=1.1,
    )
    cax = axis.inset_axes((0.72, 0.30, 0.26, 0.26))
    scalar = mpl.cm.ScalarMappable(
        norm=Normalize(
            float(config["performance_color_min"]),
            float(config["performance_color_max"]),
            clip=False,
        ),
        cmap=performance_cmap(),
    )
    colorbar = axis.figure.colorbar(
        scalar, cax=cax, orientation="horizontal", extend="both", extendfrac=0.045
    )
    scale_min = float(config["performance_color_min"])
    scale_max = float(config["performance_color_max"])
    colorbar.set_ticks([scale_min, (scale_min + scale_max) / 2, scale_max])
    colorbar.ax.tick_params(labelsize=6.5, length=1.5, pad=1)
    colorbar.set_label("OOF Spearman ρ · focused scale", fontsize=6.8, labelpad=1)


def draw_ridgeline_terrain(
    axis: Axes3D, landscape: pd.DataFrame, config: Mapping[str, Any]
) -> None:
    """Render one continuous semi-transparent interpolated performance terrain."""
    domains = display_domains(config)
    selected = landscape[
        landscape["horizon"].eq(5)
        & landscape["model_id"].eq("primary")
        & landscape["status"].eq("reliable")
    ]
    z_min = float(config["ridgeline_z_min"])
    z_max = float(config["ridgeline_z_max"])
    years = np.arange(
        int(config["landscape_year_min"]),
        int(config["mature_year_max"]["5"]) + 1,
        dtype=float,
    )
    matrix = np.full((len(domains), len(years)), np.nan, dtype=float)
    for domain_index, domain in enumerate(domains):
        group = selected[selected["domain12"].eq(domain)].set_index("window_end")
        series = group["spearman"].reindex(years.astype(int))
        matrix[domain_index] = series.to_numpy(dtype=float)
    observed_limits = []
    for row in matrix:
        observed = np.flatnonzero(np.isfinite(row))
        if len(observed) < 2:
            raise ValueError("Panel d requires two reliable years per domain")
        observed_limits.append((int(observed[0]), int(observed[-1])))
    common_start = max(start for start, _ in observed_limits)
    common_end = min(end for _, end in observed_limits)
    if common_end - common_start < 3:
        raise ValueError("Panel d lacks a shared four-year observed range")
    years = years[common_start : common_end + 1]
    matrix = matrix[:, common_start : common_end + 1]
    matrix = np.vstack(
        [
            pd.Series(row, index=years)
            .interpolate(method="linear", limit_area="inside")
            .to_numpy(dtype=float)
            for row in matrix
        ]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("Panel d interpolation left an internal domain-year gap")
    dense_years = np.linspace(years[0], years[-1], 150)
    dense_domains = np.linspace(0, len(domains) - 1, 72)
    spline = RectBivariateSpline(
        np.arange(len(domains), dtype=float), years, matrix, kx=3, ky=3, s=0
    )
    surface_z = spline(dense_domains, dense_years)
    surface_z = np.clip(
        surface_z,
        max(z_min, float(np.nanmin(matrix))),
        min(z_max, float(np.nanmax(matrix))),
    )
    surface_x, surface_y = np.meshgrid(dense_years, dense_domains)
    norm = Normalize(
        float(config["performance_color_min"]),
        float(config["performance_color_max"]),
        clip=False,
    )
    axis.plot_surface(
        surface_x,
        surface_y,
        surface_z,
        cmap=performance_cmap(),
        norm=norm,
        linewidth=0,
        alpha=0.72,
        shade=False,
        antialiased=True,
        rstride=1,
        cstride=1,
    )
    axis.contour(
        surface_x,
        surface_y,
        surface_z,
        zdir="z",
        offset=z_min,
        levels=np.linspace(0.4, 0.85, 6),
        cmap=performance_cmap(),
        norm=norm,
        linewidths=0.45,
        alpha=0.35,
    )
    highlighted = (
        selected.groupby("domain12")["spearman"]
        .mean()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )
    highlight_points: list[tuple[str, float, int, float]] = []
    for domain in highlighted:
        domain_index = domains.index(str(domain))
        domain_curve = np.asarray(
            spline(float(domain_index), dense_years), dtype=float
        ).reshape(-1)
        peak_index = int(np.nanargmax(domain_curve))
        peak_year = float(dense_years[peak_index])
        peak_z = float(np.clip(domain_curve[peak_index], z_min, z_max))
        highlight_points.append((str(domain), peak_year, domain_index, peak_z))
        axis.scatter(
            [peak_year],
            [domain_index],
            [peak_z],
            s=28,
            facecolors=WHITE,
            edgecolors=INK,
            linewidths=1.0,
            depthshade=False,
            zorder=8,
        )
    axis.set_xlim(float(years[0]), float(years[-1]))
    axis.set_ylim(0, len(domains) - 1)
    axis.set_zlim(z_min, z_max)
    axis.set_xticks([1990, 2000, 2010, 2020])
    shown = [0, 3, 7, 11]
    axis.set_yticks(shown, [str(index + 1) for index in shown])
    axis.set_zticks([0.3, 0.5, 0.7, 0.9])
    axis.tick_params(labelsize=6.5, pad=0)
    axis.set_xlabel("year", labelpad=-1)
    axis.set_zlabel("ρ", labelpad=-2)
    axis.view_init(elev=28, azim=-61)
    axis.set_box_aspect((2.35, 1.42, 1.0), zoom=1.38)
    figure = axis.get_figure()
    panel_bounds = axis.get_subplotspec().get_position(figure)
    label_positions = [
        (
            panel_bounds.x0 + 0.015 * panel_bounds.width,
            panel_bounds.y0 + 0.77 * panel_bounds.height,
        ),
        (
            panel_bounds.x0 + 0.015 * panel_bounds.width,
            panel_bounds.y0 + 0.55 * panel_bounds.height,
        ),
        (
            panel_bounds.x0 + 0.985 * panel_bounds.width,
            panel_bounds.y0 + 0.67 * panel_bounds.height,
        ),
    ]
    label_alignments = ["left", "left", "right"]
    projected_points = []
    for domain, peak_year, domain_index, peak_z in highlight_points:
        projected_x, projected_y, _ = proj_transform(
            peak_year, float(domain_index), peak_z, axis.get_proj()
        )
        projected_points.append(
            (projected_x, projected_y, domain, peak_year, domain_index, peak_z)
        )
    projected_points.sort(key=lambda item: item[0])
    for projected_point, label_position, alignment in zip(
        projected_points, label_positions, label_alignments, strict=True
    ):
        projected_x, projected_y, domain, _, _, _ = projected_point
        peak_display = axis.transData.transform((projected_x, projected_y))
        peak_figure = figure.transFigure.inverted().transform(peak_display)
        axis.annotate(
            DOMAIN_SHORT[domain],
            xy=peak_figure,
            xycoords=figure.transFigure,
            xytext=label_position,
            textcoords=figure.transFigure,
            fontsize=6.5,
            weight="bold",
            color=INK,
            ha=alignment,
            va="center",
            bbox={
                "facecolor": WHITE,
                "edgecolor": "none",
                "alpha": 0.92,
                "pad": 0.2,
            },
            arrowprops={
                "arrowstyle": "-|>",
                "color": INK,
                "linewidth": 0.7,
                "mutation_scale": 9,
                "shrinkA": 2,
                "shrinkB": 0,
            },
            annotation_clip=False,
            zorder=10,
        )
    axis.grid(False)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))


def draw_gain_board(
    figure: Figure,
    spec: mpl.gridspec.SubplotSpec,
    gains: pd.DataFrame,
    summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> list[Axes]:
    """Render the three exact D5 adjacent-set gain heatmaps."""
    domains = display_domains(config)
    labels = [str(index) for index in range(1, 13)]
    years = list(
        range(int(config["landscape_year_min"]), int(config["landscape_year_max"]) + 1)
    )
    ordered = summary.sort_values("comparison_order")
    limit = float(config["gain_color_limit"])
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    grid = spec.subgridspec(
        2,
        4,
        width_ratios=[1, 1, 1, 0.055],
        height_ratios=[0.90, 1],
        wspace=0.09,
        hspace=0.02,
    )
    axes: list[Axes] = []
    for column, row in enumerate(ordered.itertuples(index=False)):
        title_axis = figure.add_subplot(grid[0, column])
        title_axis.set_axis_off()
        comparison = str(row.comparison_label).replace(" − ", " −\n")
        median = f"{row.median_delta_spearman:+.4f}".replace("+0.", "+.").replace(
            "-0.", "−."
        )
        title_axis.text(
            0.5,
            0.96,
            comparison,
            ha="center",
            va="top",
            fontsize=6.5,
            weight="bold",
            color=INK,
            linespacing=0.95,
        )
        title_axis.text(
            0.5,
            0.08,
            f"Median {median}\n{100 * row.positive_share:.1f}% positive",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=MUTED,
            linespacing=0.95,
        )
        axis = figure.add_subplot(grid[1, column])
        group = gains[gains["comparison_label"].eq(row.comparison_label)]
        values, statuses = matrix_for_group(group, domains, years, "display_delta")
        draw_status_background(axis, statuses)
        masked = np.ma.masked_where(statuses != "reliable", values)
        axis.pcolormesh(masked, cmap=gain_cmap(), norm=norm, shading="flat", zorder=1)
        mark_clipped_cells(axis, group, domains, years)
        style_heatmap(
            axis,
            years,
            labels,
            show_y=column == 0,
            show_x=True,
            show_right_year=column == len(ordered) - 1,
            sparse_y=True,
        )
        axes.append(title_axis)
        axes.append(axis)
    cax = figure.add_subplot(grid[1, 3])
    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=gain_cmap()),
        cax=cax,
        orientation="vertical",
    )
    colorbar.set_ticks([-limit, 0.0, limit])
    colorbar.ax.tick_params(labelsize=6.5, length=1.5, pad=1)
    colorbar.ax.set_yticklabels(["−.08", "0", "+.08"])
    cax.set_title("Δρ", fontsize=6.8, pad=3, color=INK)
    axes.append(cax)
    return axes


def mark_clipped_cells(
    axis: Axes,
    group: pd.DataFrame,
    domains: Sequence[str],
    years: Sequence[int],
) -> None:
    """Disclose cells clipped by the focused symmetric gain scale."""
    domain_lookup = {domain: index for index, domain in enumerate(domains)}
    year_lookup = {year: index for index, year in enumerate(years)}
    for row in group[group["out_of_scale"].eq(True)].itertuples(index=False):
        marker = "^" if float(row.delta_spearman) > 0 else "v"
        axis.scatter(
            year_lookup[int(row.window_end)] + 0.5,
            domain_lookup[str(row.domain12)] + 0.5,
            marker=marker,
            s=4,
            color=INK,
            linewidths=0,
            zorder=4,
        )


def build_figure(
    config: Mapping[str, Any],
    score_summary: pd.DataFrame,
    deciles: pd.DataFrame,
    landscape: pd.DataFrame,
    gains: pd.DataFrame,
    gain_summary: pd.DataFrame,
) -> Figure:
    """Assemble the compact five-panel print figure."""
    configure_matplotlib()
    render = config["render"]
    figure = plt.figure(
        figsize=(float(render["width_inches"]), float(render["height_inches"]))
    )
    outer = figure.add_gridspec(
        5,
        12,
        height_ratios=[0.48, 2.50, 0.66, 1.82, 1.78],
        left=0.065,
        right=0.955,
        top=0.99,
        bottom=0.05,
        hspace=0.24,
        wspace=0.30,
    )
    header = figure.add_subplot(outer[0, :])
    header.set_axis_off()
    header.text(
        0.0,
        0.92,
        "Fig. 3 | Out-of-time validation of ASPR Score",
        fontsize=12.2,
        weight="bold",
        color=INK,
        va="top",
    )
    header.text(
        0.0,
        0.08,
        "for subsequent scientific diffusion",
        fontsize=7.8,
        color=MUTED,
        va="bottom",
    )
    top = outer[1, :].subgridspec(1, 2, width_ratios=[0.37, 0.63], wspace=0.15)
    axis_a = figure.add_subplot(top[0, 0])
    draw_panel_a(axis_a, score_summary)
    b_grid = top[0, 1].subgridspec(2, 1, height_ratios=[0.42, 1], hspace=0.04)
    b_header = figure.add_subplot(b_grid[0, 0])
    draw_panel_b_header(b_header)
    draw_panel_b_board(figure, b_grid[1, 0], deciles, config)
    c_header = outer[2, :].subgridspec(2, 1, height_ratios=[0.68, 0.32], hspace=0.02)
    c_title = figure.add_subplot(c_header[0, 0])
    draw_panel_c_title(c_title)
    c_toolbar = figure.add_subplot(c_header[1, 0])
    draw_panel_c_toolbar(c_toolbar, config)
    draw_performance_board(figure, outer[3, :], landscape, config)
    bottom = outer[4, :].subgridspec(
        2,
        2,
        height_ratios=[0.38, 1.0],
        width_ratios=[0.42, 0.58],
        hspace=0.06,
        wspace=0.12,
    )
    d_header = figure.add_subplot(bottom[0, 0])
    panel_label(
        d_header,
        "d",
        "D5 Primary16 3D performance terrain",
        "D5 × Primary 16 · interpolation within observed years",
    )
    d_axis = cast(Axes3D, figure.add_subplot(bottom[1, 0], projection="3d"))
    draw_ridgeline_terrain(d_axis, landscape, config)
    e_header = figure.add_subplot(bottom[0, 1])
    panel_label(
        e_header,
        "e",
        "Incremental value of broader feature sets",
        "D5 Δρ vs preceding set · ▲/▼ beyond ±.08",
    )
    draw_gain_board(figure, bottom[1, 1], gains, gain_summary, config)
    add_panel_frame(figure, [top[0, 0]])
    add_panel_frame(figure, [b_grid[0, 0], b_grid[1, 0]])
    add_panel_frame(figure, [c_header[0, 0], c_header[1, 0], outer[3, :]])
    add_panel_frame(figure, [bottom[0, 0], bottom[1, 0]])
    add_panel_frame(figure, [bottom[0, 1], bottom[1, 1]])
    return figure


def minimum_font_size(figure: Figure) -> float:
    """Return the smallest visible Matplotlib text size in points."""
    sizes = [
        text.get_fontsize()
        for axis in figure.axes
        for text in axis.texts
        if text.get_visible()
    ]
    for axis in figure.axes:
        sizes.extend(
            label.get_fontsize()
            for label in axis.get_xticklabels()
            if label.get_visible()
        )
        sizes.extend(
            label.get_fontsize()
            for label in axis.get_yticklabels()
            if label.get_visible()
        )
        sizes.extend([axis.xaxis.label.get_fontsize(), axis.yaxis.label.get_fontsize()])
    return float(min(sizes))


def unexpected_text_overlaps(figure: Figure) -> list[Mapping[str, Any]]:
    """Return non-trivial intersections among rendered text bounding boxes."""
    figure.canvas.draw()
    renderer = cast(FigureCanvasAgg, figure.canvas).get_renderer()
    items: list[tuple[int, str, Bbox]] = []
    for axis_index, axis in enumerate(figure.axes):
        texts = list(axis.texts)
        if axis.axison:
            texts.extend(
                [axis.title, axis.xaxis.label, axis.yaxis.label]
                + list(axis.get_xticklabels())
                + list(axis.get_yticklabels())
            )
        for item in texts:
            label = item.get_text().strip()
            if not label or not item.get_visible():
                continue
            bounds = (
                Text.get_window_extent(item, renderer)
                if isinstance(item, Annotation)
                else item.get_window_extent(renderer)
            ).expanded(1.005, 1.02)
            if bounds.width > 0 and bounds.height > 0:
                items.append((axis_index, label, bounds))
    overlaps: list[Mapping[str, Any]] = []
    for index, (axis_a, label_a, bounds_a) in enumerate(items):
        for axis_b, label_b, bounds_b in items[index + 1 :]:
            width = min(bounds_a.x1, bounds_b.x1) - max(bounds_a.x0, bounds_b.x0)
            height = min(bounds_a.y1, bounds_b.y1) - max(bounds_a.y0, bounds_b.y0)
            if width <= 0 or height <= 0:
                continue
            intersection = width * height
            smaller = min(
                bounds_a.width * bounds_a.height, bounds_b.width * bounds_b.height
            )
            ratio = intersection / smaller
            if ratio >= 0.12:
                overlaps.append(
                    {
                        "axis_a": axis_a,
                        "text_a": label_a,
                        "axis_b": axis_b,
                        "text_b": label_b,
                        "intersection_ratio": float(ratio),
                    }
                )
    return overlaps


def export_figure(
    figure: Figure, output_dir: Path, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Export editable vectors, 600-dpi PNG, and accessibility previews."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(config["render"]["dpi"])
    min_font = minimum_font_size(figure)
    text_overlaps = unexpected_text_overlaps(figure)
    artifacts: dict[str, Any] = {}
    for extension in config["render"]["formats"]:
        path = output_dir / f"figure_full.{extension}"
        figure.savefig(path, dpi=dpi, facecolor=WHITE)
        artifacts[str(extension)] = artifact_record(path)
    plt.close(figure)
    png_path = output_dir / "figure_full.png"
    grayscale_path = output_dir / "figure_full_grayscale.png"
    deuteranopia_path = output_dir / "figure_full_deuteranopia.png"
    with Image.open(png_path).convert("RGB") as image:
        image.convert("L").convert("RGB").save(grayscale_path)
        simulate_deuteranopia(image).save(deuteranopia_path)
        dimensions = [int(image.width), int(image.height)]
    artifacts["grayscale"] = artifact_record(grayscale_path)
    artifacts["deuteranopia"] = artifact_record(deuteranopia_path)
    artifacts["png"]["pixel_dimensions"] = dimensions
    artifacts["physical_size_mm"] = [
        float(config["render"]["width_mm"]),
        float(config["render"]["height_mm"]),
    ]
    artifacts["minimum_font_size_pt"] = min_font
    artifacts["unexpected_text_overlap_count"] = len(text_overlaps)
    artifacts["unexpected_text_overlaps"] = text_overlaps
    return artifacts


def simulate_deuteranopia(image: Image.Image) -> Image.Image:
    """Create a deterministic approximate deuteranopia QA preview."""
    array = np.asarray(image, dtype=float) / 255.0
    matrix = np.array(
        [
            [0.625, 0.375, 0.0],
            [0.700, 0.300, 0.0],
            [0.000, 0.300, 0.700],
        ]
    )
    transformed = np.clip(array @ matrix.T, 0.0, 1.0)
    return Image.fromarray(np.round(255 * transformed).astype(np.uint8), mode="RGB")


def artifact_record(path: Path) -> Mapping[str, Any]:
    """Return file provenance for one artifact."""
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def nature_chart_contract(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the frozen refined renderer contract."""
    return {
        "figure_id": 3,
        "figure_version": config["figure_version"],
        "surface": "220 × 220 mm editable expanded research figure",
        "panels": {
            "a": "five-node formal ASPR construction with separate formulas and claim boundary",
            "b": "twelve separate D3/D5/D8 × four-set OOF decile curves with 95% CI",
            "c": "exact 3-by-4 temporal-domain heatmap board",
            "d": "D5 Primary 16 continuous semi-transparent 3D performance terrain",
            "e": "three exact adjacent-set D5 gain heatmaps",
        },
        "data_policy": "render frozen panel tables only; no simulation, fitting, bootstrap, or numeric recomputation",
        "interpolation": "Panel c and e use exact cells only; Panel d linearly bridges internal missing years within each domain's observed endpoints, then applies a visual surface spline across the shared mature-year range; it never extrapolates beyond observed endpoints",
        "model_family": "hgb",
        "minimum_font_size_pt": float(config["minimum_font_size_pt"]),
        "palette": "blue/teal/pale-yellow/orange/red absolute scale; blue-decrease/white/orange-increase gain scale",
        "performance_scale": "focused 0.68-0.82; under/over values shown with disclosed endpoint colors",
        "domain_display_order": "descending mean reliable D5 Primary 16 annual-window Spearman; display only",
        "ridgeline_color": "relative domain-mean performance on the same blue-to-red palette",
        "claim_boundary": "predictive screening, not causality or direct novelty judgment",
    }


def nature_panel_text(deciles: pd.DataFrame) -> Mapping[str, Any]:
    """Return the five-panel caption without changing any numeric evidence."""
    top_rows = deciles.loc[deciles["prediction_decile"].eq(10)]
    d5_top = top_rows.loc[
        top_rows["horizon"].eq(5) & top_rows["model_id"].eq("primary")
    ].iloc[0]
    return {
        "title": "Out-of-time validation of ASPR Score",
        "caption": (
            "Fig. 3 | Out-of-time validation of ASPR Score. a, Publication-time indicators enter a calibrated "
            "two-part HGB; its raw expected-diffusion prediction is mapped through the mature-D5 empirical CDF "
            "to the 0–100 ASPR score. b, Twelve separate fold-local decile curves report enrichment for every "
            "D3/D5/D8 × Strict 7/Primary 16/Expanded 153/Broad T0 219 combination, with year-block bootstrap "
            "intervals and D10 share/lift callouts. The D5 Primary 16 cell reaches "
            f"{100 * d5_top['observed_top_share']:.2f}% and {d5_top['enrichment_over_baseline']:.2f}× baseline. "
            "c, Exact three-year trailing heatmaps retain the frozen D3/D5/D8 × four-set × twelve-domain × year "
            "results. d, A single semi-transparent surface visually interpolates the D5 Primary 16 mature-year "
            "terrain only within each domain's observed endpoints; the three highest mean-performance domains are "
            "labelled, while exact values remain in c. e, Exact D5 adjacent-set "
            "gain maps show local "
            "changes on a shared zero-centered scale. ASPR is a predictive screening signal, not a causal "
            "estimate or direct novelty judgment."
        ),
    }


def render_from_tables(
    config: Mapping[str, Any], output_dir: Path
) -> Mapping[str, Any]:
    """Render only from hash-frozen panel tables."""
    panel_dir = output_dir / "panel_data"
    verify_frozen_tables(panel_dir, config)
    score_summary = pd.read_csv(panel_dir / "score_summary.csv")
    deciles = pd.read_csv(panel_dir / "decile_enrichment.csv")
    landscape = pd.read_csv(panel_dir / "performance_landscape.csv")
    gains = pd.read_csv(panel_dir / "d5_gain_landscape.csv")
    gain_summary = pd.read_csv(panel_dir / "d5_gain_summary.csv")
    display_order = pd.read_csv(panel_dir / "domain_display_order.csv")
    runtime_config = dict(config)
    runtime_config["domain_display_order"] = [
        {"id": row.domain12, "label": row.domain_label}
        for row in display_order.itertuples(index=False)
    ]
    figure = build_figure(
        runtime_config, score_summary, deciles, landscape, gains, gain_summary
    )
    artifacts = export_figure(figure, output_dir, runtime_config)
    write_json(
        output_dir / "chart_contract.json", nature_chart_contract(runtime_config)
    )
    write_json(output_dir / "panel_text.json", nature_panel_text(deciles))
    inventory = {
        "figure_version": config["figure_version"],
        "panel_data_manifest": read_json(output_dir / "panel_data_manifest.json"),
        "artifacts": artifacts,
        "panel_tables": {
            path.name: artifact_record(path)
            for path in sorted(panel_dir.iterdir())
            if path.is_file()
        },
    }
    write_json(output_dir / "output_inventory.json", inventory)
    return inventory


def verify_frozen_tables(panel_dir: Path, config: Mapping[str, Any]) -> None:
    """Refuse to render when any frozen data table changed."""
    manifest_path = panel_dir.parent / "panel_data_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        expected_hashes = {
            str(name): str(record["sha256"]).removeprefix("sha256:")
            for name, record in manifest["tables"].items()
        }
    else:
        expected_hashes = {
            str(name): str(value).removeprefix("sha256:")
            for name, value in config["frozen_panel_sha256"].items()
        }
    mismatches = []
    for name, expected in expected_hashes.items():
        path = panel_dir / str(name)
        observed = (
            sha256_file(path).removeprefix("sha256:") if path.is_file() else "missing"
        )
        if observed != expected:
            mismatches.append(f"{name}: {observed} != {expected}")
    if mismatches:
        raise ValueError(f"frozen Fig.3 panel data changed: {mismatches}")


__all__ = [
    "build_figure",
    "export_figure",
    "render_from_tables",
    "verify_frozen_tables",
]
