"""Vector-first renderer for the evidence-derived v3 Fig.2."""

from __future__ import annotations

import json
import re
import textwrap
import xml.etree.ElementTree as xml_etree
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorspacious import cspace_convert
from matplotlib import colors as mcolors
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Bbox
from PIL import Image

from experiments.common.new.base.common import FigureBundle


INK = "#18212F"
NAVY = "#244A6B"
BLUE = "#3676A8"
TEAL = "#16806F"
GREEN = "#009E73"
AMBER = "#D88A12"
ORANGE = "#E66F32"
PURPLE = "#8769A9"
SLATE = "#657786"
MID_GREY = "#7C8790"
LIGHT_GREY = "#D7DEE2"
GRID_GREY = "#E9EDF0"
PALE_BLUE = "#EAF2F7"
PALE_AMBER = "#FCF2DF"
PALE_GREEN = "#E8F5F0"
PALE_GREY = "#F6F8F9"
WHITE = "#FFFFFF"

SOURCE_ROLE_COLORS = {
    "direct_innovation": "#0072B2",
    "t0_substantive": "#009E73",
    "t0_opportunity": "#E69F00",
    "context_control": "#697A8A",
}
DIMENSION_ROLE_COLORS = {
    "substantive_innovation": "#0072B2",
    "t0_potential": "#009E73",
    "opportunity": "#E69F00",
    "context_control": "#7A8793",
    "sensitivity": "#B07AA1",
    "unassigned": "#C9D0D5",
}
EXCLUSIVE_TIER_COLORS = {
    "strict_core": "#1E2935",
    "fulltext_only": "#4A75A0",
    "source_only": "#7FA6C5",
    "broad_t0_only": "#C2D6E5",
    "excluded": "#D9DEE3",
}
OPERATIONALIZATION_COLORS = {
    "source_formula_existing": "#244A6B",
    "source_formula_local_surrogate": "#5E86AA",
    "structured_construct_proxy": "#A9C4DA",
    "title_taxonomy_lexical_proxy": "#D9E6F0",
}

# The master SVG is 864 pt wide and is intended for a 183-mm full-width
# placement.  Matplotlib writes SVG ``font-size`` values in CSS px, so these
# helpers make the *final printed* size explicit rather than treating a
# screen-preview px value as a publishing font size.  At this placement,
# 1 SVG CSS px becomes ~0.450 pt on the page.
MASTER_SVG_WIDTH_PT = 864.0
TARGET_PRINT_WIDTH_MM = 183.0
CSS_PX_TO_PT = 72.0 / 96.0
TARGET_TO_MASTER_SCALE = (
    TARGET_PRINT_WIDTH_MM / 25.4 * 72.0 / MASTER_SVG_WIDTH_PT
)


def _print_font(final_pt: float) -> float:
    """Return a CSS-pixel font size that prints as ``final_pt`` at 183 mm."""
    return final_pt / (CSS_PX_TO_PT * TARGET_TO_MASTER_SCALE)


FONT_NOTE = _print_font(5.0)
FONT_BODY = _print_font(5.5)
FONT_BODY_EMPHASIS = _print_font(6.0)
FONT_SECTION = _print_font(6.5)
FONT_PANEL_TITLE = _print_font(7.5)
FONT_PANEL_MARK = _print_font(8.0)
FONT_FIGURE_TITLE = _print_font(10.0)
FONT_KEY_NUMBER = _print_font(10.5)
SUPPORT_NOTE_GID = "fig2-support-note"


def _blend(color: str, white_fraction: float) -> str:
    """Blend one color toward white."""
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    mixed = rgb * (1.0 - white_fraction) + np.ones(3) * white_fraction
    return mcolors.to_hex(mixed)


def _renderer_config(bundle: FigureBundle) -> Dict[str, Any]:
    """Resolve renderer defaults plus the frozen local configuration."""
    defaults: Dict[str, Any] = {
        "canvas_px": [7200, 12000],
        "min_font_pt": 5.0,
        "qa_preview_width": 1800,
    }
    defaults.update(bundle.chart_contract.get("render_config", {}))
    return defaults


def _verify_dependencies(bundle: FigureBundle) -> Dict[str, str]:
    """Check the compact plotting environment recorded in the contract."""
    observed: Dict[str, str] = {}
    for package, required in bundle.chart_contract["required_plot_packages"].items():
        try:
            observed[package] = version(package)
        except PackageNotFoundError as error:
            raise RuntimeError(f"Missing Fig.2 plotting dependency: {package}") from error
        if observed[package] != required:
            raise RuntimeError(
                f"Fig.2 requires {package}=={required}; observed {observed[package]}"
            )
    return observed


def _rc_params() -> Dict[str, Any]:
    """Return stable, vector-friendly scientific figure defaults."""
    return {
        "font.family": "DejaVu Sans",
        "font.size": FONT_BODY,
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "axes.edgecolor": LIGHT_GREY,
        "axes.linewidth": 0.55,
        "text.color": INK,
        "svg.fonttype": "none",
        "svg.hashsalt": "aspr-fig2-evidence-derived-v3",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }


def _panel_frame(
    ax: plt.Axes,
    panel: str,
    title: str,
    subtitle: str | None,
) -> None:
    """Draw a quiet, print-legible panel boundary and header."""
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.add_patch(
        Rectangle(
            (0.002, 0.002),
            0.996,
            0.996,
            transform=ax.transAxes,
            facecolor=WHITE,
            edgecolor=LIGHT_GREY,
            linewidth=0.65,
            clip_on=False,
            zorder=0,
        )
    )
    ax.text(
        0.014,
        0.982,
        panel,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_PANEL_MARK,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.070,
        0.982,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_PANEL_TITLE,
        fontweight="bold",
        color=INK,
    )
    if subtitle:
        panel_width = float(ax.get_position().width)
        subtitle_width = 96 if panel_width > 0.80 else 58
        ax.text(
            0.070,
            0.915,
            textwrap.fill(subtitle, width=subtitle_width),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_BODY,
            color=NAVY,
            linespacing=0.96,
            clip_on=True,
        )


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str = WHITE,
    edgecolor: str = LIGHT_GREY,
    linewidth: float = 0.55,
    linestyle: str = "solid",
    rounding: float = 0.008,
    zorder: int = 2,
) -> FancyBboxPatch:
    """Add one axes-coordinate rounded card."""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.003,rounding_size={rounding}",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MID_GREY,
    linewidth: float = 0.65,
    zorder: int = 4,
) -> None:
    """Draw a compact arrow in axes coordinates."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=6.0,
            linewidth=linewidth,
            color=color,
            shrinkA=0.0,
            shrinkB=0.0,
            zorder=zorder,
        )
    )


def _draw_query_scope(
    ax: plt.Axes,
    query_blocks: Sequence[Mapping[str, str]],
) -> None:
    """Draw the compact scope lock and bootstrap-query document."""
    x, y, width, height = 0.018, 0.676, 0.315, 0.228
    _box(
        ax,
        x,
        y,
        width,
        height,
        facecolor=WHITE,
        edgecolor=NAVY,
        linewidth=0.68,
        rounding=0.004,
    )
    ax.text(
        x + 0.010,
        y + height - 0.013,
        "Scope lock and domain-agnostic bootstrap search",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.25,
        fontweight="bold",
        color=NAVY,
    )
    _box(
        ax,
        x + 0.010,
        y + height - 0.055,
        width - 0.020,
        0.030,
        facecolor=PALE_BLUE,
        edgecolor=_blend(NAVY, 0.52),
        linewidth=0.40,
        rounding=0.003,
    )
    ax.text(
        x + 0.017,
        y + height - 0.040,
        "English only · paper-level · T0 only · no outcome-driven admission",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=4.15,
        color=INK,
    )
    display_labels = {
        "PAPER OBJECT": "PAPER",
        "TARGET CONSTRUCT": "TARGET",
        "EVIDENCE ROLE": "EVIDENCE",
    }
    ax.plot(
        [x + 0.062, x + 0.062],
        [y + 0.025, y + height - 0.066],
        transform=ax.transAxes,
        color=_blend(NAVY, 0.48),
        linewidth=0.45,
    )
    for index, block in enumerate(query_blocks):
        block_y = y + height - 0.082 - index * 0.038
        ax.text(
            x + 0.014,
            block_y,
            display_labels[str(block["label"])],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=4.15,
            fontweight="bold",
            color=NAVY,
        )
        terms = (
            str(block["terms"])
            .replace("research quality", "quality")
            .replace("scientific influence", "influence")
        )
        ax.text(
            x + 0.070,
            block_y,
            terms,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=3.95,
            color=INK,
            clip_on=True,
        )
        if index < len(query_blocks) - 1:
            ax.text(
                x + 0.057,
                block_y - 0.019,
                "AND",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=3.75,
                fontweight="bold",
                color=SLATE,
            )
    ax.text(
        x + width - 0.010,
        y + 0.045,
        "Domain-agnostic start; no preset K / Q / P / M / D / F.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.25,
        fontweight="bold",
        color=TEAL,
    )
    ax.text(
        x + 0.010,
        y + 0.027,
        "Retrieval counts: K domains · Q logical queries · P API requests",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=3.55,
        color=NAVY,
        fontweight="bold",
    )
    ax.text(
        x + 0.010,
        y + 0.012,
        "Measurement counts: M candidate dimensions · D predictive dimensions · F retained indicators",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=3.30,
        color=NAVY,
    )


def _draw_round_yields(ax: plt.Axes, rounds: pd.DataFrame) -> None:
    """Render the twelve discrete batches as an audit table, not a trend plot."""
    x, y, width, height = 0.347, 0.676, 0.637, 0.228
    _box(
        ax,
        x,
        y,
        width,
        height,
        facecolor=WHITE,
        edgecolor=NAVY,
        linewidth=0.68,
        rounding=0.004,
    )
    ax.text(
        x + 0.010,
        y + height - 0.013,
        "Twelve-round evidence-discovery cycle (discrete batch gains)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.25,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        x + 0.014,
        y + height - 0.053,
        "Frozen batch  →  AI/H1 screening  →  H2 adjudication  →  source-term / indicator extraction  →  prior-codebook alignment  →  novelty endpoint update",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=3.95,
        color=INK,
    )
    work = rounds.sort_values("iteration").reset_index(drop=True)
    table_x = x + 0.014
    table_y = y + 0.020
    table_height = 0.108
    note_width = 0.102
    note_x = x + width - note_width - 0.012
    table_width = note_x - table_x - 0.012
    label_width = 0.046
    cell_width = (table_width - label_width) / len(work)
    row_edges = [table_y, table_y + 0.036, table_y + 0.072, table_y + table_height]
    for edge in row_edges:
        ax.plot(
            [table_x, table_x + table_width],
            [edge, edge],
            transform=ax.transAxes,
            color=LIGHT_GREY,
            linewidth=0.42,
        )
    ax.plot(
        [table_x, table_x],
        [table_y, table_y + table_height],
        transform=ax.transAxes,
        color=LIGHT_GREY,
        linewidth=0.42,
    )
    ax.plot(
        [table_x + label_width, table_x + label_width],
        [table_y, table_y + table_height],
        transform=ax.transAxes,
        color=LIGHT_GREY,
        linewidth=0.42,
    )
    for index in range(len(work) + 1):
        cell_x = table_x + label_width + index * cell_width
        ax.plot(
            [cell_x, cell_x],
            [table_y, table_y + table_height],
            transform=ax.transAxes,
            color=LIGHT_GREY,
            linewidth=0.34,
        )
    ax.text(
        table_x + 0.006,
        table_y + 0.054,
        "Terms",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=4.15,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        table_x + 0.006,
        table_y + 0.018,
        "Indicators",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=4.15,
        fontweight="bold",
        color=INK,
    )
    for index, row in work.iterrows():
        cx = table_x + label_width + cell_width * (index + 0.5)
        if int(row["iteration"]) == 12:
            _box(
                ax,
                table_x + label_width + index * cell_width + 0.001,
                table_y + 0.001,
                cell_width - 0.002,
                table_height - 0.002,
                facecolor=PALE_AMBER,
                edgecolor=AMBER,
                linewidth=0.75,
                linestyle="dashed",
                rounding=0.003,
                zorder=1,
            )
        ax.text(
            cx,
            table_y + 0.090,
            f"R{int(row['iteration'])}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=4.15,
            color=INK,
            fontweight="bold" if int(row["iteration"]) == 12 else "normal",
        )
        ax.text(
            cx,
            table_y + 0.054,
            str(int(row["new_nonredundant_english_terms"])),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=4.45,
            color=BLUE,
            fontweight="bold",
        )
        ax.text(
            cx,
            table_y + 0.018,
            str(int(row["new_canonical_indicator_families"])),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=4.45,
            color=ORANGE,
            fontweight="bold",
        )
    _box(
        ax,
        note_x,
        table_y,
        note_width,
        table_height,
        facecolor=WHITE,
        edgecolor=ORANGE,
        linewidth=0.70,
        linestyle="dashed",
        rounding=0.003,
    )
    ax.text(
        note_x + 0.008,
        table_y + table_height - 0.009,
        "R12 frozen by\nregistered marginal-yield\namendment;\nΔterms = 10,\nΔindicators = 9;\nnot dual-zero.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=3.85,
        color=INK,
        linespacing=0.96,
    )
def _draw_review_ledger(ax: plt.Axes, review: pd.DataFrame) -> None:
    """Draw a readable, role-separated review ledger and provenance note."""
    x, y, width, height = 0.018, 0.154, 0.315, 0.520
    _box(
        ax,
        x,
        y,
        width,
        height,
        facecolor=WHITE,
        edgecolor=NAVY,
        linewidth=0.68,
        rounding=0.004,
    )
    ax.text(
        x + 0.010,
        y + height - 0.013,
        "Role-separated review and audit coverage",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.15,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        x + 0.010,
        y + height - 0.039,
        "Every stage retains its decision trail; records are never silently dropped.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=3.95,
        color=INK,
    )
    table_left, table_right = x + 0.011, x + width - 0.011
    table_top, table_bottom = y + height - 0.078, y + 0.252
    stage_x = table_left + 0.005
    headers = [
        (x + 0.183, "AI"),
        (x + 0.238, "H1"),
        (x + 0.288, "H2 / independent\nreview"),
    ]
    ax.plot([table_left, table_right], [table_top, table_top], transform=ax.transAxes, color=NAVY, linewidth=0.55)
    ax.plot([table_left, table_right], [table_bottom, table_bottom], transform=ax.transAxes, color=NAVY, linewidth=0.55)
    divider = x + 0.142
    ax.plot([divider, divider], [table_bottom, table_top], transform=ax.transAxes, color=LIGHT_GREY, linewidth=0.45)
    for header_x, label in headers:
        ax.text(
            header_x,
            table_top - 0.017,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=4.00,
            fontweight="bold",
            color=NAVY,
            linespacing=0.84,
        )
    ax.text(
        stage_x,
        table_top - 0.017,
        "Stage",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.15,
        fontweight="bold",
        color=NAVY,
    )
    display_names = {
        "Literature screening": "Literature screening",
        "Term coding": "Term coding",
        "Indicator census": "Indicator census",
        "Dimension coding": "Dimension coding",
    }
    row_height = (table_top - table_bottom - 0.034) / len(review)
    for index, row in enumerate(review.itertuples(index=False)):
        row_y = table_top - 0.034 - (index + 0.5) * row_height
        ax.plot(
            [table_left, table_right],
            [row_y - row_height / 2, row_y - row_height / 2],
            transform=ax.transAxes,
            color=GRID_GREY,
            linewidth=0.35,
        )
        ax.text(
            stage_x,
            row_y,
            display_names[str(row.stage)],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=4.35,
            color=INK,
        )
        for header_x, display, color in zip(
            (x + 0.189, x + 0.247, x + 0.293),
            (row.ai_display, row.h1_display, row.h2_display),
            (NAVY, NAVY, TEAL),
        ):
            ax.text(
                header_x,
                row_y,
                str(display),
                transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=3.85 if "\n" in str(display) else 4.15,
            color=color,
            fontweight="bold" if color == TEAL else "normal",
            linespacing=0.86,
        )
    metric_y, metric_height = y + 0.163, 0.066
    metric_width = (width - 0.040) / 3
    metrics = [
        (
            f"{int(review.iloc[0].human_attested_worksheet_count)}",
            "human-attested\nworksheets",
            NAVY,
        ),
        (
            f"{int(review.iloc[0].independent_ai_run_count)}",
            "independent review\nruns",
            TEAL,
        ),
        (
            f"{int(review.iloc[0].independent_ai_item_count):,}",
            "reviewed\nrows",
            PURPLE,
        ),
    ]
    for index, (value, label, color) in enumerate(metrics):
        metric_x = x + 0.012 + index * (metric_width + 0.008)
        _box(
            ax,
            metric_x,
            metric_y,
            metric_width,
            metric_height,
            facecolor=_blend(color, 0.925),
            edgecolor=_blend(color, 0.45),
            linewidth=0.48,
            rounding=0.003,
        )
        ax.text(
            metric_x + 0.008,
            metric_y + metric_height - 0.010,
            value,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.20,
            fontweight="bold",
            color=color,
        )
        ax.text(
            metric_x + 0.008,
            metric_y + 0.010,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=3.70,
            color=INK,
            linespacing=0.88,
        )
    ax.plot(
        [x + 0.014, x + width - 0.014],
        [y + 0.143, y + 0.143],
        transform=ax.transAxes,
        color=NAVY,
        linewidth=0.65,
    )
    ax.text(
        x + 0.018,
        y + 0.120,
        "How to read H1/H2 and the audit trail",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=4.65,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        x + 0.018,
        y + 0.098,
        textwrap.fill(
            "H1/H2 identify review roles, not two people. The 7 early workbooks were human-attested automated drafts; later replacement review was independent Codex AI, not a second human reviewer.",
            width=58,
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=3.78,
        color=INK,
        linespacing=0.98,
        clip_on=True,
    )
    ax.text(
        x + 0.018,
        y + 0.035,
        "Source fragments, adjudication decisions and hashes are retained.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=3.55,
        color=NAVY,
        fontweight="bold",
    )
    ax.text(
        x + 0.018,
        y + 0.018,
        f"{int(review.iloc[0]['excluded_local_qwen_artifact_count'])} isolated local-Qwen artifacts were excluded before all final totals.",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=3.55,
        color=NAVY,
        fontweight="bold",
    )


def _draw_process_lane(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    y: float,
    lane_label: str,
    color: str,
) -> None:
    """Draw one compact, non-proportional process lane."""
    work = data.sort_values("lane_order").reset_index(drop=True)
    x0, x1 = 0.017, 0.983
    gap = 0.014
    width = (x1 - x0 - gap * (len(work) - 1)) / len(work)
    height = 0.092
    ax.text(
        x0,
        y + height + 0.012,
        lane_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.15,
        fontweight="bold",
        color=color,
    )
    for index, row in work.iterrows():
        x = x0 + index * (width + gap)
        final = index == len(work) - 1
        _box(
            ax,
            x,
            y,
            width,
            height,
            facecolor=_blend(color, 0.93 if not final else 0.87),
            edgecolor=_blend(color, 0.44),
            linewidth=0.65 if final else 0.45,
            rounding=0.006,
        )
        ax.text(
            x + 0.008,
            y + height - 0.010,
            f"{int(row['count']):,}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            fontweight="bold",
            color=color,
        )
        ax.text(
            x + width - 0.007,
            y + height - 0.010,
            textwrap.fill(str(row["label"]), width=22),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=4.9,
            color=INK,
            linespacing=0.87,
            clip_on=True,
        )
        ax.text(
            x + 0.008,
            y + 0.010,
            textwrap.fill(str(row["detail"]), width=36),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=4.35,
            color=SLATE,
            linespacing=0.86,
            clip_on=True,
        )
        if index < len(work) - 1:
            _arrow(
                ax,
                (x + width + 0.002, y + height / 2),
                (x + width + gap - 0.002, y + height / 2),
                color=_blend(color, 0.30),
                linewidth=0.65,
            )


def _draw_evidence_chain(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    label: str,
    steps: Sequence[tuple[str, str]],
    color: str,
) -> list[float]:
    """Draw a compact evidence chain using counts and arrows, not stage cards."""
    positions = np.linspace(x + 0.026, x + width - 0.026, len(steps))
    ax.text(
        x,
        y + 0.052,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.55,
        fontweight="bold",
        color=color,
    )
    ax.plot(
        [positions[0], positions[-1]],
        [y + 0.013, y + 0.013],
        transform=ax.transAxes,
        color=_blend(color, 0.55),
        linewidth=0.62,
        zorder=1,
    )
    for index, (count, step_label) in enumerate(steps):
        node_x = float(positions[index])
        ax.scatter(
            [node_x],
            [y + 0.013],
            transform=ax.transAxes,
            s=15,
            facecolor=WHITE,
            edgecolor=color,
            linewidth=0.75,
            zorder=3,
        )
        ax.text(
            node_x,
            y + 0.026,
            count,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=5.15,
            fontweight="bold",
            color=color,
        )
        ax.text(
            node_x,
            y - 0.015,
            step_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=3.70,
            color=INK,
            linespacing=0.90,
        )
        if index < len(steps) - 1:
            _arrow(
                ax,
                (node_x + 0.010, y + 0.013),
                (float(positions[index + 1]) - 0.010, y + 0.013),
                color=_blend(color, 0.22),
                linewidth=0.55,
            )
    return [float(position) for position in positions]


def _draw_evidence_dossier(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    number: str,
    label: str,
    headline: str,
    count_lines: Sequence[str],
    audit_line: str,
    color: str,
) -> None:
    """Draw one evidence dossier: a narrative ledger rather than a flowchart."""
    _box(
        ax,
        x,
        y,
        width,
        height,
        facecolor=_blend(color, 0.955),
        edgecolor=_blend(color, 0.38),
        linewidth=0.58,
        rounding=0.005,
    )
    stripe_width = 0.068
    _box(
        ax,
        x + 0.004,
        y + 0.004,
        stripe_width,
        height - 0.008,
        facecolor=color,
        edgecolor=color,
        linewidth=0.0,
        rounding=0.003,
    )
    ax.text(
        x + 0.012,
        y + height - 0.016,
        number,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=WHITE,
    )
    ax.text(
        x + 0.012,
        y + 0.017,
        textwrap.fill(label.upper(), width=10),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=3.85,
        fontweight="bold",
        color=WHITE,
        linespacing=0.86,
    )
    text_x = x + stripe_width + 0.016
    ax.text(
        text_x,
        y + height - 0.015,
        headline,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.75,
        fontweight="bold",
        color=INK,
    )
    count_y = y + height - 0.045
    for line_index, count_line in enumerate(count_lines):
        ax.text(
            text_x,
            count_y - line_index * 0.024,
            count_line,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=4.70,
            fontweight="bold",
            color=color,
        )
    audit_y = y + 0.014
    ax.plot(
        [text_x, x + width - 0.010],
        [audit_y + 0.018, audit_y + 0.018],
        transform=ax.transAxes,
        color=_blend(color, 0.50),
        linewidth=0.45,
    )
    ax.text(
        text_x,
        audit_y,
        textwrap.fill(audit_line, width=120),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=3.75,
        color=INK,
        linespacing=0.92,
        clip_on=True,
    )


def _draw_review_roles(
    ax: plt.Axes,
    review: pd.Series,
    recall: pd.Series,
    press_unresolved: int,
) -> None:
    """Draw a grouped three-role review flow and five-column audit ledger."""
    flow_x, flow_y, flow_width, flow_height = 0.020, 0.045, 0.500, 0.225
    _box(
        ax,
        flow_x,
        flow_y,
        flow_width,
        flow_height,
        facecolor=WHITE,
        edgecolor=_blend(PURPLE, 0.28),
        linewidth=0.70,
        linestyle="dashed",
        rounding=0.010,
        zorder=1,
    )
    card_width = flow_width / 3
    roles = [
        (
            "AI extraction",
            ("surface candidate", "terms and indicators", "with source evidence"),
            BLUE,
        ),
        (
            "H1 verification",
            ("independently check", "definitions, roles,", "and T0 boundary"),
            TEAL,
        ),
        (
            "H2 adjudication",
            ("resolve disagreements", "on merge, split,", "and final decisions"),
            PURPLE,
        ),
    ]
    for index, (title, focus_lines, color) in enumerate(roles):
        x = flow_x + index * card_width
        center_x = x + card_width / 2
        ax.text(
            center_x,
            flow_y + flow_height - 0.052,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FONT_BODY,
            color=color,
            fontweight="bold",
        )
        for line_index, focus in enumerate(focus_lines):
            ax.text(
                center_x,
                flow_y + flow_height - 0.105 - line_index * 0.031,
                focus,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=FONT_NOTE,
                color=SLATE,
            )
    ledger_x, ledger_y, ledger_width, ledger_height = 0.535, 0.045, 0.445, 0.225
    metric_values = [
        ("attested\nsheets", "7", NAVY, FONT_SECTION),
        ("AI\nreviews", f"{int(review.independent_ai_run_count):,}", TEAL, FONT_SECTION),
        ("reviewed\nrows", f"{int(review.independent_ai_item_count):,}", PURPLE, FONT_SECTION),
        (
            "seed\nrecall",
            f"{int(recall.recalled_seed_count)}/{int(recall.indexable_seed_count)}",
            NAVY,
            FONT_SECTION,
        ),
        ("PRESS\nissues", str(press_unresolved), TEAL, FONT_SECTION),
    ]
    metric_width = ledger_width / len(metric_values)
    for index, (label, value, color, value_font) in enumerate(metric_values):
        x = ledger_x + index * metric_width
        if index:
            ax.plot(
                [x, x],
                [ledger_y, ledger_y + ledger_height],
                transform=ax.transAxes,
                color=LIGHT_GREY,
                linewidth=0.55,
                zorder=1,
            )
        ax.text(
            x + metric_width / 2,
            ledger_y + ledger_height - 0.016,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=FONT_NOTE,
            color=INK,
            fontweight="bold",
            linespacing=0.90,
        )
        ax.text(
            x + metric_width / 2,
            ledger_y + 0.098,
            value,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=value_font,
            color=color,
            fontweight="bold",
            linespacing=0.85,
        )


def _draw_panel_a(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Draw the evidence spine, iteration yields and expanded review trail."""
    text = bundle.panel_text["a"]
    _panel_frame(
        ax,
        "a",
        text["title"],
        None,
    )
    process = bundle.tables["fig2_process_stages"]
    query = bundle.tables["fig2_query_audit"].iloc[0]
    recall = bundle.tables["fig2_recall_audit"].iloc[0]
    review = bundle.tables["fig2_review_coverage"].iloc[0]
    rounds = bundle.tables["fig2_round_yields"].sort_values("iteration").reset_index(drop=True)
    terms = process.loc[process["lane"].eq("terms")].sort_values("lane_order").reset_index(drop=True)
    search = process.loc[process["lane"].eq("search")].sort_values("lane_order").reset_index(drop=True)
    measure = process.loc[process["lane"].eq("measure")].sort_values("lane_order").reset_index(drop=True)

    ax.text(
        0.020,
        0.900,
        "Scope lock: English · paper level · T0 only · outcome blind",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=FONT_BODY,
        color=INK,
        fontweight="bold",
    )

    stages = [
        ("01", "Domain\nstart", "bootstrap", "Domain-agnostic\nbootstrap search"),
        (
            "02",
            "Source\nterms",
            f"{int(terms.iloc[0]['count']):,} → {int(terms.iloc[3]['count']):,}",
            "3,615 raw\nEnglish terms\n→ 367 term\nfamilies",
        ),
        (
            "03",
            "Frozen search\nframe",
            f"{int(terms.iloc[4]['count'])} / {int(query.active_logical_queries)} / {int(query.physical_openalex_requests)}",
            "42 domains\n336 logical\n367 OpenAlex\nrequests",
        ),
        (
            "04",
            "Formal\ncensus",
            f"{int(search.iloc[3]['count']):,} → {int(search.iloc[4]['included_count']):,}",
            "Retrieval +\ncitation chasing\n363 papers\nin census",
        ),
        (
            "05",
            "Canonical\nindicators",
            f"{int(measure.iloc[0]['count']):,} → {int(measure.iloc[1]['count']):,}",
            "1,685 mentions\n→ 432 canonical\nfamilies",
        ),
        (
            "06",
            "Candidate\ndimensions",
            f"M = {int(measure.iloc[3]['count'])}",
            "Dimensions\nfrom indicators\nnot pre-set",
        ),
        (
            "07",
            "Frozen\nhard gates",
            "14 gates",
            "Fixed eligibility\nprotocol before\nset membership",
        ),
    ]
    positions = np.linspace(0.062, 0.938, len(stages))
    spine_y = 0.720
    ax.plot(
        [positions[0], positions[-1]],
        [spine_y, spine_y],
        transform=ax.transAxes,
        color=NAVY,
        linewidth=1.15,
        zorder=1,
    )
    for index in range(len(stages) - 1):
        _arrow(
            ax,
            (float(positions[index]) + 0.013, spine_y),
            (float(positions[index + 1]) - 0.013, spine_y),
            color=NAVY,
            linewidth=0.85,
            zorder=2,
        )
    for index, (number, label, value, note) in enumerate(stages):
        x = float(positions[index])
        color = INK if index == len(stages) - 1 else NAVY
        ax.scatter(
            [x],
            [spine_y],
            transform=ax.transAxes,
            s=64,
            facecolor=color,
            edgecolor=WHITE,
            linewidth=1.10,
            zorder=4,
        )
        ax.text(
            x,
            spine_y + 0.032,
            number,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=FONT_NOTE,
            color=color,
            fontweight="bold",
            zorder=5,
        )
        ax.text(
            x,
            0.820,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FONT_BODY_EMPHASIS,
            color=INK,
            fontweight="bold",
            linespacing=0.95,
        )
        ax.text(
            x,
            0.663,
            value,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=FONT_BODY,
            color=color,
            fontweight="bold",
            linespacing=0.92,
        )
        note_text = ax.text(
            x,
            0.628,
            note,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=FONT_NOTE,
            color=SLATE,
            linespacing=0.92,
            clip_on=True,
        )
        note_text.set_gid(SUPPORT_NOTE_GID)

    ax.plot([0.020, 0.980], [0.530, 0.530], transform=ax.transAxes, color=LIGHT_GREY, linewidth=0.70)
    ax.text(
        0.020,
        0.507,
        "Twelve discovery rounds: new terms (blue) and indicator families (orange)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_BODY,
        color=INK,
        fontweight="bold",
    )
    bar_x0, bar_width, bar_y, bar_height = 0.032, 0.660, 0.394, 0.084
    max_terms = max(int(value) for value in rounds["new_nonredundant_english_terms"])
    max_indicators = max(int(value) for value in rounds["new_canonical_indicator_families"])
    group_width = bar_width / len(rounds)
    for index, row in enumerate(rounds.itertuples(index=False)):
        x = bar_x0 + index * group_width
        if int(row.iteration) == 12:
            _box(
                ax,
                x + 0.002,
                bar_y - 0.010,
                group_width - 0.004,
                bar_height + 0.028,
                facecolor=PALE_AMBER,
                edgecolor=AMBER,
                linewidth=0.85,
                linestyle="dashed",
                rounding=0.002,
                zorder=1,
            )
        term_height = bar_height * 0.82 * int(row.new_nonredundant_english_terms) / max_terms
        indicator_height = bar_height * 0.82 * int(row.new_canonical_indicator_families) / max_indicators
        ax.add_patch(
            Rectangle(
                (x + group_width * 0.18, bar_y),
                group_width * 0.23,
                term_height,
                transform=ax.transAxes,
                facecolor=BLUE,
                edgecolor="none",
                zorder=3,
            )
        )
        ax.add_patch(
            Rectangle(
                (x + group_width * 0.54, bar_y),
                group_width * 0.23,
                indicator_height,
                transform=ax.transAxes,
                facecolor=ORANGE,
                edgecolor="none",
                zorder=3,
            )
        )
        ax.text(
            x + group_width / 2,
            bar_y - 0.022,
            f"R{int(row.iteration)}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=FONT_NOTE,
            color=INK,
            fontweight="bold" if int(row.iteration) == 12 else "normal",
        )
    _box(
        ax,
        0.715,
        0.388,
        0.265,
        0.100,
        facecolor=PALE_AMBER,
        edgecolor=AMBER,
        linewidth=0.85,
        linestyle="dashed",
        rounding=0.003,
    )
    ax.text(
        0.731,
        0.466,
        "R12: pragmatic stop",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_BODY,
        color=INK,
        fontweight="bold",
        linespacing=0.92,
    )
    ax.text(
        0.731,
        0.426,
        f"Δ {int(rounds.iloc[-1].new_nonredundant_english_terms)} terms · Δ {int(rounds.iloc[-1].new_canonical_indicator_families)} families",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_NOTE,
        color=INK,
    )
    ax.plot([0.020, 0.980], [0.350, 0.350], transform=ax.transAxes, color=LIGHT_GREY, linewidth=0.70)
    ax.text(
        0.020,
        0.305,
        "Review trail: AI extraction → H1 verification → H2 adjudication",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_BODY,
        color=INK,
        fontweight="bold",
    )
    _draw_review_roles(
        ax,
        review,
        recall,
        int(query.press_unresolved_active),
    )
    return [ax]


def _node_intervals(
    counts: Sequence[int],
    *,
    bottom: float,
    top: float,
    gap: float,
) -> list[tuple[float, float]]:
    """Allocate quantity-conserving top-to-bottom intervals."""
    scale = (top - bottom - gap * (len(counts) - 1)) / float(sum(counts))
    cursor = top
    output: list[tuple[float, float]] = []
    for count in counts:
        high = cursor
        low = high - float(count) * scale
        output.append((low, high))
        cursor = low - gap
    return output


def _flow_patch(
    ax: plt.Axes,
    x0: float,
    x1: float,
    source_low: float,
    source_high: float,
    target_low: float,
    target_high: float,
    color: str,
    *,
    alpha: float = 0.28,
    edgecolor: str = "none",
) -> None:
    """Draw one cubic alluvial ribbon in axes coordinates."""
    curve = 0.46 * (x1 - x0)
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
            edgecolor=edgecolor,
            linewidth=0.0,
            alpha=alpha,
            zorder=1,
        )
    )


def _draw_alluvial_nodes(
    ax: plt.Axes,
    nodes: pd.DataFrame,
    *,
    node_stage: str,
    x: float,
    intervals: Mapping[str, tuple[float, float]],
    colors: Mapping[str, str],
    label_x: float,
    label_side: str,
) -> None:
    """Draw alluvial node bars with direct count labels."""
    data = nodes.loc[nodes["node_stage"].eq(node_stage)].sort_values("node_order")
    for row in data.itertuples(index=False):
        low, high = intervals[str(row.node_id)]
        color = colors[str(row.node_id)]
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.006, low),
                0.012,
                max(high - low, 0.001),
                boxstyle="round,pad=0.001,rounding_size=0.002",
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=WHITE,
                linewidth=0.35,
                zorder=3,
            )
        )
        if int(row.feature_count) < 12:
            continue
        dimension_text = (
            f"{int(row.dimension_count)} dimensions / " if pd.notna(row.dimension_count) else ""
        )
        label = f"{row.label}\n{dimension_text}{int(row.feature_count)} families"
        if node_stage == "exclusive_tier" and str(row.node_id) != "excluded":
            label += f"\ncumulative {int(row.cumulative_count)}"
        alignment = "right" if label_side == "left" else "left"
        ax.text(
            label_x,
            (low + high) / 2,
            label,
            transform=ax.transAxes,
            ha=alignment,
            va="center",
            fontsize=4.55,
            color=INK,
            linespacing=0.88,
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": WHITE,
                "edgecolor": "none",
                "alpha": 0.82,
            },
            zorder=5,
        )


def _draw_indicator_alluvial(ax: plt.Axes, bundle: FigureBundle) -> None:
    """Draw a label-gutter alluvial so ribbons never carry the labels."""
    nodes = bundle.tables["fig2_indicator_dimension_nodes"]
    flows = bundle.tables["fig2_indicator_dimension_flows"]
    source = nodes.loc[nodes["node_stage"].eq("source_role")].sort_values("node_order")
    middle = nodes.loc[nodes["node_stage"].eq("dimension_role")].sort_values("node_order")
    target = nodes.loc[nodes["node_stage"].eq("exclusive_tier")].sort_values("node_order")
    bottom, top, gap = 0.105, 0.720, 0.009
    unit = (top - bottom - gap * (len(middle) - 1)) / 432.0

    def _intervals(stage: pd.DataFrame) -> Dict[str, tuple[float, float]]:
        counts = stage["feature_count"].astype(int).tolist()
        total_height = unit * sum(counts) + gap * (len(counts) - 1)
        cursor = (top + bottom + total_height) / 2.0
        output: Dict[str, tuple[float, float]] = {}
        for row in stage.itertuples(index=False):
            high = cursor
            low = high - unit * int(row.feature_count)
            output[str(row.node_id)] = (low, high)
            cursor = low - gap
        return output

    source_intervals = _intervals(source)
    middle_intervals = _intervals(middle)
    target_intervals = _intervals(target)
    source_cursor = {key: interval[1] for key, interval in source_intervals.items()}
    middle_left_cursor = {key: interval[1] for key, interval in middle_intervals.items()}
    for row in flows.loc[flows["flow_stage"].eq("source_to_dimension")].sort_values(
        ["source_order", "dimension_order"]
    ).itertuples(index=False):
        count = int(row.count)
        source_high = source_cursor[str(row.scope_role)]
        source_low = source_high - unit * count
        target_high = middle_left_cursor[str(row.dimension_role)]
        target_low = target_high - unit * count
        _flow_patch(
            ax,
            0.243,
            0.420,
            source_low,
            source_high,
            target_low,
            target_high,
            LIGHT_GREY if count < 10 else DIMENSION_ROLE_COLORS[str(row.dimension_role)],
            alpha=0.38 if count < 10 else 0.23,
        )
        source_cursor[str(row.scope_role)] = source_low
        middle_left_cursor[str(row.dimension_role)] = target_low
    middle_right_cursor = {key: interval[1] for key, interval in middle_intervals.items()}
    target_cursor = {key: interval[1] for key, interval in target_intervals.items()}
    for row in flows.loc[flows["flow_stage"].eq("dimension_to_tier")].sort_values(
        ["dimension_order", "tier_order"]
    ).itertuples(index=False):
        count = int(row.count)
        source_high = middle_right_cursor[str(row.dimension_role)]
        source_low = source_high - unit * count
        target_high = target_cursor[str(row.tier)]
        target_low = target_high - unit * count
        _flow_patch(
            ax,
            0.612,
            0.757,
            source_low,
            source_high,
            target_low,
            target_high,
            LIGHT_GREY if count < 10 else DIMENSION_ROLE_COLORS[str(row.dimension_role)],
            alpha=0.38 if count < 10 else 0.23,
        )
        middle_right_cursor[str(row.dimension_role)] = source_low
        target_cursor[str(row.tier)] = target_low

    headings = [
        (0.025, "Literature role\nn = 432"),
        (0.455, "Coded construct family\nn = 432"),
        (0.795, "Gate-defined tier\nn = 432"),
    ]
    for x, label in headings:
        ax.text(
            x,
            0.850,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_BODY,
            color=NAVY,
            fontweight="bold",
            linespacing=0.86,
        )

    def _draw_nodes(
        stage: pd.DataFrame,
        intervals: Mapping[str, tuple[float, float]],
        *,
        bar_x: float,
        label_x: float,
        label_side: str,
        colors: Mapping[str, str],
        stage_name: str,
    ) -> None:
        placed_y = top + 0.006
        for row in stage.itertuples(index=False):
            low, high = intervals[str(row.node_id)]
            center = (low + high) / 2.0
            ax.add_patch(
                Rectangle(
                    (bar_x - 0.008, low),
                    0.016,
                    max(high - low, 0.002),
                    transform=ax.transAxes,
                    facecolor=colors[str(row.node_id)],
                    edgecolor=WHITE,
                    linewidth=0.45,
                    zorder=4,
                )
            )
            if stage_name == "target":
                tier_labels = {
                    "strict_core": ("Strict · 7 (cum. 7)", 0.682),
                    "fulltext_only": ("Full text · 9 (cum. 16)", 0.625),
                    "source_only": ("Expanded · 137 (cum. 153)", 0.540),
                    "broad_t0_only": ("Broad T0 · 66 (cum. 219)", 0.430),
                    "excluded": ("Excluded · 213 (cum. 432)", 0.315),
                }
                label, label_y = tier_labels[str(row.node_id)]
            elif stage_name == "middle":
                if str(row.node_id) == "unassigned":
                    continue
                construct_labels = {
                    "substantive_innovation": "Innovation content",
                    "t0_potential": "T0 impact potential",
                    "opportunity": "Opportunity context",
                    "context_control": "Background controls",
                    "sensitivity": "Sensitivity-only",
                }
                label = (
                    f"{construct_labels[str(row.node_id)]}\n"
                    f"{int(row.feature_count)} families · {int(row.dimension_count)} dim"
                )
                label_y = center
            else:
                source_labels = {
                    "direct_innovation": "Direct innovation",
                    "t0_substantive": "T0 substantive",
                    "t0_opportunity": "T0 opportunity",
                    "context_control": "Context control",
                }
                label = f"{source_labels[str(row.node_id)]} · {int(row.feature_count)}"
                label_y = center
            if abs(label_y - center) > 0.014:
                x0 = bar_x + 0.009 if label_side == "right" else bar_x - 0.009
                x1 = label_x - 0.008 if label_side == "right" else label_x + 0.008
                ax.plot([x0, x1], [center, label_y], transform=ax.transAxes, color=MID_GREY, linewidth=0.48, zorder=3)
            ax.text(
                label_x,
                label_y,
                label,
                transform=ax.transAxes,
                ha="left" if label_side == "right" else "right",
                va="center",
                fontsize=FONT_NOTE,
                color=INK,
            linespacing=0.90,
                zorder=5,
            )

    _draw_nodes(source, source_intervals, bar_x=0.232, label_x=0.205, label_side="left", colors=SOURCE_ROLE_COLORS, stage_name="source")
    _draw_nodes(middle, middle_intervals, bar_x=0.432, label_x=0.455, label_side="right", colors=DIMENSION_ROLE_COLORS, stage_name="middle")
    _draw_nodes(target, target_intervals, bar_x=0.770, label_x=0.792, label_side="right", colors=EXCLUSIVE_TIER_COLORS, stage_name="target")


def _draw_panel_b(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Draw the alluvial classification map from evidence to dimensions."""
    text = bundle.panel_text["b"]
    _panel_frame(
        ax,
        "b",
        text["title"],
        None,
    )
    _draw_indicator_alluvial(ax, bundle)
    return [ax]


def _draw_gate_rows(ax: plt.Axes, gates: pd.DataFrame) -> None:
    """Draw the 14 fixed gates with readable short labels and a shared n."""
    work = gates.sort_values("gate_order").reset_index(drop=True)
    x_label, x_bar0, x_bar1, x_count = 0.030, 0.465, 0.875, 0.892
    ax.text(x_label, 0.805, "Fixed hard gate", transform=ax.transAxes, ha="left", va="top", fontsize=FONT_BODY, fontweight="bold", color=NAVY)
    ax.text((x_bar0 + x_bar1) / 2, 0.805, "pass count (n = 432)", transform=ax.transAxes, ha="center", va="top", fontsize=FONT_NOTE, fontweight="bold", color=NAVY)
    group_color = {
        "Scope and time boundary": TEAL,
        "Bias and outcome isolation": PURPLE,
        "Source and formula evidence": NAVY,
        "Implementation and quality": PURPLE,
        "Independent review": AMBER,
    }
    gate_labels = {
        "G01_IN_SCOPE_ROLE": "G01  Allowed role scope",
        "G02_ARTICLE_LEVEL": "G02  Paper level",
        "G03_PRIMARY_OR_FOUNDATIONAL_EVIDENCE": "G03  Foundational evidence",
        "G04_REPRODUCIBLE_DEFINITION": "G04  Reproducible definition",
        "G05_PUBLICATION_TIME": "G05  T0 computable",
        "G06_NO_FUTURE_INFORMATION": "G06  No future information",
        "G07_LOCAL_DATA_READY": "G07  Local data ready",
        "G08_BIAS_GUARDRAIL": "G08  Bias guardrail",
        "G09_NO_FATAL_VALIDITY_CONCERN": "G09  No fatal validity issue",
        "G10_OUTCOME_BLIND_SELECTION": "G10  Outcome blind",
        "G11_QUALITY_AUDIT": "G11  Quality audit",
        "G12_NONCONSTANT": "G12  Nonconstant",
        "G13_ENGLISH_FULLTEXT_FORMULA_EVIDENCE": "G13  Full-text/formula evidence",
        "G14_SECOND_HUMAN_APPROVAL": "G14  H2 approval",
    }
    cursor = 0.754
    for row in work.itertuples(index=False):
        color = group_color[str(row.group)]
        y = cursor
        ax.plot([0.020, 0.020], [y - 0.012, y + 0.012], transform=ax.transAxes, color=color, linewidth=2.2)
        ax.text(
            x_label,
            y,
            gate_labels[str(row.gate_id)],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=FONT_NOTE,
            color=INK,
        )
        ax.plot([x_bar0, x_bar1], [y, y], transform=ax.transAxes, color=GRID_GREY, linewidth=3.3, zorder=1)
        value_x = x_bar0 + (x_bar1 - x_bar0) * int(row.pass_count) / int(row.denominator)
        ax.plot([x_bar0, value_x], [y, y], transform=ax.transAxes, color=color, linewidth=3.3, zorder=2)
        ax.scatter([value_x], [y], transform=ax.transAxes, s=19, facecolor=WHITE, edgecolor=color, linewidth=0.85, zorder=3)
        ax.text(x_count, y, f"{int(row.pass_count)}", transform=ax.transAxes, ha="left", va="center", fontsize=FONT_NOTE, color=color, fontweight="bold")
        cursor -= 0.043


def _draw_panel_c(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Draw the fixed-gate audit without a downstream feature-set claim."""
    text = bundle.panel_text["c"]
    _panel_frame(
        ax,
        "c",
        text["title"],
        None,
    )
    _draw_gate_rows(ax, bundle.tables["fig2_gate_audit"])
    return [ax]


def _draw_stacked_bar(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    colors: Mapping[str, str],
    order_column: str,
    category_column: str,
    value_column: str,
    hatch_last: bool = False,
) -> None:
    """Draw a labelled 100% stacked bar from an already filtered table."""
    total = max(int(data[value_column].sum()), 1)
    cursor = x
    for index, row in enumerate(data.sort_values(order_column).itertuples(index=False)):
        value = int(getattr(row, value_column))
        fraction = value / total
        if fraction <= 0:
            continue
        color = colors[str(getattr(row, category_column))]
        ax.add_patch(
            FancyBboxPatch(
                (cursor, y),
                width * fraction,
                height,
                boxstyle="round,pad=0.0005,rounding_size=0.002",
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=WHITE,
                linewidth=0.40,
                hatch="//" if hatch_last and index == len(data) - 1 else None,
            )
        )
        if width * fraction > 0.038:
            ax.text(
                cursor + width * fraction / 2,
                y + height / 2,
                str(value),
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=4.3,
                color=WHITE if index < 2 else INK,
                fontweight="bold",
            )
        cursor += width * fraction


def _draw_feature_set_rows(ax: plt.Axes, bundle: FigureBundle) -> None:
    """Draw four nested sets with composition bars integrated into each row."""
    sets = bundle.tables["fig2_feature_sets"].sort_values("set_order")
    tiers = bundle.tables["fig2_operationalization_tiers"]
    ax.text(
        0.028,
        0.775,
        "7  ⊂  16  ⊂  153  ⊂  219",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=FONT_SECTION,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        0.028,
        0.715,
        "F source · L surrogate · S structured · X lexical",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=FONT_NOTE,
        color=NAVY,
    )
    labels = {
        "strict_7": "Strict core",
        "fulltext_16": "Primary set",
        "source_154": "Expanded set",
        "ultrarelaxed_221": "Broad T0 set",
    }
    y_positions = [0.600, 0.458, 0.316, 0.174]
    tier_short = {
        "source_formula_existing": "F",
        "source_formula_local_surrogate": "L",
        "structured_construct_proxy": "S",
        "title_taxonomy_lexical_proxy": "X",
    }
    for row, y in zip(sets.itertuples(index=False), y_positions):
        if bool(row.is_primary_scalable):
            face, edge, line, style, label_color = PALE_BLUE, NAVY, 1.25, "solid", NAVY
        elif bool(row.is_strict_core):
            face, edge, line, style, label_color = WHITE, INK, 1.15, "solid", INK
        elif bool(row.is_sensitivity_ceiling):
            face, edge, line, style, label_color = PALE_GREY, MID_GREY, 0.90, "dashed", INK
        else:
            face, edge, line, style, label_color = WHITE, LIGHT_GREY, 0.70, "solid", INK
        inset = 0.018
        row_width = 0.964
        _box(
            ax,
            inset,
            y - 0.050,
            row_width,
            0.100,
            facecolor=face,
            edgecolor=edge,
            linewidth=line,
            linestyle=style,
            rounding=0.002,
        )
        ax.text(
            inset + 0.014,
            y,
            labels[str(row.set_id)],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=FONT_BODY,
            color=label_color,
            fontweight="bold",
        )
        ax.text(
            0.440,
            y,
            f"{int(row.feature_count)} / {int(row.dimension_count)}",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=FONT_BODY,
            color=label_color,
            fontweight="bold",
        )
        tier_data = tiers.loc[tiers["set_id"].eq(row.set_id)].sort_values("tier_order")
        total = max(int(tier_data["feature_count"].sum()), 1)
        cursor = 0.615
        bar_width = 0.300
        composition = " ".join(
            f"{tier_short[str(item.tier)]}{int(item.feature_count)}"
            for item in tier_data.itertuples(index=False)
            if int(item.feature_count) > 0
        )
        ax.text(
            0.615,
            y + 0.024,
            composition,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=FONT_NOTE,
            color=label_color,
            fontweight="bold",
        )
        for item in tier_data.itertuples(index=False):
            value = int(item.feature_count)
            fraction = value / total
            if value <= 0:
                continue
            segment_width = bar_width * fraction
            ax.add_patch(
                Rectangle(
                    (cursor, y - 0.038),
                    segment_width,
                    0.040,
                    transform=ax.transAxes,
                    facecolor=OPERATIONALIZATION_COLORS[str(item.tier)],
                    edgecolor=WHITE,
                    linewidth=0.55,
                    hatch="//" if str(item.tier) == "title_taxonomy_lexical_proxy" else None,
                    zorder=4,
                )
            )
            cursor += segment_width


def _draw_panel_d(ax: plt.Axes, bundle: FigureBundle) -> list[plt.Axes]:
    """Draw the strict-to-scalable nested operating-set ladder."""
    _panel_frame(
        ax,
        "d",
        "Nested outcome-blind sets",
        None,
    )
    _draw_feature_set_rows(ax, bundle)
    return [ax]


def _save_figure(
    fig: plt.Figure,
    path: Path,
    *,
    dpi: int,
    bbox_inches: Bbox | None = None,
) -> None:
    """Save a deterministic raster or vector artifact."""
    kwargs: Dict[str, Any] = {
        "facecolor": WHITE,
        "edgecolor": "none",
        "pad_inches": 0,
    }
    if bbox_inches is None:
        kwargs["bbox_inches"] = Bbox.from_bounds(
            0.0,
            0.0,
            float(fig.get_figwidth()),
            float(fig.get_figheight()),
        )
    else:
        kwargs["bbox_inches"] = bbox_inches
    if path.suffix.lower() == ".png":
        kwargs["dpi"] = int(dpi)
    elif path.suffix.lower() == ".svg":
        kwargs["metadata"] = {"Date": None, "Creator": "ASPR Fig.2 v3 renderer"}
    elif path.suffix.lower() == ".pdf":
        kwargs["metadata"] = {
            "CreationDate": None,
            "ModDate": None,
            "Creator": "ASPR Fig.2 v3 renderer",
        }
    fig.savefig(path, **kwargs)


def _export_panels(
    fig: plt.Figure,
    groups: Mapping[str, Sequence[plt.Axes]],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Export full panel cells as vector or raster reuse assets."""
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outputs: Dict[str, Path] = {}
    for panel, axes in groups.items():
        extents = [axis.get_window_extent(renderer) for axis in axes]
        extent = Bbox.union(extents).transformed(fig.dpi_scale_trans.inverted())
        for extension in formats:
            path = panel_dir / f"fig02_{panel}.{extension}"
            _save_figure(fig, path, dpi=dpi, bbox_inches=extent)
            outputs[f"panel_{panel}_{extension}"] = path
    return outputs


def _cvd_preview(rgb: np.ndarray, *, cvd_type: str) -> np.ndarray:
    """Simulate one complete colour-vision-deficiency condition."""
    space = {"name": "sRGB1+CVD", "cvd_type": cvd_type, "severity": 100}
    return np.clip(cspace_convert(rgb, space, "sRGB1"), 0.0, 1.0)


def _palette_cvd_distance(cvd_type: str) -> float:
    """Return the minimum CVD-space distance for semantic role colours."""
    colors = list(SOURCE_ROLE_COLORS.values()) + list(DIMENSION_ROLE_COLORS.values())
    rgb = np.asarray([mcolors.to_rgb(color) for color in colors], dtype=float)
    transformed = _cvd_preview(rgb, cvd_type=cvd_type)
    distances = [
        float(np.linalg.norm(transformed[left] - transformed[right]))
        for left in range(len(transformed))
        for right in range(left + 1, len(transformed))
    ]
    return min(distances)


def _write_accessibility_qa(
    figure_png: Path,
    output_dir: Path,
    *,
    preview_width: int,
    dpi: int,
    package_versions: Mapping[str, str],
) -> Dict[str, Path]:
    """Write grayscale/CVD previews and a compact visual QA record."""
    qa_dir = output_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(figure_png).convert("RGB")
    original_size = image.size
    height = max(1, round(original_size[1] * preview_width / original_size[0]))
    preview = image.resize((int(preview_width), int(height)), Image.Resampling.LANCZOS)
    outputs: Dict[str, Path] = {}
    lowres = qa_dir / "figure_full_lowres_preview.png"
    preview.save(lowres)
    outputs["qa_lowres_preview"] = lowres
    grayscale = qa_dir / "figure_full_grayscale.png"
    preview.convert("L").save(grayscale)
    outputs["qa_grayscale"] = grayscale
    actual_width_px = round(183.0 / 25.4 * dpi)
    actual_height_px = max(
        1,
        round(original_size[1] * actual_width_px / original_size[0]),
    )
    actual_size_preview = image.resize(
        (actual_width_px, actual_height_px),
        Image.Resampling.LANCZOS,
    )
    actual_size_path = qa_dir / "figure_183mm_preview.png"
    actual_size_preview.save(actual_size_path)
    outputs["qa_183mm_preview"] = actual_size_path
    rgb = np.asarray(preview, dtype=float) / 255.0
    for name, cvd_type in (("deuteranopia", "deuteranomaly"), ("protanopia", "protanomaly")):
        converted = (_cvd_preview(rgb, cvd_type=cvd_type) * 255).round().astype(np.uint8)
        path = qa_dir / f"figure_full_{name}.png"
        Image.fromarray(converted, mode="RGB").save(path)
        outputs[f"qa_{name}"] = path
    record = {
        "source_png": str(figure_png.resolve()),
        "source_size_px": list(original_size),
        "preview_size_px": list(preview.size),
        "preview_183mm_at_render_dpi_px": list(actual_size_preview.size),
        "requested_physical_width_mm": 183.0,
        "render_dpi": int(dpi),
        "role_palette_min_rgb_distance": {
            "deuteranopia": _palette_cvd_distance("deuteranomaly"),
            "protanopia": _palette_cvd_distance("protanomaly"),
        },
        "non_color_encodings": [
            "direct labels and counts",
            "fixed left-to-right process order",
            "hatching for title/taxonomy lexical proxies",
            "strict, scalable and sensitivity borders",
        ],
        "packages": dict(package_versions),
    }
    record_path = qa_dir / "visual_accessibility.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs["qa_record"] = record_path
    return outputs


def _bbox_overlap_ratio(left: Bbox, right: Bbox) -> float:
    """Return overlap area divided by the smaller text bounding-box area."""
    intersection = Bbox.intersection(left, right)
    if intersection is None:
        return 0.0
    intersection_area = max(0.0, intersection.width) * max(0.0, intersection.height)
    denominator = min(max(left.width * left.height, 1.0), max(right.width * right.height, 1.0))
    return float(intersection_area / denominator)


def _write_layout_audit(
    fig: plt.Figure,
    axes: Mapping[str, plt.Axes],
    output_dir: Path,
) -> Dict[str, Path]:
    """Record conservative text-boundary and obvious text-collision checks.

    This intentionally reports, rather than silently suppresses, any possible
    layout issue.  The collision rule is conservative and ignores the normal
    adjacency of small table labels; it flags only overlaps covering at least
    45% of the smaller visible text box.
    """
    qa_dir = output_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panels: Dict[str, Any] = {}
    all_outside: list[Dict[str, Any]] = []
    all_collisions: list[Dict[str, Any]] = []
    tolerance = 1.5
    for panel, axis in axes.items():
        axis_bbox = axis.get_window_extent(renderer)
        texts = [text for text in axis.texts if text.get_visible() and text.get_text().strip()]
        text_boxes: list[tuple[str, Bbox]] = []
        outside: list[Dict[str, Any]] = []
        for text in texts:
            bbox = text.get_window_extent(renderer)
            label = text.get_text().replace("\n", " ")[:110]
            text_boxes.append((label, bbox))
            if (
                bbox.x0 < axis_bbox.x0 - tolerance
                or bbox.x1 > axis_bbox.x1 + tolerance
                or bbox.y0 < axis_bbox.y0 - tolerance
                or bbox.y1 > axis_bbox.y1 + tolerance
            ):
                outside.append({"text": label, "bbox_px": [bbox.x0, bbox.y0, bbox.x1, bbox.y1]})
        collisions: list[Dict[str, Any]] = []
        for left_index, (left_label, left_bbox) in enumerate(text_boxes):
            for right_label, right_bbox in text_boxes[left_index + 1 :]:
                ratio = _bbox_overlap_ratio(left_bbox, right_bbox)
                if ratio >= 0.45:
                    collisions.append(
                        {
                            "left": left_label,
                            "right": right_label,
                            "overlap_ratio_of_smaller_box": round(ratio, 4),
                        }
                    )
        panels[panel] = {
            "visible_text_count": len(texts),
            "outside_panel_text_count": len(outside),
            "obvious_text_collision_count": len(collisions),
            "outside_panel_text": outside,
            "obvious_text_collisions": collisions,
        }
        all_outside.extend({"panel": panel, **item} for item in outside)
        all_collisions.extend({"panel": panel, **item} for item in collisions)
    required_text = {
        "global_title": "Evidence-derived publication-time measurement architecture",
        "panel_a": "Evidence-derived search and measurement pipeline",
        "panel_b": "Indicators precede dimensions",
        "panel_c": "Fixed hard-gate audit",
        "panel_d": "Nested outcome-blind sets",
        "round12": "R12: pragmatic stop",
        "expanded153": "Expanded set",
    }
    rendered_text = "\n".join(
        text.get_text() for axis in axes.values() for text in axis.texts
    ) + "\n" + "\n".join(text.get_text() for text in fig.texts)
    record = {
        "panel_checks": panels,
        "outside_panel_text_count": len(all_outside),
        "obvious_text_collision_count": len(all_collisions),
        "required_text_present": {
            label: phrase in rendered_text for label, phrase in required_text.items()
        },
        "emphasis_contract": {
            "r12_orange_dashed": True,
            "hard_gate_audit_only": True,
            "expanded153_blue_emphasis": True,
            "broad_t0_dashed_outline": True,
            "operationalization_palette_separate_from_role_palette": True,
        },
    }
    record["passes"] = bool(
        not all_outside
        and not all_collisions
        and all(record["required_text_present"].values())
    )
    path = qa_dir / "layout_audit.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"qa_layout_audit": path}


def _enforce_print_typography(fig: plt.Figure) -> None:
    """Apply the publication-body minimum to every rendered text object."""
    for text in fig.findobj(mpl.text.Text):
        if text.get_visible() and text.get_text().strip():
            minimum = FONT_NOTE if text.get_gid() == SUPPORT_NOTE_GID else FONT_BODY
            text.set_fontsize(max(float(text.get_fontsize()), minimum))


def _write_vector_text_audit(
    output_dir: Path,
    *,
    svg_path: Path,
    pdf_path: Path,
) -> Dict[str, Path]:
    """Verify that the two publication-vector outputs retain extractable text.

    The check is deliberately modest: it confirms vector text objects rather
    than claiming that every glyph is editable in every downstream editor.
    Raster images in either vector master are reported explicitly.
    """
    qa_dir = output_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    svg_text_count = 0
    svg_image_count = 0
    svg_font_sizes_px: list[float] = []
    source_svg_width_pt = MASTER_SVG_WIDTH_PT
    if svg_path.is_file():
        root = xml_etree.parse(svg_path).getroot()
        width_attribute = str(root.attrib.get("width", ""))
        width_match = re.fullmatch(r"([0-9.]+)pt", width_attribute)
        if width_match:
            source_svg_width_pt = float(width_match.group(1))
        svg_text_count = sum(
            1
            for element in root.iter()
            if element.tag.rsplit("}", maxsplit=1)[-1] == "text"
        )
        svg_image_count = sum(
            1
            for element in root.iter()
            if element.tag.rsplit("}", maxsplit=1)[-1] == "image"
        )
        for element in root.iter():
            if element.tag.rsplit("}", maxsplit=1)[-1] != "text":
                continue
            style = str(element.attrib.get("style", ""))
            match = re.search(r"font-size:\s*([0-9.]+)px", style)
            if match:
                svg_font_sizes_px.append(float(match.group(1)))
    pdf_pages = 0
    pdf_text_character_count = 0
    if pdf_path.is_file():
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pdf_pages = len(reader.pages)
        pdf_text_character_count = sum(
            len(page.extract_text() or "") for page in reader.pages
        )
    print_scale = (TARGET_PRINT_WIDTH_MM / 25.4 * 72.0) / source_svg_width_pt
    estimated_print_sizes = [size * CSS_PX_TO_PT * print_scale for size in svg_font_sizes_px]
    typography_passes = bool(
        estimated_print_sizes
        and min(estimated_print_sizes) >= 4.95
        and float(np.median(estimated_print_sizes)) >= 5.45
    )
    record = {
        "svg_path": str(svg_path.resolve()),
        "svg_text_elements": int(svg_text_count),
        "svg_raster_image_elements": int(svg_image_count),
        "pdf_path": str(pdf_path.resolve()),
        "pdf_pages": int(pdf_pages),
        "pdf_extractable_text_characters": int(pdf_text_character_count),
        "typography": {
            "target_print_width_mm": TARGET_PRINT_WIDTH_MM,
            "source_svg_width_pt": source_svg_width_pt,
            "minimum_svg_font_px": round(min(svg_font_sizes_px), 3) if svg_font_sizes_px else None,
            "median_svg_font_px": round(float(np.median(svg_font_sizes_px)), 3) if svg_font_sizes_px else None,
            "estimated_minimum_font_pt_at_183mm": round(min(estimated_print_sizes), 3) if estimated_print_sizes else None,
            "estimated_median_font_pt_at_183mm": round(float(np.median(estimated_print_sizes)), 3) if estimated_print_sizes else None,
            "text_below_5pt_at_183mm": int(sum(size < 4.95 for size in estimated_print_sizes)),
            "typography_passes": typography_passes,
        },
        "passes": bool(
            svg_text_count > 0
            and svg_image_count == 0
            and pdf_pages == 1
            and pdf_text_character_count > 0
            and typography_passes
        ),
    }
    path = qa_dir / "vector_text_audit.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"qa_vector_text_audit": path}


def render_fig2_evidence_map(
    bundle: FigureBundle,
    output_dir: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> Dict[str, Path]:
    """Render the asymmetric four-panel Fig.2 and accessibility previews."""
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
        fig.text(
            0.012,
            0.995,
            "Evidence-derived publication-time measurement architecture",
            ha="left",
            va="top",
            fontsize=FONT_FIGURE_TITLE,
            fontweight="bold",
            color=INK,
        )
        grid = fig.add_gridspec(
            3,
            2,
            width_ratios=[1.0, 1.0],
            height_ratios=[1.75, 1.20, 1.00],
            left=0.012,
            right=0.988,
            top=0.955,
            bottom=0.013,
            wspace=0.030,
            hspace=0.040,
        )
        axes = {
            "a": fig.add_subplot(grid[0, :]),
            "b": fig.add_subplot(grid[1, :]),
            "c": fig.add_subplot(grid[2, 0]),
            "d": fig.add_subplot(grid[2, 1]),
        }
        groups = {
            "a": _draw_panel_a(axes["a"], bundle),
            "b": _draw_panel_b(axes["b"], bundle),
            "c": _draw_panel_c(axes["c"], bundle),
            "d": _draw_panel_d(axes["d"], bundle),
        }
        outputs: Dict[str, Path] = {}
        _enforce_print_typography(fig)
        outputs.update(_write_layout_audit(fig, axes, output_dir))
        for extension in formats:
            path = output_dir / f"figure_full.{extension}"
            _save_figure(fig, path, dpi=dpi)
            outputs[f"figure_full_{extension}"] = path
            architecture_path = output_dir / f"Fig2_evidence_architecture.{extension}"
            _save_figure(fig, architecture_path, dpi=dpi)
            outputs[f"architecture_{extension}"] = architecture_path
        outputs.update(_export_panels(fig, groups, output_dir, formats, dpi))
        plt.close(fig)
    png_path = output_dir / "figure_full.png"
    if png_path.is_file():
        outputs.update(
            _write_accessibility_qa(
                png_path,
                output_dir,
                preview_width=int(render_config["qa_preview_width"]),
                dpi=dpi,
                package_versions=package_versions,
            )
        )
    svg_path = output_dir / "Fig2_evidence_architecture.svg"
    pdf_path = output_dir / "Fig2_evidence_architecture.pdf"
    if svg_path.is_file() and pdf_path.is_file():
        outputs.update(
            _write_vector_text_audit(
                output_dir,
                svg_path=svg_path,
                pdf_path=pdf_path,
            )
        )
    return outputs
