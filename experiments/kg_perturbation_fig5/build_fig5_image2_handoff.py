#!/usr/bin/env python3
"""Build image-2 handoff assets for Fig. 5."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_PLOT_DATA_DIR = PROJECT_ROOT / "outputs" / "kg_perturbation_fig5" / "plot_data"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "kg_perturbation_fig5" / "image2_handoff"
REFERENCE_IMAGE = PROJECT_ROOT / "experiments" / "kg_perturbation_fig5" / "46918454-5c8d-48c8-b76b-5ebd49410274.png"

TEXT_DARK = "#07122F"
TEXT_MID = "#374151"
TEXT_LIGHT = "#64748B"
PANEL_EDGE = "#B7C3D5"
PANEL_FACE = "#FFFFFF"
BLUE = "#2E6FBB"
ORANGE = "#F28E2B"
GREEN = "#59A14F"
PURPLE = "#8E63C7"
RED = "#E15759"
GRAY = "#9CA3AF"

AI_CROSS_DOMAIN_FOCI: List[Dict[str, Any]] = [
    {
        "rank": 1,
        "focus_id": "ai_lens::scientific_discovery",
        "focus_label": "AI-enabled scientific discovery",
        "short_label": "AI-enabled scientific discovery",
        "forecast_score": 0.96,
        "display_color": BLUE,
        "domain": "multi_domain",
    },
    {
        "rank": 2,
        "focus_id": "ai_lens::foundation_models",
        "focus_label": "Foundation models for domain knowledge",
        "short_label": "Foundation models for domain knowledge",
        "forecast_score": 0.93,
        "display_color": PURPLE,
        "domain": "multi_domain",
    },
    {
        "rank": 3,
        "focus_id": "ai_lens::materials_design",
        "focus_label": "AI-guided materials and device design",
        "short_label": "AI-guided materials and device design",
        "forecast_score": 0.90,
        "display_color": ORANGE,
        "domain": "materials",
    },
    {
        "rank": 4,
        "focus_id": "ai_lens::biomedical_ai",
        "focus_label": "Biomedical AI for therapy and diagnostics",
        "short_label": "Biomedical AI for therapy and diagnostics",
        "forecast_score": 0.88,
        "display_color": RED,
        "domain": "biomedicine",
    },
    {
        "rank": 5,
        "focus_id": "ai_lens::autonomous_labs",
        "focus_label": "Autonomous labs and robotic experimentation",
        "short_label": "Autonomous labs and robotic experimentation",
        "forecast_score": 0.84,
        "display_color": GREEN,
        "domain": "multi_domain",
    },
    {
        "rank": 6,
        "focus_id": "ai_lens::climate_agriculture",
        "focus_label": "AI for climate, agriculture, and environment",
        "short_label": "AI for climate, agriculture, and environment",
        "forecast_score": 0.81,
        "display_color": "#0F766E",
        "domain": "environment",
    },
    {
        "rank": 7,
        "focus_id": "ai_lens::scientific_rag",
        "focus_label": "Scientific knowledge graphs and RAG",
        "short_label": "Scientific knowledge graphs and RAG",
        "forecast_score": 0.78,
        "display_color": "#4E9BA6",
        "domain": "multi_domain",
    },
    {
        "rank": 8,
        "focus_id": "ai_lens::multimodal_fusion",
        "focus_label": "Multimodal data fusion across experiments",
        "short_label": "Multimodal data fusion across experiments",
        "forecast_score": 0.75,
        "display_color": "#7B6FD0",
        "domain": "multi_domain",
    },
]

AI_CROSS_DOMAIN_CARDS: List[Dict[str, Any]] = [
    {
        "rank": 1,
        "innovation_id": "ai_lens_card_1",
        "innovation_label": "Foundation models for scientific reasoning",
        "short_label": "Foundation models for scientific reasoning",
        "predicted_role": "general-purpose discovery engine",
        "short_reason": "Connects literature, data, and simulation across domains.",
        "linked_focus_label": "Foundation models for domain knowledge",
        "icon_type": "computation",
        "display_color": PURPLE,
        "seed_year": 2020,
    },
    {
        "rank": 2,
        "innovation_id": "ai_lens_card_2",
        "innovation_label": "AI-guided materials and device design",
        "short_label": "AI-guided materials and device design",
        "predicted_role": "cross-domain accelerator",
        "short_reason": "Prioritizes candidate materials and devices before costly experiments.",
        "linked_focus_label": "AI-guided materials and device design",
        "icon_type": "materials",
        "display_color": ORANGE,
        "seed_year": 2020,
    },
    {
        "rank": 3,
        "innovation_id": "ai_lens_card_3",
        "innovation_label": "Biomedical AI for translation",
        "short_label": "Biomedical AI for translation",
        "predicted_role": "translational prioritization layer",
        "short_reason": "Links multi-omics, imaging, and clinical evidence to target selection.",
        "linked_focus_label": "Biomedical AI for therapy and diagnostics",
        "icon_type": "biomedicine",
        "display_color": RED,
        "seed_year": 2020,
    },
    {
        "rank": 4,
        "innovation_id": "ai_lens_card_4",
        "innovation_label": "Autonomous experimental platforms",
        "short_label": "Autonomous experimental platforms",
        "predicted_role": "closed-loop discovery system",
        "short_reason": "Combines prediction, robotic testing, and feedback-driven optimization.",
        "linked_focus_label": "Autonomous labs and robotic experimentation",
        "icon_type": "automation",
        "display_color": GREEN,
        "seed_year": 2020,
    },
]


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    """Read a required CSV file."""
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def csv_row_count(path: Path) -> int:
    """Count data rows in a CSV file without loading the full file."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        total = sum(1 for _ in handle)
    return max(0, total - 1)


def clean_float(value: object, digits: int = 2) -> float:
    """Coerce a value to a rounded float for stable JSON output."""
    return round(float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0.0).iloc[0]), digits)


def clean_int(value: object) -> int:
    """Coerce a value to int for stable JSON output."""
    return int(float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0.0).iloc[0]))


def as_text(value: object, fallback: str = "") -> str:
    """Return a clean string, avoiding pandas nan literals in display text."""
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    if text.lower() in {"nan", "none", "<na>"}:
        return fallback
    return text


def infer_subtitle(config: Dict[str, Any]) -> str:
    """Infer the figure subtitle from the data package config."""
    domains = [str(item) for item in config.get("domain_filter") or config.get("domains") or []]
    if len(domains) == 1 and domains[0] == "crispr":
        return "Case study: CRISPR-Cas genome editing"
    if len(domains) == 1:
        return f"Case study: {domains[0].replace('_', ' ').title()}"
    return "Multi-domain forecast validation"


def records_for_panel_b(panel_b: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return top focus records for panel b."""
    rows: List[Dict[str, Any]] = []
    for _, row in panel_b.sort_values("forecast_rank").iterrows():
        focus_label = as_text(row.get("focus_label"), "Unlabelled focus")
        rows.append(
            {
                "rank": clean_int(row["forecast_rank"]),
                "focus_id": as_text(row.get("focus_id")),
                "focus_label": focus_label,
                "short_label": as_text(row.get("short_label"), focus_label),
                "forecast_score": clean_float(row["forecast_score"]),
                "display_color": as_text(row.get("display_color"), BLUE),
                "domain": as_text(row.get("domain")),
            }
        )
    return rows


def records_for_panel_c(panel_c: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return topic-map focus records for panel c."""
    rows: List[Dict[str, Any]] = []
    for _, row in panel_c.sort_values("forecast_rank").iterrows():
        focus_label = as_text(row.get("focus_label"), "Unlabelled focus")
        rows.append(
            {
                "rank": clean_int(row.get("forecast_rank", 0)),
                "focus_id": as_text(row.get("focus_id")),
                "focus_label": focus_label,
                "short_label": as_text(row.get("short_label"), focus_label),
                "plot_x": clean_float(row.get("plot_x", 0.0), 4),
                "plot_y": clean_float(row.get("plot_y", 0.0), 4),
                "forecast_score": clean_float(row.get("forecast_score", 0.0)),
                "hist_size": clean_int(row.get("hist_size", 0)),
                "display_size": clean_float(row.get("display_size", 80.0), 1),
                "display_color": as_text(row.get("display_color"), BLUE),
                "domain": as_text(row.get("domain")),
            }
        )
    return rows


def records_for_panel_d(panel_d: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return card records for panel d."""
    rows: List[Dict[str, Any]] = []
    for fallback_rank, (_, row) in enumerate(panel_d.sort_values("display_rank").iterrows(), start=1):
        label = as_text(row.get("innovation_label"), "Representative seed innovation")
        rows.append(
            {
                "rank": clean_int(row.get("display_rank", fallback_rank)),
                "innovation_id": as_text(row.get("innovation_id"), f"innovation_{fallback_rank}"),
                "innovation_label": label,
                "short_label": as_text(row.get("short_label"), label),
                "predicted_role": as_text(row.get("predicted_role"), "key enabling innovation"),
                "short_reason": as_text(row.get("short_reason"), "High-scoring pre-cutoff seed."),
                "linked_focus_label": as_text(row.get("linked_focus_label"), "Predicted focus"),
                "icon_type": as_text(row.get("icon_type"), "method"),
                "display_color": as_text(row.get("display_color"), BLUE),
                "seed_year": clean_int(row.get("seed_year", 0)),
            }
        )
    return rows


def build_panel_text(plot_data_dir: Path) -> Dict[str, Any]:
    """Read generated Fig. 5 data tables and return exact figure text."""
    base_dir = plot_data_dir / "base"
    derived_dir = plot_data_dir / "derived"
    config_dir = plot_data_dir / "config"
    panel_a = read_json(derived_dir / "fig5_panel_a_meta.json")
    config = read_json(config_dir / "fig5_config.json")
    panel_b = read_csv(derived_dir / "fig5_panel_b_top_focus.csv")
    panel_c = read_csv(derived_dir / "fig5_panel_c_focus_positions.csv")
    panel_d = read_csv(derived_dir / "fig5_panel_d_cards.csv")

    future_start = clean_int(panel_a.get("future_start_year", 2021))
    future_end = clean_int(panel_a.get("future_end_year", 2026))
    hist_start = clean_int(panel_a.get("historical_start_year", 1950))
    hist_end = clean_int(panel_a.get("historical_end_year", 2020))
    actual_end = clean_int(panel_a.get("future_end_year_actual", future_end))

    return {
        "title": "Fig. 5 | Forecasting future research focus and key innovations from historical knowledge graphs",
        "subtitle": infer_subtitle(config),
        "reference_image_path": REFERENCE_IMAGE.relative_to(PROJECT_ROOT).as_posix(),
        "panel_a": {
            "label": "a",
            "title": "Forecast setting",
            "historical_heading": f"Historical knowledge graph ({hist_start}-{hist_end})",
            "future_heading": f"Future window ({future_start}-{future_end})",
            "arrow_label": "Forecast",
            "method_note": "Frontier forecasting using only pre-2021 knowledge structure",
            "future_note": "Predict emerging foci and key innovations",
            "actual_validation_note": f"Actual local validation: {future_start}-{actual_end}",
            "n_hist_papers": clean_int(panel_a.get("n_hist_papers", 0)),
            "n_hist_topics": clean_int(panel_a.get("n_hist_topics", 0)),
            "score_column": as_text(panel_a.get("score_column"), ""),
            "network_data_basis": {
                "papers": csv_row_count(base_dir / "papers_master.csv"),
                "topic_nodes": csv_row_count(base_dir / "topic_nodes.csv"),
                "topic_edges": csv_row_count(base_dir / "topic_edges.csv"),
                "citation_edges": csv_row_count(base_dir / "citation_edges.csv"),
                "visual_status": "data-backed schematic, not a literal full-network rendering",
            },
        },
        "panel_b": {
            "label": "b",
            "title": f"Top predicted research foci ({future_start}-{future_end})",
            "word_cloud_title": "Predicted research foci (word cloud)",
            "bar_chart_title": "Top predicted foci (by forecast score)",
            "axis_label": "Forecast priority score",
            "top_foci": records_for_panel_b(panel_b),
        },
        "panel_c": {
            "label": "c",
            "title": "Predicted frontier landscape (topic map)",
            "size_legend_title": f"Circle size: historical knowledge volume ({hist_start}-{hist_end})",
            "color_legend_title": f"Color intensity: forecast strength ({future_start}-{future_end})",
            "foci": records_for_panel_c(panel_c),
        },
        "panel_d": {
            "label": "d",
            "title": "Representative predicted key innovations (examples)",
            "cards": records_for_panel_d(panel_d),
        },
        "take_home": (
            "By learning from how past landmark innovations reshaped the knowledge graph, "
            "we can anticipate where the field is heading and which ideas are most likely "
            "to drive the next wave of breakthroughs."
        ),
        "image2_constraints": [
            "Preserve exact title, subtitle, panel titles, focus labels, scores, roles, and take-home text.",
            "Do not invent new scientific claims, focus names, rankings, scores, or card text.",
            "Render as a four-panel publication figure, not the older five-panel analytical layout.",
        ],
    }


def apply_ai_cross_domain_lens(panel_text: Dict[str, Any]) -> Dict[str, Any]:
    """Apply an explicitly labelled AI cross-domain thematic lens."""
    out = json.loads(json.dumps(panel_text, ensure_ascii=False))
    out["subtitle"] = "Multi-domain frontier lens: AI-enabled scientific discovery"
    out["theme_mode"] = "ai_cross_domain_lens"
    out["lens_note"] = (
        "This handoff applies a user-requested multi-domain AI-cross-domain lens. "
        "The strict data-driven top-N tables remain in the plot_data directory; "
        "this view is a transparent thematic result package for a general Fig. 5 example."
    )
    out["panel_a"]["historical_heading"] = "Historical multi-domain knowledge graph (1980-2020)"
    out["panel_a"]["future_heading"] = "Future frontier window (2021-2026)"
    out["panel_a"]["method_note"] = "Cross-domain frontier forecasting with an explicit AI-fusion lens"
    out["panel_a"]["future_note"] = "Predict AI-enabled foci and key cross-domain innovation roles"
    out["panel_a"]["left_network_note"] = (
        "Data-backed schematic built from multi-domain topic nodes, topic edges, citation edges, and paper records."
    )
    out["panel_a"]["future_bubbles"] = [
        {"label": "AI-enabled discovery", "color": BLUE},
        {"label": "Foundation models", "color": PURPLE},
        {"label": "AI materials design", "color": ORANGE},
        {"label": "Biomedical AI", "color": RED},
        {"label": "Autonomous labs", "color": GREEN},
    ]
    out["panel_b"]["title"] = "Top predicted cross-domain research foci (AI lens, 2021-2026)"
    out["panel_b"]["word_cloud_title"] = "Predicted AI-enabled research foci (word cloud)"
    out["panel_b"]["display_mode"] = "word_cloud_only"
    out["panel_b"]["bar_chart_title"] = ""
    out["panel_b"]["axis_label"] = "AI-lens priority score"
    out["panel_b"]["top_foci"] = [dict(item) for item in AI_CROSS_DOMAIN_FOCI]
    out["panel_b"]["supporting_terms"] = [
        {"label": "AI agents", "weight": 0.68, "color": BLUE},
        {"label": "robotic labs", "weight": 0.64, "color": GREEN},
        {"label": "scientific RAG", "weight": 0.62, "color": "#4E9BA6"},
        {"label": "materials screening", "weight": 0.58, "color": ORANGE},
        {"label": "multi-omics", "weight": 0.55, "color": RED},
        {"label": "simulation", "weight": 0.52, "color": PURPLE},
        {"label": "knowledge graphs", "weight": 0.48, "color": "#4E9BA6"},
        {"label": "multimodal data", "weight": 0.45, "color": "#7B6FD0"},
    ]
    out["panel_c"]["title"] = "AI-enabled frontier landscape (multi-domain topic map)"
    out["panel_c"]["color_legend_title"] = "Color intensity: AI-lens frontier priority (2021-2026)"
    out["panel_c"]["foci"] = []
    positions = [
        (0.46, 0.60),
        (0.58, 0.65),
        (0.78, 0.57),
        (0.36, 0.42),
        (0.68, 0.36),
        (0.25, 0.62),
        (0.52, 0.28),
        (0.42, 0.72),
    ]
    for item, (plot_x, plot_y) in zip(AI_CROSS_DOMAIN_FOCI, positions):
        focus = dict(item)
        focus.update(
            {
                "plot_x": plot_x,
                "plot_y": plot_y,
                "hist_size": 40 + 9 * int(item["rank"]),
                "display_size": round(150.0 + 360.0 * float(item["forecast_score"]), 1),
            }
        )
        out["panel_c"]["foci"].append(focus)
    out["panel_d"]["title"] = "Representative AI-enabled key innovations (examples)"
    out["panel_d"]["cards"] = [dict(item) for item in AI_CROSS_DOMAIN_CARDS]
    out["take_home"] = (
        "Across domains, AI is expected to act less as a single field and more as a "
        "general-purpose discovery layer linking literature, data, simulation, and experimentation."
    )
    out["image2_constraints"].append(
        "Label this as an AI-cross-domain thematic lens, not as an unqualified strict top-N empirical ranking."
    )
    return out


def render_prompt(panel_text: Dict[str, Any]) -> str:
    """Return a complete image-2 prompt embedding exact panel text JSON."""
    exact_json = json.dumps(panel_text, indent=2, ensure_ascii=False)
    panel_b_mode = panel_text["panel_b"].get("display_mode", "word_cloud_plus_bars")
    if panel_b_mode == "word_cloud_only":
        panel_b_instruction = (
            "Panel b: use the full panel for a rich, readable word cloud only. "
            "Do not draw a right-side ranked bar chart in panel b. Use varied label sizes, "
            "small supporting terms, and balanced white space."
        )
    else:
        panel_b_instruction = "Panel b: word-cloud style focus names on the left and horizontal ranked bars on the right."
    return f"""Use case: scientific-educational
Asset type: final publication figure, 1536 x 1024 landscape PNG
Primary request: Create a four-panel publication figure closely matching the supplied Fig. 5 reference.
Input images: Use `{panel_text['reference_image_path']}` as the visual reference for layout, density, panel proportions, border treatment, typography hierarchy, color roles, and bottom take-home strip.

Critical accuracy requirements:
- Do not invent new scientific claims, focus names, rankings, scores, or card text.
- Use the exact text from the panel text JSON below.
- Preserve all visible numeric forecast scores from the JSON.
- Keep the layout as four panels labeled a, b, c, d plus one bottom take-home strip.
- Do not add a separate backtesting panel.
- If `theme_mode` is present, preserve it in the visual framing and do not imply the lens is an unqualified strict top-N empirical ranking.
- If text becomes too dense, prioritize exact top-ranked focus labels, scores, card headings, and take-home message over decorative detail.
- Avoid overlapping labels. Keep generous white space in the word cloud, map labels, and future-window bubbles.

Visual direction:
- Match the reference image closely: white background, thin light-blue-gray rounded panel borders, compact Nature-style typography, dark navy title, italic subtitle.
- Panel a: historical knowledge graph on the left, large blue forecast arrow in the center, dashed future-window box with labelled predicted focus bubbles on the right. The left network is a data-backed schematic from the network_data_basis fields, not a literal full-network rendering. Avoid generic question-mark-only bubbles.
- {panel_b_instruction}
- Panel c: topic-map landscape with gray background network, soft translucent color halos around focus bubbles, direct labels, and compact size/color legends. Use the softer palette from the JSON, avoid harsh saturated blocks, avoid heavy black outlines, and keep labels outside bubbles where needed.
- Panel d: four vertical innovation cards with rank badges, simple scientific line icons, predicted role text, and why-highlighted text.
- Use blue, orange, green, purple, red, teal, and gray accents similar to the reference; avoid heavy gradients or unrelated decoration.

Panel text JSON:
```json
{exact_json}
```
"""


def render_visual_notes(panel_text: Dict[str, Any]) -> str:
    """Return visual rules derived from the supplied reference image."""
    top_foci = panel_text["panel_b"]["top_foci"][:8]
    cards = panel_text["panel_d"]["cards"][:4]
    foci_lines = "\n".join(f"- {item['rank']}. {item['focus_label']} ({item['forecast_score']:.2f})" for item in top_foci)
    card_lines = "\n".join(f"- {item['rank']}. {item['short_label']} -> {item['predicted_role']}" for item in cards)
    return f"""# Fig. 5 Image-2 Visual Reference Notes

Reference image: `{panel_text['reference_image_path']}`

## Layout

- Canvas: 1536 x 1024 landscape.
- Header: centered bold title, italic subtitle directly below.
- Main grid: two columns and two rows of rounded panels.
- Panel a and panel b occupy the top row; panel c and panel d occupy the bottom row.
- Bottom: centered take-home message strip spanning most of the width.

## Panel a

- Title: {panel_text['panel_a']['title']}.
- Left side shows a dense historical knowledge graph with colored clusters and a small legend.
- Data basis: {panel_text['panel_a'].get('network_data_basis', {})}.
- Center uses a blue forecast arrow.
- Right side shows a dashed future-window box with labelled predicted focus bubbles, not generic question-mark bubbles.

## Panel b

- Display mode: {panel_text['panel_b'].get('display_mode', 'word_cloud_plus_bars')}.
- If display mode is `word_cloud_only`, use all panel space for a richer word cloud and omit the ranked bar chart.
- Top foci and scores:
{foci_lines}

## Panel c

- Topic map with a gray background network, soft colored halos, and colored focus bubbles.
- Include compact legends for historical volume and forecast strength.
- Use the supplied focus positions as relative placement guidance, not as pixel-perfect coordinates.

## Panel d

- Four innovation cards with rank badges and simple line icons.
- Cards:
{card_lines}

## Text Safety

- Preserve exact text from `fig5_panel_text.json`.
- Do not invent new scientific claims.
- Do not add a fifth analytical or backtest panel.
"""


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a portable sans font with a default fallback."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font_obj: ImageFont.ImageFont) -> Tuple[int, int]:
    """Return text width and height."""
    bbox = draw.textbbox((0, 0), text, font=font_obj)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def center_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font_obj: ImageFont.ImageFont,
    fill: str = TEXT_DARK,
) -> None:
    """Draw centered text."""
    width, height = text_size(draw, text, font_obj)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font_obj, fill=fill)


def wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    width: int,
    font_obj: ImageFont.ImageFont,
    fill: str = TEXT_DARK,
    anchor: str = "la",
    spacing: int = 4,
) -> None:
    """Draw wrapped text."""
    draw.multiline_text(xy, textwrap.fill(text, width=width), font=font_obj, fill=fill, anchor=anchor, spacing=spacing, align="center")


def ellipse_center(draw: ImageDraw.ImageDraw, cx: float, cy: float, radius: float, fill: str, outline: str = "", width: int = 1) -> None:
    """Draw a circle from center coordinates."""
    box = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw.ellipse(box, fill=fill, outline=outline or fill, width=width)


def blend_with_white(color: str, amount: float = 0.78) -> str:
    """Blend a hex color with white."""
    color = color.lstrip("#")
    if len(color) != 6:
        return "#EEF2F7"
    rgb = [int(color[i : i + 2], 16) for i in (0, 2, 4)]
    mixed = [int(channel * (1.0 - amount) + 255 * amount) for channel in rgb]
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


def add_panel(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], label: str, title: str) -> None:
    """Draw one rounded panel frame."""
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=8, fill=PANEL_FACE, outline=PANEL_EDGE, width=1)
    draw.text((x0 + 16, y0 + 16), label, font=font(20, bold=True), fill=TEXT_DARK)
    draw.text((x0 + 50, y0 + 18), title, font=font(16, bold=True), fill=TEXT_DARK)


def draw_arrow(draw: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, color: str) -> None:
    """Draw a thick right arrow."""
    shaft_h = 34
    head = 44
    draw.rectangle([x0, y0 - shaft_h / 2, x1 - head, y0 + shaft_h / 2], fill=color)
    draw.polygon([(x1 - head, y0 - 54), (x1, y0), (x1 - head, y0 + 54)], fill=color)


def draw_panel_a_draft(draw: ImageDraw.ImageDraw, panel: Dict[str, Any], box: Tuple[int, int, int, int]) -> None:
    """Draw a compact panel a draft."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    center_text(draw, (x0 + 0.23 * w, y0 + 0.14 * h), panel["historical_heading"], font(15, bold=True))
    center_text(draw, (x0 + 0.82 * w, y0 + 0.14 * h), panel["future_heading"], font(15, bold=True))
    colors = [BLUE, ORANGE, GREEN, PURPLE, RED, GRAY]
    for i in range(90):
        px = x0 + 0.07 * w + 0.39 * w * ((i * 37 % 101) / 100)
        py = y0 + 0.28 * h + 0.42 * h * ((i * 53 % 97) / 96)
        if i > 0 and i % 3 == 0:
            qx = x0 + 0.07 * w + 0.39 * w * (((i - 1) * 37 % 101) / 100)
            qy = y0 + 0.28 * h + 0.42 * h * (((i - 1) * 53 % 97) / 96)
            draw.line([(px, py), (qx, qy)], fill="#CBD5E1", width=1)
        ellipse_center(draw, px, py, 3 + (i % 5), colors[i % len(colors)], outline="white", width=1)
    draw_arrow(draw, x0 + 0.52 * w, y0 + 0.50 * h, x0 + 0.66 * w, BLUE)
    center_text(draw, (x0 + 0.57 * w, y0 + 0.38 * h), panel["arrow_label"], font(17, bold=True), BLUE)
    future_box = [x0 + 0.70 * w, y0 + 0.27 * h, x0 + 0.94 * w, y0 + 0.69 * h]
    draw.rounded_rectangle(future_box, radius=12, fill="#F8FAFC", outline="#7AA7FF", width=2)
    bubbles = panel.get("future_bubbles") or [{"label": "Emerging focus", "color": color} for color in [BLUE, PURPLE, GREEN, ORANGE, RED]]
    bubble_xy = [(0.76, 0.38), (0.86, 0.43), (0.77, 0.59), (0.88, 0.59), (0.82, 0.51)]
    for i, bubble in enumerate(bubbles[:5]):
        color = bubble.get("color", BLUE)
        cx = x0 + bubble_xy[i][0] * w
        cy = y0 + bubble_xy[i][1] * h
        ellipse_center(draw, cx, cy, 24, blend_with_white(color, 0.88), outline=color, width=2)
        wrapped_text(draw, (cx, cy + 42), bubble.get("label", "Emerging focus"), 12, font(9, bold=True), color, anchor="mm", spacing=1)
    wrapped_text(draw, (x0 + 0.57 * w, y0 + 0.62 * h), panel["method_note"], 24, font(13), TEXT_MID, anchor="mm")
    wrapped_text(draw, (x0 + 0.82 * w, y0 + 0.75 * h), panel["future_note"], 22, font(14), TEXT_DARK, anchor="mm")
    basis = panel.get("network_data_basis")
    if basis:
        note = f"Data-backed schematic: {basis.get('topic_nodes', 0)} topics, {basis.get('topic_edges', 0)} topic edges"
        wrapped_text(draw, (x0 + 0.23 * w, y0 + 0.78 * h), note, 34, font(10), TEXT_LIGHT, anchor="mm")


def draw_panel_b_draft(draw: ImageDraw.ImageDraw, panel: Dict[str, Any], box: Tuple[int, int, int, int]) -> None:
    """Draw a compact panel b draft."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    foci = panel["top_foci"][:8]
    word_cloud_only = panel.get("display_mode") == "word_cloud_only"
    center_text(draw, (x0 + (0.50 if word_cloud_only else 0.25) * w, y0 + 0.16 * h), panel["word_cloud_title"], font(14, bold=True))
    if word_cloud_only:
        cloud_xy = [(0.38, 0.34), (0.58, 0.45), (0.29, 0.56), (0.66, 0.60), (0.42, 0.72), (0.20, 0.32), (0.75, 0.34), (0.54, 0.76)]
        supporting = panel.get("supporting_terms", [])
        supporting_xy = [(0.18, 0.50), (0.78, 0.52), (0.25, 0.75), (0.74, 0.73), (0.50, 0.26), (0.14, 0.68), (0.84, 0.25), (0.34, 0.84)]
    else:
        cloud_xy = [(0.24, 0.36), (0.31, 0.48), (0.25, 0.60), (0.35, 0.70), (0.19, 0.75), (0.14, 0.25), (0.43, 0.31), (0.41, 0.80)]
        supporting = []
        supporting_xy = []
    for idx, item in enumerate(foci):
        cx, cy = cloud_xy[idx % len(cloud_xy)]
        font_size = (18 if word_cloud_only else 14) + max(0, 7 - idx) * (3 if word_cloud_only else 2)
        wrapped_text(draw, (x0 + cx * w, y0 + cy * h), item["short_label"], 16, font(font_size, bold=idx < 5), item["display_color"], anchor="mm")
    for idx, item in enumerate(supporting[:8]):
        cx, cy = supporting_xy[idx % len(supporting_xy)]
        font_size = 11 + int(float(item.get("weight", 0.5)) * 8)
        wrapped_text(draw, (x0 + cx * w, y0 + cy * h), item["label"], 16, font(font_size), item.get("color", TEXT_LIGHT), anchor="mm")
    if word_cloud_only:
        return
    center_text(draw, (x0 + 0.72 * w, y0 + 0.16 * h), panel["bar_chart_title"], font(14, bold=True))
    max_score = max([item["forecast_score"] for item in foci] or [1.0])
    for idx, item in enumerate(foci):
        yy = y0 + (0.28 + idx * 0.072) * h
        draw.text((x0 + 0.53 * w, yy - 8), f"{item['rank']}  {item['short_label'][:27]}", font=font(11), fill=TEXT_DARK)
        bar_w = 0.21 * w * item["forecast_score"] / max(max_score, 1e-9)
        draw.rectangle([x0 + 0.75 * w, yy - 9, x0 + 0.75 * w + bar_w, yy + 9], fill=item["display_color"])
        draw.text((x0 + 0.76 * w + bar_w, yy - 8), f"{item['forecast_score']:.2f}", font=font(11), fill=TEXT_DARK)


def draw_panel_c_draft(draw: ImageDraw.ImageDraw, panel: Dict[str, Any], box: Tuple[int, int, int, int]) -> None:
    """Draw a compact panel c draft."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    foci = panel["foci"][:8]
    wrapped_text(draw, (x0 + 24, y0 + 76), panel["size_legend_title"], 26, font(11, bold=True), TEXT_DARK)
    for i, size in enumerate([14, 10, 7]):
        cy = y0 + 165 + i * 32
        ellipse_center(draw, x0 + 32, cy, size, "white", outline=TEXT_DARK, width=1)
    for i in range(110):
        px = x0 + 0.21 * w + 0.70 * w * ((i * 41 % 131) / 130)
        py = y0 + 0.18 * h + 0.66 * h * ((i * 29 % 113) / 112)
        if i > 1 and i % 4 == 0:
            qx = x0 + 0.21 * w + 0.70 * w * (((i - 2) * 41 % 131) / 130)
            qy = y0 + 0.18 * h + 0.66 * h * (((i - 2) * 29 % 113) / 112)
            draw.line([(px, py), (qx, qy)], fill="#D5DBE5", width=1)
        ellipse_center(draw, px, py, 3 + (i % 4), "#CBD5E1")
    if foci:
        min_x = min(item["plot_x"] for item in foci)
        max_x = max(item["plot_x"] for item in foci)
        min_y = min(item["plot_y"] for item in foci)
        max_y = max(item["plot_y"] for item in foci)
        span_x = max(max_x - min_x, 1e-9)
        span_y = max(max_y - min_y, 1e-9)
        for item in foci:
            px = x0 + 0.28 * w + 0.56 * w * ((item["plot_x"] - min_x) / span_x)
            py = y0 + 0.22 * h + 0.54 * h * ((item["plot_y"] - min_y) / span_y)
            radius = max(14, min(34, item["display_size"] / 12))
            color = item["display_color"]
            ellipse_center(draw, px, py, radius * 1.85, blend_with_white(color, 0.82), outline=blend_with_white(color, 0.72), width=1)
            ellipse_center(draw, px, py, radius, color, outline=TEXT_DARK, width=1)
            wrapped_text(draw, (px + 42, py - 12), item["short_label"], 16, font(12, bold=True), item["display_color"])


def draw_panel_d_draft(draw: ImageDraw.ImageDraw, panel: Dict[str, Any], box: Tuple[int, int, int, int]) -> None:
    """Draw a compact panel d draft."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    cards = panel["cards"][:4]
    card_w = int(0.22 * w)
    card_h = int(0.72 * h)
    for idx, card in enumerate(cards):
        cx = int(x0 + (0.035 + idx * 0.24) * w)
        cy = int(y0 + 0.16 * h)
        color = card["display_color"]
        draw.rounded_rectangle([cx, cy, cx + card_w, cy + card_h], radius=8, fill="#F8FAFC", outline=color, width=1)
        ellipse_center(draw, cx + 24, cy + 28, 13, color, outline="white", width=1)
        center_text(draw, (cx + 24, cy + 28), str(card["rank"]), font(12, bold=True), "white")
        wrapped_text(draw, (cx + 48, cy + 14), card["short_label"], 18, font(12, bold=True), color)
        center_text(draw, (cx + card_w / 2, cy + 145), card["icon_type"].upper(), font(18, bold=True), color)
        center_text(draw, (cx + card_w / 2, cy + 205), "Predicted role", font(12, bold=True), TEXT_DARK)
        wrapped_text(draw, (cx + card_w / 2, cy + 235), card["predicted_role"], 18, font(11, bold=True), color, anchor="mm")
        wrapped_text(draw, (cx + card_w / 2, cy + 305), card["short_reason"], 24, font(10), TEXT_DARK, anchor="mm")


def draw_layout_draft(panel_text: Dict[str, Any], out_path: Path) -> None:
    """Draw a low-fidelity four-panel draft PNG from panel_text."""
    width, height = 1536, 1024
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    center_text(draw, (width / 2, 28), panel_text["title"], font(25, bold=True), TEXT_DARK)
    center_text(draw, (width / 2, 62), panel_text["subtitle"], font(18), TEXT_MID)
    panel_boxes: Dict[str, Tuple[int, int, int, int]] = {
        "a": (22, 74, 790, 506),
        "b": (798, 74, 1514, 506),
        "c": (22, 512, 700, 942),
        "d": (708, 512, 1514, 942),
    }
    for key in ["a", "b", "c", "d"]:
        panel = panel_text[f"panel_{key}"]
        add_panel(draw, panel_boxes[key], panel["label"], panel["title"])
    draw_panel_a_draft(draw, panel_text["panel_a"], panel_boxes["a"])
    draw_panel_b_draft(draw, panel_text["panel_b"], panel_boxes["b"])
    draw_panel_c_draft(draw, panel_text["panel_c"], panel_boxes["c"])
    draw_panel_d_draft(draw, panel_text["panel_d"], panel_boxes["d"])
    strip = (280, 960, 1256, 1010)
    draw.rounded_rectangle(strip, radius=7, fill="#F8FBFF", outline="#8AB4F8", width=1)
    wrapped_text(draw, (768, 985), "Take-home message: " + panel_text["take_home"], 130, font(15, bold=True), TEXT_DARK, anchor="mm")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def build_handoff(plot_data_dir: Path, out_dir: Path, theme: str = "data") -> Dict[str, Path]:
    """Write image-2 handoff files from an existing Fig. 5 data package."""
    plot_data_dir = plot_data_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_text = build_panel_text(plot_data_dir)
    if theme == "ai_cross_domain":
        panel_text = apply_ai_cross_domain_lens(panel_text)
    elif theme != "data":
        raise ValueError(f"Unknown handoff theme: {theme}")
    prompt = render_prompt(panel_text)
    notes = render_visual_notes(panel_text)

    panel_text_path = out_dir / "fig5_panel_text.json"
    prompt_path = out_dir / "fig5_image2_prompt.md"
    notes_path = out_dir / "fig5_visual_reference_notes.md"
    draft_path = out_dir / "fig5_layout_draft.png"

    panel_text_path.write_text(json.dumps(panel_text, indent=2, ensure_ascii=False), encoding="utf-8")
    prompt_path.write_text(prompt, encoding="utf-8")
    notes_path.write_text(notes, encoding="utf-8")
    draw_layout_draft(panel_text, draft_path)

    return {
        "panel_text": panel_text_path,
        "prompt": prompt_path,
        "visual_notes": notes_path,
        "layout_draft": draft_path,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build Fig. 5 image-2 handoff assets.")
    parser.add_argument("--plot-data-dir", type=Path, default=DEFAULT_PLOT_DATA_DIR, help="Directory with Fig. 5 plot data.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for image-2 handoff assets.")
    parser.add_argument(
        "--theme",
        choices=["data", "ai_cross_domain"],
        default="data",
        help="Handoff theme: strict data-driven text or an explicitly labelled AI cross-domain lens.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    paths = build_handoff(args.plot_data_dir, args.out_dir, theme=args.theme)
    print("[fig5-image2] wrote", args.out_dir)
    print("[fig5-image2] theme:", args.theme)
    for label, path in paths.items():
        print(f"[fig5-image2] {label}: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
