#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 5 forecast-outcome validation from historical knowledge graphs.

The script turns the paper-level Fig. 3 score table into a result-oriented
forecasting figure:

1. aggregate pre-cutoff publication-day scores to future focus forecasts;
2. aggregate post-cutoff publication growth, impact, and graph perturbation to
   realized focus outcomes;
3. compare predicted and realized top focus lists;
4. choose representative pre-cutoff seed-paper cases;
5. run historical backtests over multiple cutoff windows.

It is intentionally a plotting and audit layer. For a fully strict forecasting
claim, provide a score table whose weights were learned using only information
available before each cutoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.env import load_env

load_env()
os.environ.setdefault("MPLCONFIGDIR", "/tmp/aspr_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch


DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "data" / "knowledge_corpus" / "v1_strict"
DEFAULT_CORPUS_FIG5_INPUT_DIR = DEFAULT_CORPUS_ROOT / "views" / "fig5" / "multi_domain"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fig05/old"
DEFAULT_BACKTEST_WINDOWS = [
    "1995:2000",
    "2000:2005",
    "2005:2010",
    "2010:2015",
    "2015:2020",
    "2020:2026",
]

TEXT_DARK = "#111827"
TEXT_MID = "#374151"
TEXT_LIGHT = "#6B7280"
BORDER = "#9CA3AF"
GRID = "#D1D5DB"
PANEL_FACE = "#FFFFFF"
PREDICTED_BLUE = "#2563EB"
REALIZED_GREEN = "#059669"
HIT_BLUE = "#0B4FA3"
MISSED_GRAY = "#9CA3AF"
LANDMARK_RED = "#DC2626"
UNEXPECTED_GREEN = "#10B981"
PANEL_BORDER = "#B8C2D1"
DOMAIN_COLORS = {
    "crispr": "#2563EB",
    "graphene_2d_materials": "#F97316",
    "ipsc_reprogramming": "#0F766E",
    "transformer_foundation_models": "#7C3AED",
}
FALLBACK_COLORS = ["#2563EB", "#8B5CF6", "#F97316", "#0F766E", "#DC2626", "#94A3B8"]
DOMAIN_LAYOUT_CENTERS = {
    "crispr": (0.64, 0.62),
    "graphene_2d_materials": (0.36, 0.62),
    "ipsc_reprogramming": (0.48, 0.38),
    "transformer_foundation_models": (0.72, 0.38),
}

SCORE_CANDIDATES = ("S_w_oof", "S_w", "S_equal")
RGPM_CANDIDATES = (
    "RGPM_structural_residual_tau10",
    "RGPM",
    "RGPM_v3_balanced",
    "RGPM_v2",
    "RGPM_simple",
)
STOPWORDS = {
    "and",
    "or",
    "of",
    "in",
    "for",
    "the",
    "a",
    "an",
    "to",
    "with",
    "using",
    "based",
    "research",
    "studies",
    "study",
    "applications",
    "application",
}


@dataclass
class LoadedData:
    papers: pd.DataFrame
    topics: pd.DataFrame
    score_col: str
    rgpm_col: Optional[str]
    min_year: int
    max_year: int
    warnings: List[str]


@dataclass
class Fig5Tables:
    focus: pd.DataFrame
    predicted_focus: pd.DataFrame
    realized_focus: pd.DataFrame
    alignment: pd.DataFrame
    key_innovations: pd.DataFrame
    backtest: pd.DataFrame
    summary: Dict[str, Any]


def setup_style() -> None:
    """Configure a compact Nature-style matplotlib theme."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.linewidth": 0.7,
            "axes.edgecolor": BORDER,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.color": TEXT_DARK,
        }
    )


def progress_log(message: str, enabled: bool = True) -> None:
    """Print one prefixed progress message."""
    if enabled:
        print(f"[fig5] {message}", flush=True)


def blend_with_white(color: str, amount: float = 0.86) -> str:
    """Blend a color with white."""
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    out = rgb * (1.0 - amount) + np.ones(3) * amount
    return mcolors.to_hex(out)


def wrap_text(text: object, width: int) -> str:
    """Wrap text without breaking long words."""
    if text is None:
        return ""
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def ellipsize(text: object, max_chars: int) -> str:
    """Truncate long display text with an ellipsis."""
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)].rstrip() + "..."


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    facecolor: str = "white",
    edgecolor: str = BORDER,
    linewidth: float = 0.8,
    radius: float = 0.018,
    linestyle: str = "-",
    alpha: float = 1.0,
    zorder: int = 1,
) -> mpatches.FancyBboxPatch:
    """Draw a rounded box in axes coordinates."""
    patch = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def panel_frame(ax: plt.Axes, label: str, title: str) -> None:
    """Create a framed panel with label and title."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded_box(ax, 0.0, 0.0, 1.0, 1.0, PANEL_FACE, PANEL_BORDER, 0.75, 0.020, zorder=0)
    ax.text(0.022, 0.965, label, ha="left", va="top", fontsize=15, fontweight="bold")
    ax.text(0.070, 0.955, title, ha="left", va="top", fontsize=10.2, fontweight="bold")


def draw_arrow(
    ax: plt.Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: str = "#4B5563",
    lw: float = 1.1,
    mutation_scale: float = 12.0,
    linestyle: str = "-",
    connectionstyle: str = "arc3,rad=0.0",
    zorder: int = 4,
) -> None:
    """Draw an arrow in axes coordinates."""
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=1,
        shrinkB=1,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def draw_pill(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    width: float,
    height: float = 0.052,
    fontsize: float = 6.7,
    zorder: int = 4,
) -> None:
    """Draw a small labelled pill."""
    rounded_box(ax, x, y, width, height, blend_with_white(color, 0.88), color, 0.8, height * 0.45, zorder=zorder)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        fontweight="bold",
        zorder=zorder + 1,
    )


def stable_float(value: object) -> float:
    """Return a deterministic pseudo-random float in [0, 1)."""
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def robust_zscore(values: pd.Series) -> pd.Series:
    """Robustly standardize a numeric series, returning zeros for constants."""
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric.dropna()
    out = pd.Series(np.zeros(len(values), dtype=float), index=values.index)
    if len(valid) < 2:
        return out
    median = float(valid.median())
    mad = float((valid - median).abs().median())
    scale = 1.4826 * mad if mad > 1e-9 else float(valid.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-9:
        return out
    out.loc[valid.index] = ((valid - median) / scale).clip(-4.0, 4.0)
    return out.fillna(0.0)


def percentile_score(values: pd.Series | np.ndarray) -> pd.Series:
    """Convert a numeric vector to percentile scores in [0, 1]."""
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric.dropna()
    out = pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    if valid.empty:
        return out
    if float(valid.max()) == float(valid.min()):
        out.loc[valid.index] = 0.5
        return out
    out.loc[valid.index] = valid.rank(method="average", pct=True)
    return out.fillna(0.0)


def top_tail_mean(values: pd.Series, frac: float = 0.20) -> float:
    """Mean of the highest-scoring tail of a series."""
    valid = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return 0.0
    n = max(1, int(math.ceil(len(valid) * frac)))
    return float(valid.sort_values(ascending=False).head(n).mean())


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    """Return the first existing path from a sequence."""
    for path in paths:
        if path.exists():
            return path
    return None


def default_fig3_root() -> Path:
    """Resolve the most recent local Fig. 3 root that contains outputs."""
    base = PROJECT_ROOT / "outputs" / "fig03/old"
    candidates = [
        base / "strong_evidence_tau10_v3",
        base / "strong_evidence_tau10_v2",
        base / "strong_evidence_tau10",
        base,
    ]
    path = first_existing(candidates)
    return path if path is not None else candidates[0]


def default_fig5_input_dir(fig3_root: Path) -> Path:
    """Prefer the unified corpus Fig. 5 view, then fall back to the Fig. 3 run input."""
    if DEFAULT_CORPUS_FIG5_INPUT_DIR.exists():
        return DEFAULT_CORPUS_FIG5_INPUT_DIR
    return fig3_root / "fig3_input" / "multi_domain"


def choose_column(df: pd.DataFrame, candidates: Sequence[str], name: str, required: bool = True) -> Optional[str]:
    """Pick the first available column from a candidate list."""
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"Could not find a {name} column. Tried: {', '.join(candidates)}")
    return None


def read_csv_required(path: Path, name: str) -> pd.DataFrame:
    """Read a required CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return pd.read_csv(path, low_memory=False)


def clean_label(label: object, domain: Optional[str] = None) -> str:
    """Clean a focus label for display."""
    text = str(label or "").strip()
    if domain:
        text = re.sub(rf"^{re.escape(str(domain))}\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[a-z0-9_ -]+\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text or "Unlabelled focus"


def token_set(label: object) -> set[str]:
    """Tokenize a label for semantic matching."""
    tokens = re.findall(r"[a-z0-9]+", str(label).lower())
    return {tok for tok in tokens if len(tok) > 2 and tok not in STOPWORDS}


def label_similarity(left: object, right: object) -> float:
    """Return a simple token Jaccard label similarity."""
    a = token_set(left)
    b = token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def infer_domain_from_topic(label: object) -> str:
    """Infer a domain prefix from a topic label."""
    text = str(label or "")
    if ":" in text:
        return text.split(":", 1)[0].strip()
    return ""


def prepare_topics(topics: pd.DataFrame) -> pd.DataFrame:
    """Normalize topic metadata."""
    if topics.empty:
        return pd.DataFrame(columns=["display_community", "topic_label", "topic_x", "topic_y", "topic_domain"])
    out = topics.copy()
    if "community" not in out.columns:
        raise ValueError("topics.csv must include a community column")
    out["display_community"] = pd.to_numeric(out["community"], errors="coerce").astype("Int64")
    out["topic_label"] = out.get("label", out["display_community"].astype(str)).astype(str)
    out["topic_x"] = pd.to_numeric(out.get("x", np.nan), errors="coerce")
    out["topic_y"] = pd.to_numeric(out.get("y", np.nan), errors="coerce")
    out["topic_domain"] = out["topic_label"].map(infer_domain_from_topic)
    return out[["display_community", "topic_label", "topic_x", "topic_y", "topic_domain"]]


def normalize_works(works: pd.DataFrame) -> pd.DataFrame:
    """Normalize the works input table."""
    if "id" in works.columns and "paper_id" not in works.columns:
        works = works.rename(columns={"id": "paper_id"})
    required = ["paper_id", "year", "display_community"]
    missing = [col for col in required if col not in works.columns]
    if missing:
        raise ValueError(f"works.csv is missing required columns: {missing}")
    out = works.copy()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["display_community"] = pd.to_numeric(out["display_community"], errors="coerce").astype("Int64")
    out["domain"] = out.get("domain", "all").fillna("all").astype(str)
    out["title"] = out.get("title", "").fillna("").astype(str)
    out["primary_field"] = out.get("primary_field", "").fillna("").astype(str)
    out["community_label"] = out.get("community_label", out["primary_field"]).fillna("").astype(str)
    out["cited_by_count"] = pd.to_numeric(out.get("cited_by_count", 0.0), errors="coerce").fillna(0.0)
    out["is_landmark"] = pd.to_numeric(out.get("is_landmark", 0), errors="coerce").fillna(0).astype(int)
    return out.dropna(subset=["year", "display_community"]).copy()


def normalize_scores(scores: pd.DataFrame, score_col: str, rgpm_col: Optional[str]) -> pd.DataFrame:
    """Normalize the Fig. 3 paper-level score table."""
    if "paper_id" not in scores.columns:
        raise ValueError("fig3_score_table.csv must include paper_id")
    keep_cols = ["paper_id", score_col]
    if rgpm_col is not None:
        keep_cols.append(rgpm_col)
    for optional in ["S_equal", "reference_count", "fold_id"]:
        if optional in scores.columns and optional not in keep_cols:
            keep_cols.append(optional)
    out = scores[keep_cols].copy()
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    if rgpm_col is not None:
        out[rgpm_col] = pd.to_numeric(out[rgpm_col], errors="coerce")
    return out


def load_data(
    fig3_run_dir: Path,
    fig3_input_dir: Path,
    domain_filter: Optional[Sequence[str]],
) -> LoadedData:
    """Load and merge Fig. 3 outputs into a paper table."""
    scores = read_csv_required(fig3_run_dir / "fig3_score_table.csv", "Fig. 3 score table")
    works = normalize_works(read_csv_required(fig3_input_dir / "works.csv", "Fig. 3 works table"))
    topics_path = fig3_input_dir / "topics.csv"
    topics = prepare_topics(pd.read_csv(topics_path, low_memory=False) if topics_path.exists() else pd.DataFrame())
    score_col = choose_column(scores, SCORE_CANDIDATES, "publication-day score", required=True)
    rgpm_col = choose_column(scores, RGPM_CANDIDATES, "RGPM outcome", required=False)
    score_df = normalize_scores(scores, score_col, rgpm_col)
    papers = works.merge(score_df, on="paper_id", how="left")
    papers = papers.merge(topics, on="display_community", how="left")
    if domain_filter:
        allowed = {str(item) for item in domain_filter}
        papers = papers[papers["domain"].isin(allowed)].copy()
    papers["focus_id"] = papers["domain"].astype(str) + "::" + papers["display_community"].astype(str)
    papers["focus_label"] = [
        clean_label(topic if isinstance(topic, str) and topic else fallback, domain)
        for topic, fallback, domain in zip(papers["topic_label"], papers["community_label"], papers["domain"])
    ]
    papers["topic_x"] = pd.to_numeric(papers["topic_x"], errors="coerce")
    papers["topic_y"] = pd.to_numeric(papers["topic_y"], errors="coerce")
    min_year = int(papers["year"].min()) if not papers.empty else 0
    max_year = int(papers["year"].max()) if not papers.empty else 0
    warnings = build_load_warnings(papers, score_col, rgpm_col)
    return LoadedData(
        papers=papers,
        topics=topics,
        score_col=score_col,
        rgpm_col=rgpm_col,
        min_year=min_year,
        max_year=max_year,
        warnings=warnings,
    )


def build_load_warnings(papers: pd.DataFrame, score_col: str, rgpm_col: Optional[str]) -> List[str]:
    """Build data-availability warnings."""
    warnings: List[str] = []
    if papers.empty:
        warnings.append("No papers remain after input loading and domain filtering.")
        return warnings
    score_rate = float(pd.to_numeric(papers[score_col], errors="coerce").notna().mean())
    if score_rate < 0.50:
        warnings.append(f"Only {score_rate:.1%} of papers have the selected score column {score_col}.")
    if rgpm_col is None:
        warnings.append("No RGPM column found; realized focus uses growth and citation signals only.")
    return warnings


def group_meta(papers: pd.DataFrame) -> pd.DataFrame:
    """Aggregate static focus metadata."""
    rows: List[Dict[str, Any]] = []
    for focus_id, group in papers.groupby("focus_id", sort=False):
        first = group.iloc[0]
        x = pd.to_numeric(group["topic_x"], errors="coerce").dropna()
        y = pd.to_numeric(group["topic_y"], errors="coerce").dropna()
        fallback_x = math.cos(2.0 * math.pi * stable_float(f"{focus_id}:x"))
        fallback_y = math.sin(2.0 * math.pi * stable_float(f"{focus_id}:y"))
        rows.append(
            {
                "focus_id": focus_id,
                "domain": first.get("domain", ""),
                "display_community": first.get("display_community", ""),
                "focus_label": first.get("focus_label", "Unlabelled focus"),
                "cluster_x": float(x.iloc[0]) if not x.empty else fallback_x,
                "cluster_y": float(y.iloc[0]) if not y.empty else fallback_y,
            }
        )
    return pd.DataFrame(rows)


def aggregate_historical(
    papers: pd.DataFrame,
    score_col: str,
    cutoff_year: int,
) -> pd.DataFrame:
    """Aggregate pre-cutoff focus signals."""
    hist = papers[papers["year"].astype(int) <= cutoff_year].copy()
    if hist.empty:
        return pd.DataFrame(columns=["focus_id"])
    recent = hist[hist["year"].astype(int).between(cutoff_year - 2, cutoff_year)]
    prior = hist[hist["year"].astype(int).between(cutoff_year - 7, cutoff_year - 3)]
    agg = hist.groupby("focus_id").agg(
        historical_size=("paper_id", "count"),
        historical_citations=("cited_by_count", "sum"),
        hist_scored_papers=(score_col, lambda values: int(pd.to_numeric(values, errors="coerce").notna().sum())),
        hist_mean_score=(score_col, "mean"),
        hist_max_score=(score_col, "max"),
        hist_top_tail_score=(score_col, top_tail_mean),
        hist_landmarks=("is_landmark", "sum"),
    )
    recent_counts = recent.groupby("focus_id").size().rename("recent_hist_size")
    prior_counts = prior.groupby("focus_id").size().rename("prior_hist_size")
    agg = agg.join(recent_counts, how="left").join(prior_counts, how="left")
    return agg.reset_index()


def aggregate_future(
    papers: pd.DataFrame,
    rgpm_col: Optional[str],
    validation_start: int,
    validation_end: int,
) -> pd.DataFrame:
    """Aggregate validation-window focus outcomes."""
    future = papers[papers["year"].astype(int).between(validation_start, validation_end)].copy()
    if future.empty:
        return pd.DataFrame(columns=["focus_id"])
    agg_map: Dict[str, Tuple[str, Any]] = {
        "future_papers": ("paper_id", "count"),
        "future_citations": ("cited_by_count", "sum"),
        "future_mean_citations": ("cited_by_count", "mean"),
        "future_landmarks": ("is_landmark", "sum"),
    }
    if rgpm_col is not None and rgpm_col in future.columns:
        agg_map["future_rgpm_top_tail"] = (rgpm_col, top_tail_mean)
        agg_map["future_rgpm_mean"] = (rgpm_col, "mean")
    agg = future.groupby("focus_id").agg(**agg_map)
    top_titles = future.sort_values(["cited_by_count", "year"], ascending=[False, True]).groupby("focus_id").head(1)
    top_titles = top_titles.set_index("focus_id")[["title", "year", "cited_by_count"]]
    top_titles = top_titles.rename(
        columns={
            "title": "top_future_paper",
            "year": "top_future_year",
            "cited_by_count": "top_future_citations",
        }
    )
    return agg.join(top_titles, how="left").reset_index()


def build_focus_table(
    papers: pd.DataFrame,
    score_col: str,
    rgpm_col: Optional[str],
    cutoff_year: int,
    validation_start: int,
    validation_end: int,
    min_historical_papers: int,
    min_future_papers: int,
    top_n: int,
) -> pd.DataFrame:
    """Build one focus-level forecast and outcome table."""
    meta = group_meta(papers)
    hist = aggregate_historical(papers, score_col, cutoff_year)
    future = aggregate_future(papers, rgpm_col, validation_start, validation_end)
    focus = meta.merge(hist, on="focus_id", how="left").merge(future, on="focus_id", how="left")
    focus = fill_focus_defaults(focus)
    add_focus_scores(focus, min_historical_papers=min_historical_papers)
    focus["prediction_eligible"] = focus["historical_size"] >= int(min_historical_papers)
    focus["realization_eligible"] = focus["future_papers"] >= int(min_future_papers)
    focus["predicted_rank"] = rank_focus(focus, "predicted_score", "prediction_eligible")
    focus["realized_rank"] = rank_focus(focus, "realized_score", "realization_eligible")
    focus["is_hotspot"] = focus["realized_rank"].le(top_n).fillna(False)
    focus["is_landmark_related"] = (focus["future_landmarks"] > 0) | (focus["hist_landmarks"] > 0)
    pred_top = set(focus.loc[focus["predicted_rank"].le(top_n), "focus_id"])
    real_top = set(focus.loc[focus["realized_rank"].le(top_n), "focus_id"])
    focus["forecast_category"] = [
        category_for_focus(fid, pred_top, real_top)
        for fid in focus["focus_id"]
    ]
    return focus.sort_values(["predicted_rank", "realized_rank"], na_position="last").reset_index(drop=True)


def fill_focus_defaults(focus: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric defaults after aggregation joins."""
    numeric_defaults = {
        "historical_size": 0,
        "historical_citations": 0.0,
        "hist_scored_papers": 0,
        "hist_mean_score": 0.0,
        "hist_max_score": 0.0,
        "hist_top_tail_score": 0.0,
        "hist_landmarks": 0,
        "recent_hist_size": 0,
        "prior_hist_size": 0,
        "future_papers": 0,
        "future_citations": 0.0,
        "future_mean_citations": 0.0,
        "future_landmarks": 0,
        "future_rgpm_top_tail": 0.0,
        "future_rgpm_mean": 0.0,
        "top_future_citations": 0.0,
    }
    for col, value in numeric_defaults.items():
        if col not in focus.columns:
            focus[col] = value
        focus[col] = pd.to_numeric(focus[col], errors="coerce").fillna(value)
    for col in ["top_future_paper", "top_future_year"]:
        if col not in focus.columns:
            focus[col] = ""
        focus[col] = focus[col].astype("object").where(focus[col].notna(), "")
    return focus


def add_focus_scores(focus: pd.DataFrame, min_historical_papers: int) -> None:
    """Add prediction and realization scores in place."""
    size_p = percentile_score(np.log1p(focus["historical_size"]))
    score_p = percentile_score(focus["hist_top_tail_score"])
    citation_p = percentile_score(np.log1p(focus["historical_citations"]))
    recent_p = percentile_score(np.log1p(focus["recent_hist_size"]))
    historical_growth_ratio = (focus["recent_hist_size"] + 1.0) / (focus["prior_hist_size"] + 1.0)
    historical_growth_p = percentile_score(np.log1p(historical_growth_ratio))
    historical_landmark_p = percentile_score(np.log1p(focus["hist_landmarks"]))
    low_data_penalty = ((min_historical_papers - focus["historical_size"]).clip(lower=0) / max(1, min_historical_papers))
    score_coverage = focus["hist_scored_papers"] / focus["historical_size"].clip(lower=1)
    focus["historical_growth_score"] = historical_growth_p
    focus["predicted_score"] = (
        0.20 * score_p
        + 0.20 * historical_growth_p
        + 0.20 * citation_p
        + 0.30 * recent_p
        + 0.10 * historical_landmark_p
    )
    focus["predicted_score"] -= 0.20 * low_data_penalty + 0.10 * (1.0 - score_coverage.clip(0.0, 1.0))
    future_count_p = percentile_score(np.log1p(focus["future_papers"]))
    future_cite_p = percentile_score(np.log1p(focus["future_citations"]))
    future_rgpm_p = percentile_score(focus["future_rgpm_top_tail"])
    growth_ratio = focus["future_papers"] / (focus["recent_hist_size"] + 1.0)
    growth_p = percentile_score(np.log1p(growth_ratio))
    landmark_bonus = (focus["future_landmarks"] > 0).astype(float) * 0.25
    focus["realized_score"] = 0.48 * future_count_p + 0.32 * future_cite_p + 0.12 * future_rgpm_p + 0.08 * growth_p + landmark_bonus
    focus["growth_only_score"] = 0.85 * historical_growth_p + 0.15 * size_p
    focus["citation_only_score"] = citation_p


def rank_focus(focus: pd.DataFrame, score_col: str, eligible_col: str) -> pd.Series:
    """Rank eligible foci by a score column."""
    ranks = pd.Series(pd.NA, index=focus.index, dtype="Float64")
    eligible = focus[eligible_col].fillna(False)
    if eligible.any():
        ranks.loc[eligible] = focus.loc[eligible, score_col].rank(ascending=False, method="first")
    return ranks


def category_for_focus(focus_id: str, pred_top: set[str], real_top: set[str]) -> str:
    """Return the forecast-outcome category for a focus."""
    if focus_id in pred_top and focus_id in real_top:
        return "hit"
    if focus_id in pred_top:
        return "predicted_only"
    if focus_id in real_top:
        return "unexpected_realized"
    return "background"


def build_predicted_focus_csv(focus: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Build the predicted focus output table."""
    cols = [
        "focus_id",
        "focus_label",
        "predicted_rank",
        "predicted_score",
        "cluster_x",
        "cluster_y",
        "historical_size",
        "domain",
        "forecast_category",
        "realized_rank",
    ]
    out = focus.loc[focus["predicted_rank"].notna(), cols].copy()
    out = out.sort_values("predicted_rank").head(max(top_n * 2, top_n))
    return out


def build_realized_focus_csv(focus: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Build the realized focus output table."""
    cols = [
        "focus_id",
        "focus_label",
        "realized_rank",
        "realized_score",
        "is_hotspot",
        "is_landmark_related",
        "domain",
        "future_papers",
        "future_citations",
        "future_rgpm_top_tail",
        "predicted_rank",
    ]
    out = focus.loc[focus["realized_rank"].notna(), cols].copy()
    out = out.sort_values("realized_rank").head(max(top_n * 2, top_n))
    return out


def match_predicted_to_realized(
    focus: pd.DataFrame,
    top_n: int,
    semantic_threshold: float,
) -> pd.DataFrame:
    """Build predicted-vs-realized alignment rows."""
    pred_top = focus.loc[focus["predicted_rank"].le(top_n)].sort_values("predicted_rank")
    real_top = focus.loc[focus["realized_rank"].le(top_n)].sort_values("realized_rank")
    real_by_id = real_top.set_index("focus_id")
    matched_realized: set[str] = set()
    rows: List[Dict[str, Any]] = []
    for _, pred in pred_top.iterrows():
        row, matched_id = alignment_for_prediction(pred, real_top, real_by_id, matched_realized, semantic_threshold)
        if matched_id:
            matched_realized.add(matched_id)
        rows.append(row)
    for _, real in real_top.iterrows():
        if real["focus_id"] not in matched_realized and real["focus_id"] not in set(pred_top["focus_id"]):
            rows.append(
                {
                    "predicted_focus": "",
                    "realized_focus": real["focus_label"],
                    "predicted_focus_id": "",
                    "realized_focus_id": real["focus_id"],
                    "match_score": 0.0,
                    "hit_type": "unexpected_realized",
                }
            )
    return pd.DataFrame(rows)


def alignment_for_prediction(
    pred: pd.Series,
    real_top: pd.DataFrame,
    real_by_id: pd.DataFrame,
    matched_realized: set[str],
    semantic_threshold: float,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Return one alignment row for a predicted focus."""
    focus_id = str(pred["focus_id"])
    if focus_id in real_by_id.index:
        real = real_by_id.loc[focus_id]
        return (
            {
                "predicted_focus": pred["focus_label"],
                "realized_focus": real["focus_label"],
                "predicted_focus_id": focus_id,
                "realized_focus_id": focus_id,
                "match_score": 1.0,
                "hit_type": "exact_hit",
            },
            focus_id,
        )
    best_id, best_label, best_score = best_semantic_match(pred, real_top, matched_realized)
    if best_id and best_score >= semantic_threshold:
        return (
            {
                "predicted_focus": pred["focus_label"],
                "realized_focus": best_label,
                "predicted_focus_id": focus_id,
                "realized_focus_id": best_id,
                "match_score": best_score,
                "hit_type": "semantic_hit",
            },
            best_id,
        )
    return (
        {
            "predicted_focus": pred["focus_label"],
            "realized_focus": "",
            "predicted_focus_id": focus_id,
            "realized_focus_id": "",
            "match_score": 0.0,
            "hit_type": "missed",
        },
        None,
    )


def best_semantic_match(
    pred: pd.Series,
    real_top: pd.DataFrame,
    matched_realized: set[str],
) -> Tuple[Optional[str], str, float]:
    """Find the best unmatched label-level match."""
    best_id: Optional[str] = None
    best_label = ""
    best_score = 0.0
    for _, real in real_top.iterrows():
        if str(real["focus_id"]) in matched_realized:
            continue
        if str(real.get("domain", "")) != str(pred.get("domain", "")):
            continue
        score = label_similarity(pred["focus_label"], real["focus_label"])
        if score > best_score:
            best_id = str(real["focus_id"])
            best_label = str(real["focus_label"])
            best_score = float(score)
    return best_id, best_label, best_score


def build_key_innovations(
    papers: pd.DataFrame,
    focus: pd.DataFrame,
    score_col: str,
    cutoff_year: int,
    validation_start: int,
    validation_end: int,
    case_count: int,
) -> pd.DataFrame:
    """Select representative predicted key innovation cases."""
    hit_focus = focus[focus["forecast_category"].eq("hit")].sort_values(["predicted_rank", "realized_rank"])
    used_ids = set(str(item) for item in hit_focus["focus_id"])
    fallback = focus[
        focus["predicted_rank"].notna() & ~focus["focus_id"].astype(str).isin(used_ids)
    ].sort_values("predicted_rank")
    selected_focus = pd.concat([hit_focus, fallback], ignore_index=True).head(case_count)
    rows: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(selected_focus.iterrows(), start=1):
        focus_id = str(row["focus_id"])
        hist = papers[(papers["focus_id"].eq(focus_id)) & (papers["year"].astype(int) <= cutoff_year)].copy()
        seeds = select_seed_papers(hist, score_col, limit=2)
        evidence = realized_evidence_text(row, validation_start, validation_end)
        rows.append(
            {
                "case_id": f"case_{i}",
                "case_label": row["focus_label"],
                "predicted_before_2021": bool(cutoff_year < validation_start),
                "seed_papers": seeds,
                "realized_evidence": evidence,
                "case_summary": case_summary_text(row, seeds),
            }
        )
    return pd.DataFrame(rows)


def select_seed_papers(hist: pd.DataFrame, score_col: str, limit: int = 2) -> str:
    """Format top pre-cutoff seed papers for a focus."""
    if hist.empty:
        return "No eligible pre-cutoff seed paper"
    sorted_hist = hist.sort_values([score_col, "cited_by_count"], ascending=[False, False]).head(limit)
    items = []
    for _, row in sorted_hist.iterrows():
        title = str(row.get("title", "Untitled")).strip() or "Untitled"
        items.append(f"{int(row['year'])}: {title}")
    return "; ".join(items)


def realized_evidence_text(row: pd.Series, validation_start: int, validation_end: int) -> str:
    """Build a compact realized evidence sentence."""
    papers = int(row.get("future_papers", 0))
    citations = int(round(float(row.get("future_citations", 0.0))))
    rank = row.get("realized_rank", pd.NA)
    rank_text = f"rank #{int(rank)}" if pd.notna(rank) else "not ranked"
    top_title = str(row.get("top_future_paper", "") or "").strip()
    top_part = f"; top follow-up: {top_title}" if top_title else ""
    return f"{validation_start}-{validation_end}: {papers} papers, {citations} citations, realized {rank_text}{top_part}"


def case_summary_text(row: pd.Series, seeds: str) -> str:
    """Build a compact case summary."""
    pred_rank = row.get("predicted_rank", pd.NA)
    real_rank = row.get("realized_rank", pd.NA)
    pred = f"#{int(pred_rank)}" if pd.notna(pred_rank) else "unranked"
    real = f"#{int(real_rank)}" if pd.notna(real_rank) else "unranked"
    seed_title = seeds.split(";", 1)[0]
    return f"Forecast {pred}, realized hotspot {real}; seed evidence: {seed_title}"


def parse_windows(values: Sequence[str]) -> List[Tuple[int, int, str]]:
    """Parse backtest windows such as 2000:2005."""
    windows: List[Tuple[int, int, str]] = []
    for value in values:
        match = re.match(r"^\s*(\d{4})\s*(?::|->|-)\s*(\d{4})\s*$", value)
        if not match:
            raise ValueError(f"Invalid backtest window {value!r}; use START:END")
        start = int(match.group(1))
        end = int(match.group(2))
        if end <= start:
            raise ValueError(f"Backtest window must have END > START: {value!r}")
        windows.append((start, end, f"{start}->{end}"))
    return windows


def top_k_hit_rate(pred_ids: Sequence[str], real_ids: Sequence[str], k: int) -> float:
    """Compute top-k overlap rate."""
    pred = list(pred_ids)[:k]
    real = list(real_ids)[:k]
    denom = min(k, len(real))
    if denom == 0 or not pred:
        return float("nan")
    return len(set(pred) & set(real)) / float(denom)


def ndcg_at_k(pred_ids: Sequence[str], real_ids: Sequence[str], k: int) -> float:
    """Compute binary NDCG@k from predicted focus order against realized top-k membership."""
    pred = list(pred_ids)[:k]
    real_set = set(list(real_ids)[:k])
    if not pred or not real_set:
        return float("nan")
    relevance = [1.0 if item in real_set else 0.0 for item in pred]
    dcg = sum(rel / math.log2(idx + 2.0) for idx, rel in enumerate(relevance))
    ideal_hits = min(len(real_set), len(pred), k)
    idcg = sum(1.0 / math.log2(idx + 2.0) for idx in range(ideal_hits))
    return float(dcg / idcg) if idcg else 0.0


def add_backtest_baseline_columns(backtest: pd.DataFrame) -> pd.DataFrame:
    """Add precision/NDCG and best non-graph baseline comparisons per window."""
    if backtest.empty:
        return backtest.copy()
    out = backtest.copy()
    if "precision_at_10" not in out.columns:
        out["precision_at_10"] = pd.to_numeric(out.get("top10_hit_rate"), errors="coerce")
    if "ndcg_at_10" not in out.columns:
        out["ndcg_at_10"] = np.nan
    for col in ["baseline_method", "baseline_precision_at_10", "baseline_ndcg_at_10", "delta_precision_at_10", "delta_ndcg_at_10"]:
        if col not in out.columns:
            out[col] = np.nan if col != "baseline_method" else ""
    for window, group in out.groupby("window", sort=False):
        baselines = group[~group["method"].astype(str).eq("graph_score")].copy()
        if baselines.empty:
            continue
        baselines["_baseline_score"] = pd.to_numeric(baselines["ndcg_at_10"], errors="coerce").fillna(
            pd.to_numeric(baselines["precision_at_10"], errors="coerce")
        )
        best = baselines.sort_values("_baseline_score", ascending=False).iloc[0]
        mask = out["window"].eq(window) & out["method"].astype(str).eq("graph_score")
        out.loc[mask, "baseline_method"] = str(best["method"])
        out.loc[mask, "baseline_precision_at_10"] = float(best["precision_at_10"])
        out.loc[mask, "baseline_ndcg_at_10"] = float(best["ndcg_at_10"])
        out.loc[mask, "delta_precision_at_10"] = pd.to_numeric(out.loc[mask, "precision_at_10"], errors="coerce") - float(
            best["precision_at_10"]
        )
        out.loc[mask, "delta_ndcg_at_10"] = pd.to_numeric(out.loc[mask, "ndcg_at_10"], errors="coerce") - float(best["ndcg_at_10"])
    return out


def ranked_ids(frame: pd.DataFrame, score_col: str, eligible_col: str, limit: int) -> List[str]:
    """Return focus ids ranked by a score column."""
    eligible = frame[eligible_col].fillna(False)
    ranked = frame.loc[eligible].sort_values(score_col, ascending=False).head(limit)
    return [str(item) for item in ranked["focus_id"]]


def build_backtest(
    papers: pd.DataFrame,
    score_col: str,
    rgpm_col: Optional[str],
    windows: Sequence[Tuple[int, int, str]],
    min_historical_papers: int,
    min_future_papers: int,
    top_n: int,
    seed: int,
) -> pd.DataFrame:
    """Build historical window hit-rate table."""
    rows: List[Dict[str, Any]] = []
    methods = [
        ("graph_score", "predicted_score"),
        ("growth_only", "growth_only_score"),
        ("citation_only", "citation_only_score"),
        ("random", "random_score"),
    ]
    for cutoff, end, label in windows:
        focus = build_focus_table(
            papers,
            score_col,
            rgpm_col,
            cutoff,
            cutoff + 1,
            min(end, int(papers["year"].max())),
            min_historical_papers,
            min_future_papers,
            top_n,
        )
        focus["random_score"] = [stable_float(f"{seed}:{label}:{fid}") for fid in focus["focus_id"]]
        real_ids = ranked_ids(focus, "realized_score", "realization_eligible", top_n)
        for method, col in methods:
            pred_ids = ranked_ids(focus, col, "prediction_eligible", top_n)
            rows.append(
                {
                    "window": label,
                    "method": method,
                    "top5_hit_rate": top_k_hit_rate(pred_ids, real_ids, 5),
                    "top10_hit_rate": top_k_hit_rate(pred_ids, real_ids, 10),
                    "precision_at_10": top_k_hit_rate(pred_ids, real_ids, 10),
                    "ndcg_at_10": ndcg_at_k(pred_ids, real_ids, 10),
                    "n_predicted": len(pred_ids),
                    "n_realized": len(real_ids),
                }
            )
    return add_backtest_baseline_columns(pd.DataFrame(rows))


def compute_tables(args: argparse.Namespace, data: LoadedData) -> Fig5Tables:
    """Compute all Fig. 5 tables."""
    validation_start = args.validation_start or args.cutoff_year + 1
    actual_validation_end = min(args.validation_end, data.max_year)
    warnings = list(data.warnings)
    if actual_validation_end < args.validation_end:
        warnings.append(
            f"Requested validation end year {args.validation_end}, but local data end at {data.max_year}; "
            f"plots use {validation_start}-{actual_validation_end}."
        )
    focus = build_focus_table(
        data.papers,
        data.score_col,
        data.rgpm_col,
        args.cutoff_year,
        validation_start,
        actual_validation_end,
        args.min_historical_papers,
        args.min_future_papers,
        args.top_n,
    )
    predicted = build_predicted_focus_csv(focus, args.top_n)
    realized = build_realized_focus_csv(focus, args.top_n)
    alignment = match_predicted_to_realized(focus, args.top_n, args.semantic_threshold)
    cases = build_key_innovations(
        data.papers,
        focus,
        data.score_col,
        args.cutoff_year,
        validation_start,
        actual_validation_end,
        args.case_count,
    )
    backtest = build_backtest(
        data.papers,
        data.score_col,
        data.rgpm_col,
        parse_windows(args.backtest_windows),
        args.min_historical_papers,
        args.min_future_papers,
        args.top_n,
        args.seed,
    )
    summary = build_summary(focus, alignment, backtest, args, data, validation_start, actual_validation_end, warnings)
    return Fig5Tables(focus, predicted, realized, alignment, cases, backtest, summary)


def build_summary(
    focus: pd.DataFrame,
    alignment: pd.DataFrame,
    backtest: pd.DataFrame,
    args: argparse.Namespace,
    data: LoadedData,
    validation_start: int,
    actual_validation_end: int,
    warnings: Sequence[str],
) -> Dict[str, Any]:
    """Build run summary metadata."""
    final_window = f"{args.cutoff_year}->{args.validation_end}"
    graph_final = backtest[(backtest["window"].eq(final_window)) & (backtest["method"].eq("graph_score"))]
    hit_counts = alignment["hit_type"].value_counts().to_dict() if not alignment.empty else {}
    return {
        "n_papers": int(len(data.papers)),
        "n_focus": int(len(focus)),
        "score_column": data.score_col,
        "rgpm_column": data.rgpm_col,
        "min_input_year": int(data.min_year),
        "max_input_year": int(data.max_year),
        "cutoff_year": int(args.cutoff_year),
        "validation_start": int(validation_start),
        "validation_end_requested": int(args.validation_end),
        "validation_end_actual": int(actual_validation_end),
        "top_n": int(args.top_n),
        "alignment_hit_counts": {str(k): int(v) for k, v in hit_counts.items()},
        "final_top5_hit_rate": value_or_none(graph_final["top5_hit_rate"].iloc[0]) if not graph_final.empty else None,
        "final_top10_hit_rate": value_or_none(graph_final["top10_hit_rate"].iloc[0]) if not graph_final.empty else None,
        "warnings": list(warnings),
    }


def value_or_none(value: object) -> Optional[float]:
    """Convert numeric values to float or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def build_alignment_metrics(alignment: pd.DataFrame, backtest: pd.DataFrame) -> pd.DataFrame:
    """Summarize forecast/backtest alignment metrics for manuscript traceability."""
    rows: List[Dict[str, Any]] = []
    if not alignment.empty and "hit_type" in alignment.columns:
        total = int(len(alignment))
        for hit_type, count in alignment["hit_type"].astype(str).value_counts().sort_index().items():
            rows.append(
                {
                    "metric_group": "predicted_realized_alignment",
                    "metric": f"hit_type:{hit_type}",
                    "value": float(count),
                    "denominator": total,
                }
            )
    if not backtest.empty:
        for metric in ["precision_at_10", "ndcg_at_10", "baseline_precision_at_10", "baseline_ndcg_at_10"]:
            if metric in backtest.columns:
                values = pd.to_numeric(backtest[metric], errors="coerce").dropna()
                if not values.empty:
                    rows.append(
                        {
                            "metric_group": "retrospective_backtest",
                            "metric": metric,
                            "value": float(values.mean()),
                            "denominator": int(len(values)),
                        }
                    )
        if "method" in backtest.columns:
            graph = backtest[backtest["method"].astype(str).eq("graph_score")].copy()
            for metric in [
                "precision_at_10",
                "ndcg_at_10",
                "baseline_precision_at_10",
                "baseline_ndcg_at_10",
                "delta_precision_at_10",
                "delta_ndcg_at_10",
            ]:
                if metric in graph.columns:
                    values = pd.to_numeric(graph[metric], errors="coerce").dropna()
                    if not values.empty:
                        rows.append(
                            {
                                "metric_group": "retrospective_backtest_graph_score",
                                "metric": f"graph_score_{metric}",
                                "value": float(values.mean()),
                                "denominator": int(len(values)),
                            }
                        )
    return pd.DataFrame(rows)


def build_failure_cases(alignment: pd.DataFrame) -> pd.DataFrame:
    """Extract forecast cases that were not exact or semantic hits."""
    if alignment.empty:
        return pd.DataFrame(columns=["focus_id", "hit_type", "failure_reason"])
    table = alignment.copy()
    if "hit_type" not in table.columns:
        table["hit_type"] = "unknown"
    failures = table[~table["hit_type"].astype(str).isin(["exact_hit", "semantic_hit"])].copy()
    if "failure_reason" not in failures.columns:
        failures["failure_reason"] = failures["hit_type"].astype(str).map(lambda value: f"forecast_alignment_{value}")
    expected_cols = ["focus_id", "hit_type", "failure_reason"]
    for col in expected_cols:
        if col not in failures.columns:
            failures[col] = ""
    return failures


def build_fig5_quality_report(out_dir: Path) -> Dict[str, Any]:
    """Read Fig.5 outputs and report whether forecast claims trace to CSV backtests."""
    backtest_focus = out_dir / "fig5_backtest_focus.csv"
    alignment_metrics = out_dir / "fig5_alignment_metrics.csv"
    failure_cases = out_dir / "fig5_failure_cases.csv"
    checks = {
        "backtest_table_present": int(backtest_focus.exists()),
        "alignment_metrics_present": int(alignment_metrics.exists()),
        "failure_cases_present": int(failure_cases.exists()),
        "precision_at_10_present": 0,
        "ndcg_at_10_present": 0,
        "baseline_comparison_present": 0,
        "mean_precision_delta_nonnegative": 0,
        "mean_ndcg_delta_positive": 0,
    }
    mean_precision = float("nan")
    mean_baseline_precision = float("nan")
    mean_ndcg = float("nan")
    mean_baseline_ndcg = float("nan")
    mean_delta_precision = float("nan")
    mean_delta_ndcg = float("nan")
    if backtest_focus.exists():
        table = pd.read_csv(backtest_focus)
        checks["precision_at_10_present"] = int("precision_at_10" in table.columns and table["precision_at_10"].notna().any())
        checks["ndcg_at_10_present"] = int("ndcg_at_10" in table.columns and table["ndcg_at_10"].notna().any())
        baseline_cols = [col for col in table.columns if str(col).startswith("baseline_")]
        checks["baseline_comparison_present"] = int(bool(baseline_cols))
        graph = table[table.get("method", pd.Series(dtype=str)).astype(str).eq("graph_score")].copy()
        if not graph.empty:
            mean_precision = float(pd.to_numeric(graph.get("precision_at_10", pd.Series(dtype=float)), errors="coerce").mean())
            mean_baseline_precision = float(
                pd.to_numeric(graph.get("baseline_precision_at_10", pd.Series(dtype=float)), errors="coerce").mean()
            )
            mean_ndcg = float(pd.to_numeric(graph.get("ndcg_at_10", pd.Series(dtype=float)), errors="coerce").mean())
            mean_baseline_ndcg = float(pd.to_numeric(graph.get("baseline_ndcg_at_10", pd.Series(dtype=float)), errors="coerce").mean())
            mean_delta_precision = float(pd.to_numeric(graph.get("delta_precision_at_10", pd.Series(dtype=float)), errors="coerce").mean())
            mean_delta_ndcg = float(pd.to_numeric(graph.get("delta_ndcg_at_10", pd.Series(dtype=float)), errors="coerce").mean())
            checks["mean_precision_delta_nonnegative"] = int(np.isfinite(mean_delta_precision) and mean_delta_precision >= 0.0)
            checks["mean_ndcg_delta_positive"] = int(np.isfinite(mean_delta_ndcg) and mean_delta_ndcg > 0.0)
    overall = bool(
        checks["backtest_table_present"]
        and checks["alignment_metrics_present"]
        and checks["precision_at_10_present"]
        and checks["ndcg_at_10_present"]
        and checks["baseline_comparison_present"]
    )
    return {
        "figure": "fig5",
        "overall_pass": overall,
        "status_label": (
            "forecast_backtest_beats_baseline"
            if overall and checks["mean_precision_delta_nonnegative"] and checks["mean_ndcg_delta_positive"]
            else "forecast_backtest_ready_but_baseline_underperforms"
            if overall
            else "forecast_backtest_incomplete"
        ),
        "quality_gates": {
            "checks": checks,
            "mean_precision_at_10": mean_precision,
            "mean_baseline_precision_at_10": mean_baseline_precision,
            "mean_ndcg_at_10": mean_ndcg,
            "mean_baseline_ndcg_at_10": mean_baseline_ndcg,
            "mean_delta_precision_at_10": mean_delta_precision,
            "mean_delta_ndcg_at_10": mean_delta_ndcg,
            "allowed_claim": "Forecast/backtest claims must trace to fig5_backtest_focus.csv and fig5_alignment_metrics.csv and beat no-leakage historical baselines before main-text use.",
            "forbidden_claim": "Do not use image handoff or layout-only panels as evidence for forecast performance.",
        },
    }


def write_outputs(out_dir: Path, tables: Fig5Tables, args: argparse.Namespace, data: LoadedData) -> None:
    """Write Fig. 5 tables and metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables.predicted_focus.to_csv(out_dir / "fig5_predicted_focus.csv", index=False)
    tables.realized_focus.to_csv(out_dir / "fig5_realized_focus.csv", index=False)
    tables.alignment.to_csv(out_dir / "fig5_focus_alignment.csv", index=False)
    tables.key_innovations.to_csv(out_dir / "fig5_key_innovations.csv", index=False)
    tables.backtest.to_csv(out_dir / "fig5_backtest.csv", index=False)
    tables.backtest.to_csv(out_dir / "fig5_backtest_focus.csv", index=False)
    build_alignment_metrics(tables.alignment, tables.backtest).to_csv(out_dir / "fig5_alignment_metrics.csv", index=False)
    build_failure_cases(tables.alignment).to_csv(out_dir / "fig5_failure_cases.csv", index=False)
    tables.focus.to_csv(out_dir / "fig5_focus_map.csv", index=False)
    (out_dir / "fig5_summary.json").write_text(json.dumps(tables.summary, indent=2), encoding="utf-8")
    run_config = build_run_config(args, data)
    (out_dir / "fig5_run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    report = build_fig5_quality_report(out_dir)
    (out_dir / "figure_quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_run_config(args: argparse.Namespace, data: LoadedData) -> Dict[str, Any]:
    """Build a JSON-serializable run configuration."""
    return {
        "fig3_run_dir": str(args.fig3_run_dir),
        "fig3_input_dir": str(args.fig3_input_dir),
        "out_dir": str(args.out_dir),
        "domain_filter": args.domain_filter,
        "cutoff_year": args.cutoff_year,
        "validation_start": args.validation_start,
        "validation_end": args.validation_end,
        "top_n": args.top_n,
        "case_count": args.case_count,
        "min_historical_papers": args.min_historical_papers,
        "min_future_papers": args.min_future_papers,
        "score_column": data.score_col,
        "rgpm_column": data.rgpm_col,
        "min_input_year": data.min_year,
        "max_input_year": data.max_year,
        "backtest_windows": args.backtest_windows,
        "seed": args.seed,
    }


def draw_panel_a(ax: plt.Axes, tables: Fig5Tables) -> None:
    """Draw forecasting setting panel."""
    panel_frame(ax, "a", "Forecast setting")
    focus = tables.focus
    score_col = "predicted_score" if "predicted_score" in focus.columns else "forecast_score"
    cutoff = int(tables.summary.get("cutoff_year", 2020))
    ax.text(0.28, 0.865, f"Historical knowledge graph\n(1950-{cutoff})", ha="center", va="top", fontsize=8.8, fontweight="bold")
    ax.text(0.80, 0.865, f"Future window\n({tables.summary['validation_start']}-{tables.summary['validation_end_requested']})", ha="center", va="top", fontsize=8.8, fontweight="bold")
    cloud = focus.sort_values("historical_size", ascending=False).head(130).copy()
    if not cloud.empty:
        cloud_x, cloud_y = focus_layout_coordinates(cloud, spread=0.22)
        x = 0.055 + 0.39 * cloud_x
        y = 0.19 + 0.56 * cloud_y
        sizes = 8 + 42 * normalize_series(np.log1p(cloud["historical_size"]))
        colors = [color_for_domain(domain) for domain in cloud["domain"]]
        # Deterministic nearest-neighbour scaffolding gives the left cloud a KG texture.
        coords = np.column_stack([x, y])
        for i in range(min(len(coords), 75)):
            distances = np.sum((coords - coords[i]) ** 2, axis=1)
            neighbours = np.argsort(distances)[1:3]
            for j in neighbours:
                if j <= i:
                    continue
                ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], transform=ax.transAxes, color="#CBD5E1", lw=0.35, alpha=0.42, zorder=1)
        ax.scatter(x, y, s=sizes, c=colors, alpha=0.82, edgecolor="white", linewidth=0.28, transform=ax.transAxes, zorder=2)
        important = cloud.sort_values([score_col, "realized_score"], ascending=False).head(5)
        for _, row in important.iterrows():
            idx = cloud.index.get_loc(row.name)
            ax.scatter([x[idx]], [y[idx]], s=sizes[idx] * 1.35, facecolor="none", edgecolor=color_for_domain(row.get("domain")), linewidth=1.1, transform=ax.transAxes, zorder=4)
    ax.text(0.50, 0.62, "Forecast", ha="center", va="center", fontsize=9.0, color=PREDICTED_BLUE, fontweight="bold")
    val_text = f"{tables.summary['validation_start']}-{tables.summary['validation_end_requested']}\nrealized outcomes"
    draw_arrow(ax, (0.47, 0.51), (0.64, 0.51), color=PREDICTED_BLUE, lw=2.2, mutation_scale=24)
    ax.text(0.50, 0.42, "Frontier forecasting\nusing only pre-cutoff\nknowledge structure", ha="center", va="top", fontsize=6.8, color=TEXT_MID)
    rounded_box(ax, 0.66, 0.24, 0.28, 0.49, "#F8FAFC", "#8AB4F8", 0.8, 0.018, linestyle=(0, (4, 3)), zorder=1)
    rng_points = []
    for i in range(90):
        px = 0.69 + 0.21 * stable_float(f"future-x-{i}")
        py = 0.29 + 0.36 * stable_float(f"future-y-{i}")
        rng_points.append((px, py))
    ax.scatter([p[0] for p in rng_points], [p[1] for p in rng_points], s=9, c="#D8DEE7", alpha=0.55, transform=ax.transAxes, zorder=2)
    future_colors = [PREDICTED_BLUE, "#F97316", REALIZED_GREEN, "#7C3AED", LANDMARK_RED]
    for (px, py), color in zip([(0.73, 0.61), (0.84, 0.62), (0.84, 0.40), (0.73, 0.34), (0.90, 0.29)], future_colors):
        draw_question_focus(ax, px, py, color)
    ax.text(0.80, 0.20, "Predict emerging foci\nand key innovations", ha="center", va="top", fontsize=7.1, color=TEXT_DARK)
    ax.text(0.50, 0.29, "only pre-2021\nknowledge structure", ha="center", va="center", fontsize=6.8, color=TEXT_MID)
    ax.text(
        0.50,
        0.11,
        f"Actual local validation: {tables.summary['validation_start']}-{tables.summary['validation_end_actual']}",
        ha="center",
        va="center",
        fontsize=6.5,
        color=TEXT_MID,
    )
    legend_domains = list(dict.fromkeys(focus["domain"].astype(str).tolist()))[:5]
    legend_x = 0.05
    for domain in legend_domains:
        ax.scatter([legend_x], [0.055], s=20, color=color_for_domain(domain), transform=ax.transAxes, zorder=5)
        ax.text(legend_x + 0.018, 0.055, clean_label(domain.replace("_", " ").title()), ha="left", va="center", fontsize=5.7, transform=ax.transAxes)
        legend_x += 0.17


def draw_topic_cloud(ax: plt.Axes, focus: pd.DataFrame) -> None:
    """Draw a small topic bubble cloud inside panel a."""
    cloud = focus.sort_values("historical_size", ascending=False).head(60)
    if cloud.empty:
        return
    x = 0.11 + 0.22 * normalize_series(cloud["cluster_x"])
    y = 0.47 + 0.16 * normalize_series(cloud["cluster_y"])
    size = 14 + 60 * normalize_series(np.log1p(cloud["historical_size"]))
    colors = [HIT_BLUE if c == "hit" else "#CBD5E1" for c in cloud["forecast_category"]]
    ax.scatter(x, y, s=size, c=colors, alpha=0.75, edgecolor="white", linewidth=0.35, transform=ax.transAxes, zorder=2)


def normalize_series(values: pd.Series | np.ndarray) -> np.ndarray:
    """Scale values to [0, 1], returning midpoints for constants."""
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    if np.all(np.isnan(arr)):
        return np.full_like(arr, 0.5, dtype=float)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if abs(mx - mn) < 1e-12:
        return np.full_like(arr, 0.5, dtype=float)
    return (np.nan_to_num(arr, nan=mn) - mn) / (mx - mn)


def focus_layout_coordinates(focus: pd.DataFrame, spread: float = 0.18) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic display coordinates that avoid ring-like topic layouts."""
    if focus.empty:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    raw_x = normalize_series(pd.to_numeric(focus.get("cluster_x", 0.0), errors="coerce").fillna(0.0)) - 0.5
    raw_y = normalize_series(pd.to_numeric(focus.get("cluster_y", 0.0), errors="coerce").fillna(0.0)) - 0.5
    volume = normalize_series(np.log1p(pd.to_numeric(focus.get("historical_size", 0.0), errors="coerce").fillna(0.0)))
    domains = list(dict.fromkeys(focus.get("domain", pd.Series(["all"] * len(focus))).astype(str).tolist()))
    centers: Dict[str, Tuple[float, float]] = {}
    fallback_i = 0
    for domain in domains:
        if domain in DOMAIN_LAYOUT_CENTERS:
            centers[domain] = DOMAIN_LAYOUT_CENTERS[domain]
        else:
            angle = 2.0 * math.pi * fallback_i / max(1, len(domains))
            centers[domain] = (0.52 + 0.22 * math.cos(angle), 0.52 + 0.20 * math.sin(angle))
            fallback_i += 1
    xs: List[float] = []
    ys: List[float] = []
    for pos, (_, row) in enumerate(focus.iterrows()):
        focus_id = str(row.get("focus_id", pos))
        cx, cy = centers.get(str(row.get("domain", "")), (0.52, 0.52))
        angle = 2.0 * math.pi * stable_float(f"{focus_id}:layout-angle")
        radial = math.sqrt(stable_float(f"{focus_id}:layout-radius"))
        local_x = spread * radial * math.cos(angle) + 0.09 * float(raw_x[pos])
        local_y = spread * 0.78 * radial * math.sin(angle) + 0.09 * float(raw_y[pos])
        prominence_pull = 0.035 * float(volume[pos])
        xs.append(float(np.clip(cx + local_x + prominence_pull, 0.055, 0.945)))
        ys.append(float(np.clip(cy + local_y + 0.015 * float(volume[pos]), 0.075, 0.925)))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def draw_panel_b(ax: plt.Axes, tables: Fig5Tables, top_n: int) -> None:
    """Draw predicted-vs-realized ranked focus alignment panel."""
    panel_frame(ax, "b", f"Top predicted research foci ({tables.summary['validation_start']}-{tables.summary['validation_end_requested']})")
    focus = tables.focus
    score_col = "predicted_score" if "predicted_score" in focus.columns else "forecast_score"
    pred = focus.loc[focus["predicted_rank"].le(top_n)].sort_values("predicted_rank")
    if pred.empty:
        ax.text(0.5, 0.5, "No predicted focus available", ha="center", va="center", fontsize=8, color=TEXT_LIGHT)
        return
    ax.text(0.11, 0.86, "Predicted research foci (word cloud)", fontsize=7.3, fontweight="bold", color=TEXT_DARK)
    word_positions = [
        (0.20, 0.64), (0.29, 0.54), (0.23, 0.43), (0.32, 0.35), (0.18, 0.31),
        (0.10, 0.73), (0.39, 0.71), (0.09, 0.43), (0.41, 0.47), (0.38, 0.25),
    ]
    scores = pd.to_numeric(pred[score_col], errors="coerce").fillna(0.0)
    score_norm = normalize_series(scores)
    for idx, (_, row) in enumerate(pred.head(min(top_n, len(word_positions))).iterrows()):
        x, y = word_positions[idx]
        color = color_for_domain(row.get("domain")) if idx < 5 else blend_with_white(PREDICTED_BLUE, 0.45)
        fontsize = 6.0 + 8.8 * float(score_norm[idx])
        label = ellipsize(row["focus_label"], 34)
        ax.text(x, y, label, transform=ax.transAxes, ha="center", va="center", fontsize=fontsize, color=color, fontweight="bold" if idx < 5 else "normal", alpha=0.95)
    ax.text(0.07, 0.22, "Forecast score combines pre-cutoff graph score,\nhistorical seed strength and topic-level concentration.", transform=ax.transAxes, ha="left", va="top", fontsize=5.8, color=TEXT_LIGHT)

    bar_ax = ax.inset_axes([0.55, 0.16, 0.38, 0.68])
    bar_rows = pred.head(min(8, top_n)).iloc[::-1].copy()
    bar_scores = pd.to_numeric(bar_rows[score_col], errors="coerce").fillna(0.0)
    y = np.arange(len(bar_rows))
    colors = [color_for_domain(domain) for domain in bar_rows["domain"]]
    bar_ax.barh(y, bar_scores, color=colors, alpha=0.82, height=0.55)
    bar_ax.set_yticks(y)
    labels = [f"{int(rank)}  {ellipsize(label, 31)}" for rank, label in zip(bar_rows["predicted_rank"], bar_rows["focus_label"])]
    bar_ax.set_yticklabels(labels, fontsize=6.3)
    bar_ax.set_xlim(0, max(1.0, float(bar_scores.max()) * 1.15))
    bar_ax.set_xlabel("Forecast priority score", fontsize=6.4)
    bar_ax.set_title("Top predicted foci (by forecast score)", fontsize=7.3, fontweight="bold", pad=6)
    bar_ax.grid(axis="x", color=GRID, lw=0.45, alpha=0.55)
    for spine in ["top", "right"]:
        bar_ax.spines[spine].set_visible(False)
    for yi, value in zip(y, bar_scores):
        bar_ax.text(float(value) + max(0.02, float(bar_scores.max()) * 0.02), yi, f"{value:.2f}", va="center", fontsize=6.2, color=TEXT_DARK)
    hit_count = int((tables.alignment["hit_type"] == "exact_hit").sum()) if not tables.alignment.empty else 0
    semantic_count = int((tables.alignment["hit_type"] == "semantic_hit").sum()) if not tables.alignment.empty else 0
    draw_pill(ax, 0.39, 0.055, f"{hit_count + semantic_count}/{top_n} realized hits", HIT_BLUE, 0.20, height=0.050, fontsize=6.2)


def draw_ranked_list(
    ax: plt.Axes,
    rows: pd.DataFrame,
    y_lookup: Mapping[str, float],
    rank_col: str,
    x: float,
    align: str,
) -> None:
    """Draw a compact ranked label list."""
    for _, row in rows.iterrows():
        y = y_lookup[str(row["focus_id"])]
        category = str(row.get("forecast_category", "background"))
        color = color_for_category(category)
        ax.scatter([x - 0.018], [y + 0.006], s=24, color=color, transform=ax.transAxes, zorder=5)
        rank = int(row[rank_col])
        label = wrap_text(row["focus_label"], 27)
        ax.text(x, y, f"{rank}. {label}", ha=align, va="center", fontsize=6.2, color=TEXT_DARK, transform=ax.transAxes)


def draw_alignment_lines(
    ax: plt.Axes,
    pred: pd.DataFrame,
    real: pd.DataFrame,
    pred_y: Mapping[str, float],
    real_y: Mapping[str, float],
) -> None:
    """Draw connections between matching focus ids."""
    real_ids = set(str(item) for item in real["focus_id"])
    for _, row in pred.iterrows():
        fid = str(row["focus_id"])
        if fid not in real_ids:
            continue
        y0 = pred_y[fid]
        y1 = real_y[fid]
        ax.plot([0.40, 0.59], [y0, y1], color=HIT_BLUE, lw=1.0, alpha=0.78, transform=ax.transAxes, zorder=3)


def color_for_category(category: str) -> str:
    """Return the visual color for a focus category."""
    if category == "hit":
        return HIT_BLUE
    if category == "predicted_only":
        return "#93C5FD"
    if category == "unexpected_realized":
        return UNEXPECTED_GREEN
    return "#CBD5E1"


def color_for_domain(domain: object) -> str:
    """Return a stable color for one domain label."""
    key = str(domain or "").strip()
    if key in DOMAIN_COLORS:
        return DOMAIN_COLORS[key]
    idx = int(stable_float(key) * len(FALLBACK_COLORS)) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[idx]


def draw_question_focus(ax: plt.Axes, x: float, y: float, color: str, radius: float = 0.055) -> None:
    """Draw a dashed future focus bubble."""
    circle = mpatches.Circle(
        (x, y),
        radius,
        transform=ax.transAxes,
        facecolor=blend_with_white(color, 0.92),
        edgecolor=color,
        linewidth=1.1,
        linestyle=(0, (4, 3)),
        alpha=0.95,
        zorder=5,
    )
    ax.add_patch(circle)
    ax.text(x, y, "?", transform=ax.transAxes, ha="center", va="center", fontsize=14, color=color, fontweight="bold", zorder=6)


def draw_take_home(ax: plt.Axes, text: str) -> None:
    """Draw the Fig. 5 take-home message strip."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded_box(ax, 0.16, 0.16, 0.68, 0.68, "#F8FBFF", "#8AB4F8", 0.85, 0.020, zorder=0)
    # Simple binocular icon made from vector patches, kept evidence-neutral.
    icon_x, icon_y = 0.205, 0.50
    ax.add_patch(mpatches.Circle((icon_x - 0.018, icon_y), 0.025, transform=ax.transAxes, facecolor="none", edgecolor=PREDICTED_BLUE, lw=1.4))
    ax.add_patch(mpatches.Circle((icon_x + 0.018, icon_y), 0.025, transform=ax.transAxes, facecolor="none", edgecolor=PREDICTED_BLUE, lw=1.4))
    ax.plot([icon_x - 0.006, icon_x + 0.006], [icon_y + 0.015, icon_y + 0.015], transform=ax.transAxes, color=PREDICTED_BLUE, lw=1.2)
    ax.plot([icon_x - 0.030, icon_x - 0.014], [icon_y + 0.033, icon_y + 0.018], transform=ax.transAxes, color=PREDICTED_BLUE, lw=1.2)
    ax.plot([icon_x + 0.030, icon_x + 0.014], [icon_y + 0.033, icon_y + 0.018], transform=ax.transAxes, color=PREDICTED_BLUE, lw=1.2)
    ax.text(0.255, 0.56, "Take-home message:", transform=ax.transAxes, ha="left", va="center", fontsize=9.2, fontweight="bold", color=TEXT_DARK)
    ax.text(0.405, 0.56, textwrap.fill(text, width=118), transform=ax.transAxes, ha="left", va="center", fontsize=8.7, color=TEXT_DARK)


def draw_panel_c(ax: plt.Axes, tables: Fig5Tables) -> None:
    """Draw frontier landscape map panel."""
    panel_frame(ax, "c", "Predicted frontier landscape (topic map)")
    focus = tables.focus.copy()
    score_col = "predicted_score" if "predicted_score" in focus.columns else "forecast_score"
    legend_ax = ax.inset_axes([0.035, 0.16, 0.16, 0.70])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    legend_ax.text(0.0, 0.94, "Circle size: historical\nknowledge volume\n(1950-2020)", transform=legend_ax.transAxes, fontsize=5.9, fontweight="bold", va="top")
    for y, size, label in [(0.72, 90, "Large"), (0.61, 52, "Medium"), (0.51, 26, "Small")]:
        legend_ax.scatter([0.12], [y], s=size, facecolor="white", edgecolor=TEXT_DARK, linewidth=0.7, transform=legend_ax.transAxes)
        legend_ax.text(0.28, y, label, transform=legend_ax.transAxes, va="center", fontsize=5.7)
    legend_ax.text(
        0.0,
        0.36,
        f"Color intensity:\nforecast strength\n({tables.summary['validation_start']}-{tables.summary['validation_end_requested']})",
        transform=legend_ax.transAxes,
        fontsize=5.9,
        fontweight="bold",
        va="top",
    )
    for i in range(18):
        legend_ax.add_patch(
            mpatches.Rectangle(
                (0.05, 0.06 + i * 0.011),
                0.13,
                0.012,
                transform=legend_ax.transAxes,
                facecolor=plt.cm.turbo(i / 17),
                edgecolor="none",
            )
        )
    legend_ax.text(0.24, 0.25, "High", transform=legend_ax.transAxes, fontsize=5.5, va="center")
    legend_ax.text(0.24, 0.06, "Low", transform=legend_ax.transAxes, fontsize=5.5, va="center")

    map_ax = ax.inset_axes([0.20, 0.10, 0.74, 0.78])
    focus["map_x"], focus["map_y"] = focus_layout_coordinates(focus, spread=0.21)
    hist_norm = normalize_series(np.log1p(focus["historical_size"]))
    sizes = 18 + 185 * hist_norm
    strength = normalize_series(pd.to_numeric(focus[score_col], errors="coerce").fillna(0.0))
    base_colors = [mcolors.to_hex(plt.cm.turbo(0.15 + 0.72 * float(value))) for value in strength]
    colors = [
        color_for_category(str(category)) if str(category) in {"hit", "predicted_only", "unexpected_realized"} else base_colors[i]
        for i, category in enumerate(focus["forecast_category"])
    ]
    edge_colors = [
        color_for_category(str(category)) if str(category) != "background" else ("#FFFFFF" if not bool(is_hotspot) else REALIZED_GREEN)
        for category, is_hotspot in zip(focus["forecast_category"], focus["is_hotspot"])
    ]
    coords = focus[["map_x", "map_y"]].to_numpy(dtype=float)
    if len(coords) > 4:
        for i in range(min(len(coords), 140)):
            distances = np.sum((coords - coords[i]) ** 2, axis=1)
            neighbours = np.argsort(distances)[1:3]
            for j in neighbours:
                if j <= i:
                    continue
                map_ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], color="#CBD5E1", lw=0.35, alpha=0.22, zorder=0)
    map_ax.scatter(
        focus["map_x"],
        focus["map_y"],
        s=sizes,
        c=colors,
        edgecolor=edge_colors,
        linewidth=0.9,
        alpha=0.86,
        zorder=2,
    )
    top_pred = focus[focus["predicted_rank"].le(6)].copy()
    score_max = float(pd.to_numeric(focus[score_col], errors="coerce").fillna(0.0).max())
    for _, row in top_pred.iterrows():
        score_value = float(pd.to_numeric(pd.Series([row.get(score_col, 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        radius_size = 420 + 520 * (score_value / max(score_max, 1e-9))
        map_ax.scatter([row["map_x"]], [row["map_y"]], s=radius_size, color=color_for_domain(row.get("domain")), alpha=0.11, edgecolor="none", zorder=1)
        map_ax.scatter([row["map_x"]], [row["map_y"]], s=radius_size * 0.45, color=color_for_domain(row.get("domain")), alpha=0.12, edgecolor="none", zorder=1)
    landmark = focus[focus["is_landmark_related"].fillna(False)]
    if not landmark.empty:
        map_ax.scatter(landmark["map_x"], landmark["map_y"], s=34, marker="*", color=LANDMARK_RED, zorder=4)
    label_focus = pd.concat([top_pred.head(4), select_map_labels(focus)], ignore_index=True).drop_duplicates("focus_id").head(7)
    offsets = [(18, 16), (18, -18), (-70, 16), (-70, -20), (18, 34), (-58, 34), (22, -34)]
    for i, (_, row) in enumerate(label_focus.iterrows()):
        dx, dy = offsets[i % len(offsets)]
        map_ax.annotate(
            wrap_text(row["focus_label"], 18),
            xy=(row["map_x"], row["map_y"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.7,
            color=color_for_domain(row.get("domain")),
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 0.8},
            arrowprops={"arrowstyle": "-", "color": color_for_domain(row.get("domain")), "lw": 0.55, "alpha": 0.60},
            zorder=5,
        )
    map_ax.set_xlim(0, 1)
    map_ax.set_ylim(0, 1)
    map_ax.set_xticks([])
    map_ax.set_yticks([])
    for spine in map_ax.spines.values():
        spine.set_visible(False)
    draw_map_legend(ax)


def select_map_labels(focus: pd.DataFrame) -> pd.DataFrame:
    """Select a small non-overwhelming set of map labels."""
    hits = focus[focus["forecast_category"].eq("hit")].sort_values("realized_rank").head(3)
    unexpected = focus[focus["forecast_category"].eq("unexpected_realized")].sort_values("realized_rank").head(2)
    selected = pd.concat([hits, unexpected], ignore_index=True)
    return selected.drop_duplicates("focus_id").head(5)


def draw_map_legend(ax: plt.Axes) -> None:
    """Draw the map legend in panel c."""
    items = [
        ("hit", HIT_BLUE),
        ("predicted only", "#93C5FD"),
        ("unexpected", UNEXPECTED_GREEN),
        ("background", "#CBD5E1"),
    ]
    x = 0.42
    for label, color in items:
        ax.scatter([x], [0.065], s=22, color=color, transform=ax.transAxes, zorder=5)
        ax.text(x + 0.025, 0.065, label, ha="left", va="center", fontsize=5.8, transform=ax.transAxes)
        x += 0.135


def draw_panel_d(ax: plt.Axes, tables: Fig5Tables) -> None:
    """Draw representative key innovation cases."""
    panel_frame(ax, "d", "Representative predicted key innovations (examples)")
    cases = tables.key_innovations.head(4)
    if cases.empty:
        ax.text(0.5, 0.5, "No case available", ha="center", va="center", fontsize=8, color=TEXT_LIGHT)
        return
    card_colors = [PREDICTED_BLUE, "#F97316", REALIZED_GREEN, "#7C3AED"]
    card_w = 0.215
    card_h = 0.74
    y = 0.12
    for i, (_, row) in enumerate(cases.iterrows()):
        x = 0.035 + i * 0.238
        color = card_colors[i % len(card_colors)]
        rounded_box(ax, x, y, card_w, card_h, blend_with_white(color, 0.94), blend_with_white(color, 0.45), 0.7, 0.018, zorder=1)
        ax.add_patch(mpatches.Circle((x + 0.025, y + card_h - 0.045), 0.019, transform=ax.transAxes, facecolor=color, edgecolor="white", lw=0.7, zorder=3))
        ax.text(x + 0.025, y + card_h - 0.045, str(i + 1), transform=ax.transAxes, ha="center", va="center", fontsize=7.2, color="white", fontweight="bold", zorder=4)
        ax.text(x + 0.052, y + card_h - 0.038, wrap_text(row["case_label"], 22), ha="left", va="top", fontsize=7.0, fontweight="bold", color=color, transform=ax.transAxes)
        # Lightweight abstract icon: three connected nodes inside each card.
        icon_cx, icon_cy = x + card_w / 2, y + 0.52
        icon_points = [(icon_cx - 0.035, icon_cy + 0.018), (icon_cx + 0.032, icon_cy + 0.028), (icon_cx - 0.005, icon_cy - 0.035)]
        ax.plot([p[0] for p in icon_points + [icon_points[0]]], [p[1] for p in icon_points + [icon_points[0]]], transform=ax.transAxes, color=color, lw=1.2, alpha=0.85)
        for px, py in icon_points:
            ax.add_patch(mpatches.Circle((px, py), 0.014, transform=ax.transAxes, facecolor="white", edgecolor=color, lw=1.2))
        ax.text(x + card_w / 2, y + 0.375, "Predicted role", transform=ax.transAxes, ha="center", va="center", fontsize=6.8, fontweight="bold", color=TEXT_DARK)
        ax.text(x + card_w / 2, y + 0.315, wrap_text(ellipsize(row.get("case_summary", ""), 54), 22), transform=ax.transAxes, ha="center", va="center", fontsize=6.1, color=color, fontweight="bold")
        ax.text(x + card_w / 2, y + 0.205, "Why highlighted", transform=ax.transAxes, ha="center", va="center", fontsize=6.7, fontweight="bold", color=TEXT_DARK)
        why = f"{ellipsize(row.get('seed_papers', ''), 72)} {ellipsize(row.get('realized_evidence', ''), 78)}"
        ax.text(x + card_w / 2, y + 0.105, wrap_text(why, 26), transform=ax.transAxes, ha="center", va="center", fontsize=5.8, color=TEXT_DARK)


def draw_panel_e(ax: plt.Axes, tables: Fig5Tables) -> None:
    """Draw historical backtesting reliability panel."""
    panel_frame(ax, "e", "Forecasting performance is reproducible across historical windows")
    bt = tables.backtest.copy()
    plot_ax = ax.inset_axes([0.12, 0.20, 0.82, 0.66])
    method_order = ["graph_score", "growth_only", "citation_only", "random"]
    colors = {
        "graph_score": HIT_BLUE,
        "growth_only": "#F59E0B",
        "citation_only": "#7C3AED",
        "random": "#9CA3AF",
    }
    labels = {
        "graph_score": "graph score",
        "growth_only": "growth-only",
        "citation_only": "citation-only",
        "random": "random",
    }
    windows = list(dict.fromkeys(bt["window"].tolist()))
    x_lookup = {window: i for i, window in enumerate(windows)}
    for method in method_order:
        part = bt[bt["method"].eq(method)].copy()
        if part.empty:
            continue
        xs = [x_lookup[w] for w in part["window"]]
        ys = pd.to_numeric(part["top10_hit_rate"], errors="coerce") * 100.0
        plot_ax.plot(xs, ys, marker="o", lw=1.4, ms=3.2, color=colors[method], label=labels[method])
    plot_ax.set_xticks(range(len(windows)))
    plot_ax.set_xticklabels(windows, rotation=35, ha="right", fontsize=5.7)
    plot_ax.set_ylabel("Top-10 hit rate (%)", fontsize=6.3)
    plot_ax.set_ylim(0, 105)
    plot_ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.65)
    plot_ax.legend(frameon=False, fontsize=5.7, loc="upper left")
    for spine in plot_ax.spines.values():
        spine.set_linewidth(0.55)


def draw_full_figure(tables: Fig5Tables, out_path: Path, top_n: int) -> None:
    """Draw the full five-panel Fig. 5."""
    setup_style()
    fig = plt.figure(figsize=(18.8, 11.4), dpi=300)
    fig.text(
        0.5,
        0.986,
        "Fig. 5 | Forecasting future research focus and key innovations from historical knowledge graphs",
        ha="center",
        va="top",
        fontsize=15.0,
        fontweight="bold",
    )
    subtitle = (
        "Holdout validation: pre-cutoff knowledge graphs are used to forecast "
        f"{tables.summary['validation_start']}-{tables.summary['validation_end_actual']} research hotspots and seed innovations"
    )
    fig.text(0.5, 0.958, subtitle, ha="center", va="top", fontsize=9.8, color=TEXT_MID, fontstyle="italic")
    gs = GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1.03, 1.02, 0.16],
        width_ratios=[1.04, 1.09],
        left=0.025,
        right=0.985,
        top=0.925,
        bottom=0.050,
        hspace=0.030,
        wspace=0.020,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_summary = fig.add_subplot(gs[2, :])
    draw_panel_a(ax_a, tables)
    draw_panel_b(ax_b, tables, top_n)
    draw_panel_c(ax_c, tables)
    draw_panel_d(ax_d, tables)
    draw_take_home(
        ax_summary,
        "By learning from how past landmark innovations reshaped the knowledge graph, "
        "the model highlights where the field may be heading and which pre-cutoff ideas are most likely to seed future breakthroughs.",
    )
    fig.text(
        0.5,
        0.016,
        "Note: Forecasts are topic-level aggregations of publication-day graph scores; realized hotspots combine post-cutoff growth, citation impact, graph-perturbation outcomes and landmark flags. Historical backtests are exported separately.",
        ha="center",
        va="bottom",
        fontsize=6.4,
        color=TEXT_DARK,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_panel_files(tables: Fig5Tables, out_dir: Path, top_n: int) -> None:
    """Draw individual panel PNG files."""
    panel_funcs = {
        "a": lambda ax: draw_panel_a(ax, tables),
        "b": lambda ax: draw_panel_b(ax, tables, top_n),
        "c": lambda ax: draw_panel_c(ax, tables),
        "d": lambda ax: draw_panel_d(ax, tables),
        "e": lambda ax: draw_panel_e(ax, tables),
    }
    for name, func in panel_funcs.items():
        setup_style()
        fig = plt.figure(figsize=(6.2, 4.1), dpi=300)
        ax = fig.add_subplot(111)
        func(ax)
        fig.savefig(out_dir / f"fig5_panel_{name}.png", dpi=300)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    fig3_root = default_fig3_root()
    parser = argparse.ArgumentParser(description="Build Fig. 5 forecast-outcome validation tables and figure.")
    parser.add_argument("--fig3-run-dir", type=Path, default=fig3_root / "multi_domain", help="Directory with fig3_score_table.csv.")
    parser.add_argument("--fig3-input-dir", type=Path, default=default_fig5_input_dir(fig3_root), help="Directory with works.csv and topics.csv.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--domain-filter", nargs="+", default=None, help="Optional domain names to keep, for example crispr.")
    parser.add_argument("--cutoff-year", type=int, default=2020, help="Last historical year visible to the forecast.")
    parser.add_argument("--validation-start", type=int, default=None, help="First validation year. Default: cutoff-year + 1.")
    parser.add_argument("--validation-end", type=int, default=2025, help="Requested final validation year. Use an explicit value for partial 2026 analysis.")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N focus list size for panels b/c.")
    parser.add_argument("--case-count", type=int, default=4, help="Number of key innovation case cards.")
    parser.add_argument("--min-historical-papers", type=int, default=5, help="Minimum pre-cutoff papers for a focus to be ranked.")
    parser.add_argument("--min-future-papers", type=int, default=2, help="Minimum validation-window papers for a focus to be ranked.")
    parser.add_argument("--semantic-threshold", type=float, default=0.42, help="Token Jaccard threshold for semantic focus matches.")
    parser.add_argument("--backtest-windows", nargs="+", default=DEFAULT_BACKTEST_WINDOWS, help="Backtest windows such as 2000:2005.")
    parser.add_argument("--formats", nargs="+", default=["png", "svg"], choices=["png", "svg", "pdf"], help="Full figure formats.")
    parser.add_argument("--skip-panel-files", action="store_true", help="Do not write individual panel PNG files.")
    parser.add_argument("--seed", type=int, default=2028, help="Deterministic seed for random baseline.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs.")
    return parser.parse_args()


def main() -> None:
    """Run the Fig. 5 pipeline."""
    args = parse_args()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    progress = not args.quiet
    progress_log(f"Loading Fig. 3 score table from {args.fig3_run_dir}", progress)
    data = load_data(args.fig3_run_dir, args.fig3_input_dir, args.domain_filter)
    progress_log(f"Loaded {len(data.papers):,} papers across {data.papers['focus_id'].nunique():,} foci.", progress)
    tables = compute_tables(args, data)
    progress_log(f"Writing Fig. 5 tables to {args.out_dir}", progress)
    write_outputs(args.out_dir, tables, args, data)
    for ext in args.formats:
        path = args.out_dir / f"fig5_full.{ext}"
        progress_log(f"Drawing full Fig. 5: {path}", progress)
        draw_full_figure(tables, path, args.top_n)
    if not args.skip_panel_files:
        progress_log("Drawing individual panel PNG files.", progress)
        draw_panel_files(tables, args.out_dir, args.top_n)
    for warning in tables.summary.get("warnings", []):
        progress_log(f"Warning: {warning}", progress)
    progress_log("Done.", progress)


if __name__ == "__main__":
    main()
