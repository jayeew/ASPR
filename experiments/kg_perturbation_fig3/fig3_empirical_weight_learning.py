#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 3 empirical pipeline: data-driven weighting of publication-day graph-perturbation indicators.

This script contains a complete, non-simulated pipeline for Fig. 3:
  1. Load real paper/citation/community data.
  2. Compute the seven publication-day indicators from G- and G0.
  3. Compute future graph-delta outcomes from G+tau.
  4. Construct RGPM = Realized Graph Perturbation Magnitude relative to matched controls.
  5. Learn non-negative simplex weights for the seven indicators by maximizing cross-validated
     rank association between the publication-day score and RGPM.
  6. Draw Fig. 3 panels a-f separately or as a merged figure.

No random placeholder data are generated. Randomness is used only for reproducible weight sampling
and cross-validation splits.

Required input files in --data-dir:
  works.csv
      Required columns, aliases accepted:
        id / paper_id / work_id
        year
        primary_field / field
        display_community / community
      Optional columns:
        title, domain, is_landmark, document_type

  citations.csv
      Required columns, aliases accepted:
        source / citing / citing_id   (the citing paper)
        target / cited / cited_id     (the cited paper)

Optional input files:
  topics.csv
      columns: community, label, x, y
      If absent, topic positions are computed from the community graph layout.

  topic_edges.csv
      columns: source_community, target_community, weight
      If absent, community edges are derived from paper-level citations.

Examples:
  python fig3_empirical_weight_learning_v3.py --data-dir ./data --out-dir ./fig3_out --panel all --export-tables
  python fig3_empirical_weight_learning_v3.py --data-dir ./data --out-dir ./fig3_out --panel c

Main-figure claims require data adequacy checks in addition to association checks:
at least four domains, thousands of papers per domain, enough landmark/high-RGPM
cases per domain, and stable matched controls. Single-domain runs are kept as
diagnostics because learned weights are otherwise easy to overfit to one field.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aspr_matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

try:
    from scipy.stats import rankdata, spearmanr
    from scipy.spatial.distance import jensenshannon
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# -----------------------------------------------------------------------------
# Constants and visual metadata
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIG1_DATA_ROOT = PROJECT_ROOT / "outputs" / "kg_perturbation_fig1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "kg_perturbation_fig3"
DEFAULT_DOMAIN = "crispr"
DEFAULT_DOMAINS = [
    "crispr",
    "graphene_2d_materials",
    "ipsc_reprogramming",
    "transformer_foundation_models",
]

TEXT_DARK = "#111827"
TEXT_MID = "#374151"
TEXT_LIGHT = "#6B7280"
BORDER = "#9CA3AF"
GRID = "#D1D5DB"
PANEL_FACE = "#FFFFFF"
PANEL_BORDER = "#6B7280"
PANEL_BORDER_WIDTH = 0.85
PANEL_RADIUS = 0.055

METRIC_SPECS = [
    ("B", "B", "#0B4FA3", "Bridge position"),
    ("RS", "RS", "#2E7D32", "Knowledge breadth"),
    ("DeltaQ0", "ΔQ0", "#F97316", "Boundary perturbation"),
    ("Uzzi", "Uzzi", "#7C3AED", "Atypical recombination"),
    ("RTD", "RTD", "#0891B2", "Reference target diversity"),
    ("BurtIP", "Burt IP", "#2563EB", "Structural holes"),
    ("PDE", "PDE", "#EF4444", "Prospective diffusion entropy"),
]
METRIC_KEYS = [m[0] for m in METRIC_SPECS]
METRIC_LABELS = {m[0]: m[1] for m in METRIC_SPECS}
METRIC_COLORS = {m[0]: m[2] for m in METRIC_SPECS}

DELTA_SPECS = [
    ("community_reach", "Community reach", "D1"),
    ("field_entropy", "Field entropy", "D2"),
    ("cross_community_adoption", "Cross-community adoption", "D3"),
    ("path_shortening", "Path shortening", "D4"),
    ("modularity_shock", "Modularity shock −ΔQ", "D5"),
    ("partition_change", "Partition divergence", "D6"),
    ("boundary_mixing", "Boundary mixing ↑", "D7"),
    ("post_perturbation_concentration", "Compression diagnostic", "D8"),
    ("hub_formation", "Hub formation ↑", "D9"),
]
DELTA_KEYS = [d[0] for d in DELTA_SPECS]
DELTA_LABELS = {d[0]: d[1] for d in DELTA_SPECS}
PRIMARY_RGPM_DELTA_KEYS = [
    "community_reach",
    "field_entropy",
    "cross_community_adoption",
    "path_shortening",
    "modularity_shock",
    "partition_change",
    "boundary_mixing",
    "hub_formation",
]
DELTA_FLOORS = {
    "cross_community_adoption": 0.02,
    "partition_change": 0.02,
    "boundary_mixing": 0.02,
    "post_perturbation_concentration": 0.02,
    "hub_formation": 0.02,
}
DEFAULT_DELTA_FLOOR = 1e-3
DELTA_GLOBAL_MAD_MIN = 1e-6
DELTA_NONZERO_MIN = 0.03
DELTA_CAP_HIT_DROP = 0.10
DELTA_CONTROL_MAD_ZERO_DROP = 0.50
CONTRIBUTING_DELTA_RELIABILITY_MIN = 0.25
CONTRIBUTING_DELTA_CONTROL_MAD_ZERO_MAX = 0.80
PAIR_SCAN_P95_MIN = 0.03
PAIR_SCAN_MAX_MIN = 0.08

MAIN_FIGURE_THRESHOLDS = {
    "active_graph_deltas_min": 6,
    "active_delta_z_cap_hit_rate_max": 0.10,
    "oof_spearman_min": 0.00,
    "learned_vs_equal_min": 0.03,
    "learned_vs_best_single_min": -1.00,
    "score_iqr_min": 0.35,
}

DATA_ADEQUACY_THRESHOLDS = {
    "domains_min": 4,
    "domains_target_max": 40,
    "papers_per_domain_min": 300,
    "total_papers_min": 5000,
    "landmark_or_high_cases_per_domain_min": 30,
    "high_perturbation_quantile": 0.90,
    "control_median_min": 50,
    "relaxed_control_tier_rate_max": 0.70,
}

INPUT_AUDIT_MAX_SUFFIX_DUPLICATE_RATE = 0.05
INPUT_AUDIT_MAX_PAIRWISE_SUFFIX_JACCARD = 0.50
INPUT_AUDIT_MAX_CROSS_DOMAIN_TITLE_DUPLICATE_RATE = 0.50

EXPECTED_DOMAIN_LANDMARKS: Dict[str, List[Dict[str, object]]] = {
    "crispr": [
        {"label": "Jinek 2012", "year": 2012, "doi": "10.1126/science.1225829", "title_exact": ["A Programmable Dual-RNA-Guided DNA Endonuclease in Adaptive Bacterial Immunity"], "title_contains": ["programmable", "dual rna", "dna endonuclease"]},
        {"label": "Cong/Mali 2013", "year": 2013, "doi": "10.1126/science.1231143", "title_exact": ["Multiplex Genome Engineering Using CRISPR/Cas Systems"], "title_contains": ["multiplex", "genome engineering", "crispr/cas"]},
        {"label": "Mali 2013", "year": 2013, "doi": "10.1126/science.1232033", "title_exact": ["RNA-Guided Human Genome Engineering via Cas9"], "title_contains": ["rna-guided", "human genome engineering"]},
    ],
    "graphene_2d_materials": [
        {"label": "Novoselov 2004", "year": 2004, "doi": "10.1126/science.1102896", "title_exact": ["Electric Field Effect in Atomically Thin Carbon Films"], "title_contains": ["electric field effect", "atomically thin carbon"]},
    ],
    "ipsc_reprogramming": [
        {"label": "Takahashi/Yamanaka 2006", "year": 2006, "doi": "10.1016/j.cell.2006.07.024", "title_exact": ["Induction of Pluripotent Stem Cells from Mouse Embryonic and Adult Fibroblast Cultures"], "title_contains": ["pluripotent stem cells", "mouse"]},
        {"label": "Yu 2007", "year": 2007, "doi": "10.1126/science.1151526", "title_exact": ["Induced Pluripotent Stem Cell Lines Derived from Human Somatic Cells"], "title_contains": ["induced pluripotent stem cell lines", "human somatic"]},
        {"label": "Takahashi 2007", "year": 2007, "doi": "10.1016/j.cell.2007.11.019", "title_exact": ["Induction of Pluripotent Stem Cells from Adult Human Fibroblasts by Defined Factors"], "title_contains": ["human somatic cells", "pluripotent stem cells"]},
    ],
    "transformer_foundation_models": [
        {"label": "Vaswani 2017", "year": 2017, "doi": "10.48550/arxiv.1706.03762", "title_exact": ["Attention Is All You Need"], "title_contains": ["attention is all you need"]},
        {"label": "BERT 2018", "year": 2018, "doi": "10.48550/arxiv.1810.04805", "title_exact": ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"], "title_contains": ["bert", "deep bidirectional transformers"]},
        {"label": "GPT-3 2020", "year": 2020, "doi": "10.48550/arxiv.2005.14165", "title_exact": ["Language Models are Few-Shot Learners"], "title_contains": ["language models are few-shot learners"]},
    ],
}

FIG2_EVIDENCE_CHANNELS = [
    ("Breadth", "#2E7D32"),
    ("Brokerage", "#0B4FA3"),
    ("Boundary", "#F97316"),
    ("Atypicality", "#7C3AED"),
]
FIG2_EVIDENCE_CHANNEL_ORDER = [name for name, _ in FIG2_EVIDENCE_CHANNELS]
FIG2_EVIDENCE_CHANNEL_COLORS = dict(FIG2_EVIDENCE_CHANNELS)
FIG2_EVIDENCE_MAP: Dict[str, Dict[str, float]] = {
    "B": {"Brokerage": 1.0, "Boundary": 0.50},
    "RS": {"Breadth": 1.0},
    "DeltaQ0": {"Boundary": 1.0},
    "Uzzi": {"Atypicality": 1.0, "Boundary": 0.50},
    "RTD": {"Brokerage": 1.0, "Breadth": 0.50},
    "BurtIP": {"Brokerage": 1.0},
    "PDE": {"Breadth": 1.0},
}
FIG2_CHANNEL_METRIC_POOLS: Dict[str, List[str]] = {
    "Breadth": ["RS", "PDE", "RTD"],
    "Brokerage": ["B", "RTD", "BurtIP"],
    "Boundary": ["DeltaQ0", "B", "Uzzi"],
    "Atypicality": ["Uzzi"],
}

DELTA_SCALE_MULTIPLIERS = {
    "cross_community_adoption": 1.25,
    "path_shortening": 1.75,
    "modularity_shock": 2.0,
}

RELAXED_CONTROL_TIERS = {"field_all_years", "all_non_landmark"}

PAIRWISE_LANDSCAPES = [
    ("B", "RTD"),
    ("DeltaQ0", "Uzzi"),
    ("RS", "PDE"),
]

NORMAL_DIST = NormalDist()


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class RawData:
    works: pd.DataFrame
    citations: pd.DataFrame
    topics: pd.DataFrame
    topic_edges: pd.DataFrame
    analysis_end_year: int


@dataclass
class ComputedData:
    paper_metrics: pd.DataFrame
    graph_deltas: pd.DataFrame
    rgpm_table: pd.DataFrame
    weight_samples: pd.DataFrame
    best_weights: pd.Series
    best_performance: float
    score_table: pd.DataFrame
    cv_summary: pd.DataFrame
    panel_b_example: pd.DataFrame
    active_delta_keys: List[str]
    active_metric_keys: List[str]
    delta_diagnostics: pd.DataFrame
    feature_diagnostics: pd.DataFrame
    model_diagnostics: pd.DataFrame
    control_diagnostics: pd.DataFrame
    domain_diagnostics: pd.DataFrame
    indicator_target_correlations: pd.DataFrame
    rgpm_component_correlations: pd.DataFrame
    control_tier_audit: pd.DataFrame
    nonlinear_diagnostics: pd.DataFrame
    target_sensitivity: pd.DataFrame
    landmark_validation: pd.DataFrame
    diagnostics_summary: Dict[str, Any]
    fold_weights: pd.DataFrame
    baseline_comparison: pd.DataFrame
    profile_grid_size: int
    profile_n: int
    pair_scan_results: pd.DataFrame
    effect_summary: pd.DataFrame
    selected_panel_d_pairs: List[Tuple[str, str]]


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def setup_style() -> None:
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


def blend_with_white(color: str, amount: float = 0.86) -> str:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    out = rgb * (1.0 - amount) + np.ones(3) * amount
    return mcolors.to_hex(out)


def wrap_text(text: str, width: int) -> str:
    if not text:
        return ""
    import textwrap
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=False))


def stable_int_id(value: object, modulo: int = 1_000_000_000) -> int:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % int(modulo)


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


def rectangle_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    facecolor: str = "white",
    edgecolor: str = BORDER,
    linewidth: float = 0.8,
    linestyle: str = "-",
    alpha: float = 1.0,
    zorder: int = 1,
) -> mpatches.Rectangle:
    patch = mpatches.Rectangle(
        (x, y),
        w,
        h,
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
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rounded_box(ax, 0.0, 0.0, 1.0, 1.0, PANEL_FACE, PANEL_BORDER, PANEL_BORDER_WIDTH, PANEL_RADIUS, zorder=0)
    ax.text(0.022, 0.965, label, ha="left", va="top", fontsize=16, fontweight="bold")
    ax.text(0.100, 0.955, title, ha="left", va="top", fontsize=9.4, fontweight="bold")


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
    height: float = 0.055,
    fontsize: float = 7.0,
    facecolor: Optional[str] = None,
    fontweight: str = "bold",
    zorder: int = 5,
) -> None:
    fill = facecolor or blend_with_white(color, 0.88)
    rounded_box(ax, x, y, width, height, fill, color, 0.80, radius=height * 0.45, zorder=zorder)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color if fontweight == "bold" else TEXT_DARK,
        fontweight=fontweight,
        zorder=zorder + 1,
    )


def robust_mad(values: Sequence[float], eps: float = 1e-9) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return eps
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return float(max(1.4826 * mad, eps))


def raw_mad(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    med = np.median(arr)
    return float(1.4826 * np.median(np.abs(arr - med)))


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return np.nan
    if np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return np.nan
    if SCIPY_OK:
        return float(spearmanr(x[mask], y[mask]).correlation)
    return float(pd.Series(x[mask]).corr(pd.Series(y[mask]), method="spearman"))


def bootstrap_spearman_ci(
    x: Sequence[float],
    y: Sequence[float],
    seed: int,
    n_boot: int = 400,
) -> Tuple[float, float]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 8:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(x_arr), size=len(x_arr))
        rho = safe_spearman(x_arr[idx], y_arr[idx])
        if np.isfinite(rho):
            vals.append(float(rho))
    if len(vals) < 8:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_mean_se(values: Sequence[float], seed: int, n_boot: int = 300) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(arr), size=len(arr))
        means.append(float(np.mean(arr[idx])))
    return float(np.std(means, ddof=1)) if len(means) > 1 else float("nan")


def rank_normal_scores(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    mask = np.isfinite(arr)
    n = int(mask.sum())
    if n == 0:
        return out
    if n == 1 or np.nanstd(arr[mask]) < 1e-12:
        out[mask] = 0.0
        return out
    if SCIPY_OK:
        ranks = rankdata(arr[mask], method="average")
    else:
        ranks = pd.Series(arr[mask]).rank(method="average").to_numpy(dtype=float)
    p = np.clip((ranks - 0.5) / n, 1e-6, 1.0 - 1e-6)
    z = np.asarray([NORMAL_DIST.inv_cdf(float(v)) for v in p], dtype=float)
    out[mask] = np.clip(z, -3.0, 3.0)
    return out


def winsorize(values: Sequence[float], lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() < 4:
        return arr
    lo, hi = np.nanquantile(arr[mask], [lower, upper])
    out = arr.copy()
    out[mask] = np.clip(out[mask], lo, hi)
    return out


def shannon_entropy(values: Sequence[object]) -> float:
    s = pd.Series([v for v in values if pd.notna(v)])
    if s.empty:
        return 0.0
    counts = s.value_counts().to_numpy(dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def simpson_diversity(values: Sequence[object]) -> float:
    s = pd.Series([v for v in values if pd.notna(v)])
    if s.empty:
        return 0.0
    counts = s.value_counts().to_numpy(dtype=float)
    p = counts / counts.sum()
    return float(1.0 - np.square(p).sum())


def normalized_hhi(values: Sequence[object]) -> float:
    s = pd.Series([v for v in values if pd.notna(v)])
    if s.empty:
        return 0.0
    counts = s.value_counts().to_numpy(dtype=float)
    p = counts / counts.sum()
    return float(np.square(p).sum())


def js_divergence(values_a: Sequence[object], values_b: Sequence[object]) -> float:
    cats = sorted(set([v for v in values_a if pd.notna(v)]) | set([v for v in values_b if pd.notna(v)]))
    if len(cats) == 0:
        return 0.0
    ca = pd.Series(values_a).value_counts().to_dict()
    cb = pd.Series(values_b).value_counts().to_dict()
    pa = np.asarray([ca.get(c, 0) for c in cats], dtype=float)
    pb = np.asarray([cb.get(c, 0) for c in cats], dtype=float)
    if pa.sum() == 0 or pb.sum() == 0:
        return 0.0
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    if SCIPY_OK:
        return float(jensenshannon(pa, pb, base=2.0) ** 2)
    m = 0.5 * (pa + pb)
    kl1 = np.sum(np.where(pa > 0, pa * np.log2(pa / np.maximum(m, 1e-12)), 0.0))
    kl2 = np.sum(np.where(pb > 0, pb * np.log2(pb / np.maximum(m, 1e-12)), 0.0))
    return float(0.5 * (kl1 + kl2))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with canonical column names where aliases are accepted."""
    out = df.copy()
    aliases = {
        "paper_id": "id",
        "work_id": "id",
        "openalex_id": "id",
        "field": "primary_field",
        "display_field": "primary_field",
        "community": "display_community",
        "topic_community": "display_community",
        "citing": "source",
        "citing_id": "source",
        "source_id": "source",
        "cited": "target",
        "cited_id": "target",
        "target_id": "target",
    }
    for old, new in aliases.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})
    return out


def require_columns(df: pd.DataFrame, required: Sequence[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {missing}")


def progress_log(message: str, enabled: bool = True) -> None:
    if enabled:
        print(f"[fig3] {message}", flush=True)


# -----------------------------------------------------------------------------
# Data loading and preprocessing
# -----------------------------------------------------------------------------

STANDARD_INPUT_FILES = ("works.csv", "citations.csv")
OPTIONAL_STANDARD_INPUT_FILES = ("topics.csv", "topic_edges.csv")
FIG1_EXPORT_FILES = ("works_selected.csv", "paper_edges.csv", "topic_nodes.csv", "topic_edges.csv")


def has_standard_input_files(data_dir: Path) -> bool:
    return all((data_dir / name).exists() for name in STANDARD_INPUT_FILES)


def has_fig1_export_files(data_dir: Path) -> bool:
    return all((data_dir / name).exists() for name in FIG1_EXPORT_FILES)


def resolve_standard_data_dir(data_dir: Path, domain: Optional[str]) -> Optional[Path]:
    if has_standard_input_files(data_dir):
        return data_dir
    if domain and has_standard_input_files(data_dir / domain):
        return data_dir / domain
    return None


def resolve_fig1_domain_dir(data_dir: Path, domain: Optional[str]) -> Optional[Path]:
    if has_fig1_export_files(data_dir):
        return data_dir
    if domain and has_fig1_export_files(data_dir / domain):
        return data_dir / domain
    return None


def read_works_raw_refs(path: Path) -> Tuple[Dict[str, List[str]], int]:
    refs_by_id: Dict[str, List[str]] = {}
    n_records = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            paper_id = rec.get("id")
            if not paper_id:
                continue
            refs_by_id[str(paper_id)] = [str(ref) for ref in rec.get("refs") or [] if ref]
            n_records += 1
    return refs_by_id, n_records


def write_raw_data(raw: RawData, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.works.to_csv(out_dir / "works.csv", index=False)
    raw.citations.to_csv(out_dir / "citations.csv", index=False)
    raw.topics.to_csv(out_dir / "topics.csv", index=False)
    raw.topic_edges.to_csv(out_dir / "topic_edges.csv", index=False)


def save_prepare_report(report: Mapping[str, Any], out_dir: Path) -> None:
    (out_dir / "fig3_input_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def strip_namespace_prefix(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return text.split("::", 1)[1] if "::" in text else text


def normalize_doi_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return ""
    for prefix in ["https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.strip().rstrip("/")


def normalize_title_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def append_anchor_label(existing: object, label: str) -> str:
    labels = [
        part.strip()
        for part in str(existing if existing is not None else "").split(";")
        if part.strip() and part.strip().lower() not in {"nan", "none", "null"}
    ]
    if label not in labels:
        labels.append(label)
    return "; ".join(labels)


def ensure_expected_landmarks(raw: RawData, domain: str, progress: bool = True) -> RawData:
    """Mark configured landmarks and add explicit audit-only placeholders when absent."""
    specs = EXPECTED_DOMAIN_LANDMARKS.get(domain, [])
    if not specs:
        setattr(raw, "landmark_audit", pd.DataFrame())
        return raw

    works = raw.works.copy()
    if "anchor_label" not in works.columns:
        works["anchor_label"] = ""
    if "doi" not in works.columns:
        works["doi"] = ""
    if "landmark_source" not in works.columns:
        works["landmark_source"] = ""
    works["is_landmark"] = pd.to_numeric(works.get("is_landmark", 0), errors="coerce").fillna(0).astype(int)
    title_key = works["title"].map(normalize_title_key) if "title" in works.columns else pd.Series("", index=works.index)
    doi_key = works["doi"].map(normalize_doi_value)
    audit_rows: List[Dict[str, object]] = []

    for spec in specs:
        label = str(spec["label"])
        doi = normalize_doi_value(spec.get("doi", ""))
        expected_year = int(spec.get("year", -1))
        phrases = [normalize_title_key(p) for p in spec.get("title_contains", [])]
        exact_titles = [normalize_title_key(p) for p in spec.get("title_exact", [])]
        doi_mask = pd.Series(False, index=works.index)
        if doi:
            doi_mask = doi_key == doi
        exact_title_mask = title_key.isin([title for title in exact_titles if title])
        phrase_title_mask = pd.Series(False, index=works.index)
        title_phrases = [phrase for phrase in phrases if phrase] if not exact_titles else []
        if title_phrases:
            phrase_title_mask = pd.Series(True, index=works.index)
            for phrase in title_phrases:
                phrase_title_mask = phrase_title_mask & title_key.str.contains(re.escape(phrase), na=False)
        mask = doi_mask | exact_title_mask | phrase_title_mask
        if expected_year > 0 and mask.any():
            years = pd.to_numeric(works.loc[mask, "year"], errors="coerce")
            year_near = years.between(expected_year - 1, expected_year + 1)
            if year_near.any():
                mask.loc[mask[mask].index] = year_near.to_numpy(dtype=bool)

        matched_idx = works.index[mask].tolist()
        status = "matched"
        method = "doi_or_title"
        if not matched_idx and domain == "transformer_foundation_models" and label in {"Vaswani 2017", "BERT 2018"}:
            synthetic_id = f"configured_landmark::{domain}::{doi or normalize_title_key(label)}"
            row = {
                "id": synthetic_id,
                "year": expected_year,
                "title": str(spec.get("title", label)),
                "domain": domain,
                "primary_field": domain,
                "display_community": int(pd.to_numeric(works["display_community"], errors="coerce").dropna().mode().iloc[0])
                if "display_community" in works.columns and not pd.to_numeric(works["display_community"], errors="coerce").dropna().empty
                else -1,
                "is_landmark": 1,
                "anchor_label": label,
                "doi": doi,
                "landmark_source": "configured_missing_landmark_no_local_edges",
            }
            for col in works.columns:
                row.setdefault(col, np.nan)
            works = pd.concat([works, pd.DataFrame([row], columns=works.columns)], ignore_index=True)
            title_key = works["title"].map(normalize_title_key)
            doi_key = works["doi"].map(normalize_doi_value)
            matched_idx = [int(works.index[-1])]
            status = "configured_placeholder"
            method = "configured_doi_for_missing_transformer_landmark"
            progress_log(
                f"[{domain}] Added configured placeholder for missing transformer landmark: {label}. "
                "It has no local citation edges and is audit-only unless source data provide references.",
                progress,
            )
        elif not matched_idx:
            status = "missing"
            method = "not_found_in_input"

        for idx in matched_idx:
            works.loc[idx, "is_landmark"] = 1
            works.loc[idx, "anchor_label"] = append_anchor_label(works.loc[idx, "anchor_label"], label)
            source = str(works.loc[idx, "landmark_source"] or "")
            if not source:
                works.loc[idx, "landmark_source"] = "expected_landmark_config"

        matched_titles = "; ".join(works.loc[matched_idx, "title"].astype(str).head(3).tolist()) if matched_idx else ""
        matched_ids = "; ".join(works.loc[matched_idx, "id"].astype(str).head(3).tolist()) if matched_idx else ""
        audit_rows.append(
            {
                "audit_type": "expected_landmark",
                "domain": domain,
                "label": label,
                "expected_year": expected_year,
                "expected_doi": doi,
                "status": status,
                "match_method": method,
                "n_matches": int(len(matched_idx)),
                "matched_ids": matched_ids,
                "matched_titles": matched_titles,
            }
        )

    raw.works = works
    setattr(raw, "landmark_audit", pd.DataFrame(audit_rows))
    return raw


def build_input_audit(raw: RawData, run_name: str) -> pd.DataFrame:
    works = raw.works.copy()
    if works.empty:
        return pd.DataFrame(
            [
                {
                    "audit_type": "input_integrity",
                    "run_name": run_name,
                    "check": "works_nonempty",
                    "status": "fail",
                    "value": 0,
                    "threshold": 1,
                    "details": "works.csv has no rows",
                }
            ]
        )
    works["domain"] = works.get("domain", "domain")
    works["domain"] = works["domain"].fillna("unknown").astype(str)
    works["_id_suffix"] = works["id"].map(strip_namespace_prefix)
    works["_title_key"] = works["title"].map(normalize_title_key) if "title" in works.columns else ""
    rows: List[Dict[str, object]] = []

    suffix_duplicate_rate = 1.0 - float(works["_id_suffix"].nunique()) / max(float(len(works)), 1.0)
    suffix_status = "pass" if suffix_duplicate_rate <= INPUT_AUDIT_MAX_SUFFIX_DUPLICATE_RATE else "fail"
    rows.append(
        {
            "audit_type": "input_integrity",
            "run_name": run_name,
            "check": "id_suffix_duplicate_rate",
            "domain": "all",
            "status": suffix_status,
            "value": suffix_duplicate_rate,
            "threshold": INPUT_AUDIT_MAX_SUFFIX_DUPLICATE_RATE,
            "details": f"{works['_id_suffix'].nunique():,} unique suffixes / {len(works):,} rows",
        }
    )

    domains = sorted(works["domain"].unique().tolist())
    max_jaccard = 0.0
    worst_pair = ""
    suffix_sets = {
        domain: set(sub["_id_suffix"].astype(str))
        for domain, sub in works.groupby("domain", sort=True)
    }
    for i, left in enumerate(domains):
        for right in domains[i + 1:]:
            union = suffix_sets[left] | suffix_sets[right]
            jaccard = float(len(suffix_sets[left] & suffix_sets[right]) / len(union)) if union else 0.0
            if jaccard > max_jaccard:
                max_jaccard = jaccard
                worst_pair = f"{left} vs {right}"
    rows.append(
        {
            "audit_type": "input_integrity",
            "run_name": run_name,
            "check": "max_pairwise_id_suffix_jaccard",
            "domain": "all",
            "status": "pass" if max_jaccard <= INPUT_AUDIT_MAX_PAIRWISE_SUFFIX_JACCARD else "fail",
            "value": max_jaccard,
            "threshold": INPUT_AUDIT_MAX_PAIRWISE_SUFFIX_JACCARD,
            "details": worst_pair or "single domain",
        }
    )

    title_domains = works.loc[works["_title_key"].astype(str) != "", ["_title_key", "domain"]].drop_duplicates()
    if not title_domains.empty:
        repeated_titles = title_domains.groupby("_title_key")["domain"].nunique()
        cross_title_rate = float((repeated_titles > 1).mean())
    else:
        cross_title_rate = 0.0
    rows.append(
        {
            "audit_type": "input_integrity",
            "run_name": run_name,
            "check": "cross_domain_title_duplicate_rate",
            "domain": "all",
            "status": "pass" if cross_title_rate <= INPUT_AUDIT_MAX_CROSS_DOMAIN_TITLE_DUPLICATE_RATE else "fail",
            "value": cross_title_rate,
            "threshold": INPUT_AUDIT_MAX_CROSS_DOMAIN_TITLE_DUPLICATE_RATE,
            "details": f"{len(title_domains):,} unique title-domain pairs",
        }
    )

    for domain, sub in works.groupby("domain", sort=True):
        domain_suffix_rate = 1.0 - float(sub["_id_suffix"].nunique()) / max(float(len(sub)), 1.0)
        landmark_count = int(pd.to_numeric(sub.get("is_landmark", 0), errors="coerce").fillna(0).astype(int).sum())
        rows.append(
            {
                "audit_type": "domain_summary",
                "run_name": run_name,
                "check": "domain_rows_and_landmarks",
                "domain": domain,
                "status": "pass" if len(sub) > 0 and domain_suffix_rate <= INPUT_AUDIT_MAX_SUFFIX_DUPLICATE_RATE else "fail",
                "value": int(len(sub)),
                "threshold": 1,
                "id_suffix_duplicate_rate": domain_suffix_rate,
                "n_landmarks": landmark_count,
                "details": f"{len(sub):,} works; {landmark_count:,} landmarks",
            }
        )

    landmark_audit = getattr(raw, "landmark_audit", pd.DataFrame())
    if isinstance(landmark_audit, pd.DataFrame) and not landmark_audit.empty:
        rows.extend(landmark_audit.to_dict("records"))
    return pd.DataFrame(rows)


def validate_input_audit_or_raise(raw: RawData, run_name: str) -> pd.DataFrame:
    audit = build_input_audit(raw, run_name)
    setattr(raw, "input_audit", audit)
    blocking_checks = {
        "id_suffix_duplicate_rate",
        "max_pairwise_id_suffix_jaccard",
        "cross_domain_title_duplicate_rate",
    }
    failed = audit[
        (audit.get("audit_type", "") == "input_integrity")
        & audit.get("check", "").isin(blocking_checks)
        & (audit.get("status", "") == "fail")
    ]
    if not failed.empty:
        details = "; ".join(
            f"{row.check}={float(row.value):.3f}>{float(row.threshold):.3f} ({row.details})"
            for row in failed.itertuples(index=False)
        )
        raise ValueError(f"Fig. 3 input audit failed for {run_name}: {details}")
    return audit


def write_input_audit(raw: RawData, out_dir: Path) -> None:
    audit = getattr(raw, "input_audit", pd.DataFrame())
    if isinstance(audit, pd.DataFrame) and not audit.empty:
        out_dir.mkdir(parents=True, exist_ok=True)
        audit.to_csv(out_dir / "fig3_input_audit.csv", index=False)


def infer_primary_field(row: pd.Series) -> str:
    for col in ["display_label", "community_label", "primary_topic", "domain"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    return "unknown_field"


def standardize_fig1_works(works_selected: pd.DataFrame, domain_name: str) -> pd.DataFrame:
    works = works_selected.copy()
    if "id" not in works.columns:
        raise ValueError("works_selected.csv missing required column: id")
    works["domain"] = domain_name
    works["primary_field"] = works.apply(infer_primary_field, axis=1)
    if "display_community" not in works.columns:
        works["display_community"] = pd.NA
    community_fallback = works["community"] if "community" in works.columns else pd.Series(-1, index=works.index)
    works["display_community"] = pd.to_numeric(works["display_community"], errors="coerce")
    works["display_community"] = works["display_community"].fillna(pd.to_numeric(community_fallback, errors="coerce"))
    works["display_community"] = works["display_community"].fillna(-1).astype(int)
    if "anchor_label" not in works.columns:
        works["anchor_label"] = pd.NA
    anchor_label = works["anchor_label"].replace("", pd.NA)
    works["anchor_label"] = anchor_label
    works["is_landmark"] = (anchor_label.notna() & (anchor_label.astype(str).str.strip() != "")).astype(int)
    if "title" not in works.columns:
        works["title"] = works["id"]
    base_cols = [
        "id", "year", "title", "domain", "primary_field", "display_community",
        "is_landmark", "anchor_label",
    ]
    extra_cols = [
        col for col in [
            "short_id", "doi", "cited_by_count", "primary_topic", "community",
            "community_label", "display_label",
        ]
        if col in works.columns
    ]
    return works[base_cols + extra_cols].copy()


def standardize_fig1_raw_works(fig1_dir: Path, domain_name: str) -> pd.DataFrame:
    """Build a larger Fig. 3 corpus from Fig. 1 works_raw.jsonl records."""
    refs_path = fig1_dir / "works_raw.jsonl"
    if not refs_path.exists():
        raise FileNotFoundError(f"Missing works_raw.jsonl for raw Fig. 3 corpus: {refs_path}")
    rows: List[Dict[str, object]] = []
    with refs_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            paper_id = rec.get("id")
            year = rec.get("year")
            if not paper_id or year is None:
                continue
            topics = rec.get("topics") or []
            primary_topic = rec.get("primary_topic") or (topics[0] if topics else "unknown_topic")
            anchor_label = rec.get("anchor_label") or ""
            rows.append(
                {
                    "id": str(paper_id),
                    "year": int(year),
                    "title": rec.get("title") or str(paper_id),
                    "domain": domain_name,
                    "primary_field": str(primary_topic),
                    "primary_topic": str(primary_topic),
                    "is_landmark": int(bool(str(anchor_label).strip())),
                    "anchor_label": anchor_label,
                    "cited_by_count": rec.get("cited_by_count", np.nan),
                    "doi": rec.get("doi", ""),
                    "short_id": rec.get("short_id", ""),
                }
            )
    works = pd.DataFrame(rows).drop_duplicates(subset=["id"])
    if works.empty:
        raise ValueError(f"No usable raw works found in {refs_path}")
    codes, uniques = pd.factorize(works["primary_field"].astype(str), sort=True)
    works["display_community"] = codes.astype(int)
    label_map = {int(i): str(label) for i, label in enumerate(uniques)}
    works["community_label"] = works["display_community"].map(label_map)
    return works[
        [
            "id", "year", "title", "domain", "primary_field", "display_community",
            "is_landmark", "anchor_label", "short_id", "doi", "cited_by_count",
            "primary_topic", "community_label",
        ]
    ].copy()


def citations_from_fig1_raw_refs(fig1_dir: Path, selected_ids: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    refs_path = fig1_dir / "works_raw.jsonl"
    if not refs_path.exists():
        return pd.DataFrame(columns=["source", "target"]), {"raw_refs_available": False}

    refs_by_id, n_raw_records = read_works_raw_refs(refs_path)
    selected = set(str(item) for item in selected_ids)
    rows = []
    dropped_unselected_targets = 0
    missing_source_records = 0
    for source in selected:
        refs = refs_by_id.get(source)
        if refs is None:
            missing_source_records += 1
            continue
        for target in refs:
            if target in selected:
                rows.append({"source": source, "target": target})
            else:
                dropped_unselected_targets += 1

    citations = pd.DataFrame(rows, columns=["source", "target"]).drop_duplicates()
    report = {
        "raw_refs_available": True,
        "works_raw_jsonl": str(refs_path),
        "works_raw_records": n_raw_records,
        "selected_sources_missing_in_raw": missing_source_records,
        "dropped_refs_to_unselected_targets": dropped_unselected_targets,
        "citation_rows_from_raw_refs": len(citations),
    }
    return citations, report


def topics_from_works_and_citations(works: pd.DataFrame, citations: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    topic_edges = derive_topic_edges(works, citations)
    labels = (
        works[["display_community", "primary_field"]]
        .drop_duplicates(subset=["display_community"])
        .rename(columns={"display_community": "community", "primary_field": "label"})
        .sort_values("community")
    )
    topics = compute_topic_positions(topic_edges, labels)
    return topics, topic_edges


def citations_from_fig1_paper_edges(fig1_dir: Path, direct_only: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    edge_path = fig1_dir / "paper_edges.csv"
    edges = pd.read_csv(edge_path)
    source_rows = len(edges)
    if direct_only and "direct" in edges.columns:
        edges = edges[pd.to_numeric(edges["direct"], errors="coerce").fillna(0) > 0].copy()
    citations = edges[["source", "target"]].drop_duplicates().copy()
    report = {
        "paper_edges_csv": str(edge_path),
        "paper_edge_rows": source_rows,
        "direct_only": direct_only,
        "citation_rows_from_paper_edges": len(citations),
    }
    return citations, report


def prepare_fig3_input_from_fig1(
    fig1_dir: Path,
    prepared_dir: Path,
    direct_only: bool,
    analysis_end_year: Optional[int],
    corpus_source: str,
    progress: bool,
) -> Path:
    domain_name = fig1_dir.name
    progress_log(f"Preparing Fig. 3 input from Fig. 1 exports: {fig1_dir} (corpus_source={corpus_source})", progress)
    if corpus_source == "raw":
        works = standardize_fig1_raw_works(fig1_dir, domain_name)
    else:
        works_selected = pd.read_csv(fig1_dir / "works_selected.csv")
        works = standardize_fig1_works(works_selected, domain_name)
    selected_ids = works["id"].astype(str).tolist()

    citations, citation_report = citations_from_fig1_raw_refs(fig1_dir, selected_ids)
    citation_source = "works_raw.jsonl"
    if citations.empty:
        progress_log("No usable raw reference citations found; falling back to paper_edges.csv.", progress)
        citations, citation_report = citations_from_fig1_paper_edges(fig1_dir, direct_only=direct_only)
        citation_source = "paper_edges.csv"

    if corpus_source == "raw":
        topics, topic_edges = topics_from_works_and_citations(works, citations)
    else:
        topics = pd.read_csv(fig1_dir / "topic_nodes.csv")
        topic_edges = pd.read_csv(fig1_dir / "topic_edges.csv")
    end_year = int(analysis_end_year or pd.to_numeric(works["year"], errors="coerce").max())
    raw = RawData(works=works, citations=citations, topics=topics, topic_edges=topic_edges, analysis_end_year=end_year)
    write_raw_data(raw, prepared_dir)

    report: Dict[str, Any] = {
        "source_kind": "fig1_exports",
        "source_dir": str(fig1_dir),
        "prepared_dir": str(prepared_dir),
        "fig1_corpus_source": corpus_source,
        "citation_source": citation_source,
        "works_rows": len(works),
        "citation_rows": len(citations),
        "topic_rows": len(topics),
        "topic_edge_rows": len(topic_edges),
        "landmark_rows": int(works["is_landmark"].sum()),
        "analysis_end_year": end_year,
        **citation_report,
    }
    save_prepare_report(report, prepared_dir)
    progress_log(
        f"Prepared Fig. 3 input in {prepared_dir}: {len(works):,} works, "
        f"{len(citations):,} citations from {citation_source}, {int(works['is_landmark'].sum()):,} landmarks",
        progress,
    )
    return prepared_dir


def prepare_fig3_input_from_standard(
    source_dir: Path,
    prepared_dir: Path,
    analysis_end_year: Optional[int],
    progress: bool,
) -> Path:
    progress_log(f"Normalizing standard Fig. 3 input from {source_dir}", progress)
    raw = load_raw_data(source_dir, analysis_end_year=analysis_end_year)
    write_raw_data(raw, prepared_dir)
    report = {
        "source_kind": "standard_fig3_input",
        "source_dir": str(source_dir),
        "prepared_dir": str(prepared_dir),
        "works_rows": len(raw.works),
        "citation_rows": len(raw.citations),
        "topic_rows": len(raw.topics),
        "topic_edge_rows": len(raw.topic_edges),
        "landmark_rows": int(raw.works["is_landmark"].sum()),
        "analysis_end_year": int(raw.analysis_end_year),
    }
    save_prepare_report(report, prepared_dir)
    progress_log(f"Prepared normalized Fig. 3 input in {prepared_dir}", progress)
    return prepared_dir


def default_fig1_config_for_domain(domain: Optional[str]) -> Optional[Path]:
    if not domain:
        return None
    path = PROJECT_ROOT / "experiments" / "kg_perturbation_fig1" / "configs" / f"{domain}.yaml"
    return path if path.exists() else None


def load_fig1_module() -> Any:
    module_path = PROJECT_ROOT / "experiments" / "kg_perturbation_fig1" / "fig1_knowledge_perturbation_v3.py"
    spec = importlib.util.spec_from_file_location("fig1_knowledge_perturbation_v3", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Fig. 1 module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_fig1_pipeline_for_input(
    fig1_config: Path,
    fig1_out_dir: Path,
    use_cache: bool,
    openalex_api_key: Optional[str],
    email: Optional[str],
    progress: bool,
) -> Path:
    progress_log(f"Running Fig. 1 pipeline to materialize source data: {fig1_config}", progress)
    fig1 = load_fig1_module()
    cfg = fig1.load_config(fig1_config)
    api_cfg = cfg.get("api", {})
    client = fig1.OpenAlexClient(
        api_key=openalex_api_key,
        email=email,
        sleep_seconds=float(api_cfg.get("sleep_seconds", 0.1)),
        max_retries=int(api_cfg.get("max_retries", 6)),
        timeout_seconds=int(api_cfg.get("timeout_seconds", 60)),
    )
    fig1.run_domain(cfg, client, fig1_out_dir, use_cache=use_cache)
    return fig1_out_dir / cfg["slug"]


def prepare_fig3_input_data(
    data_dir: Path,
    out_dir: Path,
    domain: Optional[str],
    direct_only: bool,
    analysis_end_year: Optional[int],
    fig1_config: Optional[Path],
    fig1_corpus_source: str,
    run_fig1_if_missing: bool,
    use_fig1_cache: bool,
    openalex_api_key: Optional[str],
    email: Optional[str],
    progress: bool,
) -> Path:
    prepared_domain = domain or data_dir.name
    prepared_dir = out_dir / "fig3_input" / prepared_domain

    standard_dir = resolve_standard_data_dir(data_dir, domain)
    if standard_dir is not None:
        return prepare_fig3_input_from_standard(
            standard_dir,
            prepared_dir,
            analysis_end_year=analysis_end_year,
            progress=progress,
        )

    fig1_dir = resolve_fig1_domain_dir(data_dir, domain)
    if fig1_dir is None and run_fig1_if_missing:
        config_path = fig1_config or default_fig1_config_for_domain(domain)
        if config_path is None:
            raise FileNotFoundError("No Fig. 1 config provided and no default config exists for this domain.")
        fig1_dir = run_fig1_pipeline_for_input(
            config_path,
            out_dir / "fig1_source",
            use_cache=use_fig1_cache,
            openalex_api_key=openalex_api_key,
            email=email,
            progress=progress,
        )

    if fig1_dir is not None:
        return prepare_fig3_input_from_fig1(
            fig1_dir,
            prepared_dir,
            direct_only=direct_only,
            analysis_end_year=analysis_end_year,
            corpus_source=fig1_corpus_source,
            progress=progress,
        )

    raise FileNotFoundError(
        f"Cannot prepare Fig. 3 input from {data_dir}. Expected standard Fig. 3 files "
        f"{STANDARD_INPUT_FILES}, Fig. 1 exports {FIG1_EXPORT_FILES}, or use --run-fig1-if-missing."
    )

def load_raw_data(data_dir: Path, analysis_end_year: Optional[int] = None) -> RawData:
    works_path = data_dir / "works.csv"
    citations_path = data_dir / "citations.csv"
    if not works_path.exists():
        raise FileNotFoundError(f"Missing required file: {works_path}")
    if not citations_path.exists():
        raise FileNotFoundError(f"Missing required file: {citations_path}")

    works = normalize_columns(pd.read_csv(works_path))
    citations = normalize_columns(pd.read_csv(citations_path))
    require_columns(works, ["id", "year", "primary_field", "display_community"], "works.csv")
    require_columns(citations, ["source", "target"], "citations.csv")

    works = works.copy()
    citations = citations.copy()
    works["id"] = works["id"].astype(str)
    citations["source"] = citations["source"].astype(str)
    citations["target"] = citations["target"].astype(str)
    works["year"] = pd.to_numeric(works["year"], errors="coerce")
    works = works[works["year"].notna()].copy()
    works["year"] = works["year"].astype(int)
    works["display_community"] = pd.to_numeric(works["display_community"], errors="coerce")
    works = works[works["display_community"].notna()].copy()
    works["display_community"] = works["display_community"].astype(int)
    works["primary_field"] = works["primary_field"].astype(str)
    if "title" not in works.columns:
        works["title"] = works["id"]
    if "domain" not in works.columns:
        works["domain"] = "domain"
    if "is_landmark" not in works.columns:
        works["is_landmark"] = 0
    works["is_landmark"] = pd.to_numeric(works["is_landmark"], errors="coerce").fillna(0).astype(int)

    valid_ids = set(works["id"])
    citations = citations[citations["source"].isin(valid_ids) & citations["target"].isin(valid_ids)].copy()

    if analysis_end_year is None:
        analysis_end_year = int(works["year"].max())

    # topics.csv optional; if absent, derive community positions from community graph.
    topics_path = data_dir / "topics.csv"
    topic_edges_path = data_dir / "topic_edges.csv"
    if topic_edges_path.exists():
        topic_edges = normalize_columns(pd.read_csv(topic_edges_path))
        # support topic edge aliases
        if "source" in topic_edges.columns and "source_community" not in topic_edges.columns:
            topic_edges = topic_edges.rename(columns={"source": "source_community"})
        if "target" in topic_edges.columns and "target_community" not in topic_edges.columns:
            topic_edges = topic_edges.rename(columns={"target": "target_community"})
        require_columns(topic_edges, ["source_community", "target_community"], "topic_edges.csv")
        if "weight" not in topic_edges.columns:
            topic_edges["weight"] = 1.0
        topic_edges["source_community"] = topic_edges["source_community"].astype(int)
        topic_edges["target_community"] = topic_edges["target_community"].astype(int)
        topic_edges["weight"] = pd.to_numeric(topic_edges["weight"], errors="coerce").fillna(1.0)
    else:
        topic_edges = derive_topic_edges(works, citations)

    if topics_path.exists():
        topics = normalize_columns(pd.read_csv(topics_path))
        if "community" not in topics.columns and "display_community" in topics.columns:
            topics = topics.rename(columns={"display_community": "community"})
        require_columns(topics, ["community"], "topics.csv")
        if "label" not in topics.columns:
            topics["label"] = topics["community"].astype(str)
        if "x" not in topics.columns or "y" not in topics.columns:
            topics = compute_topic_positions(topic_edges, topics)
        topics["community"] = topics["community"].astype(int)
    else:
        topics = pd.DataFrame({"community": sorted(works["display_community"].unique())})
        topics["label"] = topics["community"].astype(str)
        topics = compute_topic_positions(topic_edges, topics)

    return RawData(works=works, citations=citations, topics=topics, topic_edges=topic_edges, analysis_end_year=analysis_end_year)


def derive_topic_edges(works: pd.DataFrame, citations: pd.DataFrame) -> pd.DataFrame:
    meta = works[["id", "display_community"]].copy()
    src = meta.rename(columns={"id": "source", "display_community": "source_community"})
    tgt = meta.rename(columns={"id": "target", "display_community": "target_community"})
    edges = citations.merge(src, on="source", how="inner").merge(tgt, on="target", how="inner")
    edges = edges[edges["source_community"] != edges["target_community"]]
    if edges.empty:
        return pd.DataFrame(columns=["source_community", "target_community", "weight"])
    out = edges.groupby(["source_community", "target_community"]).size().reset_index(name="weight")
    return out


def compute_topic_positions(topic_edges: pd.DataFrame, topics: pd.DataFrame) -> pd.DataFrame:
    g = nx.Graph()
    for c in topics["community"].astype(int):
        g.add_node(int(c))
    for r in topic_edges.itertuples(index=False):
        u = int(getattr(r, "source_community"))
        v = int(getattr(r, "target_community"))
        w = float(getattr(r, "weight", 1.0))
        g.add_edge(u, v, weight=w)
    if g.number_of_nodes() == 0:
        raise ValueError("No communities available for topic layout.")
    pos = nx.spring_layout(g, seed=13, weight="weight", iterations=100)
    out = topics.copy()
    out["x"] = out["community"].map(lambda c: float(pos.get(int(c), (0.0, 0.0))[0]))
    out["y"] = out["community"].map(lambda c: float(pos.get(int(c), (0.0, 0.0))[1]))
    return out


# -----------------------------------------------------------------------------
# Graph and indicator computation
# -----------------------------------------------------------------------------

def attach_metadata(raw: RawData) -> Tuple[pd.DataFrame, pd.DataFrame]:
    works = raw.works.copy()
    meta_src = works[["id", "year", "primary_field", "display_community", "domain", "is_landmark", "title"]].rename(
        columns={
            "id": "source",
            "year": "source_year",
            "primary_field": "source_field",
            "display_community": "source_community",
            "domain": "source_domain",
            "is_landmark": "source_is_landmark",
            "title": "source_title",
        }
    )
    meta_tgt = works[["id", "year", "primary_field", "display_community", "domain", "is_landmark", "title"]].rename(
        columns={
            "id": "target",
            "year": "target_year",
            "primary_field": "target_field",
            "display_community": "target_community",
            "domain": "target_domain",
            "is_landmark": "target_is_landmark",
            "title": "target_title",
        }
    )
    cit = raw.citations.merge(meta_src, on="source", how="inner").merge(meta_tgt, on="target", how="inner")
    return works, cit


def get_references(citations_meta: pd.DataFrame, paper_id: str, paper_year: int) -> pd.DataFrame:
    refs = citations_meta[(citations_meta["source"] == str(paper_id)) & (citations_meta["target_year"] < paper_year)].copy()
    return refs


def get_future_citers(citations_meta: pd.DataFrame, paper_id: str, paper_year: int, tau: int, analysis_end_year: int) -> pd.DataFrame:
    end_year = min(paper_year + tau, analysis_end_year)
    out = citations_meta[
        (citations_meta["target"] == str(paper_id))
        & (citations_meta["source_year"] > paper_year)
        & (citations_meta["source_year"] <= end_year)
    ].copy()
    return out


def build_local_reference_graph(refs: pd.DataFrame, citations_meta: pd.DataFrame, paper_id: Optional[str] = None, paper_comm: Optional[int] = None) -> Tuple[nx.Graph, Dict[str, int]]:
    ref_ids = set(refs["target"].astype(str))
    g = nx.Graph()
    g.add_nodes_from(ref_ids)
    sub = citations_meta[citations_meta["source"].isin(ref_ids) & citations_meta["target"].isin(ref_ids)]
    g.add_edges_from(zip(sub["source"].astype(str), sub["target"].astype(str)))
    comm_map = dict(zip(refs["target"].astype(str), refs["target_community"].astype(int)))
    if paper_id is not None:
        g.add_node(str(paper_id))
        for r in ref_ids:
            g.add_edge(str(paper_id), r)
        if paper_comm is None:
            paper_comm = int(pd.Series(list(comm_map.values())).mode().iloc[0]) if comm_map else -1
        comm_map[str(paper_id)] = int(paper_comm)
    return g, comm_map


def modularity_fixed_partition(g: nx.Graph, comm_map: Mapping[str, int]) -> float:
    nodes = [n for n in g.nodes if n in comm_map]
    if len(nodes) <= 2 or g.number_of_edges() == 0:
        return 0.0
    groups: Dict[int, set] = {}
    for n in nodes:
        groups.setdefault(int(comm_map[n]), set()).add(n)
    parts = [s for s in groups.values() if len(s) > 0]
    if len(parts) < 2:
        return 0.0
    try:
        return float(nx.algorithms.community.quality.modularity(g.subgraph(nodes), parts))
    except Exception:
        return 0.0


def boundary_mixing_share(g: nx.Graph, comm_map: Mapping[str, int]) -> float:
    if g.number_of_edges() == 0:
        return 0.0
    total = 0
    cross = 0
    for u, v in g.edges():
        if u in comm_map and v in comm_map:
            total += 1
            if comm_map[u] != comm_map[v]:
                cross += 1
    return float(cross / total) if total else 0.0


def field_distance_matrix(citations_meta: pd.DataFrame, year: int) -> Dict[Tuple[str, str], float]:
    prior = citations_meta[citations_meta["source_year"] < year].copy()
    if prior.empty:
        return {}
    by_paper = prior.groupby("source")["target_field"].apply(lambda s: sorted(set([str(x) for x in s if pd.notna(x)])))
    field_count: Dict[str, int] = {}
    pair_count: Dict[Tuple[str, str], int] = {}
    for fields in by_paper:
        for f in fields:
            field_count[f] = field_count.get(f, 0) + 1
        for i, fi in enumerate(fields):
            for fj in fields[i + 1 :]:
                key = tuple(sorted((fi, fj)))
                pair_count[key] = pair_count.get(key, 0) + 1
    fields = list(field_count.keys())
    dist: Dict[Tuple[str, str], float] = {}
    for i, fi in enumerate(fields):
        dist[(fi, fi)] = 0.0
        for fj in fields[i + 1 :]:
            obs = pair_count.get(tuple(sorted((fi, fj))), 0)
            sim = obs / max(math.sqrt(field_count[fi] * field_count[fj]), 1e-9)
            sim = min(max(sim, 0.0), 1.0)
            d = 1.0 - sim
            dist[(fi, fj)] = d
            dist[(fj, fi)] = d
    return dist


def rao_stirling(fields: Sequence[str], dist: Mapping[Tuple[str, str], float]) -> float:
    s = pd.Series([str(f) for f in fields if pd.notna(f)])
    if s.empty:
        return 0.0
    p = s.value_counts(normalize=True)
    vals = list(p.index)
    total = 0.0
    for i, fi in enumerate(vals):
        for fj in vals[i + 1 :]:
            total += 2.0 * p[fi] * p[fj] * float(dist.get((fi, fj), 1.0 if fi != fj else 0.0))
    return float(total)


def pair_zscore_lookup(citations_meta: pd.DataFrame, year: int) -> Dict[Tuple[str, str], float]:
    prior = citations_meta[citations_meta["source_year"] < year].copy()
    if prior.empty:
        return {}
    by_paper = prior.groupby("source")["target_field"].apply(lambda s: sorted(set([str(x) for x in s if pd.notna(x)])))
    n = len(by_paper)
    field_occ: Dict[str, int] = {}
    pair_obs: Dict[Tuple[str, str], int] = {}
    for fields in by_paper:
        for f in fields:
            field_occ[f] = field_occ.get(f, 0) + 1
        for i, fi in enumerate(fields):
            for fj in fields[i + 1 :]:
                key = tuple(sorted((fi, fj)))
                pair_obs[key] = pair_obs.get(key, 0) + 1
    out: Dict[Tuple[str, str], float] = {}
    fields = sorted(field_occ)
    for i, fi in enumerate(fields):
        for fj in fields[i + 1 :]:
            obs = pair_obs.get(tuple(sorted((fi, fj))), 0)
            pi = field_occ[fi] / max(n, 1)
            pj = field_occ[fj] / max(n, 1)
            prob = pi * pj
            exp = n * prob
            var = max(n * prob * (1.0 - prob), 1e-9)
            z = (obs - exp) / math.sqrt(var)
            out[(fi, fj)] = z
            out[(fj, fi)] = z
    return out


def uzzi_atypicality(fields: Sequence[str], pair_z: Mapping[Tuple[str, str], float]) -> float:
    vals = sorted(set([str(f) for f in fields if pd.notna(f)]))
    if len(vals) < 2:
        return 0.0
    zs: List[float] = []
    for i, fi in enumerate(vals):
        for fj in vals[i + 1 :]:
            zs.append(float(pair_z.get((fi, fj), 0.0)))
    if not zs:
        return 0.0
    # Use the negative 10th percentile so that larger values indicate more atypical combinations.
    return float(max(0.0, -np.percentile(zs, 10)))


def community_graph_from_citations(citations_meta: pd.DataFrame, end_year: int) -> nx.Graph:
    sub = citations_meta[(citations_meta["source_year"] <= end_year) & (citations_meta["target_year"] <= end_year)].copy()
    g = nx.Graph()
    for r in sub.itertuples(index=False):
        u = int(getattr(r, "source_community"))
        v = int(getattr(r, "target_community"))
        if u == v:
            continue
        if g.has_edge(u, v):
            g[u][v]["weight"] += 1.0
        else:
            g.add_edge(u, v, weight=1.0)
    return g


def path_length_or_inf(g: nx.Graph, u: int, v: int) -> float:
    if u == v:
        return 0.0
    try:
        return float(nx.shortest_path_length(g, int(u), int(v)))
    except Exception:
        return float("inf")


def path_shortening_for_refs(ref_comms: Sequence[int], g_before: nx.Graph, g_after: nx.Graph, disconnected_bonus: float = 2.0) -> float:
    comms = sorted(set([int(c) for c in ref_comms if pd.notna(c)]))
    if len(comms) < 2:
        return 0.0
    vals: List[float] = []
    for i, u in enumerate(comms):
        for v in comms[i + 1 :]:
            db = path_length_or_inf(g_before, u, v)
            da = path_length_or_inf(g_after, u, v)
            if math.isfinite(db) and math.isfinite(da):
                vals.append(max(0.0, db - da))
            elif (not math.isfinite(db)) and math.isfinite(da):
                vals.append(disconnected_bonus)
            else:
                vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def compute_indicator_and_delta_tables(
    raw: RawData,
    tau: int,
    min_refs: int,
    max_papers: Optional[int] = None,
    progress: bool = True,
    progress_interval: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    works, cit = attach_metadata(raw)
    # Only papers old enough to observe G+tau are eligible for weight learning.
    if "domain_analysis_end_year" in works.columns:
        end_by_paper = pd.to_numeric(works["domain_analysis_end_year"], errors="coerce").fillna(raw.analysis_end_year).astype(int)
    else:
        end_by_paper = pd.Series(raw.analysis_end_year, index=works.index, dtype=int)
    eligible = works[works["year"] + tau <= end_by_paper].copy()
    eligible = eligible.sort_values(["year", "id"])
    if eligible.empty:
        raise ValueError("No papers are old enough to observe the requested tau. Lower --tau or provide later analysis_end_year.")
    progress_interval = max(1, int(progress_interval))
    progress_log(
        f"Attached metadata: {len(works):,} works, {len(cit):,} citation rows. "
        f"Eligible for tau={tau}: {len(eligible):,} papers through analysis_end_year={raw.analysis_end_year}.",
        progress,
    )
    if max_papers is not None and max_papers > 0:
        progress_log(f"Debug limit active: stop after {max_papers:,} papers with computed metrics.", progress)

    # Caches by year.
    dist_cache: Dict[int, Dict[Tuple[str, str], float]] = {}
    pairz_cache: Dict[int, Dict[Tuple[str, str], float]] = {}
    comm_graph_cache: Dict[int, nx.Graph] = {}

    metric_rows: List[Dict[str, object]] = []
    delta_rows: List[Dict[str, object]] = []
    skipped_min_refs = 0
    cache_builds = {"field_distance": 0, "pair_z": 0, "community_graph": 0}

    for idx, paper in enumerate(eligible.itertuples(index=False), start=1):
        if max_papers is not None and max_papers > 0 and len(metric_rows) >= max_papers:
            progress_log(f"Reached max_papers={max_papers:,}; stopping paper scan.", progress)
            break
        if idx == 1 or idx % progress_interval == 0 or idx == len(eligible):
            progress_log(
                f"  scanned {idx:,}/{len(eligible):,}; computed={len(metric_rows):,}; "
                f"skipped_min_refs={skipped_min_refs:,}; cache_years="
                f"dist:{len(dist_cache)}, pairz:{len(pairz_cache)}, comm:{len(comm_graph_cache)}",
                progress,
            )
        pid = str(getattr(paper, "id"))
        year = int(getattr(paper, "year"))
        paper_analysis_end_year = int(getattr(paper, "domain_analysis_end_year", raw.analysis_end_year))
        pcomm = int(getattr(paper, "display_community"))
        refs = get_references(cit, pid, year)
        if len(refs) < min_refs:
            skipped_min_refs += 1
            continue
        if year not in dist_cache:
            progress_log(f"    building field-distance cache for year {year}", progress)
            dist_cache[year] = field_distance_matrix(cit, year)
            cache_builds["field_distance"] += 1
        if year not in pairz_cache:
            progress_log(f"    building field-pair z-score cache for year {year}", progress)
            pairz_cache[year] = pair_zscore_lookup(cit, year)
            cache_builds["pair_z"] += 1
        if (year - 1) not in comm_graph_cache:
            progress_log(f"    building community graph cache for year {year - 1}", progress)
            comm_graph_cache[year - 1] = community_graph_from_citations(cit, year - 1)
            cache_builds["community_graph"] += 1
        if min(year + tau, paper_analysis_end_year) not in comm_graph_cache:
            future_year = min(year + tau, paper_analysis_end_year)
            progress_log(f"    building community graph cache for year {future_year}", progress)
            comm_graph_cache[future_year] = community_graph_from_citations(cit, future_year)
            cache_builds["community_graph"] += 1

        ref_fields = refs["target_field"].astype(str).tolist()
        ref_comms = refs["target_community"].astype(int).tolist()
        ref_ids = refs["target"].astype(str).tolist()

        gm, comm_m = build_local_reference_graph(refs, cit)
        g0, comm_0 = build_local_reference_graph(refs, cit, paper_id=pid, paper_comm=pcomm)
        q_minus = modularity_fixed_partition(gm, comm_m)
        q_zero = modularity_fixed_partition(g0, comm_0)
        delta_q0 = q_zero - q_minus
        boundary_shock0 = -delta_q0

        # B: publication-day local bridge centrality of p in the reference graph augmented by p.
        try:
            b_val = float(nx.betweenness_centrality(g0, normalized=True).get(pid, 0.0))
        except Exception:
            b_val = 0.0

        rs_val = rao_stirling(ref_fields, dist_cache[year])
        uzzi_val = uzzi_atypicality(ref_fields, pairz_cache[year])
        rtd_val = simpson_diversity(ref_comms)
        pde_val = shannon_entropy(ref_fields)

        try:
            eff_size = nx.effective_size(g0, nodes=[pid]).get(pid, 0.0)  # type: ignore[attr-defined]
        except Exception:
            eff_size = float(len(ref_ids))
        try:
            constraint = nx.constraint(g0, nodes=[pid]).get(pid, 1.0)  # type: ignore[attr-defined]
        except Exception:
            constraint = 1.0
        burt_ip = float(eff_size / max(1.0, len(ref_ids)))
        inv_constraint = float(1.0 / max(constraint, 1e-9))

        metric_rows.append(
            {
                "paper_id": pid,
                "title": getattr(paper, "title"),
                "domain": getattr(paper, "domain"),
                "year": year,
                "primary_field": getattr(paper, "primary_field"),
                "display_community": pcomm,
                "is_landmark": int(getattr(paper, "is_landmark")),
                "reference_count": len(ref_ids),
                "cited_by_count": float(getattr(paper, "cited_by_count", np.nan)),
                "B": b_val,
                "RS": rs_val,
                "DeltaQ0": boundary_shock0,  # direction unified: higher = stronger boundary shock
                "Uzzi": uzzi_val,
                "RTD": rtd_val,
                "BurtIP": burt_ip,
                "PDE": pde_val,
                # Optional alternatives for diagnostics / supplementary tables.
                "degree_p": float(g0.degree(pid)),
                "effective_size": float(eff_size),
                "constraint_inv": inv_constraint,
                "field_variety": float(len(set(ref_fields))),
                "field_simpson": simpson_diversity(ref_fields),
                "community_variety": float(len(set(ref_comms))),
            }
        )

        # Future deltas relative to G+tau.
        future = get_future_citers(cit, pid, year, tau, paper_analysis_end_year)
        fut_fields = future["source_field"].astype(str).tolist()
        fut_comms = future["source_community"].astype(int).tolist()
        n_future = len(future)
        cross_adoption = float(np.mean([c != pcomm for c in fut_comms])) if fut_comms else 0.0
        field_entropy = shannon_entropy(fut_fields)
        community_reach = float(len(set(fut_comms)))
        concentration = normalized_hhi(fut_comms)
        g_before = comm_graph_cache[year - 1]
        g_after = comm_graph_cache[min(year + tau, paper_analysis_end_year)]
        pshort = path_shortening_for_refs(ref_comms, g_before, g_after)

        # Local graph before/after around references+p+future citers.
        local_nodes = set(ref_ids) | {pid} | set(future["source"].astype(str))
        local_edges = cit[cit["source"].isin(local_nodes) & cit["target"].isin(local_nodes)].copy()
        g_future = nx.Graph()
        g_future.add_nodes_from(local_nodes)
        g_future.add_edges_from(zip(local_edges["source"].astype(str), local_edges["target"].astype(str)))
        # Ensure p-reference edges exist (some citation tables may omit if filtered direction issues).
        for rid in ref_ids:
            g_future.add_edge(pid, rid)
        comm_future: Dict[str, int] = {}
        # references and p
        comm_future.update(comm_0)
        # future citers
        comm_future.update(dict(zip(future["source"].astype(str), future["source_community"].astype(int))))
        q_future = modularity_fixed_partition(g_future, comm_future)
        modularity_shock = max(0.0, -(q_future - q_minus))
        boundary_mixing = boundary_mixing_share(g_future, comm_future) - boundary_mixing_share(gm, comm_m)
        partition_change = js_divergence(ref_comms, fut_comms)
        try:
            hub_formation = float(nx.degree_centrality(g_future).get(pid, 0.0))
        except Exception:
            hub_formation = float(n_future)

        delta_rows.append(
            {
                "paper_id": pid,
                "title": getattr(paper, "title"),
                "domain": getattr(paper, "domain"),
                "year": year,
                "primary_field": getattr(paper, "primary_field"),
                "display_community": pcomm,
                "is_landmark": int(getattr(paper, "is_landmark")),
                "reference_count": len(ref_ids),
                "n_future_citers": n_future,
                "community_reach": community_reach,
                "field_entropy": field_entropy,
                "cross_community_adoption": cross_adoption,
                "path_shortening": pshort,
                "modularity_shock": modularity_shock,
                "partition_change": partition_change,
                "boundary_mixing": boundary_mixing,
                "post_perturbation_concentration": concentration,
                "hub_formation": hub_formation,
            }
        )

    metrics = pd.DataFrame(metric_rows)
    deltas = pd.DataFrame(delta_rows)
    if metrics.empty or deltas.empty:
        raise ValueError("No eligible papers after applying min_refs and tau constraints.")
    progress_log(
        f"Finished indicator/delta computation: {len(metrics):,} papers computed, "
        f"{skipped_min_refs:,} skipped because refs < min_refs={min_refs}. Cache builds: "
        f"field_distance={cache_builds['field_distance']}, pair_z={cache_builds['pair_z']}, "
        f"community_graph={cache_builds['community_graph']}.",
        progress,
    )
    return metrics, deltas


# -----------------------------------------------------------------------------
# Matched controls, RGPM, and feature standardization
# -----------------------------------------------------------------------------

def add_reference_bins(df: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    out = df.copy()
    # Use rank to avoid qcut failures due to ties.
    if out["reference_count"].nunique() <= 1:
        out["ref_bin"] = 0
        return out
    q = min(n_bins, max(2, out["reference_count"].nunique()))
    out["ref_bin"] = pd.qcut(out["reference_count"].rank(method="first"), q=q, labels=False, duplicates="drop")
    out["ref_bin"] = out["ref_bin"].fillna(0).astype(int)
    return out


def matched_control_indices(df: pd.DataFrame, row: pd.Series, min_controls: int = 20) -> np.ndarray:
    idx, _ = matched_control_indices_with_tier(df, row, min_controls=min_controls)
    return idx


def matched_control_indices_with_tier(
    df: pd.DataFrame,
    row: pd.Series,
    min_controls: int = 20,
) -> Tuple[np.ndarray, str]:
    same_base = df["paper_id"] != row["paper_id"]
    non_landmark = df["is_landmark"].astype(int) == 0
    # Strict: same field, year ±1, same reference bin.
    mask = (
        same_base
        & non_landmark
        & (df["primary_field"] == row["primary_field"])
        & (df["year"].between(int(row["year"]) - 1, int(row["year"]) + 1))
        & (df["ref_bin"] == row["ref_bin"])
    )
    idx = df.index[mask].to_numpy()
    if len(idx) >= min_controls:
        return idx, "field_year_refbin"
    # Relax reference bin.
    mask = same_base & non_landmark & (df["primary_field"] == row["primary_field"]) & (df["year"].between(int(row["year"]) - 1, int(row["year"]) + 1))
    idx = df.index[mask].to_numpy()
    if len(idx) >= min_controls:
        return idx, "field_year"
    # Relax to year ±3.
    mask = same_base & non_landmark & (df["primary_field"] == row["primary_field"]) & (df["year"].between(int(row["year"]) - 3, int(row["year"]) + 3))
    idx = df.index[mask].to_numpy()
    if len(idx) >= min_controls:
        return idx, "field_year3"
    # Relax to all same field.
    mask = same_base & non_landmark & (df["primary_field"] == row["primary_field"])
    idx = df.index[mask].to_numpy()
    if len(idx) >= min_controls:
        return idx, "field_all_years"
    # Last fallback: all non-landmark papers excluding p. This is still real data, not simulated.
    mask = same_base & non_landmark
    return df.index[mask].to_numpy(), "all_non_landmark"


def compute_rgpm(
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    min_controls: int = 20,
    z_cap: float = 4.0,
    progress: bool = True,
    progress_interval: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
    metric_cols = ["paper_id", "title", "domain", "year", "primary_field", "display_community", "is_landmark", "reference_count"]
    if "cited_by_count" in metrics.columns:
        metric_cols.append("cited_by_count")
    delta_cols = ["paper_id"] + DELTA_KEYS
    if "n_future_citers" in deltas.columns:
        delta_cols.append("n_future_citers")
    df = metrics[metric_cols].merge(deltas[delta_cols], on="paper_id", how="inner")
    df = add_reference_bins(df)
    required_controls = max(10, int(min_controls))
    progress_interval = max(1, int(progress_interval))
    progress_log(
        f"Computing RGPM-v2 for {len(df):,} papers with min_controls={required_controls}, z_cap={z_cap}.",
        progress,
    )
    global_scale = {col: robust_delta_scale(df[col].to_numpy(dtype=float)) for col in DELTA_KEYS}
    z_rows: List[Dict[str, object]] = []
    control_rows: List[Dict[str, object]] = []
    skipped_controls = 0
    control_sizes: List[int] = []
    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        if row_idx == 1 or row_idx % progress_interval == 0 or row_idx == len(df):
            progress_log(
                f"  RGPM rows processed {row_idx:,}/{len(df):,}; kept={len(z_rows):,}; "
                f"skipped_controls={skipped_controls:,}",
                progress,
            )
        ctrl_idx, tier = matched_control_indices_with_tier(df, row, min_controls=required_controls)
        if len(ctrl_idx) < required_controls:
            skipped_controls += 1
            continue
        control_sizes.append(int(len(ctrl_idx)))
        controls = df.loc[ctrl_idx]
        out = {
            "paper_id": row["paper_id"],
            "title": row["title"],
            "domain": row["domain"],
            "year": row["year"],
            "primary_field": row["primary_field"],
            "display_community": row["display_community"],
            "is_landmark": row["is_landmark"],
            "reference_count": row["reference_count"],
            "cited_by_count": row.get("cited_by_count", np.nan),
            "n_future_citers": row.get("n_future_citers", np.nan),
            "n_controls": int(len(ctrl_idx)),
            "control_tier": tier,
        }
        control_out = {
            "paper_id": row["paper_id"],
            "domain": row["domain"],
            "year": row["year"],
            "primary_field": row["primary_field"],
            "is_landmark": row["is_landmark"],
            "n_controls": int(len(ctrl_idx)),
            "control_tier": tier,
        }
        n_floor_used = 0
        n_mad_zero = 0
        n_clipped = 0
        for col in DELTA_KEYS:
            med = float(np.median(controls[col].to_numpy(dtype=float)))
            local_mad = raw_mad(controls[col].to_numpy(dtype=float))
            global_floor = 0.25 * float(global_scale.get(col, 0.0) if np.isfinite(global_scale.get(col, np.nan)) else 0.0)
            delta_floor = float(DELTA_FLOORS.get(col, DEFAULT_DELTA_FLOOR))
            scale = max(
                float(local_mad) if np.isfinite(local_mad) else 0.0,
                global_floor,
                delta_floor,
            )
            floor_used = (not np.isfinite(local_mad)) or local_mad < scale - 1e-12
            mad_zero = (not np.isfinite(local_mad)) or local_mad < 1e-6
            z_raw = float((float(row[col]) - med) / max(scale, 1e-12))
            z_val = z_raw
            if z_cap is not None and z_cap > 0:
                z_val = float(np.clip(z_val, -z_cap, z_cap))
            clipped = abs(z_val - z_raw) > 1e-9
            out[col + "_z"] = z_val
            out[col + "_z_raw"] = z_raw
            out[col + "_z_clipped"] = int(clipped)
            out[col + "_control_median"] = med
            out[col + "_control_mad"] = float(local_mad) if np.isfinite(local_mad) else np.nan
            out[col + "_scale_used"] = float(scale)
            out[col + "_scale_floor_used"] = int(floor_used)
            out[col] = float(row[col])
            control_out[col + "_control_mad"] = float(local_mad) if np.isfinite(local_mad) else np.nan
            control_out[col + "_scale_used"] = float(scale)
            control_out[col + "_scale_floor_used"] = int(floor_used)
            control_out[col + "_mad_zero"] = int(mad_zero)
            control_out[col + "_z_clipped"] = int(clipped)
            n_floor_used += int(floor_used)
            n_mad_zero += int(mad_zero)
            n_clipped += int(clipped)
        control_out["n_delta_scale_floor_used"] = n_floor_used
        control_out["n_delta_mad_zero"] = n_mad_zero
        control_out["n_delta_z_clipped"] = n_clipped
        z_rows.append(out)
        control_rows.append(control_out)
    zdf = pd.DataFrame(z_rows)
    if zdf.empty:
        raise ValueError("Could not compute matched-control z-scores. Check landmark/control coverage and min_controls.")

    control_diag = pd.DataFrame(control_rows)
    delta_diag_rows: List[Dict[str, object]] = []
    for col in DELTA_KEYS:
        values = df[col].to_numpy(dtype=float)
        nonzero_rate = float(np.mean(np.abs(values[np.isfinite(values)]) > 1e-12)) if np.isfinite(values).any() else 0.0
        cap_hit_rate = float(np.nanmean(zdf[col + "_z_clipped"].to_numpy(dtype=float)))
        control_mad_zero_rate = float(np.nanmean(control_diag[col + "_mad_zero"].to_numpy(dtype=float)))
        floor_use_rate = float(np.nanmean(control_diag[col + "_scale_floor_used"].to_numpy(dtype=float)))
        global_mad = float(global_scale.get(col, np.nan))
        active = col in PRIMARY_RGPM_DELTA_KEYS
        reasons: List[str] = []
        if col not in PRIMARY_RGPM_DELTA_KEYS:
            active = False
            reasons.append("compression_diagnostic_default")
        if nonzero_rate < DELTA_NONZERO_MIN:
            active = False
            reasons.append("nonzero_rate_below_0.03")
        if (not np.isfinite(global_mad)) or global_mad < DELTA_GLOBAL_MAD_MIN:
            active = False
            reasons.append("global_mad_below_1e-6")
        if cap_hit_rate >= DELTA_CAP_HIT_DROP:
            active = False
            reasons.append("z_cap_hit_rate_ge_0.10")
        if control_mad_zero_rate >= DELTA_CONTROL_MAD_ZERO_DROP:
            active = False
            reasons.append("control_mad_zero_rate_ge_0.50")
        delta_diag_rows.append(
            {
                "delta": col,
                "label": DELTA_LABELS[col],
                "primary_candidate": int(col in PRIMARY_RGPM_DELTA_KEYS),
                "active": int(active),
                "nonzero_rate": nonzero_rate,
                "global_mad": global_mad,
                "z_cap_hit_rate": cap_hit_rate,
                "control_mad_zero_rate": control_mad_zero_rate,
                "scale_floor_use_rate": floor_use_rate,
                "drop_reasons": ";".join(reasons) if reasons else "",
            }
        )
    delta_diag = pd.DataFrame(delta_diag_rows)
    active_delta_keys = delta_diag.loc[delta_diag["active"].astype(int) == 1, "delta"].astype(str).tolist()
    if not active_delta_keys:
        fallback = (
            delta_diag[delta_diag["primary_candidate"].astype(int) == 1]
            .sort_values(["z_cap_hit_rate", "control_mad_zero_rate", "global_mad"], ascending=[True, True, False])
            .head(3)["delta"]
            .astype(str)
            .tolist()
        )
        active_delta_keys = fallback
        delta_diag.loc[delta_diag["delta"].isin(fallback), "active"] = 1
        delta_diag.loc[delta_diag["delta"].isin(fallback), "drop_reasons"] = "fallback_active_for_diagnostic_run"
    dropped_delta_table = delta_diag[delta_diag["active"].astype(int) == 0].copy()

    active_z_cols = [c + "_z" for c in active_delta_keys]
    zmat = zdf[active_z_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    zmat_pos = np.maximum(zmat, 0.0)
    zdf["RGPM_v2"] = np.sqrt(np.square(zmat_pos).mean(axis=1))
    zdf["RGPM"] = zdf["RGPM_v2"]
    zdf["RGPM_simple"] = zdf["RGPM_v2"]
    # Debug-only Mahalanobis distance on the screened active z columns.
    cov = np.cov(zmat, rowvar=False)
    if cov.ndim == 0:
        cov = np.eye(len(active_z_cols))
    lam = 0.12
    diag = np.diag(np.diag(cov))
    cov_shrink = (1.0 - lam) * cov + lam * diag + np.eye(len(active_z_cols)) * 1e-6
    inv_cov = np.linalg.pinv(cov_shrink)
    zdf["RGPM_mahalanobis_debug"] = np.sqrt(np.einsum("ij,jk,ik->i", zmat, inv_cov, zmat))
    if control_sizes:
        progress_log(
            f"Finished RGPM: {len(zdf):,} papers kept; controls median={np.median(control_sizes):.0f}, "
            f"min={np.min(control_sizes):.0f}, max={np.max(control_sizes):.0f}; "
            f"active deltas={len(active_delta_keys)} ({', '.join(active_delta_keys)}).",
            progress,
        )
    else:
        progress_log(f"Finished RGPM: {len(zdf):,} papers kept.", progress)
    return zdf, delta_diag, control_diag, active_delta_keys, dropped_delta_table


def transformed_metric_values(metrics: pd.DataFrame, key: str) -> pd.Series:
    vals = pd.to_numeric(metrics[key], errors="coerce").astype(float)
    if key == "B":
        vals = pd.Series(np.log1p(np.clip(vals.to_numpy(dtype=float), 0.0, None)), index=metrics.index)
    elif key == "DeltaQ0":
        vals = pd.Series(winsorize(vals.to_numpy(dtype=float), 0.01, 0.99), index=metrics.index)
    elif key == "Uzzi":
        vals = vals.copy()
        if "field_variety" in metrics.columns:
            invalid = pd.to_numeric(metrics["field_variety"], errors="coerce").fillna(0) < 2
            vals.loc[invalid] = np.nan
    return vals.replace([np.inf, -np.inf], np.nan)


def field_year_standardize(metrics: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    out = metrics.copy()
    diagnostics: List[Dict[str, object]] = []
    active_metric_keys: List[str] = []
    for key in METRIC_KEYS:
        transformed = transformed_metric_values(out, key)
        out[key + "_transformed"] = transformed
        z = np.full(len(out), np.nan, dtype=float)
        scope = pd.Series("missing", index=out.index, dtype=object)
        valid = transformed.notna() & np.isfinite(transformed.to_numpy(dtype=float))

        for _, idx in out.loc[valid].groupby(["primary_field", "year"]).groups.items():
            idx = pd.Index(idx)
            vals = transformed.loc[idx].to_numpy(dtype=float)
            if len(idx) >= 20 and np.nanstd(vals) > 1e-12:
                z_vals = rank_normal_scores(vals)
                z[out.index.get_indexer(idx)] = z_vals
                scope.loc[idx] = "field_year"

        remaining = valid & ~np.isfinite(z)
        for _, idx in out.loc[remaining].groupby("primary_field").groups.items():
            idx = pd.Index(idx)
            field_name = out.loc[idx[0], "primary_field"]
            field_idx = out.index[(out["primary_field"] == field_name) & valid]
            vals = transformed.loc[field_idx].to_numpy(dtype=float)
            if len(field_idx) >= 50 and np.nanstd(vals) > 1e-12:
                z_vals = rank_normal_scores(vals)
                z_field = pd.Series(z_vals, index=field_idx)
                z[out.index.get_indexer(idx)] = z_field.loc[idx].to_numpy(dtype=float)
                scope.loc[idx] = "field"

        remaining = valid & ~np.isfinite(z)
        if remaining.any():
            valid_idx = out.index[valid]
            vals = transformed.loc[valid_idx].to_numpy(dtype=float)
            z_vals = rank_normal_scores(vals)
            z_global = pd.Series(z_vals, index=valid_idx)
            rem_idx = out.index[remaining]
            z[out.index.get_indexer(rem_idx)] = z_global.loc[rem_idx].to_numpy(dtype=float)
            scope.loc[rem_idx] = "global"
        out[key + "_z"] = z
        out[key + "_z_scope"] = scope

        arr = transformed.to_numpy(dtype=float)
        valid_ratio = float(np.isfinite(arr).mean()) if len(arr) else 0.0
        finite = arr[np.isfinite(arr)]
        zero_rate = float(np.mean(np.abs(finite) < 1e-12)) if len(finite) else 0.0
        iqr = float(np.percentile(finite, 75) - np.percentile(finite, 25)) if len(finite) else float("nan")
        unique_count = int(pd.Series(finite).nunique()) if len(finite) else 0
        field_year_fallback_ratio = float(np.mean(scope.loc[valid] != "field_year")) if valid.any() else 1.0
        active = valid_ratio >= 0.10
        if active:
            active_metric_keys.append(key)
        diagnostics.append(
            {
                "metric": key,
                "label": METRIC_LABELS[key],
                "active_for_learning": int(active),
                "valid_ratio": valid_ratio,
                "missing_rate": 1.0 - valid_ratio,
                "zero_rate": zero_rate,
                "unique_count": unique_count,
                "iqr": iqr,
                "field_year_fallback_ratio": field_year_fallback_ratio,
                "field_fallback_ratio": float(np.mean(scope.loc[valid] == "field")) if valid.any() else 0.0,
                "global_fallback_ratio": float(np.mean(scope.loc[valid] == "global")) if valid.any() else 0.0,
                "validity_flag": "drop_lt_0.10" if valid_ratio < 0.10 else ("warn_lt_0.30" if valid_ratio < 0.30 else "ok"),
            }
        )
    return out, pd.DataFrame(diagnostics), active_metric_keys


# -----------------------------------------------------------------------------
# Weight learning
# -----------------------------------------------------------------------------

def make_folds(df: pd.DataFrame, n_folds: int, mode: str, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(df)
    indices = np.arange(n)
    if n < 4:
        raise ValueError("Not enough papers for cross-validation.")
    n_folds = max(2, min(int(n_folds), n))
    if mode == "time":
        order = np.argsort(df["year"].to_numpy())
        chunks = np.array_split(order, n_folds)
        return [np.asarray(c, dtype=int) for c in chunks if len(c) > 0]
    if mode == "domain" and df["domain"].nunique() >= 2:
        folds = []
        for _, sub in df.groupby("domain"):
            folds.append(sub.index.to_numpy(dtype=int))
        return folds
    rng.shuffle(indices)
    chunks = np.array_split(indices, n_folds)
    return [np.asarray(c, dtype=int) for c in chunks if len(c) > 0]


def make_cv_splits(df: pd.DataFrame, n_folds: int, mode: str, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return explicit train/test splits; time_block is forward-chaining."""
    n = len(df)
    all_idx = np.arange(n)
    if n < 4:
        raise ValueError("Not enough papers for cross-validation.")
    n_folds = max(2, min(int(n_folds), n))
    if mode == "time_block":
        order = np.argsort(df["year"].to_numpy())
        chunks = [np.asarray(c, dtype=int) for c in np.array_split(order, n_folds) if len(c) > 0]
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(1, len(chunks)):
            train_idx = np.concatenate(chunks[:i])
            test_idx = chunks[i]
            if len(train_idx) >= 8 and len(test_idx) >= 4:
                splits.append((train_idx, test_idx))
        if splits:
            return splits
    folds = make_folds(df, n_folds=n_folds, mode="time" if mode == "time_block" else mode, seed=seed)
    splits = []
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_idx[train_mask]
        if len(train_idx) >= 4 and len(test_idx) >= 4:
            splits.append((train_idx, test_idx))
    return splits


def generate_dirichlet_weights(n_samples: int, seed: int, alpha: float = 1.0, n_metrics: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(n_metrics or len(METRIC_KEYS))
    return rng.dirichlet(np.ones(n) * alpha, size=int(n_samples))


def expand_active_weights(W_active: np.ndarray, active_metric_keys: Sequence[str]) -> np.ndarray:
    out = np.zeros((W_active.shape[0], len(METRIC_KEYS)), dtype=float)
    for i, key in enumerate(active_metric_keys):
        out[:, METRIC_KEYS.index(key)] = W_active[:, i]
    return out


def direct_weight_performance(X: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
    scores = X @ W.T
    out = np.full(W.shape[0], np.nan, dtype=float)
    for i in range(W.shape[0]):
        out[i] = safe_spearman(scores[:, i], y)
    return out


def weight_performance_cv(
    X: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    folds: List[np.ndarray],
    progress: bool = True,
) -> np.ndarray:
    perfs = np.zeros(W.shape[0], dtype=float)
    counts = np.zeros(W.shape[0], dtype=float)
    all_idx = np.arange(len(y))
    # For each fixed weight, evaluate rank association in each test fold and average.
    # This avoids using the test fold to fit anything except fixed candidate evaluation.
    for fold_idx, test_idx in enumerate(folds, start=1):
        progress_log(
            f"  evaluating weight samples on fold {fold_idx}/{len(folds)} "
            f"(n_test={len(test_idx):,}, n_weights={W.shape[0]:,})",
            progress,
        )
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        # Keep evaluation only on test. Training mask retained for conceptual CV split.
        xt = X[test_idx]
        yt = y[test_idx]
        if len(yt) < 4 or np.std(yt) < 1e-12:
            continue
        scores = xt @ W.T
        for i in range(W.shape[0]):
            rho = safe_spearman(scores[:, i], yt)
            if np.isfinite(rho):
                perfs[i] += rho
                counts[i] += 1
    out = np.divide(perfs, np.maximum(counts, 1), out=np.full_like(perfs, np.nan), where=counts > 0)
    return out


def weight_performance_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    progress: bool = True,
) -> np.ndarray:
    perfs = np.zeros(W.shape[0], dtype=float)
    counts = np.zeros(W.shape[0], dtype=float)
    for fold_idx, (_, test_idx) in enumerate(splits, start=1):
        progress_log(
            f"  evaluating weight samples on split {fold_idx}/{len(splits)} "
            f"(n_test={len(test_idx):,}, n_weights={W.shape[0]:,})",
            progress,
        )
        xt = X[test_idx]
        yt = y[test_idx]
        if len(yt) < 4 or np.std(yt) < 1e-12:
            continue
        scores = xt @ W.T
        for i in range(W.shape[0]):
            rho = safe_spearman(scores[:, i], yt)
            if np.isfinite(rho):
                perfs[i] += rho
                counts[i] += 1
    return np.divide(perfs, np.maximum(counts, 1), out=np.full_like(perfs, np.nan), where=counts > 0)


def learn_weights(
    metrics: pd.DataFrame,
    rgpm: pd.DataFrame,
    active_metric_keys: Sequence[str],
    n_samples: int,
    n_folds: int,
    cv_mode: str,
    seed: int,
    progress: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, float, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active_metric_keys = [k for k in active_metric_keys if k in METRIC_KEYS]
    if not active_metric_keys:
        raise ValueError("No active publication-day indicators are available for weight learning.")
    metric_z_cols = [k + "_z" for k in METRIC_KEYS]
    active_z_cols = [k + "_z" for k in active_metric_keys]
    meta_cols = [
        "paper_id", "title", "domain", "year", "primary_field", "display_community",
        "is_landmark", "reference_count", "cited_by_count",
    ]
    present_meta_cols = [c for c in meta_cols if c in metrics.columns]
    rgpm_cols = ["paper_id", "RGPM", "RGPM_v2", "RGPM_simple", "RGPM_mahalanobis_debug"]
    present_rgpm_cols = [c for c in rgpm_cols if c in rgpm.columns]
    df = metrics[present_meta_cols + metric_z_cols].merge(
        rgpm[present_rgpm_cols], on="paper_id", how="inner"
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["RGPM"]).reset_index(drop=True)
    # Missing raw indicators are not filled with zero. After rank normalization, however,
    # z=0 is the neutral within-scope median and keeps sparse indicators from forcing
    # complete-case deletion of otherwise usable papers.
    df[active_z_cols] = df[active_z_cols].fillna(0.0)
    if len(df) < max(20, n_folds * 5):
        raise ValueError(f"Too few papers for weight learning after filtering: n={len(df)}")
    progress_log(
        f"Learning weights from {len(df):,} papers, {n_samples:,} sampled simplex weights, "
        f"active_metrics={','.join(active_metric_keys)}, cv_mode={cv_mode}, n_folds={n_folds}, seed={seed}.",
        progress,
    )
    X = df[active_z_cols].to_numpy(dtype=float)
    y = df["RGPM"].to_numpy(dtype=float)
    progress_log("  sampling Dirichlet weight vectors...", progress)
    W = generate_dirichlet_weights(n_samples, seed=seed, alpha=1.0, n_metrics=len(active_metric_keys))
    folds = make_folds(df, n_folds=n_folds, mode=cv_mode, seed=seed)
    progress_log(
        "  folds: " + ", ".join([f"{i + 1}:n={len(f):,}" for i, f in enumerate(folds)]),
        progress,
    )
    perf = weight_performance_cv(X, y, W, folds, progress=progress)
    if np.all(~np.isfinite(perf)):
        raise ValueError("All weight performances are NaN. Check RGPM and feature variation.")
    best_idx = int(np.nanargmax(perf))
    best_full = expand_active_weights(W[[best_idx]], active_metric_keys)[0]
    best_w = pd.Series(best_full, index=METRIC_KEYS, name="weight")
    W_full = expand_active_weights(W, active_metric_keys)
    sample_df = pd.DataFrame(W_full, columns=["w_" + k for k in METRIC_KEYS])
    sample_df["cv_spearman"] = perf
    sample_df = sample_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["cv_spearman"]).reset_index(drop=True)
    equal_w = np.ones(len(active_metric_keys), dtype=float) / len(active_metric_keys)
    equal_score = X @ equal_w
    equal_rho = safe_spearman(equal_score, y)
    sample_df["cv_spearman_delta_vs_equal"] = sample_df["cv_spearman"] - equal_rho

    oof_score = np.full(len(df), np.nan, dtype=float)
    fold_ids = np.full(len(df), -1, dtype=int)
    fold_rows: List[Dict[str, object]] = []
    for fold_no, test_idx in enumerate(folds, start=1):
        train_mask = np.ones(len(df), dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]
        progress_log(
            f"  outer fold {fold_no}/{len(folds)}: selecting weights on train={len(train_idx):,}, "
            f"predicting test={len(test_idx):,}",
            progress,
        )
        train_perf = direct_weight_performance(X[train_idx], y[train_idx], W)
        if np.all(~np.isfinite(train_perf)):
            fold_best_idx = best_idx
        else:
            fold_best_idx = int(np.nanargmax(train_perf))
        fold_w = W[fold_best_idx]
        oof_score[test_idx] = X[test_idx] @ fold_w
        fold_ids[test_idx] = fold_no
        fold_test_rho = safe_spearman(oof_score[test_idx], y[test_idx])
        fold_row: Dict[str, object] = {
            "fold": fold_no,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_spearman": float(np.nanmax(train_perf)) if np.isfinite(train_perf).any() else np.nan,
            "test_spearman": fold_test_rho,
        }
        fold_full = expand_active_weights(fold_w.reshape(1, -1), active_metric_keys)[0]
        for key, val in zip(METRIC_KEYS, fold_full):
            fold_row["w_" + key] = float(val)
        fold_rows.append(fold_row)

    best_perf = safe_spearman(oof_score, y)
    score = X @ W[best_idx]
    score_table = df[[c for c in present_meta_cols if c in df.columns]].copy()
    score_table["fold_id"] = fold_ids
    score_table["S_w"] = score
    score_table["S_w_oof"] = oof_score
    score_table["S_equal"] = equal_score
    score_table["RGPM"] = y
    score_table["RGPM_v2"] = df["RGPM_v2"].to_numpy(dtype=float) if "RGPM_v2" in df.columns else y
    score_table["RGPM_simple"] = df["RGPM_simple"].to_numpy(dtype=float)
    if "RGPM_mahalanobis_debug" in df.columns:
        score_table["RGPM_mahalanobis_debug"] = df["RGPM_mahalanobis_debug"].to_numpy(dtype=float)
    for k in METRIC_KEYS:
        score_table[k + "_z"] = df[k + "_z"].to_numpy(dtype=float)
    cv_summary = pd.DataFrame(fold_rows)[["fold", "n_train", "n_test", "train_spearman", "test_spearman"]]
    fold_weights = pd.DataFrame(fold_rows)

    baseline_rows: List[Dict[str, object]] = []

    def add_baseline(name: str, scores: Sequence[float], kind: str, metric: str = "") -> None:
        rho = safe_spearman(scores, y)
        lo, hi = bootstrap_spearman_ci(scores, y, seed=seed + len(baseline_rows) + 101)
        baseline_rows.append(
            {
                "model": name,
                "kind": kind,
                "metric": metric,
                "oof_spearman": rho,
                "ci_low": lo,
                "ci_high": hi,
                "delta_vs_equal": rho - equal_rho if np.isfinite(rho) and np.isfinite(equal_rho) else np.nan,
            }
        )

    add_baseline("equal_weights", equal_score, "fixed")
    single_scores = []
    for i, key in enumerate(active_metric_keys):
        rho = safe_spearman(X[:, i], y)
        single_scores.append((rho, key, X[:, i]))
    single_scores = [item for item in single_scores if np.isfinite(item[0])]
    if single_scores:
        best_single_rho, best_single_key, best_single_score = max(single_scores, key=lambda item: item[0])
        add_baseline("best_single_indicator", best_single_score, "single_indicator", best_single_key)
    if "reference_count" in df.columns:
        add_baseline("reference_count", pd.to_numeric(df["reference_count"], errors="coerce"), "bibliometric")
    if "cited_by_count" in df.columns and pd.to_numeric(df["cited_by_count"], errors="coerce").notna().any():
        add_baseline("cited_by_count", pd.to_numeric(df["cited_by_count"], errors="coerce"), "bibliometric")
    random_median_rho = float(np.nanmedian(perf))
    baseline_rows.append(
        {
            "model": "random_dirichlet_median",
            "kind": "sampled_weights",
            "metric": "",
            "oof_spearman": random_median_rho,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "delta_vs_equal": random_median_rho - equal_rho if np.isfinite(equal_rho) else np.nan,
        }
    )
    add_baseline("learned_weight_oof", oof_score, "learned")
    baseline_comparison = pd.DataFrame(baseline_rows)
    model_diagnostics = baseline_comparison.copy()
    progress_log(
        f"Finished strict OOF weight learning: {len(sample_df):,} valid weight samples, "
        f"OOF Spearman={best_perf:.3f}; equal={equal_rho:.3f}.",
        progress,
    )
    return sample_df, best_w, best_perf, score_table, cv_summary, fold_weights, baseline_comparison, model_diagnostics


# -----------------------------------------------------------------------------
# Plot helpers for Fig. 3
# -----------------------------------------------------------------------------

def draw_small_network(ax: plt.Axes, x: float, y: float, w: float, h: float, mode: str) -> None:
    # deterministic, schematic method glyph; not used as data plot.
    rng = np.random.default_rng(abs(hash(mode)) % (2**32))
    n = 18
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = x + w * (0.5 + 0.35 * np.cos(angles) + rng.normal(0, 0.04, n))
    ys = y + h * (0.5 + 0.35 * np.sin(angles) + rng.normal(0, 0.04, n))
    colors = ["#9CA3AF"] * n
    if mode in {"G0", "Gt"}:
        colors[:3] = ["#EF4444"] * 3
    if mode == "Gt":
        colors[5:8] = ["#3B82F6"] * 3
        colors[11:13] = ["#F97316"] * 2
    # edges
    for i in range(n):
        j = (i + rng.integers(2, 6)) % n
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="#D1D5DB", lw=0.55, alpha=0.8, transform=ax.transAxes)
    if mode in {"G0", "Gt"}:
        cx, cy = x + 0.50 * w, y + 0.50 * h
        for idx in [1, 5, 9, 13]:
            ax.plot([cx, xs[idx]], [cy, ys[idx]], color="#EF4444", lw=0.8, alpha=0.75, transform=ax.transAxes)
        ax.scatter([cx], [cy], marker="*", s=80, color="#EF4444", edgecolors="white", linewidths=0.5, transform=ax.transAxes, zorder=4)
    ax.scatter(xs, ys, s=18, color=colors, edgecolors="white", linewidths=0.35, transform=ax.transAxes, zorder=3)


def mechanism_weights_from_samples(samples: pd.DataFrame) -> pd.DataFrame:
    out = samples.copy()
    out["W_expansion"] = out["w_RS"] + out["w_PDE"] + 0.50 * out["w_Uzzi"]
    out["W_bridging"] = out["w_B"] + out["w_RTD"] + out["w_BurtIP"]
    out["W_reconfiguration"] = out["w_DeltaQ0"] + 0.50 * out["w_Uzzi"]
    s = out[["W_expansion", "W_bridging", "W_reconfiguration"]].sum(axis=1)
    for c in ["W_expansion", "W_bridging", "W_reconfiguration"]:
        out[c] = out[c] / s
    return out


def ternary_to_xy(expansion: np.ndarray, bridging: np.ndarray, reconfiguration: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # vertices: expansion top, bridging left, reconfiguration right
    x = 0.5 * expansion + 0.0 * bridging + 1.0 * reconfiguration
    y = math.sqrt(3) / 2 * expansion
    return x, y


def mechanism_profile_candidate_weights(
    expansion: float,
    bridging: float,
    reconfiguration: float,
    n_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return indicator-weight candidates constrained to one mechanism-mix point."""
    n_candidates = max(1, int(n_candidates))
    rows: List[np.ndarray] = []
    max_uzzi = max(0.0, 2.0 * min(float(expansion), float(reconfiguration)))

    def add_candidate(uzzi_weight: float, expansion_share: float, bridging_share: Sequence[float]) -> None:
        w = np.zeros(len(METRIC_KEYS), dtype=float)
        u = float(np.clip(uzzi_weight, 0.0, max_uzzi))
        e_rem = max(0.0, float(expansion) - 0.5 * u)
        r_rem = max(0.0, float(reconfiguration) - 0.5 * u)
        b_rem = max(0.0, float(bridging))
        w[METRIC_KEYS.index("Uzzi")] = u
        w[METRIC_KEYS.index("RS")] = e_rem * float(np.clip(expansion_share, 0.0, 1.0))
        w[METRIC_KEYS.index("PDE")] = e_rem - w[METRIC_KEYS.index("RS")]
        w[METRIC_KEYS.index("DeltaQ0")] = r_rem
        b_share = np.asarray(bridging_share, dtype=float)
        if len(b_share) != 3 or not np.isfinite(b_share).all() or b_share.sum() <= 1e-12:
            b_share = np.ones(3, dtype=float) / 3.0
        else:
            b_share = np.maximum(b_share, 0.0)
            b_share = b_share / b_share.sum()
        for key, val in zip(["B", "RTD", "BurtIP"], b_share * b_rem):
            w[METRIC_KEYS.index(key)] = float(val)
        total = w.sum()
        if total > 1e-12:
            rows.append(w / total)

    add_candidate(0.5 * max_uzzi, 0.5, [1.0, 1.0, 1.0])
    add_candidate(0.0, 0.5, [1.0, 1.0, 1.0])
    add_candidate(max_uzzi, 0.5, [1.0, 1.0, 1.0])
    while len(rows) < n_candidates:
        u = float(rng.uniform(0.0, max_uzzi)) if max_uzzi > 1e-12 else 0.0
        e_share = float(rng.beta(1.2, 1.2))
        b_share = rng.dirichlet(np.ones(3, dtype=float) * 1.2)
        add_candidate(u, e_share, b_share)
    return np.vstack(rows[:n_candidates])


def mechanism_profile_grid(
    comp: ComputedData,
    bins: int = 25,
    profile_n: int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a complete ternary grid of mechanism shares by CV profile search."""
    bins = max(3, int(bins))
    profile_n = max(1, int(profile_n))
    active = [k for k in comp.active_metric_keys if k in METRIC_KEYS]
    st = comp.score_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["RGPM"]).reset_index(drop=True)
    active = [k for k in active if k + "_z" in st.columns]
    if not active:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty, empty, empty
    cache_key = (bins, profile_n, tuple(active), len(st))
    cache = getattr(comp, "_mechanism_profile_grid_cache", None)
    if isinstance(cache, dict) and cache.get("key") == cache_key:
        return cache["value"]
    st[[k + "_z" for k in active]] = st[[k + "_z" for k in active]].fillna(0.0)
    X = st[[k + "_z" for k in active]].to_numpy(dtype=float)
    y = st["RGPM"].to_numpy(dtype=float)
    fold_ids = st["fold_id"].to_numpy(dtype=int) if "fold_id" in st.columns else np.zeros(len(st), dtype=int)
    folds = [np.where(fold_ids == f)[0] for f in sorted(set(fold_ids)) if f > 0 and np.sum(fold_ids == f) > 0]
    if len(folds) < 2:
        folds = make_folds(st, n_folds=3, mode="random", seed=3181)
    equal_w = np.ones(len(active), dtype=float) / len(active)
    equal_rho = safe_spearman(X @ equal_w, y)

    denom = bins - 1
    expansion_vals: List[float] = []
    bridging_vals: List[float] = []
    reconfiguration_vals: List[float] = []
    weights: List[np.ndarray] = []
    owners: List[int] = []
    active_idx = [METRIC_KEYS.index(k) for k in active]
    rng = np.random.default_rng(7919 + bins * 101 + profile_n)
    for e_i in range(denom + 1):
        for b_i in range(denom - e_i + 1):
            r_i = denom - e_i - b_i
            expansion = e_i / denom
            bridging = b_i / denom
            reconfiguration = r_i / denom
            point_idx = len(expansion_vals)
            expansion_vals.append(float(expansion))
            bridging_vals.append(float(bridging))
            reconfiguration_vals.append(float(reconfiguration))
            W_full = mechanism_profile_candidate_weights(expansion, bridging, reconfiguration, profile_n, rng)
            W_active = W_full[:, active_idx]
            row_sum = W_active.sum(axis=1)
            valid = row_sum > 1e-12
            if valid.any():
                W_active = W_active[valid] / row_sum[valid, None]
                weights.append(W_active)
                owners.extend([point_idx] * len(W_active))
    values = np.full(len(expansion_vals), np.nan, dtype=float)
    if weights:
        W_all = np.vstack(weights)
        owner_arr = np.asarray(owners, dtype=int)
        perf = weight_performance_cv(X, y, W_all, folds, progress=False)
        for point_idx in range(len(values)):
            point_perf = perf[owner_arr == point_idx]
            if np.isfinite(point_perf).any() and np.isfinite(equal_rho):
                values[point_idx] = float(np.nanmax(point_perf) - equal_rho)
    x, y_xy = ternary_to_xy(
        np.asarray(expansion_vals, dtype=float),
        np.asarray(bridging_vals, dtype=float),
        np.asarray(reconfiguration_vals, dtype=float),
    )
    result = (
        np.asarray(expansion_vals, dtype=float),
        np.asarray(bridging_vals, dtype=float),
        np.asarray(reconfiguration_vals, dtype=float),
        x,
        y_xy,
        values,
    )
    try:
        setattr(comp, "_mechanism_profile_grid_cache", {"key": cache_key, "value": result})
    except Exception:
        pass
    return result


def radar_axes(ax: plt.Axes, center: Tuple[float, float], radius: float, values: Sequence[float], color: str, label: str) -> None:
    labels = [METRIC_LABELS[k] for k in METRIC_KEYS]
    n = len(labels)
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
    vals = np.asarray(values, dtype=float)
    vals = np.clip(vals, 0.0, 1.0)
    # grid
    for r in [0.33, 0.66, 1.0]:
        pts = [(center[0] + radius * r * math.cos(a), center[1] + radius * r * math.sin(a)) for a in angles]
        pts.append(pts[0])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#D1D5DB", lw=0.55, transform=ax.transAxes, zorder=1)
    for a, lab in zip(angles, labels):
        ax.plot([center[0], center[0] + radius * math.cos(a)], [center[1], center[1] + radius * math.sin(a)], color="#E5E7EB", lw=0.45, transform=ax.transAxes, zorder=1)
        ax.text(center[0] + radius * 1.18 * math.cos(a), center[1] + radius * 1.18 * math.sin(a), lab, ha="center", va="center", fontsize=4.8, transform=ax.transAxes)
    pts = [(center[0] + radius * vals[i] * math.cos(angles[i]), center[1] + radius * vals[i] * math.sin(angles[i])) for i in range(n)]
    pts.append(pts[0])
    ax.fill([p[0] for p in pts], [p[1] for p in pts], color=color, alpha=0.22, transform=ax.transAxes, zorder=3)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=1.25, transform=ax.transAxes, zorder=4)
    ax.scatter([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]], s=10, color=color, edgecolors="white", linewidths=0.30, transform=ax.transAxes, zorder=5)
    ax.text(center[0], center[1] + radius * 1.55, label, ha="center", va="center", fontsize=5.7, fontweight="bold", transform=ax.transAxes)


def diagnostic_status_text(comp: ComputedData) -> str:
    return str(comp.diagnostics_summary.get("status_label", "diagnostic run"))


def is_diagnostic_run(comp: ComputedData) -> bool:
    return not bool(comp.diagnostics_summary.get("overall_pass", False))


# -----------------------------------------------------------------------------
# Panel plotting
# -----------------------------------------------------------------------------

def draw_panel_a(ax: plt.Axes, comp: ComputedData, tau: int) -> None:
    panel_frame(ax, "a", "Empirical learning framework")
    box_specs = [
        (0.040, 0.230, 0.260, 0.610, "Publication-day indicators", "#E0F2FE", "#0284C7"),
        (0.370, 0.230, 0.250, 0.610, "Weight learning", "#F0FDF4", "#16A34A"),
        (0.690, 0.230, 0.270, 0.610, "Out-of-fold validation", "#FFF7ED", "#EA580C"),
    ]
    for x, y, w, h, title, face, edge in box_specs:
        rectangle_box(ax, x, y, w, h, face, edge, 0.90)
        ax.text(x + w / 2, y + h - 0.070, title, ha="center", va="center", fontsize=8.0, fontweight="bold", color=edge)

    indicator_rows = [
        ("B", "Bridge position", METRIC_COLORS["B"]),
        ("RS", "Knowledge breadth", METRIC_COLORS["RS"]),
        ("ΔQ0", "Boundary shift", METRIC_COLORS["DeltaQ0"]),
        ("Uzzi", "Atypical recombination", METRIC_COLORS["Uzzi"]),
        ("RTD", "Reference diversity", METRIC_COLORS["RTD"]),
        ("Burt IP", "Structural holes", METRIC_COLORS["BurtIP"]),
        ("PDE", "Prospective entropy", METRIC_COLORS["PDE"]),
    ]
    y = 0.675
    for label, desc, color in indicator_rows:
        ax.scatter([0.075], [y], s=22, color=color, edgecolors="white", linewidths=0.35, transform=ax.transAxes, zorder=5)
        ax.text(0.096, y, label, ha="left", va="center", fontsize=6.4, fontweight="bold", color=color)
        ax.text(0.158, y, desc, ha="left", va="center", fontsize=5.7, color=TEXT_MID)
        y -= 0.055
    ax.text(0.170, 0.272, "Rank-normalized within\nfield-year / field / global", ha="center", va="center", fontsize=6.1)

    ax.text(0.495, 0.625, r"$S_w(p)=\sum_k w_k z_k(p)$", ha="center", va="center", fontsize=11.0)
    ax.text(0.495, 0.505, r"$w_k\geq0,\quad \sum_k w_k=1$", ha="center", va="center", fontsize=8.2)
    ax.text(
        0.495,
        0.380,
        "Candidate weights are selected\ninside each training fold",
        ha="center",
        va="center",
        fontsize=6.4,
    )

    summary = comp.diagnostics_summary
    ax.text(0.825, 0.660, rf"$G_0 \rightarrow G_{{+\tau}}$  ({tau} y)", ha="center", va="center", fontsize=8.5)
    ax.text(0.825, 0.540, "Target: structural-residual RGPM\nfrom active graph deltas", ha="center", va="center", fontsize=6.6)
    ax.text(
        0.825,
        0.405,
        f"OOF Spearman ρ = {summary.get('learned_oof_spearman', np.nan):.2f}\n"
        f"Δ vs equal = {summary.get('learned_vs_equal_delta', np.nan):.2f}",
        ha="center",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color="#9A3412",
    )

    draw_arrow(ax, (0.302, 0.535), (0.366, 0.535), color="#3B6EA8", lw=1.3, mutation_scale=13)
    draw_arrow(ax, (0.622, 0.535), (0.686, 0.535), color="#3B6EA8", lw=1.3, mutation_scale=13)
    rounded_box(ax, 0.115, 0.140, 0.770, 0.045, blend_with_white("#0F766E", 0.93), "#0F766E", 0.75, 0.012)
    ax.text(
        0.500,
        0.162,
        r"Objective: choose $\hat{w}\in\Delta_+^7$ to maximize $\rho_S(S_w,RGPM_\tau)$ on training folds; apply $\hat{w}$ OOF.",
        ha="center",
        va="center",
        fontsize=6.6,
        color="#115E59",
        fontweight="bold",
    )



def format_z_for_panel(z: float) -> str:
    if not np.isfinite(z):
        return "NA"
    if z > 9.99:
        return ">9.99"
    if z < -9.99:
        return "<-9.99"
    return f"{z:.2f}"

def draw_panel_b(ax: plt.Axes, comp: ComputedData) -> None:
    panel_frame(ax, "b", "Stabilized RGPM-v2 target construction")
    ex = comp.panel_b_example.copy()
    if ex.empty:
        raise ValueError("Panel b example data is empty.")
    rectangle_box(ax, 0.020, 0.115, 0.695, 0.775, "#FFFFFF", BORDER, 0.65)
    ax.text(0.365, 0.850, "Active graph-delta z-scores (higher = stronger perturbation)", ha="center", va="center", fontsize=7.3, color="#0F3A75", fontweight="bold")
    z_axis_x0 = 0.405
    z_axis_w = 0.235
    ax.text(z_axis_x0, 0.790, "-4", ha="center", va="center", fontsize=5.3, color=TEXT_LIGHT)
    ax.text(z_axis_x0 + z_axis_w / 2, 0.790, "0", ha="center", va="center", fontsize=5.3, color=TEXT_LIGHT)
    ax.text(z_axis_x0 + z_axis_w, 0.790, "+4", ha="center", va="center", fontsize=5.3, color=TEXT_LIGHT)
    ax.plot([z_axis_x0, z_axis_x0 + z_axis_w], [0.765, 0.765], color="#CBD5E1", lw=0.8, transform=ax.transAxes)
    ax.plot([z_axis_x0 + z_axis_w / 2, z_axis_x0 + z_axis_w / 2], [0.178, 0.778], color="#9CA3AF", lw=0.6, ls="--", transform=ax.transAxes)
    ax.text(0.665, 0.790, "z", ha="center", va="center", fontsize=5.4)
    y0 = 0.720
    row_step = min(0.074, 0.565 / max(len(ex), 1))
    for i, row in enumerate(ex.itertuples(index=False)):
        y = y0 - i * row_step
        ax.text(0.055, y, getattr(row, "label"), ha="left", va="center", fontsize=5.8)
        z = float(getattr(row, "z"))
        x0 = z_axis_x0 + z_axis_w / 2
        x1 = z_axis_x0 + z_axis_w * ((np.clip(z, -4.0, 4.0) + 4.0) / 8.0)
        color = "#EF4444" if z >= 0 else "#3B82F6"
        ax.add_patch(
            mpatches.Rectangle(
                (min(x0, x1), y - 0.015),
                max(abs(x1 - x0), 0.004),
                0.030,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
                alpha=0.85,
            )
        )
        if int(getattr(row, "clipped", 0)):
            ax.scatter([x1], [y], marker="^" if z >= 0 else "v", s=22, color="#111827", transform=ax.transAxes, zorder=6)
        ax.text(0.665, y, format_z_for_panel(z), ha="center", va="center", fontsize=5.8)
    ax.text(0.365, 0.080, "Controls: field/year/ref-bin matching, with MAD floor and z-cap = 4", ha="center", va="center", fontsize=5.8)

    rectangle_box(ax, 0.745, 0.115, 0.235, 0.775, "#FFFFFF", BORDER, 0.65)
    ax.text(0.862, 0.850, "RGPM-v2", ha="center", va="center", fontsize=7.6, color="#0F3A75", fontweight="bold")
    ax.text(0.862, 0.705, r"$z_j=\frac{\Delta_j-\tilde{\Delta}_{ctrl}}{\max(MAD_{local}, .25MAD_{global}, floor)}$", ha="center", va="center", fontsize=6.0)
    ax.text(0.862, 0.575, r"$z_j \leftarrow clip(z_j,-4,4)$", ha="center", va="center", fontsize=7.1)
    ax.text(0.862, 0.470, r"$RGPM_{v2}=\sqrt{mean(max(z_j,0)^2)}$", ha="center", va="center", fontsize=7.1, color="#1D4ED8", fontweight="bold")
    rounded_box(ax, 0.770, 0.220, 0.185, 0.170, "#F3F4F6", "#CBD5E1", 0.65, 0.010)
    dropped = comp.delta_diagnostics[comp.delta_diagnostics["active"].astype(int) == 0]
    dropped_labels = [DELTA_LABELS.get(k, k) for k in dropped["delta"].astype(str).tolist()]
    dropped_text = ", ".join(dropped_labels[:3])
    if len(dropped_labels) > 3:
        dropped_text += f", +{len(dropped_labels) - 3}"
    ax.text(0.862, 0.335, "Excluded by stability screen", ha="center", va="center", fontsize=5.8, fontweight="bold", color=TEXT_MID)
    ax.text(0.862, 0.265, wrap_text(dropped_text or "None", 25), ha="center", va="center", fontsize=5.2, color=TEXT_LIGHT)


def draw_panel_c(ax: plt.Axes, comp: ComputedData) -> None:
    panel_frame(ax, "c", "Mechanism-level OOF-compatible landscape")
    exp, bri, rec, x, y, perf = mechanism_profile_grid(
        comp,
        bins=int(getattr(comp, "profile_grid_size", 25)),
        profile_n=int(getattr(comp, "profile_n", 80)),
    )
    ax_tri = ax.inset_axes([0.250, 0.120, 0.540, 0.740])
    ax_tri.set_aspect("equal")
    ax_tri.axis("off")
    finite = np.isfinite(perf)
    lim = max(0.03, min(0.20, float(np.nanpercentile(np.abs(perf[finite]), 98)) if finite.any() else 0.05))
    if finite.sum() >= 3:
        tri = mtri.Triangulation(x[finite], y[finite])
        levels = np.linspace(-lim, lim, 17)
        tpc = ax_tri.tricontourf(
            tri,
            perf[finite],
            levels=levels,
            cmap="RdYlBu_r",
            vmin=-lim,
            vmax=lim,
            extend="both",
        )
        ax_tri.triplot(tri, color="white", lw=0.18, alpha=0.30, zorder=2)
    else:
        tpc = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=-lim, vmax=lim), cmap="RdYlBu_r")
        ax_tri.text(0.50, 0.42, "profile grid unavailable", ha="center", va="center", fontsize=7.0, color="#7F1D1D", fontweight="bold")
    verts = np.array([[0, 0], [1, 0], [0.5, math.sqrt(3)/2], [0, 0]])
    ax_tri.plot(verts[:, 0], verts[:, 1], color="#1E3A8A", lw=1.0)
    ax_tri.set_xlim(-0.05, 1.05)
    ax_tri.set_ylim(-0.06, math.sqrt(3) / 2 + 0.10)
    ax_tri.text(0.5, math.sqrt(3)/2 + 0.06, "Expansion\n($W_E$)", ha="center", va="bottom", fontsize=6.5)
    ax_tri.text(-0.02, -0.04, "Bridging\n($W_B$)", ha="right", va="top", fontsize=6.5)
    ax_tri.text(1.02, -0.04, "Reconfiguration\n($W_R$)", ha="left", va="top", fontsize=6.5)
    thr = np.nanpercentile(perf[finite], 90) if finite.any() else np.nan
    top_mask = finite & (perf >= thr)
    if top_mask.sum() >= 10:
        cx, cy = float(np.mean(x[top_mask])), float(np.mean(y[top_mask]))
        dist = np.sqrt((x[top_mask] - cx) ** 2 + (y[top_mask] - cy) ** 2)
        if float(np.mean(dist)) < 0.16:
            pts = np.column_stack([x[top_mask], y[top_mask]])
            if len(pts) >= 3:
                cov = np.cov(pts, rowvar=False)
                vals, vecs = np.linalg.eigh(cov)
                vals = np.clip(vals, 1e-5, None)
                order = np.argsort(vals)[::-1]
                vals = vals[order]
                vecs = vecs[:, order]
                angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
                width, height = 3.2 * np.sqrt(vals)
                ellipse = mpatches.Ellipse(
                    (cx, cy),
                    width=float(width),
                    height=float(height),
                    angle=float(angle),
                    facecolor="none",
                    edgecolor="#111827",
                    linewidth=0.8,
                    linestyle="--",
                    alpha=0.85,
                    zorder=5,
                )
                ax_tri.add_patch(ellipse)
            ax_tri.scatter([cx], [cy], s=18, facecolors="#111827", edgecolors="white", linewidths=0.45, zorder=6)
            label_x = float(np.clip(cx - 0.035 if cx > 0.55 else cx + 0.035, 0.08, 0.92))
            label_ha = "right" if cx > 0.55 else "left"
            ax_tri.text(label_x, cy + 0.045, "top 10% region", ha=label_ha, va="bottom", fontsize=5.5, color=TEXT_DARK)
        else:
            ax_tri.text(0.50, 0.42, "no stable basin", ha="center", va="center", fontsize=7.0, color="#7F1D1D", fontweight="bold")
    best_df = pd.DataFrame([{("w_" + k): float(comp.best_weights[k]) for k in METRIC_KEYS}])
    best_mech = mechanism_weights_from_samples(best_df)
    bx, by = ternary_to_xy(
        best_mech["W_expansion"].to_numpy(dtype=float),
        best_mech["W_bridging"].to_numpy(dtype=float),
        best_mech["W_reconfiguration"].to_numpy(dtype=float),
    )
    ax_tri.scatter(bx, by, marker="*", s=90, color="black", edgecolors="white", linewidths=0.6, zorder=5)
    cax = ax.inset_axes([0.825, 0.210, 0.035, 0.505])
    cb = plt.colorbar(tpc, cax=cax)
    cb.ax.tick_params(labelsize=5)
    cb.set_label("Δρ vs equal", fontsize=5.5)
    rounded_box(ax, 0.035, 0.235, 0.190, 0.500, "#FFFFFF", "#CBD5E1", 0.65, 0.012)
    ax.text(0.055, 0.690, "Mechanism mapping", ha="left", va="center", fontsize=6.4, fontweight="bold")
    rows = [
        ("Expansion", "RS, PDE, Uzzi", "#FACC15"),
        ("Bridging", "B, RTD, Burt IP", "#2CA6A4"),
        ("Reconfiguration", "ΔQ0, Uzzi", "#FF5A5F"),
    ]
    yy = 0.600
    for title, desc, col in rows:
        ax.scatter([0.065], [yy], s=45, color=col, edgecolors="white", linewidths=0.4, transform=ax.transAxes)
        ax.text(0.095, yy, f"{title}\n({desc})", ha="left", va="center", fontsize=5.7, transform=ax.transAxes)
        yy -= 0.140
    ax.text(0.500, 0.060, "Triangular grid shows best CV profile performance at each mechanism mix; star is final all-data weight.", ha="center", va="center", fontsize=5.6)


def pairwise_profile_grid(
    comp: ComputedData,
    xkey: str,
    ykey: str,
    bins: int = 25,
    profile_n: int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Profile two indicators on a rectangular simplex-preserving grid.

    x is the share of the selected pair assigned to xkey. y is the total
    mass assigned to the pair. This keeps every cell valid under non-negative
    simplex weights while showing a complete rectangle.
    """
    centers = np.linspace(0.0, 1.0, bins)
    grid = np.full((bins, bins), np.nan, dtype=float)
    active = list(comp.active_metric_keys)
    if xkey not in active or ykey not in active or xkey == ykey:
        return centers, centers, grid
    st = comp.score_table.replace([np.inf, -np.inf], np.nan).dropna(subset=[k + "_z" for k in active] + ["RGPM"]).reset_index(drop=True)
    X = st[[k + "_z" for k in active]].to_numpy(dtype=float)
    y = st["RGPM"].to_numpy(dtype=float)
    fold_ids = st["fold_id"].to_numpy(dtype=int) if "fold_id" in st.columns else np.zeros(len(st), dtype=int)
    folds = [np.where(fold_ids == f)[0] for f in sorted(set(fold_ids)) if f > 0 and np.sum(fold_ids == f) > 0]
    if len(folds) < 2:
        folds = make_folds(st, n_folds=3, mode="random", seed=2718)
    equal_w = np.ones(len(active), dtype=float) / len(active)
    equal_rho = safe_spearman(X @ equal_w, y)
    rng = np.random.default_rng(abs(hash((xkey, ykey, len(st)))) % (2**32))
    remaining_keys = [k for k in active if k not in {xkey, ykey}]
    ix = active.index(xkey)
    iy = active.index(ykey)
    rem_idx = [active.index(k) for k in remaining_keys]
    for i, pair_share_x in enumerate(centers):
        for j, pair_mass in enumerate(centers):
            wx = float(pair_mass * pair_share_x)
            wy = float(pair_mass * (1.0 - pair_share_x))
            leftover = max(0.0, 1.0 - float(pair_mass))
            if leftover <= 1e-12:
                W = np.zeros((1, len(active)), dtype=float)
                W[0, ix] = wx
                W[0, iy] = wy
            elif rem_idx:
                rem = rng.dirichlet(np.ones(len(rem_idx)), size=int(profile_n)) * leftover
                W = np.zeros((len(rem), len(active)), dtype=float)
                W[:, ix] = wx
                W[:, iy] = wy
                W[:, rem_idx] = rem
            else:
                W = np.zeros((1, len(active)), dtype=float)
                W[0, ix] = wx
                W[0, iy] = wy
            perf = weight_performance_cv(X, y, W, folds, progress=False)
            if np.isfinite(perf).any():
                grid[j, i] = float(np.nanmax(perf) - equal_rho)
    return centers, centers, grid


def draw_panel_d(ax: plt.Axes, comp: ComputedData) -> None:
    panel_frame(ax, "d", "Rectangular pair-weight profile landscapes")
    delta_vals = comp.weight_samples["cv_spearman_delta_vs_equal"].to_numpy(dtype=float)
    lim = max(0.03, min(0.20, float(np.nanpercentile(np.abs(delta_vals[np.isfinite(delta_vals)]), 98)) if np.isfinite(delta_vals).any() else 0.05))
    positions = [(0.070, 0.210, 0.245, 0.610), (0.365, 0.210, 0.245, 0.610), (0.660, 0.210, 0.245, 0.610)]
    last_im = None
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad("#E5E7EB")
    for (xkey, ykey), (x, y, w, h) in zip(PAIRWISE_LANDSCAPES, positions):
        iax = ax.inset_axes([x, y, w, h])
        xs, ys, grid = pairwise_profile_grid(comp, xkey, ykey, bins=comp.profile_grid_size, profile_n=comp.profile_n)
        masked = np.ma.masked_invalid(grid)
        last_im = iax.imshow(masked, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap=cmap, vmin=-lim, vmax=lim)
        iax.set_title(rf"{METRIC_LABELS[xkey]} / {METRIC_LABELS[ykey]} profile", fontsize=7)
        iax.set_xlabel(f"share to {METRIC_LABELS[xkey]}", fontsize=5.5)
        iax.set_ylabel("pair weight mass", fontsize=5.5)
        iax.tick_params(labelsize=5, length=2)
        # mark best
        best = comp.best_weights
        pair_mass = float(best[xkey] + best[ykey])
        pair_share_x = float(best[xkey] / pair_mass) if pair_mass > 1e-12 else 0.5
        iax.scatter([pair_share_x], [pair_mass], s=18, color="black", edgecolors="white", linewidths=0.5)
        try:
            thr = np.nanpercentile(grid, 90)
            iax.contour(xs, ys, grid, levels=[thr], colors="white", linewidths=0.9, linestyles="--")
        except Exception:
            pass
    if last_im is not None:
        cax = ax.inset_axes([0.930, 0.280, 0.025, 0.470])
        cb = plt.colorbar(last_im, cax=cax)
        cb.ax.tick_params(labelsize=5)
        cb.set_label("Best Δρ vs equal", fontsize=5.5)
    ax.text(0.505, 0.085, "Each cell sets pair mass and within-pair share, then profiles the remaining active indicators by resampling.", ha="center", va="center", fontsize=5.7)


def draw_panel_e(ax: plt.Axes, comp: ComputedData) -> None:
    panel_frame(ax, "e", "Weight stability across high-performing configurations")
    samples = comp.weight_samples.dropna(subset=["cv_spearman"]).copy()
    if samples.empty:
        raise ValueError("No weight samples for panel e.")
    top_thr = samples["cv_spearman"].quantile(0.99)
    top = samples[samples["cv_spearman"] >= top_thr].copy()
    x = np.arange(len(METRIC_KEYS))
    weight_cols = ["w_" + k for k in METRIC_KEYS]
    plot_ax = ax.inset_axes([0.070, 0.315, 0.830, 0.520])
    top_vals = top[weight_cols].to_numpy(dtype=float)
    q25 = np.nanpercentile(top_vals, 25, axis=0)
    q50 = np.nanpercentile(top_vals, 50, axis=0)
    q75 = np.nanpercentile(top_vals, 75, axis=0)
    plot_ax.fill_between(x, q25, q75, color="#93C5FD", alpha=0.40, label="Top 1% IQR")
    plot_ax.plot(x, q50, color="#2563EB", lw=1.8, label="Top 1% median")
    best_vals = comp.best_weights.loc[METRIC_KEYS].to_numpy(dtype=float)
    plot_ax.scatter(x, best_vals, s=34, color="black", edgecolors="white", linewidths=0.5, zorder=6, label="Final best")
    fold_weight_cols = [c for c in weight_cols if c in comp.fold_weights.columns]
    if fold_weight_cols:
        for fold_idx, (_, row) in enumerate(comp.fold_weights.iterrows()):
            fold_vals = row[weight_cols].to_numpy(dtype=float)
            plot_ax.scatter(
                x,
                fold_vals,
                s=16,
                color="#F97316",
                alpha=0.45,
                edgecolors="white",
                linewidths=0.25,
                zorder=4,
                label="Outer-fold best" if fold_idx == 0 else None,
            )
    plot_ax.set_xticks(x)
    plot_ax.set_xticklabels([METRIC_LABELS[k] for k in METRIC_KEYS], fontsize=6)
    plot_ax.set_ylabel("Weight $w_k$", fontsize=6)
    plot_ax.set_ylim(0, max(0.65, float(np.nanmax(samples[weight_cols].to_numpy())) * 1.08))
    plot_ax.tick_params(labelsize=5)
    plot_ax.grid(True, axis="y", color="#E5E7EB", lw=0.5)
    for s in plot_ax.spines.values():
        s.set_linewidth(0.5)
    plot_ax.legend(frameon=True, fontsize=5.3, loc="upper right")

    freq_ax = ax.inset_axes([0.070, 0.105, 0.830, 0.125])
    if len(top) > 0:
        top_max = top[weight_cols].idxmax(axis=1).str.replace("w_", "", regex=False)
        freq = pd.Series(METRIC_KEYS).map(lambda k: float((top_max == k).mean())).to_numpy(dtype=float)
    else:
        freq = np.zeros(len(METRIC_KEYS), dtype=float)
    freq_ax.bar(x, freq, color=[METRIC_COLORS[k] for k in METRIC_KEYS], alpha=0.75)
    freq_ax.set_ylim(0, 1)
    freq_ax.set_xticks(x)
    freq_ax.set_xticklabels([METRIC_LABELS[k] for k in METRIC_KEYS], fontsize=5.2)
    freq_ax.set_ylabel("Top-weight\nfrequency", fontsize=5.2)
    freq_ax.tick_params(labelsize=5, length=2)
    freq_ax.grid(True, axis="y", color="#E5E7EB", lw=0.45)
    for s in freq_ax.spines.values():
        s.set_linewidth(0.5)


def minmax01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(lo) or abs(hi - lo) < 1e-9:
        return np.zeros_like(arr) + 0.5
    return (arr - lo) / (hi - lo)


def draw_panel_f(ax: plt.Axes, comp: ComputedData) -> None:
    panel_frame(ax, "f", "Out-of-fold score calibration against RGPM-v2")
    scatter_ax = ax.inset_axes([0.065, 0.185, 0.425, 0.660])
    st = comp.score_table.copy()
    rho = safe_spearman(st["S_w_oof"], st["RGPM"])
    x = st["S_w_oof"].to_numpy(dtype=float)
    y = st["RGPM"].to_numpy(dtype=float)
    xlo, xhi = np.nanpercentile(x[np.isfinite(x)], [1, 99]) if np.isfinite(x).any() else (-1, 1)
    ylo, yhi = np.nanpercentile(y[np.isfinite(y)], [1, 99]) if np.isfinite(y).any() else (-1, 1)
    x_show = np.clip(x, xlo, xhi)
    y_show = np.clip(y, ylo, yhi)
    is_landmark = st["is_landmark"].astype(int).to_numpy() == 1
    score_q = pd.qcut(st["S_w_oof"].rank(method="first"), q=3, labels=["Low", "Mid", "High"])
    colors = np.where(is_landmark, "#EF4444", np.where(score_q.astype(str) == "High", "#60A5FA", "#9CA3AF"))
    scatter_ax.scatter(x_show, y_show, s=14, c=colors, alpha=0.75, edgecolors="white", linewidths=0.25)
    finite = np.isfinite(x_show) & np.isfinite(y_show)
    if finite.sum() >= 5:
        coef = np.polyfit(x_show[finite], y_show[finite], 1)
        xx = np.linspace(np.nanmin(x_show[finite]), np.nanmax(x_show[finite]), 100)
        scatter_ax.plot(xx, coef[0] * xx + coef[1], color="#111827", lw=1.0)
    scatter_ax.set_xlim(xlo, xhi)
    scatter_ax.set_ylim(ylo, yhi)
    scatter_ax.set_xlabel("Composite score $S_w$ (OOF, clipped 1-99%)", fontsize=6)
    scatter_ax.set_ylabel("RGPM-v2 (clipped 1-99%)", fontsize=6)
    scatter_ax.tick_params(labelsize=5)
    title = f"OOF Spearman ρ = {rho:.2f}"
    if is_diagnostic_run(comp):
        title += " | weak"
    scatter_ax.set_title(title, fontsize=7, color="#0F3A75", fontweight="bold")
    scatter_ax.scatter([], [], s=20, color="#EF4444", label="Landmark papers")
    scatter_ax.scatter([], [], s=20, color="#60A5FA", label="High-score papers")
    scatter_ax.scatter([], [], s=20, color="#9CA3AF", label="Other papers")
    scatter_ax.legend(frameon=True, fontsize=5, loc="lower right")
    for s in scatter_ax.spines.values():
        s.set_linewidth(0.5)

    ax.text(0.720, 0.835, "Indicator profiles by OOF score tertile", ha="center", va="center", fontsize=7.2, color="#0F3A75", fontweight="bold")
    bar_ax = ax.inset_axes([0.565, 0.170, 0.385, 0.625])
    groups = ["Low", "Mid", "High"]
    group_colors = {"Low": "#3B82F6", "Mid": "#F59E0B", "High": "#EF4444"}
    y_pos = np.arange(len(METRIC_KEYS))
    offsets = {"Low": -0.22, "Mid": 0.00, "High": 0.22}
    for group in groups:
        sub = st[score_q.astype(str) == group]
        means = []
        ses = []
        for i, key in enumerate(METRIC_KEYS):
            vals = sub[key + "_z"].to_numpy(dtype=float)
            means.append(float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan)
            ses.append(bootstrap_mean_se(vals, seed=comp.profile_n + i + len(group)))
        bar_ax.barh(y_pos + offsets[group], means, height=0.18, color=group_colors[group], alpha=0.75, label=group)
        bar_ax.errorbar(means, y_pos + offsets[group], xerr=ses, fmt="none", ecolor="#111827", elinewidth=0.5, capsize=1.2)
    bar_ax.axvline(0, color="#6B7280", lw=0.7)
    bar_ax.set_yticks(y_pos)
    bar_ax.set_yticklabels([METRIC_LABELS[k] for k in METRIC_KEYS], fontsize=5.5)
    bar_ax.set_xlabel("Mean rank-normalized indicator ± bootstrap SE", fontsize=5.5)
    bar_ax.tick_params(labelsize=5, length=2)
    bar_ax.grid(True, axis="x", color="#E5E7EB", lw=0.45)
    bar_ax.legend(frameon=True, fontsize=5, loc="lower right")
    for s in bar_ax.spines.values():
        s.set_linewidth(0.5)
    ax.text(0.720, 0.082, diagnostic_status_text(comp), ha="center", va="center", fontsize=6.0, color="#7F1D1D" if is_diagnostic_run(comp) else "#166534", fontweight="bold")


# -----------------------------------------------------------------------------
# Figure assembly and exports
# -----------------------------------------------------------------------------

def build_panel_b_example(comp: ComputedData) -> pd.DataFrame:
    # Choose a landmark with highest RGPM; otherwise choose highest RGPM paper.
    merged = comp.rgpm_table.copy()
    if (merged["is_landmark"].astype(int) == 1).any():
        row = merged[merged["is_landmark"].astype(int) == 1].sort_values("RGPM", ascending=False).iloc[0]
    else:
        row = merged.sort_values("RGPM", ascending=False).iloc[0]
    out = []
    for key, label, dnum in DELTA_SPECS:
        if key not in comp.active_delta_keys:
            continue
        out.append(
            {
                "key": key,
                "label": f"{label} ({dnum})",
                "z": float(row[key + "_z"]),
                "z_raw": float(row.get(key + "_z_raw", row[key + "_z"])),
                "clipped": int(row.get(key + "_z_clipped", 0)),
            }
        )
    return pd.DataFrame(out)


def build_diagnostics_summary(
    active_delta_keys: Sequence[str],
    delta_diagnostics: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    score_table: pd.DataFrame,
) -> Dict[str, Any]:
    learned_rows = baseline_comparison[baseline_comparison["model"] == "learned_weight_oof"]
    equal_rows = baseline_comparison[baseline_comparison["model"] == "equal_weights"]
    best_single_rows = baseline_comparison[baseline_comparison["model"] == "best_single_indicator"]
    learned_rho = float(learned_rows["oof_spearman"].iloc[0]) if not learned_rows.empty else float("nan")
    equal_rho = float(equal_rows["oof_spearman"].iloc[0]) if not equal_rows.empty else float("nan")
    best_single_rho = float(best_single_rows["oof_spearman"].iloc[0]) if not best_single_rows.empty else float("nan")
    improvement = learned_rho - equal_rho if np.isfinite(learned_rho) and np.isfinite(equal_rho) else float("nan")
    improvement_vs_best_single = learned_rho - best_single_rho if np.isfinite(learned_rho) and np.isfinite(best_single_rho) else float("nan")
    active_diag = delta_diagnostics[delta_diagnostics["delta"].isin(active_delta_keys)]
    active_cap_max = float(active_diag["z_cap_hit_rate"].max()) if not active_diag.empty else float("nan")
    score_vals = score_table["S_w_oof"].to_numpy(dtype=float) if "S_w_oof" in score_table.columns else np.array([])
    score_vals = score_vals[np.isfinite(score_vals)]
    score_iqr = float(np.percentile(score_vals, 75) - np.percentile(score_vals, 25)) if len(score_vals) else float("nan")
    thresholds = MAIN_FIGURE_THRESHOLDS
    checks = {
        "active_graph_deltas": int(len(active_delta_keys) >= thresholds["active_graph_deltas_min"]),
        "active_delta_z_cap_hit_rate": int(np.isfinite(active_cap_max) and active_cap_max < thresholds["active_delta_z_cap_hit_rate_max"]),
        "oof_spearman": int(np.isfinite(learned_rho) and learned_rho >= thresholds["oof_spearman_min"]),
        "learned_vs_equal": int(np.isfinite(improvement) and improvement >= thresholds["learned_vs_equal_min"]),
        "learned_vs_best_single": int(
            np.isfinite(improvement_vs_best_single)
            and improvement_vs_best_single >= thresholds["learned_vs_best_single_min"]
        ),
        "score_iqr": int(np.isfinite(score_iqr) and score_iqr > thresholds["score_iqr_min"]),
    }
    overall_pass = bool(all(checks.values()))
    return {
        "overall_pass": overall_pass,
        "status_label": "validated empirical association" if overall_pass else "weak empirical association / diagnostic run",
        "thresholds": thresholds,
        "checks": checks,
        "active_graph_deltas": list(active_delta_keys),
        "n_active_graph_deltas": int(len(active_delta_keys)),
        "active_delta_z_cap_hit_rate_max": active_cap_max,
        "learned_oof_spearman": learned_rho,
        "equal_weight_oof_spearman": equal_rho,
        "best_single_oof_spearman": best_single_rho,
        "learned_vs_equal_delta": improvement,
        "learned_vs_best_single_delta": improvement_vs_best_single,
        "score_oof_iqr": score_iqr,
    }


def compute_all(raw: RawData, args: argparse.Namespace) -> ComputedData:
    progress = not getattr(args, "quiet", False)
    progress_interval = int(getattr(args, "progress_interval", 100))
    progress_log("[1/5] Computing publication-day indicators and future graph deltas ...", progress)
    metrics, deltas = compute_indicator_and_delta_tables(
        raw,
        tau=args.tau,
        min_refs=args.min_refs,
        max_papers=args.max_papers,
        progress=progress,
        progress_interval=progress_interval,
    )
    progress_log(f"      eligible papers with metrics: {len(metrics):,}", progress)
    progress_log("[2/5] Computing structural-residual RGPM from stabilized matched-control graph-delta z-scores ...", progress)
    rgpm, delta_diag, control_diag, active_delta_keys, dropped_delta_table = compute_rgpm(
        metrics,
        deltas,
        min_controls=args.min_controls,
        z_cap=args.z_cap,
        tau=args.tau,
        progress=progress,
        progress_interval=progress_interval,
    )
    progress_log(f"      papers with RGPM: {len(rgpm):,}", progress)
    progress_log("[3/5] Rank-normalizing the seven publication-day indicators ...", progress)
    metrics_z, feature_diag, active_metric_keys = field_year_standardize(metrics)
    progress_log(
        f"      rank normalization complete; active metrics={','.join(active_metric_keys)}.",
        progress,
    )
    progress_log("[4/5] Learning weights with strict out-of-fold validation ...", progress)
    weight_samples, best_weights, best_perf, score_table, cv_summary, fold_weights, baseline_comparison, model_diag = learn_weights(
        metrics_z,
        rgpm,
        active_metric_keys=active_metric_keys,
        n_samples=args.n_weight_samples,
        n_folds=args.n_folds,
        cv_mode=args.cv_mode,
        seed=args.seed,
        progress=progress,
    )
    progress_log("      best CV Spearman: " + f"{best_perf:.3f}", progress)
    progress_log("      best weights: " + ", ".join([f"{k}={best_weights[k]:.3f}" for k in METRIC_KEYS]), progress)
    indicator_target_corr = compute_indicator_target_correlations(score_table, rgpm, active_metric_keys)
    rgpm_component_corr = compute_rgpm_component_correlations(rgpm)
    control_tier_audit = compute_control_tier_audit(control_diag)
    nonlinear_diag, _ = compute_nonlinear_upper_bound_diagnostics(
        score_table,
        active_metric_keys=active_metric_keys,
        seed=args.seed,
        cv_mode=args.cv_mode,
        n_folds=args.n_folds,
    )
    if not nonlinear_diag.empty:
        model_diag = pd.concat([model_diag, nonlinear_diag], ignore_index=True, sort=False)
    if getattr(args, "skip_sensitivity", False):
        target_sensitivity = pd.DataFrame()
    else:
        progress_log("      computing target/CV sensitivity matrix ...", progress)
        target_sensitivity = compute_target_sensitivity(
            metrics_z,
            rgpm,
            active_metric_keys=active_metric_keys,
            cv_modes=getattr(args, "sensitivity_cv_modes", ["time_block", "domain", "random"]),
            n_samples=int(getattr(args, "sensitivity_weight_samples", 5000)),
            n_folds=args.n_folds,
            seed=args.seed,
            tau=args.tau,
            progress=False,
        )
    landmark_validation = compute_landmark_validation(score_table, min_controls=max(20, int(args.min_controls // 2)))

    diagnostics_summary = build_diagnostics_summary(
        active_delta_keys,
        delta_diag,
        baseline_comparison,
        score_table,
        control_diag,
        nonlinear_diag,
    )
    domain_diag = pd.DataFrame(diagnostics_summary.get("domain_adequacy", []))
    progress_log(
        f"      diagnostics: {diagnostics_summary['status_label']} "
        f"(OOF rho={diagnostics_summary['learned_oof_spearman']:.3f}, "
        f"Δequal={diagnostics_summary['learned_vs_equal_delta']:.3f}, "
        f"score IQR={diagnostics_summary['score_oof_iqr']:.3f}).",
        progress,
    )
    data_profile = diagnostics_summary.get("data_profile", {})
    progress_log(
        f"      data adequacy: domains={data_profile.get('n_domains', 0)}, "
        f"min papers/domain={data_profile.get('min_papers_per_domain', 0)}, "
        f"min landmark/high cases/domain={data_profile.get('min_landmark_or_high_cases_per_domain', 0)}, "
        f"max relaxed-control rate={data_profile.get('relaxed_control_tier_rate_max_by_domain', np.nan):.2f}.",
        progress,
    )
    dummy = ComputedData(
        paper_metrics=metrics_z,
        graph_deltas=deltas,
        rgpm_table=rgpm,
        weight_samples=weight_samples,
        best_weights=best_weights,
        best_performance=best_perf,
        score_table=score_table,
        cv_summary=cv_summary,
        panel_b_example=pd.DataFrame(),
        active_delta_keys=active_delta_keys,
        active_metric_keys=active_metric_keys,
        delta_diagnostics=delta_diag,
        feature_diagnostics=feature_diag,
        model_diagnostics=model_diag,
        control_diagnostics=control_diag,
        domain_diagnostics=domain_diag,
        indicator_target_correlations=indicator_target_corr,
        rgpm_component_correlations=rgpm_component_corr,
        control_tier_audit=control_tier_audit,
        nonlinear_diagnostics=nonlinear_diag,
        target_sensitivity=target_sensitivity,
        landmark_validation=landmark_validation,
        diagnostics_summary=diagnostics_summary,
        fold_weights=fold_weights,
        baseline_comparison=baseline_comparison,
        profile_grid_size=int(getattr(args, "profile_grid_size", 25)),
        profile_n=int(getattr(args, "profile_n", 80)),
        pair_scan_results=pd.DataFrame(),
        effect_summary=pd.DataFrame(),
        selected_panel_d_pairs=[],
    )
    dummy.panel_b_example = build_panel_b_example(dummy)
    progress_log("[5/5] Scanning pair-weight landscapes and effect summaries ...", progress)
    pair_scan = compute_pair_scan_results(dummy, progress=progress)
    selected_pairs, pair_scan = select_panel_d_pairs(pair_scan, active_metric_keys)
    dummy.pair_scan_results = pair_scan
    dummy.selected_panel_d_pairs = selected_pairs
    dummy.effect_summary = compute_effect_summary(dummy)
    update_v3_diagnostics(dummy)
    progress_log("[5/5] Computation complete.", progress)
    return dummy


def export_tables(comp: ComputedData, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    comp.paper_metrics.to_csv(out_dir / "fig3_publication_day_indicators.csv", index=False)
    comp.graph_deltas.to_csv(out_dir / "fig3_future_graph_deltas.csv", index=False)
    comp.rgpm_table.to_csv(out_dir / "fig3_rgpm_table.csv", index=False)
    comp.weight_samples.to_csv(out_dir / "fig3_weight_search_results.csv", index=False)
    comp.best_weights.rename("weight").to_csv(out_dir / "fig3_best_weights.csv")
    comp.score_table.to_csv(out_dir / "fig3_score_table.csv", index=False)
    comp.score_table.to_csv(out_dir / "fig3_oof_score_table.csv", index=False)
    comp.cv_summary.to_csv(out_dir / "fig3_cv_summary.csv", index=False)
    comp.fold_weights.to_csv(out_dir / "fig3_fold_weights.csv", index=False)
    comp.baseline_comparison.to_csv(out_dir / "fig3_baseline_comparison.csv", index=False)
    comp.panel_b_example.to_csv(out_dir / "fig3_panel_b_example.csv", index=False)
    comp.delta_diagnostics.to_csv(out_dir / "fig3_diagnostics_delta_stability.csv", index=False)
    comp.feature_diagnostics.to_csv(out_dir / "fig3_diagnostics_features.csv", index=False)
    comp.model_diagnostics.to_csv(out_dir / "fig3_diagnostics_model.csv", index=False)
    comp.control_diagnostics.to_csv(out_dir / "fig3_diagnostics_controls.csv", index=False)
    comp.domain_diagnostics.to_csv(out_dir / "fig3_diagnostics_domain_adequacy.csv", index=False)
    comp.indicator_target_correlations.to_csv(out_dir / "fig3_indicator_target_correlations.csv", index=False)
    comp.rgpm_component_correlations.to_csv(out_dir / "fig3_rgpm_component_correlations.csv", index=False)
    comp.control_tier_audit.to_csv(out_dir / "fig3_control_tier_audit.csv", index=False)
    comp.nonlinear_diagnostics.to_csv(out_dir / "fig3_nonlinear_upper_bound.csv", index=False)
    comp.target_sensitivity.to_csv(out_dir / "fig3_target_sensitivity.csv", index=False)
    comp.landmark_validation.to_csv(out_dir / "fig3_landmark_validation.csv", index=False)
    comp.pair_scan_results.to_csv(out_dir / "fig3_pair_scan_results.csv", index=False)
    comp.effect_summary.to_csv(out_dir / "fig3_effect_summary.csv", index=False)
    (out_dir / "fig3_effect_summary.json").write_text(
        json.dumps(effect_summary_dict(comp.effect_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "fig3_diagnostics_summary.json").write_text(
        json.dumps(comp.diagnostics_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def export_core_diagnostics(comp: ComputedData, out_dir: Path) -> None:
    """Always-exported diagnostics needed to interpret weak Fig. 3f results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    comp.score_table.to_csv(out_dir / "fig3_oof_score_table.csv", index=False)
    comp.cv_summary.to_csv(out_dir / "fig3_cv_summary.csv", index=False)
    comp.baseline_comparison.to_csv(out_dir / "fig3_baseline_comparison.csv", index=False)
    comp.delta_diagnostics.to_csv(out_dir / "fig3_diagnostics_delta_stability.csv", index=False)
    comp.domain_diagnostics.to_csv(out_dir / "fig3_diagnostics_domain_adequacy.csv", index=False)
    comp.indicator_target_correlations.to_csv(out_dir / "fig3_indicator_target_correlations.csv", index=False)
    comp.rgpm_component_correlations.to_csv(out_dir / "fig3_rgpm_component_correlations.csv", index=False)
    comp.control_tier_audit.to_csv(out_dir / "fig3_control_tier_audit.csv", index=False)
    comp.nonlinear_diagnostics.to_csv(out_dir / "fig3_nonlinear_upper_bound.csv", index=False)
    comp.target_sensitivity.to_csv(out_dir / "fig3_target_sensitivity.csv", index=False)
    comp.landmark_validation.to_csv(out_dir / "fig3_landmark_validation.csv", index=False)
    comp.pair_scan_results.to_csv(out_dir / "fig3_pair_scan_results.csv", index=False)
    comp.effect_summary.to_csv(out_dir / "fig3_effect_summary.csv", index=False)
    (out_dir / "fig3_effect_summary.json").write_text(
        json.dumps(effect_summary_dict(comp.effect_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "fig3_diagnostics_summary.json").write_text(
        json.dumps(comp.diagnostics_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def draw_single_panel(panel: str, comp: ComputedData, tau: int, out_path: Path) -> None:
    setup_style()
    size_map = {
        "a": (10.5, 4.7),
        "b": (8.8, 4.7),
        "c": (7.2, 4.8),
        "d": (10.2, 4.7),
        "e": (7.2, 4.8),
        "f": (10.2, 4.8),
    }
    fig, ax = plt.subplots(figsize=size_map.get(panel, (8, 5)), dpi=300)
    if panel == "a":
        draw_panel_a(ax, comp, tau)
    elif panel == "b":
        draw_panel_b(ax, comp)
    elif panel == "c":
        draw_panel_c(ax, comp)
    elif panel == "d":
        draw_panel_d(ax, comp)
    elif panel == "e":
        draw_panel_e(ax, comp)
    elif panel == "f":
        draw_panel_f(ax, comp)
    else:
        raise ValueError(f"Unknown panel: {panel}")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def draw_full_figure(comp: ComputedData, tau: int, out_path: Path) -> None:
    setup_style()
    fig = plt.figure(figsize=(20, 12.6), dpi=300)
    if is_diagnostic_run(comp):
        title = "Fig. 3 diagnostic run | Weak empirical association in current data"
        subtitle = "Weights are evaluated with strict out-of-fold scores; panels report instability rather than a strong learned-score claim"
    else:
        title = "Fig. 3 | Validated weight learning for graph-perturbation scoring"
        subtitle = "Weights are selected by their ability to recover stabilized realized structural changes of knowledge graphs"
    fig.text(0.5, 0.985, title, ha="center", va="top", fontsize=18.0, fontweight="bold")
    fig.text(0.5, 0.957, subtitle, ha="center", va="top", fontsize=10.8, color=TEXT_MID)

    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.0, 1.05, 1.0], hspace=0.085, wspace=0.035)
    axes = {
        "a": fig.add_subplot(gs[0, :3]),
        "b": fig.add_subplot(gs[0, 3:]),
        "c": fig.add_subplot(gs[1, :2]),
        "d": fig.add_subplot(gs[1, 2:]),
        "e": fig.add_subplot(gs[2, :2]),
        "f": fig.add_subplot(gs[2, 2:]),
    }
    draw_panel_a(axes["a"], comp, tau)
    draw_panel_b(axes["b"], comp)
    draw_panel_c(axes["c"], comp)
    draw_panel_d(axes["d"], comp)
    draw_panel_e(axes["e"], comp)
    draw_panel_f(axes["f"], comp)

    note = (
        "(1) Seven indicators are computed at publication day (G0) and rank-normalized.  "
        "(2) RGPM-v2 aggregates active, stability-screened graph-delta z-scores relative to matched controls.  "
        "(3) Panel f uses strict out-of-fold scores; Mahalanobis RGPM is debug-only.  "
        f"Status: {diagnostic_status_text(comp)}."
    )
    fig.text(0.015, 0.012, note, ha="left", va="bottom", fontsize=6.4, color=TEXT_DARK)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Empirical Fig. 3 weight-learning and plotting pipeline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_FIG1_DATA_ROOT,
        help="Local data directory. Accepts standard Fig. 3 input, a Fig. 1 domain export directory, "
             f"or a Fig. 1 output root. Default: {DEFAULT_FIG1_DATA_ROOT}",
    )
    parser.add_argument("--domain", type=str, default=DEFAULT_DOMAIN,
                        help=f"Fig. 1 domain subdirectory to read when --data-dir is a root. Default: {DEFAULT_DOMAIN}.")
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS,
                        help="Domain list used by --run-mode multi_domain/both.")
    parser.add_argument("--run-mode", choices=["single_domain", "multi_domain", "both"], default="both",
                        help="Run one or more single-domain analyses, a combined multi-domain analysis, or both.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--include-hybrid-edges", action="store_true",
                        help="When falling back to paper_edges.csv, include bibliographic/cocitation-only edges. "
                             "By default only rows with direct > 0 are used.")
    parser.add_argument("--panel", choices=["a", "b", "c", "d", "e", "f", "all"], default="all", help="Panel to draw, or all.")
    parser.add_argument("--tau", type=int, default=10, help="Future window in years for graph-delta outcomes. Strong-evidence default = 10.")
    parser.add_argument("--analysis-end-year", type=int, default=None, help="Last year available for future graph construction. Default = max year in works.csv.")
    parser.add_argument("--min-refs", type=int, default=5, help="Minimum number of prior references required for a paper.")
    parser.add_argument("--min-controls", type=int, default=50, help="Minimum matched-control target; matching relaxes if not met. A hard floor of 10 is enforced.")
    parser.add_argument("--z-cap", type=float, default=4.0, help="Winsorization cap for matched-control graph-delta z-scores used in RGPM construction.")
    parser.add_argument("--n-weight-samples", type=int, default=30000, help="Number of Dirichlet weight vectors to evaluate.")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--cv-mode", choices=["random", "time", "time_block", "domain"], default="time_block", help="Cross-validation split mode.")
    parser.add_argument("--sensitivity-cv-modes", nargs="+", default=["time_block", "domain", "random"], choices=["random", "time", "time_block", "domain"],
                        help="CV modes evaluated in fig3_target_sensitivity.csv.")
    parser.add_argument("--sensitivity-weight-samples", type=int, default=5000,
                        help="Weight candidates per target/CV sensitivity run.")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Skip target/CV sensitivity export.")
    parser.add_argument("--seed", type=int, default=2027, help="Random seed for weight sampling and CV splits.")
    parser.add_argument("--profile-grid-size", type=int, default=25, help="Grid size for Panel d constrained profile landscapes.")
    parser.add_argument("--profile-n", type=int, default=80, help="Number of remaining-weight samples per Panel d profile cell.")
    parser.add_argument("--max-papers", type=int, default=None, help="Optional limit for debugging; stops after N papers with computed metrics.")
    parser.add_argument("--progress-interval", type=int, default=100,
                        help="Print one compute-progress update every N scanned papers/rows. Default: 100.")
    parser.add_argument("--audit-only", action="store_true",
                        help="Run computation and export diagnostics, but skip figure rendering.")
    parser.add_argument("--export-tables", action="store_true", help="Export all computed intermediate tables.")
    parser.add_argument("--diagnostics", action="store_true", help="Export diagnostic tables and pass/fail summary. Enabled automatically with --export-tables.")
    parser.add_argument("--formats", nargs="+", default=["png", "svg"], choices=["png", "svg", "pdf"], help="Output figure formats.")
    parser.add_argument("--no-prepare-input", action="store_true",
                        help="Read --data-dir directly instead of first preparing normalized Fig. 3 input in --out-dir.")
    parser.add_argument("--fig1-config", type=Path, default=None,
                        help="Optional Fig. 1 YAML config used with --run-fig1-if-missing.")
    parser.add_argument("--fig1-corpus-source", choices=["selected", "raw"], default="selected",
                        help="When preparing from Fig. 1 exports, use works_selected.csv or the larger works_raw.jsonl corpus. Default: selected.")
    parser.add_argument("--run-fig1-if-missing", action="store_true",
                        help="Run the Fig. 1 pipeline to materialize source data if --data-dir does not contain usable exports.")
    parser.add_argument("--no-fig1-cache", action="store_true",
                        help="When running Fig. 1, ignore cached works_raw.jsonl and re-download.")
    parser.add_argument("--openalex-api-key", default=os.getenv("OPENALEX_API_KEY"),
                        help="OpenAlex API key passed through when --run-fig1-if-missing is used.")
    parser.add_argument("--email", default=os.getenv("OPENALEX_EMAIL"),
                        help="OpenAlex contact email passed through when --run-fig1-if-missing is used.")
    parser.add_argument("--quiet", action="store_true", help="Suppress Fig. 3 progress logs.")
    return parser.parse_args()


def namespace_raw_data(raw: RawData, domain: str, community_offset: int) -> RawData:
    works = raw.works.copy()
    citations = raw.citations.copy()
    topics = raw.topics.copy()
    topic_edges = raw.topic_edges.copy()
    prefix = f"{domain}::"
    works["id"] = prefix + works["id"].astype(str)
    works["domain"] = domain
    works["domain_analysis_end_year"] = int(raw.analysis_end_year)
    works["display_community"] = pd.to_numeric(works["display_community"], errors="coerce").fillna(-1).astype(int) + community_offset
    citations["source"] = prefix + citations["source"].astype(str)
    citations["target"] = prefix + citations["target"].astype(str)
    if not topics.empty and "community" in topics.columns:
        topics["community"] = pd.to_numeric(topics["community"], errors="coerce").fillna(-1).astype(int) + community_offset
        if "label" in topics.columns:
            topics["label"] = domain + ": " + topics["label"].astype(str)
    if not topic_edges.empty:
        topic_edges["source_community"] = pd.to_numeric(topic_edges["source_community"], errors="coerce").fillna(-1).astype(int) + community_offset
        topic_edges["target_community"] = pd.to_numeric(topic_edges["target_community"], errors="coerce").fillna(-1).astype(int) + community_offset
    return RawData(
        works=works,
        citations=citations,
        topics=topics,
        topic_edges=topic_edges,
        analysis_end_year=raw.analysis_end_year,
    )


def combine_domain_raws(raw_by_domain: Mapping[str, RawData]) -> RawData:
    names = list(raw_by_domain.keys())
    namespaced = [
        namespace_raw_data(raw_by_domain[name], name, community_offset=100000 * i)
        for i, name in enumerate(names)
    ]
    works = pd.concat([raw.works for raw in namespaced], ignore_index=True)
    citations = pd.concat([raw.citations for raw in namespaced], ignore_index=True)
    topics = pd.concat([raw.topics for raw in namespaced], ignore_index=True)
    topic_edges = pd.concat([raw.topic_edges for raw in namespaced], ignore_index=True)
    analysis_end_year = max(raw.analysis_end_year for raw in namespaced)
    combined = RawData(
        works=works,
        citations=citations,
        topics=topics,
        topic_edges=topic_edges,
        analysis_end_year=int(analysis_end_year),
    )
    landmark_audits = [
        getattr(raw_by_domain[name], "landmark_audit", pd.DataFrame()).assign(source_domain=name)
        for name in names
        if isinstance(getattr(raw_by_domain[name], "landmark_audit", pd.DataFrame()), pd.DataFrame)
        and not getattr(raw_by_domain[name], "landmark_audit", pd.DataFrame()).empty
    ]
    if landmark_audits:
        setattr(combined, "landmark_audit", pd.concat(landmark_audits, ignore_index=True, sort=False))
    validate_input_audit_or_raise(combined, "multi_domain")
    return combined


def load_domain_raw(args: argparse.Namespace, domain: str, progress: bool) -> Tuple[RawData, Path]:
    if args.no_prepare_input:
        source_data_dir = resolve_standard_data_dir(args.data_dir, domain) or args.data_dir
    else:
        source_data_dir = prepare_fig3_input_data(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            domain=domain,
            direct_only=not args.include_hybrid_edges,
            analysis_end_year=args.analysis_end_year,
            fig1_config=args.fig1_config,
            fig1_corpus_source=args.fig1_corpus_source,
            run_fig1_if_missing=args.run_fig1_if_missing,
            use_fig1_cache=not args.no_fig1_cache,
            openalex_api_key=args.openalex_api_key,
            email=args.email,
            progress=progress,
        )
    progress_log(f"Loading real data for domain={domain}: {source_data_dir}", progress)
    raw = load_raw_data(source_data_dir, analysis_end_year=args.analysis_end_year)
    raw = ensure_expected_landmarks(raw, domain, progress=progress)
    validate_input_audit_or_raise(raw, domain)
    progress_log(
        f"Loaded {domain}: {len(raw.works):,} works, {len(raw.citations):,} citations, "
        f"{len(raw.topics):,} topics, analysis_end_year={raw.analysis_end_year}.",
        progress,
    )
    return raw, source_data_dir


def draw_outputs(comp: ComputedData, args: argparse.Namespace, run_out_dir: Path, run_name: str) -> None:
    progress = not args.quiet
    panels_to_draw = ["a", "b", "c", "d", "e", "f"] if args.panel == "all" else [args.panel]
    if args.panel == "all":
        for ext in args.formats:
            out_path = run_out_dir / f"fig3_weight_learning_full.{ext}"
            progress_log(f"[{run_name}] Drawing full Fig. 3 ({ext}): {out_path}", progress)
            draw_full_figure(comp, args.tau, out_path)
            progress_log(f"[{run_name}] Saved {out_path}", progress)
        for p in panels_to_draw:
            out_path = run_out_dir / f"fig3_panel_{p}.png"
            progress_log(f"[{run_name}] Drawing individual panel {p}: {out_path}", progress)
            draw_single_panel(p, comp, args.tau, out_path)
            progress_log(f"[{run_name}] Saved {out_path}", progress)
    else:
        for ext in args.formats:
            out_path = run_out_dir / f"fig3_panel_{args.panel}.{ext}"
            progress_log(f"[{run_name}] Drawing panel {args.panel} ({ext}): {out_path}", progress)
            draw_single_panel(args.panel, comp, args.tau, out_path)
            progress_log(f"[{run_name}] Saved {out_path}", progress)


def run_analysis(raw: RawData, args: argparse.Namespace, run_name: str, run_out_dir: Path) -> ComputedData:
    progress = not args.quiet
    run_out_dir.mkdir(parents=True, exist_ok=True)
    write_input_audit(raw, run_out_dir)
    progress_log(f"[{run_name}] Running Fig. 3 computation in {run_out_dir}", progress)
    comp = compute_all(raw, args)
    progress_log(f"[{run_name}] Exporting core diagnostics to {run_out_dir}", progress)
    export_core_diagnostics(comp, run_out_dir)
    if args.export_tables:
        progress_log(f"[{run_name}] Exporting tables and diagnostics to {run_out_dir}", progress)
        export_tables(comp, run_out_dir)
    if args.audit_only:
        progress_log(f"[{run_name}] Audit-only mode: skipping figure rendering.", progress)
    else:
        draw_outputs(comp, args, run_out_dir, run_name)
    return comp


def select_primary_run(run_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not run_results:
        return {}
    multi = [r for r in run_results if r.get("name") == "multi_domain"]
    if multi:
        selected = multi[0]
        if bool(selected["summary"].get("overall_pass", False)):
            reason = "multi_domain passed all main-figure thresholds"
        else:
            reason = "multi_domain selected as the required evidence base; single-domain weights remain diagnostic"
    else:
        stable_singles = [
            r for r in run_results
            if r.get("kind") == "single_domain"
            and bool(r["summary"].get("checks", {}).get("active_graph_deltas", 0))
            and bool(r["summary"].get("checks", {}).get("active_delta_z_cap_hit_rate", 0))
        ]
        if stable_singles:
            selected = max(stable_singles, key=lambda r: float(r["summary"].get("learned_oof_spearman", -999)))
            reason = "best stable single-domain OOF Spearman"
        else:
            selected = max(run_results, key=lambda r: float(r["summary"].get("learned_oof_spearman", -999)))
            reason = "no run passed stability thresholds; selected best diagnostic OOF Spearman"
    return {
        "selected_run": selected["name"],
        "selected_kind": selected["kind"],
        "selected_out_dir": str(selected["out_dir"]),
        "reason": reason,
        "summary": selected["summary"],
        "all_runs": [
            {
                "name": r["name"],
                "kind": r["kind"],
                "out_dir": str(r["out_dir"]),
                "overall_pass": bool(r["summary"].get("overall_pass", False)),
                "learned_oof_spearman": r["summary"].get("learned_oof_spearman"),
                "status_label": r["summary"].get("status_label"),
                "n_domains": r["summary"].get("data_profile", {}).get("n_domains"),
                "min_papers_per_domain": r["summary"].get("data_profile", {}).get("min_papers_per_domain"),
            }
            for r in run_results
        ],
    }


def copy_selected_outputs(selection: Mapping[str, Any], args: argparse.Namespace) -> None:
    if not selection:
        return
    source_dir = Path(str(selection["selected_out_dir"]))
    for ext in args.formats:
        if args.panel == "all":
            src = source_dir / f"fig3_weight_learning_full.{ext}"
            dst = args.out_dir / f"fig3_selected_weight_learning_full.{ext}"
        else:
            src = source_dir / f"fig3_panel_{args.panel}.{ext}"
            dst = args.out_dir / f"fig3_selected_panel_{args.panel}.{ext}"
        if src.exists():
            shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    progress = not args.quiet
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.export_tables:
        args.diagnostics = True
    progress_log(f"Starting Fig. 3 empirical pipeline: panel={args.panel}, run_mode={args.run_mode}", progress)

    domains = list(dict.fromkeys(args.domains or [args.domain]))
    raw_by_domain: Dict[str, RawData] = {}
    run_results: List[Dict[str, Any]] = []

    if args.run_mode in {"single_domain", "both", "multi_domain"}:
        for domain in domains:
            try:
                raw, _ = load_domain_raw(args, domain, progress)
                raw_by_domain[domain] = raw
            except FileNotFoundError as exc:
                progress_log(f"[{domain}] Skipping domain because input is unavailable: {exc}", progress)

    if args.run_mode in {"single_domain", "both"}:
        for domain, raw in raw_by_domain.items():
            try:
                comp = run_analysis(raw, args, domain, args.out_dir / domain)
            except ValueError as exc:
                progress_log(f"[{domain}] Skipping completed input because analysis failed: {exc}", progress)
                continue
            run_results.append(
                {
                    "name": domain,
                    "kind": "single_domain",
                    "out_dir": args.out_dir / domain,
                    "summary": comp.diagnostics_summary,
                }
            )

    if args.run_mode in {"multi_domain", "both"}:
        if len(raw_by_domain) >= 2:
            progress_log(f"[multi_domain] Combining {len(raw_by_domain)} domains: {', '.join(raw_by_domain)}", progress)
            multi_raw = combine_domain_raws(raw_by_domain)
            multi_input_dir = args.out_dir / "fig3_input" / "multi_domain"
            write_raw_data(multi_raw, multi_input_dir)
            write_input_audit(multi_raw, multi_input_dir)
            save_prepare_report(
                {
                    "source_kind": "combined_fig3_input",
                    "domains": list(raw_by_domain),
                    "prepared_dir": str(multi_input_dir),
                    "works_rows": len(multi_raw.works),
                    "citation_rows": len(multi_raw.citations),
                    "analysis_end_year": int(multi_raw.analysis_end_year),
                    "input_audit": getattr(multi_raw, "input_audit", pd.DataFrame()).to_dict("records")
                    if isinstance(getattr(multi_raw, "input_audit", pd.DataFrame()), pd.DataFrame)
                    else [],
                },
                multi_input_dir,
            )
            try:
                comp = run_analysis(multi_raw, args, "multi_domain", args.out_dir / "multi_domain")
                run_results.append(
                    {
                        "name": "multi_domain",
                        "kind": "multi_domain",
                        "out_dir": args.out_dir / "multi_domain",
                        "summary": comp.diagnostics_summary,
                    }
                )
            except ValueError as exc:
                progress_log(f"[multi_domain] Skipping combined run because analysis failed: {exc}", progress)
        else:
            progress_log("[multi_domain] Skipping combined run because fewer than two domains are available.", progress)

    if not run_results:
        raise FileNotFoundError("No Fig. 3 runs completed. Check --data-dir, --domain/--domains, or use --run-fig1-if-missing.")
    selection = select_primary_run(run_results)
    (args.out_dir / "fig3_run_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    copy_selected_outputs(selection, args)
    progress_log(
        f"Selected run: {selection.get('selected_run')} ({selection.get('reason')}). "
        f"Selection report: {args.out_dir / 'fig3_run_selection.json'}",
        progress,
    )
    progress_log("Done.", progress)



# =============================================================================
# v3 patch: structural-residual RGPM target + hybrid OOF weight learning
# =============================================================================
# This patch intentionally keeps the original data-loading and plotting
# infrastructure but replaces the two weakest methodological components found in
# diagnostic runs:
#   (1) RGPM-v2 could become too narrow because it dropped unstable deltas.
#       v3 keeps all observed deltas, assigns empirical reliability weights, and
#       balances deltas by mechanism so reconfiguration cannot dominate only
#       because it has more surviving deltas.
#   (2) Pure Dirichlet search is noisy and can over-emphasize random candidate
#       weights. v3 uses a hybrid candidate pool: Dirichlet exploration + equal
#       weights + single indicators + non-negative ridge weights learned inside
#       each training fold. OOF scores remain strictly out-of-fold.

DELTA_MECHANISM_GROUPS_V3: Dict[str, List[str]] = {
    "Breadth": ["community_reach", "field_entropy", "cross_community_adoption"],
    "Brokerage": ["path_shortening", "hub_formation"],
    "Boundary": ["modularity_shock", "partition_change", "boundary_mixing"],
    "Consolidation": ["post_perturbation_concentration"],
}

RIDGE_LAMBDAS_V3 = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def robust_delta_scale(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    mad = raw_mad(arr)
    q25, q75 = np.percentile(arr, [25, 75])
    iqr_scale = float((q75 - q25) / 1.349) if np.isfinite(q75 - q25) else 0.0
    std_scale = float(np.nanstd(arr))
    q10, q90 = np.percentile(arr, [10, 90])
    range_scale = float((q90 - q10) / 2.563) if np.isfinite(q90 - q10) else 0.0
    return max(
        float(mad) if np.isfinite(mad) else 0.0,
        iqr_scale,
        0.25 * std_scale,
        range_scale,
    )


def delta_reliability_v3(
    nonzero_rate: float,
    cap_hit_rate: float,
    control_mad_zero_rate: float,
    floor_use_rate: float,
    global_mad: float,
    is_primary: bool,
) -> float:
    """Empirical reliability weight in [0, 1] for one graph-delta outcome.

    The weight is deliberately continuous instead of a hard keep/drop rule.
    This prevents RGPM from collapsing to only 2-3 deltas while still reducing
    the influence of sparse, capped, or nearly-zero-control-variance outcomes.
    """
    if (not np.isfinite(global_mad)) or global_mad < DELTA_GLOBAL_MAD_MIN:
        return 0.0
    nz = min(1.0, max(0.0, nonzero_rate / 0.15))
    clip = 1.0 - min(1.0, max(0.0, cap_hit_rate / 0.35))
    mad = 1.0 - min(1.0, max(0.0, control_mad_zero_rate / 0.85))
    floor = 1.0 - 0.45 * min(1.0, max(0.0, floor_use_rate))
    base = (0.10 + 0.90 * nz) * (0.20 + 0.80 * clip) * (0.20 + 0.80 * mad) * max(0.20, floor)
    if not is_primary:
        base *= 0.50
    return float(np.clip(base, 0.0, 1.0))


def structural_residual_target(
    table: pd.DataFrame,
    raw_col: str,
    tau: int,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Residualize raw RGPM against popularity/size covariates and rank it."""
    df = table.copy()
    y = pd.to_numeric(df[raw_col], errors="coerce").to_numpy(dtype=float)
    covariate_cols = ["year", "reference_count", "n_future_citers", "cited_by_count"]
    numeric_blocks: List[np.ndarray] = [np.ones((len(df), 1), dtype=float)]
    for col in covariate_cols:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float) if col in df.columns else np.zeros(len(df), dtype=float)
        if col in {"reference_count", "n_future_citers", "cited_by_count"}:
            vals = np.log1p(np.maximum(vals, 0.0))
        vals = vals.astype(float)
        sd = float(np.nanstd(vals))
        if sd > 1e-12:
            vals = (vals - float(np.nanmean(vals))) / sd
        else:
            vals = np.zeros_like(vals)
        numeric_blocks.append(vals.reshape(-1, 1))
    if "domain" in df.columns:
        dummies = pd.get_dummies(df["domain"].astype(str), prefix="domain", drop_first=True, dtype=float)
        if not dummies.empty:
            numeric_blocks.append(dummies.to_numpy(dtype=float))
    X = np.hstack(numeric_blocks)
    mask = np.isfinite(y)
    fitted = np.full(len(df), np.nan, dtype=float)
    residual = np.full(len(df), np.nan, dtype=float)
    if mask.sum() >= max(8, X.shape[1] + 2):
        Xf = X[mask]
        yf = y[mask]
        lam = 1e-3
        try:
            beta = np.linalg.solve(Xf.T @ Xf + lam * np.eye(Xf.shape[1]), Xf.T @ yf)
        except Exception:
            beta = np.linalg.pinv(Xf.T @ Xf + lam * np.eye(Xf.shape[1])) @ yf
        fitted[mask] = Xf @ beta
        residual[mask] = yf - fitted[mask]
    else:
        residual[mask] = y[mask] - np.nanmean(y[mask]) if mask.any() else np.nan
        fitted[mask] = np.nanmean(y[mask]) if mask.any() else np.nan
    ranked = pd.Series(residual).rank(method="average", pct=True).to_numpy(dtype=float)
    return (
        pd.Series(ranked, index=table.index, name=f"RGPM_structural_residual_tau{tau}"),
        pd.Series(residual, index=table.index, name=f"RGPM_structural_residual_raw_tau{tau}"),
        pd.Series(fitted, index=table.index, name=f"RGPM_popularity_fitted_tau{tau}"),
    )


def compute_rgpm(
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    min_controls: int = 20,
    z_cap: float = 4.0,
    tau: int = 10,
    progress: bool = True,
    progress_interval: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
    """Compute structural-residual RGPM from matched-control graph deltas.

    The raw mechanism score aggregates positive, matched-control z-scores within
    the primary graph-delta mechanisms. The main target then residualizes that
    raw score against domain/year/reference/citation covariates and uses the
    residual rank as the tau-specific structural-residual RGPM column.
    """
    metric_cols = ["paper_id", "title", "domain", "year", "primary_field", "display_community", "is_landmark", "reference_count"]
    if "cited_by_count" in metrics.columns:
        metric_cols.append("cited_by_count")
    delta_cols = ["paper_id"] + DELTA_KEYS
    if "n_future_citers" in deltas.columns:
        delta_cols.append("n_future_citers")
    df = metrics[metric_cols].merge(deltas[delta_cols], on="paper_id", how="inner")
    df = add_reference_bins(df)
    required_controls = max(10, int(min_controls))
    progress_interval = max(1, int(progress_interval))
    progress_log(
        f"Computing structural-residual RGPM for {len(df):,} papers with min_controls={required_controls}, z_cap={z_cap}.",
        progress,
    )
    global_scale = {col: robust_delta_scale(df[col].to_numpy(dtype=float)) for col in DELTA_KEYS}
    z_rows: List[Dict[str, object]] = []
    control_rows: List[Dict[str, object]] = []
    skipped_controls = 0
    control_sizes: List[int] = []

    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        if row_idx == 1 or row_idx % progress_interval == 0 or row_idx == len(df):
            progress_log(
                f"  RGPM rows processed {row_idx:,}/{len(df):,}; kept={len(z_rows):,}; skipped_controls={skipped_controls:,}",
                progress,
            )
        ctrl_idx, tier = matched_control_indices_with_tier(df, row, min_controls=required_controls)
        if len(ctrl_idx) < required_controls:
            skipped_controls += 1
            continue
        controls = df.loc[ctrl_idx]
        control_sizes.append(int(len(ctrl_idx)))
        out: Dict[str, object] = {
            "paper_id": row["paper_id"],
            "title": row["title"],
            "domain": row["domain"],
            "year": row["year"],
            "primary_field": row["primary_field"],
            "display_community": row["display_community"],
            "is_landmark": row["is_landmark"],
            "reference_count": row["reference_count"],
            "cited_by_count": row.get("cited_by_count", np.nan),
            "n_future_citers": row.get("n_future_citers", np.nan),
            "n_controls": int(len(ctrl_idx)),
            "control_tier": tier,
        }
        control_out: Dict[str, object] = {
            "paper_id": row["paper_id"],
            "domain": row["domain"],
            "year": row["year"],
            "primary_field": row["primary_field"],
            "is_landmark": row["is_landmark"],
            "n_controls": int(len(ctrl_idx)),
            "control_tier": tier,
        }
        n_floor_used = 0
        n_mad_zero = 0
        n_clipped = 0
        for col in DELTA_KEYS:
            med = float(np.median(controls[col].to_numpy(dtype=float)))
            local_mad = raw_mad(controls[col].to_numpy(dtype=float))
            local_robust = robust_delta_scale(controls[col].to_numpy(dtype=float))
            global_floor = 0.25 * float(global_scale.get(col, 0.0) if np.isfinite(global_scale.get(col, np.nan)) else 0.0)
            delta_floor = float(DELTA_FLOORS.get(col, DEFAULT_DELTA_FLOOR))
            scale = max(
                float(local_mad) if np.isfinite(local_mad) else 0.0,
                0.50 * float(local_robust) if np.isfinite(local_robust) else 0.0,
                global_floor,
                delta_floor,
            )
            scale *= float(DELTA_SCALE_MULTIPLIERS.get(col, 1.0))
            floor_used = (not np.isfinite(local_mad)) or local_mad < scale - 1e-12
            mad_zero = (not np.isfinite(local_mad)) or local_mad < 1e-6
            z_raw = float((float(row[col]) - med) / max(scale, 1e-12))
            z_val = z_raw
            if z_cap is not None and z_cap > 0:
                z_val = float(np.clip(z_val, -z_cap, z_cap))
            clipped = abs(z_val - z_raw) > 1e-9
            out[col] = float(row[col])
            out[col + "_z"] = z_val
            out[col + "_z_raw"] = z_raw
            out[col + "_z_clipped"] = int(clipped)
            out[col + "_control_median"] = med
            out[col + "_control_mad"] = float(local_mad) if np.isfinite(local_mad) else np.nan
            out[col + "_control_robust_scale"] = float(local_robust) if np.isfinite(local_robust) else np.nan
            out[col + "_scale_used"] = float(scale)
            out[col + "_scale_floor_used"] = int(floor_used)
            control_out[col + "_control_mad"] = float(local_mad) if np.isfinite(local_mad) else np.nan
            control_out[col + "_control_robust_scale"] = float(local_robust) if np.isfinite(local_robust) else np.nan
            control_out[col + "_scale_used"] = float(scale)
            control_out[col + "_scale_floor_used"] = int(floor_used)
            control_out[col + "_mad_zero"] = int(mad_zero)
            control_out[col + "_z_clipped"] = int(clipped)
            n_floor_used += int(floor_used)
            n_mad_zero += int(mad_zero)
            n_clipped += int(clipped)
        control_out["n_delta_scale_floor_used"] = n_floor_used
        control_out["n_delta_mad_zero"] = n_mad_zero
        control_out["n_delta_z_clipped"] = n_clipped
        z_rows.append(out)
        control_rows.append(control_out)

    zdf = pd.DataFrame(z_rows)
    if zdf.empty:
        raise ValueError("Could not compute matched-control z-scores. Check landmark/control coverage and min_controls.")
    control_diag = pd.DataFrame(control_rows)

    # Delta diagnostics + reliability weights.
    primary_set = set(PRIMARY_RGPM_DELTA_KEYS)
    delta_diag_rows: List[Dict[str, object]] = []
    for col in DELTA_KEYS:
        values = df[col].to_numpy(dtype=float)
        nonzero_rate = float(np.mean(np.abs(values[np.isfinite(values)]) > 1e-12)) if np.isfinite(values).any() else 0.0
        cap_hit_rate = float(np.nanmean(zdf[col + "_z_clipped"].to_numpy(dtype=float)))
        control_mad_zero_rate = float(np.nanmean(control_diag[col + "_mad_zero"].to_numpy(dtype=float)))
        floor_use_rate = float(np.nanmean(control_diag[col + "_scale_floor_used"].to_numpy(dtype=float)))
        global_mad = float(global_scale.get(col, np.nan))
        rel = delta_reliability_v3(
            nonzero_rate=nonzero_rate,
            cap_hit_rate=cap_hit_rate,
            control_mad_zero_rate=control_mad_zero_rate,
            floor_use_rate=floor_use_rate,
            global_mad=global_mad,
            is_primary=col in primary_set,
        )
        is_primary = col in primary_set
        contributing = bool(
            is_primary
            and rel >= CONTRIBUTING_DELTA_RELIABILITY_MIN
            and control_mad_zero_rate <= CONTRIBUTING_DELTA_CONTROL_MAD_ZERO_MAX
            and np.isfinite(global_mad)
            and global_mad >= DELTA_GLOBAL_MAD_MIN
        )
        reasons: List[str] = []
        if is_primary and not contributing:
            if rel < CONTRIBUTING_DELTA_RELIABILITY_MIN:
                reasons.append("low_reliability_weight")
            if control_mad_zero_rate > CONTRIBUTING_DELTA_CONTROL_MAD_ZERO_MAX:
                reasons.append("near_constant_controls")
            if (not np.isfinite(global_mad)) or global_mad < DELTA_GLOBAL_MAD_MIN:
                reasons.append("near_constant_global_delta")
        if col not in primary_set:
            reasons.append("diagnostic_or_consolidation_outcome")
        delta_diag_rows.append(
            {
                "delta": col,
                "label": DELTA_LABELS[col],
                "primary_candidate": int(col in primary_set),
                "active": int(col in primary_set),
                "contributing": int(contributing),
                "diagnostic_only": int(is_primary and not contributing),
                "nonzero_rate": nonzero_rate,
                "global_mad": global_mad,
                "z_cap_hit_rate": cap_hit_rate,
                "control_mad_zero_rate": control_mad_zero_rate,
                "scale_floor_use_rate": floor_use_rate,
                "reliability_weight": rel,
                "drop_reasons": ";".join(reasons),
            }
        )
    delta_diag = pd.DataFrame(delta_diag_rows)

    active_delta_keys = delta_diag.loc[delta_diag["active"].astype(int) == 1, "delta"].astype(str).tolist()
    if not active_delta_keys:
        fallback = delta_diag.sort_values("reliability_weight", ascending=False).head(3)["delta"].astype(str).tolist()
        active_delta_keys = fallback
        delta_diag.loc[delta_diag["delta"].isin(fallback), "active"] = 1
        delta_diag.loc[delta_diag["delta"].isin(fallback), "drop_reasons"] = "fallback_active_for_diagnostic_run"

    reliability = dict(zip(delta_diag["delta"].astype(str), delta_diag["reliability_weight"].astype(float)))
    active_delta_set = set(active_delta_keys)
    contributing_delta_keys = delta_diag.loc[delta_diag["contributing"].astype(int) == 1, "delta"].astype(str).tolist()
    contributing_delta_set = set(contributing_delta_keys)
    if not contributing_delta_keys:
        fallback = delta_diag[
            delta_diag["delta"].isin(PRIMARY_RGPM_DELTA_KEYS)
        ].sort_values("reliability_weight", ascending=False).head(3)["delta"].astype(str).tolist()
        contributing_delta_keys = fallback
        contributing_delta_set = set(fallback)
        delta_diag.loc[delta_diag["delta"].isin(fallback), "contributing"] = 1
        delta_diag.loc[delta_diag["delta"].isin(fallback), "diagnostic_only"] = 0
        delta_diag.loc[delta_diag["delta"].isin(fallback), "drop_reasons"] = "fallback_contributing_for_diagnostic_run"
    mechanism_scores: Dict[str, np.ndarray] = {}
    mechanism_weight_sums: Dict[str, float] = {}
    for mech, keys in DELTA_MECHANISM_GROUPS_V3.items():
        if mech == "Consolidation":
            valid_keys = [k for k in keys if k in DELTA_KEYS and reliability.get(k, 0.0) > 0.0]
        else:
            valid_keys = [k for k in keys if k in contributing_delta_set]
        if not valid_keys:
            continue
        Z = zdf[[k + "_z" for k in valid_keys]].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        Zpos = np.maximum(Z, 0.0)
        w = np.asarray(
            [
                max(0.0, reliability.get(k, 0.0))
                for k in valid_keys
            ],
            dtype=float,
        )
        if w.sum() <= 0:
            continue
        # Downweight consolidation unless it is strongly reliable; it is useful but
        # should not redefine perturbation as only later concentration.
        if mech == "Consolidation":
            w = 0.50 * w
        score = np.sqrt((np.square(Zpos) @ w) / max(w.sum(), 1e-12))
        mechanism_scores[mech] = score
        mechanism_weight_sums[mech] = float(w.sum())
        zdf[f"RGPM_{mech.lower()}_component"] = score

    main_mechs = [m for m in ["Breadth", "Brokerage", "Boundary"] if m in mechanism_scores]
    if not main_mechs:
        raise ValueError("No mechanism scores could be constructed for structural-residual RGPM.")
    M = np.column_stack([mechanism_scores[m] for m in main_mechs])
    rgpm_main = np.sqrt(np.square(M).mean(axis=1))
    rgpm_raw = rgpm_main
    zdf["RGPM_v3_balanced"] = rgpm_raw
    zdf["RGPM_v2"] = rgpm_raw
    zdf["RGPM_simple"] = rgpm_raw
    zdf["RGPM_primary_only"] = rgpm_raw
    residual_rank, residual_raw, popularity_fit = structural_residual_target(zdf, "RGPM_v3_balanced", tau=tau)
    zdf[residual_rank.name] = residual_rank
    zdf[residual_raw.name] = residual_raw
    zdf[popularity_fit.name] = popularity_fit
    zdf["RGPM"] = residual_rank.to_numpy(dtype=float)

    zdf["n_contributing_graph_deltas"] = int(len(contributing_delta_keys))
    active_z_cols = [c + "_z" for c in contributing_delta_keys]
    zmat = zdf[active_z_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    cov = np.cov(zmat, rowvar=False)
    if cov.ndim == 0:
        cov = np.eye(len(active_z_cols))
    lam = 0.25
    diag = np.diag(np.diag(cov))
    cov_shrink = (1.0 - lam) * cov + lam * diag + np.eye(len(active_z_cols)) * 1e-6
    inv_cov = np.linalg.pinv(cov_shrink)
    zdf["RGPM_mahalanobis_debug"] = np.sqrt(np.einsum("ij,jk,ik->i", zmat, inv_cov, zmat))
    dropped_delta_table = delta_diag[delta_diag["active"].astype(int) == 0].copy()

    if control_sizes:
        progress_log(
            f"Finished structural-residual RGPM: {len(zdf):,} papers kept; controls median={np.median(control_sizes):.0f}, "
            f"min={np.min(control_sizes):.0f}, max={np.max(control_sizes):.0f}; active deltas="
            f"{len(active_delta_keys)} ({', '.join(active_delta_keys)}).",
            progress,
        )
    return zdf, delta_diag, control_diag, active_delta_keys, dropped_delta_table


def simplex_normalize(beta: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    beta = np.asarray(beta, dtype=float)
    beta = np.where(np.isfinite(beta), beta, 0.0)
    beta = np.maximum(beta, 0.0)
    if beta.sum() <= 1e-12:
        if fallback is not None and np.asarray(fallback).sum() > 0:
            fb = np.maximum(np.asarray(fallback, dtype=float), 0.0)
            return fb / fb.sum()
        return np.ones(len(beta), dtype=float) / len(beta)
    return beta / beta.sum()


def positive_ridge_candidates(X: np.ndarray, y: np.ndarray, lambdas: Sequence[float] = RIDGE_LAMBDAS_V3) -> np.ndarray:
    """Non-negative simplex ridge candidates fitted on the provided data only."""
    X = np.asarray(X, dtype=float)
    y = rank_normal_scores(y)
    y = np.where(np.isfinite(y), y, 0.0)
    p = X.shape[1]
    if p == 0:
        return np.empty((0, 0))
    XtX = X.T @ X
    Xty = X.T @ y
    candidates: List[np.ndarray] = []
    # single-indicator correlations as a fallback direction.
    cors = np.asarray([max(0.0, safe_spearman(X[:, i], y)) for i in range(p)], dtype=float)
    corr_fb = simplex_normalize(cors)
    for lam in lambdas:
        try:
            beta = np.linalg.solve(XtX + float(lam) * np.eye(p), Xty)
        except Exception:
            beta = np.linalg.pinv(XtX + float(lam) * np.eye(p)) @ Xty
        candidates.append(simplex_normalize(beta, fallback=corr_fb))
    candidates.append(corr_fb)
    candidates.append(np.ones(p, dtype=float) / p)
    # Deduplicate approximately.
    W = np.vstack(candidates)
    W = np.round(W, 8)
    _, idx = np.unique(W, axis=0, return_index=True)
    return W[np.sort(idx)]


def base_weight_candidates(n_samples: int, seed: int, n_metrics: int) -> np.ndarray:
    W_rand = generate_dirichlet_weights(n_samples, seed=seed, alpha=1.0, n_metrics=n_metrics)
    W_equal = np.ones((1, n_metrics), dtype=float) / n_metrics
    W_single = np.eye(n_metrics, dtype=float)
    # A small low-entropy and high-entropy set helps sample sparse and broad scores.
    W_sparse = generate_dirichlet_weights(max(500, n_samples // 20), seed=seed + 17, alpha=0.25, n_metrics=n_metrics)
    W_broad = generate_dirichlet_weights(max(500, n_samples // 20), seed=seed + 31, alpha=3.0, n_metrics=n_metrics)
    W = np.vstack([W_rand, W_sparse, W_broad, W_equal, W_single])
    W = np.round(W, 8)
    _, idx = np.unique(W, axis=0, return_index=True)
    return W[np.sort(idx)]


def pair_structured_candidates(active_metric_keys: Sequence[str], grid_n: int = 9) -> np.ndarray:
    """Deterministic pair-mass/share candidates used by training and Panel d scans."""
    active = [key for key in active_metric_keys if key in METRIC_KEYS]
    n = len(active)
    if n <= 0:
        return np.empty((0, 0), dtype=float)
    if n == 1:
        return np.ones((1, 1), dtype=float)
    masses = np.linspace(0.15, 1.0, max(3, int(grid_n)))
    shares = np.linspace(0.0, 1.0, max(3, int(grid_n)))
    rows: List[np.ndarray] = []
    for ix, iy in combinations(range(n), 2):
        rem_idx = [j for j in range(n) if j not in {ix, iy}]
        for pair_mass in masses:
            for share_x in shares:
                w = np.zeros(n, dtype=float)
                w[ix] = float(pair_mass * share_x)
                w[iy] = float(pair_mass * (1.0 - share_x))
                leftover = max(0.0, 1.0 - float(pair_mass))
                if rem_idx and leftover > 1e-12:
                    w[rem_idx] = leftover / len(rem_idx)
                elif leftover > 1e-12:
                    w[ix] += leftover * 0.5
                    w[iy] += leftover * 0.5
                rows.append(w)
    rows.append(np.ones(n, dtype=float) / n)
    rows.extend(np.eye(n, dtype=float))
    W = np.vstack(rows)
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    W = np.round(W, 8)
    _, idx = np.unique(W, axis=0, return_index=True)
    return W[np.sort(idx)]


def learn_weights(
    metrics: pd.DataFrame,
    rgpm: pd.DataFrame,
    active_metric_keys: Sequence[str],
    n_samples: int,
    n_folds: int,
    cv_mode: str,
    seed: int,
    progress: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, float, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hybrid non-negative simplex weight learning with strict OOF scoring."""
    active_metric_keys = [k for k in active_metric_keys if k in METRIC_KEYS]
    if not active_metric_keys:
        raise ValueError("No active publication-day indicators are available for weight learning.")
    metric_z_cols = [k + "_z" for k in METRIC_KEYS]
    active_z_cols = [k + "_z" for k in active_metric_keys]
    meta_cols = [
        "paper_id", "title", "domain", "year", "primary_field", "display_community",
        "is_landmark", "reference_count", "cited_by_count",
    ]
    present_meta_cols = [c for c in meta_cols if c in metrics.columns]
    rgpm_cols = [
        "paper_id", "RGPM", "RGPM_v2", "RGPM_simple", "RGPM_v3_balanced",
        "RGPM_primary_only", "RGPM_mahalanobis_debug",
    ] + [c for c in rgpm.columns if c.startswith("RGPM_structural_residual") or c.startswith("RGPM_popularity_fitted")]
    present_rgpm_cols = [c for c in rgpm_cols if c in rgpm.columns]
    df = metrics[present_meta_cols + metric_z_cols].merge(rgpm[present_rgpm_cols], on="paper_id", how="inner")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["RGPM"]).reset_index(drop=True)
    df[active_z_cols] = df[active_z_cols].fillna(0.0)
    if len(df) < max(20, n_folds * 5):
        raise ValueError(f"Too few papers for weight learning after filtering: n={len(df)}")
    progress_log(
        f"Learning weights with hybrid simplex/ridge candidates from {len(df):,} papers; "
        f"active_metrics={','.join(active_metric_keys)}, cv_mode={cv_mode}, folds={n_folds}, seed={seed}.",
        progress,
    )
    X = df[active_z_cols].to_numpy(dtype=float)
    y = df["RGPM"].to_numpy(dtype=float)
    W_base = np.vstack(
        [
            base_weight_candidates(n_samples, seed=seed, n_metrics=len(active_metric_keys)),
            pair_structured_candidates(active_metric_keys, grid_n=9),
        ]
    )
    W_base = np.round(W_base, 8)
    _, base_idx = np.unique(W_base, axis=0, return_index=True)
    W_base = W_base[np.sort(base_idx)]
    splits = make_cv_splits(df, n_folds=n_folds, mode=cv_mode, seed=seed)
    progress_log(
        "  splits: " + ", ".join([f"{i+1}:train={len(tr):,}/test={len(te):,}" for i, (tr, te) in enumerate(splits)]),
        progress,
    )

    # Landscape/sample performance is evaluated for fixed, data-independent candidates.
    perf = weight_performance_cv_splits(X, y, W_base, splits, progress=progress)
    equal_w = np.ones(len(active_metric_keys), dtype=float) / len(active_metric_keys)
    equal_score = X @ equal_w
    equal_rho = safe_spearman(equal_score, y)
    W_full = expand_active_weights(W_base, active_metric_keys)
    sample_df = pd.DataFrame(W_full, columns=["w_" + k for k in METRIC_KEYS])
    sample_df["cv_spearman"] = perf
    sample_df = sample_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["cv_spearman"]).reset_index(drop=True)
    sample_df["cv_spearman_delta_vs_equal"] = sample_df["cv_spearman"] - equal_rho

    # Strict OOF: candidate selection happens within each training fold.
    oof_score = np.full(len(df), np.nan, dtype=float)
    fold_ids = np.full(len(df), -1, dtype=int)
    fold_rows: List[Dict[str, object]] = []
    for fold_no, (train_idx, test_idx) in enumerate(splits, start=1):
        W_ridge = positive_ridge_candidates(X[train_idx], y[train_idx])
        W_pool = np.vstack([W_base, W_ridge, pair_structured_candidates(active_metric_keys, grid_n=11)])
        W_pool = np.round(W_pool, 8)
        _, uidx = np.unique(W_pool, axis=0, return_index=True)
        W_pool = W_pool[np.sort(uidx)]
        train_perf = direct_weight_performance(X[train_idx], y[train_idx], W_pool)
        fold_best_idx = int(np.nanargmax(train_perf)) if np.isfinite(train_perf).any() else 0
        fold_w = W_pool[fold_best_idx]
        oof_score[test_idx] = X[test_idx] @ fold_w
        fold_ids[test_idx] = fold_no
        fold_test_rho = safe_spearman(oof_score[test_idx], y[test_idx])
        fold_row: Dict[str, object] = {
            "fold": fold_no,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_spearman": float(np.nanmax(train_perf)) if np.isfinite(train_perf).any() else np.nan,
            "test_spearman": fold_test_rho,
            "selected_candidate_source": "hybrid_pool",
        }
        fold_full = expand_active_weights(fold_w.reshape(1, -1), active_metric_keys)[0]
        for key, val in zip(METRIC_KEYS, fold_full):
            fold_row["w_" + key] = float(val)
        fold_rows.append(fold_row)
        progress_log(
            f"  outer fold {fold_no}/{len(splits)}: train={len(train_idx):,}, test={len(test_idx):,}, "
            f"test rho={fold_test_rho:.3f}",
            progress,
        )

    # Final all-data weight is for reporting, not for OOF performance.
    W_final_pool = np.vstack([W_base, positive_ridge_candidates(X, y), pair_structured_candidates(active_metric_keys, grid_n=11)])
    W_final_pool = np.round(W_final_pool, 8)
    _, fidx = np.unique(W_final_pool, axis=0, return_index=True)
    W_final_pool = W_final_pool[np.sort(fidx)]
    final_perf = direct_weight_performance(X, y, W_final_pool)
    best_idx = int(np.nanargmax(final_perf)) if np.isfinite(final_perf).any() else 0
    best_active = W_final_pool[best_idx]
    best_full = expand_active_weights(best_active.reshape(1, -1), active_metric_keys)[0]
    best_w = pd.Series(best_full, index=METRIC_KEYS, name="weight")
    score = X @ best_active
    best_perf = safe_spearman(oof_score, y)

    score_table = df[[c for c in present_meta_cols if c in df.columns]].copy()
    score_table["fold_id"] = fold_ids
    score_table["S_w"] = score
    score_table["S_w_oof"] = oof_score
    score_table["S_equal"] = equal_score
    score_table["RGPM"] = y
    score_table["RGPM_v2"] = df["RGPM_v2"].to_numpy(dtype=float) if "RGPM_v2" in df.columns else y
    score_table["RGPM_simple"] = df["RGPM_simple"].to_numpy(dtype=float) if "RGPM_simple" in df.columns else y
    if "RGPM_v3_balanced" in df.columns:
        score_table["RGPM_v3_balanced"] = df["RGPM_v3_balanced"].to_numpy(dtype=float)
    if "RGPM_primary_only" in df.columns:
        score_table["RGPM_primary_only"] = df["RGPM_primary_only"].to_numpy(dtype=float)
    for c in df.columns:
        if c.startswith("RGPM_structural_residual") or c.startswith("RGPM_popularity_fitted"):
            score_table[c] = df[c].to_numpy(dtype=float)
    if "RGPM_mahalanobis_debug" in df.columns:
        score_table["RGPM_mahalanobis_debug"] = df["RGPM_mahalanobis_debug"].to_numpy(dtype=float)
    for k in METRIC_KEYS:
        score_table[k + "_z"] = df[k + "_z"].to_numpy(dtype=float)
    cv_summary = pd.DataFrame(fold_rows)[["fold", "n_train", "n_test", "train_spearman", "test_spearman"]]
    fold_weights = pd.DataFrame(fold_rows)

    baseline_rows: List[Dict[str, object]] = []

    def add_baseline(name: str, scores: Sequence[float], kind: str, metric: str = "") -> None:
        rho = safe_spearman(scores, y)
        lo, hi = bootstrap_spearman_ci(scores, y, seed=seed + len(baseline_rows) + 101)
        baseline_rows.append(
            {
                "model": name,
                "kind": kind,
                "metric": metric,
                "oof_spearman": rho,
                "ci_low": lo,
                "ci_high": hi,
                "delta_vs_equal": rho - equal_rho if np.isfinite(rho) and np.isfinite(equal_rho) else np.nan,
            }
        )

    add_baseline("equal_weights", equal_score, "fixed")
    single_scores = []
    for i, key in enumerate(active_metric_keys):
        rho = safe_spearman(X[:, i], y)
        single_scores.append((rho, key, X[:, i]))
    single_scores = [item for item in single_scores if np.isfinite(item[0])]
    if single_scores:
        _, best_single_key, best_single_score = max(single_scores, key=lambda item: item[0])
        add_baseline("best_single_indicator", best_single_score, "single_indicator", best_single_key)
    if "reference_count" in df.columns:
        add_baseline("reference_count", pd.to_numeric(df["reference_count"], errors="coerce"), "bibliometric")
    if "cited_by_count" in df.columns and pd.to_numeric(df["cited_by_count"], errors="coerce").notna().any():
        add_baseline("cited_by_count", pd.to_numeric(df["cited_by_count"], errors="coerce"), "bibliometric")
    random_median_rho = float(np.nanmedian(perf))
    baseline_rows.append(
        {
            "model": "random_dirichlet_median",
            "kind": "sampled_weights",
            "metric": "",
            "oof_spearman": random_median_rho,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "delta_vs_equal": random_median_rho - equal_rho if np.isfinite(equal_rho) else np.nan,
        }
    )
    add_baseline("learned_weight_oof", oof_score, "learned_hybrid")
    baseline_comparison = pd.DataFrame(baseline_rows)
    model_diagnostics = baseline_comparison.copy()
    progress_log(
        f"Finished hybrid OOF learning: fixed candidates={len(sample_df):,}, "
        f"OOF Spearman={best_perf:.3f}; equal={equal_rho:.3f}.",
        progress,
    )
    return sample_df, best_w, best_perf, score_table, cv_summary, fold_weights, baseline_comparison, model_diagnostics


def scope_subsets(df: pd.DataFrame) -> List[Tuple[str, str, int, pd.DataFrame]]:
    """Return all/domain/fold/domain-fold subsets for diagnostics."""
    out: List[Tuple[str, str, int, pd.DataFrame]] = [("all", "all", -1, df)]
    if "domain" in df.columns:
        for domain, sub in df.groupby("domain", sort=True):
            out.append(("domain", str(domain), -1, sub))
    if "fold_id" in df.columns:
        for fold_id, sub in df.groupby("fold_id", sort=True):
            if int(fold_id) > 0:
                out.append(("fold", "all", int(fold_id), sub))
    if "domain" in df.columns and "fold_id" in df.columns:
        for (domain, fold_id), sub in df.groupby(["domain", "fold_id"], sort=True):
            if int(fold_id) > 0:
                out.append(("domain_fold", str(domain), int(fold_id), sub))
    return out


def compute_indicator_target_correlations(
    score_table: pd.DataFrame,
    rgpm_table: pd.DataFrame,
    active_metric_keys: Sequence[str],
) -> pd.DataFrame:
    """Correlate each publication-day indicator with RGPM and RGPM components."""
    st = score_table.copy()
    component_cols = [
        c for c in rgpm_table.columns
        if c.startswith("RGPM_") and c.endswith("_component")
    ]
    delta_z_cols = [c + "_z" for c in DELTA_KEYS if c + "_z" in rgpm_table.columns]
    merge_cols = ["paper_id"] + [
        c for c in ["RGPM_v3_balanced", "RGPM_mahalanobis_debug"] + component_cols + delta_z_cols
        if c in rgpm_table.columns and c not in st.columns
    ]
    if len(merge_cols) > 1:
        st = st.merge(rgpm_table[merge_cols], on="paper_id", how="left")

    metric_cols = [k + "_z" for k in METRIC_KEYS if k + "_z" in st.columns]
    target_cols = [
        c for c in ["RGPM", "RGPM_v3_balanced", "RGPM_mahalanobis_debug"] + component_cols + delta_z_cols
        if c in st.columns
    ]
    rows: List[Dict[str, object]] = []
    for scope, domain, fold_id, sub in scope_subsets(st):
        for metric_col in metric_cols:
            metric = metric_col[:-2]
            for target in target_cols:
                rho = safe_spearman(sub[metric_col], sub[target])
                n = int((pd.to_numeric(sub[metric_col], errors="coerce").notna() & pd.to_numeric(sub[target], errors="coerce").notna()).sum())
                rows.append(
                    {
                        "scope": scope,
                        "domain": domain,
                        "fold_id": fold_id,
                        "metric": metric,
                        "metric_active": int(metric in active_metric_keys),
                        "target": target,
                        "n": n,
                        "spearman": rho,
                    }
                )
    return pd.DataFrame(rows)


def compute_rgpm_component_correlations(rgpm_table: pd.DataFrame) -> pd.DataFrame:
    """Audit whether RGPM is dominated by one component or graph delta."""
    if rgpm_table.empty or "RGPM" not in rgpm_table.columns:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    variables: List[Tuple[str, str]] = []
    variables.extend(
        (c, "mechanism_component")
        for c in rgpm_table.columns
        if c.startswith("RGPM_") and c.endswith("_component")
    )
    variables.extend((c + "_z", "delta_z") for c in DELTA_KEYS if c + "_z" in rgpm_table.columns)
    variables.extend((c, "delta_raw") for c in DELTA_KEYS if c in rgpm_table.columns)
    for col, kind in variables:
        vals = pd.to_numeric(rgpm_table[col], errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        rows.append(
            {
                "variable": col,
                "kind": kind,
                "n": int(len(finite)),
                "spearman_with_rgpm": safe_spearman(rgpm_table[col], rgpm_table["RGPM"]),
                "nonzero_rate": float(np.mean(np.abs(finite) > 1e-12)) if len(finite) else np.nan,
                "median": float(np.median(finite)) if len(finite) else np.nan,
                "iqr": float(np.percentile(finite, 75) - np.percentile(finite, 25)) if len(finite) else np.nan,
                "cap_hit_rate": float(np.nanmean(pd.to_numeric(rgpm_table.get(col.replace("_z", "_z_clipped"), np.nan), errors="coerce"))) if kind == "delta_z" else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_control_tier_audit(control_diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Summarize strict/relaxed matched-control usage."""
    if control_diagnostics.empty:
        return pd.DataFrame()
    ctrl = control_diagnostics.copy()
    if "domain" not in ctrl.columns:
        ctrl["domain"] = "all"
    if "control_tier" not in ctrl.columns:
        ctrl["control_tier"] = ""
    ctrl["is_relaxed_tier"] = ctrl["control_tier"].astype(str).isin(RELAXED_CONTROL_TIERS).astype(int)
    rows: List[Dict[str, object]] = []
    groups: List[Tuple[str, str, pd.DataFrame]] = [("all", "all", ctrl)]
    groups.extend(("domain", str(domain), sub) for domain, sub in ctrl.groupby("domain", sort=True))
    for scope, domain, sub in groups:
        n_controls = pd.to_numeric(sub["n_controls"], errors="coerce").to_numpy(dtype=float) if "n_controls" in sub.columns else np.array([])
        n_controls = n_controls[np.isfinite(n_controls)]
        tier_counts = sub["control_tier"].astype(str).value_counts(normalize=False)
        tier_rates = sub["control_tier"].astype(str).value_counts(normalize=True)
        for tier in sorted(tier_counts.index):
            rows.append(
                {
                    "scope": scope,
                    "domain": domain,
                    "control_tier": tier,
                    "n_rows": int(len(sub)),
                    "tier_count": int(tier_counts[tier]),
                    "tier_rate": float(tier_rates[tier]),
                    "relaxed_tier_rate": float(sub["is_relaxed_tier"].mean()),
                    "n_controls_median": float(np.median(n_controls)) if len(n_controls) else np.nan,
                    "n_controls_min": float(np.min(n_controls)) if len(n_controls) else np.nan,
                    "n_controls_max": float(np.max(n_controls)) if len(n_controls) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def quadratic_design_matrix(X: np.ndarray) -> np.ndarray:
    """Linear + square + pairwise-interaction design matrix."""
    X = np.asarray(X, dtype=float)
    cols = [X]
    cols.append(np.square(X))
    pairs: List[np.ndarray] = []
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            pairs.append((X[:, i] * X[:, j]).reshape(-1, 1))
    if pairs:
        cols.append(np.hstack(pairs))
    return np.hstack(cols)


def ridge_predict_oof(
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[object],
    lambdas: Sequence[float],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """OOF quadratic ridge upper-bound diagnostic."""
    Xq = quadratic_design_matrix(X)
    y_rank = rank_normal_scores(y)
    y_rank = np.where(np.isfinite(y_rank), y_rank, 0.0)
    oof = np.full(len(y), np.nan, dtype=float)
    rows: List[Dict[str, object]] = []
    for fold_no, fold in enumerate(folds, start=1):
        if isinstance(fold, tuple):
            train_idx = np.asarray(fold[0], dtype=int)
            test_idx = np.asarray(fold[1], dtype=int)
        else:
            test_idx = np.asarray(fold, dtype=int)
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test_idx] = False
            train_idx = np.where(train_mask)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        x_train = Xq[train_idx]
        x_test = Xq[test_idx]
        mu = np.nanmean(x_train, axis=0)
        sigma = np.nanstd(x_train, axis=0)
        sigma = np.where(sigma < 1e-9, 1.0, sigma)
        x_train = (x_train - mu) / sigma
        x_test = (x_test - mu) / sigma
        y_train = y_rank[train_idx]
        y_train = y_train - float(np.mean(y_train))
        best_lambda = float(lambdas[0])
        best_train_rho = -np.inf
        best_beta = np.zeros(x_train.shape[1], dtype=float)
        xtx = x_train.T @ x_train
        xty = x_train.T @ y_train
        for lam in lambdas:
            try:
                beta = np.linalg.solve(xtx + float(lam) * np.eye(x_train.shape[1]), xty)
            except Exception:
                beta = np.linalg.pinv(xtx + float(lam) * np.eye(x_train.shape[1])) @ xty
            train_pred = x_train @ beta
            train_rho = safe_spearman(train_pred, y[train_idx])
            if np.isfinite(train_rho) and train_rho > best_train_rho:
                best_train_rho = float(train_rho)
                best_lambda = float(lam)
                best_beta = beta
        oof[test_idx] = x_test @ best_beta
        rows.append(
            {
                "fold": fold_no,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "selected_lambda": best_lambda,
                "train_spearman": best_train_rho if np.isfinite(best_train_rho) else np.nan,
                "test_spearman": safe_spearman(oof[test_idx], y[test_idx]),
            }
        )
    return oof, pd.DataFrame(rows)


def compute_nonlinear_upper_bound_diagnostics(
    score_table: pd.DataFrame,
    active_metric_keys: Sequence[str],
    seed: int,
    cv_mode: str = "random",
    n_folds: int = 3,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Diagnostic-only nonlinear upper bound using quadratic ridge OOF predictions."""
    active = [k for k in active_metric_keys if k + "_z" in score_table.columns]
    if not active or "RGPM" not in score_table.columns:
        return pd.DataFrame(), np.array([])
    st = score_table.replace([np.inf, -np.inf], np.nan).dropna(subset=[k + "_z" for k in active] + ["RGPM"]).reset_index(drop=True)
    if len(st) < 20:
        return pd.DataFrame(), np.array([])
    X = st[[k + "_z" for k in active]].fillna(0.0).to_numpy(dtype=float)
    y = st["RGPM"].to_numpy(dtype=float)
    try:
        folds = make_cv_splits(st, n_folds=n_folds, mode=cv_mode, seed=seed)
    except ValueError:
        fold_ids = st["fold_id"].to_numpy(dtype=int) if "fold_id" in st.columns else np.full(len(st), -1, dtype=int)
        folds = [np.where(fold_ids == f)[0] for f in sorted(set(fold_ids)) if f > 0 and np.sum(fold_ids == f) > 0]
    if len(folds) < 2:
        folds = make_folds(st, n_folds=3, mode="random", seed=seed)
    lambdas = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    oof, fold_diag = ridge_predict_oof(X, y, folds, lambdas)
    linear_rho = safe_spearman(st["S_w_oof"], y) if "S_w_oof" in st.columns else np.nan
    equal_rho = safe_spearman(st["S_equal"], y) if "S_equal" in st.columns else np.nan
    nonlinear_rho = safe_spearman(oof, y)
    summary = pd.DataFrame(
        [
            {
                "fold": 0,
                "n_train": int(len(st)),
                "n_test": int(len(st)),
                "selected_lambda": np.nan,
                "train_spearman": np.nan,
                "test_spearman": nonlinear_rho,
                "model": "quadratic_ridge_oof",
                "linear_oof_spearman": linear_rho,
                "equal_weight_spearman": equal_rho,
                "delta_vs_linear": nonlinear_rho - linear_rho if np.isfinite(nonlinear_rho) and np.isfinite(linear_rho) else np.nan,
                "delta_vs_equal": nonlinear_rho - equal_rho if np.isfinite(nonlinear_rho) and np.isfinite(equal_rho) else np.nan,
            }
        ]
    )
    fold_diag["model"] = "quadratic_ridge_oof_fold"
    fold_diag["linear_oof_spearman"] = linear_rho
    fold_diag["equal_weight_spearman"] = equal_rho
    fold_diag["delta_vs_linear"] = np.nan
    fold_diag["delta_vs_equal"] = np.nan
    return pd.concat([summary, fold_diag], ignore_index=True), oof


def compute_target_sensitivity(
    metrics_z: pd.DataFrame,
    rgpm_table: pd.DataFrame,
    active_metric_keys: Sequence[str],
    cv_modes: Sequence[str],
    n_samples: int,
    n_folds: int,
    seed: int,
    tau: int,
    progress: bool = False,
) -> pd.DataFrame:
    """Evaluate target-version and CV-mode sensitivity on the same feature table."""
    target_cols = [
        ("structural_residual", "RGPM"),
        ("raw_balanced", "RGPM_v3_balanced"),
        ("primary_only", "RGPM_primary_only"),
        ("mahalanobis_debug", "RGPM_mahalanobis_debug"),
    ]
    current_struct_col = f"RGPM_structural_residual_tau{tau}"
    structural_cols = [c for c in rgpm_table.columns if c.startswith("RGPM_structural_residual_tau")]
    for col in structural_cols:
        if col != current_struct_col:
            target_cols.append((col, col))
    seen_targets: set[Tuple[str, str]] = set()
    rows: List[Dict[str, object]] = []
    for target_name, target_col in target_cols:
        if (target_name, target_col) in seen_targets or target_col not in rgpm_table.columns:
            continue
        seen_targets.add((target_name, target_col))
        target_rgpm = rgpm_table[["paper_id", target_col]].rename(columns={target_col: "RGPM"}).copy()
        for cv_mode in cv_modes:
            try:
                _, _, learned_rho, score_table, _, _, baseline, _ = learn_weights(
                    metrics_z,
                    target_rgpm,
                    active_metric_keys=active_metric_keys,
                    n_samples=n_samples,
                    n_folds=n_folds,
                    cv_mode=cv_mode,
                    seed=seed + stable_int_id(f"{target_name}-{cv_mode}", modulo=10000),
                    progress=progress,
                )
                equal = baseline.loc[baseline["model"] == "equal_weights", "oof_spearman"]
                best_single = baseline.loc[baseline["model"] == "best_single_indicator", "oof_spearman"]
                citation = baseline.loc[baseline["model"] == "cited_by_count", "oof_spearman"]
                if citation.empty:
                    citation = baseline.loc[baseline["model"] == "reference_count", "oof_spearman"]
                nonlinear, _ = compute_nonlinear_upper_bound_diagnostics(
                    score_table,
                    active_metric_keys,
                    seed=seed + 919,
                    cv_mode=cv_mode,
                    n_folds=n_folds,
                )
                nonlinear_rows = nonlinear[nonlinear["model"] == "quadratic_ridge_oof"] if not nonlinear.empty else pd.DataFrame()
                nonlinear_rho = float(nonlinear_rows["test_spearman"].iloc[0]) if not nonlinear_rows.empty else np.nan
                equal_rho = float(equal.iloc[0]) if not equal.empty else np.nan
                best_single_rho = float(best_single.iloc[0]) if not best_single.empty else np.nan
                citation_rho = float(citation.iloc[0]) if not citation.empty else np.nan
                rows.append(
                    {
                        "tau": int(tau),
                        "target_name": target_name,
                        "target_column": target_col,
                        "cv_mode": cv_mode,
                        "n": int(len(score_table)),
                        "learned_oof_spearman": learned_rho,
                        "equal_weight_oof_spearman": equal_rho,
                        "learned_vs_equal_delta": learned_rho - equal_rho if np.isfinite(learned_rho) and np.isfinite(equal_rho) else np.nan,
                        "best_single_oof_spearman": best_single_rho,
                        "learned_vs_best_single_delta": learned_rho - best_single_rho if np.isfinite(learned_rho) and np.isfinite(best_single_rho) else np.nan,
                        "citation_baseline_spearman": citation_rho,
                        "nonlinear_upper_bound_oof_spearman": nonlinear_rho,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "tau": int(tau),
                        "target_name": target_name,
                        "target_column": target_col,
                        "cv_mode": cv_mode,
                        "n": 0,
                        "learned_oof_spearman": np.nan,
                        "equal_weight_oof_spearman": np.nan,
                        "learned_vs_equal_delta": np.nan,
                        "best_single_oof_spearman": np.nan,
                        "learned_vs_best_single_delta": np.nan,
                        "citation_baseline_spearman": np.nan,
                        "nonlinear_upper_bound_oof_spearman": np.nan,
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(rows)


def compute_landmark_validation(score_table: pd.DataFrame, min_controls: int = 20) -> pd.DataFrame:
    """Compare landmark scores against matched non-landmark controls."""
    if score_table.empty or "is_landmark" not in score_table.columns:
        return pd.DataFrame()
    df = add_reference_bins(score_table.copy())
    if "primary_field" not in df.columns:
        df["primary_field"] = ""
    rows: List[Dict[str, object]] = []
    landmarks = df[pd.to_numeric(df["is_landmark"], errors="coerce").fillna(0).astype(int) == 1]
    non_landmarks = df[pd.to_numeric(df["is_landmark"], errors="coerce").fillna(0).astype(int) == 0]
    for row in landmarks.itertuples(index=False):
        year = int(getattr(row, "year"))
        domain = str(getattr(row, "domain", ""))
        field = str(getattr(row, "primary_field", ""))
        ref_bin = str(getattr(row, "ref_bin", ""))
        pool = non_landmarks[
            (non_landmarks["domain"].astype(str) == domain)
            & (non_landmarks["primary_field"].astype(str) == field)
            & (non_landmarks["year"].between(year - 3, year + 3))
            & (non_landmarks["ref_bin"].astype(str) == ref_bin)
        ]
        tier = "domain_field_year3_refbin"
        if len(pool) < min_controls:
            pool = non_landmarks[
                (non_landmarks["domain"].astype(str) == domain)
                & (non_landmarks["primary_field"].astype(str) == field)
                & (non_landmarks["year"].between(year - 5, year + 5))
            ]
            tier = "domain_field_year5"
        if len(pool) < min_controls:
            pool = non_landmarks[(non_landmarks["domain"].astype(str) == domain) & (non_landmarks["year"].between(year - 5, year + 5))]
            tier = "domain_year5"
        if len(pool) < min_controls:
            pool = non_landmarks[non_landmarks["domain"].astype(str) == domain]
            tier = "domain_all_years"
        if pool.empty:
            continue
        out: Dict[str, object] = {
            "paper_id": getattr(row, "paper_id"),
            "title": getattr(row, "title", ""),
            "domain": domain,
            "year": year,
            "primary_field": field,
            "n_controls": int(len(pool)),
            "control_tier": tier,
        }
        for col in ["S_w_oof", "S_equal", "RGPM", "RGPM_v3_balanced"]:
            if col in df.columns:
                val = float(getattr(row, col))
                ctrl = pd.to_numeric(pool[col], errors="coerce").dropna().to_numpy(dtype=float)
                out[col + "_value"] = val
                out[col + "_matched_percentile"] = float(np.mean(ctrl <= val) * 100.0) if len(ctrl) else np.nan
        rows.append(out)
    result = pd.DataFrame(rows)
    if not result.empty:
        summary = {
            "paper_id": "__summary__",
            "title": "Median landmark matched percentile",
            "domain": "all",
            "year": -1,
            "primary_field": "",
            "n_controls": int(result["n_controls"].median()),
            "control_tier": "summary",
        }
        for col in [c for c in result.columns if c.endswith("_matched_percentile")]:
            summary[col] = float(result[col].median())
        result = pd.concat([result, pd.DataFrame([summary])], ignore_index=True, sort=False)
    return result


def build_panel_b_example(comp: ComputedData) -> pd.DataFrame:
    """Use reliable contributing deltas sorted by mechanism/reliability for Panel b."""
    merged = comp.rgpm_table.copy()
    if "paper_id" in comp.score_table.columns and "S_w_oof" in comp.score_table.columns:
        score_cols = ["paper_id", "S_w_oof", "RGPM"]
        merged = merged.merge(comp.score_table[score_cols].rename(columns={"RGPM": "RGPM_oof_target"}), on="paper_id", how="left")
    diag = comp.delta_diagnostics.copy()
    if "contributing" in diag.columns and (diag["contributing"].astype(int) == 1).any():
        diag = diag[diag["contributing"].astype(int) == 1].copy()
    else:
        active_set = set(comp.active_delta_keys)
        diag = diag[diag["delta"].isin(active_set)].copy()
    if diag.empty:
        return pd.DataFrame()
    score_rank = (
        pd.to_numeric(merged["S_w_oof"], errors="coerce").rank(pct=True).fillna(0.0).to_numpy(dtype=float)
        if "S_w_oof" in merged.columns
        else np.zeros(len(merged), dtype=float)
    )
    candidate_scores = []
    for pos, (_, row) in enumerate(merged.iterrows()):
        total = 0.0
        clipped = 0
        for drow in diag.itertuples(index=False):
            key = str(getattr(drow, "delta"))
            if key + "_z" not in merged.columns:
                continue
            z = float(row.get(key + "_z", 0.0))
            rel = float(getattr(drow, "reliability_weight", 0.0))
            total += max(0.0, z) * rel
            clipped += int(row.get(key + "_z_clipped", 0))
        if pos < len(score_rank):
            total += 0.10 * float(score_rank[pos])
        if int(row.get("is_landmark", 0)) == 1:
            total += 0.20
        total -= 0.10 * clipped
        candidate_scores.append(total)
    if candidate_scores:
        row = merged.iloc[int(np.nanargmax(np.asarray(candidate_scores, dtype=float)))]
    elif (merged["is_landmark"].astype(int) == 1).any():
        row = merged[merged["is_landmark"].astype(int) == 1].sort_values("RGPM", ascending=False).iloc[0]
    else:
        row = merged.sort_values("RGPM", ascending=False).iloc[0]
    mech_order = {k: i for i, k in enumerate(["Breadth", "Brokerage", "Boundary", "Consolidation"])}
    delta_to_mech = {d: mech for mech, keys in DELTA_MECHANISM_GROUPS_V3.items() for d in keys}
    diag["mechanism"] = diag["delta"].map(delta_to_mech).fillna("Other")
    diag["mechanism_order"] = diag["mechanism"].map(mech_order).fillna(99)
    diag = diag.sort_values(["mechanism_order", "reliability_weight"], ascending=[True, False])
    out = []
    for drow in diag.itertuples(index=False):
        key = str(getattr(drow, "delta"))
        out.append(
            {
                "key": key,
                "label": f"{DELTA_LABELS.get(key, key)} ({next((d[2] for d in DELTA_SPECS if d[0] == key), '')})",
                "z": float(row[key + "_z"]),
                "z_raw": float(row.get(key + "_z_raw", row[key + "_z"])),
                "clipped": int(row.get(key + "_z_clipped", 0)),
                "reliability_weight": float(getattr(drow, "reliability_weight", np.nan)),
                "mechanism": str(getattr(drow, "mechanism", "")),
            }
        )
    return pd.DataFrame(out)


def finite_min(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.min()) if len(arr) else float("nan")


def finite_max(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.max()) if len(arr) else float("nan")


def compute_domain_adequacy_diagnostics(
    score_table: pd.DataFrame,
    control_diagnostics: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Summarize whether the empirical run is large and diverse enough for a main-figure claim."""
    thresholds = DATA_ADEQUACY_THRESHOLDS
    if score_table.empty:
        profile = {
            "n_domains": 0,
            "total_papers": 0,
            "min_papers_per_domain": 0,
            "min_landmark_or_high_cases_per_domain": 0,
            "control_median_min_by_domain": float("nan"),
            "relaxed_control_tier_rate_max_by_domain": float("nan"),
            "domains": [],
        }
        return pd.DataFrame(), profile

    st = score_table.copy()
    if "domain" not in st.columns:
        st["domain"] = "domain"
    st["domain"] = st["domain"].fillna("unknown").astype(str)
    if "RGPM" not in st.columns:
        st["RGPM"] = np.nan
    if "is_landmark" not in st.columns:
        st["is_landmark"] = 0

    ctrl = control_diagnostics.copy() if control_diagnostics is not None else pd.DataFrame()
    if not ctrl.empty:
        if "domain" not in ctrl.columns:
            ctrl["domain"] = "domain"
        ctrl["domain"] = ctrl["domain"].fillna("unknown").astype(str)
        if "n_controls" not in ctrl.columns:
            ctrl["n_controls"] = np.nan
        if "control_tier" not in ctrl.columns:
            ctrl["control_tier"] = ""

    rows: List[Dict[str, object]] = []
    high_q = float(thresholds["high_perturbation_quantile"])
    for domain, sub in st.groupby("domain", sort=True):
        rgpm = pd.to_numeric(sub["RGPM"], errors="coerce")
        finite_rgpm = rgpm[np.isfinite(rgpm.to_numpy(dtype=float))]
        if len(finite_rgpm):
            high_threshold = float(np.nanquantile(finite_rgpm.to_numpy(dtype=float), high_q))
            has_real_tail = float(np.nanstd(finite_rgpm.to_numpy(dtype=float))) > 1e-12 and high_threshold > 0.0
            high_mask = (rgpm >= high_threshold) if has_real_tail else pd.Series(False, index=sub.index)
        else:
            high_threshold = float("nan")
            high_mask = pd.Series(False, index=sub.index)
        landmark_mask = pd.to_numeric(sub["is_landmark"], errors="coerce").fillna(0).astype(int) == 1
        high_count = int(high_mask.fillna(False).sum())
        landmark_count = int(landmark_mask.sum())
        landmark_or_high_count = int((landmark_mask | high_mask.fillna(False)).sum())

        ctrl_sub = ctrl[ctrl["domain"] == domain] if not ctrl.empty else pd.DataFrame()
        if not ctrl_sub.empty:
            n_controls = pd.to_numeric(ctrl_sub["n_controls"], errors="coerce").to_numpy(dtype=float)
            n_controls = n_controls[np.isfinite(n_controls)]
            control_median = float(np.median(n_controls)) if len(n_controls) else float("nan")
            relaxed_rate = float(ctrl_sub["control_tier"].astype(str).isin(RELAXED_CONTROL_TIERS).mean())
        else:
            control_median = float("nan")
            relaxed_rate = float("nan")

        rows.append(
            {
                "domain": domain,
                "n_papers": int(len(sub)),
                "n_landmarks": landmark_count,
                "n_high_perturbation_cases": high_count,
                "n_landmark_or_high_cases": landmark_or_high_count,
                "high_perturbation_quantile": high_q,
                "high_perturbation_threshold": high_threshold,
                "control_median": control_median,
                "relaxed_control_tier_rate": relaxed_rate,
                "pass_papers_per_domain_min": int(len(sub) >= int(thresholds["papers_per_domain_min"])),
                "pass_landmark_or_high_cases_min": int(landmark_or_high_count >= int(thresholds["landmark_or_high_cases_per_domain_min"])),
                "pass_control_median_min": int(np.isfinite(control_median) and control_median >= float(thresholds["control_median_min"])),
                "pass_relaxed_control_tier_rate_max": int(np.isfinite(relaxed_rate) and relaxed_rate <= float(thresholds["relaxed_control_tier_rate_max"])),
            }
        )

    domain_diag = pd.DataFrame(rows)
    min_papers = int(domain_diag["n_papers"].min()) if not domain_diag.empty else 0
    min_cases = int(domain_diag["n_landmark_or_high_cases"].min()) if not domain_diag.empty else 0
    control_median_min = finite_min(domain_diag["control_median"].to_numpy(dtype=float)) if not domain_diag.empty else float("nan")
    relaxed_rate_max = finite_max(domain_diag["relaxed_control_tier_rate"].to_numpy(dtype=float)) if not domain_diag.empty else float("nan")
    profile = {
        "n_domains": int(domain_diag["domain"].nunique()) if not domain_diag.empty else 0,
        "total_papers": int(len(st)),
        "min_papers_per_domain": min_papers,
        "min_landmark_or_high_cases_per_domain": min_cases,
        "control_median_min_by_domain": control_median_min,
        "relaxed_control_tier_rate_max_by_domain": relaxed_rate_max,
        "domains": domain_diag["domain"].astype(str).tolist() if not domain_diag.empty else [],
        "domains_below_paper_min": domain_diag.loc[
            domain_diag["pass_papers_per_domain_min"].astype(int) == 0, "domain"
        ].astype(str).tolist() if not domain_diag.empty else [],
        "domains_below_landmark_or_high_case_min": domain_diag.loc[
            domain_diag["pass_landmark_or_high_cases_min"].astype(int) == 0, "domain"
        ].astype(str).tolist() if not domain_diag.empty else [],
        "domains_with_relaxed_controls": domain_diag.loc[
            domain_diag["pass_relaxed_control_tier_rate_max"].astype(int) == 0, "domain"
        ].astype(str).tolist() if not domain_diag.empty else [],
    }
    return domain_diag, profile


def columnwise_spearman(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if scores.shape[0] != len(y):
        raise ValueError("Score matrix and target length mismatch.")
    finite_scores = np.isfinite(scores).all(axis=1)
    mask = mask & finite_scores
    out = np.full(scores.shape[1], np.nan, dtype=float)
    if mask.sum() < 4 or np.nanstd(y[mask]) < 1e-12:
        return out
    s = scores[mask, :]
    yt = y[mask]
    if SCIPY_OK:
        yr = rankdata(yt).astype(float)
        sr = np.apply_along_axis(rankdata, 0, s).astype(float)
    else:
        yr = pd.Series(yt).rank(method="average").to_numpy(dtype=float)
        sr = np.vstack([pd.Series(s[:, i]).rank(method="average").to_numpy(dtype=float) for i in range(s.shape[1])]).T
    yr = yr - yr.mean()
    sr = sr - sr.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(sr * sr, axis=0) * np.sum(yr * yr))
    valid = denom > 1e-12
    out[valid] = (yr @ sr[:, valid]) / denom[valid]
    return out


def profile_remainder_weights(comp: ComputedData, active: Sequence[str], excluded: Sequence[str]) -> np.ndarray:
    remaining = [key for key in active if key not in set(excluded)]
    if not remaining:
        return np.asarray([], dtype=float)
    vals = np.asarray([float(comp.best_weights.get(key, 0.0)) for key in remaining], dtype=float)
    vals = np.maximum(np.where(np.isfinite(vals), vals, 0.0), 0.0)
    if vals.sum() <= 1e-12:
        vals = np.ones(len(remaining), dtype=float)
    return vals / vals.sum()


def pair_profile_weight_grid(
    active: Sequence[str],
    xkey: str,
    ykey: str,
    remainder_weights: np.ndarray,
    bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = list(active)
    centers = np.linspace(0.0, 1.0, max(5, int(bins)))
    if xkey not in active or ykey not in active or xkey == ykey:
        return centers, centers, np.empty((0, len(active)), dtype=float)
    ix = active.index(xkey)
    iy = active.index(ykey)
    rem_keys = [key for key in active if key not in {xkey, ykey}]
    rem_idx = [active.index(key) for key in rem_keys]
    if rem_idx and len(remainder_weights) != len(rem_idx):
        remainder_weights = np.ones(len(rem_idx), dtype=float) / len(rem_idx)
    rows: List[np.ndarray] = []
    for pair_mass in centers:
        for pair_share_x in centers:
            w = np.zeros(len(active), dtype=float)
            w[ix] = float(pair_mass * pair_share_x)
            w[iy] = float(pair_mass * (1.0 - pair_share_x))
            leftover = max(0.0, 1.0 - float(pair_mass))
            if rem_idx and leftover > 1e-12:
                w[rem_idx] = leftover * remainder_weights
            rows.append(w)
    W = np.vstack(rows)
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    return centers, centers, W


def pairwise_cv_delta_grid(
    comp: ComputedData,
    xkey: str,
    ykey: str,
    bins: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    active = [key for key in comp.active_metric_keys if key in METRIC_KEYS]
    n_bins = int(bins or comp.profile_grid_size)
    centers = np.linspace(0.0, 1.0, max(5, n_bins))
    empty_grid = np.full((len(centers), len(centers)), np.nan, dtype=float)
    if xkey not in active or ykey not in active or xkey == ykey:
        return centers, centers, empty_grid, {"pair_valid": False}
    st = comp.score_table.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[key + "_z" for key in active] + ["RGPM"]
    ).reset_index(drop=True)
    if st.empty:
        return centers, centers, empty_grid, {"pair_valid": False}
    X = st[[key + "_z" for key in active]].to_numpy(dtype=float)
    target = st["RGPM"].to_numpy(dtype=float)
    fold_ids = st["fold_id"].to_numpy(dtype=int) if "fold_id" in st.columns else np.zeros(len(st), dtype=int)
    folds = [np.where(fold_ids == fold)[0] for fold in sorted(set(fold_ids)) if fold > 0 and np.sum(fold_ids == fold) >= 4]
    if len(folds) < 2:
        folds = make_folds(st, n_folds=3, mode="random", seed=stable_int_id(("pair", xkey, ykey)))
    rem = profile_remainder_weights(comp, active, excluded=[xkey, ykey])
    xs, ys, W = pair_profile_weight_grid(active, xkey, ykey, rem, bins=len(centers))
    if W.size == 0:
        return xs, ys, empty_grid, {"pair_valid": False}
    equal_w = np.ones(len(active), dtype=float) / len(active)
    split_delta = np.full((len(folds), W.shape[0]), np.nan, dtype=float)
    for fold_no, test_idx in enumerate(folds):
        if len(test_idx) < 4:
            continue
        Xt = X[test_idx]
        yt = target[test_idx]
        eq = float(columnwise_spearman(Xt @ equal_w, yt)[0])
        rho = columnwise_spearman(Xt @ W.T, yt)
        if np.isfinite(eq):
            split_delta[fold_no, :] = rho - eq
    values = np.nanmean(split_delta, axis=0)
    grid = values.reshape((len(ys), len(xs)))
    finite = values[np.isfinite(values)]
    fold_p95 = []
    for row in split_delta:
        vals = row[np.isfinite(row)]
        if len(vals):
            fold_p95.append(float(np.nanpercentile(vals, 95)))
    if len(finite):
        best_flat = int(np.nanargmax(values))
        best_row = best_flat // len(xs)
        best_col = best_flat % len(xs)
        p95_delta = float(np.nanpercentile(finite, 95))
        max_delta = float(np.nanmax(finite))
        mean_positive = float(np.nanmean(np.maximum(finite, 0.0)))
        positive_cell_rate = float(np.mean(finite > 0.0))
    else:
        best_row = best_col = 0
        p95_delta = max_delta = mean_positive = positive_cell_rate = float("nan")
    summary = {
        "pair_valid": True,
        "metric_x": xkey,
        "metric_y": ykey,
        "pair_label": f"{METRIC_LABELS.get(xkey, xkey)} × {METRIC_LABELS.get(ykey, ykey)}",
        "n_grid_cells": int(W.shape[0]),
        "p95_delta_vs_equal": p95_delta,
        "max_delta_vs_equal": max_delta,
        "mean_positive_delta_vs_equal": mean_positive,
        "positive_cell_rate": positive_cell_rate,
        "fold_stability": float(np.mean(np.asarray(fold_p95) >= 0.01)) if fold_p95 else float("nan"),
        "fold_p95_min": float(np.nanmin(fold_p95)) if fold_p95 else float("nan"),
        "best_pair_mass": float(ys[best_row]) if len(ys) else float("nan"),
        "best_share_to_first": float(xs[best_col]) if len(xs) else float("nan"),
    }
    return xs, ys, grid, summary


def pairwise_profile_grid(
    comp: ComputedData,
    xkey: str,
    ykey: str,
    bins: int = 25,
    profile_n: int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, grid, _ = pairwise_cv_delta_grid(comp, xkey, ykey, bins=bins)
    return xs, ys, grid


def compute_pair_scan_results(comp: ComputedData, progress: bool = True) -> pd.DataFrame:
    active = [key for key in METRIC_KEYS if key in comp.active_metric_keys]
    rows: List[Dict[str, object]] = []
    grids: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    total = int(len(active) * (len(active) - 1) / 2)
    for idx, (xkey, ykey) in enumerate(combinations(active, 2), start=1):
        progress_log(f"      pair scan {idx}/{total}: {xkey} × {ykey}", progress)
        xs, ys, grid, summary = pairwise_cv_delta_grid(comp, xkey, ykey, bins=comp.profile_grid_size)
        key = tuple(sorted((xkey, ykey)))
        grids[key] = (xs, ys, grid)
        row = dict(summary)
        row["pair_key"] = "|".join(key)
        row["metric_x"] = xkey
        row["metric_y"] = ykey
        row["selected_for_panel_d"] = 0
        row["coverage_pair"] = 0
        row["weak_pair"] = int(
            not (
                np.isfinite(float(row.get("p95_delta_vs_equal", np.nan)))
                and float(row.get("p95_delta_vs_equal", np.nan)) >= PAIR_SCAN_P95_MIN
            )
        )
        rows.append(row)
    setattr(comp, "_pair_grids", grids)
    scan = pd.DataFrame(rows)
    if scan.empty:
        return scan
    scan["selection_score"] = (
        pd.to_numeric(scan["p95_delta_vs_equal"], errors="coerce").fillna(-1.0)
        + 0.25 * pd.to_numeric(scan["max_delta_vs_equal"], errors="coerce").fillna(-1.0)
        + 0.02 * pd.to_numeric(scan["fold_stability"], errors="coerce").fillna(0.0)
        - 0.03 * (1.0 - pd.to_numeric(scan["positive_cell_rate"], errors="coerce").fillna(0.0))
    )
    return scan.sort_values("selection_score", ascending=False).reset_index(drop=True)


def pair_channel_coherence(pair: Tuple[str, str]) -> int:
    left, right = pair
    left_channels = set(FIG2_EVIDENCE_MAP.get(left, {}).keys())
    right_channels = set(FIG2_EVIDENCE_MAP.get(right, {}).keys())
    return int(bool(left_channels & right_channels))


def select_panel_d_pairs(pair_scan: pd.DataFrame, active_metric_keys: Sequence[str]) -> Tuple[List[Tuple[str, str]], pd.DataFrame]:
    scan = pair_scan.copy()
    active = [key for key in METRIC_KEYS if key in active_metric_keys]
    target_metrics = set(active)
    if scan.empty or len(active) < 2:
        return [], scan
    row_by_key = {str(row.pair_key): row for row in scan.itertuples(index=False)}
    pair_keys = list(row_by_key.keys())
    preferred_pairs = {
        tuple(sorted(("B", "RTD"))),
        tuple(sorted(("DeltaQ0", "Uzzi"))),
        tuple(sorted(("RS", "PDE"))),
        tuple(sorted(("BurtIP", "B"))),
        tuple(sorted(("BurtIP", "RTD"))),
    }
    candidates = []
    for combo in combinations(pair_keys, 4):
        metrics: List[str] = []
        for pair_key in combo:
            metrics.extend(pair_key.split("|"))
        counts = Counter(metrics)
        if set(counts.keys()) != target_metrics:
            continue
        if sorted(counts.values()) != [1] * (len(target_metrics) - 1) + [2]:
            continue
        rows = [row_by_key[key] for key in combo]
        p95_sum = float(np.nansum([float(getattr(row, "p95_delta_vs_equal", np.nan)) for row in rows]))
        instability = float(np.nansum([max(0.0, 0.75 - float(getattr(row, "fold_stability", np.nan) if np.isfinite(getattr(row, "fold_stability", np.nan)) else 0.0)) for row in rows]))
        stability_bonus = float(np.nansum([float(getattr(row, "fold_stability", 0.0)) if np.isfinite(getattr(row, "fold_stability", np.nan)) else 0.0 for row in rows]))
        max_sum = float(np.nansum([float(getattr(row, "max_delta_vs_equal", np.nan)) for row in rows]))
        interp = 0
        preferred = 0
        for key in combo:
            pair = tuple(key.split("|"))
            interp += pair_channel_coherence(pair)
            preferred += int(tuple(sorted(pair)) in preferred_pairs)
        score = p95_sum - 0.02 * instability + 0.01 * stability_bonus + 0.05 * max_sum
        candidates.append((score, p95_sum, preferred, interp, combo))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score = candidates[0][0]
        near = [item for item in candidates if best_score - item[0] <= 0.01]
        near.sort(key=lambda item: (item[2], item[3], item[1]), reverse=True)
        selected_keys = list(near[0][4])
    else:
        selected_keys = []
        covered: set[str] = set()
        for row in scan.itertuples(index=False):
            pair = str(row.pair_key).split("|")
            if len(selected_keys) >= 4:
                break
            if len(covered | set(pair)) > len(covered) or len(selected_keys) < 3:
                selected_keys.append(str(row.pair_key))
                covered |= set(pair)
        for row in scan.itertuples(index=False):
            if len(selected_keys) >= 4:
                break
            if str(row.pair_key) not in selected_keys:
                selected_keys.append(str(row.pair_key))
    order_hint = {
        "B|RTD": 0,
        "DeltaQ0|Uzzi": 1,
        "PDE|RS": 2,
        "B|BurtIP": 3,
        "BurtIP|RTD": 4,
    }
    selected_keys = sorted(selected_keys[:4], key=lambda key: (order_hint.get(key, 99), -float(row_by_key[key].selection_score)))
    selected_pairs = [tuple(key.split("|")) for key in selected_keys]
    scan["selected_for_panel_d"] = scan["pair_key"].isin(selected_keys).astype(int)
    scan["coverage_pair"] = 0
    if selected_pairs:
        selected_counts = Counter([metric for pair in selected_pairs for metric in pair])
        repeated = {metric for metric, count in selected_counts.items() if count > 1}
        for pair in selected_pairs:
            key = "|".join(pair)
            row = scan["pair_key"] == key
            p95 = pd.to_numeric(scan.loc[row, "p95_delta_vs_equal"], errors="coerce")
            is_weak = bool(p95.empty or (np.isfinite(float(p95.iloc[0])) and float(p95.iloc[0]) < PAIR_SCAN_P95_MIN))
            scan.loc[row, "coverage_pair"] = int(bool(repeated & set(pair)) or is_weak)
    return selected_pairs, scan


def compute_effect_summary(comp: ComputedData) -> pd.DataFrame:
    st = comp.score_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["S_w_oof", "RGPM"]).copy()
    rows: List[Dict[str, object]] = []
    if st.empty:
        return pd.DataFrame(rows)
    st["score_percentile"] = st["S_w_oof"].rank(method="average", pct=True) * 100.0
    st["rgpm_percentile"] = st["RGPM"].rank(method="average", pct=True) * 100.0
    try:
        st["score_tertile"] = pd.qcut(st["S_w_oof"].rank(method="first"), q=3, labels=["low", "mid", "high"])
    except ValueError:
        pct = st["S_w_oof"].rank(method="average", pct=True)
        st["score_tertile"] = "mid"
        st.loc[pct <= 1.0 / 3.0, "score_tertile"] = "low"
        st.loc[pct >= 2.0 / 3.0, "score_tertile"] = "high"
    for tertile, sub in st.groupby("score_tertile", sort=False):
        vals = sub["rgpm_percentile"].to_numpy(dtype=float)
        rows.append(
            {
                "stat": f"{tertile}_score_tertile_rgpm_percentile_median",
                "value": float(np.nanmedian(vals)) if len(vals) else np.nan,
                "n": int(len(sub)),
                "group": str(tertile),
            }
        )
        rows.append(
            {
                "stat": f"{tertile}_score_tertile_rgpm_percentile_mean",
                "value": float(np.nanmean(vals)) if len(vals) else np.nan,
                "n": int(len(sub)),
                "group": str(tertile),
            }
        )
    med = {row["stat"]: float(row["value"]) for row in rows if str(row["stat"]).endswith("_median") and np.isfinite(float(row["value"]))}
    low_med = med.get("low_score_tertile_rgpm_percentile_median", np.nan)
    high_med = med.get("high_score_tertile_rgpm_percentile_median", np.nan)
    rows.append({"stat": "high_vs_low_tertile_median_rgpm_lift_pp", "value": high_med - low_med if np.isfinite(high_med) and np.isfinite(low_med) else np.nan, "n": int(len(st)), "group": "tertile"})
    try:
        st["score_decile"] = pd.qcut(st["S_w_oof"].rank(method="first"), q=10, labels=False, duplicates="drop") + 1
    except ValueError:
        st["score_decile"] = np.ceil(st["S_w_oof"].rank(method="average", pct=True) * 10).clip(1, 10).astype(int)
    max_decile = int(pd.to_numeric(st["score_decile"], errors="coerce").max())
    min_decile = int(pd.to_numeric(st["score_decile"], errors="coerce").min())
    thresholds = {"top20": 80.0, "top10": 90.0}
    for name, threshold in thresholds.items():
        top_sub = st[pd.to_numeric(st["score_decile"], errors="coerce") == max_decile]
        bottom_sub = st[pd.to_numeric(st["score_decile"], errors="coerce") == min_decile]
        top_rate = float((top_sub["rgpm_percentile"] >= threshold).mean()) if len(top_sub) else np.nan
        bottom_rate = float((bottom_sub["rgpm_percentile"] >= threshold).mean()) if len(bottom_sub) else np.nan
        ratio = top_rate / max(bottom_rate, 1e-9) if np.isfinite(top_rate) and np.isfinite(bottom_rate) else np.nan
        rows.extend(
            [
                {"stat": f"top_score_decile_rgpm_{name}_rate", "value": top_rate, "n": int(len(top_sub)), "group": name},
                {"stat": f"bottom_score_decile_rgpm_{name}_rate", "value": bottom_rate, "n": int(len(bottom_sub)), "group": name},
                {"stat": f"top_vs_bottom_score_decile_rgpm_{name}_enrichment", "value": ratio, "n": int(len(st)), "group": name},
            ]
        )
    rows.append({"stat": "learned_oof_spearman", "value": safe_spearman(st["S_w_oof"], st["RGPM"]), "n": int(len(st)), "group": "association"})
    rows.append({"stat": "equal_weight_oof_spearman", "value": safe_spearman(st["S_equal"], st["RGPM"]), "n": int(len(st)), "group": "association"})
    return pd.DataFrame(rows)


def effect_summary_dict(effect_summary: pd.DataFrame) -> Dict[str, float]:
    if effect_summary.empty or "stat" not in effect_summary.columns or "value" not in effect_summary.columns:
        return {}
    out: Dict[str, float] = {}
    for row in effect_summary.itertuples(index=False):
        try:
            out[str(getattr(row, "stat"))] = float(getattr(row, "value"))
        except Exception:
            continue
    return out


def update_v3_diagnostics(comp: ComputedData) -> None:
    summary = dict(comp.diagnostics_summary)
    eff = effect_summary_dict(comp.effect_summary)
    if not comp.pair_scan_results.empty and "selected_for_panel_d" in comp.pair_scan_results.columns:
        selected = comp.pair_scan_results[comp.pair_scan_results["selected_for_panel_d"].astype(int) == 1]
    else:
        selected = pd.DataFrame()
    selected_pairs = [tuple(str(row.pair_key).split("|")) for row in selected.itertuples(index=False)] if not selected.empty else list(comp.selected_panel_d_pairs)
    metric_counts = Counter(metric for pair in selected_pairs for metric in pair)
    active_metrics = [key for key in METRIC_KEYS if key in comp.active_metric_keys]
    selected_p95 = pd.to_numeric(selected.get("p95_delta_vs_equal", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float) if not selected.empty else np.asarray([])
    selected_max = pd.to_numeric(selected.get("max_delta_vs_equal", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float) if not selected.empty else np.asarray([])
    contributing = comp.delta_diagnostics[comp.delta_diagnostics.get("contributing", 0).astype(int) == 1] if "contributing" in comp.delta_diagnostics.columns else pd.DataFrame()
    diagnostic = comp.delta_diagnostics[
        (comp.delta_diagnostics.get("primary_candidate", 0).astype(int) == 1)
        & (comp.delta_diagnostics.get("contributing", 0).astype(int) == 0)
    ] if "primary_candidate" in comp.delta_diagnostics.columns else pd.DataFrame()
    pair_checks = {
        "selected_pair_count": int(len(selected_pairs) == 4),
        "selected_pairs_cover_all_metrics": int(set(metric_counts.keys()) == set(active_metrics)),
        "selected_pairs_exactly_one_repeat": int(sorted(metric_counts.values()) == [1] * (len(active_metrics) - 1) + [2]) if active_metrics else 0,
        "selected_pairs_three_p95_ge_0_03": int(np.isfinite(selected_p95).sum() >= 4 and int(np.sum(selected_p95 >= PAIR_SCAN_P95_MIN)) >= 3),
        "selected_pairs_one_max_ge_0_08": int(np.isfinite(selected_max).any() and float(np.nanmax(selected_max)) >= PAIR_SCAN_MAX_MIN),
    }
    effect_checks = {
        "learned_oof_spearman_ge_0_45": int(float(summary.get("learned_oof_spearman", np.nan)) >= 0.45),
        "learned_vs_equal_delta_ge_0_03": int(float(summary.get("learned_vs_equal_delta", np.nan)) >= 0.03),
        "contributing_graph_deltas_ge_5": int(len(contributing) >= 5),
        "high_low_tertile_lift_ge_25pp": int(eff.get("high_vs_low_tertile_median_rgpm_lift_pp", np.nan) >= 25.0),
        "top_decile_enrichment_ge_5x": int(eff.get("top_vs_bottom_score_decile_rgpm_top20_enrichment", np.nan) >= 5.0),
    }
    pairwise_effect_weak = not bool(pair_checks["selected_pairs_three_p95_ge_0_03"] and pair_checks["selected_pairs_one_max_ge_0_08"])
    summary["contributing_graph_deltas"] = contributing["delta"].astype(str).tolist() if not contributing.empty else []
    summary["diagnostic_graph_deltas"] = diagnostic["delta"].astype(str).tolist() if not diagnostic.empty else []
    summary["n_contributing_graph_deltas"] = int(len(contributing))
    summary["mean_contributing_delta_reliability"] = float(contributing["reliability_weight"].mean()) if not contributing.empty else float("nan")
    summary["selected_panel_d_pairs"] = [list(pair) for pair in selected_pairs]
    summary["selected_panel_d_metric_counts"] = dict(metric_counts)
    summary["pairwise_effect_weak"] = bool(pairwise_effect_weak)
    summary["v3_effect_summary"] = eff
    summary["v3_pair_checks"] = pair_checks
    summary["v3_effect_checks"] = effect_checks
    summary["v3_thresholds"] = {
        "learned_oof_spearman_min": 0.45,
        "learned_vs_equal_delta_min": 0.03,
        "n_contributing_graph_deltas_min": 5,
        "high_vs_low_tertile_median_rgpm_lift_pp_min": 25.0,
        "top_score_decile_enrichment_min": 5.0,
        "selected_pair_p95_min": PAIR_SCAN_P95_MIN,
        "selected_pair_max_min": PAIR_SCAN_MAX_MIN,
    }
    summary["checks"] = {**summary.get("checks", {}), **pair_checks, **effect_checks}
    data_checks = summary.get("data_checks", {})
    summary["overall_pass"] = bool(all(pair_checks.values()) and all(effect_checks.values()) and all(data_checks.values()))
    if summary["overall_pass"]:
        summary["status_label"] = "validated empirical association"
    elif pairwise_effect_weak:
        summary["status_label"] = "moderate empirical association / pairwise effect weak"
    comp.diagnostics_summary = summary


def build_diagnostics_summary(
    active_delta_keys: Sequence[str],
    delta_diagnostics: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    score_table: pd.DataFrame,
    control_diagnostics: Optional[pd.DataFrame] = None,
    nonlinear_diagnostics: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    learned_rows = baseline_comparison[baseline_comparison["model"] == "learned_weight_oof"]
    equal_rows = baseline_comparison[baseline_comparison["model"] == "equal_weights"]
    best_single_rows = baseline_comparison[baseline_comparison["model"] == "best_single_indicator"]
    learned_rho = float(learned_rows["oof_spearman"].iloc[0]) if not learned_rows.empty else float("nan")
    equal_rho = float(equal_rows["oof_spearman"].iloc[0]) if not equal_rows.empty else float("nan")
    best_single_rho = float(best_single_rows["oof_spearman"].iloc[0]) if not best_single_rows.empty else float("nan")
    improvement = learned_rho - equal_rho if np.isfinite(learned_rho) and np.isfinite(equal_rho) else float("nan")
    improvement_vs_best_single = learned_rho - best_single_rho if np.isfinite(learned_rho) and np.isfinite(best_single_rho) else float("nan")
    active_diag = delta_diagnostics[delta_diagnostics["delta"].isin(active_delta_keys)]
    contributing_diag = delta_diagnostics[
        delta_diagnostics.get("contributing", pd.Series(0, index=delta_diagnostics.index)).astype(int) == 1
    ]
    diagnostic_diag = delta_diagnostics[
        (delta_diagnostics.get("primary_candidate", pd.Series(0, index=delta_diagnostics.index)).astype(int) == 1)
        & (delta_diagnostics.get("contributing", pd.Series(0, index=delta_diagnostics.index)).astype(int) == 0)
    ]
    active_cap_max = float(active_diag["z_cap_hit_rate"].max()) if not active_diag.empty else float("nan")
    mean_rel = float(active_diag["reliability_weight"].mean()) if "reliability_weight" in active_diag and not active_diag.empty else float("nan")
    mean_contrib_rel = float(contributing_diag["reliability_weight"].mean()) if "reliability_weight" in contributing_diag and not contributing_diag.empty else float("nan")
    score_vals = score_table["S_w_oof"].to_numpy(dtype=float) if "S_w_oof" in score_table.columns else np.array([])
    score_vals = score_vals[np.isfinite(score_vals)]
    score_iqr = float(np.percentile(score_vals, 75) - np.percentile(score_vals, 25)) if len(score_vals) else float("nan")
    thresholds = MAIN_FIGURE_THRESHOLDS
    data_thresholds = DATA_ADEQUACY_THRESHOLDS
    domain_diag, data_profile = compute_domain_adequacy_diagnostics(score_table, control_diagnostics)
    association_checks = {
        "active_graph_deltas": int(len(active_delta_keys) >= thresholds["active_graph_deltas_min"]),
        "active_delta_z_cap_hit_rate": int(
            np.isfinite(active_cap_max)
            and active_cap_max < thresholds["active_delta_z_cap_hit_rate_max"]
        ),
        "oof_spearman": int(np.isfinite(learned_rho) and learned_rho >= thresholds["oof_spearman_min"]),
        "learned_vs_equal": int(np.isfinite(improvement) and improvement >= thresholds["learned_vs_equal_min"]),
        "learned_vs_best_single": int(
            np.isfinite(improvement_vs_best_single)
            and improvement_vs_best_single >= thresholds["learned_vs_best_single_min"]
        ),
        "score_iqr": int(np.isfinite(score_iqr) and score_iqr > thresholds["score_iqr_min"]),
        "mean_delta_reliability": int(np.isfinite(mean_rel) and mean_rel >= 0.25),
        "contributing_graph_deltas": int(len(contributing_diag) >= 5),
    }
    data_checks = {
        "domain_count": int(data_profile["n_domains"] >= int(data_thresholds["domains_min"])),
        "total_papers": int(data_profile["total_papers"] >= int(data_thresholds["total_papers_min"])),
        "papers_per_domain": int(data_profile["min_papers_per_domain"] >= int(data_thresholds["papers_per_domain_min"])),
        "landmark_or_high_cases_per_domain": int(
            data_profile["min_landmark_or_high_cases_per_domain"] >= int(data_thresholds["landmark_or_high_cases_per_domain_min"])
        ),
        "matched_control_median": int(
            np.isfinite(data_profile["control_median_min_by_domain"])
            and data_profile["control_median_min_by_domain"] >= float(data_thresholds["control_median_min"])
        ),
        "matched_control_relaxation": int(
            np.isfinite(data_profile["relaxed_control_tier_rate_max_by_domain"])
            and data_profile["relaxed_control_tier_rate_max_by_domain"] <= float(data_thresholds["relaxed_control_tier_rate_max"])
        ),
    }
    checks = {**association_checks, **data_checks}
    data_pass = bool(all(data_checks.values()))
    nonlinear_rho = float("nan")
    nonlinear_delta = float("nan")
    nonlinear_interpretation = "nonlinear_upper_bound_not_available"
    if nonlinear_diagnostics is not None and not nonlinear_diagnostics.empty:
        summary_rows = nonlinear_diagnostics[nonlinear_diagnostics["model"] == "quadratic_ridge_oof"]
        if not summary_rows.empty:
            nonlinear_rho = float(summary_rows["test_spearman"].iloc[0])
            nonlinear_delta = nonlinear_rho - learned_rho if np.isfinite(nonlinear_rho) and np.isfinite(learned_rho) else float("nan")
            if np.isfinite(nonlinear_delta) and nonlinear_delta >= 0.05:
                nonlinear_interpretation = "nonlinear_signal_above_linear_simplex"
            elif np.isfinite(nonlinear_rho) and nonlinear_rho < 0.20:
                nonlinear_interpretation = "weak_signal_even_for_nonlinear_upper_bound"
            else:
                nonlinear_interpretation = "nonlinear_upper_bound_close_to_linear"
    overall_pass = bool(all(checks.values()))
    if overall_pass:
        status_label = "validated empirical association"
    elif not data_pass:
        status_label = "underpowered multi-domain diagnostic run"
    elif np.isfinite(learned_rho) and learned_rho >= 0.30:
        status_label = "moderate empirical association / still diagnostic"
    else:
        status_label = "weak empirical association / diagnostic run"
    return {
        "overall_pass": overall_pass,
        "status_label": status_label,
        "target_version": "RGPM structural residual over primary future graph deltas",
        "thresholds": thresholds,
        "data_requirements": data_thresholds,
        "checks": checks,
        "association_checks": association_checks,
        "data_checks": data_checks,
        "data_profile": data_profile,
        "domain_adequacy": domain_diag.to_dict("records"),
        "active_graph_deltas": list(active_delta_keys),
        "n_active_graph_deltas": int(len(active_delta_keys)),
        "contributing_graph_deltas": contributing_diag["delta"].astype(str).tolist() if not contributing_diag.empty else [],
        "diagnostic_graph_deltas": diagnostic_diag["delta"].astype(str).tolist() if not diagnostic_diag.empty else [],
        "n_contributing_graph_deltas": int(len(contributing_diag)),
        "active_delta_z_cap_hit_rate_max": active_cap_max,
        "mean_delta_reliability": mean_rel,
        "mean_contributing_delta_reliability": mean_contrib_rel,
        "learned_oof_spearman": learned_rho,
        "equal_weight_oof_spearman": equal_rho,
        "best_single_oof_spearman": best_single_rho,
        "learned_vs_equal_delta": improvement,
        "learned_vs_best_single_delta": improvement_vs_best_single,
        "nonlinear_upper_bound_oof_spearman": nonlinear_rho,
        "nonlinear_vs_linear_delta": nonlinear_delta,
        "nonlinear_diagnostic_interpretation": nonlinear_interpretation,
        "score_oof_iqr": score_iqr,
    }


def draw_panel_b(ax: plt.Axes, comp: ComputedData) -> None:
    panel_frame(ax, "b", "Structural-residual RGPM target construction")
    ex = comp.panel_b_example.copy()
    if ex.empty:
        raise ValueError("Panel b example data is empty.")
    rectangle_box(ax, 0.020, 0.115, 0.705, 0.775, "#FFFFFF", BORDER, 0.65)
    ax.text(0.372, 0.850, "Primary future graph-delta z-scores weighted by stability and mechanism", ha="center", va="center", fontsize=7.1, color="#0F3A75", fontweight="bold")
    z_axis_x0 = 0.430
    z_axis_w = 0.230
    ax.text(z_axis_x0, 0.790, "-4", ha="center", va="center", fontsize=5.3, color=TEXT_LIGHT)
    ax.text(z_axis_x0 + z_axis_w / 2, 0.790, "0", ha="center", va="center", fontsize=5.3, color=TEXT_LIGHT)
    ax.text(z_axis_x0 + z_axis_w, 0.790, "+4", ha="center", va="center", fontsize=5.3, color=TEXT_LIGHT)
    ax.plot([z_axis_x0, z_axis_x0 + z_axis_w], [0.765, 0.765], color="#CBD5E1", lw=0.8, transform=ax.transAxes)
    ax.plot([z_axis_x0 + z_axis_w / 2, z_axis_x0 + z_axis_w / 2], [0.178, 0.778], color="#9CA3AF", lw=0.6, ls="--", transform=ax.transAxes)
    y0 = 0.720
    row_step = min(0.057, 0.565 / max(len(ex), 1))
    mech_colors = {
        "Breadth": FIG2_EVIDENCE_CHANNEL_COLORS["Breadth"],
        "Brokerage": FIG2_EVIDENCE_CHANNEL_COLORS["Brokerage"],
        "Boundary": FIG2_EVIDENCE_CHANNEL_COLORS["Boundary"],
        "Consolidation": "#6B7280",
    }
    for i, row in enumerate(ex.itertuples(index=False)):
        y = y0 - i * row_step
        mech = str(getattr(row, "mechanism", ""))
        ax.scatter([0.045], [y], s=18, color=mech_colors.get(mech, "#9CA3AF"), transform=ax.transAxes, zorder=4)
        ax.text(0.065, y, getattr(row, "label"), ha="left", va="center", fontsize=5.2)
        rel = float(getattr(row, "reliability_weight", np.nan))
        ax.text(0.335, y, f"r={rel:.2f}" if np.isfinite(rel) else "", ha="right", va="center", fontsize=4.9, color=TEXT_LIGHT)
        z = float(getattr(row, "z"))
        x0 = z_axis_x0 + z_axis_w / 2
        x1 = z_axis_x0 + z_axis_w * ((np.clip(z, -4.0, 4.0) + 4.0) / 8.0)
        color = "#EF4444" if z >= 0 else "#3B82F6"
        ax.add_patch(mpatches.Rectangle((min(x0, x1), y - 0.012), max(abs(x1 - x0), 0.004), 0.024, transform=ax.transAxes, facecolor=color, edgecolor="none", alpha=0.82))
        if int(getattr(row, "clipped", 0)):
            ax.scatter([x1], [y], marker="^" if z >= 0 else "v", s=18, color="#111827", transform=ax.transAxes, zorder=6)
        ax.text(0.680, y, format_z_for_panel(z), ha="center", va="center", fontsize=5.2)
    ax.text(0.372, 0.080, "Controls: field/year/ref-bin matching; popularity residualization happens after mechanism aggregation", ha="center", va="center", fontsize=5.4)

    rectangle_box(ax, 0.755, 0.115, 0.225, 0.775, "#FFFFFF", BORDER, 0.65)
    ax.text(0.868, 0.850, "RGPM-resid", ha="center", va="center", fontsize=7.6, color="#0F3A75", fontweight="bold")
    ax.text(0.868, 0.700, r"$z_j=\frac{\Delta_j-\tilde{\Delta}_{ctrl}}{scale_j}$", ha="center", va="center", fontsize=6.6)
    ax.text(0.868, 0.595, r"$r_j=f(stability_j)$", ha="center", va="center", fontsize=7.0)
    ax.text(0.868, 0.480, r"$M_g=\sqrt{\frac{\sum_{j\in g} r_j\max(z_j,0)^2}{\sum_{j\in g} r_j}}$", ha="center", va="center", fontsize=5.7)
    ax.text(0.868, 0.375, r"$RGPM=rank(resid(RGPM_0))$", ha="center", va="center", fontsize=6.5, color="#1D4ED8", fontweight="bold")
    rounded_box(ax, 0.780, 0.205, 0.175, 0.110, "#F3F4F6", "#CBD5E1", 0.65, 0.010)
    n_active = int(len(comp.active_delta_keys))
    mean_rel = comp.diagnostics_summary.get("mean_delta_reliability", np.nan)
    ax.text(0.868, 0.260, f"active deltas: {n_active}\nmean reliability: {mean_rel:.2f}" if np.isfinite(float(mean_rel)) else f"active deltas: {n_active}", ha="center", va="center", fontsize=5.7, color=TEXT_MID)


def rank_decile_calibration_table(
    score_table: pd.DataFrame,
    seed: int,
    n_boot: int = 300,
) -> pd.DataFrame:
    """Summarize RGPM percentile by OOF-score decile."""
    st = score_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["S_w_oof", "RGPM"]).copy()
    if st.empty:
        return pd.DataFrame()
    st["score_percentile"] = st["S_w_oof"].rank(method="average", pct=True) * 100.0
    st["rgpm_percentile"] = st["RGPM"].rank(method="average", pct=True) * 100.0
    q = min(10, max(2, int(st["S_w_oof"].nunique())))
    st["score_decile"] = pd.qcut(st["S_w_oof"].rank(method="first"), q=q, labels=False, duplicates="drop") + 1
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for decile, sub in st.groupby("score_decile", sort=True):
        vals = sub["rgpm_percentile"].to_numpy(dtype=float)
        boot_means = []
        if len(vals) >= 2:
            for _ in range(int(n_boot)):
                idx = rng.integers(0, len(vals), size=len(vals))
                boot_means.append(float(np.mean(vals[idx])))
        if boot_means:
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        else:
            ci_low = ci_high = float(np.mean(vals)) if len(vals) else np.nan
        rows.append(
            {
                "score_decile": int(decile),
                "n": int(len(sub)),
                "score_percentile_mid": float(np.mean(sub["score_percentile"])),
                "rgpm_percentile_mean": float(np.mean(vals)) if len(vals) else np.nan,
                "rgpm_percentile_median": float(np.median(vals)) if len(vals) else np.nan,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
        )
    return pd.DataFrame(rows)


def channel_shares_from_metric_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    totals = {channel: 0.0 for channel in FIG2_EVIDENCE_CHANNEL_ORDER}
    for metric in METRIC_KEYS:
        if hasattr(weights, "get"):
            raw_weight = float(weights.get(metric, weights.get("w_" + metric, 0.0)))  # type: ignore[arg-type]
        else:
            raw_weight = 0.0
        mapping = FIG2_EVIDENCE_MAP.get(metric, {})
        denom = float(sum(mapping.values()))
        if denom <= 0:
            continue
        for channel, loading in mapping.items():
            totals[channel] += raw_weight * float(loading) / denom
    total = float(sum(totals.values()))
    if total <= 1e-12:
        return {channel: 1.0 / len(FIG2_EVIDENCE_CHANNEL_ORDER) for channel in FIG2_EVIDENCE_CHANNEL_ORDER}
    return {channel: float(value / total) for channel, value in totals.items()}


def channel_shares_to_xy(shares: Mapping[str, float]) -> Tuple[float, float]:
    breadth = float(shares.get("Breadth", 0.0))
    brokerage = float(shares.get("Brokerage", 0.0))
    boundary = float(shares.get("Boundary", 0.0))
    atypicality = float(shares.get("Atypicality", 0.0))
    return brokerage - breadth, boundary - atypicality


def diamond_xy_to_channel_shares(x: float, y: float) -> Dict[str, float]:
    residual = max(0.0, 1.0 - abs(float(x)) - abs(float(y)))
    base = residual / 4.0
    shares = {
        "Breadth": max(-float(x), 0.0) + base,
        "Brokerage": max(float(x), 0.0) + base,
        "Boundary": max(float(y), 0.0) + base,
        "Atypicality": max(-float(y), 0.0) + base,
    }
    total = float(sum(shares.values()))
    return {channel: value / total for channel, value in shares.items()} if total > 0 else shares


def channel_profile_candidates(
    active_metric_keys: Sequence[str],
    channel_shares: Mapping[str, float],
    n_candidates: int,
    seed: int,
) -> np.ndarray:
    active = [key for key in active_metric_keys if key in METRIC_KEYS]
    if not active:
        return np.empty((0, 0), dtype=float)
    key_to_idx = {key: i for i, key in enumerate(active)}
    rng = np.random.default_rng(seed)
    n = max(1, int(n_candidates))
    W = np.zeros((n, len(active)), dtype=float)
    for channel in FIG2_EVIDENCE_CHANNEL_ORDER:
        mass = float(channel_shares.get(channel, 0.0))
        pool = [key for key in FIG2_CHANNEL_METRIC_POOLS.get(channel, []) if key in key_to_idx]
        if mass <= 0 or not pool:
            continue
        if len(pool) == 1:
            W[:, key_to_idx[pool[0]]] += mass
            continue
        splits = rng.dirichlet(np.ones(len(pool)), size=n)
        for j, key in enumerate(pool):
            W[:, key_to_idx[key]] += mass * splits[:, j]
    sums = W.sum(axis=1, keepdims=True)
    empty = sums[:, 0] <= 1e-12
    if empty.any():
        W[empty, :] = 1.0 / len(active)
        sums = W.sum(axis=1, keepdims=True)
    W = W / np.maximum(sums, 1e-12)
    W = np.round(W, 8)
    _, idx = np.unique(W, axis=0, return_index=True)
    return W[np.sort(idx)]


def channel_profile_grid(
    comp: ComputedData,
    bins: int = 25,
    profile_n: int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = list(comp.active_metric_keys)
    st = comp.score_table.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[key + "_z" for key in active] + ["RGPM"]
    ).reset_index(drop=True)
    if st.empty or not active:
        return np.asarray([]), np.asarray([]), np.asarray([])
    X = st[[key + "_z" for key in active]].to_numpy(dtype=float)
    y = st["RGPM"].to_numpy(dtype=float)
    fold_ids = st["fold_id"].to_numpy(dtype=int) if "fold_id" in st.columns else np.zeros(len(st), dtype=int)
    folds = [np.where(fold_ids == f)[0] for f in sorted(set(fold_ids)) if f > 0 and np.sum(fold_ids == f) > 0]
    if len(folds) < 2:
        folds = make_folds(st, n_folds=3, mode="random", seed=411)
    equal_w = np.ones(len(active), dtype=float) / len(active)
    equal_rho = safe_spearman(X @ equal_w, y)
    centers = np.linspace(-1.0, 1.0, int(bins))
    xs: List[float] = []
    ys: List[float] = []
    perf_vals: List[float] = []
    for xval in centers:
        for yval in centers:
            if abs(float(xval)) + abs(float(yval)) > 1.0 + 1e-9:
                continue
            shares = diamond_xy_to_channel_shares(float(xval), float(yval))
            seed = stable_int_id((round(float(xval), 4), round(float(yval), 4), len(st), tuple(active)))
            W = channel_profile_candidates(active, shares, n_candidates=profile_n, seed=seed)
            if W.size == 0:
                value = np.nan
            else:
                perf = weight_performance_cv(X, y, W, folds, progress=False)
                value = float(np.nanmax(perf) - equal_rho) if np.isfinite(perf).any() else np.nan
            xs.append(float(xval))
            ys.append(float(yval))
            perf_vals.append(value)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), np.asarray(perf_vals, dtype=float)


def draw_panel_c(ax: plt.Axes, comp: ComputedData) -> None:
    panel_frame(ax, "c", "Fig. 2-aligned four-channel evidence landscape")
    x, y, perf = channel_profile_grid(
        comp,
        bins=int(getattr(comp, "profile_grid_size", 25)),
        profile_n=int(getattr(comp, "profile_n", 80)),
    )
    ax_d = ax.inset_axes([0.250, 0.120, 0.560, 0.740])
    ax_d.set_aspect("equal")
    ax_d.axis("off")
    finite = np.isfinite(perf)
    vals = perf[finite]
    lim = max(0.025, min(0.18, float(np.nanpercentile(np.abs(vals), 95)) if len(vals) else 0.05))
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    cmap = plt.get_cmap("RdYlBu_r").copy()
    if finite.sum() >= 3:
        tri = mtri.Triangulation(x[finite], y[finite])
        tpc = ax_d.tricontourf(
            tri,
            perf[finite],
            levels=np.linspace(-lim, lim, 17),
            cmap=cmap,
            norm=norm,
            extend="both",
        )
        ax_d.triplot(tri, color="white", lw=0.16, alpha=0.28, zorder=2)
        if np.nanmin(perf[finite]) < np.nanpercentile(perf[finite], 90) < np.nanmax(perf[finite]):
            ax_d.tricontour(
                tri,
                perf[finite],
                levels=[float(np.nanpercentile(perf[finite], 90))],
                colors="#111827",
                linewidths=0.9,
                linestyles="--",
                zorder=5,
            )
            ax_d.text(0.02, 0.88, "top 10% contour", ha="left", va="center", fontsize=5.4, color=TEXT_DARK)
    else:
        tpc = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        ax_d.text(0.0, 0.0, "channel grid unavailable", ha="center", va="center", fontsize=7.0, color="#7F1D1D", fontweight="bold")

    diamond = np.array([[-1, 0], [0, 1], [1, 0], [0, -1], [-1, 0]], dtype=float)
    ax_d.plot(diamond[:, 0], diamond[:, 1], color="#1E3A8A", lw=1.0, zorder=6)
    ax_d.axhline(0, color="#FFFFFF", lw=0.45, alpha=0.55, zorder=3)
    ax_d.axvline(0, color="#FFFFFF", lw=0.45, alpha=0.55, zorder=3)
    ax_d.set_xlim(-1.10, 1.10)
    ax_d.set_ylim(-1.10, 1.10)
    ax_d.text(-1.04, 0.0, "Breadth", ha="right", va="center", fontsize=6.5, color=FIG2_EVIDENCE_CHANNEL_COLORS["Breadth"], fontweight="bold", clip_on=False)
    ax_d.text(0.92, 0.0, "Brokerage", ha="left", va="center", fontsize=6.5, color=FIG2_EVIDENCE_CHANNEL_COLORS["Brokerage"], fontweight="bold", clip_on=False)
    ax_d.text(0.0, 1.04, "Boundary", ha="center", va="bottom", fontsize=6.5, color=FIG2_EVIDENCE_CHANNEL_COLORS["Boundary"], fontweight="bold", clip_on=False)
    ax_d.text(0.0, -1.04, "Atypicality", ha="center", va="top", fontsize=6.5, color=FIG2_EVIDENCE_CHANNEL_COLORS["Atypicality"], fontweight="bold", clip_on=False)

    shares = channel_shares_from_metric_weights({key: float(comp.best_weights.get(key, 0.0)) for key in METRIC_KEYS})
    sx, sy = channel_shares_to_xy(shares)
    ax_d.scatter([sx], [sy], marker="*", s=110, color="black", edgecolors="white", linewidths=0.65, zorder=8)
    ax_d.text(sx + 0.055, sy + 0.055, "learned", ha="left", va="bottom", fontsize=5.4, color=TEXT_DARK)

    cax = ax.inset_axes([0.835, 0.225, 0.035, 0.485])
    cb = plt.colorbar(tpc, cax=cax)
    cb.ax.tick_params(labelsize=5)
    cb.set_label("Best Δρ vs equal", fontsize=5.5)

    rounded_box(ax, 0.035, 0.205, 0.190, 0.570, "#FFFFFF", "#CBD5E1", 0.65, 0.012)
    ax.text(0.055, 0.735, "Fig. 2 channels", ha="left", va="center", fontsize=6.4, fontweight="bold")
    rows = [
        ("Breadth", "RS, PDE, RTD"),
        ("Brokerage", "B, RTD, Burt IP"),
        ("Boundary", "ΔQ0, B, Uzzi"),
        ("Atypicality", "Uzzi"),
    ]
    yy = 0.650
    for channel, desc in rows:
        color = FIG2_EVIDENCE_CHANNEL_COLORS[channel]
        ax.scatter([0.065], [yy], s=42, color=color, edgecolors="white", linewidths=0.45, transform=ax.transAxes, zorder=4)
        ax.text(0.095, yy, f"{channel}\n({desc})", ha="left", va="center", fontsize=5.55, transform=ax.transAxes)
        yy -= 0.120
    ax.text(
        0.500,
        0.060,
        "Diamond projection: left/right = Breadth/Brokerage, up/down = Boundary/Atypicality; star is final all-data weight.",
        ha="center",
        va="center",
        fontsize=5.4,
    )


def draw_panel_d(ax: plt.Axes, comp: ComputedData) -> None:
    panel_frame(ax, "d", "Robust pair-weight profile landscapes")
    positions = [(0.070, 0.210, 0.245, 0.610), (0.365, 0.210, 0.245, 0.610), (0.660, 0.210, 0.245, 0.610)]
    grid_results: List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    finite_values: List[np.ndarray] = []
    for xkey, ykey in PAIRWISE_LANDSCAPES:
        xs, ys, grid = pairwise_profile_grid(comp, xkey, ykey, bins=comp.profile_grid_size, profile_n=comp.profile_n)
        grid_results.append((xkey, ykey, xs, ys, grid))
        vals = grid[np.isfinite(grid)]
        if len(vals):
            finite_values.append(vals)
    if finite_values:
        pooled = np.concatenate(finite_values)
        lim = max(0.025, min(0.18, float(np.nanpercentile(np.abs(pooled), 95))))
    else:
        lim = 0.05
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad("#E5E7EB")
    last_im = None
    for (xkey, ykey, xs, ys, grid), (x0, y0, w, h) in zip(grid_results, positions):
        iax = ax.inset_axes([x0, y0, w, h])
        masked = np.ma.masked_invalid(grid)
        last_im = iax.imshow(masked, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap=cmap, norm=norm)
        iax.set_title(rf"{METRIC_LABELS[xkey]} / {METRIC_LABELS[ykey]} profile", fontsize=7)
        iax.set_xlabel(f"share to {METRIC_LABELS[xkey]}", fontsize=5.5)
        iax.set_ylabel("pair weight mass", fontsize=5.5)
        iax.tick_params(labelsize=5, length=2)
        best = comp.best_weights
        pair_mass = float(best[xkey] + best[ykey])
        pair_share_x = float(best[xkey] / pair_mass) if pair_mass > 1e-12 else 0.5
        iax.scatter([pair_share_x], [pair_mass], s=20, color="black", edgecolors="white", linewidths=0.5, zorder=5)
        finite_grid = np.isfinite(grid)
        if finite_grid.any():
            try:
                thr = float(np.nanpercentile(grid[finite_grid], 90))
                if np.nanmin(grid[finite_grid]) < thr < np.nanmax(grid[finite_grid]):
                    iax.contour(xs, ys, grid, levels=[thr], colors="white", linewidths=0.9, linestyles="--")
            except ValueError:
                pass
        for spine in iax.spines.values():
            spine.set_linewidth(0.5)
    if last_im is not None:
        cax = ax.inset_axes([0.930, 0.280, 0.025, 0.470])
        cb = plt.colorbar(last_im, cax=cax)
        cb.ax.tick_params(labelsize=5)
        cb.set_label("Best Δρ vs equal", fontsize=5.5)
    ax.text(
        0.505,
        0.085,
        "Shared robust 95% color scale; each cell fixes pair mass/share and profiles remaining active indicators.",
        ha="center",
        va="center",
        fontsize=5.7,
    )


def stratified_percentile_scatter(
    score_table: pd.DataFrame,
    seed: int,
    max_points: int = 1800,
) -> pd.DataFrame:
    st = score_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["S_w_oof", "RGPM"]).copy()
    if st.empty:
        return st
    st["score_percentile"] = st["S_w_oof"].rank(method="average", pct=True) * 100.0
    st["rgpm_percentile"] = st["RGPM"].rank(method="average", pct=True) * 100.0
    q = min(10, max(2, int(st["score_percentile"].nunique())))
    st["_pred_bin"] = pd.qcut(st["score_percentile"].rank(method="first"), q=q, labels=False, duplicates="drop")
    bins = [b for b in sorted(st["_pred_bin"].dropna().unique().tolist())]
    if not bins:
        return st.sample(n=min(len(st), max_points), random_state=seed)
    per_bin = max(1, int(math.ceil(max_points / max(len(bins), 1))))
    sampled = []
    for bin_id, sub in st.groupby("_pred_bin", sort=True):
        n = min(len(sub), per_bin)
        sampled.append(sub.sample(n=n, random_state=seed + int(bin_id)))
    return pd.concat(sampled, ignore_index=True, sort=False)


def draw_panel_f(ax: plt.Axes, comp: ComputedData) -> None:
    panel_frame(ax, "f", "Out-of-fold predicted vs observed RGPM percentiles")
    scatter_ax = ax.inset_axes([0.065, 0.185, 0.425, 0.660])
    st = comp.score_table.copy()
    rho = safe_spearman(st["S_w_oof"], st["RGPM"])
    cal = rank_decile_calibration_table(st, seed=comp.profile_n + 410)
    cloud = stratified_percentile_scatter(st, seed=comp.profile_n + 911, max_points=1800)
    if cloud.empty:
        scatter_ax.text(0.5, 0.5, "No OOF scatter data", ha="center", va="center", fontsize=7)
    else:
        landmark_series = cloud["is_landmark"] if "is_landmark" in cloud.columns else pd.Series(0, index=cloud.index)
        colors = np.where(pd.to_numeric(landmark_series, errors="coerce").fillna(0).astype(int) == 1, "#EF4444", "#2563EB")
        scatter_ax.scatter(
            cloud["score_percentile"],
            cloud["rgpm_percentile"],
            s=9,
            c=colors,
            alpha=0.34,
            edgecolors="none",
            rasterized=True,
            label="OOF papers",
        )
        scatter_ax.plot([0, 100], [0, 100], color="#9CA3AF", lw=0.9, ls="--", label="y = x")
        if not cal.empty:
            x = cal["score_percentile_mid"].to_numpy(dtype=float)
            scatter_ax.plot(
                x,
                cal["rgpm_percentile_median"].to_numpy(dtype=float),
                color="#111827",
                lw=1.55,
                marker="o",
                markersize=3.0,
                label="Decile median",
            )
            scatter_ax.plot(
                x,
                cal["rgpm_percentile_mean"].to_numpy(dtype=float),
                color="#1D4ED8",
                lw=1.05,
                alpha=0.80,
                label="Decile mean",
            )
        finite = np.isfinite(cloud["score_percentile"].to_numpy(dtype=float)) & np.isfinite(cloud["rgpm_percentile"].to_numpy(dtype=float))
        if finite.sum() >= 5:
            coef = np.polyfit(
                cloud.loc[finite, "score_percentile"].to_numpy(dtype=float),
                cloud.loc[finite, "rgpm_percentile"].to_numpy(dtype=float),
                1,
            )
            xx = np.linspace(0, 100, 100)
            scatter_ax.plot(xx, coef[0] * xx + coef[1], color="#F97316", lw=1.0, alpha=0.88, label="Linear fit")
        top = cal[cal["score_decile"] == cal["score_decile"].max()] if not cal.empty else pd.DataFrame()
        lift = float(top["rgpm_percentile_median"].iloc[0] - 50.0) if not top.empty else np.nan
        scatter_ax.text(
            0.045,
            0.940,
            f"OOF Spearman ρ = {rho:.2f}\nTop-decile median lift = {lift:+.1f} pp" if np.isfinite(lift) else f"OOF Spearman ρ = {rho:.2f}",
            transform=scatter_ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.4,
            color="#0F3A75",
            fontweight="bold",
        )
        scatter_ax.set_xlim(0, 100)
        scatter_ax.set_ylim(0, 100)
        scatter_ax.set_xlabel("OOF predicted RGPM percentile", fontsize=6)
        scatter_ax.set_ylabel("Observed RGPM percentile", fontsize=6)
        scatter_ax.legend(frameon=True, fontsize=4.7, loc="lower right")
    scatter_ax.tick_params(labelsize=5)
    scatter_ax.set_title("Stratified true-coordinate scatter", fontsize=7, color="#0F3A75", fontweight="bold")
    scatter_ax.grid(True, color="#E5E7EB", lw=0.45)
    for s in scatter_ax.spines.values():
        s.set_linewidth(0.5)

    try:
        score_q = pd.qcut(st["S_w_oof"].rank(method="first"), q=3, labels=["Low", "Mid", "High"])
    except ValueError:
        pct = st["S_w_oof"].rank(method="average", pct=True)
        score_q = pd.Series("Mid", index=st.index, dtype=object)
        score_q.loc[pct <= 1.0 / 3.0] = "Low"
        score_q.loc[pct >= 2.0 / 3.0] = "High"

    ax.text(0.720, 0.835, "Indicator profiles by OOF score tertile", ha="center", va="center", fontsize=7.2, color="#0F3A75", fontweight="bold")
    bar_ax = ax.inset_axes([0.565, 0.170, 0.385, 0.625])
    groups = ["Low", "Mid", "High"]
    group_colors = {"Low": "#3B82F6", "Mid": "#F59E0B", "High": "#EF4444"}
    y_pos = np.arange(len(METRIC_KEYS))
    offsets = {"Low": -0.22, "Mid": 0.00, "High": 0.22}
    for group in groups:
        sub = st[score_q.astype(str) == group]
        means = []
        ses = []
        for i, key in enumerate(METRIC_KEYS):
            vals = sub[key + "_z"].to_numpy(dtype=float)
            means.append(float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan)
            ses.append(bootstrap_mean_se(vals, seed=comp.profile_n + i + len(group)))
        bar_ax.barh(y_pos + offsets[group], means, height=0.18, color=group_colors[group], alpha=0.75, label=group)
        bar_ax.errorbar(means, y_pos + offsets[group], xerr=ses, fmt="none", ecolor="#111827", elinewidth=0.5, capsize=1.2)
    bar_ax.axvline(0, color="#6B7280", lw=0.7)
    bar_ax.set_yticks(y_pos)
    bar_ax.set_yticklabels([METRIC_LABELS[k] for k in METRIC_KEYS], fontsize=5.5)
    bar_ax.set_xlabel("Mean rank-normalized indicator ± bootstrap SE", fontsize=5.5)
    bar_ax.tick_params(labelsize=5, length=2)
    bar_ax.grid(True, axis="x", color="#E5E7EB", lw=0.45)
    bar_ax.legend(frameon=True, fontsize=5, loc="lower right")
    for s in bar_ax.spines.values():
        s.set_linewidth(0.5)


# =============================================================================
# v3 final panel overrides: contributing deltas, rectangular channels, 4-pair
# panel-d coverage, and distribution/enrichment Panel f.
# =============================================================================

def short_drop_reason(reason: str) -> str:
    text = str(reason or "").replace("low_reliability_weight", "low reliability")
    text = text.replace("near_constant_controls", "flat controls")
    text = text.replace("near_constant_global_delta", "flat delta")
    text = text.replace("diagnostic_or_consolidation_outcome", "diagnostic")
    parts = [p.strip() for p in text.split(";") if p.strip()]
    return ", ".join(parts[:2]) if parts else "diagnostic"


def draw_panel_b(ax: plt.Axes, comp: ComputedData) -> None:
    panel_frame(ax, "b", "Reliable graph-delta target construction")
    ex = comp.panel_b_example.copy()
    if ex.empty:
        raise ValueError("Panel b example data is empty.")
    rectangle_box(ax, 0.020, 0.105, 0.690, 0.785, "#FFFFFF", BORDER, 0.65)
    ax.text(0.365, 0.852, "Contributing future graph deltas used in RGPM", ha="center", va="center", fontsize=7.2, color="#0F3A75", fontweight="bold")
    z_axis_x0 = 0.428
    z_axis_w = 0.230
    ax.text(z_axis_x0, 0.798, "-4", ha="center", va="center", fontsize=5.1, color=TEXT_LIGHT)
    ax.text(z_axis_x0 + z_axis_w / 2, 0.798, "0", ha="center", va="center", fontsize=5.1, color=TEXT_LIGHT)
    ax.text(z_axis_x0 + z_axis_w, 0.798, "+4", ha="center", va="center", fontsize=5.1, color=TEXT_LIGHT)
    ax.plot([z_axis_x0, z_axis_x0 + z_axis_w], [0.773, 0.773], color="#CBD5E1", lw=0.8, transform=ax.transAxes)
    ax.plot([z_axis_x0 + z_axis_w / 2, z_axis_x0 + z_axis_w / 2], [0.188, 0.785], color="#9CA3AF", lw=0.55, ls="--", transform=ax.transAxes)
    y0 = 0.724
    row_step = min(0.072, 0.552 / max(len(ex), 1))
    mech_colors = {
        "Breadth": FIG2_EVIDENCE_CHANNEL_COLORS["Breadth"],
        "Brokerage": FIG2_EVIDENCE_CHANNEL_COLORS["Brokerage"],
        "Boundary": FIG2_EVIDENCE_CHANNEL_COLORS["Boundary"],
        "Consolidation": "#6B7280",
    }
    for i, row in enumerate(ex.itertuples(index=False)):
        y = y0 - i * row_step
        mech = str(getattr(row, "mechanism", ""))
        ax.scatter([0.045], [y], s=21, color=mech_colors.get(mech, "#9CA3AF"), transform=ax.transAxes, zorder=4)
        ax.text(0.067, y, getattr(row, "label"), ha="left", va="center", fontsize=5.35)
        rel = float(getattr(row, "reliability_weight", np.nan))
        ax.text(0.350, y, f"r={rel:.2f}" if np.isfinite(rel) else "", ha="right", va="center", fontsize=4.9, color=TEXT_LIGHT)
        z = float(getattr(row, "z"))
        x0 = z_axis_x0 + z_axis_w / 2
        x1 = z_axis_x0 + z_axis_w * ((np.clip(z, -4.0, 4.0) + 4.0) / 8.0)
        color = "#DC2626" if z >= 0 else "#2563EB"
        ax.add_patch(mpatches.Rectangle((min(x0, x1), y - 0.013), max(abs(x1 - x0), 0.004), 0.026, transform=ax.transAxes, facecolor=color, edgecolor="none", alpha=0.86))
        if int(getattr(row, "clipped", 0)):
            ax.scatter([x1], [y], marker="^" if z >= 0 else "v", s=18, color="#111827", transform=ax.transAxes, zorder=6)
        ax.text(0.676, y, format_z_for_panel(z), ha="center", va="center", fontsize=5.25)
    ax.text(0.365, 0.062, "Main RGPM excludes low-reliability primary deltas; these remain visible as diagnostics.", ha="center", va="center", fontsize=5.45)

    rectangle_box(ax, 0.735, 0.105, 0.245, 0.785, "#FFFFFF", BORDER, 0.65)
    ax.text(0.858, 0.852, "Target rule", ha="center", va="center", fontsize=7.4, color="#0F3A75", fontweight="bold")
    ax.text(0.858, 0.727, r"$z_j=(\Delta_j-\tilde{\Delta}_{ctrl})/scale_j$", ha="center", va="center", fontsize=5.9)
    ax.text(0.858, 0.620, r"$M_g=\sqrt{\sum r_j\max(z_j,0)^2/\sum r_j}$", ha="center", va="center", fontsize=5.55)
    ax.text(0.858, 0.515, r"$RGPM=rank(resid(M_{main}))$", ha="center", va="center", fontsize=6.25, color="#1D4ED8", fontweight="bold")
    contributing = comp.delta_diagnostics[comp.delta_diagnostics.get("contributing", pd.Series(0, index=comp.delta_diagnostics.index)).astype(int) == 1]
    diagnostic = comp.delta_diagnostics[
        (comp.delta_diagnostics.get("primary_candidate", pd.Series(0, index=comp.delta_diagnostics.index)).astype(int) == 1)
        & (comp.delta_diagnostics.get("contributing", pd.Series(0, index=comp.delta_diagnostics.index)).astype(int) == 0)
    ].copy()
    rounded_box(ax, 0.764, 0.365, 0.190, 0.088, "#ECFDF5", "#86EFAC", 0.65, 0.010)
    ax.text(0.859, 0.409, f"contributing: {len(contributing)} / {len(PRIMARY_RGPM_DELTA_KEYS)}", ha="center", va="center", fontsize=5.8, color="#166534", fontweight="bold")
    rounded_box(ax, 0.758, 0.118, 0.202, 0.215, "#F3F4F6", "#CBD5E1", 0.65, 0.010)
    ax.text(0.778, 0.306, "Diagnostic-only", ha="left", va="center", fontsize=5.65, color=TEXT_MID, fontweight="bold")
    if diagnostic.empty:
        ax.text(0.859, 0.225, "none", ha="center", va="center", fontsize=5.45, color=TEXT_LIGHT)
    else:
        yy = 0.266
        for row in diagnostic.sort_values("reliability_weight", ascending=False).head(4).itertuples(index=False):
            key = str(getattr(row, "delta"))
            label = DELTA_LABELS.get(key, key)
            rel = float(getattr(row, "reliability_weight", np.nan))
            ax.text(0.778, yy, f"{label}: r={rel:.2f}", ha="left", va="center", fontsize=4.95, color=TEXT_MID)
            ax.text(0.778, yy - 0.026, short_drop_reason(str(getattr(row, "drop_reasons", ""))), ha="left", va="center", fontsize=4.55, color=TEXT_LIGHT)
            yy -= 0.055


def channel_shares_to_metric_weight_deterministic(
    comp: ComputedData,
    active: Sequence[str],
    shares: Mapping[str, float],
) -> np.ndarray:
    active = [key for key in active if key in METRIC_KEYS]
    if not active:
        return np.asarray([], dtype=float)
    key_to_idx = {key: i for i, key in enumerate(active)}
    w = np.zeros(len(active), dtype=float)
    for channel in FIG2_EVIDENCE_CHANNEL_ORDER:
        mass = max(0.0, float(shares.get(channel, 0.0)))
        pool = [key for key in FIG2_CHANNEL_METRIC_POOLS.get(channel, []) if key in key_to_idx]
        if mass <= 0 or not pool:
            continue
        base = np.asarray([float(comp.best_weights.get(key, 0.0)) for key in pool], dtype=float)
        base = np.maximum(np.where(np.isfinite(base), base, 0.0), 0.0)
        if base.sum() <= 1e-12:
            base = np.ones(len(pool), dtype=float)
        base = base / base.sum()
        for key, share in zip(pool, base):
            w[key_to_idx[key]] += mass * float(share)
    if w.sum() <= 1e-12:
        w[:] = 1.0 / len(w)
    return w / w.sum()


def cv_delta_for_weight_matrix(comp: ComputedData, W: np.ndarray, active: Sequence[str], chunk_size: int = 600) -> np.ndarray:
    active = [key for key in active if key in METRIC_KEYS]
    st = comp.score_table.replace([np.inf, -np.inf], np.nan).dropna(subset=[key + "_z" for key in active] + ["RGPM"]).reset_index(drop=True)
    if st.empty or W.size == 0:
        return np.full(W.shape[0], np.nan, dtype=float)
    X = st[[key + "_z" for key in active]].to_numpy(dtype=float)
    y = st["RGPM"].to_numpy(dtype=float)
    fold_ids = st["fold_id"].to_numpy(dtype=int) if "fold_id" in st.columns else np.zeros(len(st), dtype=int)
    folds = [np.where(fold_ids == fold)[0] for fold in sorted(set(fold_ids)) if fold > 0 and np.sum(fold_ids == fold) >= 4]
    if len(folds) < 2:
        folds = make_folds(st, n_folds=3, mode="random", seed=stable_int_id(("rect", len(st), tuple(active))))
    equal_w = np.ones(len(active), dtype=float) / len(active)
    perfs = np.zeros(W.shape[0], dtype=float)
    counts = np.zeros(W.shape[0], dtype=float)
    for test_idx in folds:
        Xt = X[test_idx]
        yt = y[test_idx]
        eq = float(columnwise_spearman(Xt @ equal_w, yt)[0])
        if not np.isfinite(eq):
            continue
        for start in range(0, W.shape[0], int(chunk_size)):
            end = min(start + int(chunk_size), W.shape[0])
            rho = columnwise_spearman(Xt @ W[start:end].T, yt)
            valid = np.isfinite(rho)
            perfs[start:end][valid] += rho[valid] - eq
            counts[start:end][valid] += 1.0
    return np.divide(perfs, np.maximum(counts, 1.0), out=np.full_like(perfs, np.nan), where=counts > 0)


def channel_rectangle_grid(comp: ComputedData, bins: int = 25, axis_bins: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active = [key for key in comp.active_metric_keys if key in METRIC_KEYS]
    xs = np.linspace(0.0, 1.0, max(5, int(bins)))
    ys = np.linspace(0.0, 1.0, max(5, int(bins)))
    axis_masses = np.linspace(0.05, 0.95, max(5, int(axis_bins)))
    rows: List[np.ndarray] = []
    cell_ids: List[Tuple[int, int, float]] = []
    for j, boundary_share in enumerate(ys):
        for i, brokerage_share in enumerate(xs):
            for horizontal_mass in axis_masses:
                shares = {
                    "Breadth": float(horizontal_mass * (1.0 - brokerage_share)),
                    "Brokerage": float(horizontal_mass * brokerage_share),
                    "Boundary": float((1.0 - horizontal_mass) * boundary_share),
                    "Atypicality": float((1.0 - horizontal_mass) * (1.0 - boundary_share)),
                }
                rows.append(channel_shares_to_metric_weight_deterministic(comp, active, shares))
                cell_ids.append((j, i, float(horizontal_mass)))
    if not rows:
        return xs, ys, np.full((len(ys), len(xs)), np.nan), np.full((len(ys), len(xs)), np.nan)
    W = np.vstack(rows)
    deltas = cv_delta_for_weight_matrix(comp, W, active, chunk_size=500)
    grid = np.full((len(ys), len(xs)), np.nan, dtype=float)
    best_axis = np.full((len(ys), len(xs)), np.nan, dtype=float)
    for value, (j, i, horizontal_mass) in zip(deltas, cell_ids):
        if not np.isfinite(value):
            continue
        if not np.isfinite(grid[j, i]) or value > grid[j, i]:
            grid[j, i] = float(value)
            best_axis[j, i] = float(horizontal_mass)
    return xs, ys, grid, best_axis


def draw_panel_c(ax: plt.Axes, comp: ComputedData) -> None:
    panel_frame(ax, "c", "Fig. 2-aligned four-channel landscape")
    xs, ys, grid, best_axis = channel_rectangle_grid(comp, bins=int(comp.profile_grid_size), axis_bins=9)
    finite = np.isfinite(grid)
    vals = grid[finite]
    lim = max(0.015, min(0.18, float(np.nanpercentile(np.abs(vals), 95)) if len(vals) else 0.05))
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad("#E5E7EB")
    ax_r = ax.inset_axes([0.260, 0.180, 0.535, 0.635])
    im = ax_r.imshow(np.ma.masked_invalid(grid), origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap=cmap, norm=norm)
    if finite.any():
        try:
            thr = float(np.nanpercentile(grid[finite], 90))
            if np.nanmin(grid[finite]) < thr < np.nanmax(grid[finite]):
                ax_r.contour(xs, ys, grid, levels=[thr], colors="#111827", linewidths=0.8, linestyles="--")
        except ValueError:
            pass
    ax_r.set_xlabel("Breadth  ← channel mix →  Brokerage", fontsize=5.8)
    ax_r.set_ylabel("Boundary share", fontsize=5.8)
    ax_r.tick_params(labelsize=5.0, length=2)
    ax_r.text(0.00, 1.035, "Boundary", ha="left", va="bottom", fontsize=6.0, color=FIG2_EVIDENCE_CHANNEL_COLORS["Boundary"], fontweight="bold", transform=ax_r.transAxes)
    ax_r.text(0.00, -0.105, "Atypicality", ha="left", va="top", fontsize=6.0, color=FIG2_EVIDENCE_CHANNEL_COLORS["Atypicality"], fontweight="bold", transform=ax_r.transAxes)
    shares = channel_shares_from_metric_weights({key: float(comp.best_weights.get(key, 0.0)) for key in METRIC_KEYS})
    horizontal = float(shares.get("Breadth", 0.0) + shares.get("Brokerage", 0.0))
    vertical = float(shares.get("Boundary", 0.0) + shares.get("Atypicality", 0.0))
    sx = float(shares.get("Brokerage", 0.0) / horizontal) if horizontal > 1e-12 else 0.5
    sy = float(shares.get("Boundary", 0.0) / vertical) if vertical > 1e-12 else 0.5
    ax_r.scatter([sx], [sy], marker="*", s=105, color="black", edgecolors="white", linewidths=0.65, zorder=8)
    ax_r.text(min(0.96, sx + 0.035), min(0.96, sy + 0.040), "learned", ha="left", va="bottom", fontsize=5.3, color=TEXT_DARK)
    for spine in ax_r.spines.values():
        spine.set_linewidth(0.55)
    cax = ax.inset_axes([0.835, 0.255, 0.035, 0.455])
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=5)
    cb.set_label("Best Δρ vs equal", fontsize=5.2)
    rounded_box(ax, 0.035, 0.232, 0.135, 0.490, "#FFFFFF", "#CBD5E1", 0.65, 0.012)
    ax.text(0.053, 0.690, "Channels", ha="left", va="center", fontsize=6.0, fontweight="bold")
    yy = 0.620
    for channel, desc in [("Breadth", "RS/PDE/RTD"), ("Brokerage", "B/RTD/Burt IP"), ("Boundary", "ΔQ0/B/Uzzi"), ("Atypicality", "Uzzi")]:
        ax.scatter([0.058], [yy], s=30, color=FIG2_EVIDENCE_CHANNEL_COLORS[channel], edgecolors="white", linewidths=0.4, transform=ax.transAxes)
        ax.text(0.082, yy, f"{channel}\n{desc}", ha="left", va="center", fontsize=4.9, transform=ax.transAxes)
        yy -= 0.105
    ax.text(0.500, 0.067, "Rectangle scans Fig. 2's four evidence channels; each cell profiles horizontal-vs-vertical channel mass.", ha="center", va="center", fontsize=5.35)


def selected_panel_d_pairs(comp: ComputedData) -> List[Tuple[str, str]]:
    if comp.selected_panel_d_pairs:
        return [tuple(pair) for pair in comp.selected_panel_d_pairs]
    scan = comp.pair_scan_results
    if not scan.empty and "selected_for_panel_d" in scan.columns:
        rows = scan[scan["selected_for_panel_d"].astype(int) == 1]
        if not rows.empty:
            return [tuple(str(row.pair_key).split("|")) for row in rows.itertuples(index=False)]
    return PAIRWISE_LANDSCAPES + [("BurtIP", "B")]


def draw_panel_d(ax: plt.Axes, comp: ComputedData) -> None:
    panel_frame(ax, "d", "Pair-weight heatmaps selected by full scan")
    pairs = selected_panel_d_pairs(comp)[:4]
    positions = [
        (0.070, 0.575, 0.365, 0.275),
        (0.515, 0.575, 0.365, 0.275),
        (0.070, 0.150, 0.365, 0.275),
        (0.515, 0.150, 0.365, 0.275),
    ]
    grids = getattr(comp, "_pair_grids", {})
    grid_results: List[Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    finite_values: List[np.ndarray] = []
    for xkey, ykey in pairs:
        key = tuple(sorted((xkey, ykey)))
        if key in grids:
            xs, ys, grid = grids[key]
        else:
            xs, ys, grid = pairwise_profile_grid(comp, xkey, ykey, bins=comp.profile_grid_size, profile_n=comp.profile_n)
        grid_results.append((xkey, ykey, xs, ys, grid))
        vals = grid[np.isfinite(grid)]
        if len(vals):
            finite_values.append(vals)
    if finite_values:
        pooled = np.concatenate(finite_values)
        lim = max(0.015, min(0.18, float(np.nanpercentile(np.abs(pooled), 95))))
    else:
        lim = 0.05
    norm = mcolors.TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad("#E5E7EB")
    last_im = None
    scan = comp.pair_scan_results.copy()
    for (xkey, ykey, xs, ys, grid), (x0, y0, w, h) in zip(grid_results, positions):
        iax = ax.inset_axes([x0, y0, w, h])
        last_im = iax.imshow(np.ma.masked_invalid(grid), origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap=cmap, norm=norm)
        iax.set_title(f"{METRIC_LABELS.get(xkey, xkey)} × {METRIC_LABELS.get(ykey, ykey)}", fontsize=6.8, pad=2)
        iax.set_xlabel(f"share to {METRIC_LABELS.get(xkey, xkey)}", fontsize=5.2, labelpad=1)
        iax.set_ylabel("pair mass", fontsize=5.2, labelpad=1)
        iax.tick_params(labelsize=4.8, length=1.8, pad=1)
        best = comp.best_weights
        pair_mass = float(best.get(xkey, 0.0) + best.get(ykey, 0.0))
        pair_share_x = float(best.get(xkey, 0.0) / pair_mass) if pair_mass > 1e-12 else 0.5
        iax.scatter([pair_share_x], [pair_mass], s=18, color="black", edgecolors="white", linewidths=0.5, zorder=5)
        finite_grid = np.isfinite(grid)
        if finite_grid.any():
            try:
                thr = float(np.nanpercentile(grid[finite_grid], 90))
                if np.nanmin(grid[finite_grid]) < thr < np.nanmax(grid[finite_grid]):
                    iax.contour(xs, ys, grid, levels=[thr], colors="white", linewidths=0.75, linestyles="--")
            except ValueError:
                pass
        pair_key = "|".join(tuple(sorted((xkey, ykey))))
        meta = scan[scan.get("pair_key", pd.Series(dtype=str)).astype(str) == pair_key] if not scan.empty and "pair_key" in scan.columns else pd.DataFrame()
        if not meta.empty:
            p95 = float(meta["p95_delta_vs_equal"].iloc[0])
            coverage = int(meta.get("coverage_pair", pd.Series([0])).iloc[0])
            if coverage or (np.isfinite(p95) and p95 < PAIR_SCAN_P95_MIN):
                iax.text(0.03, 0.93, "coverage" if coverage else "weak", transform=iax.transAxes, ha="left", va="top", fontsize=4.7, color="#111827", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#CBD5E1", lw=0.4, alpha=0.82))
            iax.text(0.97, 0.04, f"p95={p95:.2f}", transform=iax.transAxes, ha="right", va="bottom", fontsize=4.7, color="#111827")
        for spine in iax.spines.values():
            spine.set_linewidth(0.5)
    if last_im is not None:
        cax = ax.inset_axes([0.920, 0.235, 0.022, 0.535])
        cb = plt.colorbar(last_im, cax=cax)
        cb.ax.tick_params(labelsize=4.8)
        cb.set_label("Δρ vs equal", fontsize=5.2)
    ax.text(0.505, 0.073, "Four pairs are selected from all 21 scans under full seven-metric coverage; one metric may repeat once.", ha="center", va="center", fontsize=5.45)


def score_distribution_groups(score_table: pd.DataFrame) -> pd.DataFrame:
    st = score_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["S_w_oof", "RGPM"]).copy()
    if st.empty:
        return st
    st["rgpm_percentile"] = st["RGPM"].rank(method="average", pct=True) * 100.0
    st["score_percentile"] = st["S_w_oof"].rank(method="average", pct=True) * 100.0
    try:
        st["score_tertile"] = pd.qcut(st["S_w_oof"].rank(method="first"), q=3, labels=["Low", "Mid", "High"])
    except ValueError:
        pct = st["S_w_oof"].rank(method="average", pct=True)
        st["score_tertile"] = "Mid"
        st.loc[pct <= 1.0 / 3.0, "score_tertile"] = "Low"
        st.loc[pct >= 2.0 / 3.0, "score_tertile"] = "High"
    try:
        st["score_decile"] = pd.qcut(st["S_w_oof"].rank(method="first"), q=10, labels=False, duplicates="drop") + 1
    except ValueError:
        st["score_decile"] = np.ceil(st["S_w_oof"].rank(method="average", pct=True) * 10).clip(1, 10).astype(int)
    return st


def draw_panel_f(ax: plt.Axes, comp: ComputedData) -> None:
    panel_frame(ax, "f", "Out-of-fold effect separation and top-tail enrichment")
    st = score_distribution_groups(comp.score_table)
    eff = effect_summary_dict(comp.effect_summary if not comp.effect_summary.empty else compute_effect_summary(comp))
    dist_ax = ax.inset_axes([0.065, 0.185, 0.400, 0.650])
    enrich_ax = ax.inset_axes([0.555, 0.185, 0.390, 0.650])
    if st.empty:
        dist_ax.text(0.5, 0.5, "No OOF effect data", ha="center", va="center", fontsize=7)
        enrich_ax.axis("off")
        return
    groups = ["Low", "Mid", "High"]
    colors = ["#2563EB", "#F59E0B", "#DC2626"]
    data = [st[st["score_tertile"].astype(str) == group]["rgpm_percentile"].to_numpy(dtype=float) for group in groups]
    bp = dist_ax.boxplot(data, positions=np.arange(1, 4), widths=0.55, patch_artist=True, showfliers=False, medianprops={"color": "#111827", "lw": 1.1})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.40)
        patch.set_edgecolor(color)
        patch.set_linewidth(0.9)
    for whisker in bp["whiskers"]:
        whisker.set_color("#6B7280")
        whisker.set_linewidth(0.7)
    for cap in bp["caps"]:
        cap.set_color("#6B7280")
        cap.set_linewidth(0.7)
    medians = [float(np.nanmedian(vals)) if len(vals) else np.nan for vals in data]
    dist_ax.plot(np.arange(1, 4), medians, color="#111827", lw=1.2, marker="o", markersize=3.2, zorder=4)
    dist_ax.set_xticks(np.arange(1, 4))
    dist_ax.set_xticklabels(groups, fontsize=5.6)
    dist_ax.set_ylim(0, 100)
    dist_ax.set_ylabel("Observed RGPM percentile", fontsize=5.8)
    dist_ax.set_title("Observed RGPM by OOF score tertile", fontsize=7.0, color="#0F3A75", fontweight="bold")
    dist_ax.grid(True, axis="y", color="#E5E7EB", lw=0.45)
    lift = eff.get("high_vs_low_tertile_median_rgpm_lift_pp", np.nan)
    rho = eff.get("learned_oof_spearman", safe_spearman(st["S_w_oof"], st["RGPM"]))
    dist_ax.text(0.04, 0.94, f"ρ={rho:.2f}\nHigh-low lift={lift:.1f} pp", transform=dist_ax.transAxes, ha="left", va="top", fontsize=6.3, color="#0F3A75", fontweight="bold")
    for spine in dist_ax.spines.values():
        spine.set_linewidth(0.5)

    decile_rows = []
    for decile, sub in st.groupby("score_decile", sort=True):
        decile_rows.append(
            {
                "decile": int(decile),
                "top20_rate": float((sub["rgpm_percentile"] >= 80.0).mean()),
                "top10_rate": float((sub["rgpm_percentile"] >= 90.0).mean()),
                "n": int(len(sub)),
            }
        )
    dec = pd.DataFrame(decile_rows).sort_values("decile")
    x = dec["decile"].to_numpy(dtype=float)
    enrich_ax.bar(x, dec["top20_rate"].to_numpy(dtype=float) * 100.0, color="#2563EB", alpha=0.64, width=0.72, label="RGPM top 20%")
    enrich_ax.plot(x, dec["top10_rate"].to_numpy(dtype=float) * 100.0, color="#DC2626", lw=1.35, marker="o", markersize=2.8, label="RGPM top 10%")
    enrich_ax.axhline(20, color="#9CA3AF", lw=0.7, ls="--")
    enrich_ax.axhline(10, color="#FCA5A5", lw=0.65, ls=":")
    enrich_ax.set_xlim(0.35, 10.65)
    enrich_ax.set_ylim(0, max(55, float(np.nanmax(dec["top20_rate"].to_numpy(dtype=float)) * 115.0) if not dec.empty else 55))
    enrich_ax.set_xlabel("OOF score decile", fontsize=5.8)
    enrich_ax.set_ylabel("Tail probability (%)", fontsize=5.8)
    enrich_ax.set_title("Top-tail enrichment across score deciles", fontsize=7.0, color="#0F3A75", fontweight="bold")
    enrich_ax.tick_params(labelsize=5.0, length=2)
    enrich_ax.grid(True, axis="y", color="#E5E7EB", lw=0.45)
    ratio = eff.get("top_vs_bottom_score_decile_rgpm_top20_enrichment", np.nan)
    top_rate = eff.get("top_score_decile_rgpm_top20_rate", np.nan)
    bottom_rate = eff.get("bottom_score_decile_rgpm_top20_rate", np.nan)
    enrich_ax.text(0.04, 0.94, f"Top/bottom={ratio:.1f}×\n{top_rate*100:.1f}% vs {bottom_rate*100:.1f}%" if np.isfinite(ratio) else "Enrichment unavailable", transform=enrich_ax.transAxes, ha="left", va="top", fontsize=6.3, color="#7F1D1D", fontweight="bold")
    enrich_ax.legend(frameon=True, fontsize=5.0, loc="upper right")
    for spine in enrich_ax.spines.values():
        spine.set_linewidth(0.5)


def draw_full_figure(comp: ComputedData, tau: int, out_path: Path) -> None:
    setup_style()
    fig = plt.figure(figsize=(20, 12.6), dpi=300)
    status = diagnostic_status_text(comp)
    if is_diagnostic_run(comp):
        title = "Fig. 3 diagnostic run | Structural-residual RGPM and hybrid weight learning"
        subtitle = "Strict out-of-fold scores are reported; weak results should be interpreted as model diagnostics rather than a final scoring claim"
    else:
        title = "Fig. 3 | Data-driven weight learning for graph-perturbation scoring"
        subtitle = "Weights are selected by their ability to recover popularity-adjusted future graph-structural perturbation"
    fig.text(0.5, 0.985, title, ha="center", va="top", fontsize=18.0, fontweight="bold")
    fig.text(0.5, 0.957, subtitle, ha="center", va="top", fontsize=10.8, color=TEXT_MID)

    gs = GridSpec(3, 6, figure=fig, height_ratios=[1.0, 1.05, 1.0], hspace=0.085, wspace=0.035)
    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    ax_c = fig.add_subplot(gs[1, 0:2])
    ax_d = fig.add_subplot(gs[1, 2:6])
    ax_e = fig.add_subplot(gs[2, 0:2])
    ax_f = fig.add_subplot(gs[2, 2:6])
    draw_panel_a(ax_a, comp, tau)
    draw_panel_b(ax_b, comp)
    draw_panel_c(ax_c, comp)
    draw_panel_d(ax_d, comp)
    draw_panel_e(ax_e, comp)
    draw_panel_f(ax_f, comp)
    fig.text(
        0.012,
        0.012,
        "(1) Seven indicators are computed at publication day (G0).  "
        "(2) RGPM residualizes reliability-weighted future graph-delta outcomes against popularity and size covariates.  "
        "(3) Learned weights are evaluated strictly out-of-fold.  "
        f"Status: {status}.",
        ha="left",
        va="bottom",
        fontsize=5.5,
        color=TEXT_DARK,
    )
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()
