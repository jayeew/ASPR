#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 1 knowledge-graph perturbation pipeline
============================================

This script downloads real scholarly metadata from OpenAlex, builds a paper-level
hybrid citation knowledge graph, compresses it into a topic/community-level
backbone, computes four perturbation signatures, and draws the Fig. 1 style
multi-panel figure:

    1-5 years | 1-10 years | 1-15 years | 1-20 years | 1-25 years
    Expansion | Bridging | Reconfiguration | Compression

The code is intentionally modular so that you can replace the CRISPR config
with any other field by editing only a YAML file.

Typical usage
-------------

    cp .env.example .env  # then set OPENALEX_API_KEY or OPENALEX_API_KEYS
    python experiments/kg_perturbation_fig1/fig1_knowledge_perturbation_v3.py \
        --config experiments/kg_perturbation_fig1/configs/crispr.yaml

For a four-domain Nature-style Fig. 1:

    python experiments/kg_perturbation_fig1/fig1_knowledge_perturbation_v3.py \
        --config experiments/kg_perturbation_fig1/configs/crispr.yaml \
                 experiments/kg_perturbation_fig1/configs/graphene.yaml \
                 experiments/kg_perturbation_fig1/configs/ipsc.yaml \
                 experiments/kg_perturbation_fig1/configs/transformer.yaml

Outputs
-------

For each domain, the script writes:

    outputs/kg_perturbation_fig1/<slug>/works_raw.jsonl
    outputs/kg_perturbation_fig1/<slug>/works_selected.csv
    outputs/kg_perturbation_fig1/<slug>/paper_edges.csv
    outputs/kg_perturbation_fig1/<slug>/topic_nodes.csv
    outputs/kg_perturbation_fig1/<slug>/topic_edges.csv
    outputs/kg_perturbation_fig1/<slug>/perturbation_metrics.csv
    outputs/kg_perturbation_fig1/<slug>/fig1_<slug>_real.png/.svg/.pdf

If multiple configs are supplied, it additionally writes:

    outputs/kg_perturbation_fig1/fig1_multi_domain_real.png/.svg/.pdf

Notes
-----

1. The graph is real-data based, but the metrics are intended for Fig. 1
   conceptual/empirical visualization. For Nature-level statistical claims,
   add matched controls and null models in a separate validation script.
2. The script caches downloads. Re-running with --use-cache avoids repeated API
   calls.
3. OpenAlex search can be broad. Use balanced window quotas and field-specific
   terms in YAML to prevent recent papers from dominating the graph.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import random
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np
import pandas as pd
import requests
import yaml
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Configuration defaults
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.env import getenv
from experiments.figure_quality import (
    write_figure_quality_report,
    write_json,
    write_run_manifest,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "kg_perturbation_fig1"

SELECT_FIELDS = ",".join(
    [
        "id",
        "doi",
        "display_name",
        "title",
        "publication_year",
        "publication_date",
        "type",
        "language",
        "cited_by_count",
        "referenced_works",
        "referenced_works_count",
        "primary_topic",
        "topics",
        "keywords",
        "mesh",
        "ids",
        "fwci",
        "citation_normalized_percentile",
        "counts_by_year",
        # Uncomment in YAML by setting include_abstract: true. This field can be
        # large, so it is not requested unless needed.
        # "abstract_inverted_index",
    ]
)

SELECT_FIELDS_WITH_ABSTRACT = SELECT_FIELDS + ",abstract_inverted_index"

DEFAULT_CONFIG: Dict[str, Any] = {
    "domain_name": "CRISPR-Cas genome editing",
    "slug": "crispr",
    "search_query": '("CRISPR" OR "Cas9" OR "RNA-guided nuclease" OR "genome editing")',
    "search_groups": [],
    "start_year": 2000,
    "end_year": 2024,
    "window_size": 5,
    "custom_windows": [],
    "snapshot_years": [],
    "work_types": [],  # Example: ["article", "preprint", "review"]
    "language": "en",
    "include_abstract": False,
    "max_works_per_window": 1200,
    "max_anchor_citers": 500,
    "fetch_anchor_citers": True,
    "anchors": [],
    "relevance_filter": {
        "enabled": False,
        "positive_keywords": [],
        "strong_positive_keywords": [],
        "negative_keywords": [],
        "negative_primary_topics": [],
        "keep_anchor_citers": True,
    },
    "api": {
        "sleep_seconds": 0.10,
        "max_retries": 6,
        "timeout_seconds": 60,
        "per_page": 100,
    },
    "graph": {
        "max_papers_for_graph": 3500,
        "use_direct_citation": True,
        "use_bibliographic_coupling": True,
        "use_cocitation": True,
        "direct_weight": 3.0,
        "bibliographic_weight": 1.0,
        "cocitation_weight": 0.8,
        "min_shared_references": 2,
        "min_cocitations": 2,
        "max_reference_fanout": 80,
        "max_cocited_refs_per_paper": 80,
        "max_edges": 120000,
        "community_resolution": 1.8,
        "max_communities": 14,
        "min_community_size": 5,
        "random_seed": 42,
    },
    "metrics": {
        "betweenness_sample": 250,
        "semantic_sample": 600,
        "curve_mode": "cumulative_positive",
        "signed_compression": True,
    },
    "plot": {
        "fig_width_single": 22,
        "fig_height_single": 9.5,
        "fig_width_multi": 26,
        "row_height_multi": 3.8,
        "dpi": 300,
        "edge_keep_quantile": 0.55,
        "max_edges_per_panel": 40,
        "max_labels_per_panel": 8,
        "node_size_min": 80,
        "node_size_max": 260,
        "show_time_axis": True,
        "display_mode": "cluster_schematic",
        "display_max_topics": 9,
        "display_min_topic_size": 6,
        "display_max_backbone_edges": 18,
        "display_extra_edges": 8,
        "cluster_radius_min": 0.13,
        "cluster_radius_max": 0.24,
        "max_representative_papers": 7,
        "min_papers_per_display_topic": 3,
        "show_internal_cluster_edges": True,
        "show_panel_captions": True,
        "panel_captions": [
            "Fragmented prior knowledge",
            "Mechanistic consolidation",
            "Innovation shock: programmable Cas9",
            "Field reconfiguration",
            "Compression into translational hubs",
        ],
        "metric_x_axis": "years",
        "metric_panel_title": "Dominant parameter trajectories explaining the graph transitions in panel a",
        "metric_y_label": "Standardized\nparameter value",
        "metric_landmark_label": "",
        "dominant_parameter_ylim": [-1.5, 1.5],
        "dominant_parameters": [
            {"key": "B"},
            {"key": "RTD"},
            {"key": "Uzzi"},
            {"key": "DeltaQ"},
        ],
        "parameter_callouts": [],
        "parameter_interpretation_boxes": [],
        "parameter_box_width": 0.18,
        "parameter_box_height": 0.34,
        "parameter_box_gap": 0.035,
        "parameter_box_y": -0.56,
        "parameter_box_title_size": 7.8,
        "parameter_box_formula_size": 8.2,
        "parameter_box_description_size": 7.0,
        "parameter_box_show_icons": False,
        "parameter_box_corner_radius": 5.0,
        "parameter_box_linewidth": 0.75,
        "show_metric_connectors": True,
        "title": "Landmark papers induce measurable perturbations in citation knowledge graphs",
        "subtitle": "Expansion, bridging, reconfiguration and compression across cumulative citation-knowledge snapshots",
        "show_retrieval_date": False,
    },
}

METRIC_NAMES = ["Expansion", "Bridging", "Reconfiguration", "Compression"]
METRIC_COLORS = {
    "Expansion": "#3B82F6",
    "Bridging": "#F97316",
    "Reconfiguration": "#10B981",
    "Compression": "#8B5CF6",
}

PARAMETER_SPECS: Dict[str, Dict[str, Any]] = {
    "B": {
        "label": r"$B$ (bridge centrality)",
        "source": "B_proxy_raw",
        "color": "#1D4ED8",
    },
    "RS": {
        "label": "RS (reference span)",
        "source": "RS_proxy_raw",
        "color": "#0F766E",
    },
    "RTD": {
        "label": "RTD (reference target diversity)",
        "source": "RTD_proxy_raw",
        "color": "#F97316",
    },
    "Uzzi": {
        "label": "Uzzi novelty (-p10)",
        "source": "Uzzi_proxy_raw",
        "color": "#7C3AED",
    },
    "DeltaQ": {
        "label": r"$\Delta Q$ directionality",
        "source": "DeltaQ_directionality_raw",
        "color": "#166534",
        "center_zero": True,
    },
    "BurtIP": {
        "label": "Burt IP (structural holes)",
        "source": "BurtIP_proxy_raw",
        "color": "#0891B2",
    },
    "PDE": {
        "label": "PDE (diffusion potential)",
        "source": "PDE_proxy_raw",
        "color": "#B45309",
    },
}

STOPWORDS_EXTRA = {
    "using",
    "based",
    "study",
    "studies",
    "analysis",
    "effect",
    "effects",
    "method",
    "methods",
    "approach",
    "new",
    "novel",
    "human",
    "mouse",
    "cell",
    "cells",
    "protein",
    "proteins",
    "system",
    "systems",
    "gene",
    "genes",
    "role",
    "model",
    "models",
    "data",
    "via",
    "towards",
    "toward",
    "high",
    "low",
    "large",
    "small",
    "field",
    "biology",
    "science",
    "natural",
    "engineering",
    "genetic",
    "crispr",
    "genome",
    "genomic",
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def deep_update(base: Dict[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge a user config dictionary into defaults."""
    out = dict(base)
    for key, val in update.items():
        if isinstance(val, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(dict(out[key]), val)
        else:
            out[key] = val
    return out


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = deep_update(DEFAULT_CONFIG, user_cfg)
    if not cfg.get("slug"):
        cfg["slug"] = slugify(cfg["domain_name"])
    validate_time_windows(cfg)
    return cfg


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "domain"


def split_api_keys(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            out.extend(split_api_keys(item))
        return list(dict.fromkeys(out))
    text = str(value or "").strip()
    if not text:
        return []
    return list(dict.fromkeys(part.strip() for part in re.split(r"[,;\s]+", text) if part.strip()))


def normalize_openalex_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = str(value).strip()
    if value.lower() in {"nan", "none", "null", "<na>"}:
        return None
    m = re.search(r"(W\d+)$", value)
    if m:
        return f"https://openalex.org/{m.group(1)}"
    if value.startswith("https://openalex.org/W"):
        return value
    return value


def short_openalex_id(value: Optional[str]) -> Optional[str]:
    value = normalize_openalex_id(value)
    if not value:
        return None
    return value.rsplit("/", 1)[-1]


def normalize_doi(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = str(value).strip().lower()
    value = value.replace("https://doi.org/", "").replace("http://doi.org/", "")
    value = value.replace("doi:", "")
    return value.strip()


def normalize_text_key(value: Optional[str]) -> str:
    text = str(value or "").lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_arxiv_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip().lower()
    text = text.replace("arxiv:", "").replace("https://arxiv.org/abs/", "")
    text = text.replace("http://arxiv.org/abs/", "")
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", text)
    return m.group(1) if m else text or None


def config_data_fingerprint(cfg: Mapping[str, Any]) -> str:
    """Hash only the config parts that affect fetched/cached works."""
    keys = [
        "domain_name",
        "slug",
        "search_query",
        "search_groups",
        "start_year",
        "end_year",
        "window_size",
        "custom_windows",
        "snapshot_years",
        "language",
        "include_abstract",
        "max_works_per_window",
        "max_anchor_citers",
        "fetch_anchor_citers",
        "work_types",
        "anchors",
        "relevance_filter",
    ]
    payload = {k: cfg.get(k) for k in keys}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_manifest(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "config_hash": config_data_fingerprint(cfg),
        "domain_name": cfg.get("domain_name"),
        "slug": cfg.get("slug"),
        "search_query": cfg.get("search_query"),
        "search_groups": cfg.get("search_groups") or [],
        "start_year": cfg.get("start_year"),
        "end_year": cfg.get("end_year"),
        "window_size": cfg.get("window_size"),
        "custom_windows": cfg.get("custom_windows") or [],
        "snapshot_years": cfg.get("snapshot_years") or [],
        "anchors": cfg.get("anchors") or [],
        "retrieval_date": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }


def cache_matches_config(manifest_path: Path, cfg: Mapping[str, Any]) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(manifest.get("config_hash") or "") == config_data_fingerprint(cfg)


def year_int(value: Any) -> Optional[int]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return int(value)
    except Exception:
        return None


def make_rolling_windows(start_year: int, end_year: int, window_size: int) -> List[Tuple[int, int]]:
    windows = []
    y = start_year
    while y <= end_year:
        windows.append((y, min(end_year, y + window_size - 1)))
        y += window_size
    return windows


def parse_window_pair(value: Any) -> Tuple[int, int]:
    if isinstance(value, Mapping):
        start = year_int(value.get("start") or value.get("from") or value.get("rolling_start"))
        end = year_int(value.get("end") or value.get("to") or value.get("rolling_end"))
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        start = year_int(value[0])
        end = year_int(value[1])
    else:
        raise ValueError(f"Invalid window entry: {value!r}")
    if start is None or end is None or start > end:
        raise ValueError(f"Invalid window bounds: {value!r}")
    return int(start), int(end)


def make_rolling_windows_from_config(cfg: Mapping[str, Any]) -> List[Tuple[int, int]]:
    custom = cfg.get("custom_windows") or []
    if custom:
        return [parse_window_pair(v) for v in custom]
    return make_rolling_windows(int(cfg["start_year"]), int(cfg["end_year"]), int(cfg["window_size"]))


def make_cumulative_windows(start_year: int, rolling_windows: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [(start_year, end) for _, end in rolling_windows]


def make_cumulative_windows_from_config(
    cfg: Mapping[str, Any],
    rolling_windows: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    snapshot_years = [year_int(y) for y in (cfg.get("snapshot_years") or [])]
    snapshot_years = [int(y) for y in snapshot_years if y is not None]
    if snapshot_years:
        return [(int(cfg["start_year"]), min(int(cfg["end_year"]), y)) for y in snapshot_years]
    return make_cumulative_windows(int(cfg["start_year"]), rolling_windows)


def validate_time_windows(cfg: Mapping[str, Any]) -> None:
    rolling = make_rolling_windows_from_config(cfg)
    cumulative = make_cumulative_windows_from_config(cfg, rolling)
    if len(rolling) != len(cumulative):
        raise ValueError(
            "custom_windows and snapshot_years must have the same length when both are provided."
        )
    prev_end = None
    for start, end in rolling:
        if prev_end is not None and start <= prev_end:
            raise ValueError("custom_windows must be ordered and non-overlapping.")
        prev_end = end


def window_label(start_year: int, end_year: int, cfg_start_year: int) -> str:
    elapsed = end_year - cfg_start_year + 1
    return f"1-{elapsed}\n{cfg_start_year}-{end_year}"


def in_year_range(year: Optional[int], start: int, end: int) -> bool:
    return year is not None and start <= year <= end


def abstract_from_inverted_index(inv: Optional[Mapping[str, Sequence[int]]]) -> str:
    if not inv:
        return ""
    pairs: List[Tuple[int, str]] = []
    for word, positions in inv.items():
        if not isinstance(positions, Sequence):
            continue
        for pos in positions:
            try:
                pairs.append((int(pos), word))
            except Exception:
                continue
    if not pairs:
        return ""
    pairs.sort(key=lambda x: x[0])
    return " ".join(word for _, word in pairs)


def safe_log1p(x: float) -> float:
    if x is None or not np.isfinite(x) or x <= 0:
        return 0.0
    return float(np.log1p(x))


def robust_minmax(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray([0.0 if v is None or not np.isfinite(v) else float(v) for v in values], dtype=float)
    if len(arr) == 0:
        return arr
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def weighted_quantile_threshold(weights: Sequence[float], keep_quantile: float) -> float:
    if not weights:
        return 0.0
    keep_quantile = min(max(float(keep_quantile), 0.0), 0.99)
    return float(np.quantile(np.asarray(weights, dtype=float), keep_quantile))


def clean_label(s: str, max_len: int = 34) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    s = s.replace("Crispr", "CRISPR").replace("Cas", "Cas")
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def normalize_positions(pos: Mapping[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
    if not pos:
        return {}
    xs = np.array([float(v[0]) for v in pos.values()])
    ys = np.array([float(v[1]) for v in pos.values()])
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    xr = max(xmax - xmin, 1e-9)
    yr = max(ymax - ymin, 1e-9)
    out = {}
    for k, v in pos.items():
        out[k] = np.array([2.0 * (float(v[0]) - xmin) / xr - 1.0, 2.0 * (float(v[1]) - ymin) / yr - 1.0])
    return out


def repel_positions(pos: Dict[Any, np.ndarray], radii: Mapping[Any, float], iterations: int = 250, step: float = 0.012) -> Dict[Any, np.ndarray]:
    if len(pos) <= 1:
        return pos
    keys = list(pos.keys())
    arr = np.array([pos[k] for k in keys], dtype=float)
    r = np.array([float(radii.get(k, 0.08)) for k in keys], dtype=float)
    center_pull = 0.002
    for _ in range(iterations):
        disp = np.zeros_like(arr)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                delta = arr[i] - arr[j]
                dist = np.linalg.norm(delta) + 1e-9
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    push = (min_dist - dist) * (delta / dist) * step
                    disp[i] += push
                    disp[j] -= push
        disp -= center_pull * arr
        arr += disp
    out = {k: arr[i] for i, k in enumerate(keys)}
    return normalize_positions(out)




def score_keyword_match(text: str, keywords: Sequence[str]) -> float:
    """Soft keyword score used only for publication-layout template assignment."""
    s = " " + re.sub(r"[^a-z0-9]+", " ", str(text).lower()) + " "
    score = 0.0
    for kw in keywords or []:
        k = str(kw).lower().strip()
        if not k:
            continue
        # Give phrase hits more credit than isolated word hits.
        if " " in k:
            if k in s:
                score += 2.5 + 0.15 * len(k.split())
        else:
            if f" {k} " in s:
                score += 1.0
    return score


def display_templates(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    templates = cfg.get("plot", {}).get("topic_layout_templates") or []
    out = []
    for t in templates:
        try:
            out.append(
                {
                    "name": str(t.get("name") or t.get("label") or "Topic"),
                    "x": float(t.get("x", 0.0)),
                    "y": float(t.get("y", 0.0)),
                    "keywords": list(t.get("keywords") or []),
                }
            )
        except Exception:
            continue
    return out


def build_display_comm_map(G: nx.Graph, full_comm_map: Mapping[str, int], cfg: Dict[str, Any]) -> Dict[str, int]:
    """Select a small set of interpretable communities for the main figure.

    The old code compressed all discarded communities into one "other" community,
    creating the large isolated gray node. For a publication figure, we instead
    keep only the strongest/anchor communities and hide the long tail from the
    visual layer while keeping all papers for metrics.
    """
    pcfg = cfg.get("plot", {})
    max_topics_raw = pcfg.get("display_max_topics", cfg.get("graph", {}).get("max_communities", 9))
    if str(max_topics_raw).lower() == "auto":
        max_topics = int(pcfg.get("display_max_topics_auto_cap", 12))
    else:
        max_topics = int(max_topics_raw)
    min_size = int(pcfg.get("display_min_topic_size", cfg.get("graph", {}).get("min_community_size", 6)))
    members = community_members(full_comm_map)
    weighted_deg = dict(G.degree(weight="weight"))
    scored: List[Tuple[float, int]] = []
    anchor_comms: Set[int] = set()
    event_specs = pcfg.get("event_topic_keywords") or []
    event_specs = [{"keywords": item} if isinstance(item, str) else dict(item or {}) for item in event_specs]
    event_candidates_by_spec: List[List[Tuple[float, float, int]]] = [[] for _ in event_specs]
    event_strength: Dict[int, float] = collections.defaultdict(float)
    for c, nodes in members.items():
        has_anchor = any(G.nodes[n].get("anchor_label") for n in nodes if n in G)
        if has_anchor:
            anchor_comms.add(int(c))
        comm_text = ""
        if event_specs:
            text_parts: List[str] = []
            for n in nodes:
                if n not in G:
                    continue
                data = G.nodes[n]
                text_parts.append(str(data.get("title") or ""))
                text_parts.append(str(data.get("primary_topic") or ""))
                text_parts.extend(str(t) for t in (data.get("topics") or []))
            comm_text = " ".join(text_parts)
        if len(nodes) < min_size and not has_anchor:
            continue
        citation_score = sum(safe_log1p(float(G.nodes[n].get("cited_by_count") or 0)) for n in nodes if n in G)
        degree_score = sum(float(weighted_deg.get(n, 0.0)) for n in nodes)
        score = 3.0 * safe_log1p(len(nodes)) + 0.45 * safe_log1p(citation_score) + 0.35 * safe_log1p(degree_score)
        event_match = False
        for spec_i, spec in enumerate(event_specs):
            event_score = score_keyword_match(comm_text, spec.get("keywords") or [])
            if event_score > 0:
                event_match = True
                event_strength[int(c)] += event_score
                event_candidates_by_spec[spec_i].append((event_score, score, int(c)))
        if has_anchor:
            score += 1000.0
        if event_match:
            score += 80.0
        scored.append((score, int(c)))
    scored.sort(reverse=True)
    max_topics = max(max_topics, len(anchor_comms))

    event_comms: Set[int] = set()
    event_topics_per_spec = max(0, int(pcfg.get("event_topics_per_spec", 1)))
    for candidates in event_candidates_by_spec:
        candidates.sort(reverse=True)
        for _, _, c in candidates[:event_topics_per_spec]:
            event_comms.add(c)
    max_event_topics = int(pcfg.get("event_max_topics", max_topics))
    if len(event_comms) > max_event_topics:
        event_comms = set(
            sorted(
                event_comms,
                key=lambda c: (float(event_strength.get(c, 0.0)), len(members.get(c, []))),
                reverse=True,
            )[:max_event_topics]
        )

    keep: List[int] = []
    event_slots = max(0, max_topics - len(anchor_comms))
    ranked_event_comms = sorted(
        event_comms - anchor_comms,
        key=lambda c: (float(event_strength.get(c, 0.0)), len(members.get(c, []))),
        reverse=True,
    )[:event_slots]
    for c in sorted(anchor_comms) + ranked_event_comms:
        if c not in keep:
            keep.append(c)
    for _, c in scored:
        if c not in keep:
            keep.append(c)
        if len(keep) >= max_topics:
            break
    for c in sorted(anchor_comms):
        if c not in keep:
            if len(keep) < max_topics:
                keep.append(c)
            elif keep:
                keep[-1] = c
    keep = list(dict.fromkeys(keep))[:max_topics]
    remap = {old: new for new, old in enumerate(keep)}
    out: Dict[str, int] = {}
    for n, old in full_comm_map.items():
        if old in remap:
            out[n] = remap[old]
    return out


def deterministic_disc_points(key: Any, n: int, radius: float) -> np.ndarray:
    """Deterministic small-paper positions inside a topic bubble."""
    rng = random.Random(str(key))
    pts = []
    if n <= 0:
        return np.zeros((0, 2))
    # One central-ish point plus a ring gives a hand-crafted Nature-like cluster.
    for i in range(n):
        if i == 0:
            rr = 0.08 * radius
            theta = rng.random() * 2 * math.pi
        else:
            rr = radius * (0.34 + 0.55 * ((i - 1) / max(1, n - 1)))
            theta = 2 * math.pi * (i - 1) / max(1, n - 1) + rng.uniform(-0.25, 0.25)
        pts.append([rr * math.cos(theta), rr * math.sin(theta)])
    return np.asarray(pts, dtype=float)


def log_scaled_radius(values: Sequence[float], rmin: float, rmax: float) -> List[float]:
    arr = np.log1p(np.asarray([max(0.0, float(v)) for v in values], dtype=float))
    if len(arr) == 0:
        return []
    lo, hi = float(arr.min()), float(arr.max())
    if abs(hi - lo) < 1e-9:
        return [0.5 * (rmin + rmax)] * len(arr)
    return list(rmin + (arr - lo) / (hi - lo) * (rmax - rmin))



def filter_topic_graph_for_display(TG: nx.Graph, end_year: int, pcfg: Mapping[str, Any]) -> nx.Graph:
    """Hide weakly represented topics in early snapshots.

    A topic should not appear in the first 5-year panel just because one paper
    from a future module exists. This threshold makes expansion visually legible.
    """
    min_papers = int(pcfg.get("min_papers_per_display_topic", 1))
    if min_papers <= 1 or TG.number_of_nodes() == 0:
        return TG
    H = TG.copy()
    drop = []
    for n, d in H.nodes(data=True):
        has_anchor = bool(d.get("anchor_labels")) and (d.get("anchor_year") is None or int(d.get("anchor_year")) <= end_year)
        if not has_anchor and int(d.get("n_papers") or 0) < min_papers:
            drop.append(n)
    H.remove_nodes_from(drop)
    return H


def select_backbone_edges(
    TG: nx.Graph,
    TGprev: nx.Graph,
    pcfg: Mapping[str, Any],
    anchor_nodes: Set[int],
) -> List[Tuple[int, int, Mapping[str, Any], bool]]:
    """Return a sparse, readable edge backbone: maximum spanning tree + top extras."""
    if TG.number_of_edges() == 0:
        return []
    max_edges = int(pcfg.get("display_max_backbone_edges", 18))
    extra_edges = int(pcfg.get("display_extra_edges", 8))
    chosen: Dict[Tuple[int, int], Tuple[int, int, Mapping[str, Any]]] = {}

    def add_edge(u: int, v: int, d: Mapping[str, Any]) -> None:
        key = tuple(sorted((int(u), int(v))))
        chosen[key] = (int(u), int(v), d)

    # 1) Backbone tree for each connected component.
    for comp in nx.connected_components(TG):
        sub = TG.subgraph(comp).copy()
        if sub.number_of_edges() == 0:
            continue
        try:
            tree = nx.maximum_spanning_tree(sub, weight="weight")
            for u, v, d in tree.edges(data=True):
                add_edge(u, v, d)
        except Exception:
            pass

    # 2) Anchor incident edges, so the innovation event is visually connected.
    anchor_edges = []
    for u, v, d in TG.edges(data=True):
        if u in anchor_nodes or v in anchor_nodes:
            anchor_edges.append((float(d.get("weight", 1.0)), u, v, d))
    for _, u, v, d in sorted(anchor_edges, reverse=True)[: max(3, extra_edges)]:
        add_edge(u, v, d)

    # 3) A small number of strongest remaining edges.
    all_edges = sorted(TG.edges(data=True), key=lambda x: float(x[2].get("weight", 1.0)), reverse=True)
    for u, v, d in all_edges:
        add_edge(u, v, d)
        if len(chosen) >= max_edges:
            break

    prev_edges = {tuple(sorted((int(u), int(v)))) for u, v in TGprev.edges()}
    prev_weights = {
        tuple(sorted((int(u), int(v)))): float(d.get("weight", 0.0))
        for u, v, d in TGprev.edges(data=True)
    }
    gain_ratio = float(pcfg.get("edge_gain_highlight_ratio", 0.15))
    gain_abs = float(pcfg.get("edge_gain_highlight_min", 5.0))
    out = []
    for key, (u, v, d) in chosen.items():
        prev_weight = prev_weights.get(key, 0.0)
        weight_gain = float(d.get("weight", 0.0)) - prev_weight
        is_new = key not in prev_edges or weight_gain >= max(gain_abs, gain_ratio * max(prev_weight, 1.0))
        out.append((u, v, d, is_new))
    out.sort(key=lambda x: (x[3], float(x[2].get("weight", 1.0))), reverse=True)
    return out[:max_edges]


# -----------------------------------------------------------------------------
# OpenAlex client
# -----------------------------------------------------------------------------


class OpenAlexClient:
    """Small resilient wrapper around the OpenAlex REST API."""

    BASE_URL = "https://api.openalex.org"

    def __init__(
        self,
        api_key: str,
        api_keys: Optional[Sequence[str]] = None,
        email: Optional[str] = None,
        sleep_seconds: float = 0.1,
        max_retries: int = 6,
        timeout_seconds: int = 60,
    ):
        self.api_keys = split_api_keys([api_key, api_keys])
        self.api_key = self.api_keys[0] if self.api_keys else ""
        self._api_key_index = 0
        self.email = email
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "kg-perturbation-fig1/1.0"})

    def next_api_key(self) -> str:
        if not self.api_keys:
            return ""
        key = self.api_keys[self._api_key_index % len(self.api_keys)]
        self._api_key_index += 1
        return key

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        base_params = dict(params or {})
        url = f"{self.BASE_URL.rstrip('/')}/{path.lstrip('/')}"

        for attempt in range(self.max_retries + 1):
            params = dict(base_params)
            api_key = self.next_api_key()
            if api_key:
                params["api_key"] = api_key
            if self.email:
                params["mailto"] = self.email
            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout_seconds)
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Request failed after retries: {url}: {exc}") from exc
                time.sleep(min(60.0, 2.0**attempt))
                continue

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except Exception:
                        delay = min(60.0, 2.0**attempt)
                else:
                    delay = min(60.0, 2.0**attempt)
                time.sleep(delay)
                continue

            msg = resp.text[:1000]
            raise RuntimeError(f"OpenAlex error {resp.status_code} for {resp.url}: {msg}")

        raise RuntimeError(f"OpenAlex request exhausted retries: {url}")

    def list_works(
        self,
        search_query: Optional[str],
        from_year: Optional[int],
        to_year: Optional[int],
        max_records: int,
        work_types: Optional[Sequence[str]] = None,
        language: Optional[str] = "en",
        include_abstract: bool = False,
        extra_filters: Optional[Sequence[str]] = None,
        sort: Optional[str] = None,
        per_page: int = 100,
    ) -> List[Dict[str, Any]]:
        filters: List[str] = ["is_retracted:false", "is_paratext:false"]
        if from_year:
            filters.append(f"from_publication_date:{from_year}-01-01")
        if to_year:
            filters.append(f"to_publication_date:{to_year}-12-31")
        if language:
            filters.append(f"language:{language}")
        if work_types:
            filters.append("type:" + "|".join(work_types))
        if extra_filters:
            filters.extend(extra_filters)

        params: Dict[str, Any] = {
            "filter": ",".join(filters),
            "per_page": int(per_page),
            "cursor": "*",
            "select": SELECT_FIELDS_WITH_ABSTRACT if include_abstract else SELECT_FIELDS,
        }
        if search_query:
            params["search"] = search_query
        if sort:
            params["sort"] = sort

        out: List[Dict[str, Any]] = []
        pbar = tqdm(total=max_records, desc="OpenAlex works", unit="work", leave=False)
        while len(out) < max_records:
            data = self.get_json("/works", params=params)
            results = data.get("results", []) or []
            if not results:
                break
            remaining = max_records - len(out)
            out.extend(results[:remaining])
            pbar.update(min(len(results), remaining))
            next_cursor = (data.get("meta") or {}).get("next_cursor")
            if not next_cursor:
                break
            params["cursor"] = next_cursor
            if len(results) < per_page:
                break
        pbar.close()
        return out

    def get_work_by_doi(self, doi: str, include_abstract: bool = False) -> Optional[Dict[str, Any]]:
        doi_norm = normalize_doi(doi)
        if not doi_norm:
            return None
        select = SELECT_FIELDS_WITH_ABSTRACT if include_abstract else SELECT_FIELDS

        # OpenAlex singleton endpoint accepts DOI-like identifiers when URL encoded.
        doi_url = "https://doi.org/" + doi_norm
        encoded = urllib.parse.quote(doi_url, safe="")
        try:
            return self.get_json(f"/works/{encoded}", params={"select": select})
        except Exception:
            pass

        # Fallback to list+filter. Some historical DOI records are normalized in
        # different ways, so try both bare DOI and doi.org form.
        for candidate in [doi_norm, doi_url]:
            try:
                data = self.get_json(
                    "/works",
                    params={
                        "filter": f"doi:{candidate}",
                        "per_page": 1,
                        "select": select,
                    },
                )
                results = data.get("results", []) or []
                if results:
                    return results[0]
            except Exception:
                continue
        return None

    def get_work_by_openalex_id(self, openalex_id: str, include_abstract: bool = False) -> Optional[Dict[str, Any]]:
        sid = short_openalex_id(openalex_id)
        if not sid:
            return None
        select = SELECT_FIELDS_WITH_ABSTRACT if include_abstract else SELECT_FIELDS
        try:
            return self.get_json(f"/works/{sid}", params={"select": select})
        except Exception:
            return None

    def get_work_by_title(
        self,
        title: str,
        year: Optional[int] = None,
        include_abstract: bool = False,
    ) -> Optional[Dict[str, Any]]:
        title_key = normalize_text_key(title)
        if not title_key:
            return None
        from_year = year - 1 if year else None
        to_year = year + 1 if year else None
        try:
            records = self.list_works(
                search_query=title,
                from_year=from_year,
                to_year=to_year,
                max_records=25,
                work_types=[],
                language=None,
                include_abstract=include_abstract,
                sort=None,
                per_page=25,
            )
        except Exception:
            return None
        for rec in records:
            rec_title = rec.get("display_name") or rec.get("title")
            if normalize_text_key(rec_title) == title_key:
                return rec
        return None


# -----------------------------------------------------------------------------
# Work normalization and fetching
# -----------------------------------------------------------------------------


def extract_topic_names(work: Mapping[str, Any]) -> List[str]:
    names: List[str] = []
    primary = work.get("primary_topic") or {}
    if isinstance(primary, Mapping) and primary.get("display_name"):
        names.append(str(primary["display_name"]))
    for t in work.get("topics") or []:
        if isinstance(t, Mapping) and t.get("display_name"):
            names.append(str(t["display_name"]))
    for k in work.get("keywords") or []:
        if isinstance(k, Mapping) and k.get("display_name"):
            names.append(str(k["display_name"]))
    for m in work.get("mesh") or []:
        if isinstance(m, Mapping) and m.get("descriptor_name"):
            names.append(str(m["descriptor_name"]))
    # Keep order but drop duplicates.
    seen = set()
    uniq = []
    for name in names:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(name)
    return uniq


def normalize_work(work: Mapping[str, Any], include_abstract: bool = False) -> Dict[str, Any]:
    wid = normalize_openalex_id(work.get("id") or (work.get("ids") or {}).get("openalex"))
    title = work.get("display_name") or work.get("title") or "Untitled"
    doi = normalize_doi(work.get("doi") or (work.get("ids") or {}).get("doi"))
    year = year_int(work.get("publication_year"))
    topics = extract_topic_names(work)
    refs = [normalize_openalex_id(r) for r in (work.get("referenced_works") or [])]
    refs = [r for r in refs if r]
    abstract = abstract_from_inverted_index(work.get("abstract_inverted_index")) if include_abstract else ""
    text_parts = [title] + topics
    if abstract:
        text_parts.append(abstract)
    return {
        "id": wid,
        "short_id": short_openalex_id(wid),
        "doi": doi,
        "title": str(title),
        "year": year,
        "date": work.get("publication_date"),
        "type": work.get("type"),
        "language": work.get("language"),
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "fwci": work.get("fwci"),
        "citation_normalized_percentile": (work.get("citation_normalized_percentile") or {}).get("value")
        if isinstance(work.get("citation_normalized_percentile"), Mapping)
        else None,
        "refs": refs,
        "topics": topics,
        "primary_topic": topics[0] if topics else "",
        "text": " ".join([p for p in text_parts if p]),
        "anchor_label": "",
        "anchor_year": None,
        "anchor_citer": bool(work.get("_aspr_anchor_citer")),
        "reference_stub": bool(work.get("_aspr_reference_stub")),
    }


def resolve_anchor_matches(
    works: Mapping[str, Dict[str, Any]],
    anchors: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, List[Mapping[str, Any]]], List[Dict[str, Any]]]:
    doi_to_id = {normalize_doi(w.get("doi")): wid for wid, w in works.items() if w.get("doi")}
    short_to_id = {short_openalex_id(wid): wid for wid in works}
    title_to_id = {
        normalize_text_key(w.get("title")): wid
        for wid, w in works.items()
        if normalize_text_key(w.get("title"))
    }
    arxiv_to_id: Dict[str, str] = {}
    for wid, w in works.items():
        doi = normalize_doi(w.get("doi"))
        if not doi:
            continue
        m = re.search(r"arxiv[./](\d{4}\.\d{4,5})(?:v\d+)?", doi)
        if m:
            arxiv_to_id[m.group(1)] = wid

    matches: Dict[str, List[Mapping[str, Any]]] = collections.defaultdict(list)
    rows: List[Dict[str, Any]] = []
    for a in anchors or []:
        label = str(a.get("label") or a.get("name") or a.get("doi") or a.get("openalex_id") or "landmark")
        matched_id = None
        method = ""
        if a.get("openalex_id"):
            matched_id = short_to_id.get(short_openalex_id(a.get("openalex_id")))
            method = "openalex_id" if matched_id else ""
        if not matched_id and a.get("doi"):
            matched_id = doi_to_id.get(normalize_doi(a.get("doi")))
            method = "doi" if matched_id else ""
        arxiv_id = normalize_arxiv_id(a.get("arxiv_id") or a.get("doi"))
        if not matched_id and arxiv_id:
            matched_id = arxiv_to_id.get(arxiv_id)
            method = "arxiv_id" if matched_id else ""
        if not matched_id and a.get("title"):
            matched_id = title_to_id.get(normalize_text_key(a.get("title")))
            method = "title" if matched_id else ""
        if matched_id:
            matches[matched_id].append(a)
        rows.append(
            {
                "label": label,
                "year": year_int(a.get("year")),
                "openalex_id": a.get("openalex_id", ""),
                "doi": a.get("doi", ""),
                "arxiv_id": a.get("arxiv_id", ""),
                "title": a.get("title", ""),
                "resolved": int(bool(matched_id)),
                "resolved_id": matched_id or "",
                "resolved_title": works.get(matched_id, {}).get("title", "") if matched_id else "",
                "reference_stub": int(bool(works.get(matched_id, {}).get("reference_stub"))) if matched_id else 0,
                "method": method,
            }
        )
    return dict(matches), rows


def write_anchor_resolution_report(
    domain_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    fail_on_unresolved: bool = True,
) -> None:
    report = pd.DataFrame(list(rows))
    if not report.empty:
        report.to_csv(domain_dir / "anchor_resolution_report.csv", index=False)
    unresolved = report[report["resolved"].astype(int) == 0] if not report.empty else pd.DataFrame()
    if fail_on_unresolved and not unresolved.empty:
        labels = ", ".join(str(v) for v in unresolved["label"].tolist())
        raise RuntimeError(
            f"Unresolved landmark anchors for {domain_dir.name}: {labels}. "
            f"See {domain_dir / 'anchor_resolution_report.csv'}."
        )


def mark_anchors(works: Dict[str, Dict[str, Any]], anchors: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    for w in works.values():
        w["anchor_label"] = ""
        w["anchor_year"] = None
    matches, rows = resolve_anchor_matches(works, anchors)
    for matched_id, anchor_items in matches.items():
        labels: List[str] = []
        years: List[int] = []
        for a in anchor_items:
            label = a.get("label") or a.get("name") or a.get("doi") or a.get("openalex_id") or "landmark"
            label = str(label)
            if label not in labels:
                labels.append(label)
            a_year = year_int(a.get("year")) or year_int(works[matched_id].get("year"))
            if a_year is not None:
                years.append(a_year)
        works[matched_id]["anchor_label"] = "; ".join(labels)
        works[matched_id]["anchor_year"] = min(years) if years else works[matched_id].get("year")
    return rows


def make_anchor_reference_stub(anchor: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Create a minimal work record for OpenAlex reference IDs that cannot be dereferenced."""
    openalex_id = normalize_openalex_id(anchor.get("openalex_id"))
    title = str(anchor.get("title") or "").strip()
    year = year_int(anchor.get("year"))
    if not (openalex_id and title and year):
        return None
    doi = normalize_doi(anchor.get("doi"))
    arxiv_id = normalize_arxiv_id(anchor.get("arxiv_id") or anchor.get("doi"))
    if not doi and arxiv_id:
        doi = f"10.48550/arxiv.{arxiv_id}"
    return {
        "id": openalex_id,
        "display_name": title,
        "title": title,
        "doi": f"https://doi.org/{doi}" if doi else None,
        "publication_year": year,
        "publication_date": f"{year}-01-01",
        "type": anchor.get("type") or "preprint",
        "language": anchor.get("language") or "en",
        "cited_by_count": int(anchor.get("cited_by_count") or 0),
        "fwci": None,
        "citation_normalized_percentile": None,
        "referenced_works": [],
        "primary_topic": {"display_name": anchor.get("primary_topic") or "Transformer and foundation models"},
        "topics": [{"display_name": v} for v in (anchor.get("topics") or ["Natural Language Processing Techniques"])],
        "keywords": [{"display_name": v} for v in (anchor.get("keywords") or ["Transformer", "Self-attention"])],
        "_aspr_reference_stub": True,
    }


def apply_anchor_metadata_fallback(rec: Dict[str, Any], anchor: Mapping[str, Any]) -> Dict[str, Any]:
    title = str(rec.get("display_name") or rec.get("title") or "").strip()
    if (not title or title == "Untitled") and anchor.get("title"):
        rec["display_name"] = str(anchor["title"])
        rec["title"] = str(anchor["title"])
    if not rec.get("publication_year") and anchor.get("year"):
        rec["publication_year"] = year_int(anchor.get("year"))
    if not rec.get("publication_date") and anchor.get("year"):
        rec["publication_date"] = f"{year_int(anchor.get('year'))}-01-01"
    if not rec.get("doi") and anchor.get("doi"):
        doi = normalize_doi(anchor.get("doi"))
        rec["doi"] = f"https://doi.org/{doi}" if doi else anchor.get("doi")
    if not rec.get("topics") and anchor.get("topics"):
        rec["topics"] = [{"display_name": v} for v in (anchor.get("topics") or [])]
    if not rec.get("primary_topic") and anchor.get("primary_topic"):
        rec["primary_topic"] = {"display_name": anchor.get("primary_topic")}
    return rec


def fetch_anchor_work(client: OpenAlexClient, anchor: Mapping[str, Any], include_abstract: bool) -> Optional[Dict[str, Any]]:
    rec: Optional[Dict[str, Any]] = None
    if anchor.get("openalex_id"):
        rec = client.get_work_by_openalex_id(str(anchor["openalex_id"]), include_abstract=include_abstract)
        if rec:
            return apply_anchor_metadata_fallback(rec, anchor)
    if anchor.get("doi"):
        rec = client.get_work_by_doi(str(anchor["doi"]), include_abstract=include_abstract)
        if rec:
            return apply_anchor_metadata_fallback(rec, anchor)
    arxiv_id = normalize_arxiv_id(anchor.get("arxiv_id") or anchor.get("doi"))
    if arxiv_id:
        rec = client.get_work_by_doi(f"10.48550/arxiv.{arxiv_id}", include_abstract=include_abstract)
        if rec:
            return apply_anchor_metadata_fallback(rec, anchor)
    if anchor.get("title"):
        rec = client.get_work_by_title(
            str(anchor["title"]),
            year=year_int(anchor.get("year")),
            include_abstract=include_abstract,
        )
        if rec:
            return apply_anchor_metadata_fallback(rec, anchor)
    if anchor.get("allow_reference_stub", False):
        return make_anchor_reference_stub(anchor)
    return None


def ensure_anchor_records(
    works: Dict[str, Dict[str, Any]],
    cfg: Mapping[str, Any],
    client: OpenAlexClient,
) -> List[Dict[str, Any]]:
    anchors = cfg.get("anchors") or []
    rows = mark_anchors(works, anchors)
    unresolved = {str(row.get("label")) for row in rows if not int(row.get("resolved") or 0)}
    if not unresolved:
        return rows

    for anchor in anchors:
        label = str(anchor.get("label") or anchor.get("name") or anchor.get("doi") or anchor.get("openalex_id") or "landmark")
        if label not in unresolved:
            continue
        rec = fetch_anchor_work(client, anchor, include_abstract=bool(cfg.get("include_abstract")))
        if rec and rec.get("id"):
            rec["_aspr_anchor_seed"] = True
            norm = normalize_work(rec, include_abstract=bool(cfg.get("include_abstract")))
            if norm.get("id") and norm.get("year"):
                works[norm["id"]] = norm
    return mark_anchors(works, anchors)


def iter_search_groups(cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    groups = cfg.get("search_groups") or []
    if groups:
        out = []
        for group in groups:
            if isinstance(group, str):
                out.append({"name": group[:32], "query": group})
            elif isinstance(group, Mapping) and group.get("query"):
                out.append(dict(group))
        return out
    return [{"name": "default", "query": cfg.get("search_query", ""), "max_works_per_window": cfg.get("max_works_per_window")}]


def work_matches_relevance_filter(work: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    fcfg = cfg.get("relevance_filter") or {}
    if not fcfg.get("enabled", False):
        return True
    if work.get("anchor_label"):
        return True
    if bool(fcfg.get("keep_anchor_citers", True)) and work.get("anchor_citer"):
        return True
    text = normalize_text_key(
        " ".join(
            [
                str(work.get("title") or ""),
                str(work.get("primary_topic") or ""),
                " ".join(str(t) for t in (work.get("topics") or [])),
            ]
        )
    )
    topic_text = normalize_text_key(str(work.get("primary_topic") or ""))
    positives = [normalize_text_key(v) for v in fcfg.get("positive_keywords") or []]
    strong_positives = [normalize_text_key(v) for v in fcfg.get("strong_positive_keywords") or []]
    negatives = [normalize_text_key(v) for v in fcfg.get("negative_keywords") or []]
    negative_topics = [normalize_text_key(v) for v in fcfg.get("negative_primary_topics") or []]
    has_positive = not positives or any(p and p in text for p in positives)
    has_strong_positive = any(p and p in text for p in strong_positives)
    if has_strong_positive:
        return True
    has_negative = any(n and n in text for n in negatives)
    has_negative_topic = any(n and n in topic_text for n in negative_topics)
    return bool(has_positive and not has_negative and not has_negative_topic)


def apply_relevance_filter(works: Dict[str, Dict[str, Any]], cfg: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    if not (cfg.get("relevance_filter") or {}).get("enabled", False):
        return works
    return {wid: w for wid, w in works.items() if work_matches_relevance_filter(w, cfg)}


def mark_anchors_legacy(works: Dict[str, Dict[str, Any]], anchors: Sequence[Mapping[str, Any]]) -> None:
    doi_to_id = {normalize_doi(w.get("doi")): wid for wid, w in works.items() if w.get("doi")}
    short_to_id = {short_openalex_id(wid): wid for wid in works}
    for a in anchors or []:
        label = a.get("label") or a.get("name") or a.get("doi") or a.get("openalex_id") or "landmark"
        a_year = year_int(a.get("year"))
        matched_id = None
        if a.get("doi"):
            matched_id = doi_to_id.get(normalize_doi(a.get("doi")))
        if not matched_id and a.get("openalex_id"):
            matched_id = short_to_id.get(short_openalex_id(a.get("openalex_id")))
        if matched_id and matched_id in works:
            prev = works[matched_id].get("anchor_label")
            works[matched_id]["anchor_label"] = f"{prev}; {label}" if prev else str(label)
            works[matched_id]["anchor_year"] = a_year or works[matched_id].get("year")


def fetch_domain_works(
    cfg: Dict[str, Any],
    client: OpenAlexClient,
    out_dir: Path,
    use_cache: bool = True,
    force_cache: bool = False,
) -> Dict[str, Dict[str, Any]]:
    slug = cfg["slug"]
    domain_dir = out_dir / slug
    domain_dir.mkdir(parents=True, exist_ok=True)
    cache_path = domain_dir / "works_raw.jsonl"
    manifest_path = domain_dir / "cache_manifest.json"

    if use_cache and cache_path.exists():
        if force_cache or cache_matches_config(manifest_path, cfg):
            works: Dict[str, Dict[str, Any]] = {}
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("id"):
                        works[rec["id"]] = rec
            report_rows = ensure_anchor_records(works, cfg, client)
            works = apply_relevance_filter(works, cfg)
            report_rows = ensure_anchor_records(works, cfg, client)
            write_anchor_resolution_report(domain_dir, report_rows)
            print(f"[{slug}] Loaded {len(works):,} cached works from {cache_path}")
            return works
        print(f"[{slug}] Cache manifest mismatch; re-downloading works. Use --force-cache to override.")

    rolling = make_rolling_windows_from_config(cfg)
    raw_by_id: Dict[str, Dict[str, Any]] = {}
    api_cfg = cfg.get("api", {})
    search_groups = iter_search_groups(cfg)

    for y0, y1 in rolling:
        for group in search_groups:
            query = str(group.get("query") or "").strip()
            if not query:
                continue
            group_name = str(group.get("name") or "search")
            max_records = int(group.get("max_works_per_window") or cfg.get("max_works_per_window", 1200))
            sort = group.get("sort")
            print(f"[{slug}] Fetching {group_name} window {y0}-{y1} ...")
            records = client.list_works(
                search_query=query,
                from_year=y0,
                to_year=y1,
                max_records=max_records,
                work_types=cfg.get("work_types") or [],
                language=cfg.get("language"),
                include_abstract=bool(cfg.get("include_abstract")),
                sort=str(sort) if sort else None,
                per_page=int(api_cfg.get("per_page", 100)),
            )
            for r in records:
                wid = normalize_openalex_id(r.get("id"))
                if wid:
                    raw_by_id[wid] = r

    # Always add known landmark papers, even if the keyword query misses them.
    for a in cfg.get("anchors") or []:
        label = a.get("label") or a.get("title") or a.get("doi") or a.get("openalex_id")
        print(f"[{slug}] Fetching anchor {label} ...")
        rec = fetch_anchor_work(client, a, include_abstract=bool(cfg.get("include_abstract")))
        if rec and rec.get("id"):
            rec["_aspr_anchor_seed"] = True
            raw_by_id[normalize_openalex_id(rec["id"])] = rec

    # Optional: fetch citing papers of anchors to better capture downstream disturbance.
    if cfg.get("fetch_anchor_citers", True) and cfg.get("anchors"):
        anchor_records: List[Dict[str, Any]] = []
        for a in cfg.get("anchors") or []:
            rec = fetch_anchor_work(client, a, include_abstract=False)
            if rec and rec.get("id"):
                anchor_records.append(rec)
        for rec in anchor_records:
            sid = short_openalex_id(rec.get("id"))
            if not sid:
                continue
            print(f"[{slug}] Fetching top papers that cite anchor {sid} ...")
            citers = client.list_works(
                search_query=None,
                from_year=cfg["start_year"],
                to_year=cfg["end_year"],
                max_records=int(cfg.get("max_anchor_citers", 500)),
                work_types=cfg.get("work_types") or [],
                language=cfg.get("language"),
                include_abstract=bool(cfg.get("include_abstract")),
                extra_filters=[f"cites:{sid}"],
                sort="cited_by_count:desc",
                per_page=int(api_cfg.get("per_page", 100)),
            )
            for r in citers:
                wid = normalize_openalex_id(r.get("id"))
                if wid:
                    r["_aspr_anchor_citer"] = True
                    raw_by_id[wid] = r

    works: Dict[str, Dict[str, Any]] = {}
    for raw in raw_by_id.values():
        rec = normalize_work(raw, include_abstract=bool(cfg.get("include_abstract")))
        if rec.get("id") and rec.get("year"):
            works[rec["id"]] = rec

    report_rows = mark_anchors(works, cfg.get("anchors") or [])
    works = apply_relevance_filter(works, cfg)
    report_rows = mark_anchors(works, cfg.get("anchors") or [])
    write_anchor_resolution_report(domain_dir, report_rows)
    with open(cache_path, "w", encoding="utf-8") as f:
        for rec in works.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    manifest_path.write_text(json.dumps(cache_manifest(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{slug}] Saved {len(works):,} works to {cache_path}")
    return works


# -----------------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------------


def select_balanced_papers(
    works: Mapping[str, Dict[str, Any]],
    cfg: Dict[str, Any],
    rolling_windows: Sequence[Tuple[int, int]],
) -> Dict[str, Dict[str, Any]]:
    max_n = int(cfg["graph"].get("max_papers_for_graph", 3500))
    if len(works) <= max_n:
        return dict(works)

    selected: Dict[str, Dict[str, Any]] = {}
    quota = max(1, max_n // max(1, len(rolling_windows)))

    def score(w: Mapping[str, Any]) -> float:
        return safe_log1p(float(w.get("cited_by_count") or 0)) + (100.0 if w.get("anchor_label") else 0.0)

    for y0, y1 in rolling_windows:
        candidates = [w for w in works.values() if in_year_range(w.get("year"), y0, y1)]
        candidates.sort(key=score, reverse=True)
        for w in candidates[:quota]:
            selected[w["id"]] = w

    # Ensure anchors are always present.
    for w in works.values():
        if w.get("anchor_label"):
            selected[w["id"]] = w

    # Fill remaining slots with globally high-impact works.
    if len(selected) < max_n:
        rest = [w for w in works.values() if w["id"] not in selected]
        rest.sort(key=score, reverse=True)
        for w in rest[: max_n - len(selected)]:
            selected[w["id"]] = w

    return selected


def add_weighted_edge(G: nx.Graph, u: str, v: str, weight: float, edge_type: str, count: int = 1) -> None:
    if u == v or weight <= 0:
        return
    if G.has_edge(u, v):
        G[u][v]["weight"] += float(weight)
        G[u][v][edge_type] = G[u][v].get(edge_type, 0) + count
    else:
        G.add_edge(u, v, weight=float(weight), **{edge_type: count})


def prune_graph_edges(G: nx.Graph, max_edges: int) -> nx.Graph:
    if max_edges <= 0 or G.number_of_edges() <= max_edges:
        return G
    edges = sorted(G.edges(data=True), key=lambda x: x[2].get("weight", 1.0), reverse=True)
    keep = set(tuple(sorted((u, v))) for u, v, _ in edges[:max_edges])
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        if tuple(sorted((u, v))) in keep:
            H.add_edge(u, v, **d)
    return H


def _hybrid_edge_type(data: Mapping[str, Any]) -> str:
    """Return the primary non-direct edge family for deterministic sampling."""
    if float(data.get("direct", 0) or 0) > 0:
        return "direct"
    weights = {
        "bibliographic": float(data.get("bibliographic", 0) or 0),
        "cocitation": float(data.get("cocitation", 0) or 0),
    }
    best = max(weights.items(), key=lambda item: item[1])
    return best[0] if best[1] > 0 else "hybrid"


def _edge_year(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def apply_deterministic_hybrid_edge_sampling(G: nx.Graph, gcfg: Mapping[str, Any]) -> nx.Graph:
    """Sample hybrid edges reproducibly while preserving all direct citation edges.

    The sampling target is intentionally separate from ``max_edges``: hitting the
    deterministic target is not treated as an uncontrolled edge-cap truncation.
    """
    enabled = bool(gcfg.get("deterministic_hybrid_sampling", False))
    target_edges = int(gcfg.get("sampling_target_edges", gcfg.get("max_edges", 0)) or 0)
    seed = int(gcfg.get("sampling_seed", gcfg.get("random_seed", 42)) or 42)
    if not enabled or target_edges <= 0 or G.number_of_edges() <= target_edges:
        G.graph["edge_sampling"] = {
            "sampling_applied": 0,
            "sampling_seed": seed,
            "sampling_target_edges": target_edges,
            "raw_edges": int(G.number_of_edges()),
            "exported_edges": int(G.number_of_edges()),
            "direct_edges_preserved": int(sum(1 for _, _, data in G.edges(data=True) if float(data.get("direct", 0) or 0) > 0)),
            "hybrid_raw_edges": int(sum(1 for _, _, data in G.edges(data=True) if float(data.get("direct", 0) or 0) <= 0)),
            "hybrid_exported_edges": int(sum(1 for _, _, data in G.edges(data=True) if float(data.get("direct", 0) or 0) <= 0)),
            "hybrid_sampling_fraction": 1.0,
        }
        return G

    direct_edges: List[Tuple[str, str, Dict[str, Any]]] = []
    hybrid_edges: List[Tuple[str, str, Dict[str, Any]]] = []
    strata: Dict[Tuple[str, int, int], List[Tuple[str, str, Dict[str, Any]]]] = collections.defaultdict(list)
    for u, v, data in G.edges(data=True):
        row = (str(u), str(v), dict(data))
        if float(data.get("direct", 0) or 0) > 0:
            direct_edges.append(row)
            continue
        source_year = _edge_year(G.nodes[u].get("year"))
        target_year = _edge_year(G.nodes[v].get("year"))
        year_a, year_b = sorted((source_year, target_year))
        key = (_hybrid_edge_type(data), year_a, year_b)
        strata[key].append(row)
        hybrid_edges.append(row)

    hybrid_budget = max(0, target_edges - len(direct_edges))
    selected_hybrid: List[Tuple[str, str, Dict[str, Any]]] = []
    if hybrid_budget >= len(hybrid_edges):
        selected_hybrid = list(hybrid_edges)
    elif hybrid_budget > 0 and hybrid_edges:
        stratum_items = sorted(strata.items(), key=lambda item: (item[0][0], item[0][1], item[0][2]))
        allocations: Dict[Tuple[str, int, int], int] = {}
        remainders: List[Tuple[float, Tuple[str, int, int]]] = []
        for key, edges in stratum_items:
            exact = hybrid_budget * (len(edges) / len(hybrid_edges))
            base = min(len(edges), int(math.floor(exact)))
            allocations[key] = base
            remainders.append((exact - base, key))
        remaining = hybrid_budget - sum(allocations.values())
        for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
            if remaining <= 0:
                break
            if allocations[key] < len(strata[key]):
                allocations[key] += 1
                remaining -= 1
        for key, edges in stratum_items:
            take = allocations.get(key, 0)
            ranked = sorted(
                edges,
                key=lambda edge: hashlib.sha1(f"{seed}|{key}|{edge[0]}|{edge[1]}".encode("utf-8")).hexdigest(),
            )
            selected_hybrid.extend(ranked[:take])

    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, data in direct_edges + selected_hybrid:
        H.add_edge(u, v, **data)
    hybrid_fraction = float(len(selected_hybrid) / len(hybrid_edges)) if hybrid_edges else 1.0
    H.graph.update(G.graph)
    H.graph["edge_sampling"] = {
        "sampling_applied": 1,
        "sampling_seed": seed,
        "sampling_target_edges": target_edges,
        "raw_edges": int(G.number_of_edges()),
        "exported_edges": int(H.number_of_edges()),
        "direct_edges_preserved": int(len(direct_edges)),
        "hybrid_raw_edges": int(len(hybrid_edges)),
        "hybrid_exported_edges": int(len(selected_hybrid)),
        "hybrid_sampling_fraction": hybrid_fraction,
    }
    return H


def build_edge_sampling_manifest_row(
    domain: str,
    raw_graph: nx.Graph,
    sampled_graph: nx.Graph,
    gcfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build one row for the Fig.1 deterministic edge-sampling manifest."""
    sampling = dict(sampled_graph.graph.get("edge_sampling") or {})
    if not sampling:
        raw_edges = int(raw_graph.number_of_edges())
        exported_edges = int(sampled_graph.number_of_edges())
        hybrid_raw = int(sum(1 for _, _, data in raw_graph.edges(data=True) if float(data.get("direct", 0) or 0) <= 0))
        hybrid_exported = int(sum(1 for _, _, data in sampled_graph.edges(data=True) if float(data.get("direct", 0) or 0) <= 0))
        sampling = {
            "sampling_applied": int(exported_edges < raw_edges),
            "sampling_seed": int(gcfg.get("sampling_seed", gcfg.get("random_seed", 42)) or 42),
            "sampling_target_edges": int(gcfg.get("sampling_target_edges", gcfg.get("max_edges", 0)) or 0),
            "raw_edges": raw_edges,
            "exported_edges": exported_edges,
            "direct_edges_preserved": int(sum(1 for _, _, data in sampled_graph.edges(data=True) if float(data.get("direct", 0) or 0) > 0)),
            "hybrid_raw_edges": hybrid_raw,
            "hybrid_exported_edges": hybrid_exported,
            "hybrid_sampling_fraction": float(hybrid_exported / hybrid_raw) if hybrid_raw else 1.0,
        }
    return {
        "domain": domain,
        "sampling_applied": int(sampling.get("sampling_applied", 0)),
        "sampling_seed": int(sampling.get("sampling_seed", 0) or 0),
        "sampling_target_edges": int(sampling.get("sampling_target_edges", 0) or 0),
        "graph_max_edges": int(gcfg.get("max_edges", 0) or 0),
        "raw_edges": int(sampling.get("raw_edges", raw_graph.number_of_edges()) or 0),
        "exported_edges": int(sampling.get("exported_edges", sampled_graph.number_of_edges()) or 0),
        "direct_edges_preserved": int(sampling.get("direct_edges_preserved", 0) or 0),
        "hybrid_raw_edges": int(sampling.get("hybrid_raw_edges", 0) or 0),
        "hybrid_exported_edges": int(sampling.get("hybrid_exported_edges", 0) or 0),
        "hybrid_sampling_fraction": float(sampling.get("hybrid_sampling_fraction", 1.0) or 0.0),
    }


def build_hybrid_graph(
    selected_works: Mapping[str, Dict[str, Any]],
    all_works: Mapping[str, Dict[str, Any]],
    cfg: Dict[str, Any],
) -> nx.Graph:
    gcfg = cfg["graph"]
    node_ids = set(selected_works.keys())
    G = nx.Graph()
    for wid, w in selected_works.items():
        G.add_node(
            wid,
            title=w.get("title", ""),
            year=w.get("year"),
            cited_by_count=w.get("cited_by_count", 0),
            primary_topic=w.get("primary_topic", ""),
            topics=w.get("topics", []),
            text=w.get("text", ""),
            doi=w.get("doi", ""),
            anchor_label=w.get("anchor_label", ""),
            anchor_year=w.get("anchor_year"),
            anchor_citer=bool(w.get("anchor_citer")),
            reference_stub=bool(w.get("reference_stub")),
        )

    # Direct citation edges among selected papers.
    if gcfg.get("use_direct_citation", True):
        for wid, w in tqdm(selected_works.items(), desc="Direct citations", leave=False):
            for ref in set(w.get("refs") or []):
                if ref in node_ids:
                    add_weighted_edge(G, wid, ref, float(gcfg.get("direct_weight", 3.0)), "direct", count=1)

    # Bibliographic coupling: papers sharing references, including references not
    # selected as nodes. This captures proximity even when cited classics are not
    # in the displayed graph.
    if gcfg.get("use_bibliographic_coupling", True):
        ref_to_sources: Dict[str, List[str]] = collections.defaultdict(list)
        for wid, w in selected_works.items():
            for ref in set(w.get("refs") or []):
                ref_to_sources[ref].append(wid)

        pair_counts: Dict[Tuple[str, str], int] = collections.defaultdict(int)
        max_fanout = int(gcfg.get("max_reference_fanout", 80))
        for sources in tqdm(ref_to_sources.values(), desc="Bibliographic coupling", leave=False):
            if len(sources) < 2 or len(sources) > max_fanout:
                continue
            sources = sorted(set(sources))
            for u, v in itertools.combinations(sources, 2):
                pair_counts[(u, v)] += 1

        min_shared = int(gcfg.get("min_shared_references", 2))
        bib_weight = float(gcfg.get("bibliographic_weight", 1.0))
        for (u, v), c in tqdm(pair_counts.items(), desc="Add coupling edges", leave=False):
            if c >= min_shared:
                add_weighted_edge(G, u, v, bib_weight * math.log1p(c), "bibliographic", count=c)

    # Co-citation: selected papers jointly cited by the same paper in the fetched
    # corpus. Here all fetched works can act as citing papers.
    if gcfg.get("use_cocitation", True):
        pair_counts: Dict[Tuple[str, str], int] = collections.defaultdict(int)
        max_refs = int(gcfg.get("max_cocited_refs_per_paper", 80))
        for w in tqdm(all_works.values(), desc="Co-citation", leave=False):
            refs = sorted(set(r for r in (w.get("refs") or []) if r in node_ids))
            if len(refs) < 2 or len(refs) > max_refs:
                continue
            for u, v in itertools.combinations(refs, 2):
                pair_counts[(u, v)] += 1
        min_coc = int(gcfg.get("min_cocitations", 2))
        coc_weight = float(gcfg.get("cocitation_weight", 0.8))
        for (u, v), c in tqdm(pair_counts.items(), desc="Add co-citation edges", leave=False):
            if c >= min_coc:
                add_weighted_edge(G, u, v, coc_weight * math.log1p(c), "cocitation", count=c)

    raw_graph = G.copy()
    if bool(gcfg.get("deterministic_hybrid_sampling", False)):
        G = apply_deterministic_hybrid_edge_sampling(G, gcfg)
        G.graph["edge_sampling_manifest"] = build_edge_sampling_manifest_row(
            str(cfg.get("slug", "")),
            raw_graph,
            G,
            gcfg,
        )
    else:
        G = prune_graph_edges(G, int(gcfg.get("max_edges", 120000)))
        G.graph["edge_sampling_manifest"] = build_edge_sampling_manifest_row(
            str(cfg.get("slug", "")),
            raw_graph,
            G,
            gcfg,
        )
    return G


# -----------------------------------------------------------------------------
# Communities and topic graph
# -----------------------------------------------------------------------------


def detect_communities(G: nx.Graph, cfg: Dict[str, Any], compact: bool = False) -> Dict[str, int]:
    gcfg = cfg["graph"]
    seed = int(gcfg.get("random_seed", 42))
    resolution = float(gcfg.get("community_resolution", 1.0))

    if G.number_of_nodes() == 0:
        return {}
    if G.number_of_edges() == 0:
        return {n: i for i, n in enumerate(G.nodes())}

    try:
        communities = nx.community.louvain_communities(
            G,
            weight="weight",
            resolution=resolution,
            seed=seed,
        )
    except Exception:
        communities = nx.community.greedy_modularity_communities(G, weight="weight")

    comm_map: Dict[str, int] = {}
    for i, comm in enumerate(communities):
        for n in comm:
            comm_map[n] = i

    if compact:
        comm_map = compact_communities(G, comm_map, cfg)
    return comm_map


def compact_communities(G: nx.Graph, comm_map: Mapping[str, int], cfg: Dict[str, Any]) -> Dict[str, int]:
    gcfg = cfg["graph"]
    max_comms = int(gcfg.get("max_communities", 12))
    min_size = int(gcfg.get("min_community_size", 8))

    members: Dict[int, List[str]] = collections.defaultdict(list)
    for n, c in comm_map.items():
        members[int(c)].append(n)

    anchor_comms = {
        c for n, c in comm_map.items() if G.nodes[n].get("anchor_label")
    }
    ranked = sorted(members, key=lambda c: (len(members[c]), c), reverse=True)
    keep: List[int] = []
    for c in ranked:
        if len(members[c]) >= min_size or c in anchor_comms:
            keep.append(c)
        if len(keep) >= max_comms:
            break
    # Make sure anchor communities survive when possible.
    for c in anchor_comms:
        if c not in keep:
            if len(keep) < max_comms:
                keep.append(c)
            else:
                keep[-1] = c

    keep = list(dict.fromkeys(keep))
    remap = {old: new for new, old in enumerate(keep)}
    other_id = len(remap)
    out: Dict[str, int] = {}
    for n, old in comm_map.items():
        out[n] = remap.get(old, other_id)
    return out


def community_members(comm_map: Mapping[str, int]) -> Dict[int, List[str]]:
    members: Dict[int, List[str]] = collections.defaultdict(list)
    for n, c in comm_map.items():
        members[int(c)].append(n)
    return dict(members)


def top_terms_from_texts(texts: Sequence[str], n: int = 3) -> List[str]:
    cleaned = [t for t in texts if t and len(t.split()) > 2]
    if not cleaned:
        return []
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=3000,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b",
        )
        X = vectorizer.fit_transform(cleaned)
        scores = np.asarray(X.sum(axis=0)).ravel()
        terms = np.array(vectorizer.get_feature_names_out())
        order = np.argsort(scores)[::-1]
        out: List[str] = []
        for idx in order:
            term = terms[idx]
            parts = term.lower().split()
            if any(p in STOPWORDS_EXTRA for p in parts):
                continue
            out.append(term)
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


def make_community_labels(G: nx.Graph, comm_map: Mapping[str, int]) -> Dict[int, str]:
    members = community_members(comm_map)
    labels: Dict[int, str] = {}
    used: Set[str] = set()
    for c, nodes in sorted(members.items(), key=lambda kv: len(kv[1]), reverse=True):
        texts: List[str] = []
        topic_counts: collections.Counter[str] = collections.Counter()
        for n in nodes:
            data = G.nodes[n]
            title = str(data.get("title") or "")
            if title:
                texts.append(title)
            if data.get("text"):
                texts.append(str(data.get("text"))[:600])
            for t in data.get("topics", []) or []:
                if t and len(str(t)) > 4:
                    topic_counts[str(t)] += 1
        # Prefer TF-IDF terms because OpenAlex topic labels are often repetitive.
        terms = top_terms_from_texts(texts, n=4)
        chosen = []
        for term in terms:
            cl = clean_label(term.title(), 28)
            if cl.lower() not in used:
                chosen.append(cl)
            if len(chosen) >= 2:
                break
        if not chosen:
            for topic, _ in topic_counts.most_common(5):
                cl = clean_label(topic, 28)
                if cl.lower() not in used:
                    chosen.append(cl)
                if len(chosen) >= 2:
                    break
        if not chosen:
            chosen = [f"Topic {c + 1}"]
        label = " / ".join(chosen)
        labels[c] = clean_label(label, max_len=34)
        used.add(labels[c].lower())
    return labels


def make_topic_graph(
    G: nx.Graph,
    comm_map: Mapping[str, int],
    labels: Mapping[int, str],
    active_nodes: Optional[Set[str]] = None,
) -> nx.Graph:
    if active_nodes is None:
        active_nodes = set(G.nodes())
    else:
        active_nodes = set(active_nodes)

    TG = nx.Graph()
    comm_nodes: Dict[int, List[str]] = collections.defaultdict(list)
    for n in active_nodes:
        if n in comm_map and n in G:
            comm_nodes[int(comm_map[n])].append(n)

    for c, nodes in comm_nodes.items():
        years = [G.nodes[n].get("year") for n in nodes if G.nodes[n].get("year")]
        cited = [int(G.nodes[n].get("cited_by_count") or 0) for n in nodes]
        anchors = [G.nodes[n].get("anchor_label") for n in nodes if G.nodes[n].get("anchor_label")]
        anchor_years = [G.nodes[n].get("anchor_year") or G.nodes[n].get("year") for n in nodes if G.nodes[n].get("anchor_label")]
        topic_counter: collections.Counter[str] = collections.Counter()
        title_texts: List[str] = []
        for n in nodes:
            data = G.nodes[n]
            title_texts.append(str(data.get("title") or ""))
            if data.get("primary_topic"):
                topic_counter[str(data.get("primary_topic"))] += 2
            for t in data.get("topics", []) or []:
                topic_counter[str(t)] += 1
        semantic_terms = [labels.get(c, f"Topic {c + 1}")]
        semantic_terms.extend([t for t, _ in topic_counter.most_common(12)])
        semantic_terms.extend(top_terms_from_texts(title_texts, n=8))
        TG.add_node(
            c,
            label=labels.get(c, f"Topic {c + 1}"),
            display_label=labels.get(c, f"Topic {c + 1}"),
            n_papers=len(nodes),
            cited_by_count=sum(cited),
            first_year=min(years) if years else None,
            anchor_labels="; ".join(anchors),
            anchor_year=min([y for y in anchor_years if y] or [None]) if anchor_years else None,
            member_ids=nodes,
            semantic_text=" ".join(semantic_terms).lower(),
        )

    active = active_nodes
    for u, v, d in G.edges(data=True):
        if u not in active or v not in active:
            continue
        cu, cv = comm_map.get(u), comm_map.get(v)
        if cu is None or cv is None or cu == cv:
            continue
        w = float(d.get("weight", 1.0))
        if TG.has_edge(cu, cv):
            TG[cu][cv]["weight"] += w
            TG[cu][cv]["n_edges"] += 1
        else:
            TG.add_edge(cu, cv, weight=w, n_edges=1)
    return TG


def layout_topic_graph(TG: nx.Graph, cfg: Dict[str, Any]) -> Dict[int, np.ndarray]:
    seed = int(cfg["graph"].get("random_seed", 42))
    if TG.number_of_nodes() == 0:
        return {}
    if TG.number_of_nodes() == 1:
        n = next(iter(TG.nodes()))
        return {n: np.array([0.0, 0.0])}

    H = TG.copy()
    for u, v, d in H.edges(data=True):
        # Use log-compressed weights; otherwise one very strong relation collapses the graph.
        d["distance"] = 1.0 / max(math.log1p(float(d.get("weight", 1.0))), 1e-6)
    try:
        auto_pos = nx.kamada_kawai_layout(H, weight="distance")
    except Exception:
        auto_pos = nx.spring_layout(H, weight="weight", seed=seed, k=1.4 / np.sqrt(max(2, H.number_of_nodes())), iterations=600)
    auto_pos = normalize_positions({int(k): np.asarray(v, dtype=float) for k, v in auto_pos.items()})

    templates = display_templates(cfg)
    if not templates:
        sizes = {int(n): 0.07 + 0.11 * np.sqrt(float(TG.nodes[n].get("n_papers", 1)) / max(1.0, max(TG.nodes[m].get("n_papers", 1) for m in TG.nodes()))) for n in TG.nodes()}
        return repel_positions(dict(auto_pos), sizes, iterations=400, step=0.035)

    # Greedily assign the best matching community to each semantic slot.
    used_nodes: Set[int] = set()
    pos: Dict[int, np.ndarray] = {}
    for tmpl in templates:
        candidates = []
        for n in TG.nodes():
            if int(n) in used_nodes:
                continue
            text = f"{TG.nodes[n].get('label', '')} {TG.nodes[n].get('semantic_text', '')}"
            sc = score_keyword_match(text, tmpl.get("keywords", []))
            if sc > 0:
                candidates.append((sc, float(TG.nodes[n].get("n_papers", 1)), int(n)))
        if not candidates:
            continue
        candidates.sort(reverse=True)
        n = candidates[0][2]
        used_nodes.add(n)
        pos[n] = np.array([float(tmpl["x"]), float(tmpl["y"])], dtype=float)
        TG.nodes[n]["display_label"] = tmpl.get("name", TG.nodes[n].get("label"))
        TG.nodes[n]["layout_template"] = tmpl.get("name", "")

    # Remaining communities keep their automatic layout but are scaled into the open space.
    remaining = [int(n) for n in TG.nodes() if int(n) not in pos]
    if remaining:
        rem_auto = {n: np.asarray(auto_pos.get(n, np.array([0.0, 0.0])), dtype=float) for n in remaining}
        rem_auto = normalize_positions(rem_auto)
        # Put leftovers around a lower/central ring rather than far outside.
        for j, n in enumerate(remaining):
            base = rem_auto[n]
            if len(remaining) > 1:
                angle = 2 * math.pi * j / len(remaining) + 0.35
                ring = np.array([0.78 * math.cos(angle), 0.58 * math.sin(angle)])
                p = 0.40 * base + 0.60 * ring
            else:
                p = 0.35 * base + np.array([0.0, -0.65])
            pos[n] = p

    # Mildly blend template positions with graph positions so that heavily connected
    # communities do not look totally arbitrary.
    for n in list(pos):
        if n in auto_pos and TG.nodes[n].get("layout_template"):
            pos[n] = 0.88 * pos[n] + 0.12 * auto_pos[n]

    sizes = {int(n): 0.10 + 0.10 * np.sqrt(float(TG.nodes[n].get("n_papers", 1)) / max(1.0, max(TG.nodes[m].get("n_papers", 1) for m in TG.nodes()))) for n in TG.nodes()}
    pos = repel_positions(pos, sizes, iterations=260, step=0.018)
    return pos


# -----------------------------------------------------------------------------
# Perturbation metrics
# -----------------------------------------------------------------------------


def edge_key_set(G: nx.Graph) -> Set[Tuple[str, str]]:
    return {tuple(sorted((u, v))) for u, v in G.edges()}


def node_set_for_years(G: nx.Graph, start: int, end: int) -> Set[str]:
    return {n for n, d in G.nodes(data=True) if in_year_range(d.get("year"), start, end)}


def node_set_until_year(G: nx.Graph, end: int) -> Set[str]:
    return {n for n, d in G.nodes(data=True) if d.get("year") is not None and int(d.get("year")) <= end}


def weighted_degree_hub_concentration(G: nx.Graph, top_frac: float = 0.10) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    degrees = np.array([d for _, d in G.degree(weight="weight")], dtype=float)
    total = float(degrees.sum())
    if total <= 0:
        return 0.0
    k = max(1, int(math.ceil(len(degrees) * top_frac)))
    return float(np.sort(degrees)[-k:].sum() / total)


def avg_shortest_path_topic(TG: nx.Graph) -> float:
    if TG.number_of_nodes() <= 1 or TG.number_of_edges() == 0:
        return np.nan
    H = TG.copy()
    for u, v, d in H.edges(data=True):
        d["distance"] = 1.0 / max(float(d.get("weight", 1.0)), 1e-9)
    components = list(nx.connected_components(H))
    if not components:
        return np.nan
    largest = max(components, key=len)
    if len(largest) <= 1:
        return np.nan
    return float(nx.average_shortest_path_length(H.subgraph(largest), weight="distance"))


def partition_modularity(G: nx.Graph, comm_map: Mapping[str, int]) -> float:
    if G.number_of_edges() == 0 or G.number_of_nodes() == 0:
        return 0.0
    groups = list(community_members(comm_map).values())
    if len(groups) <= 1:
        return 0.0
    try:
        return float(nx.community.modularity(G, [set(g) for g in groups], weight="weight"))
    except Exception:
        return 0.0


def participation_coefficient(G: nx.Graph, comm_map: Mapping[str, int], node: str) -> float:
    if node not in G:
        return 0.0
    totals: Dict[int, float] = collections.defaultdict(float)
    total = 0.0
    for nbr, d in G[node].items():
        w = float(d.get("weight", 1.0))
        total += w
        totals[int(comm_map.get(nbr, -1))] += w
    if total <= 0:
        return 0.0
    return float(1.0 - sum((v / total) ** 2 for v in totals.values()))


def build_tfidf_matrix(G: nx.Graph) -> Tuple[List[str], Any]:
    node_ids = list(G.nodes())
    texts = [str(G.nodes[n].get("text") or G.nodes[n].get("title") or "") for n in node_ids]
    if not texts or all(not t.strip() for t in texts):
        return node_ids, None
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=2 if len(texts) > 20 else 1,
            max_features=6000,
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b",
        )
        X = vectorizer.fit_transform(texts)
        return node_ids, X
    except Exception:
        return node_ids, None


def semantic_dispersion(
    active_nodes: Sequence[str],
    node_to_idx: Mapping[str, int],
    X: Any,
    sample_size: int,
    seed: int,
) -> float:
    if X is None:
        return 0.0
    ids = [n for n in active_nodes if n in node_to_idx]
    if len(ids) < 3:
        return 0.0
    rng = random.Random(seed)
    if len(ids) > sample_size:
        ids = rng.sample(ids, sample_size)
    idx = [node_to_idx[n] for n in ids]
    S = cosine_similarity(X[idx])
    tri = S[np.triu_indices_from(S, k=1)]
    if len(tri) == 0:
        return 0.0
    return float(1.0 - np.mean(tri))


def primary_topic_set(G: nx.Graph, nodes: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for n in nodes:
        if n not in G:
            continue
        topic = str(G.nodes[n].get("primary_topic") or "").strip().lower()
        if topic:
            out.add(topic)
    return out


def shannon_entropy(values: Iterable[str]) -> float:
    items = [str(v).strip().lower() for v in values if str(v).strip()]
    if not items:
        return 0.0
    counts = collections.Counter(items)
    total = float(sum(counts.values()))
    return float(-sum((c / total) * math.log(max(c / total, 1e-12)) for c in counts.values()))


def edge_weight_sum(G: nx.Graph, edges: Iterable[Tuple[str, str]]) -> float:
    total = 0.0
    for u, v in edges:
        if G.has_edge(u, v):
            total += float(G[u][v].get("weight", 1.0))
    return total


def compute_perturbation_metrics(
    G: nx.Graph,
    global_comm_map: Mapping[str, int],
    labels: Mapping[int, str],
    cfg: Dict[str, Any],
    rolling_windows: Sequence[Tuple[int, int]],
    cumulative_windows: Sequence[Tuple[int, int]],
) -> pd.DataFrame:
    seed = int(cfg["graph"].get("random_seed", 42))
    bet_k = int(cfg["metrics"].get("betweenness_sample", 250))
    sem_sample = int(cfg["metrics"].get("semantic_sample", 600))

    node_order, X = build_tfidf_matrix(G)
    node_to_idx = {n: i for i, n in enumerate(node_order)}

    rows: List[Dict[str, Any]] = []
    prev_cum_nodes: Set[str] = set()
    prev_cum_edges: Set[Tuple[str, str]] = set()
    prev_partition: Dict[str, int] = {}
    prev_modularity = 0.0
    prev_path = np.nan
    prev_hub = 0.0
    prev_sem = 0.0
    prev_field_entropy = 0.0
    prev_paper_rate = 0.0
    seen_topics: Set[str] = set()

    for i, ((r0, r1), (c0, c1)) in enumerate(zip(rolling_windows, cumulative_windows)):
        roll_nodes = node_set_for_years(G, r0, r1)
        cum_nodes = node_set_until_year(G, c1)
        Gcum = G.subgraph(cum_nodes).copy()
        Groom = G.subgraph(roll_nodes).copy()
        curr_edges = edge_key_set(Gcum)

        # Expansion: new nodes, new edges, new topic labels.
        new_nodes = cum_nodes - prev_cum_nodes
        new_edges = curr_edges - prev_cum_edges
        curr_topics = primary_topic_set(G, roll_nodes)
        new_topics = curr_topics - seen_topics
        new_edge_weight = edge_weight_sum(G, new_edges)
        prev_edge_weight = edge_weight_sum(G, prev_cum_edges)
        new_cross_edges = {
            (u, v)
            for u, v in new_edges
            if global_comm_map.get(u) is not None
            and global_comm_map.get(v) is not None
            and global_comm_map.get(u) != global_comm_map.get(v)
        }
        cross_comm_weight_gain = edge_weight_sum(G, new_cross_edges)
        edge_gain_rate = new_edge_weight / max(prev_edge_weight, 1.0)
        topic_birth_rate = len(new_topics) / max(len(curr_topics), 1)
        anchor_citer_count = sum(1 for n in roll_nodes if G.nodes[n].get("anchor_citer") or G.nodes[n].get("anchor_label"))
        anchor_citer_reach = anchor_citer_count / max(len(roll_nodes), 1)
        curr_field_entropy = shannon_entropy(G.nodes[n].get("primary_topic") for n in roll_nodes if n in G)
        field_entropy_gain = curr_field_entropy - prev_field_entropy if i > 0 else 0.0
        paper_rate = len(roll_nodes) / max((r1 - r0 + 1), 1)
        paper_burst_yoy = (paper_rate / max(prev_paper_rate, 1e-9)) - 1.0 if i > 0 else 0.0
        expansion_raw = safe_log1p(len(new_nodes)) + safe_log1p(len(new_edges)) + safe_log1p(len(new_topics))

        # Bridging: cross-community edge ratio + participation + betweenness of rolling papers.
        edge_total = max(1, Gcum.number_of_edges())
        inter_edges = 0
        for u, v in Gcum.edges():
            if global_comm_map.get(u) != global_comm_map.get(v):
                inter_edges += 1
        inter_ratio = inter_edges / edge_total

        if roll_nodes:
            part_vals = [participation_coefficient(Gcum, global_comm_map, n) for n in roll_nodes if n in Gcum]
            part_mean = float(np.mean(part_vals)) if part_vals else 0.0
        else:
            part_mean = 0.0

        if Gcum.number_of_nodes() > 2 and Gcum.number_of_edges() > 0:
            try:
                k = min(bet_k, max(2, Gcum.number_of_nodes() - 1))
                bc = nx.betweenness_centrality(Gcum, k=k, seed=seed, weight="weight", normalized=True)
            except Exception:
                bc = nx.betweenness_centrality(Gcum, weight=None, normalized=True)
            roll_bc = [bc.get(n, 0.0) for n in roll_nodes]
            if roll_bc:
                # Focus on the most bridge-like new papers rather than average all papers.
                q = np.quantile(roll_bc, 0.90) if len(roll_bc) > 10 else min(roll_bc)
                bridge_bc = float(np.mean([x for x in roll_bc if x >= q]))
            else:
                bridge_bc = 0.0
        else:
            bridge_bc = 0.0
        bridging_raw = 0.50 * inter_ratio + 0.35 * part_mean + 0.15 * bridge_bc

        # Reconfiguration: community partition change + edge turnover + modularity change.
        curr_partition = detect_communities(Gcum, cfg, compact=False) if Gcum.number_of_nodes() else {}
        common = sorted(set(prev_partition) & set(curr_partition))
        if len(common) >= 3:
            prev_labels = [prev_partition[n] for n in common]
            curr_labels = [curr_partition[n] for n in common]
            ari = adjusted_rand_score(prev_labels, curr_labels)
            partition_change = max(0.0, 1.0 - float(ari))
        else:
            partition_change = 0.0
        union_edges = prev_cum_edges | curr_edges
        if i == 0:
            edge_turnover = 0.0
        else:
            edge_turnover = 1.0 - (len(prev_cum_edges & curr_edges) / len(union_edges)) if union_edges else 0.0
        curr_modularity = partition_modularity(Gcum, curr_partition) if curr_partition else 0.0
        modularity_delta = (curr_modularity - prev_modularity) if i > 0 else 0.0
        modularity_shift = abs(modularity_delta) if i > 0 else 0.0
        reconfiguration_raw = partition_change + 0.5 * edge_turnover + modularity_shift

        # Compression: shorter topic paths + hub concentration + lower semantic dispersion.
        TGcum = make_topic_graph(Gcum, global_comm_map, labels, active_nodes=set(Gcum.nodes()))
        curr_path = avg_shortest_path_topic(TGcum)
        curr_hub = weighted_degree_hub_concentration(TGcum, top_frac=0.15)
        curr_sem = semantic_dispersion(
            sorted(cum_nodes),
            node_to_idx,
            X,
            sample_size=sem_sample,
            seed=seed + i,
        )
        if i == 0:
            path_gain = 0.0
            hub_gain = 0.0
            sem_gain = 0.0
        else:
            if np.isfinite(prev_path) and np.isfinite(curr_path):
                path_gain = prev_path - curr_path
            else:
                path_gain = 0.0
            hub_gain = curr_hub - prev_hub
            sem_gain = prev_sem - curr_sem
        if not bool(cfg.get("metrics", {}).get("signed_compression", True)):
            path_gain = max(0.0, path_gain)
            hub_gain = max(0.0, hub_gain)
            sem_gain = max(0.0, sem_gain)
        compression_raw = path_gain + hub_gain + sem_gain

        rows.append(
            {
                "window_index": i + 1,
                "rolling_start": r0,
                "rolling_end": r1,
                "cumulative_start": c0,
                "cumulative_end": c1,
                "label": window_label(c0, c1, cfg["start_year"]),
                "n_rolling_papers": len(roll_nodes),
                "n_cumulative_papers": len(cum_nodes),
                "n_cumulative_edges": Gcum.number_of_edges(),
                "new_nodes": len(new_nodes),
                "new_edges": len(new_edges),
                "new_topics": len(new_topics),
                "new_edge_weight": new_edge_weight,
                "edge_gain_rate": edge_gain_rate,
                "cross_comm_weight_gain": cross_comm_weight_gain,
                "topic_birth_rate": topic_birth_rate,
                "anchor_citer_count": anchor_citer_count,
                "anchor_citer_reach": anchor_citer_reach,
                "field_entropy": curr_field_entropy,
                "field_entropy_gain": field_entropy_gain,
                "paper_burst_yoy": paper_burst_yoy,
                "intercommunity_edge_ratio": inter_ratio,
                "participation_mean": part_mean,
                "bridge_betweenness_top": bridge_bc,
                "partition_change": partition_change,
                "edge_turnover": edge_turnover,
                "modularity": curr_modularity,
                "modularity_delta": modularity_delta,
                "modularity_shift": modularity_shift,
                "topic_avg_shortest_path": curr_path,
                "hub_concentration": curr_hub,
                "semantic_dispersion": curr_sem,
                "path_gain": path_gain,
                "hub_gain": hub_gain,
                "semantic_gain": sem_gain,
                "Expansion_raw": expansion_raw,
                "Bridging_raw": bridging_raw,
                "Reconfiguration_raw": reconfiguration_raw,
                "Compression_raw": compression_raw,
            }
        )

        prev_cum_nodes = set(cum_nodes)
        prev_cum_edges = set(curr_edges)
        prev_partition = dict(curr_partition)
        prev_modularity = curr_modularity
        prev_path = curr_path
        prev_hub = curr_hub
        prev_sem = curr_sem
        prev_field_entropy = curr_field_entropy
        prev_paper_rate = paper_rate
        seen_topics |= curr_topics

    df = pd.DataFrame(rows)
    if not df.empty:
        df["B_proxy_raw"] = df["Bridging_raw"].astype(float)
        df["RTD_proxy_raw"] = df["participation_mean"].astype(float)
        df["RS_proxy_raw"] = (
            0.55 * robust_minmax(df["semantic_dispersion"].values)
            + 0.45 * robust_minmax(df["new_topics"].values)
        )
        df["Uzzi_proxy_raw"] = (
            0.45 * robust_minmax(df["participation_mean"].values)
            + 0.35 * robust_minmax(df["partition_change"].values)
            + 0.20 * robust_minmax(df["edge_turnover"].values)
        )
        df["DeltaQ_directionality_raw"] = df["modularity_delta"].astype(float)
        df["BurtIP_proxy_raw"] = (
            0.60 * robust_minmax(df["participation_mean"].values)
            + 0.40 * robust_minmax(df["intercommunity_edge_ratio"].values)
        )
        df["PDE_proxy_raw"] = (
            0.55 * robust_minmax(df["Expansion_raw"].values)
            + 0.45 * robust_minmax(df["new_edges"].values)
        )
        z_cols = [
            "Expansion_raw",
            "Bridging_raw",
            "Reconfiguration_raw",
            "Compression_raw",
            "new_edge_weight",
            "edge_gain_rate",
            "cross_comm_weight_gain",
            "topic_birth_rate",
            "anchor_citer_reach",
            "field_entropy_gain",
            "paper_burst_yoy",
        ]
        event_idx = landmark_window_index(df, cfg)
        non_event = df.index != event_idx if event_idx is not None else np.ones(len(df), dtype=bool)
        for col in z_cols:
            vals = df[col].astype(float).values
            base = vals[non_event]
            center = float(np.nanmean(base)) if len(base) else float(np.nanmean(vals))
            scale = float(np.nanstd(base)) if len(base) else float(np.nanstd(vals))
            df[f"{col}_non_event_z"] = (vals - center) / scale if scale > 1e-12 else 0.0
    curve_mode = str(cfg.get("metrics", {}).get("curve_mode", "cumulative_positive")).lower()
    for metric in METRIC_NAMES:
        raw = np.asarray(df[f"{metric}_raw"].values, dtype=float)
        if curve_mode == "raw":
            series = raw
        elif curve_mode == "cumulative_positive":
            series = np.cumsum(np.maximum(raw, 0.0))
        else:
            # cumulative delta from first window; keeps the figure narrative monotonic.
            series = np.maximum.accumulate(raw)
        df[f"{metric}_index"] = 100.0 * robust_minmax(series)
    return df


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def community_color_map(comm_ids: Sequence[int]) -> Dict[int, str]:
    cmap_names = ["tab20", "tab20b", "tab20c"]
    colors: List[Any] = []
    for name in cmap_names:
        cmap = plt.get_cmap(name)
        colors.extend([cmap(i) for i in range(cmap.N)])
    out: Dict[int, str] = {}
    for i, c in enumerate(sorted(set(int(x) for x in comm_ids))):
        out[c] = colors[i % len(colors)]
    return out


def scale_values(values: Sequence[float], vmin: float, vmax: float) -> List[float]:
    arr = np.asarray([0.0 if v is None or not np.isfinite(v) else float(v) for v in values], dtype=float)
    if len(arr) == 0:
        return []
    lo, hi = float(arr.min()), float(arr.max())
    if abs(hi - lo) < 1e-12:
        return [0.5 * (vmin + vmax)] * len(arr)
    return list(vmin + (arr - lo) / (hi - lo) * (vmax - vmin))


def normalize_parameter_key(value: Any) -> str:
    text = str(value or "").strip()
    folded = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    aliases = {
        "b": "B",
        "bridge": "B",
        "bridge_centrality": "B",
        "rs": "RS",
        "reference_span": "RS",
        "reference_spread": "RS",
        "rtd": "RTD",
        "reference_target_diversity": "RTD",
        "uzzi": "Uzzi",
        "uzzi_novelty": "Uzzi",
        "delta_q": "DeltaQ",
        "deltaq": "DeltaQ",
        "dq": "DeltaQ",
        "q_directionality": "DeltaQ",
        "burt_ip": "BurtIP",
        "burtip": "BurtIP",
        "ip": "BurtIP",
        "pde": "PDE",
    }
    return aliases.get(folded, text)


def dominant_parameter_specs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    pcfg = cfg.get("plot", {})
    raw_items = pcfg.get("dominant_parameters") or DEFAULT_CONFIG["plot"]["dominant_parameters"]
    max_params = int(pcfg.get("max_dominant_parameters", 4))
    specs: List[Dict[str, Any]] = []
    for item in raw_items:
        entry = {"key": item} if isinstance(item, str) else dict(item or {})
        key = normalize_parameter_key(entry.get("key") or entry.get("name") or entry.get("label"))
        spec = dict(PARAMETER_SPECS.get(key, {}))
        spec.update(entry)
        spec["key"] = key
        spec.setdefault("label", key)
        spec.setdefault("source", entry.get("source") or entry.get("column") or key)
        spec.setdefault("color", "#374151")
        specs.append(spec)
        if len(specs) >= max_params:
            break
    return specs


def standardize_parameter_values(values: Sequence[float], spec: Mapping[str, Any], manual_values: bool) -> np.ndarray:
    arr = np.asarray([np.nan if v is None else float(v) for v in values], dtype=float)
    if len(arr) == 0:
        return arr
    if np.isnan(arr).all():
        return np.zeros_like(arr)
    fill = float(np.nanmedian(arr))
    arr = np.where(np.isfinite(arr), arr, fill)
    if bool(spec.get("invert", False)):
        arr = -arr
    if manual_values and not bool(spec.get("standardize_values", False)):
        return arr

    mode = str(spec.get("standardize_mode") or "").lower()
    if mode == "none" or spec.get("standardize") is False:
        return arr
    if bool(spec.get("center_zero", False)):
        scale = float(np.nanstd(arr))
        return arr / scale if scale > 1e-12 else np.zeros_like(arr)
    if mode == "robust":
        center = float(np.nanmedian(arr))
        q25, q75 = np.nanpercentile(arr, [25, 75])
        scale = float(q75 - q25)
        return (arr - center) / scale if scale > 1e-12 else np.zeros_like(arr)
    center = float(np.nanmean(arr))
    scale = float(np.nanstd(arr))
    return (arr - center) / scale if scale > 1e-12 else np.zeros_like(arr)


def dominant_parameter_trajectories(
    metrics: pd.DataFrame,
    cfg: Dict[str, Any],
    allow_manual: bool = True,
) -> List[Dict[str, Any]]:
    trajectories: List[Dict[str, Any]] = []
    n = len(metrics)
    for spec in dominant_parameter_specs(cfg):
        manual = "values" in spec and spec.get("values") is not None
        if manual:
            if not allow_manual:
                raise ValueError(
                    f"dominant parameter {spec['key']} uses manual_schematic values; "
                    "main-figure mode requires computed trajectories."
                )
            raw_values = list(spec.get("values") or [])
            if len(raw_values) < n:
                raw_values = raw_values + [np.nan] * (n - len(raw_values))
            raw_values = raw_values[:n]
        else:
            source = str(spec.get("source") or "")
            if source not in metrics.columns:
                raw_values = [0.0] * n
            else:
                raw_values = metrics[source].values
        y = standardize_parameter_values(raw_values, spec, manual_values=manual)
        clip = spec.get("clip")
        if isinstance(clip, (list, tuple)) and len(clip) == 2:
            y = np.clip(y, float(clip[0]), float(clip[1]))
        trajectories.append(
            {
                "key": spec["key"],
                "label": str(spec.get("label") or spec["key"]),
                "color": str(spec.get("color") or "#374151"),
                "values": y,
                "spec": spec,
                "source_column": str(spec.get("source") or ""),
                "provenance": "manual_schematic" if manual else "computed",
                "manual_values": int(bool(manual)),
            }
        )
    return trajectories


def landmark_window_index(metrics: pd.DataFrame, cfg: Dict[str, Any]) -> Optional[int]:
    pcfg = cfg.get("plot", {})
    focus_year = year_int(pcfg.get("landmark_focus_year"))
    if focus_year is None:
        anchor_years = [year_int(a.get("year")) for a in cfg.get("anchors") or []]
        anchor_years = [y for y in anchor_years if y is not None]
        focus_year = min(anchor_years) if anchor_years else None
    if focus_year is None or metrics.empty:
        return None
    for i, row in enumerate(metrics.itertuples(index=False)):
        if int(row.rolling_start) <= focus_year <= int(row.rolling_end):
            return i
    return None


def resolve_metric_window_index(value: Any, metrics: pd.DataFrame, cfg: Dict[str, Any]) -> Optional[int]:
    if value is None or str(value).lower() == "landmark":
        return landmark_window_index(metrics, cfg)
    if isinstance(value, int):
        return value - 1 if 1 <= value <= len(metrics) else value if 0 <= value < len(metrics) else None
    text = str(value).strip()
    yr = year_int(text)
    if yr is not None:
        for i, row in enumerate(metrics.itertuples(index=False)):
            if int(row.rolling_start) <= yr <= int(row.rolling_end):
                return i
    labels = [f"{int(a)}-{int(b)}" for a, b in zip(metrics["rolling_start"], metrics["rolling_end"])]
    return labels.index(text) if text in labels else None


def draw_parameter_callouts(
    ax: plt.Axes,
    x: np.ndarray,
    trajectories: Sequence[Mapping[str, Any]],
    metrics: pd.DataFrame,
    cfg: Dict[str, Any],
) -> None:
    pcfg = cfg.get("plot", {})
    by_key = {str(t["key"]): t for t in trajectories}
    for callout in pcfg.get("parameter_callouts") or []:
        key = normalize_parameter_key(callout.get("parameter") or callout.get("key"))
        trajectory = by_key.get(key)
        if trajectory is None:
            continue
        idx = resolve_metric_window_index(callout.get("window", "landmark"), metrics, cfg)
        if idx is None or idx < 0 or idx >= len(x):
            continue
        y = np.asarray(trajectory["values"], dtype=float)
        color = str(callout.get("color") or trajectory["color"])
        dx = float(callout.get("dx", 0.0))
        dy = float(callout.get("dy", 0.25))
        text_x = float(x[idx]) + dx
        text_y = float(y[idx]) + dy
        if bool(callout.get("clip_to_ylim", True)):
            ymin, ymax = ax.get_ylim()
            span = ymax - ymin
            text_y = min(max(text_y, ymin + 0.10 * span), ymax - 0.10 * span)
        ax.annotate(
            str(callout.get("text") or ""),
            xy=(float(x[idx]), float(y[idx])),
            xytext=(text_x, text_y),
            textcoords="data",
            ha=str(callout.get("ha", "center")),
            va="center",
            fontsize=float(callout.get("fontsize", 7.5)),
            color=color,
            fontstyle=str(callout.get("fontstyle", "normal")),
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8, shrinkA=2, shrinkB=2),
            zorder=8,
        )


def draw_parameter_card_icon(ax: plt.Axes, icon: str, cx: float, cy: float, color: str, size: float = 34.0) -> None:
    """Draw a compact vector icon in a fixed-size drawing area."""
    da = DrawingArea(size, size, 0, 0, clip=False)
    half = 0.5 * size
    r = 0.44 * size
    da.add_artist(
        mpatches.Circle(
            (half, half),
            r,
            facecolor="white",
            edgecolor=color,
            linewidth=1.1,
            alpha=1.0,
        )
    )
    icon = str(icon or "bridge").lower()
    if icon == "novelty":
        for k in range(8):
            theta = 2 * math.pi * k / 8.0
            x1 = half + 0.18 * r * math.cos(theta)
            y1 = half + 0.18 * r * math.sin(theta)
            x2 = half + 0.72 * r * math.cos(theta)
            y2 = half + 0.72 * r * math.sin(theta)
            da.add_artist(plt.Line2D([x1, x2], [y1, y2], color=color, lw=1.0))
        da.add_artist(mpatches.Circle((half, half), 0.15 * r, facecolor=color, edgecolor=color))
    elif icon == "boundary":
        da.add_artist(
            mpatches.Arc(
                (half - 0.15 * r, half),
                0.85 * r,
                1.05 * r,
                angle=28,
                theta1=35,
                theta2=325,
                color=color,
                lw=1.5,
            )
        )
        da.add_artist(
            mpatches.Arc(
                (half + 0.15 * r, half),
                0.85 * r,
                1.05 * r,
                angle=28,
                theta1=215,
                theta2=145,
                color=color,
                lw=1.5,
            )
        )
        da.add_artist(plt.Line2D([half - 0.35 * r, half + 0.35 * r], [half - 0.35 * r, half + 0.35 * r], color=color, lw=1.1))
    elif icon == "target":
        for scale, lw in [(0.76, 1.0), (0.48, 1.0), (0.18, 1.2)]:
            da.add_artist(mpatches.Circle((half, half), scale * r, facecolor="none", edgecolor=color, linewidth=lw))
        da.add_artist(
            FancyArrowPatch(
                (half + 0.18 * r, half + 0.18 * r),
                (half + 0.78 * r, half + 0.78 * r),
                arrowstyle="-|>",
                mutation_scale=8,
                color=color,
                lw=1.2,
            )
        )
    else:
        x_left = half - 0.55 * r
        x_right = half + 0.55 * r
        y_base = half - 0.38 * r
        y_top = half + 0.42 * r
        for x0 in [x_left, x_right]:
            da.add_artist(plt.Line2D([x0, x0], [y_base, y_top], color=color, lw=1.3))
            da.add_artist(mpatches.Arc((x0, y_top), 0.52 * r, 0.70 * r, theta1=180, theta2=360, color=color, lw=1.3))
        da.add_artist(plt.Line2D([x_left - 0.25 * r, x_right + 0.25 * r], [y_base, y_base], color=color, lw=1.3))
        da.add_artist(plt.Line2D([x_left + 0.17 * r, x_right - 0.17 * r], [half + 0.18 * r, half + 0.18 * r], color=color, lw=1.2))

    ab = AnnotationBbox(
        da,
        (cx, cy),
        xycoords=ax.transAxes,
        frameon=False,
        box_alignment=(0.5, 0.5),
        pad=0.0,
        annotation_clip=False,
    )
    ab.set_zorder(8)
    ax.add_artist(ab)


def draw_parameter_interpretation_boxes(ax: plt.Axes, cfg: Dict[str, Any]) -> None:
    pcfg = cfg.get("plot", {})
    boxes = list(pcfg.get("parameter_interpretation_boxes") or [])
    if not boxes:
        return
    n = min(len(boxes), 4)
    box_width = min(max(float(pcfg.get("parameter_box_width", 0.18)), 0.12), 0.30)
    box_height = min(max(float(pcfg.get("parameter_box_height", 0.34)), 0.18), 0.46)
    box_gap = min(max(float(pcfg.get("parameter_box_gap", 0.035)), 0.0), 0.08)
    total_width = min(n * box_width + (n - 1) * box_gap, 0.98)
    if total_width < n * box_width:
        box_width = (total_width - (n - 1) * box_gap) / n
    left = 0.5 - 0.5 * total_width
    y = float(pcfg.get("parameter_box_y", -0.56))
    title_size = float(pcfg.get("parameter_box_title_size", 7.8))
    formula_size = float(pcfg.get("parameter_box_formula_size", 8.2))
    description_size = float(pcfg.get("parameter_box_description_size", 7.0))
    show_icons = bool(pcfg.get("parameter_box_show_icons", False))
    corner_radius = float(pcfg.get("parameter_box_corner_radius", 5.0))
    linewidth = float(pcfg.get("parameter_box_linewidth", 0.75))
    axes_width_pt = ax.get_position().width * ax.figure.get_figwidth() * 72.0
    axes_height_pt = ax.get_position().height * ax.figure.get_figheight() * 72.0
    box_width_pt = box_width * axes_width_pt
    box_height_pt = box_height * axes_height_pt
    for i, box in enumerate(boxes[:n]):
        color = str(box.get("color") or "#6B7280")
        x0 = left + i * (box_width + box_gap)
        y0 = y - 0.5 * box_height
        card = DrawingArea(box_width_pt, box_height_pt, 0, 0, clip=False)
        card.add_artist(
            mpatches.FancyBboxPatch(
                (0, 0),
                box_width_pt,
                box_height_pt,
                boxstyle=f"round,pad=0,rounding_size={corner_radius}",
                facecolor="white",
                edgecolor=color,
                linewidth=linewidth,
                alpha=0.98,
            )
        )
        card_box = AnnotationBbox(
            card,
            (x0 + 0.5 * box_width, y),
            xycoords=ax.transAxes,
            frameon=False,
            box_alignment=(0.5, 0.5),
            pad=0.0,
            annotation_clip=False,
        )
        card_box.set_zorder(5)
        ax.add_artist(card_box)

        if show_icons:
            icon_cx = x0 + 0.105 * box_width
            icon_cy = y
            draw_parameter_card_icon(
                ax,
                str(box.get("icon") or "bridge"),
                icon_cx,
                icon_cy,
                color,
                size=float(box.get("icon_size", pcfg.get("parameter_box_icon_size", 34))),
            )
            text_x = x0 + 0.205 * box_width
        else:
            text_x = x0 + 0.070 * box_width
        title = str(box.get("title") or "")
        formula = str(box.get("formula") or box.get("text") or "")
        description = str(box.get("description") or "")
        if not title and formula:
            parts = formula.split("=", 1)
            title = parts[0].strip()
            formula = parts[1].strip() if len(parts) > 1 else formula
        ax.text(
            text_x,
            y0 + 0.73 * box_height,
            title,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=float(box.get("title_size", title_size)),
            color=color,
            fontweight="bold",
            clip_on=False,
            zorder=6,
        )
        ax.text(
            text_x,
            y0 + 0.50 * box_height,
            formula,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=float(box.get("formula_size", formula_size)),
            color=str(box.get("formula_color") or "#111827"),
            fontweight="bold",
            clip_on=False,
            zorder=6,
        )
        ax.text(
            text_x,
            y0 + 0.25 * box_height,
            description,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=float(box.get("description_size", description_size)),
            color="#374151",
            linespacing=1.12,
            clip_on=False,
            zorder=6,
            wrap=True,
        )


def dominant_parameter_table(
    metrics: pd.DataFrame,
    cfg: Dict[str, Any],
    allow_manual: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    trajectories = dominant_parameter_trajectories(metrics, cfg, allow_manual=allow_manual)
    labels = [f"{int(a)}-{int(b)}" for a, b in zip(metrics["rolling_start"], metrics["rolling_end"])]
    for trajectory in trajectories:
        for i, value in enumerate(np.asarray(trajectory["values"], dtype=float)):
            rows.append(
                {
                    "parameter": trajectory["key"],
                    "label": trajectory["label"],
                    "window_index": i + 1,
                    "window": labels[i],
                    "standardized_value": value,
                    "source_column": trajectory.get("source_column", ""),
                    "provenance": trajectory.get("provenance", "computed"),
                    "manual_values": int(trajectory.get("manual_values", 0)),
                }
            )
    return pd.DataFrame(rows)


def compute_snapshot_delta_metrics(result: "DomainResult") -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    prev_nodes: Set[str] = set()
    prev_edges: Set[Tuple[str, str]] = set()
    prev_field_entropy = 0.0
    prev_paper_rate = 0.0
    seen_display_topics: Set[int] = set()
    G = result.G

    for i, ((r0, r1), (c0, c1)) in enumerate(zip(result.rolling_windows, result.cumulative_windows)):
        active = node_set_until_year(G, c1)
        rolling_nodes = node_set_for_years(G, r0, r1)
        Gcum = G.subgraph(active).copy()
        curr_edges = edge_key_set(Gcum)
        new_nodes = active - prev_nodes
        new_edges = curr_edges - prev_edges
        new_edge_weight = edge_weight_sum(G, new_edges)
        prev_edge_weight = edge_weight_sum(G, prev_edges)
        display_nodes = {n for n in active if n in result.display_comm_map}
        hidden_node_ratio = 1.0 - (len(display_nodes) / max(len(active), 1))
        TG = make_topic_graph(G, result.display_comm_map, result.display_labels, active_nodes=active)
        display_topics = {int(n) for n in TG.nodes()}
        new_display_topics = display_topics - seen_display_topics
        possible_pairs = max(1, len(display_topics) * (len(display_topics) - 1) // 2)
        display_saturation = TG.number_of_edges() / possible_pairs if len(display_topics) > 1 else 0.0
        cross_edges = {
            (u, v)
            for u, v in new_edges
            if result.comm_map.get(u) is not None
            and result.comm_map.get(v) is not None
            and result.comm_map.get(u) != result.comm_map.get(v)
        }
        cross_display_edges = {
            (u, v)
            for u, v in new_edges
            if result.display_comm_map.get(u) is not None
            and result.display_comm_map.get(v) is not None
            and result.display_comm_map.get(u) != result.display_comm_map.get(v)
        }
        curr_field_entropy = shannon_entropy(G.nodes[n].get("primary_topic") for n in rolling_nodes if n in G)
        anchor_citer_count = sum(1 for n in rolling_nodes if G.nodes[n].get("anchor_citer") or G.nodes[n].get("anchor_label"))
        paper_rate = len(rolling_nodes) / max((r1 - r0 + 1), 1)
        rows.append(
            {
                "snapshot_index": i + 1,
                "rolling_start": r0,
                "rolling_end": r1,
                "cumulative_start": c0,
                "cumulative_end": c1,
                "n_cumulative_papers": len(active),
                "n_rolling_papers": len(rolling_nodes),
                "new_nodes": len(new_nodes),
                "new_edges": len(new_edges),
                "new_edge_weight": new_edge_weight,
                "edge_gain_rate": new_edge_weight / max(prev_edge_weight, 1.0),
                "cross_community_edge_weight": edge_weight_sum(G, cross_edges),
                "cross_display_edge_weight": edge_weight_sum(G, cross_display_edges),
                "hidden_node_ratio": hidden_node_ratio,
                "displayed_topics": len(display_topics),
                "new_display_topics": len(new_display_topics),
                "display_saturation": display_saturation,
                "topic_birth_rate": len(new_display_topics) / max(len(display_topics), 1),
                "anchor_citer_count": anchor_citer_count,
                "anchor_citer_reach": anchor_citer_count / max(len(rolling_nodes), 1),
                "field_entropy": curr_field_entropy,
                "field_entropy_gain": curr_field_entropy - prev_field_entropy if i > 0 else 0.0,
                "paper_burst_yoy": (paper_rate / max(prev_paper_rate, 1e-9)) - 1.0 if i > 0 else 0.0,
            }
        )
        prev_nodes = set(active)
        prev_edges = set(curr_edges)
        prev_field_entropy = curr_field_entropy
        prev_paper_rate = paper_rate
        seen_display_topics |= display_topics

    return pd.DataFrame(rows)


def draw_snapshot(
    ax: plt.Axes,
    G: nx.Graph,
    comm_map: Mapping[str, int],
    labels: Mapping[int, str],
    pos: Mapping[int, np.ndarray],
    color_map: Mapping[int, Any],
    cfg: Dict[str, Any],
    end_year: int,
    prev_end_year: Optional[int],
    panel_label: str,
    show_ylabel: bool = False,
) -> None:
    pcfg = cfg["plot"]
    active = node_set_until_year(G, end_year)
    prev_active = node_set_until_year(G, prev_end_year) if prev_end_year else set()
    TG = make_topic_graph(G, comm_map, labels, active_nodes=active)
    TGprev = make_topic_graph(G, comm_map, labels, active_nodes=prev_active) if prev_end_year else nx.Graph()
    TG = filter_topic_graph_for_display(TG, end_year, cfg["plot"])
    if prev_end_year:
        TGprev = filter_topic_graph_for_display(TGprev, prev_end_year, cfg["plot"])

    panel_lines = str(panel_label or "").splitlines()
    year_label = panel_lines[-1] if panel_lines else str(end_year)
    ax.set_title(year_label, fontsize=10.5, pad=8, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_edgecolor("#D9D9D9")

    if show_ylabel:
        ax.text(
            -0.12,
            0.5,
            cfg["domain_name"],
            transform=ax.transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=11,
            fontweight="bold",
        )

    if TG.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No displayed topics", transform=ax.transAxes, ha="center", va="center")
        return

    nodes = [int(n) for n in TG.nodes() if int(n) in pos]
    anchor_nodes = {int(n) for n in nodes if TG.nodes[n].get("anchor_labels") and (TG.nodes[n].get("anchor_year") is None or TG.nodes[n].get("anchor_year") <= end_year)}

    # Soft background halos, drawn before edges and beads.
    rmin = float(pcfg.get("cluster_radius_min", 0.13))
    rmax = float(pcfg.get("cluster_radius_max", 0.24))
    radii_list = log_scaled_radius([TG.nodes[n].get("n_papers", 1) for n in nodes], rmin, rmax)
    radii = {n: r for n, r in zip(nodes, radii_list)}

    for n in nodes:
        x, y = pos[n]
        first_year = TG.nodes[n].get("first_year")
        is_new_topic = prev_end_year is None or (first_year is not None and first_year > prev_end_year)
        has_anchor = n in anchor_nodes
        base_color = color_map.get(n, "#9CA3AF")
        halo = mpatches.Circle(
            (x, y),
            radius=radii[n],
            facecolor=base_color,
            edgecolor=base_color,
            lw=0.9 if not has_anchor else 1.6,
            alpha=0.10 if not is_new_topic else 0.16,
            zorder=0,
        )
        ax.add_patch(halo)
        if has_anchor:
            ax.add_patch(
                mpatches.Circle(
                    (x, y),
                    radius=radii[n] * 1.28,
                    facecolor="none",
                    edgecolor="#DC2626",
                    lw=1.0,
                    linestyle="--",
                    alpha=0.55,
                    zorder=1,
                )
            )

    # Sparse curved backbone edges.
    selected_edges = select_backbone_edges(TG, TGprev, pcfg, anchor_nodes)
    for idx, (u, v, d, is_new) in enumerate(selected_edges):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = float(d.get("weight", 1.0))
        lw = 0.45 + 1.35 * min(1.0, math.log1p(w) / 7.0)
        rad = (0.12 + 0.04 * (idx % 3)) * (-1 if idx % 2 else 1)
        color = "#3F3F46" if is_new else "#9CA3AF"
        alpha = 0.55 if is_new else 0.28
        patch = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-",
            connectionstyle=f"arc3,rad={rad}",
            linewidth=lw,
            color=color,
            alpha=alpha,
            zorder=2 if is_new else 1,
            shrinkA=9,
            shrinkB=9,
        )
        ax.add_patch(patch)

    # Representative paper beads inside each topic halo.
    max_beads = int(pcfg.get("max_representative_papers", 7))
    for n in nodes:
        x, y = pos[n]
        base_color = color_map.get(n, "#9CA3AF")
        n_papers = int(TG.nodes[n].get("n_papers", 1) or 1)
        n_beads = int(np.clip(round(3 + math.log1p(n_papers)), 4, max_beads))
        pts = deterministic_disc_points(n, n_beads, radii[n] * 0.66)
        if pcfg.get("show_internal_cluster_edges", True) and len(pts) > 2:
            for j in range(1, len(pts)):
                x0, y0 = x + pts[0, 0], y + pts[0, 1]
                x1, y1 = x + pts[j, 0], y + pts[j, 1]
                ax.plot([x0, x1], [y0, y1], color="#9CA3AF", lw=0.45, alpha=0.38, zorder=3)
        ax.scatter(
            x + pts[:, 0],
            y + pts[:, 1],
            s=float(pcfg.get("node_size_min", 80)),
            color=base_color,
            edgecolors="white",
            linewidths=0.5,
            alpha=0.94,
            zorder=4,
        )

        # Anchor star and concise annotation.
        if n in anchor_nodes:
            ax.scatter([x], [y], s=210, marker="*", color="#DC2626", edgecolors="white", linewidths=0.7, zorder=7)
            short_anchor = clean_label(TG.nodes[n].get("anchor_labels", "landmark papers"), 32)
            ax.annotate(
                short_anchor,
                xy=(x, y),
                xytext=(x + 0.12, y - 0.16),
                textcoords="data",
                fontsize=6.7,
                color="#B91C1C",
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#B91C1C", lw=0.75, alpha=0.85),
                zorder=8,
            )

    # Topic labels: small and schematic-like.
    label_candidates = sorted(
        nodes,
        key=lambda n: (
            1 if n in anchor_nodes else 0,
            TG.degree(n, weight="weight"),
            TG.nodes[n].get("n_papers", 0),
        ),
        reverse=True,
    )[: int(pcfg.get("max_labels_per_panel", 8))]
    for n in label_candidates:
        x, y = pos[n]
        label = TG.nodes[n].get("display_label") or TG.nodes[n].get("label", f"Topic {n + 1}")
        # A little radial offset prevents text from sitting exactly on beads.
        offset_y = radii[n] * 0.74
        ax.text(
            x,
            y + offset_y,
            clean_label(label, 26),
            fontsize=6.7,
            ha="center",
            va="bottom",
            color="#2F2F36",
            zorder=9,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.70),
        )

    # Optional panel caption below each snapshot.
    if pcfg.get("show_panel_captions", True):
        captions = pcfg.get("panel_captions") or []
        # Determine caption index from cumulative end year.
        idx = None
        ends = [e for _, e in make_cumulative_windows_from_config(cfg, make_rolling_windows_from_config(cfg))]
        if end_year in ends:
            idx = ends.index(end_year)
        if idx is not None and idx < len(captions):
            ax.text(0.5, -0.08, str(captions[idx]), transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#5B5B66")

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if xs and ys:
        xpad = 0.18 * (max(xs) - min(xs) + 1e-9)
        ypad = 0.18 * (max(ys) - min(ys) + 1e-9)
        ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
        ax.set_ylim(min(ys) - ypad, max(ys) + ypad)
        ax.set_aspect("equal", adjustable="box")

    ax.text(
        0.02,
        0.035,
        f"n={len(active):,} papers\n{TG.number_of_nodes()} displayed topics",
        transform=ax.transAxes,
        fontsize=6.8,
        color="#5B6472",
        ha="left",
        va="bottom",
    )


def draw_metric_panel(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    cfg: Dict[str, Any],
    compact: bool = False,
    panel_label: str = "b",
) -> None:
    pcfg = cfg.get("plot", {})
    if pcfg.get("metric_x_axis", "years") == "years":
        x = 0.5 * (metrics["rolling_start"].values.astype(float) + metrics["rolling_end"].values.astype(float))
        labels = [f"{int(a)}-{int(b)}" for a, b in zip(metrics["rolling_start"], metrics["rolling_end"])]
        ax.set_xlim(float(metrics["rolling_start"].min()) - 0.5, float(metrics["rolling_end"].max()) + 1.0)
        x_label = "Rolling publication window" if cfg.get("custom_windows") else "Rolling 5-year publication window"
        ax.set_xlabel(x_label, fontsize=8.5 if not compact else 7, labelpad=8 if not compact else 4)
    else:
        x = np.arange(len(metrics))
        labels = [str(v).split("\n")[0] for v in metrics["label"].tolist()]
        ax.set_xlim(-0.5, len(metrics) - 0.5)
        ax.set_xlabel("Cumulative window", fontsize=8.5 if not compact else 7, labelpad=8 if not compact else 4)

    x = np.asarray(x, dtype=float)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8 if not compact else 6.5)

    landmark_idx = landmark_window_index(metrics, cfg)
    if landmark_idx is not None and 0 <= landmark_idx < len(metrics):
        if pcfg.get("metric_x_axis", "years") == "years":
            row = metrics.iloc[landmark_idx]
            ax.axvspan(float(row["rolling_start"]), float(row["rolling_end"]) + 1.0, color="#FEE2E2", alpha=0.55, lw=0, zorder=0)
        else:
            ax.axvspan(float(landmark_idx) - 0.5, float(landmark_idx) + 0.5, color="#FEE2E2", alpha=0.55, lw=0, zorder=0)

    allow_manual = bool(pcfg.get("allow_manual_trajectories", False))
    trajectories = dominant_parameter_trajectories(metrics, cfg, allow_manual=allow_manual)
    for trajectory in trajectories:
        y = np.asarray(trajectory["values"], dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.9 if not compact else 1.3,
            markersize=4.2 if not compact else 3.2,
            label=str(trajectory["label"]),
            color=str(trajectory["color"]),
            zorder=4,
        )
        if landmark_idx is not None and 0 <= landmark_idx < len(y):
            ax.scatter(
                [x[landmark_idx]],
                [y[landmark_idx]],
                marker="*",
                s=120 if not compact else 60,
                color=str(trajectory["color"]),
                edgecolors="white",
                linewidths=0.6,
                zorder=7,
            )

    ax.axhline(0.0, color="#4B5563", lw=0.8, alpha=0.75, zorder=1)
    ylim = pcfg.get("dominant_parameter_ylim")
    if isinstance(ylim, (list, tuple)) and len(ylim) == 2:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    else:
        vals = np.concatenate([np.asarray(t["values"], dtype=float) for t in trajectories]) if trajectories else np.array([0.0])
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
        pad = max(0.25, 0.16 * (hi - lo + 1e-9))
        ax.set_ylim(min(lo - pad, -0.2), max(hi + pad, 0.8))

    ax.set_ylabel(str(pcfg.get("metric_y_label", "Standardized\nparameter value")), fontsize=8.5 if not compact else 7)
    ax.grid(True, axis="y", alpha=0.36, linewidth=0.6, linestyle=(0, (4, 4)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    title = str(pcfg.get("metric_panel_title", "Dominant parameter trajectories"))
    title_prefix = f"{panel_label}  " if panel_label else ""
    ax.set_title(f"{title_prefix}{title}", fontsize=10.5 if not compact else 8, fontweight="bold", loc="left")

    if landmark_idx is not None and 0 <= landmark_idx < len(x):
        label = str(pcfg.get("metric_landmark_label", "")).strip()
        if label:
            ylim_now = ax.get_ylim()
            ax.text(
                float(x[landmark_idx]),
                ylim_now[1] - 0.05 * (ylim_now[1] - ylim_now[0]),
                label,
                ha="center",
                va="top",
                fontsize=7.4 if not compact else 6.2,
                color="#DC2626",
                fontweight="bold",
                zorder=9,
            )

    if not compact:
        ncol = min(4, max(1, len(trajectories)))
        ax.legend(ncol=ncol, fontsize=8, loc="upper left", frameon=False, columnspacing=1.4, handlelength=2.0)
        draw_parameter_callouts(ax, x, trajectories, metrics, cfg)
        draw_parameter_interpretation_boxes(ax, cfg)
    else:
        ax.legend(fontsize=6.3, loc="upper left", frameon=False)


def draw_top_time_axis(ax: plt.Axes, result: "DomainResult") -> None:
    cfg = result.cfg
    pcfg = cfg.get("plot", {})
    ax.set_xlim(cfg["start_year"], cfg["end_year"] + 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y = 0.48
    ax.plot([cfg["start_year"], cfg["end_year"] + 0.5], [y, y], color="#8D8D8D", lw=1.1, clip_on=False)
    ax.annotate("", xy=(cfg["end_year"] + 0.55, y), xytext=(cfg["end_year"] - 0.2, y), arrowprops=dict(arrowstyle="->", color="#8D8D8D", lw=1.1))
    tick_years = [cfg["start_year"]] + [end for _, end in result.cumulative_windows]
    tick_years = sorted(dict.fromkeys(tick_years))
    for yr in tick_years:
        ax.plot([yr, yr], [y - 0.12, y + 0.12], color="#8D8D8D", lw=0.9)
        ax.text(yr, y - 0.20, str(yr), ha="center", va="top", fontsize=8, color="#60636B")

    anchor_years = [year_int(a.get("year")) for a in cfg.get("anchors") or []]
    anchor_years = [v for v in anchor_years if v]
    if anchor_years:
        x0, x1 = min(anchor_years) - 0.2, max(anchor_years) + 0.6
        ax.axvspan(x0, x1, ymin=0.28, ymax=0.76, color="#FEE2E2", alpha=0.75, lw=0)
        ax.annotate(
            "innovation event",
            xy=((x0 + x1) / 2, y + 0.05),
            xytext=(min(anchor_years) - 3.2, 0.88),
            ha="left",
            va="center",
            fontsize=8.5,
            color="#DC2626",
            arrowprops=dict(arrowstyle="->", lw=0.8, color="#DC2626"),
        )
        event_label = str(pcfg.get("top_landmark_label") or "").strip()
        if not event_label:
            if min(anchor_years) == max(anchor_years):
                event_label = f"{min(anchor_years)}\nlandmark paper"
            else:
                event_label = f"{min(anchor_years)}-{max(anchor_years)}\nlandmark papers"
        ax.text((x0 + x1) / 2, 0.86, event_label, ha="center", va="top", fontsize=8.6, color="#B91C1C", fontweight="bold")
    ax.text(cfg["end_year"] + 0.75, y - 0.20, "yr", ha="left", va="top", fontsize=8, color="#60636B")


def draw_single_domain_figure(result: "DomainResult", out_dir: Path) -> None:
    cfg = result.cfg
    pcfg = cfg["plot"]
    domain_dir = out_dir / cfg["slug"]
    domain_dir.mkdir(parents=True, exist_ok=True)

    n_snapshots = max(1, len(result.cumulative_windows))
    fig = plt.figure(figsize=(pcfg["fig_width_single"], pcfg["fig_height_single"]), dpi=pcfg["dpi"])
    gs = GridSpec(3, n_snapshots, figure=fig, height_ratios=[0.44, 2.95, 1.35], hspace=0.30, wspace=0.13)

    title = pcfg.get("title") or "Knowledge-graph perturbation"
    subtitle = pcfg.get("subtitle") or ""
    fig.suptitle(title, y=0.992, fontsize=16, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.958, subtitle, ha="center", va="top", fontsize=10, color="#4B5563")

    axt = fig.add_subplot(gs[0, :])
    if pcfg.get("show_time_axis", True):
        draw_top_time_axis(axt, result)
    else:
        axt.axis("off")

    snapshot_axes: List[plt.Axes] = []
    for i, (_, end) in enumerate(result.cumulative_windows):
        prev_end = result.cumulative_windows[i - 1][1] if i > 0 else None
        ax = fig.add_subplot(gs[1, i])
        snapshot_axes.append(ax)
        draw_snapshot(
            ax,
            result.G,
            result.display_comm_map,
            result.display_labels,
            result.pos,
            result.color_map,
            cfg,
            end_year=end,
            prev_end_year=prev_end,
            panel_label=window_label(cfg["start_year"], end, cfg["start_year"]),
            show_ylabel=(i == 0),
        )

    axm = fig.add_subplot(gs[2, :])
    draw_metric_panel(axm, result.metrics, cfg, compact=False)
    heading_x = axm.get_position().x0
    fig.text(
        heading_x,
        0.765,
        "a  Cumulative citation knowledge graph snapshots",
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="center",
    )
    if pcfg.get("show_metric_connectors", True):
        metric_bbox = axm.get_position()
        for ax in snapshot_axes:
            bbox = ax.get_position()
            x_mid = 0.5 * (bbox.x0 + bbox.x1)
            y_top = bbox.y0 - 0.012
            y_bottom = metric_bbox.y1 + 0.012
            if y_top > y_bottom:
                fig.add_artist(
                    plt.Line2D(
                        [x_mid, x_mid],
                        [y_bottom, y_top],
                        transform=fig.transFigure,
                        color="#9CA3AF",
                        lw=0.8,
                        linestyle=(0, (2, 2)),
                        alpha=0.65,
                    )
                )

    if pcfg.get("show_retrieval_date", True):
        fig.text(
            0.995,
            0.008,
            f"Data source: OpenAlex; generated {dt.date.today().isoformat()}",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#6B7280",
        )

    for ext in ["png", "svg", "pdf"]:
        path = domain_dir / f"fig1_{cfg['slug']}_real.{ext}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_multi_domain_figure(results: Sequence["DomainResult"], out_dir: Path) -> None:
    if not results:
        return
    nrows = len(results)
    max_snapshots = max(len(r.cumulative_windows) for r in results)
    ncols = max_snapshots + 1
    pcfg = results[0].cfg["plot"]
    fig_w = float(pcfg.get("fig_width_multi", 28))
    fig_h = float(pcfg.get("row_height_multi", 4.35)) * nrows
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(pcfg.get("dpi", 300)))
    gs = GridSpec(
        nrows,
        ncols,
        figure=fig,
        width_ratios=[1.0] * max_snapshots + [1.65],
        hspace=0.28,
        wspace=0.16,
    )

    fig.suptitle(
        "Landmark discoveries induce reproducible perturbations in citation knowledge graphs",
        y=0.995,
        fontsize=16,
        fontweight="bold",
    )

    for r, result in enumerate(results):
        cfg = result.cfg
        pcfg_domain = cfg.get("plot", {})
        cfg_draw = deep_update(
            cfg,
            {
                "plot": {
                    "show_internal_cluster_edges": False,
                    "max_representative_papers": min(int(pcfg_domain.get("max_representative_papers", 6)), 5),
                    "max_labels_per_panel": min(int(pcfg_domain.get("max_labels_per_panel", 8)), 6),
                    "display_max_backbone_edges": min(int(pcfg_domain.get("display_max_backbone_edges", 15)), 12),
                    "display_extra_edges": min(int(pcfg_domain.get("display_extra_edges", 6)), 4),
                    "cluster_radius_min": min(float(pcfg_domain.get("cluster_radius_min", 0.13)), 0.10),
                    "cluster_radius_max": min(float(pcfg_domain.get("cluster_radius_max", 0.24)), 0.18),
                    "node_size_min": min(float(pcfg_domain.get("node_size_min", 64)), 46),
                }
            },
        )
        for i, (_, end) in enumerate(result.cumulative_windows):
            prev_end = result.cumulative_windows[i - 1][1] if i > 0 else None
            ax = fig.add_subplot(gs[r, i])
            draw_snapshot(
                ax,
                result.G,
                result.display_comm_map,
                result.display_labels,
                result.pos,
                result.color_map,
                cfg_draw,
                end_year=end,
                prev_end_year=prev_end,
                panel_label=window_label(cfg["start_year"], end, cfg["start_year"]) if r == 0 else "",
                show_ylabel=(i == 0),
            )
        for i in range(len(result.cumulative_windows), max_snapshots):
            ax = fig.add_subplot(gs[r, i])
            ax.axis("off")
        axm = fig.add_subplot(gs[r, ncols - 1])
        draw_metric_panel(axm, result.metrics, cfg_draw, compact=True, panel_label="")
        axm.text(
            0.02,
            0.98,
            cfg["domain_name"],
            transform=axm.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            fontweight="bold",
        )

    if pcfg.get("show_retrieval_date", False):
        fig.text(
            0.995,
            0.005,
            f"Data source: OpenAlex; generated {dt.date.today().isoformat()}",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#6B7280",
        )

    for ext in ["png", "svg", "pdf"]:
        path = out_dir / f"fig1_multi_domain_real.{ext}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Export helpers and orchestration
# -----------------------------------------------------------------------------


@dataclass
class DomainResult:
    cfg: Dict[str, Any]
    works_all: Dict[str, Dict[str, Any]]
    works_selected: Dict[str, Dict[str, Any]]
    G: nx.Graph
    comm_map: Dict[str, int]
    labels: Dict[int, str]
    display_comm_map: Dict[str, int]
    display_labels: Dict[int, str]
    TG_full: nx.Graph
    TG_display: nx.Graph
    pos: Dict[int, np.ndarray]
    color_map: Dict[int, Any]
    metrics: pd.DataFrame
    rolling_windows: List[Tuple[int, int]]
    cumulative_windows: List[Tuple[int, int]]


def count_manual_parameter_specs(cfg: Mapping[str, Any]) -> int:
    return int(
        sum(
            1
            for spec in dominant_parameter_specs(dict(cfg))
            if "values" in spec and spec.get("values") is not None
        )
    )


def build_edge_cap_diagnostics(result: DomainResult) -> Dict[str, Any]:
    cfg = result.cfg
    gcfg = cfg.get("graph", {})
    max_edges = int(gcfg.get("max_edges", 0))
    edge_count = int(result.G.number_of_edges())
    sampling = dict(result.G.graph.get("edge_sampling") or {})
    direct_edges = 0
    bibliographic_edges = 0
    cocitation_edges = 0
    hybrid_only_edges = 0
    for _, _, data in result.G.edges(data=True):
        direct = float(data.get("direct", 0) or 0)
        bibliographic = float(data.get("bibliographic", 0) or 0)
        cocitation = float(data.get("cocitation", 0) or 0)
        if direct > 0:
            direct_edges += 1
        else:
            hybrid_only_edges += 1
        if bibliographic > 0:
            bibliographic_edges += 1
        if cocitation > 0:
            cocitation_edges += 1
    sampling_applied = int(sampling.get("sampling_applied", 0) or 0)
    edge_cap_hit = int(max_edges > 0 and edge_count >= max_edges and not sampling_applied)
    return {
        "domain": cfg.get("slug"),
        "domain_name": cfg.get("domain_name"),
        "graph_max_edges": max_edges,
        "sampling_target_edges": int(sampling.get("sampling_target_edges", 0) or 0),
        "deterministic_hybrid_sampling": sampling_applied,
        "exported_edges": edge_count,
        "raw_edges_before_sampling": int(sampling.get("raw_edges", edge_count) or edge_count),
        "edge_cap_hit": edge_cap_hit,
        "direct_edges": direct_edges,
        "bibliographic_edges": bibliographic_edges,
        "cocitation_edges": cocitation_edges,
        "hybrid_only_edges": hybrid_only_edges,
        "hybrid_only_edge_ratio": float(hybrid_only_edges / edge_count) if edge_count else 0.0,
        "interpretation": "deterministic_hybrid_sampling_standardized_trajectories"
        if sampling_applied
        else ("edge_cap_hit_density_not_interpretable" if edge_cap_hit else "edge_cap_not_hit"),
    }


def effective_main_cumulative_horizon_years(cfg: Mapping[str, Any], cumulative_windows: Sequence[Tuple[int, int]]) -> int:
    """Return the main-text cumulative horizon, honoring a configured common horizon."""
    metrics_cfg = cfg.get("metrics", {}) if isinstance(cfg.get("metrics", {}), Mapping) else {}
    common = metrics_cfg.get("common_cumulative_horizon_years")
    if common is not None:
        try:
            return int(common)
        except (TypeError, ValueError):
            pass
    if not cumulative_windows:
        return 0
    return int(cumulative_windows[-1][1] - cumulative_windows[0][0] + 1)


def build_fig1_quality_gates(results: Sequence[DomainResult]) -> Dict[str, Any]:
    edge_reports = [build_edge_cap_diagnostics(result) for result in results]
    manual_counts = {str(result.cfg.get("slug")): count_manual_parameter_specs(result.cfg) for result in results}
    final_horizons = {
        str(result.cfg.get("slug")): effective_main_cumulative_horizon_years(result.cfg, result.cumulative_windows)
        for result in results
    }
    sampling_manifest_present = all(bool(result.G.graph.get("edge_sampling_manifest")) for result in results)
    checks = {
        "manual_trajectories_absent": int(all(count == 0 for count in manual_counts.values())),
        "edge_cap_not_hit_all_domains": int(all(int(report["edge_cap_hit"]) == 0 for report in edge_reports)),
        "final_cumulative_horizon_consistent": int(len(set(final_horizons.values())) <= 1),
        "edge_sampling_manifest_present": int(sampling_manifest_present),
    }
    overall = bool(all(checks.values()))
    return {
        "overall_pass": overall,
        "status_label": "main-figure ready" if overall else "diagnostic / schematic evidence",
        "checks": checks,
        "manual_trajectory_counts": manual_counts,
        "edge_cap_diagnostics": edge_reports,
        "final_cumulative_horizon_years": final_horizons,
        "thresholds": {
            "manual_trajectories_allowed_in_main": 0,
            "edge_cap_hit_allowed": 0,
            "final_cumulative_horizon_unique_values": 1,
            "edge_sampling_manifest_required": 1,
        },
    }


def export_tables(result: DomainResult, out_dir: Path) -> None:
    cfg = result.cfg
    domain_dir = out_dir / cfg["slug"]
    domain_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for wid, w in result.works_selected.items():
        rows.append(
            {
                "id": wid,
                "short_id": short_openalex_id(wid),
                "doi": w.get("doi"),
                "title": w.get("title"),
                "year": w.get("year"),
                "cited_by_count": w.get("cited_by_count"),
                "primary_topic": w.get("primary_topic"),
                "community": result.comm_map.get(wid),
                "community_label": result.labels.get(result.comm_map.get(wid), "") if wid in result.comm_map else "",
                "display_community": result.display_comm_map.get(wid),
                "display_label": result.display_labels.get(result.display_comm_map.get(wid), "") if wid in result.display_comm_map else "",
                "anchor_label": w.get("anchor_label"),
                "anchor_citer": int(bool(w.get("anchor_citer"))),
                "reference_stub": int(bool(w.get("reference_stub"))),
            }
        )
    pd.DataFrame(rows).sort_values(["year", "cited_by_count"], ascending=[True, False]).to_csv(
        domain_dir / "works_selected.csv", index=False
    )

    edge_rows = []
    for u, v, d in result.G.edges(data=True):
        edge_rows.append(
            {
                "source": u,
                "target": v,
                "weight": d.get("weight", 1.0),
                "direct": d.get("direct", 0),
                "bibliographic": d.get("bibliographic", 0),
                "cocitation": d.get("cocitation", 0),
            }
        )
    pd.DataFrame(edge_rows).to_csv(domain_dir / "paper_edges.csv", index=False)

    topic_rows = []
    for c, d in result.TG_display.nodes(data=True):
        p = result.pos.get(c, np.array([np.nan, np.nan]))
        topic_rows.append(
            {
                "community": c,
                "label": d.get("label"),
                "n_papers": d.get("n_papers"),
                "cited_by_count": d.get("cited_by_count"),
                "first_year": d.get("first_year"),
                "anchor_labels": d.get("anchor_labels"),
                "x": p[0],
                "y": p[1],
            }
        )
    pd.DataFrame(topic_rows).sort_values("n_papers", ascending=False).to_csv(domain_dir / "topic_nodes.csv", index=False)

    topic_edge_rows = []
    for u, v, d in result.TG_display.edges(data=True):
        topic_edge_rows.append(
            {
                "source_community": u,
                "target_community": v,
                "weight": d.get("weight", 1.0),
                "n_edges": d.get("n_edges", 1),
            }
        )
    pd.DataFrame(topic_edge_rows).to_csv(domain_dir / "topic_edges.csv", index=False)

    result.metrics.to_csv(domain_dir / "perturbation_metrics.csv", index=False)
    compute_snapshot_delta_metrics(result).to_csv(domain_dir / "snapshot_delta_metrics.csv", index=False)
    pd.DataFrame([result.G.graph.get("edge_sampling_manifest") or build_edge_sampling_manifest_row(cfg["slug"], result.G, result.G, cfg.get("graph", {}))]).to_csv(
        domain_dir / "fig1_edge_sampling_manifest.csv",
        index=False,
    )
    allow_manual = bool(cfg.get("plot", {}).get("allow_manual_trajectories", False))
    dominant_parameter_table(result.metrics, cfg, allow_manual=allow_manual).to_csv(
        domain_dir / "dominant_parameter_trajectories.csv",
        index=False,
    )
    write_json(domain_dir / "edge_cap_diagnostics.json", build_edge_cap_diagnostics(result))


def run_domain(
    cfg: Dict[str, Any],
    client: OpenAlexClient,
    out_dir: Path,
    use_cache: bool = True,
    force_cache: bool = False,
) -> DomainResult:
    slug = cfg["slug"]
    print(f"\n=== Running domain: {cfg['domain_name']} ({slug}) ===")
    rolling = make_rolling_windows_from_config(cfg)
    cumulative = make_cumulative_windows_from_config(cfg, rolling)

    works_all = fetch_domain_works(cfg, client, out_dir, use_cache=use_cache, force_cache=force_cache)
    works_selected = select_balanced_papers(works_all, cfg, rolling)
    print(f"[{slug}] Selected {len(works_selected):,}/{len(works_all):,} works for graph construction")

    G = build_hybrid_graph(works_selected, works_all, cfg)
    print(f"[{slug}] Paper graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # Full community map is used for quantitative metrics. A separate display
    # community map is used for the Nature-style main figure, so the long tail is
    # not collapsed into a misleading giant "other" node.
    comm_map = detect_communities(G, cfg, compact=False)
    labels = make_community_labels(G, comm_map)
    display_comm_map = build_display_comm_map(G, comm_map, cfg)
    display_labels = make_community_labels(G, display_comm_map)
    TG_full = make_topic_graph(G, comm_map, labels, active_nodes=set(G.nodes()))
    TG_display = make_topic_graph(G, display_comm_map, display_labels, active_nodes=set(G.nodes()))
    pos = layout_topic_graph(TG_display, cfg)
    color_map = community_color_map(list(TG_display.nodes()))
    metrics = compute_perturbation_metrics(G, comm_map, labels, cfg, rolling, cumulative)

    result = DomainResult(
        cfg=cfg,
        works_all=works_all,
        works_selected=works_selected,
        G=G,
        comm_map=comm_map,
        labels=labels,
        display_comm_map=display_comm_map,
        display_labels=display_labels,
        TG_full=TG_full,
        TG_display=TG_display,
        pos=pos,
        color_map=color_map,
        metrics=metrics,
        rolling_windows=rolling,
        cumulative_windows=cumulative,
    )
    export_tables(result, out_dir)
    draw_single_domain_figure(result, out_dir)
    domain_dir = out_dir / cfg["slug"]
    quality = build_fig1_quality_gates([result])
    generated = [domain_dir / f"fig1_{cfg['slug']}_real.{ext}" for ext in ["png", "svg", "pdf"]]
    write_run_manifest(
        domain_dir,
        figure="fig1",
        argv=None,
        inputs={
            "domain_config_slug": cfg.get("slug"),
            "domain_name": cfg.get("domain_name"),
        },
        domains=[str(cfg.get("slug"))],
        quality_gates=quality,
        extra={
            "rolling_windows": rolling,
            "cumulative_windows": cumulative,
            "selected_papers": len(works_selected),
            "all_papers": len(works_all),
        },
    )
    write_figure_quality_report(
        domain_dir,
        figure="fig1",
        generated_files=generated,
        quality_gates=quality,
        extra={"edge_cap_diagnostics_path": str(domain_dir / "edge_cap_diagnostics.json")},
    )
    print(f"[{slug}] Done. Outputs in {out_dir / slug}")
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw real-data Fig. 1 knowledge graph perturbation panels.")
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="One or more YAML config files. One file gives a single-domain figure; multiple files also create a multi-domain figure.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--openalex-api-key", default=getenv("OPENALEX_API_KEY"), help="OpenAlex API key.")
    parser.add_argument("--openalex-api-keys", default=getenv("OPENALEX_API_KEYS"), help="Comma/space separated OpenAlex API keys used in round-robin order.")
    parser.add_argument("--email", default=getenv("OPENALEX_EMAIL"), help="Optional contact email for API calls.")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=None,
        help="Optional unified corpus directory. When provided, materialize Fig. 1 cache files from the corpus before drawing.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached works_raw.jsonl and re-download data.")
    parser.add_argument("--force-cache", action="store_true", help="Use works_raw.jsonl even when cache_manifest.json does not match the config.")
    parser.add_argument(
        "--allow-schematic-trajectories",
        action="store_true",
        help="Allow YAML dominant_parameters.values. Without this flag, Fig. 1 main-figure mode requires computed trajectories.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load configs first so API behavior can be controlled by the first config.
    cfgs = [load_config(Path(p)) for p in args.config]
    for cfg in cfgs:
        cfg.setdefault("plot", {})["allow_manual_trajectories"] = bool(args.allow_schematic_trajectories)
    if args.corpus_dir is not None:
        from aspr.corpus import materialize_fig1_cache  # pylint: disable=import-outside-toplevel

        materialize_fig1_cache(Path(args.corpus_dir), out_dir, [cfg["slug"] for cfg in cfgs])
    first_api = cfgs[0].get("api", {})
    client = OpenAlexClient(
        api_key=args.openalex_api_key,
        api_keys=split_api_keys(args.openalex_api_keys),
        email=args.email,
        sleep_seconds=float(first_api.get("sleep_seconds", 0.1)),
        max_retries=int(first_api.get("max_retries", 6)),
        timeout_seconds=int(first_api.get("timeout_seconds", 60)),
    )

    results = []
    for cfg in cfgs:
        result = run_domain(
            cfg,
            client,
            out_dir,
            use_cache=True if args.corpus_dir is not None else not args.no_cache,
            force_cache=True if args.corpus_dir is not None else bool(args.force_cache),
        )
        results.append(result)

    if len(results) > 1:
        draw_multi_domain_figure(results, out_dir)
        print(f"Multi-domain figure written to {out_dir}")

    quality = build_fig1_quality_gates(results)
    generated_files: List[Path] = []
    if len(results) > 1:
        generated_files.extend(out_dir / f"fig1_multi_domain_real.{ext}" for ext in ["png", "svg", "pdf"])
    write_run_manifest(
        out_dir,
        figure="fig1",
        argv=sys.argv if argv is None else list(argv),
        inputs={
            "config": [str(Path(p)) for p in args.config],
            "corpus_dir": str(args.corpus_dir) if args.corpus_dir is not None else None,
            "allow_schematic_trajectories": bool(args.allow_schematic_trajectories),
        },
        domains=[str(result.cfg.get("slug")) for result in results],
        quality_gates=quality,
        extra={
            "selected_papers_by_domain": {
                str(result.cfg.get("slug")): len(result.works_selected)
                for result in results
            },
        },
    )
    write_figure_quality_report(
        out_dir,
        figure="fig1",
        generated_files=generated_files,
        quality_gates=quality,
        extra={"domain_reports": [str(out_dir / str(result.cfg.get("slug")) / "figure_quality_report.json") for result in results]},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
