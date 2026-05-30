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

    export OPENALEX_API_KEY="YOUR_KEY"
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Configuration defaults
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    "start_year": 2000,
    "end_year": 2024,
    "window_size": 5,
    "work_types": [],  # Example: ["article", "preprint", "review"]
    "language": "en",
    "include_abstract": False,
    "max_works_per_window": 1200,
    "max_anchor_citers": 500,
    "fetch_anchor_citers": True,
    "anchors": [],
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
        "title": "Landmark papers induce measurable perturbations in citation knowledge graphs",
        "subtitle": "Expansion, bridging, reconfiguration and compression across cumulative citation-knowledge snapshots",
        "show_retrieval_date": True,
    },
}

METRIC_NAMES = ["Expansion", "Bridging", "Reconfiguration", "Compression"]
METRIC_COLORS = {
    "Expansion": "#3B82F6",
    "Bridging": "#F97316",
    "Reconfiguration": "#10B981",
    "Compression": "#8B5CF6",
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
    return cfg


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "domain"


def normalize_openalex_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = str(value).strip()
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


def make_cumulative_windows(start_year: int, rolling_windows: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [(start_year, end) for _, end in rolling_windows]


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
    max_topics = int(pcfg.get("display_max_topics", cfg.get("graph", {}).get("max_communities", 9)))
    min_size = int(pcfg.get("display_min_topic_size", cfg.get("graph", {}).get("min_community_size", 6)))
    members = community_members(full_comm_map)
    weighted_deg = dict(G.degree(weight="weight"))
    scored: List[Tuple[float, int]] = []
    anchor_comms: Set[int] = set()
    for c, nodes in members.items():
        has_anchor = any(G.nodes[n].get("anchor_label") for n in nodes if n in G)
        if has_anchor:
            anchor_comms.add(int(c))
        if len(nodes) < min_size and not has_anchor:
            continue
        citation_score = sum(safe_log1p(float(G.nodes[n].get("cited_by_count") or 0)) for n in nodes if n in G)
        degree_score = sum(float(weighted_deg.get(n, 0.0)) for n in nodes)
        score = 3.0 * safe_log1p(len(nodes)) + 0.45 * safe_log1p(citation_score) + 0.35 * safe_log1p(degree_score)
        if has_anchor:
            score += 1000.0
        scored.append((score, int(c)))
    scored.sort(reverse=True)
    keep: List[int] = []
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
    out = []
    for key, (u, v, d) in chosen.items():
        is_new = key not in prev_edges
        out.append((u, v, d, is_new))
    out.sort(key=lambda x: (x[3], float(x[2].get("weight", 1.0))), reverse=False)
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
        email: Optional[str] = None,
        sleep_seconds: float = 0.1,
        max_retries: int = 6,
        timeout_seconds: int = 60,
    ):
        if not api_key:
            raise ValueError(
                "OPENALEX_API_KEY is required. Create a free key at openalex.org/settings/api "
                "and pass it with --openalex-api-key or export OPENALEX_API_KEY."
            )
        self.api_key = api_key
        self.email = email
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "kg-perturbation-fig1/1.0"})

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = dict(params or {})
        params["api_key"] = self.api_key
        if self.email:
            params["mailto"] = self.email
        url = f"{self.BASE_URL.rstrip('/')}/{path.lstrip('/')}"

        for attempt in range(self.max_retries + 1):
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
    }


def mark_anchors(works: Dict[str, Dict[str, Any]], anchors: Sequence[Mapping[str, Any]]) -> None:
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
) -> Dict[str, Dict[str, Any]]:
    slug = cfg["slug"]
    domain_dir = out_dir / slug
    domain_dir.mkdir(parents=True, exist_ok=True)
    cache_path = domain_dir / "works_raw.jsonl"

    if use_cache and cache_path.exists():
        works: Dict[str, Dict[str, Any]] = {}
        with open(cache_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("id"):
                    works[rec["id"]] = rec
        mark_anchors(works, cfg.get("anchors") or [])
        print(f"[{slug}] Loaded {len(works):,} cached works from {cache_path}")
        return works

    rolling = make_rolling_windows(cfg["start_year"], cfg["end_year"], cfg["window_size"])
    raw_by_id: Dict[str, Dict[str, Any]] = {}
    api_cfg = cfg.get("api", {})

    for y0, y1 in rolling:
        print(f"[{slug}] Fetching search window {y0}-{y1} ...")
        records = client.list_works(
            search_query=cfg["search_query"],
            from_year=y0,
            to_year=y1,
            max_records=int(cfg["max_works_per_window"]),
            work_types=cfg.get("work_types") or [],
            language=cfg.get("language"),
            include_abstract=bool(cfg.get("include_abstract")),
            sort=None,  # default OpenAlex search ranking mixes relevance and citations
            per_page=int(api_cfg.get("per_page", 100)),
        )
        for r in records:
            wid = normalize_openalex_id(r.get("id"))
            if wid:
                raw_by_id[wid] = r

    # Always add known landmark papers, even if the keyword query misses them.
    for a in cfg.get("anchors") or []:
        rec = None
        if a.get("doi"):
            print(f"[{slug}] Fetching anchor DOI {a.get('doi')} ...")
            rec = client.get_work_by_doi(str(a["doi"]), include_abstract=bool(cfg.get("include_abstract")))
        elif a.get("openalex_id"):
            print(f"[{slug}] Fetching anchor OpenAlex ID {a.get('openalex_id')} ...")
            rec = client.get_work_by_openalex_id(str(a["openalex_id"]), include_abstract=bool(cfg.get("include_abstract")))
        if rec and rec.get("id"):
            raw_by_id[normalize_openalex_id(rec["id"])] = rec

    # Optional: fetch citing papers of anchors to better capture downstream disturbance.
    if cfg.get("fetch_anchor_citers", True) and cfg.get("anchors"):
        anchor_records: List[Dict[str, Any]] = []
        for a in cfg.get("anchors") or []:
            rec = None
            if a.get("doi"):
                rec = client.get_work_by_doi(str(a["doi"]), include_abstract=False)
            elif a.get("openalex_id"):
                rec = client.get_work_by_openalex_id(str(a["openalex_id"]), include_abstract=False)
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
                    raw_by_id[wid] = r

    works: Dict[str, Dict[str, Any]] = {}
    for raw in raw_by_id.values():
        rec = normalize_work(raw, include_abstract=bool(cfg.get("include_abstract")))
        if rec.get("id") and rec.get("year"):
            works[rec["id"]] = rec

    mark_anchors(works, cfg.get("anchors") or [])
    with open(cache_path, "w", encoding="utf-8") as f:
        for rec in works.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
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

    G = prune_graph_edges(G, int(gcfg.get("max_edges", 120000)))
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
        modularity_shift = abs(curr_modularity - prev_modularity) if i > 0 else 0.0
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
                path_gain = max(0.0, prev_path - curr_path)
            else:
                path_gain = 0.0
            hub_gain = max(0.0, curr_hub - prev_hub)
            sem_gain = max(0.0, prev_sem - curr_sem)
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
                "intercommunity_edge_ratio": inter_ratio,
                "participation_mean": part_mean,
                "bridge_betweenness_top": bridge_bc,
                "partition_change": partition_change,
                "edge_turnover": edge_turnover,
                "modularity": curr_modularity,
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
        seen_topics |= curr_topics

    df = pd.DataFrame(rows)
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

    ax.set_title(panel_label.replace("1-", "0-"), fontsize=10.5, pad=8, fontweight="bold")
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
        ends = [e for _, e in make_cumulative_windows(cfg["start_year"], make_rolling_windows(cfg["start_year"], cfg["end_year"], cfg["window_size"]))]
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


def draw_metric_panel(ax: plt.Axes, metrics: pd.DataFrame, cfg: Dict[str, Any], compact: bool = False) -> None:
    pcfg = cfg.get("plot", {})
    if pcfg.get("metric_x_axis", "years") == "years":
        x = 0.5 * (metrics["rolling_start"].values.astype(float) + metrics["rolling_end"].values.astype(float))
        labels = [f"{int(a)}-{int(b)}" for a, b in zip(metrics["rolling_start"], metrics["rolling_end"])]
        ax.set_xlim(float(cfg["start_year"]), float(cfg["end_year"] + 1))
        ax.set_xlabel("Publication year / rolling citation-graph window", fontsize=8.5 if not compact else 7)
    else:
        x = np.arange(len(metrics))
        labels = [str(v).split("\n")[0] for v in metrics["label"].tolist()]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8 if not compact else 7)
        ax.set_xlabel("Cumulative window", fontsize=8.5 if not compact else 7)

    for metric in METRIC_NAMES:
        y = metrics[f"{metric}_index"].values
        # Use 0-1 axis in compact schematic style; keep 0-100 if requested.
        if pcfg.get("metric_scale", "unit") == "unit":
            y = y / 100.0
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=1.9 if not compact else 1.3,
            markersize=4.2 if not compact else 3.2,
            label={
                "Expansion": "Expansion: new nodes / edges",
                "Bridging": "Bridging: cross-community paths",
                "Reconfiguration": "Reconfiguration: community turnover",
                "Compression": "Compression: path shortening",
            }.get(metric, metric) if not compact else metric,
            color=METRIC_COLORS[metric],
        )

    unit = pcfg.get("metric_scale", "unit") == "unit"
    ax.set_ylim(-0.03 if unit else -5, 1.08 if unit else 105)
    ax.set_ylabel("Normalized\nperturbation score" if unit else "Perturbation\nintensity", fontsize=8.5 if not compact else 7)
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("b  Perturbation fingerprint across rolling 5-year intervals", fontsize=10.5 if not compact else 8, fontweight="bold", loc="left")

    # Mark the landmark event interval.
    anchor_years = [year_int(a.get("year")) for a in cfg.get("anchors") or []]
    anchor_years = [y for y in anchor_years if y]
    if anchor_years:
        y0, y1 = min(anchor_years) - 0.6, max(anchor_years) + 0.9
        ax.axvspan(y0, y1, color="#FEE2E2", alpha=0.55, lw=0)
        ylim = ax.get_ylim()
        ax.text((y0 + y1) / 2, ylim[1] * 0.96, "landmark\ninnovation", ha="center", va="top", fontsize=7.4, color="#DC2626", fontweight="bold")

    if not compact:
        ax.legend(ncol=2, fontsize=8, loc="upper left", frameon=False, columnspacing=1.6)
        ax.text(
            0.985,
            0.47,
            "Operational readout for a real dataset:\n"
            "Expansion = growth of field-specific nodes/edges\n"
            "Bridging = increase in inter-module paths\n"
            "Reconfiguration = modularity/community change\n"
            "Compression = shorter semantic/citation paths",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=7.2,
            color="#4B5563",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#D1D5DB", alpha=0.90),
        )
    else:
        ax.legend(fontsize=6.5, loc="upper left", frameon=False)


def draw_top_time_axis(ax: plt.Axes, result: "DomainResult") -> None:
    cfg = result.cfg
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
        ax.text((x0 + x1) / 2, 0.86, "2012-2013\nlandmark papers", ha="center", va="top", fontsize=8.6, color="#B91C1C", fontweight="bold")
    ax.text(cfg["end_year"] + 0.75, y - 0.20, "yr", ha="left", va="top", fontsize=8, color="#60636B")


def draw_single_domain_figure(result: "DomainResult", out_dir: Path) -> None:
    cfg = result.cfg
    pcfg = cfg["plot"]
    domain_dir = out_dir / cfg["slug"]
    domain_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(pcfg["fig_width_single"], pcfg["fig_height_single"]), dpi=pcfg["dpi"])
    gs = GridSpec(3, 5, figure=fig, height_ratios=[0.44, 2.95, 1.35], hspace=0.30, wspace=0.13)

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
    fig.text(0.012, 0.765, "a", fontsize=12, fontweight="bold")
    fig.text(0.03, 0.765, "Cumulative citation knowledge graph snapshots", fontsize=11, fontweight="bold")

    for i, (_, end) in enumerate(result.cumulative_windows):
        prev_end = result.cumulative_windows[i - 1][1] if i > 0 else None
        ax = fig.add_subplot(gs[1, i])
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
    ncols = len(results[0].cumulative_windows) + 1
    pcfg = results[0].cfg["plot"]
    fig_w = float(pcfg.get("fig_width_multi", 26))
    fig_h = float(pcfg.get("row_height_multi", 4.1)) * nrows
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(pcfg.get("dpi", 300)))
    gs = GridSpec(
        nrows,
        ncols,
        figure=fig,
        width_ratios=[1, 1, 1, 1, 1, 1.15],
        hspace=0.25,
        wspace=0.13,
    )

    fig.suptitle(
        "Landmark discoveries induce reproducible perturbations in citation knowledge graphs",
        y=0.995,
        fontsize=16,
        fontweight="bold",
    )

    for r, result in enumerate(results):
        cfg = result.cfg
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
                cfg,
                end_year=end,
                prev_end_year=prev_end,
                panel_label=window_label(cfg["start_year"], end, cfg["start_year"]) if r == 0 else "",
                show_ylabel=(i == 0),
            )
        axm = fig.add_subplot(gs[r, ncols - 1])
        draw_metric_panel(axm, result.metrics, cfg, compact=True)
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


def run_domain(cfg: Dict[str, Any], client: OpenAlexClient, out_dir: Path, use_cache: bool = True) -> DomainResult:
    slug = cfg["slug"]
    print(f"\n=== Running domain: {cfg['domain_name']} ({slug}) ===")
    rolling = make_rolling_windows(cfg["start_year"], cfg["end_year"], cfg["window_size"])
    cumulative = make_cumulative_windows(cfg["start_year"], rolling)

    works_all = fetch_domain_works(cfg, client, out_dir, use_cache=use_cache)
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
    parser.add_argument("--openalex-api-key", default=os.getenv("OPENALEX_API_KEY"), help="OpenAlex API key.")
    parser.add_argument("--email", default=os.getenv("OPENALEX_EMAIL"), help="Optional contact email for API calls.")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached works_raw.jsonl and re-download data.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load configs first so API behavior can be controlled by the first config.
    cfgs = [load_config(Path(p)) for p in args.config]
    first_api = cfgs[0].get("api", {})
    client = OpenAlexClient(
        api_key=args.openalex_api_key,
        email=args.email,
        sleep_seconds=float(first_api.get("sleep_seconds", 0.1)),
        max_retries=int(first_api.get("max_retries", 6)),
        timeout_seconds=int(first_api.get("timeout_seconds", 60)),
    )

    results = []
    for cfg in cfgs:
        result = run_domain(cfg, client, out_dir, use_cache=not args.no_cache)
        results.append(result)

    if len(results) > 1:
        draw_multi_domain_figure(results, out_dir)
        print(f"Multi-domain figure written to {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
