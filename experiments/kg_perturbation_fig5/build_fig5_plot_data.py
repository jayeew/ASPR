#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Fig. 5 plotting data tables without drawing the figure.

The script turns the local strict Fig. 3 multi-domain run into a stable data
interface for manual/external Fig. 5 drawing:

```
outputs/kg_perturbation_fig5/plot_data/
├── base/
├── derived/
└── config/
```

It intentionally keeps plotting concerns out of the pipeline. The output files
are small, explicit contracts for panel a/b/c/d.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIG3_ROOT = PROJECT_ROOT / "outputs" / "kg_perturbation_fig3_strict_broad10"
DEFAULT_FIG3_RUN_DIR = DEFAULT_FIG3_ROOT / "multi_domain"
DEFAULT_FIG3_INPUT_DIR = DEFAULT_FIG3_ROOT / "fig3_input" / "multi_domain"
DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "data" / "knowledge_corpus" / "v1_strict"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "kg_perturbation_fig5" / "plot_data"

SCORE_CANDIDATES = ("S_w_oof", "S_w", "S_equal")
RGPM_CANDIDATES = (
    "RGPM_structural_residual_tau10",
    "RGPM",
    "RGPM_v3_balanced",
    "RGPM_v2",
    "RGPM_simple",
)
PALETTE = [
    "#2563EB",
    "#059669",
    "#D97706",
    "#DC2626",
    "#7C3AED",
    "#0891B2",
    "#4D7C0F",
    "#BE123C",
    "#4338CA",
    "#0F766E",
]
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
    "analysis",
    "approach",
    "application",
    "applications",
}


def read_csv_required(path: Path, name: str) -> pd.DataFrame:
    """Read a required CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return pd.read_csv(path, low_memory=False)


def read_csv_optional(path: Path) -> pd.DataFrame:
    """Read an optional CSV file."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def column_or_default(frame: pd.DataFrame, col: Optional[str], default: object = "") -> pd.Series:
    """Return an existing column or a full-length default series."""
    if col and col in frame.columns:
        return frame[col]
    return pd.Series([default] * len(frame), index=frame.index)


def choose_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    label: str,
    min_coverage: float = 0.05,
    required: bool = True,
) -> Optional[str]:
    """Choose the first usable column from candidates."""
    for col in candidates:
        if col in frame.columns and float(frame[col].notna().mean()) >= min_coverage:
            return col
    if required:
        raise ValueError(f"Could not find a usable {label} column. Tried: {', '.join(candidates)}")
    return None


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    """Return the first existing path from a list."""
    for path in paths:
        if path.exists():
            return path
    return None


def stable_float(value: object) -> float:
    """Return a deterministic pseudo-random float in [0, 1)."""
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def clean_id(value: object) -> str:
    """Normalize ids read from CSV cells."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0", text):
        return text[:-2]
    return text


def make_topic_id(domain: object, community: object) -> str:
    """Create a domain-qualified topic id."""
    return f"{clean_id(domain)}::{clean_id(community)}"


def clean_label(label: object, domain: Optional[object] = None) -> str:
    """Clean labels for display."""
    if pd.isna(label):
        label = ""
    text = re.sub(r"\s+", " ", str(label or "")).strip()
    if domain:
        text = re.sub(rf"^{re.escape(str(domain))}\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^[a-z0-9_ -]+\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("_", " ").strip(" -:/")
    return text or "Unlabelled focus"


def shorten(text: object, max_chars: int = 46) -> str:
    """Shorten a label without splitting words when possible."""
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return textwrap.shorten(cleaned, width=max_chars, placeholder="...")


def title_case_slug(value: object) -> str:
    """Turn a domain slug into a display label."""
    text = str(value or "").replace("_", " ").strip()
    return text.title() if text else "Unknown"


def json_list(values: Iterable[object]) -> str:
    """Serialize a compact list cell for CSV output."""
    return json.dumps([str(item) for item in values if str(item).strip()], ensure_ascii=False)


def keyword_list(label: object, max_items: int = 6) -> str:
    """Extract keyword tokens from a label."""
    tokens = re.findall(r"[A-Za-z0-9]+", str(label).lower())
    keep: List[str] = []
    for token in tokens:
        if len(token) <= 2 or token in STOPWORDS or token in keep:
            continue
        keep.append(token)
    return json_list(keep[:max_items])


def percentile_score(values: pd.Series | np.ndarray) -> pd.Series:
    """Convert numeric values to percentile scores in [0, 1]."""
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


def minmax(values: pd.Series | np.ndarray) -> pd.Series:
    """Min-max normalize numeric values to [0, 1]."""
    series = pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric.dropna()
    out = pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    if valid.empty:
        return out
    span = float(valid.max() - valid.min())
    if span <= 1e-12:
        out.loc[valid.index] = 0.5
        return out
    out.loc[valid.index] = (valid - float(valid.min())) / span
    return out.fillna(0.0)


def top_tail_mean(values: pd.Series, frac: float = 0.20) -> float:
    """Mean of the highest-scoring tail of a numeric series."""
    valid = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return 0.0
    n = max(1, int(math.ceil(len(valid) * frac)))
    return float(valid.sort_values(ascending=False).head(n).mean())


def mode_or_empty(values: pd.Series) -> str:
    """Return the most common non-empty string."""
    cleaned = values.dropna().astype(str)
    cleaned = cleaned[cleaned.str.strip().ne("")]
    if cleaned.empty:
        return ""
    return str(cleaned.mode().iloc[0])


def coalesce_columns(frame: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    """Coalesce candidate columns into one string series."""
    out = pd.Series([""] * len(frame), index=frame.index, dtype="object")
    for col in candidates:
        if col not in frame.columns:
            continue
        series = frame[col].fillna("").astype(str)
        series = series.where(~series.str.lower().isin({"nan", "none", "<na>"}), "")
        out = out.where(out.astype(str).str.strip().ne(""), series)
    return out


def parse_domain_filter(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Normalize an optional domain filter."""
    if not values:
        return None
    domains: List[str] = []
    for value in values:
        domains.extend(part.strip() for part in str(value).split(",") if part.strip())
    return domains or None


def category_for_label(label: object, domain: object) -> str:
    """Assign a simple focus category for color grouping."""
    lower = str(label).lower()
    if any(token in lower for token in ["model", "algorithm", "learning", "prediction", "simulation"]):
        return "computation"
    if any(token in lower for token in ["therapy", "clinical", "patient", "disease", "vaccine"]):
        return "application"
    if any(token in lower for token in ["method", "editing", "sequencing", "synthesis", "measurement"]):
        return "method"
    if any(token in lower for token in ["material", "device", "cell", "protein", "rna", "dna"]):
        return "platform"
    return str(domain)


def color_map_for_groups(groups: Sequence[str]) -> Dict[str, str]:
    """Map color groups to a deterministic palette."""
    unique = sorted({str(group) for group in groups if str(group).strip()})
    return {group: PALETTE[i % len(PALETTE)] for i, group in enumerate(unique)}


def role_for_text(text: object) -> str:
    """Assign a fixed innovation role label."""
    lower = str(text).lower()
    if any(token in lower for token in ["platform", "atlas", "database", "library"]):
        return "platform innovation"
    if any(token in lower for token in ["delivery", "therapy", "clinical", "vaccine", "vector"]):
        return "translational bottleneck breaker"
    if any(token in lower for token in ["model", "algorithm", "learning", "prediction", "simulation"]):
        return "method accelerator"
    if any(token in lower for token in ["hybrid", "interface", "cross", "coupled", "multi"]):
        return "paradigm connector"
    if any(token in lower for token in ["frontier", "emerging", "novel"]):
        return "next frontier"
    return "key enabling innovation"


def icon_for_text(text: object) -> str:
    """Assign a coarse icon type for downstream drawing."""
    lower = str(text).lower()
    if any(token in lower for token in ["model", "algorithm", "learning", "ai", "simulation"]):
        return "computation"
    if any(token in lower for token in ["delivery", "vector", "therapy", "vaccine"]):
        return "delivery"
    if any(token in lower for token in ["platform", "atlas", "database", "library"]):
        return "platform"
    if any(token in lower for token in ["cell", "clinical", "patient", "disease"]):
        return "application"
    if any(token in lower for token in ["cross", "hybrid", "interface", "coupled"]):
        return "cross_domain"
    return "method"


def resolve_default_run_dir() -> Path:
    """Resolve the best local Fig. 3 run directory."""
    candidates = [
        DEFAULT_FIG3_RUN_DIR,
        PROJECT_ROOT / "outputs" / "kg_perturbation_fig3_audit" / "multi_domain",
        PROJECT_ROOT / "outputs" / "kg_perturbation_fig3" / "strong_evidence_tau10_v3" / "multi_domain",
        PROJECT_ROOT / "outputs" / "kg_perturbation_fig3" / "strong_evidence_tau10_v2" / "multi_domain",
        PROJECT_ROOT / "outputs" / "kg_perturbation_fig3" / "strong_evidence_tau10" / "multi_domain",
    ]
    return first_existing(candidates) or candidates[0]


def resolve_default_input_dir(run_dir: Path) -> Path:
    """Resolve the Fig. 3 input directory matching a run."""
    if DEFAULT_FIG3_INPUT_DIR.exists():
        return DEFAULT_FIG3_INPUT_DIR
    candidates = [
        run_dir.parent / "fig3_input" / "multi_domain",
        PROJECT_ROOT / "outputs" / "kg_perturbation_fig3_audit" / "fig3_input" / "multi_domain",
        DEFAULT_CORPUS_ROOT / "views" / "fig5" / "multi_domain",
        DEFAULT_CORPUS_ROOT,
    ]
    return first_existing(candidates) or candidates[0]


def load_domain_display_names(domains: Sequence[str]) -> Dict[str, str]:
    """Load domain display names when the corpus registry is available."""
    mapping = {domain: title_case_slug(domain) for domain in domains}
    path = DEFAULT_CORPUS_ROOT / "domains.csv"
    registry = read_csv_optional(path)
    if {"slug", "display_name"}.issubset(registry.columns):
        mapping.update(dict(zip(registry["slug"].astype(str), registry["display_name"].astype(str))))
    return mapping


def build_work_topic_labels(works: pd.DataFrame) -> pd.Series:
    """Build one clean topic label per paper."""
    labels = pd.Series([""] * len(works), index=works.index, dtype="object")
    for col in ["display_label", "community_label", "primary_topic", "primary_field"]:
        if col not in works.columns:
            continue
        series = works[col].fillna("").astype(str)
        series = series.where(~series.str.lower().isin({"nan", "none", "<na>"}), "")
        labels = labels.where(labels.astype(str).str.strip().ne(""), series)
    return pd.Series([clean_label(label, domain) for label, domain in zip(labels, works["domain"])], index=works.index)


def load_inputs(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, Optional[str]]:
    """Load and normalize core inputs."""
    scores = read_csv_required(args.fig3_run_dir / "fig3_score_table.csv", "Fig. 3 score table")
    works = read_csv_required(args.fig3_input_dir / "works.csv", "Fig. 3 works table")
    topics = read_csv_optional(args.fig3_input_dir / "topics.csv")
    citations = read_csv_optional(args.fig3_input_dir / "citations.csv")
    score_col = choose_column(scores, [args.score_col] if args.score_col else SCORE_CANDIDATES, "score")
    rgpm_col = choose_column(scores, RGPM_CANDIDATES, "RGPM", min_coverage=0.01, required=False)
    if "id" in works.columns and "paper_id" not in works.columns:
        works = works.rename(columns={"id": "paper_id"})
    missing = [col for col in ["paper_id", "year", "domain", "display_community"] if col not in works.columns]
    if missing:
        raise ValueError(f"works.csv is missing required columns: {missing}")
    domain_filter = parse_domain_filter(args.domain_filter)
    if domain_filter:
        works = works[works["domain"].astype(str).isin(set(domain_filter))].copy()
    works["paper_id"] = works["paper_id"].astype(str)
    works["domain"] = works["domain"].astype(str)
    works["year"] = pd.to_numeric(works["year"], errors="coerce").astype("Int64")
    works = works.dropna(subset=["year", "display_community"]).copy()
    works["community_id"] = works["display_community"].map(clean_id)
    works["topic_id"] = [make_topic_id(d, c) for d, c in zip(works["domain"], works["community_id"])]
    for col in ["cited_by_count", "reference_count", "is_landmark"]:
        works[col] = pd.to_numeric(column_or_default(works, col, 0), errors="coerce").fillna(0)
    score_cols = ["paper_id", str(score_col)]
    if rgpm_col:
        score_cols.append(rgpm_col)
    works = works.merge(scores[[col for col in score_cols if col in scores.columns]], on="paper_id", how="left")
    works["selected_score"] = pd.to_numeric(column_or_default(works, str(score_col), np.nan), errors="coerce")
    works["rgpm_score"] = pd.to_numeric(column_or_default(works, rgpm_col, np.nan), errors="coerce")
    works["topic_label"] = build_work_topic_labels(works)
    if works.empty:
        raise ValueError("No works remain after input loading and domain filtering.")
    return works.reset_index(drop=True), topics, citations, scores, str(score_col), rgpm_col


def normalize_topics(topics: pd.DataFrame, works: pd.DataFrame) -> pd.DataFrame:
    """Normalize topic metadata, falling back to work communities."""
    if topics.empty or "community" not in topics.columns:
        base = works[["domain", "community_id", "topic_id", "topic_label"]].drop_duplicates("topic_id")
        base["local_x"] = np.nan
        base["local_y"] = np.nan
        base["source_topic_id"] = ""
        return base
    allowed_domains = set(works["domain"].astype(str).unique())
    out = topics.copy()
    out["domain"] = column_or_default(out, "domain", "").astype(str)
    out = out[out["domain"].isin(allowed_domains)].copy()
    out["community_id"] = out["community"].map(clean_id)
    out["topic_id"] = [make_topic_id(d, c) for d, c in zip(out["domain"], out["community_id"])]
    out["topic_label"] = [clean_label(label, domain) for label, domain in zip(column_or_default(out, "label", ""), out["domain"])]
    out["local_x"] = pd.to_numeric(column_or_default(out, "x", np.nan), errors="coerce")
    out["local_y"] = pd.to_numeric(column_or_default(out, "y", np.nan), errors="coerce")
    out["source_topic_id"] = out["topic_id"].astype(str)
    return out[["domain", "community_id", "topic_id", "topic_label", "local_x", "local_y", "source_topic_id"]].drop_duplicates("topic_id")


def domain_centers(domains: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    """Return deterministic circular domain centers."""
    if len(domains) <= 1:
        return {domain: (0.0, 0.0) for domain in domains}
    radius = 2.2
    return {
        domain: (
            radius * math.cos((2.0 * math.pi * idx) / len(domains)),
            radius * math.sin((2.0 * math.pi * idx) / len(domains)),
        )
        for idx, domain in enumerate(domains)
    }


def local_coordinates(row: pd.Series) -> Tuple[float, float]:
    """Return local topic coordinates with stable fallback."""
    x = row.get("local_x", np.nan)
    y = row.get("local_y", np.nan)
    if pd.notna(x) and pd.notna(y):
        return float(x), float(y)
    key = row.get("topic_id", row.name)
    angle = 2.0 * math.pi * stable_float(f"{key}:angle")
    radius = 0.25 + 0.75 * stable_float(f"{key}:radius")
    return radius * math.cos(angle), radius * math.sin(angle)


def build_topic_nodes(works: pd.DataFrame, topics: pd.DataFrame, cutoff_year: int) -> pd.DataFrame:
    """Build domain-qualified topic nodes for plotting."""
    hist = works[works["year"].astype(int) <= cutoff_year]
    topic_meta = normalize_topics(topics, works)
    total = works.groupby("topic_id").agg(
        n_papers_total=("paper_id", "count"),
        avg_year=("year", "mean"),
        field=("primary_field", mode_or_empty),
        fallback_label=("topic_label", mode_or_empty),
        domain_work=("domain", mode_or_empty),
        community_id_work=("community_id", mode_or_empty),
    )
    hist_counts = hist.groupby("topic_id").size().rename("n_papers_hist")
    counts = total.join(hist_counts, how="left").reset_index()
    nodes = topic_meta.merge(counts, on="topic_id", how="outer", suffixes=("_topic", "_work"))
    nodes["domain"] = coalesce_columns(nodes, ["domain", "domain_topic", "domain_work"])
    nodes["community_id"] = coalesce_columns(nodes, ["community_id", "community_id_topic", "community_id_work"])
    nodes["topic_label"] = coalesce_columns(nodes, ["fallback_label", "topic_label", "topic_label_topic"])
    nodes["topic_label"] = [clean_label(label, domain) for label, domain in zip(nodes["topic_label"], nodes["domain"])]
    nodes["short_label"] = nodes["topic_label"].map(lambda value: shorten(value, 34))
    nodes["n_papers_hist"] = pd.to_numeric(column_or_default(nodes, "n_papers_hist", 0), errors="coerce").fillna(0).astype(int)
    nodes["n_papers_total"] = pd.to_numeric(column_or_default(nodes, "n_papers_total", 0), errors="coerce").fillna(0).astype(int)
    nodes["avg_year"] = pd.to_numeric(column_or_default(nodes, "avg_year", np.nan), errors="coerce")
    nodes["field"] = column_or_default(nodes, "field", "").fillna("").astype(str)
    nodes["local_x"] = pd.to_numeric(column_or_default(nodes, "local_x", np.nan), errors="coerce")
    nodes["local_y"] = pd.to_numeric(column_or_default(nodes, "local_y", np.nan), errors="coerce")
    nodes["raw_community_id"] = nodes["community_id"].astype(str)
    nodes["source_topic_id"] = column_or_default(nodes, "source_topic_id", "").fillna("").astype(str)
    display_names = load_domain_display_names(sorted(works["domain"].astype(str).unique()))
    nodes["domain_label"] = nodes["domain"].map(display_names).fillna(nodes["domain"].map(title_case_slug))
    centers = domain_centers(sorted(nodes["domain"].astype(str).unique()))
    scale = 0.34 if len(centers) > 1 else 1.0
    local_xy = [local_coordinates(row) for _, row in nodes.iterrows()]
    nodes["local_x"] = [item[0] for item in local_xy]
    nodes["local_y"] = [item[1] for item in local_xy]
    nodes["x"] = [centers.get(str(domain), (0.0, 0.0))[0] + scale * x for domain, x in zip(nodes["domain"], nodes["local_x"])]
    nodes["y"] = [centers.get(str(domain), (0.0, 0.0))[1] + scale * y for domain, y in zip(nodes["domain"], nodes["local_y"])]
    nodes["topic_keywords"] = nodes["topic_label"].map(keyword_list)
    cols = [
        "topic_id",
        "topic_label",
        "short_label",
        "n_papers_hist",
        "n_papers_total",
        "avg_year",
        "field",
        "domain",
        "domain_label",
        "x",
        "y",
        "local_x",
        "local_y",
        "community_id",
        "raw_community_id",
        "source_topic_id",
        "topic_keywords",
    ]
    return nodes[cols].sort_values(["domain", "n_papers_hist", "topic_label"], ascending=[True, False, True]).reset_index(drop=True)


def build_papers_master(works: pd.DataFrame) -> pd.DataFrame:
    """Build the common paper-level base table."""
    out = pd.DataFrame()
    out["paper_id"] = works["paper_id"].astype(str)
    out["title"] = column_or_default(works, "title", "").fillna("").astype(str)
    out["abstract"] = column_or_default(works, "abstract", "").fillna("").astype(str)
    out["year"] = works["year"].astype(int)
    out["venue"] = column_or_default(works, "venue", "").fillna("").astype(str)
    out["authors"] = column_or_default(works, "authors", "").fillna("").astype(str)
    out["references"] = column_or_default(works, "referenced_works", "").fillna("").astype(str)
    out["cited_by_count"] = pd.to_numeric(column_or_default(works, "cited_by_count", 0), errors="coerce").fillna(0).astype(int)
    out["keywords"] = works["topic_label"].map(keyword_list)
    out["field"] = column_or_default(works, "primary_field", "").fillna("").astype(str)
    out["domain"] = works["domain"].astype(str)
    out["topic_id"] = works["topic_id"].astype(str)
    out["community_id"] = works["community_id"].astype(str)
    out["topic_label"] = works["topic_label"].astype(str)
    out["is_landmark"] = pd.to_numeric(column_or_default(works, "is_landmark", 0), errors="coerce").fillna(0).astype(int)
    out["reference_count"] = pd.to_numeric(column_or_default(works, "reference_count", 0), errors="coerce").fillna(0).astype(int)
    out["selected_score"] = pd.to_numeric(column_or_default(works, "selected_score", np.nan), errors="coerce")
    out["rgpm_score"] = pd.to_numeric(column_or_default(works, "rgpm_score", np.nan), errors="coerce")
    return out


def build_citation_edges(citations: pd.DataFrame, works: pd.DataFrame) -> pd.DataFrame:
    """Build paper citation edges with source and target years."""
    cols = ["source_paper_id", "target_paper_id", "source_year", "target_year"]
    if citations.empty or not {"source", "target"}.issubset(citations.columns):
        return pd.DataFrame(columns=cols)
    year_map = works.set_index("paper_id")["year"].astype(int).to_dict()
    out = pd.DataFrame()
    out["source_paper_id"] = citations["source"].astype(str)
    out["target_paper_id"] = citations["target"].astype(str)
    out["source_year"] = out["source_paper_id"].map(year_map)
    out["target_year"] = out["target_paper_id"].map(year_map)
    return out[cols]


def build_forecast_focus(works: pd.DataFrame, topic_nodes: pd.DataFrame, cutoff_year: int, min_historical_papers: int) -> pd.DataFrame:
    """Aggregate historical topic signals into forecast focus rows."""
    hist = works[works["year"].astype(int) <= cutoff_year].copy()
    recent = hist[hist["year"].astype(int).between(cutoff_year - 2, cutoff_year)]
    prior = hist[hist["year"].astype(int).between(cutoff_year - 7, cutoff_year - 3)]
    agg = hist.groupby("topic_id").agg(
        focus_id=("topic_id", "first"),
        historical_size=("paper_id", "count"),
        historical_citations=("cited_by_count", "sum"),
        hist_scored_papers=("selected_score", lambda values: int(pd.to_numeric(values, errors="coerce").notna().sum())),
        hist_mean_score=("selected_score", "mean"),
        hist_max_score=("selected_score", "max"),
        hist_top_tail_score=("selected_score", top_tail_mean),
    )
    agg = agg.join(recent.groupby("topic_id").size().rename("recent_hist_size"), how="left")
    agg = agg.join(prior.groupby("topic_id").size().rename("prior_hist_size"), how="left").reset_index(drop=True)
    focus = topic_nodes.merge(agg, left_on="topic_id", right_on="focus_id", how="left")
    focus["focus_id"] = focus["topic_id"].astype(str)
    focus["focus_label"] = focus["topic_label"].astype(str)
    for col in [
        "historical_size",
        "historical_citations",
        "hist_scored_papers",
        "hist_mean_score",
        "hist_max_score",
        "hist_top_tail_score",
        "recent_hist_size",
        "prior_hist_size",
    ]:
        focus[col] = pd.to_numeric(column_or_default(focus, col, 0), errors="coerce").fillna(0)
    focus["score_coverage"] = focus["hist_scored_papers"] / focus["historical_size"].clip(lower=1)
    score_p = percentile_score(focus["hist_top_tail_score"])
    size_p = percentile_score(np.log1p(focus["historical_size"]))
    citation_p = percentile_score(np.log1p(focus["historical_citations"]))
    momentum_p = percentile_score(np.log1p((focus["recent_hist_size"] + 1.0) / (focus["prior_hist_size"] + 1.0)))
    low_data_penalty = ((min_historical_papers - focus["historical_size"]).clip(lower=0) / max(1, min_historical_papers))
    focus["forecast_score_raw"] = 0.62 * score_p + 0.18 * size_p + 0.10 * citation_p + 0.10 * momentum_p
    focus["forecast_score_raw"] -= 0.18 * low_data_penalty + 0.08 * (1.0 - focus["score_coverage"].clip(0.0, 1.0))
    eligible = focus["historical_size"] >= int(min_historical_papers)
    focus["forecast_score"] = 0.0
    if eligible.any():
        focus.loc[eligible, "forecast_score"] = minmax(focus.loc[eligible, "forecast_score_raw"]).to_numpy()
    focus["forecast_rank"] = pd.Series(pd.NA, index=focus.index, dtype="Float64")
    focus.loc[eligible, "forecast_rank"] = focus.loc[eligible, "forecast_score"].rank(ascending=False, method="first")
    focus["short_label"] = focus["focus_label"].map(lambda value: shorten(value, 34))
    focus["source_topic_ids"] = focus["focus_id"].map(lambda value: json_list([value]))
    focus["keyword_list"] = focus["focus_label"].map(keyword_list)
    focus["category"] = [category_for_label(label, domain) for label, domain in zip(focus["focus_label"], focus["domain"])]
    focus["focus_group"] = focus["category"].astype(str)
    colors = color_map_for_groups(focus["focus_group"])
    focus["display_color"] = focus["focus_group"].map(colors)
    focus["description"] = [
        f"{shorten(label, 40)} supported by {int(size)} pre-cutoff papers."
        for label, size in zip(focus["focus_label"], focus["historical_size"])
    ]
    cols = [
        "focus_id",
        "focus_label",
        "short_label",
        "forecast_score",
        "forecast_score_raw",
        "forecast_rank",
        "source_topic_ids",
        "keyword_list",
        "category",
        "focus_group",
        "description",
        "domain",
        "domain_label",
        "historical_size",
        "historical_citations",
        "hist_top_tail_score",
        "hist_mean_score",
        "hist_scored_papers",
        "score_coverage",
        "recent_hist_size",
        "prior_hist_size",
        "n_papers_hist",
        "x",
        "y",
        "display_color",
    ]
    return focus[cols].sort_values(["forecast_rank", "focus_label"], na_position="last").reset_index(drop=True)


def build_focus_topic_mapping(forecast_focus: pd.DataFrame) -> pd.DataFrame:
    """Build a focus-topic mapping table."""
    out = pd.DataFrame()
    out["focus_id"] = forecast_focus["focus_id"].astype(str)
    out["topic_id"] = forecast_focus["focus_id"].astype(str)
    out["contribution_weight"] = 1.0
    out["is_primary"] = 1
    return out


def qualify_topic_edges(edges: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Qualify community-level topic edges for one domain."""
    required = {"source_community", "target_community", "weight"}
    if edges.empty or not required.issubset(edges.columns):
        return pd.DataFrame(columns=["source_topic_id", "target_topic_id", "edge_weight", "edge_type"])
    out = pd.DataFrame()
    out["source_topic_id"] = [make_topic_id(domain, value) for value in edges["source_community"]]
    out["target_topic_id"] = [make_topic_id(domain, value) for value in edges["target_community"]]
    out["edge_weight"] = pd.to_numeric(edges["weight"], errors="coerce").fillna(0.0)
    out["edge_type"] = "topic_citation"
    return out


def build_topic_edges(fig3_input_dir: Path, works: pd.DataFrame, topic_nodes: pd.DataFrame, forecast_focus: pd.DataFrame, frontier_top_n: int) -> pd.DataFrame:
    """Build domain-qualified topic edges and frontier flags."""
    frames: List[pd.DataFrame] = []
    domains = sorted(works["domain"].astype(str).unique())
    if fig3_input_dir.name == "multi_domain":
        for domain in domains:
            frames.append(qualify_topic_edges(read_csv_optional(fig3_input_dir.parent / domain / "topic_edges.csv"), domain))
    elif len(domains) == 1:
        frames.append(qualify_topic_edges(read_csv_optional(fig3_input_dir / "topic_edges.csv"), domains[0]))
    edges = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    cols = ["source_topic_id", "target_topic_id", "edge_weight", "edge_type", "source_forecast_score", "target_forecast_score", "is_predicted_frontier"]
    if edges.empty:
        return pd.DataFrame(columns=cols)
    valid_topics = set(topic_nodes["topic_id"].astype(str))
    edges = edges[edges["source_topic_id"].isin(valid_topics) & edges["target_topic_id"].isin(valid_topics)].copy()
    if edges.empty:
        return pd.DataFrame(columns=cols)
    score_map = forecast_focus.set_index("focus_id")["forecast_score"].to_dict()
    top_topics = set(forecast_focus.loc[forecast_focus["forecast_rank"].le(frontier_top_n), "focus_id"].astype(str))
    q80 = float(pd.to_numeric(edges["edge_weight"], errors="coerce").quantile(0.80))
    edges["source_forecast_score"] = edges["source_topic_id"].map(score_map).fillna(0.0)
    edges["target_forecast_score"] = edges["target_topic_id"].map(score_map).fillna(0.0)
    edges["is_predicted_frontier"] = [
        int((src in top_topics and dst in top_topics) or ((src in top_topics or dst in top_topics) and float(weight) >= q80))
        for src, dst, weight in zip(edges["source_topic_id"], edges["target_topic_id"], edges["edge_weight"])
    ]
    return edges[cols].sort_values(["is_predicted_frontier", "edge_weight"], ascending=[False, False]).reset_index(drop=True)


def build_panel_b_top_focus(forecast_focus: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Build Panel b direct plotting data."""
    top = forecast_focus[forecast_focus["forecast_rank"].notna()].sort_values("forecast_rank").head(top_n).copy()
    top["color_group"] = top["focus_group"]
    cols = [
        "forecast_rank",
        "focus_id",
        "focus_label",
        "short_label",
        "forecast_score",
        "forecast_score_raw",
        "color_group",
        "display_color",
        "keyword_list",
        "domain",
    ]
    return top[cols].reset_index(drop=True)


def build_panel_c_focus_positions(
    forecast_focus: pd.DataFrame,
    topic_nodes: pd.DataFrame,
    mapping: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Build weighted focus positions for Panel c."""
    top_focus = forecast_focus[forecast_focus["forecast_rank"].le(top_n)].copy()
    joined = mapping.merge(topic_nodes, on="topic_id", how="left")
    joined["contribution_weight"] = pd.to_numeric(joined["contribution_weight"], errors="coerce").fillna(1.0)
    focus_by_id = top_focus.set_index("focus_id")
    rows: List[Dict[str, Any]] = []
    for focus_id, group in joined[joined["focus_id"].isin(set(top_focus["focus_id"]))].groupby("focus_id"):
        weights = group["contribution_weight"].clip(lower=0.0)
        total = float(weights.sum()) or 1.0
        focus = focus_by_id.loc[focus_id]
        norm = float(focus.get("forecast_score", 0.0))
        rows.append(
            {
                "focus_id": str(focus_id),
                "focus_label": focus["focus_label"],
                "short_label": focus["short_label"],
                "forecast_rank": int(focus["forecast_rank"]),
                "plot_x": float((pd.to_numeric(group["x"], errors="coerce").fillna(0.0) * weights).sum() / total),
                "plot_y": float((pd.to_numeric(group["y"], errors="coerce").fillna(0.0) * weights).sum() / total),
                "forecast_score": norm,
                "hist_size": int(float(focus.get("historical_size", 0))),
                "n_support_topics": int(len(group)),
                "support_topic_ids": json_list(group["topic_id"]),
                "display_size": round(70.0 + 430.0 * math.sqrt(max(0.0, norm)), 3),
                "display_color_value": norm,
                "display_color": focus["display_color"],
                "domain": focus["domain"],
                "color_group": focus["focus_group"],
            }
        )
    return pd.DataFrame(rows).sort_values("forecast_rank").reset_index(drop=True)


def innovation_row(row: pd.Series, rank: int) -> Dict[str, Any]:
    """Build one predicted innovation candidate row."""
    title = str(row.get("title", "") or "Untitled seed paper")
    focus_label = str(row.get("focus_label", "") or row.get("topic_label", ""))
    combined = f"{title} {focus_label}"
    paper_id = str(row.get("paper_id", ""))
    focus_id = str(row.get("focus_id", row.get("topic_id", "")))
    return {
        "innovation_id": f"innovation_{rank}",
        "innovation_label": shorten(title, 72),
        "short_label": shorten(title, 42),
        "forecast_rank": rank,
        "predicted_role": role_for_text(combined),
        "short_reason": f"High-scoring seed for {shorten(focus_label, 32)}.",
        "linked_focus_id": focus_id,
        "linked_focus_label": focus_label,
        "linked_topic_ids": json_list([focus_id]),
        "representative_papers": json_list([paper_id]),
        "icon_type": icon_for_text(combined),
        "description": f"{shorten(title, 58)} anchors the predicted focus {shorten(focus_label, 34)}.",
        "seed_year": int(row["year"]),
        "seed_score": float(row.get("selected_score_filled", 0.0)),
        "color_group": row.get("focus_group", ""),
        "display_color": row.get("display_color", ""),
    }


def build_forecast_innovations(
    works: pd.DataFrame,
    forecast_focus: pd.DataFrame,
    cutoff_year: int,
    top_focus_limit: int,
) -> pd.DataFrame:
    """Select representative pre-cutoff seed papers as key innovations."""
    top_focus = forecast_focus[forecast_focus["forecast_rank"].le(top_focus_limit)].copy()
    focus_ids = set(top_focus["focus_id"].astype(str))
    seeds = works[(works["topic_id"].isin(focus_ids)) & (works["year"].astype(int) <= cutoff_year)].copy()
    if seeds.empty:
        return pd.DataFrame()
    seeds["selected_score_filled"] = pd.to_numeric(seeds["selected_score"], errors="coerce")
    seeds["selected_score_filled"] = seeds["selected_score_filled"].fillna(percentile_score(seeds["cited_by_count"]))
    seeds = seeds.merge(
        top_focus[["focus_id", "focus_label", "short_label", "forecast_rank", "focus_group", "display_color"]],
        left_on="topic_id",
        right_on="focus_id",
        how="left",
    )
    seeds = seeds.sort_values(["forecast_rank", "selected_score_filled", "cited_by_count"], ascending=[True, False, False])
    rows: List[Dict[str, Any]] = []
    used_focus: set[str] = set()
    for _, row in seeds.iterrows():
        if str(row["focus_id"]) in used_focus:
            continue
        rows.append(innovation_row(row, len(rows) + 1))
        used_focus.add(str(row["focus_id"]))
    return pd.DataFrame(rows)


def build_panel_d_cards(forecast_innovations: pd.DataFrame, card_count: int) -> pd.DataFrame:
    """Build card-ready rows for Panel d."""
    cols = [
        "display_rank",
        "innovation_id",
        "innovation_label",
        "short_label",
        "predicted_role",
        "short_reason",
        "linked_focus_label",
        "icon_type",
        "color_group",
        "display_color",
        "representative_papers",
        "seed_year",
    ]
    if forecast_innovations.empty:
        return pd.DataFrame(columns=cols)
    top = forecast_innovations.sort_values("forecast_rank").head(card_count).copy()
    top["display_rank"] = range(1, len(top) + 1)
    return top[cols].reset_index(drop=True)


def build_panel_a_meta(
    works: pd.DataFrame,
    topic_nodes: pd.DataFrame,
    forecast_focus: pd.DataFrame,
    forecast_innovations: pd.DataFrame,
    args: argparse.Namespace,
    score_col: str,
) -> Dict[str, Any]:
    """Build metadata for Panel a."""
    hist = works[works["year"].astype(int) <= args.cutoff_year]
    available_max_year = int(works["year"].max())
    return {
        "historical_start_year": int(hist["year"].min()) if not hist.empty else int(args.historical_start_year),
        "historical_start_year_requested": int(args.historical_start_year),
        "historical_end_year": int(args.cutoff_year),
        "future_start_year": int(args.future_start_year or args.cutoff_year + 1),
        "future_end_year": int(args.future_end_year),
        "future_end_year_actual": int(min(args.future_end_year, available_max_year)),
        "n_hist_papers": int(len(hist)),
        "n_hist_topics": int((topic_nodes["n_papers_hist"] > 0).sum()),
        "n_predicted_focus": int(forecast_focus["forecast_rank"].notna().sum()),
        "n_predicted_innovations": int(len(forecast_innovations)),
        "n_domains": int(works["domain"].nunique()),
        "score_column": score_col,
        "warnings": [],
    }


def relpath(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_config(args: argparse.Namespace, works: pd.DataFrame, score_col: str, rgpm_col: Optional[str]) -> Dict[str, Any]:
    """Build a JSON-serializable run configuration."""
    return {
        "fig3_run_dir": relpath(args.fig3_run_dir),
        "fig3_input_dir": relpath(args.fig3_input_dir),
        "out_dir": relpath(args.out_dir),
        "score_column": score_col,
        "rgpm_column": rgpm_col,
        "cutoff_year": int(args.cutoff_year),
        "historical_start_year": int(args.historical_start_year),
        "future_start_year": int(args.future_start_year or args.cutoff_year + 1),
        "future_end_year": int(args.future_end_year),
        "top_focus_n": int(args.top_focus_n),
        "panel_c_top_n": int(args.panel_c_top_n),
        "innovation_card_n": int(args.innovation_card_n),
        "min_historical_papers": int(args.min_historical_papers),
        "domain_filter": parse_domain_filter(args.domain_filter),
        "n_input_papers": int(len(works)),
        "domains": sorted(works["domain"].astype(str).unique()),
        "topic_id_policy": "domain::display_community",
        "layout_policy": "domain-offset local topic layout for multi-domain inputs",
    }


def write_outputs(out_dir: Path, tables: Dict[str, Any]) -> None:
    """Write base, derived, and config files."""
    base_dir = out_dir / "base"
    derived_dir = out_dir / "derived"
    config_dir = out_dir / "config"
    for path in [base_dir, derived_dir, config_dir]:
        path.mkdir(parents=True, exist_ok=True)
    tables["papers_master"].to_csv(base_dir / "papers_master.csv", index=False)
    tables["citation_edges"].to_csv(base_dir / "citation_edges.csv", index=False)
    tables["topic_nodes"].to_csv(base_dir / "topic_nodes.csv", index=False)
    tables["topic_edges"].to_csv(base_dir / "topic_edges.csv", index=False)
    tables["focus_topic_mapping"].to_csv(base_dir / "focus_topic_mapping.csv", index=False)
    tables["forecast_focus"].to_csv(derived_dir / "forecast_focus.csv", index=False)
    tables["forecast_innovations"].to_csv(derived_dir / "forecast_innovations.csv", index=False)
    tables["panel_b"].to_csv(derived_dir / "fig5_panel_b_top_focus.csv", index=False)
    tables["panel_c"].to_csv(derived_dir / "fig5_panel_c_focus_positions.csv", index=False)
    tables["panel_d"].to_csv(derived_dir / "fig5_panel_d_cards.csv", index=False)
    (derived_dir / "fig5_panel_a_meta.json").write_text(json.dumps(tables["panel_a"], indent=2, ensure_ascii=False), encoding="utf-8")
    (config_dir / "fig5_config.json").write_text(json.dumps(tables["config"], indent=2, ensure_ascii=False), encoding="utf-8")


def compute_tables(args: argparse.Namespace) -> Dict[str, Any]:
    """Compute all Fig. 5 plotting data tables."""
    works, topics, citations, _scores, score_col, rgpm_col = load_inputs(args)
    topic_nodes = build_topic_nodes(works, topics, args.cutoff_year)
    papers_master = build_papers_master(works)
    citation_edges = build_citation_edges(citations, works)
    forecast_focus = build_forecast_focus(works, topic_nodes, args.cutoff_year, args.min_historical_papers)
    focus_topic_mapping = build_focus_topic_mapping(forecast_focus)
    topic_edges = build_topic_edges(args.fig3_input_dir, works, topic_nodes, forecast_focus, args.panel_c_top_n)
    panel_b = build_panel_b_top_focus(forecast_focus, args.top_focus_n)
    panel_c = build_panel_c_focus_positions(forecast_focus, topic_nodes, focus_topic_mapping, args.panel_c_top_n)
    forecast_innovations = build_forecast_innovations(works, forecast_focus, args.cutoff_year, args.top_focus_n)
    panel_d = build_panel_d_cards(forecast_innovations, args.innovation_card_n)
    panel_a = build_panel_a_meta(works, topic_nodes, forecast_focus, forecast_innovations, args, score_col)
    config = build_config(args, works, score_col, rgpm_col)
    return {
        "papers_master": papers_master,
        "citation_edges": citation_edges,
        "topic_nodes": topic_nodes,
        "topic_edges": topic_edges,
        "focus_topic_mapping": focus_topic_mapping,
        "forecast_focus": forecast_focus,
        "forecast_innovations": forecast_innovations,
        "panel_b": panel_b,
        "panel_c": panel_c,
        "panel_d": panel_d,
        "panel_a": panel_a,
        "config": config,
    }


def print_summary(out_dir: Path, tables: Dict[str, Any]) -> None:
    """Print a short command-line summary."""
    print("[fig5-data] wrote", out_dir)
    print("[fig5-data] papers:", len(tables["papers_master"]))
    print("[fig5-data] topics:", len(tables["topic_nodes"]))
    print("[fig5-data] topic edges:", len(tables["topic_edges"]))
    print("[fig5-data] forecast foci:", len(tables["forecast_focus"]))
    print("[fig5-data] innovation candidates:", len(tables["forecast_innovations"]))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    default_run_dir = resolve_default_run_dir()
    parser = argparse.ArgumentParser(description="Build Fig. 5 plotting data tables.")
    parser.add_argument("--fig3-run-dir", type=Path, default=default_run_dir, help="Directory containing fig3_score_table.csv.")
    parser.add_argument("--fig3-input-dir", type=Path, default=resolve_default_input_dir(default_run_dir), help="Directory containing works/topics/topic_edges/citations CSVs.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output root for fig5 plot data.")
    parser.add_argument("--domain-filter", nargs="+", help="Optional domain slugs or comma-separated domain list.")
    parser.add_argument("--score-col", default=None, help="Optional score column override; defaults to S_w_oof then S_w.")
    parser.add_argument("--historical-start-year", type=int, default=1950, help="Conceptual historical window start year.")
    parser.add_argument("--cutoff-year", type=int, default=2020, help="Historical graph cutoff year T0.")
    parser.add_argument("--future-start-year", type=int, default=None, help="Future window start year; default cutoff + 1.")
    parser.add_argument("--future-end-year", type=int, default=2026, help="Requested future window end year.")
    parser.add_argument("--top-focus-n", type=int, default=10, help="Number of top foci for Panel b.")
    parser.add_argument("--panel-c-top-n", type=int, default=8, help="Number of highlighted foci for Panel c.")
    parser.add_argument("--innovation-card-n", type=int, default=4, help="Number of card rows for Panel d.")
    parser.add_argument("--min-historical-papers", type=int, default=5, help="Minimum pre-cutoff papers for ranking a focus.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    args.fig3_run_dir = args.fig3_run_dir.resolve()
    args.fig3_input_dir = args.fig3_input_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    tables = compute_tables(args)
    write_outputs(args.out_dir, tables)
    print_summary(args.out_dir, tables)


if __name__ == "__main__":
    main(sys.argv[1:])
