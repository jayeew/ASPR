#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "knowledge_corpus" / "v1_strict"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "corpus_screening" / "v2_strict"
FIG3_SCRIPT = PROJECT_ROOT / "experiments" / "kg_perturbation_fig3" / "fig3_empirical_weight_learning.py"

BIOMED_HINTS = (
    "medicine",
    "biology",
    "genetic",
    "cell",
    "oncology",
    "microbiology",
    "immunology",
    "biochemistry",
    "genomics",
    "therapy",
    "vaccine",
    "organoid",
    "crispr",
)
MATERIALS_HINTS = (
    "materials",
    "energy",
    "chemical",
    "battery",
    "perovskite",
    "graphene",
    "electrocatalysis",
    "hydrogen",
    "thin_films",
)
AI_HINTS = (
    "computer science",
    "artificial intelligence",
    "transformer",
    "diffusion",
    "foundation",
    "graph_neural",
    "recommendation",
)
PHYSICS_HINTS = (
    "physics",
    "astronomy",
    "exoplanet",
    "gravitational",
    "gamma_ray",
    "topological",
    "quantum",
)
SOCIAL_HINTS = (
    "economic",
    "policy",
    "banking",
    "finance",
    "poverty",
    "inequality",
    "gender",
    "labor",
    "causal",
    "bayesian",
)


@dataclass
class CorpusTables:
    works: pd.DataFrame
    citations: pd.DataFrame
    topics: pd.DataFrame
    domains: pd.DataFrame
    landmarks: pd.DataFrame
    quality_report: Dict[str, Any]


def progress_log(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[screen] {message}", flush=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_doi(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"", "nan", "none", "null", "<na>"}:
        return ""
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text, flags=re.I)
    return text.replace("doi:", "").strip()


def normalize_id(value: object) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    if text.startswith("https://openalex.org/"):
        return text.rstrip("/")
    match = re.search(r"(W\d+)$", text)
    return f"https://openalex.org/{match.group(1)}" if match else text


def normalize_title(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def nonempty_text(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.lower() not in {"nan", "none", "null", "<na>"}


def safe_numeric(series: Any, default: float = 0.0) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    return pd.Series(dtype=float)


def percentile_rank(values: pd.Series, high_is_good: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series(0.5, index=values.index)
    rank = numeric.rank(pct=True, method="average")
    return rank if high_is_good else 1.0 - rank


def minmax(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(0.0, index=values.index)
    lo = float(numeric.min())
    hi = float(numeric.max())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=values.index)
    return ((numeric - lo) / (hi - lo)).clip(0.0, 1.0).fillna(0.0)


def clipped_ratio(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    if not math.isfinite(float(value)):
        return 0.0
    return float(max(0.0, min(1.0, value / cap)))


def clipped_ratio_series(values: pd.Series, cap: float) -> pd.Series:
    if cap <= 0:
        return pd.Series(0.0, index=values.index)
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return (numeric / float(cap)).clip(lower=0.0, upper=1.0)


def load_corpus(corpus_dir: Path) -> CorpusTables:
    works = read_csv(corpus_dir / "works.csv")
    citations = read_csv(corpus_dir / "citations.csv")
    topics = read_csv(corpus_dir / "topics.csv")
    domains = read_csv(corpus_dir / "domains.csv")
    landmarks = read_csv(corpus_dir / "landmarks.csv")
    quality_path = corpus_dir / "quality_report.json"
    quality_report = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
    if works.empty:
        raise FileNotFoundError(f"No works.csv found in {corpus_dir}")
    for col in ["id", "domain", "title", "doi", "anchor_label", "display_topic_label", "primary_field"]:
        if col not in works.columns:
            works[col] = ""
    for col in ["year", "display_community", "is_landmark", "cited_by_count", "reference_count", "partial_2026"]:
        if col not in works.columns:
            works[col] = 0
    if citations.empty:
        citations = pd.DataFrame(columns=["source", "target"])
    for col in ["source", "target"]:
        if col not in citations.columns:
            citations[col] = ""
    return CorpusTables(works, citations, topics, domains, landmarks, quality_report)


def complete_works(works: pd.DataFrame, complete_end_year: int) -> pd.DataFrame:
    out = works.copy()
    out["id"] = out["id"].map(normalize_id)
    out["doi_norm"] = out["doi"].map(normalize_doi)
    out["title_norm"] = out["title"].map(normalize_title)
    out["year"] = safe_numeric(out["year"]).astype(int)
    out["display_community"] = safe_numeric(out["display_community"], -1).astype(int)
    out["is_landmark"] = safe_numeric(out["is_landmark"]).astype(int)
    out["cited_by_count"] = safe_numeric(out["cited_by_count"]).astype(float)
    out["reference_count"] = safe_numeric(out["reference_count"]).astype(float)
    out["partial_2026"] = safe_numeric(out["partial_2026"]).astype(int)
    out = out[(out["year"] <= int(complete_end_year)) & (out["partial_2026"] == 0)].copy()
    return out.reset_index(drop=True)


def _landmark_keys(landmarks: pd.DataFrame, complete_end_year: int) -> pd.DataFrame:
    if landmarks.empty:
        return pd.DataFrame(columns=["domain", "id_norm", "doi_norm", "title_norm", "label", "has_label"])
    lm = landmarks.copy()
    for col in ["domain", "id", "doi", "title", "label"]:
        if col not in lm.columns:
            lm[col] = ""
    if "include_main" in lm.columns:
        lm = lm[safe_numeric(lm["include_main"], 1).astype(int) != 0].copy()
    if "year" in lm.columns:
        lm = lm[safe_numeric(lm["year"], complete_end_year).astype(int) <= int(complete_end_year)].copy()
    lm["id_norm"] = lm["id"].map(normalize_id)
    lm["doi_norm"] = lm["doi"].map(normalize_doi)
    lm["title_norm"] = lm["title"].map(normalize_title)
    lm["has_label"] = lm["label"].map(nonempty_text)
    return lm[["domain", "id_norm", "doi_norm", "title_norm", "label", "has_label"]].drop_duplicates()


def parse_year_from_label(value: object) -> Optional[int]:
    match = re.search(r"(19\d{2}|20\d{2})", str(value or ""))
    return int(match.group(1)) if match else None


def annotate_reliable_anchors(
    works: pd.DataFrame,
    landmarks: pd.DataFrame,
    complete_end_year: int = 2025,
    noisy_ratio: float = 0.25,
) -> pd.DataFrame:
    """Mark reliable anchor papers while correcting noisy legacy landmark flags."""
    out = complete_works(works, complete_end_year)
    lm = _landmark_keys(landmarks, complete_end_year)
    out["anchor_label_clean"] = out["anchor_label"].fillna("").astype(str).str.strip()
    out["has_anchor_label"] = out["anchor_label_clean"].map(nonempty_text)
    if "anchor_policy" in out.columns and out["anchor_policy"].fillna("").astype(str).eq("strict").any():
        is_landmark = pd.to_numeric(out.get("is_landmark", 0), errors="coerce").fillna(0).astype(int)
        legacy = pd.to_numeric(out.get("legacy_is_landmark", is_landmark), errors="coerce").fillna(0).astype(int)
        out["domain_anchor_flags_noisy"] = legacy.groupby(out["domain"].astype(str)).transform("mean") > float(noisy_ratio)
        out["anchor_label_representative"] = out["has_anchor_label"] & (is_landmark == 1)
        out["landmark_id_match"] = False
        out["landmark_doi_match"] = False
        out["landmark_title_match"] = False
        out["landmark_labeled_match"] = out["anchor_label_representative"]
        out["landmark_match"] = out["landmark_labeled_match"]
        out["is_landmark_reliable"] = is_landmark == 1
        out["reliable_anchor"] = is_landmark
        if "reliable_anchor_source" not in out.columns:
            out["reliable_anchor_source"] = np.where(is_landmark == 1, "strict_is_landmark", "")
        else:
            out["reliable_anchor_source"] = out["reliable_anchor_source"].fillna("")
            out.loc[(is_landmark == 1) & (out["reliable_anchor_source"].astype(str) == ""), "reliable_anchor_source"] = "strict_is_landmark"
        return out
    out["landmark_id_match"] = False
    out["landmark_doi_match"] = False
    out["landmark_title_match"] = False
    out["landmark_labeled_match"] = False
    if not lm.empty:
        keyed = lm.groupby("domain", sort=False)
        for domain, lsub in keyed:
            mask = out["domain"].astype(str) == str(domain)
            if not mask.any():
                continue
            ids = {v for v in lsub["id_norm"].astype(str) if nonempty_text(v)}
            dois = {v for v in lsub["doi_norm"].astype(str) if nonempty_text(v)}
            titles = {v for v in lsub["title_norm"].astype(str) if nonempty_text(v)}
            labeled = lsub[lsub["has_label"].astype(bool)]
            labeled_ids = {v for v in labeled["id_norm"].astype(str) if nonempty_text(v)}
            labeled_dois = {v for v in labeled["doi_norm"].astype(str) if nonempty_text(v)}
            labeled_titles = {v for v in labeled["title_norm"].astype(str) if nonempty_text(v)}
            id_match = out.loc[mask, "id"].isin(ids)
            doi_match = out.loc[mask, "doi_norm"].isin(dois)
            title_match = out.loc[mask, "title_norm"].isin(titles)
            labeled_match = (
                out.loc[mask, "id"].isin(labeled_ids)
                | out.loc[mask, "doi_norm"].isin(labeled_dois)
                | out.loc[mask, "title_norm"].isin(labeled_titles)
            )
            out.loc[mask, "landmark_id_match"] = id_match.to_numpy(dtype=bool)
            out.loc[mask, "landmark_doi_match"] = doi_match.to_numpy(dtype=bool)
            out.loc[mask, "landmark_title_match"] = title_match.to_numpy(dtype=bool)
            out.loc[mask, "landmark_labeled_match"] = labeled_match.to_numpy(dtype=bool)

    ratios = (
        out.groupby("domain", sort=False)
        .agg(
            is_landmark_ratio=("is_landmark", "mean"),
            anchor_label_ratio=("has_anchor_label", "mean"),
            registry_match_ratio=("landmark_id_match", "mean"),
        )
        .reset_index()
    )
    generic_landmark_match = out["landmark_id_match"] | out["landmark_doi_match"] | out["landmark_title_match"]
    registry_ratios = generic_landmark_match.groupby(out["domain"].astype(str)).mean()
    ratios = ratios.merge(registry_ratios.rename("registry_match_ratio"), left_on="domain", right_index=True, how="left", suffixes=("", "_actual"))
    ratios["registry_match_ratio"] = ratios["registry_match_ratio_actual"].fillna(ratios["registry_match_ratio"]).fillna(0.0)
    noisy_domains = set(
        ratios[
            (ratios["is_landmark_ratio"] > float(noisy_ratio))
            | (ratios["anchor_label_ratio"] > float(noisy_ratio))
            | (ratios["registry_match_ratio"] > float(noisy_ratio))
        ]["domain"].astype(str)
    )
    out["domain_anchor_flags_noisy"] = out["domain"].astype(str).isin(noisy_domains)
    out["anchor_label_representative"] = False

    labeled = out[out["has_anchor_label"]].copy()
    if not labeled.empty:
        labeled["label_year"] = labeled["anchor_label_clean"].map(parse_year_from_label)
        reps: List[pd.DataFrame] = []
        for (domain, label), group in labeled.groupby(["domain", "anchor_label_clean"], sort=False):
            group = group.copy()
            label_years = [int(v) for v in group["label_year"].dropna().unique()]
            if str(domain) in noisy_domains and label_years:
                year = label_years[0]
                near = group[(group["year"] >= year - 1) & (group["year"] <= year + 1)].copy()
                if not near.empty:
                    group = near
            keep_n = 2 if str(domain) in noisy_domains else len(group)
            group = group.sort_values(["cited_by_count", "reference_count"], ascending=[False, False]).head(keep_n)
            reps.append(group)
        if reps:
            rep_ids = set(pd.concat(reps)["id"].astype(str))
            out["anchor_label_representative"] = out["id"].astype(str).isin(rep_ids)

    out["landmark_match"] = out["landmark_labeled_match"] | (
        generic_landmark_match & ~out["domain_anchor_flags_noisy"]
    )
    out["is_landmark_reliable"] = (out["is_landmark"].astype(int) == 1) & ~out["domain_anchor_flags_noisy"]
    out["reliable_anchor"] = (
        out["landmark_match"] | out["anchor_label_representative"] | out["is_landmark_reliable"]
    ).astype(int)
    source = np.where(out["landmark_labeled_match"], "landmarks_csv_labeled", "")
    source = np.where(
        (source == "") & generic_landmark_match & ~out["domain_anchor_flags_noisy"],
        "landmarks_csv",
        source,
    )
    source = np.where((source == "") & out["anchor_label_representative"], "anchor_label_representative", source)
    source = np.where((source == "") & out["is_landmark_reliable"], "is_landmark", source)
    out["reliable_anchor_source"] = source
    return out


def annotate_citations(citations: pd.DataFrame, works: pd.DataFrame) -> pd.DataFrame:
    if citations.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "source_domain",
                "target_domain",
                "source_year",
                "source_community",
                "target_community",
            ]
        )
    meta = works.set_index("id")[["domain", "year", "display_community"]].to_dict("index")
    out = citations[["source", "target"]].copy()
    out["source"] = out["source"].map(normalize_id)
    out["target"] = out["target"].map(normalize_id)
    out["source_domain"] = out["source"].map(lambda x: meta.get(x, {}).get("domain", ""))
    out["target_domain"] = out["target"].map(lambda x: meta.get(x, {}).get("domain", ""))
    out["source_year"] = out["source"].map(lambda x: meta.get(x, {}).get("year", np.nan))
    out["source_community"] = out["source"].map(lambda x: meta.get(x, {}).get("display_community", np.nan))
    out["target_community"] = out["target"].map(lambda x: meta.get(x, {}).get("display_community", np.nan))
    return out[(out["source_domain"] != "") & (out["target_domain"] != "")].reset_index(drop=True)


def topic_shift(pre: pd.DataFrame, post: pd.DataFrame) -> float:
    if pre.empty or post.empty:
        return 0.0
    pre_counts = pre["display_community"].value_counts(normalize=True)
    post_counts = post["display_community"].value_counts(normalize=True)
    keys = sorted(set(pre_counts.index).union(set(post_counts.index)))
    return float(0.5 * sum(abs(float(post_counts.get(k, 0.0)) - float(pre_counts.get(k, 0.0))) for k in keys))


def window_edge_stats(citations: pd.DataFrame, ids: set[str], n_papers: int) -> Tuple[float, float]:
    if not ids or citations.empty or n_papers <= 0:
        return 0.0, 0.0
    sub = citations[citations["source"].isin(ids)].copy()
    if sub.empty:
        return 0.0, 0.0
    density = float(len(sub) / max(1, n_papers))
    cross = safe_numeric(sub["source_community"], -1).astype(int) != safe_numeric(sub["target_community"], -1).astype(int)
    return density, float(cross.mean())


def event_metrics_for_anchor(
    anchor: Mapping[str, Any],
    works_domain: pd.DataFrame,
    citations_domain: pd.DataFrame,
    window: int,
) -> Dict[str, Any]:
    year = int(anchor.get("year", 0) or 0)
    anchor_id = str(anchor.get("id") or "")
    pre = works_domain[(works_domain["year"] >= year - int(window)) & (works_domain["year"] < year)].copy()
    post = works_domain[(works_domain["year"] > year) & (works_domain["year"] <= year + int(window))].copy()
    pre_ids = set(pre["id"].astype(str))
    post_ids = set(post["id"].astype(str))
    pre_density, pre_cross = window_edge_stats(citations_domain, pre_ids, len(pre))
    post_density, post_cross = window_edge_stats(citations_domain, post_ids, len(post))
    pre_topics = set(pre["display_community"].astype(int)) if not pre.empty else set()
    post_topics = set(post["display_community"].astype(int)) if not post.empty else set()
    new_topic_share = float(len(post_topics - pre_topics) / max(1, len(post_topics)))
    anchor_citers = citations_domain[
        (citations_domain["target"] == anchor_id)
        & (safe_numeric(citations_domain["source_year"], 0).astype(int) > year)
        & (safe_numeric(citations_domain["source_year"], 0).astype(int) <= year + int(window))
    ].copy()
    citer_count = int(anchor_citers["source"].nunique()) if not anchor_citers.empty else 0
    citer_topic_breadth = int(anchor_citers["source_community"].nunique()) if not anchor_citers.empty else 0
    shift = topic_shift(pre, post)
    cross_delta = abs(post_cross - pre_cross)
    density_delta = abs(post_density - pre_density)
    raw_score = (
        0.30 * shift
        + 0.18 * cross_delta
        + 0.16 * new_topic_share
        + 0.16 * clipped_ratio(math.log1p(citer_count), math.log1p(500))
        + 0.10 * clipped_ratio(citer_topic_breadth, 50)
        + 0.10 * clipped_ratio(density_delta, 8)
    )
    return {
        "event_window": int(window),
        "pre_papers": int(len(pre)),
        "post_papers": int(len(post)),
        "topic_shift": shift,
        "new_topic_share": new_topic_share,
        "cross_share_pre": pre_cross,
        "cross_share_post": post_cross,
        "cross_share_abs_delta": cross_delta,
        "citation_density_pre": pre_density,
        "citation_density_post": post_density,
        "citation_density_abs_delta": density_delta,
        "anchor_citers": citer_count,
        "anchor_citer_topic_breadth": citer_topic_breadth,
        "event_proxy_raw": raw_score,
    }


def compute_paper_event_candidates(
    works: pd.DataFrame,
    citations: pd.DataFrame,
    top_papers_per_domain: int,
    max_anchor_candidates_per_domain: int = 200,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    same_domain_citations = citations[citations["source_domain"] == citations["target_domain"]].copy()
    for domain, sub in works.groupby("domain", sort=True):
        sub = sub.copy()
        csub = same_domain_citations[same_domain_citations["source_domain"].astype(str) == str(domain)].copy()
        anchors = sub[sub["reliable_anchor"].astype(int) == 1].copy()
        if anchors.empty:
            anchors = sub.sort_values("cited_by_count", ascending=False).head(max(5, top_papers_per_domain)).copy()
            anchors["reliable_anchor_source"] = "fallback_high_cited"
        else:
            anchors = anchors.sort_values(["cited_by_count", "reference_count"], ascending=[False, False]).head(
                int(max_anchor_candidates_per_domain)
            )
        for _, anchor in anchors.iterrows():
            best: Dict[str, Any] = {}
            for window in (5, 10):
                metrics = event_metrics_for_anchor(anchor, sub, csub, window)
                if not best or metrics["event_proxy_raw"] > best["event_proxy_raw"]:
                    best = metrics
            rows.append(
                {
                    "domain": domain,
                    "paper_id": anchor.get("id", ""),
                    "title": anchor.get("title", ""),
                    "year": int(anchor.get("year", 0) or 0),
                    "doi": anchor.get("doi_norm", ""),
                    "display_topic_label": anchor.get("display_topic_label", ""),
                    "primary_field": anchor.get("primary_field", ""),
                    "anchor_label": anchor.get("anchor_label_clean", ""),
                    "reliable_anchor": int(anchor.get("reliable_anchor", 0) or 0),
                    "reliable_anchor_source": anchor.get("reliable_anchor_source", ""),
                    "cited_by_count": float(anchor.get("cited_by_count", 0) or 0),
                    "reference_count": float(anchor.get("reference_count", 0) or 0),
                    **best,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["event_proxy_score"] = minmax(out["event_proxy_raw"])
    out["citation_proxy_score"] = out.groupby("domain")["cited_by_count"].transform(lambda s: percentile_rank(s))
    out["paper_proxy_score"] = (
        0.62 * out["event_proxy_score"]
        + 0.18 * out["citation_proxy_score"]
        + 0.12 * clipped_series(out["anchor_citers"], 500)
        + 0.08 * clipped_series(out["anchor_citer_topic_breadth"], 50)
    )
    out["domain_event_rank"] = out.groupby("domain")["paper_proxy_score"].rank(method="first", ascending=False).astype(int)
    out["global_event_rank"] = out["paper_proxy_score"].rank(method="first", ascending=False).astype(int)
    out = out[out["domain_event_rank"] <= int(top_papers_per_domain)].copy()
    return out.sort_values(["global_event_rank", "domain_event_rank"]).reset_index(drop=True)


def clipped_series(values: pd.Series, cap: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return (numeric / float(cap)).clip(0.0, 1.0)


def domain_category(row: Mapping[str, Any]) -> str:
    text = " ".join(
        str(row.get(col, ""))
        for col in ["domain", "display_name", "field_name", "subfield_name", "primary_field", "query"]
    ).lower()
    if any(h in text for h in AI_HINTS):
        return "ai_cs"
    if any(h in text for h in BIOMED_HINTS):
        return "biomed"
    if any(h in text for h in MATERIALS_HINTS):
        return "materials_energy"
    if any(h in text for h in PHYSICS_HINTS):
        return "physics_space"
    if any(h in text for h in SOCIAL_HINTS):
        return "social_econ"
    return "methods_other"


def quality_by_domain(quality_report: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    rows = quality_report.get("domains") or []
    return {str(row.get("domain")): row for row in rows if row.get("domain")}


def compute_domain_proxy_scores(
    works: pd.DataFrame,
    citations: pd.DataFrame,
    domains: pd.DataFrame,
    paper_events: pd.DataFrame,
    quality_report: Mapping[str, Any],
    min_domain_papers: int,
    min_citations_per_work: float,
) -> pd.DataFrame:
    qmap = quality_by_domain(quality_report)
    rows: List[Dict[str, Any]] = []
    same_domain = citations[citations["source_domain"] == citations["target_domain"]].copy()
    domain_meta = domains.set_index("slug").to_dict("index") if not domains.empty and "slug" in domains.columns else {}
    for domain, sub in works.groupby("domain", sort=True):
        sub = sub.copy()
        ids = set(sub["id"].astype(str))
        csub = same_domain[same_domain["source"].isin(ids)].copy()
        doi = sub["doi_norm"].replace("", np.nan)
        duplicate_doi_rate = float(doi.duplicated().sum() / max(1, doi.notna().sum()))
        years = safe_numeric(sub["year"])
        event_sub = paper_events[paper_events["domain"].astype(str) == str(domain)] if not paper_events.empty else pd.DataFrame()
        reliable = sub[sub["reliable_anchor"].astype(int) == 1]
        qrow = qmap.get(str(domain), {})
        meta = domain_meta.get(str(domain), {})
        row = {
            "domain": domain,
            "display_name": meta.get("display_name", domain),
            "category": domain_category({"domain": domain, **meta}),
            "n_works": int(len(sub)),
            "year_min": int(years.min()) if len(sub) else 0,
            "year_max": int(years.max()) if len(sub) else 0,
            "year_span": int(years.max() - years.min()) if len(sub) else 0,
            "topic_count": int(sub["display_community"].nunique()),
            "citation_rows": int(len(csub)),
            "citation_rows_per_work": float(len(csub) / max(1, len(sub))),
            "duplicate_doi_rate": duplicate_doi_rate,
            "partial_2026_rows": int(safe_numeric(sub.get("partial_2026", 0)).sum()),
            "reliable_anchor_count": int(len(reliable)),
            "anchor_flag_noisy": bool(sub["domain_anchor_flags_noisy"].any()),
            "quality_gate_pass": bool(qrow.get("passes", False)),
            "quality_topic_coverage": float(qrow.get("topic_coverage", np.nan)) if qrow else np.nan,
            "top_event_proxy": float(event_sub["paper_proxy_score"].max()) if not event_sub.empty else 0.0,
            "mean_top5_event_proxy": float(event_sub.head(5)["paper_proxy_score"].mean()) if not event_sub.empty else 0.0,
            "top_anchor_citers": int(event_sub["anchor_citers"].max()) if not event_sub.empty else 0,
        }
        row["eligible_proxy"] = int(
            row["n_works"] >= int(min_domain_papers)
            and row["citation_rows_per_work"] >= float(min_citations_per_work)
            and row["reliable_anchor_count"] > 0
            and row["partial_2026_rows"] == 0
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["data_quality_score"] = (
        0.22 * clipped_series(out["n_works"], 3000)
        + 0.22 * clipped_series(out["citation_rows_per_work"], 15)
        + 0.14 * clipped_series(out["topic_count"], 120)
        + 0.14 * clipped_series(out["year_span"], 30)
        + 0.18 * (1.0 - clipped_series(out["duplicate_doi_rate"], 0.12))
        + 0.10 * out["quality_gate_pass"].astype(float)
    )
    out["anchor_score"] = clipped_series(np.log1p(out["reliable_anchor_count"]), math.log1p(12))
    out["event_score"] = minmax(out["top_event_proxy"])
    out["overall_proxy_score"] = (
        0.38 * out["data_quality_score"]
        + 0.36 * out["event_score"]
        + 0.18 * out["anchor_score"]
        + 0.08 * minmax(out["mean_top5_event_proxy"])
    )
    out.loc[out["eligible_proxy"] == 0, "overall_proxy_score"] *= 0.65
    return out.sort_values("overall_proxy_score", ascending=False).reset_index(drop=True)


def set_score(domains: Sequence[str], domain_scores: pd.DataFrame) -> Dict[str, Any]:
    frame = domain_scores.set_index("domain").loc[list(domains)].copy()
    categories = sorted(frame["category"].astype(str).unique().tolist())
    diversity = min(1.0, len(categories) / min(4, max(1, len(frame))))
    score = (
        0.45 * float(frame["overall_proxy_score"].mean())
        + 0.20 * float(frame["top_event_proxy"].max())
        + 0.15 * diversity
        + 0.10 * float(frame["data_quality_score"].mean())
        + 0.10 * min(1.0, float(frame["reliable_anchor_count"].sum()) / 20.0)
    )
    return {
        "set_proxy_score": score,
        "mean_domain_proxy_score": float(frame["overall_proxy_score"].mean()),
        "max_event_proxy": float(frame["top_event_proxy"].max()),
        "mean_data_quality_score": float(frame["data_quality_score"].mean()),
        "category_diversity_score": diversity,
        "categories": ",".join(categories),
        "n_reliable_anchors": int(frame["reliable_anchor_count"].sum()),
        "min_domain_works": int(frame["n_works"].min()),
        "mean_citations_per_work": float(frame["citation_rows_per_work"].mean()),
    }


def generate_candidate_domain_sets(
    domain_scores: pd.DataFrame,
    sizes: Sequence[int],
    top_domains: int,
    beam_width: int,
) -> pd.DataFrame:
    eligible = domain_scores[domain_scores["eligible_proxy"].astype(int) == 1].head(int(top_domains)).copy()
    if eligible.empty:
        eligible = domain_scores.head(int(top_domains)).copy()
    names = eligible["domain"].astype(str).tolist()
    max_size = min(max(int(s) for s in sizes), len(names)) if names else 0
    beams: List[Tuple[str, ...]] = [tuple()]
    records: List[Dict[str, Any]] = []
    score_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for size in range(1, max_size + 1):
        expanded: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        for current in beams:
            for name in names:
                if name in current:
                    continue
                candidate = tuple(sorted(current + (name,)))
                if len(candidate) != size:
                    continue
                if candidate not in score_cache:
                    score_cache[candidate] = set_score(candidate, domain_scores)
                expanded[candidate] = score_cache[candidate]
        ranked = sorted(expanded.items(), key=lambda item: item[1]["set_proxy_score"], reverse=True)
        beams = [item[0] for item in ranked[: int(beam_width)]]
        if size in {int(s) for s in sizes}:
            for rank, (candidate, metrics) in enumerate(ranked[: int(beam_width)], start=1):
                cid = f"set{size:02d}_rank{rank:02d}"
                records.append(
                    {
                        "candidate_id": cid,
                        "set_size": size,
                        "proxy_rank": rank,
                        "domains": " ".join(candidate),
                        "domain_count": len(candidate),
                        **metrics,
                    }
                )
    out = pd.DataFrame(records)
    return out.sort_values(["set_size", "set_proxy_score"], ascending=[True, False]).reset_index(drop=True)


def strict_fixed_candidate_definitions() -> Dict[str, List[str]]:
    return {
        "strict_core6_finance": [
            "crispr",
            "financial_markets_and_investment_strategies",
            "graphene_2d_materials",
            "ipsc_reprogramming",
            "perovskite_solar_cells",
            "transformer_foundation_models",
        ],
        "strict_core6_finance_no_transformer": [
            "crispr",
            "financial_markets_and_investment_strategies",
            "graphene_2d_materials",
            "ipsc_reprogramming",
            "perovskite_solar_cells",
            "autophagy_in_disease_and_therapy",
        ],
        "strict_core6_income": [
            "crispr",
            "income_poverty_and_inequality",
            "graphene_2d_materials",
            "ipsc_reprogramming",
            "perovskite_solar_cells",
            "transformer_foundation_models",
        ],
        "strict_core6_monetary": [
            "crispr",
            "monetary_policy_and_economic_impact",
            "graphene_2d_materials",
            "ipsc_reprogramming",
            "perovskite_solar_cells",
            "transformer_foundation_models",
        ],
        "strict_no_transformer8": [
            "crispr",
            "graphene_2d_materials",
            "ipsc_reprogramming",
            "perovskite_solar_cells",
            "financial_markets_and_investment_strategies",
            "income_poverty_and_inequality",
            "autophagy_in_disease_and_therapy",
            "ubiquitin_and_proteasome_pathways",
        ],
        "strict_broad10": [
            "crispr",
            "graphene_2d_materials",
            "ipsc_reprogramming",
            "perovskite_solar_cells",
            "financial_markets_and_investment_strategies",
            "income_poverty_and_inequality",
            "autophagy_in_disease_and_therapy",
            "ubiquitin_and_proteasome_pathways",
            "genome_wide_association_studies",
            "gravitational_waves",
        ],
    }


def append_strict_fixed_candidate_sets(
    candidate_sets: pd.DataFrame,
    domain_scores: pd.DataFrame,
) -> pd.DataFrame:
    available = set(domain_scores["domain"].astype(str))
    rows: List[Dict[str, Any]] = []
    for candidate_id, domains in strict_fixed_candidate_definitions().items():
        missing = [domain for domain in domains if domain not in available]
        if missing:
            continue
        ordered = tuple(domains)
        metrics = set_score(ordered, domain_scores)
        rows.append(
            {
                "candidate_id": candidate_id,
                "set_size": len(ordered),
                "proxy_rank": 0,
                "domains": " ".join(ordered),
                "domain_count": len(ordered),
                "fixed_candidate": 1,
                **metrics,
            }
        )
    out = candidate_sets.copy()
    if "fixed_candidate" not in out.columns:
        out["fixed_candidate"] = 0
    if rows:
        fixed = pd.DataFrame(rows)
        out = pd.concat([fixed, out], ignore_index=True, sort=False)
        out = out.drop_duplicates("candidate_id", keep="first")
    return out.sort_values(["fixed_candidate", "set_proxy_score"], ascending=[False, False]).reset_index(drop=True)


def fig3_command(args: argparse.Namespace, domains: Sequence[str], run_dir: Path) -> List[str]:
    command = [
        sys.executable,
        str(FIG3_SCRIPT),
        "--data-dir",
        str(args.corpus_dir / "views" / "fig3"),
        "--domains",
        *domains,
        "--run-mode",
        "multi_domain",
        "--out-dir",
        str(run_dir),
        "--audit-only",
        "--export-tables",
        "--skip-sensitivity",
        "--n-weight-samples",
        str(args.fig3_weight_samples),
        "--min-controls",
        str(args.fig3_min_controls),
        "--tau",
        str(args.fig3_tau),
        "--delta-variant",
        str(args.fig3_delta_variant),
        "--progress-interval",
        str(args.fig3_progress_interval),
    ]
    if args.max_papers:
        command.extend(["--max-papers", str(args.max_papers)])
    if args.quiet:
        command.append("--quiet")
    return command


def fig3_subprocess_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    thread_limit = args.fig3_thread_limit
    if thread_limit is None and int(args.fig3_parallel_runs) > 1:
        thread_limit = max(1, (os.cpu_count() or 1) // int(args.fig3_parallel_runs))
    if thread_limit is not None and int(thread_limit) > 0:
        value = str(int(thread_limit))
        for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            env[name] = value
    return env


def run_one_fig3_candidate(args: argparse.Namespace, row: Mapping[str, Any]) -> Dict[str, Any]:
    candidate_id = str(row["candidate_id"])
    domains = str(row["domains"]).split()
    run_dir = args.out_dir / "fig3_runs" / candidate_id
    summary_path = run_dir / "multi_domain" / "fig3_diagnostics_summary.json"
    command = fig3_command(args, domains, run_dir)
    if not summary_path.exists() or args.force_fig3:
        progress_log(f"Running Fig. 3 audit for {candidate_id}: {', '.join(domains)}", args.quiet)
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "fig3_run.log").open("w", encoding="utf-8") as log_handle:
            proc = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=fig3_subprocess_env(args),
                check=False,
            )
        return_code = int(proc.returncode)
    else:
        progress_log(f"Reusing Fig. 3 run for {candidate_id}", args.quiet)
        return_code = 0
    parsed = parse_fig3_run(run_dir / "multi_domain", candidate_id, row)
    parsed["fig3_return_code"] = return_code
    parsed["fig3_command"] = " ".join(command)
    parsed["fig3_run_dir"] = str(run_dir / "multi_domain")
    return parsed


def run_fig3_candidates(args: argparse.Namespace, candidate_sets: pd.DataFrame) -> pd.DataFrame:
    selected = candidate_sets.copy()
    if args.strict_pass_target and "fixed_candidate" in selected.columns:
        selected = selected.sort_values(["fixed_candidate", "set_proxy_score"], ascending=[False, False])
    else:
        selected = selected.sort_values("set_proxy_score", ascending=False)
    selected = selected.head(int(args.max_fig3_runs)).copy()
    records = selected.to_dict("records")
    if not records:
        return pd.DataFrame()
    parallel_runs = max(1, int(args.fig3_parallel_runs))
    if parallel_runs == 1:
        rows = [run_one_fig3_candidate(args, row) for row in records]
    else:
        progress_log(
            f"Running Fig. 3 audits with parallel_runs={parallel_runs}, "
            f"thread_limit={args.fig3_thread_limit or max(1, (os.cpu_count() or 1) // parallel_runs)}",
            args.quiet,
        )
        rows = []
        with ThreadPoolExecutor(max_workers=parallel_runs) as executor:
            futures = {executor.submit(run_one_fig3_candidate, args, row): str(row["candidate_id"]) for row in records}
            for future in as_completed(futures):
                candidate_id = futures[future]
                try:
                    rows.append(future.result())
                    progress_log(f"Finished Fig. 3 audit for {candidate_id}", args.quiet)
                except Exception as exc:
                    progress_log(f"Fig. 3 audit failed for {candidate_id}: {exc}", args.quiet)
                    rows.append({"candidate_id": candidate_id, "fig3_return_code": -1, "fig3_error": str(exc)})
    order = {str(row["candidate_id"]): idx for idx, row in enumerate(records)}
    return pd.DataFrame(rows).sort_values("candidate_id", key=lambda s: s.map(order)).reset_index(drop=True)




def read_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _baseline_value(path: Path, model: str) -> float:
    df = read_csv(path)
    if df.empty or "model" not in df.columns or "oof_spearman" not in df.columns:
        return float("nan")
    sub = df[df["model"].astype(str) == model]
    return float(pd.to_numeric(sub["oof_spearman"], errors="coerce").dropna().iloc[0]) if not sub.empty else float("nan")


def parse_fig3_run(run_dir: Path, candidate_id: str, proxy_row: Mapping[str, Any]) -> Dict[str, Any]:
    summary = read_json_dict(run_dir / "fig3_diagnostics_summary.json")
    cv = read_csv(run_dir / "fig3_cv_summary.csv")
    score = read_csv(run_dir / "fig3_score_table.csv")
    landmark = read_csv(run_dir / "fig3_landmark_validation.csv")
    effect = read_csv(run_dir / "fig3_effect_summary.csv")
    baseline_path = run_dir / "fig3_baseline_comparison.csv"
    learned = float(summary.get("learned_oof_spearman", np.nan))
    if not math.isfinite(learned) and not cv.empty and "test_spearman" in cv.columns:
        learned = float(pd.to_numeric(cv["test_spearman"], errors="coerce").mean())
    equal = _baseline_value(baseline_path, "equal_weights")
    best_single = _baseline_value(baseline_path, "best_single_indicator")
    delta_equal = learned - equal if math.isfinite(learned) and math.isfinite(equal) else 0.0
    delta_best = learned - best_single if math.isfinite(learned) and math.isfinite(best_single) else 0.0
    score_iqr = float(summary.get("score_iqr", np.nan))
    if not math.isfinite(score_iqr) and not score.empty and "S_w_oof" in score.columns:
        s = pd.to_numeric(score["S_w_oof"], errors="coerce")
        score_iqr = float(s.quantile(0.75) - s.quantile(0.25))
    landmark_sep = 0.0
    if not landmark.empty:
        cols = [c for c in landmark.columns if c.endswith("_matched_percentile")]
        if cols:
            vals = pd.concat([pd.to_numeric(landmark[c], errors="coerce") for c in cols], axis=0)
            landmark_sep = float(vals.dropna().mean() / 100.0) if vals.notna().any() else 0.0
    elif not score.empty and "is_landmark" in score.columns and "RGPM" in score.columns:
        lm = score[pd.to_numeric(score["is_landmark"], errors="coerce").fillna(0).astype(int) == 1]
        if not lm.empty:
            landmark_sep = float(percentile_rank(score["RGPM"]).loc[lm.index].mean())
    effect_lookup: Dict[str, float] = {}
    if not effect.empty and {"stat", "value"}.issubset(effect.columns):
        for row in effect.to_dict("records"):
            value = pd.to_numeric(row.get("value"), errors="coerce")
            effect_lookup[str(row.get("stat"))] = float(value) if pd.notna(value) else float("nan")
    data_profile = summary.get("data_profile", {}) if isinstance(summary.get("data_profile", {}), dict) else {}
    checks = summary.get("checks", {}) if isinstance(summary.get("checks", {}), dict) else {}
    data_checks = summary.get("data_checks", {}) if isinstance(summary.get("data_checks", {}), dict) else {}
    n_contributing = int(summary.get("n_contributing_graph_deltas", 0) or 0)
    total_papers = int(data_profile.get("total_papers", 0) or 0)
    min_papers = int(data_profile.get("min_papers_per_domain", 0) or 0)
    relaxed_rate = float(data_profile.get("relaxed_control_tier_rate_max_by_domain", np.nan))
    top20_enrichment = float(effect_lookup.get("top_vs_bottom_score_decile_rgpm_top20_enrichment", np.nan))
    high_low_lift = float(effect_lookup.get("high_vs_low_tertile_median_rgpm_lift_pp", np.nan))
    spearman_score = clipped_ratio(learned + 0.05, 0.40)
    improvement_score = clipped_ratio(max(delta_equal, 0.0), 0.08) * 0.6 + clipped_ratio(max(delta_best, 0.0), 0.05) * 0.4
    data_quality = float(proxy_row.get("mean_data_quality_score", 0.0) or 0.0)
    event_proxy = float(proxy_row.get("max_event_proxy", 0.0) or 0.0)
    diversity = float(proxy_row.get("category_diversity_score", 0.0) or 0.0)
    final = (
        0.35 * spearman_score
        + 0.15 * improvement_score
        + 0.20 * landmark_sep
        + 0.15 * event_proxy
        + 0.10 * data_quality
        + 0.05 * diversity
    )
    return {
        "candidate_id": candidate_id,
        "learned_oof_spearman": learned,
        "equal_weight_oof_spearman": equal,
        "best_single_oof_spearman": best_single,
        "delta_vs_equal": delta_equal,
        "delta_vs_best_single": delta_best,
        "score_iqr": score_iqr,
        "landmark_percentile_mean": landmark_sep,
        "fig3_overall_pass": bool(summary.get("overall_pass", False)),
        "fig3_status_label": summary.get("status_label", ""),
        "n_contributing_graph_deltas": n_contributing,
        "active_delta_z_cap_hit_rate_max": float(summary.get("active_delta_z_cap_hit_rate_max", np.nan)),
        "mean_delta_reliability": float(summary.get("mean_delta_reliability", np.nan)),
        "total_effective_papers": total_papers,
        "min_effective_papers_per_domain": min_papers,
        "relaxed_control_tier_rate_max_by_domain": relaxed_rate,
        "top20_enrichment": top20_enrichment,
        "high_low_tertile_rgpm_lift_pp": high_low_lift,
        "data_checks_pass_count": int(sum(int(bool(v)) for v in data_checks.values())),
        "checks_pass_count": int(sum(int(bool(v)) for v in checks.values())),
        "delta_variant": str(summary.get("delta_variant", "")),
        "fig3_screening_score": final,
    }


def attach_fig3_paper_scores(papers: pd.DataFrame, run_matrix: pd.DataFrame) -> pd.DataFrame:
    if papers.empty or run_matrix.empty:
        papers = papers.copy()
        papers["RGPM_percentile"] = np.nan
        papers["S_w_oof_percentile"] = np.nan
        return papers
    best = run_matrix.sort_values("fig3_screening_score", ascending=False).head(1)
    if best.empty:
        return papers
    run_dir = Path(str(best.iloc[0].get("fig3_run_dir", "")))
    score = read_csv(run_dir / "fig3_score_table.csv")
    if score.empty or "paper_id" not in score.columns:
        papers = papers.copy()
        papers["RGPM_percentile"] = np.nan
        papers["S_w_oof_percentile"] = np.nan
        return papers
    score = score.copy()
    score["paper_openalex_id"] = score["paper_id"].astype(str).str.replace(r"^[^:]+::", "", regex=True)
    if "RGPM" in score.columns:
        score["RGPM_percentile"] = percentile_rank(score["RGPM"])
    if "S_w_oof" in score.columns:
        score["S_w_oof_percentile"] = percentile_rank(score["S_w_oof"])
    keep = ["paper_openalex_id", "RGPM", "S_w_oof", "RGPM_percentile", "S_w_oof_percentile"]
    keep = [c for c in keep if c in score.columns]
    out = papers.merge(score[keep], left_on="paper_id", right_on="paper_openalex_id", how="left")
    return out.drop(columns=["paper_openalex_id"], errors="ignore")


def build_recommended_outputs(
    args: argparse.Namespace,
    domain_scores: pd.DataFrame,
    paper_events: pd.DataFrame,
    candidate_sets: pd.DataFrame,
    run_matrix: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sets = candidate_sets.copy()
    if not run_matrix.empty:
        sets = sets.merge(run_matrix, on="candidate_id", how="left", suffixes=("", "_fig3"))
        if getattr(args, "strict_pass_target", False):
            def series_col(name: str, default: object = np.nan) -> pd.Series:
                return sets[name] if name in sets.columns else pd.Series(default, index=sets.index)

            overall_pass = series_col("fig3_overall_pass", False).fillna(False).astype(bool).astype(float)
            learned = pd.to_numeric(series_col("learned_oof_spearman"), errors="coerce").fillna(0.0)
            delta_equal = pd.to_numeric(series_col("delta_vs_equal"), errors="coerce").fillna(0.0)
            score_iqr = pd.to_numeric(series_col("score_iqr"), errors="coerce").fillna(0.0)
            contributing = pd.to_numeric(series_col("n_contributing_graph_deltas"), errors="coerce").fillna(0.0)
            total_papers = pd.to_numeric(series_col("total_effective_papers"), errors="coerce").fillna(0.0)
            min_papers = pd.to_numeric(series_col("min_effective_papers_per_domain"), errors="coerce").fillna(0.0)
            relaxed = pd.to_numeric(series_col("relaxed_control_tier_rate_max_by_domain"), errors="coerce").fillna(1.0)
            top20 = pd.to_numeric(series_col("top20_enrichment"), errors="coerce").fillna(0.0)
            high_low = pd.to_numeric(series_col("high_low_tertile_rgpm_lift_pp"), errors="coerce").fillna(0.0)
            proxy = pd.to_numeric(series_col("set_proxy_score"), errors="coerce").fillna(0.0)
            strict_quality = (
                0.22 * clipped_ratio_series(learned, 0.55)
                + 0.14 * clipped_ratio_series(delta_equal.clip(lower=0.0), 0.18)
                + 0.12 * clipped_ratio_series(score_iqr, 0.60)
                + 0.12 * clipped_ratio_series(contributing, 6.0)
                + 0.10 * clipped_ratio_series(total_papers, 8000.0)
                + 0.10 * clipped_ratio_series(min_papers, 700.0)
                + 0.08 * clipped_ratio_series((0.70 - relaxed).clip(lower=0.0), 0.70)
                + 0.06 * clipped_ratio_series(top20, 12.0)
                + 0.04 * clipped_ratio_series(high_low, 60.0)
                + 0.02 * proxy
            )
            sets["strict_screening_score"] = 2.0 * overall_pass + strict_quality
            sets["final_screening_score"] = sets["strict_screening_score"]
        else:
            sets["final_screening_score"] = sets["fig3_screening_score"].fillna(sets["set_proxy_score"])
    else:
        sets["final_screening_score"] = sets["set_proxy_score"]
        if getattr(args, "strict_pass_target", False):
            sets["strict_screening_score"] = sets["final_screening_score"]
    sets = sets.sort_values("final_screening_score", ascending=False).reset_index(drop=True)
    sets["final_rank"] = np.arange(1, len(sets) + 1)

    papers = paper_events.copy()
    papers = attach_fig3_paper_scores(papers, run_matrix)
    papers["RGPM_percentile"] = pd.to_numeric(papers.get("RGPM_percentile", np.nan), errors="coerce")
    papers["S_w_oof_percentile"] = pd.to_numeric(papers.get("S_w_oof_percentile", np.nan), errors="coerce")
    papers["final_paper_score"] = (
        0.46 * papers["paper_proxy_score"]
        + 0.24 * papers["RGPM_percentile"].fillna(0.5)
        + 0.20 * papers["S_w_oof_percentile"].fillna(0.5)
        + 0.10 * percentile_rank(papers["cited_by_count"])
    )
    papers["proxy_rank"] = papers["paper_proxy_score"].rank(method="first", ascending=False).astype(int)
    papers["final_rank"] = papers["final_paper_score"].rank(method="first", ascending=False).astype(int)
    papers["recommended_for"] = papers.apply(recommendation_label, axis=1)
    papers["recommendation_reason"] = papers.apply(recommendation_reason, axis=1)
    papers = papers.sort_values("final_paper_score", ascending=False).head(int(args.recommended_papers)).reset_index(drop=True)
    return sets, papers


def recommendation_label(row: Mapping[str, Any]) -> str:
    labels: List[str] = []
    if float(row.get("event_proxy_score", 0) or 0) >= 0.75:
        labels.append("fig1_best_event_cases")
    if int(row.get("reliable_anchor", 0) or 0) == 1 and float(row.get("paper_proxy_score", 0) or 0) >= 0.65:
        labels.append("fig2_best_landmark_cases")
    if float(row.get("RGPM_percentile", 0) or 0) >= 0.75 or float(row.get("S_w_oof_percentile", 0) or 0) >= 0.75:
        labels.append("fig3_best_calibration_cases")
    if int(row.get("year", 0) or 0) <= 2020 and float(row.get("anchor_citer_topic_breadth", 0) or 0) >= 5:
        labels.append("fig5_best_forecast_domains")
    return ";".join(labels or ["general_candidate"])


def recommendation_reason(row: Mapping[str, Any]) -> str:
    parts = [
        f"event_proxy={float(row.get('event_proxy_score', 0) or 0):.3f}",
        f"topic_shift={float(row.get('topic_shift', 0) or 0):.3f}",
        f"citers={int(row.get('anchor_citers', 0) or 0)}",
    ]
    if pd.notna(row.get("RGPM_percentile", np.nan)):
        parts.append(f"RGPM_pct={float(row.get('RGPM_percentile')):.3f}")
    if pd.notna(row.get("S_w_oof_percentile", np.nan)):
        parts.append(f"S_w_pct={float(row.get('S_w_oof_percentile')):.3f}")
    return ", ".join(parts)


def write_reproduce_commands(args: argparse.Namespace, sets: pd.DataFrame) -> None:
    path = args.out_dir / "reproduce_commands.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    command_sets = sets.copy()
    if args.strict_pass_target and "fixed_candidate" in command_sets.columns:
        command_sets = command_sets.sort_values(
            ["fixed_candidate", "final_screening_score"],
            ascending=[False, False],
        )
    for _, row in command_sets.head(int(args.max_fig3_runs)).iterrows():
        run_dir = args.out_dir / "fig3_runs" / str(row["candidate_id"])
        domains = str(row["domains"]).split()
        command = fig3_command(args, domains, run_dir)
        lines.append("# " + str(row["candidate_id"]))
        lines.append(" ".join(command))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    args: argparse.Namespace,
    domain_scores: pd.DataFrame,
    candidate_sets: pd.DataFrame,
    recommended_sets: pd.DataFrame,
    recommended_papers: pd.DataFrame,
    run_matrix: pd.DataFrame,
) -> None:
    report_json = {
        "corpus_dir": str(args.corpus_dir),
        "stage": args.stage,
        "complete_end_year": int(args.complete_end_year),
        "n_domains_scored": int(len(domain_scores)),
        "n_candidate_sets": int(len(candidate_sets)),
        "n_fig3_runs": int(len(run_matrix)),
        "strict_pass_target": bool(getattr(args, "strict_pass_target", False)),
        "n_fig3_overall_pass": int(run_matrix["fig3_overall_pass"].fillna(False).astype(bool).sum()) if "fig3_overall_pass" in run_matrix else 0,
        "top_domain_sets": recommended_sets.head(10).to_dict("records"),
        "top_papers": recommended_papers.head(30).to_dict("records"),
    }
    write_json(args.out_dir / "screening_report.json", report_json)
    lines = [
        "# Corpus Screening Report",
        "",
        f"- Corpus: `{args.corpus_dir}`",
        f"- Stage: `{args.stage}`",
        f"- Complete end year: `{args.complete_end_year}`",
        f"- Domains scored: {len(domain_scores)}",
        f"- Candidate domain sets: {len(candidate_sets)}",
        f"- Fig3 runs parsed: {len(run_matrix)}",
        f"- Strict pass target: {bool(getattr(args, 'strict_pass_target', False))}",
        "",
        "## Top Domain Sets",
        "",
    ]
    for _, row in recommended_sets.head(10).iterrows():
        lines.append(
            f"- **{row['candidate_id']}** score={float(row['final_screening_score']):.3f} "
            f"domains=`{row['domains']}`"
        )
    lines.extend(["", "## Top Papers", ""])
    for _, row in recommended_papers.head(20).iterrows():
        lines.append(
            f"- **{row['domain']} ({int(row['year'])})** {row['title']} "
            f"[{row['recommended_for']}] score={float(row['final_paper_score']):.3f}"
        )
    (args.out_dir / "screening_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_proxy_stage(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    progress_log(f"Loading corpus from {args.corpus_dir}", args.quiet)
    corpus = load_corpus(args.corpus_dir)
    works = annotate_reliable_anchors(corpus.works, corpus.landmarks, args.complete_end_year)
    citations = annotate_citations(corpus.citations, works)
    paper_events = compute_paper_event_candidates(
        works,
        citations,
        top_papers_per_domain=args.top_papers_per_domain,
        max_anchor_candidates_per_domain=args.max_anchor_candidates_per_domain,
    )
    domain_scores = compute_domain_proxy_scores(
        works,
        citations,
        corpus.domains,
        paper_events,
        corpus.quality_report,
        min_domain_papers=args.min_domain_papers,
        min_citations_per_work=args.min_citations_per_work,
    )
    candidate_sets = generate_candidate_domain_sets(
        domain_scores,
        sizes=args.candidate_set_sizes,
        top_domains=args.top_domains,
        beam_width=args.beam_width,
    )
    if args.strict_pass_target and not args.no_strict_fixed_candidates:
        candidate_sets = append_strict_fixed_candidate_sets(candidate_sets, domain_scores)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    domain_scores.to_csv(args.out_dir / "domain_proxy_scores.csv", index=False)
    paper_events.to_csv(args.out_dir / "paper_event_candidates.csv", index=False)
    candidate_sets.to_csv(args.out_dir / "candidate_domain_sets.csv", index=False)
    progress_log(
        f"Proxy stage wrote {len(domain_scores)} domains, {len(paper_events)} paper candidates, "
        f"{len(candidate_sets)} domain sets.",
        args.quiet,
    )
    return domain_scores, paper_events, candidate_sets


def load_proxy_outputs(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    domain_scores = read_csv(args.out_dir / "domain_proxy_scores.csv")
    paper_events = read_csv(args.out_dir / "paper_event_candidates.csv")
    candidate_sets = read_csv(args.out_dir / "candidate_domain_sets.csv")
    if domain_scores.empty or candidate_sets.empty:
        return run_proxy_stage(args)
    return domain_scores, paper_events, candidate_sets


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen ASPR corpus domains and papers for strong figure candidates.")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stage", choices=["proxy", "fig3", "all"], default="all")
    parser.add_argument("--complete-end-year", type=int, default=2025)
    parser.add_argument("--candidate-set-sizes", nargs="+", type=int, default=[4, 6, 8, 12])
    parser.add_argument("--beam-width", type=int, default=12)
    parser.add_argument("--top-domains", type=int, default=24)
    parser.add_argument("--top-papers-per-domain", type=int, default=20)
    parser.add_argument("--recommended-papers", type=int, default=100)
    parser.add_argument("--min-domain-papers", type=int, default=1500)
    parser.add_argument("--min-citations-per-work", type=float, default=1.0)
    parser.add_argument("--max-anchor-candidates-per-domain", type=int, default=200)
    parser.add_argument("--max-fig3-runs", type=int, default=5)
    parser.add_argument(
        "--fig3-parallel-runs",
        type=int,
        default=1,
        help="Number of Fig. 3 candidate subprocesses to run concurrently.",
    )
    parser.add_argument(
        "--fig3-thread-limit",
        type=int,
        default=None,
        help="Optional OMP/BLAS thread cap per Fig. 3 subprocess. Defaults to CPU cores / parallel runs.",
    )
    parser.add_argument("--fig3-weight-samples", type=int, default=8000)
    parser.add_argument("--fig3-min-controls", type=int, default=30)
    parser.add_argument("--fig3-tau", type=int, default=10)
    parser.add_argument(
        "--fig3-delta-variant",
        choices=["matched_control_v3", "domain_residual_v2"],
        default="matched_control_v3",
    )
    parser.add_argument("--fig3-progress-interval", type=int, default=1000)
    parser.add_argument("--max-papers", type=int, default=None, help="Optional Fig3 debugging cap passed through to --max-papers.")
    parser.add_argument("--strict-pass-target", action="store_true", help="Rank Fig. 3 runs by strict validation pass criteria first.")
    parser.add_argument("--no-strict-fixed-candidates", action="store_true", help="Do not inject fixed strict-validation domain sets.")
    parser.add_argument("--force-fig3", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.stage in {"proxy", "all"}:
        domain_scores, paper_events, candidate_sets = run_proxy_stage(args)
    else:
        domain_scores, paper_events, candidate_sets = load_proxy_outputs(args)

    run_matrix = pd.DataFrame()
    if args.stage in {"fig3", "all"} and not candidate_sets.empty:
        run_matrix = run_fig3_candidates(args, candidate_sets)
        run_matrix.to_csv(args.out_dir / "fig3_run_matrix.csv", index=False)
    elif (args.out_dir / "fig3_run_matrix.csv").exists():
        run_matrix = read_csv(args.out_dir / "fig3_run_matrix.csv")

    recommended_sets, recommended_papers = build_recommended_outputs(
        args,
        domain_scores,
        paper_events,
        candidate_sets,
        run_matrix,
    )
    recommended_sets.to_csv(args.out_dir / "recommended_domain_sets.csv", index=False)
    recommended_papers.to_csv(args.out_dir / "recommended_papers.csv", index=False)
    write_reproduce_commands(args, recommended_sets)
    write_report(args, domain_scores, candidate_sets, recommended_sets, recommended_papers, run_matrix)
    progress_log(f"Screening complete. Outputs in {args.out_dir}", args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
