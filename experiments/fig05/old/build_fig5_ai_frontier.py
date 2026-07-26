#!/usr/bin/env python3
"""Build source-backed 2024-2026 AI/AI-enabled frontier data for Fig. 5."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.figure_quality import write_json, write_run_manifest


DEFAULT_LOCAL_PAPERS = PROJECT_ROOT / "outputs" / "fig05/old" / "plot_data" / "base" / "papers_master.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "fig05/old" / "ai_frontier"
OPENALEX_WORKS_URL = "https://api.openalex.org/works"

AI_PATTERNS: Sequence[Tuple[str, str, str]] = (
    ("large_language_model", "large language model", r"\b(large language model|large language models|llm|llms|gpt-[0-9a-z]+|chatgpt)\b"),
    ("foundation_model", "foundation model", r"\b(foundation model|foundation models)\b"),
    ("generative_ai", "generative AI", r"\b(generative ai|generative artificial intelligence)\b"),
    ("diffusion_model", "diffusion model", r"\b(diffusion model|diffusion models)\b"),
    ("deep_learning", "deep learning", r"\bdeep learning\b"),
    ("machine_learning", "machine learning", r"\bmachine learning\b"),
    ("artificial_intelligence", "artificial intelligence", r"\bartificial intelligence\b|\bAI\b"),
    ("neural_network", "neural network", r"\bneural network|neural networks\b"),
    ("multimodal_ai", "multimodal AI", r"\b(multimodal|vision-language|vision language)\b"),
    ("graph_neural_network", "graph neural network", r"\bgraph neural network|graph neural networks|GNN\b"),
    ("reinforcement_learning", "reinforcement learning", r"\breinforcement learning\b"),
    ("self_supervised", "self-supervised learning", r"\bself-supervised|self supervised\b"),
)

APPLICATION_PATTERNS: Sequence[Tuple[str, str, str]] = (
    ("scientific_discovery", "AI for scientific discovery", r"\b(scientific discovery|science|research automation)\b"),
    ("materials", "AI-enabled materials discovery", r"\b(materials?|perovskite|battery|catalyst|crystal|polymer|graphene|mxene)\b"),
    ("biomedicine", "AI-enabled biomedicine", r"\b(drug discovery|drug development|protein|clinical|medical|healthcare|pathology|biomedical|disease|cancer|antimicrobial)\b"),
    ("genomics", "AI-enabled genomics and cells", r"\b(genomic|genomics|single-cell|single cell|gene|cellular|transcriptomic)\b"),
    ("chemistry", "AI-enabled chemistry", r"\b(chemistry|molecular|molecule|reaction|retrosynthesis)\b"),
    ("astronomy", "AI-enabled astronomy", r"\b(exoplanet|astronomy|astrophysics|telescope|transit)\b"),
    ("climate", "AI-enabled climate and Earth science", r"\b(climate|weather|earth system|remote sensing)\b"),
    ("general_methods", "General-purpose AI methods", r"\b(evaluation|benchmark|agent|retrieval|reasoning|planning)\b"),
)

OPENALEX_QUERIES: Sequence[Tuple[str, str]] = (
    ("scientific_discovery", '"large language model" "scientific discovery"'),
    ("scientific_discovery", '"foundation model" "scientific discovery"'),
    ("scientific_discovery", '"generative artificial intelligence" science'),
    ("materials", '"AI" "materials discovery"'),
    ("materials", '"machine learning" "materials discovery"'),
    ("materials", '"graph neural network" "materials science"'),
    ("biomedicine", '"artificial intelligence" "drug discovery"'),
    ("biomedicine", '"deep learning" "protein design"'),
    ("biomedicine", '"diffusion model" protein design'),
    ("genomics", '"machine learning" "single-cell genomics"'),
    ("genomics", '"foundation model" genomics'),
    ("chemistry", '"large language model" chemistry'),
    ("astronomy", '"machine learning" exoplanet detection'),
    ("climate", '"artificial intelligence" climate science'),
    ("general_methods", '"multimodal foundation model" science'),
)


def compact_text(value: object) -> str:
    """Return whitespace-normalized text."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def stable_float(value: object) -> float:
    """Return a deterministic pseudo-random float in [0, 1)."""
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def title_norm(value: object) -> str:
    """Normalize titles for deduplication."""
    return re.sub(r"[^a-z0-9]+", " ", compact_text(value).lower()).strip()


def relpath(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def matched_patterns(text: str, patterns: Sequence[Tuple[str, str, str]]) -> List[Tuple[str, str]]:
    """Return pattern ids and labels matched by text."""
    out: List[Tuple[str, str]] = []
    lowered = text.lower()
    for key, label, pattern in patterns:
        if key == "artificial_intelligence" and re.search(r"\b(non[- ]?ai|not ai|without ai)\b", lowered):
            continue
        if re.search(pattern, text, flags=re.IGNORECASE) and key not in {item[0] for item in out}:
            out.append((key, label))
    return out


def classify_theme(text: str, fallback: str = "general_methods") -> Tuple[str, str]:
    """Classify one paper into a reader-facing AI frontier theme."""
    matches = matched_patterns(text, APPLICATION_PATTERNS)
    if matches:
        return matches[0]
    safe_fallback = fallback if fallback in {"scientific_discovery", "general_methods"} else "general_methods"
    for key, label, _ in APPLICATION_PATTERNS:
        if key == safe_fallback:
            return key, label
    return "general_methods", "General-purpose AI methods"


def column_or_blank(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a string column or blanks."""
    if column not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index)
    return frame[column].fillna("").astype(str)


def local_frontier_rows(path: Path, start_year: int, end_year: int) -> List[Dict[str, Any]]:
    """Extract strict AI rows from the local Fig. 5 paper table."""
    if not path.exists():
        return []
    papers = pd.read_csv(path, low_memory=False)
    years = pd.to_numeric(papers.get("year"), errors="coerce")
    text = (
        column_or_blank(papers, "title")
        + " "
        + column_or_blank(papers, "topic_label")
        + " "
        + column_or_blank(papers, "display_topic_label")
        + " "
        + column_or_blank(papers, "keywords")
    )
    rows: List[Dict[str, Any]] = []
    for idx, row in papers[years.between(start_year, end_year)].iterrows():
        evidence_text = compact_text(text.loc[idx])
        ai_terms = matched_patterns(evidence_text, AI_PATTERNS)
        if not ai_terms:
            continue
        theme_key, theme_label = classify_theme(evidence_text, fallback=str(row.get("domain") or "general_methods"))
        rows.append(
            {
                "source": "local_fig5_papers_master",
                "source_query": "",
                "source_url": relpath(path),
                "paper_id": compact_text(row.get("paper_id") or row.get("id") or f"local_{idx}"),
                "doi": compact_text(row.get("doi")),
                "title": compact_text(row.get("title")),
                "year": int(years.loc[idx]),
                "cited_by_count": int(float(row.get("cited_by_count") or 0)),
                "domain": compact_text(row.get("domain")),
                "topic_label": compact_text(row.get("topic_label") or row.get("display_topic_label")),
                "ai_term_ids": ";".join(key for key, _ in ai_terms),
                "ai_terms": ";".join(label for _, label in ai_terms),
                "theme_id": theme_key,
                "theme_label": theme_label,
                "title_topic_ai_match": 1,
            }
        )
    return rows


def openalex_select_fields() -> str:
    """Return a compact OpenAlex select clause."""
    return "id,doi,display_name,publication_year,cited_by_count,primary_topic,topics"


def openalex_topic_text(work: Mapping[str, Any]) -> str:
    """Return compact primary-topic and topic labels from one OpenAlex work."""
    values: List[str] = []
    primary = work.get("primary_topic") or {}
    if isinstance(primary, Mapping):
        values.append(compact_text(primary.get("display_name")))
    for topic in work.get("topics") or []:
        if isinstance(topic, Mapping):
            values.append(compact_text(topic.get("display_name")))
    return "; ".join([value for value in values if value])


def normalize_openalex_work(work: Mapping[str, Any], query: str, query_theme: str, source_url: str) -> Optional[Dict[str, Any]]:
    """Normalize one OpenAlex work into the frontier row contract."""
    title = compact_text(work.get("display_name"))
    year = work.get("publication_year")
    if not title or year is None:
        return None
    topic_text = openalex_topic_text(work)
    evidence_text = compact_text(f"{title} {topic_text}")
    ai_terms = matched_patterns(evidence_text, AI_PATTERNS)
    if not ai_terms:
        return None
    theme_key, theme_label = classify_theme(evidence_text, fallback=query_theme)
    return {
        "source": "openalex_works_search",
        "source_query": query,
        "source_url": source_url,
        "paper_id": compact_text(work.get("id")),
        "doi": compact_text(work.get("doi")),
        "title": title,
        "year": int(year),
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "domain": "openalex",
        "topic_label": topic_text,
        "ai_term_ids": ";".join(key for key, _ in ai_terms),
        "ai_terms": ";".join(label for _, label in ai_terms),
        "theme_id": theme_key,
        "theme_label": theme_label,
        "title_topic_ai_match": 1,
    }


def fetch_openalex_rows(
    *,
    start_date: str,
    end_date: str,
    per_query: int,
    timeout: int,
    mailto: Optional[str],
    sleep_seconds: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetch source-backed AI frontier rows from OpenAlex."""
    rows: List[Dict[str, Any]] = []
    query_reports: List[Dict[str, Any]] = []
    for query_theme, query in OPENALEX_QUERIES:
        params = {
            "search": query,
            "filter": f"from_publication_date:{start_date},to_publication_date:{end_date}",
            "per-page": int(per_query),
            "select": openalex_select_fields(),
        }
        if mailto:
            params["mailto"] = mailto
        started = time.time()
        try:
            response = requests.get(OPENALEX_WORKS_URL, params=params, timeout=timeout)
            status_code = int(response.status_code)
            payload = response.json() if response.ok else {}
        except (requests.RequestException, json.JSONDecodeError) as exc:
            query_reports.append(
                {
                    "query": query,
                    "query_theme": query_theme,
                    "status": "failed",
                    "error": str(exc),
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            )
            continue
        source_url = response.url
        normalized: List[Dict[str, Any]] = []
        for work in payload.get("results") or []:
            row = normalize_openalex_work(work, query, query_theme, source_url)
            if row:
                normalized.append(row)
        rows.extend(normalized)
        query_reports.append(
            {
                "query": query,
                "query_theme": query_theme,
                "status": "ok" if response.ok else "http_error",
                "status_code": status_code,
                "source_url": source_url,
                "result_count": int((payload.get("meta") or {}).get("count") or 0),
                "accepted_rows": len(normalized),
                "elapsed_seconds": round(time.time() - started, 3),
            }
        )
        if sleep_seconds > 0:
            time.sleep(float(sleep_seconds))
    return rows, query_reports


def dedupe_rows(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Dedupe frontier rows by OpenAlex id, DOI, then normalized title."""
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["_dedupe_key"] = [
        compact_text(row.get("paper_id")) or compact_text(row.get("doi")) or title_norm(row.get("title"))
        for row in rows
    ]
    frame["_source_priority"] = frame["source"].map({"openalex_works_search": 0, "local_fig5_papers_master": 1}).fillna(2)
    frame = frame.sort_values(["_dedupe_key", "_source_priority", "cited_by_count"], ascending=[True, True, False])
    frame = frame.drop_duplicates("_dedupe_key", keep="first").drop(columns=["_dedupe_key", "_source_priority"])
    return frame.reset_index(drop=True)


def percentile(series: pd.Series) -> pd.Series:
    """Return percentile rank in [0, 1]."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if numeric.max() == numeric.min():
        return pd.Series([0.5] * len(numeric), index=numeric.index)
    return numeric.rank(method="average", pct=True)


def add_scores_and_positions(frame: pd.DataFrame, end_year: int) -> pd.DataFrame:
    """Add frontier scores and deterministic point-cloud coordinates."""
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    cite_score = percentile(out["cited_by_count"].map(lambda value: math.log1p(float(value or 0))))
    years = pd.to_numeric(out["year"], errors="coerce").fillna(end_year)
    recency = ((years - float(years.min())) / max(1.0, float(years.max() - years.min()))).clip(0.0, 1.0)
    term_richness = out["ai_term_ids"].astype(str).map(lambda value: min(1.0, max(1, len([v for v in value.split(";") if v])) / 3.0))
    out["frontier_score"] = (0.48 * cite_score + 0.30 * recency + 0.22 * term_richness).round(6)
    theme_order = {theme: idx for idx, theme in enumerate(sorted(out["theme_id"].astype(str).unique()))}
    n_themes = max(1, len(theme_order))
    xs: List[float] = []
    ys: List[float] = []
    for _, row in out.iterrows():
        idx = theme_order[str(row["theme_id"])]
        angle = 2.0 * math.pi * idx / n_themes
        radius = 0.58 + 0.16 * (stable_float(row["paper_id"]) - 0.5)
        jitter_angle = 2.0 * math.pi * stable_float(str(row["paper_id"]) + "j")
        jitter = 0.10 * (stable_float(str(row["paper_id"]) + "r") - 0.5)
        xs.append(round(radius * math.cos(angle) + jitter * math.cos(jitter_angle), 6))
        ys.append(round(radius * math.sin(angle) + jitter * math.sin(jitter_angle), 6))
    out["plot_x"] = xs
    out["plot_y"] = ys
    out["display_size"] = (50.0 + 520.0 * out["frontier_score"].clip(0.0, 1.0).pow(0.65)).round(3)
    out["display_alpha"] = (0.48 + 0.42 * out["frontier_score"].clip(0.0, 1.0)).round(3)
    return out.sort_values(["frontier_score", "cited_by_count", "year"], ascending=[False, False, False]).reset_index(drop=True)


def build_term_table(frontier: pd.DataFrame) -> pd.DataFrame:
    """Aggregate matched AI terms for figure labels and audit."""
    rows: List[Dict[str, Any]] = []
    if frontier.empty:
        return pd.DataFrame(columns=["term_id", "term_label", "paper_count", "theme_count", "top_paper_title"])
    labels = {key: label for key, label, _ in AI_PATTERNS}
    for term_id, group in frontier.assign(term_id=frontier["ai_term_ids"].str.split(";")).explode("term_id").groupby("term_id"):
        term = compact_text(term_id)
        if not term:
            continue
        top = group.sort_values("frontier_score", ascending=False).iloc[0]
        rows.append(
            {
                "term_id": term,
                "term_label": labels.get(term, term.replace("_", " ")),
                "paper_count": int(len(group)),
                "theme_count": int(group["theme_id"].nunique()),
                "top_paper_title": top["title"],
                "top_paper_year": int(top["year"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["paper_count", "theme_count", "term_label"], ascending=[False, False, True])


def select_point_cloud(frontier: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    """Select a high-score but theme-balanced point cloud."""
    if frontier.empty:
        return frontier.copy()
    target = int(n_rows)
    themes = sorted(frontier["theme_id"].astype(str).unique())
    per_theme = max(4, target // max(1, len(themes)) // 2)
    selected_parts: List[pd.DataFrame] = []
    selected_ids: set[str] = set()
    for theme in themes:
        group = frontier[frontier["theme_id"].astype(str).eq(theme)].sort_values("frontier_score", ascending=False).head(per_theme)
        selected_parts.append(group)
        selected_ids.update(group["paper_id"].astype(str))
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    remaining = frontier[~frontier["paper_id"].astype(str).isin(selected_ids)].sort_values("frontier_score", ascending=False)
    point_cloud = pd.concat([selected, remaining.head(max(0, target - len(selected)))], ignore_index=True)
    return point_cloud.sort_values(["frontier_score", "cited_by_count", "year"], ascending=[False, False, False]).head(target).reset_index(drop=True)


def build_quality_gates(frontier: pd.DataFrame, point_cloud: pd.DataFrame, terms: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate whether the AI frontier data can support a Fig. 5 AI-hotspot claim."""
    if frontier.empty:
        checks = {
            "ai_frontier_evidence_rows_min80": 0,
            "point_cloud_rows_min60": 0,
            "ai_terms_min8": 0,
            "top_points_ai_relevance_all": 0,
            "top_points_2024_2026_all": 0,
            "theme_coverage_min5": 0,
            "point_cloud_theme_coverage_min5": 0,
            "source_urls_present": 0,
        }
    else:
        top = point_cloud.head(min(60, len(point_cloud))).copy()
        top_years = pd.to_numeric(top["year"], errors="coerce")
        checks = {
            "ai_frontier_evidence_rows_min80": int(len(frontier) >= 80),
            "point_cloud_rows_min60": int(len(point_cloud) >= 60),
            "ai_terms_min8": int(len(terms) >= 8),
            "top_points_ai_relevance_all": int(top["title_topic_ai_match"].astype(int).eq(1).all()),
            "top_points_2024_2026_all": int(top_years.between(2024, 2026).all()),
            "theme_coverage_min5": int(frontier["theme_id"].nunique() >= 5),
            "point_cloud_theme_coverage_min5": int(point_cloud["theme_id"].nunique() >= 5),
            "source_urls_present": int(frontier["source_url"].astype(str).str.len().gt(0).all()),
        }
    overall = bool(all(checks.values()))
    return {
        "overall_pass": overall,
        "status_label": "source_backed_ai_frontier_ready" if overall else "ai_frontier_data_gap",
        "checks": checks,
        "thresholds": {
            "ai_frontier_evidence_rows": 80,
            "point_cloud_rows": 60,
            "ai_terms": 8,
            "theme_coverage": 5,
            "point_cloud_theme_coverage": 5,
            "top_points_year_range": "2024-2026",
        },
        "counts": {
            "frontier_rows": int(len(frontier)),
            "point_cloud_rows": int(len(point_cloud)),
            "ai_terms": int(len(terms)),
            "themes": int(frontier["theme_id"].nunique()) if not frontier.empty else 0,
            "point_cloud_themes": int(point_cloud["theme_id"].nunique()) if not point_cloud.empty else 0,
            "openalex_rows": int(frontier["source"].eq("openalex_works_search").sum()) if not frontier.empty else 0,
            "local_rows": int(frontier["source"].eq("local_fig5_papers_master").sum()) if not frontier.empty else 0,
        },
    }


def render_image2_prompt(point_cloud: pd.DataFrame, terms: pd.DataFrame, gates: Mapping[str, Any]) -> str:
    """Write a concise visual handoff prompt for a non-Python Fig. 5 redraw."""
    top_terms = terms.head(10)["term_label"].tolist() if not terms.empty else []
    top_points = point_cloud.head(18)[["title", "year", "theme_label", "frontier_score", "source"]].to_dict("records") if not point_cloud.empty else []
    payload = {
        "visual_goal": "Nature-style Fig. 5 AI/AI-enabled frontier point cloud; beautiful dense point cloud is the primary visual.",
        "data_contract": {
            "point_cloud_csv": "ai_frontier_point_cloud.csv",
            "term_csv": "ai_frontier_terms.csv",
            "quality_status": gates.get("status_label"),
            "no_take_home_footer": True,
            "delete_low_value_panels": True,
        },
        "top_ai_terms": top_terms,
        "top_points": top_points,
    }
    return (
        "# Fig. 5 AI Frontier Image Prompt\n\n"
        "Create a polished Nature-style figure with a dense AI/AI-enabled science frontier point cloud as the dominant panel. "
        "Use the exact CSV-backed labels and do not add a take-home-message footer. Fuse small explanatory panels into compact side annotations.\n\n"
        "```json\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
        + "\n```\n"
    )


def write_outputs(
    out_dir: Path,
    frontier: pd.DataFrame,
    point_cloud: pd.DataFrame,
    terms: pd.DataFrame,
    query_reports: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    """Write all AI frontier deliverables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frontier.to_csv(out_dir / "ai_frontier_papers.csv", index=False)
    point_cloud.to_csv(out_dir / "ai_frontier_point_cloud.csv", index=False)
    terms.to_csv(out_dir / "ai_frontier_terms.csv", index=False)
    write_json(out_dir / "openalex_query_report.json", {"queries": list(query_reports)})
    write_json(out_dir / "ai_frontier_quality_report.json", gates)
    (out_dir / "fig5_ai_frontier_image2_prompt.md").write_text(render_image2_prompt(point_cloud, terms, gates), encoding="utf-8")
    write_run_manifest(
        out_dir,
        figure="fig5_ai_frontier",
        argv=sys.argv,
        inputs={"local_papers": str(args.local_papers)},
        domains=sorted(frontier["theme_id"].unique()) if not frontier.empty else [],
        quality_gates=gates,
        extra={
            "start_date": args.start_date,
            "end_date": args.end_date,
            "openalex_enabled": not args.offline,
            "per_query": int(args.per_query),
            "query_count": len(OPENALEX_QUERIES),
            "visual_policy": "data for image model / design redraw; no Python final Fig.5 hard plot",
        },
    )


def build_frontier(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
    """Build all Fig. 5 AI frontier tables."""
    start_year = int(args.start_date[:4])
    end_year = int(args.end_date[:4])
    rows = local_frontier_rows(args.local_papers, start_year, end_year)
    query_reports: List[Dict[str, Any]] = []
    if not args.offline:
        online_rows, query_reports = fetch_openalex_rows(
            start_date=args.start_date,
            end_date=args.end_date,
            per_query=args.per_query,
            timeout=args.timeout,
            mailto=args.mailto,
            sleep_seconds=args.sleep_seconds,
        )
        rows.extend(online_rows)
    frontier = dedupe_rows(rows)
    if not frontier.empty:
        years = pd.to_numeric(frontier["year"], errors="coerce")
        frontier = frontier[years.between(start_year, end_year)].copy()
    frontier = add_scores_and_positions(frontier, end_year)
    point_cloud = select_point_cloud(frontier, int(args.point_cloud_rows))
    terms = build_term_table(frontier)
    gates = build_quality_gates(frontier, point_cloud, terms)
    return frontier, point_cloud, terms, query_reports, gates


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build source-backed Fig. 5 AI frontier data.")
    parser.add_argument("--local-papers", type=Path, default=DEFAULT_LOCAL_PAPERS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default=dt.date.today().isoformat())
    parser.add_argument("--per-query", type=int, default=30)
    parser.add_argument("--point-cloud-rows", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--mailto", default=None)
    parser.add_argument("--offline", action="store_true", help="Use local rows only; quality gates may remain blocked.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    frontier, point_cloud, terms, query_reports, gates = build_frontier(args)
    write_outputs(args.out_dir, frontier, point_cloud, terms, query_reports, gates, args)
    print(f"[fig5-ai-frontier] wrote {args.out_dir}")
    print(f"[fig5-ai-frontier] rows={len(frontier)} point_cloud={len(point_cloud)} terms={len(terms)}")
    print(f"[fig5-ai-frontier] status={gates['status_label']}")


if __name__ == "__main__":
    main()
