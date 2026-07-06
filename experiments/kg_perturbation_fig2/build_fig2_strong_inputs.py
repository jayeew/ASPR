from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.kg_perturbation_fig2.build_fig2_reference_closure import build_reference_closure


MATCH_COLUMNS_EXACT = ["domain", "publication_year", "article_type", "reference_count_decile", "venue_family"]
MATCH_COLUMNS_RELAXED = ["domain", "publication_year", "article_type", "reference_count_decile"]
RELAXED_CONTROL_TIERS = {"relaxed_without_venue", "domain_year_article", "domain_year", "domain_all_years"}
CONTROL_TIER_COLUMNS = {
    "exact": MATCH_COLUMNS_EXACT,
    "relaxed_without_venue": MATCH_COLUMNS_RELAXED,
    "domain_year_article": ["domain", "publication_year", "article_type"],
    "domain_year": ["domain", "publication_year"],
    "domain_all_years": ["domain"],
}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def read_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_corpus_root(source: Path) -> Path:
    """Resolve either a corpus root or a views/fig2 directory to the corpus root."""
    source = source.resolve()
    if (source / "works.csv").exists() and (source / "citations.csv").exists():
        works_cols = set(pd.read_csv(source / "works.csv", nrows=0).columns)
        if "referenced_works" in works_cols:
            return source
    if source.name == "fig2" and source.parent.name == "views":
        candidate = source.parent.parent
        if (candidate / "works.csv").exists():
            return candidate
    candidate = source
    for parent in [source.parent, source.parent.parent if source.parent != source.parent.parent else source.parent]:
        if (parent / "works.csv").exists() and (parent / "citations.csv").exists():
            return parent
    raise FileNotFoundError(f"Cannot infer corpus root from {source}")


def normalize_works_for_fig2(works: pd.DataFrame) -> pd.DataFrame:
    """Add the standard Fig.2/Fig.3 input columns while preserving reference metadata."""
    out = works.copy()
    if "id" not in out.columns:
        raise ValueError("works table must include id")
    out["id"] = out["id"].astype(str)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out = out[out["year"].notna()].copy()
    out["year"] = out["year"].astype(int)
    if "display_community" not in out.columns:
        out["display_community"] = pd.factorize(out.get("primary_field", out["domain"]).astype(str))[0] + 1
    out["display_community"] = pd.to_numeric(out["display_community"], errors="coerce").fillna(0).astype(int)
    out["analysis_community"] = pd.to_numeric(out.get("analysis_community", out["display_community"]), errors="coerce").fillna(out["display_community"]).astype(int)
    out["community"] = pd.to_numeric(out.get("community", out["display_community"]), errors="coerce").fillna(out["display_community"]).astype(int)
    if "primary_field" not in out.columns:
        out["primary_field"] = out["domain"].astype(str)
    if "is_landmark" not in out.columns:
        out["is_landmark"] = 0
    if "title" not in out.columns:
        out["title"] = out["id"]
    if "short_id" not in out.columns:
        out["short_id"] = out["id"].str.rsplit("/", n=1).str[-1]
    if "doi" not in out.columns:
        out["doi"] = ""
    if "cited_by_count" not in out.columns:
        out["cited_by_count"] = 0
    if "reference_count" not in out.columns:
        out["reference_count"] = 0
    if "source_dataset" not in out.columns:
        out["source_dataset"] = "unknown"
    if "legacy_is_landmark" not in out.columns:
        out["legacy_is_landmark"] = out["is_landmark"]
    if "reliable_anchor_source" not in out.columns:
        out["reliable_anchor_source"] = ""
    if "anchor_policy" not in out.columns:
        out["anchor_policy"] = ""
    if "anchor_label" not in out.columns:
        out["anchor_label"] = ""
    topic_label = out.get("display_topic_label", out.get("primary_topic", out["primary_field"]))
    out["primary_topic"] = topic_label.astype(str)
    out["community_label"] = out.get("community_label", out["primary_topic"]).astype(str)
    out["display_label"] = out.get("display_label", out["primary_topic"]).astype(str)
    if "is_closure_node" not in out.columns:
        out["is_closure_node"] = 0
    if "domain_analysis_end_year" not in out.columns:
        out["domain_analysis_end_year"] = out.groupby("domain")["year"].transform("max")
    return out


def add_matching_columns(eligible: pd.DataFrame, reference_count_bins: int = 10) -> pd.DataFrame:
    out = eligible.copy()
    out["publication_year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    out["article_type"] = out.get("document_type", "unknown")
    out["article_type"] = out["article_type"].fillna("unknown").astype(str)
    out["venue_family"] = out.get("source_dataset", "unknown")
    out["venue_family"] = out["venue_family"].fillna("unknown").astype(str)
    refs = pd.to_numeric(out.get("reference_count", 0), errors="coerce").fillna(0)
    if refs.nunique() <= 1:
        out["reference_count_decile"] = 0
    else:
        q = min(max(2, int(reference_count_bins)), refs.nunique())
        out["reference_count_decile"] = pd.qcut(refs.rank(method="first"), q=q, labels=False, duplicates="drop")
        out["reference_count_decile"] = out["reference_count_decile"].fillna(0).astype(int)
    out["reference_count_bins"] = int(reference_count_bins)
    return out


def select_control_tier(frame: pd.DataFrame, row: pd.Series, min_controls: int) -> tuple[str, int]:
    base = (frame["id"].astype(str) != str(row["id"])) & (pd.to_numeric(frame["is_landmark"], errors="coerce").fillna(0).astype(int) == 0)
    tiers = [
        ("exact", MATCH_COLUMNS_EXACT),
        ("relaxed_without_venue", MATCH_COLUMNS_RELAXED),
        ("domain_year_article", ["domain", "publication_year", "article_type"]),
        ("domain_year", ["domain", "publication_year"]),
        ("domain_all_years", ["domain"]),
    ]
    for tier, columns in tiers:
        mask = base.copy()
        for column in columns:
            mask &= frame[column].astype(str) == str(row[column])
        count = int(mask.sum())
        if count >= int(min_controls) or tier == "domain_all_years":
            return tier, count
    return "no_controls", 0


def _key_from_row(row: pd.Series, columns: Sequence[str]) -> tuple[object, ...]:
    return tuple(row[column] for column in columns)


def compute_control_tier_count_maps(frame: pd.DataFrame) -> Dict[str, Dict[tuple[object, ...], int]]:
    """Precompute non-landmark control counts for each matching tier."""
    if not set(MATCH_COLUMNS_EXACT).issubset(frame.columns):
        frame = add_matching_columns(frame)
    non_landmark = frame[pd.to_numeric(frame["is_landmark"], errors="coerce").fillna(0).astype(int) == 0].copy()
    count_maps: Dict[str, Dict[tuple[object, ...], int]] = {}
    for tier, columns in CONTROL_TIER_COLUMNS.items():
        if non_landmark.empty:
            count_maps[tier] = {}
            continue
        counts = non_landmark.groupby(columns, dropna=False).size()
        count_maps[tier] = {tuple(key if isinstance(key, tuple) else (key,)): int(value) for key, value in counts.items()}
    return count_maps


def select_control_tier_from_counts(
    row: pd.Series,
    count_maps: Mapping[str, Mapping[tuple[object, ...], int]],
    min_controls: int,
) -> tuple[str, int]:
    is_non_landmark = int(pd.to_numeric(pd.Series([row.get("is_landmark", 0)]), errors="coerce").fillna(0).iloc[0]) == 0
    for tier, columns in CONTROL_TIER_COLUMNS.items():
        key = _key_from_row(row, columns)
        count = int(count_maps.get(tier, {}).get(key, 0))
        if is_non_landmark:
            count = max(0, count - 1)
        if count >= int(min_controls) or tier == "domain_all_years":
            return tier, count
    return "no_controls", 0


def build_control_tier_audit(eligible: pd.DataFrame, min_controls: int = 20, reference_count_bins: int = 10) -> pd.DataFrame:
    frame = add_matching_columns(eligible, reference_count_bins=reference_count_bins)
    count_maps = compute_control_tier_count_maps(frame)
    rows: List[dict[str, object]] = []
    for _, row in frame.iterrows():
        tier, n_controls = select_control_tier_from_counts(row, count_maps=count_maps, min_controls=min_controls)
        rows.append(
            {
                "paper_id": row["id"],
                "domain": row["domain"],
                "publication_year": int(row["publication_year"]),
                "article_type": row["article_type"],
                "reference_count_decile": int(row["reference_count_decile"]),
                "reference_count_bins": int(row["reference_count_bins"]),
                "venue_family": row["venue_family"],
                "control_tier": tier,
                "n_controls": int(n_controls),
                "relaxed_tier": int(tier in RELAXED_CONTROL_TIERS or tier == "no_controls"),
            }
        )
    return pd.DataFrame(rows)


def copy_or_make_topics(corpus_root: Path, out_dir: Path, works: pd.DataFrame) -> None:
    topics_path = corpus_root / "topics.csv"
    topic_edges_path = corpus_root / "topic_edges.csv"
    if topics_path.exists():
        topics = pd.read_csv(topics_path)
    else:
        topic_rows = []
        for community, sub in works.groupby("display_community", sort=True):
            topic_rows.append(
                {
                    "community": int(community),
                    "label": str(sub["primary_topic"].iloc[0]),
                    "x": 0.0,
                    "y": 0.0,
                    "domain": str(sub["domain"].iloc[0]),
                    "topic_id": str(community),
                }
            )
        topics = pd.DataFrame(topic_rows)
    if topic_edges_path.exists():
        topic_edges = pd.read_csv(topic_edges_path)
    else:
        topic_edges = pd.DataFrame(columns=["source_community", "target_community", "weight"])
    topics.to_csv(out_dir / "topics.csv", index=False)
    topic_edges.to_csv(out_dir / "topic_edges.csv", index=False)


def write_domain_views(corpus_root: Path, out_dir: Path, works: pd.DataFrame, citations: pd.DataFrame) -> None:
    for domain, domain_works in works.groupby("domain", sort=True):
        domain_dir = out_dir / str(domain)
        domain_dir.mkdir(parents=True, exist_ok=True)
        ids = set(domain_works["id"].astype(str))
        domain_citations = citations[
            citations["source"].astype(str).isin(ids) & citations["target"].astype(str).isin(ids)
        ].copy()
        domain_works.to_csv(domain_dir / "works.csv", index=False)
        domain_citations.to_csv(domain_dir / "citations.csv", index=False)
        copy_or_make_topics(corpus_root, domain_dir, domain_works)
    multi = out_dir / "multi_domain"
    multi.mkdir(parents=True, exist_ok=True)
    works.to_csv(multi / "works.csv", index=False)
    citations.to_csv(multi / "citations.csv", index=False)
    copy_or_make_topics(corpus_root, multi, works)


def build_strong_inputs(
    source: Path,
    out_dir: Path,
    pre_cutoff_max_year: int,
    future_window_start: int,
    future_window_end: int,
    min_total_eligible: int = 8000,
    min_controls: int = 20,
    reference_count_bins: int = 10,
) -> Dict[str, object]:
    """Materialize Fig.2 strong-input tables from the full no-leakage corpus."""
    corpus_root = infer_corpus_root(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    works = normalize_works_for_fig2(pd.read_csv(corpus_root / "works.csv"))
    citations = pd.read_csv(corpus_root / "citations.csv")
    citations["source"] = citations["source"].astype(str)
    citations["target"] = citations["target"].astype(str)
    works = works[pd.to_numeric(works["year"], errors="coerce") <= int(future_window_end)].copy()
    eligible = works[pd.to_numeric(works["year"], errors="coerce") <= int(pre_cutoff_max_year)].copy()
    eligible = eligible[pd.to_numeric(eligible.get("is_closure_node", 0), errors="coerce").fillna(0).astype(int) == 0].copy()
    eligible.to_csv(out_dir / "fig2_eligible_papers.csv", index=False)
    closure, closure_report = build_reference_closure(eligible)
    closure.to_csv(out_dir / "reference_closure_table.csv", index=False)
    closure_report.to_csv(out_dir / "fig2_reference_closure_report.csv", index=False)
    control_audit = build_control_tier_audit(
        eligible,
        min_controls=min_controls,
        reference_count_bins=reference_count_bins,
    )
    control_audit.to_csv(out_dir / "fig2_control_tier_audit.csv", index=False)
    write_domain_views(corpus_root, out_dir, works, citations)
    relaxed_ratio = float(control_audit["relaxed_tier"].mean()) if not control_audit.empty else 1.0
    min_closure = float(pd.to_numeric(closure_report["coverage_materialized"], errors="coerce").min()) if not closure_report.empty else 0.0
    summary = {
        "source": str(source),
        "corpus_root": str(corpus_root),
        "pre_cutoff_max_year": int(pre_cutoff_max_year),
        "future_window_start": int(future_window_start),
        "future_window_end": int(future_window_end),
        "eligible_papers": int(len(eligible)),
        "min_total_eligible": int(min_total_eligible),
        "min_controls": int(min_controls),
        "reference_count_bins": int(reference_count_bins),
        "eligible_gate_pass": int(len(eligible) >= int(min_total_eligible)),
        "domains": sorted(eligible["domain"].astype(str).unique().tolist()),
        "relaxed_control_tier_ratio": relaxed_ratio,
        "control_gate_pass": int(relaxed_ratio <= 0.25),
        "min_reference_closure_coverage": min_closure,
        "reference_closure_gate_pass": int((closure_report["quality_gate_pass"].astype(int).min() if not closure_report.empty else 0) == 1),
    }
    atomic_write_text(out_dir / "fig2_strong_input_manifest.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strong Fig.2 input tables from the locked corpus.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pre-cutoff-max-year", type=int, default=2018)
    parser.add_argument("--future-window-start", type=int, default=2019)
    parser.add_argument("--future-window-end", type=int, default=2025)
    parser.add_argument("--min-total-eligible", type=int, default=8000)
    parser.add_argument("--min-controls", type=int, default=20)
    parser.add_argument("--reference-count-bins", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = build_strong_inputs(
        source=args.source,
        out_dir=args.out_dir,
        pre_cutoff_max_year=int(args.pre_cutoff_max_year),
        future_window_start=int(args.future_window_start),
        future_window_end=int(args.future_window_end),
        min_total_eligible=int(args.min_total_eligible),
        min_controls=int(args.min_controls),
        reference_count_bins=int(args.reference_count_bins),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
