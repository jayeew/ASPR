from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.corpus import (  # noqa: E402
    DEFAULT_COMPLETE_END_YEAR,
    DEFAULT_CORPUS_DIR,
    apply_strict_anchor_policy,
    audit_corpus,
    make_views,
    normalize_doi,
    normalize_openalex_id,
    normalize_title,
    short_openalex_id,
    slugify,
    stable_int_id,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "publication_corpus_v2"
DEFAULT_V2_CORPUS_DIR = PROJECT_ROOT / "data" / "knowledge_corpus" / "v2_publication"
DEFAULT_REPAIRED_SOURCE_DIR = PROJECT_ROOT / "data" / "knowledge_corpus" / "v1_strict_landmark_repaired"
DEFAULT_EXTERNAL_SOURCE_DIR = PROJECT_ROOT / "data" / "knowledge_corpus" / "v1_strict_landmark_external"
DEFAULT_TOPUP_SOURCE_DIR = PROJECT_ROOT / "data" / "knowledge_corpus" / "v1_strict_landmark_external_topup"
DEFAULT_DEDUP_SOURCE_DIR = PROJECT_ROOT / "data" / "knowledge_corpus" / "v1_strict_landmark_external_topup_dedup"
OPENALEX_WORK_SELECT = ",".join(
    [
        "id",
        "doi",
        "display_name",
        "publication_year",
        "type",
        "cited_by_count",
        "referenced_works",
        "primary_topic",
    ]
)
TRUSTED_LANDMARK_SOURCES = {
    "manual",
    "manual_landmark",
    "nobel",
    "award",
    "awards",
    "curated",
    "expert",
    "external_authority",
    "strict_manual",
    "strict_manual_v3",
}
LEGACY_LANDMARK_SOURCES = {"fig1_anchor"}
PUBLICATION_MAIN_TARGET = "main_candidate"
FIGURE_LOGIC_POLICY = "fixed_consumer_contract"
FIGURE_VIEW_CONTRACT: Dict[str, Sequence[str]] = {
    "fig1": ("works.csv", "citations.csv", "topics.csv", "topic_edges.csv"),
    "fig2": ("works.csv", "citations.csv", "topics.csv", "topic_edges.csv"),
    "fig3": ("works.csv", "citations.csv"),
    "fig5": ("works.csv", "citations.csv"),
}
DOMAIN_STATUS_VALUES = {
    "main_ready",
    "repair_landmark",
    "repair_closure",
    "repair_topic",
    "too_recent_for_main",
    "supplement_only",
    "drop",
}
LEAKAGE_KEYWORDS: Dict[str, Sequence[str]] = {
    "ipsc_reprogramming": ("crispr", "cas9", "cas12", "base editing", "prime editing"),
    "transformer_foundation_models": ("alphafold", "protein folding", "rna", "genomics", "drug discovery"),
    "graphene_2d_materials": ("protein", "rna", "clinical", "psychology"),
    "perovskite_solar_cells": ("cas9", "rna", "clinical", "patient"),
}


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file, returning an empty frame for absent or empty inputs."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write compact, deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_now() -> str:
    """Return a second-resolution UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def nonempty(value: object) -> str:
    """Normalize common empty sentinels to an empty string."""
    text = str(value or "").strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def safe_numeric(value: object, default: float = 0.0) -> float:
    """Convert one value to float without leaking pandas NA values."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return float(default)
    return float(number)


def _ensure_landmark_columns(landmarks: pd.DataFrame) -> pd.DataFrame:
    out = landmarks.copy()
    for col in [
        "domain",
        "landmark_source",
        "source_id",
        "label",
        "id",
        "doi",
        "title",
        "year",
        "match_confidence",
        "include_main",
    ]:
        if col not in out.columns:
            out[col] = ""
    return out


def clean_landmark_registry(
    landmarks: pd.DataFrame,
    max_landmarks_per_domain: int = 5,
    complete_end_year: int = DEFAULT_COMPLETE_END_YEAR,
) -> pd.DataFrame:
    """
    Build the v2 publication landmark seed registry.

    Legacy Fig. 1 anchors are allowed only when they are named and exactly
    identifiable; blank inherited anchors are dropped so they cannot recreate
    the v1 landmark inflation problem.
    """
    columns = [
        "domain",
        "landmark_source",
        "accepted_landmark_source",
        "source_id",
        "label",
        "id",
        "doi",
        "title",
        "year",
        "match_confidence",
        "include_main",
        "needs_manual_confirmation",
        "evidence_key",
    ]
    if landmarks.empty:
        return pd.DataFrame(columns=columns)

    out = _ensure_landmark_columns(landmarks)
    out["domain"] = out["domain"].map(slugify)
    out["landmark_source"] = out["landmark_source"].map(lambda value: nonempty(value).lower())
    out["source_id"] = out["source_id"].map(nonempty)
    out["label"] = out["label"].map(nonempty)
    out["label_norm"] = out["label"].map(normalize_title)
    out["id"] = out["id"].map(normalize_openalex_id)
    out["doi"] = out["doi"].map(normalize_doi)
    out["title"] = out["title"].map(nonempty)
    out["title_norm"] = out["title"].map(normalize_title)
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["match_confidence"] = pd.to_numeric(out["match_confidence"], errors="coerce").fillna(1.0)
    out["include_main"] = pd.to_numeric(out["include_main"], errors="coerce").fillna(1).astype(int)
    out = out[(out["include_main"] == 1) & (out["label"] != "")].copy()
    out = out[out["year"].isna() | (out["year"] <= int(complete_end_year))].copy()
    out["has_exact_evidence"] = (out["id"] != "") | (out["doi"] != "") | (out["title_norm"] != "")

    trusted = out["landmark_source"].isin(TRUSTED_LANDMARK_SOURCES)
    legacy_named = out["landmark_source"].isin(LEGACY_LANDMARK_SOURCES) & out["has_exact_evidence"]
    out = out[trusted | legacy_named].copy()
    if out.empty:
        return pd.DataFrame(columns=columns)

    out["needs_manual_confirmation"] = (~out["landmark_source"].isin(TRUSTED_LANDMARK_SOURCES)).astype(int)
    out["accepted_landmark_source"] = out["landmark_source"].where(
        out["needs_manual_confirmation"] == 0,
        "legacy_fig1_labeled_seed",
    )
    out["evidence_key"] = out["id"].where(out["id"] != "", out["doi"].where(out["doi"] != "", out["title_norm"]))
    out["_event_key"] = (
        out["label_norm"].astype(str)
        + "|"
        + out["year"].fillna(-1).astype(int).astype(str)
        + "|"
        + out["title_norm"].astype(str)
    )
    out["_source_priority"] = out["needs_manual_confirmation"]
    out["_has_id"] = out["id"].astype(str).ne("").astype(int)
    out = out.sort_values(
        ["_source_priority", "domain", "year", "match_confidence", "_has_id", "label"],
        ascending=[True, True, True, False, False, True],
    )
    out = out.drop_duplicates(["domain", "evidence_key"], keep="first")
    out = out.drop_duplicates(["domain", "_event_key"], keep="first")
    out = out.groupby("domain", group_keys=False).head(int(max_landmarks_per_domain)).reset_index(drop=True)
    return out[columns].copy()


def _quality_by_domain(corpus_dir: Path) -> Dict[str, Mapping[str, Any]]:
    path = corpus_dir / "quality_report.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {str(row.get("domain")): row for row in payload.get("domains", []) if row.get("domain")}


def _domain_display_lookup(domains: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    if domains.empty:
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for row in domains.to_dict("records"):
        slug = slugify(row.get("slug") or row.get("domain") or row.get("display_name"))
        out[slug] = {
            "display_name": nonempty(row.get("display_name")) or slug,
            "field_name": nonempty(row.get("field_name")),
            "subfield_name": nonempty(row.get("subfield_name")),
            "seed_source": nonempty(row.get("seed_source")),
        }
    return out


def _duplicate_doi_rate(frame: pd.DataFrame) -> float:
    doi = frame.get("doi", pd.Series("", index=frame.index)).map(normalize_doi)
    doi = doi[doi.astype(str) != ""]
    return float(doi.duplicated().mean()) if len(doi) else 0.0


def deduplicate_domain_dois(works: pd.DataFrame, citations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Drop duplicate DOI rows within each domain, preferring landmark/high-citation rows."""
    if works.empty:
        return works.copy(), citations.copy(), {"dropped_duplicate_works": 0, "input_works": 0, "output_works": 0}
    out = works.copy()
    out["_original_order"] = range(len(out))
    out["_doi_norm"] = out.get("doi", pd.Series("", index=out.index)).map(normalize_doi)
    out["_is_landmark_sort"] = pd.to_numeric(out.get("is_landmark", 0), errors="coerce").fillna(0).astype(int)
    out["_cited_sort"] = pd.to_numeric(out.get("cited_by_count", 0), errors="coerce").fillna(0)
    out["_domain_norm"] = out.get("domain", pd.Series("", index=out.index)).map(slugify)
    with_doi = out[out["_doi_norm"].astype(str).ne("")].copy()
    without_doi = out[out["_doi_norm"].astype(str).eq("")].copy()
    kept_with_doi = (
        with_doi.sort_values(
            ["_domain_norm", "_doi_norm", "_is_landmark_sort", "_cited_sort", "_original_order"],
            ascending=[True, True, False, False, True],
        )
        .drop_duplicates(["_domain_norm", "_doi_norm"], keep="first")
        .copy()
    )
    kept = pd.concat([kept_with_doi, without_doi], ignore_index=True, sort=False)
    kept = kept.sort_values("_original_order").copy()
    keep_ids = set(kept.get("id", pd.Series(dtype=str)).astype(str))
    dedup_citations = citations.copy()
    if not dedup_citations.empty and "source" in dedup_citations.columns:
        dedup_citations = dedup_citations[dedup_citations["source"].astype(str).isin(keep_ids)].copy()
    helper_cols = ["_original_order", "_doi_norm", "_is_landmark_sort", "_cited_sort", "_domain_norm"]
    kept = kept.drop(columns=[col for col in helper_cols if col in kept.columns]).reset_index(drop=True)
    report = {
        "input_works": int(len(works)),
        "output_works": int(len(kept)),
        "dropped_duplicate_works": int(len(works) - len(kept)),
        "citation_rows_after_source_filter": int(len(dedup_citations)),
    }
    return kept, dedup_citations.reset_index(drop=True), report


def _topic_coverage(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    label = frame.get("display_topic_label", frame.get("primary_topic", pd.Series("", index=frame.index)))
    return float(label.fillna("").astype(str).str.strip().ne("").mean())


def _local_reference_closure(citations: pd.DataFrame, ids: set[str]) -> float:
    if citations.empty or not {"source", "target"}.issubset(citations.columns):
        return 0.0
    sub = citations[citations["source"].astype(str).isin(ids)]
    if sub.empty:
        return 0.0
    return float(sub["target"].astype(str).isin(ids).mean())


def _clean_landmark_ids(clean_landmarks: pd.DataFrame) -> set[str]:
    if clean_landmarks.empty:
        return set()
    return set(clean_landmarks["id"].astype(str)) - {""}


def match_clean_landmarks_to_works(works: pd.DataFrame, clean_landmarks: pd.DataFrame) -> pd.DataFrame:
    """Return clean landmark rows that exactly match a work in the same domain."""
    if works.empty or clean_landmarks.empty:
        return pd.DataFrame(columns=list(clean_landmarks.columns) + ["matched_work_id", "matched_work_year", "matched_by"])

    w = works.copy()
    w["domain"] = w.get("domain", pd.Series("", index=w.index)).map(slugify)
    w["id_norm"] = w.get("id", pd.Series("", index=w.index)).map(normalize_openalex_id)
    w["doi_norm"] = w.get("doi", pd.Series("", index=w.index)).map(normalize_doi)
    w["title_norm"] = w.get("title", pd.Series("", index=w.index)).map(normalize_title)
    w["year"] = pd.to_numeric(w.get("year", pd.Series(dtype=float)), errors="coerce")

    rows: List[Dict[str, Any]] = []
    for row in clean_landmarks.to_dict("records"):
        domain = slugify(row.get("domain"))
        sub = w[w["domain"].astype(str) == domain]
        if sub.empty:
            continue
        keys = [
            ("id", "id_norm", normalize_openalex_id(row.get("id"))),
            ("doi", "doi_norm", normalize_doi(row.get("doi"))),
            ("title", "title_norm", normalize_title(row.get("title"))),
        ]
        match = pd.DataFrame()
        matched_by = ""
        for label, col, value in keys:
            if not value:
                continue
            match = sub[sub[col].astype(str) == value]
            if not match.empty:
                matched_by = label
                break
        if match.empty:
            continue
        work = match.sort_values(["year", "id_norm"]).iloc[0]
        out = dict(row)
        out["id"] = normalize_openalex_id(out.get("id")) or str(work.get("id_norm") or "")
        out["doi"] = normalize_doi(out.get("doi")) or normalize_doi(work.get("doi"))
        out["matched_work_id"] = str(work.get("id_norm") or "")
        out["matched_work_year"] = int(work.get("year")) if pd.notna(work.get("year")) else out.get("year", "")
        out["matched_by"] = matched_by
        rows.append(out)
    return pd.DataFrame(rows)


def _normalized_work_index(works: pd.DataFrame) -> pd.DataFrame:
    out = works.copy()
    out["domain"] = out.get("domain", pd.Series("", index=out.index)).map(slugify)
    out["id_norm"] = out.get("id", pd.Series("", index=out.index)).map(normalize_openalex_id)
    out["doi_norm"] = out.get("doi", pd.Series("", index=out.index)).map(normalize_doi)
    out["title_norm"] = out.get("title", pd.Series("", index=out.index)).map(normalize_title)
    out["year"] = pd.to_numeric(out.get("year", pd.Series(dtype=float)), errors="coerce")
    return out


def _exact_work_match(work_index: pd.DataFrame, landmark: Mapping[str, Any]) -> tuple[pd.DataFrame, str]:
    keys = [
        ("id", "id_norm", normalize_openalex_id(landmark.get("id"))),
        ("doi", "doi_norm", normalize_doi(landmark.get("doi"))),
        ("title", "title_norm", normalize_title(landmark.get("title"))),
    ]
    for label, col, value in keys:
        if not value or col not in work_index.columns:
            continue
        match = work_index[work_index[col].astype(str) == value].copy()
        if not match.empty:
            return match.sort_values(["domain", "year", "id_norm"]).copy(), label
    return pd.DataFrame(), ""


def build_global_landmark_repair_plan(works: pd.DataFrame, clean_landmarks: pd.DataFrame) -> pd.DataFrame:
    """Find clean landmarks that can be repaired by exact matches in another source domain."""
    columns = [
        "domain",
        "label",
        "year",
        "repair_status",
        "matched_by",
        "matched_work_id",
        "source_domain",
        "source_year",
        "source_doi",
        "source_title",
    ]
    if works.empty or clean_landmarks.empty:
        return pd.DataFrame(columns=columns)

    work_index = _normalized_work_index(works)
    rows: List[Dict[str, Any]] = []
    for landmark in clean_landmarks.to_dict("records"):
        domain = slugify(landmark.get("domain"))
        same_domain = work_index[work_index["domain"].astype(str) == domain].copy()
        same_match, same_by = _exact_work_match(same_domain, landmark)
        if not same_match.empty:
            work = same_match.iloc[0]
            status = "matched_in_domain"
            matched_by = same_by
        else:
            global_match, global_by = _exact_work_match(work_index, landmark)
            if not global_match.empty:
                global_match = global_match[global_match["domain"].astype(str) != domain].copy()
            if global_match.empty:
                rows.append(
                    {
                        "domain": domain,
                        "label": nonempty(landmark.get("label")),
                        "year": landmark.get("year", ""),
                        "repair_status": "external_fetch_required",
                        "matched_by": "",
                        "matched_work_id": "",
                        "source_domain": "",
                        "source_year": "",
                        "source_doi": "",
                        "source_title": "",
                    }
                )
                continue
            work = global_match.iloc[0]
            status = "global_source_match_available"
            matched_by = global_by

        rows.append(
            {
                "domain": domain,
                "label": nonempty(landmark.get("label")),
                "year": landmark.get("year", ""),
                "repair_status": status,
                "matched_by": matched_by,
                "matched_work_id": str(work.get("id_norm") or ""),
                "source_domain": str(work.get("domain") or ""),
                "source_year": "" if pd.isna(work.get("year")) else int(work.get("year")),
                "source_doi": normalize_doi(work.get("doi")),
                "source_title": nonempty(work.get("title")),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _dominant_value(frame: pd.DataFrame, column: str, default: object = "") -> object:
    if frame.empty or column not in frame.columns:
        return default
    values = frame[column].dropna()
    values = values[values.astype(str).str.strip() != ""]
    if values.empty:
        return default
    return values.mode().iloc[0]


def augment_corpus_with_global_landmark_repairs(
    source_corpus_dir: Path,
    target_corpus_dir: Path,
    domains: Optional[Sequence[str]] = None,
    max_landmarks_per_domain: int = 5,
) -> Dict[str, Any]:
    """Copy globally misplaced landmark rows into their target domains with audit columns."""
    works = read_csv(source_corpus_dir / "works.csv")
    citations = read_csv(source_corpus_dir / "citations.csv")
    topics = read_csv(source_corpus_dir / "topics.csv")
    topic_edges = read_csv(source_corpus_dir / "topic_edges.csv")
    domain_table = read_csv(source_corpus_dir / "domains.csv")
    landmarks = read_csv(source_corpus_dir / "landmarks.csv")
    clean_landmarks = clean_landmark_registry(landmarks, max_landmarks_per_domain=max_landmarks_per_domain)
    plan = build_global_landmark_repair_plan(works, clean_landmarks)

    selected_domains = {slugify(domain) for domain in domains} if domains else None
    repair_rows = plan[plan["repair_status"].astype(str) == "global_source_match_available"].copy()
    if selected_domains is not None:
        repair_rows = repair_rows[repair_rows["domain"].astype(str).isin(selected_domains)].copy()

    out_works = works.copy()
    if not out_works.empty:
        out_works["domain"] = out_works["domain"].map(slugify)
        out_works["id"] = out_works.get("id", pd.Series("", index=out_works.index)).map(normalize_openalex_id)
    work_index = _normalized_work_index(out_works)
    additions: List[Dict[str, Any]] = []
    for row in repair_rows.to_dict("records"):
        target_domain = str(row.get("domain") or "")
        matched_id = str(row.get("matched_work_id") or "")
        already_in_target = (
            (work_index["domain"].astype(str) == target_domain)
            & (work_index["id_norm"].astype(str) == matched_id)
        ).any()
        if already_in_target:
            continue
        source_match = out_works[
            (out_works["domain"].astype(str) == str(row.get("source_domain") or ""))
            & (out_works["id"].astype(str) == matched_id)
        ]
        if source_match.empty:
            continue
        source = source_match.iloc[0].to_dict()
        target_sub = out_works[out_works["domain"].astype(str) == target_domain]
        source["domain"] = target_domain
        source["is_landmark"] = 1
        source["anchor_label"] = nonempty(row.get("label"))
        source["source_dataset"] = "v2_global_landmark_repair"
        source["landmark_repair_source_domain"] = row.get("source_domain", "")
        source["landmark_repair_source_id"] = matched_id
        source["landmark_repair_matched_by"] = row.get("matched_by", "")
        source["display_community"] = _dominant_value(target_sub, "display_community", source.get("display_community", ""))
        source["display_topic_label"] = _dominant_value(target_sub, "display_topic_label", source.get("display_topic_label", ""))
        source["primary_field"] = _dominant_value(target_sub, "primary_field", source.get("primary_field", ""))
        additions.append(source)

    if additions:
        out_works = pd.concat([out_works, pd.DataFrame(additions)], ignore_index=True, sort=False)

    target_corpus_dir.mkdir(parents=True, exist_ok=True)
    views_dir = target_corpus_dir / "views"
    if views_dir.exists():
        shutil.rmtree(views_dir)
    out_works.to_csv(target_corpus_dir / "works.csv", index=False)
    citations.to_csv(target_corpus_dir / "citations.csv", index=False)
    topics.to_csv(target_corpus_dir / "topics.csv", index=False)
    topic_edges.to_csv(target_corpus_dir / "topic_edges.csv", index=False)
    domain_table.to_csv(target_corpus_dir / "domains.csv", index=False)
    landmarks.to_csv(target_corpus_dir / "landmarks.csv", index=False)
    plan.to_csv(target_corpus_dir / "landmark_repair_plan.csv", index=False)

    manifest = {
        "created_at": utc_now(),
        "artifact_kind": "v2_global_landmark_repair_source",
        "source_corpus_dir": str(source_corpus_dir),
        "target_corpus_dir": str(target_corpus_dir),
        "domains_requested": sorted(selected_domains) if selected_domains is not None else [],
        "n_repaired_landmarks": int(len(additions)),
        "n_global_matches_available": int(len(repair_rows)),
        "repair_plan": "landmark_repair_plan.csv",
    }
    write_json(target_corpus_dir / "global_landmark_repair_manifest.json", manifest)
    return manifest


def _openalex_work_url(identifier: object) -> str:
    ident = str(identifier or "").strip()
    if ident.lower().startswith("10.") or "doi.org" in ident.lower():
        doi = normalize_doi(ident)
        key = urllib.parse.quote(f"https://doi.org/{doi}", safe="")
    else:
        key = short_openalex_id(ident)
    return f"https://api.openalex.org/works/{key}?select={urllib.parse.quote(OPENALEX_WORK_SELECT, safe=',')}"


def fetch_openalex_work(identifier: object, timeout_seconds: int = 60) -> Optional[Dict[str, Any]]:
    """Fetch one OpenAlex work by DOI or OpenAlex ID using the standard library."""
    if not nonempty(identifier):
        return None
    url = _openalex_work_url(identifier)
    request = urllib.request.Request(url, headers={"User-Agent": "ASPR publication corpus builder"})
    try:
        with urllib.request.urlopen(request, timeout=int(timeout_seconds)) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except urllib.error.URLError:
        return None


def search_openalex_work_by_title(title: object, timeout_seconds: int = 60) -> Optional[Dict[str, Any]]:
    clean_title = nonempty(title)
    if not clean_title:
        return None
    params = urllib.parse.urlencode(
        {
            "search": clean_title,
            "per-page": 1,
            "select": OPENALEX_WORK_SELECT,
        }
    )
    request = urllib.request.Request(
        f"https://api.openalex.org/works?{params}",
        headers={"User-Agent": "ASPR publication corpus builder"},
    )
    try:
        with urllib.request.urlopen(request, timeout=int(timeout_seconds)) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError:
        return None
    results = payload.get("results") or []
    if not results:
        return None
    work = results[0]
    return work if normalize_title(work.get("display_name")) == normalize_title(clean_title) else None


def _load_external_records_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _external_records_from_openalex(
    source_corpus_dir: Path,
    domains: Optional[Sequence[str]] = None,
    timeout_seconds: int = 60,
    max_landmarks_per_domain: int = 5,
) -> List[Dict[str, Any]]:
    works = read_csv(source_corpus_dir / "works.csv")
    landmarks = clean_landmark_registry(
        read_csv(source_corpus_dir / "landmarks.csv"),
        max_landmarks_per_domain=max_landmarks_per_domain,
    )
    plan_path = source_corpus_dir / "landmark_repair_plan.csv"
    plan = read_csv(plan_path) if plan_path.exists() else build_global_landmark_repair_plan(works, landmarks)
    selected_domains = {slugify(domain) for domain in domains} if domains else None
    todo = plan[plan["repair_status"].astype(str) == "external_fetch_required"].copy()
    if selected_domains is not None:
        todo = todo[todo["domain"].astype(str).isin(selected_domains)].copy()

    records: List[Dict[str, Any]] = []
    for row in todo.to_dict("records"):
        domain = slugify(row.get("domain"))
        label = nonempty(row.get("label"))
        lm = landmarks[
            (landmarks["domain"].astype(str) == domain)
            & (landmarks["label"].astype(str) == label)
        ]
        if lm.empty:
            continue
        landmark = lm.iloc[0].to_dict()
        matched_by = ""
        work = None
        for key in ["id", "doi"]:
            value = landmark.get(key)
            if nonempty(value):
                work = fetch_openalex_work(value, timeout_seconds=timeout_seconds)
                if work:
                    matched_by = key
                    break
        if work is None:
            work = search_openalex_work_by_title(landmark.get("title"), timeout_seconds=timeout_seconds)
            matched_by = "title" if work else ""
        if work:
            records.append({"domain": domain, "label": label, "matched_by": matched_by, "work": work})
    return records


def _work_record_to_external_row(record: Mapping[str, Any], target_sub: pd.DataFrame) -> Dict[str, Any]:
    work = record.get("work") or {}
    domain = slugify(record.get("domain"))
    work_id = normalize_openalex_id(work.get("id"))
    refs = [normalize_openalex_id(ref) for ref in (work.get("referenced_works") or []) if normalize_openalex_id(ref)]
    topic = work.get("primary_topic") or {}
    topic_field = topic.get("field") or {}
    year = int(safe_numeric(work.get("publication_year"), default=0))
    return {
        "id": work_id,
        "short_id": short_openalex_id(work_id),
        "doi": normalize_doi(work.get("doi")),
        "title": nonempty(work.get("display_name")) or work_id,
        "year": year,
        "domain": domain,
        "primary_field": _dominant_value(target_sub, "primary_field", topic_field.get("display_name", "")),
        "display_community": _dominant_value(target_sub, "display_community", 0),
        "display_topic_id": normalize_openalex_id(topic.get("id")),
        "display_topic_label": _dominant_value(target_sub, "display_topic_label", topic.get("display_name", "")),
        "legacy_is_landmark": 1,
        "is_landmark": 1,
        "anchor_label": nonempty(record.get("label")),
        "reliable_anchor_source": "external_openalex_exact",
        "anchor_policy": "strict",
        "document_type": nonempty(work.get("type")),
        "cited_by_count": int(pd.to_numeric(work.get("cited_by_count"), errors="coerce") or 0),
        "reference_count": int(len(refs)),
        "source_provider": "openalex",
        "source_dataset": "v2_external_landmark_fetch",
        "fetched_at": utc_now(),
        "referenced_works": json.dumps(refs, ensure_ascii=False),
        "partial_2026": int(year >= 2026),
        "external_landmark_fetch_matched_by": nonempty(record.get("matched_by")),
    }


def _work_record_to_topup_row(record: Mapping[str, Any], target_sub: pd.DataFrame) -> Dict[str, Any]:
    work = record.get("work") or {}
    domain = slugify(record.get("domain"))
    work_id = normalize_openalex_id(work.get("id"))
    refs = [normalize_openalex_id(ref) for ref in (work.get("referenced_works") or []) if normalize_openalex_id(ref)]
    topic = work.get("primary_topic") or {}
    topic_field = topic.get("field") or {}
    year = int(safe_numeric(work.get("publication_year"), default=0))
    return {
        "id": work_id,
        "short_id": short_openalex_id(work_id),
        "doi": normalize_doi(work.get("doi")),
        "title": nonempty(work.get("display_name")) or work_id,
        "year": year,
        "domain": domain,
        "primary_field": _dominant_value(target_sub, "primary_field", topic_field.get("display_name", "")),
        "display_community": _dominant_value(target_sub, "display_community", 0),
        "display_topic_id": normalize_openalex_id(topic.get("id")),
        "display_topic_label": _dominant_value(target_sub, "display_topic_label", topic.get("display_name", "")),
        "legacy_is_landmark": 0,
        "is_landmark": 0,
        "anchor_label": "",
        "reliable_anchor_source": "",
        "anchor_policy": "strict",
        "document_type": nonempty(work.get("type")),
        "cited_by_count": int(safe_numeric(work.get("cited_by_count"), default=0)),
        "reference_count": int(len(refs)),
        "source_provider": "openalex",
        "source_dataset": "v2_openalex_topup",
        "fetched_at": utc_now(),
        "referenced_works": json.dumps(refs, ensure_ascii=False),
        "partial_2026": int(year >= 2026),
    }


def _parse_referenced_works(value: object) -> List[str]:
    """Parse a referenced_works cell from JSON, list, or loose string form."""
    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        text = nonempty(value)
        if not text:
            return []
        try:
            parsed = json.loads(text)
            raw_items = parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            raw_items = re.findall(r"https://openalex\.org/W\d+", text)
    return list(dict.fromkeys(normalize_openalex_id(item) for item in raw_items if normalize_openalex_id(item)))


def _work_record_to_reference_support_row(
    work: Mapping[str, Any],
    domain: str,
    target_sub: pd.DataFrame,
) -> Dict[str, Any]:
    topic = work.get("primary_topic") or {}
    topic_field = topic.get("field") or {}
    work_id = normalize_openalex_id(work.get("id"))
    refs = [normalize_openalex_id(ref) for ref in (work.get("referenced_works") or []) if normalize_openalex_id(ref)]
    year = int(safe_numeric(work.get("publication_year"), default=0))
    display_topic_id = normalize_openalex_id(topic.get("id"))
    return {
        "id": work_id,
        "short_id": short_openalex_id(work_id),
        "doi": normalize_doi(work.get("doi")),
        "title": nonempty(work.get("display_name")) or work_id,
        "year": year,
        "domain": domain,
        "primary_field": nonempty(topic_field.get("display_name")) or _dominant_value(target_sub, "primary_field", ""),
        "display_community": stable_int_id(display_topic_id or topic.get("display_name") or work_id, modulo=100000),
        "display_topic_id": display_topic_id,
        "display_topic_label": nonempty(topic.get("display_name")) or _dominant_value(target_sub, "display_topic_label", ""),
        "legacy_is_landmark": 0,
        "is_landmark": 0,
        "anchor_label": "",
        "reliable_anchor_source": "",
        "anchor_policy": "strict",
        "document_type": nonempty(work.get("type")),
        "cited_by_count": int(safe_numeric(work.get("cited_by_count"), default=0)),
        "reference_count": int(len(refs)),
        "source_provider": "openalex",
        "source_dataset": "v2_landmark_reference_support",
        "fetched_at": utc_now(),
        "referenced_works": json.dumps(refs, ensure_ascii=False),
        "partial_2026": int(year >= 2026),
        "reference_support_work": 1,
    }


def augment_corpus_with_external_landmark_records(
    source_corpus_dir: Path,
    target_corpus_dir: Path,
    records: Sequence[Mapping[str, Any]],
    domains: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Append externally fetched exact landmark works to target-domain corpora."""
    works = read_csv(source_corpus_dir / "works.csv")
    citations = read_csv(source_corpus_dir / "citations.csv")
    topics = read_csv(source_corpus_dir / "topics.csv")
    topic_edges = read_csv(source_corpus_dir / "topic_edges.csv")
    domain_table = read_csv(source_corpus_dir / "domains.csv")
    landmarks = read_csv(source_corpus_dir / "landmarks.csv")
    selected_domains = {slugify(domain) for domain in domains} if domains else None

    out_works = works.copy()
    if not out_works.empty:
        out_works["domain"] = out_works["domain"].map(slugify)
        out_works["id"] = out_works.get("id", pd.Series("", index=out_works.index)).map(normalize_openalex_id)
    if citations.empty:
        citations = pd.DataFrame(columns=["source", "target", "relation", "source_dataset"])
    out_citations = citations.copy()

    additions: List[Dict[str, Any]] = []
    citation_additions: List[Dict[str, Any]] = []
    for record in records:
        domain = slugify(record.get("domain"))
        if selected_domains is not None and domain not in selected_domains:
            continue
        work = record.get("work") or {}
        work_id = normalize_openalex_id(work.get("id"))
        if not work_id:
            continue
        already_in_target = (
            (out_works.get("domain", pd.Series("", index=out_works.index)).astype(str) == domain)
            & (out_works.get("id", pd.Series("", index=out_works.index)).astype(str) == work_id)
        ).any()
        if already_in_target:
            continue
        target_sub = out_works[out_works.get("domain", pd.Series("", index=out_works.index)).astype(str) == domain]
        new_row = _work_record_to_external_row(record, target_sub)
        additions.append(new_row)
        refs = json.loads(new_row["referenced_works"])
        citation_additions.extend(
            {
                "source": work_id,
                "target": ref,
                "relation": "reference",
                "source_dataset": "v2_external_landmark_fetch",
            }
            for ref in refs
        )

    if additions:
        out_works = pd.concat([out_works, pd.DataFrame(additions)], ignore_index=True, sort=False)
    if citation_additions:
        out_citations = pd.concat([out_citations, pd.DataFrame(citation_additions)], ignore_index=True, sort=False)
        out_citations = out_citations.drop_duplicates(["source", "target"], keep="first").reset_index(drop=True)

    target_corpus_dir.mkdir(parents=True, exist_ok=True)
    views_dir = target_corpus_dir / "views"
    if views_dir.exists():
        shutil.rmtree(views_dir)
    out_works.to_csv(target_corpus_dir / "works.csv", index=False)
    out_citations.to_csv(target_corpus_dir / "citations.csv", index=False)
    topics.to_csv(target_corpus_dir / "topics.csv", index=False)
    topic_edges.to_csv(target_corpus_dir / "topic_edges.csv", index=False)
    domain_table.to_csv(target_corpus_dir / "domains.csv", index=False)
    landmarks.to_csv(target_corpus_dir / "landmarks.csv", index=False)
    source_plan = source_corpus_dir / "landmark_repair_plan.csv"
    if source_plan.exists():
        shutil.copy2(source_plan, target_corpus_dir / "landmark_repair_plan.csv")

    manifest = {
        "created_at": utc_now(),
        "artifact_kind": "v2_external_landmark_fetch_source",
        "source_corpus_dir": str(source_corpus_dir),
        "target_corpus_dir": str(target_corpus_dir),
        "domains_requested": sorted(selected_domains) if selected_domains is not None else [],
        "n_external_landmarks_added": int(len(additions)),
        "n_external_reference_edges_added": int(len(citation_additions)),
    }
    write_json(target_corpus_dir / "external_landmark_fetch_manifest.json", manifest)
    return manifest


def augment_corpus_with_topup_work_records(
    source_corpus_dir: Path,
    target_corpus_dir: Path,
    records: Sequence[Mapping[str, Any]],
    domains: Optional[Sequence[str]] = None,
    min_local_refs: int = 0,
    local_references_only: bool = False,
) -> Dict[str, Any]:
    """Append ordinary OpenAlex works used only to top up near-threshold domains."""
    works = read_csv(source_corpus_dir / "works.csv")
    citations = read_csv(source_corpus_dir / "citations.csv")
    topics = read_csv(source_corpus_dir / "topics.csv")
    topic_edges = read_csv(source_corpus_dir / "topic_edges.csv")
    domain_table = read_csv(source_corpus_dir / "domains.csv")
    landmarks = read_csv(source_corpus_dir / "landmarks.csv")
    selected_domains = {slugify(domain) for domain in domains} if domains else None

    out_works = works.copy()
    if not out_works.empty:
        out_works["domain"] = out_works["domain"].map(slugify)
        out_works["id"] = out_works.get("id", pd.Series("", index=out_works.index)).map(normalize_openalex_id)
    out_citations = citations.copy() if not citations.empty else pd.DataFrame(columns=["source", "target", "relation", "source_dataset"])

    additions: List[Dict[str, Any]] = []
    citation_additions: List[Dict[str, Any]] = []
    existing_ids_by_domain = {
        domain: set(group.get("id", pd.Series(dtype=str)).astype(str))
        for domain, group in out_works.groupby(out_works.get("domain", pd.Series(dtype=str)).astype(str))
    }
    for record in records:
        domain = slugify(record.get("domain"))
        if selected_domains is not None and domain not in selected_domains:
            continue
        work = dict(record.get("work") or {})
        work_id = normalize_openalex_id(work.get("id"))
        year = int(safe_numeric(work.get("publication_year"), default=0))
        if not work_id or year > DEFAULT_COMPLETE_END_YEAR:
            continue
        refs = [normalize_openalex_id(ref) for ref in (work.get("referenced_works") or []) if normalize_openalex_id(ref)]
        local_refs = [ref for ref in refs if ref in existing_ids_by_domain.get(domain, set())]
        if len(local_refs) < int(min_local_refs):
            continue
        if local_references_only:
            work["referenced_works"] = local_refs
        already_in_domain = (
            (out_works.get("domain", pd.Series("", index=out_works.index)).astype(str) == domain)
            & (out_works.get("id", pd.Series("", index=out_works.index)).astype(str) == work_id)
        ).any()
        if already_in_domain:
            continue
        target_sub = out_works[out_works.get("domain", pd.Series("", index=out_works.index)).astype(str) == domain]
        new_row = _work_record_to_topup_row({**record, "work": work}, target_sub)
        additions.append(new_row)
        refs = json.loads(new_row["referenced_works"])
        citation_additions.extend(
            {
                "source": work_id,
                "target": ref,
                "relation": "reference",
                "source_dataset": "v2_openalex_topup",
            }
            for ref in refs
        )

    if additions:
        out_works = pd.concat([out_works, pd.DataFrame(additions)], ignore_index=True, sort=False)
    if citation_additions:
        out_citations = pd.concat([out_citations, pd.DataFrame(citation_additions)], ignore_index=True, sort=False)
        out_citations = out_citations.drop_duplicates(["source", "target"], keep="first").reset_index(drop=True)

    target_corpus_dir.mkdir(parents=True, exist_ok=True)
    views_dir = target_corpus_dir / "views"
    if views_dir.exists():
        shutil.rmtree(views_dir)
    out_works.to_csv(target_corpus_dir / "works.csv", index=False)
    out_citations.to_csv(target_corpus_dir / "citations.csv", index=False)
    topics.to_csv(target_corpus_dir / "topics.csv", index=False)
    topic_edges.to_csv(target_corpus_dir / "topic_edges.csv", index=False)
    domain_table.to_csv(target_corpus_dir / "domains.csv", index=False)
    landmarks.to_csv(target_corpus_dir / "landmarks.csv", index=False)
    for sidecar in ["landmark_repair_plan.csv", "global_landmark_repair_manifest.json", "external_landmark_fetch_manifest.json"]:
        source_sidecar = source_corpus_dir / sidecar
        if source_sidecar.exists():
            shutil.copy2(source_sidecar, target_corpus_dir / sidecar)

    manifest = {
        "created_at": utc_now(),
        "artifact_kind": "v2_openalex_topup_source",
        "source_corpus_dir": str(source_corpus_dir),
        "target_corpus_dir": str(target_corpus_dir),
        "domains_requested": sorted(selected_domains) if selected_domains is not None else [],
        "n_topup_works_added": int(len(additions)),
        "n_topup_reference_edges_added": int(len(citation_additions)),
        "local_references_only": bool(local_references_only),
        "min_local_refs": int(min_local_refs),
    }
    write_json(target_corpus_dir / "openalex_topup_manifest.json", manifest)
    return manifest


def _existing_work_to_reference_support_row(row: Mapping[str, Any], domain: str, target_sub: pd.DataFrame) -> Dict[str, Any]:
    out = dict(row)
    out["id"] = normalize_openalex_id(out.get("id"))
    out["doi"] = normalize_doi(out.get("doi"))
    out["domain"] = domain
    out["legacy_is_landmark"] = 0
    out["is_landmark"] = 0
    out["anchor_label"] = ""
    out["source_dataset"] = "v2_landmark_reference_support"
    out["reference_support_work"] = 1
    out["reference_support_copied_from_domain"] = nonempty(row.get("domain"))
    if not nonempty(out.get("display_topic_label")):
        out["display_topic_label"] = _dominant_value(target_sub, "display_topic_label", "")
    if not nonempty(out.get("primary_field")):
        out["primary_field"] = _dominant_value(target_sub, "primary_field", "")
    if not nonempty(out.get("display_community")):
        out["display_community"] = _dominant_value(target_sub, "display_community", 0)
    refs = _parse_referenced_works(out.get("referenced_works"))
    out["referenced_works"] = json.dumps(refs, ensure_ascii=False)
    out["reference_count"] = int(safe_numeric(out.get("reference_count"), default=len(refs)) or len(refs))
    return out


def augment_corpus_with_landmark_reference_support(
    source_corpus_dir: Path,
    target_corpus_dir: Path,
    domains: Optional[Sequence[str]] = None,
    min_internal_refs: int = 5,
    max_support_refs_per_landmark: int = 8,
    timeout_seconds: int = 60,
    max_landmarks_per_domain: int = 5,
) -> Dict[str, Any]:
    """Add cited reference target works so clean landmarks have local prior-reference support."""
    works = read_csv(source_corpus_dir / "works.csv")
    citations = read_csv(source_corpus_dir / "citations.csv")
    topics = read_csv(source_corpus_dir / "topics.csv")
    topic_edges = read_csv(source_corpus_dir / "topic_edges.csv")
    domain_table = read_csv(source_corpus_dir / "domains.csv")
    landmarks = read_csv(source_corpus_dir / "landmarks.csv")
    clean_landmarks = clean_landmark_registry(landmarks, max_landmarks_per_domain=max_landmarks_per_domain)
    selected_domains = {slugify(domain) for domain in domains} if domains else None

    if works.empty or "domain" not in works.columns:
        raise ValueError(f"{source_corpus_dir / 'works.csv'} has no usable domain rows")

    out_works = works.copy()
    out_works["domain"] = out_works.get("domain", pd.Series("", index=out_works.index)).map(slugify)
    out_works["id"] = out_works.get("id", pd.Series("", index=out_works.index)).map(normalize_openalex_id)
    out_works["year"] = pd.to_numeric(out_works.get("year", pd.Series(dtype=float)), errors="coerce")
    out_citations = citations.copy() if not citations.empty else pd.DataFrame(columns=["source", "target", "relation", "source_dataset"])
    for col in ["source", "target"]:
        if col not in out_citations.columns:
            out_citations[col] = ""
        out_citations[col] = out_citations[col].map(normalize_openalex_id)
    if "relation" not in out_citations.columns:
        out_citations["relation"] = "reference"
    if "source_dataset" not in out_citations.columns:
        out_citations["source_dataset"] = "unknown"

    report_rows: List[Dict[str, Any]] = []
    support_additions: List[Dict[str, Any]] = []
    citation_additions: List[Dict[str, Any]] = []
    fetched_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    target_domains = sorted(selected_domains or set(out_works["domain"].dropna().astype(str)))
    for domain in target_domains:
        target_sub = out_works[out_works["domain"].astype(str) == domain].copy()
        if target_sub.empty:
            continue
        domain_landmarks = clean_landmarks[clean_landmarks["domain"].astype(str) == domain].copy()
        matched = match_clean_landmarks_to_works(target_sub, domain_landmarks)
        for landmark in matched.to_dict("records"):
            source_id = normalize_openalex_id(landmark.get("matched_work_id") or landmark.get("id"))
            source_year = int(safe_numeric(landmark.get("matched_work_year", landmark.get("year")), default=0))
            if not source_id or source_year <= 0:
                continue

            def current_domain_ids() -> set[str]:
                return set(out_works[out_works["domain"].astype(str) == domain]["id"].astype(str))

            def current_internal_ref_count() -> int:
                ids = current_domain_ids()
                domain_year = dict(
                    zip(
                        out_works[out_works["domain"].astype(str) == domain]["id"].astype(str),
                        pd.to_numeric(
                            out_works[out_works["domain"].astype(str) == domain]["year"],
                            errors="coerce",
                        ),
                    )
                )
                existing = out_citations[
                    (out_citations["source"].astype(str) == source_id)
                    & (out_citations["target"].astype(str).isin(ids))
                ].copy()
                if existing.empty:
                    return 0
                years = existing["target"].astype(str).map(domain_year)
                return int((pd.to_numeric(years, errors="coerce") < source_year).sum())

            before_refs = current_internal_ref_count()
            if before_refs >= int(min_internal_refs):
                report_rows.append(
                    {
                        "domain": domain,
                        "label": nonempty(landmark.get("label")),
                        "matched_work_id": source_id,
                        "source_year": source_year,
                        "refs_before": before_refs,
                        "refs_after": before_refs,
                        "support_works_added": 0,
                        "citation_edges_added": 0,
                        "status": "already_sufficient",
                    }
                )
                continue

            source_row = out_works[(out_works["domain"].astype(str) == domain) & (out_works["id"].astype(str) == source_id)]
            ref_ids: List[str] = []
            if not source_row.empty:
                ref_ids = _parse_referenced_works(source_row.iloc[0].get("referenced_works"))
            if not ref_ids:
                work_key = nonempty(landmark.get("doi")) or source_id
                fetched_cache.setdefault(work_key, fetch_openalex_work(work_key, timeout_seconds=timeout_seconds))
                fetched_source = fetched_cache.get(work_key)
                ref_ids = [
                    normalize_openalex_id(ref)
                    for ref in ((fetched_source or {}).get("referenced_works") or [])
                    if normalize_openalex_id(ref)
                ]
                if fetched_source is not None:
                    refs_json = json.dumps(ref_ids, ensure_ascii=False)
                    mask = out_works["id"].astype(str) == source_id
                    out_works.loc[mask, "referenced_works"] = refs_json
                    out_works.loc[mask, "reference_count"] = len(ref_ids)

            added_support = 0
            added_edges = 0
            for ref_id in ref_ids:
                if current_internal_ref_count() >= int(min_internal_refs):
                    break
                if added_support >= int(max_support_refs_per_landmark):
                    break
                if not ref_id:
                    continue

                domain_ids = current_domain_ids()
                ref_year = None
                if ref_id in domain_ids:
                    ref_rows = out_works[(out_works["domain"].astype(str) == domain) & (out_works["id"].astype(str) == ref_id)]
                    ref_year = int(safe_numeric(ref_rows.iloc[0].get("year"), default=0)) if not ref_rows.empty else None
                else:
                    global_rows = out_works[out_works["id"].astype(str) == ref_id].copy()
                    if not global_rows.empty:
                        candidate = global_rows.iloc[0].to_dict()
                        ref_year = int(safe_numeric(candidate.get("year"), default=0))
                        if ref_year and ref_year < source_year:
                            support_additions.append(_existing_work_to_reference_support_row(candidate, domain, target_sub))
                            out_works = pd.concat([out_works, pd.DataFrame([support_additions[-1]])], ignore_index=True, sort=False)
                            added_support += 1
                    else:
                        fetched_cache.setdefault(ref_id, fetch_openalex_work(ref_id, timeout_seconds=timeout_seconds))
                        ref_work = fetched_cache.get(ref_id)
                        ref_year = int(safe_numeric((ref_work or {}).get("publication_year"), default=0))
                        if ref_work is not None and ref_year and ref_year < source_year:
                            support_row = _work_record_to_reference_support_row(ref_work, domain, target_sub)
                            support_additions.append(support_row)
                            out_works = pd.concat([out_works, pd.DataFrame([support_row])], ignore_index=True, sort=False)
                            added_support += 1

                if ref_year is None or ref_year >= source_year:
                    continue
                edge_exists = (
                    (out_citations["source"].astype(str) == source_id)
                    & (out_citations["target"].astype(str) == ref_id)
                ).any()
                if not edge_exists:
                    edge = {
                        "source": source_id,
                        "target": ref_id,
                        "relation": "reference",
                        "source_dataset": "v2_landmark_reference_support",
                    }
                    citation_additions.append(edge)
                    out_citations = pd.concat([out_citations, pd.DataFrame([edge])], ignore_index=True, sort=False)
                    added_edges += 1

            after_refs = current_internal_ref_count()
            report_rows.append(
                {
                    "domain": domain,
                    "label": nonempty(landmark.get("label")),
                    "matched_work_id": source_id,
                    "source_year": source_year,
                    "refs_before": before_refs,
                    "refs_after": after_refs,
                    "support_works_added": added_support,
                    "citation_edges_added": added_edges,
                    "status": "repaired" if after_refs >= int(min_internal_refs) else "insufficient_reference_support",
                }
            )

    out_citations = out_citations.drop_duplicates(["source", "target"], keep="first").reset_index(drop=True)
    target_corpus_dir.mkdir(parents=True, exist_ok=True)
    views_dir = target_corpus_dir / "views"
    if views_dir.exists():
        shutil.rmtree(views_dir)
    out_works.to_csv(target_corpus_dir / "works.csv", index=False)
    out_citations.to_csv(target_corpus_dir / "citations.csv", index=False)
    topics.to_csv(target_corpus_dir / "topics.csv", index=False)
    topic_edges.to_csv(target_corpus_dir / "topic_edges.csv", index=False)
    domain_table.to_csv(target_corpus_dir / "domains.csv", index=False)
    landmarks.to_csv(target_corpus_dir / "landmarks.csv", index=False)
    for sidecar in [
        "landmark_repair_plan.csv",
        "global_landmark_repair_manifest.json",
        "external_landmark_fetch_manifest.json",
        "openalex_topup_manifest.json",
        "doi_dedupe_manifest.json",
        "metadata_repair_manifest.json",
    ]:
        source_sidecar = source_corpus_dir / sidecar
        if source_sidecar.exists():
            shutil.copy2(source_sidecar, target_corpus_dir / sidecar)
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(target_corpus_dir / "landmark_reference_support_report.csv", index=False)
    manifest = {
        "created_at": utc_now(),
        "artifact_kind": "v2_landmark_reference_support_source",
        "source_corpus_dir": str(source_corpus_dir),
        "target_corpus_dir": str(target_corpus_dir),
        "domains_requested": sorted(selected_domains) if selected_domains is not None else [],
        "min_internal_refs": int(min_internal_refs),
        "max_support_refs_per_landmark": int(max_support_refs_per_landmark),
        "n_support_works_added": int(sum(int(row.get("support_works_added", 0)) for row in report_rows)),
        "n_reference_edges_added": int(sum(int(row.get("citation_edges_added", 0)) for row in report_rows)),
        "n_landmarks_repaired": int((report_df.get("status", pd.Series(dtype=str)).astype(str) == "repaired").sum())
        if not report_df.empty
        else 0,
        "report": "landmark_reference_support_report.csv",
    }
    write_json(target_corpus_dir / "landmark_reference_support_manifest.json", manifest)
    return manifest


def repair_corpus_metadata(
    source_corpus_dir: Path,
    target_corpus_dir: Path,
    domains: Optional[Sequence[str]] = None,
    max_landmarks_per_domain: int = 5,
) -> Dict[str, Any]:
    """Repair metadata-only issues without changing citation or figure logic."""
    works = read_csv(source_corpus_dir / "works.csv")
    landmarks = read_csv(source_corpus_dir / "landmarks.csv")
    if works.empty:
        raise ValueError(f"{source_corpus_dir / 'works.csv'} has no works")

    target_domains = {slugify(domain) for domain in (domains or []) if str(domain).strip()}
    out_works = works.copy()
    out_works["domain"] = out_works.get("domain", pd.Series("", index=out_works.index)).map(slugify)
    if "display_topic_label" not in out_works.columns:
        out_works["display_topic_label"] = ""
    fill_mask = out_works["display_topic_label"].fillna("").astype(str).str.strip().eq("")
    if target_domains:
        fill_mask &= out_works["domain"].astype(str).isin(target_domains)

    fallback = pd.Series("", index=out_works.index, dtype=object)
    for col in ["primary_topic", "primary_field"]:
        if col in out_works.columns:
            candidate = out_works[col].fillna("").astype(str).str.strip()
            fallback = fallback.where(fallback.astype(str).str.strip().ne(""), candidate)
    filled_mask = fill_mask & fallback.astype(str).str.strip().ne("")
    out_works.loc[filled_mask, "display_topic_label"] = fallback[filled_mask]

    clean_landmarks = clean_landmark_registry(landmarks, max_landmarks_per_domain=max_landmarks_per_domain)

    target_corpus_dir.mkdir(parents=True, exist_ok=True)
    views_dir = target_corpus_dir / "views"
    if views_dir.exists():
        shutil.rmtree(views_dir)
    out_works.to_csv(target_corpus_dir / "works.csv", index=False)
    clean_landmarks.to_csv(target_corpus_dir / "landmarks.csv", index=False)
    for name in [
        "citations.csv",
        "topics.csv",
        "topic_edges.csv",
        "domains.csv",
        "landmark_repair_plan.csv",
        "global_landmark_repair_manifest.json",
        "external_landmark_fetch_manifest.json",
        "openalex_topup_manifest.json",
        "doi_dedupe_manifest.json",
    ]:
        src = source_corpus_dir / name
        if src.exists():
            shutil.copy2(src, target_corpus_dir / name)

    manifest = {
        "created_at": utc_now(),
        "artifact_kind": "v2_metadata_repair_source",
        "source_corpus_dir": str(source_corpus_dir),
        "target_corpus_dir": str(target_corpus_dir),
        "domains_requested": sorted(target_domains),
        "n_topic_labels_filled": int(filled_mask.sum()),
        "n_landmarks_input": int(len(landmarks)),
        "n_landmarks_output": int(len(clean_landmarks)),
    }
    write_json(target_corpus_dir / "metadata_repair_manifest.json", manifest)
    return manifest


def fetch_openalex_works_for_query(
    query: object,
    max_records: int,
    timeout_seconds: int = 60,
) -> List[Dict[str, Any]]:
    clean_query = nonempty(query)
    if not clean_query or max_records <= 0:
        return []
    out: List[Dict[str, Any]] = []
    cursor = "*"
    per_page = min(max(int(max_records), 10), 200)
    while len(out) < int(max_records) and cursor:
        params = urllib.parse.urlencode(
            {
                "search": clean_query,
                "per-page": per_page,
                "cursor": cursor,
                "select": OPENALEX_WORK_SELECT,
            }
        )
        request = urllib.request.Request(
            f"https://api.openalex.org/works?{params}",
            headers={"User-Agent": "ASPR publication corpus builder"},
        )
        try:
            with urllib.request.urlopen(request, timeout=int(timeout_seconds)) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError:
            break
        results = payload.get("results") or []
        if not results:
            break
        for work in results:
            year = int(safe_numeric(work.get("publication_year"), default=0))
            if 0 < year <= DEFAULT_COMPLETE_END_YEAR:
                out.append(work)
            if len(out) >= int(max_records):
                break
        cursor = (payload.get("meta") or {}).get("next_cursor")
    return out


def openalex_search_query_from_domain_query(query: object) -> str:
    """Flatten simple quoted OR seed queries into OpenAlex search text."""
    text = nonempty(query)
    if not text:
        return ""
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        return " ".join(dict.fromkeys(part.strip() for part in quoted if part.strip()))
    text = re.sub(r"\b(OR|AND)\b", " ", text, flags=re.I)
    text = text.replace('"', " ")
    return re.sub(r"\s+", " ", text).strip()


def _topup_records_from_openalex(
    source_corpus_dir: Path,
    domains: Sequence[str],
    target_papers: int = 2500,
    max_extra_per_domain: int = 25,
    timeout_seconds: int = 60,
    min_local_refs: int = 0,
    local_references_only: bool = False,
) -> List[Dict[str, Any]]:
    works = read_csv(source_corpus_dir / "works.csv")
    domain_table = read_csv(source_corpus_dir / "domains.csv")
    if works.empty or "domain" not in works.columns:
        return []
    works = works.copy()
    works["domain"] = works["domain"].map(slugify)
    existing_ids = set(works.get("id", pd.Series(dtype=str)).map(normalize_openalex_id).astype(str))
    domain_table = domain_table.copy() if not domain_table.empty else pd.DataFrame()
    if not domain_table.empty and "slug" in domain_table.columns:
        domain_table["slug"] = domain_table["slug"].map(slugify)
    records: List[Dict[str, Any]] = []
    for domain in [slugify(item) for item in domains]:
        domain_works = works[works["domain"].astype(str) == domain].copy()
        current_n = int(len(domain_works))
        needed = int(target_papers) - current_n
        if needed <= 0:
            continue
        domain_ids = set(domain_works.get("id", pd.Series(dtype=str)).map(normalize_openalex_id).astype(str))
        domain_meta = domain_table[domain_table.get("slug", pd.Series(dtype=str)).astype(str) == domain]
        query = domain
        if not domain_meta.empty:
            query = nonempty(domain_meta.iloc[0].get("query")) or nonempty(domain_meta.iloc[0].get("display_name")) or domain
        query = openalex_search_query_from_domain_query(query)
        candidates = fetch_openalex_works_for_query(
            query,
            max_records=int(max_extra_per_domain),
            timeout_seconds=timeout_seconds,
        )
        added = 0
        for work in candidates:
            work_id = normalize_openalex_id(work.get("id"))
            if not work_id or work_id in existing_ids:
                continue
            work = dict(work)
            refs = [normalize_openalex_id(ref) for ref in (work.get("referenced_works") or []) if normalize_openalex_id(ref)]
            local_refs = [ref for ref in refs if ref in domain_ids]
            if len(local_refs) < int(min_local_refs):
                continue
            if local_references_only:
                work["referenced_works"] = local_refs
            records.append({"domain": domain, "work": work})
            existing_ids.add(work_id)
            added += 1
            if added >= needed:
                break
    return records


def _min_controls_for_matched_landmarks(
    works: pd.DataFrame,
    matched_landmarks: pd.DataFrame,
    year_window: int = 5,
) -> Optional[int]:
    if works.empty or matched_landmarks.empty:
        return None
    landmark_ids = set(matched_landmarks.get("matched_work_id", pd.Series(dtype=str)).astype(str)) - {""}
    controls = works[~works.get("id", pd.Series("", index=works.index)).astype(str).isin(landmark_ids)].copy()
    control_years = pd.to_numeric(controls.get("year", pd.Series(dtype=float)), errors="coerce")
    counts: List[int] = []
    for row in matched_landmarks.to_dict("records"):
        year = safe_numeric(row.get("matched_work_year", row.get("year")), default=float("nan"))
        if not math.isfinite(year):
            continue
        counts.append(int(control_years.between(int(year) - year_window, int(year) + year_window).sum()))
    return min(counts) if counts else None


def _min_controls_for_landmarks(works: pd.DataFrame, clean_landmarks: pd.DataFrame, year_window: int = 5) -> Optional[int]:
    if works.empty or clean_landmarks.empty:
        return None
    years = pd.to_numeric(works.get("year", pd.Series(dtype=float)), errors="coerce")
    landmark_ids = _clean_landmark_ids(clean_landmarks)
    controls = works[~works.get("id", pd.Series("", index=works.index)).astype(str).isin(landmark_ids)].copy()
    control_years = pd.to_numeric(controls.get("year", pd.Series(dtype=float)), errors="coerce")
    counts: List[int] = []
    for row in clean_landmarks.to_dict("records"):
        year = safe_numeric(row.get("year"), default=float("nan"))
        if not math.isfinite(year):
            continue
        mask = control_years.between(int(year) - year_window, int(year) + year_window)
        counts.append(int(mask.sum()))
    if not counts and years.notna().any():
        return None
    return min(counts) if counts else None


def _topic_leakage_rate(domain: str, frame: pd.DataFrame) -> float:
    keywords = LEAKAGE_KEYWORDS.get(domain, ())
    if frame.empty or not keywords:
        return 0.0
    text = (
        frame.get("title", pd.Series("", index=frame.index)).fillna("").astype(str)
        + " "
        + frame.get("display_topic_label", pd.Series("", index=frame.index)).fillna("").astype(str)
        + " "
        + frame.get("primary_field", pd.Series("", index=frame.index)).fillna("").astype(str)
    ).str.lower()
    pattern = "|".join(pd.Series(list(keywords)).map(lambda value: str(value).replace("|", r"\|")))
    return float(text.str.contains(pattern, regex=True, na=False).mean()) if pattern else 0.0


def _role_from_failures(failures: Sequence[str], publication_score: float) -> str:
    if not failures:
        return PUBLICATION_MAIN_TARGET
    rebuild_reasons = {
        "duplicate_doi",
        "landmark_controls",
        "legacy_landmark_inflation",
        "missing_clean_landmark",
        "n_works",
        "reference_closure",
        "topic_coverage",
        "topic_leakage",
        "unmatched_clean_landmark",
    }
    if publication_score >= 0.50 and any(reason in rebuild_reasons for reason in failures):
        return "rebuild_needed"
    return "exclude_for_now"


def _split_reasons(value: object) -> set[str]:
    text = nonempty(value)
    if not text:
        return set()
    return {part.strip() for part in text.split(";") if part.strip()}


def _domain_family(field_name: object, display_name: object = "", domain: object = "") -> str:
    text = " ".join([nonempty(field_name), nonempty(display_name), nonempty(domain)]).lower()
    if any(
        token in text
        for token in [
            "banking",
            "economic",
            "economics",
            "finance",
            "financial",
            "income",
            "inequality",
            "monetary",
            "poverty",
            "regulation",
        ]
    ):
        return "social_economics_policy"
    if any(token in text for token in ["biology", "medicine", "biomedical", "genetics", "genomics", "microbiome"]):
        return "biology_biomedicine"
    if any(token in text for token in ["materials", "chemistry", "chemical", "nanotechnology", "perovskite", "graphene"]):
        return "materials_chemistry"
    if any(token in text for token in ["physics", "astronomy", "astrophysics", "cosmology", "gamma", "supernova"]):
        return "physics_astronomy"
    if any(token in text for token in ["computer", "artificial intelligence", "machine learning", "measurement", "imaging"]):
        return "computational_methods"
    return "other_science"


def classify_domain_status(row: Mapping[str, Any]) -> tuple[str, str]:
    """Map diagnostics into one publication decision status."""
    failures = _split_reasons(row.get("failure_reasons"))
    fig3_failures = _split_reasons(row.get("fig3_readiness_failures"))
    role = nonempty(row.get("recommended_role"))
    excluded = bool(row.get("excluded_by_policy", False))
    fig3_ready_raw = row.get("fig3_ready", True)
    fig3_ready = bool(fig3_ready_raw) if not pd.isna(fig3_ready_raw) else True

    if excluded:
        return "drop", "excluded_by_policy"
    if role == PUBLICATION_MAIN_TARGET and not failures and fig3_ready:
        return "main_ready", "passes_all_current_data_gates"
    if {"topic_coverage", "topic_leakage"} & failures:
        return "repair_topic", ";".join(sorted({"topic_coverage", "topic_leakage"} & failures))
    if "reference_closure" in failures:
        return "repair_closure", "reference_closure"
    if "duplicate_doi" in failures:
        return "repair_closure", "duplicate_doi"
    if {"missing_clean_landmark", "unmatched_clean_landmark", "landmark_controls"} & failures:
        return "repair_landmark", ";".join(
            sorted({"missing_clean_landmark", "unmatched_clean_landmark", "landmark_controls"} & failures)
        )
    if "fig3_metric_landmarks" in fig3_failures:
        return "repair_landmark", "fig3_metric_landmarks"
    if fig3_failures == {"fig3_metric_papers"}:
        return "too_recent_for_main", "fig3_metric_papers"
    if "n_works" in failures:
        return "supplement_only", "n_works"
    if role == PUBLICATION_MAIN_TARGET and fig3_failures:
        return "supplement_only", ";".join(sorted(fig3_failures))
    return "drop", ";".join(sorted(failures | fig3_failures)) or "low_publication_score"


def build_domain_status_table(
    diagnostics: pd.DataFrame,
    fig3_readiness: Optional[pd.DataFrame] = None,
    clean_landmarks: Optional[pd.DataFrame] = None,
    excluded_domains: Optional[Sequence[str]] = None,
    excluded_families: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build the auditable domain inclusion/exclusion table for publication selection."""
    if diagnostics.empty:
        return pd.DataFrame()
    out = diagnostics.copy()
    out["domain"] = out["domain"].map(slugify)
    if fig3_readiness is not None and not fig3_readiness.empty:
        fig3 = fig3_readiness.copy()
        fig3["domain"] = fig3["domain"].map(slugify)
        merge_cols = [
            "domain",
            "fig3_ready",
            "fig3_readiness_failures",
            "fig3_analysis_end_year",
            "fig3_eligible_metric_papers",
            "fig3_eligible_metric_landmarks",
        ]
        out = out.merge(fig3[[col for col in merge_cols if col in fig3.columns]], on="domain", how="left")
    else:
        out["fig3_ready"] = True
        out["fig3_readiness_failures"] = ""
        out["fig3_analysis_end_year"] = DEFAULT_COMPLETE_END_YEAR
        out["fig3_eligible_metric_papers"] = ""
        out["fig3_eligible_metric_landmarks"] = ""

    excluded = {slugify(domain) for domain in (excluded_domains or []) if str(domain).strip()}
    out["excluded_by_policy"] = out["domain"].astype(str).isin(excluded)

    landmark_years: Dict[str, int] = {}
    landmark_policy: Dict[str, str] = {}
    if clean_landmarks is not None and not clean_landmarks.empty:
        lm = clean_landmarks.copy()
        lm["domain"] = lm["domain"].map(slugify)
        lm["year"] = pd.to_numeric(lm.get("year", pd.Series(dtype=float)), errors="coerce")
        for domain, group in lm.groupby("domain"):
            years = group["year"].dropna()
            if not years.empty:
                landmark_years[str(domain)] = int(years.min())
            ids = group.get("id", pd.Series("", index=group.index)).map(normalize_openalex_id)
            dois = group.get("doi", pd.Series("", index=group.index)).map(normalize_doi)
            if ids.astype(str).ne("").any() or dois.astype(str).ne("").any():
                landmark_policy[str(domain)] = "validated_doi_or_openalex"
            elif len(group):
                landmark_policy[str(domain)] = "title_year_seed"

    statuses: List[str] = []
    reasons: List[str] = []
    for record in out.to_dict("records"):
        status, reason = classify_domain_status(record)
        statuses.append(status)
        reasons.append(reason)
    out["status"] = statuses
    out["reason_for_inclusion_or_exclusion"] = reasons
    out["domain_id"] = out["domain"]
    out["family"] = [
        _domain_family(row.get("field_name"), row.get("display_name"), row.get("domain"))
        for row in out.to_dict("records")
    ]
    excluded_family_set = {nonempty(family) for family in (excluded_families or []) if nonempty(family)}
    if excluded_family_set:
        family_excluded = out["family"].astype(str).isin(excluded_family_set)
        out.loc[family_excluded, "status"] = "drop"
        out.loc[family_excluded, "reason_for_inclusion_or_exclusion"] = "excluded_family"
    out["event_year"] = out["domain"].astype(str).map(landmark_years)
    out["analysis_end_year"] = out.get("fig3_analysis_end_year", DEFAULT_COMPLETE_END_YEAR)
    out["landmark_policy"] = out["domain"].astype(str).map(landmark_policy).fillna("no_clean_landmark")
    out["notes"] = out["reason_for_inclusion_or_exclusion"]

    preferred = [
        "domain_id",
        "display_name",
        "family",
        "status",
        "event_year",
        "analysis_end_year",
        "landmark_policy",
        "n_works",
        "local_reference_closure",
        "topic_coverage",
        "duplicate_doi_rate",
        "clean_landmark_rows",
        "matched_clean_landmark_rows",
        "fig3_ready",
        "fig3_eligible_metric_papers",
        "fig3_eligible_metric_landmarks",
        "recommended_role",
        "failure_reasons",
        "fig3_readiness_failures",
        "publication_score",
        "reason_for_inclusion_or_exclusion",
        "notes",
    ]
    columns = [col for col in preferred if col in out.columns]
    status_rank = {
        "main_ready": 0,
        "repair_landmark": 1,
        "repair_closure": 2,
        "repair_topic": 3,
        "too_recent_for_main": 4,
        "supplement_only": 5,
        "drop": 6,
    }
    sortable = out[columns].copy()
    sortable["_status_rank"] = sortable["status"].astype(str).map(status_rank).fillna(99).astype(int)
    sortable = sortable.sort_values(
        ["_status_rank", "publication_score", "domain_id"],
        ascending=[True, False, True],
    ).drop(columns=["_status_rank"])
    return sortable.reset_index(drop=True)


def build_publication_target_roster(status_table: pd.DataFrame, top_domains: int = 12) -> Dict[str, Any]:
    """Create a machine-readable roster from main-ready domains."""
    if status_table.empty:
        rows: List[Dict[str, Any]] = []
    else:
        ready = status_table[status_table["status"].astype(str) == "main_ready"].copy()
        if "publication_score" in ready.columns:
            ready = ready.sort_values(["publication_score", "domain_id"], ascending=[False, True])
        rows = []
        for row in ready.head(int(top_domains)).to_dict("records"):
            rows.append(
                {
                    "domain_id": nonempty(row.get("domain_id")),
                    "family": nonempty(row.get("family")),
                    "status": nonempty(row.get("status")),
                    "event_year": ""
                    if pd.isna(row.get("event_year"))
                    else int(safe_numeric(row.get("event_year"), default=0)),
                    "analysis_end_year": int(safe_numeric(row.get("analysis_end_year"), DEFAULT_COMPLETE_END_YEAR)),
                    "landmark_policy": nonempty(row.get("landmark_policy")),
                    "notes": nonempty(row.get("notes")),
                }
            )
    family_counts: Dict[str, int] = {}
    for row in rows:
        family = row.get("family") or "other_science"
        family_counts[family] = family_counts.get(family, 0) + 1
    return {
        "created_at": utc_now(),
        "artifact_kind": "publication_target_domain_roster",
        "figure_logic_policy": FIGURE_LOGIC_POLICY,
        "n_domains": len(rows),
        "family_counts": family_counts,
        "domains": rows,
    }


def build_domain_diagnostics(
    corpus_dir: Path,
    min_papers: int = 2500,
    topic_coverage_target: float = 0.95,
    duplicate_doi_max: float = 0.015,
    reference_closure_target: float = 0.80,
    min_controls_per_landmark: int = 50,
    max_landmarks_per_domain: int = 5,
) -> pd.DataFrame:
    """Score each source domain for v2 publication-corpus suitability."""
    works = read_csv(corpus_dir / "works.csv")
    citations = read_csv(corpus_dir / "citations.csv")
    landmarks = read_csv(corpus_dir / "landmarks.csv")
    domains = read_csv(corpus_dir / "domains.csv")
    clean_landmarks = clean_landmark_registry(landmarks, max_landmarks_per_domain=max_landmarks_per_domain)
    quality = _quality_by_domain(corpus_dir)
    display_lookup = _domain_display_lookup(domains)

    if works.empty or "domain" not in works.columns:
        return pd.DataFrame()

    works = works.copy()
    works["domain"] = works["domain"].map(slugify)
    if "id" in works.columns:
        works["id"] = works["id"].map(normalize_openalex_id)
    rows: List[Dict[str, Any]] = []
    all_domains = sorted(set(works["domain"].astype(str)) | set(display_lookup))
    for domain in all_domains:
        sub = works[works["domain"].astype(str) == domain].copy()
        ids = set(sub.get("id", pd.Series(dtype=str)).astype(str))
        csub = citations[citations.get("source", pd.Series(dtype=str)).astype(str).isin(ids)] if not citations.empty else pd.DataFrame()
        lm = clean_landmarks[clean_landmarks["domain"].astype(str) == domain].copy()
        matched_lm = match_clean_landmarks_to_works(sub, lm)
        raw_landmark_rate = float(
            pd.to_numeric(sub.get("is_landmark", pd.Series(0, index=sub.index)), errors="coerce").fillna(0).astype(int).mean()
        ) if len(sub) else 0.0
        legacy = landmarks[landmarks.get("domain", pd.Series(dtype=str)).map(slugify) == domain].copy() if not landmarks.empty else pd.DataFrame()
        legacy_sources = legacy.get("landmark_source", pd.Series(dtype=str)).fillna("").astype(str).str.lower()
        legacy_fig1_rows = int(legacy_sources.isin(LEGACY_LANDMARK_SOURCES).sum()) if not legacy.empty else 0
        legacy_labels = legacy.get("label", pd.Series("", index=legacy.index)).fillna("").astype(str).str.strip()
        blank_legacy_fig1_rows = int((legacy_sources.isin(LEGACY_LANDMARK_SOURCES) & legacy_labels.eq("")).sum()) if not legacy.empty else 0
        min_controls = _min_controls_for_matched_landmarks(sub, matched_lm)
        topic_coverage = _topic_coverage(sub)
        duplicate_rate = _duplicate_doi_rate(sub)
        reference_closure = _local_reference_closure(citations, ids)
        leakage_rate = _topic_leakage_rate(domain, sub)

        failures: List[str] = []
        if len(sub) < int(min_papers):
            failures.append("n_works")
        if duplicate_rate >= float(duplicate_doi_max):
            failures.append("duplicate_doi")
        if topic_coverage < float(topic_coverage_target):
            failures.append("topic_coverage")
        if reference_closure < float(reference_closure_target):
            failures.append("reference_closure")
        if lm.empty:
            failures.append("missing_clean_landmark")
        elif len(matched_lm) < len(lm):
            failures.append("unmatched_clean_landmark")
        if min_controls is None or min_controls < int(min_controls_per_landmark):
            failures.append("landmark_controls")
        if raw_landmark_rate > 0.25 or blank_legacy_fig1_rows > max(0, len(lm)):
            failures.append("legacy_landmark_inflation")
        if leakage_rate > 0.05:
            failures.append("topic_leakage")

        quality_row = quality.get(domain, {})
        display = display_lookup.get(domain, {})
        score = (
            0.22 * min(len(sub) / max(1, float(min_papers)), 1.0)
            + 0.18 * min(topic_coverage / max(0.01, float(topic_coverage_target)), 1.0)
            + 0.18 * max(0.0, 1.0 - duplicate_rate / max(0.001, float(duplicate_doi_max)))
            + 0.16 * min(reference_closure / max(0.01, float(reference_closure_target)), 1.0)
            + 0.14 * min(len(matched_lm) / 2.0, 1.0)
            + 0.12 * min((min_controls or 0) / max(1, float(min_controls_per_landmark)), 1.0)
        )
        rows.append(
            {
                "domain": domain,
                "display_name": display.get("display_name", domain),
                "field_name": display.get("field_name", ""),
                "seed_source": display.get("seed_source", ""),
                "n_works": int(len(sub)),
                "citation_rows": int(len(csub)),
                "duplicate_doi_rate": duplicate_rate,
                "topic_coverage": topic_coverage,
                "local_reference_closure": reference_closure,
                "raw_landmark_rate": raw_landmark_rate,
                "legacy_fig1_anchor_rows": legacy_fig1_rows,
                "blank_legacy_fig1_anchor_rows": blank_legacy_fig1_rows,
                "clean_landmark_rows": int(len(lm)),
                "matched_clean_landmark_rows": int(len(matched_lm)),
                "unmatched_clean_landmark_rows": int(max(0, len(lm) - len(matched_lm))),
                "min_controls_pm5_years": "" if min_controls is None else int(min_controls),
                "topic_leakage_rate": leakage_rate,
                "v1_quality_gate_pass": bool(quality_row.get("passes", False)),
                "recommended_role": _role_from_failures(failures, score),
                "failure_reasons": ";".join(failures),
                "publication_score": round(float(score), 6),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["recommended_role", "publication_score", "domain"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_fig3_readiness_diagnostics(
    corpus_dir: Path,
    analysis_end_year: int = DEFAULT_COMPLETE_END_YEAR,
    tau: int = 10,
    min_refs: int = 5,
    min_metric_papers: int = 300,
    min_metric_landmarks: int = 1,
    max_landmarks_per_domain: int = 5,
) -> pd.DataFrame:
    """Estimate whether each domain can support Fig. 3 tau-year validation."""
    works = read_csv(corpus_dir / "works.csv")
    citations = read_csv(corpus_dir / "citations.csv")
    landmarks = clean_landmark_registry(
        read_csv(corpus_dir / "landmarks.csv"),
        max_landmarks_per_domain=max_landmarks_per_domain,
    )
    if works.empty or "domain" not in works.columns:
        return pd.DataFrame()

    works = works.copy()
    works["domain"] = works["domain"].map(slugify)
    works["id"] = works.get("id", pd.Series("", index=works.index)).map(normalize_openalex_id)
    works["year_num"] = pd.to_numeric(works.get("year", pd.Series(float("nan"), index=works.index)), errors="coerce")
    if citations.empty:
        source_ref_counts = pd.Series(dtype=int)
    else:
        ctmp = citations.copy()
        if "source" not in ctmp.columns:
            ctmp["source"] = ""
        ctmp["source"] = ctmp["source"].map(normalize_openalex_id)
        source_ref_counts = ctmp.groupby("source").size()

    works["citation_ref_count"] = works["id"].map(source_ref_counts).fillna(0).astype(int)
    works["fig3_ref_count"] = works["citation_ref_count"]
    cutoff_year = int(analysis_end_year) - int(tau)

    rows: List[Dict[str, Any]] = []
    for domain in sorted(works["domain"].dropna().astype(str).unique()):
        sub = works[works["domain"].astype(str) == domain].copy()
        eligible = sub[
            (sub["year_num"] <= cutoff_year)
            & (sub["fig3_ref_count"] >= int(min_refs))
        ].copy()
        domain_landmarks = landmarks[landmarks["domain"].astype(str) == domain].copy()
        matched_landmarks = match_clean_landmarks_to_works(sub, domain_landmarks)
        landmark_ids = (
            set(matched_landmarks.get("id", pd.Series(dtype=str)).map(normalize_openalex_id).astype(str))
            if not matched_landmarks.empty
            else set()
        )
        eligible_landmark_count = int(eligible["id"].astype(str).isin(landmark_ids).sum())
        failures: List[str] = []
        if len(eligible) < int(min_metric_papers):
            failures.append("fig3_metric_papers")
        if eligible_landmark_count < int(min_metric_landmarks):
            failures.append("fig3_metric_landmarks")
        rows.append(
            {
                "domain": domain,
                "fig3_analysis_end_year": int(analysis_end_year),
                "fig3_tau": int(tau),
                "fig3_cutoff_year": cutoff_year,
                "fig3_min_refs": int(min_refs),
                "fig3_total_works": int(len(sub)),
                "fig3_eligible_works": int((sub["year_num"] <= cutoff_year).sum()),
                "fig3_eligible_metric_papers": int(len(eligible)),
                "fig3_clean_landmarks": int(len(domain_landmarks)),
                "fig3_matched_clean_landmarks": int(len(matched_landmarks)),
                "fig3_eligible_metric_landmarks": eligible_landmark_count,
                "fig3_metric_paper_rate": float(len(eligible) / len(sub)) if len(sub) else 0.0,
                "fig3_ready": not failures,
                "fig3_readiness_failures": ";".join(failures),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["fig3_ready", "fig3_eligible_metric_landmarks", "fig3_eligible_metric_papers", "domain"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def write_seed_bundle(
    corpus_dir: Path,
    out_dir: Path,
    diagnostics: pd.DataFrame,
    clean_landmarks: pd.DataFrame,
    thresholds: Mapping[str, Any],
    top_domains: int = 12,
    excluded_domains: Optional[Sequence[str]] = None,
    fig3_readiness: Optional[pd.DataFrame] = None,
    require_fig3_ready: bool = False,
    target_roster_path: Optional[Path] = None,
    excluded_families: Optional[Sequence[str]] = None,
) -> None:
    """Write v2 seed artifacts consumed by the next corpus-building step."""
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(out_dir / "domain_diagnostics.csv", index=False)
    if fig3_readiness is not None:
        fig3_readiness.to_csv(out_dir / "fig3_readiness_diagnostics.csv", index=False)
    excluded_domain_set = {slugify(domain) for domain in (excluded_domains or []) if str(domain).strip()}
    status_table = build_domain_status_table(
        diagnostics=diagnostics,
        fig3_readiness=fig3_readiness,
        clean_landmarks=clean_landmarks,
        excluded_domains=excluded_domains,
        excluded_families=excluded_families,
    )
    if excluded_families and not status_table.empty:
        family_excluded_domains = status_table[
            status_table["reason_for_inclusion_or_exclusion"].astype(str).eq("excluded_family")
        ]["domain_id"].astype(str)
        excluded_domain_set.update(family_excluded_domains)
    status_table.to_csv(out_dir / "domain_status.csv", index=False)
    status_table.to_csv(out_dir / "domain_inclusion_table.csv", index=False)
    roster = build_publication_target_roster(status_table, top_domains=top_domains)
    write_json(out_dir / "publication_target_domains.json", roster)
    if target_roster_path is not None:
        write_json(target_roster_path, roster)
    main_candidates = diagnostics[diagnostics["recommended_role"] == PUBLICATION_MAIN_TARGET].copy()
    if require_fig3_ready and fig3_readiness is not None and not fig3_readiness.empty:
        ready_domains = set(
            fig3_readiness[fig3_readiness["fig3_ready"].astype(bool)]["domain"].astype(str).map(slugify)
        )
        main_candidates = main_candidates[
            main_candidates["domain"].astype(str).map(slugify).isin(ready_domains)
        ].copy()
    if excluded_domain_set:
        domain_series = main_candidates["domain"].astype(str).map(slugify)
        excluded_candidates = main_candidates[domain_series.isin(excluded_domain_set)].copy()
        main_candidates = main_candidates[~domain_series.isin(excluded_domain_set)].copy()
    else:
        excluded_candidates = pd.DataFrame(columns=diagnostics.columns)
    candidates = main_candidates.head(int(top_domains)).copy()
    candidates.to_csv(out_dir / "candidate_domains.csv", index=False)
    excluded_candidates.to_csv(out_dir / "excluded_candidate_domains.csv", index=False)
    rebuild_queue = diagnostics[diagnostics["recommended_role"] != PUBLICATION_MAIN_TARGET].copy()
    if not rebuild_queue.empty:
        rebuild_queue.insert(0, "rebuild_rank", range(1, len(rebuild_queue) + 1))
    rebuild_queue.to_csv(out_dir / "rebuild_queue.csv", index=False)
    candidate_domains = set(candidates["domain"].astype(str))
    seed_landmarks = clean_landmarks[clean_landmarks["domain"].astype(str).isin(candidate_domains)].copy()
    seed_landmarks.to_csv(out_dir / "landmark_registry_v2_seed.csv", index=False)
    write_json(
        out_dir / "v2_publication_seed_manifest.json",
        {
            "created_at": utc_now(),
            "source_corpus_dir": str(corpus_dir),
            "target_corpus_dir": str(DEFAULT_V2_CORPUS_DIR),
            "artifact_kind": "v2_publication_data_layer_seed",
            "figure_logic_policy": FIGURE_LOGIC_POLICY,
            "figure_view_contract": {key: list(value) for key, value in FIGURE_VIEW_CONTRACT.items()},
            "figure_contract": "Keeps existing views/fig1|fig2|fig3|fig5 CSV contract; no figure drawing logic changes.",
            "thresholds": dict(thresholds),
            "excluded_domains": sorted(excluded_domain_set),
            "excluded_families": sorted(nonempty(family) for family in (excluded_families or []) if nonempty(family)),
            "require_fig3_ready": bool(require_fig3_ready),
            "n_domains_scored": int(len(diagnostics)),
            "n_main_ready_domains": int((status_table.get("status", pd.Series(dtype=str)).astype(str) == "main_ready").sum())
            if not status_table.empty
            else 0,
            "n_candidate_domains": int(len(candidates)),
            "n_excluded_candidate_domains": int(len(excluded_candidates)),
            "n_seed_landmarks": int(len(seed_landmarks)),
            "candidate_domains": candidates["domain"].astype(str).tolist(),
            "outputs": [
                "domain_diagnostics.csv",
                "fig3_readiness_diagnostics.csv" if fig3_readiness is not None else "",
                "domain_status.csv",
                "domain_inclusion_table.csv",
                "publication_target_domains.json",
                "candidate_domains.csv",
                "excluded_candidate_domains.csv",
                "rebuild_queue.csv",
                "landmark_registry_v2_seed.csv",
            ],
        },
    )


def _candidate_domains_from_seed(seed_dir: Path, top_domains: int) -> List[str]:
    candidates = read_csv(seed_dir / "candidate_domains.csv")
    if candidates.empty:
        diagnostics = read_csv(seed_dir / "domain_diagnostics.csv")
        if not diagnostics.empty and {"domain", "recommended_role"}.issubset(diagnostics.columns):
            candidates = diagnostics[diagnostics["recommended_role"].astype(str) == PUBLICATION_MAIN_TARGET].copy()
    if candidates.empty or "domain" not in candidates.columns:
        return []
    return candidates["domain"].map(slugify).drop_duplicates().head(int(top_domains)).tolist()


def _filter_topics_for_domains(topics: pd.DataFrame, works: pd.DataFrame) -> pd.DataFrame:
    if topics.empty:
        return topics.copy()
    if "domain" in topics.columns:
        selected_domains = set(works["domain"].astype(str))
        return topics[topics["domain"].astype(str).isin(selected_domains)].copy()
    selected_communities = set(pd.to_numeric(works.get("display_community", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
    out = topics.copy()
    if "community" in out.columns:
        out["community"] = pd.to_numeric(out["community"], errors="coerce").fillna(-1).astype(int)
        return out[out["community"].isin(selected_communities)].copy()
    return out


def _filter_topic_edges_for_topics(topic_edges: pd.DataFrame, topics: pd.DataFrame) -> pd.DataFrame:
    if topic_edges.empty or topics.empty or "community" not in topics.columns:
        return topic_edges.copy()
    out = topic_edges.copy()
    source_col = "source_community" if "source_community" in out.columns else "source"
    target_col = "target_community" if "target_community" in out.columns else "target"
    if source_col not in out.columns or target_col not in out.columns:
        return out
    communities = set(pd.to_numeric(topics["community"], errors="coerce").dropna().astype(int))
    out[source_col] = pd.to_numeric(out[source_col], errors="coerce").fillna(-1).astype(int)
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce").fillna(-1).astype(int)
    return out[out[source_col].isin(communities) & out[target_col].isin(communities)].copy()


def _standard_landmark_columns(landmarks: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "domain",
        "landmark_source",
        "source_id",
        "label",
        "id",
        "doi",
        "title",
        "year",
        "match_confidence",
        "include_main",
        "accepted_landmark_source",
        "needs_manual_confirmation",
        "evidence_key",
    ]
    out = landmarks.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = ""
    return out[cols].copy()


def materialize_candidate_corpus(
    source_corpus_dir: Path,
    seed_dir: Path,
    target_corpus_dir: Path,
    top_domains: int = 12,
    min_papers_per_domain: int = 2500,
    anchor_policy: str = "strict",
) -> Dict[str, Any]:
    """
    Materialize a Fig1-Fig5-compatible v2 seed corpus from vetted candidate domains.

    Root citations keep all references made by selected works, including targets
    outside the selected work set, so reference-closure diagnostics remain honest.
    Figure views are still written by the existing corpus view generator.
    """
    candidate_domains = _candidate_domains_from_seed(seed_dir, top_domains=top_domains)
    if not candidate_domains:
        raise ValueError(f"No candidate domains found under {seed_dir}")

    works = read_csv(source_corpus_dir / "works.csv")
    citations = read_csv(source_corpus_dir / "citations.csv")
    topics = read_csv(source_corpus_dir / "topics.csv")
    topic_edges = read_csv(source_corpus_dir / "topic_edges.csv")
    domains = read_csv(source_corpus_dir / "domains.csv")
    seed_landmarks = read_csv(seed_dir / "landmark_registry_v2_seed.csv")
    if seed_landmarks.empty:
        seed_landmarks = clean_landmark_registry(read_csv(source_corpus_dir / "landmarks.csv"))

    if works.empty or "domain" not in works.columns:
        raise ValueError(f"{source_corpus_dir / 'works.csv'} has no usable domain rows")

    works = works.copy()
    works["domain"] = works["domain"].map(slugify)
    works["id"] = works.get("id", pd.Series("", index=works.index)).map(normalize_openalex_id)
    selected_domain_set = set(candidate_domains)
    works = works[works["domain"].astype(str).isin(selected_domain_set)].copy()
    if works.empty:
        raise ValueError("Candidate domains have no matching works in the source corpus")

    landmarks = _standard_landmark_columns(seed_landmarks)
    landmarks["domain"] = landmarks["domain"].map(slugify)
    landmarks = landmarks[landmarks["domain"].astype(str).isin(selected_domain_set)].copy()
    if landmarks.empty:
        raise ValueError("Candidate domains have no clean seed landmarks")

    matched_landmarks = match_clean_landmarks_to_works(works, landmarks)
    landmark_counts = landmarks.groupby("domain").size()
    matched_counts = matched_landmarks.groupby("domain").size() if not matched_landmarks.empty else pd.Series(dtype=int)
    fully_matched_domains = [
        domain
        for domain in candidate_domains
        if int(matched_counts.get(domain, 0)) == int(landmark_counts.get(domain, 0)) and int(landmark_counts.get(domain, 0)) > 0
    ]
    excluded_unmatched_domains = [
        domain for domain in candidate_domains if domain not in set(fully_matched_domains)
    ]
    if not fully_matched_domains:
        raise ValueError("Candidate domains have no fully matched clean landmarks")

    candidate_domains = fully_matched_domains
    selected_domain_set = set(candidate_domains)
    works = works[works["domain"].astype(str).isin(selected_domain_set)].copy()
    landmarks = _standard_landmark_columns(
        matched_landmarks[matched_landmarks["domain"].astype(str).isin(selected_domain_set)].copy()
    )

    works = apply_strict_anchor_policy(works, landmarks, complete_end_year=DEFAULT_COMPLETE_END_YEAR)
    selected_ids = set(works["id"].astype(str))
    if citations.empty:
        citations = pd.DataFrame(columns=["source", "target", "relation", "source_dataset"])
    else:
        citations = citations.copy()
        for col in ["source", "target"]:
            if col not in citations.columns:
                citations[col] = ""
            citations[col] = citations[col].map(normalize_openalex_id)
        citations = citations[citations["source"].astype(str).isin(selected_ids)].drop_duplicates().copy()

    topics = _filter_topics_for_domains(topics, works)
    topic_edges = _filter_topic_edges_for_topics(topic_edges, topics)
    if not domains.empty:
        domains = domains.copy()
        domain_col = "slug" if "slug" in domains.columns else "domain"
        if domain_col in domains.columns:
            domains[domain_col] = domains[domain_col].map(slugify)
            domains = domains[domains[domain_col].astype(str).isin(selected_domain_set)].copy()
    if domains.empty:
        domains = pd.DataFrame({"slug": candidate_domains, "display_name": candidate_domains})

    target_corpus_dir.mkdir(parents=True, exist_ok=True)
    views_dir = target_corpus_dir / "views"
    if views_dir.exists():
        shutil.rmtree(views_dir)
    works.to_csv(target_corpus_dir / "works.csv", index=False)
    citations.to_csv(target_corpus_dir / "citations.csv", index=False)
    topics.to_csv(target_corpus_dir / "topics.csv", index=False)
    topic_edges.to_csv(target_corpus_dir / "topic_edges.csv", index=False)
    domains.to_csv(target_corpus_dir / "domains.csv", index=False)
    landmarks.to_csv(target_corpus_dir / "landmarks.csv", index=False)

    quality_report = audit_corpus(target_corpus_dir, min_papers_per_domain=min_papers_per_domain)
    make_views(target_corpus_dir, anchor_policy=anchor_policy)

    manifest = {
        "created_at": utc_now(),
        "artifact_kind": "v2_publication_candidate_corpus",
        "source_corpus_dir": str(source_corpus_dir),
        "seed_dir": str(seed_dir),
        "target_corpus_dir": str(target_corpus_dir),
        "figure_contract": "Root tables and views remain compatible with existing Fig1-Fig5 readers.",
        "anchor_policy": anchor_policy,
        "domains": candidate_domains,
        "n_domains": int(len(candidate_domains)),
        "n_works": int(len(works)),
        "n_citations_source_refs": int(len(citations)),
        "n_landmarks": int(len(landmarks)),
        "excluded_unmatched_landmark_domains": excluded_unmatched_domains,
        "quality_overall_pass": bool(quality_report.get("overall_pass", False)),
        "min_papers_per_domain": int(min_papers_per_domain),
    }
    write_json(target_corpus_dir / "manifest.json", manifest)
    write_json(target_corpus_dir / "publication_corpus_v2_manifest.json", manifest)
    return manifest


def diagnose_command(args: argparse.Namespace) -> None:
    explicit_excluded_domains = [slugify(domain) for domain in (args.exclude_domains or []) if str(domain).strip()]
    excluded_prefixes = [slugify(prefix) for prefix in (args.exclude_domain_prefix or []) if str(prefix).strip()]
    excluded_families = [nonempty(family) for family in (args.exclude_family or []) if nonempty(family)]
    thresholds = {
        "min_papers": int(args.min_papers),
        "topic_coverage_target": float(args.topic_coverage_target),
        "duplicate_doi_max": float(args.duplicate_doi_max),
        "reference_closure_target": float(args.reference_closure_target),
        "min_controls_per_landmark": int(args.min_controls_per_landmark),
        "max_landmarks_per_domain": int(args.max_landmarks_per_domain),
        "top_domains": int(args.top_domains),
        "excluded_domains": explicit_excluded_domains,
        "excluded_domain_prefixes": excluded_prefixes,
        "excluded_families": excluded_families,
        "require_fig3_ready": bool(args.require_fig3_ready),
        "fig3_analysis_end_year": int(args.fig3_analysis_end_year),
        "fig3_tau": int(args.fig3_tau),
        "fig3_min_refs": int(args.fig3_min_refs),
        "fig3_min_metric_papers": int(args.fig3_min_metric_papers),
        "fig3_min_metric_landmarks": int(args.fig3_min_metric_landmarks),
    }
    diagnostics = build_domain_diagnostics(
        corpus_dir=args.corpus_dir,
        min_papers=args.min_papers,
        topic_coverage_target=args.topic_coverage_target,
        duplicate_doi_max=args.duplicate_doi_max,
        reference_closure_target=args.reference_closure_target,
        min_controls_per_landmark=args.min_controls_per_landmark,
        max_landmarks_per_domain=args.max_landmarks_per_domain,
    )
    clean_landmarks = clean_landmark_registry(
        read_csv(args.corpus_dir / "landmarks.csv"),
        max_landmarks_per_domain=args.max_landmarks_per_domain,
    )
    fig3_readiness = build_fig3_readiness_diagnostics(
        corpus_dir=args.corpus_dir,
        analysis_end_year=args.fig3_analysis_end_year,
        tau=args.fig3_tau,
        min_refs=args.fig3_min_refs,
        min_metric_papers=args.fig3_min_metric_papers,
        min_metric_landmarks=args.fig3_min_metric_landmarks,
        max_landmarks_per_domain=args.max_landmarks_per_domain,
    )
    excluded_domains = list(explicit_excluded_domains)
    if excluded_prefixes and not diagnostics.empty and "domain" in diagnostics.columns:
        for domain in diagnostics["domain"].astype(str).map(slugify):
            if any(domain.startswith(prefix) for prefix in excluded_prefixes):
                excluded_domains.append(domain)
    excluded_domains = sorted(set(excluded_domains))
    write_seed_bundle(
        args.corpus_dir,
        args.out_dir,
        diagnostics,
        clean_landmarks,
        thresholds,
        top_domains=args.top_domains,
        excluded_domains=excluded_domains,
        fig3_readiness=fig3_readiness,
        require_fig3_ready=args.require_fig3_ready,
        target_roster_path=args.target_roster_path,
        excluded_families=excluded_families,
    )
    if not args.quiet:
        candidates = read_csv(args.out_dir / "candidate_domains.csv")
        print(
            f"[publication-v2] scored {len(diagnostics)} domains; "
            f"{len(candidates)} main candidates; wrote {args.out_dir}",
            flush=True,
        )


def materialize_command(args: argparse.Namespace) -> None:
    manifest = materialize_candidate_corpus(
        source_corpus_dir=args.source_corpus_dir,
        seed_dir=args.seed_dir,
        target_corpus_dir=args.target_corpus_dir,
        top_domains=args.top_domains,
        min_papers_per_domain=args.min_papers_per_domain,
        anchor_policy=args.anchor_policy,
    )
    if not args.quiet:
        print(
            f"[publication-v2] materialized {manifest['n_domains']} domains, "
            f"{manifest['n_works']} works under {args.target_corpus_dir}",
            flush=True,
        )


def repair_global_landmarks_command(args: argparse.Namespace) -> None:
    manifest = augment_corpus_with_global_landmark_repairs(
        source_corpus_dir=args.source_corpus_dir,
        target_corpus_dir=args.target_corpus_dir,
        domains=args.domains,
        max_landmarks_per_domain=args.max_landmarks_per_domain,
    )
    if not args.quiet:
        print(
            f"[publication-v2] repaired {manifest['n_repaired_landmarks']} globally misplaced landmarks "
            f"under {args.target_corpus_dir}",
            flush=True,
        )


def fetch_external_landmarks_command(args: argparse.Namespace) -> None:
    if args.fetched_records_jsonl:
        records = _load_external_records_jsonl(args.fetched_records_jsonl)
    else:
        records = _external_records_from_openalex(
            source_corpus_dir=args.source_corpus_dir,
            domains=args.domains,
            timeout_seconds=args.timeout_seconds,
            max_landmarks_per_domain=args.max_landmarks_per_domain,
        )
    manifest = augment_corpus_with_external_landmark_records(
        source_corpus_dir=args.source_corpus_dir,
        target_corpus_dir=args.target_corpus_dir,
        records=records,
        domains=args.domains,
    )
    if not args.quiet:
        print(
            f"[publication-v2] added {manifest['n_external_landmarks_added']} external landmark works "
            f"under {args.target_corpus_dir}",
            flush=True,
        )


def topup_openalex_works_command(args: argparse.Namespace) -> None:
    if args.topup_records_jsonl:
        records = _load_external_records_jsonl(args.topup_records_jsonl)
    else:
        records = _topup_records_from_openalex(
            source_corpus_dir=args.source_corpus_dir,
            domains=args.domains or [],
            target_papers=args.target_papers,
            max_extra_per_domain=args.max_extra_per_domain,
            timeout_seconds=args.timeout_seconds,
            min_local_refs=args.min_local_refs,
            local_references_only=args.local_references_only,
        )
    manifest = augment_corpus_with_topup_work_records(
        source_corpus_dir=args.source_corpus_dir,
        target_corpus_dir=args.target_corpus_dir,
        records=records,
        domains=args.domains,
        min_local_refs=args.min_local_refs,
        local_references_only=args.local_references_only,
    )
    if not args.quiet:
        print(
            f"[publication-v2] added {manifest['n_topup_works_added']} top-up works "
            f"under {args.target_corpus_dir}",
            flush=True,
        )


def repair_metadata_command(args: argparse.Namespace) -> None:
    manifest = repair_corpus_metadata(
        source_corpus_dir=args.source_corpus_dir,
        target_corpus_dir=args.target_corpus_dir,
        domains=args.domains,
        max_landmarks_per_domain=args.max_landmarks_per_domain,
    )
    if not args.quiet:
        print(
            f"[publication-v2] filled {manifest['n_topic_labels_filled']} topic labels "
            f"and wrote {manifest['n_landmarks_output']} clean landmarks under {args.target_corpus_dir}",
            flush=True,
        )


def add_landmark_reference_support_command(args: argparse.Namespace) -> None:
    manifest = augment_corpus_with_landmark_reference_support(
        source_corpus_dir=args.source_corpus_dir,
        target_corpus_dir=args.target_corpus_dir,
        domains=args.domains,
        min_internal_refs=args.min_internal_refs,
        max_support_refs_per_landmark=args.max_support_refs_per_landmark,
        timeout_seconds=args.timeout_seconds,
        max_landmarks_per_domain=args.max_landmarks_per_domain,
    )
    if not args.quiet:
        print(
            f"[publication-v2] added {manifest['n_support_works_added']} support works "
            f"and repaired {manifest['n_landmarks_repaired']} landmarks under {args.target_corpus_dir}",
            flush=True,
        )


def dedupe_dois_command(args: argparse.Namespace) -> None:
    works = read_csv(args.source_corpus_dir / "works.csv")
    citations = read_csv(args.source_corpus_dir / "citations.csv")
    dedup_works, dedup_citations, report = deduplicate_domain_dois(works, citations)
    args.target_corpus_dir.mkdir(parents=True, exist_ok=True)
    views_dir = args.target_corpus_dir / "views"
    if views_dir.exists():
        shutil.rmtree(views_dir)
    dedup_works.to_csv(args.target_corpus_dir / "works.csv", index=False)
    dedup_citations.to_csv(args.target_corpus_dir / "citations.csv", index=False)
    for name in [
        "topics.csv",
        "topic_edges.csv",
        "domains.csv",
        "landmarks.csv",
        "landmark_repair_plan.csv",
        "global_landmark_repair_manifest.json",
        "external_landmark_fetch_manifest.json",
        "openalex_topup_manifest.json",
    ]:
        src = args.source_corpus_dir / name
        if src.exists():
            shutil.copy2(src, args.target_corpus_dir / name)
    manifest = {
        "created_at": utc_now(),
        "artifact_kind": "v2_domain_doi_dedupe_source",
        "source_corpus_dir": str(args.source_corpus_dir),
        "target_corpus_dir": str(args.target_corpus_dir),
        **report,
    }
    write_json(args.target_corpus_dir / "doi_dedupe_manifest.json", manifest)
    if not args.quiet:
        print(
            f"[publication-v2] dropped {report['dropped_duplicate_works']} duplicate DOI works "
            f"under {args.target_corpus_dir}",
            flush=True,
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose and seed the v2 publication corpus data layer.")
    sub = parser.add_subparsers(dest="command", required=True)
    diagnose = sub.add_parser("diagnose", help="Write domain diagnostics and v2 seed artifacts.")
    diagnose.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    diagnose.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    diagnose.add_argument("--min-papers", type=int, default=2500)
    diagnose.add_argument("--topic-coverage-target", type=float, default=0.95)
    diagnose.add_argument("--duplicate-doi-max", type=float, default=0.015)
    diagnose.add_argument("--reference-closure-target", type=float, default=0.80)
    diagnose.add_argument("--min-controls-per-landmark", type=int, default=50)
    diagnose.add_argument("--max-landmarks-per-domain", type=int, default=5)
    diagnose.add_argument("--top-domains", type=int, default=12)
    diagnose.add_argument("--exclude-domains", nargs="*", default=None)
    diagnose.add_argument("--exclude-domain-prefix", action="append", default=None)
    diagnose.add_argument("--exclude-family", action="append", default=None)
    diagnose.add_argument("--require-fig3-ready", action="store_true")
    diagnose.add_argument("--fig3-analysis-end-year", type=int, default=DEFAULT_COMPLETE_END_YEAR)
    diagnose.add_argument("--fig3-tau", type=int, default=10)
    diagnose.add_argument("--fig3-min-refs", type=int, default=5)
    diagnose.add_argument("--fig3-min-metric-papers", type=int, default=300)
    diagnose.add_argument("--fig3-min-metric-landmarks", type=int, default=1)
    diagnose.add_argument("--target-roster-path", type=Path, default=None)
    diagnose.add_argument("--quiet", action="store_true")
    diagnose.set_defaults(func=diagnose_command)

    materialize = sub.add_parser("materialize", help="Write a Fig1-Fig5-compatible v2 corpus from seed artifacts.")
    materialize.add_argument("--source-corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    materialize.add_argument("--seed-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    materialize.add_argument("--target-corpus-dir", type=Path, default=DEFAULT_V2_CORPUS_DIR)
    materialize.add_argument("--top-domains", type=int, default=12)
    materialize.add_argument("--min-papers-per-domain", type=int, default=2500)
    materialize.add_argument("--anchor-policy", choices=["strict"], default="strict")
    materialize.add_argument("--quiet", action="store_true")
    materialize.set_defaults(func=materialize_command)

    repair = sub.add_parser("repair-global-landmarks", help="Copy exact global-source landmark matches into target domains.")
    repair.add_argument("--source-corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    repair.add_argument("--target-corpus-dir", type=Path, default=DEFAULT_REPAIRED_SOURCE_DIR)
    repair.add_argument("--domains", nargs="*", default=None)
    repair.add_argument("--max-landmarks-per-domain", type=int, default=5)
    repair.add_argument("--quiet", action="store_true")
    repair.set_defaults(func=repair_global_landmarks_command)

    external = sub.add_parser("fetch-external-landmarks", help="Fetch exact external OpenAlex landmark works into target domains.")
    external.add_argument("--source-corpus-dir", type=Path, default=DEFAULT_REPAIRED_SOURCE_DIR)
    external.add_argument("--target-corpus-dir", type=Path, default=DEFAULT_EXTERNAL_SOURCE_DIR)
    external.add_argument("--domains", nargs="*", default=None)
    external.add_argument("--fetched-records-jsonl", type=Path, default=None)
    external.add_argument("--timeout-seconds", type=int, default=60)
    external.add_argument("--max-landmarks-per-domain", type=int, default=5)
    external.add_argument("--quiet", action="store_true")
    external.set_defaults(func=fetch_external_landmarks_command)

    topup = sub.add_parser("topup-openalex-works", help="Top up near-threshold domains with a small number of OpenAlex works.")
    topup.add_argument("--source-corpus-dir", type=Path, default=DEFAULT_EXTERNAL_SOURCE_DIR)
    topup.add_argument("--target-corpus-dir", type=Path, default=DEFAULT_TOPUP_SOURCE_DIR)
    topup.add_argument("--domains", nargs="*", default=None)
    topup.add_argument("--topup-records-jsonl", type=Path, default=None)
    topup.add_argument("--target-papers", type=int, default=2500)
    topup.add_argument("--max-extra-per-domain", type=int, default=25)
    topup.add_argument("--min-local-refs", type=int, default=0)
    topup.add_argument("--local-references-only", action="store_true")
    topup.add_argument("--timeout-seconds", type=int, default=60)
    topup.add_argument("--quiet", action="store_true")
    topup.set_defaults(func=topup_openalex_works_command)

    metadata = sub.add_parser("repair-metadata", help="Fill missing topic labels and write a clean landmark registry.")
    metadata.add_argument("--source-corpus-dir", type=Path, default=DEFAULT_DEDUP_SOURCE_DIR)
    metadata.add_argument("--target-corpus-dir", type=Path, required=True)
    metadata.add_argument("--domains", nargs="*", default=None)
    metadata.add_argument("--max-landmarks-per-domain", type=int, default=5)
    metadata.add_argument("--quiet", action="store_true")
    metadata.set_defaults(func=repair_metadata_command)

    support = sub.add_parser(
        "add-landmark-reference-support",
        help="Add reference target works so clean landmarks have local prior-reference support.",
    )
    support.add_argument("--source-corpus-dir", type=Path, default=DEFAULT_DEDUP_SOURCE_DIR)
    support.add_argument("--target-corpus-dir", type=Path, required=True)
    support.add_argument("--domains", nargs="*", default=None)
    support.add_argument("--min-internal-refs", type=int, default=5)
    support.add_argument("--max-support-refs-per-landmark", type=int, default=8)
    support.add_argument("--timeout-seconds", type=int, default=60)
    support.add_argument("--max-landmarks-per-domain", type=int, default=5)
    support.add_argument("--quiet", action="store_true")
    support.set_defaults(func=add_landmark_reference_support_command)

    dedupe = sub.add_parser("dedupe-dois", help="Deduplicate DOI rows within each domain while preserving root-table compatibility.")
    dedupe.add_argument("--source-corpus-dir", type=Path, default=DEFAULT_TOPUP_SOURCE_DIR)
    dedupe.add_argument("--target-corpus-dir", type=Path, default=DEFAULT_DEDUP_SOURCE_DIR)
    dedupe.add_argument("--quiet", action="store_true")
    dedupe.set_defaults(func=dedupe_dois_command)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
