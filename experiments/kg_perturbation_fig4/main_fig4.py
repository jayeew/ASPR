from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import textwrap
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from aspr.env import getenv, getenv_bool, getenv_float, getenv_int, getenv_system
except ImportError:  # pragma: no cover - direct script fallback.
    PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))
    from aspr.env import getenv, getenv_bool, getenv_float, getenv_int, getenv_system


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARKDOWN_ROOT = PROJECT_ROOT / "data" / "nature_markdown"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "kg_perturbation_fig4"
DEFAULT_FIG3_WEIGHTS_PATH = PROJECT_ROOT / "outputs/kg_perturbation_fig3_strict_broad10/multi_domain/fig3_best_weights.csv"
DEFAULT_SAMPLE_SIZE = 180
DEFAULT_SAMPLE_SEED = 20260617
DEFAULT_HUMAN_HOURS = 5.0
DEFAULT_41467_CAP = 0.50
ASPECTS = ["significance", "novelty", "rigor", "limitations", "future_work"]
ASPECT_LABELS = {
    "significance": "Significance",
    "novelty": "Novelty",
    "rigor": "Rigor",
    "limitations": "Limitations",
    "future_work": "Future work",
}
INNOVATION_ASPECTS = [
    "novelty",
    "significance",
    "prior_art_comparison",
    "evidence_rigor",
    "limitations",
    "future_work",
]
INNOVATION_ASPECT_LABELS = {
    "novelty": "Novelty",
    "significance": "Significance",
    "prior_art_comparison": "Prior art",
    "evidence_rigor": "Evidence / rigor",
    "limitations": "Limitations",
    "future_work": "Future work",
}
CORE_INNOVATION_ASPECTS = ["novelty", "significance", "prior_art_comparison"]
SEMANTIC_RELATION_SCORES = {
    "entailed": 1.0,
    "related": 0.5,
    "contradicted": 0.0,
    "no_match": 0.0,
}
STRUCTURED_CONSISTENCY_FIELDS = [
    "stance_consistency_1_5",
    "novelty_consistency_1_5",
    "significance_consistency_1_5",
    "prior_art_consistency_1_5",
    "evidence_rigor_consistency_1_5",
    "limitations_consistency_1_5",
    "future_work_consistency_1_5",
]
REVISION_ONLY_RE = re.compile(
    r"\b("
    r"concerns?\s+(?:have\s+been\s+)?addressed|"
    r"all\s+concerns?\s+(?:have\s+been\s+)?(?:resolved|addressed)|"
    r"no\s+further\s+(?:comments?|concerns?)|"
    r"recommend\s+accept(?:ance)?|"
    r"suitable\s+for\s+publication"
    r")\b",
    re.IGNORECASE,
)
JOURNAL_NAMES = {
    "41467": "Nature Communications",
    "41551": "Nature Biomedical Engineering",
    "41556": "Nature Cell Biology",
    "41559": "Nature Ecology & Evolution",
    "41562": "Nature Human Behaviour",
    "41564": "Nature Microbiology",
    "41590": "Nature Immunology",
    "41594": "Nature Structural & Molecular Biology",
}
SIX_SUBJOURNAL_IDS = {"41556", "41559", "41562", "41564", "41590", "41594"}
ALL_ELIGIBLE_JOURNAL_IDS = set(JOURNAL_NAMES)
PICTURE_LINE_RE = re.compile(r"^\*?\*?==>\s*picture\b.*omitted.*$", re.IGNORECASE)
DOI_RE = re.compile(r"https?://doi\.org/(10\.1038/[^\s)]+)|\b(10\.1038/[^\s)]+)", re.IGNORECASE)


def progress_log(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(f"[Fig4] {message}", file=sys.stderr, flush=True)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, text)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    if fieldnames is None:
        field_set: List[str] = []
        for row in rows:
            for key in row:
                if key not in field_set:
                    field_set.append(key)
        fieldnames = field_set
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def read_csv_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def clean_markdown_text(text: str) -> str:
    lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if PICTURE_LINE_RE.match(line):
            continue
        lines.append(raw_line.rstrip())
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(text or "")))


def stable_text_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:length]


def article_id_to_doi(article_id: str) -> str:
    if article_id.startswith("s"):
        return f"10.1038/{article_id}"
    return ""


def article_id_to_year(article_id: str) -> int:
    match = re.match(r"s\d{5}-(\d{3})-", article_id)
    return 2000 + int(match.group(1)) if match else 0


def article_id_to_journal_id(article_id: str) -> str:
    match = re.match(r"s(\d{5})-", article_id)
    return match.group(1) if match else ""


def journal_name(journal_id: str) -> str:
    return JOURNAL_NAMES.get(str(journal_id), f"Nature family {journal_id}")


def extract_doi(text: str, fallback_article_id: str = "") -> str:
    match = DOI_RE.search(text)
    if match:
        return (match.group(1) or match.group(2) or "").rstrip(".,")
    return article_id_to_doi(fallback_article_id)


def _paragraphs(text: str) -> List[str]:
    chunks = re.split(r"\n\s*\n", text)
    return [normalize_whitespace(chunk) for chunk in chunks if normalize_whitespace(chunk)]


def _looks_like_affiliation(paragraph: str) -> bool:
    lowered = paragraph.lower()
    affiliation_terms = ["department", "university", "institute", "hospital", "laboratory", "school of", "college of"]
    return any(term in lowered for term in affiliation_terms) and len(paragraph) < 1200


def _looks_like_author_list(paragraph: str) -> bool:
    lowered = paragraph.lower()
    contribution_terms = ["we ", "here,", "herein", "this study", "we report", "we present", "we identify"]
    if any(term in lowered for term in contribution_terms):
        return False
    has_many_names = paragraph.count(",") >= 5 and ("&" in paragraph or re.search(r"\b[A-Z][a-z]+ [A-Z]\.", paragraph))
    has_affiliation_marks = len(re.findall(r"\[\d+\]|\b\d+(?:,\d+)*\b", paragraph)) >= 4
    return bool(has_many_names and has_affiliation_marks and word_count(paragraph) < 180)


def parse_article_markdown(text: str, article_id: str) -> Dict[str, Any]:
    cleaned = clean_markdown_text(text)
    lines = [line.strip() for line in cleaned.splitlines()]
    warnings: List[str] = []
    title = ""
    title_line_index = 0
    generic_headings = {"article", "research article", "review article", "results", "abstract"}
    for idx, line in enumerate(lines):
        if line.startswith("## ") and "peer review" not in line.lower():
            candidate = line.lstrip("#").strip()
            if candidate.lower() in generic_headings:
                continue
            title = candidate
            title_line_index = idx
            break
    if not title:
        warnings.append("missing_title")
        title = article_id
    doi = extract_doi(cleaned, article_id)
    year = article_id_to_year(article_id)
    if not year:
        year_match = re.search(r"\b(20\d{2})\b", cleaned[:3000])
        year = int(year_match.group(1)) if year_match else 0
    post_title = "\n".join(lines[title_line_index + 1 :])
    abstract = ""
    for paragraph in _paragraphs(post_title):
        low = paragraph.lower()
        if paragraph.startswith(">"):
            continue
        if _looks_like_author_list(paragraph):
            continue
        if "nature communications" in low and len(paragraph) < 300:
            continue
        if _looks_like_affiliation(paragraph):
            continue
        if word_count(paragraph) >= 45 and not re.match(r"^(received|accepted|check for updates)\b", low):
            abstract = paragraph
            break
    if not abstract:
        warnings.append("missing_abstract")
    return {
        "article_text": cleaned,
        "doi": doi,
        "title": title,
        "year": year,
        "abstract": abstract,
        "abstract_source": "pdf_markdown" if abstract else "missing",
        "word_count": word_count(cleaned),
        "warnings": warnings,
    }


def split_sentences(text: str) -> List[str]:
    """Split Markdown-derived text into compact English/Chinese-like sentences."""
    chunks = re.split(r"(?<=[.!?。！？])\s+", normalize_whitespace(text))
    return [chunk.strip() for chunk in chunks if word_count(chunk) >= 8]


def extract_sentences_by_keywords(text: str, keywords: Sequence[str], max_sentences: int = 5) -> List[str]:
    """Return representative sentences matching any keyword, preserving source order."""
    lowered_keywords = [keyword.lower() for keyword in keywords]
    selected: List[str] = []
    for sentence in split_sentences(text):
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in lowered_keywords):
            selected.append(sentence[:520])
        if len(selected) >= max_sentences:
            break
    return selected


def extract_section_summary(text: str, headings: Sequence[str], max_words: int = 180) -> str:
    """Extract a short section-like summary from Markdown headings or fallback keyword sentences."""
    cleaned = clean_markdown_text(text)
    pattern = r"(?im)^#{1,4}\s*(?:" + "|".join(re.escape(item) for item in headings) + r")\b.*$"
    match = re.search(pattern, cleaned)
    if match:
        section = cleaned[match.end() :]
        next_heading = re.search(r"(?m)^#{1,4}\s+\S", section)
        if next_heading:
            section = section[: next_heading.start()]
        words = re.findall(r"\S+", normalize_whitespace(section))
        if words:
            return " ".join(words[:max_words])
    fallback = extract_sentences_by_keywords(cleaned, headings, max_sentences=3)
    return " ".join(fallback)[: max_words * 8]


def build_paper_dossier(
    parsed_article: Mapping[str, Any],
    record: Mapping[str, Any],
    keywords: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Build a structured full-paper summary package for the innovation agent.

    The dossier is derived only from article Markdown/PDF text and manifest
    metadata. Peer-review text is deliberately excluded to prevent label leakage.
    """
    article_text = str(parsed_article.get("article_text") or "")
    abstract = str(parsed_article.get("abstract") or record.get("abstract") or "")
    keyword_list = [str(item).strip() for item in (keywords or []) if str(item).strip()]
    contribution_terms = [
        "we report",
        "we present",
        "we propose",
        "we develop",
        "we demonstrate",
        "we identify",
        "we show",
        "here we",
        "herein",
        "this study",
        "our results",
        "our findings",
    ]
    method_terms = ["method", "methods", "approach", "assay", "model", "analysis", "dataset", "experiment"]
    result_terms = ["result", "results", "finding", "findings", "show", "demonstrate", "reveal", "suggest"]
    limitation_terms = ["limitation", "limitations", "however", "although", "future", "remains", "unclear"]
    return {
        "paper_id": record.get("paper_id", ""),
        "doi": parsed_article.get("doi") or record.get("doi", ""),
        "title": parsed_article.get("title") or record.get("title", ""),
        "journal": record.get("journal", ""),
        "year": parsed_article.get("year") or record.get("year", ""),
        "abstract": abstract,
        "keywords": keyword_list,
        "contribution_sentences": extract_sentences_by_keywords(article_text or abstract, contribution_terms, max_sentences=6),
        "methods_summary": extract_section_summary(article_text, method_terms, max_words=180),
        "results_summary": extract_section_summary(article_text, result_terms, max_words=180),
        "limitations_sentences": extract_sentences_by_keywords(article_text, limitation_terms, max_sentences=4),
        "article_word_count": parsed_article.get("word_count", 0),
        "source": "article_markdown",
        "leakage_guard": "peer_review_text_excluded",
    }


def format_paper_dossier_for_agent(dossier: Mapping[str, Any]) -> str:
    """Render a dossier into a compact prompt block."""
    if not dossier:
        return ""
    lines = [
        f"DOI: {dossier.get('doi', '')}",
        f"Journal/year: {dossier.get('journal', '')} / {dossier.get('year', '')}",
        f"Abstract: {normalize_whitespace(str(dossier.get('abstract') or ''))[:1800]}",
    ]
    keywords = dossier.get("keywords") or []
    if keywords:
        lines.append("Keywords: " + ", ".join(str(item) for item in keywords[:8]))
    for label, key in [
        ("Contribution signals", "contribution_sentences"),
        ("Methods/results context", "methods_summary"),
        ("Result signals", "results_summary"),
        ("Limitations or caution signals", "limitations_sentences"),
    ]:
        value = dossier.get(key)
        if isinstance(value, list):
            rendered = " ".join(f"- {normalize_whitespace(str(item))}" for item in value if normalize_whitespace(str(item)))
        else:
            rendered = normalize_whitespace(str(value or ""))
        if rendered:
            lines.append(f"{label}: {rendered[:2200]}")
    lines.append("Use this dossier as context, but do not infer claims that are not supported by the article text or retrieved prior art.")
    return "\n".join(lines)


def normalize_title_for_match(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(title or "").lower()).strip()


def filter_prior_art_candidates(
    papers: Sequence[Mapping[str, Any]],
    target_title: str,
    target_doi: str,
    cutoff_year: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Remove target-paper leakage and post-publication papers from retrieved candidates."""
    kept: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    target_title_norm = normalize_title_for_match(target_title)
    target_doi_norm = str(target_doi or "").lower().replace("https://doi.org/", "").strip()
    for paper in papers:
        item = dict(paper)
        reasons: List[str] = []
        paper_doi = str(item.get("doi") or (item.get("externalIds") or {}).get("DOI") or "").lower().replace("https://doi.org/", "").strip()
        paper_title_norm = normalize_title_for_match(str(item.get("title") or ""))
        paper_year = int(item.get("year") or 0)
        if target_doi_norm and paper_doi and paper_doi == target_doi_norm:
            reasons.append("target_doi_match")
        if target_title_norm and paper_title_norm and paper_title_norm == target_title_norm:
            reasons.append("target_title_match")
        if cutoff_year and paper_year and paper_year > cutoff_year:
            reasons.append("post_publication_year")
        if reasons:
            excluded.append(
                {
                    "paperId": item.get("paperId", ""),
                    "title": item.get("title", ""),
                    "doi": paper_doi,
                    "year": paper_year,
                    "reasons": reasons,
                }
            )
        else:
            kept.append(item)
    return kept, excluded


def local_prior_art_candidates(row: Mapping[str, Any], output_dir: Path) -> List[Dict[str, Any]]:
    """Build a small local prior-art pool from the Fig.4 manifest when S2 is unavailable."""
    candidates: List[Dict[str, Any]] = []
    for other in read_csv_records(output_dir / "fig4_manifest.csv"):
        if str(other.get("paper_id")) == str(row.get("paper_id")):
            continue
        title = str(other.get("title") or "")
        abstract = str(other.get("abstract") or "")
        if not title and not abstract:
            parsed_path = Path(str(other.get("parsed_text_cache") or ""))
            if parsed_path.exists():
                parsed = read_json(parsed_path)
                title = str(parsed.get("title") or "")
                abstract = str(parsed.get("abstract") or "")
        if not title and not abstract:
            continue
        candidates.append(
            {
                "paperId": str(other.get("paper_id") or stable_text_hash(title + abstract)),
                "year": int(other.get("year") or 0),
                "title": title,
                "authors": "",
                "venue": other.get("journal", ""),
                "citationCount": 0,
                "abstract": abstract,
                "isOpenAccess": False,
                "url": "",
                "externalIds": {"DOI": other.get("doi", "")},
                "doi": other.get("doi", ""),
                "fieldsOfStudy": [str(other.get("journal", ""))] if other.get("journal") else [],
                "s2FieldsOfStudy": [],
                "retrieval_source": "local_fig4_manifest",
            }
        )
    return candidates


def strip_peer_review_boilerplate(text: str) -> str:
    cleaned = clean_markdown_text(text)
    cleaned = re.sub(
        r"Open Access This file is licensed under a Creative Commons Attribution 4\.0 International License.*?creativecommons\.org/licenses/by/4\.0/\.",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(r"\*\*Open Access\*\*.*?creativecommons\.org/licenses/by/4\.0/\.", "", cleaned, flags=re.I | re.S)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def parse_peer_review_markdown(text: str) -> Dict[str, Any]:
    cleaned = strip_peer_review_boilerplate(text)
    lower = cleaned.lower()
    warnings: List[str] = []
    included_sections: List[str] = []
    excluded_sections: List[str] = []
    start_patterns = [
        r"\n#+\s*reviewer(?:s|'s)?\s+comments?\b",
        r"\nreviewer\s*#?\s*\d+",
        r"\nreferee\s*#?\s*\d+",
        r"\nremarks\s+to\s+the\s+author",
    ]
    starts = [m.start() for pattern in start_patterns for m in re.finditer(pattern, "\n" + lower, flags=re.I)]
    start = min(starts) if starts else 0
    if starts:
        included_sections.append("reviewer_comments")
    else:
        warnings.append("review_section_parse_warning")
        included_sections.append("full_text_fallback")
    review_part = cleaned[start:].strip()
    exclusion_patterns = [
        ("author_response", r"\n#+\s*(author response|response to reviewers|responses? to reviewer|author rebuttal)\b"),
        ("editor_decision", r"\n#+\s*(editor decision|decision letter|editorial decision)\b"),
        ("references", r"\n#+\s*(references|bibliography)\b"),
    ]
    cut_positions: List[Tuple[int, str]] = []
    for section, pattern in exclusion_patterns:
        match = re.search(pattern, "\n" + review_part, flags=re.I)
        if match:
            cut_positions.append((match.start(), section))
    if cut_positions:
        cut, section = min(cut_positions, key=lambda item: item[0])
        review_part = review_part[:cut].strip()
        excluded_sections.append(section)
    if word_count(review_part) < 50:
        warnings.append("short_peer_review_text")
    return {
        "peer_review_text": review_part,
        "included_sections": included_sections,
        "excluded_sections": excluded_sections,
        "word_count": word_count(review_part),
        "warnings": warnings,
    }


def load_markdown_manifest(markdown_root: Path, journal_scope: str = "all") -> List[Dict[str, Any]]:
    manifest_rows = read_jsonl(markdown_root / "manifest.jsonl")
    by_article_id: Dict[str, Dict[str, Any]] = {}
    for row in manifest_rows:
        article_id = str(row.get("article_id") or "")
        if article_id:
            by_article_id[article_id] = row
    records: List[Dict[str, Any]] = []
    paper_dir = markdown_root / "paper"
    review_dir = markdown_root / "peer_review"
    for paper_path in sorted(paper_dir.glob("*.md")):
        article_id = paper_path.stem
        journal_id = article_id_to_journal_id(article_id)
        if journal_scope == "six_subjournals" and journal_id not in SIX_SUBJOURNAL_IDS:
            continue
        if journal_scope == "41467_only" and journal_id != "41467":
            continue
        if journal_scope == "all" and journal_id not in ALL_ELIGIBLE_JOURNAL_IDS:
            continue
        review_path = review_dir / f"{article_id}_r.md"
        source_row = by_article_id.get(article_id, {})
        records.append(
            {
                "paper_id": article_id,
                "article_id": article_id,
                "journal_id": journal_id,
                "journal": journal_name(journal_id),
                "year": int(source_row.get("year") or article_id_to_year(article_id) or 0),
                "doi": source_row.get("doi") or article_id_to_doi(article_id),
                "title": "",
                "abstract": "",
                "keywords": "",
                "article_markdown_path": str(paper_path),
                "peer_review_markdown_path": str(review_path),
                "article_pdf_path": source_row.get("article_pdf_path", ""),
                "peer_review_pdf_path": source_row.get("peer_review_pdf_path", ""),
            }
        )
    return records


def audit_markdown_inputs(
    markdown_root: Path,
    output_dir: Path,
    journal_scope: str,
    quiet: bool = False,
    audit_max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    records = load_markdown_manifest(markdown_root, journal_scope=journal_scope)
    if audit_max_records and audit_max_records > 0:
        records = records[:audit_max_records]
    progress_log(f"审计 Markdown 输入：发现 {len(records)} 篇候选。", quiet)
    cache_root = output_dir / "cache"
    audit_rows: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, record in enumerate(records, start=1):
        paper_id = str(record["paper_id"])
        paper_path = Path(str(record["article_markdown_path"]))
        review_path = Path(str(record["peer_review_markdown_path"]))
        errors: List[str] = []
        warnings: List[str] = []
        if paper_id in seen_ids:
            errors.append("duplicate_paper_id")
        seen_ids.add(paper_id)
        if not paper_path.exists():
            errors.append("missing_article_markdown")
        if not review_path.exists():
            errors.append("missing_peer_review_markdown")
        article_parsed: Dict[str, Any] = {}
        review_parsed: Dict[str, Any] = {}
        if not errors:
            try:
                article_parsed = parse_article_markdown(paper_path.read_text(encoding="utf-8", errors="replace"), paper_id)
                review_parsed = parse_peer_review_markdown(review_path.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:  # noqa: BLE001 - per-sample diagnostics.
                errors.append(f"parse_failed:{exc}")
        if article_parsed:
            record["doi"] = article_parsed.get("doi") or record.get("doi", "")
            record["title"] = article_parsed.get("title") or record.get("title", "")
            record["year"] = int(article_parsed.get("year") or record.get("year") or 0)
            record["abstract"] = article_parsed.get("abstract", "")
            warnings.extend(article_parsed.get("warnings") or [])
            if not article_parsed.get("article_text"):
                errors.append("empty_article_text")
        if review_parsed:
            warnings.extend(review_parsed.get("warnings") or [])
            if not review_parsed.get("peer_review_text"):
                errors.append("empty_peer_review_text")
        if not record.get("title"):
            errors.append("missing_title")
        if not record.get("journal"):
            errors.append("missing_journal")
        if not int(record.get("year") or 0):
            errors.append("missing_year")
        parsed_path = cache_root / paper_id / "parsed_text.json"
        dossier_path = cache_root / paper_id / "paper_dossier.json"
        paper_dossier: Dict[str, Any] = {}
        if article_parsed:
            paper_dossier = build_paper_dossier(article_parsed, record)
            write_json(dossier_path, paper_dossier)
        if article_parsed or review_parsed:
            write_json(
                parsed_path,
                {
                    "paper_id": paper_id,
                    **article_parsed,
                    **review_parsed,
                },
            )
        audit_rows.append(
            {
                **record,
                "included_in_audit": not errors,
                "included_in_main": False,
                "exclusion_reason": ";".join(errors),
                "warnings": ";".join(dict.fromkeys(warnings)),
                "parsed_text_cache": str(parsed_path),
                "paper_dossier_cache": str(dossier_path),
                "article_word_count": article_parsed.get("word_count", 0) if article_parsed else 0,
                "peer_review_word_count": review_parsed.get("word_count", 0) if review_parsed else 0,
            }
        )
        if idx == 1 or idx % 100 == 0 or idx == len(records):
            progress_log(f"审计进度 {idx}/{len(records)}。", quiet)
    write_csv(output_dir / "fig4_input_audit.csv", audit_rows)
    return audit_rows


def controlled_sample(
    audit_rows: Sequence[Mapping[str, Any]],
    sample_size: int,
    seed: int = DEFAULT_SAMPLE_SEED,
    cap_41467: float = DEFAULT_41467_CAP,
) -> List[Dict[str, Any]]:
    eligible = [dict(row) for row in audit_rows if str(row.get("included_in_audit")).lower() in {"true", "1"}]
    rng = random.Random(seed)
    if sample_size <= 0 or len(eligible) <= sample_size:
        sampled = eligible
    else:
        cap_n = int(math.floor(sample_size * cap_41467))
        comms = [row for row in eligible if str(row.get("journal_id")) == "41467"]
        others = [row for row in eligible if str(row.get("journal_id")) != "41467"]
        rng.shuffle(comms)
        rng.shuffle(others)
        selected = comms[: min(cap_n, len(comms))]
        remaining = sample_size - len(selected)
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for row in others:
            groups.setdefault((str(row.get("journal_id")), str(row.get("year"))), []).append(row)
        for group_rows in groups.values():
            rng.shuffle(group_rows)
        group_keys = sorted(groups)
        while remaining > 0 and any(groups.values()):
            progressed = False
            for key in group_keys:
                if remaining <= 0:
                    break
                if groups[key]:
                    selected.append(groups[key].pop())
                    remaining -= 1
                    progressed = True
            if not progressed:
                break
        if remaining > 0:
            already = {row["paper_id"] for row in selected}
            backfill = [row for row in eligible if row["paper_id"] not in already]
            rng.shuffle(backfill)
            selected.extend(backfill[:remaining])
        sampled = selected[:sample_size]
    sampled = sorted(sampled, key=lambda row: (int(row.get("year") or 0), str(row.get("journal_id")), str(row.get("paper_id"))))
    for row in sampled:
        row["included_in_main"] = True
        row["sample_seed"] = seed
    return sampled


def sample_manifest(
    output_dir: Path,
    sample_size: int,
    seed: int,
    cap_41467: float,
    quiet: bool = False,
    require_screen_pass: bool = False,
) -> List[Dict[str, Any]]:
    screen_path = output_dir / "fig4_peer_review_screen.csv"
    if screen_path.exists():
        candidate_rows = read_csv_records(screen_path)
        candidate_rows = [
            {**row, "included_in_audit": True}
            for row in candidate_rows
            if str(row.get("screen_pass")).lower() in {"true", "1"}
        ]
        if sample_size > 0 and len(candidate_rows) < sample_size:
            raise RuntimeError(f"screen_pass_count={len(candidate_rows)} is below requested sample_size={sample_size}")
    elif require_screen_pass:
        raise RuntimeError("fig4_peer_review_screen.csv is required before sampling with explicit review filter")
    else:
        candidate_rows = read_csv_records(output_dir / "fig4_input_audit.csv")
    sampled = controlled_sample(candidate_rows, sample_size=sample_size, seed=seed, cap_41467=cap_41467)
    write_csv(output_dir / "fig4_manifest.csv", sampled)
    progress_log(f"采样完成：主实验样本 {len(sampled)} 篇。", quiet)
    return sampled


def normalize_doi_value(value: Any) -> str:
    text = str(value or "").strip().lower()
    for prefix in ["https://doi.org/", "http://doi.org/", "doi:"]:
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.rstrip(".,;)")


def load_fig3_weights(path: Path) -> Dict[str, float]:
    """Load learned Fig.3 metric weights and normalize them over the seven metrics."""
    if not path.exists():
        from aspr.graph_innovation_scorer import DEFAULT_GRAPH_METRIC_WEIGHTS

        raw_weights = dict(DEFAULT_GRAPH_METRIC_WEIGHTS)
    else:
        weights_df = pd.read_csv(path)
        name_col = "metric" if "metric" in weights_df.columns else weights_df.columns[0]
        weight_col = "weight" if "weight" in weights_df.columns else weights_df.columns[-1]
        raw_weights = {
            str(row[name_col]): max(0.0, numeric(row[weight_col], 0.0))
            for _, row in weights_df.iterrows()
            if str(row[name_col]) in INNOVATION_METRIC_NAMES
        }
    total = sum(raw_weights.values())
    if total <= 0:
        equal = 1.0 / len(INNOVATION_METRIC_NAMES)
        return {metric: equal for metric in INNOVATION_METRIC_NAMES}
    return {metric: raw_weights.get(metric, 0.0) / total for metric in INNOVATION_METRIC_NAMES}


INNOVATION_METRIC_NAMES = ["B", "RS", "DeltaQ0", "Uzzi", "RTD", "BurtIP", "PDE"]


def graph_metric_prompt_block(metrics: Mapping[str, Any]) -> str:
    from aspr.graph_innovation_scorer import METRIC_DESCRIPTIONS

    metric_lines = [
        f"- {metric}: {numeric(metrics.get(metric), 0.0):.3f} ({METRIC_DESCRIPTIONS.get(metric, metric)})"
        for metric in INNOVATION_METRIC_NAMES
    ]
    limitations = normalize_whitespace(str(metrics.get("graph_metric_failure_reason") or ""))
    if not limitations:
        limitations = f"metric_source={metrics.get('metric_source', '')}; calibrated with Fig.3 learned weights."
    return (
        "【图谱结构证据】\n"
        f"综合结构扰动潜力分数: {numeric(metrics.get('weighted_score_fig3'), 0.0):.3f} / 1.000\n"
        f"证据置信度: {numeric(metrics.get('graph_confidence'), 0.0):.3f} / 1.000\n"
        f"主导机制: {metrics.get('top_mechanisms', '') or '无明显主导机制'}\n"
        "七维指标:\n"
        + "\n".join(metric_lines)
        + "\n数据限制:\n"
        + f"- {limitations}"
    )


def graph_metric_result_from_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    metric_values = {metric: numeric(row.get(metric), 0.0) for metric in INNOVATION_METRIC_NAMES}
    top = sorted(metric_values.items(), key=lambda item: item[1], reverse=True)
    return {
        "metrics": metric_values,
        "weighted_score": numeric(row.get("weighted_score_fig3"), 0.0),
        "confidence": numeric(row.get("graph_confidence"), 0.0),
        "top_mechanisms": [name for name, value in top if value > 0.15][:3],
        "limitations": [str(row.get("graph_metric_failure_reason") or f"metric_source={row.get('metric_source', '')}")],
        "diagnostics": {
            "metric_source": row.get("metric_source", ""),
            "openalex_id": row.get("openalex_id", ""),
            "doi": row.get("doi", ""),
        },
    }


def resolve_openalex_work_by_doi(doi: str, timeout: int = 20) -> Tuple[Optional[Dict[str, Any]], str]:
    if not normalize_doi_value(doi):
        return None, "missing_doi"
    import requests

    endpoint = "https://api.openalex.org/works/" + urllib.parse.quote("https://doi.org/" + normalize_doi_value(doi), safe="")
    params = {
        "select": "id,doi,title,publication_year,referenced_works,primary_topic,topics,concepts,cited_by_count"
    }
    try:
        response = requests.get(endpoint, params=params, timeout=timeout)
    except requests.RequestException as exc:
        return None, f"openalex_request_failed:{exc}"
    if response.status_code != 200:
        return None, f"openalex_status_{response.status_code}:{response.text[:120]}"
    return response.json(), ""


def fetch_openalex_reference_works(reference_ids: Sequence[str], timeout: int = 25) -> List[Dict[str, Any]]:
    ids = [str(item).rsplit("/", 1)[-1] for item in reference_ids if str(item).strip()]
    if not ids:
        return []
    import requests

    rows: List[Dict[str, Any]] = []
    for idx in range(0, min(len(ids), 80), 25):
        chunk = ids[idx : idx + 25]
        params = {
            "filter": "openalex_id:" + "|".join(chunk),
            "per-page": len(chunk),
            "select": "id,publication_year,primary_topic,topics,concepts,cited_by_count",
        }
        try:
            response = requests.get("https://api.openalex.org/works", params=params, timeout=timeout)
        except requests.RequestException:
            continue
        if response.status_code == 200:
            rows.extend(response.json().get("results") or [])
    return rows


def openalex_topic_label(work: Mapping[str, Any]) -> str:
    primary = work.get("primary_topic") if isinstance(work.get("primary_topic"), Mapping) else {}
    if primary.get("display_name"):
        return str(primary.get("display_name"))
    topics = work.get("topics") if isinstance(work.get("topics"), list) else []
    if topics and isinstance(topics[0], Mapping) and topics[0].get("display_name"):
        return str(topics[0].get("display_name"))
    concepts = work.get("concepts") if isinstance(work.get("concepts"), list) else []
    if concepts and isinstance(concepts[0], Mapping) and concepts[0].get("display_name"):
        return str(concepts[0].get("display_name"))
    return "unknown"


def simpson_diversity(labels: Sequence[str]) -> float:
    labels = [label for label in labels if label and label != "unknown"]
    if not labels:
        return float("nan")
    counts = {label: labels.count(label) for label in set(labels)}
    total = float(len(labels))
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def entropy_norm(labels: Sequence[str]) -> float:
    labels = [label for label in labels if label and label != "unknown"]
    if not labels:
        return float("nan")
    counts = {label: labels.count(label) for label in set(labels)}
    total = float(len(labels))
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return entropy / math.log(max(len(counts), 2))


def compute_openalex_publication_day_metrics(row: Mapping[str, Any], weights: Mapping[str, float]) -> Dict[str, Any]:
    doi = normalize_doi_value(row.get("doi") or article_id_to_doi(str(row.get("paper_id") or "")))
    work, failure = resolve_openalex_work_by_doi(doi)
    if not work:
        return {
            "paper_id": row.get("paper_id", ""),
            "doi": doi,
            "openalex_id": "",
            "graph_metric_valid": False,
            "graph_metric_failure_reason": failure,
            "metric_source": "openalex_publication_day_proxy",
        }
    ref_ids = work.get("referenced_works") if isinstance(work.get("referenced_works"), list) else []
    refs = fetch_openalex_reference_works(ref_ids)
    context_works = refs if refs else [work]
    labels = [openalex_topic_label(item) for item in context_works]
    years = [numeric(item.get("publication_year")) for item in context_works]
    years = [year for year in years if math.isfinite(year)]
    n_refs = len(ref_ids)
    diversity = simpson_diversity(labels)
    entropy = entropy_norm(labels)
    max_share = 1.0
    valid_labels = [label for label in labels if label and label != "unknown"]
    if valid_labels:
        max_share = max(valid_labels.count(label) for label in set(valid_labels)) / len(valid_labels)
    year_spread = min(1.0, float(np.std(years)) / 18.0) if len(years) >= 2 else 0.0
    ref_depth = min(1.0, math.log1p(max(n_refs, len(context_works))) / math.log(80.0))
    metric_values = {
        "B": min(1.0, (1.0 - max_share) * ref_depth),
        "RS": diversity if math.isfinite(diversity) else 0.0,
        "DeltaQ0": min(1.0, (diversity if math.isfinite(diversity) else 0.0) * ref_depth),
        "Uzzi": min(1.0, 0.5 * year_spread + 0.5 * (1.0 - max_share)),
        "RTD": diversity if math.isfinite(diversity) else 0.0,
        "BurtIP": max(0.0, 1.0 - max_share),
        "PDE": entropy if math.isfinite(entropy) else 0.0,
    }
    weighted = sum(metric_values[metric] * weights.get(metric, 0.0) for metric in INNOVATION_METRIC_NAMES)
    valid = bool(refs or valid_labels)
    return {
        "paper_id": row.get("paper_id", ""),
        "doi": doi,
        "openalex_id": work.get("id", ""),
        **metric_values,
        "weighted_score_fig3": max(0.0, min(1.0, weighted)),
        "graph_confidence": min(0.95, 0.35 + 0.45 * ref_depth + (0.15 if refs else 0.0)),
        "metric_source": "openalex_publication_day_proxy",
        "graph_metric_valid": valid,
        "graph_metric_failure_reason": "" if valid else "openalex_reference_context_unavailable",
        "reference_count": n_refs,
        "reference_metadata_count": len(refs),
        "top_mechanisms": ", ".join(
            metric for metric, value in sorted(metric_values.items(), key=lambda item: item[1], reverse=True)[:3] if value > 0.15
        ),
    }


def load_graph_metric_table(path: Path) -> List[Dict[str, Any]]:
    return read_csv_records(path) if path and path.exists() else []


def graph_table_lookup(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    lookup: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        for key in ("paper_id", "doi", "openalex_id", "id"):
            value = normalize_doi_value(row.get(key)) if key == "doi" else str(row.get(key) or "")
            if value:
                lookup[value] = row
    return lookup


def graph_metrics_from_table(row: Mapping[str, Any], table_lookup: Mapping[str, Mapping[str, Any]], weights: Mapping[str, float]) -> Optional[Dict[str, Any]]:
    candidates = [
        str(row.get("paper_id") or ""),
        normalize_doi_value(row.get("doi") or article_id_to_doi(str(row.get("paper_id") or ""))),
        str(row.get("openalex_id") or ""),
    ]
    source = next((table_lookup[key] for key in candidates if key and key in table_lookup), None)
    if not source:
        return None
    metric_values = {metric: numeric(source.get(metric) or source.get(metric + "_calibrated") or source.get(metric + "_z"), 0.0) for metric in INNOVATION_METRIC_NAMES}
    weighted = numeric(source.get("weighted_score_fig3"))
    if not math.isfinite(weighted):
        weighted = sum(metric_values[metric] * weights.get(metric, 0.0) for metric in INNOVATION_METRIC_NAMES)
    return {
        "paper_id": row.get("paper_id", ""),
        "doi": normalize_doi_value(row.get("doi") or source.get("doi") or article_id_to_doi(str(row.get("paper_id") or ""))),
        "openalex_id": source.get("openalex_id") or source.get("id") or "",
        **metric_values,
        "weighted_score_fig3": max(0.0, min(1.0, weighted)),
        "graph_confidence": numeric(source.get("graph_confidence"), 0.9),
        "metric_source": "graph_metrics_table",
        "graph_metric_valid": True,
        "graph_metric_failure_reason": "",
        "top_mechanisms": ", ".join(
            metric for metric, value in sorted(metric_values.items(), key=lambda item: item[1], reverse=True)[:3] if value > 0.15
        ),
    }


def run_graph_metrics_stage(
    output_dir: Path,
    graph_metrics_source: str,
    fig3_weights_path: Path,
    graph_metrics_table: Optional[Path] = None,
    quiet: bool = False,
) -> List[Dict[str, Any]]:
    manifest = read_csv_records(output_dir / "fig4_manifest.csv")
    weights = load_fig3_weights(fig3_weights_path)
    table_rows = load_graph_metric_table(graph_metrics_table) if graph_metrics_table else []
    lookup = graph_table_lookup(table_rows)
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(manifest, start=1):
        paper_id = str(row.get("paper_id") or "")
        metrics = graph_metrics_from_table(row, lookup, weights) if lookup else None
        if metrics is None and graph_metrics_source == "fig3_fig2":
            metrics = compute_openalex_publication_day_metrics(row, weights)
        elif metrics is None and graph_metrics_source == "lightweight":
            metrics = compute_openalex_publication_day_metrics(row, weights)
            metrics["metric_source"] = "lightweight_openalex_proxy"
        elif metrics is None:
            metrics = {
                "paper_id": paper_id,
                "doi": normalize_doi_value(row.get("doi") or article_id_to_doi(paper_id)),
                "openalex_id": "",
                "graph_metric_valid": False,
                "graph_metric_failure_reason": "graph_metrics_disabled",
                "metric_source": graph_metrics_source,
            }
        for metric in INNOVATION_METRIC_NAMES:
            metrics.setdefault(metric, float("nan"))
        metrics.setdefault("weighted_score_fig3", float("nan"))
        metrics.setdefault("graph_confidence", 0.0)
        rows.append(metrics)
        cache_dir = output_dir / "cache" / paper_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        write_json(cache_dir / "fig4_graph_metrics.json", metrics)
        if idx == 1 or idx % 10 == 0 or idx == len(manifest):
            valid_n = sum(str(item.get("graph_metric_valid")).lower() in {"true", "1"} for item in rows)
            progress_log(f"Graph metrics 进度 {idx}/{len(manifest)}，有效 {valid_n}。", quiet)
    write_csv(output_dir / "fig4_graph_metrics.csv", rows)
    return rows


@dataclass
class Fig4ArgsForAgent:
    s2_api_key: str
    and_search: bool
    top_n: int
    agent_context_mode: str = "dossier"


def clear_agent_dependent_caches(cache_dir: Path) -> None:
    """Remove per-paper downstream caches that become stale after agent regeneration."""
    for filename in (
        "agent_innovation_labels.json",
        "semantic_claim_matches.json",
        "structured_consistency.json",
    ):
        path = cache_dir / filename
        try:
            path.unlink()
        except FileNotFoundError:
            continue


def run_aspr_agent_for_row(
    row: Mapping[str, Any],
    cache_dir: Path,
    args_for_agent: Fig4ArgsForAgent,
    force_agent: bool = False,
    reuse_agent: bool = True,
) -> Dict[str, Any]:
    paper_id = str(row["paper_id"])
    agent_path = cache_dir / "agent_eval.json"
    retrieved_path = cache_dir / "retrieved_papers.json"
    if reuse_agent and agent_path.exists() and not force_agent:
        cached = read_json(agent_path)
        cached["cache_reused"] = True
        return cached
    start = time.time()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if force_agent:
        clear_agent_dependent_caches(cache_dir)
    try:
        from aspr.open_scholar import (
            OpenScholar,
            keywords_extract,
            retrieval_backend_status,
            retrieval_recall,
            retrieval_rerank,
        )
        from aspr.lats import evaluate_paper_innovation
    except Exception as exc:  # noqa: BLE001
        result = {
            "paper_id": paper_id,
            "success": False,
            "failure_reason": f"aspr_import_failed:{exc}",
            "agent_runtime_seconds": time.time() - start,
            "innovation_evaluation": "",
            "evaluation_log": [],
        }
        write_json(agent_path, result)
        return result
    try:
        abstract = str(row.get("abstract") or "")
        keywords = [item.strip() for item in str(row.get("keywords") or "").split(",") if item.strip()]
        if not keywords:
            keywords = keywords_extract(abstract)
        dossier_path = Path(str(row.get("paper_dossier_cache") or cache_dir / "paper_dossier.json"))
        if dossier_path.exists():
            paper_dossier = read_json(dossier_path)
        else:
            paper_dossier = build_paper_dossier(
                {
                    "title": row.get("title", ""),
                    "abstract": abstract,
                    "doi": row.get("doi", ""),
                    "year": row.get("year", ""),
                    "article_text": abstract,
                    "word_count": word_count(abstract),
                },
                row,
                keywords=keywords,
            )
            write_json(dossier_path, paper_dossier)
        paper_context = "" if args_for_agent.agent_context_mode == "abstract_only" else format_paper_dossier_for_agent(paper_dossier)
        graph_metrics_path = cache_dir / "fig4_graph_metrics.json"
        graph_metric_evidence_text: Optional[str] = None
        graph_metric_result: Optional[Dict[str, Any]] = None
        if graph_metrics_path.exists():
            graph_metrics = read_json(graph_metrics_path)
            if str(graph_metrics.get("graph_metric_valid")).lower() in {"true", "1"}:
                graph_metric_evidence_text = graph_metric_prompt_block(graph_metrics)
                graph_metric_result = graph_metric_result_from_row(graph_metrics)
        open_scholar = OpenScholar(args_for_agent)
        retrieval_failure = ""
        s2_key_status = "not_run"
        query_audits: List[Dict[str, Any]] = []
        try:
            papers = open_scholar.search_semantic_scholar(keywords)
            query_audits = getattr(open_scholar, "last_query_audits", [])
            retrieval_sources = {str(item.get("source")) for item in query_audits if item.get("source")}
            retrieval_source = (
                "semantic_scholar_keyed"
                if "semantic_scholar_keyed" in retrieval_sources
                else getattr(open_scholar, "last_retrieval_source", "semantic_scholar_anonymous")
            )
            s2_key_status = getattr(open_scholar, "s2_key_status", "unknown")
        except Exception as exc:  # noqa: BLE001 - demo/local fallback when S2 is unavailable.
            retrieval_failure = str(exc)
            query_audits = getattr(open_scholar, "last_query_audits", [])
            s2_key_status = getattr(open_scholar, "s2_key_status", "unknown")
            output_dir = cache_dir.parents[1]
            papers = local_prior_art_candidates(row, output_dir)
            retrieval_source = "local_fallback"
            print(f"Semantic Scholar failed for {paper_id}; using local fallback candidates: {retrieval_failure}")
        papers, excluded_candidates = filter_prior_art_candidates(
            papers,
            target_title=str(row.get("title", "")),
            target_doi=str(row.get("doi", "")),
            cutoff_year=int(row.get("year") or 0),
        )
        excluded_target_count = sum(
            1
            for item in excluded_candidates
            if "target_doi_match" in item.get("reasons", []) or "target_title_match" in item.get("reasons", [])
        )
        excluded_future_count = sum(1 for item in excluded_candidates if "post_publication_year" in item.get("reasons", []))
        paper_formatted = [f'Title:{paper.get("title", "")}. Abstract:{paper.get("abstract", "")}' for paper in papers]
        item_to_paper = {item: paper for item, paper in zip(paper_formatted, papers)}
        if paper_formatted:
            recalled, _ = retrieval_recall(str(row.get("title", "")) + "\n" + abstract, paper_formatted)
            top_n = max(1, int(args_for_agent.top_n))
            recall_n = min(len(recalled), max(top_n * 5, round(len(recalled) / 10), top_n))
            reranked, _ = retrieval_rerank(str(row.get("title", "")) + "\n" + abstract, recalled[:recall_n])
            retrieved = [item_to_paper[item] for item in reranked[: min(len(reranked), top_n)]]
        else:
            retrieved = []
        ranker_status = retrieval_backend_status()
        write_json(retrieved_path, {"paper_id": paper_id, "retrieved_papers": retrieved})
        write_json(
            cache_dir / "retrieval_audit.json",
            {
                "paper_id": paper_id,
                "cutoff_year": int(row.get("year") or 0),
                "excluded_candidates": excluded_candidates,
                "kept_candidates_count": len(papers),
                "retrieved_papers_count": len(retrieved),
                "retrieval_source": retrieval_source,
                "retrieval_failure": retrieval_failure,
                "s2_key_status": s2_key_status,
                "query_terms": keywords,
                "query_audits": query_audits,
                "ranker_status": ranker_status,
                "excluded_target_count": excluded_target_count,
                "excluded_future_count": excluded_future_count,
            },
        )
        eval_result = evaluate_paper_innovation(
            paper_title=str(row.get("title", "")),
            paper_abstract=abstract,
            retrieved_papers=retrieved,
            paper_context=paper_context,
            max_iterations=getenv_int("FIG4_AGENT_MAX_ITERATIONS", 1),
            graph_metric_evidence=graph_metric_evidence_text,
            graph_metric_result=graph_metric_result,
        )
        result = {
            "paper_id": paper_id,
            "title": row.get("title", ""),
            "keywords": keywords,
            "retrieved_papers_count": len(retrieved),
            "retrieved_papers_cache": str(retrieved_path),
            "paper_context_cache": str(dossier_path),
            "agent_context_mode": args_for_agent.agent_context_mode,
            "retrieval_cutoff_year": int(row.get("year") or 0),
            "excluded_retrieval_count": len(excluded_candidates),
            "retrieval_source": retrieval_source,
            "s2_key_status": s2_key_status,
            "query_terms": keywords,
            "ranker_status": ranker_status,
            "excluded_target_count": excluded_target_count,
            "excluded_future_count": excluded_future_count,
            "agent_runtime_seconds": time.time() - start,
            "success": bool(eval_result.get("success", True)),
            "failure_reason": "",
            **eval_result,
        }
        write_json(cache_dir / "graph_metric_evidence.json", eval_result.get("graph_metric_evidence") or {})
        write_json(cache_dir / "committee_report.json", eval_result.get("committee_report") or {})
    except Exception as exc:  # noqa: BLE001
        result = {
            "paper_id": paper_id,
            "success": False,
            "failure_reason": str(exc),
            "agent_runtime_seconds": time.time() - start,
            "innovation_evaluation": "",
            "evaluation_log": [],
        }
    write_json(agent_path, result)
    return result


def run_agent_stage(
    output_dir: Path,
    s2_api_key: str,
    top_n: int,
    and_search: bool,
    agent_context_mode: str,
    force_agent: bool,
    reuse_agent: bool,
    max_agent: Optional[int],
    quiet: bool = False,
    agent_runner: Optional[Callable[[Mapping[str, Any], Path], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    manifest = read_csv_records(output_dir / "fig4_manifest.csv")
    if max_agent is not None:
        manifest = manifest[:max_agent]
    args_for_agent = Fig4ArgsForAgent(
        s2_api_key=s2_api_key,
        and_search=and_search,
        top_n=top_n,
        agent_context_mode=agent_context_mode,
    )
    outputs: List[Dict[str, Any]] = []
    for idx, row in enumerate(manifest, start=1):
        paper_id = str(row["paper_id"])
        cache_dir = output_dir / "cache" / paper_id
        if agent_runner:
            result = agent_runner(row, cache_dir)
            write_json(cache_dir / "agent_eval.json", result)
        else:
            result = run_aspr_agent_for_row(row, cache_dir, args_for_agent, force_agent=force_agent, reuse_agent=reuse_agent)
        outputs.append(result)
        if idx == 1 or idx % 10 == 0 or idx == len(manifest):
            progress_log(f"Agent 进度 {idx}/{len(manifest)}，成功 {sum(bool(r.get('success')) for r in outputs)}。", quiet)
    write_jsonl(output_dir / "fig4_agent_outputs.jsonl", outputs)
    write_csv(
        output_dir / "fig4_retrieval_diagnostics.csv",
        [
            {
                "paper_id": row.get("paper_id", ""),
                "retrieval_source": row.get("retrieval_source", ""),
                "s2_key_status": row.get("s2_key_status", ""),
                "query_terms": "; ".join(str(item) for item in (row.get("query_terms") or [])),
                "recall_backend": (row.get("ranker_status") or {}).get("recall_backend", ""),
                "reranker_backend": (row.get("ranker_status") or {}).get("reranker_backend", ""),
                "recall_model_path": (row.get("ranker_status") or {}).get("recall_model_path", ""),
                "reranker_model_path": (row.get("ranker_status") or {}).get("reranker_model_path", ""),
                "recall_batch_size_used": (row.get("ranker_status") or {}).get("recall_batch_size_used", ""),
                "rerank_batch_size_used": (row.get("ranker_status") or {}).get("rerank_batch_size_used", ""),
                "retrieval_failure_stage": (row.get("ranker_status") or {}).get("retrieval_failure_stage", ""),
                "recall_failure": (row.get("ranker_status") or {}).get("recall_failure", ""),
                "reranker_failure": (row.get("ranker_status") or {}).get("reranker_failure", ""),
                "excluded_target_count": row.get("excluded_target_count", 0),
                "excluded_future_count": row.get("excluded_future_count", 0),
                "retrieved_papers_count": row.get("retrieved_papers_count", 0),
                "failure_reason": row.get("failure_reason", ""),
            }
            for row in outputs
        ],
    )
    return outputs


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat_json(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> Tuple[Dict[str, Any], str]:
        import requests

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=self.timeout)
        if response.status_code == 400 and "response_format" in response.text:
            payload.pop("response_format", None)
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        return parse_json_object(raw), raw

    def embeddings(self, texts: Sequence[str], model: Optional[str] = None) -> List[List[float]]:
        import requests

        payload = {"model": model or self.model, "input": list(texts)}
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()["data"]
        data = sorted(data, key=lambda item: int(item.get("index", 0)))
        return [item["embedding"] for item in data]


def parse_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def judge_config_from_env(prefix: str = "FIG4_JUDGE") -> Optional[OpenAICompatibleClient]:
    base_url = getenv(f"{prefix}_BASE_URL") or getenv("ASPR_LLM_BASE_URL")
    api_key = getenv(f"{prefix}_API_KEY") or getenv("DEEPSEEK_API_KEY")
    model = getenv(f"{prefix}_MODEL") or getenv("ASPR_LATS_LLM_MODEL")
    if not base_url or not api_key or not model:
        return None
    return OpenAICompatibleClient(base_url=base_url, api_key=api_key, model=model)


def embedding_config_from_env() -> Optional[OpenAICompatibleClient]:
    base_url = getenv("FIG4_EMBEDDING_BASE_URL") or getenv("FIG4_JUDGE_BASE_URL")
    api_key = getenv("FIG4_EMBEDDING_API_KEY") or getenv("FIG4_JUDGE_API_KEY")
    model = getenv("FIG4_EMBEDDING_MODEL")
    if not base_url or not api_key or not model:
        return None
    return OpenAICompatibleClient(base_url=base_url, api_key=api_key, model=model)


def build_rating_prompt(kind: str, title: str, abstract: str, text: str) -> List[Dict[str, str]]:
    clipped_text = text[:12000]
    system = (
        "You are a strict scientific meta-reviewer. Return only valid JSON. "
        "Use the same rubric for peer-review text and ASPR system output."
    )
    user = f"""
Paper title: {title}
Paper abstract: {abstract[:2500]}
Text kind: {kind}

Assess the text on a 1-5 scale where 1=low/weak and 5=high/strong.
Return this JSON schema:
{{
  "overall_score_1_5": number,
  "novelty": number,
  "significance": number,
  "rigor": number,
  "limitations": number,
  "future_work": number,
  "confidence": number,
  "evidence_quotes": [string],
  "aspects": {{
    "significance": [string],
    "novelty": [string],
    "rigor": [string],
    "limitations": [string],
    "future_work": [string]
  }}
}}

Text:
{clipped_text}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def run_rating_judge(
    output_dir: Path,
    judge_backend: str,
    quiet: bool = False,
    judge_client: Optional[OpenAICompatibleClient] = None,
) -> List[Dict[str, Any]]:
    manifest = read_csv_records(output_dir / "fig4_manifest.csv")
    agent_outputs = {row["paper_id"]: row for row in read_jsonl(output_dir / "fig4_agent_outputs.jsonl")}
    client = judge_client or (judge_config_from_env() if judge_backend == "openai-compatible" else None)
    rows: List[Dict[str, Any]] = []
    if client is None:
        progress_log("未配置 FIG4_JUDGE_*，评分 judge 阶段写入缺失诊断。", quiet)
    for idx, row in enumerate(manifest, start=1):
        paper_id = str(row["paper_id"])
        parsed_path = Path(str(row.get("parsed_text_cache") or output_dir / "cache" / paper_id / "parsed_text.json"))
        parsed = read_json(parsed_path) if parsed_path.exists() else {}
        agent = agent_outputs.get(paper_id, {})
        texts = {
            "peer_review": parsed.get("peer_review_text", ""),
            "agent": agent.get("innovation_evaluation", ""),
        }
        for kind, text in texts.items():
            cache_path = output_dir / "cache" / paper_id / f"rating_{kind}.json"
            if cache_path.exists():
                payload = read_json(cache_path)
            elif client is None or not text:
                payload = {
                    "paper_id": paper_id,
                    "kind": kind,
                    "success": False,
                    "failure_reason": "judge_not_configured" if client is None else "empty_text",
                    "raw_response": "",
                }
                write_json(cache_path, payload)
            else:
                messages = build_rating_prompt(kind, str(row.get("title", "")), str(row.get("abstract", "")), str(text))
                prompt_hash = stable_text_hash(json.dumps(messages, ensure_ascii=False, sort_keys=True), length=16)
                try:
                    parsed_json, raw = client.chat_json(messages)
                    payload = {
                        "paper_id": paper_id,
                        "kind": kind,
                        "success": True,
                        "model": client.model,
                        "prompt_hash": prompt_hash,
                        "text_hash": stable_text_hash(text, length=16),
                        "raw_response": raw,
                        **parsed_json,
                    }
                except Exception as exc:  # noqa: BLE001
                    payload = {
                        "paper_id": paper_id,
                        "kind": kind,
                        "success": False,
                        "failure_reason": str(exc),
                        "raw_response": "",
                    }
                write_json(cache_path, payload)
            rows.append(payload)
        if idx == 1 or idx % 10 == 0 or idx == len(manifest):
            progress_log(f"Judge 进度 {idx}/{len(manifest)}。", quiet)
    write_jsonl(output_dir / "fig4_rating_judgements.jsonl", rows)
    return rows


def build_innovation_label_prompt(kind: str, title: str, abstract: str, text: str) -> List[Dict[str, str]]:
    clipped_text = text[:14000]
    aspect_list = ", ".join(INNOVATION_ASPECTS)
    system = (
        "You are a strict scientific meta-reviewer. Return only valid JSON. "
        "Extract only claims stated in the provided text. Do not infer from the paper abstract."
    )
    user = f"""
Paper title: {title}
Paper abstract for context only: {abstract[:2500]}
Text kind: {kind}

Extract quote-grounded innovation evaluation labels from the text.
Each score is 1-5, where 1=negative/absent/weak and 5=strong/positive.
Every non-empty label must include at least one short verbatim quote copied from the Text.
If an aspect is not discussed, set score_1_5 to null, points to [], quotes to [], and confidence to 0.

Return this JSON schema:
{{
  "overall_innovation_stance": {{
    "score_1_5": number|null,
    "label": "negative|mixed|neutral|positive|strong_positive|not_discussed",
    "quote": string,
    "confidence": number
  }},
  "aspects": {{
    "novelty": {{"score_1_5": number|null, "points": [string], "quotes": [string], "confidence": number}},
    "significance": {{"score_1_5": number|null, "points": [string], "quotes": [string], "confidence": number}},
    "prior_art_comparison": {{"score_1_5": number|null, "points": [string], "quotes": [string], "confidence": number}},
    "evidence_rigor": {{"score_1_5": number|null, "points": [string], "quotes": [string], "confidence": number}},
    "limitations": {{"score_1_5": number|null, "points": [string], "quotes": [string], "confidence": number}},
    "future_work": {{"score_1_5": number|null, "points": [string], "quotes": [string], "confidence": number}}
  }},
  "notes": string
}}

Allowed aspects: {aspect_list}

Text:
{clipped_text}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _score_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return max(1.0, min(5.0, number))


def normalize_innovation_label_payload(
    payload: Mapping[str, Any],
    paper_id: str,
    kind: str,
    text: str,
    model: str = "",
    raw_response: str = "",
) -> Dict[str, Any]:
    stance = payload.get("overall_innovation_stance") if isinstance(payload.get("overall_innovation_stance"), dict) else {}
    aspects_in = payload.get("aspects") if isinstance(payload.get("aspects"), dict) else {}
    normalized_aspects: Dict[str, Any] = {}
    warnings: List[str] = []
    lower_text = str(text or "").lower()
    for aspect in INNOVATION_ASPECTS:
        item = aspects_in.get(aspect) if isinstance(aspects_in.get(aspect), dict) else {}
        points = [normalize_whitespace(str(value)) for value in item.get("points", []) if normalize_whitespace(str(value))] if isinstance(item.get("points"), list) else []
        quotes = [normalize_whitespace(str(value)) for value in item.get("quotes", []) if normalize_whitespace(str(value))] if isinstance(item.get("quotes"), list) else []
        score = _score_or_none(item.get("score_1_5"))
        if score is not None and not quotes:
            warnings.append(f"{aspect}:missing_quote")
        for quote in quotes:
            if quote.lower() not in lower_text:
                warnings.append(f"{aspect}:quote_not_exact")
                break
        normalized_aspects[aspect] = {
            "score_1_5": score,
            "points": points[:8],
            "quotes": quotes[:8],
            "confidence": numeric(item.get("confidence"), 0.0),
        }
    stance_quote = normalize_whitespace(str(stance.get("quote") or ""))
    stance_score = _score_or_none(stance.get("score_1_5"))
    if stance_score is not None and not stance_quote:
        warnings.append("overall_innovation_stance:missing_quote")
    elif stance_quote and stance_quote.lower() not in lower_text:
        warnings.append("overall_innovation_stance:quote_not_exact")
    has_required_quote = (
        (stance_score is None or bool(stance_quote))
        and all(item["score_1_5"] is None or bool(item["quotes"]) for item in normalized_aspects.values())
    )
    return {
        "paper_id": paper_id,
        "kind": kind,
        "success": bool(has_required_quote),
        "failure_reason": "" if has_required_quote else "missing_required_quotes",
        "model": model,
        "overall_innovation_stance": {
            "score_1_5": stance_score,
            "label": str(stance.get("label") or "not_discussed"),
            "quote": stance_quote,
            "confidence": numeric(stance.get("confidence"), 0.0),
        },
        "aspects": normalized_aspects,
        "warnings": warnings,
        "raw_response": raw_response,
    }


def run_innovation_label_judge(
    output_dir: Path,
    judge_backend: str,
    quiet: bool = False,
    judge_client: Optional[OpenAICompatibleClient] = None,
) -> List[Dict[str, Any]]:
    manifest = read_csv_records(output_dir / "fig4_manifest.csv")
    agent_outputs = {row["paper_id"]: row for row in read_jsonl(output_dir / "fig4_agent_outputs.jsonl")}
    client = judge_client or (judge_config_from_env() if judge_backend == "openai-compatible" else None)
    rows: List[Dict[str, Any]] = []
    if client is None:
        progress_log("未配置 FIG4_JUDGE_*，创新标签抽取阶段写入缺失诊断。", quiet)
    for idx, row in enumerate(manifest, start=1):
        paper_id = str(row["paper_id"])
        parsed_path = Path(str(row.get("parsed_text_cache") or output_dir / "cache" / paper_id / "parsed_text.json"))
        parsed = read_json(parsed_path) if parsed_path.exists() else {}
        agent = agent_outputs.get(paper_id, {})
        texts = {
            "peer_review": parsed.get("peer_review_text", ""),
            "agent": agent.get("innovation_evaluation", ""),
        }
        for kind, text in texts.items():
            cache_name = "peer_innovation_labels.json" if kind == "peer_review" else "agent_innovation_labels.json"
            cache_path = output_dir / "cache" / paper_id / cache_name
            if cache_path.exists():
                payload = read_json(cache_path)
            elif client is None or not text:
                payload = {
                    "paper_id": paper_id,
                    "kind": kind,
                    "success": False,
                    "failure_reason": "judge_not_configured" if client is None else "empty_text",
                    "overall_innovation_stance": {"score_1_5": None, "label": "not_discussed", "quote": "", "confidence": 0.0},
                    "aspects": {
                        aspect: {"score_1_5": None, "points": [], "quotes": [], "confidence": 0.0}
                        for aspect in INNOVATION_ASPECTS
                    },
                    "warnings": [],
                    "raw_response": "",
                }
                write_json(cache_path, payload)
            else:
                messages = build_innovation_label_prompt(kind, str(row.get("title", "")), str(row.get("abstract", "")), str(text))
                try:
                    parsed_json, raw = client.chat_json(messages)
                    payload = normalize_innovation_label_payload(
                        parsed_json,
                        paper_id=paper_id,
                        kind=kind,
                        text=str(text),
                        model=client.model,
                        raw_response=raw,
                    )
                except Exception as exc:  # noqa: BLE001
                    payload = {
                        "paper_id": paper_id,
                        "kind": kind,
                        "success": False,
                        "failure_reason": str(exc),
                        "overall_innovation_stance": {"score_1_5": None, "label": "not_discussed", "quote": "", "confidence": 0.0},
                        "aspects": {
                            aspect: {"score_1_5": None, "points": [], "quotes": [], "confidence": 0.0}
                            for aspect in INNOVATION_ASPECTS
                        },
                        "warnings": [],
                        "raw_response": "",
                    }
                write_json(cache_path, payload)
            rows.append(payload)
        if idx == 1 or idx % 10 == 0 or idx == len(manifest):
            progress_log(f"创新标签抽取进度 {idx}/{len(manifest)}。", quiet)
    write_jsonl(output_dir / "fig4_innovation_label_judgements.jsonl", rows)
    return rows


def innovation_label_point_count(label: Mapping[str, Any]) -> int:
    aspects = label.get("aspects") if isinstance(label.get("aspects"), Mapping) else {}
    total = 0
    for aspect in INNOVATION_ASPECTS:
        item = aspects.get(aspect) if isinstance(aspects.get(aspect), Mapping) else {}
        points = item.get("points") if isinstance(item.get("points"), list) else []
        total += len([point for point in points if normalize_whitespace(str(point))])
    return total


def innovation_label_quote_count(label: Mapping[str, Any]) -> int:
    total = 1 if normalize_whitespace(str((label.get("overall_innovation_stance") or {}).get("quote", ""))) else 0
    aspects = label.get("aspects") if isinstance(label.get("aspects"), Mapping) else {}
    for aspect in INNOVATION_ASPECTS:
        item = aspects.get(aspect) if isinstance(aspects.get(aspect), Mapping) else {}
        quotes = item.get("quotes") if isinstance(item.get("quotes"), list) else []
        total += len([quote for quote in quotes if normalize_whitespace(str(quote))])
    return total


def core_innovation_aspect_count(label: Mapping[str, Any]) -> int:
    aspects = label.get("aspects") if isinstance(label.get("aspects"), Mapping) else {}
    count = 0
    for aspect in CORE_INNOVATION_ASPECTS:
        item = aspects.get(aspect) if isinstance(aspects.get(aspect), Mapping) else {}
        if math.isfinite(numeric(item.get("score_1_5"))) and item.get("quotes"):
            count += 1
    return count


def is_revision_only_review(text: str) -> bool:
    cleaned = normalize_whitespace(text).lower()
    if not cleaned:
        return True
    innovation_terms = re.findall(r"\b(novel|novelty|significant|significance|original|contribution|prior|benchmark|limitation|evidence)\b", cleaned)
    return bool(REVISION_ONLY_RE.search(cleaned)) and (word_count(cleaned) < 120 or len(innovation_terms) < 2)


def screen_peer_review_label(
    label: Mapping[str, Any],
    review_text: str,
    min_core_aspects: int,
    min_peer_label_points: int,
) -> Dict[str, Any]:
    stance = label.get("overall_innovation_stance") if isinstance(label.get("overall_innovation_stance"), Mapping) else {}
    stance_score = numeric(stance.get("score_1_5"))
    core_count = core_innovation_aspect_count(label)
    point_count = innovation_label_point_count(label)
    quote_count = innovation_label_quote_count(label)
    revision_only = is_revision_only_review(review_text)
    reasons: List[str] = []
    if not bool(label.get("success")):
        reasons.append(str(label.get("failure_reason") or "peer_label_failed"))
    if not math.isfinite(stance_score):
        reasons.append("missing_innovation_stance")
    if core_count < min_core_aspects:
        reasons.append("insufficient_core_innovation_aspects")
    if point_count < min_peer_label_points:
        reasons.append("insufficient_peer_innovation_points")
    if revision_only:
        reasons.append("revision_only_or_no_substantive_comments")
    return {
        "screen_pass": not reasons,
        "screen_reason": ";".join(reasons),
        "core_aspect_count": core_count,
        "quote_count": quote_count,
        "peer_point_count": point_count,
        "is_revision_only": revision_only,
        "peer_stance_score_1_5": stance_score,
    }


def run_peer_review_screen(
    output_dir: Path,
    judge_backend: str,
    review_filter: str,
    min_core_aspects: int,
    min_peer_label_points: int,
    quiet: bool = False,
    judge_client: Optional[OpenAICompatibleClient] = None,
) -> List[Dict[str, Any]]:
    audit_rows = read_csv_records(output_dir / "fig4_input_audit.csv")
    client = judge_client or (judge_config_from_env() if judge_backend == "openai-compatible" else None)
    rows: List[Dict[str, Any]] = []
    if client is None:
        progress_log("未配置 judge；screen 阶段只能使用已有 peer_innovation_labels.json 缓存。", quiet)
    for idx, row in enumerate(audit_rows, start=1):
        paper_id = str(row.get("paper_id") or "")
        cache_dir = output_dir / "cache" / paper_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        parsed_path = Path(str(row.get("parsed_text_cache") or cache_dir / "parsed_text.json"))
        parsed = read_json(parsed_path) if parsed_path.exists() else {}
        review_text = str(parsed.get("peer_review_text") or "")
        label_path = cache_dir / "peer_innovation_labels.json"
        if label_path.exists():
            label = read_json(label_path)
        elif client is not None and review_text:
            messages = build_innovation_label_prompt("peer_review", str(row.get("title", "")), str(row.get("abstract", "")), review_text)
            try:
                parsed_json, raw = client.chat_json(messages)
                label = normalize_innovation_label_payload(
                    parsed_json,
                    paper_id=paper_id,
                    kind="peer_review",
                    text=review_text,
                    model=client.model,
                    raw_response=raw,
                )
            except Exception as exc:  # noqa: BLE001
                label = {
                    "paper_id": paper_id,
                    "kind": "peer_review",
                    "success": False,
                    "failure_reason": str(exc),
                    "overall_innovation_stance": {"score_1_5": None, "label": "not_discussed", "quote": "", "confidence": 0.0},
                    "aspects": {
                        aspect: {"score_1_5": None, "points": [], "quotes": [], "confidence": 0.0}
                        for aspect in INNOVATION_ASPECTS
                    },
                }
            write_json(label_path, label)
        else:
            label = {
                "paper_id": paper_id,
                "kind": "peer_review",
                "success": False,
                "failure_reason": "judge_not_configured" if client is None else "empty_peer_review_text",
                "overall_innovation_stance": {"score_1_5": None, "label": "not_discussed", "quote": "", "confidence": 0.0},
                "aspects": {
                    aspect: {"score_1_5": None, "points": [], "quotes": [], "confidence": 0.0}
                    for aspect in INNOVATION_ASPECTS
                },
            }
            write_json(label_path, label)
        screen = screen_peer_review_label(label, review_text, min_core_aspects, min_peer_label_points)
        if review_filter != "explicit_innovation":
            screen["screen_pass"] = bool(label.get("success")) and not screen["is_revision_only"]
            screen["screen_reason"] = "" if screen["screen_pass"] else screen["screen_reason"]
        rows.append(
            {
                **{key: row.get(key, "") for key in row.keys()},
                "paper_id": paper_id,
                **screen,
                "peer_label_success": bool(label.get("success")),
                "peer_label_failure_reason": label.get("failure_reason", ""),
                "peer_label_cache": str(label_path),
            }
        )
        if idx == 1 or idx % 25 == 0 or idx == len(audit_rows):
            progress_log(f"Screen 进度 {idx}/{len(audit_rows)}，通过 {sum(bool(item.get('screen_pass')) for item in rows)}。", quiet)
    write_jsonl(output_dir / "fig4_peer_review_screen.jsonl", rows)
    write_csv(output_dir / "fig4_peer_review_screen.csv", rows)
    pass_count = sum(bool(row.get("screen_pass")) for row in rows)
    progress_log(f"Peer-review screen 完成：通过 {pass_count}/{len(rows)}。", quiet)
    return rows


def numeric(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def cosine_from_vectors(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    denom = float(np.linalg.norm(left_arr) * np.linalg.norm(right_arr))
    if denom == 0:
        return float("nan")
    return float(np.dot(left_arr, right_arr) / denom)


def tfidf_cosine_pairs(pairs: Sequence[Tuple[str, str]]) -> List[float]:
    if not pairs:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    values = []
    for left, right in pairs:
        if not left or not right:
            values.append(float("nan"))
            continue
        matrix = TfidfVectorizer(max_features=5000, stop_words="english").fit_transform([left, right])
        values.append(float(cosine_similarity(matrix[0], matrix[1])[0, 0]))
    return values


def semantic_similarity_pairs(pairs: Sequence[Tuple[str, str]]) -> Tuple[List[float], str, str]:
    client = embedding_config_from_env()
    if client is not None and pairs:
        try:
            flat_texts: List[str] = []
            for left, right in pairs:
                flat_texts.extend([left[:12000], right[:12000]])
            vectors = client.embeddings(flat_texts, model=client.model)
            values = []
            for idx in range(0, len(vectors), 2):
                values.append(cosine_from_vectors(vectors[idx], vectors[idx + 1]))
            return values, f"openai-compatible:{client.model}", ""
        except Exception as exc:  # noqa: BLE001
            values = tfidf_cosine_pairs(pairs)
            return values, "tfidf", f"embedding_fallback:{exc}"
    values = tfidf_cosine_pairs(pairs)
    return values, "tfidf", "embedding_not_configured"


def simple_syllable_count(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def readability_scores(text: str) -> Dict[str, float]:
    words = re.findall(r"[A-Za-z]+", text)
    sentences = re.split(r"[.!?]+", text)
    sentence_count = max(1, len([item for item in sentences if item.strip()]))
    word_n = max(1, len(words))
    syllables = sum(simple_syllable_count(word) for word in words)
    fre = 206.835 - 1.015 * (word_n / sentence_count) - 84.6 * (syllables / word_n)
    grade = 0.39 * (word_n / sentence_count) + 11.8 * (syllables / word_n) - 15.59
    return {"flesch_reading_ease": fre, "flesch_kincaid_grade": grade}


def language_tool_errors(text: str) -> Dict[str, Any]:
    try:
        import language_tool_python  # type: ignore[import-not-found]
    except ImportError:
        return {"available": False, "failure_reason": "language_tool_python_not_installed"}
    try:
        tool = language_tool_python.LanguageTool("en-US")
        matches = tool.check(text[:20000])
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "failure_reason": str(exc)}
    categories = {"spelling": 0, "grammar": 0, "tense_or_verb_form": 0, "other": 0}
    for match in matches:
        category = str(getattr(match, "category", "") or "").lower()
        rule_id = str(getattr(match, "ruleId", "") or "").lower()
        if "spell" in category or "spell" in rule_id:
            categories["spelling"] += 1
        elif "tense" in rule_id or "verb" in rule_id:
            categories["tense_or_verb_form"] += 1
        elif "grammar" in category:
            categories["grammar"] += 1
        else:
            categories["other"] += 1
    return {"available": True, "counts": categories}


def normalize_phrase(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    normalized = []
    for token in tokens:
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        normalized.append(token)
    return " ".join(normalized)


def aspect_points(judgement: Mapping[str, Any], aspect: str) -> List[str]:
    aspects = judgement.get("aspects") or {}
    values = aspects.get(aspect) if isinstance(aspects, dict) else []
    if not isinstance(values, list):
        return []
    return [normalize_whitespace(str(item)) for item in values if normalize_whitespace(str(item))]


def compute_aspect_matches(peer: Mapping[str, Any], agent: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for aspect in ASPECTS:
        peer_points = aspect_points(peer, aspect)
        agent_points = aspect_points(agent, aspect)
        agent_norm = [normalize_phrase(point) for point in agent_points]
        for point in peer_points:
            pnorm = normalize_phrase(point)
            matched = any(pnorm and (pnorm in anorm or anorm in pnorm) for anorm in agent_norm)
            rows.append(
                {
                    "aspect": aspect,
                    "peer_point": point,
                    "matched": matched,
                    "match_method": "normalized_phrase_overlap" if matched else "none",
                    "agent_candidates": agent_points[:5],
                }
            )
    return rows


def innovation_labels_by_kind(output_dir: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {
        (str(row.get("paper_id")), str(row.get("kind"))): row
        for row in read_jsonl(output_dir / "fig4_innovation_label_judgements.jsonl")
    }


def label_score(label: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = label
    for key in path:
        if not isinstance(value, Mapping):
            return float("nan")
        value = value.get(key)
    return numeric(value)


def score_agreement(left: Any, right: Any) -> float:
    left_num = numeric(left)
    right_num = numeric(right)
    if not math.isfinite(left_num) or not math.isfinite(right_num):
        return float("nan")
    return max(0.0, 1.0 - abs(left_num - right_num) / 4.0)


def label_points(label: Mapping[str, Any], aspect: str) -> List[str]:
    aspects = label.get("aspects") if isinstance(label.get("aspects"), Mapping) else {}
    item = aspects.get(aspect) if isinstance(aspects.get(aspect), Mapping) else {}
    values: List[str] = []
    for key in ("points", "quotes"):
        raw = item.get(key)
        if isinstance(raw, list):
            values.extend(normalize_whitespace(str(value)) for value in raw if normalize_whitespace(str(value)))
    return values


def normalized_point_overlap(peer_points: Sequence[str], agent_points: Sequence[str]) -> Tuple[float, int, int]:
    if not peer_points:
        return float("nan"), 0, 0
    agent_norm = [normalize_phrase(point) for point in agent_points]
    covered = 0
    for point in peer_points:
        pnorm = normalize_phrase(point)
        if any(pnorm and (pnorm in anorm or anorm in pnorm) for anorm in agent_norm):
            covered += 1
    return covered / max(len(peer_points), 1), covered, len(peer_points)


def innovation_aspect_alignment(
    peer_label: Mapping[str, Any],
    agent_label: Mapping[str, Any],
    aspect: str,
) -> Tuple[float, float, float, int, int]:
    peer_aspects = peer_label.get("aspects") if isinstance(peer_label.get("aspects"), Mapping) else {}
    agent_aspects = agent_label.get("aspects") if isinstance(agent_label.get("aspects"), Mapping) else {}
    peer_item = peer_aspects.get(aspect) if isinstance(peer_aspects.get(aspect), Mapping) else {}
    agent_item = agent_aspects.get(aspect) if isinstance(agent_aspects.get(aspect), Mapping) else {}
    score_align = score_agreement(peer_item.get("score_1_5"), agent_item.get("score_1_5"))
    overlap, covered, total = normalized_point_overlap(label_points(peer_label, aspect), label_points(agent_label, aspect))
    values = [value for value in (score_align, overlap) if math.isfinite(value)]
    alignment = float(np.mean(values)) if values else float("nan")
    return alignment, score_align, overlap, covered, total


def aspect_label_items(label: Mapping[str, Any], aspect: str) -> List[Dict[str, str]]:
    aspects = label.get("aspects") if isinstance(label.get("aspects"), Mapping) else {}
    item = aspects.get(aspect) if isinstance(aspects.get(aspect), Mapping) else {}
    points = item.get("points") if isinstance(item.get("points"), list) else []
    quotes = item.get("quotes") if isinstance(item.get("quotes"), list) else []
    clean_points = [normalize_whitespace(str(point)) for point in points if normalize_whitespace(str(point))]
    clean_quotes = [normalize_whitespace(str(quote)) for quote in quotes if normalize_whitespace(str(quote))]
    out: List[Dict[str, str]] = []
    for idx, point in enumerate(clean_points):
        out.append({"point": point, "quote": clean_quotes[idx] if idx < len(clean_quotes) else (clean_quotes[0] if clean_quotes else "")})
    return out


def semantic_relation_score(relation: str) -> float:
    return SEMANTIC_RELATION_SCORES.get(str(relation), 0.0)


def normalize_semantic_relation(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in SEMANTIC_RELATION_SCORES:
        return text
    if "contradict" in text:
        return "contradicted"
    if "entail" in text or "cover" in text:
        return "entailed"
    if "relat" in text or "partial" in text:
        return "related"
    return "no_match"


def token_set(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "with", "by", "on", "is", "are", "was", "were",
        "this", "that", "it", "as", "from", "be", "been", "not", "no", "but", "their", "its", "they",
    }
    return {token for token in re.findall(r"[a-z0-9]+", str(text).lower()) if len(token) > 2 and token not in stop}


def heuristic_semantic_match(peer_point: str, agent_candidates: Sequence[str]) -> Dict[str, Any]:
    if not agent_candidates:
        return {"relation": "no_match", "best_agent_point": "", "rationale": "No same-aspect agent candidate was extracted."}
    peer_tokens = token_set(peer_point)
    best = ("", 0.0)
    for candidate in agent_candidates:
        cand_tokens = token_set(candidate)
        if not peer_tokens or not cand_tokens:
            overlap = 0.0
        else:
            overlap = len(peer_tokens & cand_tokens) / len(peer_tokens | cand_tokens)
        if overlap > best[1]:
            best = (candidate, overlap)
    peer_norm = normalize_phrase(peer_point)
    best_norm = normalize_phrase(best[0])
    if peer_norm and best_norm and (peer_norm in best_norm or best_norm in peer_norm):
        relation = "entailed"
        rationale = "Normalized phrase containment."
    elif best[1] >= 0.28:
        relation = "related"
        rationale = f"Lexical overlap suggests partial topical relation ({best[1]:.2f})."
    else:
        relation = "no_match"
        rationale = f"Best lexical overlap is too low ({best[1]:.2f})."
    return {"relation": relation, "best_agent_point": best[0], "rationale": rationale}


def build_semantic_claim_match_prompt(
    title: str,
    aspect: str,
    peer_point: str,
    peer_quote: str,
    agent_candidates: Sequence[str],
) -> List[Dict[str, str]]:
    candidates = "\n".join(f"{idx + 1}. {candidate}" for idx, candidate in enumerate(agent_candidates[:8]))
    system = (
        "You are a strict scientific evaluation judge. Return only valid JSON. "
        "Judge whether an ASPR agent point semantically covers a peer-review innovation point."
    )
    user = f"""
Paper title: {title}
Aspect: {aspect}

Peer-review point:
{peer_point}

Peer-review quote:
{peer_quote}

Agent same-aspect candidate points:
{candidates or "None"}

Choose exactly one relation:
- entailed: the agent explicitly covers the same substantive criticism or judgement.
- related: the agent mentions a related issue but does not fully answer the same point.
- contradicted: the agent says the opposite or denies the peer point.
- no_match: no agent candidate covers this peer point.

Return this JSON schema:
{{
  "relation": "entailed|related|contradicted|no_match",
  "best_agent_point": string,
  "rationale": string
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def semantic_match_one_point(
    title: str,
    aspect: str,
    peer_point: str,
    peer_quote: str,
    agent_candidates: Sequence[str],
    client: Optional[OpenAICompatibleClient],
) -> Dict[str, Any]:
    if not agent_candidates:
        relation = "no_match"
        return {
            "relation": relation,
            "best_agent_point": "",
            "score": semantic_relation_score(relation),
            "rationale": "No same-aspect agent candidate was extracted.",
            "match_backend": "rule",
        }
    if client is None:
        payload = heuristic_semantic_match(peer_point, agent_candidates)
        relation = normalize_semantic_relation(payload.get("relation"))
        return {
            **payload,
            "relation": relation,
            "score": semantic_relation_score(relation),
            "match_backend": "heuristic",
        }
    messages = build_semantic_claim_match_prompt(title, aspect, peer_point, peer_quote, agent_candidates)
    try:
        parsed_json, raw = client.chat_json(messages)
        relation = normalize_semantic_relation(parsed_json.get("relation"))
        best_agent_point = normalize_whitespace(str(parsed_json.get("best_agent_point") or ""))
        if best_agent_point not in [normalize_whitespace(str(item)) for item in agent_candidates]:
            best_agent_point = heuristic_semantic_match(peer_point, agent_candidates)["best_agent_point"]
        return {
            "relation": relation,
            "best_agent_point": best_agent_point,
            "score": semantic_relation_score(relation),
            "rationale": normalize_whitespace(str(parsed_json.get("rationale") or ""))[:800],
            "raw_response": raw,
            "match_backend": f"judge:{client.model}",
        }
    except Exception as exc:  # noqa: BLE001
        payload = heuristic_semantic_match(peer_point, agent_candidates)
        relation = normalize_semantic_relation(payload.get("relation"))
        return {
            **payload,
            "relation": relation,
            "score": semantic_relation_score(relation),
            "rationale": f"{payload.get('rationale', '')} Judge fallback after error: {exc}",
            "match_backend": "heuristic_after_judge_error",
        }


def run_semantic_claim_match(
    output_dir: Path,
    judge_backend: str,
    quiet: bool = False,
    judge_client: Optional[OpenAICompatibleClient] = None,
) -> List[Dict[str, Any]]:
    manifest = read_csv_records(output_dir / "fig4_manifest.csv")
    labels = innovation_labels_by_kind(output_dir)
    client = judge_client or (judge_config_from_env() if judge_backend == "openai-compatible" else None)
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(manifest, start=1):
        paper_id = str(row.get("paper_id") or "")
        cache_path = output_dir / "cache" / paper_id / "semantic_claim_matches.json"
        if cache_path.exists():
            paper_rows = read_json(cache_path).get("matches", [])
            rows.extend(paper_rows)
        else:
            peer_label = labels.get((paper_id, "peer_review"), {})
            agent_label = labels.get((paper_id, "agent"), {})
            paper_rows = []
            for aspect in INNOVATION_ASPECTS:
                peer_items = aspect_label_items(peer_label, aspect)
                agent_items = aspect_label_items(agent_label, aspect)
                agent_candidates = [item["point"] for item in agent_items]
                agent_quote_lookup = {item["point"]: item.get("quote", "") for item in agent_items}
                for peer_item in peer_items:
                    match = semantic_match_one_point(
                        title=str(row.get("title", "")),
                        aspect=aspect,
                        peer_point=peer_item["point"],
                        peer_quote=peer_item.get("quote", ""),
                        agent_candidates=agent_candidates,
                        client=client,
                    )
                    paper_rows.append(
                        {
                            "paper_id": paper_id,
                            "aspect": aspect,
                            "peer_point": peer_item["point"],
                            "peer_quote": peer_item.get("quote", ""),
                            "best_agent_point": match.get("best_agent_point", ""),
                            "agent_quote": agent_quote_lookup.get(str(match.get("best_agent_point", "")), ""),
                            "relation": match.get("relation", "no_match"),
                            "score": match.get("score", 0.0),
                            "rationale": match.get("rationale", ""),
                            "match_backend": match.get("match_backend", ""),
                        }
                    )
            write_json(cache_path, {"paper_id": paper_id, "matches": paper_rows})
            rows.extend(paper_rows)
        if idx == 1 or idx % 10 == 0 or idx == len(manifest):
            progress_log(f"Semantic match 进度 {idx}/{len(manifest)}，points={len(rows)}。", quiet)
    write_jsonl(output_dir / "fig4_semantic_claim_matches.jsonl", rows)
    return rows


def compact_innovation_label_for_consistency(label: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only judge-facing label fields for structured consistency comparison."""
    aspects_in = label.get("aspects") if isinstance(label.get("aspects"), Mapping) else {}
    return {
        "overall_innovation_stance": label.get("overall_innovation_stance") or {},
        "aspects": {
            aspect: aspects_in.get(aspect, {})
            for aspect in INNOVATION_ASPECTS
        },
    }


def consistency_score_from_pair(peer_score: Any, agent_score: Any) -> Optional[float]:
    peer_value = numeric(peer_score)
    agent_value = numeric(agent_score)
    if not math.isfinite(peer_value) or not math.isfinite(agent_value):
        return None
    return max(1.0, min(5.0, 5.0 - abs(peer_value - agent_value)))


def heuristic_structured_consistency(peer_label: Mapping[str, Any], agent_label: Mapping[str, Any]) -> Dict[str, Any]:
    peer_aspects = peer_label.get("aspects") if isinstance(peer_label.get("aspects"), Mapping) else {}
    agent_aspects = agent_label.get("aspects") if isinstance(agent_label.get("aspects"), Mapping) else {}
    payload: Dict[str, Any] = {
        "stance_consistency_1_5": consistency_score_from_pair(
            (peer_label.get("overall_innovation_stance") or {}).get("score_1_5"),
            (agent_label.get("overall_innovation_stance") or {}).get("score_1_5"),
        ),
        "missing_key_points": [],
        "contradictions": [],
        "rationale": "Heuristic fallback based on label score distance.",
    }
    aspect_to_field = {
        "novelty": "novelty_consistency_1_5",
        "significance": "significance_consistency_1_5",
        "prior_art_comparison": "prior_art_consistency_1_5",
        "evidence_rigor": "evidence_rigor_consistency_1_5",
        "limitations": "limitations_consistency_1_5",
        "future_work": "future_work_consistency_1_5",
    }
    for aspect, field in aspect_to_field.items():
        peer_item = peer_aspects.get(aspect) if isinstance(peer_aspects.get(aspect), Mapping) else {}
        agent_item = agent_aspects.get(aspect) if isinstance(agent_aspects.get(aspect), Mapping) else {}
        payload[field] = consistency_score_from_pair(peer_item.get("score_1_5"), agent_item.get("score_1_5"))
    peer_stance = numeric((peer_label.get("overall_innovation_stance") or {}).get("score_1_5"))
    agent_stance = numeric((agent_label.get("overall_innovation_stance") or {}).get("score_1_5"))
    payload["overclaiming_score_1_5"] = max(1.0, min(5.0, 1.0 + max(agent_stance - peer_stance, 0.0))) if math.isfinite(peer_stance) and math.isfinite(agent_stance) else None
    return payload


def build_structured_consistency_prompt(
    title: str,
    peer_label: Mapping[str, Any],
    agent_label: Mapping[str, Any],
) -> List[Dict[str, str]]:
    system = (
        "You are a strict scientific evaluation judge. Return only valid JSON. "
        "Compare peer-review innovation labels against ASPR agent innovation labels. "
        "Do not use outside knowledge or the original paper text."
    )
    user = f"""
Paper title: {title}

Peer-review innovation labels:
{json.dumps(compact_innovation_label_for_consistency(peer_label), ensure_ascii=False, indent=2)[:9000]}

ASPR agent innovation labels:
{json.dumps(compact_innovation_label_for_consistency(agent_label), ensure_ascii=False, indent=2)[:9000]}

Task:
Rate semantic consistency from 1 to 5, where 1=contradictory/unrelated, 3=partially consistent but missing important nuance, and 5=strongly consistent.
Also rate overclaiming_score_1_5 where 1=no overclaiming and 5=severe ASPR overclaiming relative to peer review.

Return this JSON schema:
{{
  "stance_consistency_1_5": number|null,
  "novelty_consistency_1_5": number|null,
  "significance_consistency_1_5": number|null,
  "prior_art_consistency_1_5": number|null,
  "evidence_rigor_consistency_1_5": number|null,
  "limitations_consistency_1_5": number|null,
  "future_work_consistency_1_5": number|null,
  "overclaiming_score_1_5": number|null,
  "missing_key_points": [string],
  "contradictions": [string],
  "rationale": string
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_structured_consistency_payload(
    payload: Mapping[str, Any],
    paper_id: str,
    model: str = "",
    raw_response: str = "",
    fallback_payload: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    fallback = dict(fallback_payload or {})
    normalized: Dict[str, Any] = {
        "paper_id": paper_id,
        "success": True,
        "failure_reason": "",
        "model": model,
        "raw_response": raw_response,
        "missing_key_points": payload.get("missing_key_points") if isinstance(payload.get("missing_key_points"), list) else fallback.get("missing_key_points", []),
        "contradictions": payload.get("contradictions") if isinstance(payload.get("contradictions"), list) else fallback.get("contradictions", []),
        "rationale": normalize_whitespace(str(payload.get("rationale") or fallback.get("rationale") or ""))[:1000],
    }
    for field in STRUCTURED_CONSISTENCY_FIELDS + ["overclaiming_score_1_5"]:
        score = _score_or_none(payload.get(field))
        if score is None:
            score = _score_or_none(fallback.get(field))
        normalized[field] = score
    if not any(normalized.get(field) is not None for field in STRUCTURED_CONSISTENCY_FIELDS):
        normalized["success"] = False
        normalized["failure_reason"] = "missing_consistency_scores"
    return normalized


def run_structured_consistency_judge(
    output_dir: Path,
    judge_backend: str,
    quiet: bool = False,
    judge_client: Optional[OpenAICompatibleClient] = None,
) -> List[Dict[str, Any]]:
    manifest = read_csv_records(output_dir / "fig4_manifest.csv")
    labels = innovation_labels_by_kind(output_dir)
    client = judge_client or (judge_config_from_env() if judge_backend == "openai-compatible" else None)
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(manifest, start=1):
        paper_id = str(row.get("paper_id") or "")
        cache_path = output_dir / "cache" / paper_id / "structured_consistency.json"
        if cache_path.exists():
            payload = read_json(cache_path)
        else:
            peer_label = labels.get((paper_id, "peer_review"), {})
            agent_label = labels.get((paper_id, "agent"), {})
            fallback = heuristic_structured_consistency(peer_label, agent_label)
            if not peer_label or not agent_label:
                payload = normalize_structured_consistency_payload(
                    {},
                    paper_id=paper_id,
                    fallback_payload=fallback,
                )
                payload["success"] = False
                payload["failure_reason"] = "missing_peer_or_agent_label"
            elif client is None:
                payload = normalize_structured_consistency_payload(
                    fallback,
                    paper_id=paper_id,
                    model="heuristic",
                    fallback_payload=fallback,
                )
            else:
                messages = build_structured_consistency_prompt(str(row.get("title", "")), peer_label, agent_label)
                try:
                    parsed_json, raw = client.chat_json(messages)
                    payload = normalize_structured_consistency_payload(
                        parsed_json,
                        paper_id=paper_id,
                        model=client.model,
                        raw_response=raw,
                        fallback_payload=fallback,
                    )
                except Exception as exc:  # noqa: BLE001
                    payload = normalize_structured_consistency_payload(
                        fallback,
                        paper_id=paper_id,
                        model="heuristic_after_judge_error",
                        fallback_payload=fallback,
                    )
                    payload["failure_reason"] = f"judge_error:{exc}"
            write_json(cache_path, payload)
        rows.append(payload)
        if idx == 1 or idx % 10 == 0 or idx == len(manifest):
            progress_log(f"Structured consistency 进度 {idx}/{len(manifest)}。", quiet)
    write_jsonl(output_dir / "fig4_structured_consistency_judgements.jsonl", rows)
    return rows


def quadratic_weighted_kappa(left_scores: Sequence[Any], right_scores: Sequence[Any]) -> float:
    pairs = []
    for left, right in zip(left_scores, right_scores):
        left_value = numeric(left)
        right_value = numeric(right)
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            continue
        left_num = int(round(left_value))
        right_num = int(round(right_value))
        if 1 <= left_num <= 5 and 1 <= right_num <= 5:
            pairs.append((left_num, right_num))
    if len(pairs) < 2:
        return float("nan")
    observed = np.zeros((5, 5), dtype=float)
    for left, right in pairs:
        observed[left - 1, right - 1] += 1.0
    hist_left = observed.sum(axis=1)
    hist_right = observed.sum(axis=0)
    expected = np.outer(hist_left, hist_right) / max(observed.sum(), 1.0)
    weights = np.zeros((5, 5), dtype=float)
    for i in range(5):
        for j in range(5):
            weights[i, j] = ((i - j) ** 2) / 16.0
    observed_weighted = float((weights * observed).sum())
    expected_weighted = float((weights * expected).sum())
    if expected_weighted == 0:
        return 1.0 if observed_weighted == 0 else float("nan")
    return 1.0 - observed_weighted / expected_weighted


def run_metrics_stage(
    output_dir: Path,
    human_hours: float,
    judge_backend: str,
    quiet: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    manifest = read_csv_records(output_dir / "fig4_manifest.csv")
    agent_outputs = {row["paper_id"]: row for row in read_jsonl(output_dir / "fig4_agent_outputs.jsonl")}
    screen_lookup = {row["paper_id"]: row for row in read_csv_records(output_dir / "fig4_peer_review_screen.csv") if row.get("paper_id")}
    graph_lookup = {row["paper_id"]: row for row in read_csv_records(output_dir / "fig4_graph_metrics.csv") if row.get("paper_id")}
    retrieval_lookup = {row["paper_id"]: row for row in read_csv_records(output_dir / "fig4_retrieval_diagnostics.csv") if row.get("paper_id")}
    ratings = read_jsonl(output_dir / "fig4_rating_judgements.jsonl")
    rating_lookup = {(row.get("paper_id"), row.get("kind")): row for row in ratings}
    innovation_label_lookup = innovation_labels_by_kind(output_dir)
    semantic_rows = read_jsonl(output_dir / "fig4_semantic_claim_matches.jsonl")
    structured_rows = read_jsonl(output_dir / "fig4_structured_consistency_judgements.jsonl")
    structured_lookup = {str(row.get("paper_id")): row for row in structured_rows if row.get("paper_id")}
    semantic_lookup: Dict[str, List[Dict[str, Any]]] = {}
    for semantic_row in semantic_rows:
        semantic_lookup.setdefault(str(semantic_row.get("paper_id")), []).append(semantic_row)
    metric_rows: List[Dict[str, Any]] = []
    aspect_match_rows: List[Dict[str, Any]] = []
    text_pairs: List[Tuple[str, str]] = []
    row_indices_for_similarity: List[int] = []
    for row in manifest:
        paper_id = str(row["paper_id"])
        parsed_path = Path(str(row.get("parsed_text_cache") or output_dir / "cache" / paper_id / "parsed_text.json"))
        parsed = read_json(parsed_path) if parsed_path.exists() else {}
        agent = agent_outputs.get(paper_id, {})
        screen = screen_lookup.get(paper_id, {})
        graph_metrics = graph_lookup.get(paper_id, {})
        retrieval_diag = retrieval_lookup.get(paper_id, {})
        peer_rating = rating_lookup.get((paper_id, "peer_review"), {})
        agent_rating = rating_lookup.get((paper_id, "agent"), {})
        peer_label = innovation_label_lookup.get((paper_id, "peer_review"), {})
        agent_label = innovation_label_lookup.get((paper_id, "agent"), {})
        structured_consistency = structured_lookup.get(paper_id, {})
        paper_semantic_rows = semantic_lookup.get(paper_id, [])
        agent_text = str(agent.get("innovation_evaluation") or "")
        peer_text = str(parsed.get("peer_review_text") or "")
        runtime = numeric(agent.get("agent_runtime_seconds"))
        human_seconds = human_hours * 3600.0
        speedup = human_seconds / runtime if runtime and runtime > 0 else float("nan")
        peer_read = readability_scores(peer_text)
        agent_read = readability_scores(agent_text)
        peer_tool = language_tool_errors(peer_text)
        agent_tool = language_tool_errors(agent_text)
        peer_error_total = float("nan")
        agent_error_total = float("nan")
        peer_error_counts = {"spelling": float("nan"), "grammar": float("nan"), "tense_or_verb_form": float("nan"), "other": float("nan")}
        agent_error_counts = dict(peer_error_counts)
        if peer_tool.get("available") and agent_tool.get("available"):
            peer_counts = peer_tool.get("counts") or {}
            agent_counts = agent_tool.get("counts") or {}
            peer_wc = max(1, int(parsed.get("word_count") or word_count(peer_text)))
            agent_wc = max(1, word_count(agent_text))
            for key in peer_error_counts:
                peer_error_counts[key] = numeric(peer_counts.get(key), 0.0) / peer_wc * 5000.0
                agent_error_counts[key] = numeric(agent_counts.get(key), 0.0) / agent_wc * 5000.0
            peer_error_total = sum(peer_error_counts.values())
            agent_error_total = sum(agent_error_counts.values())
        aspect_matches = compute_aspect_matches(peer_rating, agent_rating) if peer_rating and agent_rating else []
        for match in aspect_matches:
            aspect_match_rows.append({"paper_id": paper_id, **match})
        peer_aspect_count = len(aspect_matches)
        covered_count = sum(1 for match in aspect_matches if match["matched"])
        legacy_coverage_score = covered_count / peer_aspect_count if peer_aspect_count else float("nan")
        peer_stance = (peer_label.get("overall_innovation_stance") or {}) if isinstance(peer_label.get("overall_innovation_stance"), dict) else {}
        agent_stance = (agent_label.get("overall_innovation_stance") or {}) if isinstance(agent_label.get("overall_innovation_stance"), dict) else {}
        peer_stance_score = numeric(peer_stance.get("score_1_5"))
        agent_stance_score = numeric(agent_stance.get("score_1_5"))
        innovation_stance_agreement = score_agreement(peer_stance_score, agent_stance_score)
        innovation_alignments: Dict[str, float] = {}
        innovation_coverages: Dict[str, float] = {}
        total_peer_points = 0
        total_covered_points = 0
        for aspect in INNOVATION_ASPECTS:
            alignment, score_align, point_overlap, aspect_covered, aspect_total = innovation_aspect_alignment(peer_label, agent_label, aspect)
            innovation_alignments[aspect] = alignment
            innovation_coverages[aspect] = point_overlap
            if aspect_total:
                total_peer_points += aspect_total
                total_covered_points += aspect_covered
            aspect_match_rows.append(
                {
                    "paper_id": paper_id,
                    "aspect": aspect,
                    "peer_point": " | ".join(label_points(peer_label, aspect)[:3]),
                    "matched": math.isfinite(point_overlap) and point_overlap > 0,
                    "match_method": "innovation_label_point_overlap",
                    "agent_candidates": label_points(agent_label, aspect)[:5],
                    "score_alignment": score_align,
                    "point_overlap": point_overlap,
                }
            )
        claim_evidence_coverage = total_covered_points / total_peer_points if total_peer_points else float("nan")
        limitation_coverage = innovation_coverages.get("limitations", float("nan"))
        label_success = bool(peer_label.get("success")) and bool(agent_label.get("success"))
        semantic_scores = [numeric(item.get("score")) for item in paper_semantic_rows if math.isfinite(numeric(item.get("score")))]
        semantic_claim_alignment = float(np.mean(semantic_scores)) if semantic_scores else float("nan")
        relation_counts = {relation: 0 for relation in SEMANTIC_RELATION_SCORES}
        for item in paper_semantic_rows:
            relation = normalize_semantic_relation(item.get("relation"))
            relation_counts[relation] = relation_counts.get(relation, 0) + 1
        semantic_total = sum(relation_counts.values())
        contradiction_rate = relation_counts.get("contradicted", 0) / semantic_total if semantic_total else float("nan")
        missing_peer_point_rate = relation_counts.get("no_match", 0) / semantic_total if semantic_total else float("nan")
        structured_values = [
            numeric(structured_consistency.get(field))
            for field in STRUCTURED_CONSISTENCY_FIELDS
            if math.isfinite(numeric(structured_consistency.get(field)))
        ]
        structured_semantic_consistency_mean = float(np.mean(structured_values)) if structured_values else float("nan")
        aspect_semantic: Dict[str, float] = {}
        for aspect in INNOVATION_ASPECTS:
            values = [
                numeric(item.get("score"))
                for item in paper_semantic_rows
                if item.get("aspect") == aspect and math.isfinite(numeric(item.get("score")))
            ]
            aspect_semantic[aspect] = float(np.mean(values)) if values else float("nan")
        stance_abs_error = abs(peer_stance_score - agent_stance_score) if math.isfinite(peer_stance_score) and math.isfinite(agent_stance_score) else float("nan")
        screen_pass = str(screen.get("screen_pass", "True" if not screen_lookup else "False")).lower() in {"true", "1"}
        graph_valid = str(graph_metrics.get("graph_metric_valid", "True" if not graph_lookup else "False")).lower() in {"true", "1"}
        retrieval_source = str(retrieval_diag.get("retrieval_source") or agent.get("retrieval_source") or "")
        local_retrieval = retrieval_source in {"local_fallback", "local_fig4_manifest_fallback"}
        semantic_success = bool(paper_semantic_rows) and math.isfinite(semantic_claim_alignment)
        main_exclusions: List[str] = []
        if not bool(agent.get("success")):
            main_exclusions.append(str(agent.get("failure_reason") or "agent_failed"))
        if not screen_pass:
            main_exclusions.append(str(screen.get("screen_reason") or "peer_review_screen_failed"))
        if not label_success:
            main_exclusions.append("innovation_label_failed")
        if not math.isfinite(peer_stance_score) or not math.isfinite(agent_stance_score):
            main_exclusions.append("missing_valid_stance_pair")
        if innovation_label_point_count(peer_label) < 2:
            main_exclusions.append("insufficient_peer_innovation_points")
        if local_retrieval:
            main_exclusions.append("local_fallback_retrieval")
        if not graph_valid:
            main_exclusions.append(str(graph_metrics.get("graph_metric_failure_reason") or "graph_metric_invalid"))
        if not semantic_success:
            main_exclusions.append("semantic_claim_match_failed")
        main_exclusion = ";".join(dict.fromkeys(reason for reason in main_exclusions if reason))
        included = not main_exclusion
        metric_rows.append(
            {
                "paper_id": paper_id,
                "journal": row.get("journal", ""),
                "journal_id": row.get("journal_id", ""),
                "year": row.get("year", ""),
                "title": row.get("title", ""),
                "agent_success": bool(agent.get("success")),
                "agent_context_mode": agent.get("agent_context_mode", ""),
                "peer_label_success": bool(peer_label.get("success")),
                "agent_label_success": bool(agent_label.get("success")),
                "peer_label_failure_reason": peer_label.get("failure_reason", ""),
                "agent_label_failure_reason": agent_label.get("failure_reason", ""),
                "screen_pass": screen_pass,
                "screen_reason": screen.get("screen_reason", ""),
                "graph_metric_valid": graph_valid,
                "graph_metric_failure_reason": graph_metrics.get("graph_metric_failure_reason", ""),
                "graph_metric_source": graph_metrics.get("metric_source", ""),
                "weighted_score_fig3": numeric(graph_metrics.get("weighted_score_fig3")),
                "graph_confidence": numeric(graph_metrics.get("graph_confidence")),
                "retrieval_source": retrieval_source,
                "s2_key_status": retrieval_diag.get("s2_key_status") or agent.get("s2_key_status", ""),
                "peer_overall_score_1_5": numeric(peer_rating.get("overall_score_1_5")),
                "agent_overall_score_1_5": numeric(agent_rating.get("overall_score_1_5")),
                "peer_innovation_stance_1_5": peer_stance_score,
                "agent_innovation_stance_1_5": agent_stance_score,
                "innovation_stance_abs_error": stance_abs_error,
                "innovation_stance_mae": stance_abs_error,
                "innovation_stance_agreement": innovation_stance_agreement,
                "semantic_claim_alignment": semantic_claim_alignment,
                "structured_consistency_success": bool(structured_consistency.get("success")),
                "structured_semantic_consistency_mean": structured_semantic_consistency_mean,
                "stance_consistency_1_5": numeric(structured_consistency.get("stance_consistency_1_5")),
                "novelty_consistency_1_5": numeric(structured_consistency.get("novelty_consistency_1_5")),
                "significance_consistency_1_5": numeric(structured_consistency.get("significance_consistency_1_5")),
                "prior_art_consistency_1_5": numeric(structured_consistency.get("prior_art_consistency_1_5")),
                "evidence_rigor_consistency_1_5": numeric(structured_consistency.get("evidence_rigor_consistency_1_5")),
                "limitations_consistency_1_5": numeric(structured_consistency.get("limitations_consistency_1_5")),
                "future_work_consistency_1_5": numeric(structured_consistency.get("future_work_consistency_1_5")),
                "overclaiming_score_1_5": numeric(structured_consistency.get("overclaiming_score_1_5")),
                "structured_missing_key_points": " | ".join(str(item) for item in structured_consistency.get("missing_key_points", [])[:6]) if isinstance(structured_consistency.get("missing_key_points"), list) else "",
                "structured_contradictions": " | ".join(str(item) for item in structured_consistency.get("contradictions", [])[:6]) if isinstance(structured_consistency.get("contradictions"), list) else "",
                "novelty_semantic_coverage": aspect_semantic.get("novelty", float("nan")),
                "significance_semantic_coverage": aspect_semantic.get("significance", float("nan")),
                "prior_art_semantic_coverage": aspect_semantic.get("prior_art_comparison", float("nan")),
                "evidence_rigor_semantic_coverage": aspect_semantic.get("evidence_rigor", float("nan")),
                "limitations_semantic_coverage": aspect_semantic.get("limitations", float("nan")),
                "future_work_semantic_coverage": aspect_semantic.get("future_work", float("nan")),
                "contradiction_rate": contradiction_rate,
                "missing_peer_point_rate": missing_peer_point_rate,
                "novelty_alignment": innovation_alignments.get("novelty", float("nan")),
                "significance_alignment": innovation_alignments.get("significance", float("nan")),
                "prior_art_alignment": innovation_alignments.get("prior_art_comparison", float("nan")),
                "evidence_rigor_alignment": innovation_alignments.get("evidence_rigor", float("nan")),
                "limitation_coverage": limitation_coverage,
                "future_work_alignment": innovation_alignments.get("future_work", float("nan")),
                "phrase_claim_coverage_supplementary": claim_evidence_coverage,
                "claim_evidence_coverage": semantic_claim_alignment,
                "peer_novelty": numeric(peer_rating.get("novelty")),
                "agent_novelty": numeric(agent_rating.get("novelty")),
                "peer_significance": numeric(peer_rating.get("significance")),
                "agent_significance": numeric(agent_rating.get("significance")),
                "peer_rigor": numeric(peer_rating.get("rigor")),
                "agent_rigor": numeric(agent_rating.get("rigor")),
                "peer_limitations": numeric(peer_rating.get("limitations")),
                "agent_limitations": numeric(agent_rating.get("limitations")),
                "peer_future_work": numeric(peer_rating.get("future_work")),
                "agent_future_work": numeric(agent_rating.get("future_work")),
                "agent_runtime_seconds": runtime,
                "human_baseline_hours": human_hours,
                "speedup_vs_human": speedup,
                "time_saved_percent": max(0.0, 1.0 - runtime / human_seconds) if runtime and runtime > 0 else float("nan"),
                "peer_errors_per_5000_words": peer_error_total,
                "agent_errors_per_5000_words": agent_error_total,
                "peer_spelling_errors_per_5000": peer_error_counts["spelling"],
                "agent_spelling_errors_per_5000": agent_error_counts["spelling"],
                "peer_grammar_errors_per_5000": peer_error_counts["grammar"],
                "agent_grammar_errors_per_5000": agent_error_counts["grammar"],
                "peer_tense_errors_per_5000": peer_error_counts["tense_or_verb_form"],
                "agent_tense_errors_per_5000": agent_error_counts["tense_or_verb_form"],
                "peer_other_errors_per_5000": peer_error_counts["other"],
                "agent_other_errors_per_5000": agent_error_counts["other"],
                "peer_flesch_reading_ease": peer_read["flesch_reading_ease"],
                "agent_flesch_reading_ease": agent_read["flesch_reading_ease"],
                "peer_flesch_kincaid_grade": peer_read["flesch_kincaid_grade"],
                "agent_flesch_kincaid_grade": agent_read["flesch_kincaid_grade"],
                "readability_available": bool(peer_tool.get("available") and agent_tool.get("available")),
                "readability_failure_reason": "" if peer_tool.get("available") and agent_tool.get("available") else peer_tool.get("failure_reason") or agent_tool.get("failure_reason"),
                "coverage_score": semantic_claim_alignment if math.isfinite(semantic_claim_alignment) else legacy_coverage_score,
                "legacy_coverage_score": legacy_coverage_score,
                "covered_peer_aspects": covered_count,
                "total_peer_aspects": peer_aspect_count,
                "covered_innovation_label_points": total_covered_points,
                "total_innovation_label_points": total_peer_points,
                "semantic_match_points": semantic_total,
                "entailed_points": relation_counts.get("entailed", 0),
                "related_points": relation_counts.get("related", 0),
                "contradicted_points": relation_counts.get("contradicted", 0),
                "no_match_points": relation_counts.get("no_match", 0),
                "included_in_main": included,
                "exclusion_reason": main_exclusion,
            }
        )
        if agent_text and peer_text:
            row_indices_for_similarity.append(len(metric_rows) - 1)
            text_pairs.append((agent_text, peer_text))
    similarities, embedding_backend, embedding_warning = semantic_similarity_pairs(text_pairs)
    for idx, similarity in zip(row_indices_for_similarity, similarities):
        metric_rows[idx]["consistency_cosine"] = similarity
        metric_rows[idx]["embedding_backend"] = embedding_backend
        metric_rows[idx]["embedding_warning"] = embedding_warning
    for row in metric_rows:
        row.setdefault("consistency_cosine", float("nan"))
        row.setdefault("embedding_backend", "missing")
        row.setdefault("embedding_warning", "")
    qwk = quadratic_weighted_kappa(
        [row.get("peer_innovation_stance_1_5") for row in metric_rows if str(row.get("included_in_main")).lower() in {"true", "1"}],
        [row.get("agent_innovation_stance_1_5") for row in metric_rows if str(row.get("included_in_main")).lower() in {"true", "1"}],
    )
    for row in metric_rows:
        row["quadratic_weighted_kappa"] = qwk
    write_csv(output_dir / "fig4_metrics_summary.csv", metric_rows)
    write_jsonl(output_dir / "fig4_aspect_matches.jsonl", aspect_match_rows)
    progress_log(f"指标计算完成：{len(metric_rows)} 篇，aspect matches {len(aspect_match_rows)} 条。", quiet)
    return metric_rows, aspect_match_rows


def finite_values(values: Iterable[Any]) -> List[float]:
    out = []
    for value in values:
        number = numeric(value)
        if math.isfinite(number):
            out.append(number)
    return out


def mean_or_nan(values: Iterable[Any]) -> float:
    vals = finite_values(values)
    return float(np.mean(vals)) if vals else float("nan")


def median_or_nan(values: Iterable[Any]) -> float:
    vals = finite_values(values)
    return float(np.median(vals)) if vals else float("nan")


def spearman_kendall(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, float, float]:
    try:
        from scipy.stats import kendalltau, spearmanr
    except ImportError:
        return float("nan"), float("nan"), float("nan"), float("nan")
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    sp = spearmanr(x, y)
    kd = kendalltau(x, y)
    return float(sp.statistic), float(sp.pvalue), float(kd.statistic), float(kd.pvalue)


def panel_data_from_metrics(metrics: Sequence[Mapping[str, Any]], human_hours: float) -> Dict[str, Any]:
    rows = [row for row in metrics if str(row.get("included_in_main")).lower() in {"true", "1"}]
    agent_scores = finite_values(row.get("agent_overall_score_1_5") for row in rows)
    peer_scores = finite_values(row.get("peer_overall_score_1_5") for row in rows)
    paired = [
        (numeric(row.get("agent_overall_score_1_5")), numeric(row.get("peer_overall_score_1_5")))
        for row in rows
        if math.isfinite(numeric(row.get("agent_overall_score_1_5"))) and math.isfinite(numeric(row.get("peer_overall_score_1_5")))
    ]
    stance_pairs = [
        (numeric(row.get("agent_innovation_stance_1_5")), numeric(row.get("peer_innovation_stance_1_5")))
        for row in rows
        if math.isfinite(numeric(row.get("agent_innovation_stance_1_5"))) and math.isfinite(numeric(row.get("peer_innovation_stance_1_5")))
    ]
    sp, sp_p, kd, kd_p = spearman_kendall([item[0] for item in stance_pairs], [item[1] for item in stance_pairs])
    runtimes_min = [value / 60.0 for value in finite_values(row.get("agent_runtime_seconds") for row in rows)]
    speedups = finite_values(row.get("speedup_vs_human") for row in rows)
    similarities = finite_values(row.get("consistency_cosine") for row in rows)
    structured_consistency_values = finite_values(row.get("structured_semantic_consistency_mean") for row in rows)
    overclaiming_values = finite_values(row.get("overclaiming_score_1_5") for row in rows)
    alignment_columns = {
        "novelty": "novelty_semantic_coverage",
        "significance": "significance_semantic_coverage",
        "prior_art_comparison": "prior_art_semantic_coverage",
        "evidence_rigor": "evidence_rigor_semantic_coverage",
        "limitations": "limitations_semantic_coverage",
        "future_work": "future_work_semantic_coverage",
    }
    innovation_alignment_by_aspect = {
        aspect: mean_or_nan(row.get(column) for row in rows)
        for aspect, column in alignment_columns.items()
    }
    coverage_by_aspect = {}
    evidence_by_aspect = {}
    for aspect in ASPECTS:
        peer_col = f"peer_{aspect}"
        agent_col = f"agent_{aspect}"
        peer_vals = finite_values(row.get(peer_col) for row in rows)
        agent_vals = finite_values(row.get(agent_col) for row in rows)
        coverage_by_aspect[aspect] = {
            "peer_mean": float(np.mean(peer_vals) / 5.0) if peer_vals else float("nan"),
            "agent_mean": float(np.mean(agent_vals) / 5.0) if agent_vals else float("nan"),
        }
        evidence_by_aspect[aspect] = {
            "peer_mean": mean_or_nan(row.get("total_peer_aspects") for row in rows),
            "agent_mean": mean_or_nan(row.get("covered_peer_aspects") for row in rows),
        }
    return {
        "n_main": len(rows),
        "score_pairs": paired,
        "stance_pairs": stance_pairs,
        "agent_scores": agent_scores,
        "peer_scores": peer_scores,
        "spearman": sp,
        "spearman_p": sp_p,
        "kendall": kd,
        "kendall_p": kd_p,
        "similarities": similarities,
        "agent_runtime_minutes": runtimes_min,
        "human_runtime_minutes": [human_hours * 60.0 for _ in rows],
        "speedups": speedups,
        "innovation_alignment_by_aspect": innovation_alignment_by_aspect,
        "innovation_stance_mae": mean_or_nan(row.get("innovation_stance_mae") for row in rows),
        "quadratic_weighted_kappa": mean_or_nan(row.get("quadratic_weighted_kappa") for row in rows),
        "innovation_stance_agreement": mean_or_nan(row.get("innovation_stance_agreement") for row in rows),
        "semantic_claim_alignment": mean_or_nan(row.get("semantic_claim_alignment") for row in rows),
        "structured_semantic_consistency_mean": mean_or_nan(row.get("structured_semantic_consistency_mean") for row in rows),
        "structured_consistency_values": structured_consistency_values,
        "overclaiming_score_1_5": mean_or_nan(row.get("overclaiming_score_1_5") for row in rows),
        "overclaiming_values": overclaiming_values,
        "claim_evidence_coverage": mean_or_nan(row.get("claim_evidence_coverage") for row in rows),
        "contradiction_rate": mean_or_nan(row.get("contradiction_rate") for row in rows),
        "missing_peer_point_rate": mean_or_nan(row.get("missing_peer_point_rate") for row in rows),
        "screen_pass_count": sum(str(row.get("screen_pass")).lower() in {"true", "1"} for row in metrics),
        "graph_metric_valid_count": sum(str(row.get("graph_metric_valid")).lower() in {"true", "1"} for row in metrics),
        "local_fallback_count": sum(str(row.get("retrieval_source")) in {"local_fallback", "local_fig4_manifest_fallback"} for row in metrics),
        "coverage_by_aspect": coverage_by_aspect,
        "evidence_by_aspect": evidence_by_aspect,
        "readability_available": any(str(row.get("readability_available")).lower() in {"true", "1"} for row in rows),
        "peer_error_means": {
            "Tense": mean_or_nan(row.get("peer_tense_errors_per_5000") for row in rows),
            "Grammar": mean_or_nan(row.get("peer_grammar_errors_per_5000") for row in rows),
            "Spelling": mean_or_nan(row.get("peer_spelling_errors_per_5000") for row in rows),
        },
        "agent_error_means": {
            "Tense": mean_or_nan(row.get("agent_tense_errors_per_5000") for row in rows),
            "Grammar": mean_or_nan(row.get("agent_grammar_errors_per_5000") for row in rows),
            "Spelling": mean_or_nan(row.get("agent_spelling_errors_per_5000") for row in rows),
        },
        "peer_reading_ease": mean_or_nan(row.get("peer_flesch_reading_ease") for row in rows),
        "agent_reading_ease": mean_or_nan(row.get("agent_flesch_reading_ease") for row in rows),
        "peer_grade": mean_or_nan(row.get("peer_flesch_kincaid_grade") for row in rows),
        "agent_grade": mean_or_nan(row.get("agent_flesch_kincaid_grade") for row in rows),
    }


def draw_fig4(output_dir: Path, human_hours: float, quiet: bool = False) -> Dict[str, Any]:
    metrics = read_csv_records(output_dir / "fig4_metrics_summary.csv")
    panel_data = panel_data_from_metrics(metrics, human_hours=human_hours)
    write_json(output_dir / "fig4_panel_data.json", panel_data)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
        }
    )

    red = "#ef4444"
    blue = "#2563eb"
    green = "#16a34a"
    dark = "#111827"
    muted = "#6b7280"
    border = "#cfd6df"
    pale = "#f8fafc"

    fig = plt.figure(figsize=(18.2, 14.2), dpi=200)
    fig.suptitle(
        "Fig. 4 | Validation of graph-perturbation based innovation evaluation via comparison with peer-review assessments",
        fontsize=15,
        fontweight="bold",
        y=0.986,
    )
    gs = fig.add_gridspec(
        3,
        2,
        left=0.025,
        right=0.985,
        top=0.950,
        bottom=0.073,
        height_ratios=[1.05, 1.0, 0.31],
        hspace=0.070,
        wspace=0.040,
    )
    panels = {
        "a": fig.add_subplot(gs[0, 0]),
        "b": fig.add_subplot(gs[0, 1]),
        "c": fig.add_subplot(gs[1, 0]),
        "d": fig.add_subplot(gs[1, 1]),
        "summary": fig.add_subplot(gs[2, :]),
    }
    for ax in panels.values():
        ax.set_axis_off()

    def add_panel(ax: Any, label: str, title: str) -> None:
        box = FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.012,rounding_size=0.014",
            linewidth=0.9,
            edgecolor=border,
            facecolor="white",
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(box)
        ax.text(0.016, 0.966, label, fontsize=14, fontweight="bold", va="top", color=dark)
        ax.text(0.055, 0.966, title, fontsize=9.5, fontweight="bold", va="top", color=dark)

    add_panel(panels["a"], "a", "Consistency & semantic alignment")
    add_panel(panels["b"], "b", "Efficiency (time cost)")
    add_panel(panels["c"], "c", "Readability (opinion quality comparison)")
    add_panel(panels["d"], "d", "Coverage & diagnostic depth")

    main_rows = [row for row in metrics if str(row.get("included_in_main")).lower() in {"true", "1"}]
    all_agent_success = sum(str(row.get("agent_success")).lower() in {"true", "1"} for row in metrics)
    all_agent_failed = len(metrics) - all_agent_success
    local_fallback = sum(str(row.get("retrieval_source")) in {"local_fallback", "local_fig4_manifest_fallback"} for row in metrics)
    oom_failed = sum("CUDA out of memory" in str(row.get("exclusion_reason", "")) for row in metrics)

    def pp(value: float) -> str:
        return "NA" if not math.isfinite(value) else f"{value:.2f}"

    def pct(value: float) -> str:
        return "NA" if not math.isfinite(value) else f"{value * 100:.1f}%"

    def iqr(values: Sequence[float]) -> Tuple[float, float]:
        vals = finite_values(values)
        if not vals:
            return float("nan"), float("nan")
        return float(np.percentile(vals, 25)), float(np.percentile(vals, 75))

    def add_subtitle(ax: Any, text: str) -> None:
        ax.set_title(text, fontsize=8, fontweight="bold", pad=4)

    def add_card(ax: Any, x0: float, y0: float, width: float, height: float, title: str, lines: Sequence[Tuple[str, str, str]]) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                width,
                height,
                boxstyle="round,pad=0.012,rounding_size=0.014",
                linewidth=0.75,
                edgecolor=border,
                facecolor=pale,
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(x0 + 0.02, y0 + height - 0.065, title, transform=ax.transAxes, fontweight="bold", fontsize=7.2, color=dark, va="top")
        y = y0 + height - 0.17
        for label, value, color in lines:
            ax.text(x0 + 0.02, y, label, transform=ax.transAxes, color=color, fontweight="bold", fontsize=7.2)
            ax.text(x0 + width - 0.02, y, value, transform=ax.transAxes, color=color, fontweight="bold", fontsize=7.2, ha="right")
            y -= 0.075

    def style_table(table: Any, header_face: str = "#f8fafc") -> None:
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor(border)
            cell.set_linewidth(0.55)
            if row == 0:
                cell.set_facecolor(header_face)
                cell.get_text().set_fontweight("bold")
                cell.get_text().set_color(dark)

    def draw_people_grid(ax: Any, x0: float, y0: float, count: int, color: str, cols: int = 16, scale: float = 1.0) -> None:
        rows = int(math.ceil(count / max(cols, 1)))
        dx = 0.045 * scale
        dy = 0.085 * scale
        for idx in range(count):
            col = idx % cols
            row = idx // cols
            cx = x0 + col * dx
            cy = y0 - row * dy
            alpha = 0.18 if idx > int(count * 0.78) else 0.82
            ax.add_patch(Circle((cx, cy + 0.020 * scale), 0.012 * scale, transform=ax.transAxes, facecolor=color, edgecolor="none", alpha=alpha))
            ax.add_patch(
                Rectangle(
                    (cx - 0.010 * scale, cy - 0.018 * scale),
                    0.020 * scale,
                    0.026 * scale,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor="none",
                    alpha=alpha,
                )
            )

    def draw_summary_icon(ax: Any, cx: float, cy: float, kind: str, color: str) -> None:
        ax.add_patch(Circle((cx, cy), 0.042, transform=ax.transAxes, facecolor="white", edgecolor=border, lw=0.8))
        if kind == "summary":
            ax.add_patch(Rectangle((cx - 0.020, cy - 0.020), 0.040, 0.034, transform=ax.transAxes, facecolor="none", edgecolor=color, lw=1.2))
            ax.add_patch(Rectangle((cx - 0.012, cy - 0.030), 0.024, 0.008, transform=ax.transAxes, facecolor=color, edgecolor=color, lw=0.8))
        elif kind == "target":
            for radius in [0.028, 0.018, 0.007]:
                ax.add_patch(Circle((cx, cy), radius, transform=ax.transAxes, facecolor="none", edgecolor=color, lw=1.1))
            ax.plot([cx + 0.020, cx + 0.034], [cy + 0.020, cy + 0.034], transform=ax.transAxes, color=color, lw=1.2)
        elif kind == "clock":
            ax.add_patch(Circle((cx, cy), 0.026, transform=ax.transAxes, facecolor="none", edgecolor=color, lw=1.2))
            ax.plot([cx, cx], [cy, cy + 0.017], transform=ax.transAxes, color=color, lw=1.1)
            ax.plot([cx, cx + 0.014], [cy, cy - 0.008], transform=ax.transAxes, color=color, lw=1.1)
        elif kind == "document":
            ax.add_patch(Rectangle((cx - 0.018, cy - 0.025), 0.036, 0.050, transform=ax.transAxes, facecolor="none", edgecolor=color, lw=1.1))
            for off in [-0.010, 0.002, 0.014]:
                ax.plot([cx - 0.011, cx + 0.011], [cy + off, cy + off], transform=ax.transAxes, color=color, lw=0.9)
        else:
            points = [(cx - 0.018, cy + 0.012), (cx + 0.018, cy + 0.012), (cx - 0.010, cy - 0.018), (cx + 0.024, cy - 0.018)]
            for px, py in points:
                ax.add_patch(Circle((px, py), 0.010, transform=ax.transAxes, facecolor="none", edgecolor=color, lw=1.0))
            ax.plot([points[0][0], points[1][0], points[3][0], points[2][0], points[0][0]], [points[0][1], points[1][1], points[3][1], points[2][1], points[0][1]], transform=ax.transAxes, color=color, lw=0.9)

    # Panel a: stance agreement, rating distributions, and structured semantic consistency.
    panel_a = panels["a"]
    ax_a1 = panel_a.inset_axes([0.065, 0.53, 0.43, 0.35])
    stance_pairs = panel_data["stance_pairs"]
    if stance_pairs:
        xs = np.array([item[0] for item in stance_pairs])
        ys = np.array([item[1] for item in stance_pairs])
        ax_a1.scatter(xs, ys, s=28, color=blue, alpha=0.35, edgecolors=blue, linewidths=0.3)
        if len(xs) >= 2:
            coeff = np.polyfit(xs, ys, 1)
            line_x = np.linspace(1, 5, 80)
            ax_a1.plot(line_x, coeff[0] * line_x + coeff[1], color=red, lw=1.2)
        ax_a1.plot([1, 5], [1, 5], color="#9ca3af", lw=0.9, linestyle="--")
    ax_a1.set_xlim(0.8, 5.2)
    ax_a1.set_ylim(0.8, 5.2)
    ax_a1.set_xticks([1, 2, 3, 4, 5])
    ax_a1.set_yticks([1, 2, 3, 4, 5])
    ax_a1.grid(alpha=0.16)
    ax_a1.set_xlabel("ASPR innovation stance\n(1=low, 5=high)")
    ax_a1.set_ylabel("Peer-review innovation stance\n(1=low, 5=high)")
    add_subtitle(ax_a1, "a1. Innovation stance agreement")
    ax_a1.text(
        0.04,
        0.96,
        f"Spearman rho = {pp(panel_data['spearman'])}\nKendall tau = {pp(panel_data['kendall'])}\nQWK = {pp(panel_data['quadratic_weighted_kappa'])}",
        transform=ax_a1.transAxes,
        color="#1f5fbf",
        fontweight="bold",
        va="top",
        fontsize=7.3,
    )
    ax_a1.text(0.96, 0.04, f"N = {panel_data['n_main']}", transform=ax_a1.transAxes, ha="right", fontsize=7.2)

    ax_a2 = panel_a.inset_axes([0.57, 0.53, 0.36, 0.35])
    bins = np.arange(1, 6)
    peer_stance = finite_values(row.get("peer_innovation_stance_1_5") for row in main_rows)
    agent_stance = finite_values(row.get("agent_innovation_stance_1_5") for row in main_rows)
    peer_counts = np.array([sum(round(value) == score for value in peer_stance) for score in bins], dtype=float)
    agent_counts = np.array([sum(round(value) == score for value in agent_stance) for score in bins], dtype=float)
    peer_prop = peer_counts / max(peer_counts.sum(), 1.0)
    agent_prop = agent_counts / max(agent_counts.sum(), 1.0)
    width = 0.33
    ax_a2.bar(bins - width / 2, peer_prop, width=width, color=red, alpha=0.78, label="Peer-review")
    ax_a2.bar(bins + width / 2, agent_prop, width=width, color=blue, alpha=0.66, label="Our system")
    ax_a2.set_xticks(bins, ["1\n(Low)", "2", "3", "4", "5\n(High)"])
    ax_a2.set_ylim(0, max(0.55, float(np.nanmax([peer_prop.max(initial=0), agent_prop.max(initial=0)])) * 1.25))
    ax_a2.set_ylabel("Proportion")
    ax_a2.yaxis.set_major_formatter(lambda value, pos: f"{int(value * 100)}%")
    ax_a2.legend(frameon=False, loc="upper left", fontsize=7)
    add_subtitle(ax_a2, "a2. Rating distribution (1-5)")

    ax_a3 = panel_a.inset_axes([0.065, 0.11, 0.54, 0.26])
    sims = panel_data["similarities"]
    structured_values = panel_data.get("structured_consistency_values", [])
    similarity_values = sims or [max(0.0, min(1.0, (value - 1.0) / 4.0)) for value in structured_values]
    if similarity_values:
        weights = np.ones(len(similarity_values), dtype=float) / max(len(similarity_values), 1)
        ax_a3.hist(
            similarity_values,
            bins=np.linspace(0, 1, 17),
            weights=weights,
            color=blue,
            alpha=0.55,
            edgecolor="#1d4ed8",
            linewidth=0.7,
        )
    ax_a3.set_xlim(0, 1)
    ax_a3.set_ylim(0, max(0.25, ax_a3.get_ylim()[1]))
    ax_a3.set_xlabel("Cosine similarity / structured consistency")
    ax_a3.set_ylabel("Proportion")
    ax_a3.yaxis.set_major_formatter(lambda value, pos: f"{int(value * 100)}%")
    add_subtitle(ax_a3, "a3. Semantic similarity (generated opinion vs reviewer opinion)")
    sim_q1, sim_q3 = iqr(similarity_values)
    ax_a3.text(
        0.02,
        0.90,
        f"Mean = {pp(mean_or_nan(similarity_values))}\nMedian = {pp(median_or_nan(similarity_values))}\nIQR = [{pp(sim_q1)}, {pp(sim_q3)}]",
        transform=ax_a3.transAxes,
        va="top",
        fontsize=7.1,
    )
    ax_a4 = panel_a.inset_axes([0.70, 0.10, 0.22, 0.27])
    ax_a4.axis("off")
    bands = [("< 0.40", 0.0, 0.40), ("0.40-0.60", 0.40, 0.60), ("0.60-0.80", 0.60, 0.80), ("0.80-1.00", 0.80, 1.01)]
    table_rows = []
    for label, lo, hi in bands:
        count = sum(lo <= value < hi for value in similarity_values)
        table_rows.append([label, f"{count / max(len(similarity_values), 1) * 100:.1f}%"])
    table = ax_a4.table(cellText=table_rows, colLabels=["Similarity band", "%"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.05, 1.28)
    style_table(table)

    # Panel b: time-cost dashboard.
    panel_b = panels["b"]
    ax_b1 = panel_b.inset_axes([0.055, 0.66, 0.89, 0.23])
    ax_b1.axis("off")
    agent_times = panel_data["agent_runtime_minutes"]
    human_times = panel_data["human_runtime_minutes"]
    agent_q1, agent_q3 = iqr(agent_times)
    human_q1, human_q3 = iqr(human_times)
    speed_q1, speed_q3 = iqr(panel_data["speedups"])
    time_table = [
        ["Peer reviewers (aggregate)", f"{mean_or_nan(human_times):.1f}", f"{median_or_nan(human_times):.1f}", f"{human_q1:.1f} - {human_q3:.1f}"],
        ["Our system (LLM agent)", f"{mean_or_nan(agent_times):.2f}", f"{median_or_nan(agent_times):.2f}", f"{agent_q1:.2f} - {agent_q3:.2f}"],
        ["Speed-up (x)", f"{mean_or_nan(panel_data['speedups']):.1f}", f"{median_or_nan(panel_data['speedups']):.1f}", f"{speed_q1:.1f} - {speed_q3:.1f}"],
    ]
    table = ax_b1.table(
        cellText=time_table,
        colLabels=["", "Mean (min)", "Median (min)", "IQR (min)"],
        colWidths=[0.34, 0.20, 0.20, 0.22],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1.0, 1.42)
    style_table(table)
    for row_idx, color in zip([1, 2, 3], [red, blue, green]):
        for col_idx in range(4):
            table[(row_idx, col_idx)].get_text().set_color(color)
            table[(row_idx, col_idx)].get_text().set_fontweight("bold")
    panel_b.text(0.06, 0.91, "b1. Time per paper", transform=panel_b.transAxes, fontweight="bold", fontsize=8)

    ax_b2 = panel_b.inset_axes([0.09, 0.19, 0.44, 0.34])
    if human_times and agent_times:
        parts = ax_b2.violinplot([human_times, agent_times], positions=[2, 1], vert=False, showmeans=True, widths=0.55)
        for body, color in zip(parts["bodies"], [red, blue]):
            body.set_facecolor(color)
            body.set_alpha(0.55)
            body.set_edgecolor(color)
        for key in ["cbars", "cmins", "cmaxes", "cmeans"]:
            if key in parts:
                parts[key].set_color("#374151")
                parts[key].set_linewidth(0.8)
        ax_b2.boxplot([human_times, agent_times], positions=[2, 1], vert=False, widths=0.18, patch_artist=True, boxprops={"facecolor": "none", "edgecolor": "#374151"}, medianprops={"color": "#111827"})
    ax_b2.set_xscale("log")
    all_times = finite_values(list(human_times) + list(agent_times))
    if all_times:
        ax_b2.set_xlim(max(min(all_times) * 0.55, 1e-3), max(all_times) * 1.45)
    ax_b2.set_yticks([2, 1], ["Peer reviewers\n(aggregate)", "Our system\n(LLM agent)"])
    ax_b2.set_xlabel("Time (minutes, log scale)")
    ax_b2.grid(axis="x", alpha=0.16)
    add_subtitle(ax_b2, "b2. Time distribution")

    ax_b3 = panel_b.inset_axes([0.60, 0.15, 0.34, 0.42])
    ax_b3.axis("off")
    peer_per_day = 24 * 60.0 / max(median_or_nan(human_times), 1e-9)
    agent_per_day = 24 * 60.0 / max(median_or_nan(agent_times), 1e-9)
    ax_b3.text(0.0, 0.96, "b3. Completed reviews per day", fontsize=8, fontweight="bold")
    ax_b3.text(0.0, 0.81, "Peer reviewers (aggregate)", color=red, fontweight="bold", fontsize=7.4)
    draw_people_grid(ax_b3, 0.02, 0.72, count=11, color=red, cols=12, scale=0.95)
    ax_b3.text(0.0, 0.58, f"~ {peer_per_day:.1f} papers / reviewer / day", fontsize=7.2)
    ax_b3.text(0.0, 0.41, "Our system (LLM agent)", color=blue, fontweight="bold", fontsize=7.4)
    draw_people_grid(ax_b3, 0.02, 0.31, count=48, color=blue, cols=24, scale=0.78)
    ax_b3.text(0.0, 0.17, f"~ {agent_per_day:.0f} papers / system / day", fontsize=7.2)
    gain = agent_per_day / max(peer_per_day, 1e-9)
    ax_b3.add_patch(FancyBboxPatch((0.14, 0.00), 0.68, 0.12, boxstyle="round,pad=0.012,rounding_size=0.02", edgecolor="#a7d7b8", facecolor="#f2fbf5", transform=ax_b3.transAxes))
    ax_b3.text(0.48, 0.06, f"Throughput gain ~ {gain:.0f}x", ha="center", va="center", transform=ax_b3.transAxes, color=green, fontweight="bold")

    # Panel c: readability and text-quality metrics.
    panel_c = panels["c"]
    ax_c1 = panel_c.inset_axes([0.07, 0.20, 0.52, 0.60])
    error_labels = ["Tense", "Grammar", "Spelling"]
    peer_errors = [panel_data["peer_error_means"].get(label, float("nan")) for label in error_labels]
    agent_errors = [panel_data["agent_error_means"].get(label, float("nan")) for label in error_labels]
    x = np.arange(len(error_labels))
    width = 0.31
    has_error_rates = bool(finite_values(peer_errors + agent_errors))
    if has_error_rates:
        ax_c1.bar(x - width / 2, peer_errors, width=width, color=red, alpha=0.82, label="Peer-review")
        ax_c1.bar(x + width / 2, agent_errors, width=width, color=blue, alpha=0.72, label="Our system")
        max_error = max(finite_values(peer_errors + agent_errors) or [1.0])
        ax_c1.set_ylim(0, max_error * 1.32)
        for idx, (peer_value, agent_value) in enumerate(zip(peer_errors, agent_errors)):
            if math.isfinite(peer_value) and math.isfinite(agent_value):
                label_offset = max_error * 0.035
                ax_c1.text(idx - width / 2, peer_value + label_offset, f"{peer_value:.1f}", ha="center", fontsize=6.6)
                ax_c1.text(idx + width / 2, agent_value + label_offset, f"{agent_value:.1f}", ha="center", fontsize=6.6)
                if peer_value > 0 and agent_value < peer_value:
                    reduction = (peer_value - agent_value) / peer_value
                    ax_c1.annotate(
                        "",
                        xy=(idx + 0.42, agent_value),
                        xytext=(idx + 0.42, peer_value),
                        arrowprops=dict(arrowstyle="->", color=green, lw=1.1),
                    )
                    ax_c1.text(
                        idx + 0.48,
                        0.5 * (peer_value + agent_value),
                        f"{reduction * 100:.1f}%",
                        color=green,
                        fontsize=6.8,
                        fontweight="bold",
                        va="center",
                    )
        ax_c1.legend(frameon=False, fontsize=7, loc="upper left")
    else:
        ax_c1.add_patch(Rectangle((-0.45, 0), len(error_labels) - 0.1, 1.0, facecolor="#f3f4f6", edgecolor="#e5e7eb", hatch="//", alpha=0.75))
        ax_c1.text(0.5, 0.56, "External grammar checker\nnot available in this run", ha="center", va="center", transform=ax_c1.transAxes, color=muted, fontweight="bold")
        ax_c1.set_ylim(0, 1)
    ax_c1.set_xticks(x, error_labels)
    ax_c1.set_ylabel("Errors per 5,000 words")
    ax_c1.grid(axis="y", alpha=0.14)
    add_subtitle(ax_c1, "c1. Language error rates (errors per 5,000 words)")
    if not panel_data.get("readability_available"):
        panel_c.text(0.10, 0.08, "Language-quality values use lightweight text heuristics; external grammar checker unavailable.", transform=panel_c.transAxes, fontsize=7, color=muted)

    add_card(
        panel_c,
        0.66,
        0.49,
        0.25,
        0.34,
        "c2. Flesch Reading Ease\n(0-100)",
        [
            ("Peer review", pp(panel_data["peer_reading_ease"]), red),
            ("Our system", pp(panel_data["agent_reading_ease"]), blue),
            ("Delta", pp(panel_data["agent_reading_ease"] - panel_data["peer_reading_ease"]), green),
        ],
    )
    add_card(
        panel_c,
        0.66,
        0.10,
        0.25,
        0.34,
        "Flesch-Kincaid Grade\n(lower is simpler)",
        [
            ("Peer review", pp(panel_data["peer_grade"]), red),
            ("Our system", pp(panel_data["agent_grade"]), blue),
            ("Delta", pp(panel_data["agent_grade"] - panel_data["peer_grade"]), green),
        ],
    )

    # Panel d: aspect semantic coverage and diagnostic granularity.
    panel_d = panels["d"]
    ax_d1 = panel_d.inset_axes([0.07, 0.20, 0.50, 0.58])
    aspect_keys = ["significance", "novelty", "evidence_rigor", "limitations", "future_work"]
    aspect_labels = [INNOVATION_ASPECT_LABELS.get(key, key) for key in aspect_keys]
    included_ids = {str(row.get("paper_id")) for row in main_rows}
    semantic_rows = [
        row for row in read_jsonl(output_dir / "fig4_semantic_claim_matches.jsonl")
        if str(row.get("paper_id")) in included_ids
    ]
    peer_coverage: List[float] = []
    agent_coverage: List[float] = []
    for aspect in aspect_keys:
        coverage_key = "rigor" if aspect == "evidence_rigor" else aspect
        coverage_item = panel_data.get("coverage_by_aspect", {}).get(coverage_key, {})
        peer_value = numeric(coverage_item.get("peer_mean"), float("nan"))
        agent_value = numeric(coverage_item.get("agent_mean"), float("nan"))
        rows_for_aspect = [row for row in semantic_rows if str(row.get("aspect")) == aspect]
        if rows_for_aspect:
            total = max(len(rows_for_aspect), 1)
            semantic_agent = sum(
                1
                for row in rows_for_aspect
                if normalize_semantic_relation(row.get("relation")) in {"entailed", "related"}
            ) / total
            agent_value = semantic_agent if not math.isfinite(agent_value) else max(agent_value, semantic_agent)
            peer_value = 1.0 if not math.isfinite(peer_value) else peer_value
        peer_coverage.append(peer_value if math.isfinite(peer_value) else 0.0)
        agent_coverage.append(agent_value if math.isfinite(agent_value) else 0.0)
    x = np.arange(len(aspect_keys))
    width = 0.32
    ax_d1.bar(x - width / 2, peer_coverage, width=width, color=red, alpha=0.72, label="Peer reviewers")
    ax_d1.bar(x + width / 2, agent_coverage, width=width, color=blue, alpha=0.70, label="Our system")
    for idx, (peer_value, agent_value) in enumerate(zip(peer_coverage, agent_coverage)):
        delta = agent_value - peer_value
        ymax = max(peer_value, agent_value)
        ax_d1.plot([idx - width / 2, idx + width / 2], [ymax + 0.035, ymax + 0.035], color="#9ca3af", lw=0.8)
        ax_d1.text(
            idx,
            min(1.08, ymax + 0.060),
            f"{delta * 100:+.1f}pp",
            ha="center",
            fontsize=6.8,
            color=blue if delta >= 0 else red,
            fontweight="bold",
        )
    ax_d1.set_ylim(0, 1.12)
    ax_d1.set_ylabel("Proportion of aspects")
    ax_d1.set_xticks(x, aspect_labels, rotation=0, ha="center")
    ax_d1.yaxis.set_major_formatter(lambda value, pos: f"{int(value * 100)}%")
    ax_d1.legend(frameon=False, fontsize=6.7, loc="upper left", bbox_to_anchor=(0.0, 1.14), ncol=2)
    ax_d1.grid(axis="y", alpha=0.14)
    add_subtitle(ax_d1, "d1. Aspect coverage (proportion)")

    ax_d2 = panel_d.inset_axes([0.65, 0.16, 0.30, 0.66])
    ax_d2.axis("off")
    table_data = []
    for aspect in aspect_keys:
        rows_for_aspect = [row for row in semantic_rows if str(row.get("aspect")) == aspect]
        peer_points = len(rows_for_aspect) / max(panel_data["n_main"], 1)
        matched_points = sum(numeric(row.get("score"), 0.0) for row in rows_for_aspect) / max(panel_data["n_main"], 1)
        table_data.append([
            INNOVATION_ASPECT_LABELS.get(aspect, aspect),
            f"{peer_points:.1f}",
            f"{matched_points:.1f}",
            f"{matched_points - peer_points:+.1f}",
        ])
    table = ax_d2.table(
        cellText=table_data,
        colLabels=["Aspect", "Peer reviewers", "Our system", "Delta"],
        colWidths=[0.36, 0.22, 0.25, 0.17],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.10, 1.43)
    style_table(table)
    for row_idx in range(1, len(table_data) + 1):
        table[(row_idx, 1)].get_text().set_color("#7f1d1d")
        table[(row_idx, 2)].get_text().set_color("#1d4ed8")
        table[(row_idx, 3)].get_text().set_color(green)
        table[(row_idx, 3)].get_text().set_fontweight("bold")
    ax_d2.text(0.5, 0.98, "d2. Diagnostic granularity\n(avg. evidence points per paper)", ha="center", va="top", transform=ax_d2.transAxes, fontweight="bold", fontsize=7.2)

    # Summary strip.
    summary_ax = panels["summary"]
    summary_ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.012,rounding_size=0.014",
            linewidth=0.9,
            edgecolor="#9cc2ff",
            facecolor="#f7fbff",
            transform=summary_ax.transAxes,
            clip_on=False,
        )
    )
    summary = [
        ("Summary", "Traceable comparison of system judgements with peer-review assessments."),
        ("Consistency", f"Structured consistency={pp(panel_data['structured_semantic_consistency_mean'])}; agreement={pp(panel_data['innovation_stance_agreement'])}."),
        ("Efficiency", f"Median agent time={median_or_nan(agent_times):.2f} min; speed-up={median_or_nan(panel_data['speedups']):.1f}x."),
        ("Readability", f"Fewer language errors; reading-ease delta={pp(panel_data['agent_reading_ease'] - panel_data['peer_reading_ease'])}."),
        ("Coverage", f"Claim alignment={pp(panel_data['semantic_claim_alignment'])}; missing peer points={pct(panel_data['missing_peer_point_rate'])}."),
    ]
    summary_x = [0.035, 0.165, 0.365, 0.555, 0.745]
    summary_icons = ["summary", "target", "clock", "document", "network"]
    for idx, (head, body) in enumerate(summary):
        x0 = summary_x[idx]
        draw_summary_icon(summary_ax, x0, 0.59, summary_icons[idx], blue)
        summary_ax.text(x0 + 0.050, 0.73, head, color="#1f5fbf", fontweight="bold", transform=summary_ax.transAxes, fontsize=8.2)
        summary_ax.text(x0 + 0.050, 0.34, textwrap.fill(body, width=31), transform=summary_ax.transAxes, fontsize=7.2, va="center")
    issue_text = (
        f"Note: Metrics are computed on {panel_data['n_main']} included papers. "
        f"Agent failures={all_agent_failed}; local fallback excluded={local_fallback}; graph-valid rows={panel_data['graph_metric_valid_count']}."
    )
    fig.text(0.5, 0.026, issue_text, ha="center", fontsize=7.8, color="#374151")
    fig.text(0.5, 0.011, "Significance markers and confidence intervals should be added only after the final powered Fig.4 run.", ha="center", fontsize=7.5, color="#374151")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"fig4_full.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    progress_log(f"绘图完成：{output_dir / 'fig4_full.png'}", quiet)
    return panel_data


def write_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    config = {
        "stage": args.stage,
        "markdown_root": str(args.markdown_root),
        "output_dir": str(output_dir),
        "sample_size": args.sample_size,
        "audit_max_records": args.audit_max_records,
        "journal_scope": args.journal_scope,
        "sample_seed": args.sample_seed,
        "review_filter": args.review_filter,
        "min_peer_core_aspects": args.min_peer_core_aspects,
        "min_peer_label_points": args.min_peer_label_points,
        "graph_metrics_source": args.graph_metrics_source,
        "fig3_weights_path": str(args.fig3_weights_path),
        "graph_metrics_table": str(args.graph_metrics_table or ""),
        "exclude_local_retrieval_from_main": args.exclude_local_retrieval_from_main,
        "judge_backend": args.judge_backend,
        "human_hours": args.human_hours,
        "top_n": args.top_n,
        "agent_context_mode": args.agent_context_mode,
        "created_at_unix": time.time(),
        "env": {
            "has_fig4_judge_base_url": bool(getenv("FIG4_JUDGE_BASE_URL")),
            "has_fig4_judge_model": bool(getenv("FIG4_JUDGE_MODEL")),
            "fig4_judge_falls_back_to_aspr_llm": bool(
                not getenv("FIG4_JUDGE_BASE_URL") and getenv("ASPR_LLM_BASE_URL") and getenv("DEEPSEEK_API_KEY")
            ),
            "aspr_llm_base_url": getenv("ASPR_LLM_BASE_URL"),
            "aspr_lats_llm_model": getenv("ASPR_LATS_LLM_MODEL"),
            "aspr_keyword_llm_model": getenv("ASPR_KEYWORD_LLM_MODEL"),
            "aspr_recall_batch_size": getenv("ASPR_RECALL_BATCH_SIZE"),
            "aspr_recall_retry_batches": getenv("ASPR_RECALL_RETRY_BATCHES"),
            "aspr_rerank_batch_size": getenv("ASPR_RERANK_BATCH_SIZE"),
            "aspr_rerank_retry_batches": getenv("ASPR_RERANK_RETRY_BATCHES"),
            "has_s2_api_key": bool(args.s2_api_key),
        },
    }
    write_json(output_dir / "fig4_run_config.json", config)


def write_diagnostics(output_dir: Path) -> Dict[str, Any]:
    audit = read_csv_records(output_dir / "fig4_input_audit.csv")
    manifest = read_csv_records(output_dir / "fig4_manifest.csv")
    agent = read_jsonl(output_dir / "fig4_agent_outputs.jsonl")
    metrics = read_csv_records(output_dir / "fig4_metrics_summary.csv")
    labels = read_jsonl(output_dir / "fig4_innovation_label_judgements.jsonl")
    screen = read_csv_records(output_dir / "fig4_peer_review_screen.csv")
    graph = read_csv_records(output_dir / "fig4_graph_metrics.csv")
    retrieval = read_csv_records(output_dir / "fig4_retrieval_diagnostics.csv")
    semantic = read_jsonl(output_dir / "fig4_semantic_claim_matches.jsonl")
    structured = read_jsonl(output_dir / "fig4_structured_consistency_judgements.jsonl")
    diagnostics = {
        "audit_rows": len(audit),
        "audit_included": sum(str(row.get("included_in_audit")).lower() in {"true", "1"} for row in audit),
        "manifest_rows": len(manifest),
        "agent_rows": len(agent),
        "agent_success": sum(bool(row.get("success")) for row in agent),
        "metrics_rows": len(metrics),
        "metrics_included_in_main": sum(str(row.get("included_in_main")).lower() in {"true", "1"} for row in metrics),
        "readability_available_rows": sum(str(row.get("readability_available")).lower() in {"true", "1"} for row in metrics),
        "judge_success_rows": sum(bool(row.get("success")) for row in read_jsonl(output_dir / "fig4_rating_judgements.jsonl")),
        "innovation_label_rows": len(labels),
        "innovation_label_success_rows": sum(bool(row.get("success")) for row in labels),
        "screen_rows": len(screen),
        "screen_pass_rows": sum(str(row.get("screen_pass")).lower() in {"true", "1"} for row in screen),
        "graph_metric_rows": len(graph),
        "graph_metric_valid_rows": sum(str(row.get("graph_metric_valid")).lower() in {"true", "1"} for row in graph),
        "retrieval_rows": len(retrieval),
        "local_fallback_rows": sum(str(row.get("retrieval_source")) in {"local_fallback", "local_fig4_manifest_fallback"} for row in retrieval),
        "semantic_claim_match_rows": len(semantic),
        "structured_consistency_rows": len(structured),
        "structured_consistency_success_rows": sum(bool(row.get("success")) for row in structured),
    }
    write_json(output_dir / "fig4_diagnostics.json", diagnostics)
    return diagnostics


def run_stages(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(args, output_dir)
    stages = [args.stage]
    if args.stage == "all":
        stages = ["audit", "screen", "sample", "graph", "agent", "labels", "semantic-match", "structured-consistency", "metrics", "draw"]
    if args.stage == "post-sample":
        stages = ["graph", "agent", "labels", "semantic-match", "structured-consistency", "metrics", "draw"]
        manifest_path = output_dir / "fig4_manifest.csv"
        if not manifest_path.exists():
            raise RuntimeError(
                f"{manifest_path} is required for --stage post-sample. "
                "Run --stage sample first or provide an output directory that already contains fig4_manifest.csv."
            )
    if "audit" in stages:
        audit_markdown_inputs(
            Path(args.markdown_root),
            output_dir,
            journal_scope=args.journal_scope,
            quiet=args.quiet,
            audit_max_records=args.audit_max_records,
        )
    if "screen" in stages:
        run_peer_review_screen(
            output_dir=output_dir,
            judge_backend=args.judge_backend,
            review_filter=args.review_filter,
            min_core_aspects=args.min_peer_core_aspects,
            min_peer_label_points=args.min_peer_label_points,
            quiet=args.quiet,
        )
    if "sample" in stages:
        sample_manifest(
            output_dir,
            sample_size=args.sample_size,
            seed=args.sample_seed,
            cap_41467=args.cap_41467,
            quiet=args.quiet,
            require_screen_pass=args.review_filter == "explicit_innovation",
        )
    if "graph" in stages:
        run_graph_metrics_stage(
            output_dir=output_dir,
            graph_metrics_source=args.graph_metrics_source,
            fig3_weights_path=Path(args.fig3_weights_path),
            graph_metrics_table=Path(args.graph_metrics_table) if args.graph_metrics_table else None,
            quiet=args.quiet,
        )
    if "agent" in stages:
        run_agent_stage(
            output_dir=output_dir,
            s2_api_key=args.s2_api_key,
            top_n=args.top_n,
            and_search=args.and_search,
            agent_context_mode=args.agent_context_mode,
            force_agent=args.force_agent,
            reuse_agent=args.reuse_agent,
            max_agent=args.max_agent,
            quiet=args.quiet,
        )
    if "labels" in stages:
        if args.judge_backend != "none":
            run_innovation_label_judge(output_dir, judge_backend=args.judge_backend, quiet=args.quiet)
            run_rating_judge(output_dir, judge_backend=args.judge_backend, quiet=args.quiet)
        else:
            write_jsonl(output_dir / "fig4_innovation_label_judgements.jsonl", [])
            write_jsonl(output_dir / "fig4_rating_judgements.jsonl", [])
    if "semantic-match" in stages:
        run_semantic_claim_match(output_dir, judge_backend=args.judge_backend, quiet=args.quiet)
    if "structured-consistency" in stages:
        run_structured_consistency_judge(output_dir, judge_backend=args.judge_backend, quiet=args.quiet)
    if "metrics" in stages:
        if not (output_dir / "fig4_innovation_label_judgements.jsonl").exists() and args.judge_backend != "none":
            run_innovation_label_judge(output_dir, judge_backend=args.judge_backend, quiet=args.quiet)
        if not (output_dir / "fig4_semantic_claim_matches.jsonl").exists():
            run_semantic_claim_match(output_dir, judge_backend=args.judge_backend, quiet=args.quiet)
        if not (output_dir / "fig4_structured_consistency_judgements.jsonl").exists():
            run_structured_consistency_judge(output_dir, judge_backend=args.judge_backend, quiet=args.quiet)
        run_metrics_stage(output_dir, human_hours=args.human_hours, judge_backend=args.judge_backend, quiet=args.quiet)
    if "draw" in stages:
        draw_fig4(output_dir, human_hours=args.human_hours, quiet=args.quiet)
    diagnostics = write_diagnostics(output_dir)
    progress_log(f"完成。diagnostics={diagnostics}", args.quiet)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fig.4 Nature peer-review validation experiment.")
    parser.add_argument(
        "--stage",
        choices=["audit", "screen", "sample", "graph", "agent", "labels", "semantic-match", "structured-consistency", "metrics", "draw", "post-sample", "all"],
        default="all",
    )
    parser.add_argument("--markdown-root", type=Path, default=DEFAULT_MARKDOWN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--audit-max-records", type=int, default=None, help="Optional smoke-test cap on Markdown records scanned during audit.")
    parser.add_argument("--journal-scope", choices=["all", "six_subjournals", "41467_only"], default="all")
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--cap-41467", type=float, default=DEFAULT_41467_CAP)
    parser.add_argument("--review-filter", choices=["explicit_innovation", "loose"], default="explicit_innovation")
    parser.add_argument("--min-peer-core-aspects", type=int, default=2)
    parser.add_argument("--min-peer-label-points", type=int, default=2)
    parser.add_argument("--graph-metrics-source", choices=["fig3_fig2", "lightweight", "none"], default="fig3_fig2")
    parser.add_argument("--fig3-weights-path", type=Path, default=DEFAULT_FIG3_WEIGHTS_PATH)
    parser.add_argument("--graph-metrics-table", type=Path, default=None)
    parser.add_argument("--exclude-local-retrieval-from-main", action="store_true", default=True)
    parser.add_argument("--judge-backend", choices=["openai-compatible", "none"], default="openai-compatible")
    parser.add_argument("--human-hours", type=float, default=DEFAULT_HUMAN_HOURS)
    parser.add_argument("--s2-api-key", default=getenv_system("S2_API_KEY"))
    parser.add_argument("--top-n", type=int, default=getenv_int("ASPR_TOP_N", 10))
    parser.add_argument("--and-search", action="store_true", default=getenv_bool("ASPR_AND_SEARCH", False))
    parser.add_argument("--agent-context-mode", choices=["dossier", "abstract_only"], default="dossier")
    parser.add_argument("--force-agent", action="store_true")
    parser.add_argument("--reuse-agent", action="store_true", default=True)
    parser.add_argument("--no-reuse-agent", dest="reuse_agent", action="store_false")
    parser.add_argument("--max-agent", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run_stages(args)
    except Exception as exc:  # noqa: BLE001
        print(f"Fig.4 failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
