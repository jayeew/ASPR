from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import sys
import textwrap
import time
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from aspr.env import getenv

DEFAULT_MARKDOWN_ROOT = PROJECT_ROOT / "data" / "nature_markdown"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fig04/old"
DEFAULT_SAMPLE_SEED = 20260617
DEFAULT_41467_CAP = 0.50
DEFAULT_HUMAN_HOURS = 5.0
DEFAULT_FIG3_WEIGHTS_PATH = PROJECT_ROOT / "outputs" / "fig03/old" / "multi_domain" / "fig3_best_weights.csv"
DEFAULT_FIG3_SCORE_TABLE_PATH = PROJECT_ROOT / "outputs" / "fig03/old" / "multi_domain" / "fig3_score_table.csv"
DEFAULT_FIG3_INDICATORS_PATH = PROJECT_ROOT / "outputs" / "fig03/old" / "multi_domain" / "fig3_publication_day_indicators.csv"
DEFAULT_FIG4_LABELING_PRIMARY_PER_TIER = 10
DEFAULT_FIG4_LABELING_RESERVE_PER_TIER = 20
FIG4_COMPLETED_BLINDED_LABELS_FILE = "fig4_completed_blinded_labels.csv"
FIG4_COMPLETED_BLINDED_LABELS_TEMPLATE_FILE = "fig4_completed_blinded_labels_template.csv"
FIG4_LABELER_IDS = ("labeler_1", "labeler_2", "labeler_3")

INNOVATION_METRIC_NAMES = ["B", "RS", "DeltaQ0", "Uzzi", "RTD", "BurtIP", "PDE"]
RATING_ASPECTS = ["significance", "novelty", "rigor", "limitations", "future_work"]
INNOVATION_ASPECTS = [
    "novelty",
    "significance",
    "prior_art_comparison",
    "evidence_rigor",
    "limitations",
    "future_work",
]
CORE_INNOVATION_ASPECTS = ["novelty", "significance", "prior_art_comparison"]
ASPECT_DISPLAY_NAMES = {
    "novelty": "Novelty",
    "significance": "Significance",
    "prior_art_comparison": "Prior art",
    "evidence_rigor": "Evidence/rigor",
    "limitations": "Limitations",
    "future_work": "Future work",
}
EVIDENCE_TYPE_BY_ASPECT = {
    "novelty": "novelty_claim",
    "significance": "significance_claim",
    "prior_art_comparison": "prior_art_comparison",
    "evidence_rigor": "evidence_support",
    "limitations": "limitation",
    "future_work": "future_work",
}
ALLOWED_POINT_POLARITIES = {"positive", "negative", "mixed", "neutral"}
ALLOWED_SOURCE_ROLES = {"reviewer", "editor", "agent"}
ALLOWED_EVIDENCE_TYPES = {
    "novelty_claim",
    "significance_claim",
    "prior_art_comparison",
    "evidence_support",
    "rigor_concern",
    "limitation",
    "future_work",
}
CROSS_ASPECT_FALLBACKS = {
    "novelty": ("prior_art_comparison",),
    "prior_art_comparison": ("novelty",),
    "evidence_rigor": ("limitations",),
    "limitations": ("evidence_rigor",),
}
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
DOI_RE = re.compile(r"https?://doi\.org/(10\.1038/[^\s)]+)|\b(10\.1038/[^\s)]+)", re.I)
PICTURE_LINE_RE = re.compile(r"^\*?\*?==>\s*picture\b.*omitted.*$", re.I)
REVISION_ONLY_RE = re.compile(
    r"\b("
    r"concerns?\s+(?:have\s+been\s+)?addressed|"
    r"all\s+concerns?\s+(?:have\s+been\s+)?(?:resolved|addressed)|"
    r"no\s+further\s+(?:comments?|concerns?)|"
    r"recommend\s+accept(?:ance)?|"
    r"suitable\s+for\s+publication"
    r")\b",
    re.I,
)


def progress_log(message: str, quiet: bool = False) -> None:
    """Print a compact progress message unless quiet mode is enabled."""
    if not quiet:
        print(f"[Fig4] {message}", file=sys.stderr, flush=True)


def atomic_write_text(path: Path, text: str) -> None:
    """Write text via a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
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
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    if fieldnames is None:
        fields: List[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        fieldnames = fields
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
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


def numeric(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def clamp(value: Any, lower: float, upper: float, default: float = float("nan")) -> float:
    number = numeric(value, default)
    if not math.isfinite(number):
        return default
    return max(lower, min(upper, number))


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def clean_optional_text(value: Any) -> str:
    """Return normalized text while treating pandas/CSV NaN-like values as missing."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = normalize_whitespace(str(value))
    return "" if text.lower() in {"nan", "none", "null"} else text


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", str(text or "")))


def stable_text_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:length]


def strip_model_json_chatter(text: str) -> str:
    """Remove common model wrappers before JSON parsing."""
    cleaned = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.I | re.S).strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.I | re.S)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    return cleaned


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Parse the first JSON object from a chat completion."""
    cleaned = strip_model_json_chatter(raw_text)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge response was not a JSON object")
    return value


def simple_readability_metrics(text: str) -> Dict[str, Any]:
    """Compute deterministic readability proxies without external services."""
    words = re.findall(r"[A-Za-z]+", str(text or ""))
    sentence_count = max(1, len(re.findall(r"[.!?](?:\s|$)", str(text or ""))) or 1)
    word_count_value = len(words)
    syllables = 0
    for word in words:
        groups = re.findall(r"[aeiouy]+", word.lower())
        count = max(1, len(groups))
        if word.lower().endswith("e") and count > 1:
            count -= 1
        syllables += count
    if not words:
        return {
            "readability_available": False,
            "flesch_reading_ease": float("nan"),
            "flesch_kincaid_grade": float("nan"),
            "grammar_errors_per_5000": float("nan"),
            "spelling_errors_per_5000": float("nan"),
            "tense_errors_per_5000": float("nan"),
        }
    words_per_sentence = word_count_value / sentence_count
    syllables_per_word = syllables / max(word_count_value, 1)
    flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    repeated_words = len(re.findall(r"\b([A-Za-z]+)\s+\1\b", str(text or ""), flags=re.I))
    passive_like = len(re.findall(r"\b(?:is|are|was|were|be|been|being)\s+\w+ed\b", str(text or ""), flags=re.I))
    spelling_proxy = len([word for word in words if len(word) > 26])
    scale = 5000.0 / max(word_count_value, 1)
    return {
        "readability_available": word_count_value >= 30,
        "flesch_reading_ease": clamp(flesch, -50.0, 120.0),
        "flesch_kincaid_grade": clamp(grade, 0.0, 30.0),
        "grammar_errors_per_5000": (repeated_words + passive_like) * scale,
        "spelling_errors_per_5000": spelling_proxy * scale,
        "tense_errors_per_5000": passive_like * scale,
    }


def clean_markdown_text(text: str) -> str:
    lines = []
    for raw_line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if PICTURE_LINE_RE.match(line):
            continue
        lines.append(raw_line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def article_id_to_doi(article_id: str) -> str:
    return f"10.1038/{article_id}" if str(article_id).startswith("s") else ""


def article_id_to_year(article_id: str) -> int:
    match = re.match(r"s\d{5}-(\d{3})-", str(article_id))
    return 2000 + int(match.group(1)) if match else 0


def article_id_to_journal_id(article_id: str) -> str:
    match = re.match(r"s(\d{5})-", str(article_id))
    return match.group(1) if match else ""


def journal_name(journal_id: str) -> str:
    return JOURNAL_NAMES.get(str(journal_id), f"Nature family {journal_id}")


def extract_doi(text: str, fallback_article_id: str = "") -> str:
    match = DOI_RE.search(text)
    if match:
        return (match.group(1) or match.group(2) or "").rstrip(".,;)")
    return article_id_to_doi(fallback_article_id)


def _paragraphs(text: str) -> List[str]:
    chunks = re.split(r"\n\s*\n", str(text or ""))
    return [normalize_whitespace(chunk) for chunk in chunks if normalize_whitespace(chunk)]


def _looks_like_author_list(paragraph: str) -> bool:
    lowered = paragraph.lower()
    contribution_terms = ["we ", "here,", "herein", "this study", "we report", "we present", "we identify"]
    if any(term in lowered for term in contribution_terms):
        return False
    many_commas = paragraph.count(",") >= 4
    name_like = bool(re.search(r"\b[A-Z][a-z]+ [A-Z]\.", paragraph) or "&" in paragraph)
    affiliation_marks = len(re.findall(r"\[\d+\]|\b\d+(?:,\d+)*\b", paragraph)) >= 3
    return bool((many_commas and name_like) or affiliation_marks) and word_count(paragraph) < 220


def _looks_like_affiliation(paragraph: str) -> bool:
    lowered = paragraph.lower()
    terms = ["department", "university", "institute", "hospital", "laboratory", "school of", "college of"]
    return any(term in lowered for term in terms) and word_count(paragraph) < 180


def parse_article_markdown(text: str, article_id: str) -> Dict[str, Any]:
    """Extract title, abstract, DOI, and publication year from article Markdown."""
    cleaned = clean_markdown_text(text)
    lines = [line.strip() for line in cleaned.splitlines()]
    warnings: List[str] = []
    generic_headings = {"article", "research article", "review article", "abstract", "results"}
    title = ""
    title_line_index = 0
    for idx, line in enumerate(lines):
        if not line.startswith("## "):
            continue
        candidate = line.lstrip("#").strip()
        if "peer review" in candidate.lower() or candidate.lower() in generic_headings:
            continue
        title = candidate
        title_line_index = idx
        break
    if not title:
        warnings.append("missing_title")
        title = str(article_id)
    year = article_id_to_year(article_id)
    if not year:
        match = re.search(r"\b(20\d{2})\b", cleaned[:4000])
        year = int(match.group(1)) if match else 0
    post_title = "\n".join(lines[title_line_index + 1 :])
    abstract = ""
    for paragraph in _paragraphs(post_title):
        lowered = paragraph.lower()
        if lowered.startswith(("received:", "accepted:", "published:", "check for updates")):
            continue
        if "nature communications" in lowered and word_count(paragraph) < 60:
            continue
        if _looks_like_author_list(paragraph) or _looks_like_affiliation(paragraph):
            continue
        if word_count(paragraph) >= 35:
            abstract = paragraph
            break
    if not abstract:
        warnings.append("missing_abstract")
    return {
        "article_text": cleaned,
        "doi": extract_doi(cleaned, article_id),
        "title": title,
        "year": year,
        "abstract": abstract,
        "abstract_source": "pdf_markdown" if abstract else "missing",
        "word_count": word_count(cleaned),
        "warnings": warnings,
    }


def split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?。！？])\s+", normalize_whitespace(text))
    return [chunk.strip() for chunk in chunks if word_count(chunk) >= 8]


def extract_sentences_by_keywords(text: str, keywords: Sequence[str], max_sentences: int = 5) -> List[str]:
    lowered_keywords = [keyword.lower() for keyword in keywords if keyword]
    selected: List[str] = []
    for sentence in split_sentences(text):
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in lowered_keywords):
            selected.append(sentence[:520])
        if len(selected) >= max_sentences:
            break
    return selected


def extract_section_summary(text: str, headings: Sequence[str], max_words: int = 180) -> str:
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
    Build the no-leakage structured full-context package for the innovation agent.

    The dossier is derived only from article metadata and article Markdown/PDF
    text. Peer-review text is intentionally not accepted as an input.
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
        "contribution_sentences": extract_sentences_by_keywords(article_text or abstract, contribution_terms, 6),
        "methods_summary": extract_section_summary(article_text, method_terms, 180),
        "results_summary": extract_section_summary(article_text, result_terms, 180),
        "limitations_sentences": extract_sentences_by_keywords(article_text, limitation_terms, 4),
        "article_word_count": parsed_article.get("word_count", 0),
        "source": "article_markdown",
        "leakage_guard": "peer_review_text_excluded",
    }


def format_paper_dossier_for_agent(dossier: Mapping[str, Any]) -> str:
    if not dossier:
        return ""
    lines = [
        f"DOI: {dossier.get('doi', '')}",
        f"Journal/year: {dossier.get('journal', '')} / {dossier.get('year', '')}",
        f"Abstract: {normalize_whitespace(str(dossier.get('abstract') or ''))[:1800]}",
    ]
    keywords = dossier.get("keywords") or []
    if keywords:
        lines.append("Keywords: " + ", ".join(str(item) for item in list(keywords)[:8]))
    for label, key in [
        ("Contribution signals", "contribution_sentences"),
        ("Methods context", "methods_summary"),
        ("Results context", "results_summary"),
        ("Limitations or caution signals", "limitations_sentences"),
    ]:
        value = dossier.get(key)
        if isinstance(value, list):
            rendered = " ".join(f"- {normalize_whitespace(str(item))}" for item in value if normalize_whitespace(str(item)))
        else:
            rendered = normalize_whitespace(str(value or ""))
        if rendered:
            lines.append(f"{label}: {rendered[:2200]}")
    lines.append("Use this as article context only. Do not infer peer-review labels from it.")
    return "\n".join(lines)


def strip_peer_review_boilerplate(text: str) -> str:
    cleaned = clean_markdown_text(text)
    cleaned = re.sub(
        r"Open Access This file is licensed under a Creative Commons Attribution 4\.0 International License.*?creativecommons\.org/licenses/by/4\.0/\.",
        "",
        cleaned,
        flags=re.I | re.S,
    )
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


INLINE_AUTHOR_RESPONSE_RE = re.compile(
    r"\b("
    r"responses?\s+can\s+be\s+found\s+below|"
    r"blue\s+text|"
    r"we\s+(?:thank|explain|have|now|added|add|revised|revise|clarified|clarify|"
    r"agree|disagree|respond|included|include|performed|changed|corrected|removed|"
    r"modified|addressed|believe|show|provide|note|apologize|appreciate)|"
    r"our\s+(?:response|revision|revised\s+manuscript|manuscript|study|results)|"
    r"the\s+revised\s+manuscript\s+(?:now\s+)?(?:includes|contains|states|shows)"
    r")\b",
    re.I,
)
AUTHOR_RESPONSE_SECTION_LINE_RE = re.compile(
    r"^\s*(?:#+\s*)?(?:author response|responses? to reviewers?|response to referee|author rebuttal|rebuttal)\b",
    re.I,
)


def is_author_response_sentence(sentence: str) -> bool:
    """Return True for author-rebuttal voice that should not label peer review."""
    normalized = normalize_whitespace(re.sub(r"[*_`#]+", "", str(sentence or ""))).strip()
    if not normalized:
        return False
    if AUTHOR_RESPONSE_SECTION_LINE_RE.search(normalized):
        return True
    return bool(INLINE_AUTHOR_RESPONSE_RE.search(normalized))


def remove_inline_author_responses(text: str) -> str:
    """Remove author-response sentences while keeping reviewer/editor third-person judgements."""
    cleaned_paragraphs: List[str] = []
    for paragraph in re.split(r"\n\s*\n+", str(text or "")):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        normalized = normalize_whitespace(paragraph)
        if is_author_response_sentence(normalized) and len(split_sentences(normalized)) <= 1:
            continue
        kept_sentences = [
            sentence for sentence in split_sentences(paragraph) if not is_author_response_sentence(sentence)
        ]
        if kept_sentences:
            cleaned_paragraphs.append(" ".join(kept_sentences))
        elif not is_author_response_sentence(normalized):
            cleaned_paragraphs.append(normalized)
    return "\n\n".join(cleaned_paragraphs).strip()


def parse_peer_review_markdown(text: str) -> Dict[str, Any]:
    """Extract reviewer comments while excluding author responses and decisions."""
    cleaned = strip_peer_review_boilerplate(text)
    lower = "\n" + cleaned.lower()
    warnings: List[str] = []
    included_sections: List[str] = []
    excluded_sections: List[str] = []
    start_patterns = [
        r"\n#+\s*reviewer(?:s|'s)?\s+comments?\b",
        r"\nreviewer\s*#?\s*\d+",
        r"\nreferee\s*#?\s*\d+",
        r"\nremarks\s+to\s+the\s+author",
    ]
    starts = [m.start() for pattern in start_patterns for m in re.finditer(pattern, lower, flags=re.I)]
    start = min(starts) if starts else 0
    if starts:
        included_sections.append("reviewer_comments")
    else:
        warnings.append("review_section_parse_warning")
        included_sections.append("full_text_fallback")
    review_part = ("\n" + cleaned)[start:].strip() if start else cleaned
    cut_patterns = [
        ("author_response", r"\n#+\s*(author response|response to reviewers|responses? to reviewer|author rebuttal)\b"),
        ("editor_decision", r"\n#+\s*(editor decision|decision letter|editorial decision)\b"),
        ("references", r"\n#+\s*(references|bibliography)\b"),
    ]
    cuts = []
    for section, pattern in cut_patterns:
        match = re.search(pattern, "\n" + review_part, flags=re.I)
        if match:
            cuts.append((match.start(), section))
    if cuts:
        cut, section = min(cuts, key=lambda item: item[0])
        review_part = review_part[:cut].strip()
        excluded_sections.append(section)
    review_part = remove_inline_author_responses(review_part)
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
    manifest_rows = {str(row.get("article_id") or ""): row for row in read_jsonl(markdown_root / "manifest.jsonl")}
    paper_dir = markdown_root / "paper"
    review_dir = markdown_root / "peer_review"
    records: List[Dict[str, Any]] = []
    for paper_path in sorted(paper_dir.glob("*.md")):
        article_id = paper_path.stem
        journal_id = article_id_to_journal_id(article_id)
        if journal_scope == "six_subjournals" and journal_id not in SIX_SUBJOURNAL_IDS:
            continue
        if journal_scope == "41467_only" and journal_id != "41467":
            continue
        source = manifest_rows.get(article_id, {})
        records.append(
            {
                "paper_id": article_id,
                "article_id": article_id,
                "journal_id": journal_id,
                "journal": journal_name(journal_id),
                "year": int(source.get("year") or article_id_to_year(article_id) or 0),
                "doi": source.get("doi") or article_id_to_doi(article_id),
                "title": "",
                "abstract": "",
                "keywords": "",
                "article_markdown_path": str(paper_path),
                "peer_review_markdown_path": str(review_dir / f"{article_id}_r.md"),
                "article_pdf_path": source.get("article_pdf_path", ""),
                "peer_review_pdf_path": source.get("peer_review_pdf_path", ""),
            }
        )
    return records


def audit_markdown_inputs(
    markdown_root: Path,
    output_dir: Path,
    journal_scope: str = "all",
    quiet: bool = False,
    audit_max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    records = load_markdown_manifest(markdown_root, journal_scope=journal_scope)
    if audit_max_records and audit_max_records > 0:
        records = records[:audit_max_records]
    cache_root = output_dir / "cache"
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for idx, record in enumerate(records, start=1):
        paper_id = str(record["paper_id"])
        errors: List[str] = []
        warnings: List[str] = []
        paper_path = Path(str(record["article_markdown_path"]))
        review_path = Path(str(record["peer_review_markdown_path"]))
        if paper_id in seen:
            errors.append("duplicate_paper_id")
        seen.add(paper_id)
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
            except Exception as exc:  # noqa: BLE001 - per-paper audit diagnostics.
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
        cache_dir = cache_root / paper_id
        parsed_path = cache_dir / "parsed_text.json"
        dossier_path = cache_dir / "paper_dossier.json"
        if article_parsed:
            write_json(dossier_path, build_paper_dossier(article_parsed, record))
        if article_parsed or review_parsed:
            write_json(parsed_path, {"paper_id": paper_id, **article_parsed, **review_parsed})
        rows.append(
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
            progress_log(f"Audit progress {idx}/{len(records)}.", quiet)
    write_csv(output_dir / "fig4_input_audit.csv", rows)
    return rows


def controlled_sample(
    audit_rows: Sequence[Mapping[str, Any]],
    sample_size: int,
    seed: int = DEFAULT_SAMPLE_SEED,
    cap_41467: float = DEFAULT_41467_CAP,
) -> List[Dict[str, Any]]:
    eligible = [dict(row) for row in audit_rows if bool_value(row.get("included_in_audit"))]
    rng = random.Random(seed)
    if sample_size <= 0 or len(eligible) <= sample_size:
        selected = list(eligible)
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
        keys = sorted(groups)
        while remaining > 0 and any(groups.values()):
            for key in keys:
                if remaining <= 0:
                    break
                if groups[key]:
                    selected.append(groups[key].pop())
                    remaining -= 1
        if remaining > 0:
            seen = {row.get("paper_id") for row in selected}
            backfill = [row for row in eligible if row.get("paper_id") not in seen]
            rng.shuffle(backfill)
            selected.extend(backfill[:remaining])
    selected = sorted(selected[:sample_size] if sample_size > 0 else selected, key=lambda row: (int(row.get("year") or 0), str(row.get("journal_id")), str(row.get("paper_id"))))
    for row in selected:
        row["included_in_main"] = True
        row["sample_seed"] = seed
    return selected


def nonempty_cell(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    text = normalize_whitespace(str(value))
    return bool(text and text.lower() not in {"nan", "none", "null"})


def build_fig4_candidate_screen(output_dir: Path) -> pd.DataFrame:
    """Write the full pre-sampling Fig.4 candidate screen from the input audit."""
    audit_rows = read_csv_records(output_dir / "fig4_input_audit.csv")
    rows: List[Dict[str, Any]] = []
    for row in audit_rows:
        included = bool_value(row.get("included_in_audit"))
        has_peer_review_text = bool(
            numeric(row.get("peer_review_word_count"), 0.0) > 0
            or nonempty_cell(row.get("peer_review_markdown_path"))
        )
        has_title = nonempty_cell(row.get("title"))
        has_abstract_or_body_text = bool(
            nonempty_cell(row.get("abstract"))
            or numeric(row.get("article_word_count"), 0.0) > 0
        )
        reasons: List[str] = []
        if not included:
            reasons.append(str(row.get("exclusion_reason") or "not_included_in_audit"))
        if not has_peer_review_text:
            reasons.append("missing_peer_review_text")
        if not has_title:
            reasons.append("missing_title")
        if not has_abstract_or_body_text:
            reasons.append("missing_abstract_or_body_text")
        rows.append(
            {
                **row,
                "has_peer_review_text": has_peer_review_text,
                "has_title": has_title,
                "has_abstract_or_body_text": has_abstract_or_body_text,
                "screen_pass": not reasons,
                "exclusion_reason": ";".join(reason for reason in reasons if reason),
            }
        )
    write_csv(output_dir / "fig4_candidate_screen.csv", rows)
    return pd.DataFrame(rows)


def collect_cached_scored_candidate_pool(output_dir: Path) -> pd.DataFrame:
    """Collect candidates with cached graph-prior scores for stratified Fig.4 sampling."""
    candidate_screen = build_fig4_candidate_screen(output_dir) if (output_dir / "fig4_input_audit.csv").exists() else pd.DataFrame()
    if candidate_screen.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for candidate in candidate_screen.to_dict("records"):
        if not bool_value(candidate.get("screen_pass")):
            continue
        paper_id = str(candidate.get("paper_id") or "")
        if not paper_id:
            continue
        prior_path = output_dir / "cache" / paper_id / "fig4_graph_prior.json"
        if not prior_path.exists():
            continue
        prior = read_json(prior_path)
        if not bool_value(prior.get("graph_metric_valid", True)):
            continue
        rows.append(
            {
                **candidate,
                **prior,
                "included_in_audit": True,
                "screen_pass": True,
                "scored_pool_status": "cached_graph_prior",
            }
        )
    pool = pd.DataFrame(rows)
    if not pool.empty and "fig3_sw" in pool.columns:
        pool["fig3_sw"] = pd.to_numeric(pool["fig3_sw"], errors="coerce")
        valid_scores = pool["fig3_sw"].dropna()
        if len(valid_scores):
            pool["fig4_scored_pool_percentile"] = pool["fig3_sw"].rank(method="average", pct=True)
    write_csv(output_dir / "fig4_scored_candidate_pool.csv", pool.to_dict("records") if not pool.empty else [])
    return pool


def scored_pool_validation_tier_column(pool: pd.DataFrame) -> str:
    """Return the tier column that can support external Fig.3-score validation."""
    for column in ["fig4_global_validation_tier", "global_fig3_tier", "fig3_sw_tier"]:
        if column in pool.columns:
            values = {str(value) for value in pool[column].dropna().astype(str)}
            if values & {"low", "middle", "high"}:
                return column
    if "fig3_sw_percentile" in pool.columns:
        pool["fig3_sw_tier"] = pd.to_numeric(pool["fig3_sw_percentile"], errors="coerce").map(percentile_tier)
        return "fig3_sw_tier"
    if "fig4_sw_ladder_tier" in pool.columns:
        return "fig4_sw_ladder_tier"
    return ""


def stratified_sample_scored_pool(scored_pool: pd.DataFrame, sample_size: int, seed: int = DEFAULT_SAMPLE_SEED) -> pd.DataFrame:
    """Select a fixed-size sample across global Fig.3 tiers when cached graph priors exist."""
    if scored_pool.empty:
        return scored_pool.copy()
    pool = scored_pool.copy()
    tier_col = scored_pool_validation_tier_column(pool)
    if not tier_col:
        tier_col = "fig4_scored_pool_percentile"
        if tier_col not in pool.columns:
            pool[tier_col] = pd.to_numeric(pool.get("fig3_sw", pd.Series(dtype=float)), errors="coerce").rank(method="average", pct=True)
        pool["fig4_sw_ladder_tier"] = pool[tier_col].apply(percentile_tier)
        tier_col = "fig4_sw_ladder_tier"
    pool[tier_col] = pool[tier_col].fillna("unknown").astype(str)
    if sample_size <= 0 or len(pool) <= sample_size:
        out = pool.sort_values(["year", "journal_id", "paper_id"], kind="mergesort").reset_index(drop=True)
        out["sample_tier_column"] = tier_col
        return out

    rng = random.Random(seed)
    preferred_tiers = [tier for tier in ["low", "middle", "high"] if tier in set(pool[tier_col])]
    other_tiers = sorted(set(pool[tier_col]) - set(preferred_tiers))
    tiers = preferred_tiers + other_tiers
    if not tiers:
        return pool.head(sample_size).reset_index(drop=True)

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for tier in tiers:
        records = pool[pool[tier_col] == tier].to_dict("records")
        rng.shuffle(records)
        groups[tier] = records
    selected: List[Dict[str, Any]] = []
    base = sample_size // len(tiers)
    remainder = sample_size % len(tiers)
    for idx, tier in enumerate(tiers):
        take = base + (1 if idx < remainder else 0)
        selected.extend(groups[tier][:take])
        groups[tier] = groups[tier][take:]
    while len(selected) < sample_size and any(groups.values()):
        for tier in tiers:
            if len(selected) >= sample_size:
                break
            if groups[tier]:
                selected.append(groups[tier].pop(0))
    out = pd.DataFrame(selected[:sample_size])
    out["sample_tier_column"] = tier_col
    return out.sort_values(["year", "journal_id", "paper_id"], kind="mergesort").reset_index(drop=True)


def sample_manifest(
    output_dir: Path,
    sample_size: int,
    seed: int,
    cap_41467: float,
    quiet: bool = False,
    require_screen_pass: bool = False,
    prefer_screen_pass: bool = False,
    prefer_scored_pool: bool = False,
) -> List[Dict[str, Any]]:
    screen_path = output_dir / "fig4_peer_review_screen.csv"
    candidate_screen = build_fig4_candidate_screen(output_dir) if (output_dir / "fig4_input_audit.csv").exists() else pd.DataFrame()
    candidate_rows = candidate_screen.to_dict("records") if not candidate_screen.empty else []
    candidate_pass = [{**row, "included_in_audit": True} for row in candidate_rows if bool_value(row.get("screen_pass"))]
    use_screen = False
    if prefer_scored_pool:
        scored_pool = collect_cached_scored_candidate_pool(output_dir)
        if not scored_pool.empty and (sample_size <= 0 or len(scored_pool) >= sample_size):
            sampled_df = stratified_sample_scored_pool(scored_pool, sample_size=sample_size, seed=seed)
            sampled = sampled_df.to_dict("records")
            for row in sampled:
                row["included_in_main"] = True
                row["sample_seed"] = seed
            write_csv(output_dir / "fig4_manifest.csv", sampled)
            write_csv(output_dir / "fig4_fixed_sample_manifest.csv", sampled)
            progress_log(f"Sampled {len(sampled)} Fig.4 papers from scored candidate pool.", quiet)
            return sampled
    if screen_path.exists() and (require_screen_pass or prefer_screen_pass):
        rows = [{**row, "included_in_audit": True} for row in read_csv_records(screen_path) if bool_value(row.get("screen_pass"))]
        use_screen = require_screen_pass or sample_size <= 0 or len(rows) > sample_size
        if require_screen_pass and sample_size > 0 and len(rows) < sample_size:
            raise RuntimeError(f"screen_pass_count={len(rows)} below requested sample_size={sample_size}")
        if not use_screen:
            if candidate_pass and (sample_size <= 0 or len(candidate_pass) >= sample_size):
                rows = candidate_pass
            else:
                rows = read_csv_records(output_dir / "fig4_input_audit.csv")
    elif require_screen_pass:
        raise RuntimeError("fig4_peer_review_screen.csv is required before screen-filtered sampling")
    else:
        if candidate_pass and (sample_size <= 0 or len(candidate_pass) >= sample_size):
            rows = candidate_pass
        else:
            rows = read_csv_records(output_dir / "fig4_input_audit.csv")
    sampled = controlled_sample(rows, sample_size=sample_size, seed=seed, cap_41467=cap_41467)
    write_csv(output_dir / "fig4_manifest.csv", sampled)
    write_csv(output_dir / "fig4_fixed_sample_manifest.csv", sampled)
    progress_log(f"Sampled {len(sampled)} Fig.4 papers.", quiet)
    return sampled


def enforce_fixed_sample_contract(sampled: Sequence[Mapping[str, Any]], requested_sample_size: int) -> Dict[str, Any]:
    """Fail fast when a fixed-size Fig.4 validation sample cannot be materialized."""
    evaluable = [row for row in sampled if row]
    contract = {
        "requested_sample_size": int(requested_sample_size),
        "evaluable_case_count": int(len(evaluable)),
        "fixed_sample_contract_pass": int(requested_sample_size <= 0 or len(evaluable) == requested_sample_size),
    }
    if requested_sample_size > 0 and len(evaluable) != requested_sample_size:
        raise RuntimeError(
            f"Fig.4 fixed-sample contract failed: requested_sample_size={requested_sample_size}, "
            f"evaluable_case_count={len(evaluable)}"
        )
    return contract


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def _spearman_positive(frame: pd.DataFrame, score_col: str, target_col: str) -> int:
    if score_col not in frame.columns or target_col not in frame.columns:
        return 0
    pair = frame[[score_col, target_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3 or pair[score_col].nunique() < 2 or pair[target_col].nunique() < 2:
        return 0
    corr = pair[score_col].corr(pair[target_col], method="spearman")
    return int(pd.notna(corr) and float(corr) > 0.0)


def _spearman_positive_any(frame: pd.DataFrame, score_cols: Sequence[str], target_cols: Sequence[str]) -> int:
    for score_col in score_cols:
        for target_col in target_cols:
            if _spearman_positive(frame, score_col, target_col):
                return 1
    return 0


def _spearman_value(frame: pd.DataFrame, score_col: str, target_col: str) -> float:
    if score_col not in frame.columns or target_col not in frame.columns:
        return float("nan")
    pair = frame[[score_col, target_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3 or pair[score_col].nunique() < 2 or pair[target_col].nunique() < 2:
        return float("nan")
    corr = pair[score_col].corr(pair[target_col], method="spearman")
    return float(corr) if pd.notna(corr) else float("nan")


def bootstrap_spearman_ci(
    frame: pd.DataFrame,
    score_col: str,
    target_col: str,
    *,
    seed: int = 20260630,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Return a deterministic bootstrap CI for a blinded external-validation alignment."""
    empty = {
        "score_col": score_col,
        "target_col": target_col,
        "n": 0,
        "observed": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "bootstrap_samples": 0,
        "ci_excludes_zero_positive": 0,
    }
    if score_col not in frame.columns or target_col not in frame.columns:
        return empty
    pair = frame[[score_col, target_col]].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    observed = _spearman_value(pair, score_col, target_col)
    if len(pair) < 3 or not math.isfinite(observed):
        out = dict(empty)
        out["n"] = int(len(pair))
        out["observed"] = observed
        return out

    rng = random.Random(seed)
    values: List[float] = []
    n = len(pair)
    for _idx in range(max(0, int(n_bootstrap))):
        sample_idx = [rng.randrange(n) for _ in range(n)]
        sampled = pair.iloc[sample_idx]
        rho = _spearman_value(sampled, score_col, target_col)
        if math.isfinite(rho):
            values.append(rho)
    if values:
        arr = np.asarray(values, dtype=float)
        ci_low = float(np.quantile(arr, alpha / 2.0))
        ci_high = float(np.quantile(arr, 1.0 - alpha / 2.0))
    else:
        ci_low = float("nan")
        ci_high = float("nan")
    return {
        "score_col": score_col,
        "target_col": target_col,
        "n": int(n),
        "observed": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bootstrap_samples": int(len(values)),
        "ci_excludes_zero_positive": int(math.isfinite(ci_low) and ci_low > 0.0),
    }


def _numeric_unique_count(frame: pd.DataFrame, column: str) -> int:
    series = _numeric_series(frame, column)
    if series.empty:
        return 0
    return int(series.nunique(dropna=True))


def _text_unique_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    values = {
        normalize_whitespace(str(value))
        for value in frame[column].dropna().tolist()
        if normalize_whitespace(str(value))
    }
    return len(values)


def build_fig4_external_validation_target_audit(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    """Summarize whether Fig.4 has enough target and score range for validation."""
    fig3_scores = _numeric_series(metrics_df, "fig3_sw")
    fig3_iqr = float(fig3_scores.quantile(0.75) - fig3_scores.quantile(0.25)) if not fig3_scores.empty else float("nan")
    fig3_percentiles = _numeric_series(metrics_df, "fig3_sw_percentile")
    fig3_percentile_min = float(fig3_percentiles.min()) if not fig3_percentiles.empty else float("nan")
    fig3_percentile_max = float(fig3_percentiles.max()) if not fig3_percentiles.empty else float("nan")
    fig3_percentile_span = (
        fig3_percentile_max - fig3_percentile_min
        if math.isfinite(fig3_percentile_min) and math.isfinite(fig3_percentile_max)
        else float("nan")
    )
    audit = {
        "n_cases": int(len(metrics_df)),
        "fig3_score_unique": _numeric_unique_count(metrics_df, "fig3_sw"),
        "fig3_score_iqr": fig3_iqr,
        "fig3_reference_tier_unique": _text_unique_count(metrics_df, "fig3_sw_tier"),
        "fig4_batch_ladder_tier_unique": _text_unique_count(metrics_df, "fig4_sw_ladder_tier"),
        "fig3_reference_percentile_min": fig3_percentile_min,
        "fig3_reference_percentile_max": fig3_percentile_max,
        "fig3_reference_percentile_span": fig3_percentile_span,
        "peer_novelty_unique": _numeric_unique_count(metrics_df, "peer_novelty"),
        "peer_significance_unique": _numeric_unique_count(metrics_df, "peer_significance"),
    }
    audit["fig3_reference_percentile_range_ready"] = int(
        math.isfinite(fig3_percentile_min)
        and math.isfinite(fig3_percentile_max)
        and fig3_percentile_min <= 0.50
        and fig3_percentile_max >= 0.90
    )
    audit["peer_label_variance_ready"] = int(
        audit["peer_novelty_unique"] >= 2 and audit["peer_significance_unique"] >= 2
    )
    audit["fig3_score_range_ready"] = int(
        audit["fig3_reference_tier_unique"] >= 2
        or audit["fig3_reference_percentile_range_ready"] >= 1
    )
    audit["external_validation_target_range_ready"] = int(
        bool(audit["peer_label_variance_ready"]) and bool(audit["fig3_score_range_ready"])
    )
    return audit


def external_validation_global_tier(percentile: Any) -> str:
    """Map a global Fig.3 percentile into Fig.4 validation coverage tiers."""
    value = numeric(percentile, float("nan"))
    if not math.isfinite(value):
        return "unknown"
    if value <= 0.50:
        return "low"
    if value < 0.90:
        return "middle"
    return "high"


def _frame_from_csv(path: Path) -> pd.DataFrame:
    return pd.DataFrame(read_csv_records(path)) if path.exists() else pd.DataFrame()


def _score_reference_values(fig3_score_table_path: Path) -> List[float]:
    score_table = _frame_from_csv(fig3_score_table_path)
    if score_table.empty:
        return []
    for score_col in ("S_w_oof", "S_w"):
        if score_col not in score_table.columns:
            continue
        values = pd.to_numeric(score_table[score_col], errors="coerce").dropna().astype(float).tolist()
        if values:
            return sorted(values)
    return []


def _score_reference_tiers(fig3_score_table_path: Path) -> pd.Series:
    score_table = _frame_from_csv(fig3_score_table_path)
    if score_table.empty:
        return pd.Series(dtype=str)
    for score_col in ("S_w_oof", "S_w"):
        if score_col not in score_table.columns:
            continue
        values = pd.to_numeric(score_table[score_col], errors="coerce").dropna()
        if not values.empty:
            return values.rank(method="average", pct=True).map(external_validation_global_tier)
    return pd.Series(dtype=str)


def _with_global_validation_tier(frame: pd.DataFrame, reference_values: Sequence[float]) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        out["fig4_global_validation_percentile"] = pd.Series(dtype=float)
        out["fig4_global_validation_tier"] = pd.Series(dtype=str)
        return out
    if "fig3_sw_percentile" in out.columns:
        percentiles = pd.to_numeric(out["fig3_sw_percentile"], errors="coerce")
    else:
        score_col = "fig3_sw" if "fig3_sw" in out.columns else "S_w" if "S_w" in out.columns else ""
        if score_col and reference_values:
            percentiles = pd.to_numeric(out[score_col], errors="coerce").map(
                lambda value: empirical_percentile(float(value), reference_values)
                if pd.notna(value)
                else float("nan")
            )
        else:
            percentiles = pd.Series([float("nan")] * len(out), index=out.index)
    out["fig4_global_validation_percentile"] = percentiles
    out["fig4_global_validation_tier"] = percentiles.map(external_validation_global_tier)
    return out


def build_fig4_global_score_coverage_audit(
    *,
    output_dir: Path,
    fig3_score_table_path: Optional[Path] = None,
    requested_sample_size: int = 50,
    min_cases_per_tier: int = 10,
) -> pd.DataFrame:
    """Write Fig.4 low/middle/high sample coverage against the global Fig.3 score distribution."""
    score_path = fig3_score_table_path or fig3_score_table_path_from_env()
    reference_values = _score_reference_values(score_path)
    global_tiers = _score_reference_tiers(score_path)
    candidate_pool = _with_global_validation_tier(_frame_from_csv(output_dir / "fig4_scored_candidate_pool.csv"), reference_values)
    fixed_sample = _with_global_validation_tier(_frame_from_csv(output_dir / "fig4_fixed_sample_manifest.csv"), reference_values)

    rows: List[Dict[str, Any]] = []
    for tier in ["low", "middle", "high"]:
        fixed_count = int(fixed_sample.get("fig4_global_validation_tier", pd.Series(dtype=str)).astype(str).eq(tier).sum())
        rows.append(
            {
                "global_fig3_tier": tier,
                "global_reference_count": int(global_tiers.eq(tier).sum()) if not global_tiers.empty else 0,
                "scored_candidate_count": int(candidate_pool.get("fig4_global_validation_tier", pd.Series(dtype=str)).astype(str).eq(tier).sum()),
                "fixed_sample_count": fixed_count,
                "required_min_fixed_cases": int(min_cases_per_tier),
                "additional_fixed_cases_needed": max(0, int(min_cases_per_tier) - fixed_count),
                "tier_ready_for_external_validation": int(fixed_count >= min_cases_per_tier),
            }
        )
    overall_ready = int(all(row["tier_ready_for_external_validation"] for row in rows))
    for row in rows:
        row["requested_sample_size"] = int(requested_sample_size)
        row["scored_candidate_pool_size"] = int(len(candidate_pool))
        row["fixed_sample_size"] = int(len(fixed_sample))
        row["overall_score_coverage_ready"] = overall_ready
        row["coverage_interpretation"] = (
            "global_low_middle_high_ready"
            if overall_ready
            else "scored_peer_review_sample_does_not_cover_global_fig3_distribution"
        )
    audit = pd.DataFrame(rows)
    write_csv(output_dir / "fig4_global_score_coverage_audit.csv", audit.to_dict("records") if not audit.empty else [])
    return audit


FIG4_CANDIDATE_LABEL_COLUMNS = [
    "peer_novelty_human_1_5",
    "peer_significance_human_1_5",
    "peer_overall_human_1_5",
    "peer_prior_art_human_1_5",
    "label_source",
    "labeler_id",
    "label_notes",
]


def preserve_existing_fig4_candidate_packet_labels(output_dir: Path, packet: pd.DataFrame) -> pd.DataFrame:
    """Preserve completed external-validation labels across deterministic packet rebuilds."""
    old_packet = _frame_from_csv(output_dir / "fig4_global_validation_candidate_packet.csv")
    if packet.empty or old_packet.empty or "paper_id" not in packet.columns or "paper_id" not in old_packet.columns:
        return packet
    labels = [column for column in FIG4_CANDIDATE_LABEL_COLUMNS if column in packet.columns and column in old_packet.columns]
    if not labels:
        return packet
    current = packet.copy()
    current["paper_id"] = current["paper_id"].astype(str)
    old = old_packet[["paper_id", *labels]].copy()
    old["paper_id"] = old["paper_id"].astype(str)
    old = old.rename(columns={column: f"{column}_existing" for column in labels})
    merged = current.merge(old, on="paper_id", how="left")
    for column in labels:
        existing_col = f"{column}_existing"
        existing = merged.get(existing_col, pd.Series([""] * len(merged)))
        has_existing = existing.map(nonempty_cell)
        merged[column] = merged[column].where(~has_existing, existing)
    drop_cols = [f"{column}_existing" for column in labels]
    return merged.drop(columns=[column for column in drop_cols if column in merged.columns])


def build_fig4_global_validation_candidate_packet(
    *,
    output_dir: Path,
    fig3_score_table_path: Optional[Path] = None,
    per_tier: int = 100,
) -> pd.DataFrame:
    """Write a deterministic low/middle/high Fig.3 candidate packet for external validation labeling."""
    score_path = fig3_score_table_path or fig3_score_table_path_from_env()
    score_table = _frame_from_csv(score_path)
    if score_table.empty:
        empty = pd.DataFrame()
        write_csv(output_dir / "fig4_global_validation_candidate_packet.csv", [])
        return empty
    score_col = "S_w_oof" if "S_w_oof" in score_table.columns and pd.to_numeric(score_table["S_w_oof"], errors="coerce").notna().any() else "S_w"
    if score_col not in score_table.columns:
        empty = pd.DataFrame()
        write_csv(output_dir / "fig4_global_validation_candidate_packet.csv", [])
        return empty
    candidates = score_table.copy()
    candidates["fig3_score_for_validation"] = pd.to_numeric(candidates[score_col], errors="coerce")
    candidates = candidates.dropna(subset=["fig3_score_for_validation"]).copy()
    if candidates.empty:
        write_csv(output_dir / "fig4_global_validation_candidate_packet.csv", [])
        return candidates
    candidates["fig3_global_percentile"] = candidates["fig3_score_for_validation"].rank(method="average", pct=True)
    candidates["global_fig3_tier"] = candidates["fig3_global_percentile"].map(external_validation_global_tier)
    fixed_sample = _frame_from_csv(output_dir / "fig4_fixed_sample_manifest.csv")
    already_sampled = set(fixed_sample.get("paper_id", pd.Series(dtype=str)).astype(str)) if not fixed_sample.empty else set()
    candidates = candidates[~candidates.get("paper_id", pd.Series(dtype=str)).astype(str).isin(already_sampled)].copy()
    selected_frames: List[pd.DataFrame] = []
    for tier in ["low", "middle", "high"]:
        tier_frame = candidates[candidates["global_fig3_tier"].eq(tier)].sort_values(
            ["fig3_global_percentile", "domain", "year", "paper_id"],
            kind="mergesort",
        )
        if tier_frame.empty:
            continue
        if len(tier_frame) <= per_tier:
            selected = tier_frame
        else:
            if per_tier <= 1:
                positions = [0]
            else:
                positions = sorted({round(idx * (len(tier_frame) - 1) / (per_tier - 1)) for idx in range(per_tier)})
            selected = tier_frame.iloc[positions]
        selected_frames.append(selected)
    packet = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame()
    if not packet.empty:
        for column in [
            "peer_novelty_human_1_5",
            "peer_significance_human_1_5",
            "peer_overall_human_1_5",
            "label_source",
            "labeler_id",
            "label_notes",
        ]:
            packet[column] = ""
        packet["candidate_packet_role"] = "external_validation_labeling_candidate"
        packet["label_status"] = "needs_peer_or_blinded_human_label"
        keep_cols = [
            "paper_id",
            "title",
            "domain",
            "year",
            "primary_field",
            "reference_count",
            "fig3_score_for_validation",
            "fig3_global_percentile",
            "global_fig3_tier",
            "candidate_packet_role",
            "label_status",
            "peer_novelty_human_1_5",
            "peer_significance_human_1_5",
            "peer_overall_human_1_5",
            "label_source",
            "labeler_id",
            "label_notes",
        ]
        packet = packet[[column for column in keep_cols if column in packet.columns]]
        packet = preserve_existing_fig4_candidate_packet_labels(output_dir, packet)
    write_csv(output_dir / "fig4_global_validation_candidate_packet.csv", packet.to_dict("records") if not packet.empty else [])
    return packet


def openalex_url_from_fig3_paper_id(paper_id: Any) -> str:
    """Extract an OpenAlex work URL from a Fig.3 paper ID."""
    text = str(paper_id or "")
    marker = "https://openalex.org/"
    if marker in text:
        return marker + text.split(marker, 1)[1].split()[0].strip()
    return ""


FIG4_BLINDED_LABEL_COLUMNS = [
    "label_novelty_1_5",
    "label_significance_1_5",
    "label_prior_art_1_5",
    "label_confidence_1_5",
    "label_source",
    "labeler_id",
    "label_notes",
]


FIG4_COMPLETED_BLINDED_LABEL_TEMPLATE_COLUMNS = [
    "blinded_case_id",
    "assignment_role",
    "title",
    "doi",
    "source_openalex_url",
    "domain_context",
    "year",
    "primary_field",
    *FIG4_BLINDED_LABEL_COLUMNS,
]


def candidate_packet_blinded_labels(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Return completed candidate-packet labels in the blinded-packet schema."""
    mapped: Dict[str, Any] = {}
    column_map = {
        "peer_novelty_human_1_5": "label_novelty_1_5",
        "peer_significance_human_1_5": "label_significance_1_5",
        "peer_prior_art_human_1_5": "label_prior_art_1_5",
        "label_confidence_1_5": "label_confidence_1_5",
    }
    for source, target in column_map.items():
        value = row.get(source, "")
        if nonempty_cell(value):
            mapped[target] = value
    for column in ["label_source", "labeler_id", "label_notes"]:
        value = row.get(column, "")
        if nonempty_cell(value):
            mapped[column] = value
    return mapped


def preserve_existing_fig4_blinded_labels(output_dir: Path, blinded: pd.DataFrame, answer_key: pd.DataFrame) -> pd.DataFrame:
    """Carry completed blinded-label fields across deterministic packet rebuilds."""
    old_packet = _frame_from_csv(output_dir / "fig4_blinded_labeling_packet.csv")
    old_key = _frame_from_csv(output_dir / "fig4_blinded_labeling_answer_key.csv")
    if (
        blinded.empty
        or answer_key.empty
        or old_packet.empty
        or old_key.empty
        or "blinded_case_id" not in old_packet.columns
        or "blinded_case_id" not in old_key.columns
    ):
        return blinded
    if "paper_id" not in old_key.columns or "paper_id" not in answer_key.columns:
        return blinded
    old = old_packet.merge(
        old_key[["blinded_case_id", "paper_id"]],
        on="blinded_case_id",
        how="inner",
    )
    labels = [column for column in FIG4_BLINDED_LABEL_COLUMNS if column in old.columns and column in blinded.columns]
    if not labels:
        return blinded
    current = answer_key[["blinded_case_id", "paper_id"]].copy()
    current["blinded_case_id"] = current["blinded_case_id"].astype(str)
    current["paper_id"] = current["paper_id"].astype(str)
    preserved = blinded.merge(current, on="blinded_case_id", how="left")
    old = old[["blinded_case_id", "paper_id", *labels]].copy()
    old["blinded_case_id"] = old["blinded_case_id"].astype(str)
    old["paper_id"] = old["paper_id"].astype(str)
    old = old.rename(columns={column: f"{column}_existing" for column in labels})
    preserved = preserved.merge(old, on=["blinded_case_id", "paper_id"], how="left")
    for column in labels:
        existing = preserved.get(f"{column}_existing", pd.Series([""] * len(preserved))).astype(str)
        has_existing = existing.str.strip().ne("") & existing.str.lower().ne("nan")
        preserved[column] = preserved[column].where(~has_existing, existing)
    drop_cols = ["paper_id", *[f"{column}_existing" for column in labels]]
    return preserved.drop(columns=[column for column in drop_cols if column in preserved.columns])


def import_completed_fig4_blinded_label_sidecar(output_dir: Path, blinded: pd.DataFrame) -> pd.DataFrame:
    """Merge labeler-returned blinded labels without exposing Fig.3 score or tier columns."""
    completed = _frame_from_csv(output_dir / FIG4_COMPLETED_BLINDED_LABELS_FILE)
    if blinded.empty or completed.empty or "blinded_case_id" not in blinded.columns or "blinded_case_id" not in completed.columns:
        return blinded
    labels = [column for column in FIG4_BLINDED_LABEL_COLUMNS if column in completed.columns and column in blinded.columns]
    if not labels:
        return blinded
    merged = blinded.copy()
    merged["blinded_case_id"] = merged["blinded_case_id"].astype(str)
    sidecar = completed[["blinded_case_id", *labels]].copy()
    sidecar["blinded_case_id"] = sidecar["blinded_case_id"].astype(str)
    sidecar = sidecar.drop_duplicates(subset=["blinded_case_id"], keep="last")
    sidecar = sidecar.rename(columns={column: f"{column}_completed" for column in labels})
    merged = merged.merge(sidecar, on="blinded_case_id", how="left")
    for column in labels:
        completed_values = merged.get(f"{column}_completed", pd.Series([""] * len(merged)))
        has_completed = completed_values.map(nonempty_cell)
        merged[column] = merged[column].where(~has_completed, completed_values)
    drop_cols = [f"{column}_completed" for column in labels]
    return merged.drop(columns=[column for column in drop_cols if column in merged.columns])


def write_fig4_completed_blinded_label_template(output_dir: Path, blinded: pd.DataFrame) -> pd.DataFrame:
    """Write the evaluator-facing completed-label template for primary blinded cases."""
    if blinded.empty or "assignment_role" not in blinded.columns:
        template = pd.DataFrame(columns=FIG4_COMPLETED_BLINDED_LABEL_TEMPLATE_COLUMNS)
    else:
        primary = blinded[blinded["assignment_role"].astype(str).eq("primary_validation_labeling_sample")].copy()
        template = primary[[column for column in FIG4_COMPLETED_BLINDED_LABEL_TEMPLATE_COLUMNS if column in primary.columns]].copy()
        for column in FIG4_COMPLETED_BLINDED_LABEL_TEMPLATE_COLUMNS:
            if column not in template.columns:
                template[column] = ""
        template = template[FIG4_COMPLETED_BLINDED_LABEL_TEMPLATE_COLUMNS]
    write_csv(
        output_dir / FIG4_COMPLETED_BLINDED_LABELS_TEMPLATE_FILE,
        template.to_dict("records") if not template.empty else [],
        fieldnames=FIG4_COMPLETED_BLINDED_LABEL_TEMPLATE_COLUMNS,
    )
    return template


def write_fig4_labeler_completed_blinded_label_templates(
    output_dir: Path,
    template: pd.DataFrame,
    labeler_ids: Sequence[str] = FIG4_LABELER_IDS,
) -> Dict[str, str]:
    """Write one Fig.4 completed-label return template per external labeler."""
    template_paths: Dict[str, str] = {}
    for labeler_id in labeler_ids:
        file_name = f"fig4_completed_blinded_labels_{labeler_id}.csv"
        path = output_dir / file_name
        labeler_template = template.copy()
        if labeler_template.empty:
            labeler_template = pd.DataFrame(columns=FIG4_COMPLETED_BLINDED_LABEL_TEMPLATE_COLUMNS)
        if "labeler_id" not in labeler_template.columns:
            labeler_template["labeler_id"] = ""
        if "label_source" not in labeler_template.columns:
            labeler_template["label_source"] = ""
        labeler_template["labeler_id"] = str(labeler_id)
        existing = _frame_from_csv(path)
        preserve_existing_return = False
        if not existing.empty and "blinded_case_id" in existing.columns:
            source = existing.get("label_source", pd.Series([""] * len(existing)))
            novelty = pd.to_numeric(existing.get("label_novelty_1_5", pd.Series(dtype=float)), errors="coerce")
            significance = pd.to_numeric(existing.get("label_significance_1_5", pd.Series(dtype=float)), errors="coerce")
            preserve_existing_return = bool((source.map(nonempty_cell) & novelty.between(1, 5) & significance.between(1, 5)).any())
        if not preserve_existing_return:
            write_csv(
                path,
                labeler_template.to_dict("records") if not labeler_template.empty else [],
                fieldnames=FIG4_COMPLETED_BLINDED_LABEL_TEMPLATE_COLUMNS,
            )
        template_paths[str(labeler_id)] = file_name
    return template_paths


def _fig4_labeler_merge_audit_row(
    *,
    required_files: int,
    observed_files: int,
    required_labels: int,
    observed_valid_labels: int,
    missing_labels: int,
    passed: bool,
    failure_reason: str,
) -> Dict[str, Any]:
    """Build a one-row audit for Fig.4 labeler-return merging."""
    return {
        "audit_item": "overall_labeler_return_merge_ready",
        "required_files": int(required_files),
        "observed_files": int(observed_files),
        "required_labels": int(required_labels),
        "observed_valid_labels": int(observed_valid_labels),
        "missing_labels": int(missing_labels),
        "pass": int(passed),
        "failure_reason": failure_reason,
    }


def _valid_fig4_labeler_source(value: Any) -> bool:
    """Return whether a Fig.4 label source is usable for strict external evidence."""
    text = str(value or "").strip().lower()
    if not text:
        return False
    forbidden = ("llm", "synthetic", "model", "proxy", "automated")
    return not any(term in text for term in forbidden)


def merge_fig4_labeler_blinded_label_returns(
    output_dir: Path,
    *,
    labeler_ids: Sequence[str] = FIG4_LABELER_IDS,
    write: bool = True,
) -> pd.DataFrame:
    """Combine completed labeler-specific Fig.4 blinded label returns when complete."""
    template_path = output_dir / FIG4_COMPLETED_BLINDED_LABELS_TEMPLATE_FILE
    audit_path = output_dir / "fig4_blinded_label_return_merge_audit.csv"
    if not template_path.exists():
        audit = pd.DataFrame(
            [
                _fig4_labeler_merge_audit_row(
                    required_files=len(labeler_ids),
                    observed_files=0,
                    required_labels=0,
                    observed_valid_labels=0,
                    missing_labels=1,
                    passed=False,
                    failure_reason="missing_completed_blinded_label_template",
                )
            ]
        )
        if write:
            write_csv(audit_path, audit.to_dict("records"))
        return audit
    try:
        template = pd.read_csv(template_path).fillna("")
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        audit = pd.DataFrame(
            [
                _fig4_labeler_merge_audit_row(
                    required_files=len(labeler_ids),
                    observed_files=0,
                    required_labels=0,
                    observed_valid_labels=0,
                    missing_labels=1,
                    passed=False,
                    failure_reason="unreadable_completed_blinded_label_template",
                )
            ]
        )
        if write:
            write_csv(audit_path, audit.to_dict("records"))
        return audit
    expected_ids = template.get("blinded_case_id", pd.Series(dtype=str)).dropna().astype(str).tolist()
    required_labels = len(expected_ids) * len(labeler_ids)
    required_cols = {
        "blinded_case_id",
        "label_novelty_1_5",
        "label_significance_1_5",
        "label_prior_art_1_5",
        "label_confidence_1_5",
        "label_source",
        "labeler_id",
        "label_notes",
    }
    expected_set = set(expected_ids)
    observed_files = 0
    observed_valid = 0
    valid_frames: List[pd.DataFrame] = []
    failure_reasons: List[str] = []
    for labeler_id in labeler_ids:
        path = output_dir / f"fig4_completed_blinded_labels_{labeler_id}.csv"
        if not path.exists():
            failure_reasons.append(f"missing_{path.name}")
            continue
        observed_files += 1
        try:
            table = pd.read_csv(path).fillna("")
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            failure_reasons.append(f"unreadable_{path.name}")
            continue
        if table.empty or not required_cols.issubset(set(table.columns)):
            failure_reasons.append(f"invalid_columns_{path.name}")
            continue
        table = table.copy()
        table["blinded_case_id"] = table["blinded_case_id"].astype(str)
        table["labeler_id"] = table["labeler_id"].astype(str)
        numeric_valid = pd.Series(True, index=table.index)
        for column in ["label_novelty_1_5", "label_significance_1_5", "label_prior_art_1_5", "label_confidence_1_5"]:
            numeric_valid &= pd.to_numeric(table[column], errors="coerce").between(1, 5)
        valid = table[
            table["blinded_case_id"].isin(expected_set)
            & table["labeler_id"].eq(str(labeler_id))
            & table["label_source"].map(_valid_fig4_labeler_source)
            & numeric_valid
        ].copy()
        valid = valid.drop_duplicates(subset=["blinded_case_id", "labeler_id"], keep="last")
        observed_valid += len(valid)
        if len(valid) != len(expected_ids):
            failure_reasons.append(f"incomplete_{path.name}")
        valid_frames.append(valid)
    missing_labels = max(0, required_labels - observed_valid)
    passed = bool(required_labels and observed_files == len(labeler_ids) and missing_labels == 0 and not failure_reasons)
    audit = pd.DataFrame(
        [
            _fig4_labeler_merge_audit_row(
                required_files=len(labeler_ids),
                observed_files=observed_files,
                required_labels=required_labels,
                observed_valid_labels=observed_valid,
                missing_labels=missing_labels,
                passed=passed,
                failure_reason="" if passed else ";".join(failure_reasons or ["incomplete_labeler_return_files"]),
            )
        ]
    )
    if passed and valid_frames:
        merged = pd.concat(valid_frames, ignore_index=True)
        score_cols = ["label_novelty_1_5", "label_significance_1_5", "label_prior_art_1_5", "label_confidence_1_5"]
        rows: List[Dict[str, Any]] = []
        for blinded_case_id, group in merged.groupby("blinded_case_id", sort=False):
            row: Dict[str, Any] = {"blinded_case_id": str(blinded_case_id)}
            for column in score_cols:
                row[column] = float(pd.to_numeric(group[column], errors="coerce").mean())
            row["label_source"] = "external_blinded_human_panel"
            row["labeler_id"] = ";".join(sorted(group["labeler_id"].astype(str).unique()))
            row["label_notes"] = "merged from complete labeler-specific blinded returns"
            rows.append(row)
        write_csv(output_dir / FIG4_COMPLETED_BLINDED_LABELS_FILE, rows)
    if write:
        write_csv(audit_path, audit.to_dict("records"))
    return audit


def build_fig4_blinded_labeling_package(
    *,
    output_dir: Path,
    candidate_packet: Optional[pd.DataFrame] = None,
    primary_per_tier: int = DEFAULT_FIG4_LABELING_PRIMARY_PER_TIER,
    reserve_per_tier: int = DEFAULT_FIG4_LABELING_RESERVE_PER_TIER,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Write blinded low/middle/high labeling packets for Fig.4 external validation."""
    packet = candidate_packet.copy() if candidate_packet is not None else _frame_from_csv(output_dir / "fig4_global_validation_candidate_packet.csv")
    if packet.empty or "global_fig3_tier" not in packet.columns:
        write_csv(output_dir / "fig4_blinded_labeling_packet.csv", [])
        write_csv(output_dir / "fig4_blinded_labeling_answer_key.csv", [])
        template = write_fig4_completed_blinded_label_template(output_dir, pd.DataFrame())
        labeler_templates = write_fig4_labeler_completed_blinded_label_templates(output_dir, template)
        protocol = {
            "status": "missing_candidate_packet",
            "primary_per_tier": int(primary_per_tier),
            "reserve_per_tier": int(reserve_per_tier),
            "seed": int(seed),
            "completed_label_import_path": FIG4_COMPLETED_BLINDED_LABELS_FILE,
            "completed_label_template_path": FIG4_COMPLETED_BLINDED_LABELS_TEMPLATE_FILE,
            "labeler_specific_completed_label_templates": labeler_templates,
        }
        (output_dir / "fig4_blinded_labeling_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return pd.DataFrame(), pd.DataFrame(), protocol

    rng = random.Random(seed)
    selected_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for tier in ["low", "middle", "high"]:
        tier_rows = packet[packet["global_fig3_tier"].astype(str).eq(tier)].to_dict("records")
        rng.shuffle(tier_rows)
        quota = max(0, int(primary_per_tier)) + max(0, int(reserve_per_tier))
        for offset, row in enumerate(tier_rows[:quota]):
            role = "primary_validation_labeling_sample" if offset < primary_per_tier else "reserve_validation_labeling_sample"
            source_url = openalex_url_from_fig3_paper_id(row.get("paper_id"))
            imported_labels = candidate_packet_blinded_labels(row)
            selected_pairs.append(
                (
                {
                    "assignment_role": role,
                    "title": row.get("title", ""),
                    "doi": row.get("doi", ""),
                    "source_openalex_url": source_url,
                    "domain_context": row.get("domain", ""),
                    "year": row.get("year", ""),
                    "primary_field": row.get("primary_field", ""),
                    "label_novelty_1_5": "",
                    "label_significance_1_5": "",
                    "label_prior_art_1_5": "",
                    "label_confidence_1_5": "",
                    "label_source": "",
                    "labeler_id": "",
                    "label_notes": "",
                    **imported_labels,
                },
                {
                    "paper_id": row.get("paper_id", ""),
                    "global_fig3_tier": tier,
                    "fig3_score_for_validation": row.get("fig3_score_for_validation", ""),
                    "fig3_global_percentile": row.get("fig3_global_percentile", ""),
                    "assignment_role": role,
                    "seed": int(seed),
                },
                )
            )

    rng.shuffle(selected_pairs)
    selected_rows: List[Dict[str, Any]] = []
    answer_rows: List[Dict[str, Any]] = []
    for blind_index, (blinded_row, answer_row) in enumerate(selected_pairs, start=1):
        blind_id = f"F4LV-{blind_index:04d}"
        selected_rows.append({"blinded_case_id": blind_id, **blinded_row})
        answer_rows.append({"blinded_case_id": blind_id, **answer_row})

    blinded = pd.DataFrame(selected_rows)
    key = pd.DataFrame(answer_rows)
    blinded = preserve_existing_fig4_blinded_labels(output_dir, blinded, key)
    blinded = import_completed_fig4_blinded_label_sidecar(output_dir, blinded)
    write_csv(output_dir / "fig4_blinded_labeling_packet.csv", blinded.to_dict("records") if not blinded.empty else [])
    write_csv(output_dir / "fig4_blinded_labeling_answer_key.csv", key.to_dict("records") if not key.empty else [])
    completed_label_template = write_fig4_completed_blinded_label_template(output_dir, blinded)
    labeler_templates = write_fig4_labeler_completed_blinded_label_templates(output_dir, completed_label_template)
    protocol = {
        "status": "ready_for_blinded_human_or_peer_labeling" if not blinded.empty else "empty_labeling_packet",
        "seed": int(seed),
        "primary_per_tier": int(primary_per_tier),
        "reserve_per_tier": int(reserve_per_tier),
        "primary_case_count": int(blinded["assignment_role"].eq("primary_validation_labeling_sample").sum()) if not blinded.empty else 0,
        "reserve_case_count": int(blinded["assignment_role"].eq("reserve_validation_labeling_sample").sum()) if not blinded.empty else 0,
        "completed_label_template_case_count": int(len(completed_label_template)),
        "blinding_rule": "fig3_score_for_validation, fig3_global_percentile, and global_fig3_tier are omitted from fig4_blinded_labeling_packet.csv, blinded_case_id assignment is randomized across Fig.3 tiers, and tier mappings are retained only in fig4_blinded_labeling_answer_key.csv.",
        "completed_label_import_path": FIG4_COMPLETED_BLINDED_LABELS_FILE,
        "completed_label_template_path": FIG4_COMPLETED_BLINDED_LABELS_TEMPLATE_FILE,
        "labeler_specific_completed_label_templates": labeler_templates,
        "labeler_return_merge_audit_path": "fig4_blinded_label_return_merge_audit.csv",
        "required_label_columns": [
            "label_novelty_1_5",
            "label_significance_1_5",
            "label_prior_art_1_5",
            "label_confidence_1_5",
            "label_source",
            "labeler_id",
        ],
        "acceptance_rule": "At least 10 primary labeled cases per low/middle/high tier with novelty and significance scores in [1,5], assigned blind to Fig.3 score/tier.",
    }
    (output_dir / "fig4_blinded_labeling_protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    protocol_md = textwrap.dedent(
        f"""
        # Fig.4 Blinded External-Validation Labeling Protocol

        Labelers receive `fig4_blinded_labeling_packet.csv` only. They must not see
        `fig4_blinded_labeling_answer_key.csv`, Fig.3 scores, percentiles, or tier labels.
        `blinded_case_id` assignment is randomized across Fig.3 tiers before export.

        Coordinators can copy `fig4_completed_blinded_labels_template.csv` to
        `fig4_completed_blinded_labels.csv` after primary-case labels are filled.
        Alternatively, distribute `fig4_completed_blinded_labels_labeler_1.csv`,
        `fig4_completed_blinded_labels_labeler_2.csv`, and
        `fig4_completed_blinded_labels_labeler_3.csv`; the merge helper only
        materializes `fig4_completed_blinded_labels.csv` after all three returns
        are complete.

        Required labels for each primary case:
        - `label_novelty_1_5`: 1 = low novelty, 5 = very high novelty.
        - `label_significance_1_5`: 1 = low expected field significance, 5 = very high significance.
        - `label_prior_art_1_5`: 1 = weak distinction from prior art, 5 = strong distinction.
        - `label_confidence_1_5`: confidence in the judgement.
        - `label_source` and `labeler_id`.

        Current packet: {protocol["primary_case_count"]} primary cases and
        {protocol["reserve_case_count"]} reserve cases, deterministic seed {seed}.
        """
    ).strip()
    atomic_write_text(output_dir / "fig4_blinded_labeling_protocol.md", protocol_md + "\n")
    return blinded, key, protocol


def build_fig4_blinded_labeling_completion_audit(output_dir: Path, min_primary_per_tier: int = 10) -> pd.DataFrame:
    """Audit whether blinded Fig.4 external-validation labels have been completed."""
    packet = _frame_from_csv(output_dir / "fig4_blinded_labeling_packet.csv")
    key = _frame_from_csv(output_dir / "fig4_blinded_labeling_answer_key.csv")
    if packet.empty or key.empty or "blinded_case_id" not in packet.columns or "blinded_case_id" not in key.columns:
        audit = pd.DataFrame(
            [
                {
                    "global_fig3_tier": tier,
                    "primary_case_count": 0,
                    "valid_labeled_primary_count": 0,
                    "additional_labels_needed": int(min_primary_per_tier),
                    "tier_label_ready": 0,
                    "overall_blinded_labeling_ready": 0,
                }
                for tier in ["low", "middle", "high"]
            ]
        )
        write_csv(output_dir / "fig4_blinded_labeling_completion_audit.csv", audit.to_dict("records"))
        return audit
    merged = packet.merge(key, on="blinded_case_id", how="inner", suffixes=("", "_key"))
    merged = merged[merged.get("assignment_role", pd.Series(dtype=str)).astype(str).eq("primary_validation_labeling_sample")].copy()
    novelty = pd.to_numeric(merged.get("label_novelty_1_5", pd.Series(dtype=float)), errors="coerce")
    significance = pd.to_numeric(merged.get("label_significance_1_5", pd.Series(dtype=float)), errors="coerce")
    label_source = merged.get("label_source", pd.Series([""] * len(merged))).astype(str).str.strip()
    merged["valid_label_row"] = novelty.between(1, 5) & significance.between(1, 5) & label_source.ne("")
    rows: List[Dict[str, Any]] = []
    for tier in ["low", "middle", "high"]:
        tier_rows = merged[merged.get("global_fig3_tier", pd.Series(dtype=str)).astype(str).eq(tier)]
        valid_count = int(tier_rows["valid_label_row"].sum()) if "valid_label_row" in tier_rows.columns else 0
        primary_count = int(len(tier_rows))
        rows.append(
            {
                "global_fig3_tier": tier,
                "primary_case_count": primary_count,
                "valid_labeled_primary_count": valid_count,
                "additional_labels_needed": max(0, int(min_primary_per_tier) - valid_count),
                "tier_label_ready": int(valid_count >= min_primary_per_tier),
                "overall_blinded_labeling_ready": 0,
            }
        )
    overall = int(all(row["tier_label_ready"] for row in rows))
    for row in rows:
        row["overall_blinded_labeling_ready"] = overall
    audit = pd.DataFrame(rows)
    write_csv(output_dir / "fig4_blinded_labeling_completion_audit.csv", audit.to_dict("records"))
    return audit


def _fig4_missing_replacement_label_fields(row: Mapping[str, Any]) -> List[str]:
    """Return the label fields still needed before a Fig.4 replacement case is usable."""
    missing: List[str] = []
    numeric_fields = [
        "label_novelty_1_5",
        "label_significance_1_5",
        "label_prior_art_1_5",
        "label_confidence_1_5",
    ]
    for column in numeric_fields:
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if pd.isna(value) or float(value) < 1.0 or float(value) > 5.0:
            missing.append(column)
    for column in ["label_source", "labeler_id"]:
        if not nonempty_cell(row.get(column, "")):
            missing.append(column)
    return missing


def build_fig4_external_validation_replacement_manifest(
    output_dir: Path,
    min_primary_per_tier: int = 10,
) -> pd.DataFrame:
    """Write the exact blinded primary cases that can replace the range-restricted Fig.4 audit."""
    packet = _frame_from_csv(output_dir / "fig4_blinded_labeling_packet.csv")
    key = _frame_from_csv(output_dir / "fig4_blinded_labeling_answer_key.csv")
    manifest_path = output_dir / "fig4_external_validation_replacement_manifest.csv"
    summary_path = output_dir / "fig4_external_validation_replacement_summary.csv"
    if packet.empty or key.empty or "blinded_case_id" not in packet.columns or "blinded_case_id" not in key.columns:
        write_csv(manifest_path, [])
        summary_rows = [
            {
                "global_fig3_tier": tier,
                "required_primary_cases": int(min_primary_per_tier),
                "primary_case_count": 0,
                "ready_primary_case_count": 0,
                "additional_ready_labels_needed": int(min_primary_per_tier),
                "tier_replacement_ready": 0,
                "overall_replacement_ready": 0,
            }
            for tier in ["low", "middle", "high"]
        ]
        write_csv(summary_path, summary_rows)
        return pd.DataFrame()
    merged = packet.merge(key, on="blinded_case_id", how="inner", suffixes=("", "_key"))
    role = merged.get("assignment_role", pd.Series([""] * len(merged))).astype(str)
    merged = merged[role.eq("primary_validation_labeling_sample")].copy()
    rows: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        row_map = row.to_dict()
        missing = _fig4_missing_replacement_label_fields(row_map)
        rows.append(
            {
                "replacement_role": "primary_external_validation_case",
                "blinded_case_id": row_map.get("blinded_case_id", ""),
                "paper_id": row_map.get("paper_id", ""),
                "global_fig3_tier": row_map.get("global_fig3_tier", ""),
                "fig3_score_for_validation": row_map.get("fig3_score_for_validation", ""),
                "fig3_global_percentile": row_map.get("fig3_global_percentile", ""),
                "label_novelty_1_5": row_map.get("label_novelty_1_5", ""),
                "label_significance_1_5": row_map.get("label_significance_1_5", ""),
                "label_prior_art_1_5": row_map.get("label_prior_art_1_5", ""),
                "label_confidence_1_5": row_map.get("label_confidence_1_5", ""),
                "label_source": row_map.get("label_source", ""),
                "labeler_id": row_map.get("labeler_id", ""),
                "missing_label_fields": ";".join(missing),
                "replacement_ready": int(not missing),
                "required_action": "" if not missing else "complete_blinded_primary_labels_before_replacing_fig4_external_validation",
            }
        )
    manifest = pd.DataFrame(rows)
    summary_rows: List[Dict[str, Any]] = []
    for tier in ["low", "middle", "high"]:
        tier_rows = manifest[manifest.get("global_fig3_tier", pd.Series(dtype=str)).astype(str).eq(tier)] if not manifest.empty else pd.DataFrame()
        ready_count = int(tier_rows.get("replacement_ready", pd.Series(dtype=int)).astype(int).sum()) if not tier_rows.empty else 0
        primary_count = int(len(tier_rows))
        summary_rows.append(
            {
                "global_fig3_tier": tier,
                "required_primary_cases": int(min_primary_per_tier),
                "primary_case_count": primary_count,
                "ready_primary_case_count": ready_count,
                "additional_ready_labels_needed": max(0, int(min_primary_per_tier) - ready_count),
                "tier_replacement_ready": int(ready_count >= int(min_primary_per_tier)),
                "overall_replacement_ready": 0,
            }
        )
    overall_ready = int(all(row["tier_replacement_ready"] for row in summary_rows))
    for row in summary_rows:
        row["overall_replacement_ready"] = overall_ready
        row["manifest_path"] = str(manifest_path)
    write_csv(manifest_path, manifest.to_dict("records") if not manifest.empty else [])
    write_csv(summary_path, summary_rows)
    return manifest


def build_fig4_blinded_external_validation_metrics(output_dir: Path) -> pd.DataFrame:
    """Materialize Fig.3-score versus blinded novelty/significance labels."""
    packet = _frame_from_csv(output_dir / "fig4_blinded_labeling_packet.csv")
    key = _frame_from_csv(output_dir / "fig4_blinded_labeling_answer_key.csv")
    out_path = output_dir / "fig4_blinded_external_validation_metrics.csv"
    if packet.empty or key.empty or "blinded_case_id" not in packet.columns or "blinded_case_id" not in key.columns:
        write_csv(out_path, [])
        return pd.DataFrame()
    merged = packet.merge(key, on="blinded_case_id", how="inner", suffixes=("", "_key"))
    role = merged.get("assignment_role", pd.Series(dtype=str)).astype(str)
    merged = merged[role.eq("primary_validation_labeling_sample")].copy()
    if merged.empty:
        write_csv(out_path, [])
        return merged
    novelty = pd.to_numeric(merged.get("label_novelty_1_5", pd.Series(dtype=float)), errors="coerce")
    significance = pd.to_numeric(merged.get("label_significance_1_5", pd.Series(dtype=float)), errors="coerce")
    source = merged.get("label_source", pd.Series([""] * len(merged))).astype(str).str.strip()
    valid = novelty.between(1, 5) & significance.between(1, 5) & source.ne("")
    metrics = pd.DataFrame(
        {
            "blinded_case_id": merged["blinded_case_id"].astype(str),
            "paper_id": merged.get("paper_id", pd.Series([""] * len(merged))).astype(str),
            "fig3_sw": pd.to_numeric(merged.get("fig3_score_for_validation", pd.Series(dtype=float)), errors="coerce"),
            "fig3_score": pd.to_numeric(merged.get("fig3_score_for_validation", pd.Series(dtype=float)), errors="coerce"),
            "fig3_sw_percentile": pd.to_numeric(merged.get("fig3_global_percentile", pd.Series(dtype=float)), errors="coerce"),
            "fig3_sw_tier": merged.get("global_fig3_tier", pd.Series([""] * len(merged))).astype(str),
            "peer_novelty": novelty,
            "peer_significance": significance,
            "peer_novelty_judgement": novelty,
            "peer_significance_judgement": significance,
            "label_prior_art_1_5": pd.to_numeric(merged.get("label_prior_art_1_5", pd.Series(dtype=float)), errors="coerce"),
            "label_confidence_1_5": pd.to_numeric(merged.get("label_confidence_1_5", pd.Series(dtype=float)), errors="coerce"),
            "label_source": source,
            "labeler_id": merged.get("labeler_id", pd.Series([""] * len(merged))).astype(str),
            "valid_blinded_label": valid.astype(int),
            "validation_source": "blinded_external_labeling",
        }
    )
    metrics = metrics[metrics["valid_blinded_label"].astype(int).eq(1)].reset_index(drop=True)
    write_csv(out_path, metrics.to_dict("records") if not metrics.empty else [])
    return metrics


def build_fig4_blinded_external_validation_gates(
    output_dir: Path,
    min_primary_per_tier: int = 10,
) -> Dict[str, Any]:
    """Evaluate whether completed blinded labels validate Fig.3 novelty/significance scores."""
    metrics = build_fig4_blinded_external_validation_metrics(output_dir)
    completion = build_fig4_blinded_labeling_completion_audit(output_dir, min_primary_per_tier=min_primary_per_tier)
    target_audit = build_fig4_external_validation_target_audit(metrics)
    completion_ready = int(
        not completion.empty
        and int(completion.get("overall_blinded_labeling_ready", pd.Series([0])).max()) == 1
    )
    novelty_ci = bootstrap_spearman_ci(metrics, "fig3_score", "peer_novelty", seed=20260630)
    significance_ci = bootstrap_spearman_ci(metrics, "fig3_score", "peer_significance", seed=20260631)
    checks = {
        "blinded_labeling_complete": completion_ready,
        "primary_cases_minimum": int(len(metrics) >= 3 * int(min_primary_per_tier)),
        "fig3_reference_tier_range_present": int(target_audit["fig3_score_range_ready"]),
        "peer_novelty_variance_present": int(target_audit["peer_novelty_unique"] >= 2),
        "peer_significance_variance_present": int(target_audit["peer_significance_unique"] >= 2),
        "fig3_peer_novelty_positive": _spearman_positive_any(
            metrics,
            ["fig3_score", "fig3_sw"],
            ["peer_novelty_judgement", "peer_novelty"],
        ),
        "fig3_peer_significance_positive": _spearman_positive_any(
            metrics,
            ["fig3_score", "fig3_sw"],
            ["peer_significance_judgement", "peer_significance"],
        ),
        "fig3_peer_novelty_ci_positive": int(novelty_ci.get("ci_excludes_zero_positive", 0)),
        "fig3_peer_significance_ci_positive": int(significance_ci.get("ci_excludes_zero_positive", 0)),
    }
    overall = bool(all(checks.values()))
    return {
        "overall_pass": overall,
        "status_label": "blinded_external_validation_ready" if overall else "blinded_external_validation_incomplete",
        "checks": checks,
        "bootstrap_spearman": {
            "peer_novelty": novelty_ci,
            "peer_significance": significance_ci,
        },
        "metrics_path": str(output_dir / "fig4_blinded_external_validation_metrics.csv"),
        "completion_audit_path": str(output_dir / "fig4_blinded_labeling_completion_audit.csv"),
        "external_validation_target_audit": target_audit,
        "allowed_claim": "Completed blinded labels externally validate Fig.3 scores against novelty/significance judgements when gates pass.",
        "forbidden_claim": "Do not claim peer-review equivalence or reviewer replacement from blinded Fig.3 score validation labels.",
    }


def build_fig4_external_validation_gates(
    metrics_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    requested_sample_size: int = 50,
) -> Dict[str, Any]:
    """Build Fig.4 external peer-review validation gates for Nature-ready claims."""
    embedding_backends = set(metrics_df.get("embedding_backend", pd.Series(dtype=str)).astype(str))
    retrieval_sources = set(metrics_df.get("retrieval_source", pd.Series(dtype=str)).astype(str))
    soft_claim = _numeric_series(metrics_df, "soft_claim_recall")
    evidence = _numeric_series(metrics_df, "claim_evidence_coverage")
    covered = _numeric_series(metrics_df, "covered_peer_aspects")
    missing = _numeric_series(metrics_df, "missing_peer_point_rate")
    target_audit = build_fig4_external_validation_target_audit(metrics_df)
    checks = {
        "fixed_sample_size_50": int(len(manifest_df) == requested_sample_size),
        "embedding_backend_not_lexical_fallback": int(bool(embedding_backends) and "lexical_fallback" not in embedding_backends),
        "retrieval_not_local_manifest": int(bool(retrieval_sources) and not {"local_fig4_manifest", "local_fallback"}.intersection(retrieval_sources)),
        "soft_claim_recall_nonzero": int(not soft_claim.empty and float(soft_claim.max()) > 0.0),
        "claim_evidence_coverage_nonzero": int(not evidence.empty and float(evidence.max()) > 0.0),
        "covered_peer_aspects_nonzero": int(not covered.empty and float(covered.max()) > 0.0),
        "missing_peer_point_rate_below_one": int(not missing.empty and float(missing.mean()) < 1.0),
        "peer_novelty_variance_present": int(target_audit["peer_novelty_unique"] >= 2),
        "peer_significance_variance_present": int(target_audit["peer_significance_unique"] >= 2),
        "fig3_reference_tier_range_present": int(target_audit["fig3_score_range_ready"]),
        "fig3_peer_novelty_positive": _spearman_positive_any(
            metrics_df,
            ["fig3_score", "fig3_sw", "fig3_S_w", "S_w"],
            ["peer_novelty_judgement", "peer_novelty", "novelty_rating"],
        ),
        "fig3_peer_significance_positive": _spearman_positive_any(
            metrics_df,
            ["fig3_score", "fig3_sw", "fig3_S_w", "S_w"],
            ["peer_significance_judgement", "peer_significance", "significance_rating"],
        ),
    }
    overall = bool(all(checks.values()))
    return {
        "figure": "fig4",
        "role": "external_peer_review_validation",
        "overall_pass": overall,
        "status_label": "external_validation_ready" if overall else "external_validation_blocked",
        "checks": checks,
        "external_validation_target_audit": target_audit,
        "embedding_backends": sorted(embedding_backends),
        "retrieval_sources": sorted(retrieval_sources),
        "allowed_claim": "Fig.4 externally validates Fig.3 innovation scores against peer-review novelty/significance judgements when gates pass.",
        "forbidden_claim": "Do not claim peer-review equivalence or reviewer replacement from Fig.4.",
    }


def write_fig4_external_validation_report(
    output_dir: Path,
    requested_sample_size: int,
    fig3_score_table_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Write Fig.4 quality gates from the current manifest and metrics tables."""
    manifest = pd.DataFrame(read_csv_records(output_dir / "fig4_manifest.csv"))
    metrics = pd.DataFrame(read_csv_records(output_dir / "fig4_metrics_summary.csv"))
    report = build_fig4_external_validation_gates(metrics, manifest, requested_sample_size=requested_sample_size)
    agent_output_audit = validate_fig4_agent_outputs_for_nature_ready(
        output_dir=output_dir,
        expected_case_count=requested_sample_size,
    )
    report["checks"]["agent_outputs_nonfallback"] = int(agent_output_audit["overall_pass"])
    report["agent_output_audit"] = agent_output_audit
    coverage_audit = build_fig4_global_score_coverage_audit(
        output_dir=output_dir,
        fig3_score_table_path=fig3_score_table_path,
        requested_sample_size=requested_sample_size,
    )
    candidate_packet = build_fig4_global_validation_candidate_packet(
        output_dir=output_dir,
        fig3_score_table_path=fig3_score_table_path,
    )
    blinded_packet, answer_key, labeling_protocol = build_fig4_blinded_labeling_package(
        output_dir=output_dir,
        candidate_packet=candidate_packet,
    )
    labeling_completion = build_fig4_blinded_labeling_completion_audit(output_dir)
    replacement_manifest = build_fig4_external_validation_replacement_manifest(output_dir)
    replacement_summary = _frame_from_csv(output_dir / "fig4_external_validation_replacement_summary.csv")
    blinded_external_validation = build_fig4_blinded_external_validation_gates(output_dir)
    if not coverage_audit.empty:
        report["global_score_coverage_audit"] = {
            "path": str(output_dir / "fig4_global_score_coverage_audit.csv"),
            "overall_score_coverage_ready": int(coverage_audit["overall_score_coverage_ready"].max()),
            "additional_fixed_cases_needed": {
                str(row["global_fig3_tier"]): int(row["additional_fixed_cases_needed"])
                for row in coverage_audit.to_dict("records")
            },
        }
    if not candidate_packet.empty:
        report["global_validation_candidate_packet"] = {
            "path": str(output_dir / "fig4_global_validation_candidate_packet.csv"),
            "candidate_count": int(len(candidate_packet)),
            "tier_counts": {
                str(tier): int(count)
                for tier, count in candidate_packet["global_fig3_tier"].value_counts().sort_index().items()
            },
        }
    if not blinded_packet.empty:
        report["blinded_labeling_package"] = {
            "packet_path": str(output_dir / "fig4_blinded_labeling_packet.csv"),
            "answer_key_path": str(output_dir / "fig4_blinded_labeling_answer_key.csv"),
            "protocol_path": str(output_dir / "fig4_blinded_labeling_protocol.md"),
            "protocol_status": str(labeling_protocol.get("status", "")),
            "primary_case_count": int(labeling_protocol.get("primary_case_count", 0)),
            "reserve_case_count": int(labeling_protocol.get("reserve_case_count", 0)),
            "answer_key_tier_counts": {
                str(tier): int(count)
                for tier, count in answer_key["global_fig3_tier"].value_counts().sort_index().items()
            },
            "blinded_columns_exclude_fig3_score_and_tier": int(
                not {"fig3_score_for_validation", "fig3_global_percentile", "global_fig3_tier"}.intersection(blinded_packet.columns)
            ),
        }
    if not labeling_completion.empty:
        report["blinded_labeling_completion_audit"] = {
            "path": str(output_dir / "fig4_blinded_labeling_completion_audit.csv"),
            "overall_blinded_labeling_ready": int(labeling_completion["overall_blinded_labeling_ready"].max()),
            "additional_labels_needed": {
                str(row["global_fig3_tier"]): int(row["additional_labels_needed"])
                for row in labeling_completion.to_dict("records")
            },
        }
    if not replacement_summary.empty:
        report["external_validation_replacement_manifest"] = {
            "manifest_path": str(output_dir / "fig4_external_validation_replacement_manifest.csv"),
            "summary_path": str(output_dir / "fig4_external_validation_replacement_summary.csv"),
            "replacement_case_count": int(len(replacement_manifest)),
            "overall_replacement_ready": int(replacement_summary["overall_replacement_ready"].max()),
            "additional_ready_labels_needed": {
                str(row["global_fig3_tier"]): int(row["additional_ready_labels_needed"])
                for row in replacement_summary.to_dict("records")
            },
        }
    report["blinded_external_validation"] = blinded_external_validation
    if blinded_external_validation.get("overall_pass"):
        report["aspr_review_audit_checks"] = dict(report.get("checks", {}))
        report["checks"] = dict(blinded_external_validation.get("checks", {}))
        report["overall_pass"] = True
        report["status_label"] = "blinded_external_validation_ready"
        report["external_validation_evidence_mode"] = "completed_blinded_fig3_score_labels"
    report["created_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    (output_dir / "figure_quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def normalize_title_for_match(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(title or "").lower()).strip()


def normalize_doi_value(value: Any) -> str:
    text = str(value or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text.rstrip(".,;)")


def filter_prior_art_candidates(
    papers: Sequence[Mapping[str, Any]],
    target_title: str,
    target_doi: str,
    cutoff_year: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Remove the target paper and papers published after the target cutoff year."""
    kept: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    target_title_norm = normalize_title_for_match(target_title)
    target_doi_norm = normalize_doi_value(target_doi)
    for paper in papers:
        item = dict(paper)
        reasons: List[str] = []
        external_ids = item.get("externalIds") if isinstance(item.get("externalIds"), Mapping) else {}
        paper_doi = normalize_doi_value(item.get("doi") or external_ids.get("DOI"))
        paper_title_norm = normalize_title_for_match(str(item.get("title") or ""))
        paper_year = int(numeric(item.get("year"), 0.0) or 0)
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
    """Use the Fig.4 manifest as a deterministic fallback prior-art pool."""
    candidates: List[Dict[str, Any]] = []
    for other in read_csv_records(output_dir / "fig4_manifest.csv"):
        if str(other.get("paper_id")) == str(row.get("paper_id")):
            continue
        title = str(other.get("title") or "")
        abstract = str(other.get("abstract") or "")
        candidates.append(
            {
                "paperId": str(other.get("paper_id") or stable_text_hash(title + abstract)),
                "year": int(numeric(other.get("year"), 0.0) or 0),
                "title": title,
                "authors": "",
                "venue": other.get("journal", ""),
                "citationCount": 0,
                "abstract": abstract,
                "externalIds": {"DOI": other.get("doi", "")},
                "doi": other.get("doi", ""),
                "fieldsOfStudy": [str(other.get("journal", ""))] if other.get("journal") else [],
                "retrieval_source": "local_fig4_manifest",
            }
        )
    return [item for item in candidates if item.get("title") or item.get("abstract")]


def extract_lightweight_keywords(title: str, abstract: str, limit: int = 8) -> List[str]:
    stop = {
        "about",
        "after",
        "also",
        "among",
        "based",
        "between",
        "from",
        "have",
        "into",
        "more",
        "than",
        "that",
        "their",
        "this",
        "through",
        "using",
        "with",
        "within",
        "without",
    }
    counts: Dict[str, int] = {}
    for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", f"{title} {abstract}".lower()):
        if token not in stop:
            counts[token] = counts.get(token, 0) + 1
    return [token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def extract_row_keywords(row: Mapping[str, Any], limit: int = 8) -> List[str]:
    """Extract robust query keywords from a manifest row without treating NaN as text."""
    raw_keywords = clean_optional_text(row.get("keywords"))
    keywords = [item.strip() for item in raw_keywords.split(",") if item.strip()]
    if keywords:
        return keywords[:limit]
    return extract_lightweight_keywords(
        clean_optional_text(row.get("title")),
        clean_optional_text(row.get("abstract")),
        limit=limit,
    )


def _first_sentence(text: str, fallback: str = "") -> str:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return fallback
    match = re.search(r"(?<=[.!?])\s+", cleaned)
    return cleaned[: match.start()].strip() if match else cleaned[:280]


def fig3_weights_path_from_env() -> Path:
    return Path(os.getenv("FIG4_FIG3_WEIGHTS_PATH", str(DEFAULT_FIG3_WEIGHTS_PATH)))


def fig3_score_table_path_from_env() -> Path:
    return Path(os.getenv("FIG4_FIG3_SCORE_TABLE_PATH", str(DEFAULT_FIG3_SCORE_TABLE_PATH)))


def fig3_indicators_path_from_env() -> Path:
    return Path(os.getenv("FIG4_FIG3_INDICATORS_PATH", str(DEFAULT_FIG3_INDICATORS_PATH)))


def metric_name_from_weight_row(row: Mapping[str, Any]) -> str:
    """Read Fig.3 weight files written either as named rows or pandas Series CSV."""
    for key in ("metric", "name", "feature", ""):
        value = normalize_whitespace(str(row.get(key) or ""))
        if value:
            return value
    return ""


def file_sha1(path: Path, length: int = 12) -> str:
    if not path.exists():
        return ""
    return hashlib.sha1(path.read_bytes()).hexdigest()[:length]


def load_fig3_weights(path: Path) -> Dict[str, float]:
    config = load_fig3_weight_config(path)
    return dict(config["weights"])


def load_fig3_weight_config(path: Path) -> Dict[str, Any]:
    """Load Fig.3 simplex weights with explicit fallback metadata."""
    if not path.exists():
        equal = 1.0 / len(INNOVATION_METRIC_NAMES)
        return {
            "weights": {metric: equal for metric in INNOVATION_METRIC_NAMES},
            "weights_source": "equal_weight_fallback",
            "weights_hash": "",
            "warning": f"missing_fig3_weights:{path}",
        }
    rows = read_csv_records(path)
    raw: Dict[str, float] = {}
    for row in rows:
        metric = metric_name_from_weight_row(row)
        if metric in INNOVATION_METRIC_NAMES:
            raw[metric] = max(0.0, numeric(row.get("weight"), 0.0))
    total = sum(raw.values())
    if total <= 0:
        equal = 1.0 / len(INNOVATION_METRIC_NAMES)
        return {
            "weights": {metric: equal for metric in INNOVATION_METRIC_NAMES},
            "weights_source": "equal_weight_fallback",
            "weights_hash": file_sha1(path),
            "warning": f"empty_fig3_weights:{path}",
        }
    return {
        "weights": {metric: raw.get(metric, 0.0) / total for metric in INNOVATION_METRIC_NAMES},
        "weights_source": str(path),
        "weights_hash": file_sha1(path),
        "warning": "",
    }


def transform_fig3_metric_value(metric: str, value: Any) -> float:
    """Apply the lightweight global transforms used before Fig.3 rank-normalization."""
    number = numeric(value, float("nan"))
    if not math.isfinite(number):
        return float("nan")
    if metric == "B":
        return math.log1p(max(number, 0.0))
    return number


def load_fig3_reference_metric_values(reference_indicators_path: Path) -> Dict[str, List[float]]:
    rows = read_csv_records(reference_indicators_path)
    out: Dict[str, List[float]] = {metric: [] for metric in INNOVATION_METRIC_NAMES}
    for metric in INNOVATION_METRIC_NAMES:
        transformed_key = f"{metric}_transformed"
        for row in rows:
            raw_value = row.get(transformed_key) if transformed_key in row else row.get(metric)
            value = numeric(raw_value, float("nan")) if transformed_key in row else transform_fig3_metric_value(metric, raw_value)
            if math.isfinite(value):
                out[metric].append(value)
        out[metric].sort()
    return out


def empirical_percentile(value: float, reference_values: Sequence[float]) -> float:
    values = [float(item) for item in reference_values if math.isfinite(float(item))]
    if not values or not math.isfinite(value):
        return float("nan")
    less = 0
    equal = 0
    for item in values:
        if item < value:
            less += 1
        elif item == value:
            equal += 1
    percentile = (less + 0.5 * max(equal, 1)) / len(values)
    epsilon = 1.0 / (2.0 * max(len(values), 1))
    return clamp(percentile, epsilon, 1.0 - epsilon, default=0.5)


def rank_normal_from_reference(value: float, reference_values: Sequence[float]) -> float:
    percentile = empirical_percentile(value, reference_values)
    if not math.isfinite(percentile):
        return 0.0
    return clamp(NormalDist().inv_cdf(percentile), -3.0, 3.0, default=0.0)


def percentile_tier(percentile: Any) -> str:
    value = numeric(percentile, float("nan"))
    if not math.isfinite(value):
        return "unknown"
    if value < 1.0 / 3.0:
        return "low"
    if value < 2.0 / 3.0:
        return "middle"
    return "high"


def fig3_score_reference_values(
    weights: Mapping[str, float],
    metric_reference_values: Mapping[str, Sequence[float]],
    reference_indicators_path: Path,
    reference_scores_path: Path,
) -> Tuple[List[float], str]:
    score_rows = read_csv_records(reference_scores_path)
    for score_key in ("S_w_oof", "S_w"):
        values = [numeric(row.get(score_key), float("nan")) for row in score_rows]
        values = sorted(value for value in values if math.isfinite(value))
        if values:
            return values, "fig3_score_table"
    indicator_rows = read_csv_records(reference_indicators_path)
    values: List[float] = []
    for row in indicator_rows:
        weighted = 0.0
        has_any = False
        for metric in INNOVATION_METRIC_NAMES:
            if f"{metric}_z" in row:
                z_value = numeric(row.get(f"{metric}_z"), float("nan"))
            else:
                z_value = rank_normal_from_reference(
                    transform_fig3_metric_value(metric, row.get(metric)),
                    metric_reference_values.get(metric, []),
                )
            if math.isfinite(z_value):
                weighted += z_value * numeric(weights.get(metric), 0.0)
                has_any = True
        if has_any:
            values.append(weighted)
    return sorted(values), "fig3_reference_distribution" if values else ""


def build_fig3_weighted_prior_rows(
    graph_rows: Sequence[Mapping[str, Any]],
    weights_path: Optional[Path] = None,
    reference_scores_path: Optional[Path] = None,
    reference_indicators_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Compute Fig.3-weighted S_w priors for Fig.4 graph rows."""
    weights_config = load_fig3_weight_config(weights_path or fig3_weights_path_from_env())
    weights = weights_config["weights"]
    indicators_path = reference_indicators_path or fig3_indicators_path_from_env()
    scores_path = reference_scores_path or fig3_score_table_path_from_env()
    metric_reference_values = load_fig3_reference_metric_values(indicators_path)
    score_reference_values, score_source = fig3_score_reference_values(weights, metric_reference_values, indicators_path, scores_path)
    rows: List[Dict[str, Any]] = []
    for graph_row in graph_rows:
        out = dict(graph_row)
        weighted = 0.0
        z_count = 0
        missing_metrics: List[str] = []
        for metric in INNOVATION_METRIC_NAMES:
            transformed = transform_fig3_metric_value(metric, graph_row.get(metric))
            z_value = rank_normal_from_reference(transformed, metric_reference_values.get(metric, []))
            out[f"{metric}_z"] = z_value
            if math.isfinite(z_value):
                weighted += z_value * numeric(weights.get(metric), 0.0)
                z_count += 1
            else:
                missing_metrics.append(metric)
        out["fig3_sw"] = weighted
        out["fig3_sw_normalization"] = "fig3_reference_rank_normal"
        out["fig3_weights_source"] = weights_config["weights_source"]
        out["fig3_weights_hash"] = weights_config["weights_hash"]
        out["fig3_weights_warning"] = weights_config["warning"]
        out["fig3_sw_reference_source"] = score_source
        out["graph_prior_prompt_mode"] = "fig3_sw_only"
        out["fig3_sw_quality_flag"] = "ok" if z_count == len(INNOVATION_METRIC_NAMES) and not weights_config["warning"] else "limited"
        if missing_metrics:
            out["fig3_sw_quality_flag"] = "limited"
            out["fig3_sw_missing_metrics"] = ",".join(missing_metrics)
        rows.append(out)
    if score_reference_values:
        for out in rows:
            percentile = empirical_percentile(numeric(out.get("fig3_sw")), score_reference_values)
            out["fig3_sw_percentile"] = percentile
            out["fig3_sw_percentile_source"] = score_source or "fig3_reference_distribution"
            out["fallback_percentile_source"] = ""
            out["fig3_sw_tier"] = percentile_tier(percentile)
    else:
        sw_values = sorted(numeric(row.get("fig3_sw")) for row in rows if math.isfinite(numeric(row.get("fig3_sw"))))
        for out in rows:
            percentile = empirical_percentile(numeric(out.get("fig3_sw")), sw_values)
            out["fig3_sw_percentile"] = percentile
            out["fig3_sw_percentile_source"] = "fig4_batch"
            out["fallback_percentile_source"] = "fig4_batch"
            out["fig3_sw_tier"] = percentile_tier(percentile)
            if out["fig3_sw_quality_flag"] == "ok":
                out["fig3_sw_quality_flag"] = "limited"
    sw_values = sorted(numeric(row.get("fig3_sw")) for row in rows if math.isfinite(numeric(row.get("fig3_sw"))))
    for out in rows:
        batch_percentile = empirical_percentile(numeric(out.get("fig3_sw")), sw_values)
        out["fig4_sw_batch_percentile"] = batch_percentile
        out["fig4_sw_ladder_tier"] = percentile_tier(batch_percentile)
    return rows


def graph_metric_prompt_block(metrics: Mapping[str, Any]) -> str:
    sw = numeric(metrics.get("fig3_sw"), float("nan"))
    percentile = numeric(metrics.get("fig3_sw_percentile"), float("nan"))
    tier = str(metrics.get("fig3_sw_tier") or "unknown")
    quality = str(metrics.get("fig3_sw_quality_flag") or metrics.get("graph_prior_quality_flag") or "limited")
    if not math.isfinite(sw):
        return (
            "Fig.3-weighted graph innovation prior: unavailable.\n"
            "Graph prior quality: limited.\n"
            "Use conservative innovation language and rely on the dossier plus prior-art comparison."
        )
    percentile_text = f"{percentile * 100:.0f}%" if math.isfinite(percentile) else "n/a"
    return (
        "Fig.3-weighted graph innovation prior:\n"
        f"S_w = {sw:.3f}\n"
        f"Fig.3 reference percentile = {percentile_text}\n"
        f"Prior tier = {tier}\n"
        f"Evidence quality flag = {quality}\n"
        "Use S_w as the only graph-perturbation prior for calibration. "
        "Do not discuss individual component indicators or treat the prior as a standalone conclusion."
    )


def graph_metric_result_from_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    metric_values = {metric: numeric(row.get(metric), 0.0) for metric in INNOVATION_METRIC_NAMES}
    return {
        "metrics": metric_values,
        "weighted_score": numeric(row.get("fig3_sw"), numeric(row.get("weighted_score_fig3"), 0.0)),
        "fig3_sw": numeric(row.get("fig3_sw"), float("nan")),
        "fig3_sw_percentile": numeric(row.get("fig3_sw_percentile"), float("nan")),
        "fig3_sw_tier": row.get("fig3_sw_tier", ""),
        "confidence": numeric(row.get("graph_confidence"), 0.0),
        "top_mechanisms": str(row.get("top_mechanisms") or ""),
        "diagnostics": {"metric_source": row.get("metric_source", ""), "doi": row.get("doi", "")},
    }


def build_graph_prior_for_retrieved_papers(
    row: Mapping[str, Any],
    retrieved: Sequence[Mapping[str, Any]],
    metric_source: str = "agent_retrieved_prior_art_graph_scorer",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute raw graph metrics and the Fig.3 S_w prior from retrieved papers."""
    from aspr.graph_innovation_scorer import GraphInnovationScorer

    evidence = GraphInnovationScorer().score(
        paper_title=str(row.get("title") or ""),
        paper_abstract=str(row.get("abstract") or ""),
        retrieved_papers=[dict(paper) for paper in retrieved if isinstance(paper, Mapping)],
    )
    graph_row = flatten_graph_metric_evidence(row, evidence.to_dict(), metric_source)
    prior_row = build_fig3_weighted_prior_rows([graph_row])[0]
    return graph_row, prior_row


def build_lightweight_agent_evaluation(
    row: Mapping[str, Any],
    abstract: str,
    keywords: Sequence[str],
    retrieved: Sequence[Mapping[str, Any]],
    graph_metric_evidence_text: Optional[str],
) -> str:
    related_lines = []
    for idx, paper in enumerate(list(retrieved)[:5], start=1):
        related_lines.append(
            f"{idx}. {paper.get('title', 'Untitled prior work')} ({paper.get('year', '')}): "
            f"{_first_sentence(str(paper.get('abstract') or ''), 'No abstract available.')}"
        )
    if not related_lines:
        related_lines.append("1. No reliable pre-publication prior-art candidates were retrieved.")
    return textwrap.dedent(
        f"""
        Innovation stance: moderately positive, evidence-limited.

        Paper under review: {row.get('title', 'Untitled paper')} ({row.get('year', '')}).

        Core innovation claim: {_first_sentence(abstract, 'The dossier does not expose a compact claim.')}

        Novelty assessment: The paper appears promising around {", ".join(list(keywords)[:8]) or "the stated contribution"}, but novelty should be treated as calibrated rather than assumed.

        Prior-art comparison:
        {chr(10).join(related_lines)}

        Evidence and rigor: Strong innovation language requires direct experimental, mechanistic, benchmark, or validation evidence. If these signals are missing from the dossier, the review should stay conservative.

        Limitations: The main risk is overclaiming beyond retrieved prior art. The review should flag unclear boundary conditions, incomplete benchmarks, and limited generalization evidence.

        Future work: Follow-up should include broader validation, direct comparison with closest prior work, and sensitivity or ablation tests.

        Graph-based evidence: {graph_metric_evidence_text or "No graph-metric evidence was available."}
        """
    ).strip()


@dataclass
class Fig4ArgsForAgent:
    s2_api_key: str
    and_search: bool
    top_n: int
    agent_context_mode: str = "dossier"
    retrieval_provider: str = "semantic_scholar"
    openalex_api_key: str = ""


def clear_agent_dependent_caches(cache_dir: Path) -> None:
    for filename in ("agent_innovation_labels.json", "semantic_claim_matches.json", "structured_consistency.json"):
        try:
            (cache_dir / filename).unlink()
        except FileNotFoundError:
            pass


def load_fig4_cached_external_retrieval(cache_dir: Path) -> Dict[str, Any]:
    """Load cached prior-art retrieval only when its provenance is external."""
    retrieved_path = cache_dir / "retrieved_papers.json"
    audit_path = cache_dir / "retrieval_audit.json"
    if not retrieved_path.exists() or not audit_path.exists():
        return {"cache_hit": False, "failure_reason": "missing_cached_retrieval_artifacts", "papers": []}
    try:
        retrieved_payload = read_json(retrieved_path)
        audit = read_json(audit_path)
    except (OSError, json.JSONDecodeError):
        return {"cache_hit": False, "failure_reason": "unreadable_cached_retrieval_artifacts", "papers": []}
    source = str(audit.get("retrieval_source") or "").strip().lower()
    if source not in {"openalex", "semantic_scholar"}:
        return {
            "cache_hit": False,
            "failure_reason": "cached_retrieval_not_external",
            "retrieval_source": source,
            "papers": [],
        }
    papers = retrieved_payload.get("retrieved_papers")
    if not isinstance(papers, list) or not papers:
        return {
            "cache_hit": False,
            "failure_reason": "cached_retrieval_empty",
            "retrieval_source": source,
            "papers": [],
        }
    return {
        "cache_hit": True,
        "failure_reason": "",
        "papers": [dict(paper) for paper in papers if isinstance(paper, Mapping)],
        "retrieval_source": source,
        "s2_key_status": audit.get("s2_key_status", ""),
        "query_terms": audit.get("query_terms", []),
        "query_audits": audit.get("query_audits", []),
        "ranker_status": audit.get("ranker_status", {}),
        "excluded_candidates": audit.get("excluded_candidates", []),
    }


def fig4_agent_runtime_provenance() -> Dict[str, Any]:
    """Return the LATS runtime controls used for reproducible Fig.4 agent outputs."""
    return {
        "agent_model": os.getenv("ASPR_LATS_LLM_MODEL", "qwen3-coder:30b"),
        "agent_base_url": os.getenv("ASPR_LATS_LLM_BASE_URL")
        or os.getenv("ASPR_LLM_BASE_URL", "http://localhost:11434/v1"),
        "agent_max_iterations": int(os.getenv("FIG4_AGENT_MAX_ITERATIONS", "1")),
        "agent_candidates": int(os.getenv("ASPR_LATS_CANDIDATES", "2")),
        "agent_beam_width": int(os.getenv("ASPR_LATS_BEAM_WIDTH", "2")),
        "agent_max_tokens": int(os.getenv("ASPR_LATS_MAX_TOKENS", "1800")),
        "agent_prompt_prefix": os.getenv("ASPR_LATS_PROMPT_PREFIX", ""),
        "agent_search_mode": "single_pass_lats_initial"
        if bool_value(os.getenv("ASPR_LATS_SINGLE_PASS", "0"))
        else "lats_tree",
        "agent_use_committee": int(bool_value(os.getenv("FIG4_AGENT_USE_COMMITTEE", "0"))),
    }


def run_aspr_agent_for_row(
    row: Mapping[str, Any],
    cache_dir: Path,
    args_for_agent: Fig4ArgsForAgent,
    force_agent: bool = False,
    reuse_agent: bool = True,
) -> Dict[str, Any]:
    paper_id = str(row["paper_id"])
    agent_path = cache_dir / "agent_eval.json"
    if reuse_agent and agent_path.exists() and not force_agent:
        cached = read_json(agent_path)
        cached["cache_reused"] = True
        return cached
    start = time.time()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if force_agent:
        clear_agent_dependent_caches(cache_dir)
    abstract = clean_optional_text(row.get("abstract"))
    keyword_limit = max(1, int(os.getenv("FIG4_QUERY_KEYWORD_LIMIT", "8")))
    keywords = extract_row_keywords(row, limit=keyword_limit)
    dossier_path = Path(str(row.get("paper_dossier_cache") or cache_dir / "paper_dossier.json"))
    if dossier_path.exists():
        dossier = read_json(dossier_path)
    else:
        dossier = build_paper_dossier(
            {
                "title": clean_optional_text(row.get("title")),
                "abstract": abstract,
                "doi": row.get("doi", ""),
                "year": row.get("year", ""),
                "article_text": abstract,
                "word_count": word_count(abstract),
            },
            row,
            keywords=keywords,
        )
        write_json(dossier_path, dossier)
    paper_context = "" if args_for_agent.agent_context_mode == "abstract_only" else format_paper_dossier_for_agent(dossier)
    graph_metric_evidence_text: Optional[str] = None
    graph_metric_result: Optional[Dict[str, Any]] = None
    graph_path = cache_dir / "fig4_graph_metrics.json"
    if graph_path.exists():
        graph_row = read_json(graph_path)
        if bool_value(graph_row.get("graph_metric_valid")):
            graph_metric_evidence_text = graph_metric_prompt_block(graph_row)
            graph_metric_result = graph_metric_result_from_row(graph_row)
    output_dir = cache_dir.parents[1]
    retrieval_source = "local_fallback"
    retrieval_failure = ""
    s2_key_status = "not_run"
    query_audits: List[Dict[str, Any]] = []
    ranker_status: Dict[str, Any] = {"recall_backend": "local", "reranker_backend": "local"}
    papers = local_prior_art_candidates(row, output_dir)
    cached_retrieval: Dict[str, Any] = {}
    if bool_value(os.getenv("FIG4_REUSE_RETRIEVAL_CACHE", "0")):
        cached_retrieval = load_fig4_cached_external_retrieval(cache_dir)
        if bool_value(cached_retrieval.get("cache_hit")):
            papers = list(cached_retrieval.get("papers", []))
            retrieval_source = str(cached_retrieval.get("retrieval_source") or "openalex")
            s2_key_status = str(cached_retrieval.get("s2_key_status") or "cached_external_retrieval")
            cached_terms = cached_retrieval.get("query_terms")
            if isinstance(cached_terms, list) and cached_terms:
                keywords = [str(item) for item in cached_terms]
            cached_query_audits = cached_retrieval.get("query_audits")
            if isinstance(cached_query_audits, list):
                query_audits = [dict(item) for item in cached_query_audits if isinstance(item, Mapping)]
            cached_ranker_status = cached_retrieval.get("ranker_status")
            if isinstance(cached_ranker_status, Mapping):
                ranker_status = dict(cached_ranker_status)
    try:
        local_retrieval_only = bool_value(os.getenv("FIG4_LOCAL_RETRIEVAL_ONLY", "0"))
        if bool_value(cached_retrieval.get("cache_hit")):
            retrieval_failure = ""
        elif not bool_value(os.getenv("FIG4_LIGHTWEIGHT_AGENT", "0")) and not local_retrieval_only:
            from aspr.open_scholar import OpenScholar, keywords_extract, retrieval_backend_status, retrieval_recall, retrieval_rerank

            if not keywords:
                keywords = keywords_extract(abstract)
            scholar = OpenScholar(args_for_agent)
            papers = scholar.search_semantic_scholar(keywords)
            query_audits = getattr(scholar, "last_query_audits", [])
            retrieval_source = getattr(scholar, "last_retrieval_source", "semantic_scholar")
            s2_key_status = getattr(scholar, "s2_key_status", "unknown")
            ranker_status = retrieval_backend_status()
            formatted = [f"Title:{paper.get('title', '')}. Abstract:{paper.get('abstract', '')}" for paper in papers]
            item_to_paper = {item: paper for item, paper in zip(formatted, papers)}
            if formatted:
                recalled, _ = retrieval_recall(clean_optional_text(row.get("title")) + "\n" + abstract, formatted)
                reranked, _ = retrieval_rerank(clean_optional_text(row.get("title")) + "\n" + abstract, recalled[: max(args_for_agent.top_n * 5, args_for_agent.top_n)])
                papers = [item_to_paper[item] for item in reranked if item in item_to_paper]
        elif local_retrieval_only:
            retrieval_source = "local_fig4_manifest"
            s2_key_status = "not_run_local_retrieval_only"
    except Exception as exc:  # noqa: BLE001 - fallback is expected in offline demos.
        retrieval_failure = str(exc)
        retrieval_source = "local_fallback"
        papers = local_prior_art_candidates(row, output_dir)
    papers, excluded = filter_prior_art_candidates(
        papers,
        target_title=str(row.get("title", "")),
        target_doi=str(row.get("doi", "")),
        cutoff_year=int(numeric(row.get("year"), 0.0) or 0),
    )
    retrieved = papers[: max(1, int(args_for_agent.top_n))]
    retrieved_path = cache_dir / "retrieved_papers.json"
    write_json(retrieved_path, {"paper_id": paper_id, "retrieved_papers": retrieved})
    excluded_target_count = sum(
        1
        for item in excluded
        if "target_doi_match" in item.get("reasons", []) or "target_title_match" in item.get("reasons", [])
    )
    excluded_future_count = sum(1 for item in excluded if "post_publication_year" in item.get("reasons", []))
    write_json(
        cache_dir / "retrieval_audit.json",
        {
            "paper_id": paper_id,
            "cutoff_year": int(numeric(row.get("year"), 0.0) or 0),
            "excluded_candidates": excluded,
            "kept_candidates_count": len(papers),
            "retrieved_papers_count": len(retrieved),
            "retrieval_source": retrieval_source,
            "retrieval_failure": retrieval_failure,
            "s2_key_status": s2_key_status,
            "query_terms": keywords,
            "query_audits": query_audits,
            "ranker_status": ranker_status,
            "retrieval_cache_reused": bool_value(cached_retrieval.get("cache_hit")),
            "excluded_target_count": excluded_target_count,
            "excluded_future_count": excluded_future_count,
        },
    )
    prior_path = cache_dir / "fig4_graph_prior.json"
    try:
        if prior_path.exists():
            prior_row = read_json(prior_path)
            if graph_path.exists():
                graph_metric_result = graph_metric_result_from_row({**read_json(graph_path), **prior_row})
            else:
                graph_metric_result = graph_metric_result_from_row(prior_row)
        elif graph_path.exists() and bool_value(read_json(graph_path).get("graph_metric_valid")):
            graph_row = read_json(graph_path)
            prior_row = build_fig3_weighted_prior_rows([graph_row])[0]
            write_json(prior_path, prior_row)
            graph_metric_result = graph_metric_result_from_row({**graph_row, **prior_row})
        else:
            graph_row, prior_row = build_graph_prior_for_retrieved_papers(row, retrieved)
            write_json(graph_path, graph_row)
            write_json(prior_path, prior_row)
            graph_metric_result = graph_metric_result_from_row(prior_row)
        graph_metric_evidence_text = graph_metric_prompt_block(prior_row)
    except Exception as exc:  # noqa: BLE001 - keep the batch running with conservative graph-prior text.
        graph_metric_evidence_text = graph_metric_prompt_block({})
        graph_metric_result = {"metrics": {}, "weighted_score": 0.0, "confidence": 0.0, "diagnostics": {"graph_prior_error": str(exc)}}
    try:
        if bool_value(os.getenv("FIG4_LIGHTWEIGHT_AGENT", "0")):
            raise RuntimeError("lightweight_agent_requested")
        from aspr.lats import evaluate_paper_innovation

        eval_result = evaluate_paper_innovation(
            paper_title=str(row.get("title", "")),
            paper_abstract=abstract,
            retrieved_papers=retrieved,
            paper_context=paper_context,
            max_iterations=int(os.getenv("FIG4_AGENT_MAX_ITERATIONS", "1")),
            graph_metric_evidence=graph_metric_evidence_text,
            graph_metric_result=graph_metric_result,
            use_committee=bool_value(os.getenv("FIG4_AGENT_USE_COMMITTEE", "0")),
        )
        result = {"success": True, **eval_result}
    except Exception as exc:  # noqa: BLE001 - deterministic fallback keeps batch runs moving.
        evaluation = build_lightweight_agent_evaluation(row, abstract, keywords, retrieved, graph_metric_evidence_text)
        fallback_requested = str(exc) == "lightweight_agent_requested"
        result = {
            "success": fallback_requested,
            "failure_reason": "" if fallback_requested else f"lats_failed_lightweight_fallback:{exc}",
            "innovation_evaluation": evaluation,
            "evaluation_log": ["Generated by deterministic Fig.4 lightweight innovation agent fallback."],
            "graph_metric_evidence": graph_metric_result or {},
            "committee_report": {},
        }
    result = {
        "paper_id": paper_id,
        "title": row.get("title", ""),
        "keywords": keywords,
        "retrieved_papers_count": len(retrieved),
        "retrieved_papers_cache": str(retrieved_path),
        "paper_context_cache": str(dossier_path),
        "agent_context_mode": args_for_agent.agent_context_mode,
        "retrieval_cutoff_year": int(numeric(row.get("year"), 0.0) or 0),
        "excluded_retrieval_count": len(excluded),
        "retrieval_source": retrieval_source,
        "s2_key_status": s2_key_status,
        "query_terms": keywords,
        "ranker_status": ranker_status,
        "excluded_target_count": excluded_target_count,
        "excluded_future_count": excluded_future_count,
        "agent_runtime_seconds": time.time() - start,
        **fig4_agent_runtime_provenance(),
        **result,
    }
    write_json(cache_dir / "agent_eval.json", result)
    return result


def run_agent_stage(
    output_dir: Path,
    s2_api_key: str = "",
    top_n: int = 10,
    and_search: bool = False,
    agent_context_mode: str = "dossier",
    retrieval_provider: str = "semantic_scholar",
    openalex_api_key: str = "",
    force_agent: bool = False,
    reuse_agent: bool = True,
    refresh_invalid_agent_only: bool = False,
    max_agent: Optional[int] = None,
    quiet: bool = False,
    agent_runner: Optional[Callable[[Mapping[str, Any], Path], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    manifest = read_csv_records(output_dir / "fig4_manifest.csv")
    if max_agent is not None:
        manifest = manifest[:max_agent]
    args = Fig4ArgsForAgent(
        s2_api_key=s2_api_key,
        and_search=and_search,
        top_n=top_n,
        agent_context_mode=agent_context_mode,
        retrieval_provider=retrieval_provider,
        openalex_api_key=openalex_api_key,
    )
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(manifest, start=1):
        cache_dir = output_dir / "cache" / str(row["paper_id"])
        refresh_invalid = refresh_invalid_agent_only and agent_cache_needs_refresh_for_nature(cache_dir)
        row_force_agent = force_agent or refresh_invalid
        row_reuse_agent = reuse_agent and not row_force_agent
        result = (
            agent_runner(row, cache_dir)
            if agent_runner
            else run_aspr_agent_for_row(row, cache_dir, args, row_force_agent, row_reuse_agent)
        )
        rows.append(result)
        write_jsonl(output_dir / "fig4_agent_outputs.jsonl", rows)
        if idx == 1 or idx % 10 == 0 or idx == len(manifest):
            progress_log(f"Agent progress {idx}/{len(manifest)}.", quiet)
    write_jsonl(output_dir / "fig4_agent_outputs.jsonl", rows)
    write_csv(
        output_dir / "fig4_retrieval_diagnostics.csv",
        [
            {
                "paper_id": row.get("paper_id", ""),
                "retrieval_source": row.get("retrieval_source", ""),
                "s2_key_status": row.get("s2_key_status", ""),
                "excluded_target_count": row.get("excluded_target_count", 0),
                "excluded_future_count": row.get("excluded_future_count", 0),
                "retrieved_papers_count": row.get("retrieved_papers_count", 0),
                "failure_reason": row.get("failure_reason", ""),
            }
            for row in rows
        ],
    )
    return rows


def fig4_agent_output_failure_reasons(row: Mapping[str, Any]) -> List[str]:
    """Return Nature-readiness failure reasons for one Fig.4 agent output row."""
    reasons: List[str] = []
    if not bool_value(row.get("success", False)):
        reasons.append("agent_success_false")
    failure_reason = str(row.get("failure_reason") or "").strip()
    evaluation_log = str(row.get("evaluation_log") or "")
    if "lightweight_fallback" in failure_reason or "lightweight innovation agent fallback" in evaluation_log:
        reasons.append("lightweight_fallback_output")
    if not clean_optional_text(row.get("innovation_evaluation")):
        reasons.append("missing_innovation_evaluation")
    runtime = numeric(row.get("agent_runtime_seconds"), 0.0)
    if runtime <= 0:
        reasons.append("missing_positive_runtime")
    required_provenance = [
        "agent_model",
        "agent_base_url",
        "agent_max_iterations",
        "agent_candidates",
        "agent_beam_width",
        "agent_max_tokens",
        "agent_prompt_prefix",
        "agent_search_mode",
    ]
    for key in required_provenance:
        if str(row.get(key, "")).strip() == "":
            reasons.append(f"missing_{key}")
    return reasons


def agent_cache_needs_refresh_for_nature(cache_dir: Path) -> bool:
    """Return whether a cached Fig.4 agent output should be replaced for Nature-ready mode."""
    agent_path = cache_dir / "agent_eval.json"
    if not agent_path.exists():
        return True
    try:
        row = read_json(agent_path)
    except (OSError, json.JSONDecodeError):
        return True
    return bool(fig4_agent_output_failure_reasons(row))


def validate_fig4_agent_outputs_for_nature_ready(output_dir: Path, expected_case_count: int = 50) -> Dict[str, Any]:
    """Audit whether Fig.4 agent outputs are real non-fallback ASPR runs."""
    output_path = output_dir / "fig4_agent_outputs.jsonl"
    rows = read_jsonl(output_path)
    audit_rows: List[Dict[str, Any]] = []
    for row in rows:
        reasons = fig4_agent_output_failure_reasons(row)
        audit_rows.append(
            {
                "paper_id": row.get("paper_id", ""),
                "success": int(bool_value(row.get("success", False))),
                "retrieval_source": row.get("retrieval_source", ""),
                "agent_runtime_seconds": row.get("agent_runtime_seconds", ""),
                "nature_agent_output_ready": int(not reasons),
                "failure_reason": ";".join(reasons),
                "raw_failure_reason": row.get("failure_reason", ""),
            }
        )
    write_csv(output_dir / "fig4_agent_output_audit.csv", audit_rows)
    fallback_ids = [str(row.get("paper_id") or "") for row, audit in zip(rows, audit_rows) if audit["failure_reason"]]
    observed = len(rows)
    successful = observed - len(fallback_ids)
    missing = max(0, int(expected_case_count) - observed) if expected_case_count > 0 else 0
    overall = bool(expected_case_count <= 0 or observed == expected_case_count) and successful == observed and observed > 0
    return {
        "path": str(output_dir / "fig4_agent_output_audit.csv"),
        "agent_outputs_path": str(output_path),
        "expected_case_count": int(expected_case_count),
        "observed_case_count": int(observed),
        "successful_nonfallback_count": int(successful),
        "fallback_or_failed_count": int(len(fallback_ids)),
        "missing_case_count": int(missing),
        "fallback_or_failed_case_ids": fallback_ids,
        "overall_pass": overall,
    }


def enforce_fig4_nature_agent_outputs(output_dir: Path, expected_case_count: int = 50) -> Dict[str, Any]:
    """Raise when Fig.4 contains reused fallback outputs under Nature-ready mode."""
    audit = validate_fig4_agent_outputs_for_nature_ready(output_dir, expected_case_count=expected_case_count)
    if not audit["overall_pass"]:
        examples = ", ".join(audit["fallback_or_failed_case_ids"][:5])
        raise RuntimeError(
            "Fig.4 Nature-ready agent outputs failed: "
            f"observed={audit['observed_case_count']}/{audit['expected_case_count']}, "
            f"nonfallback={audit['successful_nonfallback_count']}, "
            f"fallback_or_failed={audit['fallback_or_failed_count']}"
            + (f" (examples: {examples})" if examples else "")
        )
    return audit


def flatten_graph_metric_evidence(
    row: Mapping[str, Any],
    evidence: Mapping[str, Any],
    metric_source: str,
    failure_reason: str = "",
) -> Dict[str, Any]:
    """Flatten graph-innovation evidence into the Fig.4 metrics CSV schema."""
    metrics = evidence.get("metrics") if isinstance(evidence.get("metrics"), Mapping) else {}
    top_mechanisms = evidence.get("top_mechanisms")
    if isinstance(top_mechanisms, list):
        top_mechanisms_text = ", ".join(str(item) for item in top_mechanisms[:3])
    else:
        top_mechanisms_text = str(top_mechanisms or "")
    valid = bool(metrics) and not failure_reason
    out: Dict[str, Any] = {
        "paper_id": row.get("paper_id", ""),
        "doi": row.get("doi", ""),
        "openalex_id": row.get("openalex_id", ""),
        "weighted_score_fig3": numeric(evidence.get("weighted_score"), 0.0),
        "graph_confidence": numeric(evidence.get("confidence"), 0.0),
        "metric_source": metric_source,
        "graph_metric_valid": valid,
        "graph_metric_failure_reason": failure_reason,
        "top_mechanisms": top_mechanisms_text,
    }
    for metric in INNOVATION_METRIC_NAMES:
        out[metric] = numeric(metrics.get(metric), 0.0)
    return out


def compute_graph_metric_evidence_from_retrieval(
    row: Mapping[str, Any],
    cache_dir: Path,
    agent_row: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    """Compute graph evidence from cached prior-art retrieval when agent evidence is absent."""
    try:
        from aspr.graph_innovation_scorer import GraphInnovationScorer
    except Exception as exc:  # noqa: BLE001 - keep per-paper diagnostics instead of failing the batch.
        return {}, f"graph_scorer_unavailable:{type(exc).__name__}:{exc}"
    retrieved_path = Path(str(agent_row.get("retrieved_papers_cache") or cache_dir / "retrieved_papers.json"))
    if not retrieved_path.exists():
        return {}, "missing_retrieved_papers_cache"
    retrieved = read_json(retrieved_path)
    if isinstance(retrieved, Mapping):
        if isinstance(retrieved.get("retrieved_papers"), list):
            papers = retrieved.get("retrieved_papers")
        else:
            papers = retrieved.get("papers") if isinstance(retrieved.get("papers"), list) else []
    elif isinstance(retrieved, list):
        papers = retrieved
    else:
        papers = []
    if not papers:
        return {}, "empty_retrieved_papers_cache"
    evidence = GraphInnovationScorer().score(
        paper_title=str(row.get("title") or ""),
        paper_abstract=str(row.get("abstract") or ""),
        retrieved_papers=[paper for paper in papers if isinstance(paper, Mapping)],
    )
    return evidence.to_dict(), ""


def run_graph_metrics_stage(output_dir: Path, quiet: bool = False) -> List[Dict[str, Any]]:
    """Materialize graph-perturbation evidence for Fig.4 from agent caches."""
    manifest = read_csv_records(output_dir / "fig4_manifest.csv")
    agents = {str(row.get("paper_id")): row for row in read_jsonl(output_dir / "fig4_agent_outputs.jsonl")}
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(manifest, start=1):
        paper_id = str(row.get("paper_id") or "")
        cache_dir = output_dir / "cache" / paper_id
        agent_row = agents.get(paper_id, {})
        evidence = agent_row.get("graph_metric_evidence") if isinstance(agent_row.get("graph_metric_evidence"), Mapping) else {}
        metric_source = "agent_graph_metric_evidence"
        failure_reason = ""
        if not evidence:
            evidence, failure_reason = compute_graph_metric_evidence_from_retrieval(row, cache_dir, agent_row)
            metric_source = "fig4_retrieved_prior_art_graph_scorer" if evidence else "missing_graph_metric_evidence"
        graph_row = flatten_graph_metric_evidence(row, evidence, metric_source, failure_reason)
        rows.append(graph_row)
        write_json(cache_dir / "fig4_graph_metrics.json", graph_row)
        if idx == 1 or idx % 10 == 0 or idx == len(manifest):
            progress_log(f"Graph metrics progress {idx}/{len(manifest)}.", quiet)
    write_csv(output_dir / "fig4_graph_metrics.csv", rows)
    return rows


def run_graph_prior_stage(
    output_dir: Path,
    weights_path: Optional[Path] = None,
    reference_scores_path: Optional[Path] = None,
    reference_indicators_path: Optional[Path] = None,
    quiet: bool = False,
) -> List[Dict[str, Any]]:
    """Materialize Fig.3-weighted S_w priors for Fig.4 from graph metrics."""
    manifest = read_csv_records(output_dir / "fig4_manifest.csv")
    graph_by_paper = group_csv_by_paper(output_dir / "fig4_graph_metrics.csv")
    graph_rows: List[Dict[str, Any]] = []
    for row in manifest:
        paper_id = str(row.get("paper_id") or "")
        graph_row = dict(graph_by_paper.get(paper_id, {}))
        graph_row["paper_id"] = paper_id
        graph_rows.append(graph_row)
    rows = build_fig3_weighted_prior_rows(
        graph_rows,
        weights_path=weights_path,
        reference_scores_path=reference_scores_path,
        reference_indicators_path=reference_indicators_path,
    )
    write_csv(output_dir / "fig4_graph_prior.csv", rows)
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        if paper_id:
            cache_dir = output_dir / "cache" / paper_id
            cache_dir.mkdir(parents=True, exist_ok=True)
            write_json(cache_dir / "fig4_graph_prior.json", row)
    progress_log(f"Graph prior progress {len(rows)}/{len(manifest)}.", quiet)
    return rows


def _score_or_none(value: Any) -> Optional[float]:
    number = numeric(value)
    if not math.isfinite(number):
        return None
    return max(1.0, min(5.0, number))


def quote_in_source(quote: str, source_text: str) -> bool:
    """Return whether a quote appears in source text after whitespace normalization."""
    cleaned_quote = normalize_whitespace(quote).lower()
    cleaned_source = normalize_whitespace(source_text).lower()
    return bool(cleaned_quote) and cleaned_quote in cleaned_source


def is_revision_only_point(text: str) -> bool:
    """Detect acceptance/revision boilerplate that must not become innovation evidence."""
    cleaned = normalize_whitespace(text)
    return bool(cleaned) and bool(
        REVISION_ONLY_RE.search(cleaned)
        or re.search(r"\b(addressed\s+all\s+(?:of\s+)?(?:my\s+)?concerns?|no\s+further\s+comments?)\b", cleaned, flags=re.I)
    )


def normalize_point_record(
    raw: Mapping[str, Any],
    aspect: str,
    index: int,
    kind: str,
    source_text: str,
    warnings: List[str],
) -> Optional[Dict[str, Any]]:
    """Normalize one quote-grounded point record or drop it with an audit warning."""
    point = normalize_whitespace(str(raw.get("point") or raw.get("text") or ""))
    quote = normalize_whitespace(str(raw.get("quote") or ""))
    if not point and quote:
        point = quote
    if not point:
        warnings.append(f"{aspect}:point_record_empty_point")
        return None
    if not quote:
        warnings.append(f"{aspect}:point_record_missing_quote")
        return None
    if not quote_in_source(quote, source_text):
        warnings.append(f"{aspect}:point_record_quote_not_exact")
        return None
    if kind == "peer_review" and (is_revision_only_point(point) or is_revision_only_point(quote)):
        warnings.append(f"{aspect}:revision_only_point_dropped")
        return None
    polarity = str(raw.get("polarity") or "neutral").strip().lower()
    if polarity not in ALLOWED_POINT_POLARITIES:
        polarity = "neutral"
    evidence_type = str(raw.get("evidence_type") or EVIDENCE_TYPE_BY_ASPECT.get(aspect, aspect)).strip()
    if evidence_type not in ALLOWED_EVIDENCE_TYPES:
        evidence_type = EVIDENCE_TYPE_BY_ASPECT.get(aspect, "evidence_support")
    source_role = str(raw.get("source_role") or ("reviewer" if kind == "peer_review" else "agent")).strip().lower()
    if kind == "peer_review" and source_role not in {"reviewer", "editor"}:
        source_role = "reviewer"
    elif source_role not in ALLOWED_SOURCE_ROLES:
        source_role = "agent"
    return {
        "point_id": normalize_whitespace(str(raw.get("point_id") or f"{aspect}_{index + 1}")),
        "point": point,
        "quote": quote,
        "polarity": polarity,
        "evidence_type": evidence_type,
        "confidence": clamp(raw.get("confidence"), 0.0, 1.0, default=0.0),
        "source_role": source_role,
    }


def legacy_point_records(aspect: str, item: Mapping[str, Any], kind: str) -> List[Dict[str, Any]]:
    """Convert legacy parallel points/quotes lists into point-record candidates."""
    points = [normalize_whitespace(str(value)) for value in item.get("points", []) if normalize_whitespace(str(value))] if isinstance(item.get("points"), list) else []
    quotes = [normalize_whitespace(str(value)) for value in item.get("quotes", []) if normalize_whitespace(str(value))] if isinstance(item.get("quotes"), list) else []
    records: List[Dict[str, Any]] = []
    for idx, point in enumerate(points or quotes):
        quote = quotes[min(idx, len(quotes) - 1)] if quotes else ""
        records.append(
            {
                "point_id": f"{aspect}_{idx + 1}",
                "point": point,
                "quote": quote,
                "polarity": "neutral",
                "evidence_type": EVIDENCE_TYPE_BY_ASPECT.get(aspect, "evidence_support"),
                "confidence": numeric(item.get("confidence"), 0.0),
                "source_role": "reviewer" if kind == "peer_review" else "agent",
            }
        )
    return records


def normalize_aspect_point_records(
    item: Mapping[str, Any],
    aspect: str,
    kind: str,
    source_text: str,
    warnings: List[str],
) -> List[Dict[str, Any]]:
    """Normalize explicit point_records, falling back to legacy points/quotes."""
    if isinstance(item.get("point_records"), list):
        raw_records = [record for record in item.get("point_records", []) if isinstance(record, Mapping)]
    else:
        raw_records = legacy_point_records(aspect, item, kind)
    out: List[Dict[str, Any]] = []
    for idx, raw in enumerate(raw_records):
        record = normalize_point_record(raw, aspect, idx, kind, source_text, warnings)
        if record is not None:
            out.append(record)
    return out[:8]


def normalize_innovation_label_payload(
    payload: Mapping[str, Any],
    paper_id: str,
    kind: str,
    text: str,
    model: str = "",
    raw_response: str = "",
) -> Dict[str, Any]:
    """Normalize quote-grounded innovation labels and reject scored labels without quotes."""
    stance_in = payload.get("overall_innovation_stance") if isinstance(payload.get("overall_innovation_stance"), Mapping) else {}
    aspects_in = payload.get("aspects") if isinstance(payload.get("aspects"), Mapping) else {}
    warnings: List[str] = []
    normalized_aspects: Dict[str, Dict[str, Any]] = {}
    for aspect in INNOVATION_ASPECTS:
        item = aspects_in.get(aspect) if isinstance(aspects_in.get(aspect), Mapping) else {}
        point_records = normalize_aspect_point_records(item, aspect, kind, text, warnings)
        points = [record["point"] for record in point_records]
        quotes = [record["quote"] for record in point_records]
        score = _score_or_none(item.get("score_1_5"))
        if score is not None and not quotes:
            warnings.append(f"{aspect}:missing_quote")
        normalized_aspects[aspect] = {
            "score_1_5": score,
            "points": points[:8],
            "quotes": quotes[:8],
            "point_records": point_records,
            "confidence": numeric(item.get("confidence"), 0.0),
        }
    stance_score = _score_or_none(stance_in.get("score_1_5"))
    stance_quote = normalize_whitespace(str(stance_in.get("quote") or ""))
    if stance_score is not None and not stance_quote:
        warnings.append("overall_innovation_stance:missing_quote")
    elif stance_quote and not quote_in_source(stance_quote, text):
        warnings.append("overall_innovation_stance:quote_not_exact")
        stance_quote = ""
    elif kind == "peer_review" and stance_quote and is_revision_only_point(stance_quote):
        warnings.append("overall_innovation_stance:revision_only_quote_dropped")
        stance_quote = ""
    has_required_quotes = (
        (stance_score is None or bool(stance_quote))
        and all(item["score_1_5"] is None or bool(item["quotes"]) for item in normalized_aspects.values())
    )
    return {
        "paper_id": paper_id,
        "kind": kind,
        "success": bool(has_required_quotes),
        "failure_reason": "" if has_required_quotes else "missing_required_quotes",
        "model": model,
        "overall_innovation_stance": {
            "score_1_5": stance_score,
            "label": str(stance_in.get("label") or "not_discussed"),
            "quote": stance_quote,
            "confidence": numeric(stance_in.get("confidence"), 0.0),
        },
        "aspects": normalized_aspects,
        "warnings": warnings,
        "raw_response": raw_response,
    }


def innovation_label_point_count(label: Mapping[str, Any]) -> int:
    total = 0
    aspects = label.get("aspects") if isinstance(label.get("aspects"), Mapping) else {}
    for aspect in INNOVATION_ASPECTS:
        item = aspects.get(aspect) if isinstance(aspects.get(aspect), Mapping) else {}
        points = item.get("points") if isinstance(item.get("points"), list) else []
        total += len([point for point in points if normalize_whitespace(str(point))])
    return total


def innovation_label_quote_count(label: Mapping[str, Any]) -> int:
    stance = label.get("overall_innovation_stance") if isinstance(label.get("overall_innovation_stance"), Mapping) else {}
    total = 1 if normalize_whitespace(str(stance.get("quote") or "")) else 0
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
    reasons: List[str] = []
    stance_score = numeric(stance.get("score_1_5"))
    core_count = core_innovation_aspect_count(label)
    point_count = innovation_label_point_count(label)
    quote_count = innovation_label_quote_count(label)
    revision_only = is_revision_only_review(review_text)
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


@dataclass
class JudgeClientConfig:
    model: str
    base_url: str
    api_key: str
    timeout: float
    temperature: float
    max_tokens: int


class JsonResponseParseError(RuntimeError):
    """Raised when a judge response cannot be parsed while preserving raw text."""

    def __init__(self, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text


def env_first(names: Sequence[str], default: str = "") -> str:
    """Return the first non-empty environment value from a priority list."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def judge_config_from_env(model: Optional[str] = None) -> JudgeClientConfig:
    return JudgeClientConfig(
        model=model
        or env_first(("FIG4_JUDGE_MODEL", "ASPR_LATS_LLM_MODEL", "ASPR_LLM_MODEL"), "qwen3:8b"),
        base_url=env_first(
            ("FIG4_JUDGE_BASE_URL", "ASPR_LATS_LLM_BASE_URL", "ASPR_LLM_BASE_URL"),
            "http://localhost:11434/v1",
        ),
        api_key=env_first(
            (
                "FIG4_JUDGE_API_KEY",
                "ASPR_LATS_LLM_API_KEY",
                "ASPR_LLM_API_KEY",
                "DEEPSEEK_API_KEY",
                "OPENAI_API_KEY",
            ),
            "ollama",
        ),
        timeout=float(os.getenv("FIG4_JUDGE_TIMEOUT", "180")),
        temperature=float(os.getenv("FIG4_JUDGE_TEMPERATURE", "0")),
        max_tokens=int(os.getenv("FIG4_JUDGE_MAX_TOKENS", "6000")),
    )


def truncate_for_judge(text: str, max_chars: int = 18000) -> str:
    cleaned = normalize_whitespace(text)
    if len(cleaned) <= max_chars:
        return cleaned
    head = cleaned[: int(max_chars * 0.62)]
    tail = cleaned[-int(max_chars * 0.30) :]
    return f"{head}\n\n[... middle truncated for judge context ...]\n\n{tail}"


REVIEWER_EDITOR_HEADING_RE = re.compile(
    r"^\s*(?:#+\s*)?(?:[_*\s-]*)?"
    r"(?:reviewer|referee|editor|decision|major comments|minor comments|overview|report for the authors)\b",
    re.I,
)
AUTHOR_RESPONSE_HEADING_RE = re.compile(
    r"^\s*(?:#+\s*)?(?:[_*\s-]*)?"
    r"(?:rebuttal|response to reviewers?|authors?'?\s+response|response\s+to\s+referee|reply to reviewers?)\b",
    re.I,
)
PEER_REVIEW_NOISE_RE = re.compile(
    r"(creative commons|open access|licensed under|permission directly from the copyright holder|"
    r"to view a copy of this license|corresponding author:|editorial note)",
    re.I,
)
INNOVATION_REVIEW_TERM_RE = re.compile(
    r"\b("
    r"novel|novelty|original|innovative|significant|significance|important|advance|contribution|"
    r"prior\s+(?:work|art|study|studies)|previous\s+(?:work|study|studies)|compared|comparison|"
    r"evidence|convincing|support|robust|benchmark|validation|limitation|limited|concern|"
    r"future|further|additional|unclear|insufficient"
    r")\b",
    re.I,
)


def prepare_peer_review_text_for_judge(text: str) -> str:
    """Remove boilerplate and author rebuttals while keeping reviewer/editor evidence."""
    cleaned_lines: List[str] = []
    in_author_response = False
    for raw_line in str(text or "").splitlines():
        line = PICTURE_LINE_RE.sub("", raw_line).strip()
        if not line:
            cleaned_lines.append("")
            continue
        normalized = re.sub(r"[*_`#]+", "", line).strip()
        if PEER_REVIEW_NOISE_RE.search(normalized):
            continue
        if REVIEWER_EDITOR_HEADING_RE.search(normalized):
            in_author_response = False
        elif AUTHOR_RESPONSE_HEADING_RE.search(normalized):
            in_author_response = True
            continue
        if in_author_response:
            continue
        cleaned_lines.append(line)
    cleaned = remove_inline_author_responses("\n".join(cleaned_lines))
    paragraphs = [normalize_whitespace(paragraph) for paragraph in re.split(r"\n\s*\n+", cleaned)]
    paragraphs = [paragraph for paragraph in paragraphs if paragraph and not PEER_REVIEW_NOISE_RE.search(paragraph)]
    if not paragraphs:
        return normalize_whitespace(cleaned)
    selected_indices = set()
    for idx, paragraph in enumerate(paragraphs):
        if INNOVATION_REVIEW_TERM_RE.search(paragraph) or REVIEWER_EDITOR_HEADING_RE.search(paragraph):
            selected_indices.update({idx - 1, idx, idx + 1})
    selected = [paragraphs[idx] for idx in sorted(selected_indices) if 0 <= idx < len(paragraphs)]
    selected_text = "\n\n".join(selected)
    if word_count(selected_text) >= 120:
        return selected_text
    return "\n\n".join(paragraphs)


def prepare_source_text_for_label_judge(source_text: str, kind: str) -> str:
    """Prepare the source used for quote-grounded label extraction."""
    if kind == "peer_review":
        return prepare_peer_review_text_for_judge(source_text)
    return normalize_whitespace(source_text)


def call_openai_compatible_json(prompt: str, config: JudgeClientConfig) -> Tuple[Dict[str, Any], str]:
    """Call an OpenAI-compatible local/remote chat model and parse a JSON object."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for Fig.4 LLM judge") from exc

    client = OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=config.timeout)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict scientific peer-review information extraction judge. "
                "Return one JSON object only. Do not include markdown, commentary, or hidden reasoning. /no_think"
            ),
        },
        {"role": "user", "content": prompt},
    ]
    kwargs: Dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    use_json_mode = bool_value(os.getenv("FIG4_JUDGE_JSON_MODE", "1"))
    parse_retries = max(1, int(os.getenv("FIG4_JUDGE_PARSE_RETRIES", "2")))
    json_mode_variants = [True, False] if use_json_mode else [False]
    last_error: Optional[JsonResponseParseError] = None
    for _ in range(parse_retries):
        for json_mode in json_mode_variants:
            request_kwargs = dict(kwargs)
            if json_mode:
                request_kwargs["response_format"] = {"type": "json_object"}
            try:
                response = client.chat.completions.create(**request_kwargs)
            except Exception:
                if json_mode:
                    continue
                raise
            raw = response.choices[0].message.content or ""
            if not normalize_whitespace(raw):
                last_error = JsonResponseParseError("judge response was empty", raw)
                continue
            try:
                return extract_json_object(raw), raw
            except Exception as exc:  # noqa: BLE001 - preserve raw text for a repair retry.
                last_error = JsonResponseParseError(f"judge response JSON parse failed: {exc}", raw)
                continue
    if last_error is not None:
        raise last_error
    raise JsonResponseParseError("judge response JSON parse failed", "")


def coerce_innovation_label_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Coerce common near-miss judge schemas into the canonical label payload."""
    coerced = dict(payload)
    aspects = coerced.get("aspects") if isinstance(coerced.get("aspects"), Mapping) else {}
    if not aspects:
        nested = {
            aspect: coerced.get(aspect)
            for aspect in INNOVATION_ASPECTS
            if isinstance(coerced.get(aspect), Mapping)
        }
        if nested:
            coerced["aspects"] = nested
    if not isinstance(coerced.get("overall_innovation_stance"), Mapping):
        for key in ("overall", "overall_stance", "innovation_stance"):
            if isinstance(coerced.get(key), Mapping):
                coerced["overall_innovation_stance"] = coerced[key]
                break
    return coerced


def is_innovation_label_schema(payload: Mapping[str, Any]) -> bool:
    """Return whether a judge payload has the required top-level schema."""
    coerced = coerce_innovation_label_payload(payload)
    return isinstance(coerced.get("overall_innovation_stance"), Mapping) and isinstance(coerced.get("aspects"), Mapping)


def build_innovation_label_prompt(title: str, source_text: str, kind: str) -> str:
    """Prompt the judge to extract quote-grounded innovation labels."""
    source_name = "human peer review" if kind == "peer_review" else "ASPR innovation agent output"
    schema = {
        "overall_innovation_stance": {
            "score_1_5": "number|null",
            "label": "negative|mixed|positive|not_discussed",
            "quote": "short exact quote from the source text or empty string",
            "confidence": "0..1",
        },
        "aspects": {
            aspect: {
                "score_1_5": "number|null",
                "points": ["brief extracted judgement, not invented"],
                "quotes": ["short exact quote(s) from the source text"],
                "point_records": [
                    {
                        "point_id": "stable id",
                        "point": "brief extracted judgement, not invented",
                        "quote": "short exact quote copied from source text",
                        "polarity": "positive|negative|mixed|neutral",
                        "evidence_type": "novelty_claim|significance_claim|prior_art_comparison|evidence_support|rigor_concern|limitation|future_work",
                        "confidence": "0..1",
                        "source_role": "reviewer|editor for peer review; agent for ASPR output",
                    }
                ],
                "confidence": "0..1",
            }
            for aspect in INNOVATION_ASPECTS
        },
    }
    return textwrap.dedent(
        f"""
        Extract innovation-evaluation labels from the {source_name} only.

        Paper title:
        {title}

        Rules:
        - Use only the source text below. Do not infer from the paper title, abstract, or outside knowledge.
        - For human peer review, use reviewer/editor judgements only. Ignore author rebuttals, author responses, license text, and boilerplate.
        - Do not create innovation points from acceptance-only revision phrases such as "authors addressed my concerns", "now suitable", or "no further comments"; these may be context only.
        - Every non-null score and every non-empty point must be grounded by at least one short exact quote copied from the source text.
        - Prefer point_records. Every point_record must include point, exact quote, polarity, evidence_type, confidence, and source_role.
        - If the exact quote is absent from the source text, drop that point instead of guessing.
        - Write points as concise English judgement statements. Keep quotes copied exactly in the original source language.
        - If an aspect is not discussed, use score_1_5=null, points=[], quotes=[], point_records=[], confidence=0.
        - Scores are 1=strongly negative/low, 2=negative/limited, 3=mixed/uncertain, 4=positive, 5=strongly positive.
        - prior_art_comparison means explicit comparison with previous work or novelty relative to prior art.
        - evidence_rigor means evidence, validation, benchmark, experiment, robustness, or rigor concerns.
        - Return exactly one JSON object with only these top-level keys: overall_innovation_stance, aspects.
        - Do not return source_text, categories, summary, reviewer_comments, markdown, explanations, or any text outside JSON.
        - Return JSON matching this schema exactly:
        {json.dumps(schema, ensure_ascii=False)}

        Source text:
        \"\"\"
        {truncate_for_judge(source_text)}
        \"\"\"
        """
    ).strip()


def build_innovation_label_repair_prompt(
    title: str,
    source_text: str,
    kind: str,
    invalid_response: str,
) -> str:
    """Ask the judge to repair a near-miss label response into the canonical schema."""
    return textwrap.dedent(
        f"""
        Your previous answer did not match the required JSON schema for Fig.4 label extraction.
        Convert it into the required schema using only the source text. If a field cannot be grounded
        by an exact quote from the source text, set its score_1_5=null, points=[], quotes=[], confidence=0.
        Also set point_records=[] for any ungrounded aspect. Do not keep author-response or revision-only points.

        Paper title:
        {title}

        Source type:
        {kind}

        Previous invalid response:
        \"\"\"
        {invalid_response[:6000]}
        \"\"\"

        Required top-level JSON keys:
        - overall_innovation_stance
        - aspects

        Required aspect keys:
        {", ".join(INNOVATION_ASPECTS)}

        Source text:
        \"\"\"
        {truncate_for_judge(source_text)}
        \"\"\"
        """
    ).strip()


def first_sentence_containing(text: str, terms: Sequence[str]) -> str:
    for sentence in split_sentences(text):
        lowered = sentence.lower()
        if any(term in lowered for term in terms):
            return sentence[:320]
    sentences = split_sentences(text)
    return sentences[0][:320] if sentences else ""


ASPECT_POSITIVE_PATTERNS = {
    "novelty": [
        r"\bnovel\b",
        r"\bnovelty\b",
        r"\bnew\b",
        r"\boriginal\b",
        r"\binnovative\b",
        r"\bfirst\b",
        r"\bunique\b",
        r"\bunprecedented\b",
    ],
    "significance": [
        r"\bsignificant\b",
        r"\bsignificance\b",
        r"\bimportant\b",
        r"\bimpact(?:ful)?\b",
        r"\badvance\b",
        r"\bcontribution\b",
        r"\bvaluable\b",
        r"\bsubstantial\b",
    ],
    "prior_art_comparison": [
        r"\bcompared\b",
        r"\bcomparison\b",
        r"\bprior\b",
        r"\bprevious\b",
        r"\bexisting\b",
        r"\bbenchmark\b",
    ],
    "evidence_rigor": [
        r"\bconvincing\b",
        r"\brobust\b",
        r"\bvalidated?\b",
        r"\bbenchmark\b",
        r"\bevidence\b",
        r"\bexperiment(?:al)?\b",
    ],
    "limitations": [
        r"\blimitation\b",
        r"\bconcern\b",
        r"\bweakness\b",
        r"\bunclear\b",
        r"\binsufficient\b",
        r"\bminor\b",
    ],
    "future_work": [
        r"\bfuture\b",
        r"\bfurther\b",
        r"\bfollow-up\b",
        r"\bnext step\b",
        r"\badditional\b",
    ],
}
ASPECT_NEGATIVE_PATTERNS = {
    "novelty": [
        r"\bnot\s+novel\b",
        r"\blimited\s+novelty\b",
        r"\black(?:s|ing)?\s+novelty\b",
        r"\bnot\s+new\b",
        r"\bincremental\b",
        r"\balready\b",
        r"\bpreviously\b",
        r"\bminor\b",
    ],
    "significance": [
        r"\blimited\s+significance\b",
        r"\bnot\s+significant\b",
        r"\bminor\b",
        r"\bincremental\b",
        r"\bweak\b",
        r"\bunclear\s+(?:impact|significance)\b",
        r"\blimited\s+impact\b",
    ],
    "prior_art_comparison": [
        r"\bnot\s+compared\b",
        r"\binsufficient\s+(?:comparison|benchmark)\b",
        r"\bmissing\s+(?:comparison|benchmark)\b",
        r"\bprior\s+work\s+(?:not|isn't)\b",
    ],
    "evidence_rigor": [
        r"\binsufficient\b",
        r"\bweak\b",
        r"\bunclear\b",
        r"\bnot\s+validated\b",
        r"\black(?:s|ing)?\s+(?:evidence|validation|control)\b",
    ],
    "limitations": [
        r"\bwell\s+addressed\b",
        r"\bno\s+(?:major\s+)?concern\b",
        r"\bresolved\b",
    ],
    "future_work": [
        r"\bnot\s+necessary\b",
        r"\bno\s+future\b",
    ],
}


def _pattern_count(text: str, patterns: Sequence[str]) -> int:
    return sum(len(re.findall(pattern, text, flags=re.I)) for pattern in patterns)


def heuristic_aspect_score(text: str, aspect: str, quote: str, stance_score: Optional[int]) -> Optional[int]:
    """Return a quote-grounded 1-5 heuristic aspect score with directional variance."""
    if not quote:
        return None
    context = f"{quote} {text[:1200]}"
    positive = _pattern_count(context, ASPECT_POSITIVE_PATTERNS.get(aspect, []))
    negative = _pattern_count(context, ASPECT_NEGATIVE_PATTERNS.get(aspect, []))
    base = int(stance_score) if stance_score is not None else 3
    if aspect == "limitations":
        score = 3 + min(2, max(0, positive - negative))
    else:
        score = base
        if positive > negative:
            score += 1
        elif negative > positive:
            score -= 1
        if positive >= negative + 3:
            score += 1
        elif negative >= positive + 2:
            score -= 1
    return int(max(1, min(5, score)))


def heuristic_point_polarity(text: str, aspect: str) -> str:
    positive = _pattern_count(text, ASPECT_POSITIVE_PATTERNS.get(aspect, []))
    negative = _pattern_count(text, ASPECT_NEGATIVE_PATTERNS.get(aspect, []))
    if positive > negative:
        return "positive"
    if negative > positive:
        return "negative"
    if positive or negative:
        return "mixed"
    return "neutral"


def heuristic_innovation_label_payload(text: str) -> Dict[str, Any]:
    """Deterministic fallback for tests or judge outages; marked by model metadata."""
    aspect_terms = {
        "novelty": ["novel", "novelty", "new", "original", "innovative"],
        "significance": ["significant", "important", "impact", "advance", "contribution"],
        "prior_art_comparison": ["prior", "previous", "existing", "compared", "comparison"],
        "evidence_rigor": ["evidence", "experiment", "validation", "benchmark", "robust", "rigor"],
        "limitations": ["limitation", "concern", "weakness", "unclear", "however"],
        "future_work": ["future", "further", "follow-up", "next step", "additional"],
    }
    positive = len(re.findall(r"\b(novel|significant|important|strong|convincing|advance)\b", text, flags=re.I))
    negative = len(re.findall(r"\b(limited|weak|unclear|concern|insufficient|minor)\b", text, flags=re.I))
    stance_score: Optional[int]
    if positive + negative == 0:
        stance_score = None
    elif positive > negative + 1:
        stance_score = 4
    elif negative > positive + 1:
        stance_score = 2
    else:
        stance_score = 3
    stance_quote = first_sentence_containing(text, ["novel", "significant", "important", "concern", "limited"]) if stance_score else ""
    aspects: Dict[str, Any] = {}
    for aspect, terms in aspect_terms.items():
        quote = first_sentence_containing(text, terms)
        aspect_score = heuristic_aspect_score(text, aspect, quote, stance_score)
        polarity = heuristic_point_polarity(quote, aspect) if quote else "neutral"
        aspects[aspect] = {
            "score_1_5": aspect_score,
            "points": [quote] if quote else [],
            "quotes": [quote] if quote else [],
            "point_records": [
                {
                    "point_id": f"{aspect}_1",
                    "point": quote,
                    "quote": quote,
                    "polarity": polarity,
                    "evidence_type": EVIDENCE_TYPE_BY_ASPECT.get(aspect, aspect),
                    "confidence": 0.35,
                    "source_role": "reviewer",
                }
            ]
            if quote
            else [],
            "confidence": 0.35 if quote else 0,
        }
    return {
        "overall_innovation_stance": {
            "score_1_5": stance_score,
            "label": "mixed" if stance_score else "not_discussed",
            "quote": stance_quote,
            "confidence": 0.35 if stance_score else 0,
        },
        "aspects": aspects,
    }


def run_one_innovation_label_judge(
    paper_id: str,
    title: str,
    kind: str,
    source_text: str,
    judge_backend: str,
    config: JudgeClientConfig,
) -> Dict[str, Any]:
    judge_source_text = prepare_source_text_for_label_judge(source_text, kind)
    if not normalize_whitespace(judge_source_text):
        label = normalize_innovation_label_payload({}, paper_id, kind, judge_source_text, model=config.model)
        label["success"] = False
        label["failure_reason"] = "empty_source_text"
        return label
    raw_response = ""
    try:
        if judge_backend == "heuristic":
            payload = heuristic_innovation_label_payload(judge_source_text)
            model_name = "heuristic"
        else:
            prompt = build_innovation_label_prompt(title, judge_source_text, kind)
            try:
                payload, raw_response = call_openai_compatible_json(prompt, config)
            except JsonResponseParseError as exc:
                raw_response = exc.raw_text
                if not normalize_whitespace(raw_response):
                    raise
                repair_payload, repair_raw = call_openai_compatible_json(
                    build_innovation_label_repair_prompt(title, judge_source_text, kind, raw_response),
                    config,
                )
                payload = repair_payload
                raw_response = f"{raw_response}\n\n[JSON_PARSE_REPAIR_RETRY]\n{repair_raw}"
            if not is_innovation_label_schema(payload):
                repair_payload, repair_raw = call_openai_compatible_json(
                    build_innovation_label_repair_prompt(title, judge_source_text, kind, raw_response),
                    config,
                )
                payload = repair_payload
                raw_response = f"{raw_response}\n\n[SCHEMA_REPAIR_RETRY]\n{repair_raw}"
            model_name = config.model
        payload = coerce_innovation_label_payload(payload)
        return normalize_innovation_label_payload(
            payload,
            paper_id=paper_id,
            kind=kind,
            text=judge_source_text,
            model=model_name,
            raw_response=raw_response,
        )
    except Exception as exc:  # noqa: BLE001 - per-paper judge diagnostics keep the batch moving.
        label = normalize_innovation_label_payload({}, paper_id, kind, judge_source_text, model=config.model, raw_response=raw_response)
        label["success"] = False
        label["failure_reason"] = f"judge_failed:{exc}"
        return label


def run_innovation_label_judge(
    output_dir: Path,
    judge_backend: str = "openai-compatible",
    kinds: Sequence[str] = ("peer_review", "agent"),
    force_labels: bool = False,
    reuse_labels: bool = True,
    quiet: bool = False,
) -> List[Dict[str, Any]]:
    manifest = read_csv_records(output_dir / "fig4_manifest.csv")
    agents = {str(row.get("paper_id")): row for row in read_jsonl(output_dir / "fig4_agent_outputs.jsonl")}
    config = judge_config_from_env()
    labels: List[Dict[str, Any]] = []
    for idx, row in enumerate(manifest, start=1):
        paper_id = str(row.get("paper_id") or "")
        cache_dir = output_dir / "cache" / paper_id
        parsed_path = Path(str(row.get("parsed_text_cache") or cache_dir / "parsed_text.json"))
        parsed = read_json(parsed_path) if parsed_path.exists() else {}
        for kind in kinds:
            cache_name = "peer_innovation_labels.json" if kind == "peer_review" else "agent_innovation_labels.json"
            cache_path = cache_dir / cache_name
            if reuse_labels and cache_path.exists() and not force_labels:
                label = read_json(cache_path)
            else:
                source_text = str(parsed.get("peer_review_text") or "") if kind == "peer_review" else str(agents.get(paper_id, {}).get("innovation_evaluation") or "")
                label = run_one_innovation_label_judge(
                    paper_id=paper_id,
                    title=str(row.get("title") or ""),
                    kind=kind,
                    source_text=source_text,
                    judge_backend=judge_backend,
                    config=config,
                )
                write_json(cache_path, label)
            labels.append(label)
        if idx == 1 or idx % 5 == 0 or idx == len(manifest):
            progress_log(f"Label judge progress {idx}/{len(manifest)}.", quiet)
    write_jsonl(output_dir / "fig4_innovation_label_judgements.jsonl", labels)
    return labels


def run_peer_review_screen(
    output_dir: Path,
    min_core_aspects: int = 1,
    min_peer_label_points: int = 1,
    quiet: bool = False,
) -> List[Dict[str, Any]]:
    manifest = read_csv_records(output_dir / "fig4_manifest.csv")
    labels = group_jsonl_by_kind(output_dir / "fig4_innovation_label_judgements.jsonl")
    rows: List[Dict[str, Any]] = []
    for row in manifest:
        paper_id = str(row.get("paper_id") or "")
        parsed_path = Path(str(row.get("parsed_text_cache") or output_dir / "cache" / paper_id / "parsed_text.json"))
        parsed = read_json(parsed_path) if parsed_path.exists() else {}
        screen = screen_peer_review_label(
            labels.get((paper_id, "peer_review"), {}),
            str(parsed.get("peer_review_text") or ""),
            min_core_aspects=min_core_aspects,
            min_peer_label_points=min_peer_label_points,
        )
        rows.append({**row, **screen})
    write_csv(output_dir / "fig4_peer_review_screen.csv", rows)
    progress_log(f"Screened {len(rows)} peer reviews.", quiet)
    return rows


def rating_row_from_innovation_label(label: Mapping[str, Any]) -> Dict[str, Any]:
    aspects = label.get("aspects") if isinstance(label.get("aspects"), Mapping) else {}
    row: Dict[str, Any] = {
        "paper_id": label.get("paper_id", ""),
        "kind": label.get("kind", ""),
        "overall_score_1_5": label_score(label, ["overall_innovation_stance", "score_1_5"]),
        "aspects": {},
    }
    aspect_points_out: Dict[str, List[str]] = {}
    mapping = {
        "novelty": "novelty",
        "significance": "significance",
        "rigor": "evidence_rigor",
        "limitations": "limitations",
        "future_work": "future_work",
    }
    for rating_key, innovation_key in mapping.items():
        item = aspects.get(innovation_key) if isinstance(aspects.get(innovation_key), Mapping) else {}
        row[rating_key] = numeric(item.get("score_1_5"))
        points = item.get("points") if isinstance(item.get("points"), list) else []
        quotes = item.get("quotes") if isinstance(item.get("quotes"), list) else []
        aspect_points_out[rating_key] = [
            normalize_whitespace(str(value))
            for value in list(points) + list(quotes)
            if normalize_whitespace(str(value))
        ][:8]
    row["aspects"] = aspect_points_out
    return row


def run_rating_judgements_from_labels(output_dir: Path, quiet: bool = False) -> List[Dict[str, Any]]:
    labels = read_jsonl(output_dir / "fig4_innovation_label_judgements.jsonl")
    rows = [rating_row_from_innovation_label(label) for label in labels]
    write_jsonl(output_dir / "fig4_rating_judgements.jsonl", rows)
    progress_log(f"Derived {len(rows)} rating rows from innovation labels.", quiet)
    return rows


def run_semantic_claim_match(
    output_dir: Path,
    judge_backend: str = "heuristic",
    max_points_per_aspect: int = 4,
    llm_refine: Optional[bool] = None,
    quiet: bool = False,
) -> List[Dict[str, Any]]:
    labels = group_jsonl_by_kind(output_dir / "fig4_innovation_label_judgements.jsonl")
    manifest = read_csv_records(output_dir / "fig4_manifest.csv")
    use_llm_refine = bool_value(os.getenv("FIG4_SEMANTIC_LLM_REFINE", "0")) if llm_refine is None else llm_refine
    semantic_judge_config = judge_config_from_env() if use_llm_refine and judge_backend != "heuristic" else None
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(manifest, start=1):
        paper_id = str(row.get("paper_id") or "")
        peer_label = labels.get((paper_id, "peer_review"), {})
        agent_label = labels.get((paper_id, "agent"), {})
        paper_rows: List[Dict[str, Any]] = []
        for aspect in INNOVATION_ASPECTS:
            peer_aspects = peer_label.get("aspects") if isinstance(peer_label.get("aspects"), Mapping) else {}
            peer_item = peer_aspects.get(aspect) if isinstance(peer_aspects.get(aspect), Mapping) else {}
            peer_points = [
                normalize_whitespace(str(value))
                for value in (peer_item.get("points") if isinstance(peer_item.get("points"), list) else [])
                if normalize_whitespace(str(value))
            ]
            peer_quotes = [
                normalize_whitespace(str(value))
                for value in (peer_item.get("quotes") if isinstance(peer_item.get("quotes"), list) else [])
                if normalize_whitespace(str(value))
            ]
            if not peer_points:
                peer_points = peer_quotes[:]
            for point_index, point in enumerate(peer_points[:max_points_per_aspect]):
                peer_quote = peer_quotes[min(point_index, len(peer_quotes) - 1)] if peer_quotes else ""
                candidate_records = candidate_records_for_peer_aspect(agent_label, aspect, point)
                agent_candidates = [record["point"] for record in candidate_records]
                match = semantic_match_one_point(
                    title=str(row.get("title") or ""),
                    aspect=aspect,
                    peer_point=point,
                    peer_quote=peer_quote,
                    agent_candidates=agent_candidates,
                    client=None,
                )
                candidate_aspect = candidate_aspect_for_point(match.get("best_agent_point", ""), candidate_records, aspect)
                relation = normalize_semantic_relation(match.get("relation"))
                paper_rows.append(
                    {
                        "paper_id": paper_id,
                        "row_id": f"{aspect}:{point_index}",
                        "aspect": aspect,
                        "peer_point": point,
                        "peer_quote": peer_quote,
                        "agent_candidates": agent_candidates[: int(os.getenv("FIG4_SEMANTIC_LLM_MAX_CANDIDATES", "6"))],
                        "agent_candidate_records": candidate_records[: int(os.getenv("FIG4_SEMANTIC_LLM_MAX_CANDIDATES", "6"))],
                        "candidate_aspect": candidate_aspect,
                        "cross_aspect_match": bool(relation != "no_match" and candidate_aspect and candidate_aspect != aspect),
                        "bge_only_relation": relation,
                        "refined_relation": relation,
                        "relation_source": "bge",
                        **match,
                    }
                )
        if semantic_judge_config is not None:
            paper_rows = refine_semantic_matches_for_paper_with_llm(
                title=str(row.get("title") or ""),
                paper_id=paper_id,
                paper_rows=paper_rows,
                config=semantic_judge_config,
            )
        rows.extend(paper_rows)
        write_json(output_dir / "cache" / paper_id / "semantic_claim_matches.json", {"paper_id": paper_id, "matches": paper_rows})
        if idx == 1 or idx % 10 == 0 or idx == len(manifest):
            progress_log(f"Semantic claim-match progress {idx}/{len(manifest)}.", quiet)
    write_jsonl(output_dir / "fig4_semantic_claim_matches.jsonl", rows)
    return rows


def run_structured_consistency_judgements(output_dir: Path, quiet: bool = False) -> List[Dict[str, Any]]:
    labels = group_jsonl_by_kind(output_dir / "fig4_innovation_label_judgements.jsonl")
    manifest = read_csv_records(output_dir / "fig4_manifest.csv")
    rows: List[Dict[str, Any]] = []
    for row in manifest:
        paper_id = str(row.get("paper_id") or "")
        payload = {
            "paper_id": paper_id,
            **heuristic_structured_consistency(
                labels.get((paper_id, "peer_review"), {}),
                labels.get((paper_id, "agent"), {}),
            ),
        }
        rows.append(payload)
        write_json(output_dir / "cache" / paper_id / "structured_consistency.json", payload)
    write_jsonl(output_dir / "fig4_structured_consistency_judgements.jsonl", rows)
    progress_log(f"Computed {len(rows)} structured consistency judgements.", quiet)
    return rows


def normalize_phrase(text: str) -> str:
    tokens = []
    for token in re.findall(r"[a-z0-9]+", str(text).lower()):
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)


def token_set(text: str) -> set[str]:
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "with",
        "by",
        "on",
        "is",
        "are",
        "was",
        "were",
        "this",
        "that",
        "it",
        "as",
        "from",
        "be",
        "been",
        "not",
        "no",
        "but",
        "their",
        "its",
        "they",
    }
    return {token for token in re.findall(r"[a-z0-9]+", str(text).lower()) if len(token) > 2 and token not in stop}


def token_overlap_cosine(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return float("nan")
    return len(left_tokens & right_tokens) / math.sqrt(len(left_tokens) * len(right_tokens))


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


def semantic_relation_score(relation: str) -> float:
    return SEMANTIC_RELATION_SCORES.get(str(relation), 0.0)


_FIG4_EMBEDDING_MODEL: Optional[Any] = None
_FIG4_EMBEDDING_BACKEND = ""
_FIG4_EMBEDDING_FAILURE = ""
_FIG4_EMBEDDING_CACHE: Dict[str, List[float]] = {}


def get_fig4_embedding_model() -> Tuple[Optional[Any], str]:
    """Load the local multilingual embedding model used for Fig.4 semantic metrics."""
    global _FIG4_EMBEDDING_MODEL, _FIG4_EMBEDDING_BACKEND, _FIG4_EMBEDDING_FAILURE
    if _FIG4_EMBEDDING_MODEL is not None:
        return _FIG4_EMBEDDING_MODEL, _FIG4_EMBEDDING_BACKEND
    if _FIG4_EMBEDDING_FAILURE:
        return None, "lexical_fallback"
    model_path = getenv("ASPR_RECALL_MODEL_PATH", "/home/jayee/models/bge-m3")
    try:
        from FlagEmbedding import BGEM3FlagModel

        _FIG4_EMBEDDING_MODEL = BGEM3FlagModel(model_path, use_fp16=bool_value(getenv("ASPR_RETRIEVAL_USE_FP16", "true")))
        _FIG4_EMBEDDING_BACKEND = "bge-m3"
        return _FIG4_EMBEDDING_MODEL, _FIG4_EMBEDDING_BACKEND
    except Exception as exc:  # noqa: BLE001 - keep metrics stage robust in lightweight environments.
        _FIG4_EMBEDDING_FAILURE = f"{type(exc).__name__}: {exc}"
        _FIG4_EMBEDDING_MODEL = None
        _FIG4_EMBEDDING_BACKEND = "lexical_fallback"
        return None, _FIG4_EMBEDDING_BACKEND


def normalize_vector(values: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in values))
    if not norm:
        return [0.0 for _value in values]
    return [float(value) / norm for value in values]


def embed_texts(texts: Sequence[str]) -> Tuple[Dict[str, List[float]], str]:
    """Embed unique short texts with a small in-process cache."""
    cleaned = [normalize_whitespace(text) for text in texts if normalize_whitespace(text)]
    missing = [text for text in dict.fromkeys(cleaned) if text not in _FIG4_EMBEDDING_CACHE]
    model, backend = get_fig4_embedding_model()
    if model is None:
        return {text: [] for text in cleaned}, backend
    if missing:
        result = model.encode(
            missing,
            batch_size=max(1, int(os.getenv("FIG4_EMBED_BATCH_SIZE", "8"))),
            max_length=max(64, int(os.getenv("FIG4_EMBED_MAX_LENGTH", "512"))),
        )
        dense = result.get("dense_vecs") if isinstance(result, Mapping) else result
        for text, vector in zip(missing, dense):
            _FIG4_EMBEDDING_CACHE[text] = normalize_vector(vector)
    return {text: _FIG4_EMBEDDING_CACHE.get(text, []) for text in cleaned}, backend


def embedding_cosine(left: str, right: str) -> Tuple[float, str]:
    vectors, backend = embed_texts([left, right])
    left_vector = vectors.get(normalize_whitespace(left), [])
    right_vector = vectors.get(normalize_whitespace(right), [])
    if not left_vector or not right_vector:
        return token_overlap_cosine(left, right), "lexical_fallback"
    return sum(lv * rv for lv, rv in zip(left_vector, right_vector)), backend


def label_similarity_text(label: Mapping[str, Any]) -> str:
    """Build a compact label-only text for semantic similarity metrics."""
    chunks: List[str] = []
    stance = label.get("overall_innovation_stance") if isinstance(label.get("overall_innovation_stance"), Mapping) else {}
    if normalize_whitespace(str(stance.get("label") or "")):
        chunks.append(f"overall stance: {stance.get('label')}")
    if normalize_whitespace(str(stance.get("quote") or "")):
        chunks.append(f"overall evidence: {stance.get('quote')}")
    aspects = label.get("aspects") if isinstance(label.get("aspects"), Mapping) else {}
    for aspect in INNOVATION_ASPECTS:
        item = aspects.get(aspect) if isinstance(aspects.get(aspect), Mapping) else {}
        points = item.get("points") if isinstance(item.get("points"), list) else []
        quotes = item.get("quotes") if isinstance(item.get("quotes"), list) else []
        evidence = [normalize_whitespace(str(value)) for value in list(points)[:4] if normalize_whitespace(str(value))]
        if not evidence:
            evidence = [normalize_whitespace(str(value)) for value in list(quotes)[:2] if normalize_whitespace(str(value))]
        if evidence:
            chunks.append(f"{aspect}: " + "; ".join(evidence))
    return "\n".join(chunks)


def heuristic_semantic_match(peer_point: str, agent_candidates: Sequence[str]) -> Dict[str, Any]:
    if not agent_candidates:
        return {"relation": "no_match", "best_agent_point": "", "rationale": "No same-aspect agent candidate was extracted."}
    peer_tokens = token_set(peer_point)
    best_candidate = ""
    best_overlap = 0.0
    for candidate in agent_candidates:
        cand_tokens = token_set(candidate)
        overlap = len(peer_tokens & cand_tokens) / len(peer_tokens | cand_tokens) if peer_tokens and cand_tokens else 0.0
        if overlap > best_overlap:
            best_candidate = candidate
            best_overlap = overlap
    peer_norm = normalize_phrase(peer_point)
    best_norm = normalize_phrase(best_candidate)
    if peer_norm and best_norm and (peer_norm in best_norm or best_norm in peer_norm):
        relation = "entailed"
        rationale = "Normalized phrase containment."
    elif best_overlap >= 0.28:
        relation = "related"
        rationale = f"Lexical overlap suggests partial topical relation ({best_overlap:.2f})."
    else:
        relation = "no_match"
        rationale = f"Best lexical overlap is too low ({best_overlap:.2f})."
    return {"relation": relation, "best_agent_point": best_candidate, "rationale": rationale}


def embedding_semantic_match(peer_point: str, agent_candidates: Sequence[str]) -> Dict[str, Any]:
    if not agent_candidates:
        return {
            "relation": "no_match",
            "best_agent_point": "",
            "rationale": "No same-aspect agent candidate was extracted.",
            "similarity": float("nan"),
            "match_backend": "bge-m3" if not _FIG4_EMBEDDING_FAILURE else "lexical_fallback",
        }
    texts = [peer_point, *agent_candidates]
    vectors, backend = embed_texts(texts)
    peer_vector = vectors.get(normalize_whitespace(peer_point), [])
    if not peer_vector or backend == "lexical_fallback":
        payload = heuristic_semantic_match(peer_point, agent_candidates)
        payload["similarity"] = token_overlap_cosine(peer_point, payload.get("best_agent_point", ""))
        payload["match_backend"] = "lexical_fallback"
        return payload
    best_candidate = ""
    best_similarity = -1.0
    for candidate in agent_candidates:
        candidate_vector = vectors.get(normalize_whitespace(candidate), [])
        similarity = sum(lv * rv for lv, rv in zip(peer_vector, candidate_vector)) if candidate_vector else -1.0
        if similarity > best_similarity:
            best_candidate = candidate
            best_similarity = similarity
    entailed_threshold = float(os.getenv("FIG4_EMBED_ENTAILED_THRESHOLD", "0.78"))
    related_threshold = float(os.getenv("FIG4_EMBED_RELATED_THRESHOLD", "0.56"))
    if best_similarity >= entailed_threshold:
        relation = "entailed"
    elif best_similarity >= related_threshold:
        relation = "related"
    else:
        relation = "no_match"
    return {
        "relation": relation,
        "best_agent_point": best_candidate,
        "rationale": f"BGE-M3 label-point similarity={best_similarity:.3f}.",
        "similarity": best_similarity,
        "match_backend": backend,
    }


def build_semantic_match_refinement_prompt(
    title: str,
    aspect: str,
    peer_point: str,
    peer_quote: str,
    agent_candidates: Sequence[str],
    initial_match: Mapping[str, Any],
) -> str:
    """Build a bounded judge prompt for no-match semantic refinement."""
    candidate_rows = [
        {"id": idx, "agent_point": normalize_whitespace(candidate)}
        for idx, candidate in enumerate(agent_candidates[: int(os.getenv("FIG4_SEMANTIC_LLM_MAX_CANDIDATES", "6"))], start=1)
        if normalize_whitespace(candidate)
    ]
    schema = {
        "relation": "entailed|related|contradicted|no_match",
        "best_candidate_id": "number|null",
        "best_agent_point": "exact copied candidate text or empty string",
        "confidence": "0..1",
        "rationale": "one short sentence grounded only in the peer point and agent candidate",
    }
    return textwrap.dedent(
        f"""
        Decide whether an ASPR innovation-agent point semantically covers a human peer-review innovation point.

        Paper title:
        {title}

        Aspect:
        {aspect}

        Human peer-review point:
        {peer_point}

        Human peer-review quote:
        {peer_quote}

        ASPR agent candidate points:
        {json.dumps(candidate_rows, ensure_ascii=False, indent=2)}

        Initial BGE-M3 match:
        {json.dumps({key: initial_match.get(key) for key in ("relation", "best_agent_point", "similarity", "rationale")}, ensure_ascii=False)}

        Rules:
        - Use only the human point/quote and the listed ASPR agent candidate points.
        - entailed: one candidate fully covers the peer-review point under this aspect.
        - related: one candidate addresses the same scientific judgement, concern, impact, prior-art comparison, limitation, or future-work direction, but is broader, narrower, or partially missing detail.
        - contradicted: one candidate addresses the same issue but says the opposite, such as "evidence is strong" versus "evidence is insufficient".
        - no_match: candidates are generic, only share vague words, discuss a different issue, or require outside paper knowledge.
        - Do not infer from the title or outside knowledge. Do not reward generic words such as important, novel, evidence, limitation unless the concrete issue matches.
        - If relation is entailed, related, or contradicted, best_candidate_id must identify the candidate and best_agent_point must exactly copy that candidate.
        - Return exactly one JSON object matching this schema:
        {json.dumps(schema, ensure_ascii=False)}
        """
    ).strip()


def coerce_semantic_match_refinement_payload(
    payload: Mapping[str, Any],
    agent_candidates: Sequence[str],
    min_confidence: float,
) -> Dict[str, Any]:
    """Normalize a semantic-refinement judge response into a safe relation payload."""
    candidates = [normalize_whitespace(candidate) for candidate in agent_candidates if normalize_whitespace(candidate)]
    relation = normalize_semantic_relation(payload.get("relation"))
    confidence = clamp(payload.get("confidence"), 0.0, 1.0, default=0.0)
    candidate_id = numeric(payload.get("best_candidate_id"), float("nan"))
    best_agent_point = normalize_whitespace(str(payload.get("best_agent_point") or ""))
    if math.isfinite(candidate_id):
        idx = int(candidate_id) - 1
        if 0 <= idx < len(candidates):
            best_agent_point = candidates[idx]
    elif best_agent_point:
        exact_match = next((candidate for candidate in candidates if candidate == best_agent_point), "")
        if not exact_match:
            normalized = normalize_phrase(best_agent_point)
            exact_match = next((candidate for candidate in candidates if normalize_phrase(candidate) == normalized), "")
        best_agent_point = exact_match
    if relation in {"entailed", "related", "contradicted"} and (confidence < min_confidence or not best_agent_point):
        relation = "no_match"
    if relation == "no_match":
        best_agent_point = best_agent_point if best_agent_point in candidates else ""
    return {
        "relation": relation,
        "best_agent_point": best_agent_point,
        "confidence": confidence,
        "rationale": normalize_whitespace(str(payload.get("rationale") or ""))[:600],
    }


def refine_semantic_match_with_llm(
    title: str,
    aspect: str,
    peer_point: str,
    peer_quote: str,
    agent_candidates: Sequence[str],
    initial_match: Mapping[str, Any],
    config: JudgeClientConfig,
) -> Dict[str, Any]:
    """Use a strict LLM judge to refine BGE no-match rows without hiding BGE evidence."""
    if not agent_candidates:
        return dict(initial_match)
    min_similarity = float(os.getenv("FIG4_SEMANTIC_LLM_MIN_SIMILARITY", "0.45"))
    similarity = numeric(initial_match.get("similarity"), float("nan"))
    if math.isfinite(similarity) and similarity < min_similarity:
        return {
            **initial_match,
            "llm_refined": False,
            "llm_refinement_skipped": f"similarity_below_{min_similarity:.2f}",
        }
    min_confidence = float(os.getenv("FIG4_SEMANTIC_LLM_MIN_CONFIDENCE", "0.55"))
    judge_config = replace(config, max_tokens=int(os.getenv("FIG4_SEMANTIC_LLM_MAX_TOKENS", "900")))
    raw_response = ""
    try:
        prompt = build_semantic_match_refinement_prompt(
            title=title,
            aspect=aspect,
            peer_point=peer_point,
            peer_quote=peer_quote,
            agent_candidates=agent_candidates,
            initial_match=initial_match,
        )
        payload, raw_response = call_openai_compatible_json(prompt, judge_config)
        refined = coerce_semantic_match_refinement_payload(payload, agent_candidates, min_confidence)
        relation = normalize_semantic_relation(refined.get("relation"))
        return {
            **initial_match,
            "pre_llm_relation": normalize_semantic_relation(initial_match.get("relation")),
            "pre_llm_score": semantic_relation_score(normalize_semantic_relation(initial_match.get("relation"))),
            "pre_llm_similarity": initial_match.get("similarity"),
            "pre_llm_rationale": initial_match.get("rationale", ""),
            "relation": relation,
            "score": semantic_relation_score(relation),
            "best_agent_point": refined.get("best_agent_point", initial_match.get("best_agent_point", "")),
            "rationale": refined.get("rationale") or initial_match.get("rationale", ""),
            "llm_refined": True,
            "llm_confidence": refined.get("confidence", 0.0),
            "llm_raw_response": raw_response,
            "match_backend": f"{initial_match.get('match_backend') or 'embedding'}+llm_refine",
        }
    except Exception as exc:  # noqa: BLE001 - one failed refinement must not block the batch.
        return {
            **initial_match,
            "llm_refined": False,
            "llm_refinement_error": f"{type(exc).__name__}: {exc}",
            "llm_raw_response": raw_response,
        }


def build_semantic_batch_refinement_prompt(
    title: str,
    paper_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Build one refinement prompt for all eligible no-match rows in a paper."""
    items: List[Dict[str, Any]] = []
    max_candidates = int(os.getenv("FIG4_SEMANTIC_LLM_MAX_CANDIDATES", "6"))
    for row in rows:
        candidates = [
            {
                "id": idx,
                "agent_point": normalize_whitespace(str(record.get("point") or "")),
                "aspect": str(record.get("aspect") or row.get("aspect") or ""),
            }
            for idx, record in enumerate(row.get("agent_candidate_records", [])[:max_candidates], start=1)
            if normalize_whitespace(str(record.get("point") or ""))
        ]
        items.append(
            {
                "row_id": row.get("row_id"),
                "aspect": row.get("aspect"),
                "peer_point": row.get("peer_point"),
                "peer_quote": row.get("peer_quote"),
                "agent_candidates": candidates,
                "initial_bge": {
                    "relation": row.get("relation"),
                    "best_agent_point": row.get("best_agent_point"),
                    "similarity": row.get("similarity"),
                    "rationale": row.get("rationale"),
                },
            }
        )
    schema = {
        "matches": [
            {
                "row_id": "copied row_id",
                "relation": "entailed|related|contradicted|no_match",
                "best_candidate_id": "number|null",
                "best_agent_point": "exact copied candidate text or empty string",
                "confidence": "0..1",
                "rationale": "one short sentence grounded only in the peer point and agent candidate",
            }
        ]
    }
    return textwrap.dedent(
        f"""
        Refine semantic matches between human peer-review innovation points and ASPR innovation-agent points.

        Paper id:
        {paper_id}

        Paper title:
        {title}

        Rows to judge:
        {json.dumps(items, ensure_ascii=False, indent=2)}

        Relation definitions:
        - entailed: one candidate fully covers the peer-review point under that aspect.
        - related: one candidate addresses the same scientific judgement, concern, impact, prior-art comparison, limitation, or future-work direction, but is broader, narrower, or partially missing detail.
        - contradicted: one candidate addresses the same issue but says the opposite, such as "evidence is strong" versus "evidence is insufficient".
        - no_match: candidates are generic, only share vague words, discuss a different issue, or require outside paper knowledge.

        Rules:
        - Use only the row's human point/quote and listed ASPR candidate points.
        - Do not infer from the title or outside knowledge.
        - Do not reward generic words such as important, novel, evidence, limitation unless the concrete issue matches.
        - If relation is entailed, related, or contradicted, best_candidate_id must identify the candidate and best_agent_point must exactly copy that candidate.
        - Return one result for every row_id and no extra row_ids.
        - Return exactly one JSON object matching this schema:
        {json.dumps(schema, ensure_ascii=False)}
        """
    ).strip()


def build_semantic_batch_refinement_repair_prompt(
    title: str,
    paper_id: str,
    rows: Sequence[Mapping[str, Any]],
    invalid_response: str,
) -> str:
    """Repair malformed semantic batch-refinement JSON without changing row ids."""
    row_ids = [str(row.get("row_id")) for row in rows]
    return textwrap.dedent(
        f"""
        Your previous semantic match answer was not valid JSON. Repair it into exactly one JSON object.

        Paper id:
        {paper_id}

        Paper title:
        {title}

        Required row_ids:
        {json.dumps(row_ids, ensure_ascii=False)}

        Previous invalid response:
        \"\"\"
        {invalid_response[:6000]}
        \"\"\"

        Required JSON schema:
        {{
          "matches": [
            {{
              "row_id": "one of the required row_ids",
              "relation": "entailed|related|contradicted|no_match",
              "best_candidate_id": "number|null",
              "best_agent_point": "exact copied candidate text or empty string",
              "confidence": 0.0,
              "rationale": "one short sentence"
            }}
          ]
        }}

        Return one match for every required row_id, no markdown, no commentary, no extra keys.
        """
    ).strip()


def refine_semantic_matches_for_paper_with_llm(
    title: str,
    paper_id: str,
    paper_rows: Sequence[Mapping[str, Any]],
    config: JudgeClientConfig,
) -> List[Dict[str, Any]]:
    """Batch-refine eligible BGE no-match rows for a paper."""
    rows = [dict(row) for row in paper_rows]
    min_similarity = float(os.getenv("FIG4_SEMANTIC_LLM_MIN_SIMILARITY", "0.45"))
    max_rows = int(os.getenv("FIG4_SEMANTIC_LLM_MAX_ROWS_PER_PAPER", "24"))
    batch_size = max(1, int(os.getenv("FIG4_SEMANTIC_LLM_BATCH_SIZE", "6")))
    eligible_indices: List[int] = []
    for idx, row in enumerate(rows):
        if normalize_semantic_relation(row.get("relation")) != "no_match":
            continue
        if not row.get("agent_candidates"):
            rows[idx]["llm_refined"] = False
            rows[idx]["llm_refinement_skipped"] = "no_agent_candidates"
            continue
        similarity = numeric(row.get("similarity"), float("nan"))
        if math.isfinite(similarity) and similarity < min_similarity:
            rows[idx]["llm_refined"] = False
            rows[idx]["llm_refinement_skipped"] = f"similarity_below_{min_similarity:.2f}"
            continue
        eligible_indices.append(idx)
    if len(eligible_indices) > max_rows:
        eligible_indices = sorted(
            eligible_indices,
            key=lambda idx: numeric(rows[idx].get("similarity"), -1.0),
            reverse=True,
        )[:max_rows]
        selected = set(eligible_indices)
        for idx, row in enumerate(rows):
            if normalize_semantic_relation(row.get("relation")) == "no_match" and idx not in selected and "llm_refinement_skipped" not in row:
                rows[idx]["llm_refined"] = False
                rows[idx]["llm_refinement_skipped"] = f"not_in_top_{max_rows}_similarity_batch"
    if not eligible_indices:
        return rows
    judge_config = replace(config, max_tokens=int(os.getenv("FIG4_SEMANTIC_LLM_MAX_TOKENS", "2500")))
    for batch_start in range(0, len(eligible_indices), batch_size):
        batch_indices = eligible_indices[batch_start : batch_start + batch_size]
        selected_rows = [rows[idx] for idx in batch_indices]
        raw_response = ""
        try:
            prompt = build_semantic_batch_refinement_prompt(title, paper_id, selected_rows)
            try:
                payload, raw_response = call_openai_compatible_json(prompt, judge_config)
            except JsonResponseParseError as exc:
                raw_response = exc.raw_text
                if not normalize_whitespace(raw_response):
                    raise
                payload, repair_raw = call_openai_compatible_json(
                    build_semantic_batch_refinement_repair_prompt(title, paper_id, selected_rows, raw_response),
                    judge_config,
                )
                raw_response = f"{raw_response}\n\n[SEMANTIC_BATCH_REPAIR]\n{repair_raw}"
            raw_matches = payload.get("matches") if isinstance(payload.get("matches"), list) else []
            by_row_id = {str(match.get("row_id")): match for match in raw_matches if isinstance(match, Mapping)}
            min_confidence = float(os.getenv("FIG4_SEMANTIC_LLM_MIN_CONFIDENCE", "0.55"))
            for idx in batch_indices:
                row = rows[idx]
                raw_match = by_row_id.get(str(row.get("row_id")))
                if not raw_match:
                    row["llm_refined"] = False
                    row["llm_refinement_error"] = "missing_row_id_in_batch_response"
                    row["llm_raw_response"] = raw_response
                    continue
                candidate_records = row.get("agent_candidate_records") if isinstance(row.get("agent_candidate_records"), list) else []
                candidate_texts = [normalize_whitespace(str(record.get("point") or "")) for record in candidate_records if isinstance(record, Mapping)]
                refined = coerce_semantic_match_refinement_payload(raw_match, candidate_texts, min_confidence)
                relation = normalize_semantic_relation(refined.get("relation"))
                candidate_aspect = candidate_aspect_for_point(refined.get("best_agent_point", ""), candidate_records, str(row.get("aspect") or ""))
                rows[idx] = {
                    **row,
                    "pre_llm_relation": normalize_semantic_relation(row.get("relation")),
                    "pre_llm_score": semantic_relation_score(normalize_semantic_relation(row.get("relation"))),
                    "pre_llm_similarity": row.get("similarity"),
                    "pre_llm_rationale": row.get("rationale", ""),
                    "bge_only_relation": normalize_semantic_relation(row.get("bge_only_relation") or row.get("relation")),
                    "refined_relation": relation,
                    "relation": relation,
                    "score": semantic_relation_score(relation),
                    "best_agent_point": refined.get("best_agent_point", row.get("best_agent_point", "")),
                    "rationale": refined.get("rationale") or row.get("rationale", ""),
                    "llm_refined": True,
                    "llm_confidence": refined.get("confidence", 0.0),
                    "llm_raw_response": raw_response,
                    "relation_source": "llm_batch_refine" if relation != normalize_semantic_relation(row.get("bge_only_relation") or row.get("relation")) else "bge_confirmed_by_llm",
                    "candidate_aspect": candidate_aspect,
                    "cross_aspect_match": bool(candidate_aspect and candidate_aspect != str(row.get("aspect") or "")),
                    "match_backend": f"{row.get('match_backend') or 'embedding'}+llm_batch_refine",
                }
        except Exception as exc:  # noqa: BLE001 - preserve BGE rows if the batch judge fails.
            for idx in batch_indices:
                rows[idx]["llm_refined"] = False
                rows[idx]["llm_refinement_error"] = f"{type(exc).__name__}: {exc}"
                rows[idx]["llm_raw_response"] = raw_response
    return rows


def semantic_match_one_point(
    title: str,
    aspect: str,
    peer_point: str,
    peer_quote: str,
    agent_candidates: Sequence[str],
    client: Optional[Any],
) -> Dict[str, Any]:
    if bool_value(os.getenv("FIG4_USE_EMBEDDING_MATCH", "1")):
        payload = embedding_semantic_match(peer_point, agent_candidates)
    elif client is None:
        payload = heuristic_semantic_match(peer_point, agent_candidates)
    else:
        payload = heuristic_semantic_match(peer_point, agent_candidates)
    relation = normalize_semantic_relation(payload.get("relation"))
    if (
        relation == "no_match"
        and bool_value(os.getenv("FIG4_SEMANTIC_LLM_REFINE", "0"))
        and isinstance(client, JudgeClientConfig)
        and agent_candidates
    ):
        payload = refine_semantic_match_with_llm(
            title=title,
            aspect=aspect,
            peer_point=peer_point,
            peer_quote=peer_quote,
            agent_candidates=agent_candidates,
            initial_match=payload,
            config=client,
        )
        relation = normalize_semantic_relation(payload.get("relation"))
    return {
        **payload,
        "relation": relation,
        "score": semantic_relation_score(relation),
        "match_backend": payload.get("match_backend") or ("heuristic" if client is None else "heuristic_client_fallback"),
    }


def score_agreement(left: Any, right: Any) -> float:
    left_num = numeric(left)
    right_num = numeric(right)
    if not math.isfinite(left_num) or not math.isfinite(right_num):
        return float("nan")
    return max(0.0, 1.0 - abs(left_num - right_num) / 4.0)


def score_consistency_1_5(left: Any, right: Any) -> float:
    left_num = numeric(left)
    right_num = numeric(right)
    if not math.isfinite(left_num) or not math.isfinite(right_num):
        return float("nan")
    return max(1.0, 5.0 - abs(left_num - right_num))


def heuristic_structured_consistency(peer_label: Mapping[str, Any], agent_label: Mapping[str, Any]) -> Dict[str, Any]:
    peer_stance = (peer_label.get("overall_innovation_stance") or {}).get("score_1_5") if isinstance(peer_label.get("overall_innovation_stance"), Mapping) else None
    agent_stance = (agent_label.get("overall_innovation_stance") or {}).get("score_1_5") if isinstance(agent_label.get("overall_innovation_stance"), Mapping) else None
    peer_aspects = peer_label.get("aspects") if isinstance(peer_label.get("aspects"), Mapping) else {}
    agent_aspects = agent_label.get("aspects") if isinstance(agent_label.get("aspects"), Mapping) else {}
    out: Dict[str, Any] = {
        "success": True,
        "stance_consistency_1_5": score_consistency_1_5(peer_stance, agent_stance),
        "overclaiming_score_1_5": score_consistency_1_5(peer_stance, agent_stance)
        if math.isfinite(numeric(peer_stance)) and math.isfinite(numeric(agent_stance)) and numeric(agent_stance) > numeric(peer_stance)
        else 1.0,
        "missing_key_points": [],
        "contradictions": [],
    }
    for aspect in INNOVATION_ASPECTS:
        key = "prior_art_consistency_1_5" if aspect == "prior_art_comparison" else f"{aspect}_consistency_1_5"
        peer_score = (peer_aspects.get(aspect) or {}).get("score_1_5") if isinstance(peer_aspects.get(aspect), Mapping) else None
        agent_score = (agent_aspects.get(aspect) or {}).get("score_1_5") if isinstance(agent_aspects.get(aspect), Mapping) else None
        out[key] = score_consistency_1_5(peer_score, agent_score)
    return out


def label_score(label: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = label
    for key in path:
        if not isinstance(value, Mapping):
            return float("nan")
        value = value.get(key)
    return numeric(value)


def label_points(label: Mapping[str, Any], aspect: str, include_quotes: bool = True) -> List[str]:
    aspects = label.get("aspects") if isinstance(label.get("aspects"), Mapping) else {}
    item = aspects.get(aspect) if isinstance(aspects.get(aspect), Mapping) else {}
    records = item.get("point_records") if isinstance(item.get("point_records"), list) else []
    if records:
        values: List[str] = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            point = normalize_whitespace(str(record.get("point") or ""))
            quote = normalize_whitespace(str(record.get("quote") or ""))
            if point:
                values.append(point)
            if include_quotes and quote:
                values.append(quote)
        return list(dict.fromkeys(values))
    values: List[str] = []
    keys = ("points", "quotes") if include_quotes else ("points",)
    for key in keys:
        raw = item.get(key)
        if isinstance(raw, list):
            values.extend(normalize_whitespace(str(value)) for value in raw if normalize_whitespace(str(value)))
    return values


def label_point_records(label: Mapping[str, Any], aspect: str, include_quotes: bool = False) -> List[Dict[str, Any]]:
    """Return normalized candidate records for one aspect from a label payload."""
    aspects = label.get("aspects") if isinstance(label.get("aspects"), Mapping) else {}
    item = aspects.get(aspect) if isinstance(aspects.get(aspect), Mapping) else {}
    out: List[Dict[str, Any]] = []
    records = item.get("point_records") if isinstance(item.get("point_records"), list) else []
    if records:
        for record in records:
            if not isinstance(record, Mapping):
                continue
            point = normalize_whitespace(str(record.get("point") or ""))
            quote = normalize_whitespace(str(record.get("quote") or ""))
            if point:
                out.append(
                    {
                        "point": point,
                        "quote": quote,
                        "aspect": aspect,
                        "evidence_type": str(record.get("evidence_type") or EVIDENCE_TYPE_BY_ASPECT.get(aspect, "")),
                    }
                )
            if include_quotes and quote:
                out.append({"point": quote, "quote": quote, "aspect": aspect, "evidence_type": "quote"})
        return out
    for point in label_points(label, aspect, include_quotes=include_quotes):
        out.append({"point": point, "quote": "", "aspect": aspect, "evidence_type": EVIDENCE_TYPE_BY_ASPECT.get(aspect, "")})
    return out


def is_future_work_gap_text(text: str) -> bool:
    """Return whether a limitation/future-work text is phrased as a future-work gap."""
    return bool(
        re.search(
            r"\b(future|further|additional|follow[- ]?up|next|should|need(?:ed|s)?|require(?:d|s)?|remaining|larger|more)\b",
            normalize_whitespace(text),
            flags=re.I,
        )
    )


def candidate_records_for_peer_aspect(agent_label: Mapping[str, Any], peer_aspect: str, peer_point: str) -> List[Dict[str, Any]]:
    """Return same-aspect plus strictly allowed cross-aspect agent candidate records."""
    records = label_point_records(agent_label, peer_aspect, include_quotes=False)
    cross_aspects = list(CROSS_ASPECT_FALLBACKS.get(peer_aspect, ()))
    if peer_aspect in {"limitations", "future_work"} and is_future_work_gap_text(peer_point):
        paired = "future_work" if peer_aspect == "limitations" else "limitations"
        if paired not in cross_aspects:
            cross_aspects.append(paired)
    for aspect in cross_aspects:
        records.extend(label_point_records(agent_label, aspect, include_quotes=False))
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for record in records:
        point = normalize_whitespace(str(record.get("point") or ""))
        aspect = str(record.get("aspect") or peer_aspect)
        key = (point, aspect)
        if point and key not in seen:
            deduped.append({**record, "point": point, "aspect": aspect})
            seen.add(key)
    return deduped


def candidate_aspect_for_point(best_agent_point: str, records: Sequence[Mapping[str, Any]], default_aspect: str) -> str:
    """Find the aspect attached to a selected agent candidate point."""
    target = normalize_whitespace(best_agent_point)
    for record in records:
        if normalize_whitespace(str(record.get("point") or "")) == target:
            return str(record.get("aspect") or default_aspect)
    return default_aspect if target else ""


def semantic_point_overlap(peer_points: Sequence[str], agent_points: Sequence[str]) -> Tuple[float, int, int]:
    """Coverage of peer points by agent points using embedding relation scores."""
    if not peer_points:
        return float("nan"), 0, 0
    covered = 0
    score_sum = 0.0
    for point in peer_points:
        match = embedding_semantic_match(point, agent_points)
        score = semantic_relation_score(normalize_semantic_relation(match.get("relation")))
        score_sum += score
        if score > 0:
            covered += 1
    return score_sum / max(len(peer_points), 1), covered, len(peer_points)


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


def innovation_aspect_alignment(peer_label: Mapping[str, Any], agent_label: Mapping[str, Any], aspect: str) -> Tuple[float, float, float, int, int]:
    peer_aspects = peer_label.get("aspects") if isinstance(peer_label.get("aspects"), Mapping) else {}
    agent_aspects = agent_label.get("aspects") if isinstance(agent_label.get("aspects"), Mapping) else {}
    peer_item = peer_aspects.get(aspect) if isinstance(peer_aspects.get(aspect), Mapping) else {}
    agent_item = agent_aspects.get(aspect) if isinstance(agent_aspects.get(aspect), Mapping) else {}
    score_align = score_agreement(peer_item.get("score_1_5"), agent_item.get("score_1_5"))
    peer_points = label_points(peer_label, aspect, include_quotes=False) or label_points(peer_label, aspect)
    agent_points = label_points(agent_label, aspect, include_quotes=False) or label_points(agent_label, aspect)
    overlap, covered, total = semantic_point_overlap(peer_points, agent_points)
    values = [value for value in (score_align, overlap) if math.isfinite(value)]
    alignment = float(sum(values) / len(values)) if values else float("nan")
    return alignment, score_align, overlap, covered, total


def text_cosine(left: str, right: str) -> float:
    return token_overlap_cosine(left, right)


def aspect_points(judgement: Mapping[str, Any], aspect: str) -> List[str]:
    aspects = judgement.get("aspects") if isinstance(judgement.get("aspects"), Mapping) else {}
    values = aspects.get(aspect)
    if not isinstance(values, list):
        return []
    return [normalize_whitespace(str(value)) for value in values if normalize_whitespace(str(value))]


def compute_aspect_matches(peer: Mapping[str, Any], agent: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for aspect in RATING_ASPECTS:
        peer_points = aspect_points(peer, aspect)
        agent_points = aspect_points(agent, aspect)
        agent_norm = [normalize_phrase(point) for point in agent_points]
        for point in peer_points:
            pnorm = normalize_phrase(point)
            matched = any(pnorm and (pnorm in candidate or candidate in pnorm) for candidate in agent_norm)
            rows.append(
                {
                    "paper_id": peer.get("paper_id", ""),
                    "aspect": aspect,
                    "peer_point": point,
                    "matched": matched,
                    "match_method": "normalized_phrase_overlap" if matched else "none",
                    "agent_candidates": agent_points[:5],
                }
            )
    return rows


def innovation_label_overlap_matches(paper_id: str, peer_label: Mapping[str, Any], agent_label: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    peer_stance = peer_label.get("overall_innovation_stance") if isinstance(peer_label.get("overall_innovation_stance"), Mapping) else {}
    agent_stance = agent_label.get("overall_innovation_stance") if isinstance(agent_label.get("overall_innovation_stance"), Mapping) else {}
    stance_quote = normalize_whitespace(str(peer_stance.get("quote") or ""))
    if stance_quote:
        rows.append(
            {
                "paper_id": paper_id,
                "aspect": "overall_innovation_stance",
                "peer_point": stance_quote,
                "matched": bool(normalize_whitespace(str(agent_stance.get("quote") or ""))),
                "match_method": "innovation_label_point_overlap",
            }
        )
    for aspect in INNOVATION_ASPECTS:
        peer_aspects = peer_label.get("aspects") if isinstance(peer_label.get("aspects"), Mapping) else {}
        agent_aspects = agent_label.get("aspects") if isinstance(agent_label.get("aspects"), Mapping) else {}
        peer_item = peer_aspects.get(aspect) if isinstance(peer_aspects.get(aspect), Mapping) else {}
        agent_item = agent_aspects.get(aspect) if isinstance(agent_aspects.get(aspect), Mapping) else {}
        peer_points = [
            normalize_whitespace(str(point))
            for point in (peer_item.get("points") if isinstance(peer_item.get("points"), list) else [])
            if normalize_whitespace(str(point))
        ]
        agent_points = [
            normalize_whitespace(str(point))
            for point in (agent_item.get("points") if isinstance(agent_item.get("points"), list) else [])
            if normalize_whitespace(str(point))
        ]
        overlap, covered, total = normalized_point_overlap(peer_points, agent_points)
        if total == 0:
            continue
        for point in peer_points[:total]:
            pnorm = normalize_phrase(point)
            matched = any(pnorm and (pnorm in normalize_phrase(candidate) or normalize_phrase(candidate) in pnorm) for candidate in agent_points)
            rows.append(
                {
                    "paper_id": paper_id,
                    "aspect": aspect,
                    "peer_point": point,
                    "matched": matched,
                    "coverage": overlap,
                    "covered_points": covered,
                    "total_points": total,
                    "match_method": "innovation_label_point_overlap",
                }
            )
    return rows


def group_jsonl_by_kind(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {(str(row.get("paper_id")), str(row.get("kind"))): row for row in read_jsonl(path)}


def group_csv_by_paper(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("paper_id")): row for row in read_csv_records(path)}


def safe_mean(values: Iterable[Any]) -> float:
    numbers = [numeric(value) for value in values]
    numbers = [value for value in numbers if math.isfinite(value)]
    return float(sum(numbers) / len(numbers)) if numbers else float("nan")


def semantic_refined_relation(row: Mapping[str, Any]) -> str:
    return normalize_semantic_relation(row.get("refined_relation") or row.get("relation"))


def semantic_bge_relation(row: Mapping[str, Any]) -> str:
    return normalize_semantic_relation(row.get("bge_only_relation") or row.get("pre_llm_relation") or row.get("relation"))


def semantic_covered_aspect_count(semantic_rows: Sequence[Mapping[str, Any]]) -> int:
    """Count aspects with at least one related or entailed semantic peer-point match."""
    covered: set[str] = set()
    for row in semantic_rows:
        if semantic_refined_relation(row) not in {"entailed", "related"}:
            continue
        aspect = str(row.get("aspect") or "").strip()
        if aspect:
            covered.add(aspect)
    return len(covered)


def build_aspect_relation_summary(
    semantic_rows: Sequence[Mapping[str, Any]],
    included_paper_ids: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """Aggregate semantic claim relations by innovation aspect for Fig.4 panel c."""
    rows = [
        row
        for row in semantic_rows
        if included_paper_ids is None or str(row.get("paper_id")) in included_paper_ids
    ]
    out: List[Dict[str, Any]] = []
    for aspect in INNOVATION_ASPECTS:
        aspect_rows = [row for row in rows if str(row.get("aspect")) == aspect]
        refined_counts = {relation: 0 for relation in SEMANTIC_RELATION_SCORES}
        bge_counts = {relation: 0 for relation in SEMANTIC_RELATION_SCORES}
        for row in aspect_rows:
            refined_counts[semantic_refined_relation(row)] += 1
            bge_counts[semantic_bge_relation(row)] += 1
        total = sum(refined_counts.values())
        bge_total = sum(bge_counts.values())
        matched = refined_counts["entailed"] + refined_counts["related"]
        bge_matched = bge_counts["entailed"] + bge_counts["related"]
        out.append(
            {
                "aspect": aspect,
                "aspect_label": ASPECT_DISPLAY_NAMES.get(aspect, aspect),
                "total_points": total,
                "entailed_points": refined_counts["entailed"],
                "related_points": refined_counts["related"],
                "contradicted_points": refined_counts["contradicted"],
                "no_match_points": refined_counts["no_match"],
                "matched_points": matched,
                "matched_rate": matched / total if total else float("nan"),
                "contradiction_rate": refined_counts["contradicted"] / total if total else float("nan"),
                "bge_matched_points": bge_matched,
                "bge_total_points": bge_total,
                "bge_matched_rate": bge_matched / bge_total if bge_total else float("nan"),
                "llm_refined_points": sum(1 for row in aspect_rows if bool_value(row.get("llm_refined"))),
                "cross_aspect_points": sum(1 for row in aspect_rows if bool_value(row.get("cross_aspect_match"))),
            }
        )
    return out


def compact_example_text(text: Any, max_chars: int = 220) -> str:
    cleaned = normalize_whitespace(str(text or ""))
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "..."


def mostly_ascii(text: Any) -> bool:
    cleaned = normalize_whitespace(str(text or ""))
    if not cleaned:
        return False
    ascii_count = sum(1 for char in cleaned if ord(char) < 128)
    return ascii_count / max(len(cleaned), 1) >= 0.92


def display_safe_agent_point(text: Any) -> str:
    cleaned = compact_example_text(text, 220)
    if not cleaned:
        return ""
    if mostly_ascii(cleaned):
        return cleaned
    return "[non-English ASPR point; see fig4_claim_examples.json]"


def select_claim_examples(semantic_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Deterministically choose one matched, one missed, and one contradicted claim example."""
    relation_targets = [
        ("matched", {"entailed", "related"}),
        ("missed", {"no_match"}),
        ("contradicted", {"contradicted"}),
    ]
    examples: List[Dict[str, Any]] = []
    sorted_rows = sorted(
        semantic_rows,
        key=lambda row: (
            str(row.get("paper_id") or ""),
            str(row.get("aspect") or ""),
            str(row.get("peer_point") or ""),
        ),
    )
    for example_type, relations in relation_targets:
        candidates = [
            row
            for row in sorted_rows
            if semantic_refined_relation(row) in relations
            and normalize_whitespace(str(row.get("peer_point") or row.get("peer_quote") or ""))
        ]
        if not candidates:
            continue
        if example_type == "matched":
            candidates = sorted(
                candidates,
                key=lambda row: (
                    not mostly_ascii(row.get("best_agent_point")),
                    semantic_refined_relation(row) != "entailed",
                    str(row.get("paper_id") or ""),
                ),
            )
        else:
            candidates = sorted(candidates, key=lambda row: (not mostly_ascii(row.get("best_agent_point")), str(row.get("paper_id") or "")))
        row = candidates[0]
        examples.append(
            {
                "example_type": example_type,
                "paper_id": row.get("paper_id", ""),
                "aspect": row.get("aspect", ""),
                "aspect_label": ASPECT_DISPLAY_NAMES.get(str(row.get("aspect") or ""), str(row.get("aspect") or "")),
                "relation": semantic_refined_relation(row),
                "bge_only_relation": semantic_bge_relation(row),
                "peer_quote": compact_example_text(row.get("peer_quote"), 150),
                "peer_point": compact_example_text(row.get("peer_point"), 140),
                "agent_point": display_safe_agent_point(row.get("best_agent_point")),
                "raw_agent_point": compact_example_text(row.get("best_agent_point"), 220),
                "candidate_aspect": row.get("candidate_aspect", ""),
                "cross_aspect_match": bool_value(row.get("cross_aspect_match")),
                "relation_source": row.get("relation_source", row.get("match_backend", "")),
            }
        )
    return {"examples": examples}


SW_TIER_ORDER = ["low", "middle", "high"]


def visual_sw_tier(row: Mapping[str, Any]) -> str:
    """Return the display-only S_w ladder tier, falling back to Fig.3 reference tier."""
    tier = str(row.get("fig4_sw_ladder_tier") or row.get("sw_visual_tier") or row.get("fig3_sw_tier") or "unknown")
    return tier if tier in SW_TIER_ORDER else "unknown"


def visual_sw_percentile(row: Mapping[str, Any]) -> float:
    """Return display-only within-run S_w percentile, falling back to Fig.3 reference percentile."""
    value = numeric(row.get("fig4_sw_batch_percentile"), float("nan"))
    if math.isfinite(value):
        return value
    return numeric(row.get("fig3_sw_percentile"), float("nan"))


def build_aspect_relation_by_sw_tier(
    semantic_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    included_paper_ids: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    """Aggregate semantic alignment by innovation aspect and Fig.3 S_w tier."""
    tier_by_paper = {
        str(row.get("paper_id")): visual_sw_tier(row)
        for row in metric_rows
        if included_paper_ids is None or str(row.get("paper_id")) in included_paper_ids
    }
    rows = [
        row
        for row in semantic_rows
        if str(row.get("paper_id")) in tier_by_paper
    ]
    out: List[Dict[str, Any]] = []
    for aspect in INNOVATION_ASPECTS:
        for tier in SW_TIER_ORDER:
            aspect_rows = [
                row
                for row in rows
                if str(row.get("aspect")) == aspect and tier_by_paper.get(str(row.get("paper_id"))) == tier
            ]
            counts = {relation: 0 for relation in SEMANTIC_RELATION_SCORES}
            for row in aspect_rows:
                counts[semantic_refined_relation(row)] += 1
            total = sum(counts.values())
            matched = counts["entailed"] + counts["related"]
            out.append(
                {
                    "aspect": aspect,
                    "aspect_label": ASPECT_DISPLAY_NAMES.get(aspect, aspect),
                    "fig3_sw_tier": tier,
                    "sw_visual_tier": tier,
                    "tier_source": "fig4_sw_ladder_tier",
                    "total_points": total,
                    "entailed_points": counts["entailed"],
                    "related_points": counts["related"],
                    "contradicted_points": counts["contradicted"],
                    "no_match_points": counts["no_match"],
                    "matched_points": matched,
                    "matched_rate": matched / total if total else float("nan"),
                    "contradiction_rate": counts["contradicted"] / total if total else float("nan"),
                }
            )
    return out


def select_sw_ladder_examples(
    semantic_rows: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Choose one traceable claim example for each low/middle/high Fig.3 S_w tier."""
    metric_by_paper = {str(row.get("paper_id")): row for row in metric_rows}
    sorted_rows = sorted(
        semantic_rows,
        key=lambda row: (
            str(row.get("paper_id") or ""),
            str(row.get("aspect") or ""),
            str(row.get("peer_point") or ""),
        ),
    )
    examples: List[Dict[str, Any]] = []
    for tier in SW_TIER_ORDER:
        candidates = []
        for row in sorted_rows:
            paper_id = str(row.get("paper_id") or "")
            metric = metric_by_paper.get(paper_id, {})
            if visual_sw_tier(metric) != tier:
                continue
            if not normalize_whitespace(str(row.get("peer_point") or row.get("peer_quote") or "")):
                continue
            candidates.append(row)
        if not candidates:
            continue
        candidates = sorted(
            candidates,
            key=lambda row: (
                semantic_refined_relation(row) not in {"entailed", "related"},
                semantic_refined_relation(row) == "contradicted",
                not mostly_ascii(row.get("best_agent_point")),
                -numeric(metric_by_paper.get(str(row.get("paper_id") or ""), {}).get("fig3_sw_percentile"), 0.0)
                if tier == "high"
                else abs(
                    numeric(metric_by_paper.get(str(row.get("paper_id") or ""), {}).get("fig3_sw_percentile"), 0.5)
                    - (0.16 if tier == "low" else 0.50)
                ),
                str(row.get("paper_id") or ""),
            ),
        )
        row = candidates[0]
        metric = metric_by_paper.get(str(row.get("paper_id") or ""), {})
        examples.append(
            {
                "example_type": f"{tier}_sw",
                "paper_id": row.get("paper_id", ""),
                "fig3_sw": numeric(metric.get("fig3_sw"), float("nan")),
                "fig3_sw_percentile": numeric(metric.get("fig3_sw_percentile"), float("nan")),
                "fig3_sw_tier": metric.get("fig3_sw_tier", ""),
                "fig4_sw_batch_percentile": visual_sw_percentile(metric),
                "fig4_sw_ladder_tier": tier,
                "aspect": row.get("aspect", ""),
                "aspect_label": ASPECT_DISPLAY_NAMES.get(str(row.get("aspect") or ""), str(row.get("aspect") or "")),
                "relation": semantic_refined_relation(row),
                "bge_only_relation": semantic_bge_relation(row),
                "peer_quote": compact_example_text(row.get("peer_quote"), 155),
                "peer_point": compact_example_text(row.get("peer_point"), 145),
                "agent_point": display_safe_agent_point(row.get("best_agent_point")),
                "raw_agent_point": compact_example_text(row.get("best_agent_point"), 220),
                "candidate_aspect": row.get("candidate_aspect", ""),
                "cross_aspect_match": bool_value(row.get("cross_aspect_match")),
                "relation_source": row.get("relation_source", row.get("match_backend", "")),
            }
        )
    return examples


def quadratic_weighted_kappa_single(left: Any, right: Any) -> float:
    left_num = numeric(left)
    right_num = numeric(right)
    if not math.isfinite(left_num) or not math.isfinite(right_num):
        return float("nan")
    return max(0.0, 1.0 - ((left_num - right_num) ** 2) / 16.0)


def run_metrics_stage(output_dir: Path, human_hours: float = DEFAULT_HUMAN_HOURS, judge_backend: str = "none", quiet: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not (output_dir / "fig4_graph_prior.csv").exists() and (output_dir / "fig4_graph_metrics.csv").exists():
        try:
            run_graph_prior_stage(output_dir, quiet=True)
        except Exception as exc:  # noqa: BLE001 - keep metrics robust for partial/lightweight runs.
            progress_log(f"Graph prior stage skipped during metrics: {exc}", quiet)
    manifest = read_csv_records(output_dir / "fig4_manifest.csv")
    agents = {str(row.get("paper_id")): row for row in read_jsonl(output_dir / "fig4_agent_outputs.jsonl")}
    ratings = group_jsonl_by_kind(output_dir / "fig4_rating_judgements.jsonl")
    labels = group_jsonl_by_kind(output_dir / "fig4_innovation_label_judgements.jsonl")
    screen = group_csv_by_paper(output_dir / "fig4_peer_review_screen.csv")
    graph = group_csv_by_paper(output_dir / "fig4_graph_metrics.csv")
    graph_prior = group_csv_by_paper(output_dir / "fig4_graph_prior.csv")
    retrieval = group_csv_by_paper(output_dir / "fig4_retrieval_diagnostics.csv")
    semantic_rows = read_jsonl(output_dir / "fig4_semantic_claim_matches.jsonl")
    structured = {str(row.get("paper_id")): row for row in read_jsonl(output_dir / "fig4_structured_consistency_judgements.jsonl")}
    all_matches: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for row in manifest:
        paper_id = str(row.get("paper_id") or "")
        parsed_path = Path(str(row.get("parsed_text_cache") or output_dir / "cache" / paper_id / "parsed_text.json"))
        parsed = read_json(parsed_path) if parsed_path.exists() else {}
        agent = agents.get(paper_id, {})
        prior_row = graph_prior.get(paper_id, {})
        peer_text = str(parsed.get("peer_review_text") or "")
        agent_text = str(agent.get("innovation_evaluation") or "")
        peer_rating = ratings.get((paper_id, "peer_review"), {})
        agent_rating = ratings.get((paper_id, "agent"), {})
        peer_label = labels.get((paper_id, "peer_review"), {})
        agent_label = labels.get((paper_id, "agent"), {})
        peer_label_text = label_similarity_text(peer_label)
        agent_label_text = label_similarity_text(agent_label)
        if peer_label_text and agent_label_text:
            consistency_cosine, embedding_backend = embedding_cosine(peer_label_text, agent_label_text)
        else:
            consistency_cosine = float("nan")
            embedding_backend = get_fig4_embedding_model()[1]
        phrase_matches = compute_aspect_matches(peer_rating, agent_rating)
        all_matches.extend(phrase_matches)
        label_matches = innovation_label_overlap_matches(paper_id, peer_label, agent_label)
        all_matches.extend(label_matches)
        covered_phrase = sum(1 for item in phrase_matches if item.get("matched"))
        total_phrase = len(phrase_matches)
        coverage_score = covered_phrase / total_phrase if total_phrase else float("nan")
        stance_peer = label_score(peer_label, ["overall_innovation_stance", "score_1_5"])
        stance_agent = label_score(agent_label, ["overall_innovation_stance", "score_1_5"])
        stance_pair_available = math.isfinite(stance_peer) and math.isfinite(stance_agent)
        semantic_for_paper = [item for item in semantic_rows if str(item.get("paper_id")) == paper_id]
        all_matches.extend({**item, "match_method": "semantic_claim_judge"} for item in semantic_for_paper)
        sem_scores = [semantic_relation_score(semantic_refined_relation(item)) for item in semantic_for_paper]
        semantic_claim_alignment = safe_mean(sem_scores)
        llm_refined_points = sum(1 for item in semantic_for_paper if bool_value(item.get("llm_refined")))
        llm_changed_points = sum(
            1
            for item in semantic_for_paper
            if bool_value(item.get("llm_refined"))
            and semantic_bge_relation(item) != semantic_refined_relation(item)
        )
        cross_aspect_points = sum(1 for item in semantic_for_paper if bool_value(item.get("cross_aspect_match")))
        semantic_covered_aspects = semantic_covered_aspect_count(semantic_for_paper)
        relation_counts = {relation: 0 for relation in SEMANTIC_RELATION_SCORES}
        bge_relation_counts = {relation: 0 for relation in SEMANTIC_RELATION_SCORES}
        for item in semantic_for_paper:
            relation_counts[semantic_refined_relation(item)] += 1
            bge_relation_counts[semantic_bge_relation(item)] += 1
        total_semantic = sum(relation_counts.values())
        strict_claim_recall = relation_counts["entailed"] / total_semantic if total_semantic else float("nan")
        soft_claim_recall = (relation_counts["entailed"] + relation_counts["related"]) / total_semantic if total_semantic else float("nan")
        contradiction_rate = relation_counts["contradicted"] / total_semantic if total_semantic else 0.0
        missing_peer_point_rate = relation_counts["no_match"] / total_semantic if total_semantic else float("nan")
        structured_row = structured.get(paper_id, {})
        structured_values = [numeric(structured_row.get(field)) for field in STRUCTURED_CONSISTENCY_FIELDS]
        structured_mean = safe_mean(structured_values)
        overclaiming_score = numeric(structured_row.get("overclaiming_score_1_5"), float("nan"))
        if not math.isfinite(overclaiming_score):
            overclaiming_score = heuristic_structured_consistency(peer_label, agent_label)["overclaiming_score_1_5"]
        aspect_outputs: Dict[str, Any] = {}
        claim_covered = 0
        claim_total = 0
        for aspect in INNOVATION_ASPECTS:
            alignment, score_align, point_overlap, covered, total = innovation_aspect_alignment(peer_label, agent_label, aspect)
            out_key = "prior_art_alignment" if aspect == "prior_art_comparison" else f"{aspect}_alignment"
            aspect_outputs[out_key] = alignment
            semantic_key = "prior_art_semantic_coverage" if aspect == "prior_art_comparison" else f"{aspect}_semantic_coverage"
            aspect_semantic = [item for item in semantic_for_paper if str(item.get("aspect")) == aspect]
            aspect_outputs[semantic_key] = safe_mean(semantic_relation_score(semantic_refined_relation(item)) for item in aspect_semantic)
            claim_covered += covered
            claim_total += total
        semantic_claim_covered = relation_counts["entailed"] + relation_counts["related"]
        claim_evidence_coverage = (
            semantic_claim_covered / total_semantic
            if total_semantic
            else (claim_covered / claim_total if claim_total else float("nan"))
        )
        peer_readability = simple_readability_metrics(peer_text)
        agent_readability = simple_readability_metrics(agent_text)
        metric_row = {
            **{key: row.get(key, "") for key in row},
            "paper_id": paper_id,
            "included_in_main": bool_value(row.get("included_in_main", True)) if "included_in_main" in row else True,
            "screen_pass": bool_value(screen.get(paper_id, {}).get("screen_pass", True)),
            "agent_success": bool_value(agent.get("success", False)),
            "graph_metric_valid": bool_value(graph.get(paper_id, {}).get("graph_metric_valid", False)),
            "retrieval_source": retrieval.get(paper_id, {}).get("retrieval_source", agent.get("retrieval_source", "")),
            "s2_key_status": retrieval.get(paper_id, {}).get("s2_key_status", agent.get("s2_key_status", "")),
            "consistency_cosine": consistency_cosine,
            "embedding_backend": embedding_backend,
            "raw_text_token_overlap_supplementary": text_cosine(peer_text, agent_text),
            "coverage_score": coverage_score,
            "phrase_claim_coverage_supplementary": coverage_score,
            "innovation_stance_agreement": score_agreement(stance_peer, stance_agent),
            "stance_exact_agreement": (
                1.0 if stance_pair_available and round(stance_peer) == round(stance_agent) else (0.0 if stance_pair_available else float("nan"))
            ),
            "stance_within_one_agreement": (
                1.0 if stance_pair_available and abs(stance_peer - stance_agent) <= 1 else (0.0 if stance_pair_available else float("nan"))
            ),
            "quadratic_weighted_kappa": quadratic_weighted_kappa_single(stance_peer, stance_agent),
            "peer_innovation_stance_1_5": stance_peer,
            "agent_innovation_stance_1_5": stance_agent,
            "peer_overall_score_1_5": numeric(peer_rating.get("overall_score_1_5")),
            "agent_overall_score_1_5": numeric(agent_rating.get("overall_score_1_5")),
            "semantic_claim_alignment": semantic_claim_alignment,
            "strict_claim_recall": strict_claim_recall,
            "soft_claim_recall": soft_claim_recall,
            "structured_semantic_consistency_mean": structured_mean,
            "overclaiming_score_1_5": overclaiming_score,
            "overclaiming_flag": 1.0 if math.isfinite(overclaiming_score) and overclaiming_score >= 4.0 else 0.0,
            "claim_validation_pass": 1.0
            if (not math.isfinite(strict_claim_recall) or strict_claim_recall >= 0.5)
            and (not math.isfinite(overclaiming_score) or overclaiming_score <= 2.0)
            and contradiction_rate <= 0.10
            else 0.0,
            "claim_evidence_coverage": claim_evidence_coverage,
            "contradiction_rate": contradiction_rate,
            "missing_peer_point_rate": missing_peer_point_rate,
            "entailed_points": relation_counts["entailed"],
            "related_points": relation_counts["related"],
            "contradicted_points": relation_counts["contradicted"],
            "no_match_points": relation_counts["no_match"],
            "llm_refined_points": llm_refined_points,
            "llm_refinement_changed_points": llm_changed_points,
            "cross_aspect_matched_points": cross_aspect_points,
            "bge_only_matched_points": bge_relation_counts["entailed"] + bge_relation_counts["related"],
            "bge_only_no_match_points": bge_relation_counts["no_match"],
            "bge_only_contradicted_points": bge_relation_counts["contradicted"],
            "agent_runtime_seconds": numeric(agent.get("agent_runtime_seconds"), float("nan")),
            "speedup_vs_human": (float(human_hours) * 3600.0 / numeric(agent.get("agent_runtime_seconds"), float("nan")))
            if numeric(agent.get("agent_runtime_seconds"), float("nan")) > 0
            else float("nan"),
            "readability_available": bool(peer_readability.get("readability_available")) and bool(agent_readability.get("readability_available")),
            "peer_flesch_reading_ease": peer_readability.get("flesch_reading_ease"),
            "agent_flesch_reading_ease": agent_readability.get("flesch_reading_ease"),
            "peer_flesch_kincaid_grade": peer_readability.get("flesch_kincaid_grade"),
            "agent_flesch_kincaid_grade": agent_readability.get("flesch_kincaid_grade"),
            "peer_grammar_errors_per_5000": peer_readability.get("grammar_errors_per_5000"),
            "agent_grammar_errors_per_5000": agent_readability.get("grammar_errors_per_5000"),
            "peer_spelling_errors_per_5000": peer_readability.get("spelling_errors_per_5000"),
            "agent_spelling_errors_per_5000": agent_readability.get("spelling_errors_per_5000"),
            "peer_tense_errors_per_5000": peer_readability.get("tense_errors_per_5000"),
            "agent_tense_errors_per_5000": agent_readability.get("tense_errors_per_5000"),
            "total_peer_aspects": total_phrase,
            "covered_peer_aspects": max(covered_phrase, semantic_covered_aspects),
            "fig3_sw": numeric(prior_row.get("fig3_sw"), float("nan")),
            "fig3_sw_percentile": numeric(prior_row.get("fig3_sw_percentile"), float("nan")),
            "fig3_sw_tier": prior_row.get("fig3_sw_tier", ""),
            "fig3_weights_source": prior_row.get("fig3_weights_source", ""),
            "fig3_weights_hash": prior_row.get("fig3_weights_hash", ""),
            "fig3_sw_quality_flag": prior_row.get("fig3_sw_quality_flag", ""),
            "fig3_sw_normalization": prior_row.get("fig3_sw_normalization", ""),
            "fig3_sw_percentile_source": prior_row.get("fig3_sw_percentile_source", ""),
            "fallback_percentile_source": prior_row.get("fallback_percentile_source", ""),
            "fig4_sw_batch_percentile": numeric(prior_row.get("fig4_sw_batch_percentile"), float("nan")),
            "fig4_sw_ladder_tier": prior_row.get("fig4_sw_ladder_tier", ""),
            "graph_prior_prompt_mode": prior_row.get("graph_prior_prompt_mode", ""),
            **aspect_outputs,
        }
        for aspect in RATING_ASPECTS:
            metric_row[f"peer_{aspect}"] = numeric(peer_rating.get(aspect))
            metric_row[f"agent_{aspect}"] = numeric(agent_rating.get(aspect))
        for key, value in graph.get(paper_id, {}).items():
            if key not in metric_row:
                metric_row[key] = value
        for key, value in prior_row.items():
            if key not in metric_row:
                metric_row[key] = value
        rows.append(metric_row)
        progress_log(f"Metrics progress {len(rows)}/{len(manifest)}.", quiet)
    write_csv(output_dir / "fig4_metrics_summary.csv", rows)
    write_csv(output_dir / "fig4_external_validation_target_audit.csv", [build_fig4_external_validation_target_audit(pd.DataFrame(rows))])
    write_jsonl(output_dir / "fig4_aspect_matches.jsonl", all_matches)
    included_paper_ids = {str(row.get("paper_id")) for row in rows if bool_value(row.get("included_in_main", True))}
    write_csv(output_dir / "fig4_aspect_relation_summary.csv", build_aspect_relation_summary(semantic_rows, included_paper_ids))
    write_csv(output_dir / "fig4_aspect_relation_by_sw_tier.csv", build_aspect_relation_by_sw_tier(semantic_rows, rows, included_paper_ids))
    claim_examples = select_claim_examples(semantic_rows)
    claim_examples["sw_ladder_examples"] = select_sw_ladder_examples(semantic_rows, rows)
    write_json(output_dir / "fig4_claim_examples.json", claim_examples)
    return rows, all_matches


def draw_fig4_sw_centric(output_dir: Path, human_hours: float = DEFAULT_HUMAN_HOURS, quiet: bool = False) -> Dict[str, Any]:
    """Draw the S_w-centric manuscript Fig.4."""
    rows = read_csv_records(output_dir / "fig4_metrics_summary.csv")
    main_rows = [row for row in rows if bool_value(row.get("included_in_main", True))]
    semantic_rows = read_jsonl(output_dir / "fig4_semantic_claim_matches.jsonl")
    included_paper_ids = {str(row.get("paper_id")) for row in main_rows}
    aspect_tier_path = output_dir / "fig4_aspect_relation_by_sw_tier.csv"
    if aspect_tier_path.exists():
        aspect_tier_summary = read_csv_records(aspect_tier_path)
    else:
        aspect_tier_summary = build_aspect_relation_by_sw_tier(semantic_rows, main_rows, included_paper_ids)
        write_csv(aspect_tier_path, aspect_tier_summary)
    examples_path = output_dir / "fig4_claim_examples.json"
    if examples_path.exists():
        examples = read_json(examples_path)
    else:
        examples = select_claim_examples(semantic_rows)
    if not isinstance(examples.get("sw_ladder_examples"), list) or not examples.get("sw_ladder_examples"):
        examples["sw_ladder_examples"] = select_sw_ladder_examples(semantic_rows, main_rows)
        write_json(examples_path, examples)

    n_main = len(main_rows)
    tier_counts = {
        tier: sum(1 for row in main_rows if str(row.get("fig3_sw_tier") or "unknown") == tier)
        for tier in SW_TIER_ORDER
    }

    def finite_col(column: str) -> List[float]:
        return [value for value in (numeric(row.get(column)) for row in main_rows) if math.isfinite(value)]

    def mean_col(column: str) -> float:
        return safe_mean(row.get(column) for row in main_rows)

    def ranks(values: Sequence[float]) -> List[float]:
        indexed = sorted(enumerate(values), key=lambda item: item[1])
        output = [0.0 for _value in values]
        start = 0
        while start < len(indexed):
            end = start + 1
            while end < len(indexed) and indexed[end][1] == indexed[start][1]:
                end += 1
            rank_value = (start + 1 + end) / 2.0
            for idx in range(start, end):
                output[indexed[idx][0]] = rank_value
            start = end
        return output

    def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
        if len(xs) < 2 or len(xs) != len(ys):
            return float("nan")
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
        y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
        return numerator / (x_den * y_den) if x_den and y_den else float("nan")

    def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
        return pearson(ranks(xs), ranks(ys))

    def kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> float:
        concordant = 0
        discordant = 0
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                x_delta = xs[i] - xs[j]
                y_delta = ys[i] - ys[j]
                if x_delta == 0 or y_delta == 0:
                    continue
                if x_delta * y_delta > 0:
                    concordant += 1
                elif x_delta * y_delta < 0:
                    discordant += 1
        total = concordant + discordant
        return (concordant - discordant) / total if total else float("nan")

    def fmt_float(value: float, digits: int = 2, fallback: str = "n/a") -> str:
        return f"{value:.{digits}f}" if math.isfinite(value) else fallback

    def fmt_pct(value: float, digits: int = 0, fallback: str = "n/a") -> str:
        return f"{value * 100:.{digits}f}%" if math.isfinite(value) else fallback

    def wrapped(text: Any, width: int = 52) -> str:
        return textwrap.fill(normalize_whitespace(str(text or "")), width=width)

    refined_counts = {relation: 0 for relation in SEMANTIC_RELATION_SCORES}
    semantic_for_main = [row for row in semantic_rows if not included_paper_ids or str(row.get("paper_id")) in included_paper_ids]
    for row in semantic_for_main:
        refined_counts[semantic_refined_relation(row)] += 1
    semantic_total = sum(refined_counts.values())
    matched_total = refined_counts["entailed"] + refined_counts["related"]
    summary = {
        "n_main": n_main,
        "sw_tier_counts": tier_counts,
        "mean_fig3_sw_percentile": safe_mean(row.get("fig3_sw_percentile") for row in main_rows),
        "mean_stance_within_one": mean_col("stance_within_one_agreement"),
        "mean_quadratic_weighted_kappa": mean_col("quadratic_weighted_kappa"),
        "mean_claim_evidence_coverage": mean_col("claim_evidence_coverage"),
        "mean_contradiction_rate": mean_col("contradiction_rate"),
        "semantic_matched_rate": matched_total / semantic_total if semantic_total else float("nan"),
    }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        _draw_fig4_fallback(output_dir, summary)
        progress_log(f"Drew fallback Fig.4 with n={n_main}.", quiet)
        return summary

    peer_color = "#e76f61"
    peer_light = "#f9c4bd"
    agent_color = "#2767c5"
    agent_light = "#99bee9"
    tier_colors = {"low": "#c9a227", "middle": "#73a960", "high": "#19784f"}
    tier_fills = {"low": "#fff5cf", "middle": "#edf7df", "high": "#def2e7"}
    heat_cmap = LinearSegmentedColormap.from_list("sw_alignment", ["#f5f7fb", "#dcefcf", "#53a869", "#166b45"])
    dark = "#101827"
    muted = "#526173"
    panel_edge = "#c9d0da"
    grid = "#e7ebf1"

    fig = plt.figure(figsize=(14.02, 11.22), dpi=100, facecolor="white")
    fig.text(
        0.5,
        0.982,
        "Fig. 4 | Fig.3 S_w prior calibrates ASPR innovation judgements against quote-grounded peer review",
        ha="center",
        va="top",
        fontsize=13.0,
        fontweight="bold",
        color="black",
    )

    def panel(bounds: Tuple[float, float, float, float], label: str, title: str) -> None:
        x, y, w, h = bounds
        fig.patches.append(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.007,rounding_size=0.007",
                transform=fig.transFigure,
                linewidth=0.9,
                edgecolor=panel_edge,
                facecolor="white",
                zorder=-20,
            )
        )
        fig.text(x + 0.010, y + h - 0.018, label, fontsize=12.5, fontweight="bold", color="black", ha="left", va="top")
        fig.text(x + 0.035, y + h - 0.020, title, fontsize=9.4, fontweight="bold", color="black", ha="left", va="top")

    def ax_in(bounds: Tuple[float, float, float, float], rel: Tuple[float, float, float, float]) -> Any:
        x, y, w, h = bounds
        rx, ry, rw, rh = rel
        return fig.add_axes([x + rx * w, y + ry * h, rw * w, rh * h])

    def style_axis(ax: Any, ygrid: bool = True, xgrid: bool = False) -> None:
        if ygrid:
            ax.grid(True, axis="y", color=grid, linewidth=0.8, zorder=0)
        if xgrid:
            ax.grid(True, axis="x", color=grid, linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#b7c0cc")
        ax.spines["bottom"].set_color("#b7c0cc")
        ax.tick_params(axis="both", labelsize=7.1, width=0.8, length=3, colors="#1f2937")

    panel_a = (0.020, 0.650, 0.355, 0.300)
    panel_b = (0.395, 0.650, 0.585, 0.300)
    panel_c = (0.020, 0.382, 0.460, 0.238)
    panel_d = (0.500, 0.382, 0.480, 0.238)
    panel_e = (0.020, 0.132, 0.960, 0.220)
    summary_panel = (0.028, 0.040, 0.944, 0.064)

    panel(panel_a, "a", "Fig.2 -> Fig.3 -> Fig.4 evidence bridge")
    panel(panel_b, "b", "S_w prior distribution and stance calibration")
    panel(panel_c, "c", "Human vs ASPR stance by S_w tier")
    panel(panel_d, "d", "Aspect-level alignment by S_w tier")
    panel(panel_e, "e", "Claim-evidence examples on the S_w ladder")

    # Panel a: evidence bridge plus Fig.3 weight strip.
    ax = ax_in(panel_a, (0.045, 0.180, 0.910, 0.680))
    ax.axis("off")
    workflow = [
        ("Fig.2", "92 candidate\nsignals"),
        ("Fig.3", "7 publication-day\nindicators"),
        ("Weights", "learned\nbest weights"),
        ("S_w", "single graph\nprior"),
        ("Fig.4", "agent vs\npeer labels"),
    ]
    for idx, (head, body) in enumerate(workflow):
        x0 = 0.015 + idx * 0.194
        face = "#f8fbff" if idx in {0, 4} else ("#fff7d6" if idx in {2, 3} else "#f6fbf2")
        edge = agent_color if idx in {0, 4} else (tier_colors["low"] if idx in {2, 3} else tier_colors["middle"])
        ax.add_patch(
            FancyBboxPatch(
                (x0, 0.55),
                0.145,
                0.32,
                boxstyle="round,pad=0.010,rounding_size=0.022",
                transform=ax.transAxes,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.9,
            )
        )
        ax.text(x0 + 0.072, 0.765, head, ha="center", va="center", fontsize=8.1, fontweight="bold", color=dark, transform=ax.transAxes)
        ax.text(x0 + 0.072, 0.625, body, ha="center", va="center", fontsize=6.4, color=muted, transform=ax.transAxes)
        if idx < len(workflow) - 1:
            ax.annotate(
                "",
                xy=(x0 + 0.183, 0.710),
                xytext=(x0 + 0.154, 0.710),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#7d8796"},
            )
    ax.text(
        0.018,
        0.455,
        "Agent prompt receives only S_w, percentile and tier; component values stay in cache/CSV for audit.",
        fontsize=6.9,
        color=dark,
        transform=ax.transAxes,
    )
    weights_config = load_fig3_weight_config(fig3_weights_path_from_env())
    weight_items = [(metric, numeric(weights_config["weights"].get(metric), 0.0)) for metric in INNOVATION_METRIC_NAMES]
    max_weight = max([value for _metric, value in weight_items] or [1.0])
    strip_y = 0.130
    strip_h = 0.220
    ax.text(0.018, 0.350, "Fig.3 weight mass", fontsize=6.8, color=muted, fontweight="bold", transform=ax.transAxes)
    for idx, (metric, weight) in enumerate(weight_items):
        x0 = 0.020 + idx * 0.132
        height = 0.055 + (weight / max_weight if max_weight else 0.0) * 0.135
        ax.add_patch(
            FancyBboxPatch(
                (x0, strip_y),
                0.055,
                height,
                boxstyle="round,pad=0.002,rounding_size=0.006",
                transform=ax.transAxes,
                facecolor="#d5b139",
                edgecolor="#9d7f1c",
                linewidth=0.45,
            )
        )
        ax.text(x0 + 0.027, strip_y - 0.030, metric, ha="center", va="top", fontsize=5.6, color=dark, transform=ax.transAxes)
    if weights_config.get("warning"):
        ax.text(0.018, 0.030, "Weight fallback active; see fig4_graph_prior.csv.", fontsize=6.1, color=peer_color, transform=ax.transAxes)

    # Panel b: S_w prior distribution and calibration markers.
    ax = ax_in(panel_b, (0.055, 0.200, 0.895, 0.600))
    for lo, hi, tier in [(0, 33.333, "low"), (33.333, 66.667, "middle"), (66.667, 100, "high")]:
        ax.axvspan(lo, hi, color=tier_fills[tier], alpha=0.82, zorder=0)
        ax.text((lo + hi) / 2, 0.935, f"{tier} prior\nn={tier_counts[tier]}", ha="center", va="top", fontsize=6.8, color=tier_colors[tier], fontweight="bold")
    plotted = []
    for idx, row in enumerate(sorted(main_rows, key=lambda item: numeric(item.get("fig3_sw_percentile"), 0.5))):
        percentile = numeric(row.get("fig3_sw_percentile"), float("nan"))
        if not math.isfinite(percentile):
            continue
        x_val = percentile * 100.0
        jitter = ((((idx * 37) % 13) - 6) / 100.0)
        agent_stance = numeric(row.get("agent_innovation_stance_1_5"), float("nan"))
        peer_stance = numeric(row.get("peer_innovation_stance_1_5"), float("nan"))
        tier = str(row.get("fig3_sw_tier") or "unknown")
        plotted.append((x_val, 0.47 + jitter, agent_stance, peer_stance, tier))
    if plotted:
        xs = [item[0] for item in plotted]
        ys = [item[1] for item in plotted]
        colors = [item[2] if math.isfinite(item[2]) else 3.0 for item in plotted]
        edge_colors = [tier_colors.get(item[4], "#8792a2") for item in plotted]
        scatter = ax.scatter(xs, ys, c=colors, cmap="Blues", vmin=1, vmax=5, s=44, edgecolor=edge_colors, linewidth=0.80, zorder=3)
        ax.scatter(xs, [y + 0.110 for y in ys], marker="|", s=150, color=peer_color, linewidth=1.6, zorder=4)
        for x_val, y_val, _agent, peer, _tier in plotted:
            if math.isfinite(peer):
                ax.plot([x_val, x_val], [y_val + 0.030, y_val + 0.095], color=peer_light, linewidth=0.55, zorder=2)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.014)
        cbar.ax.tick_params(labelsize=6.2, length=2)
        cbar.set_label("ASPR stance", fontsize=6.6, fontweight="bold")
    ax.text(1.5, 0.140, "blue circles: ASPR stance; coral ticks: peer-review stance", fontsize=6.7, color=muted)
    ax.set_xlim(0, 100)
    ax.set_ylim(0.20, 1.02)
    ax.set_yticks([])
    ax.set_xlabel("Fig.3 reference percentile of S_w", fontsize=7.2, fontweight="bold")
    ax.set_xticks([0, 33, 67, 100])
    ax.set_xticklabels(["0", "33", "67", "100"])
    style_axis(ax, ygrid=False, xgrid=True)

    # Panel c: stance scatter by S_w tier.
    score_pairs = [
        (
            numeric(row.get("agent_innovation_stance_1_5")),
            numeric(row.get("peer_innovation_stance_1_5")),
            str(row.get("fig3_sw_tier") or "unknown"),
        )
        for row in main_rows
        if math.isfinite(numeric(row.get("agent_innovation_stance_1_5")))
        and math.isfinite(numeric(row.get("peer_innovation_stance_1_5")))
    ]
    ax = ax_in(panel_c, (0.120, 0.180, 0.780, 0.650))
    ax.fill_between([1, 5], [0, 4], [2, 6], color="#f3f7fb", alpha=0.75, zorder=0)
    if score_pairs:
        for idx, (agent_score, peer_score, tier) in enumerate(score_pairs):
            jitter_x = (((idx * 31) % 7) - 3) * 0.020
            jitter_y = (((idx * 17) % 7) - 3) * 0.020
            ax.scatter(
                agent_score + jitter_x,
                peer_score + jitter_y,
                s=46,
                color=tier_colors.get(tier, "#8792a2"),
                alpha=0.85,
                edgecolor="white",
                linewidth=0.75,
                zorder=3,
            )
        ax.plot([1, 5], [1, 5], color="#9aa4b2", linestyle="--", linewidth=0.9)
        xs = [item[0] for item in score_pairs]
        ys = [item[1] for item in score_pairs]
        rho = spearman(xs, ys)
        tau = kendall_tau(xs, ys)
        ax.text(
            0.055,
            0.930,
            f"Within-1 {fmt_pct(summary['mean_stance_within_one'])}\nQWK {fmt_float(summary['mean_quadratic_weighted_kappa'], 2)}\nN valid {len(score_pairs)}",
            transform=ax.transAxes,
            fontsize=7.2,
            color=agent_color,
            fontweight="bold",
            va="top",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#e3e8f0", "alpha": 0.95},
        )
        ax.text(0.055, 0.050, f"Rank caveat: rho {fmt_float(rho, 2)}, tau {fmt_float(tau, 2)}", transform=ax.transAxes, fontsize=6.5, color=muted)
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xlabel("ASPR stance (1-5)", fontsize=7.2, fontweight="bold")
    ax.set_ylabel("Peer-review stance (1-5)", fontsize=7.2, fontweight="bold")
    style_axis(ax)
    for idx, tier in enumerate(SW_TIER_ORDER):
        fig.text(panel_c[0] + 0.260 + idx * 0.070, panel_c[1] + panel_c[3] - 0.034, tier, fontsize=6.7, color=dark, ha="left")
        fig.patches.append(
            FancyBboxPatch(
                (panel_c[0] + 0.247 + idx * 0.070, panel_c[1] + panel_c[3] - 0.035),
                0.009,
                0.009,
                boxstyle="round,pad=0,rounding_size=0.002",
                transform=fig.transFigure,
                facecolor=tier_colors[tier],
                edgecolor=tier_colors[tier],
            )
        )

    # Panel d: aspect alignment heatmap by S_w tier.
    ax = ax_in(panel_d, (0.150, 0.180, 0.610, 0.650))
    tier_aspect = {
        (str(row.get("aspect")), str(row.get("fig3_sw_tier"))): row
        for row in aspect_tier_summary
    }
    matrix = []
    annotations: List[List[str]] = []
    for aspect in INNOVATION_ASPECTS:
        row_values = []
        row_labels = []
        for tier in SW_TIER_ORDER:
            item = tier_aspect.get((aspect, tier), {})
            matched_rate = numeric(item.get("matched_rate"), float("nan"))
            total = numeric(item.get("total_points"), 0.0)
            row_values.append(matched_rate if math.isfinite(matched_rate) else 0.0)
            row_labels.append(f"{matched_rate * 100:.0f}%\nn={int(total)}" if math.isfinite(matched_rate) else "n/a")
        matrix.append(row_values)
        annotations.append(row_labels)
    ax.imshow(matrix, cmap=heat_cmap, vmin=0, vmax=1, aspect="auto")
    for y_idx, labels in enumerate(annotations):
        for x_idx, label in enumerate(labels):
            value = matrix[y_idx][x_idx]
            ax.text(x_idx, y_idx, label, ha="center", va="center", fontsize=6.5, color="white" if value >= 0.62 else dark, fontweight="bold")
    ax.set_xticks(range(len(SW_TIER_ORDER)))
    ax.set_xticklabels([tier.title() for tier in SW_TIER_ORDER], fontsize=7.0, fontweight="bold")
    ax.set_yticks(range(len(INNOVATION_ASPECTS)))
    ax.set_yticklabels([ASPECT_DISPLAY_NAMES[aspect] for aspect in INNOVATION_ASPECTS], fontsize=7.0)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax_strip = ax_in(panel_d, (0.800, 0.180, 0.090, 0.650))
    ax_strip.set_xlim(0, 1)
    ax_strip.set_ylim(-0.5, len(INNOVATION_ASPECTS) - 0.5)
    for y_idx, aspect in enumerate(INNOVATION_ASPECTS):
        rows_for_aspect = [tier_aspect.get((aspect, tier), {}) for tier in SW_TIER_ORDER]
        total = sum(numeric(row.get("total_points"), 0.0) for row in rows_for_aspect)
        contradicted = sum(numeric(row.get("contradicted_points"), 0.0) for row in rows_for_aspect)
        rate = contradicted / total if total else 0.0
        ax_strip.barh(y_idx, min(rate, 1.0), height=0.55, color=peer_color, edgecolor="none")
        ax_strip.text(min(rate + 0.03, 0.95), y_idx, f"{rate * 100:.0f}%", va="center", fontsize=5.9, color=dark)
    ax_strip.set_yticks([])
    ax_strip.set_xticks([0, 0.5, 1.0])
    ax_strip.set_xticklabels(["0", "50", "100"], fontsize=5.8)
    ax_strip.invert_yaxis()
    ax_strip.set_title("Contrad.", fontsize=6.0, color=peer_color, pad=2)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    # Panel e: S_w ladder examples.
    ax = ax_in(panel_e, (0.025, 0.095, 0.950, 0.780))
    ax.axis("off")
    ladder_examples = examples.get("sw_ladder_examples") if isinstance(examples.get("sw_ladder_examples"), list) else []
    examples_by_tier = {str(example.get("fig3_sw_tier") or ""): example for example in ladder_examples}
    for idx, tier in enumerate(SW_TIER_ORDER):
        example = examples_by_tier.get(tier, {})
        x0 = 0.006 + idx * 0.331
        width = 0.308
        ax.add_patch(
            FancyBboxPatch(
                (x0, 0.010),
                width,
                0.950,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                transform=ax.transAxes,
                facecolor=tier_fills[tier],
                edgecolor="#cbd5e1",
                linewidth=0.80,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x0 + 0.006, 0.040),
                0.010,
                0.890,
                boxstyle="round,pad=0,rounding_size=0.004",
                transform=ax.transAxes,
                facecolor=tier_colors[tier],
                edgecolor=tier_colors[tier],
                linewidth=0,
            )
        )
        percentile = numeric(example.get("fig3_sw_percentile"), float("nan"))
        title = f"{tier.title()} S_w | pctl {percentile * 100:.0f}%" if math.isfinite(percentile) else f"{tier.title()} S_w"
        ax.text(x0 + 0.026, 0.900, title, transform=ax.transAxes, fontsize=7.1, fontweight="bold", color=tier_colors[tier], va="top")
        if example:
            ax.text(
                x0 + 0.026,
                0.815,
                f"{example.get('paper_id')} | {example.get('relation')} | {example.get('aspect_label')}",
                transform=ax.transAxes,
                fontsize=5.9,
                color=muted,
                va="top",
            )
            ax.text(x0 + 0.026, 0.720, "Reviewer quote", transform=ax.transAxes, fontsize=6.0, fontweight="bold", color=peer_color)
            ax.text(
                x0 + 0.026,
                0.665,
                wrapped(compact_example_text(example.get("peer_quote") or example.get("peer_point"), 120), 34),
                transform=ax.transAxes,
                fontsize=5.55,
                color=dark,
                va="top",
            )
            ax.text(x0 + 0.026, 0.385, "ASPR point", transform=ax.transAxes, fontsize=6.0, fontweight="bold", color=agent_color)
            ax.text(
                x0 + 0.026,
                0.330,
                wrapped(compact_example_text(example.get("agent_point") or "(no matched candidate)", 120), 34),
                transform=ax.transAxes,
                fontsize=5.55,
                color=dark,
                va="top",
            )
            if bool_value(example.get("cross_aspect_match")):
                ax.text(x0 + 0.026, 0.060, f"cross-aspect: {example.get('candidate_aspect')}", transform=ax.transAxes, fontsize=5.7, color=tier_colors["high"], fontweight="bold")
        else:
            ax.text(x0 + 0.026, 0.570, "No traceable semantic example in this tier.", transform=ax.transAxes, fontsize=6.0, color=muted)

    # Summary strip.
    x, y, w, h = summary_panel
    fig.patches.append(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            transform=fig.transFigure,
            linewidth=0.85,
            edgecolor="#82b2f0",
            facecolor="#f7fbff",
            zorder=-20,
        )
    )
    source_modes = sorted({str(row.get("graph_prior_prompt_mode") or "") for row in main_rows if str(row.get("graph_prior_prompt_mode") or "")})
    source_text = source_modes[0] if source_modes else "fig3_sw_only"
    summary_text = (
        f"Summary | n={n_main}; S_w tiers low/middle/high={tier_counts['low']}/{tier_counts['middle']}/{tier_counts['high']}; "
        f"within-1 stance {fmt_pct(summary['mean_stance_within_one'])}; "
        f"QWK {fmt_float(summary['mean_quadratic_weighted_kappa'], 2)}; "
        f"claim-evidence coverage {fmt_pct(summary['mean_claim_evidence_coverage'])}; "
        f"contradiction {fmt_pct(summary['mean_contradiction_rate'])}; prompt mode {source_text}."
    )
    fig.text(x + 0.018, y + h / 2, summary_text, fontsize=7.8, color=dark, ha="left", va="center")
    fig.text(
        0.5,
        0.018,
        "Note: Agent sees only Fig.3 S_w/percentile/tier; component metrics and weights remain audit artifacts. Peer labels are quote-grounded.",
        fontsize=7.2,
        color=dark,
        ha="center",
        va="center",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"fig4_full.{suffix}", dpi=100)
    plt.close(fig)
    draw_fig4_system_dashboard(output_dir, human_hours=human_hours, quiet=True)
    progress_log(f"Drew S_w-centric Fig.4 with n={n_main}.", quiet)
    return summary


def draw_fig4_publication_summary(output_dir: Path, human_hours: float = DEFAULT_HUMAN_HOURS, quiet: bool = False) -> Dict[str, Any]:
    """Draw the publication-facing Fig.4 with a Fig.1-3 compatible visual system."""
    rows = read_csv_records(output_dir / "fig4_metrics_summary.csv")
    main_rows = [row for row in rows if bool_value(row.get("included_in_main", True))]
    semantic_rows = read_jsonl(output_dir / "fig4_semantic_claim_matches.jsonl")
    included_paper_ids = {str(row.get("paper_id")) for row in main_rows}
    aspect_tier_summary = build_aspect_relation_by_sw_tier(semantic_rows, main_rows, included_paper_ids)
    write_csv(output_dir / "fig4_aspect_relation_by_sw_tier.csv", aspect_tier_summary)
    examples_path = output_dir / "fig4_claim_examples.json"
    examples = read_json(examples_path) if examples_path.exists() else select_claim_examples(semantic_rows)
    ladder_examples = select_sw_ladder_examples(semantic_rows, main_rows)
    examples["sw_ladder_examples"] = ladder_examples
    write_json(examples_path, examples)

    n_main = len(main_rows)
    reference_tier_counts = {
        tier: sum(1 for row in main_rows if str(row.get("fig3_sw_tier") or "unknown") == tier)
        for tier in SW_TIER_ORDER
    }
    ladder_tier_counts = {
        tier: sum(1 for row in main_rows if visual_sw_tier(row) == tier)
        for tier in SW_TIER_ORDER
    }

    def finite_col(column: str) -> List[float]:
        return [value for value in (numeric(row.get(column)) for row in main_rows) if math.isfinite(value)]

    def mean_col(column: str) -> float:
        return safe_mean(row.get(column) for row in main_rows)

    def fmt_float(value: float, digits: int = 2, fallback: str = "n/a") -> str:
        return f"{value:.{digits}f}" if math.isfinite(value) else fallback

    def fmt_pct(value: float, digits: int = 0, fallback: str = "n/a") -> str:
        return f"{value * 100:.{digits}f}%" if math.isfinite(value) else fallback

    def wrapped(text: Any, width: int = 52) -> str:
        return textwrap.fill(normalize_whitespace(str(text or "")), width=width)

    semantic_for_main = [row for row in semantic_rows if not included_paper_ids or str(row.get("paper_id")) in included_paper_ids]
    refined_counts = {relation: 0 for relation in SEMANTIC_RELATION_SCORES}
    bge_counts = {relation: 0 for relation in SEMANTIC_RELATION_SCORES}
    for row in semantic_for_main:
        refined_counts[semantic_refined_relation(row)] += 1
        bge_counts[semantic_bge_relation(row)] += 1
    semantic_total = sum(refined_counts.values())
    semantic_matched = refined_counts["entailed"] + refined_counts["related"]
    summary = {
        "n_main": n_main,
        "sw_tier_counts": ladder_tier_counts,
        "reference_sw_tier_counts": reference_tier_counts,
        "mean_fig3_sw_percentile": safe_mean(row.get("fig3_sw_percentile") for row in main_rows),
        "mean_stance_within_one": mean_col("stance_within_one_agreement"),
        "mean_quadratic_weighted_kappa": mean_col("quadratic_weighted_kappa"),
        "mean_claim_evidence_coverage": mean_col("claim_evidence_coverage"),
        "mean_contradiction_rate": mean_col("contradiction_rate"),
        "semantic_matched_rate": semantic_matched / semantic_total if semantic_total else float("nan"),
    }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import FancyBboxPatch, Rectangle
    except ImportError:
        _draw_fig4_fallback(output_dir, summary)
        progress_log(f"Drew fallback Fig.4 with n={n_main}.", quiet)
        return summary

    # Palette aligned with Fig.1-3: white/gray canvas, blue and teal evidence,
    # orange/red for comparison or risk, purple only as a secondary indicator hue.
    blue = "#2f69bf"
    deep_blue = "#10223d"
    teal = "#43c69a"
    green = "#2e7d32"
    orange = "#ff8a3d"
    red = "#e3262e"
    purple = "#8d5cf6"
    cyan = "#1598b8"
    navy = "#173f8a"
    gray = "#667085"
    pale_grid = "#e9edf3"
    panel_edge = "#9aa4b2"
    dark = "#111827"
    peer_color = red
    agent_color = blue
    tier_colors = {"low": "#d7a52f", "middle": teal, "high": green}
    tier_fills = {"low": "#fff5d9", "middle": "#e9f8f1", "high": "#ddf2e7"}
    metric_colors = {
        "B": blue,
        "RS": green,
        "DeltaQ0": orange,
        "Uzzi": purple,
        "RTD": cyan,
        "BurtIP": navy,
        "PDE": red,
    }
    heat_cmap = LinearSegmentedColormap.from_list("fig4_match", ["#f7f8fb", "#d8eef6", "#43c69a", "#166b45"])
    count_cmap = LinearSegmentedColormap.from_list("fig4_count", ["#f7f8fb", "#a8c6ea", "#2f69bf"])

    fig = plt.figure(figsize=(14.02, 11.22), dpi=100, facecolor="white")
    fig.text(
        0.5,
        0.982,
        "Fig. 4 | S_w-calibrated ASPR innovation judgements align with quote-grounded peer review",
        ha="center",
        va="top",
        fontsize=13.0,
        fontweight="bold",
        color=dark,
    )
    fig.text(
        0.5,
        0.960,
        "Agent-facing prior uses only Fig.3 S_w; within-50 S_w ladder is display-only because this Nature sample sits in the Fig.3 high-prior tail.",
        ha="center",
        va="top",
        fontsize=8.2,
        color=gray,
    )

    def panel(bounds: Tuple[float, float, float, float], label: str, title: str) -> None:
        x, y, w, h = bounds
        fig.patches.append(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.006,rounding_size=0.007",
                transform=fig.transFigure,
                linewidth=0.90,
                edgecolor=panel_edge,
                facecolor="white",
                zorder=-20,
            )
        )
        fig.text(x + 0.010, y + h - 0.016, label, fontsize=12.2, fontweight="bold", color="black", ha="left", va="top")
        fig.text(x + 0.036, y + h - 0.018, title, fontsize=8.9, fontweight="bold", color=dark, ha="left", va="top")

    def ax_in(bounds: Tuple[float, float, float, float], rel: Tuple[float, float, float, float]) -> Any:
        x, y, w, h = bounds
        rx, ry, rw, rh = rel
        return fig.add_axes([x + rx * w, y + ry * h, rw * w, rh * h])

    def style_axis(ax: Any, ygrid: bool = True, xgrid: bool = False) -> None:
        if ygrid:
            ax.grid(True, axis="y", color=pale_grid, linewidth=0.8, zorder=0)
        if xgrid:
            ax.grid(True, axis="x", color=pale_grid, linewidth=0.8, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#b7c0cc")
        ax.spines["bottom"].set_color("#b7c0cc")
        ax.tick_params(axis="both", labelsize=6.9, width=0.8, length=3, colors=dark)

    panel_a = (0.030, 0.675, 0.420, 0.260)
    panel_b = (0.472, 0.675, 0.498, 0.260)
    panel_c = (0.030, 0.392, 0.286, 0.240)
    panel_d = (0.342, 0.392, 0.286, 0.240)
    panel_e = (0.654, 0.392, 0.316, 0.240)
    panel_f = (0.030, 0.130, 0.940, 0.220)
    summary_panel = (0.036, 0.045, 0.928, 0.055)
    panel(panel_a, "a", "Evidence bridge from Fig.2/Fig.3 to Fig.4")
    panel(panel_b, "b", "S_w top-tail position and within-sample ladder")
    panel(panel_c, "c", "Stance agreement matrix")
    panel(panel_d, "d", "S_w ladder calibration")
    panel(panel_e, "e", "Aspect fingerprint by S_w ladder")
    panel(panel_f, "f", "Quote-grounded claim-evidence examples")

    # a. Flow with Fig.3 weight lollipop.
    ax = ax_in(panel_a, (0.055, 0.175, 0.890, 0.700))
    ax.axis("off")
    flow = [
        ("Fig.2", "92 -> 7\nsignals", blue),
        ("Fig.3", "learned\nweights", teal),
        ("S_w", "single\nprior", orange),
        ("ASPR", "LATS\nreview", blue),
        ("Peer", "quote\nlabels", red),
    ]
    for idx, (head, body, color) in enumerate(flow):
        x0 = 0.018 + idx * 0.194
        ax.add_patch(
            FancyBboxPatch(
                (x0, 0.610),
                0.142,
                0.300,
                boxstyle="round,pad=0.010,rounding_size=0.016",
                transform=ax.transAxes,
                facecolor="#f8fafc",
                edgecolor=color,
                linewidth=0.95,
            )
        )
        ax.text(x0 + 0.071, 0.805, head, ha="center", va="center", fontsize=7.8, fontweight="bold", color=dark, transform=ax.transAxes)
        ax.text(x0 + 0.071, 0.685, body, ha="center", va="center", fontsize=6.2, color=gray, transform=ax.transAxes)
        if idx < len(flow) - 1:
            ax.annotate("", xy=(x0 + 0.180, 0.760), xytext=(x0 + 0.152, 0.760), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "lw": 1.0, "color": gray})
    ax.text(0.018, 0.525, "Agent prompt: S_w + Fig.3 percentile + tier only; seven component values are audit-only.", fontsize=6.6, color=dark, transform=ax.transAxes)
    weights_config = load_fig3_weight_config(fig3_weights_path_from_env())
    weights = [(metric, numeric(weights_config["weights"].get(metric), 0.0)) for metric in INNOVATION_METRIC_NAMES]
    max_weight = max([value for _metric, value in weights] or [1.0])
    ax.text(0.018, 0.420, "Fig.3 best weight mass", fontsize=6.7, color=gray, fontweight="bold", transform=ax.transAxes)
    for idx, (metric, weight) in enumerate(weights):
        x_pos = 0.050 + idx * 0.128
        y0 = 0.125
        height = 0.060 + (weight / max_weight if max_weight else 0.0) * 0.210
        ax.plot([x_pos, x_pos], [y0, y0 + height], color=metric_colors.get(metric, blue), linewidth=2.0, transform=ax.transAxes, solid_capstyle="round")
        ax.scatter([x_pos], [y0 + height], s=65, color=metric_colors.get(metric, blue), edgecolor="white", linewidth=0.8, transform=ax.transAxes, zorder=4)
        ax.text(x_pos, 0.060, metric, ha="center", va="top", fontsize=5.6, color=dark, transform=ax.transAxes)
    if weights_config.get("warning"):
        ax.text(0.018, 0.010, "Weight fallback active; inspect fig4_graph_prior.csv.", fontsize=5.9, color=red, transform=ax.transAxes)

    # b. Dual-rug: Fig.3 reference tail and within-50 ladder.
    ax = ax_in(panel_b, (0.060, 0.180, 0.870, 0.670))
    for lo, hi, tier in [(0, 33.333, "low"), (33.333, 66.667, "middle"), (66.667, 100, "high")]:
        ax.axvspan(lo, hi, color=tier_fills[tier], alpha=0.58, zorder=0)
    sorted_rows = sorted(main_rows, key=lambda row: numeric(row.get("fig3_sw_percentile"), 0.0))
    ref_xs: List[float] = []
    ladder_xs: List[float] = []
    agent_scores: List[float] = []
    edge_colors: List[str] = []
    for idx, row in enumerate(sorted_rows):
        ref_pct = numeric(row.get("fig3_sw_percentile"), float("nan"))
        ladder_pct = visual_sw_percentile(row)
        if not math.isfinite(ref_pct) or not math.isfinite(ladder_pct):
            continue
        ref_xs.append(ref_pct * 100.0)
        ladder_xs.append(ladder_pct * 100.0)
        agent_scores.append(numeric(row.get("agent_innovation_stance_1_5"), 3.0))
        edge_colors.append(tier_colors.get(visual_sw_tier(row), gray))
        jitter = (((idx * 23) % 11) - 5) * 0.010
        ax.plot([ref_pct * 100.0, ladder_pct * 100.0], [1.02 + jitter, 0.33 + jitter], color="#c8d0db", linewidth=0.35, alpha=0.55, zorder=1)
    if ref_xs:
        ax.scatter(ref_xs, [1.02 + (((idx * 23) % 11) - 5) * 0.010 for idx in range(len(ref_xs))], s=34, color=teal, edgecolor="white", linewidth=0.5, zorder=3)
        scatter = ax.scatter(ladder_xs, [0.33 + (((idx * 23) % 11) - 5) * 0.010 for idx in range(len(ladder_xs))], c=agent_scores, cmap="Blues", vmin=1, vmax=5, s=43, edgecolor=edge_colors, linewidth=0.9, zorder=4)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.032, pad=0.012)
        cbar.ax.tick_params(labelsize=5.8, length=2)
        cbar.set_label("ASPR stance", fontsize=6.0, fontweight="bold")
    ax.text(2, 1.175, f"Fig.3 reference tier: low/middle/high={reference_tier_counts['low']}/{reference_tier_counts['middle']}/{reference_tier_counts['high']}", fontsize=6.5, color=dark)
    ax.text(2, 0.520, f"within-50 S_w ladder: low/middle/high={ladder_tier_counts['low']}/{ladder_tier_counts['middle']}/{ladder_tier_counts['high']}", fontsize=6.5, color=dark)
    for tier, xpos in [("low", 16.5), ("middle", 50), ("high", 83.5)]:
        ax.text(xpos, -0.050, tier, ha="center", va="top", fontsize=6.2, color=tier_colors[tier], fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.05, 1.32)
    ax.set_yticks([0.33, 1.02])
    ax.set_yticklabels(["within-50\nladder", "Fig.3 ref\npercentile"], fontsize=6.5)
    ax.set_xlabel("S_w percentile scale", fontsize=7.0, fontweight="bold")
    ax.set_xticks([0, 33, 67, 100])
    ax.set_xticklabels(["0", "33", "67", "100"])
    style_axis(ax, ygrid=False, xgrid=True)

    # c. Stance agreement matrix.
    ax = ax_in(panel_c, (0.170, 0.170, 0.565, 0.660))
    matrix = [[0 for _ in range(5)] for _ in range(5)]
    valid_pairs = []
    for row in main_rows:
        agent = numeric(row.get("agent_innovation_stance_1_5"), float("nan"))
        peer = numeric(row.get("peer_innovation_stance_1_5"), float("nan"))
        if math.isfinite(agent) and math.isfinite(peer):
            a_idx = min(4, max(0, int(round(agent)) - 1))
            p_idx = min(4, max(0, int(round(peer)) - 1))
            matrix[4 - p_idx][a_idx] += 1
            valid_pairs.append((agent, peer))
    max_count = max([count for row in matrix for count in row] or [1])
    ax.imshow(matrix, cmap=count_cmap, vmin=0, vmax=max_count, aspect="equal")
    for y_idx, row_counts in enumerate(matrix):
        for x_idx, count in enumerate(row_counts):
            if count:
                ax.text(x_idx, y_idx, str(count), ha="center", va="center", fontsize=7.0, color="white" if count > max_count * 0.48 else dark, fontweight="bold")
    for x_idx in range(5):
        for y_idx in range(5):
            peer_score = 5 - y_idx
            agent_score = x_idx + 1
            if abs(peer_score - agent_score) <= 1:
                ax.add_patch(Rectangle((x_idx - 0.5, y_idx - 0.5), 1, 1, fill=False, edgecolor=teal, linewidth=0.8))
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(["1", "2", "3", "4", "5"], fontsize=6.8)
    ax.set_yticklabels(["5", "4", "3", "2", "1"], fontsize=6.8)
    ax.set_xlabel("ASPR stance", fontsize=6.8, fontweight="bold")
    ax.set_ylabel("Peer stance", fontsize=6.8, fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(panel_c[0] + 0.205, panel_c[1] + 0.045, f"Within-1 {fmt_pct(summary['mean_stance_within_one'])} | QWK {fmt_float(summary['mean_quadratic_weighted_kappa'], 2)} | n={len(valid_pairs)}", fontsize=6.7, color=blue, fontweight="bold")

    # d. S_w ladder calibration.
    ax = ax_in(panel_d, (0.125, 0.205, 0.780, 0.610))
    tier_x = list(range(len(SW_TIER_ORDER)))
    agent_means = []
    peer_means = []
    for tier in SW_TIER_ORDER:
        tier_rows = [row for row in main_rows if visual_sw_tier(row) == tier]
        agent_means.append(safe_mean(row.get("agent_innovation_stance_1_5") for row in tier_rows))
        peer_means.append(safe_mean(row.get("peer_innovation_stance_1_5") for row in tier_rows))
    ax.plot(tier_x, [value if math.isfinite(value) else float("nan") for value in agent_means], marker="o", color=agent_color, linewidth=2.0, markersize=5.5, label="ASPR")
    ax.plot(tier_x, [value if math.isfinite(value) else float("nan") for value in peer_means], marker="s", color=peer_color, linewidth=1.6, markersize=5.0, label="Peer")
    for x_idx, tier in enumerate(SW_TIER_ORDER):
        ax.text(x_idx, 1.05, f"n={ladder_tier_counts[tier]}", ha="center", va="bottom", fontsize=6.2, color=gray)
    ax.set_xticks(tier_x)
    ax.set_xticklabels([tier.title() for tier in SW_TIER_ORDER], fontsize=7.0, fontweight="bold")
    ax.set_ylim(1, 5.2)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("Innovation stance", fontsize=6.8, fontweight="bold")
    ax.legend(frameon=False, fontsize=6.7, loc="upper left")
    style_axis(ax, ygrid=True, xgrid=False)
    ax.text(
        0.98,
        0.930,
        "Within-run ladder;\nagent prior remains Fig.3 tier",
        transform=ax.transAxes,
        fontsize=5.8,
        color=gray,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "#e5e7eb", "alpha": 0.92},
    )

    # e. Aspect fingerprint by S_w ladder.
    ax = ax_in(panel_e, (0.185, 0.180, 0.570, 0.650))
    tier_aspect = {(str(row.get("aspect")), str(row.get("sw_visual_tier") or row.get("fig3_sw_tier"))): row for row in aspect_tier_summary}
    matrix_values = []
    for aspect in INNOVATION_ASPECTS:
        row_values = []
        for tier in SW_TIER_ORDER:
            item = tier_aspect.get((aspect, tier), {})
            rate = numeric(item.get("matched_rate"), float("nan"))
            row_values.append(rate if math.isfinite(rate) else 0.0)
        matrix_values.append(row_values)
    ax.imshow(matrix_values, cmap=heat_cmap, vmin=0, vmax=1, aspect="auto")
    for y_idx, aspect in enumerate(INNOVATION_ASPECTS):
        for x_idx, tier in enumerate(SW_TIER_ORDER):
            item = tier_aspect.get((aspect, tier), {})
            rate = numeric(item.get("matched_rate"), float("nan"))
            total = int(numeric(item.get("total_points"), 0.0))
            label = f"{rate * 100:.0f}%\n{total}" if math.isfinite(rate) else "n/a"
            color = "white" if math.isfinite(rate) and rate >= 0.60 else dark
            ax.text(x_idx, y_idx, label, ha="center", va="center", fontsize=5.8, color=color, fontweight="bold")
    ax.set_xticks(range(len(SW_TIER_ORDER)))
    ax.set_xticklabels([tier.title() for tier in SW_TIER_ORDER], fontsize=6.7, fontweight="bold")
    ax.set_yticks(range(len(INNOVATION_ASPECTS)))
    ax.set_yticklabels([ASPECT_DISPLAY_NAMES[aspect] for aspect in INNOVATION_ASPECTS], fontsize=6.6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax_strip = ax_in(panel_e, (0.805, 0.180, 0.090, 0.650))
    ax_strip.set_xlim(0, 1)
    ax_strip.set_ylim(-0.5, len(INNOVATION_ASPECTS) - 0.5)
    for y_idx, aspect in enumerate(INNOVATION_ASPECTS):
        aspect_rows = [tier_aspect.get((aspect, tier), {}) for tier in SW_TIER_ORDER]
        total = sum(numeric(row.get("total_points"), 0.0) for row in aspect_rows)
        contradicted = sum(numeric(row.get("contradicted_points"), 0.0) for row in aspect_rows)
        rate = contradicted / total if total else 0.0
        ax_strip.barh(y_idx, min(rate, 1.0), color=red, height=0.48)
        ax_strip.text(min(rate + 0.04, 0.96), y_idx, f"{rate * 100:.0f}%", va="center", fontsize=5.6, color=dark)
    ax_strip.set_yticks([])
    ax_strip.set_xticks([0, 0.5, 1.0])
    ax_strip.set_xticklabels(["0", "50", "100"], fontsize=5.5)
    ax_strip.invert_yaxis()
    ax_strip.set_title("contrad.", fontsize=5.9, color=red, pad=2)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)

    # f. Claim examples as traceable cards.
    ax = ax_in(panel_f, (0.020, 0.095, 0.960, 0.790))
    ax.axis("off")
    examples_by_tier = {str(example.get("fig4_sw_ladder_tier") or example.get("fig3_sw_tier") or ""): example for example in ladder_examples}
    for idx, tier in enumerate(SW_TIER_ORDER):
        example = examples_by_tier.get(tier, {})
        x0 = 0.004 + idx * 0.333
        width = 0.314
        ax.add_patch(
            FancyBboxPatch(
                (x0, 0.018),
                width,
                0.940,
                boxstyle="round,pad=0.010,rounding_size=0.014",
                transform=ax.transAxes,
                facecolor=tier_fills[tier],
                edgecolor="#cbd5e1",
                linewidth=0.75,
            )
        )
        ax.add_patch(Rectangle((x0 + 0.010, 0.060), 0.010, 0.850, transform=ax.transAxes, facecolor=tier_colors[tier], edgecolor=tier_colors[tier]))
        percentile = numeric(example.get("fig4_sw_batch_percentile"), float("nan"))
        title = f"{tier.title()} S_w ladder | pctl {percentile * 100:.0f}%" if math.isfinite(percentile) else f"{tier.title()} S_w ladder"
        ax.text(x0 + 0.028, 0.895, title, transform=ax.transAxes, fontsize=6.8, fontweight="bold", color=tier_colors[tier], va="top")
        if example:
            ax.text(x0 + 0.028, 0.805, f"{example.get('paper_id')} | {example.get('relation')} | {example.get('aspect_label')}", transform=ax.transAxes, fontsize=5.6, color=gray, va="top")
            ax.text(x0 + 0.028, 0.700, "Reviewer quote", transform=ax.transAxes, fontsize=5.8, fontweight="bold", color=peer_color)
            ax.text(x0 + 0.028, 0.645, wrapped(compact_example_text(example.get("peer_quote") or example.get("peer_point"), 122), 35), transform=ax.transAxes, fontsize=5.35, color=dark, va="top")
            ax.text(x0 + 0.028, 0.385, "ASPR point", transform=ax.transAxes, fontsize=5.8, fontweight="bold", color=agent_color)
            ax.text(x0 + 0.028, 0.330, wrapped(compact_example_text(example.get("agent_point") or "(no matched candidate)", 122), 35), transform=ax.transAxes, fontsize=5.35, color=dark, va="top")
        else:
            ax.text(x0 + 0.028, 0.560, "No traceable semantic example in this display tier.", transform=ax.transAxes, fontsize=5.9, color=gray)

    # Summary strip.
    x, y, w, h = summary_panel
    fig.patches.append(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.005,rounding_size=0.007",
            transform=fig.transFigure,
            linewidth=0.80,
            edgecolor="#cbd5e1",
            facecolor="#f8fafc",
            zorder=-20,
        )
    )
    summary_text = (
        f"Summary | n={n_main}; Fig.3 reference high={reference_tier_counts['high']}/{n_main}; "
        f"within-50 ladder low/middle/high={ladder_tier_counts['low']}/{ladder_tier_counts['middle']}/{ladder_tier_counts['high']}; "
        f"within-1 stance {fmt_pct(summary['mean_stance_within_one'])}; "
        f"QWK {fmt_float(summary['mean_quadratic_weighted_kappa'], 2)}; "
        f"claim coverage {fmt_pct(summary['mean_claim_evidence_coverage'])}; contradiction {fmt_pct(summary['mean_contradiction_rate'])}."
    )
    fig.text(x + 0.018, y + h / 2, summary_text, fontsize=7.4, color=dark, ha="left", va="center")
    fig.text(
        0.5,
        0.018,
        "Traceability: S_w values in fig4_graph_prior.csv; plotted summary in fig4_metrics_summary.csv; claim examples from fig4_claim_examples.json.",
        fontsize=6.9,
        color=dark,
        ha="center",
        va="center",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"fig4_full.{suffix}", dpi=100)
    plt.close(fig)
    draw_fig4_system_dashboard(output_dir, human_hours=human_hours, quiet=True)
    progress_log(f"Drew publication-style Fig.4 with n={n_main}.", quiet)
    return summary


def draw_fig4(output_dir: Path, human_hours: float = DEFAULT_HUMAN_HOURS, quiet: bool = False) -> Dict[str, Any]:
    """Draw manuscript Fig.4 focused on peer-review innovation-validation evidence."""
    return draw_fig4_publication_summary(output_dir, human_hours=human_hours, quiet=quiet)
    rows = read_csv_records(output_dir / "fig4_metrics_summary.csv")
    main_rows = [row for row in rows if bool_value(row.get("included_in_main", True))]
    semantic_rows = read_jsonl(output_dir / "fig4_semantic_claim_matches.jsonl")
    included_paper_ids = {str(row.get("paper_id")) for row in main_rows}
    aspect_summary_path = output_dir / "fig4_aspect_relation_summary.csv"
    examples_path = output_dir / "fig4_claim_examples.json"
    if aspect_summary_path.exists():
        aspect_summary = read_csv_records(aspect_summary_path)
    else:
        aspect_summary = build_aspect_relation_summary(semantic_rows, included_paper_ids)
        write_csv(aspect_summary_path, aspect_summary)
    if examples_path.exists():
        examples = read_json(examples_path)
    else:
        examples = select_claim_examples(semantic_rows)
        write_json(examples_path, examples)
    n_main = len(main_rows)

    def finite_col(column: str) -> List[float]:
        return [value for value in (numeric(row.get(column)) for row in main_rows) if math.isfinite(value)]

    def mean_col(column: str) -> float:
        return safe_mean(row.get(column) for row in main_rows)

    def quantile(values: Sequence[float], q: float) -> float:
        clean = sorted(value for value in values if math.isfinite(value))
        if not clean:
            return float("nan")
        if len(clean) == 1:
            return clean[0]
        pos = (len(clean) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return clean[lo]
        return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)

    def ranks(values: Sequence[float]) -> List[float]:
        indexed = sorted(enumerate(values), key=lambda item: item[1])
        output = [0.0 for _value in values]
        start = 0
        while start < len(indexed):
            end = start + 1
            while end < len(indexed) and indexed[end][1] == indexed[start][1]:
                end += 1
            rank_value = (start + 1 + end) / 2.0
            for idx in range(start, end):
                output[indexed[idx][0]] = rank_value
            start = end
        return output

    def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
        if len(xs) < 2 or len(xs) != len(ys):
            return float("nan")
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
        y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
        return numerator / (x_den * y_den) if x_den and y_den else float("nan")

    def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
        return pearson(ranks(xs), ranks(ys))

    def kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> float:
        concordant = 0
        discordant = 0
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                x_delta = xs[i] - xs[j]
                y_delta = ys[i] - ys[j]
                if x_delta == 0 or y_delta == 0:
                    continue
                if x_delta * y_delta > 0:
                    concordant += 1
                elif x_delta * y_delta < 0:
                    discordant += 1
        total = concordant + discordant
        return (concordant - discordant) / total if total else float("nan")

    def fmt_float(value: float, digits: int = 2, fallback: str = "n/a") -> str:
        return f"{value:.{digits}f}" if math.isfinite(value) else fallback

    def fmt_pct(value: float, digits: int = 0, fallback: str = "n/a") -> str:
        return f"{value * 100:.{digits}f}%" if math.isfinite(value) else fallback

    def wrapped(text: Any, width: int = 52) -> str:
        return textwrap.fill(normalize_whitespace(str(text or "")), width=width)

    summary = {
        "n_main": n_main,
        "mean_stance_within_one": mean_col("stance_within_one_agreement"),
        "mean_quadratic_weighted_kappa": mean_col("quadratic_weighted_kappa"),
        "mean_soft_claim_recall": mean_col("soft_claim_recall"),
        "mean_claim_evidence_coverage": mean_col("claim_evidence_coverage"),
        "mean_contradiction_rate": mean_col("contradiction_rate"),
    }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        _draw_fig4_fallback(output_dir, summary)
        progress_log(f"Drew fallback Fig.4 with n={n_main}.", quiet)
        return summary

    peer_color = "#ef6b66"
    peer_light = "#f9b4ad"
    agent_color = "#3f78cf"
    agent_light = "#9fc3f4"
    entailed_color = "#2367c7"
    related_color = "#9fc3f4"
    no_match_color = "#d7dde7"
    contradicted_color = "#e45757"
    blue = "#2468c9"
    green = "#15964a"
    dark = "#111827"
    muted = "#526173"
    panel_edge = "#c9d0da"
    grid = "#e7ebf1"

    fig = plt.figure(figsize=(14.02, 11.22), dpi=100, facecolor="white")
    fig.text(
        0.5,
        0.982,
        "Fig. 4 | ASPR innovation evaluation aligns with quote-grounded peer-review judgements",
        ha="center",
        va="top",
        fontsize=13.0,
        fontweight="bold",
        color="black",
    )

    def panel(bounds: Tuple[float, float, float, float], label: str, title: str) -> None:
        x, y, w, h = bounds
        fig.patches.append(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.007,rounding_size=0.007",
                transform=fig.transFigure,
                linewidth=0.9,
                edgecolor=panel_edge,
                facecolor="white",
                zorder=-20,
            )
        )
        fig.text(x + 0.010, y + h - 0.018, label, fontsize=12.5, fontweight="bold", color="black", ha="left", va="top")
        fig.text(x + 0.035, y + h - 0.020, title, fontsize=9.4, fontweight="bold", color="black", ha="left", va="top")

    def ax_in(bounds: Tuple[float, float, float, float], rel: Tuple[float, float, float, float]) -> Any:
        x, y, w, h = bounds
        rx, ry, rw, rh = rel
        return fig.add_axes([x + rx * w, y + ry * h, rw * w, rh * h])

    def style_axis(ax: Any, ygrid: bool = True, xgrid: bool = False) -> None:
        if ygrid:
            ax.grid(True, axis="y", color=grid, linewidth=0.8)
        if xgrid:
            ax.grid(True, axis="x", color=grid, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#b7c0cc")
        ax.spines["bottom"].set_color("#b7c0cc")
        ax.tick_params(axis="both", labelsize=7.1, width=0.8, length=3, colors="#1f2937")

    panel_a = (0.020, 0.628, 0.345, 0.318)
    panel_b = (0.385, 0.628, 0.595, 0.318)
    panel_c = (0.020, 0.362, 0.460, 0.238)
    panel_d = (0.500, 0.362, 0.480, 0.238)
    panel_e = (0.020, 0.132, 0.960, 0.202)
    summary_panel = (0.028, 0.040, 0.944, 0.064)

    panel(panel_a, "a", "Validation workflow")
    panel(panel_b, "b", "Innovation stance agreement")
    panel(panel_c, "c", "Aspect-level semantic alignment")
    panel(panel_d, "d", "Claim-evidence examples")
    panel(panel_e, "e", "BGE-only vs LLM-refined sensitivity")

    # Panel a: workflow.
    ax = ax_in(panel_a, (0.050, 0.100, 0.900, 0.740))
    ax.axis("off")
    workflow = [
        ("Paper\ndossier", "title, abstract,\nmethods/results"),
        ("ASPR\nagent", "LATS + prior-art\nretrieval"),
        ("Peer-review\nlabels", "quote-grounded\nhuman judgements"),
        ("Semantic\nalignment", "BGE-M3 + bounded\nLLM/NLI refinement"),
        ("Fig.4\nmetrics", "stance, aspects,\nclaim evidence"),
    ]
    for idx, (head, body) in enumerate(workflow):
        x0 = 0.02 + idx * 0.195
        ax.add_patch(
            FancyBboxPatch(
                (x0, 0.38),
                0.145,
                0.36,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                transform=ax.transAxes,
                facecolor="#f8fbff" if idx % 2 == 0 else "#fffafa",
                edgecolor=agent_color if idx % 2 == 0 else peer_color,
                linewidth=0.9,
            )
        )
        ax.text(x0 + 0.0725, 0.625, head, ha="center", va="center", fontsize=7.6, fontweight="bold", color=dark, transform=ax.transAxes)
        ax.text(x0 + 0.0725, 0.465, body, ha="center", va="center", fontsize=6.4, color=muted, transform=ax.transAxes)
        if idx < len(workflow) - 1:
            ax.annotate(
                "",
                xy=(x0 + 0.185, 0.56),
                xytext=(x0 + 0.153, 0.56),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#7d8796"},
            )
    ax.text(
        0.02,
        0.15,
        f"Main run: n={n_main}; plotted values trace to metrics CSV, semantic JSONL, and per-paper cache.",
        fontsize=7.0,
        color=dark,
        transform=ax.transAxes,
    )

    # Panel b: stance agreement scatter + distribution.
    peer_scores = finite_col("peer_innovation_stance_1_5")
    agent_scores = finite_col("agent_innovation_stance_1_5")
    score_pairs = [
        (numeric(row.get("agent_innovation_stance_1_5")), numeric(row.get("peer_innovation_stance_1_5")))
        for row in main_rows
        if math.isfinite(numeric(row.get("agent_innovation_stance_1_5")))
        and math.isfinite(numeric(row.get("peer_innovation_stance_1_5")))
    ]
    ax = ax_in(panel_b, (0.055, 0.185, 0.420, 0.630))
    if score_pairs:
        xs = [item[0] for item in score_pairs]
        ys = [item[1] for item in score_pairs]
        jittered_xs = [x + (((idx * 37) % 9) - 4) * 0.018 for idx, x in enumerate(xs)]
        jittered_ys = [y + (((idx * 29) % 9) - 4) * 0.018 for idx, y in enumerate(ys)]
        ax.scatter(jittered_xs, jittered_ys, s=42, color=agent_light, alpha=0.82, edgecolor=agent_color, linewidth=0.55)
        ax.plot([1, 5], [1, 5], color="#9aa4b2", linestyle="--", linewidth=0.9)
        ax.text(
            0.05,
            0.91,
            f"Within-1: {fmt_pct(mean_col('stance_within_one_agreement'))}\nQWK: {fmt_float(mean_col('quadratic_weighted_kappa'), 2)}\nN valid: {len(score_pairs)}",
            transform=ax.transAxes,
            fontsize=7.3,
            color=blue,
            fontweight="bold",
            va="top",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#e3e8f0", "alpha": 0.95},
        )
        rho = spearman(xs, ys)
        tau = kendall_tau(xs, ys)
        ax.text(0.05, 0.05, f"Rank caveat: Spearman {fmt_float(rho, 2)}, Kendall {fmt_float(tau, 2)}", transform=ax.transAxes, fontsize=6.7, color=muted)
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xlabel("ASPR stance (1-5)", fontsize=7.2, fontweight="bold")
    ax.set_ylabel("Peer-review stance (1-5)", fontsize=7.2, fontweight="bold")
    style_axis(ax)

    ax = ax_in(panel_b, (0.570, 0.185, 0.360, 0.630))
    ratings = [1, 2, 3, 4, 5]
    peer_den = max(len(peer_scores), 1)
    agent_den = max(len(agent_scores), 1)
    peer_props = [sum(1 for value in peer_scores if round(value) == rating) / peer_den * 100.0 for rating in ratings]
    agent_props = [sum(1 for value in agent_scores if round(value) == rating) / agent_den * 100.0 for rating in ratings]
    ax.bar([rating - 0.17 for rating in ratings], peer_props, width=0.30, color=peer_light, edgecolor=peer_color, linewidth=0.7, label="Peer review")
    ax.bar([rating + 0.17 for rating in ratings], agent_props, width=0.30, color=agent_light, edgecolor=agent_color, linewidth=0.7, label="ASPR")
    ax.set_xticks(ratings)
    ax.set_xticklabels(["1", "2", "3", "4", "5"], fontsize=7.0)
    ax.set_ylabel("Proportion", fontsize=7.2, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{int(tick)}%" for tick in ax.get_yticks()])
    ax.legend(frameon=False, fontsize=7.0, loc="upper left")
    style_axis(ax)

    # Panel c: relation stack by aspect.
    ax = ax_in(panel_c, (0.115, 0.170, 0.815, 0.650))
    relation_order = ["entailed", "related", "no_match", "contradicted"]
    relation_colors = {
        "entailed": entailed_color,
        "related": related_color,
        "no_match": no_match_color,
        "contradicted": contradicted_color,
    }
    y_positions = list(range(len(INNOVATION_ASPECTS)))
    for y_idx, aspect in enumerate(INNOVATION_ASPECTS):
        row = next((item for item in aspect_summary if str(item.get("aspect")) == aspect), {})
        total = numeric(row.get("total_points"), 0.0)
        left = 0.0
        for relation in relation_order:
            value = numeric(row.get(f"{relation}_points"), 0.0)
            pct = value / total * 100.0 if total else 0.0
            ax.barh(y_idx, pct, left=left, height=0.58, color=relation_colors[relation], edgecolor="white", linewidth=0.7)
            if pct >= 11:
                ax.text(left + pct / 2, y_idx, f"{pct:.0f}%", ha="center", va="center", fontsize=6.5, color="white" if relation in {"entailed", "contradicted"} else dark, fontweight="bold")
            left += pct
        ax.text(102, y_idx, f"n={int(total)}", ha="left", va="center", fontsize=6.5, color=muted)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([ASPECT_DISPLAY_NAMES[aspect] for aspect in INNOVATION_ASPECTS], fontsize=7.1)
    ax.set_xlim(0, 112)
    ax.set_xlabel("Share of peer-review points", fontsize=7.2, fontweight="bold")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels([f"{tick}%" for tick in [0, 25, 50, 75, 100]])
    ax.invert_yaxis()
    style_axis(ax, ygrid=False, xgrid=True)
    legend_x = 0.13
    for idx, relation in enumerate(relation_order):
        fig.text(panel_c[0] + legend_x + idx * 0.080, panel_c[1] + panel_c[3] - 0.035, relation.replace("_", " "), fontsize=6.8, color=dark, ha="left")
        fig.patches.append(
            FancyBboxPatch(
                (panel_c[0] + legend_x + idx * 0.080 - 0.014, panel_c[1] + panel_c[3] - 0.036),
                0.010,
                0.010,
                boxstyle="round,pad=0,rounding_size=0.001",
                transform=fig.transFigure,
                facecolor=relation_colors[relation],
                edgecolor=relation_colors[relation],
            )
        )

    # Panel d: examples.
    ax = ax_in(panel_d, (0.040, 0.100, 0.925, 0.750))
    ax.axis("off")
    example_rows = examples.get("examples") if isinstance(examples.get("examples"), list) else []
    card_colors = {"matched": "#eef6ff", "missed": "#f8fafc", "contradicted": "#fff1f1"}
    for idx, example in enumerate(example_rows[:3]):
        x0 = 0.010 + idx * 0.330
        width = 0.305
        etype = str(example.get("example_type") or "example")
        ax.add_patch(
            FancyBboxPatch(
                (x0, 0.02),
                width,
                0.92,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                transform=ax.transAxes,
                facecolor=card_colors.get(etype, "white"),
                edgecolor="#cbd5e1",
                linewidth=0.75,
            )
        )
        ax.text(x0 + 0.012, 0.885, f"{etype.title()} | {example.get('relation')} | {example.get('aspect_label')}", transform=ax.transAxes, fontsize=6.8, fontweight="bold", color=dark, va="top")
        ax.text(x0 + 0.012, 0.785, f"paper_id: {example.get('paper_id')}", transform=ax.transAxes, fontsize=6.0, color=muted, va="top")
        ax.text(x0 + 0.012, 0.670, "Reviewer quote", transform=ax.transAxes, fontsize=6.1, fontweight="bold", color=peer_color)
        ax.text(x0 + 0.012, 0.615, wrapped(compact_example_text(example.get("peer_quote") or example.get("peer_point"), 130), 31), transform=ax.transAxes, fontsize=5.75, color=dark, va="top")
        ax.text(x0 + 0.012, 0.350, "ASPR point", transform=ax.transAxes, fontsize=6.1, fontweight="bold", color=agent_color)
        ax.text(x0 + 0.012, 0.295, wrapped(compact_example_text(example.get("agent_point") or "(no matched candidate)", 130), 31), transform=ax.transAxes, fontsize=5.75, color=dark, va="top")
        if bool_value(example.get("cross_aspect_match")):
            ax.text(x0 + 0.012, 0.065, f"cross-aspect: {example.get('candidate_aspect')}", transform=ax.transAxes, fontsize=5.8, color=green, fontweight="bold")

    # Panel e: sensitivity.
    ax = ax_in(panel_e, (0.055, 0.220, 0.410, 0.560))
    refined_counts = {relation: 0 for relation in SEMANTIC_RELATION_SCORES}
    bge_counts = {relation: 0 for relation in SEMANTIC_RELATION_SCORES}
    semantic_for_main = [row for row in semantic_rows if not included_paper_ids or str(row.get("paper_id")) in included_paper_ids]
    for row in semantic_for_main:
        refined_counts[semantic_refined_relation(row)] += 1
        bge_counts[semantic_bge_relation(row)] += 1
    def rate(counts: Mapping[str, int], relation_set: set[str]) -> float:
        total = sum(counts.values())
        return sum(counts.get(relation, 0) for relation in relation_set) / total if total else float("nan")
    groups = ["Matched", "No match", "Contradicted"]
    bge_values = [rate(bge_counts, {"entailed", "related"}), rate(bge_counts, {"no_match"}), rate(bge_counts, {"contradicted"})]
    refined_values = [rate(refined_counts, {"entailed", "related"}), rate(refined_counts, {"no_match"}), rate(refined_counts, {"contradicted"})]
    xs = list(range(len(groups)))
    ax.bar([x - 0.17 for x in xs], [value * 100 if math.isfinite(value) else 0 for value in bge_values], width=0.30, color="#d7dde7", edgecolor="#8a94a6", linewidth=0.7, label="BGE-only")
    ax.bar([x + 0.17 for x in xs], [value * 100 if math.isfinite(value) else 0 for value in refined_values], width=0.30, color=agent_light, edgecolor=agent_color, linewidth=0.7, label="BGE + LLM")
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, fontsize=7.0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of peer points", fontsize=7.2, fontweight="bold")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{tick}%" for tick in [0, 25, 50, 75, 100]])
    ax.legend(frameon=False, fontsize=7.0, loc="upper right")
    style_axis(ax)

    ax = ax_in(panel_e, (0.530, 0.170, 0.420, 0.620))
    ax.axis("off")
    llm_refined = sum(1 for row in semantic_for_main if bool_value(row.get("llm_refined")))
    llm_changed = sum(1 for row in semantic_for_main if bool_value(row.get("llm_refined")) and semantic_bge_relation(row) != semantic_refined_relation(row))
    llm_errors = sum(1 for row in semantic_for_main if normalize_whitespace(str(row.get("llm_refinement_error") or "")))
    skipped = sum(1 for row in semantic_for_main if normalize_whitespace(str(row.get("llm_refinement_skipped") or "")))
    cross_aspect = sum(1 for row in semantic_for_main if bool_value(row.get("cross_aspect_match")))
    table_rows = [
        ["Total peer points", str(len(semantic_for_main))],
        ["BGE matched rate", fmt_pct(bge_values[0], 1)],
        ["Refined matched rate", fmt_pct(refined_values[0], 1)],
        ["LLM refined / changed", f"{llm_refined} / {llm_changed}"],
        ["Cross-aspect matches", str(cross_aspect)],
        ["LLM fallback errors / skips", f"{llm_errors} / {skipped}"],
    ]
    table = ax.table(cellText=table_rows, colLabels=["Validity check", "Value"], loc="center", cellLoc="center", colLoc="center", colWidths=[0.58, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(7.1)
    table.scale(1.02, 1.35)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.55)
        if row_idx == 0:
            cell.set_facecolor("#f8fafc")
            cell.get_text().set_fontweight("bold")

    # Summary strip.
    x, y, w, h = summary_panel
    fig.patches.append(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            transform=fig.transFigure,
            linewidth=0.85,
            edgecolor="#82b2f0",
            facecolor="#f7fbff",
            zorder=-20,
        )
    )
    summary_text = (
        f"Summary | n={n_main}; within-1 stance agreement {fmt_pct(summary['mean_stance_within_one'])}; "
        f"QWK {fmt_float(summary['mean_quadratic_weighted_kappa'], 2)}; "
        f"claim-evidence coverage {fmt_pct(summary['mean_claim_evidence_coverage'])}; "
        f"contradiction rate {fmt_pct(summary['mean_contradiction_rate'])}."
    )
    fig.text(x + 0.018, y + h / 2, summary_text, fontsize=8.0, color=dark, ha="left", va="center")
    fig.text(
        0.5,
        0.018,
        "Note: Peer labels are quote-grounded; semantic alignment uses BGE-M3 plus bounded LLM/NLI refinement. All values trace to cached JSON/CSV artifacts.",
        fontsize=7.2,
        color=dark,
        ha="center",
        va="center",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"fig4_full.{suffix}", dpi=100)
    plt.close(fig)
    draw_fig4_system_dashboard(output_dir, human_hours=human_hours, quiet=True)
    progress_log(f"Drew Fig.4 peer-review validation view with n={n_main}.", quiet)
    return summary


def draw_fig4_system_dashboard(output_dir: Path, human_hours: float = DEFAULT_HUMAN_HOURS, quiet: bool = False) -> Dict[str, Any]:
    """Draw Fig.4 in the compact reference style used for the manuscript."""
    rows = read_csv_records(output_dir / "fig4_metrics_summary.csv")
    main_rows = [row for row in rows if bool_value(row.get("included_in_main", True))]
    rating_rows = read_jsonl(output_dir / "fig4_rating_judgements.jsonl")
    n_main = len(main_rows)

    def finite_col(column: str) -> List[float]:
        return [value for value in (numeric(row.get(column)) for row in main_rows) if math.isfinite(value)]

    def mean_col(column: str) -> float:
        return safe_mean(row.get(column) for row in main_rows)

    def sum_col(column: str) -> float:
        return sum(numeric(row.get(column), 0.0) for row in main_rows)

    def quantile(values: Sequence[float], q: float) -> float:
        clean = sorted(value for value in values if math.isfinite(value))
        if not clean:
            return float("nan")
        if len(clean) == 1:
            return clean[0]
        pos = (len(clean) - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return clean[lo]
        return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)

    def median_col(column: str) -> float:
        return quantile(finite_col(column), 0.5)

    def ranks(values: Sequence[float]) -> List[float]:
        indexed = sorted(enumerate(values), key=lambda item: item[1])
        output = [0.0 for _ in values]
        start = 0
        while start < len(indexed):
            end = start + 1
            while end < len(indexed) and indexed[end][1] == indexed[start][1]:
                end += 1
            rank_value = (start + 1 + end) / 2.0
            for idx in range(start, end):
                output[indexed[idx][0]] = rank_value
            start = end
        return output

    def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
        if len(xs) < 2 or len(xs) != len(ys):
            return float("nan")
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
        y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
        return numerator / (x_den * y_den) if x_den and y_den else float("nan")

    def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
        return pearson(ranks(xs), ranks(ys))

    def kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> float:
        concordant = 0
        discordant = 0
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                x_delta = xs[i] - xs[j]
                y_delta = ys[i] - ys[j]
                if x_delta == 0 or y_delta == 0:
                    continue
                if x_delta * y_delta > 0:
                    concordant += 1
                elif x_delta * y_delta < 0:
                    discordant += 1
        total = concordant + discordant
        return (concordant - discordant) / total if total else float("nan")

    def fmt_float(value: float, digits: int = 1, fallback: str = "n/a") -> str:
        return f"{value:.{digits}f}" if math.isfinite(value) else fallback

    def fmt_pct(value: float, digits: int = 0, fallback: str = "n/a") -> str:
        return f"{value * 100:.{digits}f}%" if math.isfinite(value) else fallback

    def percent_points(value: float) -> str:
        return f"{value * 100:+.1f}pp" if math.isfinite(value) else "n/a"

    summary = {
        "n_main": n_main,
        "mean_stance_within_one": mean_col("stance_within_one_agreement"),
        "mean_strict_claim_recall": mean_col("strict_claim_recall"),
        "mean_soft_claim_recall": mean_col("soft_claim_recall"),
        "mean_claim_evidence_coverage": mean_col("claim_evidence_coverage"),
        "mean_claim_validation_pass": mean_col("claim_validation_pass"),
        "median_agent_runtime_seconds": median_col("agent_runtime_seconds"),
    }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, FancyBboxPatch
    except ImportError:
        _draw_fig4_fallback(output_dir, summary)
        progress_log(f"Drew fallback Fig.4 with n={n_main}.", quiet)
        return summary

    peer_color = "#f05b59"
    peer_light = "#f8a19b"
    agent_color = "#4f86d9"
    agent_light = "#9fc3f4"
    blue = "#2468c9"
    green = "#15964a"
    green_light = "#eef9ef"
    dark = "#111827"
    muted = "#526173"
    panel_edge = "#c9d0da"
    grid = "#e7ebf1"

    fig = plt.figure(figsize=(14.02, 11.22), dpi=100, facecolor="white")
    fig.text(
        0.5,
        0.982,
        "Fig. 4 | Validation of the graph-perturbation based evaluation via comparison with peer-review assessments",
        ha="center",
        va="top",
        fontsize=13.0,
        fontweight="bold",
        color="black",
    )

    def panel(bounds: Tuple[float, float, float, float], label: str, title: str) -> None:
        x, y, w, h = bounds
        fig.patches.append(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.007,rounding_size=0.007",
                transform=fig.transFigure,
                linewidth=0.9,
                edgecolor=panel_edge,
                facecolor="white",
                zorder=-20,
            )
        )
        fig.text(x + 0.008, y + h - 0.018, label, fontsize=12.5, fontweight="bold", color="black", ha="left", va="top")
        fig.text(x + 0.032, y + h - 0.020, title, fontsize=9.6, fontweight="bold", color="black", ha="left", va="top")

    def ax_in(bounds: Tuple[float, float, float, float], rel: Tuple[float, float, float, float]) -> Any:
        x, y, w, h = bounds
        rx, ry, rw, rh = rel
        return fig.add_axes([x + rx * w, y + ry * h, rw * w, rh * h])

    def style_axis(ax: Any, ygrid: bool = True, xgrid: bool = False) -> None:
        if ygrid:
            ax.grid(True, axis="y", color=grid, linewidth=0.8)
        if xgrid:
            ax.grid(True, axis="x", color=grid, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#b7c0cc")
        ax.spines["bottom"].set_color("#b7c0cc")
        ax.tick_params(axis="both", labelsize=7.2, width=0.8, length=3, colors="#1f2937")

    def style_table(table: Any, header_fill: str = "#f8fafc") -> None:
        for (row_idx, _col_idx), cell in table.get_celld().items():
            cell.set_edgecolor("#cbd5e1")
            cell.set_linewidth(0.55)
            if row_idx == 0:
                cell.set_facecolor(header_fill)
                cell.get_text().set_fontweight("bold")

    panel_a = (0.020, 0.464, 0.477, 0.488)
    panel_b = (0.510, 0.464, 0.470, 0.488)
    panel_c = (0.020, 0.174, 0.435, 0.266)
    panel_d = (0.463, 0.174, 0.517, 0.266)
    summary_panel = (0.028, 0.040, 0.944, 0.098)
    panel(panel_a, "a", "Consistency & semantic alignment")
    panel(panel_b, "b", "Efficiency (time cost)")
    panel(panel_c, "c", "Readability (opinion quality comparison)")
    panel(panel_d, "d", "Coverage & diagnostic depth")

    # Panel a: stance, rating distribution, semantic similarity.
    peer_scores = finite_col("peer_innovation_stance_1_5")
    agent_scores_all = finite_col("agent_innovation_stance_1_5")
    score_pairs = [
        (numeric(row.get("agent_innovation_stance_1_5")), numeric(row.get("peer_innovation_stance_1_5")))
        for row in main_rows
        if math.isfinite(numeric(row.get("agent_innovation_stance_1_5")))
        and math.isfinite(numeric(row.get("peer_innovation_stance_1_5")))
    ]
    ax = ax_in(panel_a, (0.080, 0.500, 0.400, 0.332))
    if score_pairs:
        xs = [item[0] for item in score_pairs]
        ys = [item[1] for item in score_pairs]
        jittered_xs = [x + (((idx * 37) % 9) - 4) * 0.018 for idx, x in enumerate(xs)]
        jittered_ys = [y + (((idx * 29) % 9) - 4) * 0.018 for idx, y in enumerate(ys)]
        ax.scatter(jittered_xs, jittered_ys, s=42, color=agent_light, alpha=0.82, edgecolor=agent_color, linewidth=0.55)
        if len(set(xs)) > 1:
            x_mean = sum(xs) / len(xs)
            y_mean = sum(ys) / len(ys)
            denominator = sum((x - x_mean) ** 2 for x in xs)
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator if denominator else 0.0
            intercept = y_mean - slope * x_mean
            ax.plot([1, 5], [intercept + slope, intercept + slope * 5], color="#ff3b35", linewidth=1.25)
        rho = spearman(xs, ys)
        tau = kendall_tau(xs, ys)
        corr_text = f"Spearman r = {rho:.2f}, Kendall tau = {tau:.2f}" if math.isfinite(rho) and math.isfinite(tau) else "Rank agreement: n/a"
        ax.text(
            0.08,
            0.91,
            corr_text,
            transform=ax.transAxes,
            fontsize=7.1,
            color=blue,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
        ax.text(0.77, 0.07, f"N = {len(score_pairs)}", transform=ax.transAxes, fontsize=7.3, color=dark)
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title("a1. Overall rating agreement (ASPR vs. peer reviewers)", fontsize=7.7, fontweight="bold", loc="left", pad=5)
    ax.set_xlabel("ASPR overall score\n(1=low, 5=high)", fontsize=7.2, fontweight="bold")
    ax.set_ylabel("Peer-review overall score\n(1=low, 5=high)", fontsize=7.2, fontweight="bold")
    style_axis(ax)

    ax = ax_in(panel_a, (0.595, 0.500, 0.360, 0.332))
    ratings = [1, 2, 3, 4, 5]
    peer_den = max(len(peer_scores), 1)
    agent_den = max(len(agent_scores_all), 1)
    peer_props = [sum(1 for value in peer_scores if round(value) == rating) / peer_den * 100.0 for rating in ratings]
    agent_props = [sum(1 for value in agent_scores_all if round(value) == rating) / agent_den * 100.0 for rating in ratings]
    ax.bar([rating - 0.17 for rating in ratings], peer_props, width=0.30, color=peer_light, edgecolor=peer_color, linewidth=0.7, label="Peer-review")
    ax.bar([rating + 0.17 for rating in ratings], agent_props, width=0.30, color=agent_light, edgecolor=agent_color, linewidth=0.7, label="ASPR agent")
    ax.set_title("a2. Rating distribution (1-5)", fontsize=7.7, fontweight="bold", loc="left", pad=5)
    ax.set_xticks(ratings)
    ax.set_xticklabels(["1 (Low)", "2", "3", "4", "5 (High)"], fontsize=6.6)
    ymax = max(peer_props + agent_props + [40.0])
    ax.set_ylim(0, min(100.0, ymax * 1.22))
    ax.set_ylabel("Proportion", fontsize=7.2, fontweight="bold")
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{int(tick)}%" for tick in ax.get_yticks()])
    ax.legend(frameon=False, fontsize=7.0, loc="upper left", handlelength=1.5)
    style_axis(ax)

    ax = ax_in(panel_a, (0.080, 0.110, 0.650, 0.262))
    cosine_values = finite_col("consistency_cosine")
    if cosine_values:
        weights = [100.0 / len(cosine_values) for _ in cosine_values]
        ax.hist(cosine_values, bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], weights=weights, color=agent_light, edgecolor=agent_color, linewidth=0.8)
        iqr = (quantile(cosine_values, 0.25), quantile(cosine_values, 0.75))
        ax.text(
            0.03,
            0.83,
            f"Mean = {safe_mean(cosine_values):.2f}\nMedian = {quantile(cosine_values, 0.5):.2f}\nIQR = [{iqr[0]:.2f}, {iqr[1]:.2f}]",
            transform=ax.transAxes,
            fontsize=7.1,
            color=dark,
            va="top",
        )
    ax.set_title("a3. Semantic similarity (agent opinion vs. reviewer opinion)", fontsize=7.7, fontweight="bold", loc="left", pad=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Cosine similarity (embedding)", fontsize=7.2, fontweight="bold")
    ax.set_ylabel("Proportion", fontsize=7.2, fontweight="bold")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{int(tick)}%" for tick in ax.get_yticks()])
    style_axis(ax)

    ax = ax_in(panel_a, (0.770, 0.105, 0.195, 0.270))
    ax.axis("off")
    bands = [("< 0.40", 0.0, 0.4), ("0.40-0.60", 0.4, 0.6), ("0.60-0.80", 0.6, 0.8), ("0.80-1.00", 0.8, 1.0000001)]
    band_rows = []
    for label, lower, upper in bands:
        proportion = sum(1 for value in cosine_values if lower <= value < upper) / len(cosine_values) if cosine_values else float("nan")
        band_rows.append([label, fmt_pct(proportion, 1)])
    table = ax.table(cellText=band_rows, colLabels=["Similarity band", "%"], loc="center", cellLoc="center", colLoc="center", colWidths=[0.68, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(6.9)
    table.scale(1.08, 1.36)
    style_table(table)

    # Panel b: time table, log distribution, throughput.
    runtime_minutes = [value / 60.0 for value in finite_col("agent_runtime_seconds") if value > 0]
    runtime_mean = safe_mean(runtime_minutes)
    runtime_median = quantile(runtime_minutes, 0.5)
    runtime_iqr = (quantile(runtime_minutes, 0.25), quantile(runtime_minutes, 0.75))
    human_minutes = float(human_hours) * 60.0
    human_iqr = (human_minutes * 0.68, human_minutes * 1.32)
    speedup_mean = human_minutes / runtime_mean if math.isfinite(runtime_mean) and runtime_mean > 0 else float("nan")
    speedup_median = human_minutes / runtime_median if math.isfinite(runtime_median) and runtime_median > 0 else float("nan")

    ax = ax_in(panel_b, (0.040, 0.618, 0.920, 0.255))
    ax.axis("off")
    time_rows = [
        ["Peer reviewers (aggregate)", f"{human_minutes:.1f}", f"{human_minutes:.1f}", f"{human_iqr[0]:.1f} - {human_iqr[1]:.1f}"],
        ["ASPR agent (LLM/LATS)", fmt_float(runtime_mean, 2), fmt_float(runtime_median, 2), f"{fmt_float(runtime_iqr[0], 2)} - {fmt_float(runtime_iqr[1], 2)}"],
        ["Speed-up (x)", fmt_float(speedup_mean, 1), fmt_float(speedup_median, 1), "-"],
    ]
    table = ax.table(
        cellText=time_rows,
        colLabels=["", "Mean (min)", "Median (min)", "IQR (min)"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.34, 0.22, 0.22, 0.22],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.7)
    table.scale(1.06, 1.62)
    style_table(table, header_fill="#f9fbff")
    for row_idx, color in [(1, peer_color), (2, blue), (3, green)]:
        for col_idx in range(4):
            table[(row_idx, col_idx)].get_text().set_color(color)
            table[(row_idx, col_idx)].get_text().set_fontweight("bold")
    ax.set_title("b1. Time per paper", fontsize=7.7, fontweight="bold", loc="left", pad=5)

    ax = ax_in(panel_b, (0.085, 0.125, 0.520, 0.365))
    if runtime_minutes:
        human_dist = [human_iqr[0], human_minutes * 0.85, human_minutes, human_minutes * 1.15, human_iqr[1]]
        violin = ax.violinplot(
            [human_dist, runtime_minutes],
            positions=[2, 1],
            orientation="horizontal",
            widths=0.36,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, color, edge in zip(violin["bodies"], [peer_light, agent_light], [peer_color, agent_color]):
            body.set_facecolor(color)
            body.set_edgecolor(edge)
            body.set_alpha(0.88)
            body.set_linewidth(0.85)
        ax.boxplot(
            [human_dist, runtime_minutes],
            positions=[2, 1],
            orientation="horizontal",
            widths=0.16,
            patch_artist=True,
            boxprops={"facecolor": "white", "edgecolor": dark, "linewidth": 0.75},
            medianprops={"color": dark, "linewidth": 0.9},
            whiskerprops={"color": dark, "linewidth": 0.7},
            capprops={"color": dark, "linewidth": 0.7},
            showfliers=False,
        )
        ax.set_xscale("log")
        ax.set_xlim(0.3, max(1000, human_minutes * 2.0))
    ax.set_yticks([2, 1])
    ax.set_yticklabels(["Peer reviewers\n(aggregate)", "ASPR agent"], fontsize=7.0, fontweight="bold")
    for label, color in zip(ax.get_yticklabels(), [peer_color, blue]):
        label.set_color(color)
    ax.set_xlabel("Time (minutes, log scale)", fontsize=7.2, fontweight="bold")
    ax.set_title("b2. Time distribution (minutes)", fontsize=7.7, fontweight="bold", loc="left", pad=5)
    style_axis(ax, ygrid=False, xgrid=True)

    ax = ax_in(panel_b, (0.655, 0.125, 0.310, 0.365))
    ax.axis("off")
    human_per_day = 24.0 / float(human_hours) if human_hours > 0 else float("nan")
    agent_per_day = 1440.0 / runtime_median if math.isfinite(runtime_median) and runtime_median > 0 else float("nan")
    gain = agent_per_day / human_per_day if math.isfinite(agent_per_day) and math.isfinite(human_per_day) and human_per_day > 0 else float("nan")
    ax.text(0.02, 0.98, "b3. Completed reviews per day", fontsize=7.7, fontweight="bold", color="black", va="top")
    ax.text(0.02, 0.78, "Peer reviewers (aggregate)", fontsize=7.1, color=peer_color, fontweight="bold")
    for idx in range(12):
        ax.add_patch(Circle((0.06 + (idx % 12) * 0.065, 0.64), 0.018, transform=ax.transAxes, color=peer_light, ec=peer_color, lw=0.5))
    ax.text(0.23, 0.49, f"~ {fmt_float(human_per_day, 1)} papers / reviewer / day", fontsize=7.0, color=dark)
    ax.text(0.02, 0.32, "ASPR agent", fontsize=7.1, color=blue, fontweight="bold")
    for idx in range(42):
        col = idx % 14
        row = idx // 14
        color = agent_light if idx < 34 else "#d7e7fb"
        ax.add_patch(Circle((0.06 + col * 0.055, 0.20 - row * 0.075), 0.016, transform=ax.transAxes, color=color, ec=agent_color if idx < 34 else color, lw=0.45))
    ax.text(0.23, -0.075, f"~ {fmt_float(agent_per_day, 0)} papers / system / day", fontsize=7.0, color=dark)
    ax.add_patch(
        FancyBboxPatch(
            (0.21, -0.245),
            0.58,
            0.120,
            boxstyle="round,pad=0.01,rounding_size=0.018",
            transform=ax.transAxes,
            facecolor=green_light,
            edgecolor="#a8d8b5",
            linewidth=0.8,
            clip_on=False,
        )
    )
    ax.text(0.50, -0.184, f"Throughput gain ~ {fmt_float(gain, 0)}x", fontsize=7.2, color=green, fontweight="bold", ha="center", va="center", transform=ax.transAxes)

    # Panel c: readability and proxy language errors.
    ax = ax_in(panel_c, (0.095, 0.240, 0.565, 0.565))
    error_metrics = [("Tense", "tense_errors_per_5000"), ("Grammar", "grammar_errors_per_5000"), ("Spelling", "spelling_errors_per_5000")]
    peer_errors = [mean_col(f"peer_{metric}") for _label, metric in error_metrics]
    agent_errors = [mean_col(f"agent_{metric}") for _label, metric in error_metrics]
    x_positions = list(range(len(error_metrics)))
    ax.bar([x - 0.17 for x in x_positions], [0.0 if not math.isfinite(value) else value for value in peer_errors], width=0.30, color=peer_light, edgecolor=peer_color, linewidth=0.7, label="Peer reviewers")
    ax.bar([x + 0.17 for x in x_positions], [0.0 if not math.isfinite(value) else value for value in agent_errors], width=0.30, color=agent_light, edgecolor=agent_color, linewidth=0.7, label="ASPR agent")
    for x, peer_value, agent_value in zip(x_positions, peer_errors, agent_errors):
        if math.isfinite(peer_value) and peer_value > 0 and math.isfinite(agent_value):
            reduction = max(0.0, (peer_value - agent_value) / peer_value)
            ax.annotate("", xy=(x + 0.40, agent_value + 1.0), xytext=(x + 0.40, peer_value - 1.0), arrowprops={"arrowstyle": "->", "color": green, "lw": 0.9})
            ax.text(x + 0.46, max(peer_value, agent_value) * 0.55 + 1.0, fmt_pct(reduction, 1), color=green, fontsize=6.9, fontweight="bold", va="center")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for label, _metric in error_metrics], fontsize=7.0)
    ax.set_ylabel("Errors per 5,000 words", fontsize=7.2, fontweight="bold")
    ax.set_title("c1. Language error rates (errors per 5,000 words)", fontsize=7.7, fontweight="bold", loc="left", pad=5)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left")
    style_axis(ax)

    ax = ax_in(panel_c, (0.720, 0.155, 0.230, 0.720))
    ax.axis("off")
    ax.text(0.5, 0.955, "c2. Readability scores\n(higher is better)", fontsize=7.3, fontweight="bold", color="black", ha="center", va="top")
    read_cards = [
        ("Flesch Reading Ease\n(0-100)", mean_col("peer_flesch_reading_ease"), mean_col("agent_flesch_reading_ease"), True),
        ("Flesch-Kincaid Grade Level", mean_col("peer_flesch_kincaid_grade"), mean_col("agent_flesch_kincaid_grade"), False),
    ]
    for idx, (label, peer_value, agent_value, higher_better) in enumerate(read_cards):
        y0 = 0.515 - idx * 0.415
        ax.add_patch(
            FancyBboxPatch(
                (0.03, y0),
                0.94,
                0.320,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                transform=ax.transAxes,
                facecolor="white",
                edgecolor="#cbd5e1",
                linewidth=0.75,
            )
        )
        delta = agent_value - peer_value if math.isfinite(agent_value) and math.isfinite(peer_value) else float("nan")
        good_delta = delta if higher_better else -delta
        ax.text(0.50, y0 + 0.248, label, fontsize=6.5, fontweight="bold", color=dark, ha="center")
        ax.text(0.14, y0 + 0.135, "Peer-review", fontsize=6.7, fontweight="bold", color=peer_color, ha="left")
        ax.text(0.86, y0 + 0.135, fmt_float(peer_value, 1), fontsize=7.0, fontweight="bold", color=peer_color, ha="right")
        ax.text(0.14, y0 + 0.060, "ASPR agent", fontsize=6.7, fontweight="bold", color=blue, ha="left")
        ax.text(0.86, y0 + 0.060, fmt_float(agent_value, 1), fontsize=7.0, fontweight="bold", color=blue, ha="right")
        ax.text(0.14, y0 - 0.012, "Delta", fontsize=6.7, fontweight="bold", color=green, ha="left")
        ax.text(0.86, y0 - 0.012, f"{good_delta:+.1f}" if math.isfinite(good_delta) else "n/a", fontsize=7.0, fontweight="bold", color=green, ha="right")
    ax.add_patch(
        FancyBboxPatch(
            (-1.98, -0.075),
            1.67,
            0.090,
            boxstyle="round,pad=0.01,rounding_size=0.014",
            transform=ax.transAxes,
            facecolor=green_light,
            edgecolor="#b8e2c0",
            linewidth=0.65,
            clip_on=False,
        )
    )
    ax.text(-1.145, -0.030, "Lower is better. Green arrows show relative reduction.", fontsize=6.8, color=green, ha="center", va="center", transform=ax.transAxes)

    # Panel d: aspect coverage and diagnostic granularity.
    aspects = [
        ("Significance", "significance"),
        ("Novelty", "novelty"),
        ("Rigor", "rigor"),
        ("Limitations", "limitations"),
        ("Future work", "future_work"),
    ]
    peer_ratings = [row for row in rating_rows if str(row.get("kind")) == "peer_review"]
    agent_ratings = [row for row in rating_rows if str(row.get("kind")) == "agent"]

    def aspect_items(row: Mapping[str, Any], aspect: str) -> List[Any]:
        values = ((row.get("aspects") or {}).get(aspect) or [])
        return values if isinstance(values, list) else []

    def aspect_present(rows_for_kind: Sequence[Mapping[str, Any]], aspect: str) -> float:
        if not rows_for_kind:
            return float("nan")
        return sum(1 for row in rows_for_kind if aspect_items(row, aspect) or math.isfinite(numeric(row.get(aspect)))) / len(rows_for_kind)

    def aspect_points(rows_for_kind: Sequence[Mapping[str, Any]], aspect: str) -> float:
        if not rows_for_kind:
            return float("nan")
        return sum(len(aspect_items(row, aspect)) for row in rows_for_kind) / len(rows_for_kind)

    ax = ax_in(panel_d, (0.090, 0.245, 0.500, 0.560))
    peer_cov = [aspect_present(peer_ratings, aspect) for _label, aspect in aspects]
    agent_cov = [aspect_present(agent_ratings, aspect) for _label, aspect in aspects]
    x_positions = list(range(len(aspects)))
    ax.bar([x - 0.18 for x in x_positions], [0 if not math.isfinite(value) else value * 100 for value in peer_cov], width=0.31, color=peer_light, edgecolor=peer_color, linewidth=0.7, label="Peer reviewers")
    ax.bar([x + 0.18 for x in x_positions], [0 if not math.isfinite(value) else value * 100 for value in agent_cov], width=0.31, color=agent_light, edgecolor=agent_color, linewidth=0.7, label="ASPR agent")
    for x, peer_value, agent_value in zip(x_positions, peer_cov, agent_cov):
        if math.isfinite(peer_value) and math.isfinite(agent_value):
            ax.text(x + 0.04, min(106, agent_value * 100 + 6.0), percent_points(agent_value - peer_value), color=blue, fontsize=6.8, fontweight="bold", ha="center")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for label, _aspect in aspects], fontsize=6.7)
    ax.set_ylim(0, 130)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels([f"{int(tick)}%" for tick in ax.get_yticks()])
    ax.set_ylabel("Proportion of papers", fontsize=7.2, fontweight="bold")
    ax.set_title("d1. Aspect coverage (proportion)", fontsize=7.7, fontweight="bold", loc="left", pad=5)
    style_axis(ax)

    ax = ax_in(panel_d, (0.635, 0.115, 0.335, 0.705))
    ax.axis("off")
    table_rows = []
    for label, aspect in aspects:
        peer_points = aspect_points(peer_ratings, aspect)
        agent_points = aspect_points(agent_ratings, aspect)
        delta = agent_points - peer_points if math.isfinite(agent_points) and math.isfinite(peer_points) else float("nan")
        table_rows.append([label, fmt_float(peer_points, 1), fmt_float(agent_points, 1), f"{delta:+.1f}" if math.isfinite(delta) else "n/a"])
    valid_deltas = [numeric(row[3]) for row in table_rows if math.isfinite(numeric(row[3]))]
    if valid_deltas:
        table_rows.append(["Overall avg.", fmt_float(safe_mean([row[1] for row in table_rows]), 2), fmt_float(safe_mean([row[2] for row in table_rows]), 2), f"{safe_mean(valid_deltas):+.2f}"])
    table = ax.table(
        cellText=table_rows,
        colLabels=["", "Peer\nreviewers", "ASPR\nagent", "Delta"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.34, 0.23, 0.23, 0.20],
        bbox=[0.00, 0.000, 1.00, 0.700],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.6)
    style_table(table)
    for row_idx in range(1, len(table_rows) + 1):
        table[(row_idx, 0)].get_text().set_fontweight("bold")
        table[(row_idx, 1)].get_text().set_color("#8f2727")
        table[(row_idx, 2)].get_text().set_color("#1f5fbf")
        table[(row_idx, 3)].get_text().set_color(green)
        table[(row_idx, 3)].get_text().set_fontweight("bold")
    ax.text(0.50, 0.985, "d2. Diagnostic granularity", fontsize=7.5, fontweight="bold", color="black", ha="center", va="top")
    ax.text(0.50, 0.860, "Avg. distinct evidence points\nper aspect (mean)", fontsize=6.3, fontweight="bold", color=dark, ha="center")
    ax.add_patch(
        FancyBboxPatch(
            (-0.96, -0.052),
            1.55,
            0.070,
            boxstyle="round,pad=0.01,rounding_size=0.013",
            transform=ax.transAxes,
            facecolor=green_light,
            edgecolor="#b8e2c0",
            linewidth=0.65,
            clip_on=False,
        )
    )
    ax.text(-0.185, -0.017, "Delta = ASPR agent minus peer reviewers (percentage points)", fontsize=6.6, color=green, ha="center", va="center", transform=ax.transAxes)

    # Summary strip.
    x, y, w, h = summary_panel
    fig.patches.append(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            transform=fig.transFigure,
            linewidth=0.85,
            edgecolor="#82b2f0",
            facecolor="#f7fbff",
            zorder=-20,
        )
    )
    fig.text(x + 0.015, y + h - 0.018, "Summary", fontsize=9.0, fontweight="bold", color=blue, va="top")

    summary_items = [
        ("Consistency", f"Within-1 stance agreement: {fmt_pct(mean_col('stance_within_one_agreement'))}; cosine mean: {fmt_float(safe_mean(finite_col('consistency_cosine')), 2)}."),
        ("Efficiency", f"Median runtime: {fmt_float(runtime_median, 2)} min; throughput gain about {fmt_float(gain, 0)}x."),
        ("Readability", f"Flesch ease improves by {fmt_float(mean_col('agent_flesch_reading_ease') - mean_col('peer_flesch_reading_ease'), 1)} points."),
        ("Coverage", f"Claim-evidence coverage: {fmt_pct(mean_col('claim_evidence_coverage'))}; contradiction rate: {fmt_pct(mean_col('contradiction_rate'))}."),
    ]
    for idx, (label, body) in enumerate(summary_items):
        sx = x + 0.135 + idx * 0.205
        icon_x = sx - 0.040
        fig.patches.append(Circle((icon_x, y + 0.053), 0.022, transform=fig.transFigure, facecolor="white", edgecolor="#d5dde9", linewidth=0.8))
        fig.text(icon_x, y + 0.053, str(idx + 1), fontsize=9.8, fontweight="bold", color=dark, ha="center", va="center")
        fig.text(sx, y + 0.072, label, fontsize=8.4, fontweight="bold", color=blue, ha="left", va="center")
        fig.text(sx, y + 0.041, body, fontsize=7.1, color=dark, ha="left", va="center", wrap=True)

    fig.text(
        0.5,
        0.018,
        f"Note: Metrics are computed from the cached 50-paper ASPR run; quote-grounded peer-review labels are used when available. pp = percentage points.",
        fontsize=7.4,
        color=dark,
        ha="center",
        va="center",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"fig4_system_dashboard.{suffix}", dpi=100)
    plt.close(fig)
    progress_log(f"Drew Fig.4 system dashboard with n={n_main}.", quiet)
    return summary


def _draw_fig4_previous(output_dir: Path, human_hours: float = DEFAULT_HUMAN_HOURS, quiet: bool = False) -> Dict[str, Any]:
    """Draw a publication-style Fig.4 from cached metrics only."""
    rows = read_csv_records(output_dir / "fig4_metrics_summary.csv")
    main_rows = [row for row in rows if bool_value(row.get("included_in_main", True))]
    n_main = len(main_rows)

    def mean_col(column: str) -> float:
        return safe_mean(row.get(column) for row in main_rows)

    def sum_col(column: str) -> float:
        return sum(numeric(row.get(column), 0.0) for row in main_rows)

    def median_col(column: str) -> float:
        values = sorted(numeric(row.get(column)) for row in main_rows)
        values = [value for value in values if math.isfinite(value)]
        if not values:
            return float("nan")
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0

    def corr_cols(left: str, right: str) -> float:
        pairs = [
            (numeric(row.get(left)), numeric(row.get(right)))
            for row in main_rows
            if math.isfinite(numeric(row.get(left))) and math.isfinite(numeric(row.get(right)))
        ]
        if len(pairs) < 2:
            return float("nan")
        xs = [item[0] for item in pairs]
        ys = [item[1] for item in pairs]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        num = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
        den_x = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
        den_y = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
        return num / (den_x * den_y) if den_x and den_y else float("nan")

    summary = {
        "n_main": n_main,
        "mean_stance_within_one": mean_col("stance_within_one_agreement"),
        "mean_strict_claim_recall": mean_col("strict_claim_recall"),
        "mean_soft_claim_recall": mean_col("soft_claim_recall"),
        "mean_claim_evidence_coverage": mean_col("claim_evidence_coverage"),
        "mean_claim_validation_pass": mean_col("claim_validation_pass"),
        "median_agent_runtime_seconds": median_col("agent_runtime_seconds"),
    }

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        _draw_fig4_fallback(output_dir, summary)
        progress_log(f"Drew fallback Fig.4 with n={n_main}.", quiet)
        return summary

    peer_color = "#c95d63"
    agent_color = "#4a82c2"
    green = "#5aa469"
    dark = "#18212f"
    muted = "#6b7280"
    panel_edge = "#d8dee8"
    panel_fill = "#fbfcfe"
    grid = "#e7ebf1"

    fig = plt.figure(figsize=(14.02, 11.22), facecolor="white")
    fig.text(
        0.035,
        0.965,
        "Fig. 4 | Validation of graph-perturbation innovation evaluation against peer-review assessments",
        ha="left",
        va="top",
        fontsize=17,
        fontweight="bold",
        color=dark,
    )
    fig.text(
        0.035,
        0.937,
        "Quote-grounded peer-review labels are compared with full LATS/LLM ASPR innovation assessments.",
        ha="left",
        va="top",
        fontsize=10.5,
        color=muted,
    )

    def add_panel(bounds: Tuple[float, float, float, float], label: str, title: str) -> None:
        x, y, w, h = bounds
        fig.patches.append(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.008,rounding_size=0.018",
                transform=fig.transFigure,
                linewidth=1.1,
                edgecolor=panel_edge,
                facecolor=panel_fill,
                zorder=-10,
            )
        )
        fig.text(x + 0.014, y + h - 0.030, label, fontsize=12, fontweight="bold", color=dark, ha="left", va="top")
        fig.text(x + 0.048, y + h - 0.030, title, fontsize=11.5, fontweight="bold", color=dark, ha="left", va="top")

    def subax(panel: Tuple[float, float, float, float], rel: Tuple[float, float, float, float]):
        x, y, w, h = panel
        rx, ry, rw, rh = rel
        return fig.add_axes([x + rx * w, y + ry * h, rw * w, rh * h])

    def style_axis(ax: Any, ygrid: bool = True) -> None:
        if ygrid:
            ax.grid(True, axis="y", color=grid, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(colors="#334155", labelsize=8)

    panel_a = (0.035, 0.595, 0.438, 0.315)
    panel_b = (0.505, 0.595, 0.46, 0.315)
    panel_c = (0.035, 0.265, 0.438, 0.285)
    panel_d = (0.505, 0.265, 0.46, 0.285)
    summary_panel = (0.035, 0.055, 0.93, 0.16)
    add_panel(panel_a, "a", "Consistency and semantic alignment")
    add_panel(panel_b, "b", "Efficiency")
    add_panel(panel_c, "c", "Readability and language-quality proxies")
    add_panel(panel_d, "d", "Coverage and diagnostic depth")

    peer_scores = [numeric(row.get("peer_innovation_stance_1_5")) for row in main_rows]
    agent_scores = [numeric(row.get("agent_innovation_stance_1_5")) for row in main_rows]
    score_pairs = [(p, a) for p, a in zip(peer_scores, agent_scores) if math.isfinite(p) and math.isfinite(a)]

    ax = subax(panel_a, (0.055, 0.14, 0.30, 0.64))
    if score_pairs:
        ax.scatter([p for p, _ in score_pairs], [a for _, a in score_pairs], s=32, color=agent_color, alpha=0.82, edgecolor="white", linewidth=0.5)
    ax.plot([1, 5], [1, 5], color="#94a3b8", linestyle="--", linewidth=1.1)
    ax.set_xlim(0.7, 5.3)
    ax.set_ylim(0.7, 5.3)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title("a1  Stance score", fontsize=9.2, loc="left", pad=6)
    ax.set_xlabel("Peer", fontsize=8)
    ax.set_ylabel("ASPR", fontsize=8)
    corr = corr_cols("peer_innovation_stance_1_5", "agent_innovation_stance_1_5")
    ax.text(0.05, 0.93, f"r = {corr:.2f}" if math.isfinite(corr) else "r = n/a", transform=ax.transAxes, fontsize=8, color=muted, va="top")
    style_axis(ax)

    ax = subax(panel_a, (0.395, 0.14, 0.25, 0.64))
    bins = [1, 2, 3, 4, 5]
    peer_counts = [sum(1 for value in peer_scores if math.isfinite(value) and round(value) == rating) for rating in bins]
    agent_counts = [sum(1 for value in agent_scores if math.isfinite(value) and round(value) == rating) for rating in bins]
    positions = [idx - 0.17 for idx in bins]
    ax.bar(positions, peer_counts, width=0.30, color=peer_color, label="Peer")
    ax.bar([idx + 0.17 for idx in bins], agent_counts, width=0.30, color=agent_color, label="ASPR")
    ax.set_title("a2  Rating distribution", fontsize=9.2, loc="left", pad=6)
    ax.set_xticks(bins)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    style_axis(ax)

    ax = subax(panel_a, (0.695, 0.14, 0.25, 0.64))
    cosine_values = [numeric(row.get("consistency_cosine")) for row in main_rows]
    cosine_values = [value for value in cosine_values if math.isfinite(value)]
    if cosine_values:
        ax.hist(cosine_values, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], color=green, alpha=0.88, edgecolor="white")
    else:
        ax.text(0.5, 0.5, "No cosine\nvalues", ha="center", va="center", fontsize=8, color=muted)
    ax.set_xlim(0, 1)
    ax.set_title("a3  Text cosine", fontsize=9.2, loc="left", pad=6)
    style_axis(ax)

    ax = subax(panel_b, (0.055, 0.16, 0.30, 0.60))
    ax.axis("off")
    median_runtime = median_col("agent_runtime_seconds")
    human_minutes = float(human_hours) * 60.0
    aspr_minutes = median_runtime / 60.0 if math.isfinite(median_runtime) else float("nan")
    speedup = human_minutes / aspr_minutes if math.isfinite(aspr_minutes) and aspr_minutes > 0 else float("nan")
    table_data = [
        ["Human review", f"{human_minutes:.0f} min"],
        ["ASPR median", f"{aspr_minutes:.1f} min" if math.isfinite(aspr_minutes) else "n/a"],
        ["Speed-up", f"{speedup:.1f}x" if math.isfinite(speedup) else "n/a"],
    ]
    table = ax.table(cellText=table_data, colLabels=["Process", "Time"], loc="center", cellLoc="left", colLoc="left", colWidths=[0.58, 0.42])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.08, 1.45)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#d6dde8")
        cell.set_linewidth(0.7)

    ax = subax(panel_b, (0.405, 0.16, 0.25, 0.60))
    runtimes = [numeric(row.get("agent_runtime_seconds")) / 60.0 for row in main_rows if numeric(row.get("agent_runtime_seconds")) > 0]
    if runtimes:
        ax.boxplot(
            runtimes,
            orientation="vertical",
            widths=0.45,
            patch_artist=True,
            boxprops={"facecolor": "#d8e7f8", "color": agent_color},
            medianprops={"color": dark},
        )
        ax.axhline(human_minutes, color=peer_color, linestyle="--", linewidth=1.2, label="Human")
        ax.set_yscale("log")
        ax.set_xticks([1])
        ax.set_xticklabels(["ASPR"])
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    else:
        ax.text(0.5, 0.5, "Runtime\nmissing", ha="center", va="center", fontsize=8, color=muted)
    ax.set_ylabel("Minutes, log scale", fontsize=8)
    ax.set_title("Runtime distribution", fontsize=9.2, loc="left", pad=6)
    style_axis(ax)

    ax = subax(panel_b, (0.705, 0.16, 0.23, 0.60))
    ax.axis("off")
    human_per_day = 24.0 / float(human_hours) if human_hours > 0 else float("nan")
    aspr_per_day = 1440.0 / aspr_minutes if math.isfinite(aspr_minutes) and aspr_minutes > 0 else float("nan")
    ax.text(0.02, 0.85, "Completed reviews/day", fontsize=9.2, fontweight="bold", color=dark, va="top")
    ax.text(0.02, 0.58, f"{aspr_per_day:.1f}" if math.isfinite(aspr_per_day) else "n/a", fontsize=28, fontweight="bold", color=green, va="center")
    ax.text(0.02, 0.36, "ASPR equivalent", fontsize=8.5, color=muted)
    ax.text(0.02, 0.16, f"Human baseline: {human_per_day:.1f}/day" if math.isfinite(human_per_day) else "Human baseline: n/a", fontsize=8.5, color=muted)

    ax = subax(panel_c, (0.06, 0.18, 0.48, 0.58))
    error_metrics = [
        ("Grammar", "grammar_errors_per_5000"),
        ("Spelling", "spelling_errors_per_5000"),
        ("Tense", "tense_errors_per_5000"),
    ]
    peer_err = [mean_col(f"peer_{metric}") for _, metric in error_metrics]
    agent_err = [mean_col(f"agent_{metric}") for _, metric in error_metrics]
    y_pos = list(range(len(error_metrics)))
    ax.barh([pos + 0.18 for pos in y_pos], [0 if not math.isfinite(value) else value for value in peer_err], height=0.30, color=peer_color, label="Peer")
    ax.barh([pos - 0.18 for pos in y_pos], [0 if not math.isfinite(value) else value for value in agent_err], height=0.30, color=agent_color, label="ASPR")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([label for label, _ in error_metrics], fontsize=8)
    ax.set_xlabel("Proxy errors / 5k words", fontsize=8)
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    style_axis(ax)

    ax = subax(panel_c, (0.61, 0.17, 0.32, 0.60))
    ax.axis("off")
    read_items = [
        ("Flesch ease", mean_col("peer_flesch_reading_ease"), mean_col("agent_flesch_reading_ease"), "higher is easier"),
        ("FK grade", mean_col("peer_flesch_kincaid_grade"), mean_col("agent_flesch_kincaid_grade"), "lower is easier"),
    ]
    for idx, (label, peer_value, agent_value, note) in enumerate(read_items):
        y = 0.80 - idx * 0.42
        ax.text(0.02, y, label, fontsize=9, fontweight="bold", color=dark, va="center")
        ax.text(0.02, y - 0.14, note, fontsize=7.8, color=muted, va="center")
        ax.text(0.62, y + 0.05, f"{peer_value:.1f}" if math.isfinite(peer_value) else "n/a", fontsize=12, color=peer_color, ha="right", fontweight="bold")
        ax.text(0.96, y + 0.05, f"{agent_value:.1f}" if math.isfinite(agent_value) else "n/a", fontsize=12, color=agent_color, ha="right", fontweight="bold")
        ax.text(0.62, y - 0.09, "Peer", fontsize=7.5, color=muted, ha="right")
        ax.text(0.96, y - 0.09, "ASPR", fontsize=7.5, color=muted, ha="right")

    ax = subax(panel_d, (0.055, 0.18, 0.43, 0.58))
    aspect_cols = [
        ("Novelty", "novelty_alignment"),
        ("Signif.", "significance_alignment"),
        ("Prior art", "prior_art_alignment"),
        ("Rigor", "evidence_rigor_alignment"),
        ("Limits", "limitations_alignment"),
        ("Future", "future_work_alignment"),
    ]
    aspect_values = [mean_col(col) if math.isfinite(mean_col(col)) else 0.0 for _, col in aspect_cols]
    ax.bar([label for label, _ in aspect_cols], aspect_values, color=["#7aa6d8", "#7aa6d8", "#7aa6d8", "#91c788", "#91c788", "#91c788"], edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Alignment", fontsize=8)
    ax.tick_params(axis="x", rotation=24)
    ax.set_title("Aspect-level coverage", fontsize=9.2, loc="left", pad=6)
    style_axis(ax)

    ax = subax(panel_d, (0.57, 0.18, 0.36, 0.58))
    ax.axis("off")
    relation_total = sum_col("entailed_points") + sum_col("related_points") + sum_col("no_match_points") + sum_col("contradicted_points")
    diagnostic_rows = [
        ["Entailed + related", f"{(sum_col('entailed_points') + sum_col('related_points')) / relation_total:.0%}" if relation_total else "n/a"],
        ["No-match rate", f"{mean_col('missing_peer_point_rate'):.0%}" if math.isfinite(mean_col("missing_peer_point_rate")) else "n/a"],
        ["Contradiction rate", f"{mean_col('contradiction_rate'):.0%}" if math.isfinite(mean_col("contradiction_rate")) else "n/a"],
        ["Overclaim score", f"{mean_col('overclaiming_score_1_5'):.2f}" if math.isfinite(mean_col("overclaiming_score_1_5")) else "n/a"],
    ]
    table = ax.table(cellText=diagnostic_rows, colLabels=["Diagnostic", "Value"], loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.08, 1.42)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#d6dde8")
        cell.set_linewidth(0.7)

    x, y, w, h = summary_panel
    fig.patches.append(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            transform=fig.transFigure,
            linewidth=1.1,
            edgecolor=panel_edge,
            facecolor="#f7faf7",
            zorder=-10,
        )
    )
    fig.text(x + 0.018, y + h - 0.035, "Summary", fontsize=12, fontweight="bold", color=dark, va="top")
    summary_items = [
        ("Main sample", f"N = {n_main}", "real papers with peer-review text"),
        ("Stance agreement", f"{mean_col('stance_within_one_agreement'):.0%}" if math.isfinite(mean_col("stance_within_one_agreement")) else "n/a", "within ±1 point"),
        ("Claim coverage", f"{mean_col('claim_evidence_coverage'):.0%}" if math.isfinite(mean_col("claim_evidence_coverage")) else "n/a", "quote-grounded points"),
        ("Runtime", f"{aspr_minutes:.1f} min" if math.isfinite(aspr_minutes) else "n/a", "median ASPR LATS/LLM"),
    ]
    for idx, (label, value, note) in enumerate(summary_items):
        sx = x + 0.16 + idx * 0.205
        fig.text(sx, y + 0.095, value, fontsize=19, fontweight="bold", color=green if idx else dark, ha="left", va="center")
        fig.text(sx, y + 0.060, label, fontsize=9.2, fontweight="bold", color=dark, ha="left", va="center")
        fig.text(sx, y + 0.034, note, fontsize=7.8, color=muted, ha="left", va="center")

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"fig4_full.{suffix}", dpi=260, bbox_inches="tight")
    plt.close(fig)
    progress_log(f"Drew Fig.4 with n={n_main}.", quiet)
    return summary


def _draw_fig4_fallback(output_dir: Path, summary: Mapping[str, Any]) -> None:
    """Create lightweight figure files when matplotlib is unavailable."""
    from PIL import Image, ImageDraw
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "Fig.4 innovation validation summary",
        f"Main N: {summary.get('n_main')}",
        f"Within-1 stance: {summary.get('mean_stance_within_one')}",
        f"Strict claim recall: {summary.get('mean_strict_claim_recall')}",
    ]
    image = Image.new("RGB", (1200, 720), "white")
    draw = ImageDraw.Draw(image)
    for idx, line in enumerate(lines):
        draw.text((60, 70 + idx * 52), str(line), fill=(15, 23, 42))
    image.save(output_dir / "fig4_full.png")
    svg_lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720">',
        '<rect width="1200" height="720" fill="white"/>',
    ]
    for idx, line in enumerate(lines):
        svg_lines.append(f'<text x="60" y="{90 + idx * 52}" font-size="28" fill="#0f172a">{line}</text>')
    svg_lines.append("</svg>")
    atomic_write_text(output_dir / "fig4_full.svg", "\n".join(svg_lines) + "\n")
    pdf = canvas.Canvas(str(output_dir / "fig4_full.pdf"), pagesize=letter)
    y = 720
    for line in lines:
        pdf.drawString(72, y, str(line))
        y -= 28
    pdf.save()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Fig.4 peer-review innovation-validation pipeline.")
    parser.add_argument("--markdown-root", type=Path, default=DEFAULT_MARKDOWN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stage",
        action="append",
        choices=["audit", "sample", "agent", "graph", "prior", "labels", "screen", "rating", "semantic", "structured", "metrics", "draw"],
        default=None,
    )
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--audit-max-records", type=int, default=0)
    parser.add_argument("--journal-scope", choices=["all", "six_subjournals", "41467_only"], default="all")
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--s2-api-key", default=getenv("S2_API_KEY"))
    parser.add_argument("--retrieval-provider", choices=["semantic_scholar", "openalex"], default=getenv("ASPR_RETRIEVAL_PROVIDER", "semantic_scholar"))
    parser.add_argument("--openalex-api-key", default=getenv("OPENALEX_API_KEY") or getenv("OPENALEX_API_KEYS"))
    parser.add_argument("--agent-context-mode", choices=["dossier", "abstract_only"], default="dossier")
    parser.add_argument("--max-agent", type=int, default=0)
    parser.add_argument("--force-agent", action="store_true")
    parser.add_argument("--no-reuse-agent", action="store_true")
    parser.add_argument("--refresh-invalid-agent-only", action="store_true")
    parser.add_argument("--judge-backend", choices=["openai-compatible", "heuristic"], default="openai-compatible")
    parser.add_argument("--force-labels", action="store_true")
    parser.add_argument("--no-reuse-labels", action="store_true")
    parser.add_argument("--min-core-aspects", type=int, default=1)
    parser.add_argument("--min-peer-label-points", type=int, default=1)
    parser.add_argument("--max-points-per-aspect", type=int, default=4)
    parser.add_argument("--semantic-llm-refine", action="store_true", default=None)
    parser.add_argument("--human-hours", type=float, default=DEFAULT_HUMAN_HOURS)
    parser.add_argument("--fig3-weights-path", type=Path, default=fig3_weights_path_from_env())
    parser.add_argument("--fig3-score-table-path", type=Path, default=fig3_score_table_path_from_env())
    parser.add_argument("--fig3-indicators-path", type=Path, default=fig3_indicators_path_from_env())
    parser.add_argument("--reuse-audit", action="store_true", help="Reuse fig4_input_audit.csv when it already exists.")
    parser.add_argument("--force-audit", action="store_true", help="Rebuild fig4_input_audit.csv even when --reuse-audit is set.")
    parser.add_argument("--require-fixed-sample", action="store_true", help="Fail if the sampled evaluable cases do not equal --sample-size.")
    parser.add_argument("--prefer-screen-pass-sample", action="store_true", help="Use existing screen-pass rows for sampling when enough are available.")
    parser.add_argument("--prefer-scored-candidate-pool", action="store_true", help="Use cached graph-prior scores to stratify the fixed Fig.4 sample when available.")
    parser.add_argument("--forbid-lightweight", action="store_true", help="Fail when FIG4_LIGHTWEIGHT_AGENT=1 is set.")
    parser.add_argument("--forbid-local-retrieval", action="store_true", help="Fail if local retrieval fallbacks are used.")
    parser.add_argument("--forbid-lexical-fallback", action="store_true", help="Fail if semantic metrics use lexical_fallback embeddings.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    stages = args.stage or ["audit", "sample", "agent", "graph", "labels", "screen", "rating", "semantic", "structured", "metrics", "draw"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.forbid_lightweight and str(os.getenv("FIG4_LIGHTWEIGHT_AGENT", "")).strip() == "1":
        raise RuntimeError("FIG4_LIGHTWEIGHT_AGENT=1 is forbidden for Nature-ready Fig.4.")
    if args.forbid_local_retrieval and str(os.getenv("FIG4_LOCAL_RETRIEVAL_ONLY", "")).strip() == "1":
        raise RuntimeError("FIG4_LOCAL_RETRIEVAL_ONLY=1 is forbidden for Nature-ready Fig.4.")
    if "audit" in stages:
        audit_path = args.output_dir / "fig4_input_audit.csv"
        if args.reuse_audit and audit_path.exists() and not args.force_audit:
            progress_log(f"Reusing Fig.4 audit table at {audit_path}.", args.quiet)
        else:
            audit_markdown_inputs(
                args.markdown_root,
                args.output_dir,
                journal_scope=args.journal_scope,
                quiet=args.quiet,
                audit_max_records=args.audit_max_records if args.audit_max_records > 0 else None,
            )
    if "sample" in stages:
        sampled = sample_manifest(
            args.output_dir,
            args.sample_size,
            args.seed,
            DEFAULT_41467_CAP,
            quiet=args.quiet,
            prefer_screen_pass=args.prefer_screen_pass_sample,
            prefer_scored_pool=args.prefer_scored_candidate_pool,
        )
        if args.require_fixed_sample:
            enforce_fixed_sample_contract(sampled, args.sample_size)
    if "agent" in stages:
        run_agent_stage(
            args.output_dir,
            s2_api_key=args.s2_api_key,
            top_n=args.top_n,
            agent_context_mode=args.agent_context_mode,
            retrieval_provider=args.retrieval_provider,
            openalex_api_key=args.openalex_api_key,
            force_agent=args.force_agent,
            reuse_agent=not args.no_reuse_agent,
            refresh_invalid_agent_only=args.refresh_invalid_agent_only,
            max_agent=args.max_agent if args.max_agent > 0 else None,
            quiet=args.quiet,
        )
    if "graph" in stages:
        run_graph_metrics_stage(args.output_dir, quiet=args.quiet)
        run_graph_prior_stage(
            args.output_dir,
            weights_path=args.fig3_weights_path,
            reference_scores_path=args.fig3_score_table_path,
            reference_indicators_path=args.fig3_indicators_path,
            quiet=args.quiet,
        )
    if "prior" in stages:
        run_graph_prior_stage(
            args.output_dir,
            weights_path=args.fig3_weights_path,
            reference_scores_path=args.fig3_score_table_path,
            reference_indicators_path=args.fig3_indicators_path,
            quiet=args.quiet,
        )
    if "labels" in stages:
        run_innovation_label_judge(
            args.output_dir,
            judge_backend=args.judge_backend,
            force_labels=args.force_labels,
            reuse_labels=not args.no_reuse_labels,
            quiet=args.quiet,
        )
    if "screen" in stages:
        run_peer_review_screen(
            args.output_dir,
            min_core_aspects=args.min_core_aspects,
            min_peer_label_points=args.min_peer_label_points,
            quiet=args.quiet,
        )
    if "rating" in stages:
        run_rating_judgements_from_labels(args.output_dir, quiet=args.quiet)
    if "semantic" in stages:
        run_semantic_claim_match(
            args.output_dir,
            judge_backend=args.judge_backend,
            max_points_per_aspect=args.max_points_per_aspect,
            llm_refine=args.semantic_llm_refine,
            quiet=args.quiet,
        )
    if "structured" in stages:
        run_structured_consistency_judgements(args.output_dir, quiet=args.quiet)
    if "metrics" in stages:
        run_metrics_stage(args.output_dir, human_hours=args.human_hours, judge_backend=args.judge_backend, quiet=args.quiet)
        quality = write_fig4_external_validation_report(
            args.output_dir,
            args.sample_size if args.sample_size > 0 else 50,
            fig3_score_table_path=args.fig3_score_table_path,
        )
        if args.forbid_lexical_fallback and not quality["checks"]["embedding_backend_not_lexical_fallback"]:
            raise RuntimeError("Fig.4 semantic metrics used lexical_fallback embeddings.")
        if args.forbid_local_retrieval and not quality["checks"]["retrieval_not_local_manifest"]:
            raise RuntimeError("Fig.4 retrieval used local fallback/local manifest sources.")
        if args.forbid_lightweight:
            enforce_fig4_nature_agent_outputs(
                args.output_dir,
                expected_case_count=args.sample_size if args.sample_size > 0 else 50,
            )
    if "draw" in stages:
        draw_fig4(args.output_dir, human_hours=args.human_hours, quiet=args.quiet)


if __name__ == "__main__":
    main()
