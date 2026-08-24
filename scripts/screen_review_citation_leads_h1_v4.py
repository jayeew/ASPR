"""Create the blinded H1 screen for review-citation leads in v4."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = (
    Path(__file__).resolve().parents[1]
    / "innovation_impact_feature_selection"
    / "evidence_derived_v4_rebuild"
)
INPUT_PATH = ROOT / "outputs" / "review_citation_lead_screening_input_v4.csv"
OUTPUT_PATH = ROOT / "outputs" / "review_citation_lead_screening_H1_completed_v4.csv"
MANIFEST_PATH = (
    ROOT / "outputs" / "review_citation_lead_screening_H1_completed_v4.manifest.json"
)
EDITABLE_FIELDS = {"screen_decision", "evidence_span", "rationale"}


REVIEWS: dict[int, tuple[str, str, str]] = {
    1: (
        "include_definition_or_review",
        "A new form of document coupling called co-citation is defined",
        "Retain for full-text recovery: this is an original definition of a document-level citation-network relation that can underpin paper-level knowledge-position or novelty constructs at publication time.",
    ),
    2: (
        "include_definition_or_review",
        "Bibliographic coupling between scientific papers",
        "Retain for full-text recovery: the foundational paper defines a relation among scientific papers based on their references, a valid publication-time input to paper-level novelty or knowledge-position constructs.",
    ),
    3: (
        "exclude_not_relevant",
        "a rhetorical move analysis of the research abstracts",
        "Exclude: it analyzes journal-abstract rhetoric within a field but the available metadata does not define, apply, or validate a paper-level innovation or publication-time impact construct.",
    ),
    4: (
        "include_definition_or_review",
        "Diversity and dissimilarity coefficients: A unified approach",
        "Retain for full-text recovery: this foundational definition of diversity/dissimilarity coefficients may supply an original mathematical basis for publication-time reference or topic diversity constructs.",
    ),
    5: (
        "exclude_not_relevant",
        "a corpus study of evaluative that in abstracts",
        "Exclude: it is a linguistic corpus study of evaluative phrasing and the available metadata gives no paper-level innovation or publication-time impact definition, application, or validation.",
    ),
    6: (
        "include_definition_or_review",
        "Free online availability substantially increases a paper's impact",
        "Retain for full-text recovery: it directly evaluates online availability as a publication-time condition associated with a paper's later impact.",
    ),
    7: (
        "include_definition_or_review",
        "To measure the effect of free access to the scientific literature on article downloads and citations.",
        "Retain for full-text recovery: a randomized article-level application tests open access at publication against readership and citations, directly relevant to defining or validating a T0 availability construct.",
    ),
    8: (
        "uncertain",
        "whether language used in science abstracts can skew towards the use of strikingly positive and negative words over time",
        "Uncertain: it supplies a possible publication-time abstract-language measurement, but the metadata does not establish that the measure defines, predicts, or validates paper-level innovation or impact.",
    ),
    9: (
        "include_definition_or_review",
        "An introduction to co-word analysis",
        "Retain for full-text recovery: it is an original co-word-analysis source that may define a text-derived publication-time knowledge-position or novelty construct.",
    ),
    10: (
        "include_definition_or_review",
        "The Impact of Article Titles on Citation Hits",
        "Retain for full-text recovery: it applies paper-title features as publication-time predictors of article citation, directly relevant to a T0 textual-impact construct.",
    ),
    11: (
        "include_definition_or_review",
        "Citation Advantage of Open Access Articles",
        "Retain for full-text recovery: it applies and adjusts an article-level open-access indicator when estimating early citation differences, a direct validation lead for T0 access features.",
    ),
    12: (
        "uncertain",
        "a generative probabilistic model for collections of discrete data such as text corpora",
        "Uncertain: LDA is a generic document-text representation method and may support later feature construction, but the metadata alone does not connect it to a paper-level innovation or impact construct.",
    ),
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read all CSV rows and preserve the input schema."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def validate_rows(
    original: list[dict[str, str]],
    completed: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    """Verify decision values and preserve every non-review field."""
    assert len(original) == len(completed) == 12
    assert set(REVIEWS) == set(range(1, 13))
    allowed = {"include_definition_or_review", "exclude_not_relevant", "uncertain"}
    for before, after in zip(original, completed, strict=True):
        for field in fieldnames:
            if field not in EDITABLE_FIELDS:
                assert before[field] == after[field], field
        assert after["screen_decision"] in allowed
        assert after["evidence_span"]
        assert after["rationale"]


def write_rows(rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write the completed review CSV without adding or dropping fields."""
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Apply the independent H1 screening decisions and produce an audit manifest."""
    original, fieldnames = read_rows(INPUT_PATH)
    assert EDITABLE_FIELDS.issubset(fieldnames)
    completed = [dict(row) for row in original]
    for index, row in enumerate(completed, start=1):
        decision, evidence_span, rationale = REVIEWS[index]
        row["screen_decision"] = decision
        row["evidence_span"] = evidence_span
        row["rationale"] = rationale
    validate_rows(original, completed, fieldnames)
    write_rows(completed, fieldnames)
    counts = Counter(row["screen_decision"] for row in completed)
    manifest: dict[str, Any] = {
        "schema": "review_citation_lead_screening_h1_manifest_v4",
        "reviewer": "H1",
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "output_sha256": sha256(OUTPUT_PATH),
        "source_count": len(completed),
        "screen_decision_counts": dict(sorted(counts.items())),
        "qwen_or_ollama_used": False,
        "read_ai_or_h2_or_later_results": False,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
