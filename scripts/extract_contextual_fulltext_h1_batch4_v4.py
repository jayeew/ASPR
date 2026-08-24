"""Create blinded H1 source reviews and T0-safe indicator mentions for batch 4."""

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
INPUT_PATH = ROOT / "outputs" / "contextual_fulltext_extraction_input_batch4_v4.csv"
SOURCE_REVIEW_PATH = (
    ROOT / "outputs" / "contextual_fulltext_source_review_H1_batch4_v4.csv"
)
MENTIONS_PATH = (
    ROOT / "outputs" / "contextual_fulltext_indicator_mentions_H1_batch4_v4.csv"
)
MANIFEST_PATH = (
    ROOT / "outputs" / "contextual_fulltext_extraction_H1_batch4_v4.manifest.json"
)
EDITABLE_SOURCE_FIELDS = {"source_disposition", "source_notes"}
MENTION_FIELDS = [
    "record_key",
    "raw_name_en",
    "canonical_name_en",
    "source_role",
    "formula_location",
    "evidence_span",
    "formula",
    "parameters",
    "required_data",
    "maximum_information_time",
    "scope_role",
    "requires_future",
    "extraction_notes",
]


REVIEWS: dict[str, tuple[str, str]] = {
    "doi:10.1007/s10734-016-9995-x": (
        "review_discovery_only",
        (
            "Broad discussion of research-impact measurement, effects, and limitations. It does not authorise "
            "a specific paper-level T0 formula or application in the source text; citations are discovery leads only."
        ),
    ),
    "doi:10.1007/s11024-015-9274-5": (
        "no_relevant_indicator",
        (
            "Qualitative case study of Journal Impact Factor use in biomedical evaluation. The discussed JIF is "
            "journal-level and the source supplies no paper-level T0 indicator formula or application."
        ),
    ),
    "doi:10.1007/s11192-016-2111-2": (
        "review_discovery_only",
        (
            "Literature overview of cited-reference evaluation approaches. It is useful for terminology and original "
            "sources, but it supplies no source-authorised paper-level T0 formula/application for extraction."
        ),
    ),
    "doi:10.1007/s11192-017-2528-2": (
        "review_discovery_only",
        (
            "The empirical focus is journal citation networks and journal-level interdisciplinarity. Its discussion of "
            "citing-document diversity is contextual only; no paper-level T0 formula/application is authorised here."
        ),
    ),
    "doi:10.1007/s11192-023-04822-2": (
        "formula_or_application",
        (
            "Explicitly applies the Rao-Stirling reference-discipline index to individual papers and proposes an "
            "RS uncertainty estimate. Both are T0-computable only when citation-category mappings and similarity "
            "data are frozen using information available no later than each focal paper's publication time."
        ),
    ),
    "doi:10.1007/s11192-025-05234-0": (
        "review_discovery_only",
        (
            "Systematic review of novelty measures. It restates and compares methods but, under the brief, cannot be "
            "the sole authority for an extracted formula; cited original definition/application sources are discovery leads."
        ),
    ),
    "doi:10.1016/j.joi.2024.101546": (
        "review_discovery_only",
        (
            "Narrative bibliometrics discussion of responsible evaluation and indicator contexts. It provides discovery "
            "terminology but no explicit paper-level T0 formula or application authorised by the source."
        ),
    ),
}


MENTIONS: list[dict[str, str]] = [
    {
        "record_key": "doi:10.1007/s11192-023-04822-2",
        "raw_name_en": "Rao-Stirling diversity index (RS index)",
        "canonical_name_en": "rao_stirling_reference_diversity",
        "source_role": "validation",
        "formula_location": "Data and method, p. 6112, equation immediately after ‘It is defined as:’",
        "evidence_span": "The Rao-Stirling diversity index (RS index), as applied to the disciplines of a paper’s cited references",
        "formula": "RS_index = 1 - sum_{i,j}(S_ij * P_i * P_j)",
        "parameters": "P_i: proportion of a focal paper's references citing Subject Category i; S_ij: pairwise category similarity. The source uses the Rafols et al. cosine-similarity matrix updated with 2015 publication data.",
        "required_data": "Focal paper cited references; journal-to-Web of Science Subject Category mapping; pairwise Subject Category similarity matrix.",
        "maximum_information_time": "T0 only if cited-reference records, category mapping, and S_ij are frozen using information available no later than the focal paper publication date.",
        "scope_role": "control",
        "requires_future": "false",
        "extraction_notes": "Paper-level application/validation of a pre-existing index. Do not reuse the source's 2015-updated similarity matrix for earlier focal papers; construct or freeze a pre-T0 matrix instead.",
    },
    {
        "record_key": "doi:10.1007/s11192-023-04822-2",
        "raw_name_en": "RS uncertainty estimate (RSunc)",
        "canonical_name_en": "rao_stirling_bootstrap_uncertainty",
        "source_role": "original_application",
        "formula_location": "A more effective uncertainty estimate, p. 6115, equation immediately after ‘as follows:’",
        "evidence_span": "We propose that a more effective uncertainty estimate combines a relative size of |RSci| and the log of the number of references",
        "formula": "RS_unc = |RS_ci| / (RS + 1) + 1 / log(|R_i| + 1), where |RS_ci| = RS_uci - RS_lci",
        "parameters": "RS: Rao-Stirling index; RS_lci and RS_uci: lower and upper bootstrap confidence bounds; |R_i|: number of categorized-journal references. The source uses 500 bootstrap replicates and a 95% bias-corrected confidence interval.",
        "required_data": "All inputs for the Rao-Stirling index plus focal-paper recognized Subject Categories; bootstrap resamples of size N equal to the number of recognized categories/references.",
        "maximum_information_time": "T0 only if all reference categories and the similarity matrix used to calculate each bootstrap RS value are frozen using information available no later than publication.",
        "scope_role": "control",
        "requires_future": "false",
        "extraction_notes": "Paper-level measurement-quality control. The source suggests RSunc <= 0.5 as a middle-ground inclusion threshold; the threshold is contextual rather than a universal feature rule.",
    },
]


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read input rows and preserve their field order."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def validate_text_hashes(rows: list[dict[str, str]]) -> None:
    """Verify each independently reviewed local English full text."""
    for row in rows:
        assert sha256(Path(row["text_path"])) == row["text_sha256"], row["record_key"]


def validate_source_reviews(
    source: list[dict[str, str]], completed: list[dict[str, str]], fieldnames: list[str]
) -> None:
    """Ensure source-review output changes only the two designated fields."""
    allowed = {
        "formula_or_application",
        "review_discovery_only",
        "no_relevant_indicator",
    }
    assert len(source) == len(completed) == 7
    assert set(REVIEWS) == {row["record_key"] for row in source}
    for before, after in zip(source, completed, strict=True):
        for field in fieldnames:
            if field not in EDITABLE_SOURCE_FIELDS:
                assert before[field] == after[field], field
        assert after["source_disposition"] in allowed
        assert after["source_notes"]


def validate_mentions(mentions: list[dict[str, str]], source_keys: set[str]) -> None:
    """Check the mention contract and T0/future guardrail."""
    allowed_roles = {
        "original_definition",
        "original_application",
        "validation",
        "review_discovery",
        "mathematical_foundation",
    }
    assert len(mentions) == 2
    for mention in mentions:
        assert list(mention) == MENTION_FIELDS
        assert mention["record_key"] in source_keys
        assert mention["source_role"] in allowed_roles
        assert mention["requires_future"] == "false"
        assert mention["formula"] and mention["evidence_span"]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write a UTF-8 CSV with fixed schema."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Create the independent H1 review, mentions, and audit manifest."""
    source, fieldnames = read_rows(INPUT_PATH)
    assert EDITABLE_SOURCE_FIELDS.issubset(fieldnames)
    validate_text_hashes(source)
    completed = [dict(row) for row in source]
    for row in completed:
        disposition, notes = REVIEWS[row["record_key"]]
        row["source_disposition"] = disposition
        row["source_notes"] = notes
    validate_source_reviews(source, completed, fieldnames)
    validate_mentions(MENTIONS, {row["record_key"] for row in source})
    write_csv(SOURCE_REVIEW_PATH, completed, fieldnames)
    write_csv(MENTIONS_PATH, MENTIONS, MENTION_FIELDS)
    manifest: dict[str, Any] = {
        "schema": "contextual_fulltext_extraction_h1_batch4_manifest_v4",
        "reviewer": "H1",
        "input_path": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "source_review_path": str(SOURCE_REVIEW_PATH),
        "source_review_sha256": sha256(SOURCE_REVIEW_PATH),
        "indicator_mentions_path": str(MENTIONS_PATH),
        "indicator_mentions_sha256": sha256(MENTIONS_PATH),
        "source_count": len(completed),
        "indicator_mention_count": len(MENTIONS),
        "source_disposition_counts": dict(
            sorted(Counter(row["source_disposition"] for row in completed).items())
        ),
        "text_sha256_verified": {
            row["record_key"]: row["text_sha256"] for row in source
        },
        "qwen_or_ollama_used": False,
        "read_ai_or_h2_or_other_batch4_outputs": False,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
