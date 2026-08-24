"""Write the independent English title-and-abstract H1 screen for batch 11."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Final

ROOT: Final = Path("/home/jayee/workspace/ASPR")
INPUT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_source_screening_input_batch11_v4.csv"
)
OUTPUT_PATH: Final = (
    ROOT / "outputs/contextual_source_screening_H1_batch11_completed_v4.csv"
)
MANIFEST_PATH: Final = (
    ROOT / "outputs/contextual_source_screening_H1_batch11_completed_v4.manifest.json"
)
H1_FIELDS: Final = ["H1_decision", "H1_rationale"]

INCLUDE_IDS: Final = {
    41,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    63,
    68,
    69,
    74,
    94,
    97,
    100,
    104,
    111,
    117,
    119,
}
UNCERTAIN_IDS: Final = {57, 72, 77, 89}

INCLUDE_RATIONALES: Final = {
    41: (
        "Applies publication counts and scientific impact to assess the relationship "
        "between doctoral funding and publication outcomes."
    ),
    46: (
        "Validates named paper-level disruption-index variants against expert "
        "milestone assignments for physics papers."
    ),
    47: (
        "Provides a metric-focused review of science-of-science measures and their "
        "links to underlying mechanisms."
    ),
    48: (
        "Describes policy-document citation data and evaluates field-normalized "
        "citation metrics against research-impact assessment."
    ),
    49: (
        "Proposes and evaluates Article Scientific Prestige, an eigenvector-based "
        "metric for the impact of individual articles."
    ),
    50: (
        "Applies article-level citation impact, disruptiveness, novelty, and "
        "atypicality measures to characterize academic and industry AI research."
    ),
    51: (
        "Operationalizes and analyzes scientific novelty using a large corpus of "
        "doctoral research documents."
    ),
    52: (
        "Measures paper-level scientific contributions from citation contexts and "
        "tests their relationship with division-of-labor inputs."
    ),
    53: (
        "Applies paper-level innovation measures, including novel idea combinations "
        "and interdisciplinarity, to prizewinner research records."
    ),
    54: (
        "Tests citation, novelty, and interdisciplinarity characteristics of papers "
        "that receive explicit critical letters."
    ),
    63: (
        "Reviews the construction, limitations, and uses of impact factor and other "
        "citation metrics in evaluating publication quality."
    ),
    68: (
        "Reviews publication metrics and alternative metrics for scholarly productivity "
        "and publication strategy."
    ),
    69: (
        "Treats citations as reusable scholarly data objects for quantifying citation "
        "records and institutional research output."
    ),
    74: "Directly summarizes limitations of citation analysis for measuring research impact.",
    94: (
        "Develops sentence-context identification for citations, directly supporting "
        "citation-context information and content analysis."
    ),
    97: (
        "Reviews conceptual, qualitative, and quantitative approaches to societal "
        "impact of humanities scholarship and limitations of bibliometric-only measures."
    ),
    100: (
        "Develops NLP-driven methods that forecast future impact of scientific "
        "publications from document features."
    ),
    104: (
        "Defines and validates an eight-category scheme for citation function and "
        "polarity in biomedical papers."
    ),
    111: (
        "Experiments with a content-distribution channel and measures paper reach, "
        "usage, and scholarly impact."
    ),
    117: (
        "Tests common abstract-readability formulas against online attention for "
        "research articles."
    ),
    119: (
        "Directly examines document-level assessment of contributions to Sustainable "
        "Development Goals and its interpretation-dependent measurement boundary."
    ),
}
UNCERTAIN_RATIONALES: Final = {
    57: "The title suggests publication-system diversity, but the supplied record contains no abstract or operational detail to establish a paper-level indicator.",
    72: "The title indicates a bibliometrics-to-altmetrics review, but the supplied abstract is absent and no paper-level measure can be verified from permitted evidence.",
    77: "The record is a grant proposal concerning altmetrics rather than a completed definition, application, validation, or review with verifiable operational detail.",
    89: "The title concerns factors associated with future article citation impact, but the supplied record provides no abstract or metric definition to verify its scope.",
}
EXCLUDE_RATIONALE: Final = (
    "Title and abstract describe a substantive-domain study, organizational or journal-level "
    "evaluation, generic field mapping, or non-paper-level topic without an evident "
    "paper-level innovation, quality, or T0 potential-impact indicator definition, "
    "application, validation, or review."
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a UTF-8 CSV while retaining its original field order."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    """Write a CSV with a fixed schema."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def decision_for(index: int) -> tuple[str, str]:
    """Return the H1 screen decision and rationale for a batch row number."""
    if index in INCLUDE_IDS:
        return "include_definition_or_review", INCLUDE_RATIONALES[index]
    if index in UNCERTAIN_IDS:
        return "uncertain", UNCERTAIN_RATIONALES[index]
    return "exclude_not_relevant", EXCLUDE_RATIONALE


def main() -> None:
    """Create the frozen-column-preserving batch-11 H1 screen."""
    frozen_columns, input_rows = read_csv(INPUT_PATH)
    if len(input_rows) != 120:
        raise ValueError("Expected exactly 120 batch-11 source rows")
    if INCLUDE_IDS & UNCERTAIN_IDS:
        raise ValueError("Screening decision registries overlap")

    output_rows: list[dict[str, str]] = []
    for index, row in enumerate(input_rows, start=1):
        decision, rationale = decision_for(index)
        output_rows.append({**row, "H1_decision": decision, "H1_rationale": rationale})
    write_csv(OUTPUT_PATH, [*frozen_columns, *H1_FIELDS], output_rows)

    counts = {
        decision: sum(row["H1_decision"] == decision for row in output_rows)
        for decision in (
            "include_definition_or_review",
            "exclude_not_relevant",
            "uncertain",
        )
    }
    manifest = {
        "artifact": "contextual_source_screening_H1_batch11_completed_v4",
        "reviewer": "H1",
        "input_path": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "output_sha256": sha256(OUTPUT_PATH),
        "source_rows": len(output_rows),
        "h1_field_count": len(H1_FIELDS),
        "decision_counts": counts,
        "qwen_or_ollama_used": False,
        "blind_review_constraints": [
            "Used only the English title and abstract in the batch-11 input.",
            "Copied frozen input fields verbatim without adjudicating AI/H2 content.",
            "Did not use AI/H2 output, prior-batch output, Qwen, or Ollama.",
            "Included only direct paper-level innovation, quality, or T0 potential-impact indicator definitions, applications, validations, or reviews.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
