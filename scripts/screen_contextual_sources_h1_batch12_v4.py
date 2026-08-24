"""Write the independent English title-and-abstract H1 screen for batch 12."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Final

ROOT: Final = Path("/home/jayee/workspace/ASPR")
INPUT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_source_screening_input_batch12_v4.csv"
)
OUTPUT_PATH: Final = (
    ROOT / "outputs/contextual_source_screening_H1_batch12_completed_v4.csv"
)
MANIFEST_PATH: Final = (
    ROOT / "outputs/contextual_source_screening_H1_batch12_completed_v4.manifest.json"
)
H1_FIELDS: Final = ["H1_decision", "H1_rationale", "H1_evidence_span"]

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
    41: "Applies publication productivity and scientific impact to funded researchers' publication outcomes.",
    46: "Validates named paper-level disruption-index variants against expert milestone assignments.",
    47: "Reviews the construction and use of science metrics in relation to scientific mechanisms.",
    48: "Assesses policy-document citation metrics and their relationship to research-impact evaluation.",
    49: "Proposes Article Scientific Prestige, an eigenvector-based metric for individual-article impact.",
    50: "Applies article citation impact, disruptiveness, novelty, and atypicality measures to AI research.",
    51: "Operationalizes scientific novelty and tests its variation in a large research-document corpus.",
    52: "Measures paper-level scientific contributions from citation contexts and analyzes their determinants.",
    53: "Uses explicit innovation measures, including novel idea combinations and interdisciplinarity, for prizewinner papers.",
    54: "Analyzes citation, novelty, and interdisciplinarity characteristics of papers receiving critical letters.",
    63: "Reviews the construction, limitations, and evaluative uses of impact factor and citation metrics.",
    68: "Reviews scholarly publication metrics and alternative metrics used in publication and career decisions.",
    69: "Defines citations as data objects for quantifying citation records and scholarly output.",
    74: "Directly reviews limitations of citation analysis for measuring research impact.",
    94: "Develops sentence-context identification for citations, directly supporting citation-content indicators.",
    97: "Reviews conceptual, qualitative, and quantitative methods for assessing societal research impact.",
    100: "Develops NLP methods that forecast future impact of scientific publications from document features.",
    104: "Defines and validates an eight-category citation-function and polarity scheme for biomedical papers.",
    111: "Experiments with content distribution and measures paper reach, usage, and scholarly impact.",
    117: "Tests abstract-readability measures against online attention for research articles.",
    119: "Directly examines document-level evaluation of Sustainable Development Goal contributions.",
}
UNCERTAIN_RATIONALES: Final = {
    57: "The title suggests publication-system diversity, but the supplied abstract has no operational detail establishing a paper-level indicator.",
    72: "The title indicates a bibliometrics-to-altmetrics review, but the supplied abstract is absent and no measure can be verified.",
    77: "The record is a grant proposal about altmetrics, not a completed definition, application, validation, or review with operational evidence.",
    89: "The title concerns future article citation impact, but the supplied record has no abstract or metric definition to verify scope.",
}
EXCLUDE_RATIONALE: Final = (
    "Title and abstract concern a substantive-domain study, organizational or journal-level "
    "evaluation, generic field map, or non-paper-level topic without a direct paper-level "
    "innovation, quality, or T0 potential-impact indicator definition, application, "
    "validation, or review."
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


def evidence_span(row: dict[str, str]) -> str:
    """Return a compact verbatim title-and-abstract evidence span."""
    abstract = " ".join(row["abstract"].split())
    if not abstract:
        return f"Title: {row['title']}"
    sentence = abstract.split(". ", maxsplit=1)[0]
    return f"Title: {row['title']} | Abstract: {sentence}"


def decision_for(index: int) -> tuple[str, str]:
    """Return the independent H1 decision and rationale for one row number."""
    if index in INCLUDE_IDS:
        return "include_definition_or_review", INCLUDE_RATIONALES[index]
    if index in UNCERTAIN_IDS:
        return "uncertain", UNCERTAIN_RATIONALES[index]
    return "exclude_not_relevant", EXCLUDE_RATIONALE


def main() -> None:
    """Create the batch-12 frozen-column-preserving H1 screen."""
    frozen_columns, input_rows = read_csv(INPUT_PATH)
    if len(input_rows) != 120:
        raise ValueError("Expected exactly 120 batch-12 source rows")
    if INCLUDE_IDS & UNCERTAIN_IDS:
        raise ValueError("Screening decision registries overlap")

    output_rows: list[dict[str, str]] = []
    for index, row in enumerate(input_rows, start=1):
        decision, rationale = decision_for(index)
        output_rows.append(
            {
                **row,
                "H1_decision": decision,
                "H1_rationale": rationale,
                "H1_evidence_span": evidence_span(row),
            }
        )
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
        "artifact": "contextual_source_screening_H1_batch12_completed_v4",
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
            "Used only the English title and abstract in the batch-12 input.",
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
