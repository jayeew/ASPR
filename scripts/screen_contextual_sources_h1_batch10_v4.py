"""Write the independent English title-and-abstract H1 screen for batch 10."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Final

ROOT: Final = Path("/home/jayee/workspace/ASPR")
INPUT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_source_screening_input_batch10_v4.csv"
)
OUTPUT_PATH: Final = (
    ROOT / "outputs/contextual_source_screening_H1_batch10_completed_v4.csv"
)
MANIFEST_PATH: Final = (
    ROOT / "outputs/contextual_source_screening_H1_batch10_completed_v4.manifest.json"
)
H1_FIELDS: Final = ["H1_decision", "H1_rationale"]

INCLUDE_IDS: Final = {
    10,
    11,
    13,
    21,
    22,
    26,
    27,
    32,
    33,
    35,
    36,
    37,
    39,
    55,
    60,
    70,
    108,
    110,
    111,
    112,
    113,
    116,
}
UNCERTAIN_IDS: Final = {8, 28, 40, 41}

INCLUDE_RATIONALES: Final = {
    10: (
        "Describes the Norwegian Model, a weighted publication indicator that "
        "operationalizes scholarly production for evaluation and funding."
    ),
    11: (
        "Directly contrasts article-level download and citation rankings to examine "
        "interest and influence among published papers."
    ),
    13: (
        "Applies paper-level measures including readability, similarity, reference "
        "counts, and bibliographic accuracy to assess manuscript quality."
    ),
    21: (
        "Examines citance segmentation and annotation reliability, directly relevant "
        "to content-based citation indicators."
    ),
    22: (
        "Uses manuscript-level editorial, peer-review, and citation-impact variables "
        "to study acceptance disparities."
    ),
    26: (
        "Develops and applies citation-direction and journal-relation indicators for "
        "research and knowledge evaluation."
    ),
    27: (
        "Introduces and discusses a named metric, progressive scholarly acceptance, "
        "for tracking recognition of medical innovations."
    ),
    32: (
        "Directly critiques journal impact-factor construction and its use as an "
        "evaluation measure."
    ),
    33: (
        "Discusses impact factor, h-index, and other bibliometric indices as measures "
        "of scientific output and evaluation."
    ),
    35: (
        "Empirically analyzes top-cited articles and citation patterns, including "
        "authorship and research-method characteristics."
    ),
    36: (
        "Documents how academic evaluation draws on publication counts, impact "
        "factors, and authorship position from researchers' CVs."
    ),
    37: (
        "Reviews the digital infrastructures that produce citation, altmetric, and "
        "identifier data for indicator-based research evaluation."
    ),
    39: (
        "Uses scientific publications to examine collaboration and project-level "
        "quality, novelty, and interdisciplinarity under an exogenous travel shock."
    ),
    55: (
        "Provides a conceptual review of research social-impact metrics and proposes "
        "a framework for articulating and measuring scholarly social aims."
    ),
    60: (
        "Tests article-level citation rates against author count, publication type, "
        "SJR, affiliations, and collaboration characteristics."
    ),
    70: (
        "Compares the effects of explicit paper-level non-content variables, including "
        "reference and author counts, on citation impact."
    ),
    108: (
        "Evaluates individual-paper citation impact using collaboration type, "
        "corresponding-author country, journal impact factor, and normalized citations."
    ),
    110: (
        "Models article citation frequency with authorship, editorial, access, and "
        "contextual paper variables."
    ),
    111: (
        "Argues for a self-citation index and directly addresses citation-based "
        "measurement of scientific success."
    ),
    112: (
        "Measures article-level open-access status and bibliometric indicators in a "
        "defined publication corpus."
    ),
    113: (
        "Examines differences in citation patterns by article type and field for use "
        "in academic evaluation."
    ),
    116: (
        "Uses full-text paper analysis to measure writing style, abstract readability, "
        "publication trends, and institutional co-authorship."
    ),
}
UNCERTAIN_RATIONALES: Final = {
    8: "The title signals a citation-based article ranking, but the abstract is absent; the indicator definition and paper-level scope cannot be verified from the permitted evidence.",
    28: "The title concerns bibliometrics, but the supplied abstract is truncated before any specific indicator definition or application can be established.",
    40: "The supplied abstract is incomplete; although the title suggests a writing-style and scholarly-impact study, the metric operationalization cannot be verified.",
    41: "The title and supplied abstract are inconsistent, leaving the actual study and any paper-level indicator content indeterminate.",
}
EXCLUDE_RATIONALE: Final = (
    "Title and abstract describe a substantive-domain study, field-level bibliometric "
    "mapping, or non-paper-level research topic without an evident paper-indicator "
    "definition, application, or directly relevant metric review."
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
    """Write CSV data with a fixed schema."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def decision_for(index: int) -> tuple[str, str]:
    """Return the independent H1 decision and rationale for one row number."""
    if index in INCLUDE_IDS:
        return "include_definition_or_review", INCLUDE_RATIONALES[index]
    if index in UNCERTAIN_IDS:
        return "uncertain", UNCERTAIN_RATIONALES[index]
    return "exclude_not_relevant", EXCLUDE_RATIONALE


def main() -> None:
    """Create the batch-10 H1 screen while preserving all frozen columns."""
    frozen_columns, input_rows = read_csv(INPUT_PATH)
    if len(input_rows) != 120:
        raise ValueError("Expected exactly 120 batch-10 source rows")
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
        "artifact": "contextual_source_screening_H1_batch10_completed_v4",
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
            "Used only the English title and abstract in the batch-10 input.",
            "Copied frozen input fields verbatim without adjudicating AI/H2 content.",
            "Did not use AI/H2 output, prior-batch output, Qwen, or Ollama.",
            "Included only direct paper-level indicator definitions, applications, or relevant metric reviews; generic topical bibliometric mappings were excluded.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
