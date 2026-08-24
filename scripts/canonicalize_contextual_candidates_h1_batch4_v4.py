"""Create the independent H1 canonicalization for contextual candidate batch 4."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path("/home/jayee/workspace/ASPR")
INPUT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_candidate_canonicalization_input_batch4_v4.csv"
)
OUTPUT_PATH: Final = (
    ROOT / "outputs/contextual_candidate_canonicalization_H1_batch4_completed_v4.csv"
)
MANIFEST_PATH: Final = (
    ROOT
    / "outputs/contextual_candidate_canonicalization_H1_batch4_completed_v4.manifest.json"
)
H1_FIELDS: Final = [
    "H1_family_name_en",
    "H1_merge_or_split_reason",
    "H1_formula_reproducible",
    "H1_t0_computable",
    "H1_scope_role",
    "H1_missing_rule_status",
    "H1_promotion_decision",
    "H1_rationale",
]


def review(
    family: str,
    merge_reason: str,
    reproducible: str,
    scope_role: str,
    decision: str,
    rationale: str,
) -> dict[str, str]:
    """Build one schema-complete H1 review row."""
    return {
        "H1_family_name_en": family,
        "H1_merge_or_split_reason": merge_reason,
        "H1_formula_reproducible": reproducible,
        "H1_t0_computable": "1",
        "H1_scope_role": scope_role,
        "H1_missing_rule_status": "absent",
        "H1_promotion_decision": decision,
        "H1_rationale": rationale,
    }


REVIEWS: Final[dict[str, dict[str, str]]] = {
    "CFT_c47e432a527622ce": review(
        "Paper research-design and content classifications",
        "Article type is a mutually exclusive content coding distinct from methodology, study design, and temporal-context encodings; retain it as an atomic family member.",
        "1",
        "context_control",
        "promote_for_formalization",
        "The source fixes seven article types and precedence rules for mixed/theory/review cases. It is publication-time and formula-reproducible, though an implementation must specify handling of unreadable or uncodable papers.",
    ),
    "CFT_f62b83243f98d35b": review(
        "Static paper bibliographic metadata",
        "Author count is an atomic byline count; keep it separate from references and issue metadata, while harmonizing their shared publication-metadata contract.",
        "1",
        "context_control",
        "promote_for_formalization",
        "The formula is an explicit count of the article byline and is available at T0. The source does not resolve consortium/group-author treatment, so that missing-data rule must be set during formalization.",
    ),
    "CFT_da264ea490d92a9c": review(
        "Paper research-design and content classifications",
        "Data-source categories are a non-exclusive methodological coding distinct from study design and hypothesis-testing method; do not collapse the category indicators into one score.",
        "0",
        "context_control",
        "retain_evidence_gap",
        "The source lists the categories and T0 text inputs, but does not define the subjective/objective boundary or an ambiguity/missing-data rule. Those omissions prevent a reproducible cross-paper coding contract.",
    ),
    "CFT_0d43241beb639aed": review(
        "Paper methodological-validity assessment",
        "The endogeneity flag is a distinct methodology-quality assessment, not a synonym for article type, analysis method, or study design.",
        "0",
        "context_control",
        "retain_evidence_gap",
        "The binary rule depends on a twelve-threat checklist named but not operationalized in this candidate record, along with reviewer judgment. It is publication-time in principle, but needs the complete checklist and adjudication rules.",
    ),
    "CFT_66fc9ed4348e96ef": review(
        "Paper research-design and content classifications",
        "Hypothesis-testing-method indicators are non-exclusive method codes and should remain separate from design, data source, and scale-type features.",
        "0",
        "context_control",
        "retain_evidence_gap",
        "The source enumerates method categories but does not provide a complete coding protocol for borderline or multiple analyses beyond non-exclusivity. A formal coding manual is required before implementation.",
    ),
    "CFT_efe86ddd32862286": review(
        "Static paper bibliographic metadata",
        "Issue number is a categorical publication-metadata control distinct from journal identity and other static document counts.",
        "1",
        "context_control",
        "promote_for_formalization",
        "The source explicitly encodes issue category with k-1 dummies, and issue metadata is available at publication. A contract should choose treatment for special, supplement, or missing issues.",
    ),
    "CFT_fe3f0eea960e7fdf": review(
        "Author-team prior scholarly standing",
        "Lagged citation and publication averages are distinct prior-standing measures sharing author identity resolution and a strict pre-publication cutoff; retain both rather than merge them.",
        "1",
        "t0_opportunity",
        "retain_evidence_gap",
        "The source gives the team mean over citation histories before the publication year. It is T0-computable only with a dated citation snapshot and resolved author identities; neither missing-profile nor disambiguation rules are supplied.",
    ),
    "CFT_720bc2db6b4d3751": review(
        "Author-team prior scholarly standing",
        "Lagged publication and citation averages are complementary prior-standing measures sharing the same team/time window, not interchangeable measures.",
        "1",
        "t0_opportunity",
        "retain_evidence_gap",
        "The source explicitly averages author publication histories prior to the focal publication year. A formal version still needs identity resolution, database coverage, and missing-profile rules.",
    ),
    "CFT_0ed697789d32b804": review(
        "Paper research-design and content classifications",
        "Leadership-school labels are a source-specific topical taxonomy; do not merge them with generic article type or method categories.",
        "0",
        "context_control",
        "retain_evidence_gap",
        "The nine labels and k-1 encoding are explicit, but hybrid assignment and cross-domain portability depend on an unstated coding protocol. It remains a source-specific evidence-gap candidate.",
    ),
    "CFT_b528bb4021449378": review(
        "Paper research-design and content classifications",
        "Scale-type indicators are non-exclusive measurement choices, distinct from the statistical-analysis and study-design variables.",
        "1",
        "context_control",
        "retain_evidence_gap",
        "The categories and their basic definitions are source-authorized and publication-time, but the record supplies no rule for absent measurement detail, mixed scales, or ambiguous adaptations.",
    ),
    "CFT_e09a591561cb8914": review(
        "Paper discipline classification",
        "The paper-level CLC code is a categorical discipline assignment, distinct from journal classification and from a derived multidisciplinarity score.",
        "1",
        "context_control",
        "retain_evidence_gap",
        "The source authorizes an author-supplied CLC code subject to editorial modification at submission/publication. Missing, multiple-code, hierarchy-version, and unavailable-code handling are not specified.",
    ),
    "CFT_eedebcf65f24a45f": review(
        "Static paper bibliographic metadata",
        "Reference count is a static document count distinct from byline count and issue category; preserve its footnote-inclusive definition as a separate member of the metadata family.",
        "1",
        "context_control",
        "promote_for_formalization",
        "The source explicitly counts references from the reference list and footnotes at T0. The formal contract should make duplicate and malformed-reference handling explicit.",
    ),
    "CFT_9b031e6f2b397199": review(
        "Paper research-design and content classifications",
        "Sample location is a non-exclusive geographic study-context coding, distinct from author-affiliation geography and collaboration scope.",
        "0",
        "context_control",
        "retain_evidence_gap",
        "The source permits multiple location indicators but does not define geographic boundaries, multi-site coding precedence, or unknown-location treatment. It needs a coding contract before promotion.",
    ),
    "CFT_d87ce799245280ac": review(
        "Editorial opportunity context",
        "Senior-editor identity is an editorial-assignment control and must remain separate from author prestige or journal metadata; it cannot serve as novelty evidence.",
        "1",
        "t0_opportunity",
        "retain_evidence_gap",
        "The source gives the k-1 editor-category construction and a joint House-Tosi category. Its T0 availability depends on an editorial-assignment source, with no rule for special issues or unknown assignments.",
    ),
    "CFT_7acd06bf4888fec8": review(
        "Paper research-design and content classifications",
        "Study-count category is a capped, mutually exclusive design encoding; it is distinct from article type and number of authors.",
        "1",
        "context_control",
        "promote_for_formalization",
        "The five categories and k-1 representation are explicit and observable in the paper at T0. Formalization should specify the treatment of unclear study boundaries and absent methods sections.",
    ),
    "CFT_10c0d651ba9e7610": review(
        "Paper research-design and content classifications",
        "Study-design indicators are non-exclusive collection-method codes; retain them separately from data source and hypothesis-testing method.",
        "0",
        "context_control",
        "retain_evidence_gap",
        "The source lists the design categories but does not supply sufficient rules for mixed, borderline, or incompletely reported designs. A reusable coder protocol is missing.",
    ),
    "CFT_b0d5c4c71e593252": review(
        "Paper research-design and content classifications",
        "Temporal context is a non-exclusive study-time design encoding distinct from publication year or article age.",
        "1",
        "context_control",
        "promote_for_formalization",
        "The source defines the four categories, including two-time-period and longitudinal distinctions, from methods text available at T0. It still needs a default for unclear or missing temporal detail.",
    ),
}


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows() -> tuple[list[str], list[dict[str, str]]]:
    """Read the frozen input columns and candidate rows."""
    with INPUT_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Batch 4 input has no header.")
        return reader.fieldnames, list(reader)


def validate(
    input_fields: list[str], before: list[dict[str, str]], after: list[dict[str, str]]
) -> None:
    """Ensure only H1 fields were added and every H1 decision is valid."""
    decisions = {"promote_for_formalization", "retain_evidence_gap", "reject"}
    scopes = {
        "direct_innovation",
        "context_control",
        "t0_opportunity",
        "t0_substantive",
    }
    if len(before) != len(after) != 17:
        raise ValueError("Expected 17 batch-4 candidates.")
    if {row["candidate_id"] for row in before} != set(REVIEWS):
        raise ValueError("Candidate IDs do not match the independent H1 reviews.")
    for original, completed in zip(before, after, strict=True):
        for field in input_fields:
            if original[field] != completed[field]:
                raise ValueError(f"Frozen input field changed: {field}")
        if set(completed) != {*input_fields, *H1_FIELDS}:
            raise ValueError("Output schema changed unexpectedly.")
        if completed["H1_promotion_decision"] not in decisions:
            raise ValueError("Invalid H1 promotion decision.")
        if completed["H1_scope_role"] not in scopes:
            raise ValueError("Invalid H1 scope role.")
        if completed["H1_formula_reproducible"] not in {"0", "1"}:
            raise ValueError("Invalid formula reproducibility flag.")
        if completed["H1_t0_computable"] != "1":
            raise ValueError("A retained batch-4 candidate is not T0-computable.")
        if completed["H1_missing_rule_status"] not in {"absent", "explicit"}:
            raise ValueError("Invalid missing-rule status.")
        if not all(completed[field] for field in H1_FIELDS):
            raise ValueError("An H1 field is blank.")


def write_csv(fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write the completed canonicalization CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Create and verify the independent H1 candidate canonicalization."""
    fields, rows = read_rows()
    if len(rows) != 17:
        raise ValueError(f"Expected 17 input rows, found {len(rows)}.")
    completed = [{**row, **REVIEWS[row["candidate_id"]]} for row in rows]
    validate(fields, rows, completed)
    output_fields = [*fields, *H1_FIELDS]
    write_csv(output_fields, completed)
    decisions = Counter(row["H1_promotion_decision"] for row in completed)
    manifest: dict[str, Any] = {
        "artifact": OUTPUT_PATH.name,
        "artifact_sha256": sha256(OUTPUT_PATH),
        "batch": 4,
        "candidate_count": len(completed),
        "input": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "h1_fields": H1_FIELDS,
        "promotion_decision_counts": dict(sorted(decisions.items())),
        "reviewer": "H1",
        "schema": "contextual_candidate_canonicalization_h1_batch4_v4",
        "blind_review_constraints": [
            "Only batch-4 frozen candidate fields were used to determine H1 reviews.",
            "No AI, H2, Qwen, or Ollama results were consulted.",
            "Frozen input columns were retained unchanged; only the eight H1 fields were added.",
            "Opportunity and editorial controls were not treated as novelty evidence.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
