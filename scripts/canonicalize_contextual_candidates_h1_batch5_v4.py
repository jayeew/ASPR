"""Create the independent H1 canonicalization for contextual candidate batch 5."""

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
    "contextual_candidate_canonicalization_input_batch5_v4.csv"
)
OUTPUT_PATH: Final = (
    ROOT / "outputs/contextual_candidate_canonicalization_H1_batch5_completed_v4.csv"
)
MANIFEST_PATH: Final = (
    ROOT
    / "outputs/contextual_candidate_canonicalization_H1_batch5_completed_v4.manifest.json"
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
    decision: str,
    rationale: str,
) -> dict[str, str]:
    """Build a schema-complete H1 review for a T0 context-control candidate."""
    return {
        "H1_family_name_en": family,
        "H1_merge_or_split_reason": merge_reason,
        "H1_formula_reproducible": reproducible,
        "H1_t0_computable": "1",
        "H1_scope_role": "context_control",
        "H1_missing_rule_status": "absent",
        "H1_promotion_decision": decision,
        "H1_rationale": rationale,
    }


REVIEWS: Final[dict[str, dict[str, str]]] = {
    "CFT_2775fafaf169ca97": review(
        "Static paper bibliographic metadata",
        "This is the same focal-paper author count as the second author-count candidate; merge both source mentions under one count construct rather than treating source wording as a variant feature.",
        "1",
        "promote_for_formalization",
        "The source explicitly counts authors per publication at T0. A formal contract needs group/consortium-author and missing-byline handling, but the target quantity is unambiguous.",
    ),
    "CFT_77e50876d1e4a282": review(
        "Static paper bibliographic metadata",
        "This byline count is formula-equivalent to the first author-count candidate; merge the duplicate evidence under author_count, preserving the shared listed-author definition.",
        "1",
        "promote_for_formalization",
        "The source states that total byline authors are counted at publication. It is a direct T0 metadata control, not a novelty measure.",
    ),
    "CFT_ec4b6193121a3f0c": review(
        "First-author affiliation geography",
        "First-author continent is a positional affiliation attribute, distinct from all-author international collaboration and from a paper's topical subfield.",
        "0",
        "retain_evidence_gap",
        "The source defines continent dummies with Oceania as reference, but leaves multi-affiliation first authors, unresolved countries, and missing affiliations unspecified. Those rules are necessary for reproducible coding.",
    ),
    "CFT_8b1b9b194be425a6": review(
        "Paper funding status",
        "Funded/nonfunded is a binary acknowledgement/funding-status control distinct from author geography, open access, and journal identity.",
        "1",
        "promote_for_formalization",
        "The source directly specifies a binary funded/nonfunded indicator obtainable at publication. Formalization should fail closed when funding information is absent or ambiguous rather than treating it as nonfunded.",
    ),
    "CFT_87576570954d2346": review(
        "Paper collaboration geographic scope",
        "Domestic/international status is a binary all-author affiliation-country projection; retain it separately from first-author region and from country-count diversity.",
        "1",
        "promote_for_formalization",
        "The source explicitly maps one country to domestic and more than one country to international at T0. Missing affiliation-country and country-normalization rules remain to be fixed.",
    ),
    "CFT_d91a731d0c26055b": review(
        "Journal identity",
        "Journal dummies form a categorical venue-identity scheme, distinct from journal-impact metrics and issue position; the source's sampled journal universe is not a universal taxonomy.",
        "0",
        "retain_evidence_gap",
        "The one-hot formula is explicit, but a reusable implementation needs a frozen venue identifier, reference-category policy, and treatment of journals outside the source's eleven-journal sample.",
    ),
    "CFT_6ac180ce9bfd53a3": review(
        "Paper research-design and content classifications",
        "Methodology orientation is a source-coded content category distinct from publication form, subfield, and collaboration metadata; do not merge those constructs.",
        "0",
        "retain_evidence_gap",
        "The source names dummy coding and a reference group but relies on an external methodology guideline. Its categories, adjudication process, and missing/ambiguous-method rule are not fully supplied here.",
    ),
    "CFT_1cc3cb5d39df9ac8": review(
        "Static paper bibliographic metadata",
        "This cited-reference count is formula-equivalent to the second reference-count candidate; merge both under one focal-paper reference_count construct rather than split on wording.",
        "1",
        "promote_for_formalization",
        "The source directly counts cited references at publication. A formal contract still needs reference parsing and missing-reference-list handling.",
    ),
    "CFT_04cc02d4beb257b7": review(
        "Static paper bibliographic metadata",
        "This count of cited references at the article end is equivalent to the first reference-count candidate; merge the two source applications under reference_count.",
        "1",
        "promote_for_formalization",
        "The source supplies the same publication-time focal-reference count as the paired candidate. It is a context control only.",
    ),
    "CFT_fd3c18369d1d9ca7": review(
        "Paper research-design and content classifications",
        "Applied-linguistics subfield is a source-specific topical classification, distinct from publication venue, general field metadata, and methodology orientation.",
        "0",
        "retain_evidence_gap",
        "The source specifies dummy coding and a reference group but delegates the category system to an external definition. A reproducible category dictionary and rules for multi-topic or uncodable papers are missing.",
    ),
    "CFT_0238f457adf2f3ef": review(
        "Static paper bibliographic metadata",
        "Title character length is a distinct static text count; punctuation and spaces are explicit parts of this count, not an interchangeable title-punctuation feature.",
        "1",
        "promote_for_formalization",
        "The source directly counts every title character including punctuation and spaces, using only publication-time title text. Missing-title and Unicode-normalization policy should be fixed at formalization.",
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
            raise ValueError("Batch 5 input has no header.")
        return reader.fieldnames, list(reader)


def validate(
    input_fields: list[str], before: list[dict[str, str]], after: list[dict[str, str]]
) -> None:
    """Ensure only H1 fields were appended and all judgments are valid."""
    decisions = {"promote_for_formalization", "retain_evidence_gap", "reject"}
    if len(before) != 11 or len(after) != 11:
        raise ValueError("Expected 11 batch-5 candidates.")
    if {row["candidate_id"] for row in before} != set(REVIEWS):
        raise ValueError("Candidate IDs do not match the independent H1 reviews.")
    for original, completed in zip(before, after, strict=True):
        for field in input_fields:
            if original[field] != completed[field]:
                raise ValueError(f"Frozen input field changed: {field}")
        if set(completed) != {*input_fields, *H1_FIELDS}:
            raise ValueError("Unexpected output schema.")
        if completed["H1_formula_reproducible"] not in {"0", "1"}:
            raise ValueError("Invalid H1 formula reproducibility value.")
        if completed["H1_t0_computable"] != "1":
            raise ValueError("A batch-5 candidate is not T0-computable.")
        if completed["H1_promotion_decision"] not in decisions:
            raise ValueError("Invalid H1 promotion decision.")
        if completed["H1_scope_role"] != "context_control":
            raise ValueError("Unexpected scope role.")
        if not all(completed[field] for field in H1_FIELDS):
            raise ValueError("An H1 field is blank.")


def write_csv(fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write the completed H1 canonicalization CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Create the blind H1 batch-5 canonicalization and manifest."""
    fields, rows = read_rows()
    completed = [{**row, **REVIEWS[row["candidate_id"]]} for row in rows]
    validate(fields, rows, completed)
    write_csv([*fields, *H1_FIELDS], completed)
    decisions = Counter(row["H1_promotion_decision"] for row in completed)
    manifest: dict[str, Any] = {
        "artifact": OUTPUT_PATH.name,
        "artifact_sha256": sha256(OUTPUT_PATH),
        "batch": 5,
        "candidate_count": len(completed),
        "input": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "h1_fields": H1_FIELDS,
        "promotion_decision_counts": dict(sorted(decisions.items())),
        "reviewer": "H1",
        "schema": "contextual_candidate_canonicalization_h1_batch5_v4",
        "blind_review_constraints": [
            "Only batch-5 frozen candidate formula/evidence fields were used for H1 judgments.",
            "No AI, H2, Qwen, or Ollama results were consulted.",
            "Frozen columns were preserved unchanged; only the eight H1 fields were appended.",
            "All retained candidates are context controls, never novelty evidence.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
