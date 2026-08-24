"""Create the independent H1 family pre-consolidation for batch 3."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path("/home/jayee/workspace/ASPR")
INPUT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_candidate_canonicalization_input_batch3_v4.csv"
)
OUTPUT_PATH: Final = (
    ROOT / "outputs/contextual_candidate_canonicalization_H1_batch3_completed_v4.csv"
)
MANIFEST_PATH: Final = (
    ROOT
    / "outputs/contextual_candidate_canonicalization_H1_batch3_completed_v4.manifest.json"
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


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review(
    family: str,
    merge_reason: str,
    formula_reproducible: str,
    t0_computable: str,
    missing_rule_status: str,
    promotion_decision: str,
    rationale: str,
) -> dict[str, str]:
    """Build one H1 eight-field pre-consolidation decision."""
    return {
        "H1_family_name_en": family,
        "H1_merge_or_split_reason": merge_reason,
        "H1_formula_reproducible": formula_reproducible,
        "H1_t0_computable": t0_computable,
        "H1_scope_role": "context_control",
        "H1_missing_rule_status": missing_rule_status,
        "H1_promotion_decision": promotion_decision,
        "H1_rationale": rationale,
    }


REVIEWS: Final[dict[str, dict[str, str]]] = {
    "CFT_a203e78cdff6e8fb": review(
        "Authorship team size",
        "Standalone author-count member in this batch; merge only exact focal-paper byline-count variants, not institution, country, or author-position measures.",
        "1",
        "1",
        "stated",
        "promote_for_formalization",
        "The source extracts 'Author number per record' and verifies that complete author lists are available; it excludes publications without authors and describes the PubMed byline-limit history (Methods—Data retrieval, pp. 1303–1304).",
    ),
    "CFT_01ebf955fc5d5c9e": review(
        "Manuscript language coding",
        "Standalone ordinal language-code member; do not merge with title language, translation status, or language-model text features.",
        "1",
        "1",
        "absent",
        "promote_for_formalization",
        "The source explicitly assigns rank 1 to English, 2 to bilingual English/Other papers, and 3 to all other languages (p. 6040); no rule is provided for missing, multiple non-English, or indeterminate language records.",
    ),
    "CFT_a669196cccff29b1": review(
        "Reference-list temporal profile",
        "Keep mean reference age distinct from median age, age dispersion, recency thresholds, and the source's signed average-reference-year offset variants.",
        "0",
        "1",
        "absent",
        "retain_evidence_gap",
        "Table 1 names 'Mean age of references in years' (p. 6040), but the source supplies neither the age arithmetic, year/date granularity, nor an undated-reference rule.",
    ),
    "CFT_251674ec3cf6883e": review(
        "Paper length",
        "Standalone page-length member; do not merge with word, character, abstract, title, supplement, or article-number measures.",
        "0",
        "1",
        "absent",
        "retain_evidence_gap",
        "Table 1 defines Length only as 'Length of the paper in pages' (p. 6040). It does not state a version-of-record, start/end-page, article-number, or supplement convention.",
    ),
    "CFT_596f832ef25cc770": review(
        "Publication time",
        "Standalone publication-year categorical-control member; do not merge with online-first date, issue date, acceptance date, or citation-window age.",
        "1",
        "1",
        "absent",
        "promote_for_formalization",
        "Table 1 directly defines Year as 'Year of publication' (p. 6040). The operational construct is clear and publication-time observable, though a precedence rule for multiple publication dates remains unstated.",
    ),
    "CFT_400d27297aa6d879": review(
        "Reference-list size",
        "Standalone reference-count member; merge only exact focal-bibliography count variants, not reference age, novelty, self-citation, or source-field diversity measures.",
        "1",
        "1",
        "absent",
        "promote_for_formalization",
        "Table 1 defines References as 'Number of references' (p. 6040), a direct focal-paper count. It gives no duplicate, unlinked-reference, or missing-bibliography rule.",
    ),
    "CFT_7299d2743bc905c4": review(
        "Title length",
        "Standalone title word-count member; do not merge with abstract length, paper pages, character count, punctuation count, or readability measures.",
        "0",
        "1",
        "absent",
        "retain_evidence_gap",
        "Table 1 defines Title as 'Length of the title in words' (p. 6040), but gives no tokenization, punctuation, hyphenation, multilingual-title, or missing-title rule.",
    ),
}


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    """Write deterministic UTF-8 CSV output."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Verify batch assets and write the complete H1 review."""
    with INPUT_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        input_columns = reader.fieldnames
        input_rows = list(reader)
    if input_columns is None:
        raise ValueError("Input CSV has no header")
    if len(input_rows) != 7:
        raise ValueError(f"Expected 7 candidates, found {len(input_rows)}")
    if {row["candidate_id"] for row in input_rows} != set(REVIEWS):
        raise ValueError(
            "Review registry does not exactly cover the input candidate IDs"
        )
    for row in input_rows:
        if sha256(Path(row["local_path"])) != row["fulltext_sha256"]:
            raise ValueError(f"Full-text hash mismatch for {row['candidate_id']}")

    output_rows = [{**row, **REVIEWS[row["candidate_id"]]} for row in input_rows]
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    write_csv(OUTPUT_PATH, [*input_columns, *H1_FIELDS], output_rows)
    manifest: dict[str, Any] = {
        "artifact": "contextual_candidate_canonicalization_H1_batch3_completed_v4",
        "reviewer": "H1",
        "input_path": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "output_sha256": sha256(OUTPUT_PATH),
        "candidate_rows": len(output_rows),
        "h1_field_count": len(H1_FIELDS),
        "promotion_decision_counts": {
            "promote_for_formalization": sum(
                row["H1_promotion_decision"] == "promote_for_formalization"
                for row in output_rows
            ),
            "retain_evidence_gap": sum(
                row["H1_promotion_decision"] == "retain_evidence_gap"
                for row in output_rows
            ),
        },
        "fulltext_sha256_verified": True,
        "qwen_or_ollama_used": False,
        "blind_review_constraints": [
            "Used only the batch-3 input and its specified English full texts.",
            "Did not use AI/H2 or prior batch output as evidence.",
            "Kept formula-sensitive candidates as evidence gaps where source operational detail is incomplete.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
