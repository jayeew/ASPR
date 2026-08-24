"""Create the blind H1 canonicalization review for contextual candidates."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Final

ROOT: Final = Path("/home/jayee/workspace/ASPR")
INPUT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_candidate_canonicalization_input_v4.csv"
)
OUTPUT_PATH: Final = (
    ROOT / "outputs/contextual_candidate_canonicalization_H1_completed_v4.csv"
)
MANIFEST_PATH: Final = (
    ROOT / "outputs/contextual_candidate_canonicalization_H1_completed_v4.manifest.json"
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

REVIEWS: Final[dict[str, dict[str, str]]] = {
    "CFT_ba1f992964aea925": {
        "H1_family_name_en": "Abstract readability",
        "H1_merge_or_split_reason": "Gunning Fog and Flesch are distinct named scales within one abstract-readability family; the source invokes rather than operationally specifies either scale.",
        "H1_formula_reproducible": "0",
        "H1_t0_computable": "1",
        "H1_scope_role": "t0_substantive",
        "H1_missing_rule_status": "absent",
        "H1_promotion_decision": "retain_evidence_gap",
        "H1_rationale": "The full text says it uses Gunning Fog and Flesch indices to compute abstract difficulty (p. 3), but gives no equations, tokenization convention, or rule for absent or malformed abstracts.",
    },
    "CFT_fe43839fab9ef08d": {
        "H1_family_name_en": "Data-paper abstract rhetorical-move profile",
        "H1_merge_or_split_reason": "Shares one sentence-level rhetorical-move codebook with the structural candidate; retain composition as a distinct aggregate projection of the same annotations.",
        "H1_formula_reproducible": "1",
        "H1_t0_computable": "1",
        "H1_scope_role": "t0_substantive",
        "H1_missing_rule_status": "absent",
        "H1_promotion_decision": "promote_for_formalization",
        "H1_rationale": "Table 1 defines the nine moves; the study double-codes sentences and explicitly gives half credit when two moves co-exist (pp. 6–7). It does not define a general rule for missing or unclassifiable abstracts.",
    },
    "CFT_bc25889b7a715533": {
        "H1_family_name_en": "Data-paper abstract rhetorical-move profile",
        "H1_merge_or_split_reason": "Shares the same sentence-level codebook as rhetorical-move composition, but sequence/position is a separate structural projection and needs a separately fixed encoding.",
        "H1_formula_reproducible": "0",
        "H1_t0_computable": "1",
        "H1_scope_role": "t0_substantive",
        "H1_missing_rule_status": "absent",
        "H1_promotion_decision": "retain_evidence_gap",
        "H1_rationale": "The source asks what order the moves occur in and supplies the codebook (pp. 3, 6), but does not fix one reusable sequence representation, aggregation, or missing-abstract rule.",
    },
    "CFT_af5b714bad01896d": {
        "H1_family_name_en": "Author-team prior scholarly standing",
        "H1_merge_or_split_reason": "Mean and median author h-index are complementary aggregations of one prior-standing construct; keep both only if author identity resolution and citation-source policy are fixed together.",
        "H1_formula_reproducible": "1",
        "H1_t0_computable": "1",
        "H1_scope_role": "context_control",
        "H1_missing_rule_status": "absent",
        "H1_promotion_decision": "retain_evidence_gap",
        "H1_rationale": "The regression table names mean and median authors' h-index at publication time (p. 8), but the full text does not operationalize author disambiguation, citation snapshot, or missing author-profile handling.",
    },
    "CFT_0b0678f184289f1b": {
        "H1_family_name_en": "Static bibliographic and collaboration attributes",
        "H1_merge_or_split_reason": "Document length, references, authors, institutions, countries, and collaboration status are distinct atomic members of one static-metadata family; do not collapse their values into one score.",
        "H1_formula_reproducible": "1",
        "H1_t0_computable": "1",
        "H1_scope_role": "context_control",
        "H1_missing_rule_status": "absent",
        "H1_promotion_decision": "promote_for_formalization",
        "H1_rationale": "The article explicitly defines examples such as title characters and number of institutions in its variable table (p. 5). A formal contract still needs source-wide null and multi-affiliation conventions.",
    },
    "CFT_1c4f7b29076b5adb": {
        "H1_family_name_en": "Coauthor affiliation-country diversity",
        "H1_merge_or_split_reason": "This is a country-distribution diversity metric, distinct from binary or tiered geographic collaboration scope; retain it as the intensity/diversity member of the collaboration family.",
        "H1_formula_reproducible": "1",
        "H1_t0_computable": "1",
        "H1_scope_role": "context_control",
        "H1_missing_rule_status": "explicit",
        "H1_promotion_decision": "promote_for_formalization",
        "H1_rationale": "The source specifies true diversity and country-capital distances (p. 12) and explicitly retains records with at least one author-address link when some links are missing (p. 7).",
    },
    "CFT_45cf9532bd6751c1": {
        "H1_family_name_en": "Data availability statement access level",
        "H1_merge_or_split_reason": "The four ordered data-availability categories form one access-level family; retain category membership rather than treating the labels as interchangeable with repository-use measures.",
        "H1_formula_reproducible": "1",
        "H1_t0_computable": "1",
        "H1_scope_role": "t0_opportunity",
        "H1_missing_rule_status": "explicit",
        "H1_promotion_decision": "promote_for_formalization",
        "H1_rationale": "The full text defines categories 0–3, including no/access-restricted, on-request, in-paper/supplement, and repository statements (p. 4); no statement is explicitly represented in the scheme.",
    },
    "CFT_9bb0adca06dc9458": {
        "H1_family_name_en": "Author-order contribution allocation",
        "H1_merge_or_split_reason": "Harmonic credit is one author-order allocation rule; retain it separately from equal or fractional counting because it makes a different contribution assumption.",
        "H1_formula_reproducible": "1",
        "H1_t0_computable": "1",
        "H1_scope_role": "context_control",
        "H1_missing_rule_status": "absent",
        "H1_promotion_decision": "promote_for_formalization",
        "H1_rationale": "The source supplies the harmonic-credit equation for the i-th author (p. 2). It does not state how equal-contribution notes, group authors, or missing author order are handled.",
    },
    "CFT_bcf1d22c9ca85f4d": {
        "H1_family_name_en": "Paper collaboration geographic scope",
        "H1_merge_or_split_reason": "No-collaboration, national, and international indicators are a mutually exclusive collaboration-scope scheme; merge with the other geographic variants while preserving their different reference-country conventions.",
        "H1_formula_reproducible": "1",
        "H1_t0_computable": "1",
        "H1_scope_role": "t0_opportunity",
        "H1_missing_rule_status": "absent",
        "H1_promotion_decision": "promote_for_formalization",
        "H1_rationale": "The paper defines one institution as no collaboration, multiple institutions in one country as national, and at least two countries as international (p. 11), without a missing-affiliation rule.",
    },
    "CFT_5b2ac3bf3e278b00": {
        "H1_family_name_en": "Paper collaboration geographic scope",
        "H1_merge_or_split_reason": "A focal-country binary is a jurisdiction-relative variant of the shared collaboration-scope family; retain it only with the focal country recorded in the definition.",
        "H1_formula_reproducible": "1",
        "H1_t0_computable": "1",
        "H1_scope_role": "context_control",
        "H1_missing_rule_status": "absent",
        "H1_promotion_decision": "promote_for_formalization",
        "H1_rationale": "International collaboration exists when any author affiliation is outside Malaysia (p. 3). The rule is executable when the focal country is parameterized, but it supplies no missing-affiliation treatment.",
    },
    "CFT_2af3d91460bdc3e5": {
        "H1_family_name_en": "Paper collaboration geographic scope",
        "H1_merge_or_split_reason": "Domestic/international coauthor status is another binary projection of the collaboration-scope family; it differs from the focal-country rule by treating a paper with both domestic and international affiliations as international.",
        "H1_formula_reproducible": "1",
        "H1_t0_computable": "1",
        "H1_scope_role": "context_control",
        "H1_missing_rule_status": "absent",
        "H1_promotion_decision": "promote_for_formalization",
        "H1_rationale": "The full text treats publications with domestic and international affiliations as internationally co-authored (p. 5). It does not state how absent affiliations or unresolved countries are classified.",
    },
    "CFT_9e48f779930b95a7": {
        "H1_family_name_en": "Journal channel group",
        "H1_merge_or_split_reason": "The three journal groups are study-sample strata, not a general article-level channel taxonomy; do not merge them into a reusable journal-quality or open-access feature.",
        "H1_formula_reproducible": "0",
        "H1_t0_computable": "0",
        "H1_scope_role": "context_control",
        "H1_missing_rule_status": "absent",
        "H1_promotion_decision": "reject",
        "H1_rationale": "The source constructs groups from a dated Beall list, a five-journal comparison set, and five PLOS journals (pp. 5–6), without a stable, versioned assignment rule for arbitrary journals at T0.",
    },
    "CFT_ec0dba2f36ed9ffa": {
        "H1_family_name_en": "Paper collaboration geographic scope",
        "H1_merge_or_split_reason": "Local, national, and international collaboration are a three-level geographic-scope taxonomy; merge with related collaboration-scope variants while retaining this finer tiering as a distinct encoding.",
        "H1_formula_reproducible": "1",
        "H1_t0_computable": "1",
        "H1_scope_role": "t0_opportunity",
        "H1_missing_rule_status": "explicit",
        "H1_promotion_decision": "promote_for_formalization",
        "H1_rationale": "The article defines local as one organization/one country, national as multiple organizations in one country, and international as at least two countries (p. 5), and removes records with no affiliation information (p. 8).",
    },
    "CFT_92b58b4720134960": {
        "H1_family_name_en": "Reference-category diversity components",
        "H1_merge_or_split_reason": "Balance and variety derive from the same cited-reference category distribution; retain them as distinct diversity components rather than duplicate features.",
        "H1_formula_reproducible": "1",
        "H1_t0_computable": "1",
        "H1_scope_role": "direct_innovation",
        "H1_missing_rule_status": "absent",
        "H1_promotion_decision": "promote_for_formalization",
        "H1_rationale": "The source defines balance as normalized Shannon diversity using category proportions (pp. 7–8). It leaves unclassified references and multi-category allocation unspecified.",
    },
    "CFT_07a2f9b1d4e5b0ac": {
        "H1_family_name_en": "Reference-category diversity components",
        "H1_merge_or_split_reason": "Variety and balance share the reference-category support/distribution; retain variety as the distinct richness component rather than merging it into balance.",
        "H1_formula_reproducible": "1",
        "H1_t0_computable": "1",
        "H1_scope_role": "direct_innovation",
        "H1_missing_rule_status": "absent",
        "H1_promotion_decision": "promote_for_formalization",
        "H1_rationale": "The paper defines variety as the number of distinctive Web of Science categories among cited articles (p. 7). It does not prescribe treatment for unclassified or multiply classified references.",
    },
    "CFT_74c1bc5c748413a3": {
        "H1_family_name_en": "Static bibliographic and collaboration attributes",
        "H1_merge_or_split_reason": "Author count, keywords, abstract length, pages, and references are atomic members of the same static-metadata family as related bibliographic candidates; retain each field rather than aggregating them.",
        "H1_formula_reproducible": "1",
        "H1_t0_computable": "1",
        "H1_scope_role": "context_control",
        "H1_missing_rule_status": "absent",
        "H1_promotion_decision": "promote_for_formalization",
        "H1_rationale": "The source describes static features as immutable and tabulates keyword count and abstract word count (pp. 4, 6). It gives no common null, version, or affiliation-normalization rule.",
    },
    "CFT_3cf4d70b79c90ecc": {
        "H1_family_name_en": "Static bibliographic and collaboration attributes",
        "H1_merge_or_split_reason": "Reference count, team size, and institution count overlap the static bibliographic/team family; retain the components separately and harmonize their counting contract with related candidates.",
        "H1_formula_reproducible": "1",
        "H1_t0_computable": "1",
        "H1_scope_role": "context_control",
        "H1_missing_rule_status": "absent",
        "H1_promotion_decision": "retain_evidence_gap",
        "H1_rationale": "The variable table defines reference count, team size, and institution count (p. 5), but institution count is nullable and the text provides no missing-value or institution-identity rule.",
    },
    "CFT_95a4b8900709d9db": {
        "H1_family_name_en": "Rao–Stirling reference-category diversity and uncertainty",
        "H1_merge_or_split_reason": "Rao–Stirling diversity and its confidence interval share one category distribution and similarity matrix; retain uncertainty as a linked quality/guard output, not as an interchangeable diversity score.",
        "H1_formula_reproducible": "1",
        "H1_t0_computable": "0",
        "H1_scope_role": "context_control",
        "H1_missing_rule_status": "explicit",
        "H1_promotion_decision": "retain_evidence_gap",
        "H1_rationale": "The paper defines Rao–Stirling from category proportions and pairwise similarities (p. 5), but its similarity matrix was updated with 2015 publication data; recognized-category conditioning is explicit, yet no T0-versioned matrix contract is supplied.",
    },
    "CFT_81052a8be849aab5": {
        "H1_family_name_en": "Rao–Stirling reference-category diversity and uncertainty",
        "H1_merge_or_split_reason": "This is the uncertainty companion of the Rao–Stirling estimate, sharing the same reference categories and similarity matrix; retain it as a derivative guard rather than merge it into the base score.",
        "H1_formula_reproducible": "1",
        "H1_t0_computable": "0",
        "H1_scope_role": "context_control",
        "H1_missing_rule_status": "explicit",
        "H1_promotion_decision": "retain_evidence_gap",
        "H1_rationale": "The source derives uncertainty by bootstrapping recognized subject-category references (p. 6), but the underlying similarity matrix uses a 2015 update and no T0-frozen source version is specified.",
    },
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_input() -> tuple[list[dict[str, str]], list[str]]:
    """Read the frozen input CSV without interpreting its prior-review fields."""
    with INPUT_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        return list(reader), reader.fieldnames


def validate_rows(rows: list[dict[str, str]]) -> None:
    """Verify all candidate PDFs match the frozen full-text SHA values."""
    if len(rows) != len(REVIEWS):
        raise ValueError(f"Expected {len(REVIEWS)} rows, found {len(rows)}")
    observed_ids = {row["candidate_id"] for row in rows}
    if observed_ids != set(REVIEWS):
        raise ValueError("Input candidate IDs do not match the blind-review registry")
    for row in rows:
        pdf_path = Path(row["local_path"])
        actual_sha = sha256_file(pdf_path)
        if actual_sha != row["fulltext_sha256"]:
            raise ValueError(f"PDF SHA mismatch for {row['candidate_id']}: {pdf_path}")


def write_output(rows: list[dict[str, str]], input_fields: list[str]) -> str:
    """Append H1 fields while preserving every frozen input field unchanged."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_fields = [*input_fields, *H1_FIELDS]
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **REVIEWS[row["candidate_id"]]})
    return sha256_file(OUTPUT_PATH)


def write_manifest(rows: list[dict[str, str]], output_sha: str) -> None:
    """Write provenance for the completed blind-review artifact."""
    manifest = {
        "artifact": str(OUTPUT_PATH),
        "artifact_sha256": output_sha,
        "input": str(INPUT_PATH),
        "input_sha256": sha256_file(INPUT_PATH),
        "row_count": len(rows),
        "fulltext_sha256_verified": True,
        "qwen_or_ollama_used": False,
        "review_scope": "H1 family pre-consolidation only; not final feature selection.",
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    """Build and record the blind H1 review artifact."""
    rows, input_fields = read_input()
    validate_rows(rows)
    output_sha = write_output(rows, input_fields)
    write_manifest(rows, output_sha)


if __name__ == "__main__":
    main()
