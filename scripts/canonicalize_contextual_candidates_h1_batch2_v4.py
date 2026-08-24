"""Create independent H1 family pre-consolidation for batch-2 candidates."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Final

ROOT: Final = Path("/home/jayee/workspace/ASPR")
INPUT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_candidate_canonicalization_input_batch2_v4.csv"
)
OUTPUT_PATH: Final = (
    ROOT / "outputs/contextual_candidate_canonicalization_H1_batch2_completed_v4.csv"
)
MANIFEST_PATH: Final = (
    ROOT
    / "outputs/contextual_candidate_canonicalization_H1_batch2_completed_v4.manifest.json"
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
    formula_reproducible: str,
    t0_computable: str,
    scope_role: str,
    missing_rule_status: str,
    promotion_decision: str,
    rationale: str,
) -> dict[str, str]:
    """Return one schema-conformant H1 pre-consolidation review."""
    return {
        "H1_family_name_en": family,
        "H1_merge_or_split_reason": merge_reason,
        "H1_formula_reproducible": formula_reproducible,
        "H1_t0_computable": t0_computable,
        "H1_scope_role": scope_role,
        "H1_missing_rule_status": missing_rule_status,
        "H1_promotion_decision": promotion_decision,
        "H1_rationale": rationale,
    }


REVIEWS: Final[dict[str, dict[str, str]]] = {
    "CFT_d889836ce56d5065": review(
        "CSO ontology-based paper topic representation",
        "Duplicate of the CSO topic-set candidate: both describe the classifier's paper-level research-topic output; keep the semantic relevance score as a distinct supporting component.",
        "0",
        "0",
        "t0_substantive",
        "absent",
        "retain_evidence_gap",
        "The paper says the classifier maps title, abstract, and keywords to CSO topics (pp. 91, 95), but the text depends on a trained embedding model and CSO assets without a T0-frozen artifact/version contract.",
    ),
    "CFT_88b0d5a0bef4c8ef": review(
        "Peer-review decision and process history",
        "Duplicate of the separately named initial-editorial-decision candidate: both use the same four first-decision categories; retain turnaround, submissions, reviewers, and revision effort as distinct process members.",
        "1",
        "1",
        "t0_opportunity",
        "absent",
        "promote_for_formalization",
        "The source explicitly classifies first decisions as immediate accept, minor revision, major revision, or revise-and-resubmit (p. 1093). It does not state treatment for absent or unresolved initial decision records.",
    ),
    "CFT_14abd803b2912e4e": review(
        "Peer-review decision and process history",
        "Duplicate of peer-review turnaround-time-days: both measure the same received-for-review to acceptance interval, distinct from decision category and revision effort.",
        "1",
        "1",
        "t0_opportunity",
        "absent",
        "promote_for_formalization",
        "Turnaround time is explicitly defined as time in review from receipt to acceptance (p. 1093); it is observable by acceptance/final publication, with no stated missing-date rule.",
    ),
    "CFT_3f4927e07de30081": review(
        "Reference-field interdisciplinarity components",
        "Balance and variety share the focal paper's reference-field distribution but are separate diversity components, not duplicates; keep them split from disparity, RS, and DIV composites.",
        "1",
        "1",
        "direct_innovation",
        "explicit",
        "promote_for_formalization",
        "Table 1 defines balance as 1 minus Gini for the reference-field distribution (p. 4); the article explicitly excludes publications with fewer than three references before calculation.",
    ),
    "CFT_d9a027d47d0fa555": review(
        "Reference-field interdisciplinarity components",
        "Variety and balance use the same reference-field support/distribution but measure different dimensions; retain variety separately rather than merge values.",
        "1",
        "1",
        "direct_innovation",
        "explicit",
        "promote_for_formalization",
        "Table 1 defines variety as n/N, where n is reference-field categories and N=292 (p. 4); the source explicitly excludes papers with fewer than three references before calculation.",
    ),
    "CFT_09c47725eedf8096": review(
        "Text length and title–abstract lexical overlap",
        "Abstract character length is a distinct length component in the shared text-form family; do not merge it with title length or lexical-overlap ratios.",
        "1",
        "1",
        "context_control",
        "absent",
        "promote_for_formalization",
        "Table 1 defines abstract length as characters irrespective of spaces and punctuation (p. 3694); no rule is supplied for missing or malformed abstracts.",
    ),
    "CFT_4ac9dffb4f336831": review(
        "Text length and title–abstract lexical overlap",
        "This title-to-abstract repetition ratio shares tokenized title/abstract inputs with the reverse coverage ratio, but is a directional non-duplicate projection.",
        "1",
        "1",
        "t0_substantive",
        "absent",
        "promote_for_formalization",
        "Table 1 defines it as title-word repetition frequency in the abstract divided by abstract length without stop words (p. 3693); the source does not specify a stop-word or null-text rule.",
    ),
    "CFT_3217066e07b373b1": review(
        "Static bibliographic and document-structure attributes",
        "Duplicate of paper length (number of pages) from the second source; both are article page counts, while figures, tables, authors, references, and document type remain distinct family members.",
        "1",
        "1",
        "context_control",
        "absent",
        "promote_for_formalization",
        "The article directly operationalizes article length as number of article pages (Table 1, p. 3694), with no pagination or missing-page convention stated.",
    ),
    "CFT_36bd47b47e562a9b": review(
        "Static bibliographic and document-structure attributes",
        "Duplicate of article_page_count from the other source: both measure paper length in pages and should be consolidated to one canonical page-count definition.",
        "1",
        "1",
        "context_control",
        "absent",
        "promote_for_formalization",
        "Table 11 names paper length as number of pages (p. 8405); the source supplies no rule for article-numbered documents or missing pagination.",
    ),
    "CFT_229be64efded8380": review(
        "Static bibliographic and document-structure attributes",
        "Duplicate of all author-count candidates across three sources; the raw labels differ but each is a simple focal-paper authorship count.",
        "1",
        "1",
        "context_control",
        "absent",
        "promote_for_formalization",
        "Table 1 defines co-authors as number of authors (p. 3694), without a group-author or missing-byline rule.",
    ),
    "CFT_e018de3163123a9b": review(
        "Static bibliographic and document-structure attributes",
        "Duplicate of author_count candidates in the same static-metadata family; retain no source-specific alternate measure unless a counting policy differs.",
        "1",
        "1",
        "context_control",
        "absent",
        "promote_for_formalization",
        "The article uses the count/number of authors as a paper control (pp. 1093–1094), but does not give a group-author or missing-byline treatment.",
    ),
    "CFT_2db7fe601f83d7d0": review(
        "Static bibliographic and document-structure attributes",
        "Duplicate of the other author-count candidates; all measure focal-paper author count rather than a distinct collaboration diversity construct.",
        "1",
        "1",
        "context_control",
        "absent",
        "promote_for_formalization",
        "Table 11 names number of authors as a paper-extrinsic predictor (p. 8405), with no identity, consortium, or missing-author convention.",
    ),
    "CFT_6349de9d481d1df1": review(
        "Static bibliographic and document-structure attributes",
        "Duplicate of author_count: team size is explicitly the number of co-authors in publication i, not a different team-composition measure.",
        "1",
        "1",
        "context_control",
        "absent",
        "promote_for_formalization",
        "The regression definition gives team_size as the number of co-authors in publication i (p. 4); it has no group-authorship or missing-record policy.",
    ),
    "CFT_baf488321880a2a5": review(
        "Reference-list temporal profile",
        "Reference age is a temporal summary of the cited-reference set, distinct from reference count, self-citation, and reference-field diversity.",
        "1",
        "1",
        "context_control",
        "absent",
        "promote_for_formalization",
        "Table 1 states average reference age as average reference publication year minus focal-paper publication year (p. 3694), without treatment for incomplete reference years.",
    ),
    "CFT_5489a530e80df294": review(
        "CSO ontology-based paper topic representation",
        "Duplicate of CSO Classifier research topics: both are the classifier output topic set; keep the relevance score separately as a per-topic ranking component.",
        "0",
        "0",
        "t0_substantive",
        "absent",
        "retain_evidence_gap",
        "The classifier takes paper title, abstract, and keywords and returns ontology topics (pp. 91, 95), but full reproduction and historic T0 use require a frozen CSO/embedding/model artifact not specified in the paper.",
    ),
    "CFT_97d13470b05f1890": review(
        "CSO ontology-based paper topic representation",
        "The relevance score is a distinct per-topic ranking member that supports the shared CSO topic set; it should not be merged numerically with the output membership vector.",
        "1",
        "0",
        "t0_substantive",
        "absent",
        "retain_evidence_gap",
        "The source defines relevance as identification frequency times number of unique triggering n-grams, with a direct-mention maximum-score override (p. 97); T0 use still requires a frozen classifier/embedding artifact.",
    ),
    "CFT_0deb8e04424d2b44": review(
        "Static bibliographic and document-structure attributes",
        "Document type is a categorical static metadata member, distinct from length, authorship, reference, figure, and table counts.",
        "1",
        "1",
        "context_control",
        "absent",
        "promote_for_formalization",
        "Table 1 records document type as Review versus Article (p. 3695); the source gives no rule for other or missing document types.",
    ),
    "CFT_cf4190f1eb347d13": review(
        "Static bibliographic and document-structure attributes",
        "Duplicate of number of figures from the second source: both are focal-paper figure counts, separate from table count and article length.",
        "1",
        "1",
        "context_control",
        "absent",
        "promote_for_formalization",
        "Table 1 directly applies number of figures (p. 3694), without defining how multipart, supplemental, or missing figures are counted.",
    ),
    "CFT_e6bb7d4e17c9218c": review(
        "Static bibliographic and document-structure attributes",
        "Duplicate of figure_count from the other source: it is the same focal-paper structural count and should share one definition contract.",
        "1",
        "1",
        "context_control",
        "absent",
        "promote_for_formalization",
        "Table 11 names number of figures as a paper-extrinsic predictor (p. 8405), but supplies no multipart/supplement handling.",
    ),
    "CFT_202163a521c5f471": review(
        "Peer-review decision and process history",
        "Duplicate of Original decision: it uses the same four-category first editorial-decision field and should be consolidated to that definition.",
        "1",
        "1",
        "t0_opportunity",
        "absent",
        "promote_for_formalization",
        "The full text gives the four initial decision categories (p. 1093) but has no rule for missing or unresolved decision records.",
    ),
    "CFT_1aebf274e1d619eb": review(
        "Reference-list self-citation profile",
        "Journal self-citations within the reference list are distinct from author self-citations and total reference count, though all share the cited-reference data source.",
        "1",
        "1",
        "context_control",
        "absent",
        "promote_for_formalization",
        "Table 1 defines this as the number of times the target journal is cited in the article references (p. 3694), without journal-identity matching or missing-outlet rules.",
    ),
    "CFT_9cdb6648eeb597e7": review(
        "Static bibliographic and document-structure attributes",
        "Keyword count is a separate static metadata count; it should not be merged with title/abstract lexical measures or topic classifications.",
        "1",
        "1",
        "context_control",
        "absent",
        "promote_for_formalization",
        "The source explicitly uses number/count of keywords as a paper variable (p. 1093), but does not state a delimiter, controlled-keyword, or missing-keyword policy.",
    ),
    "CFT_d16bd6f597661fef": review(
        "Peer-review decision and process history",
        "Non-editor reviewer count is a distinct review-process intensity count, not interchangeable with submissions, turnaround time, decision category, or revision effort.",
        "1",
        "1",
        "t0_opportunity",
        "absent",
        "promote_for_formalization",
        "The publisher data field is number of reviewers in addition to the editor involved in review (p. 1094); no handling is specified for missing/informal reviews.",
    ),
    "CFT_18f4dbeb8d152546": review(
        "Peer-review decision and process history",
        "Duplicate of turnaround time: it is the same received-for-review to acceptance measure and should merge to peer_review_turnaround_time_days.",
        "1",
        "1",
        "t0_opportunity",
        "absent",
        "promote_for_formalization",
        "Turnaround is explicitly the interval from receipt for review to acceptance (p. 1093), observable no later than acceptance/final publication; no missing-date policy is given.",
    ),
    "CFT_e897bda2a6ff6074": review(
        "Publication-language international orientation",
        "This is a jurisdiction-specific language/location encoding, not a duplicate of affiliation-country collaboration; retain separately pending a generalizable territorial rule.",
        "1",
        "1",
        "t0_opportunity",
        "absent",
        "retain_evidence_gap",
        "The source calls non-Dutch publications international and also Dutch papers published in Belgium/former Dutch colonies international (p. 425); the country scope and historic territorial rule are not generally reusable.",
    ),
    "CFT_7b16db0737deb0d0": review(
        "Static bibliographic and document-structure attributes",
        "Duplicate of the two other reference-count candidates: all are counts of references in the focal paper, not average reference age or self-citation profiles.",
        "1",
        "1",
        "context_control",
        "absent",
        "promote_for_formalization",
        "Table 1 directly records number of references (p. 3694), without an explicit reference-list parsing or missing-reference policy.",
    ),
    "CFT_b825068902f2938b": review(
        "Static bibliographic and document-structure attributes",
        "Duplicate of reference_count candidates: its raw label differs but it is the same focal-paper reference total and should use one shared contract.",
        "1",
        "1",
        "context_control",
        "absent",
        "promote_for_formalization",
        "Table 11 names number of references as a paper-extrinsic predictor (p. 8405), with no reference parsing or missing-list rule.",
    ),
    "CFT_7c87d8eb4b4b6bb1": review(
        "Static bibliographic and document-structure attributes",
        "Duplicate of reference_count candidates: the source explicitly defines references_count_i as the number of references in publication i.",
        "1",
        "1",
        "context_control",
        "absent",
        "promote_for_formalization",
        "The regression variables define references_count_i as number of references in publication i (p. 4), without a missing-list or parsing rule.",
    ),
    "CFT_f7a477f40a9ceb5e": review(
        "Peer-review decision and process history",
        "Revision effort is a derived binary grouping of initial editorial decision categories; retain separately from the underlying four-level decision where analytical granularity matters.",
        "1",
        "1",
        "t0_opportunity",
        "absent",
        "promote_for_formalization",
        "The full text defines low effort as accept/minor revision and high effort as major revision/revise-and-resubmit (p. 1097); it lacks a missing-decision policy.",
    ),
    "CFT_21ac5535ef55a900": review(
        "Peer-review decision and process history",
        "Submission count is a distinct accumulated review-process count, not a synonym for reviewer count, turnaround duration, or revision effort.",
        "1",
        "1",
        "t0_opportunity",
        "absent",
        "promote_for_formalization",
        "The prepared publisher data include number of submissions (p. 1094), observable by acceptance/final publication but without a withdrawal/resubmission or missing-data rule.",
    ),
    "CFT_e3b2a9794c9be05f": review(
        "Static bibliographic and document-structure attributes",
        "Table count is a distinct structural count in the shared static-metadata family, not a duplicate of figure count or page length.",
        "1",
        "1",
        "context_control",
        "absent",
        "promote_for_formalization",
        "Table 1 directly applies number of tables (p. 3694) but does not specify multipart, supplemental, or missing-table treatment.",
    ),
    "CFT_0fdf9cb04c94213d": review(
        "Text length and title–abstract lexical overlap",
        "Title colon presence is a distinct punctuation feature in the title/text family; it cannot be merged with word count or title–abstract overlap without losing its binary content form.",
        "0",
        "1",
        "context_control",
        "absent",
        "retain_evidence_gap",
        "Table 11 applies “Colons in title” (p. 8405), but provides no coding rule for colon characters, title variants, or missing titles.",
    ),
    "CFT_b33398155a7c2b76": review(
        "Text length and title–abstract lexical overlap",
        "Title word count is a separate length component in the shared title/abstract form family; it is not a duplicate of abstract character count or lexical overlap ratios.",
        "1",
        "1",
        "context_control",
        "absent",
        "promote_for_formalization",
        "Table 1 defines title length as number of title words without punctuation (p. 3694); no tokenization, punctuation, or missing-title policy is stated.",
    ),
    "CFT_c8806d1f087a649d": review(
        "Text length and title–abstract lexical overlap",
        "This reverse title-coverage ratio uses the same texts as abstract-title repetition but is a distinct directional projection and should remain split.",
        "1",
        "1",
        "t0_substantive",
        "absent",
        "promote_for_formalization",
        "Table 1 defines it as number of title words in the abstract divided by title length without stop words (p. 3693); the source gives no stop-word or null-text rule.",
    ),
}


def sha256_file(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_input() -> tuple[list[dict[str, str]], list[str]]:
    """Read frozen candidate rows."""
    with INPUT_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        return list(reader), reader.fieldnames


def validate_rows(rows: list[dict[str, str]]) -> None:
    """Verify candidate coverage and every referenced full-text SHA."""
    if len(rows) != len(REVIEWS):
        raise ValueError(f"Expected {len(REVIEWS)} rows, found {len(rows)}")
    if {row["candidate_id"] for row in rows} != set(REVIEWS):
        raise ValueError("Candidate IDs do not match the independent H1 registry")
    for row in rows:
        if sha256_file(Path(row["local_path"])) != row["fulltext_sha256"]:
            raise ValueError(f"Full-text SHA mismatch for {row['candidate_id']}")


def write_output(rows: list[dict[str, str]], fields: list[str]) -> str:
    """Preserve frozen columns and append H1 pre-consolidation fields."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[*fields, *H1_FIELDS], extrasaction="raise"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **REVIEWS[row["candidate_id"]]})
    return sha256_file(OUTPUT_PATH)


def write_manifest(rows: list[dict[str, str]], output_sha: str) -> None:
    """Record blind-review provenance and output integrity."""
    decisions = [REVIEWS[row["candidate_id"]]["H1_promotion_decision"] for row in rows]
    manifest = {
        "input": str(INPUT_PATH),
        "input_sha256": sha256_file(INPUT_PATH),
        "artifact": str(OUTPUT_PATH),
        "artifact_sha256": output_sha,
        "row_count": len(rows),
        "fulltext_sha256_verified": True,
        "promotion_counts": {
            value: decisions.count(value) for value in sorted(set(decisions))
        },
        "qwen_or_ollama_used": False,
        "review_scope": "Independent H1 family pre-consolidation only; not final feature selection.",
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    """Build the batch-2 H1 candidate canonicalization artifact."""
    rows, fields = read_input()
    validate_rows(rows)
    output_sha = write_output(rows, fields)
    write_manifest(rows, output_sha)


if __name__ == "__main__":
    main()
