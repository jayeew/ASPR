"""Create the independent H1 formula and local-T0 mapping formalization for batch 4."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path("/home/jayee/workspace/ASPR")
SOURCE_DIR: Final = (
    ROOT / "innovation_impact_feature_selection/evidence_derived_v4_rebuild"
)
INPUT_PATH: Final = SOURCE_DIR / "outputs/contextual_formalization_input_batch4_v4.csv"
INVENTORY_PATH: Final = SOURCE_DIR / "outputs/local_t0_input_inventory_v4.json"
AUDIT_PATH: Final = SOURCE_DIR / "outputs/operational_equivalence_audit_v4.json"
PROTOCOL_PATH: Final = SOURCE_DIR / "protocol_amendment_v4_operational_equivalence.json"
OUTPUT_PATH: Final = (
    ROOT / "outputs/contextual_formalization_H1_batch4_completed_v4.csv"
)
MANIFEST_PATH: Final = (
    ROOT / "outputs/contextual_formalization_H1_batch4_completed_v4.manifest.json"
)
H1_FIELDS: Final = [
    "H1_canonical_name_en",
    "H1_label_zh",
    "H1_formula",
    "H1_units",
    "H1_parameters",
    "H1_direction",
    "H1_missing_rule",
    "H1_required_data_json",
    "H1_research_group",
    "H1_research_group_evidence",
    "H1_data_match_decision",
    "H1_local_source_ids_json",
    "H1_local_columns_json",
    "H1_derivation_description",
    "H1_formalization_decision",
    "H1_rationale",
]


def json_value(value: object) -> str:
    """Serialize a structured CSV cell deterministically."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def review(**values: str) -> dict[str, str]:
    """Build one schema-complete H1 formalization review."""
    if set(values) != set(H1_FIELDS):
        raise ValueError("H1 review does not match the 16-field contract.")
    return values


def unavailable_review(
    canonical: str,
    label: str,
    formula: str,
    units: str,
    parameters: str,
    direction: str,
    required_data: list[str],
    research_group: str,
    group_evidence: str,
    rationale: str,
) -> dict[str, str]:
    """Build a fail-closed review when no source-identical local field exists."""
    return review(
        H1_canonical_name_en=canonical,
        H1_label_zh=label,
        H1_formula=formula,
        H1_units=units,
        H1_parameters=parameters,
        H1_direction=direction,
        H1_missing_rule="Return missing. The source-required input is not present in the approved local T0 inventory; do not impute, recode to zero, or substitute a proxy.",
        H1_required_data_json=json_value(required_data),
        H1_research_group=research_group,
        H1_research_group_evidence=group_evidence,
        H1_data_match_decision="no_local_match_retain_evidence_gap",
        H1_local_source_ids_json=json_value([]),
        H1_local_columns_json=json_value([]),
        H1_derivation_description="No authorized local T0 mapping exists. The protocol prohibits adjacent or correlated metadata as a substitute for the source quantity.",
        H1_formalization_decision="retain_evidence_gap",
        H1_rationale=rationale,
    )


REVIEWS: Final[dict[str, dict[str, str]]] = {
    "CFT_f62b83243f98d35b": review(
        H1_canonical_name_en="author_count",
        H1_label_zh="作者人数",
        H1_formula="author_count(p) = expm1(log_author_count(p)); this is the audited exact numeric representation of the source quantity, the number of authors listed on focal paper p.",
        H1_units="count of listed authors per focal paper",
        H1_parameters="focal paper; audited OpenAlex author-count representation; no institution, country, or collaboration proxy",
        H1_direction="Higher value denotes a larger authorship team; context control only.",
        H1_missing_rule="Return missing when log_author_count is missing or p is outside the audited overlap; never impute or recode to zero.",
        H1_required_data_json=json_value(
            ["control_features.paper_id", "control_features.log_author_count"]
        ),
        H1_research_group="Static paper bibliographic metadata",
        H1_research_group_evidence="Source evidence: “The total number of coauthors listed in the by-line of the article.”",
        H1_data_match_decision="audited_exact_numeric_representation_equivalence",
        H1_local_source_ids_json=json_value(["control_features"]),
        H1_local_columns_json=json_value(["paper_id", "log_author_count"]),
        H1_derivation_description="Compute expm1(log_author_count). The operational-equivalence audit reports 411489 overlap rows, 1.0 exact equality rate, and maximum absolute difference 2.2737367544323206e-12.",
        H1_formalization_decision="promote_for_formalization",
        H1_rationale="The source formula is a focal-paper byline count at publication. The local transform is explicitly authorized as an exact, outcome-blind representation equivalence; no proxy is used.",
    ),
    "CFT_c47e432a527622ce": unavailable_review(
        "article_type",
        "文章类型",
        "article_type_indicator_k(p) = 1 if focal paper p belongs to source category k in {quantitative, qualitative, theory, review, commentary/discussion, methodological, agent-based simulation}; otherwise 0, with k-1 model dummies.",
        "binary indicator per article-type category",
        "source precedence rules for mixed qualitative/quantitative, meta-analysis, theory-plus-empirical, and review-plus-theory papers",
        "Nominal category profile; no ordinal direction.",
        ["focal paper full text or source-matched article-type coding"],
        "Paper research-design and content classifications",
        "Source evidence: “We used k - 1 dummy variables to model article type, including quantitative, qualitative, theory, review, commentary/discussion, methodological, and agent-based simulation articles.”",
        "The source-authorized category system is T0 but the approved inventory has no article-type/full-text coding field. Venue, subfield, or title metadata would be prohibited proxies.",
    ),
    "CFT_b528bb4021449378": unavailable_review(
        "measurement_scale_type",
        "测量量表类型",
        "measurement_scale_type_k(p) = 1 if p uses source scale category k in {new, original, modified, not_applicable}; otherwise 0; categories may co-occur.",
        "binary indicator per scale-type category",
        "source definitions for new, original, modified, and not-applicable scale use",
        "Nominal non-exclusive profile; no ordinal direction.",
        ["focal paper methods/measures text", "source-matched scale coding"],
        "Paper research-design and content classifications",
        "Source evidence: “we used the following independent categories, each modeled separately as one dummy variable: new scale, original scale, modified scale, or not applicable.”",
        "No approved local methods-text, measures-text, or scale-coding field exists. No bibliographic or topic field may be substituted.",
    ),
    "CFT_fe3f0eea960e7fdf": unavailable_review(
        "lagged_author_citation_average",
        "作者既往引文均值",
        "lagged_author_citation_average(p) = mean(citations(a, before publication_year(p)) for a in authors(p)).",
        "mean pre-publication citation count per listed author",
        "author identity resolution; strict before-publication-year cutoff; full author citation histories",
        "Higher value denotes greater pre-publication team citation standing; opportunity context only.",
        [
            "focal paper authors",
            "author citation histories strictly before focal-paper T0",
        ],
        "Author-team prior scholarly standing",
        "Source evidence: “We averaged the citations of all coauthors for a particular article to reflect how well the authors were collectively cited.”",
        "The inventory has no author-level pre-publication citation histories or a mean aggregation. log_team_prior_nature_output_max is a different maximum output quantity and cannot be used as a proxy.",
    ),
    "CFT_720bc2db6b4d3751": unavailable_review(
        "lagged_author_publication_average",
        "作者既往发表均值",
        "lagged_author_publication_average(p) = mean(publication_count(a, before publication_year(p)) for a in authors(p)).",
        "mean pre-publication publication count per listed author",
        "author identity resolution; strict before-publication-year cutoff; full author publication histories",
        "Higher value denotes greater pre-publication team publication standing; opportunity context only.",
        [
            "focal paper authors",
            "author publication histories strictly before focal-paper T0",
        ],
        "Author-team prior scholarly standing",
        "Source evidence: “we tallied the number of articles authors had published and averaged this number for author teams.”",
        "The inventory has no author-level pre-publication publication histories or a team mean. Any team-output maximum is a non-equivalent proxy and is prohibited.",
    ),
    "CFT_efe86ddd32862286": unavailable_review(
        "issue_number",
        "期号",
        "issue_number_indicator_k(p) = 1 if focal paper p has issue category k; otherwise 0, represented by k-1 dummies.",
        "binary indicator per issue category",
        "journal issue identity; source reference category",
        "Nominal issue category; no ordinal direction.",
        ["focal paper journal issue metadata"],
        "Static paper bibliographic metadata",
        "Source evidence: “We used k - 1 dummy variables to control for unobserved heterogeneity due to issue number.”",
        "The approved inventory has venue_family but no focal-paper issue number. Venue identity is not an issue-number substitute.",
    ),
    "CFT_eedebcf65f24a45f": review(
        H1_canonical_name_en="reference_count",
        H1_label_zh="参考文献数量",
        H1_formula="reference_count(p) = expm1(log_reference_count(p)); this is the audited exact numeric representation of the source count of focal-paper references.",
        H1_units="count of cited references per focal paper",
        H1_parameters="focal paper; audited backward-reference edge representation; no citation, reference-age, or popularity proxy",
        H1_direction="Higher value denotes more focal-paper references; context control only.",
        H1_missing_rule="Return missing when log_reference_count is missing or no backward edges are observed; never recode an unobserved edge set to zero.",
        H1_required_data_json=json_value(
            ["control_features.paper_id", "control_features.log_reference_count"]
        ),
        H1_research_group="Static paper bibliographic metadata",
        H1_research_group_evidence="Source evidence: “We tallied the number of cited references in the article as reported in the reference list (or footnoted in some cases).”",
        H1_data_match_decision="audited_exact_numeric_representation_equivalence_with_audited_coverage",
        H1_local_source_ids_json=json_value(["control_features"]),
        H1_local_columns_json=json_value(["paper_id", "log_reference_count"]),
        H1_derivation_description="Compute expm1(log_reference_count) only in audited coverage. The audit reports 354485 overlap rows, 1.0 exact equality, maximum absolute difference 2.7284841053187847e-12, and 0.8614668643223408 control coverage.",
        H1_formalization_decision="promote_for_formalization",
        H1_rationale="The source specifies a focal-paper backward-reference count at publication. The approved audit authorizes this exact representation transform and explicitly requires missing, rather than zero, when no edges are observed.",
    ),
    "CFT_b0d5c4c71e593252": unavailable_review(
        "study_temporal_context",
        "研究时间语境",
        "study_temporal_context_k(p) = 1 if p has source category k in {cross_sectional, two_time_periods, longitudinal, not_applicable}; otherwise 0; categories may co-occur.",
        "binary indicator per temporal-context category",
        "source definitions for cross-sectional, two-time-period, longitudinal, and not-applicable contexts",
        "Nominal non-exclusive profile; no ordinal direction.",
        ["focal paper methods text", "source-matched temporal-context coding"],
        "Paper research-design and content classifications",
        "Source evidence: “we used the following independent categories, each modeled separately as one dummy variable: cross-sectional, two time periods, longitudinal, or not applicable.”",
        "No local methods-text or temporal-study-design coding field is present. Publication year and reference years cannot substitute for the paper's study temporal context.",
    ),
    "CFT_7acd06bf4888fec8": unavailable_review(
        "study_count_category",
        "研究数量类别",
        "study_count_category_k(p) = 1 if p belongs to source category k in {one, two, three, four_or_more, not_applicable}; otherwise 0, represented by k-1 dummies.",
        "binary indicator per study-count category",
        "source capped study-count categories and study-boundary coding",
        "Ordered capped category profile; higher categories denote more studies up to four-or-more.",
        ["focal paper full text", "source-matched study-count coding"],
        "Paper research-design and content classifications",
        "Source evidence: “We used the following five categories (modeled as k-1 dummy variables): one, two, three, four or more studies, or not applicable.”",
        "No approved local full-text or study-count field exists. Author count, reference count, or paper length would be prohibited substitutes.",
    ),
}


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON support artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows() -> tuple[list[str], list[dict[str, str]]]:
    """Read the frozen batch input."""
    with INPUT_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Batch 4 input has no header.")
        return reader.fieldnames, list(reader)


def validate_support_artifacts(
    inventory: dict[str, Any], audit: dict[str, Any], protocol: dict[str, Any]
) -> None:
    """Validate the local inventory and the narrow equivalence authority."""
    controls = inventory["sources"]["control_features"]
    if (
        controls["sha256"]
        != "51daf5fcf8278210beaccedd18f96fac71b3a45bce1ed3b7c7ee77cb18f423d9"
    ):
        raise ValueError("Unexpected control-feature inventory hash.")
    if audit["author_count"]["status"] != "exact_numeric_representation_equivalence":
        raise ValueError("Author-count equivalence audit is unavailable.")
    if (
        audit["reference_count"]["status"]
        != "exact_numeric_representation_equivalence_with_audited_coverage"
    ):
        raise ValueError("Reference-count equivalence audit is unavailable.")
    if (
        "author_count = expm1(log_author_count), subject to the operational-equivalence audit"
        not in protocol["initially_eligible_quantities"]
    ):
        raise ValueError("Protocol does not authorize the author-count transform.")


def validate(
    input_fields: list[str], before: list[dict[str, str]], after: list[dict[str, str]]
) -> None:
    """Verify frozen-field preservation and H1 decisions."""
    data_decisions = {
        "audited_exact_numeric_representation_equivalence",
        "audited_exact_numeric_representation_equivalence_with_audited_coverage",
        "no_local_match_retain_evidence_gap",
    }
    formalization_decisions = {
        "promote_for_formalization",
        "retain_evidence_gap",
        "reject",
    }
    if len(before) != 9 or len(after) != 9:
        raise ValueError("Expected nine formalization candidates.")
    if {row["candidate_id"] for row in before} != set(REVIEWS):
        raise ValueError("Candidate IDs do not match the H1 reviews.")
    for original, completed in zip(before, after, strict=True):
        for field in input_fields:
            if original[field] != completed[field]:
                raise ValueError(f"Frozen field changed: {field}")
        if set(completed) != {*input_fields, *H1_FIELDS}:
            raise ValueError("Unexpected H1 output schema.")
        if completed["H1_data_match_decision"] not in data_decisions:
            raise ValueError("Invalid data match decision.")
        if completed["H1_formalization_decision"] not in formalization_decisions:
            raise ValueError("Invalid formalization decision.")
        if not all(completed[field] for field in H1_FIELDS):
            raise ValueError("An H1 field is blank.")


def write_csv(fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write the completed formalization output."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Create the independent H1 formalization artifact and provenance manifest."""
    inventory = read_json(INVENTORY_PATH)
    audit = read_json(AUDIT_PATH)
    protocol = read_json(PROTOCOL_PATH)
    validate_support_artifacts(inventory, audit, protocol)
    fields, rows = read_rows()
    completed = [{**row, **REVIEWS[row["candidate_id"]]} for row in rows]
    validate(fields, rows, completed)
    write_csv([*fields, *H1_FIELDS], completed)
    decisions = Counter(row["H1_formalization_decision"] for row in completed)
    matches = Counter(row["H1_data_match_decision"] for row in completed)
    manifest: dict[str, Any] = {
        "artifact": OUTPUT_PATH.name,
        "artifact_sha256": sha256(OUTPUT_PATH),
        "batch": 4,
        "candidate_count": len(completed),
        "formalization_decision_counts": dict(sorted(decisions.items())),
        "data_match_decision_counts": dict(sorted(matches.items())),
        "input": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "support_artifacts": {
            "inventory_sha256": sha256(INVENTORY_PATH),
            "audit_sha256": sha256(AUDIT_PATH),
            "protocol_sha256": sha256(PROTOCOL_PATH),
        },
        "reviewer": "H1",
        "schema": "contextual_formalization_h1_batch4_v4",
        "blind_review_constraints": [
            "Used only batch-4 source formula/evidence fields and the approved local T0 inventory, equivalence audit, and protocol amendment.",
            "Did not consult AI, H2, Qwen, or Ollama results.",
            "Only audited formula-preserving representations were mapped locally; no construct proxies were used.",
            "Rows without an exact local T0 match are explicit fail-closed evidence gaps.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
