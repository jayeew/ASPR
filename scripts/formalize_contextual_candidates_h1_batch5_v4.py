"""Create the independent H1 formula and local-T0 mapping formalization for batch 5."""

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
INPUT_PATH: Final = SOURCE_DIR / "outputs/contextual_formalization_input_batch5_v4.csv"
INVENTORY_PATH: Final = SOURCE_DIR / "outputs/local_t0_input_inventory_v4.json"
AUDIT_PATH: Final = SOURCE_DIR / "outputs/operational_equivalence_audit_v4.json"
PROTOCOL_PATH: Final = SOURCE_DIR / "protocol_amendment_v4_operational_equivalence.json"
OUTPUT_PATH: Final = (
    ROOT / "outputs/contextual_formalization_H1_batch5_completed_v4.csv"
)
MANIFEST_PATH: Final = (
    ROOT / "outputs/contextual_formalization_H1_batch5_completed_v4.manifest.json"
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
    """Serialize one structured CSV value deterministically."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def review(**values: str) -> dict[str, str]:
    """Build a schema-complete H1 formalization review."""
    if set(values) != set(H1_FIELDS):
        raise ValueError("H1 review does not match the 16-field contract.")
    return values


def exact_count_review(
    label: str,
    evidence: str,
    count_name: str,
    log_name: str,
    audit_status: str,
    audit_details: str,
    missing_rule: str,
) -> dict[str, str]:
    """Build a formula-preserving exact numeric representation mapping."""
    return review(
        H1_canonical_name_en=count_name,
        H1_label_zh=label,
        H1_formula=f"{count_name}(p) = expm1({log_name}(p)); this is the audited exact numeric representation of the source focal-paper count.",
        H1_units=f"count per focal paper ({'listed authors' if count_name == 'author_count' else 'cited references'})",
        H1_parameters="focal paper; audited exact numeric representation; no adjacent metadata or construct proxy",
        H1_direction=f"Higher value denotes more {'listed authors' if count_name == 'author_count' else 'cited references'}; context control only.",
        H1_missing_rule=missing_rule,
        H1_required_data_json=json_value(
            ["control_features.paper_id", f"control_features.{log_name}"]
        ),
        H1_research_group="Static paper bibliographic metadata",
        H1_research_group_evidence=evidence,
        H1_data_match_decision=audit_status,
        H1_local_source_ids_json=json_value(["control_features"]),
        H1_local_columns_json=json_value(["paper_id", log_name]),
        H1_derivation_description=f"Compute expm1({log_name}). {audit_details}",
        H1_formalization_decision="promote_for_formalization",
        H1_rationale="The source defines the same focal-paper count at T0. The local transform is authorized by the operational-equivalence audit and does not use a proxy.",
    )


def unavailable_review(
    canonical: str,
    label: str,
    formula: str,
    units: str,
    parameters: str,
    direction: str,
    required_data: list[str],
    research_group: str,
    evidence: str,
    rationale: str,
) -> dict[str, str]:
    """Build a fail-closed formalization review for absent source-identical inputs."""
    return review(
        H1_canonical_name_en=canonical,
        H1_label_zh=label,
        H1_formula=formula,
        H1_units=units,
        H1_parameters=parameters,
        H1_direction=direction,
        H1_missing_rule="Return missing. The approved local T0 inventory lacks the source-required input; never impute, recode to zero, or substitute a proxy.",
        H1_required_data_json=json_value(required_data),
        H1_research_group=research_group,
        H1_research_group_evidence=evidence,
        H1_data_match_decision="no_local_match_retain_evidence_gap",
        H1_local_source_ids_json=json_value([]),
        H1_local_columns_json=json_value([]),
        H1_derivation_description="No authorized local T0 mapping exists. The protocol permits only exact source-formula-preserving representations, not adjacent metadata or correlated proxy fields.",
        H1_formalization_decision="retain_evidence_gap",
        H1_rationale=rationale,
    )


AUTHOR_REVIEW: Final = exact_count_review(
    "作者人数",
    "Source evidence: the focal publication/article author list is counted.",
    "author_count",
    "log_author_count",
    "audited_exact_numeric_representation_equivalence",
    "The audit reports 411489 overlap rows, 1.0 exact equality rate, and maximum absolute difference 2.2737367544323206e-12.",
    "Return missing when log_author_count is missing or p is outside audited overlap; never impute or recode to zero.",
)
REFERENCE_REVIEW: Final = exact_count_review(
    "参考文献数量",
    "Source evidence: cited references in the focal publication/article are counted.",
    "reference_count",
    "log_reference_count",
    "audited_exact_numeric_representation_equivalence_with_audited_coverage",
    "The audit reports 354485 overlap rows, 1.0 exact equality, maximum absolute difference 2.7284841053187847e-12, and 0.8614668643223408 control coverage.",
    "Return missing when log_reference_count is missing or no backward edges are observed; never recode an unobserved edge set to zero.",
)

REVIEWS: Final[dict[str, dict[str, str]]] = {
    "CFT_2775fafaf169ca97": AUTHOR_REVIEW,
    "CFT_77e50876d1e4a282": AUTHOR_REVIEW,
    "CFT_87576570954d2346": unavailable_review(
        "international_collaboration",
        "国际合作",
        "international_collaboration(p) = 1 if the number of author-affiliation countries for p is greater than one; 0 if it is one.",
        "binary indicator",
        "author-affiliation country links; country normalization; complete affiliation coverage",
        "1 denotes international and 0 denotes domestic collaboration; context control only.",
        ["focal paper author-affiliation country links"],
        "Paper collaboration geographic scope",
        "Source evidence: “Each article is classified as either domestic (one country) or international (more than one country) based on the data of author affiliation(s).”",
        "The inventory has openalex_country_count and log_country_count, but no equivalence audit establishes that either exactly reproduces the source's author-affiliation country rule. The protocol explicitly prohibits country-count proxy use.",
    ),
    "CFT_8b1b9b194be425a6": unavailable_review(
        "funding_status",
        "资助状态",
        "funding_status(p) = 1 if focal paper p is funded; 0 if nonfunded.",
        "binary indicator",
        "focal paper funding information and source-matched funding classification",
        "1 denotes funded and 0 denotes nonfunded; context control only.",
        ["focal paper funding information"],
        "Paper funding status",
        "Source evidence: “Each article is classified as either funded or nonfunded.”",
        "No funding statement or source-matched funding-status field exists in the approved local T0 inventory. Venue, author, or institution metadata are not substitutes.",
    ),
    "CFT_d91a731d0c26055b": review(
        H1_canonical_name_en="journal_identity",
        H1_label_zh="期刊身份",
        H1_formula="journal_identity_u(p) = 1 if papers_common.source_id(p) equals frozen venue identifier u; 0 otherwise.",
        H1_units="binary indicator per venue identifier",
        H1_parameters="focal paper; frozen venue/source identifier u; reference-category policy",
        H1_direction="Nominal one-hot venue profile; no ordinal direction.",
        H1_missing_rule="Return missing when source_id is absent or no frozen venue identifier is available; never infer a venue from venue_family or another proxy.",
        H1_required_data_json=json_value(
            ["papers_common.paper_id", "papers_common.source_id"]
        ),
        H1_research_group="Journal identity",
        H1_research_group_evidence="Source evidence: “Ju = 1 if the publication p is published by the journal u; 0 otherwise.”",
        H1_data_match_decision="direct_local_t0_field_exact_semantic_match",
        H1_local_source_ids_json=json_value(["papers_common"]),
        H1_local_columns_json=json_value(["paper_id", "source_id"]),
        H1_derivation_description="Use the direct T0 venue/source identifier and emit one indicator per frozen identifier u. This is a direct semantic match, not an aggregation or proxy transform.",
        H1_formalization_decision="promote_for_formalization",
        H1_rationale="The source formula is an equality test on a paper's journal identity, and the inventory supplies a focal-paper source identifier at T0. The category dictionary must be frozen before use.",
    ),
    "CFT_04cc02d4beb257b7": REFERENCE_REVIEW,
    "CFT_1cc3cb5d39df9ac8": REFERENCE_REVIEW,
    "CFT_0238f457adf2f3ef": unavailable_review(
        "title_character_length",
        "标题字符长度",
        "title_character_length(p) = count every character in title(p), including punctuation and spaces.",
        "character count per focal paper title",
        "full title string; character encoding and normalization; punctuation and spaces retained",
        "Higher value denotes a longer title under the stated character rule; context control only.",
        ["focal paper title string"],
        "Static paper bibliographic metadata",
        "Source evidence: “The total number of characters in each title, including punctuations and spaces, is counted.”",
        "The inventory has title_word_count but no title string or character-count field. Word count is not formula-equivalent to a punctuation-and-space-inclusive character count and is prohibited as a proxy.",
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
            raise ValueError("Batch 5 input has no header.")
        return reader.fieldnames, list(reader)


def validate_support_artifacts(
    inventory: dict[str, Any], audit: dict[str, Any], protocol: dict[str, Any]
) -> None:
    """Validate direct and audited local-T0 mapping authorities."""
    source_columns = {
        column["name"] for column in inventory["sources"]["papers_common"]["columns"]
    }
    if {"paper_id", "source_id"} - source_columns:
        raise ValueError("Required direct venue fields are missing from the inventory.")
    if audit["author_count"]["exact_equality_rate"] != 1.0:
        raise ValueError("Author-count audit is not exact.")
    if audit["reference_count"]["exact_equality_rate"] != 1.0:
        raise ValueError("Reference-count audit is not exact.")
    if (
        "country_count as author-country diversity proxy"
        not in protocol["explicitly_not_authorized"]
    ):
        raise ValueError("Country-count proxy prohibition is unavailable.")


def validate(
    input_fields: list[str], before: list[dict[str, str]], after: list[dict[str, str]]
) -> None:
    """Verify frozen-field preservation and all H1 review decisions."""
    matches = {
        "audited_exact_numeric_representation_equivalence",
        "audited_exact_numeric_representation_equivalence_with_audited_coverage",
        "direct_local_t0_field_exact_semantic_match",
        "no_local_match_retain_evidence_gap",
    }
    decisions = {"promote_for_formalization", "retain_evidence_gap", "reject"}
    if len(before) != 8 or len(after) != 8:
        raise ValueError("Expected eight batch-5 candidates.")
    if {row["candidate_id"] for row in before} != set(REVIEWS):
        raise ValueError("Candidate IDs do not match the H1 reviews.")
    for original, completed in zip(before, after, strict=True):
        for field in input_fields:
            if original[field] != completed[field]:
                raise ValueError(f"Frozen field changed: {field}")
        if set(completed) != {*input_fields, *H1_FIELDS}:
            raise ValueError("Unexpected H1 output schema.")
        if completed["H1_data_match_decision"] not in matches:
            raise ValueError("Invalid H1 data match decision.")
        if completed["H1_formalization_decision"] not in decisions:
            raise ValueError("Invalid H1 formalization decision.")
        if not all(completed[field] for field in H1_FIELDS):
            raise ValueError("An H1 field is blank.")


def write_csv(fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write the completed formalization CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Create the independent H1 batch-5 formalization and manifest."""
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
        "batch": 5,
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
        "schema": "contextual_formalization_h1_batch5_v4",
        "blind_review_constraints": [
            "Used only batch-5 source formula/evidence fields and the approved local T0 inventory, audit, and protocol amendment.",
            "Did not consult AI, H2, Qwen, or Ollama results.",
            "Only exact audited representations or direct source-identical local fields were mapped.",
            "Missing source inputs remain fail-closed evidence gaps; no proxies were substituted.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
