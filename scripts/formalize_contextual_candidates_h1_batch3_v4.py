"""Write the independent H1 formalization review for contextual batch 3."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Final

ROOT: Final = Path("/home/jayee/workspace/ASPR")
INPUT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_formalization_input_batch3_v4.csv"
)
INVENTORY_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "local_t0_input_inventory_v4.json"
)
AUDIT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "operational_equivalence_audit_v4.json"
)
PROTOCOL_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/"
    "protocol_amendment_v4_operational_equivalence.json"
)
OUTPUT_PATH: Final = (
    ROOT / "outputs/contextual_formalization_H1_batch3_completed_v4.csv"
)
MANIFEST_PATH: Final = (
    ROOT / "outputs/contextual_formalization_H1_batch3_completed_v4.manifest.json"
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
    """Encode one structured CSV value deterministically."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review(**values: str) -> dict[str, str]:
    """Build a schema-complete H1 review row."""
    if set(values) != set(H1_FIELDS):
        raise ValueError("H1 review does not match the 16-field contract")
    return values


REVIEWS: Final[dict[str, dict[str, str]]] = {
    "CFT_a203e78cdff6e8fb": review(
        H1_canonical_name_en="author_count",
        H1_label_zh="作者人数",
        H1_formula=(
            "author_count(p) = expm1(log_author_count(p)); this is the audited "
            "numeric representation of the source quantity, the number of listed "
            "authors in focal paper p."
        ),
        H1_units="count of listed authors per focal paper",
        H1_parameters=(
            "focal paper; audited T0 OpenAlex author-count representation; complete "
            "source-record scope; no proxy for institution or country count"
        ),
        H1_direction="Higher value denotes a larger authorship team; contextual control only.",
        H1_missing_rule=(
            "Return missing when log_author_count is missing or the paper is outside "
            "the audited overlap; never impute or recode to zero. The audit records "
            "411489 overlap rows with 1.0 exact equality."
        ),
        H1_required_data_json=json_value(
            [
                "control_features.log_author_count",
                "target_openalex_metadata.openalex_author_count",
            ]
        ),
        H1_research_group=(
            "All distinct PubMed entries retrieved for publication years 2000–2021; "
            "the source analyzes 17,015,001 biomedical articles published 2000–2020."
        ),
        H1_research_group_evidence=(
            "The source Methods (p. 1303) says its extraction returned all distinct "
            "PubMed entries for 2000–2021 with 'Author number per record' and a "
            "complete-author-list check; it excludes records without listed authors."
        ),
        H1_data_match_decision="audited_exact_numeric_representation_equivalence",
        H1_local_source_ids_json=json_value(
            ["control_features", "target_openalex_metadata"]
        ),
        H1_local_columns_json=json_value(
            [
                "control_features.log_author_count",
                "target_openalex_metadata.openalex_author_count",
                "target_openalex_metadata.paper_id",
            ]
        ),
        H1_derivation_description=(
            "The operational-equivalence audit derives expm1(log_author_count) and "
            "reports exact equality to openalex_author_count for all 411489 audited "
            "overlap rows (rate 1.0; maximum absolute difference "
            "2.2737367544323206e-12), using only T0, outcome-blind inputs."
        ),
        H1_formalization_decision="promote_for_formalization",
        H1_rationale=(
            "The original definition is a focal-paper author count, and the active "
            "amendment specifically authorizes this exact audited representation "
            "transform. T0 provenance and a fail-closed missing rule are recorded."
        ),
    ),
    "CFT_01ebf955fc5d5c9e": review(
        H1_canonical_name_en="language_rank",
        H1_label_zh="论文语言等级",
        H1_formula="language_rank(p) = 1 if English; 2 if bilingual; 3 otherwise.",
        H1_units="ordered language-category code",
        H1_parameters="English/bilingual/other coding boundary and manuscript language.",
        H1_direction="Higher code denotes the source's less-English language category.",
        H1_missing_rule=(
            "Absent in the source. Do not impute records with indeterminate or "
            "multi-language status."
        ),
        H1_required_data_json=json_value(["focal-paper manuscript language"]),
        H1_research_group=(
            "PubMed biomedical papers assembled from the PubMed Knowledge Graph and "
            "NIH Open Citation Collection; the source's citation dataset covers "
            "1998–2018."
        ),
        H1_research_group_evidence=(
            "Source Table 1 (p. 6040) defines Language as a categorical rank: "
            "1 English, 2 English/Other, 3 Other."
        ),
        H1_data_match_decision="no_local_match_retain_evidence_gap",
        H1_local_source_ids_json=json_value([]),
        H1_local_columns_json=json_value([]),
        H1_derivation_description=(
            "The T0 inventory contains no source-matched manuscript-language field "
            "or audited transformation for this three-level coding."
        ),
        H1_formalization_decision="retain_evidence_gap",
        H1_rationale=(
            "The source definition is clear but lacks a local, source-matched T0 "
            "representation and missing-data rule; it is not one of the two "
            "authorized audited count transforms."
        ),
    ),
    "CFT_596f832ef25cc770": review(
        H1_canonical_name_en="publication_year",
        H1_label_zh="发表年份",
        H1_formula="publication_year(p) = publication year of focal paper p.",
        H1_units="calendar year",
        H1_parameters=(
            "publication-date precedence when online-first, issue, acceptance, and "
            "other dates disagree"
        ),
        H1_direction="Later calendar year denotes later publication timing; contextual control only.",
        H1_missing_rule=(
            "Absent in the source. Do not impute missing or conflicting publication "
            "dates until a source-matched date-precedence rule is set."
        ),
        H1_required_data_json=json_value(
            ["source-matched focal-paper publication date"]
        ),
        H1_research_group=(
            "PubMed biomedical papers assembled from the PubMed Knowledge Graph and "
            "NIH Open Citation Collection; the source's citation dataset covers "
            "1998–2018."
        ),
        H1_research_group_evidence=(
            "Source Table 1 (p. 6040) specifies Year as the categorical 'Year of "
            "publication.'"
        ),
        H1_data_match_decision="local_publication_year_not_audited_retain_evidence_gap",
        H1_local_source_ids_json=json_value(["control_features", "papers_common"]),
        H1_local_columns_json=json_value(
            ["control_features.publication_year", "papers_common.publication_year"]
        ),
        H1_derivation_description=(
            "Local T0 tables store publication_year, but no operational-equivalence "
            "audit demonstrates that their date precedence and missing handling match "
            "the source's categorical year."
        ),
        H1_formalization_decision="retain_evidence_gap",
        H1_rationale=(
            "A similarly named local field is not enough: source date precedence and "
            "missing treatment are unresolved, and no audit authorizes this mapping."
        ),
    ),
    "CFT_400d27297aa6d879": review(
        H1_canonical_name_en="reference_count",
        H1_label_zh="参考文献数量",
        H1_formula=(
            "reference_count(p) = expm1(log_reference_count(p)); this is the "
            "audited numeric representation of the source quantity, the number of "
            "references in focal paper p."
        ),
        H1_units="count of focal-paper backward references",
        H1_parameters=(
            "focal-paper backward reference edges; audited coverage only; no "
            "substitution of an unobserved bibliography with zero"
        ),
        H1_direction="Higher value denotes a larger focal-paper reference list; contextual control only.",
        H1_missing_rule=(
            "No observed backward edges is missing/unknown, never zero. Rows outside "
            "the audited coverage remain missing. The audit records 354485 overlap "
            "rows and 0.8614668643223408 control coverage."
        ),
        H1_required_data_json=json_value(
            [
                "control_features.log_reference_count",
                "paper_references.paper_id",
                "paper_references.reference_id",
            ]
        ),
        H1_research_group=(
            "PubMed biomedical papers assembled from the PubMed Knowledge Graph and "
            "NIH Open Citation Collection; the source's citation dataset covers "
            "1998–2018."
        ),
        H1_research_group_evidence=(
            "Source Table 1 (p. 6040) defines References as the numeric 'Number of "
            "references'; the empirical setting states that the PubMed Knowledge "
            "Graph supplies citation relationships among PubMed papers."
        ),
        H1_data_match_decision="audited_exact_numeric_representation_equivalence",
        H1_local_source_ids_json=json_value(["control_features", "paper_references"]),
        H1_local_columns_json=json_value(
            [
                "control_features.log_reference_count",
                "paper_references.paper_id",
                "paper_references.reference_id",
            ]
        ),
        H1_derivation_description=(
            "The operational-equivalence audit derives expm1(log_reference_count) "
            "and reports exact equality to the focal-paper backward-edge count for "
            "all 354485 audited overlap rows (rate 1.0; maximum absolute difference "
            "2.7284841053187847e-12), with the stated fail-closed missing rule."
        ),
        H1_formalization_decision="promote_for_formalization",
        H1_rationale=(
            "The source quantity, T0 backward-edge count, audited inverse transform, "
            "and missing rule all match the active amendment's admissibility test."
        ),
    ),
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read one UTF-8 CSV while retaining its original field order."""
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    """Write CSV records in a fixed schema."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Validate evidence inputs and write the frozen-column-preserving review."""
    input_columns, input_rows = read_csv(INPUT_PATH)
    if len(input_rows) != 4:
        raise ValueError("Expected exactly four batch-3 candidates")
    candidate_ids = {row["candidate_id"] for row in input_rows}
    if candidate_ids != set(REVIEWS):
        raise ValueError("Review registry does not exactly cover the input candidates")

    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if inventory["schema_version"] != "local_t0_input_inventory_v4":
        raise ValueError("Unexpected local T0 inventory schema")
    if audit["schema_version"] != "operational_equivalence_audit_v4":
        raise ValueError("Unexpected operational-equivalence audit schema")
    if protocol["schema_version"] != "protocol_amendment_v4_operational_equivalence":
        raise ValueError("Unexpected operational-equivalence protocol schema")
    if audit["outcome_columns_used"] or inventory["outcome_columns_used"]:
        raise ValueError("Formalization evidence must remain outcome blind")
    if audit["author_count"]["exact_equality_rate"] != 1.0:
        raise ValueError("Author-count audit is not exact")
    if audit["reference_count"]["exact_equality_rate"] != 1.0:
        raise ValueError("Reference-count audit is not exact")

    for row in input_rows:
        fulltext = Path(row["fulltext_local_path"])
        if sha256(fulltext) != row["fulltext_sha256"]:
            raise ValueError(f"Full-text hash mismatch for {row['candidate_id']}")

    output_rows = [{**row, **REVIEWS[row["candidate_id"]]} for row in input_rows]
    write_csv(OUTPUT_PATH, [*input_columns, *H1_FIELDS], output_rows)
    decision_counts = {
        decision: sum(
            row["H1_formalization_decision"] == decision for row in output_rows
        )
        for decision in ("promote_for_formalization", "retain_evidence_gap")
    }
    manifest = {
        "artifact": "contextual_formalization_H1_batch3_completed_v4",
        "reviewer": "H1",
        "input_path": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "inventory_path": str(INVENTORY_PATH),
        "inventory_sha256": sha256(INVENTORY_PATH),
        "operational_equivalence_audit_path": str(AUDIT_PATH),
        "operational_equivalence_audit_sha256": sha256(AUDIT_PATH),
        "protocol_amendment_path": str(PROTOCOL_PATH),
        "protocol_amendment_sha256": sha256(PROTOCOL_PATH),
        "output_path": str(OUTPUT_PATH),
        "output_sha256": sha256(OUTPUT_PATH),
        "candidate_rows": len(output_rows),
        "h1_field_count": len(H1_FIELDS),
        "formalization_decision_counts": decision_counts,
        "fulltext_sha256_verified": True,
        "qwen_or_ollama_used": False,
        "blind_review_constraints": [
            "Used only the batch-3 input, listed English full texts, local T0 inventory, operational-equivalence audit, and active protocol amendment.",
            "Copied frozen input fields verbatim without adjudicating AI/H2 content.",
            "Did not use AI/H2 review output, Qwen, or Ollama.",
            "Promoted only the two quantities explicitly authorized by the exact representation audit; all remaining candidates retain an evidence gap.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
