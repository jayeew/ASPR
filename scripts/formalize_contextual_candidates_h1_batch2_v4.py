"""Write the independent H1 formalization review for contextual batch 2."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path("/home/jayee/workspace/ASPR")
INPUT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_formalization_input_batch2_v4.csv"
)
INVENTORY_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "local_t0_input_inventory_v4.json"
)
OUTPUT_PATH: Final = (
    ROOT / "outputs/contextual_formalization_H1_batch2_completed_v4.csv"
)
MANIFEST_PATH: Final = (
    ROOT / "outputs/contextual_formalization_H1_batch2_completed_v4.manifest.json"
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

PDFS: Final = {
    "doi:10.1007/s11135-022-01480-z": ROOT
    / "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_open_fulltexts_v4/694d673d5e0b61ddd948.pdf",
    "doi:10.1007/s11192-021-04128-1": ROOT
    / "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_open_fulltexts_v4/351d0f7158e241c13a81.pdf",
    "doi:10.1057/s41599-024-02915-8": ROOT
    / "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "open_fulltexts/415b76e2cf915e971a7f.pdf",
    "doi:10.1007/s11192-017-2630-5": ROOT
    / "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_open_fulltexts_v4/5485bca2b93e0ee5c059.pdf",
}


def json_value(value: object) -> str:
    """Encode one structured CSV cell deterministically."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review(**values: str) -> dict[str, str]:
    """Build a schema-complete H1 review row."""
    if set(values) != set(H1_FIELDS):
        raise ValueError("H1 review does not match the 16-field contract")
    return values


def author_review(group: str, evidence: str) -> dict[str, str]:
    """Formalize author count without treating local OpenAlex count as equivalent."""
    return review(
        H1_canonical_name_en="author_count",
        H1_label_zh="作者人数",
        H1_formula="author_count(p) = number of authors on focal paper p; source operationalization is 'No. of authors' / 'number of co-authors in publication i'.",
        H1_units="count of authors per paper",
        H1_parameters="focal-paper byline; source database author-record policy",
        H1_direction="Higher value denotes a larger authorship team; contextual control only.",
        H1_missing_rule="Absent. Do not impute; retain missing when no complete source-author byline is available. The source does not state handling of group/consortium authors, duplicate identities, or partial bylines.",
        H1_required_data_json=json_value(["focal-paper complete author byline"]),
        H1_research_group=group,
        H1_research_group_evidence=evidence,
        H1_data_match_decision="local_raw_author_count_not_strictly_equivalent_retain_evidence_gap",
        H1_local_source_ids_json=json_value(
            ["target_openalex_metadata", "control_features"]
        ),
        H1_local_columns_json=json_value(
            [
                "target_openalex_metadata.openalex_author_count",
                "target_openalex_metadata.openalex_metadata_found",
                "control_features.log_author_count",
            ]
        ),
        H1_derivation_description="Local OpenAlex exposes a raw author count and a transformed log count, but the source does not specify group-author, duplicate, or incomplete-byline policy. Cross-database count provenance is therefore not strictly equivalent.",
        H1_formalization_decision="retain_evidence_gap",
        H1_rationale="The source applies author count, but neither its count boundary nor the local OpenAlex counting boundary is fully specified. Do not silently substitute or invert the transformed local feature.",
    )


def page_review(group: str, evidence: str) -> dict[str, str]:
    """Record paper-length evidence and its absent local T0 mapping."""
    return review(
        H1_canonical_name_en="article_page_count",
        H1_label_zh="论文页数",
        H1_formula="Source operationalization: 'No. of article pages' / 'Paper length (number of pages)'; no page-range arithmetic or article-number rule is stated.",
        H1_units="pages per paper",
        H1_parameters="version of record; page/article-number convention; supplement policy",
        H1_direction="Higher value denotes a longer paper; contextual control only.",
        H1_missing_rule="Absent. Do not impute; retain missing for article-number-only records or unavailable page ranges until a source-authorized conversion is supplied.",
        H1_required_data_json=json_value(
            ["focal-paper version-of-record pagination or page count"]
        ),
        H1_research_group=group,
        H1_research_group_evidence=evidence,
        H1_data_match_decision="no_local_match_retain_evidence_gap",
        H1_local_source_ids_json=json_value([]),
        H1_local_columns_json=json_value([]),
        H1_derivation_description="The inventory contains no focal-paper page count, start page, end page, or version-of-record pagination field.",
        H1_formalization_decision="retain_evidence_gap",
        H1_rationale="No local T0 field can reproduce the source's page-count construct, and its treatment of article numbers is not specified.",
    )


def workflow_review(
    canonical: str,
    label: str,
    formula: str,
    units: str,
    parameters: str,
    direction: str,
    missing_rule: str,
    required_data: list[str],
) -> dict[str, str]:
    """Record a source-authorized publisher-workflow construct as an evidence gap."""
    return review(
        H1_canonical_name_en=canonical,
        H1_label_zh=label,
        H1_formula=formula,
        H1_units=units,
        H1_parameters=parameters,
        H1_direction=direction,
        H1_missing_rule=missing_rule,
        H1_required_data_json=json_value(required_data),
        H1_research_group="Original articles submitted to one anonymous Business and Management journal, 2010–2015; workflow analyses focus on papers with a formal acceptance decision.",
        H1_research_group_evidence="The source says its publisher supplied peer-review records for all papers over six years, identifies 598 submissions, and restricts turnaround/workflow analysis to papers formally accepted at the March 2017 download (Methods pp. 1094–1096).",
        H1_data_match_decision="no_local_match_retain_evidence_gap",
        H1_local_source_ids_json=json_value([]),
        H1_local_columns_json=json_value([]),
        H1_derivation_description="The inventory has publication metadata, backward references, and author/affiliation counts only; it has no publisher editorial-event history, decision trail, submission sequence, or reviewer panel field.",
        H1_formalization_decision="retain_evidence_gap",
        H1_rationale="This is a source-authorized pre-publication workflow construct but cannot be mapped to a local outcome-blind T0 field.",
    )


def reference_count_review(group: str, evidence: str) -> dict[str, str]:
    """Record reference-list-size evidence without conflating local citation edges with it."""
    return review(
        H1_canonical_name_en="reference_count",
        H1_label_zh="参考文献数量",
        H1_formula="reference_count(p) = number of references in focal publication p; source operationalization is 'No. of references' / 'references_count_i the number of references in publication i'.",
        H1_units="count of references per paper",
        H1_parameters="focal bibliography; cited-work record linkage and duplicate-reference policy",
        H1_direction="Higher value denotes a larger reference list; contextual control only.",
        H1_missing_rule="Absent. Do not impute; retain missing when the focal bibliography is unavailable or incomplete. Source selection exclusions for missing records are not a reusable missing-value rule.",
        H1_required_data_json=json_value(["complete focal-paper bibliography"]),
        H1_research_group=group,
        H1_research_group_evidence=evidence,
        H1_data_match_decision="local_backward_edges_not_strictly_equivalent_retain_evidence_gap",
        H1_local_source_ids_json=json_value(["paper_references", "control_features"]),
        H1_local_columns_json=json_value(
            [
                "paper_references.paper_id",
                "paper_references.reference_id",
                "control_features.log_reference_count",
            ]
        ),
        H1_derivation_description="Local paper_references can yield a linked-edge count and control_features stores only a log transform. Neither inventory entry establishes equality to the source database's complete bibliography or duplicate/unlinked-reference treatment.",
        H1_formalization_decision="retain_evidence_gap",
        H1_rationale="A linked-edge count or log-transformed count is not a demonstrated strict substitute for the source's reference-list count.",
    )


REVIEWS: Final[dict[str, dict[str, str]]] = {
    "CFT_229be64efded8380": author_review(
        "Highly cited papers in the Journal of the American Medical Informatics Association, extracted from Web of Science on 1 June 2019.",
        "The source states that data for highly cited papers of the Journal of the American Medical Informatics Association were extracted from Web of Science, and Table 1 defines 'Co-authors: No. of authors' (Methods §3.1–3.2; Table 1, p. 3694).",
    ),
    "CFT_2db7fe601f83d7d0": author_review(
        "Publication corpora of positive-deviant and non-positive-deviant researchers affiliated with Egyptian universities; final corpus 876 unique publications.",
        "The source's Stage 3 uses papers as the unit of analysis, retrieves co-authors through Scopus, and lists 'Number of authors' as a paper-extrinsic predictor (pp. 8404–8405).",
    ),
    "CFT_6349de9d481d1df1": author_review(
        "COVID-19 journal publications in the OpenAlex/Overton focal dataset from January 2020 through December 2021.",
        "The source defines team_size_i as 'the number of co-authors in publication i' and applies it as a control variable (Variables and regression models, p. 4).",
    ),
    "CFT_e018de3163123a9b": author_review(
        "Original articles in one anonymous Business and Management journal; author count was cross-checked with Scopus.",
        "The source lists 'Number of Authors' among fields cross-checked with Scopus and uses count of authors as a control (Data set, p. 1095; Developing a model, p. 1094).",
    ),
    "CFT_3217066e07b373b1": page_review(
        "Highly cited papers in the Journal of the American Medical Informatics Association, extracted from Web of Science on 1 June 2019.",
        "Table 1 labels the feature 'Article length: No. of article pages' (p. 3694).",
    ),
    "CFT_36bd47b47e562a9b": page_review(
        "Publication corpora of positive-deviant and non-positive-deviant researchers affiliated with Egyptian universities; final corpus 876 unique publications.",
        "Table 11 applies 'Paper length (number of pages)' as a paper-extrinsic predictor (p. 8405).",
    ),
    "CFT_14abd803b2912e4e": workflow_review(
        "peer_review_turnaround_time_days",
        "同行评审周转时长",
        "turnaround_time(p) = time paper p spends in review from receipt for review to acceptance.",
        "elapsed time; source results report days",
        "receipt-for-review timestamp; acceptance timestamp; calendar/time-zone convention",
        "Higher value denotes longer time in review.",
        "Absent. Do not impute; retain missing if either endpoint is unavailable. The source excludes papers still in revision or reject-and-resubmit at download rather than defining a reusable missing rule.",
        ["publisher receipt-for-review timestamp", "publisher acceptance timestamp"],
    ),
    "CFT_18f4dbeb8d152546": workflow_review(
        "peer_review_turnaround_time_days",
        "同行评审周转时长",
        "turnaround_time(p) = time paper p spends in review from receipt for review to acceptance.",
        "elapsed time; source results report days",
        "receipt-for-review timestamp; acceptance timestamp; calendar/time-zone convention",
        "Higher value denotes longer time in review.",
        "Absent. Do not impute; retain missing if either endpoint is unavailable. The source excludes papers still in revision or reject-and-resubmit at download rather than defining a reusable missing rule.",
        ["publisher receipt-for-review timestamp", "publisher acceptance timestamp"],
    ),
    "CFT_202163a521c5f471": workflow_review(
        "initial_editorial_decision_category",
        "首次编辑决定类别",
        "initial_editorial_decision(p) ∈ {Accept, Minor Revision, Major Revision, Reject and Resubmit}.",
        "nominal four-category editorial decision",
        "first recorded editorial decision only",
        "No scalar direction; categorical workflow history.",
        "Absent. Do not impute or recode; retain missing when a first editorial decision is not recorded. The source's accepted-paper restriction is not a missing-data rule.",
        ["publisher first editorial decision record"],
    ),
    "CFT_21ac5535ef55a900": workflow_review(
        "submission_count",
        "投稿次数",
        "submission_count(p) = publisher-recorded number of submissions for p.",
        "count of submissions",
        "journal transfer, resubmission, and withdrawal inclusion policy",
        "Higher value denotes more publisher-recorded submissions.",
        "Absent. Do not impute; retain missing if workflow history is unavailable. The source does not define treatment of transfers, withdrawals, or resubmissions to other journals.",
        ["publisher submission-event history"],
    ),
    "CFT_88b0d5a0bef4c8ef": workflow_review(
        "initial_editorial_decision_category",
        "首次编辑决定类别",
        "initial_editorial_decision(p) ∈ {Accept, Minor Revision, Major Revision, Reject and Resubmit}.",
        "nominal four-category editorial decision",
        "first recorded editorial decision only",
        "No scalar direction; categorical workflow history.",
        "Absent. Do not impute or recode; retain missing when a first editorial decision is not recorded. The source's accepted-paper restriction is not a missing-data rule.",
        ["publisher first editorial decision record"],
    ),
    "CFT_d16bd6f597661fef": workflow_review(
        "non_editor_reviewer_count",
        "非编辑审稿人数量",
        "non_editor_reviewer_count(p) = number of reviewers in addition to the editor involved in review of p.",
        "count of non-editor reviewers",
        "reviewer identity/repeat-reviewer policy; editor inclusion excluded by source",
        "Higher value denotes more non-editor reviewers involved.",
        "Absent. Do not impute; retain missing if reviewer records are unavailable. The source does not specify treatment of repeated reviewers or anonymous panel records.",
        ["publisher reviewer-assignment history", "editor role marker"],
    ),
    "CFT_f7a477f40a9ceb5e": workflow_review(
        "revision_effort_category",
        "修订工作量类别",
        "revision_effort(p) = low if initial decision ∈ {Accept, Minor Revision}; high if initial decision ∈ {Major Revision, Reject and Resubmit}.",
        "binary low/high revision-effort category",
        "source's four-category first decision and stated two-group collapse",
        "High denotes the source's major-revision/reject-and-resubmit group.",
        "Absent. Do not impute; retain missing if original decision is missing or outside the four stated labels. Do not apply the grouping to recoded decision taxonomies without source-equivalent provenance.",
        ["publisher first editorial decision record"],
    ),
    "CFT_7b16db0737deb0d0": reference_count_review(
        "Highly cited papers in the Journal of the American Medical Informatics Association, extracted from Web of Science on 1 June 2019.",
        "Table 1 applies 'No. of references' among document-related factors extracted from Web of Science (Methods §3.2; Table 1, p. 3694).",
    ),
    "CFT_7c87d8eb4b4b6bb1": reference_count_review(
        "COVID-19 journal publications in the OpenAlex/Overton focal dataset from January 2020 through December 2021.",
        "The source defines references_count_i as 'the number of references in publication i' (Variables and regression models, p. 4).",
    ),
    "CFT_b825068902f2938b": reference_count_review(
        "Publication corpora of positive-deviant and non-positive-deviant researchers affiliated with Egyptian universities; final corpus 876 unique publications.",
        "The source retrieves publication information with the Scopus API and lists 'Number of references' as a paper-extrinsic predictor (pp. 8404–8405).",
    ),
    "CFT_baf488321880a2a5": review(
        H1_canonical_name_en="average_reference_publication_year_offset",
        H1_label_zh="参考文献平均发表年份差",
        H1_formula="average_reference_year_offset(p) = mean(publication_year(r) for r in references(p)) - publication_year(p).",
        H1_units="years; source-signed year offset",
        H1_parameters="focal publication year; reference publication years; reference inclusion policy",
        H1_direction="Higher value (less negative) denotes references newer on average under the source's signed expression; contextual control only.",
        H1_missing_rule="Table 1 reports zero missing values for the source dataset but gives no rule for an undated or unlinked reference. Do not impute; retain missing if any required included reference year is unavailable until a source-equivalent exclusion rule exists.",
        H1_required_data_json=json_value(
            [
                "focal-paper publication year",
                "complete focal-paper bibliography",
                "publication year for each included reference",
            ]
        ),
        H1_research_group="Highly cited papers in the Journal of the American Medical Informatics Association, extracted from Web of Science on 1 June 2019.",
        H1_research_group_evidence="Table 1 defines 'Average age of the references' as 'Average year of publication of references for each document minus the year of publication of that document' (p. 3694).",
        H1_data_match_decision="local_reference_year_inputs_not_strictly_equivalent_retain_evidence_gap",
        H1_local_source_ids_json=json_value(
            [
                "papers_common",
                "paper_references",
                "reference_metadata",
                "control_features",
            ]
        ),
        H1_local_columns_json=json_value(
            [
                "papers_common.publication_year",
                "paper_references.reference_id",
                "reference_metadata.reference_year",
                "control_features.reference_age_median",
                "control_features.reference_age_iqr",
            ]
        ),
        H1_derivation_description="Raw local reference years could support a mean offset only after linked-edge coverage and undated-reference treatment are demonstrated. The directly materialized local controls are median and IQR, not the source mean or signed convention.",
        H1_formalization_decision="retain_evidence_gap",
        H1_rationale="The source formula is explicit, but the inventory does not establish strict bibliography/reference-year coverage equivalence or a compatible missing-year rule.",
    ),
}


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    """Write deterministic UTF-8 CSV output."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Verify permitted assets and write batch-2 H1 formalizations."""
    with INPUT_PATH.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        input_columns = reader.fieldnames
        input_rows = list(reader)
    if input_columns is None:
        raise ValueError("Input CSV has no header")
    if len(input_rows) != 17:
        raise ValueError(f"Expected 17 candidates, found {len(input_rows)}")
    if {row["candidate_id"] for row in input_rows} != set(REVIEWS):
        raise ValueError(
            "Review registry does not exactly cover the input candidate IDs"
        )

    inventory: dict[str, Any] = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    if inventory["schema_version"] != "local_t0_input_inventory_v4":
        raise ValueError("Unexpected local T0 inventory schema")
    for row in input_rows:
        fulltext = Path(row["fulltext_local_path"])
        if sha256(fulltext) != row["fulltext_sha256"]:
            raise ValueError(f"Full-text hash mismatch for {row['candidate_id']}")

    output_rows = [{**row, **REVIEWS[row["candidate_id"]]} for row in input_rows]
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    write_csv(OUTPUT_PATH, [*input_columns, *H1_FIELDS], output_rows)
    manifest = {
        "artifact": "contextual_formalization_H1_batch2_completed_v4",
        "reviewer": "H1",
        "input_path": str(INPUT_PATH),
        "input_sha256": sha256(INPUT_PATH),
        "inventory_path": str(INVENTORY_PATH),
        "inventory_sha256": sha256(INVENTORY_PATH),
        "output_path": str(OUTPUT_PATH),
        "output_sha256": sha256(OUTPUT_PATH),
        "candidate_rows": len(output_rows),
        "h1_field_count": len(H1_FIELDS),
        "formalization_decision_counts": {
            "retain_evidence_gap": sum(
                row["H1_formalization_decision"] == "retain_evidence_gap"
                for row in output_rows
            )
        },
        "fulltext_sha256_verified": True,
        "qwen_or_ollama_used": False,
        "blind_review_constraints": [
            "Used only the batch-2 input, listed English full texts, and local_t0_input_inventory_v4.json.",
            "Did not use AI/H2 output or prior batch output as evidence.",
            "Retained evidence gaps whenever a local field was absent, transformed, coverage-ambiguous, or not strictly source-equivalent.",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
