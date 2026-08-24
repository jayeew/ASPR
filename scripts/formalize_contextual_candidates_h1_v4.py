"""Create the independent H1 formalization review for five contextual candidates."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Final

ROOT: Final = Path("/home/jayee/workspace/ASPR")
INPUT_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_formalization_input_v4.csv"
)
INPUT_MANIFEST_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "contextual_formalization_input_manifest_v4.json"
)
INVENTORY_PATH: Final = ROOT / (
    "innovation_impact_feature_selection/evidence_derived_v4_rebuild/outputs/"
    "local_t0_input_inventory_v4.json"
)
OUTPUT_PATH: Final = ROOT / "outputs/contextual_formalization_H1_completed_v4.csv"
MANIFEST_PATH: Final = (
    ROOT / "outputs/contextual_formalization_H1_completed_v4.manifest.json"
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
    """Encode a structured field deterministically for CSV storage."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def formalization(**values: str) -> dict[str, str]:
    """Build one complete H1 formalization row using the manifest field contract."""
    expected = set(H1_FIELDS)
    if set(values) != expected:
        raise ValueError("Formalization registry does not match required H1 fields")
    return values


REVIEWS: Final[dict[str, dict[str, str]]] = {
    "CFT_fe43839fab9ef08d": formalization(
        H1_canonical_name_en="data_paper_abstract_rhetorical_move_composition",
        H1_label_zh="数据论文摘要修辞动作构成",
        H1_formula="Source-authorized procedure, not a single equation: code every abstract sentence with Table 1's nine moves; where a sentence has two moves, assign 0.5 to each move when counting move frequencies.",
        H1_units="Nine-move count vector or normalized move-share vector per abstract",
        H1_parameters="Nine Table-1 labels; sentence segmentation policy; source fractional weight=0.5 for each of two co-existing moves",
        H1_direction="No scalar direction; composition vector",
        H1_missing_rule="Absent. The source reports coder disagreement resolution and fractional treatment for two-move sentences, but no rule for missing abstracts or unclassifiable sentences.",
        H1_required_data_json=json_value(
            [
                "focal-paper abstract text",
                "sentence boundaries",
                "Table 1 rhetorical-move codebook",
            ]
        ),
        H1_research_group="Data-paper abstracts",
        H1_research_group_evidence="The source states that data papers combine IMRaD- and data-oriented moves and that two coders classified all abstract sentences using the modified scheme (Methods §3.2–3.3).",
        H1_data_match_decision="no_local_match_retain_evidence_gap",
        H1_local_source_ids_json=json_value([]),
        H1_local_columns_json=json_value([]),
        H1_derivation_description="No inventory source exposes focal-paper title, abstract text, sentence annotations, or a rhetorical-move codebook. Paper identity alone cannot derive this composition.",
        H1_formalization_decision="retain_evidence_gap",
        H1_rationale="The coding procedure and fractional multi-move rule are source-authorized, but local T0 inputs lack abstract text and annotations; do not synthesize a text feature from unavailable fields.",
    ),
    "CFT_1c4f7b29076b5adb": formalization(
        H1_canonical_name_en="coauthor_affiliated_country_true_diversity",
        H1_label_zh="合作者所属国家真多样性",
        H1_formula="AF_D = 1 − Σ_i,j(S^a_ij P^a_i P^a_j), where S^a_ij = 1 − D^a_ij / Max_D.",
        H1_units="Dimensionless true diversity; higher values indicate greater affiliation-country diversity",
        H1_parameters="P^a_i=fractional author share in country i; D^a_ij=great-circle distance between country capitals; Max_D=maximum Earth-surface distance",
        H1_direction="Higher = more diverse across affiliated countries",
        H1_missing_rule="The source explicitly retains a publication with missing author-address links when it has at least one link; no complete local rule is supplied for unresolved country assignments or all-link-missing papers.",
        H1_required_data_json=json_value(
            [
                "focal-paper author-to-country affiliation links",
                "fractional author shares by country",
                "fixed country-capital coordinate table",
            ]
        ),
        H1_research_group="Publications with author-address affiliation links",
        H1_research_group_evidence="The paper defines affiliated-country diversity from author-country distributions and reports retaining publications with at least one author-address link (Control variables, pp. 7765–7771).",
        H1_data_match_decision="partial_local_counts_only_retain_evidence_gap",
        H1_local_source_ids_json=json_value(["target_openalex_metadata"]),
        H1_local_columns_json=json_value(["paper_id", "openalex_country_count"]),
        H1_derivation_description="The inventory provides only an aggregate country count. It lacks author-to-country links, fractional country shares, and the specified capital-coordinate table, so AF_D cannot be derived without inventing inputs.",
        H1_formalization_decision="retain_evidence_gap",
        H1_rationale="The mathematical construction is explicit, but the local inventory does not contain the distribution and distances required by the source formula; aggregate country count is not construct-equivalent.",
    ),
    "CFT_9bb0adca06dc9458": formalization(
        H1_canonical_name_en="harmonic_authorship_credit",
        H1_label_zh="谐波署名信用分配",
        H1_formula="credit_i = (1/i) / Σ_(k=1)^N(1/k).",
        H1_units="Fraction of one publication credit allocated to byline author i; credits sum to 1",
        H1_parameters="i=author rank; N=number of coauthors",
        H1_direction="Earlier byline rank receives higher credit",
        H1_missing_rule="Derivable only for a nonempty ordered byline. The source does not specify group/consortium authors, equal-contribution statements, or unavailable author order.",
        H1_required_data_json=json_value(
            ["focal-paper ordered author byline", "focal-paper author count"]
        ),
        H1_research_group="General publication-level authorship-credit method",
        H1_research_group_evidence="Equation (1) defines harmonic credit for the i-th author of a publication with N coauthors (p. 1); it is presented as a general source-level correction method.",
        H1_data_match_decision="partial_author_count_only_retain_evidence_gap",
        H1_local_source_ids_json=json_value(["target_openalex_metadata"]),
        H1_local_columns_json=json_value(
            ["paper_id", "openalex_author_count", "openalex_author_ids"]
        ),
        H1_derivation_description="The inventory has author count and IDs but no documented ordered byline. Without verified author order, it cannot assign rank-specific harmonic credit.",
        H1_formalization_decision="retain_evidence_gap",
        H1_rationale="The original formula is fully explicit, but rank is indispensable and is not available as a local, documented T0 field. No rank ordering is inferred from the author-ID list.",
    ),
    "CFT_ec0dba2f36ed9ffa": formalization(
        H1_canonical_name_en="paper_collaboration_geographic_scope",
        H1_label_zh="论文合作地理范围",
        H1_formula="local if organization_count=1 and country_count=1; national if organization_count>1 and country_count=1; international if country_count>=2.",
        H1_units="Nominal categorical value: local, national, or international",
        H1_parameters="organization_count; country_count",
        H1_direction="No ordinal direction; mutually exclusive geographic-scope categories",
        H1_missing_rule="Explicit: papers with no affiliation information (neither country nor organization) are discarded. Apply unavailable/exclude when local country or institution counts are missing or nonpositive.",
        H1_required_data_json=json_value(
            [
                "focal-paper institution count",
                "focal-paper country count",
                "paper identifier for join",
            ]
        ),
        H1_research_group="Library and Information Science papers with affiliation information",
        H1_research_group_evidence="The source defines local/national/international collaboration from organization and country counts and reports discarding 1,801 papers with neither country nor organization affiliation information (Methodology, p. 7521).",
        H1_data_match_decision="matched_with_explicit_missing_rule",
        H1_local_source_ids_json=json_value(
            ["papers_common", "target_openalex_metadata"]
        ),
        H1_local_columns_json=json_value(
            [
                "papers_common.paper_id",
                "target_openalex_metadata.paper_id",
                "target_openalex_metadata.openalex_institution_count",
                "target_openalex_metadata.openalex_country_count",
            ]
        ),
        H1_derivation_description="Join target_openalex_metadata to papers_common on paper_id. If either count is null or <=0, return unavailable/exclude. Otherwise apply the source's three mutually exclusive count rules exactly.",
        H1_formalization_decision="promote_for_formalization",
        H1_rationale="The source's categorical rules and exclusion condition are explicit, and the inventory contains exact T0 institution and country counts keyed by paper_id. This is a data-contract match, subject to normal metadata coverage checks.",
    ),
    "CFT_45cf9532bd6751c1": formalization(
        H1_canonical_name_en="data_availability_statement_access_category",
        H1_label_zh="数据可用性声明访问类别",
        H1_formula="Categorical coding: 0=not available or access restricted; 1=available on request; 2=available with paper or supplementary files; 3=available in a repository.",
        H1_units="Ordered four-category data-access label (0–3), not a continuous score",
        H1_parameters="DAS category labels 0, 1, 2, 3",
        H1_direction="Higher category represents more directly accessible data under the source coding",
        H1_missing_rule="Explicit source category 0 covers data not available/access restricted; no local missing-DAS field exists to distinguish absent statement from restricted access.",
        H1_required_data_json=json_value(
            [
                "focal-paper data-availability statement text",
                "repository link or identifier when stated",
            ]
        ),
        H1_research_group="PLOS publications with data-availability statements",
        H1_research_group_evidence="The source says it identified four DAS categories based on statement content and analyzes PLOS articles with data-availability statements (Materials and methods, Table 1, pp. 4–5).",
        H1_data_match_decision="no_local_match_retain_evidence_gap",
        H1_local_source_ids_json=json_value([]),
        H1_local_columns_json=json_value([]),
        H1_derivation_description="No inventory source provides focal-paper DAS text, repository URL/identifier, or a precomputed DAS class. Open-access status is not construct-equivalent to the four DAS categories.",
        H1_formalization_decision="retain_evidence_gap",
        H1_rationale="The source coding is explicit, including the unavailable/restricted category, but the local T0 inventory lacks the statement-level content needed to apply it. Do not substitute open-access metadata.",
    ),
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_input() -> tuple[list[dict[str, str]], list[str]]:
    """Read the frozen formalization worklist."""
    with INPUT_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        return list(reader), reader.fieldnames


def validate_inputs(rows: list[dict[str, str]]) -> None:
    """Verify input manifests, candidate coverage, and full-text hashes."""
    input_manifest = json.loads(INPUT_MANIFEST_PATH.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    if input_manifest["required_reviewer_fields"] != [
        field.replace("H1_", "{ROLE}_") for field in H1_FIELDS
    ]:
        raise ValueError("H1 fields do not match the worklist manifest contract")
    if input_manifest["inventory_sha256"] != sha256_file(INVENTORY_PATH):
        raise ValueError("Local T0 inventory SHA mismatch")
    if inventory["outcome_columns_used"]:
        raise ValueError("Inventory is not outcome blind")
    if len(rows) != len(REVIEWS) or {row["candidate_id"] for row in rows} != set(
        REVIEWS
    ):
        raise ValueError("Formalization rows do not match the H1 registry")
    for row in rows:
        if sha256_file(Path(row["fulltext_local_path"])) != row["fulltext_sha256"]:
            raise ValueError(f"Full-text SHA mismatch for {row['candidate_id']}")


def write_output(rows: list[dict[str, str]], fields: list[str]) -> str:
    """Copy frozen fields and append H1 formalization fields."""
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
    """Write provenance and decisions for the H1 formalization artifact."""
    decisions = [
        REVIEWS[row["candidate_id"]]["H1_formalization_decision"] for row in rows
    ]
    manifest = {
        "input": str(INPUT_PATH),
        "input_sha256": sha256_file(INPUT_PATH),
        "input_worklist_manifest": str(INPUT_MANIFEST_PATH),
        "input_worklist_manifest_sha256": sha256_file(INPUT_MANIFEST_PATH),
        "local_t0_inventory": str(INVENTORY_PATH),
        "local_t0_inventory_sha256": sha256_file(INVENTORY_PATH),
        "artifact": str(OUTPUT_PATH),
        "artifact_sha256": output_sha,
        "row_count": len(rows),
        "fulltext_sha256_verified": True,
        "formalization_counts": {
            value: decisions.count(value) for value in sorted(set(decisions))
        },
        "qwen_or_ollama_used": False,
        "review_scope": "Independent H1 formula, missing-rule, research-group, and local T0-data formalization only; no final selection.",
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    """Build the H1 contextual formalization artifact."""
    rows, fields = read_input()
    validate_inputs(rows)
    output_sha = write_output(rows, fields)
    write_manifest(rows, output_sha)


if __name__ == "__main__":
    main()
