"""Create the H2 v4 local-data correspondence adjudication artifact."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "data_correspondence_H2_v4.csv"
INVENTORY = ROOT / "outputs" / "local_input_inventory_v4.csv"
H1_AUDIT = ROOT / "outputs" / "data_audit_H1_v4.csv"
FEATURE_REPORT = ROOT / "outputs" / "evidence_features_v4_report.json"
OUTPUT = ROOT / "outputs" / "data_correspondence_H2_completed_v4.csv"
MANIFEST = ROOT / "outputs" / "data_correspondence_H2_completed_v4.manifest.json"

METHOD = (
    "Feature-wise H2 adjudication against the frozen local T0/T0-1 inventory, "
    "the H1 data audit, and the materialization implementation. Exact decisions "
    "require formula-equivalent inputs and derivation; similar names do not suffice. "
    "Co-citation similarity is not treated as ordinary directed citation-profile "
    "similarity, and category-set disjointness is not treated as single-label inequality."
)


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_json(value: Any) -> str:
    """Encode audit fields deterministically as JSON."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def mapping() -> dict[str, dict[str, str]]:
    """Return the independently adjudicated mapping for every frozen feature."""
    materialized_counts = [
        "innovation_candidate_features:paper_id",
        "innovation_candidate_features:field_gini_balance",
        "paper_references:paper_id",
        "paper_references:reference_id",
        "reference_metadata:reference_id",
        "reference_metadata:field_id",
    ]
    return {
        "EF0001": {
            "match_decision": "construct_mismatch",
            "local_source_ids_json": compact_json([]),
            "local_columns_json": compact_json([]),
            "derivation_description": "No approved derivation. The available five-year prior field-citation profiles produce directed citation-profile cosine distances, not a subject-category co-citation similarity matrix.",
            "construct_equivalence_notes": "Rejected: the required d_ij is one minus subject-category co-citation cosine similarity. Ordinary directed field-citation profiles are a different relation and cannot be substituted on a name or cosine-formula resemblance.",
            "reason": "No local T0/T0-1 subject-category co-citation similarity input is inventoried; H1 also marked the apparent local disparity column unavailable for this construct.",
        },
        "EF0002": {
            "match_decision": "exact_materialized",
            "local_source_ids_json": compact_json(["innovation_candidate_features", "paper_references", "reference_metadata"]),
            "local_columns_json": compact_json(materialized_counts),
            "derivation_description": "Join each T0 focal backward-citation edge to its referenced-work field_id; count nonmissing field_id values by paper; compute 1 minus the Gini coefficient. The audited materialization is innovation_candidate_features.field_gini_balance.",
            "construct_equivalence_notes": "The formula is exactly 1-Gini over cited-reference category counts. The local field_id is the explicit one-category-per-mapped-reference taxonomy used to form that count distribution; no future information is used.",
            "reason": "H1 audited the materialized column as nonconstant and retained missingness; the raw components and formula are traceable to T0 inputs.",
        },
        "EF0003": {
            "match_decision": "construct_mismatch",
            "local_source_ids_json": compact_json([]),
            "local_columns_json": compact_json([]),
            "derivation_description": "No approved derivation. The local implementation compares a focal primary field with a reference field rather than evaluating disjointness of two subject-category sets.",
            "construct_equivalence_notes": "Rejected: a single-label inequality does not establish that the focal and cited-work category sets are disjoint. The required multi-label inputs are not available in the frozen inventory.",
            "reason": "H1 marked the local scalar comparison unavailable for the stated set-disjointness construct; no same-construct T0 source was found.",
        },
        "EF0004": {
            "match_decision": "exact_materialized",
            "local_source_ids_json": compact_json(["innovation_candidate_features", "paper_references", "reference_metadata"]),
            "local_columns_json": compact_json([
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:field_gini_simpson",
                "paper_references:paper_id",
                "paper_references:reference_id",
                "reference_metadata:reference_id",
                "reference_metadata:field_id",
            ]),
            "derivation_description": "Join T0 citation edges to referenced field_id values, form per-paper category proportions p_i, and compute 1-sum_i(p_i^2). The audited materialization is innovation_candidate_features.field_gini_simpson.",
            "construct_equivalence_notes": "The local derivation is the stated Gini-Simpson formula over the explicit reference category-count distribution and uses only T0 source fields.",
            "reason": "H1 audited the materialized column as nonconstant with explicit missingness; the underlying formula and inputs are traceable.",
        },
        "EF0005": {
            "match_decision": "exact_derivable",
            "local_source_ids_json": compact_json(["paper_references"]),
            "local_columns_json": compact_json(["paper_references:paper_id", "paper_references:reference_id"]),
            "derivation_description": "For each complete T0 citation edge (paper_id, reference_id), count the citing paper's outgoing reference edges ref(p_i) and assign weight 1/ref(p_i) to that edge.",
            "construct_equivalence_notes": "This is exactly the stated edge-level reciprocal-reference-count formula from the complete local backward-citation edge table.",
            "reason": "H1 audited the edge-level inputs and derivation. This correspondence does not create a paper-level scalar: subsequent paper-level applicability requires a separately specified and audited aggregation, so it must not be treated as a direct paper-level feature.",
        },
        "EF0006": {
            "match_decision": "no_match",
            "local_source_ids_json": compact_json([]),
            "local_columns_json": compact_json([]),
            "derivation_description": "No derivation is available from the frozen sources.",
            "construct_equivalence_notes": "The formula requires both field-level keyword frequency and a focal-paper keyword count. title_word_count is not a keyword count and cannot be used as a proxy without changing the construct.",
            "reason": "Neither required keyword input is present in the T0/T0-1 inventory; H1 marked the feature unavailable.",
        },
        "EF0007": {
            "match_decision": "exact_materialized",
            "local_source_ids_json": compact_json(["innovation_candidate_features", "paper_references", "reference_metadata", "field_citation_events"]),
            "local_columns_json": compact_json([
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:rao_stirling_integration",
                "paper_references:paper_id",
                "paper_references:reference_id",
                "reference_metadata:reference_id",
                "reference_metadata:field_id",
                "field_citation_events:source_year",
                "field_citation_events:source_field_id",
                "field_citation_events:target_field_id",
                "field_citation_events:citation_count",
            ]),
            "derivation_description": "Form cited-reference field proportions p_i at T0. For each focal year, construct a five-year strictly-prior field dissimilarity matrix d_ij=1-cosine from field citation-profile vectors, then compute the ordered-pair Rao-Stirling sum; the audited materialization is innovation_candidate_features.rao_stirling_integration.",
            "construct_equivalence_notes": "The frozen formula requires a supplied subject-category dissimilarity matrix, not specifically a co-citation matrix. The implementation explicitly supplies its T0-1 field-category distance matrix and applies the stated Rao-Stirling algebra. This approval does not equate that matrix with EF0001's required co-citation similarity.",
            "reason": "H1 audited the materialized value and its outcome-blind T0/T0-1 inputs. The formula’s generic dissimilarity-matrix requirement is met by the documented local field-category distance construction.",
        },
        "EF0008": {
            "match_decision": "exact_materialized",
            "local_source_ids_json": compact_json(["innovation_candidate_features", "paper_references", "reference_metadata"]),
            "local_columns_json": compact_json([
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:field_shannon_entropy",
                "paper_references:paper_id",
                "paper_references:reference_id",
                "reference_metadata:reference_id",
                "reference_metadata:field_id",
            ]),
            "derivation_description": "Join T0 citation edges to referenced field_id values, form per-paper category proportions p_i, and compute -sum_i p_i log(p_i). The audited materialization is innovation_candidate_features.field_shannon_entropy.",
            "construct_equivalence_notes": "The local field_id count distribution directly provides the categorical input required by the Shannon formula; no distance proxy or future information is used.",
            "reason": "H1 audited the materialized column as nonconstant with explicit missingness, and the formula is directly traceable to T0 inputs.",
        },
        "EF0009": {
            "match_decision": "exact_materialized",
            "local_source_ids_json": compact_json(["innovation_candidate_features", "paper_references", "reference_metadata"]),
            "local_columns_json": compact_json([
                "innovation_candidate_features:paper_id",
                "innovation_candidate_features:field_variety",
                "paper_references:paper_id",
                "paper_references:reference_id",
                "reference_metadata:reference_id",
                "reference_metadata:field_id",
            ]),
            "derivation_description": "Join T0 citation edges to referenced field_id values and count the distinct nonmissing field_id categories per paper. The audited materialization is innovation_candidate_features.field_variety.",
            "construct_equivalence_notes": "The local value is exactly the count of distinct referenced subject-category labels under the documented field_id taxonomy, using T0 reference metadata.",
            "reason": "H1 audited the materialized column as nonconstant with explicit missingness; all formula inputs are available at T0.",
        },
    }


def main() -> None:
    """Write the adjudicated CSV and manifest without changing protected payloads."""
    with INPUT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError("The H2 input CSV has no header.")
    adjudication = mapping()
    input_ids = {row["feature_id"] for row in rows}
    if input_ids != set(adjudication):
        raise ValueError("The frozen CSV features do not match the H2 adjudication set.")
    protected_fields = [
        "alias_names_evidence", "formula_evidence", "required_data_evidence",
        "maximum_information_time_evidence", "scope_role_evidence", "article_level_evidence",
        "t0_computable_evidence", "requires_future_evidence", "research_groups_evidence",
        "mention_ids_evidence", "local_input_inventory_path", "local_input_inventory_sha256",
        "ai_payload_json", "h1_payload_json",
    ]
    protected_before = [{name: row[name] for name in protected_fields} for row in rows]
    for row in rows:
        row["reviewer_role"] = "H2"
        row.update(adjudication[row["feature_id"]])
    if protected_before != [{name: row[name] for name in protected_fields} for row in rows]:
        raise AssertionError("Protected evidence or payload fields changed.")
    allowed = {
        "exact_materialized", "exact_derivable", "candidate_formula_completion",
        "no_match", "construct_mismatch", "future_only", "insufficient_evidence",
    }
    if any(row["match_decision"] not in allowed for row in rows):
        raise ValueError("An invalid match decision was assigned.")
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    completed_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    manifest = {
        "schema_version": "data_correspondence_h2_manifest_v4",
        "status": "complete",
        "reviewer_role": "H2",
        "reviewer_id": "H2_LOCAL_DATA_MAPPING_ADJUDICATOR_V4",
        "model": "codex_configured_default",
        "model_digest": "codex-thread:/root/h2_adjudication",
        "method": METHOD,
        "review_completed_at": completed_at,
        "outcome_columns_used": False,
        "database_accessed": False,
        "qwen_or_ollama_used": False,
        "input_artifacts": {
            str(INPUT): sha256(INPUT),
            str(INVENTORY): sha256(INVENTORY),
            str(H1_AUDIT): sha256(H1_AUDIT),
            str(FEATURE_REPORT): sha256(FEATURE_REPORT),
        },
        "output_artifact": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
        "decision_counts": dict(sorted(Counter(row["match_decision"] for row in rows).items())),
        "feature_count": len(rows),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
