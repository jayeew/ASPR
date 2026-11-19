"""Write the final H2 adjudication for v4 feature operationalization."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "operationalization_H2_v4.csv"
CORRESPONDENCE = ROOT / "outputs" / "data_correspondence_H2_completed_v4.csv"
IMPLEMENTATION = ROOT / "materialize_evidence_features_v4.py"
TEST_SCRIPT = ROOT / "test_evidence_features_v4.py"
TEST_ARTIFACT = ROOT / "outputs" / "evidence_features_v4_tests.json"
MATRIX = ROOT / "outputs" / "evidence_features_v4.parquet"
REPORT = ROOT / "outputs" / "evidence_features_v4_report.json"
OUTPUT = ROOT / "outputs" / "operationalization_H2_completed_v4.csv"
MANIFEST = ROOT / "outputs" / "operationalization_H2_completed_v4.manifest.json"

IMPLEMENTATION_PATH = str(IMPLEMENTATION)
TEST_ARTIFACT_PATH = str(TEST_ARTIFACT)
MATRIX_PATH = str(MATRIX)
COMMON_PROTOCOL = (
    "Use only focal papers with a valid publication year and eligible flag. Join focal "
    "reference edges to metadata; retain only references with known reference_year strictly "
    "earlier than the focal publication_year; exclude empty field_id values from category "
    "counts; never impute a field label."
)
COMMON_COVERAGE = (
    "References lacking usable pre-publication metadata or a nonempty field_id do not "
    "contribute to the category statistic. The v4 matrix retains mapped_reference_count "
    "and total_reference_count; if no mapped category remains, emit missing rather than a filled value."
)
COMMON_INPUTS = [
    "paper_references:paper_id",
    "paper_references:reference_id",
    "papers_common:natural_science_eligible",
    "papers_common:paper_id",
    "papers_common:publication_year",
    "reference_metadata:field_id",
    "reference_metadata:reference_id",
    "reference_metadata:reference_year",
]


def digest(path: Path) -> str:
    """Return the SHA-256 digest of one artifact."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def encoded(value: Any) -> str:
    """Encode a structured CSV cell reproducibly."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def approved_payloads(implementation_sha: str, test_sha: str, matrix_sha: str) -> dict[str, dict[str, str]]:
    """Return complete H2 project payloads for formula-equivalent features."""
    shared = {
        "protocol_missing_rule": COMMON_PROTOCOL,
        "incomplete_coverage_rule": COMMON_COVERAGE,
        "implementation_path": IMPLEMENTATION_PATH,
        "implementation_sha256": implementation_sha,
        "test_artifact_path": TEST_ARTIFACT_PATH,
        "test_artifact_sha256": test_sha,
        "input_snapshot_path": MATRIX_PATH,
        "input_snapshot_sha256": matrix_sha,
        "decision": "approve",
    }
    outputs: dict[str, dict[str, str]] = {}
    outputs["EF0002"] = {
        **shared,
        "denominator_zero_rule": "No probability denominator is used. If the mapped category-count vector is empty, emit missing; otherwise compute Gini on occupied category counts only.",
        "observed_empty_set_rule": "No mapped reference category: missing (NaN). One occupied category is observed data and returns 1.0 for 1-Gini.",
        "transform_and_unit_rule": "Compute 1-Gini from untransformed occupied reference-field counts; retain the proportion-scale value in [0,1], with no additional normalization or log transform.",
        "input_columns_json": encoded(COMMON_INPUTS),
        "reason": "Approved: the project rule defines the observable category-count vector as nonmissing mapped reference categories, emits missing for zero mapped categories, and never imputes. The direct 1-Gini formula is applied at T0 and mapped_reference_count/total_reference_count make partial coverage transparent; the H1-reviewed deterministic test covers the empty and partial-coverage branches.",
    }
    outputs["EF0004"] = {
        **shared,
        "denominator_zero_rule": "When the mapped category-count total is zero, emit missing; otherwise set p_i=x_i/sum_i x_i using the mapped-reference total.",
        "observed_empty_set_rule": "No mapped reference category: missing (NaN). One occupied category is observed data and returns 0.0 for 1-sum_i p_i^2.",
        "transform_and_unit_rule": "Compute 1-sum_i p_i^2 from untransformed occupied reference-field counts; retain the proportion-scale value in [0,1], with no additional transform.",
        "input_columns_json": encoded(COMMON_INPUTS),
        "reason": "Approved: the project rule uses only the observed mapped reference-category distribution, emits missing at zero mapping, and preserves mapped/total counts without imputation. The materializer applies the stated Gini-Simpson formula at T0; the H1-reviewed test explicitly covers the partial-coverage branch.",
    }
    outputs["EF0007"] = {
        **shared,
        "denominator_zero_rule": "When no mapped reference category exists, emit missing. Otherwise use p_i=x_i/N where N is the mapped-reference count; the ordered-pair sum is implemented as twice the unordered-pair sum.",
        "observed_empty_set_rule": "No mapped reference category: missing (NaN). One occupied category with at least one mapped reference has an observed empty cross-category sum and returns 0.0; missing required distances for two or more occupied categories return missing.",
        "incomplete_coverage_rule": "References lacking usable pre-publication metadata or a nonempty field_id do not contribute. For publication year y, use only field-citation events from y-5 through y-1; if any occupied field pair lacks a local distance, emit missing. The matrix retains mapped_reference_count and total_reference_count; no distance or category is imputed.",
        "transform_and_unit_rule": "Compute sum_{i!=j} p_i p_j d_ij exactly as 2 times the unordered-pair sum; d_ij is the local five-year strictly-prior cosine distance in [0,1]. Retain the untransformed dissimilarity-weighted diversity value.",
        "input_columns_json": encoded([
            "field_citation_events:citation_count",
            "field_citation_events:source_field_id",
            "field_citation_events:source_year",
            "field_citation_events:target_field_id",
            *COMMON_INPUTS,
        ]),
        "reason": "Approved: the original formula requires a category-dissimilarity matrix but does not require EF0001's co-citation construction. The implementation applies the exact ordered-pair Rao-Stirling algebra to a documented T0-1 five-year field-category distance matrix; zero mapping and missing pair distances yield missing, no categories/distances are imputed, and mapped/total coverage remains visible. H1 reviewed the pair, missing-distance, and partial-coverage tests.",
    }
    outputs["EF0008"] = {
        **shared,
        "denominator_zero_rule": "When the mapped category-count total is zero, emit missing; otherwise set p_i=x_i/sum_i x_i using the mapped-reference total.",
        "observed_empty_set_rule": "No mapped reference category: missing (NaN). One occupied category is observed data and returns 0.0 Shannon entropy.",
        "transform_and_unit_rule": "Compute -sum_i p_i ln(p_i) using the natural logarithm from untransformed occupied reference-field counts; retain the result in nats with no further transform.",
        "input_columns_json": encoded(COMMON_INPUTS),
        "reason": "Approved: the observable mapped category-count vector is explicit, zero mapping yields missing, and partial reference coverage is reported rather than imputed or hidden. The materializer computes the stated natural-log Shannon formula from T0 inputs; H1 reviewed the empty and partial-coverage test branches.",
    }
    outputs["EF0009"] = {
        **shared,
        "denominator_zero_rule": "No denominator is used; count distinct nonempty mapped reference-category labels.",
        "observed_empty_set_rule": "No mapped reference category: missing (NaN). A nonempty set returns its observed cardinality, including 1 for a single category.",
        "transform_and_unit_rule": "Return the untransformed integer count of distinct nonempty reference field_id labels; do not normalize by a taxonomy size or apply a logarithm.",
        "input_columns_json": encoded(COMMON_INPUTS),
        "reason": "Approved: the project rule defines variety over observed mapped reference categories, preserves total and mapped counts, returns missing for zero mapping, and does not impute labels. The materializer directly counts distinct T0 field_id values; H1 reviewed the empty and partial-coverage test branches.",
    }
    return outputs


def excluded_reasons() -> dict[str, str]:
    """Return exclusion rationales that retain no non-applicable operational payload."""
    return {
        "EF0001": "Exclude: both AI and H1 reject the available directed field-citation-profile cosine distance as a substitute for the formula's subject-category co-citation cosine similarity. The project coverage rule cannot repair that construct mismatch.",
        "EF0003": "Exclude: both AI and H1 reject a single focal/reference field inequality as the formula's required comparison of two subject-category sets for disjointness. Coverage handling cannot supply the missing set-valued construct.",
        "EF0005": "Exclude: both AI and H1 reject this as an article-level operationalization because the formula is an outgoing citation-edge weight and the v4 matrix has no documented aggregation to one paper-level scalar.",
        "EF0006": "Exclude: both AI and H1 reject this because the frozen T0 inventory has neither a paper-keyword relation nor field keyword frequency and focal keyword-count inputs, and the v4 matrix has no EF0006 output.",
    }


def main() -> None:
    """Create a protected-field-preserving H2 CSV and its provenance manifest."""
    implementation_sha = digest(IMPLEMENTATION)
    test_sha = digest(TEST_ARTIFACT)
    matrix_sha = digest(MATRIX)
    approved = approved_payloads(implementation_sha, test_sha, matrix_sha)
    excluded = excluded_reasons()
    with INPUT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError("Operationalization input has no CSV header.")
    expected = set(approved) | set(excluded)
    if {row["feature_id"] for row in rows} != expected:
        raise ValueError("Operationalization input does not contain the expected nine features.")
    protected = [
        "formula_mention_id", "source_role", "literature_formula_location",
        "literature_evidence_span", "literature_formula", "literature_units",
        "literature_parameters", "literature_direction", "source_reported_missing_rule",
        "literature_required_data", "literature_maximum_information_time",
        "fulltext_source_url", "fulltext_sha256", "ai_payload_json", "h1_payload_json",
    ]
    protected_before = [{key: row[key] for key in protected} for row in rows]
    project_fields = [
        "protocol_missing_rule", "denominator_zero_rule", "observed_empty_set_rule",
        "incomplete_coverage_rule", "transform_and_unit_rule", "input_columns_json",
        "implementation_path", "implementation_sha256", "test_artifact_path",
        "test_artifact_sha256", "input_snapshot_path", "input_snapshot_sha256",
        "decision", "reason",
    ]
    for row in rows:
        row["reviewer_role"] = "H2"
        for field in project_fields:
            row[field] = ""
        feature_id = row["feature_id"]
        if feature_id in approved:
            row.update(approved[feature_id])
        else:
            row["decision"] = "exclude"
            row["reason"] = excluded[feature_id]
    if protected_before != [{key: row[key] for key in protected} for row in rows]:
        raise AssertionError("Protected literature or AI/H1 payload fields changed.")
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    decision_counts = Counter(row["decision"] for row in rows)
    manifest = {
        "schema_version": "operationalization_h2_manifest_v4",
        "status": "complete",
        "reviewer_role": "H2",
        "reviewer_id": "H2_FINAL_FORMULA_OPERATIONALIZATION_ADJUDICATOR_V4",
        "model": "codex_configured_default",
        "model_digest": "codex-thread:/root/h2_adjudication",
        "review_completed_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "method": "Adjudicate each frozen formula against implementation, H1 review, test artifact, matrix, and H2 correspondence. Approve only formula-equivalent T0/T0-1 operationalizations using the declared observed-mapped-category rule: zero mapped categories is missing, no labels/distances are imputed, and mapped/total counts remain in the matrix. Retain exclusions for construct or unit mismatches.",
        "outcome_columns_used": False,
        "database_accessed": False,
        "qwen_or_ollama_used": False,
        "input_artifacts": {
            str(INPUT): digest(INPUT),
            str(CORRESPONDENCE): digest(CORRESPONDENCE),
            str(IMPLEMENTATION): implementation_sha,
            str(TEST_SCRIPT): digest(TEST_SCRIPT),
            str(TEST_ARTIFACT): test_sha,
            str(MATRIX): matrix_sha,
            str(REPORT): digest(REPORT),
        },
        "h1_review_evidence": {
            "approved_features": sorted(approved),
            "test_artifact": str(TEST_ARTIFACT),
            "test_artifact_sha256": test_sha,
            "review_basis": "Each approved H1 payload records approve and the same hash-verified test artifact; its test list includes empty_mapped_reference_set_missing and incomplete_category_coverage_uses_observed_categories_without_imputation.",
        },
        "output_artifact": str(OUTPUT),
        "output_sha256": digest(OUTPUT),
        "decision_counts": dict(sorted(decision_counts.items())),
        "feature_count": len(rows),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
