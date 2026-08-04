from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from common import sha256_file, write_csv, write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "human_tasks"
    / "formal_terminal_operationalization_AI_v3.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "independent_codex_review_v3"
    / "formal_terminal_operationalization_AI_REVIEWED_v3.csv"
)
DEFAULT_PROTOCOL = ROOT / "PRIMARY_CODEX_OPERATIONALIZATION_PROTOCOL_V3.json"
IMPLEMENTATION = ROOT / "materialize_backward_citation_age_v3.py"
OPERATIONALIZATION_DIR = ROOT / "outputs" / "operationalizations"
REVIEWER_ID = "primary_codex_operationalization_v3"
MODEL = "codex_configured_default"
MODEL_DIGEST = "codex-thread:019fa728-bf6c-7453-9af8-9ade78756aae"
COMPLETED_AT = "2026-07-30T08:15:00+00:00"
RUN_ID = "primary_codex_operationalization_v3_20260730T081500Z"

PROVENANCE_FIELDS = (
    "draft_method",
    "independent_ai_review_status",
    "independent_ai_reviewer_id",
    "independent_ai_reviewed_at",
    "independent_ai_review_action",
    "independent_ai_run_id",
    "independent_ai_model",
    "independent_ai_prompt_sha256",
)


def _approved_backward_age_payload() -> Dict[str, str]:
    """Return the exact source-to-local operationalization for EF0052."""
    test_path = (
        OPERATIONALIZATION_DIR
        / "backward_citation_age_mean_v3_test.json"
    ).resolve()
    input_snapshot_path = (
        OPERATIONALIZATION_DIR
        / "backward_citation_age_mean_v3_input_snapshot.json"
    ).resolve()
    paths = (IMPLEMENTATION.resolve(), test_path, input_snapshot_path)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "protocol_missing_rule": (
            "Return missing when the focal publication year is absent, "
            "the paper has no backward-citation edges, or no edge has a "
            "known reference year producing a nonnegative age. No values "
            "are imputed."
        ),
        "denominator_zero_rule": (
            "The denominator is the number of references with a known "
            "year and nonnegative age. When it is zero, return missing; "
            "never substitute zero."
        ),
        "observed_empty_set_rule": (
            "A paper with zero recorded references returns missing with "
            "reference_edge_count=0. An empty reference set is not age 0."
        ),
        "incomplete_coverage_rule": (
            "Exclude references with missing years or apparent future "
            "years from numerator and denominator, compute the mean over "
            "remaining references, and report valid/missing/future counts "
            "plus coverage. Same-year references are valid zeroes."
        ),
        "transform_and_unit_rule": (
            "For every valid reference compute focal publication year "
            "minus referenced publication year, then take the untransformed "
            "arithmetic mean in years. Higher values mean an older cited "
            "knowledge base."
        ),
        "input_columns_json": json.dumps(
            [
                "papers_common_all.parquet:paper_id",
                "papers_common_all.parquet:publication_year",
                "paper_references.parquet:paper_id",
                "paper_references.parquet:reference_id",
                "reference_metadata.parquet:reference_id",
                "reference_metadata.parquet:reference_year",
            ],
            ensure_ascii=False,
        ),
        "implementation_path": str(IMPLEMENTATION.resolve()),
        "implementation_sha256": sha256_file(IMPLEMENTATION),
        "test_artifact_path": str(test_path),
        "test_artifact_sha256": sha256_file(test_path),
        "input_snapshot_path": str(input_snapshot_path),
        "input_snapshot_sha256": sha256_file(input_snapshot_path),
        "decision": "approve",
        "reason": (
            "The source explicitly defines paper-level mean backward "
            "citation age as publication-year differences averaged over "
            "a focal article's references. The implementation computes "
            "that statistic from frozen raw citation edges and years, "
            "keeps source non-reporting separate from project missing "
            "rules, passes deterministic edge-case tests, and uses no "
            "future outcomes."
        ),
    }


def _exclusion_reasons() -> Dict[str, str]:
    """Return evidence-based exclusions for non-operationalizable formulas."""
    return {
        "EF0017": (
            "Exclude: the additive dive definition requires variety and "
            "balance entropies plus mutual information from a joint "
            "cross-tabulation of multiple categorical random variables. "
            "The frozen local reference metadata supplies one field label "
            "per cited work but no source-equivalent joint assignment. "
            "The existing field_div_index is a different multiplicative "
            "Rao/Stirling-style feature and cannot be substituted."
        ),
        "EF0083": (
            "Exclude: the source's C_i is a scientist-node clustering "
            "coefficient computed in the authors' pre-T0 collaboration "
            "network. The frozen corpus has no author identity/history "
            "table sufficient to reconstruct that network; its available "
            "paper bibliographic-coupling clustering is a different graph "
            "and construct."
        ),
        "EF0240": (
            "Exclude: Concepts counts n-grams selected as the top 0.01% "
            "using their subsequent corpus-wide mentions and is assigned "
            "to field-period pairs. That selection uses post-birth "
            "information, is not a paper-level T0 feature as defined, and "
            "the frozen inputs also lack the required complete MEDLINE "
            "text/MeSH history."
        ),
        "EF0319": (
            "Exclude: relative algebraic connectivity requires the "
            "weighted scientist collaboration network from t-3 to t-1, "
            "Louvain communities, and each scientist's giant component. "
            "The frozen corpus lacks author identities and prior "
            "collaboration histories; no local paper-network feature is "
            "mathematically or construct-equivalent."
        ),
    }


def review_rows(
    input_path: Path,
    protocol_path: Path,
) -> List[Dict[str, Any]]:
    """Review every exported operationalization row without a count target."""
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    expected = {"EF0017", "EF0052", "EF0083", "EF0240", "EF0319"}
    observed = {str(row.get("feature_id") or "") for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError(
            f"Unexpected operationalization candidates: {sorted(observed)}"
        )
    prompt_hash = sha256_file(protocol_path)
    exclusions = _exclusion_reasons()
    approved = _approved_backward_age_payload()
    reviewed: List[Dict[str, Any]] = []
    for row in rows:
        feature_id = str(row["feature_id"])
        if feature_id == "EF0052":
            row.update(approved)
            action = "approve_exact_source_correspondence"
        else:
            row["decision"] = "exclude"
            row["reason"] = exclusions[feature_id]
            action = "exclude_no_exact_operationalization"
        row.update(
            {
                "draft_method": (
                    "primary_codex_source_to_frozen_input_review"
                ),
                "independent_ai_review_status": "complete",
                "independent_ai_reviewer_id": REVIEWER_ID,
                "independent_ai_reviewed_at": COMPLETED_AT,
                "independent_ai_review_action": action,
                "independent_ai_run_id": RUN_ID,
                "independent_ai_model": MODEL,
                "independent_ai_prompt_sha256": prompt_hash,
            }
        )
        reviewed.append(row)
    return reviewed


def main() -> None:
    """Write the reviewed artifact and its exact-hash manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    protocol_path = args.protocol.resolve()
    rows = review_rows(input_path, protocol_path)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        input_fields = list(csv.DictReader(handle).fieldnames or [])
    output_fields = input_fields + [
        field for field in PROVENANCE_FIELDS if field not in input_fields
    ]
    write_csv(output_path, rows, output_fields)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = {
        "run_id": RUN_ID,
        "action": "primary_source_to_local_operationalization_review",
        "artifact_path": str(output_path),
        "artifact_sha256": sha256_file(output_path),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "reviewer_role": "AI",
        "reviewer_id": REVIEWER_ID,
        "model": MODEL,
        "model_digest": MODEL_DIGEST,
        "protocol_path": str(protocol_path),
        "prompt_sha256": sha256_file(protocol_path),
        "parameters": {
            "rows": len(rows),
            "approved": sum(row["decision"] == "approve" for row in rows),
            "excluded": sum(row["decision"] == "exclude" for row in rows),
            "source_reported_values_rewritten": False,
            "target_count_influence": False,
            "model_outcomes_used": False,
            "qwen_or_ollama_used": False,
            "local_or_external_llm_api_used": False,
        },
        "item_count": len(rows),
        "completed_at": COMPLETED_AT,
        "status": "complete",
        "target_count_influence": False,
        "qwen_or_ollama_used": False,
        "prohibited_sources_used": False,
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "artifact_path": str(output_path),
                "artifact_sha256": sha256_file(output_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "approved": manifest["parameters"]["approved"],
                "excluded": manifest["parameters"]["excluded"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
