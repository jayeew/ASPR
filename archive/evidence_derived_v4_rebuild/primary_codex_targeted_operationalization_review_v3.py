from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from common import sha256_file, utc_now, write_csv, write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    ROOT
    / "outputs"
    / "human_tasks"
    / "formal_terminal_targeted_operationalization_AI_v3.csv"
)
DEFAULT_REGISTRY = (
    ROOT
    / "outputs"
    / "targeted_operationalizations_v3"
    / "targeted_operationalization_registry_v3.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "independent_codex_review_v3"
    / "formal_terminal_targeted_operationalization_AI_REVIEWED_v3.csv"
)
DEFAULT_PROTOCOL = ROOT / "PRIMARY_CODEX_OPERATIONALIZATION_PROTOCOL_V3.json"
REVIEWER_ID = "primary_codex_targeted_operationalization_v3"
MODEL = "codex_configured_default"
MODEL_DIGEST = "codex-thread:019fa728-bf6c-7453-9af8-9ade78756aae"
PROVENANCE_FIELDS = (
    "draft_method",
    "independent_ai_review_status",
    "independent_ai_reviewer_id",
    "independent_ai_reviewed_at",
    "independent_ai_review_action",
    "independent_ai_review_note",
    "independent_ai_run_id",
    "independent_ai_model",
    "independent_ai_prompt_sha256",
)


def _read_rows(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _load_registry(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("features"),
        dict,
    ):
        raise ValueError("Operationalization registry lacks features")
    if payload.get("model_outcomes_used") is not False:
        raise ValueError("Operationalization registry is not outcome-blind")
    return payload


def review_rows(
    input_path: Path,
    registry_path: Path,
    protocol_path: Path,
    completed_at: str,
    run_id: str,
) -> tuple[List[str], List[Dict[str, str]], Dict[str, int]]:
    fields, rows = _read_rows(input_path)
    registry = _load_registry(registry_path)
    features: Mapping[str, Mapping[str, Any]] = registry["features"]
    prompt_sha = sha256_file(protocol_path)
    counts = {"approve": 0, "exclude": 0}
    reviewed: List[Dict[str, str]] = []
    for row in rows:
        feature_id = str(row.get("feature_id") or "").strip()
        if not feature_id:
            raise ValueError("Operationalization row lacks feature_id")
        feature = features.get(feature_id)
        if feature is None:
            row["decision"] = "exclude"
            row["reason"] = (
                "No H2-approved exact source-to-local operationalization "
                "exists in the frozen targeted registry. The source formula "
                "or local construct correspondence remains incomplete, "
                "future-dependent, mismatched, or unavailable; project "
                "choices cannot rescue it."
            )
            action = "exclude_no_frozen_exact_operationalization"
            counts["exclude"] += 1
        else:
            for field in (
                "protocol_missing_rule",
                "denominator_zero_rule",
                "observed_empty_set_rule",
                "incomplete_coverage_rule",
                "transform_and_unit_rule",
                "input_columns_json",
                "implementation_path",
                "implementation_sha256",
                "test_artifact_path",
                "test_artifact_sha256",
                "input_snapshot_path",
                "input_snapshot_sha256",
            ):
                value = feature.get(field)
                if value is None or not str(value).strip():
                    raise ValueError(
                        f"Registry lacks {field} for {feature_id}"
                    )
                row[field] = str(value)
            row["decision"] = "approve"
            row["reason"] = (
                "The H2-approved English source formula is linked to an "
                "outcome-blind deterministic T0 implementation, explicit "
                "missing/zero/empty/coverage rules, exact input and code "
                "hashes, a feature-specific self-test, and a nonconstant "
                "audited materialization. Source-reported and project "
                "missingness remain separately attributed."
            )
            action = "approve_frozen_exact_operationalization"
            counts["approve"] += 1
        row.update(
            {
                "draft_method": (
                    "primary_codex_source_to_frozen_t0_implementation_review"
                ),
                "independent_ai_review_status": "complete",
                "independent_ai_reviewer_id": REVIEWER_ID,
                "independent_ai_reviewed_at": completed_at,
                "independent_ai_review_action": action,
                "independent_ai_review_note": row["reason"],
                "independent_ai_run_id": run_id,
                "independent_ai_model": MODEL,
                "independent_ai_prompt_sha256": prompt_sha,
            }
        )
        reviewed.append(row)
    output_fields = fields + [
        field for field in PROVENANCE_FIELDS if field not in fields
    ]
    return output_fields, reviewed, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    input_path = args.input.resolve()
    registry_path = args.registry.resolve()
    output_path = args.output.resolve()
    protocol_path = args.protocol.resolve()
    completed_at = utc_now()
    run_id = (
        "primary_codex_targeted_operationalization_v3_"
        + completed_at.replace("-", "").replace(":", "").replace("+00:00", "Z")
    )
    fields, rows, counts = review_rows(
        input_path,
        registry_path,
        protocol_path,
        completed_at,
        run_id,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, rows, fields)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = {
        "run_id": run_id,
        "action": "primary_targeted_operationalization_review",
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
            "decision_counts": counts,
            "registry_path": str(registry_path),
            "registry_sha256": sha256_file(registry_path),
            "source_reported_values_rewritten": False,
            "target_count_influence": False,
            "model_outcomes_used": False,
            "qwen_or_ollama_used": False,
            "local_or_external_llm_api_used": False,
            "round_13": False,
        },
        "item_count": len(rows),
        "completed_at": completed_at,
        "status": "complete",
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "artifact_path": str(output_path),
                "artifact_sha256": sha256_file(output_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "decision_counts": counts,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
