"""Orchestrate Stage A/B/C gates without manufacturing unavailable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .rescue_stage_gates import evaluate_gate1, evaluate_gate2_confirmatory
from .run_stage_a_validation import run_validation


def run_rescue_plan(
    output_dir: Path,
    *,
    stage_a_gear_evidence_path: Path | None = None,
    stage_a_per_decile: int = 20,
    gate1_path: Path | None = None,
    gate2_temporal_path: Path | None = None,
    gate2_domain_path: Path | None = None,
    gate2_policy_path: Path | None = None,
    gate2_no_graph_policy_path: Path | None = None,
    stage_c_randomized_data_path: Path | None = None,
) -> dict[str, Any]:
    """Run every implemented stage and return explicit readiness statuses."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_a = run_validation(
        output_dir / "stage_a",
        per_decile=stage_a_per_decile,
        gear_evidence_path=stage_a_gear_evidence_path,
    )
    gate1 = (
        evaluate_gate1(pd.read_parquet(gate1_path))
        if gate1_path is not None and gate1_path.is_file()
        else _missing("gate1", "matched claim-level mechanism dataset unavailable")
    )
    gate2 = (
        evaluate_gate2_confirmatory(
            _read_frame(gate2_temporal_path),
            _read_frame(gate2_domain_path),
            _read_frame(gate2_policy_path),
            _read_frame(gate2_no_graph_policy_path),
        )
        if gate2_temporal_path is not None
        and gate2_domain_path is not None
        and gate2_policy_path is not None
        and gate2_no_graph_policy_path is not None
        and gate2_temporal_path.is_file()
        and gate2_domain_path.is_file()
        and gate2_policy_path.is_file()
        and gate2_no_graph_policy_path.is_file()
        else _missing(
            "gate2",
            "frozen temporal/domain integration or paired policy holdouts unavailable",
        )
    )
    if gate2.get("status") == "passed":
        gate2["action_policy_runtime_candidate"] = _runtime_policy_binding(
            gate2_policy_path,
            gate2_no_graph_policy_path,
            stage_c_randomized_data_path,
        )
    (output_dir / "gate2_report.json").write_text(
        json.dumps(gate2, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "contract": "gear_graph_calibrated_rescue_plan_v1",
        "stage_a": {
            "implementation": "complete",
            "validation": stage_a["conclusion"],
        },
        "stage_b": {
            "implementation": "complete",
            "validation": gate1,
        },
        "stage_c": {
            "implementation": "complete",
            "validation": gate2,
        },
        "overall_claim_allowed": bool(
            stage_a["conclusion"]["claim_allowed"]
            and gate1.get("claim_allowed")
            and gate2.get("claim_allowed")
        ),
    }
    (output_dir / "rescue_plan_status.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _read_frame(path: Path) -> pd.DataFrame:
    return (
        pd.read_csv(path) if path.suffix.casefold() == ".csv" else pd.read_parquet(path)
    )


def _missing(gate: str, reason: str) -> dict[str, Any]:
    return {
        "gate": gate,
        "status": "not_identifiable",
        "reason": reason,
        "claim_allowed": False,
    }


def _runtime_policy_binding(
    graph_policy_path: Path | None,
    no_graph_policy_path: Path | None,
    randomized_data_path: Path | None,
) -> dict[str, Any]:
    if any(
        path is None or not path.is_file()
        for path in (
            graph_policy_path,
            no_graph_policy_path,
            randomized_data_path,
        )
    ):
        raise ValueError("passed Gate 2 lacks complete runtime policy inputs")
    assert graph_policy_path is not None
    assert no_graph_policy_path is not None
    assert randomized_data_path is not None
    policy_report_path = graph_policy_path.parent / "policy_holdout_report.json"
    if not policy_report_path.is_file():
        raise ValueError("passed Gate 2 lacks the Graph policy training report")
    policy_report = json.loads(policy_report_path.read_text(encoding="utf-8"))
    candidate = policy_report.get("runtime_candidate") or {}
    model_path = graph_policy_path.parent / str(candidate.get("model_path", ""))
    replay_path = graph_policy_path.parent / str(candidate.get("replay_path", ""))
    if not model_path.is_file() or not replay_path.is_file():
        raise ValueError("passed Gate 2 lacks the frozen Q model or replay")
    expected_model_hash = _sha256(model_path)
    expected_replay_hash = _sha256(replay_path)
    if (
        candidate.get("model_sha256") != expected_model_hash
        or candidate.get("replay_sha256") != expected_replay_hash
    ):
        raise ValueError("Graph policy training report artifact hash mismatch")
    return {
        "model_sha256": expected_model_hash,
        "replay_sha256": expected_replay_hash,
        "q_model_family": candidate.get("q_model_family"),
        "feature_schema_version": candidate.get("feature_schema_version"),
        "feature_family": candidate.get("feature_family"),
        "development_data_sha256": policy_report.get("development_input_sha256"),
        "randomized_data_sha256": _sha256(randomized_data_path),
        "graph_policy_sha256": _sha256(graph_policy_path),
        "no_graph_policy_sha256": _sha256(no_graph_policy_path),
        "future_features_used": candidate.get("future_features_used"),
        "future_outcomes_used_at_inference": candidate.get(
            "future_outcomes_used_at_inference"
        ),
        "sealed_holdout_used_for_fitting": candidate.get(
            "sealed_holdout_used_for_fitting"
        ),
        "training_rows": candidate.get("training_rows"),
        "training_scope": candidate.get("training_scope"),
        "gear_evidence_gap_status": candidate.get("gear_evidence_gap_status"),
    }


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage-a-gear-evidence", type=Path)
    parser.add_argument("--stage-a-per-decile", type=int, default=20)
    parser.add_argument("--gate1-data", type=Path)
    parser.add_argument("--gate2-temporal", type=Path)
    parser.add_argument("--gate2-domain", type=Path)
    parser.add_argument("--gate2-policy", type=Path)
    parser.add_argument("--gate2-no-graph-policy", type=Path)
    parser.add_argument("--stage-c-randomized-data", type=Path)
    args = parser.parse_args()
    result = run_rescue_plan(
        args.output_dir,
        stage_a_gear_evidence_path=args.stage_a_gear_evidence,
        stage_a_per_decile=args.stage_a_per_decile,
        gate1_path=args.gate1_data,
        gate2_temporal_path=args.gate2_temporal,
        gate2_domain_path=args.gate2_domain,
        gate2_policy_path=args.gate2_policy,
        gate2_no_graph_policy_path=args.gate2_no_graph_policy,
        stage_c_randomized_data_path=args.stage_c_randomized_data,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
