from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import pandas as pd

from experiments.gear.evaluation.audit_rescue_completion import (
    CompletionArtifacts,
    audit_rescue_completion,
)
from gear.claim_attribution import T0_FEATURE_NAMES
from gear.graph_action_policy import ACTION_POLICY_FEATURES


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> CompletionArtifacts:
    paths = {
        item.name: tmp_path / f"{item.name}.json"
        for item in fields(CompletionArtifacts)
    }
    code = tmp_path / "model.py"
    config = tmp_path / "model.json"
    model = tmp_path / "model.bin"
    annotations = tmp_path / "annotations.jsonl"
    code.write_text("MODEL = 1\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    model.write_bytes(b"model")
    annotations.write_text("{}\n", encoding="utf-8")
    digest = "sha256:" + "a" * 64
    source_digest = digest
    _write_json(
        paths["frozen_replay_manifest"],
        {
            "contract": "gear_graph_rescue_frozen_replay_v1",
            "replay_id": "fixture",
            "runtime_code_sha256": digest,
            "stage_ab_runtime_config_sha256": digest,
            "stage_c_runtime_config_sha256": digest,
            "rescue_source_sha256": source_digest,
            "rescue_source_file_count": 1,
            "inputs": {"config": {"path": str(config), "sha256": _sha(config)}},
        },
    )
    _write_json(
        paths["source_fingerprint_audit"],
        {
            "contract": "gear_rescue_source_fingerprint_audit_v1",
            "passed": True,
            "source_sha256": source_digest,
            "source_file_count": 1,
        },
    )
    for stage in ("stage_a", "stage_b", "stage_c"):
        _write_json(
            paths[f"{stage}_runtime_audit"],
            {
                "contract": "gear_runtime_cohort_fingerprint_audit_v1",
                "passed": True,
                "cases": 100,
                "runtime_code_sha256": digest,
                "runtime_config_sha256": digest,
                "runtime_source_file_count": 1,
                "config_version": "test",
                "status_counts": {"completed": 100},
            },
        )
    _write_json(
        paths["stage_a_validation"],
        {
            "conclusion": {
                "stage_a_established": True,
                "claim_allowed": True,
                "verdict": "supported",
            },
            "integration": {
                "real_hgb": {"integration_value": 0.2, "joint_spearman": 0.4},
                "shuffled_hgb": {"integration_value": 0.05, "joint_spearman": 0.1},
                "real_minus_shuffled_value": 0.15,
            },
        },
    )
    _write_json(
        paths["stage_a_gate0"],
        {"status": "passed", "checks": {f"c{i}": True for i in range(7)}},
    )
    three_arm = tmp_path / "stage_a_three_arm.csv"
    pd.DataFrame(
        {
            "paper_id": [f"W{i}" for i in range(100)],
            "gear_evidence_score": [i / 100 for i in range(100)],
            "joint_structural_score": [i / 99 for i in range(100)],
            "shuffled_structural_score": [(99 - i) / 99 for i in range(100)],
            "future_structural_outcome": [i / 99 for i in range(100)],
        }
    ).to_csv(three_arm, index=False)
    paths["stage_a_three_arm"] = three_arm
    _write_json(
        paths["hgb_p_validation"],
        {
            "contract": "gear_claim_a_bounded_validation_v1",
            "status": "supported_with_limitations",
            "claim_allowed": True,
            "uses_future_features": False,
            "claim_boundaries": {
                "hgb_d_oof": "supported",
                "hgb_p_forward_temporal": "supported",
                "hgb_p_leave_one_domain_out": "supported",
                "registered_target_coverage": "supported",
                "worst_group_consistency": "not_claimed",
            },
            "structural_heads": {
                axis: {
                    head: {"spearman_ci95_low": 0.01, "real_minus_permuted": 0.1}
                    for head in ("d_excess", "perturbation")
                }
                for axis in ("forward_temporal_latest", "leave_one_domain_out")
            },
            "coverage": {
                key: {"passed": True}
                for key in ("stage_b_241", "stage_c_150", "runtime_10")
            },
        },
    )
    _write_json(
        paths["stage_b_evidence_audit"],
        {
            "contract": "gear_rescue_randomized_outcome_audit_v1",
            "passed": True,
            "target_papers": 241,
            "claim_evidence_target_papers": 241,
            "paper_evidence_target_papers": 241,
        },
    )
    _write_json(
        paths["claim_adoption_summary"],
        {
            "contract": "gear_real_claim_adoption_labels_v1",
            "papers_resolved": 241,
            "papers_labeled": 241,
            "failed_papers": [],
            "labeler_blind_to_hgb": True,
            "claim_labels": 1434,
        },
    )
    _write_json(
        paths["claim_gate_coverage_audit"],
        {"passed": True, "resolved_label_papers": 241},
    )
    gate1 = {
        "gate": "gate1",
        "status": "passed",
        "claim_allowed": True,
        "papers": 144,
        "score_deciles": 8,
        "checks": {"attribution": True, "shuffle": True},
    }
    _write_json(paths["gate1_temporal"], gate1)
    _write_json(paths["gate1_domain"], gate1)
    _write_json(
        paths["stage_c_randomized_audit"],
        {
            "contract": "gear_rescue_randomized_outcome_audit_v1",
            "passed": True,
            "cases": 150,
            "split_counts": {"development": 90, "confirmatory_holdout": 60},
        },
    )
    action_counts = {
        **{f"development:{action}": 15 for action in _actions()},
        **{f"confirmatory_holdout:{action}": 10 for action in _actions()},
    }
    _write_json(
        paths["stage_c_outcome_report"],
        {
            "contract": "gear_randomized_action_outcomes_v1",
            "manifest_cases": 150,
            "included_cases": 150,
            "excluded_cases": 0,
            "identifiable": True,
            "all_actions_observed": True,
            "all_executions_verified": True,
            "matched_resource_caps": True,
            "resource_caps_sha256": digest,
            "action_counts": action_counts,
        },
    )
    holdout_sha = "sha256:" + "b" * 64
    policy = {
        "contract": "gear_selective_graph_policy_holdout_v2",
        "development_rows": 90,
        "confirmatory_holdout_rows": 60,
        "development_input_sha256": holdout_sha,
        "holdout_input_sha256": holdout_sha,
    }
    _write_json(
        paths["policy_graph_report"], {**policy, "policy_feature_set": "graph_features"}
    )
    _write_json(
        paths["policy_no_graph_report"],
        {**policy, "policy_feature_set": "no_graph_features"},
    )
    gate2_checks = {
        name: True
        for name in (
            "temporal_joint_beats_gear_only",
            "temporal_real_hgb_beats_shuffle",
            "domain_joint_beats_gear_only",
            "domain_real_hgb_beats_shuffle",
            "policy_positive_lcb_or_abstains",
            "wrong_correction_not_worse",
            "unsupported_claim_not_worse",
            "cost_not_worse",
            "graph_policy_beats_no_graph_or_both_abstain",
            "paired_policy_contrast_finite",
        )
    }
    _write_json(
        paths["gate2_report"],
        {
            "contract": "gear_gate2_dual_holdout_and_paired_policy_v2",
            "gate": "gate2",
            "status": "passed",
            "claim_allowed": True,
            "paired_holdout_rows": 60,
            "checks": gate2_checks,
            "graph_vs_no_graph_policy": {
                "lcb_95": 0.1,
                "both_abstain": False,
                "paired_switch_dr_sensitivity": {},
            },
            "guardrails": {
                "wrong_correction_pass": True,
                "unsupported_claim_pass": True,
                "cost_pass": True,
            },
        },
    )
    structural_assets = {}
    for name in (
        "model",
        "feature_registry",
        "training_reference",
        "prediction_table",
        "validation_report",
        "runtime_replay",
        "temporal_oof_audit",
        "domain_oof_audit",
        "coverage_audit",
    ):
        target = tmp_path / f"structural_{name}.json"
        target.write_text(
            (
                json.dumps(
                    {
                        "contract": "gear_structural_head_coverage_audit_v1",
                        "passed": True,
                        "stage_b_241": {
                            "expected_papers": 241,
                            "available_papers": 241,
                            "passed": True,
                        },
                        "stage_c_150": {
                            "expected_papers": 150,
                            "available_papers": 150,
                            "passed": True,
                        },
                        "runtime_10": {
                            "expected_papers": 10,
                            "available_papers": 10,
                            "passed": True,
                        },
                    }
                )
                + "\n"
                if name == "coverage_audit"
                else (
                    json.dumps(
                        {
                            "contract": "gear_structural_head_validation_v1",
                            "status": "supported",
                            "promotion_passed": True,
                            "uses_future_features": False,
                        }
                    )
                    + "\n"
                    if name == "validation_report"
                    else "{}\n"
                )
            ),
            encoding="utf-8",
        )
        structural_assets[name] = {
            "file": target.name,
            "sha256": _sha(target),
            "size_bytes": target.stat().st_size,
        }
    _write_json(
        paths["structural_head_manifest"],
        {
            "contract": "gear_structural_head_release_v1",
            "release_id": "s1",
            "status": "promoted",
            "feature_time_basis": "T0_only",
            "uses_future_features": False,
            "historical_prediction_policy": "strict_oof_only",
            "assets": structural_assets,
        },
    )
    gate1_bound = []
    for axis in ("temporal", "domain"):
        target = tmp_path / f"bound_gate1_{axis}.json"
        _write_json(
            target, {"claim_attribution_runtime_candidate": {"evaluation_axis": axis}}
        )
        gate1_bound.append(target)
    _write_json(
        paths["claim_attribution_manifest"],
        {
            "contract": "gear_claim_attribution_release_v1",
            "release_id": "c1",
            "status": "promoted",
            "feature_names": list(T0_FEATURE_NAMES),
            "training_features_t0_only": True,
            "development_only": True,
            "sealed_holdout_labels_used": False,
            "future_contexts_used_at_inference": False,
            "model_path": model.name,
            "model_sha256": _sha(model),
            "replay_path": config.name,
            "replay_sha256": _sha(config),
            "gate1_report_paths": [item.name for item in gate1_bound],
            "gate1_report_sha256": [_sha(item) for item in gate1_bound],
        },
    )
    action_refs = {}
    for stem in (
        "model",
        "replay",
        "development_data",
        "randomized_data",
        "graph_policy",
        "no_graph_policy",
    ):
        target = tmp_path / f"action_{stem}.json"
        target.write_text("{}\n", encoding="utf-8")
        action_refs[f"{stem}_path"] = target.name
        action_refs[f"{stem}_sha256"] = _sha(target)
    action_refs["gate2_report_path"] = paths["gate2_report"].name
    action_refs["gate2_report_sha256"] = _sha(paths["gate2_report"])
    for stem, target in (
        ("frozen_replay_manifest", paths["frozen_replay_manifest"]),
        ("source_fingerprint_audit", paths["source_fingerprint_audit"]),
        ("stage_a_runtime_audit", paths["stage_a_runtime_audit"]),
        ("stage_b_runtime_audit", paths["stage_b_runtime_audit"]),
        ("stage_c_runtime_audit", paths["stage_c_runtime_audit"]),
    ):
        action_refs[f"{stem}_path"] = target.name
        action_refs[f"{stem}_sha256"] = _sha(target)
    _write_json(
        paths["action_policy_manifest"],
        {
            "contract": "gear_graph_action_policy_release_v1",
            "release_id": "a1",
            "status": "promoted",
            "feature_family": "graph_features",
            "feature_names": list(ACTION_POLICY_FEATURES),
            "development_rows": 90,
            "randomized_rows": 150,
            "future_features_used": False,
            "future_outcomes_used_at_inference": False,
            "sealed_holdout_used_for_fitting": False,
            **action_refs,
        },
    )
    _write_json(
        paths["expert_pack_manifest"],
        {
            "contract": "gear_independent_review_pack_manifest_v1",
            "status": "ready_for_review",
            "review_policy": {
                "minimum_sessions_per_task": 1,
                "ai_sessions_accepted": True,
                "human_reviewer_required": False,
                "blinding_required": False,
                "reviewer_calibration_required": False,
                "adjudication_required": False,
            },
        },
    )
    _write_json(
        paths["expert_pack_validation"],
        {
            "contract": "gear_independent_review_pack_validation_v1",
            "valid": True,
            "completed_annotations_validated": True,
            "claim_b_tasks": 30,
            "claim_c_tasks": 30,
        },
    )
    _write_json(
        paths["final_rescue_status"],
        {
            "contract": "gear_graph_calibrated_rescue_plan_v1",
            "overall_claim_allowed": True,
            **{
                stage: {
                    "implementation": "complete",
                    "validation": {"status": "passed", "claim_allowed": True},
                }
                for stage in ("stage_a", "stage_b", "stage_c")
            },
        },
    )
    return CompletionArtifacts(**paths)


def _actions() -> tuple[str, ...]:
    return (
        "baseline",
        "antecedent_falsification",
        "remote_mechanism_analogue",
        "cross_field_pathway",
        "topology_expansion",
        "opportunity_attribution_audit",
    )


def test_completion_audit_passes_only_complete_bound_artifacts(tmp_path: Path) -> None:
    result = audit_rescue_completion(_fixture(tmp_path))
    assert result["claim_allowed"] is True
    assert result["blockers"] == []
    assert all(check["passed"] for check in result["checks"].values())


def test_completion_audit_accepts_registered_fail_closed_abstention(
    tmp_path: Path,
) -> None:
    artifacts = _fixture(tmp_path)
    manifest = json.loads(artifacts.action_policy_manifest.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "abstained",
            "runtime_enabled": False,
            "target_action": "baseline",
            "abstention_reason": "all development uplift lower bounds non-positive",
        }
    )
    for stem in (
        "model",
        "replay",
        "frozen_replay_manifest",
        "source_fingerprint_audit",
        "stage_a_runtime_audit",
        "stage_b_runtime_audit",
        "stage_c_runtime_audit",
    ):
        manifest.pop(f"{stem}_path")
        manifest.pop(f"{stem}_sha256")
    graph = json.loads(artifacts.policy_graph_report.read_text(encoding="utf-8"))
    no_graph = json.loads(artifacts.policy_no_graph_report.read_text(encoding="utf-8"))
    for policy in (graph, no_graph):
        policy["selective_abstain"] = True
        policy["target_action_counts"] = {"baseline": 60}
    graph["rules"] = {"topology_expansion": {"development_average_uplift_lcb": -0.01}}
    _write_json(artifacts.policy_graph_report, graph)
    _write_json(artifacts.policy_no_graph_report, no_graph)
    gate2 = json.loads(artifacts.gate2_report.read_text(encoding="utf-8"))
    gate2["graph_vs_no_graph_policy"]["both_abstain"] = True
    _write_json(artifacts.gate2_report, gate2)
    for stem, target in (
        ("graph_policy", artifacts.policy_graph_report),
        ("no_graph_policy", artifacts.policy_no_graph_report),
        ("gate2_report", artifacts.gate2_report),
    ):
        manifest[f"{stem}_path"] = target.name
        manifest[f"{stem}_sha256"] = _sha(target)
    _write_json(artifacts.action_policy_manifest, manifest)
    result = audit_rescue_completion(artifacts)
    assert result["checks"]["promoted_action_policy"]["passed"] is True
    assert result["checks"]["promoted_action_policy"]["evidence"]["deployment"] == (
        "fail_closed_baseline"
    )


def test_completion_audit_fails_closed_on_missing_and_partial(tmp_path: Path) -> None:
    artifacts = _fixture(tmp_path)
    Path(artifacts.gate1_domain).unlink()
    hgb = json.loads(artifacts.hgb_p_validation.read_text(encoding="utf-8"))
    hgb["claim_boundaries"]["hgb_p_leave_one_domain_out"] = "not_claimed"
    _write_json(artifacts.hgb_p_validation, hgb)
    result = audit_rescue_completion(artifacts)
    assert result["claim_allowed"] is False
    assert {row["check"] for row in result["blockers"]} >= {
        "hgb_p_claim_a",
        "gate1_domain",
    }


def test_completion_audit_detects_hash_and_pairing_mismatch(tmp_path: Path) -> None:
    artifacts = _fixture(tmp_path)
    runtime = json.loads(artifacts.stage_b_runtime_audit.read_text(encoding="utf-8"))
    runtime["runtime_code_sha256"] = "sha256:" + "c" * 64
    _write_json(artifacts.stage_b_runtime_audit, runtime)
    policy = json.loads(artifacts.policy_no_graph_report.read_text(encoding="utf-8"))
    policy["holdout_input_sha256"] = "sha256:" + "d" * 64
    _write_json(artifacts.policy_no_graph_report, policy)
    promotion = json.loads(
        artifacts.structural_head_manifest.read_text(encoding="utf-8")
    )
    asset = promotion["assets"]["model"]
    (artifacts.structural_head_manifest.parent / asset["file"]).write_bytes(
        b"mutated-after-promotion"
    )
    result = audit_rescue_completion(artifacts)
    assert result["checks"]["frozen_runtime_replay"]["passed"] is False
    assert result["checks"]["paired_graph_policy_gate2"]["passed"] is False
    assert result["checks"]["promoted_structural_head"]["passed"] is False


def test_completion_audit_accepts_ai_session_without_human_blind_or_calibration_gate(
    tmp_path: Path,
) -> None:
    artifacts = _fixture(tmp_path)
    result = audit_rescue_completion(artifacts)
    check = result["checks"]["independent_claim_b_c_review"]
    assert result["claim_allowed"] is True
    assert check["passed"] is True
    assert check["evidence"]["protocol"] == "independent_session"
    assert check["evidence"]["ai_sessions_accepted"] is True
