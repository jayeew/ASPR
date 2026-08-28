from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from gear.graph_action_policy import (
    ACTION_POLICY_FEATURE_SCHEMA,
    ACTION_POLICY_FEATURES,
    ALL_ACTIONS,
    GRAPH_ACTIONS,
    ActionPolicyRule,
    FrozenGraphActionSelector,
    GraphActionQModel,
    load_graph_action_policy_release,
    promote_graph_action_policy_release,
)
from gear.graph_prior_contracts import ClaimInventoryEntry, GraphSignalBundle


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _model() -> GraphActionQModel:
    intercepts: dict[str, float] = dict.fromkeys(ALL_ACTIONS, 0.0)
    intercepts["antecedent_falsification"] = 0.5
    return GraphActionQModel(
        feature_schema_version=ACTION_POLICY_FEATURE_SCHEMA,
        feature_family="graph_features",
        feature_names=list(ACTION_POLICY_FEATURES),
        intercepts=intercepts,
        coefficients={
            action: [0.0] * len(ACTION_POLICY_FEATURES) for action in ALL_ACTIONS
        },
        rules={
            action: ActionPolicyRule(
                uplift_margin=0.1,
                development_rows=15,
                development_average_uplift=0.5,
                development_average_uplift_lcb=0.2,
                development_positive_uplift_pass=True,
                wrong_correction_pass=True,
                unsupported_claim_pass=True,
                cost_pass=True,
            )
            for action in GRAPH_ACTIONS
        },
        selection_rule="max_positive_q_minus_baseline_minus_uplift_margin_v1",
        tie_break="uplift_lcb_then_uplift_then_action_lexicographic_v1",
        fallback_action="abstain",
        future_features_used=False,
        training_rows=90,
        training_scope="development_only",
        sealed_holdout_used_for_fitting=False,
        gear_evidence_gap_status=(
            "phase_one_excluded_not_available_at_pre_retrieval_decision"
        ),
    )


def _randomized() -> pd.DataFrame:
    rows = []
    index = 0
    for split, count in (("development", 15), ("confirmatory_holdout", 10)):
        for action in ALL_ACTIONS:
            for _ in range(count):
                rows.append(
                    {
                        "paper_id": f"P{index}",
                        "context_id": f"C{index}",
                        "experiment_split": split,
                        "logged_action": action,
                        "propensity": 1.0 / 6.0,
                        "matched_budget": 20,
                        "policy_fold_id": (
                            str(index % 3) if split == "development" else "holdout"
                        ),
                    }
                )
                index += 1
    return pd.DataFrame(rows)


def _policy(
    randomized: pd.DataFrame,
    family: str,
    development_sha: str,
    model: GraphActionQModel,
) -> pd.DataFrame:
    frame = randomized[randomized["experiment_split"].eq("confirmatory_holdout")].copy()
    for feature in ACTION_POLICY_FEATURES:
        frame[feature] = 0.0
    q_values = model.predict([0.0] * len(ACTION_POLICY_FEATURES))
    for action in ALL_ACTIONS:
        frame[f"q_{action}"] = q_values[action]
    decision = model.decision([0.0] * len(ACTION_POLICY_FEATURES))
    frame["target_action"] = decision.action if decision.selected else "baseline"
    frame["outcome"] = 0.5
    frame["q_logged"] = [
        q_values[action] for action in frame["logged_action"].astype(str)
    ]
    frame["q_target"] = q_values[str(frame["target_action"].iloc[0])]
    frame["q_baseline"] = q_values["baseline"]
    frame["wrong_correction"] = False
    frame["unsupported_claim"] = False
    frame["realized_cost"] = 10.0
    frame["policy_feature_set"] = family
    frame["policy_development_input_sha256"] = development_sha
    frame["policy_holdout_input_sha256"] = "sha256:" + "a" * 64
    return frame


def _candidate(root: Path) -> dict[str, Path]:
    paths = {
        name: root / filename
        for name, filename in {
            "model": "model.json",
            "replay": "replay.json",
            "development": "development.parquet",
            "randomized": "randomized.parquet",
            "graph": "graph_policy.parquet",
            "no_graph": "no_graph_policy.parquet",
            "gate2": "gate2.json",
            "frozen": "freeze.json",
            "source_audit": "source_audit.json",
            "stage_a_audit": "stage_a_audit.json",
            "stage_b_audit": "stage_b_audit.json",
            "stage_c_audit": "stage_c_audit.json",
        }.items()
    }
    paths["release"] = root / "release_bundle" / "manifest.json"
    model = _model()
    paths["model"].write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    features = [0.0] * len(ACTION_POLICY_FEATURES)
    paths["replay"].write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "features": features,
                        "expected_q_values": model.predict(features),
                        "expected_decision": model.decision(features).model_dump(
                            mode="json"
                        ),
                    }
                ]
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    randomized = _randomized()
    development = randomized[randomized["experiment_split"].eq("development")]
    randomized.to_parquet(paths["randomized"], index=False)
    development.to_parquet(paths["development"], index=False)
    development_sha = _sha(paths["development"])
    _policy(randomized, "graph_features", development_sha, model).to_parquet(
        paths["graph"], index=False
    )
    _policy(randomized, "no_graph_features", development_sha, model).to_parquet(
        paths["no_graph"], index=False
    )
    binding = {
        "model_sha256": _sha(paths["model"]),
        "q_model_family": "linear_t0_v1",
        "feature_schema_version": ACTION_POLICY_FEATURE_SCHEMA,
        "feature_family": "graph_features",
        "development_data_sha256": development_sha,
        "randomized_data_sha256": _sha(paths["randomized"]),
        "graph_policy_sha256": _sha(paths["graph"]),
        "no_graph_policy_sha256": _sha(paths["no_graph"]),
        "future_features_used": False,
        "future_outcomes_used_at_inference": False,
        "sealed_holdout_used_for_fitting": False,
        "training_rows": 90,
        "training_scope": "development_only",
        "gear_evidence_gap_status": (
            "phase_one_excluded_not_available_at_pre_retrieval_decision"
        ),
    }
    paths["gate2"].write_text(
        json.dumps(
            {
                "contract": "gear_gate2_dual_holdout_and_paired_policy_v2",
                "status": "passed",
                "claim_allowed": True,
                "checks": {"all_registered_checks": True},
                "guardrails": {
                    "wrong_correction_pass": True,
                    "unsupported_claim_pass": True,
                    "cost_pass": True,
                },
                "graph_vs_no_graph_policy": {
                    "lcb_95": 0.1,
                    "both_abstain": False,
                    "paired_switch_dr_sensitivity": {"value": 0.1},
                },
                "action_policy_runtime_candidate": binding,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    digest = "sha256:" + "f" * 64
    paths["frozen"].write_text(
        json.dumps(
            {
                "contract": "gear_graph_rescue_frozen_replay_v1",
                "runtime_code_sha256": digest,
                "rescue_source_sha256": digest,
                "rescue_source_file_count": 12,
                "stage_ab_runtime_config_sha256": digest,
                "stage_c_runtime_config_sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    paths["source_audit"].write_text(
        json.dumps(
            {
                "contract": "gear_rescue_source_fingerprint_audit_v1",
                "passed": True,
                "source_sha256": digest,
                "source_file_count": 12,
            }
        ),
        encoding="utf-8",
    )
    for key, cases in (
        ("stage_a_audit", 120),
        ("stage_b_audit", 241),
        ("stage_c_audit", 150),
    ):
        paths[key].write_text(
            json.dumps(
                {
                    "contract": "gear_runtime_cohort_fingerprint_audit_v1",
                    "passed": True,
                    "cases": cases,
                    "runtime_code_sha256": digest,
                    "runtime_config_sha256": digest,
                    "runtime_source_file_count": 100,
                }
            ),
            encoding="utf-8",
        )
    return paths


def _promote(paths: dict[str, Path], release_id: str = "policy-test-v1"):
    return promote_graph_action_policy_release(
        model_path=paths["model"],
        replay_path=paths["replay"],
        development_data_path=paths["development"],
        randomized_data_path=paths["randomized"],
        graph_policy_path=paths["graph"],
        no_graph_policy_path=paths["no_graph"],
        gate2_report_path=paths["gate2"],
        frozen_replay_manifest_path=paths["frozen"],
        source_fingerprint_audit_path=paths["source_audit"],
        stage_a_runtime_audit_path=paths["stage_a_audit"],
        stage_b_runtime_audit_path=paths["stage_b_audit"],
        stage_c_runtime_audit_path=paths["stage_c_audit"],
        output_path=paths["release"],
        release_id=release_id,
    )


def test_model_has_exact_t0_schema_and_conservative_rule() -> None:
    decision = _model().decision([0.0] * len(ACTION_POLICY_FEATURES))
    assert decision.action == "antecedent_falsification"
    assert decision.predicted_uplift == pytest.approx(0.5)
    assert decision.uplift_lcb == pytest.approx(0.4)

    payload = _model().model_dump(mode="json")
    payload["feature_names"] = [*ACTION_POLICY_FEATURES, "future_outcome"]
    for action in ALL_ACTIONS:
        payload["coefficients"][action].append(0.0)
    with pytest.raises(ValueError, match="frozen T0 feature schema"):
        GraphActionQModel.model_validate(payload)


def test_missing_release_abstains_and_is_limited() -> None:
    decision = FrozenGraphActionSelector(None).decide(SimpleNamespace())
    assert decision.action == "abstain"
    assert decision.policy_status == "limited"


def test_complete_release_is_hash_bound_replayed_and_lazy(tmp_path: Path) -> None:
    paths = _candidate(tmp_path)
    promoted = _promote(paths)
    assert load_graph_action_policy_release(paths["release"])[0] == promoted
    state = SimpleNamespace(
        claim_inventory=[
            ClaimInventoryEntry(
                claim_id="C1",
                claim_type="method_claim",
                text="method",
                manuscript_evidence_keys=["P:1"],
                centrality=1.0,
            )
        ],
        cutoff_date=SimpleNamespace(year=2020),
        graph_signal_bundle=GraphSignalBundle(
            paper_id="P",
            expected_diffusion=0.5,
            field_year_base=0.1,
            reliability=1.0,
            shrunk_diffusion=0.5,
            structural_contribution_share=0.8,
            opportunity_context_share=0.2,
            perturbation_potential=0.4,
        ),
    )
    decision = FrozenGraphActionSelector(paths["release"]).decide(state)
    assert decision.action == "antecedent_falsification"
    assert decision.policy_release_id == "policy-test-v1"
    missing_p = SimpleNamespace(
        **{
            **state.__dict__,
            "graph_signal_bundle": state.graph_signal_bundle.model_copy(
                update={"perturbation_potential": None}
            ),
        }
    )
    limited = FrozenGraphActionSelector(paths["release"]).decide(missing_p)
    assert limited.action == "abstain"
    assert limited.policy_status == "limited"
    assert limited.reason == "action_policy_unavailable:ValueError"
    published_model = paths["release"].parent / promoted.model_path
    published_model.write_text(
        published_model.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_graph_action_policy_release(paths["release"])


def test_promotion_rejects_partial_stage_c(tmp_path: Path) -> None:
    paths = _candidate(tmp_path)
    pd.read_parquet(paths["randomized"]).iloc[:30].to_parquet(
        paths["randomized"], index=False
    )
    with pytest.raises(ValueError, match="complete 150/90"):
        _promote(paths, "must-not-publish")


def test_promotion_rejects_unbound_gate2(tmp_path: Path) -> None:
    paths = _candidate(tmp_path)
    report = json.loads(paths["gate2"].read_text(encoding="utf-8"))
    report.pop("action_policy_runtime_candidate")
    paths["gate2"].write_text(json.dumps(report) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not hash-bound"):
        _promote(paths, "must-not-publish")


def test_promotion_rejects_runtime_outside_frozen_replay_atomically(
    tmp_path: Path,
) -> None:
    paths = _candidate(tmp_path)
    audit = json.loads(paths["stage_c_audit"].read_text(encoding="utf-8"))
    audit["runtime_code_sha256"] = "sha256:" + "e" * 64
    paths["stage_c_audit"].write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(ValueError, match="Stage C runtime cohort"):
        _promote(paths, "must-not-publish")

    assert not paths["release"].parent.exists()
