from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.gear.evaluation.collect_randomized_action_outcomes import (
    collect_outcomes,
)
from experiments.gear.evaluation.graph_action_randomized_runner import ACTIONS
from experiments.gear.evaluation.run_action_policy_evaluation import (
    run_policy_evaluation,
)
from gear.graph_action_policy import (
    ACTION_POLICY_FEATURE_SCHEMA,
    ACTION_POLICY_FEATURES,
    GraphActionQModel,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_collect_outcomes_requires_executed_assignment(tmp_path: Path) -> None:
    manifest = {
        "randomization_precedes_outcomes": True,
        "cases": [
            {
                "case_id": "W1",
                "paper_id": "https://openalex.org/W1",
                "context_id": "CTX-1",
                "assigned_action": "antecedent_falsification",
                "propensity": 1 / 6,
                "matched_budget": 20,
                "experiment_split": "development",
                "score_decile": 7,
                "metadata": {"domain": "physics"},
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    run_dir = tmp_path / "runs" / "W1"
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "review_bundle.json",
        {
            "state": {
                "graph_action_decision": {
                    "action": "antecedent_falsification",
                    "reason": "preassigned_randomized_action",
                },
                "claim_inventory": [{"claim_id": "C-1"}, {"claim_id": "C-2"}],
                "canonical_points": {
                    "P-1": {
                        "point_id": "P-1",
                        "contribution_id": "C-1",
                        "relation_evidence_keys": ["R:1"],
                        "novelty_resolution": "antecedent_found",
                    }
                },
                "resource_ledger": {
                    "caps": {
                        "provider_searches": 4,
                        "direct_fetches": 4,
                        "neighbor_expansions": 1,
                        "fulltext_candidates": 6,
                        "relation_classifications": 8,
                    },
                    "logical_provider_searches": 2,
                    "logical_direct_fetches": 1,
                    "logical_neighbor_expansions": 0,
                    "relation_classification_calls": 2,
                    "retrieval_model_calls": 0,
                },
                "action_budget": {"total_actions_max": 20, "actions_used": 4},
                "graph_guidance_plan": {
                    "controller_state": {
                        "selected_graph_action": "antecedent_falsification"
                    },
                    "claim_guidance": [
                        {
                            "missions": [
                                {
                                    "mission_type": "antecedent_falsification",
                                    "query_roles": ["author_terminology"],
                                }
                            ]
                        }
                    ],
                },
            }
        },
    )
    _write_json(
        run_dir / "validation_report.json",
        {"semantic_verification_available": True, "unsupported_major_count": 0},
    )
    (run_dir / "evidence_trace.jsonl").write_text(
        json.dumps({"evidence_id": "R:1", "kind": "prior_relation"}) + "\n",
        encoding="utf-8",
    )

    report = collect_outcomes(manifest_path, tmp_path / "runs", tmp_path / "out")

    assert report["included_cases"] == 1
    row = pd.read_parquet(tmp_path / "out" / "development_action_log.parquet").iloc[0]
    assert row["useful_relation_yield"] == 0.5
    assert row["correction_quality"] == 1.0
    assert row["claim_recall_gain"] == 0.5
    assert not bool(row["wrong_correction"])
    assert row["realized_cost"] == 5.0


def test_collect_outcomes_fails_closed_on_action_mismatch(tmp_path: Path) -> None:
    manifest = {
        "randomization_precedes_outcomes": True,
        "cases": [
            {
                "case_id": "W1",
                "paper_id": "W1",
                "context_id": "CTX-1",
                "assigned_action": "baseline",
                "propensity": 1 / 6,
                "matched_budget": 20,
                "experiment_split": "confirmatory_holdout",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    run_dir = tmp_path / "runs" / "W1"
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "review_bundle.json",
        {
            "state": {
                "graph_action_decision": {
                    "action": "topology_expansion",
                    "reason": "preassigned_randomized_action",
                }
            }
        },
    )
    _write_json(
        run_dir / "validation_report.json",
        {"semantic_verification_available": True},
    )
    (run_dir / "evidence_trace.jsonl").write_text("", encoding="utf-8")

    report = collect_outcomes(manifest_path, tmp_path / "runs", tmp_path / "out")

    assert report["included_cases"] == 0
    audit = pd.read_parquet(tmp_path / "out" / "randomized_action_audit.parquet")
    assert audit.iloc[0]["exclusion_reason"] == "assignment_runtime_mismatch"


def test_collect_outcomes_normalizes_mixed_policy_fold_ids(tmp_path: Path) -> None:
    cases = [
        {
            "case_id": "W-dev",
            "paper_id": "W-dev",
            "context_id": "CTX-dev",
            "assigned_action": "baseline",
            "propensity": 1 / 6,
            "matched_budget": 20,
            "experiment_split": "development",
            "policy_fold_id": 0,
        },
        {
            "case_id": "W-holdout",
            "paper_id": "W-holdout",
            "context_id": "CTX-holdout",
            "assigned_action": "baseline",
            "propensity": 1 / 6,
            "matched_budget": 20,
            "experiment_split": "confirmatory_holdout",
            "policy_fold_id": "holdout",
        },
    ]
    manifest_path = tmp_path / "manifest.json"
    _write_json(
        manifest_path,
        {"randomization_precedes_outcomes": True, "cases": cases},
    )
    for case in cases:
        run_dir = tmp_path / "runs" / str(case["case_id"])
        run_dir.mkdir(parents=True)
        _write_json(
            run_dir / "review_bundle.json",
            {
                "state": {
                    "graph_action_decision": {
                        "action": "baseline",
                        "reason": "preassigned_randomized_action",
                    },
                    "claim_inventory": [],
                    "canonical_points": {},
                    "resource_ledger": {
                        "caps": {
                            "provider_searches": 4,
                            "direct_fetches": 4,
                            "neighbor_expansions": 1,
                            "fulltext_candidates": 6,
                            "relation_classifications": 8,
                        },
                        "logical_provider_searches": 1,
                    },
                    "action_budget": {"total_actions_max": 20, "actions_used": 1},
                    "graph_guidance_plan": {
                        "controller_state": {"selected_graph_action": "baseline"},
                        "claim_guidance": [
                            {
                                "missions": [
                                    {
                                        "mission_type": "antecedent_falsification",
                                        "query_roles": ["author_terminology"],
                                    }
                                ]
                            }
                        ],
                    },
                }
            },
        )
        _write_json(
            run_dir / "validation_report.json",
            {"semantic_verification_available": True, "unsupported_major_count": 0},
        )
        (run_dir / "evidence_trace.jsonl").write_text("", encoding="utf-8")

    report = collect_outcomes(manifest_path, tmp_path / "runs", tmp_path / "out")

    assert report["included_cases"] == 2
    log = pd.read_parquet(tmp_path / "out" / "randomized_action_log.parquet")
    assert set(log["policy_fold_id"]) == {"0", "holdout"}


def test_collect_outcomes_rejects_declared_but_unexecuted_topology(
    tmp_path: Path,
) -> None:
    case = {
        "case_id": "W1",
        "paper_id": "W1",
        "context_id": "CTX-1",
        "assigned_action": "topology_expansion",
        "propensity": 1 / 6,
        "matched_budget": 20,
        "experiment_split": "development",
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(
        manifest_path,
        {"randomization_precedes_outcomes": True, "cases": [case]},
    )
    run_dir = tmp_path / "runs" / "W1"
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "review_bundle.json",
        {
            "state": {
                "graph_action_decision": {
                    "action": "topology_expansion",
                    "reason": "preassigned_randomized_action",
                },
                "claim_inventory": [{"claim_id": "C1"}],
                "canonical_points": {},
                "action_budget": {"total_actions_max": 20, "actions_used": 3},
                "resource_ledger": {
                    "caps": {
                        "provider_searches": 4,
                        "direct_fetches": 4,
                        "neighbor_expansions": 1,
                        "fulltext_candidates": 6,
                        "relation_classifications": 8,
                    },
                    "logical_provider_searches": 2,
                    "logical_neighbor_expansions": 0,
                },
                "graph_guidance_plan": {
                    "controller_state": {"selected_graph_action": "topology_expansion"},
                    "claim_guidance": [
                        {
                            "missions": [
                                {
                                    "mission_type": "topology_expansion",
                                    "query_roles": ["author_terminology"],
                                }
                            ]
                        }
                    ],
                },
            }
        },
    )
    _write_json(
        run_dir / "validation_report.json",
        {"semantic_verification_available": True, "unsupported_major_count": 0},
    )
    (run_dir / "evidence_trace.jsonl").write_text("", encoding="utf-8")

    report = collect_outcomes(manifest_path, tmp_path / "runs", tmp_path / "out")

    assert report["included_cases"] == 0
    audit = pd.read_parquet(tmp_path / "out" / "randomized_action_audit.parquet")
    assert (
        audit.iloc[0]["exclusion_reason"]
        == "assigned_action_not_executed:no_topology_expansion"
    )


def test_policy_evaluation_scores_never_fit_holdout(tmp_path: Path) -> None:
    development_rows = []
    holdout_rows = []
    for action_index, action in enumerate(ACTIONS):
        for index in range(15):
            development_rows.append(
                {
                    "paper_id": f"development-{action_index}-{index}",
                    "context_id": f"development-context-{action_index}-{index}",
                    "experiment_split": "development",
                    "logged_action": action,
                    "outcome": float(action_index == 1),
                    "wrong_correction": False,
                    "unsupported_claim": False,
                    "realized_cost": 1.0,
                    "propensity": 1 / 6,
                    "matched_budget": 20,
                    "policy_fold_id": index % 3,
                    "feature": float(index) / 15,
                }
            )
        for index in range(10):
            holdout_rows.append(
                {
                    "paper_id": f"holdout-{action_index}-{index}",
                    "context_id": f"holdout-context-{action_index}-{index}",
                    "experiment_split": "confirmatory_holdout",
                    "logged_action": action,
                    "outcome": float(action_index == 1),
                    "wrong_correction": False,
                    "unsupported_claim": False,
                    "realized_cost": 1.0,
                    "propensity": 1 / 6,
                    "matched_budget": 20,
                    "policy_fold_id": "holdout",
                    "feature": float(index) / 10,
                }
            )
    development = tmp_path / "development.parquet"
    holdout = tmp_path / "holdout.parquet"
    pd.DataFrame(development_rows).to_parquet(development, index=False)
    pd.DataFrame(holdout_rows).to_parquet(holdout, index=False)

    report = run_policy_evaluation(
        development,
        holdout,
        tmp_path / "policy",
        feature_columns=["feature"],
    )

    assert report["development_rows"] == 90
    assert report["confirmatory_holdout_rows"] == 60
    assert report["doubly_robust"]["n"] == 60
    assert report["paired_doubly_robust_uplift"]["n"] == 60
    assert report["development_input_sha256"].startswith("sha256:")
    assert report["holdout_input_sha256"].startswith("sha256:")
    scored = pd.read_parquet(tmp_path / "policy" / "policy_holdout_scored.parquet")
    assert scored["q_logged"].notna().all()
    assert scored["policy_holdout_input_sha256"].nunique() == 1


def test_graph_policy_evaluation_emits_frozen_linear_q_candidate(
    tmp_path: Path,
) -> None:
    development_rows = []
    holdout_rows = []
    for action_index, action in enumerate(ACTIONS):
        for index in range(15):
            row = _policy_row(
                split="development",
                action=action,
                action_index=action_index,
                index=index,
            )
            row["policy_fold_id"] = index % 3
            development_rows.append(row)
        for index in range(10):
            holdout_rows.append(
                _policy_row(
                    split="confirmatory_holdout",
                    action=action,
                    action_index=action_index,
                    index=index,
                )
            )
    development_path = tmp_path / "development.parquet"
    holdout_path = tmp_path / "holdout.parquet"
    pd.DataFrame(development_rows).to_parquet(development_path, index=False)
    pd.DataFrame(holdout_rows).to_parquet(holdout_path, index=False)

    output_dir = tmp_path / "graph_policy"
    report = run_policy_evaluation(
        development_path,
        holdout_path,
        output_dir,
        feature_columns=list(ACTION_POLICY_FEATURES),
    )

    model_path = output_dir / "graph_action_q_model.json"
    replay_path = output_dir / "graph_action_policy_replay.json"
    model = GraphActionQModel.model_validate_json(model_path.read_text())
    replay = json.loads(replay_path.read_text())
    assert report["runtime_candidate"]["model_path"] == model_path.name
    assert report["runtime_candidate"]["replay_path"] == replay_path.name
    assert report["runtime_candidate"]["feature_schema_version"] == (
        ACTION_POLICY_FEATURE_SCHEMA
    )
    assert model.feature_names == list(ACTION_POLICY_FEATURES)
    assert model.training_rows == 90
    assert len(replay["rows"]) == 60
    assert all(len(row["features"]) == 9 for row in replay["rows"])
    scored = pd.read_parquet(output_dir / "policy_holdout_scored.parquet")
    assert scored[list(ACTION_POLICY_FEATURES)].notna().all().all()


def _policy_row(
    *, split: str, action: str, action_index: int, index: int
) -> dict[str, object]:
    prefix = "development" if split == "development" else "holdout"
    scale = float(action_index * 20 + index + 1)
    return {
        "paper_id": f"{prefix}-{action_index}-{index}",
        "context_id": f"{prefix}-context-{action_index}-{index}",
        "experiment_split": split,
        "logged_action": action,
        "outcome": float(action_index == 1) + scale / 1000.0,
        "wrong_correction": False,
        "unsupported_claim": False,
        "realized_cost": 1.0,
        "propensity": 1 / 6,
        "matched_budget": 20,
        "policy_fold_id": "holdout",
        "claim_count": 1.0 + scale,
        "mean_claim_centrality": (index + 1) / 20.0,
        "publication_year": 2000.0 + float(index),
        "graph_shrunk_diffusion": scale / 200.0,
        "graph_reliability": 0.8,
        "graph_structural_share": 0.4,
        "graph_opportunity_share": 0.2,
        "graph_perturbation_potential": scale / 300.0,
        "graph_prediction_uncertainty": 0.2,
    }


def test_policy_evaluation_rejects_split_overlap(tmp_path: Path) -> None:
    rows = []
    for action_index, action in enumerate(ACTIONS):
        for index in range(15):
            rows.append(
                {
                    "paper_id": f"paper-{action_index}-{index}",
                    "context_id": f"context-{action_index}-{index}",
                    "experiment_split": "development",
                    "logged_action": action,
                    "outcome": 0.0,
                    "wrong_correction": False,
                    "unsupported_claim": False,
                    "realized_cost": 1.0,
                    "propensity": 1 / 6,
                    "matched_budget": 20,
                    "policy_fold_id": index % 3,
                    "feature": 0.0,
                }
            )
    development = pd.DataFrame(rows)
    holdout = development.groupby("logged_action").head(10).copy()
    holdout["experiment_split"] = "confirmatory_holdout"
    holdout["policy_fold_id"] = "holdout"
    development_path = tmp_path / "development.parquet"
    holdout_path = tmp_path / "holdout.parquet"
    development.to_parquet(development_path, index=False)
    holdout.to_parquet(holdout_path, index=False)

    try:
        run_policy_evaluation(
            development_path,
            holdout_path,
            tmp_path / "policy",
            feature_columns=["feature"],
        )
    except ValueError as exc:
        assert "paper_id overlap" in str(exc)
    else:
        raise AssertionError("split overlap did not fail closed")
