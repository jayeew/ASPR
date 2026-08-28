from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from pytest import MonkeyPatch, approx, raises

from experiments.gear.evaluation import build_gate1_mechanism_dataset as module
from experiments.gear.evaluation.build_gate2_integration_frame import (
    build_integration_frame,
)
from gear.claim_attribution import FEATURE_SCHEMA_VERSION, T0_FEATURE_NAMES


def test_gate1_dataset_uses_cross_fitted_claim_attribution(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    labels = []
    oof = []
    scores = []
    perturbation = []
    for paper_index in range(12):
        paper_id = f"W{paper_index}"
        for claim_index in range(2):
            labels.append(
                {
                    "paper_id": paper_id,
                    "claim_id": f"C{claim_index}",
                    "claim_centrality": 1.0 - claim_index * 0.2,
                    "claim_type": "method_claim",
                    "pathway_hypothesis": "local_method_adoption",
                    "claim_t0_schema_version": FEATURE_SCHEMA_VERSION,
                    "manuscript_validity": 1.0,
                    "evidence_coverage": 0.5,
                    "antecedent_risk": 0.0,
                    "residual_novelty": 1.0,
                    "mechanism_validity": 0.7,
                    "attribution_weight": 0.5,
                    "future_adoption": float(claim_index == paper_index % 2),
                    "context_observation_status": "resolved",
                }
            )
        oof.append(
            {
                "paper_id": paper_id,
                "outer_fold_id": paper_index % 3,
                "domain12": f"domain-{paper_index % 2}",
                "publication_year": 2000 + paper_index,
                "future_uptake": 1.0,
                "realized_diffusion_target": paper_index / 12,
                "expected_diffusion_score": paper_index / 12,
            }
        )
        scores.append(
            {
                "paper_id": paper_id,
                "prospective_5y_diffusion_percentile": paper_index * 8.0,
            }
        )
        perturbation.append(
            {
                "paper_id": paper_id,
                "perturbation_target_fold": paper_index / 12,
                "perturbation_head_p": paper_index / 12,
                "shuffled_perturbation_head_p": 1.0 - paper_index / 12,
            }
        )
    paths = {
        "labels": tmp_path / "labels.parquet",
        "oof": tmp_path / "oof.parquet",
        "score": tmp_path / "score.parquet",
        "perturbation": tmp_path / "perturbation.parquet",
    }
    pd.DataFrame(labels).to_parquet(paths["labels"], index=False)
    pd.DataFrame(oof).to_parquet(paths["oof"], index=False)
    pd.DataFrame(scores).to_parquet(paths["score"], index=False)
    pd.DataFrame(perturbation).to_parquet(paths["perturbation"], index=False)
    monkeypatch.setattr(module, "OOF_PATH", paths["oof"])
    monkeypatch.setattr(module, "SCORE_PATH", paths["score"])
    anatomy = pd.DataFrame(
        {
            "paper_id": [f"W{index}" for index in range(12)],
            "anatomy_limited": False,
            "uptake_contribution__substantive_innovation": [0.6] * 12,
            "conditional_contribution__substantive_innovation": [0.1] * 12,
            "uptake_contribution__t0_potential": [0.1] * 12,
            "conditional_contribution__t0_potential": [0.1] * 12,
            "uptake_contribution__opportunity": [0.05] * 12,
            "conditional_contribution__opportunity": [0.05] * 12,
            "uptake_contribution__context": [0.05] * 12,
            "conditional_contribution__context": [0.05] * 12,
        }
    )
    monkeypatch.setattr(
        module,
        "load_forecast_analog_index",
        lambda _: SimpleNamespace(table=lambda: anatomy),
    )
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps(
            {
                "selection_uses_future_outcomes": False,
                "cases": [
                    {
                        "paper_id": f"W{index}",
                        "integration_split": (
                            "temporal_holdout" if index >= 9 else "development"
                        ),
                    }
                    for index in range(12)
                ],
            }
        ),
        encoding="utf-8",
    )

    report = module.build_gate1_dataset(
        paths["labels"],
        paths["perturbation"],
        tmp_path / "out",
        split_manifest_path=split_path,
    )

    assert report["papers"] == 12
    assert report["future_features_used_for_training"] is False
    assert (
        report["future_labels_role"]
        == "development_training_target_and_evaluation_target"
    )
    assert report["claim_attribution"]["holdout_labels_used_for_training"] is False
    output = pd.read_parquet(tmp_path / "out" / "gate1_mechanism_dataset.parquet")
    assert output.groupby("paper_id")["attribution_weight"].sum().round(10).eq(1).all()
    assert set(T0_FEATURE_NAMES).issubset(output.columns)
    assert (
        output["structural_score_at_one"] >= output["structural_score_at_zero"]
    ).all()


def test_gate1_dataset_fails_closed_without_formal_claim_type() -> None:
    frame = pd.DataFrame(
        {
            "claim_centrality": [1.0],
            "pathway_hypothesis": ["local_method_adoption"],
            "claim_t0_schema_version": [FEATURE_SCHEMA_VERSION],
            **{f"anatomy_role__{role}": [0.25] for role in module.FORECAST_ROLES},
        }
    )
    with raises(ValueError, match="columns missing"):
        module._materialize_exact_t0_features(frame)


def test_gate2_integration_is_one_row_per_paper(tmp_path: Path) -> None:
    claims = pd.DataFrame(
        {
            "paper_id": ["W1", "W1", "W2"],
            "domain12": ["physics", "physics", "chemistry"],
            "publication_year": [2020, 2020, 2019],
            "outer_fold_id": [2, 2, 1],
            "structural_innovation_score": [0.2, 0.3, 0.1],
            "shuffled_structural_score": [0.1, 0.2, 0.1],
            "structural_score_at_zero": [0.01, 0.02, 0.01],
            "future_structural_outcome": [0.7, 0.7, 0.2],
        }
    )
    path = tmp_path / "gate1.parquet"
    claims.to_parquet(path, index=False)

    report = build_integration_frame(path, tmp_path / "gate2")

    assert report["papers"] == 2
    output = pd.read_parquet(tmp_path / "gate2" / "gate2_integration_frame.parquet")
    assert output["paper_id"].is_unique
    assert output.set_index("paper_id").at["W1", "joint_structural_score"] == approx(
        0.44
    )
