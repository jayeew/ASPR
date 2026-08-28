from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest import MonkeyPatch

from experiments.gear.evaluation import run_stage_a_validation as validation_module
from experiments.gear.evaluation.run_stage_a_validation import _attach_gear_evidence
from experiments.gear.evaluation.stage_a_dataset import (
    build_score_stratified_cohort,
    cohort_quality,
)
from experiments.gear.evaluation.stage_a_gate0 import evaluate_gate0
from experiments.gear.evaluation.stage_a_three_arm import run_three_arm_experiment


def _population(rows_per_decile: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for decile in range(10):
        for offset in range(rows_per_decile):
            rows.append(
                {
                    "paper_id": f"paper-{decile}-{offset}",
                    "domain12": f"domain-{offset % 3}",
                    "publication_year": 2000 + offset,
                    "prospective_5y_diffusion_percentile": decile * 10 + offset,
                    "expected_diffusion_score": (decile * 10 + offset) / 100,
                    "realized_diffusion_target": (decile + offset / 10) / 10,
                    "score_decile": decile,
                    "stable_key": f"{offset:02d}-{decile:02d}",
                }
            )
    return pd.DataFrame(rows)


def test_cohort_builder_covers_deciles_without_evidence() -> None:
    cohort = build_score_stratified_cohort(_population(), per_decile=2)
    quality = cohort_quality(cohort)

    assert quality["rows"] == 20
    assert quality["score_deciles_covered"] == 10
    assert quality["minimum_per_decile"] == 2
    assert quality["integration_eligible"] == 0


def test_three_arm_runner_fails_closed_without_matched_evidence() -> None:
    _, result = run_three_arm_experiment(pd.DataFrame({"paper_id": ["paper-1"]}))

    assert result["status"] == "not_identifiable"
    assert result["claim_allowed"] is False


def test_three_arm_runner_estimates_complete_wide_range_cohort() -> None:
    generator = np.random.default_rng(20260827)
    graph = np.tile(np.linspace(0.01, 0.99, 10), 12)
    evidence = generator.uniform(0.2, 1.0, len(graph))
    frame = pd.DataFrame(
        {
            "paper_id": [f"paper-{index}" for index in range(len(graph))],
            "domain12": [f"domain-{index % 4}" for index in range(len(graph))],
            "publication_year": [2000 + index % 20 for index in range(len(graph))],
            "gear_evidence_score": evidence,
            "mechanism_validity": np.ones(len(graph)),
            "antecedent_risk": np.zeros(len(graph)),
            "score_decile": np.tile(np.arange(10), 12),
            "graph_expected_diffusion": graph,
            "future_structural_outcome": evidence * (0.1 + 0.9 * graph),
        }
    )

    arms, result = run_three_arm_experiment(frame)

    assert len(arms) == 120
    assert result["status"] == "estimated"
    assert result["score_deciles"] == 10
    assert result["real_hgb"]["integration_value"] > 0.0


def test_three_arm_range_uses_frozen_graph_percentile_deciles() -> None:
    generator = np.random.default_rng(20260828)
    percentile_deciles = np.tile(np.arange(10), 12)
    graph = generator.uniform(0.25, 0.75, len(percentile_deciles))
    evidence = generator.uniform(0.2, 1.0, len(percentile_deciles))
    frame = pd.DataFrame(
        {
            "paper_id": [f"paper-{index}" for index in range(len(graph))],
            "domain12": [f"domain-{index % 4}" for index in range(len(graph))],
            "publication_year": [2000 + index % 20 for index in range(len(graph))],
            "gear_evidence_score": evidence,
            "mechanism_validity": np.ones(len(graph)),
            "antecedent_risk": np.zeros(len(graph)),
            "score_decile": percentile_deciles,
            "graph_expected_diffusion": graph,
            "future_structural_outcome": evidence * (0.1 + 0.9 * graph),
        }
    )

    _, result = run_three_arm_experiment(frame)

    assert result["status"] == "estimated"
    assert result["score_deciles"] == 10


def test_gate0_passes_for_full_realistic_score_range() -> None:
    result = evaluate_gate0(_population())

    assert result["status"] == "passed"
    assert all(result["checks"].values())


def test_real_gear_evidence_marks_only_blinded_rows_eligible() -> None:
    cohort = build_score_stratified_cohort(_population(), per_decile=2)
    evidence = pd.DataFrame(
        {
            "paper_id": ["paper-0-0", "paper-1-0"],
            "gear_evidence_score": [0.4, 0.5],
            "mechanism_validity": [0.7, 0.7],
            "antecedent_risk": [0.0, 0.0],
            "evidence_coverage": [0.8, 0.8],
            "gear_run_path": ["run-a", "run-b"],
            "blinded_to_future_outcome": [True, False],
        }
    )

    attached = _attach_gear_evidence(cohort, evidence)

    assert attached["gear_evidence_available"].sum() == 2
    assert attached["integration_eligible"].sum() == 1


def test_explicit_missing_gear_evidence_does_not_fall_back(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(validation_module, "load_stage_a_population", _population)

    with pytest.raises(FileNotFoundError):
        validation_module.run_validation(
            tmp_path / "out",
            gear_evidence_path=tmp_path / "missing.parquet",
        )
