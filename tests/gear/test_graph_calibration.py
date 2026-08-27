from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from gear.graph_calibration import (
    ROLE_FEATURES,
    ForecastAnalogIndex,
    calibration_tensions,
    compute_anatomy,
)
from gear.graph_prior_contracts import ForecastAnatomy
from gear.review_contracts import PointSeverity, ReviewAspect, ReviewPoint


class _IdentityCalibrator:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return values


class _RawModel:
    def predict_raw(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = frame.to_numpy(dtype=float)
        return values[:, :8].sum(axis=1), values[:, 8:].sum(axis=1)


def _model() -> dict[str, object]:
    return {
        "model": _RawModel(),
        "uptake_calibrator": _IdentityCalibrator(),
        "conditional_calibrator": _IdentityCalibrator(),
    }


def _anatomy() -> ForecastAnatomy:
    roles = list(ROLE_FEATURES)
    return ForecastAnatomy(
        paper_id="target",
        target_field="biology",
        uptake_percentile=80.0,
        conditional_diffusion_percentile=80.0,
        expected_diffusion_percentile=80.0,
        uptake_role_contributions=dict(zip(roles, [1.0, 2.0, 3.0, 4.0])),
        conditional_role_contributions=dict(zip(roles, [5.0, 6.0, 7.0, 8.0])),
        role_coverage=dict.fromkeys(roles, 1.0),
        baseline_id="field:biology",
        feature_input_sha256="sha256:target",
        anatomy_release_id="release",
    )


def test_group_shapley_is_efficient_and_uses_no_outcomes() -> None:
    names = [name for group in ROLE_FEATURES.values() for name in group]
    target = pd.DataFrame([{"paper_id": "target", **dict.fromkeys(names, 1.0)}])
    baseline = pd.DataFrame([{name: 0.0 for name in names}])
    anatomy = compute_anatomy(
        _model(),
        target,
        baseline,
        feature_names=names,
        uptake_reference=np.array([0.0, 100.0]),
        conditional_reference=np.array([0.0, 100.0]),
        expected_reference=np.array([0.0, 10_000.0]),
        baseline_ids=["field:test"],
        release_id="release",
    ).iloc[0]
    assert sum(
        anatomy[f"uptake_contribution__{role}"] for role in ROLE_FEATURES
    ) == pytest.approx(8.0)
    assert sum(
        anatomy[f"conditional_contribution__{role}"] for role in ROLE_FEATURES
    ) == pytest.approx(8.0)
    assert not any(
        "outcome" in column or "citation" in column for column in anatomy.index
    )


def test_analog_index_is_cutoff_safe_and_shuffled_changes_entry(tmp_path) -> None:
    target = _anatomy()
    roles = list(ROLE_FEATURES)
    columns = [
        *(f"uptake_contribution__{role}" for role in roles),
        *(f"conditional_contribution__{role}" for role in roles),
    ]
    vector = np.arange(1.0, 9.0)
    rows = []
    for index in range(8):
        row = dict(zip(columns, np.roll(vector, index)))
        row.update(
            paper_id=f"W{index}",
            title="Target mechanism evidence",
            title_sha256=f"sha256:{index}",
            publication_year=2019,
            field="biology",
        )
        rows.append(row)
    rows.append(
        {
            **dict(zip(columns, vector)),
            "paper_id": "W-post",
            "title": "Target mechanism evidence",
            "title_sha256": "sha256:post",
            "publication_year": 2021,
            "field": "chemistry",
        }
    )
    index_path = tmp_path / "index.parquet"
    pd.DataFrame(rows).to_parquet(index_path)
    manifest = {
        "source_snapshot_id": "frozen",
        "source_snapshot_sha256": "sha256:frozen",
    }
    analogs = ForecastAnalogIndex(index_path, manifest)
    correct = analogs.select(
        target,
        claim_id="claim-a",
        terms=["target", "mechanism"],
        cutoff_date=date(2021, 1, 1),
        target_field="biology",
    )
    shuffled = analogs.select(
        target,
        claim_id="claim-a",
        terms=["target", "mechanism"],
        cutoff_date=date(2021, 1, 1),
        target_field="biology",
        shuffled=True,
    )
    assert correct and shuffled
    assert correct[0].work_id != shuffled[0].work_id
    assert all(seed.publication_year < 2021 for seed in correct)
    assert all(seed.text_version == "frozen_metadata_title_v1" for seed in correct)


def test_tension_is_process_only_and_graph_card_cannot_be_review_evidence() -> None:
    tensions = calibration_tensions(_anatomy())
    assert {item.kind for item in tensions} == {"opportunity_dominant"}
    with pytest.raises(ValueError, match="review evidence keys"):
        ReviewPoint(
            point_id="point",
            aspect=ReviewAspect.NOVELTY_PRIOR_ART,
            text="Needs evidence.",
            severity=PointSeverity.MAJOR,
            evidence_keys=["G:INFLUENCE"],
        )
