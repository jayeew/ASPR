from __future__ import annotations

import sys

import numpy as np
import pytest

from gear.calibration import FEATURE_ROLES, CalibrationService
from gear.contracts import CalibrationMode
from gear.nature_multihorizon.t0_runtime_v3 import (
    ContextSnapshot,
    MaterializationReplayReport,
    TargetT0Record,
)


def test_exact_fig3_lookup_replays_components():
    legacy_module = sys.modules.get("aspr")
    packet = CalibrationService().exact_lookup("W104864265")
    assert sys.modules.get("aspr") is legacy_module
    assert packet.reliability.mode == CalibrationMode.EXACT_LOOKUP
    assert packet.forecast.raw_expected_diffusion == pytest.approx(
        packet.forecast.p_uptake * packet.forecast.conditional_diffusion,
        abs=1e-14,
    )
    assert 0.0 <= packet.forecast.aspr_score_0_100 <= 100.0
    assert "EF0197" not in FEATURE_ROLES["substantive_innovation"]
    assert packet.provenance.deprecated_fig4_to_fig10_used is False
    assert (
        packet.provenance.calibration_release_id
        == "pgc-v3-d5-fulltext16-80f673c0-93e2e0dd"
    )


def test_all_411490_official_rows_replay_exactly():
    report = CalibrationService().validate_official_replay(batch_size=75_000)
    assert report["row_count"] == 411_490
    assert report["maximum_absolute_raw_error"] <= 1e-10
    assert report["maximum_product_identity_error"] <= np.finfo(float).eps
    assert report["passed"] is True


def test_new_unpublished_paper_can_use_promoted_runtime_inference(paper_ir):
    paper = paper_ir.model_copy(
        update={
            "metadata": paper_ir.metadata.model_copy(
                update={"openalex_id": None, "doi": None}
            )
        }
    )
    packet = CalibrationService(
        doi_resolver=lambda _: None,
        runtime_target_builder=lambda _paper, _cutoff: TargetT0Record(
            paper_id="new-manuscript",
            publication_year=2026,
            title="A new evidence controller",
            author_count=2,
            metadata_observed=False,
        ),
    ).build_packet(paper)
    assert packet.reliability.mode == CalibrationMode.ELIGIBLE_INFERENCE
    assert packet.forecast.aspr_score_0_100 is not None
    assert "historical_context_lag" in packet.reliability.drift_flags


def test_deprecated_asset_path_fails_closed(gear_config):
    with pytest.raises(ValueError, match="deprecated evidence path"):
        gear_config.validate_asset_path(
            gear_config.resolve_path("outputs/fig10/old/result.json")
        )
    with pytest.raises(ValueError, match="deprecated evidence path"):
        gear_config.validate_asset_path(
            gear_config.resolve_path("outputs/fig4/result.json")
        )


def test_online_inference_rejects_a_report_that_differs_from_frozen_gate():
    tampered = MaterializationReplayReport(
        row_count=411_490,
        missingness_identical=True,
        categorical_identical=True,
        numeric_within_tolerance=True,
        numeric_max_absolute_error=0.0,
        raw_prediction_within_tolerance=True,
        raw_prediction_max_absolute_error=1e-12,
    )
    with pytest.raises(ValueError, match="has not passed frozen replay"):
        CalibrationService().eligible_inference(
            TargetT0Record(
                paper_id="W-fabricated",
                publication_year=2026,
                title="Fabricated gate",
            ),
            ContextSnapshot(source_max_year=2025),
            replay_report=tampered,
        )


def test_persisted_runtime_context_is_hash_pinned_and_promoted():
    service = CalibrationService()
    context, report = service.load_approved_runtime_context()
    assert context.source_max_year == 2022
    assert report.row_count == 411_490
    assert report.numeric_max_absolute_error == 0.0
    assert report.raw_prediction_max_absolute_error == 0.0
    assert report.eligible_inference is True
