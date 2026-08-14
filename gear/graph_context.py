"""Build the safe graph projection consumed by the GEAR reviewer."""

from __future__ import annotations

from .contracts import (
    CalibrationMode,
    CalibrationPacketV3,
    SubmissionCalibrationPacketV1,
)
from .review_contracts import GraphReviewContext


def build_graph_review_context(
    calibration: CalibrationPacketV3 | SubmissionCalibrationPacketV1,
) -> GraphReviewContext:
    """Project Fig.1-Fig.3 fields without opportunity or context controls."""
    reliability = calibration.reliability
    return GraphReviewContext(
        paper_id=calibration.paper_id,
        substantive_innovation=dict(calibration.measurement.substantive_innovation),
        t0_potential=dict(calibration.measurement.t0_potential),
        p_uptake=calibration.forecast.p_uptake,
        conditional_diffusion=calibration.forecast.conditional_diffusion,
        d5_percentile=calibration.forecast.aspr_score_0_100,
        applicability_mode=reliability.mode.value,
        feature_coverage=reliability.feature_coverage,
        overall_oof_spearman=reliability.overall_oof_spearman,
        fold_oof_spearman=reliability.fold_oof_spearman,
        domain_oof_spearman=reliability.domain_oof_spearman,
        drift_flags=list(reliability.drift_flags),
        limited=(
            reliability.mode
            in {CalibrationMode.PROFILE_ONLY, CalibrationMode.UNAVAILABLE}
            or bool(reliability.quality_flags)
        ),
    )


__all__ = ["build_graph_review_context"]
