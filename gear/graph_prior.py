"""Lazy, text-free adapter around the frozen Fig.1-Fig.3 calibration service."""

from __future__ import annotations

from datetime import date
from typing import Any, Callable, Optional

from .calibration import CalibrationService
from .config import GearConfig, load_config
from .contracts import CalibrationMode, PaperIR
from .graph_prior_contracts import (
    FeatureSetAudit,
    GraphPriorAudit,
    GraphPriorProvenance,
    GraphPriorResult,
)


class GraphPriorService:
    """Execute the required Graph branch without exposing its feature payload."""

    def __init__(
        self,
        config: Optional[GearConfig] = None,
        *,
        calibration_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.config = config or load_config()
        self._calibration_factory = calibration_factory
        self._calibration: Optional[Any] = None
        self.last_failure: Optional[str] = None
        self.last_audit: Optional[GraphPriorAudit] = None
        self.last_packet: Optional[Any] = None

    def _service(self) -> Any:
        if self._calibration is None:
            self._calibration = (
                self._calibration_factory()
                if self._calibration_factory is not None
                else CalibrationService(self.config)
            )
        return self._calibration

    def score(self, paper_ir: PaperIR, cutoff_date: date) -> GraphPriorResult:
        self.last_failure = None
        self.last_packet = None
        try:
            packet = self._service().build_packet(paper_ir, cutoff_date=cutoff_date)
        except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as exc:
            self.last_failure = f"{type(exc).__name__}:{exc}"
            self.last_audit = GraphPriorAudit(paper_id=paper_ir.paper_id)
            return self._unavailable(paper_ir.paper_id, self.last_failure)
        self.last_packet = packet
        result = graph_prior_from_calibration(packet)
        measurement = packet.measurement
        feature_values = {
            **measurement.substantive_innovation,
            **measurement.t0_potential,
            **measurement.opportunity,
            **measurement.context_control,
        }
        self.last_audit = GraphPriorAudit(
            paper_id=paper_ir.paper_id,
            feature_values=feature_values,
            p_uptake=packet.forecast.p_uptake,
            conditional_diffusion=packet.forecast.conditional_diffusion,
            feature_sets=[
                FeatureSetAudit(
                    feature_set="fulltext_16",
                    expected_dimension=16,
                    observed_dimension=len(feature_values),
                    coverage=packet.reliability.feature_coverage,
                    model_id=result.model_id,
                    score_0_100=result.score_0_100,
                    quality_flags=(
                        []
                        if len(feature_values) == 16
                        else ["materialized_feature_count_not_16"]
                    ),
                )
            ],
        )
        return result

    @staticmethod
    def _unavailable(paper_id: str, reason: str) -> GraphPriorResult:
        return GraphPriorResult(
            paper_id=paper_id,
            status="unavailable",
            quality_flags=[reason],
        )


def graph_prior_from_calibration(packet: Any) -> GraphPriorResult:
    mode = packet.reliability.mode
    score = packet.forecast.aspr_score_0_100
    available = (
        mode
        in {
            CalibrationMode.EXACT_LOOKUP,
            CalibrationMode.ELIGIBLE_INFERENCE,
        }
        and score is not None
    )
    status = mode.value if available else "unavailable"
    provenance = packet.provenance
    release_id = provenance.calibration_release_id
    model_id = (
        f"{release_id}:d5:fulltext_16:hgb" if release_id else "d5:fulltext_16:hgb"
    )
    return GraphPriorResult(
        paper_id=packet.paper_id,
        status=status,
        score_0_100=score if available else None,
        model_id=model_id if available else None,
        feature_coverage=packet.reliability.feature_coverage,
        drift_flags=list(packet.reliability.drift_flags),
        quality_flags=list(packet.reliability.quality_flags),
        provenance=GraphPriorProvenance(
            calibration_release_id=release_id,
            model_id=model_id if available else None,
            model_sha256=provenance.model_sha256,
            score_table_sha256=provenance.score_table_sha256,
            feature_matrix_sha256=provenance.feature_matrix_sha256,
            evidence_policy=provenance.evidence_policy,
        ),
    )


__all__ = ["GraphPriorService", "graph_prior_from_calibration"]
