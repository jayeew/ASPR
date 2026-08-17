"""Transparent process-level sufficiency status; no untrained probability."""

from __future__ import annotations

from typing import List, Literal

from pydantic import Field

from .contracts import StrictModel
from .review_contracts import ProcessFeatures, ReviewStateV2


class ProcessDiagnostic(StrictModel):
    contract: Literal["aspr_process_diagnostic_v1"] = "aspr_process_diagnostic_v1"
    paper_id: str
    status: Literal["sufficient", "limited", "unavailable"]
    reasons: List[str] = Field(default_factory=list)
    blocking_reasons: List[str] = Field(default_factory=list)
    advisories: List[str] = Field(default_factory=list)
    features: ProcessFeatures


def diagnose_process(state: ReviewStateV2) -> ProcessDiagnostic:
    features = state.process_features
    reasons: List[str] = []
    advisories: List[str] = []
    if not features.agent_review_available:
        reasons.append("agent_reviewer_unavailable")
    if not features.graph_score_available:
        reasons.append("graph_prior_unavailable")
    if not features.semantic_verifier_passed:
        reasons.append("semantic_verification_unavailable_or_failed")
    if state.unresolved_target_ids:
        reasons.append("unresolved_review_targets")
    if not features.stability_passed and any(
        point.stability_status != "not_required"
        for point in state.canonical_points.values()
    ):
        reasons.append("high_risk_stability_not_passed")
    if state.failures:
        reasons.append("required_component_failure")
    advisory_codes = {
        "citation_expansion_has_no_seed",
        "distant_candidate_rejected_before_relation_store",
        "insufficient_coverage_downgraded_to_question",
        "single_antecedent_downgraded_to_question",
    }
    for point in state.canonical_points.values():
        advisories.extend(
            note
            for note in point.validation_notes
            if note in advisory_codes or note.startswith("local_ranker_degraded:")
        )
    status: Literal["sufficient", "limited", "unavailable"]
    if not features.agent_review_available:
        status = "unavailable"
    elif reasons:
        status = "limited"
    else:
        status = "sufficient"
    return ProcessDiagnostic(
        paper_id=state.paper_id,
        status=status,
        reasons=reasons,
        blocking_reasons=reasons,
        advisories=list(dict.fromkeys(advisories)),
        features=features,
    )


__all__ = ["ProcessDiagnostic", "diagnose_process"]
