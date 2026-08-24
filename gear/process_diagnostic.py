"""Transparent process-level sufficiency status; no untrained probability."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .contracts import EvidenceReadiness, PaperIR, StrictModel
from .review_contracts import ProcessFeatures, ReviewStateV3


class ProcessDiagnostic(StrictModel):
    contract: Literal["aspr_process_diagnostic_v1"] = "aspr_process_diagnostic_v1"
    paper_id: str
    status: Literal["sufficient", "limited", "unavailable"]
    reasons: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    advisories: list[str] = Field(default_factory=list)
    features: ProcessFeatures


def diagnose_process(
    state: ReviewStateV3,
    paper_ir: PaperIR | None = None,
) -> ProcessDiagnostic:
    features = state.process_features
    reasons: list[str] = []
    advisories: list[str] = []
    if not features.agent_review_available:
        reasons.append("agent_reviewer_unavailable")
    if not features.graph_score_available:
        reasons.append("graph_unavailable")
    if not features.semantic_verifier_passed:
        reasons.append("semantic_verification_unavailable_or_failed")
    failure_reason_map = {
        "agent_reviewer_unavailable": "agent_reviewer_unavailable",
        "retrieval_unavailable": "retrieval_unavailable",
        "qwen_required_unavailable": "qwen_required_unavailable",
        "evidence_integrity_failed": "evidence_integrity_failed",
    }
    reasons.extend(
        failure_reason_map[failure.reason]
        for failure in state.failures
        if failure.reason in failure_reason_map
    )
    if paper_ir is not None:
        quality = paper_ir.quality_report
        if not quality.semantic_extraction_ready:
            advisories.append("semantic_extraction_degraded")
        if quality.document_only_ratio > 0.80:
            advisories.append("document_only_sections")
        if quality.evidence_readiness == EvidenceReadiness.UNAVAILABLE:
            reasons.append("paper_evidence_unavailable")
        elif quality.evidence_readiness == EvidenceReadiness.LIMITED:
            reasons.extend(
                reason
                for reason in quality.blocking_reasons
                if reason != "semantic_extraction_degraded"
            )
        advisories.extend(quality.advisories)
        if (
            paper_ir.method_result_ledger.main_results
            and not paper_ir.method_result_ledger.figures_tables
        ):
            advisories.append("result_spans_have_no_table_figure_anchor")
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
    if not features.agent_review_available or (
        paper_ir is not None
        and paper_ir.quality_report.evidence_readiness == EvidenceReadiness.UNAVAILABLE
    ):
        status = "unavailable"
    elif reasons:
        status = "limited"
    else:
        status = "sufficient"
    return ProcessDiagnostic(
        paper_id=state.paper_id,
        status=status,
        reasons=list(dict.fromkeys(reasons)),
        blocking_reasons=list(dict.fromkeys(reasons)),
        advisories=list(dict.fromkeys(advisories)),
        features=features,
    )


__all__ = ["ProcessDiagnostic", "diagnose_process"]
