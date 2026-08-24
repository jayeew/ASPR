"""Deterministic, evidence-addressable grounding cards for the review workflow."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from .contracts import ClaimType, PaperIR
from .review_contracts import ReviewModel
from .trace import EvidenceStore, sha256_value


class ContributionCard(ReviewModel):
    contribution_id: str
    claim_id: str
    claim: str
    novelty_type: Literal[
        "conceptual", "method", "application", "empirical", "resource"
    ]
    essential_facets: list[str] = Field(default_factory=list)
    paper_evidence_keys: list[str] = Field(min_length=1)


class AuditFinding(ReviewModel):
    finding_id: str
    stage: Literal["method", "result_logic", "coverage"]
    contribution_id: str | None = None
    observation: str
    why_it_matters: str
    suggested_action: str
    confidence: float = Field(ge=0.0, le=1.0)
    paper_evidence_keys: list[str] = Field(min_length=1)


class NoveltyDelta(ReviewModel):
    contribution_id: str
    status: Literal["supported", "incremental", "not_supported", "uncertain"]
    shared_base: list[str] = Field(default_factory=list)
    residual_delta: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    paper_evidence_keys: list[str] = Field(min_length=1)


class GroundingReport(ReviewModel):
    contract: Literal["aspr_grounding_report_v1"] = "aspr_grounding_report_v1"
    paper_id: str
    contributions: list[ContributionCard] = Field(default_factory=list)
    findings: list[AuditFinding] = Field(default_factory=list)
    novelty_deltas: list[NoveltyDelta] = Field(default_factory=list)
    stage_hashes: dict[str, str] = Field(default_factory=dict)
    limited_reasons: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class GroundingWorkflow:
    """Build safe P0/P1 cards without model-memory or graph semantics.

    The cards deliberately remain separate from final review points until the
    Aggregator and Verifier consume them.  This makes a failed new stage visible
    in artifacts without silently changing the reviewer output.
    """

    max_contributions: int = 8

    def run(self, paper_ir: PaperIR, store: EvidenceStore) -> GroundingReport:
        contributions = self._contributions(paper_ir)
        findings = self._findings(paper_ir, contributions)
        deltas = self._novelty_deltas(paper_ir, contributions)
        report = GroundingReport(
            paper_id=paper_ir.paper_id,
            contributions=contributions,
            findings=findings,
            novelty_deltas=deltas,
            limited_reasons=self._limited_reasons(paper_ir),
        )
        hashes = {
            "contribution_decomposer": sha256_value(contributions),
            "internal_auditors": sha256_value(findings),
            "novelty_grounder": sha256_value(deltas),
            "evidence_aggregator": sha256_value(report),
        }
        report = report.model_copy(update={"stage_hashes": hashes})
        store.add_evidence("GW:REPORT", "grounding_report", report)
        for card in contributions:
            store.add_evidence(
                f"GW:C:{card.contribution_id}", "contribution_card", card
            )
        for finding in findings:
            store.add_evidence(f"GW:F:{finding.finding_id}", "audit_finding", finding)
        for delta in deltas:
            store.add_evidence(f"GW:N:{delta.contribution_id}", "novelty_delta", delta)
        return report

    def _contributions(self, paper_ir: PaperIR) -> list[ContributionCard]:
        priority = {
            ClaimType.NOVELTY: 0,
            ClaimType.METHOD: 1,
            ClaimType.RESULT: 2,
            ClaimType.CAUSAL: 3,
            ClaimType.SCOPE: 4,
            ClaimType.SIGNIFICANCE: 5,
        }
        cards: list[ContributionCard] = []
        for claim in sorted(
            paper_ir.claims, key=lambda item: priority[item.claim_type]
        ):
            if len(cards) >= self.max_contributions:
                break
            novelty_type = {
                ClaimType.NOVELTY: "conceptual",
                ClaimType.METHOD: "method",
                ClaimType.RESULT: "empirical",
                ClaimType.CAUSAL: "empirical",
                ClaimType.SCOPE: "application",
                ClaimType.SIGNIFICANCE: "application",
            }[claim.claim_type]
            identity = f"{paper_ir.paper_id}|{claim.claim_id}"
            cards.append(
                ContributionCard(
                    contribution_id="CON-"
                    + hashlib.sha256(identity.encode()).hexdigest()[:16],
                    claim_id=claim.claim_id,
                    claim=claim.text,
                    novelty_type=novelty_type,
                    essential_facets=_facets(claim.text),
                    paper_evidence_keys=[f"P:{claim.span_id}"],
                )
            )
        return cards

    @staticmethod
    def _findings(
        paper_ir: PaperIR, contributions: list[ContributionCard]
    ) -> list[AuditFinding]:
        findings: list[AuditFinding] = []
        result_keys = [
            f"P:{span_id}" for span_id in paper_ir.claim_ledger.result_span_ids
        ]
        method_keys = [
            f"P:{span_id}" for span_id in paper_ir.claim_ledger.method_span_ids
        ]
        if (
            result_keys
            and not paper_ir.method_result_ledger.baselines_metrics_statistics
        ):
            findings.append(
                _finding(
                    "result_logic",
                    None,
                    "The result ledger contains outcome claims but no anchored baseline, metric, or statistical span.",
                    "Without a linked comparison or uncertainty anchor, result interpretation is limited.",
                    "Anchor the main result to its comparator, metric, and statistical evidence.",
                    result_keys[0],
                )
            )
        if method_keys and not paper_ir.method_result_ledger.design_comparator:
            findings.append(
                _finding(
                    "method",
                    None,
                    "The method ledger contains implementation evidence but no explicit design or comparator anchor.",
                    "The applicable scope of the method cannot be audited from implementation prose alone.",
                    "State the comparison design, controls, and applicability boundary explicitly.",
                    method_keys[0],
                )
            )
        if contributions and not paper_ir.method_result_ledger.stated_limitations:
            findings.append(
                _finding(
                    "coverage",
                    contributions[0].contribution_id,
                    "No limitations span was detected for the principal contribution.",
                    "A reviewer cannot distinguish acknowledged scope from an unexamined limitation.",
                    "Add a bounded limitations statement for the principal contribution.",
                    contributions[0].paper_evidence_keys[0],
                )
            )
        return findings

    @staticmethod
    def _novelty_deltas(
        paper_ir: PaperIR, contributions: list[ContributionCard]
    ) -> list[NoveltyDelta]:
        novelty_ids = set(paper_ir.claim_ledger.novelty_claim_ids)
        return [
            NoveltyDelta(
                contribution_id=card.contribution_id,
                status="uncertain" if card.claim_id in novelty_ids else "supported",
                residual_delta=[card.claim] if card.claim_id in novelty_ids else [],
                confidence=0.35 if card.claim_id in novelty_ids else 0.60,
                paper_evidence_keys=card.paper_evidence_keys,
            )
            for card in contributions
        ]

    @staticmethod
    def _limited_reasons(paper_ir: PaperIR) -> list[str]:
        return list(
            dict.fromkeys(
                [
                    *paper_ir.quality_report.blocking_reasons,
                    *(
                        ["semantic_extraction_degraded"]
                        if not paper_ir.quality_report.semantic_extraction_ready
                        else []
                    ),
                ]
            )
        )


def _facets(text: str) -> list[str]:
    words = [word for word in text.replace("-", " ").split() if len(word) > 3]
    return words[:6]


def _finding(
    stage: Literal["method", "result_logic", "coverage"],
    contribution_id: str | None,
    observation: str,
    why_it_matters: str,
    suggested_action: str,
    key: str,
) -> AuditFinding:
    identity = f"{stage}|{contribution_id}|{observation}|{key}"
    return AuditFinding(
        finding_id="FND-" + hashlib.sha256(identity.encode()).hexdigest()[:16],
        stage=stage,
        contribution_id=contribution_id,
        observation=observation,
        why_it_matters=why_it_matters,
        suggested_action=suggested_action,
        confidence=0.70,
        paper_evidence_keys=[key],
    )


__all__ = [
    "AuditFinding",
    "ContributionCard",
    "GroundingReport",
    "GroundingWorkflow",
    "NoveltyDelta",
]
