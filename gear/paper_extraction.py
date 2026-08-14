"""Hybrid semantic extraction layered over deterministic PaperIR spans."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

from pydantic import ValidationError

from .contracts import ClaimStrength, PaperClaim, PaperIR
from .paper_compiler import build_claim_ledger
from .review_contracts import PaperSpecificRubric


class HybridPaperExtractor:
    """Accept semantic objects only when they resolve to immutable PaperIR spans."""

    def __init__(
        self,
        *,
        generator: Optional[Callable[[Dict[str, Any]], Mapping[str, Any]]] = None,
    ) -> None:
        self.generator = generator
        self.last_failures: List[str] = []

    def enrich(self, paper_ir: PaperIR) -> PaperIR:
        self.last_failures = []
        if self.generator is None:
            return self._degraded_fallback(paper_ir, "semantic_extractor_unavailable")
        payload = {
            "paper_id": paper_ir.paper_id,
            "spans": [
                {
                    "span_id": span.span_id,
                    "section_path": span.section_path,
                    "text": span.text,
                }
                for span in paper_ir.spans
            ],
            "output": {
                "claims": [
                    {
                        "claim_id": "",
                        "claim_type": "novelty_claim",
                        "span_id": "S-*",
                        "text": "exact or entailed atomic claim",
                        "strength": "moderate",
                        "dependency_span_ids": [],
                        "required_evidence": [],
                    }
                ]
            },
        }
        try:
            raw = self.generator(payload)
            claims = [PaperClaim.model_validate(item) for item in raw.get("claims", [])]
            self._validate_claim_spans(claims, paper_ir)
        except (ValidationError, ValueError, TypeError, KeyError) as exc:
            return self._degraded_fallback(
                paper_ir, f"semantic_extraction_failed:{type(exc).__name__}"
            )
        if not claims:
            return self._degraded_fallback(paper_ir, "semantic_extraction_empty")
        return paper_ir.model_copy(
            update={
                "claims": claims,
                "claim_ledger": build_claim_ledger(
                    claims, paper_ir.method_result_ledger, paper_ir.references
                ),
            }
        )

    @staticmethod
    def _validate_claim_spans(claims: List[PaperClaim], paper_ir: PaperIR) -> None:
        span_map = paper_ir.span_map()
        for claim in claims:
            span = span_map.get(claim.span_id)
            if span is None:
                raise ValueError(
                    f"semantic claim references unknown span: {claim.span_id}"
                )
            if not claim.text.strip() or claim.text.strip() not in span.text:
                raise ValueError(
                    f"semantic claim text does not resolve inside span: {claim.claim_id}"
                )
            if any(span_id not in span_map for span_id in claim.dependency_span_ids):
                raise ValueError(
                    f"semantic claim dependency references unknown span: {claim.claim_id}"
                )

    def _degraded_fallback(self, paper_ir: PaperIR, reason: str) -> PaperIR:
        self.last_failures.append(reason)
        flags = list(
            dict.fromkeys(
                [*paper_ir.quality_flags, f"semantic_extraction_degraded:{reason}"]
            )
        )
        return paper_ir.model_copy(update={"quality_flags": flags})


class PaperRubricBuilder:
    """Build a graph- and human-review-blind rubric from observable paper structure."""

    def build(self, paper_ir: PaperIR) -> PaperSpecificRubric:
        ledger = paper_ir.method_result_ledger
        novelty_checks = [
            "nearest prior work is identified for each strong novelty claim",
            "contribution differences are explicit and evidence-bounded",
        ]
        if any(claim.strength == ClaimStrength.STRONG for claim in paper_ir.claims):
            novelty_checks.append(
                "strong novelty wording receives counterfactual search"
            )
        methodology = ["method assumptions and design are supported by paper spans"]
        if ledger.dataset_sample:
            methodology.append("dataset or sample construction is adequately specified")
        if ledger.design_comparator:
            methodology.append("comparators and controls are appropriate")
        experiments = ["reported results support the stated conclusions"]
        if ledger.baselines_metrics_statistics:
            experiments.append("baselines, metrics, and statistics are adequate")
        if ledger.ablation_robustness:
            experiments.append(
                "ablation and robustness evidence is interpreted correctly"
            )
        reproducibility = ["major critique cites stable manuscript spans"]
        if ledger.figures_tables:
            reproducibility.append("numeric claims cite result or table spans")
        return PaperSpecificRubric(
            paper_id=paper_ir.paper_id,
            novelty_checks=novelty_checks,
            methodology_checks=methodology,
            experiment_checks=experiments,
            reproducibility_checks=reproducibility,
        )


__all__ = ["HybridPaperExtractor", "PaperRubricBuilder"]
