"""Hybrid semantic extraction layered over deterministic PaperIR spans."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import ValidationError

from .config import GearConfig
from .contracts import (
    ClaimStrength,
    PaperClaim,
    PaperIR,
    StrictModel,
)
from .model_client import ModelClientUnavailableError, build_json_model_client
from .paper_compiler import build_claim_ledger
from .review_contracts import PaperSpecificRubric


class ClaimExtractionResponse(StrictModel):
    """Schema-constrained response returned by the semantic claim extractor."""

    claims: list[PaperClaim]


class HybridPaperExtractor:
    """Accept semantic objects only when they resolve to immutable PaperIR spans."""

    def __init__(
        self,
        *,
        generator: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self.generator = generator
        self.last_failures: list[str] = []

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
                        "text": "atomic claim grounded in the referenced span",
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
        except (
            ModelClientUnavailableError,
            ValidationError,
            ValueError,
            TypeError,
            KeyError,
        ) as exc:
            return self._degraded_fallback(paper_ir, _extraction_failure_reason(exc))
        if not claims:
            return self._degraded_fallback(paper_ir, "semantic_extraction_empty")
        report = paper_ir.quality_report.model_copy(
            update={"semantic_extraction_ready": True}
        )
        return paper_ir.model_copy(
            update={
                "claims": claims,
                "claim_ledger": build_claim_ledger(
                    claims, paper_ir.method_result_ledger, paper_ir.references
                ),
                "quality_report": report,
            }
        )

    @staticmethod
    def _validate_claim_spans(claims: list[PaperClaim], paper_ir: PaperIR) -> None:
        """Validate references, not wording identity.

        Semantic extraction is allowed to condense or paraphrase a supplied span.
        The immutable span ID remains the evidence anchor; requiring the generated
        claim text to be a byte-for-byte substring rejected otherwise valid atomic
        claims and added no grounding guarantee beyond that anchor.
        """
        span_map = paper_ir.span_map()
        for claim in claims:
            if claim.span_id not in span_map:
                raise ValueError(
                    f"semantic claim references unknown span: {claim.span_id}"
                )
            if not claim.text.strip():
                raise ValueError(f"semantic claim text is empty: {claim.claim_id}")
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
        report = paper_ir.quality_report.model_copy(
            update={
                "semantic_extraction_ready": False,
                "advisories": list(
                    dict.fromkeys(
                        [
                            *paper_ir.quality_report.advisories,
                            "semantic_extraction_degraded",
                        ]
                    )
                ),
            }
        )
        return paper_ir.model_copy(
            update={"quality_flags": flags, "quality_report": report}
        )


def configured_paper_extractor(config: GearConfig) -> HybridPaperExtractor:
    """Build a model-backed extractor without creating a client at import time."""
    client: Any = None

    def generate(payload: dict[str, Any]) -> Mapping[str, Any]:
        nonlocal client
        if client is None:
            client = build_json_model_client(config)
        result = client.generate_json(
            system=(
                "Extract only atomic manuscript claims grounded in supplied spans. "
                "Concise paraphrases are allowed. Preserve the supporting span_id; "
                "do not infer missing facts or use outside knowledge."
            ),
            user=json.dumps(payload, ensure_ascii=False),
            response_schema=ClaimExtractionResponse.model_json_schema(),
        )
        claims = result.get("claims")
        if isinstance(claims, list):
            return {**result, "claims": claims[: config.max_claims]}
        return result

    return HybridPaperExtractor(generator=generate)


def _extraction_failure_reason(exc: Exception) -> str:
    detail = " ".join(str(exc).split())[:240]
    suffix = f":{detail}" if detail else ""
    return f"semantic_extraction_failed:{type(exc).__name__}{suffix}"


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


__all__ = [
    "ClaimExtractionResponse",
    "HybridPaperExtractor",
    "PaperRubricBuilder",
    "configured_paper_extractor",
]
