"""Structured GEAR reviewer executed through isolated Codex CLI sessions."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, List, Mapping, Optional

from pydantic import ValidationError

from .config import GearConfig, load_config
from .model_client import (
    JsonModelClient,
    ModelClientUnavailableError,
    build_json_model_client,
)
from .contracts import PaperIR
from .review_contracts import (
    ContextClaim,
    ContextSpan,
    CriticRunMetadata,
    CriticSource,
    GraphReviewContext,
    NoveltyAssessment,
    NoveltyJudgment,
    ReviewContextPack,
    ReviewSummary,
    StructuredReview,
)

CODEX_REVIEW_PROMPT = """You are the paper-internal GEAR draft reviewer.
Return one JSON object satisfying StructuredReview. Produce exactly five logical
parts: summary, novelty, strengths, weaknesses, and questions. Do not output a
rating, decision, recommendation, accept/reject language, reviewer identity, or
source quote identity. Every scientific point must be atomic and cite supplied
P:S-* paper evidence keys. Major points require evidence. The safe graph context
may change inspection priority, retrieval budget, and wording confidence only; it
cannot establish novelty, quality, causality, or acceptance and it is never an
evidence key. Mark novelty/prior-art points external_verification_required=true.
The novelty judgment must be positive for supporting-only points, negative for
limiting-only points, mixed for both, and not_discussed for neither.
"""

FORBIDDEN_DECISION_TEXT = re.compile(
    r"\b(?:accept(?:ed|ance)?|reject(?:ed|ion)?|major revision|minor revision|"
    r"recommendation|rating|score\s*[=:])\b",
    re.I,
)


def build_review_context_pack(
    paper_ir: PaperIR,
    graph: GraphReviewContext,
    *,
    max_spans: int = 60,
) -> ReviewContextPack:
    """Select stable claim/method/result spans without raw review material."""
    span_map = paper_ir.span_map()
    selected: List[str] = []
    for claim in paper_ir.claims:
        if claim.span_id not in selected:
            selected.append(claim.span_id)
    ledger = paper_ir.method_result_ledger
    for field_name in (
        "research_question",
        "dataset_sample",
        "design_comparator",
        "model_algorithm",
        "baselines_metrics_statistics",
        "ablation_robustness",
        "main_results",
        "stated_limitations",
    ):
        for span_id in getattr(ledger, field_name):
            if span_id in span_map and span_id not in selected:
                selected.append(span_id)
    if not selected:
        selected.extend(span.span_id for span in paper_ir.spans[:max_spans])
    selected = selected[:max_spans]
    return ReviewContextPack(
        paper_id=paper_ir.paper_id,
        paper_sha256=paper_ir.paper_sha256,
        claims=[
            ContextClaim(
                claim_id=claim.claim_id,
                claim_type=claim.claim_type.value,
                evidence_key=f"P:{claim.span_id}",
                text=claim.text,
            )
            for claim in paper_ir.claims
        ],
        spans=[
            ContextSpan(
                evidence_key=f"P:{span_map[span_id].span_id}",
                span_id=span_map[span_id].span_id,
                page=span_map[span_id].page,
                section_path=list(span_map[span_id].section_path),
                text=span_map[span_id].text,
                text_sha256=span_map[span_id].text_sha256,
            )
            for span_id in selected
        ],
        graph=graph,
    )


def limited_review(paper_ir: PaperIR) -> StructuredReview:
    """Return an explicit, evidence-bound empty review when Codex is unavailable."""
    if not paper_ir.spans:
        raise ValueError("cannot build even a limited review without a paper span")
    first_span = paper_ir.spans[0]
    return StructuredReview(
        paper_id=paper_ir.paper_id,
        summary=ReviewSummary(
            text=(
                "The manuscript was compiled, but the isolated Codex review session "
                "was unavailable; no independent structured assessment was generated."
            ),
            evidence_keys=[f"P:{first_span.span_id}"],
        ),
        novelty=NoveltyAssessment(
            judgment=NoveltyJudgment.NOT_DISCUSSED,
            supporting_points=[],
            limiting_points=[],
        ),
        strengths=[],
        weaknesses=[],
        questions=[],
    )


class CodexCliCritic:
    """Generate one complete StructuredReview in an isolated Codex CLI session."""

    def __init__(
        self,
        config: Optional[GearConfig] = None,
        *,
        generator: Optional[Callable[[str, str], Mapping[str, Any]]] = None,
        client: Optional[JsonModelClient] = None,
    ) -> None:
        self.config = config or load_config()
        self.client = client or build_json_model_client(self.config)
        self.generator = generator
        self.last_failures: List[str] = []
        self.model_name = self.config.codex_cli.model

    @property
    def metadata(self) -> CriticRunMetadata:
        return CriticRunMetadata(
            critic_source=(
                CriticSource.UNAVAILABLE
                if self.last_failures
                else (
                    CriticSource.CODEX_CLI
                    if self.config.model_backend == "codex_cli"
                    else CriticSource.OPENAI_COMPATIBLE_API
                )
            ),
            model_id=self.model_name,
        )

    def review(
        self,
        paper_ir: PaperIR,
        graph_context: GraphReviewContext,
    ) -> StructuredReview:
        self.last_failures = []
        if not paper_ir.spans:
            raise ValueError("paper_ir_has_no_reviewable_spans")
        pack = build_review_context_pack(paper_ir, graph_context)
        user = json.dumps(
            {
                "context": pack.model_dump(mode="json"),
                "output_schema": StructuredReview.model_json_schema(),
            },
            ensure_ascii=False,
        )
        try:
            payload = self._generate(CODEX_REVIEW_PROMPT, user)
            raw_review = payload.get("review", payload)
            review = StructuredReview.model_validate(raw_review)
            self._validate_review(review, paper_ir)
            return review
        except (ModelClientUnavailableError, ValidationError, ValueError, TypeError) as exc:
            self.last_failures.append(f"codex_cli_unavailable_or_invalid:{exc}")
            return limited_review(paper_ir)

    def _generate(self, system: str, user: str) -> Mapping[str, Any]:
        if self.generator is not None:
            return self.generator(system, user)
        return self.client.generate_json(
            system=system,
            user=user,
            response_schema=StructuredReview.model_json_schema(),
        )

    @staticmethod
    def _validate_review(review: StructuredReview, paper_ir: PaperIR) -> None:
        if review.paper_id != paper_ir.paper_id:
            raise ValueError("Codex review paper_id mismatch")
        valid_keys = {f"P:{span.span_id}" for span in paper_ir.spans}
        evidence_lists = [
            review.summary.evidence_keys,
            *(point.evidence_keys for point in review.all_points()),
        ]
        unknown = sorted(
            key for keys in evidence_lists for key in keys if key not in valid_keys
        )
        if unknown:
            raise ValueError(f"Codex review contains unknown evidence keys: {unknown}")
        texts = [review.summary.text, *(point.text for point in review.all_points())]
        if any(FORBIDDEN_DECISION_TEXT.search(text) for text in texts):
            raise ValueError("Codex review contains decision/recommendation language")
        novelty_points = [
            *review.novelty.supporting_points,
            *review.novelty.limiting_points,
        ]
        if any(not point.external_verification_required for point in novelty_points):
            raise ValueError("novelty points must request external verification")


def stable_point_id(section: str, text: str) -> str:
    identity = f"{section}|{text.strip()}"
    return "RP-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:18]


__all__ = [
    "CODEX_REVIEW_PROMPT",
    "CodexCliCritic",
    "FORBIDDEN_DECISION_TEXT",
    "build_review_context_pack",
    "limited_review",
    "stable_point_id",
]
