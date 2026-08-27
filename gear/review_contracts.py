"""Strict ASPR-GEAR contracts shared by labels, critics, and runtime."""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from .contracts import (
    FailureRecord,
    PaperIR,
    ReviewStatus,
    StrictModel,
)
from .graph_prior_contracts import (
    GraphRuntimePacket,
    ResourceLedger,
    RetrievalGuidancePlan,
    RetrievalRoutingPlan,
)

SCHEMA_VERSION: Literal["aspr_gear"] = "aspr_gear"
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[\u3400-\u9fff]")


def word_count(text: str) -> int:
    """Count Latin tokens and CJK characters conservatively for output limits."""
    return len(_WORD_RE.findall(str(text or "")))


class ReviewModel(StrictModel):
    """Base model for the only supported review payloads."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    schema_version: Literal["aspr_gear"] = SCHEMA_VERSION
    schema_revision: Literal["evidence_state_delta_v2"] = "evidence_state_delta_v2"


class ReviewAspect(str, Enum):
    CONTRIBUTION = "contribution"
    NOVELTY_PRIOR_ART = "novelty_prior_art"
    METHOD = "method"
    EXPERIMENT_EVIDENCE = "experiment_evidence"
    RESULTS_CONCLUSION = "results_conclusion"
    PRESENTATION_REPRODUCIBILITY = "presentation_reproducibility"
    OTHER = "other"


class PointSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class NoveltyJudgment(str, Enum):
    POSITIVE = "positive"
    MIXED = "mixed"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"
    NOT_DISCUSSED = "not_discussed"


class NoveltyVerificationStatus(str, Enum):
    """Evidence status for a novelty direction, kept separate from direction."""

    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    NOT_ASSESSED = "not_assessed"


class NoveltyEvidenceStatus(str, Enum):
    NOT_ASSESSED = "not_assessed"
    MANUSCRIPT_SUPPORTED = "manuscript_supported"
    EVIDENCE_QUALIFIED = "evidence_qualified"
    EVIDENCE_CHALLENGED = "evidence_challenged"
    INCONCLUSIVE = "inconclusive"


class CriticSource(str, Enum):
    CODEX_CLI = "codex_cli"
    OPENAI_COMPATIBLE_API = "openai_compatible_api"
    UNAVAILABLE = "unavailable"


class PointValidationStatus(str, Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    EXTERNALLY_VALIDATED = "externally_validated"
    UNRESOLVED = "unresolved"
    REJECTED = "rejected"


class ReviewSource(str, Enum):
    AGENT = "agent_reviewer"
    ASPR_QWEN = "aspr_qwen"


class ReviewPhase(str, Enum):
    INITIALIZED = "initialized"
    SOURCES_READY = "sources_ready"
    FUSED = "fused"
    EVIDENCE_GATHERING = "evidence_gathering"
    EVIDENCE_FINALIZED = "evidence_finalized"
    VERIFIED = "verified"
    COMPILED = "compiled"


class EvidenceAction(str, Enum):
    SEARCH_PRIOR_ART = "search_prior_art"
    COUNTERFACTUAL_SEARCH = "counterfactual_search"
    CITATION_EXPAND = "citation_expand"
    VERIFY_POINT = "verify_point"
    STABILITY_TEST = "stability_test"
    FINALIZE = "finalize"


class ReviewSummary(ReviewModel):
    text: str = Field(min_length=1)
    evidence_keys: list[str] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def summary_limit(cls, value: str) -> str:
        if word_count(value) > 150:
            raise ValueError("summary must not exceed 150 words")
        return value.strip()

    @field_validator("evidence_keys")
    @classmethod
    def summary_evidence(cls, value: list[str]) -> list[str]:
        return _validate_evidence_keys(value)


class ReviewPoint(ReviewModel):
    point_id: str = Field(min_length=1)
    aspect: ReviewAspect
    text: str = Field(min_length=1)
    severity: PointSeverity = PointSeverity.NONE
    suggested_action: str = ""
    why_it_matters: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_keys: list[str] = Field(default_factory=list)
    external_verification_required: bool = False

    @field_validator("text")
    @classmethod
    def point_limit(cls, value: str) -> str:
        if word_count(value) > 120:
            raise ValueError("review point must not exceed 120 words")
        return value.strip()

    @field_validator("evidence_keys")
    @classmethod
    def point_evidence(cls, value: list[str]) -> list[str]:
        return _validate_evidence_keys(value)

    @model_validator(mode="after")
    def major_requires_evidence(self) -> ReviewPoint:
        if (
            self.severity in {PointSeverity.MAJOR, PointSeverity.CRITICAL}
            and not self.evidence_keys
        ):
            raise ValueError("major and critical review points require evidence_keys")
        return self


class NoveltyAssessment(ReviewModel):
    judgment: NoveltyJudgment
    verification_status: NoveltyVerificationStatus = (
        NoveltyVerificationStatus.NOT_ASSESSED
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    supporting_points: list[ReviewPoint] = Field(default_factory=list, max_length=3)
    limiting_points: list[ReviewPoint] = Field(default_factory=list, max_length=3)
    uncertain_points: list[ReviewPoint] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def validate_novelty_points(self) -> NoveltyAssessment:
        # Direction is an assessment, while the point lists are the evidence that
        # survived verification.  They deliberately need not imply the same enum:
        # incomplete retrieval may soften or remove a point without erasing the
        # graph-blind reviewer's original direction.
        for point in [
            *self.supporting_points,
            *self.limiting_points,
            *self.uncertain_points,
        ]:
            if point.aspect not in {
                ReviewAspect.CONTRIBUTION,
                ReviewAspect.NOVELTY_PRIOR_ART,
            }:
                raise ValueError("novelty points require a novelty/contribution aspect")
        return self


def infer_novelty_judgment(
    supporting_points: list[ReviewPoint],
    limiting_points: list[ReviewPoint],
    uncertain_points: list[ReviewPoint] | None = None,
) -> NoveltyJudgment:
    uncertain_points = uncertain_points or []
    if supporting_points and limiting_points:
        return NoveltyJudgment.MIXED
    if supporting_points and uncertain_points:
        return NoveltyJudgment.MIXED
    if limiting_points and uncertain_points:
        return NoveltyJudgment.MIXED
    if supporting_points:
        return NoveltyJudgment.POSITIVE
    if limiting_points:
        return NoveltyJudgment.NEGATIVE
    if uncertain_points:
        return NoveltyJudgment.UNCERTAIN
    return NoveltyJudgment.NOT_DISCUSSED


class StructuredReview(ReviewModel):
    """The single label/draft/final-review contract for GEAR."""

    paper_id: str = Field(min_length=1)
    summary: ReviewSummary
    novelty: NoveltyAssessment
    strengths: list[ReviewPoint] = Field(default_factory=list)
    weaknesses: list[ReviewPoint] = Field(default_factory=list)
    questions: list[ReviewPoint] = Field(default_factory=list)

    def all_points(self) -> list[ReviewPoint]:
        return [
            *self.novelty.supporting_points,
            *self.novelty.limiting_points,
            *self.novelty.uncertain_points,
            *self.strengths,
            *self.weaknesses,
            *self.questions,
        ]

    @model_validator(mode="after")
    def review_limits(self) -> StructuredReview:
        points = self.all_points()
        if len(points) > 24:
            raise ValueError("StructuredReview permits at most 24 atomic points")
        point_ids = [point.point_id for point in points]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("review point IDs must be unique within a paper")
        return self


class PaperSpecificRubric(ReviewModel):
    """Deterministic, generation-time rubric built only from PaperIR."""

    paper_id: str
    paper_type: str = "scientific_manuscript"
    novelty_checks: list[str] = Field(default_factory=list)
    methodology_checks: list[str] = Field(default_factory=list)
    experiment_checks: list[str] = Field(default_factory=list)
    reproducibility_checks: list[str] = Field(default_factory=list)


class BranchReview(ReviewModel):
    """Common graph-blind output contract for Agent and ASPR-Qwen."""

    contract: Literal["aspr_branch_review_v2"] = "aspr_branch_review_v2"
    paper_id: str
    source: ReviewSource
    model_id: str
    prompt_sha256: str
    input_sha256: str
    graph_blind: Literal[True] = True
    summary: ReviewSummary
    novelty: NoveltyAssessment
    strengths: list[ReviewPoint] = Field(default_factory=list)
    weaknesses: list[ReviewPoint] = Field(default_factory=list)
    questions: list[ReviewPoint] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)

    def all_points(self) -> list[ReviewPoint]:
        return [
            *self.novelty.supporting_points,
            *self.novelty.limiting_points,
            *self.novelty.uncertain_points,
            *self.strengths,
            *self.weaknesses,
            *self.questions,
        ]

    @classmethod
    def from_structured(
        cls,
        review: StructuredReview,
        *,
        source: ReviewSource,
        model_id: str,
        prompt_sha256: str,
        input_sha256: str,
        failures: list[str] | None = None,
    ) -> BranchReview:
        return cls(
            paper_id=review.paper_id,
            source=source,
            model_id=model_id,
            prompt_sha256=prompt_sha256,
            input_sha256=input_sha256,
            summary=review.summary,
            novelty=review.novelty,
            strengths=review.strengths,
            weaknesses=review.weaknesses,
            questions=review.questions,
            failures=list(failures or []),
        )


class CanonicalReviewPoint(ReviewModel):
    point_id: str
    section: Literal[
        "novelty_support", "novelty_limit", "strengths", "weaknesses", "questions"
    ]
    initial_section: (
        Literal[
            "novelty_support", "novelty_limit", "strengths", "weaknesses", "questions"
        ]
        | None
    ) = None
    aspect: ReviewAspect
    severity: PointSeverity
    proposition: str
    resolved_proposition: str | None = None
    suggested_action: str | None = None
    source_point_ids: dict[ReviewSource, list[str]] = Field(default_factory=dict)
    paper_evidence_keys: list[str] = Field(default_factory=list)
    relation_evidence_keys: list[str] = Field(default_factory=list)
    coverage_evidence_keys: list[str] = Field(default_factory=list)
    novelty_resolution: Literal[
        "not_applicable",
        "antecedent_found",
        "incremental_or_parallel",
        "bounded_no_antecedent",
        "inconclusive",
        "search_failed",
    ] = "not_applicable"
    contribution_id: str | None = None
    residual_delta: list[str] = Field(default_factory=list)
    shared_base: list[str] = Field(default_factory=list)
    novelty_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    agent_support: bool
    qwen_support: bool | None = None
    qwen_conflict: bool = False
    graph_tension: bool = False
    graph_tension_score: float = Field(default=0.0, ge=0.0, le=1.0)
    graph_focus_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    graph_extra_counterfactual_actions: int = Field(default=0, ge=0, le=2)
    novelty_claim_id: str | None = None
    requires_external_evidence: bool = False
    validation_status: PointValidationStatus = PointValidationStatus.PENDING
    stability_status: Literal["not_required", "pending", "stable", "unstable"] = (
        "not_required"
    )
    validation_notes: list[str] = Field(default_factory=list)
    normal_search_done: bool = False
    counterfactual_search_done: bool = False
    counterfactual_search_count: int = Field(default=0, ge=0)
    citation_expanded: bool = False
    semantic_verified: bool = False
    retained: bool = True


class ReviewCorrectionEventV1(ReviewModel):
    contract: Literal["aspr_review_correction_event_v1"] = (
        "aspr_review_correction_event_v1"
    )
    point_id: str
    before_text: str
    after_text: str
    before_section: str
    after_section: str
    before_direction: NoveltyJudgment | None = None
    after_direction: NoveltyJudgment | None = None
    trigger_relation_ids: list[str]
    trigger_mission_ids: list[str] = Field(default_factory=list)
    correction_type: Literal[
        "direct_antecedent_challenge",
        "partial_antecedent_refinement",
        "residual_novelty_refinement",
        "attribution_scope_refinement",
        "confidence_downgrade",
        "confidence_upgrade",
        "prior_work_added_only",
    ]
    confidence_change: float = Field(ge=-1.0, le=1.0)


class EvidenceBudget(ReviewModel):
    normal_per_claim_max: int = Field(default=4, ge=0)
    counterfactual_per_claim_max: int = Field(default=1, ge=0)
    citation_per_claim_max: int = Field(default=1, ge=0)
    relation_cards_max: int = Field(default=24, ge=1)
    total_actions_max: int = Field(default=48, ge=1)
    actions_used: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def bounded_actions(self) -> EvidenceBudget:
        if self.actions_used > self.total_actions_max:
            raise ValueError("evidence action budget exceeded")
        return self


class ProcessFeatures(ReviewModel):
    agent_review_available: bool = False
    qwen_review_available: bool = False
    graph_score_available: bool = False
    retrieval_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    independent_prior_count: int = Field(default=0, ge=0)
    relation_conflict: bool = False
    counterfactual_completed: bool = False
    counterfactual_changed_judgment: bool = False
    stability_passed: bool = False
    graph_text_tension: bool = False
    semantic_verifier_passed: bool = False
    failure_count: int = Field(default=0, ge=0)


class FusionMatch(ReviewModel):
    agent_point_id: str | None = None
    qwen_point_id: str | None = None
    relation: Literal["SAME_POINT", "PARTIAL", "CONTRADICTORY", "NO_MATCH"]


class FusionReport(ReviewModel):
    contract: Literal["aspr_fusion_report_v3"] = "aspr_fusion_report_v3"
    paper_id: str
    matches: list[FusionMatch] = Field(default_factory=list)
    canonical_point_ids: list[str] = Field(default_factory=list)
    graph_tension_scores: dict[str, float] = Field(default_factory=dict)
    graph_focus_weights: dict[str, float] = Field(default_factory=dict)
    graph_triggered_actions: dict[str, list[str]] = Field(default_factory=dict)
    graph_guided_point_ids: list[str] = Field(default_factory=list)
    graph_query_replacements: dict[str, str] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)


class ReviewState(ReviewModel):
    contract: Literal["gear_review_state"] = "gear_review_state"
    state_id: str
    phase: ReviewPhase
    paper_id: str
    paper_sha256: str
    cutoff_date: date
    rubric: PaperSpecificRubric
    branch_reviews: dict[ReviewSource, BranchReview] = Field(default_factory=dict)
    graph_result_evidence_key: str | None = None
    graph_result: GraphRuntimePacket | None = None
    graph_guidance_plan: RetrievalGuidancePlan | None = None
    retrieval_routing_plans: dict[str, RetrievalRoutingPlan] = Field(
        default_factory=dict
    )
    resource_ledger: ResourceLedger | None = None
    novelty_direction: NoveltyJudgment | None = None
    novelty_verification_status: NoveltyVerificationStatus = (
        NoveltyVerificationStatus.NOT_ASSESSED
    )
    novelty_direction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    novelty_evidence_status: NoveltyEvidenceStatus = NoveltyEvidenceStatus.NOT_ASSESSED
    aspr_evidence_assessments: dict[
        str,
        Literal[
            "NOT_CHALLENGED",
            "CHALLENGED",
            "REFINED",
            "INCONCLUSIVE",
            "NOT_APPLICABLE",
        ],
    ] = Field(default_factory=dict)
    canonical_points: dict[str, CanonicalReviewPoint] = Field(default_factory=dict)
    retrieved_work_evidence_keys: list[str] = Field(default_factory=list)
    relation_evidence_keys: list[str] = Field(default_factory=list)
    correction_event_evidence_keys: list[str] = Field(default_factory=list)
    unresolved_target_ids: list[str] = Field(default_factory=list)
    action_budget: EvidenceBudget = Field(default_factory=EvidenceBudget)
    process_features: ProcessFeatures = Field(default_factory=ProcessFeatures)
    failures: list[FailureRecord] = Field(default_factory=list)
    finalized: bool = False

    @field_validator("graph_result", mode="before")
    @classmethod
    def migrate_graph_result(cls, value: Any) -> Any:
        return None if value is None else GraphRuntimePacket.model_validate(value)

    @model_validator(mode="after")
    def state_invariants(self) -> ReviewState:
        agent = self.branch_reviews.get(ReviewSource.AGENT)
        if self.phase != ReviewPhase.INITIALIZED and agent is None:
            raise ValueError("Agent Reviewer branch is required")
        if (
            self.graph_result is not None
            and self.graph_result.paper_id != self.paper_id
        ):
            raise ValueError("Graph result paper_id mismatch")
        if any(key != point.point_id for key, point in self.canonical_points.items()):
            raise ValueError("canonical point identity mismatch")
        if self.finalized and self.phase not in {
            ReviewPhase.EVIDENCE_FINALIZED,
            ReviewPhase.VERIFIED,
            ReviewPhase.COMPILED,
        }:
            raise ValueError("finalized state has an invalid phase")
        return self


class GraphReviewContext(ReviewModel):
    """Safe Fig.1-Fig.3 projection; opportunity/control never enter this model."""

    contract: Literal["graph_review_context"] = "graph_review_context"
    paper_id: str
    substantive_innovation: dict[str, float | int | str | bool | None] = Field(
        default_factory=dict
    )
    t0_potential: dict[str, float | int | str | bool | None] = Field(
        default_factory=dict
    )
    p_uptake: float | None = Field(default=None, ge=0.0, le=1.0)
    conditional_diffusion: float | None = Field(default=None, ge=0.0, le=1.0)
    d5_percentile: float | None = Field(default=None, ge=0.0, le=100.0)
    applicability_mode: str
    feature_coverage: float = Field(ge=0.0, le=1.0)
    overall_oof_spearman: float | None = None
    fold_oof_spearman: float | None = None
    domain_oof_spearman: float | None = None
    drift_flags: list[str] = Field(default_factory=list)
    limited: bool = False


class ContextSpan(ReviewModel):
    evidence_key: str
    span_id: str
    page: int = Field(ge=1)
    section_path: list[str] = Field(default_factory=list)
    text: str
    text_sha256: str


class ContextClaim(ReviewModel):
    claim_id: str
    claim_type: str
    evidence_key: str
    text: str


class ReviewContextPack(ReviewModel):
    contract: Literal["review_context_pack"] = "review_context_pack"
    paper_id: str
    paper_sha256: str
    claims: list[ContextClaim]
    spans: list[ContextSpan]
    graph: GraphReviewContext


class CriticRunMetadata(ReviewModel):
    critic_source: CriticSource
    model_id: str


class VerificationIssue(ReviewModel):
    issue_id: str
    code: str
    message: str
    point_id: str | None = None
    repairable: bool = False


class VerificationReport(ReviewModel):
    passed: bool
    limited: bool = False
    issues: list[VerificationIssue] = Field(default_factory=list)
    semantic_verification_available: bool = False
    graph_semantic_violation_count: int = Field(default=0, ge=0)
    unsupported_major_count: int = Field(default=0, ge=0)


class ReviewBundle(ReviewModel):
    contract: Literal["gear_review_bundle"] = "gear_review_bundle"
    status: ReviewStatus
    paper_ir: PaperIR
    critic: CriticRunMetadata
    structured_review: StructuredReview
    review_markdown: str
    verification: VerificationReport
    output_files: dict[str, str] = Field(default_factory=dict)
    agent_review: BranchReview | None = None
    qwen_review: BranchReview | None = None
    graph_result: GraphRuntimePacket | None = None
    fusion_report: FusionReport | None = None
    state: ReviewState | None = None
    process_diagnostic: dict[str, Any] | None = None
    grounding_report: dict[str, Any] | None = None

    @field_validator("graph_result", mode="before")
    @classmethod
    def migrate_graph_result(cls, value: Any) -> Any:
        return None if value is None else GraphRuntimePacket.model_validate(value)


def _validate_evidence_keys(value: list[str]) -> list[str]:
    unique_keys = list(dict.fromkeys(value))
    for key in unique_keys:
        if not re.fullmatch(
            r"(?:P:S-[A-Za-z0-9_-]+|R:[A-Za-z0-9:_-]+|COV:[A-Za-z0-9:_-]+)",
            key,
        ):
            raise ValueError(
                "review evidence keys must reference paper spans, relations, "
                "or search coverage"
            )
    return unique_keys


__all__ = [
    "SCHEMA_VERSION",
    "BranchReview",
    "CanonicalReviewPoint",
    "ContextClaim",
    "ContextSpan",
    "CriticRunMetadata",
    "CriticSource",
    "EvidenceAction",
    "EvidenceBudget",
    "FusionMatch",
    "FusionReport",
    "GraphReviewContext",
    "NoveltyAssessment",
    "NoveltyEvidenceStatus",
    "NoveltyJudgment",
    "NoveltyVerificationStatus",
    "PaperSpecificRubric",
    "PointSeverity",
    "PointValidationStatus",
    "ProcessFeatures",
    "ReviewAspect",
    "ReviewBundle",
    "ReviewContextPack",
    "ReviewPhase",
    "ReviewPoint",
    "ReviewSource",
    "ReviewState",
    "ReviewSummary",
    "StructuredReview",
    "VerificationIssue",
    "VerificationReport",
    "infer_novelty_judgment",
    "word_count",
]
