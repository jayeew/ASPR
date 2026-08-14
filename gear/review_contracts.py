"""Strict ASPR-GEAR contracts shared by labels, critics, and runtime."""

from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from pydantic import ConfigDict, Field, field_validator, model_validator

from .contracts import (
    CalibrationPacketV3,
    FailureRecord,
    PaperIR,
    ReviewStatus,
    StrictModel,
    SubmissionCalibrationPacketV1,
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


class NoveltyJudgment(str, Enum):
    POSITIVE = "positive"
    MIXED = "mixed"
    NEGATIVE = "negative"
    NOT_DISCUSSED = "not_discussed"


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


class ReviewSummary(ReviewModel):
    text: str = Field(min_length=1)
    evidence_keys: List[str] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def summary_limit(cls, value: str) -> str:
        if word_count(value) > 150:
            raise ValueError("summary must not exceed 150 words")
        return value.strip()

    @field_validator("evidence_keys")
    @classmethod
    def summary_evidence(cls, value: List[str]) -> List[str]:
        return _validate_evidence_keys(value)


class ReviewPoint(ReviewModel):
    point_id: str = Field(min_length=1)
    aspect: ReviewAspect
    text: str = Field(min_length=1)
    severity: PointSeverity = PointSeverity.NONE
    suggested_action: str = ""
    evidence_keys: List[str] = Field(default_factory=list)
    external_verification_required: bool = False

    @field_validator("text")
    @classmethod
    def point_limit(cls, value: str) -> str:
        if word_count(value) > 120:
            raise ValueError("review point must not exceed 120 words")
        return value.strip()

    @field_validator("evidence_keys")
    @classmethod
    def point_evidence(cls, value: List[str]) -> List[str]:
        return _validate_evidence_keys(value)

    @model_validator(mode="after")
    def major_requires_evidence(self) -> "ReviewPoint":
        if self.severity == PointSeverity.MAJOR and not self.evidence_keys:
            raise ValueError("major review points require evidence_keys")
        return self


class NoveltyAssessment(ReviewModel):
    judgment: NoveltyJudgment
    supporting_points: List[ReviewPoint] = Field(default_factory=list, max_length=3)
    limiting_points: List[ReviewPoint] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def deterministic_judgment(self) -> "NoveltyAssessment":
        expected = infer_novelty_judgment(self.supporting_points, self.limiting_points)
        if self.judgment != expected:
            raise ValueError(
                f"novelty judgment must be {expected.value} for the supplied points"
            )
        for point in [*self.supporting_points, *self.limiting_points]:
            if point.aspect not in {
                ReviewAspect.CONTRIBUTION,
                ReviewAspect.NOVELTY_PRIOR_ART,
            }:
                raise ValueError("novelty points require a novelty/contribution aspect")
        return self


def infer_novelty_judgment(
    supporting_points: List[ReviewPoint],
    limiting_points: List[ReviewPoint],
) -> NoveltyJudgment:
    if supporting_points and limiting_points:
        return NoveltyJudgment.MIXED
    if supporting_points:
        return NoveltyJudgment.POSITIVE
    if limiting_points:
        return NoveltyJudgment.NEGATIVE
    return NoveltyJudgment.NOT_DISCUSSED


class StructuredReview(ReviewModel):
    """The single label/draft/final-review contract for GEAR."""

    paper_id: str = Field(min_length=1)
    summary: ReviewSummary
    novelty: NoveltyAssessment
    strengths: List[ReviewPoint] = Field(default_factory=list)
    weaknesses: List[ReviewPoint] = Field(default_factory=list)
    questions: List[ReviewPoint] = Field(default_factory=list)

    def all_points(self) -> List[ReviewPoint]:
        return [
            *self.novelty.supporting_points,
            *self.novelty.limiting_points,
            *self.strengths,
            *self.weaknesses,
            *self.questions,
        ]

    @model_validator(mode="after")
    def review_limits(self) -> "StructuredReview":
        points = self.all_points()
        if len(points) > 24:
            raise ValueError("StructuredReview permits at most 24 atomic points")
        point_ids = [point.point_id for point in points]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("review point IDs must be unique within a paper")
        return self


class GraphReviewContext(ReviewModel):
    """Safe Fig.1-Fig.3 projection; opportunity/control never enter this model."""

    contract: Literal["graph_review_context"] = "graph_review_context"
    paper_id: str
    substantive_innovation: Dict[str, Union[float, int, str, bool, None]] = Field(
        default_factory=dict
    )
    t0_potential: Dict[str, Union[float, int, str, bool, None]] = Field(
        default_factory=dict
    )
    p_uptake: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    conditional_diffusion: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    d5_percentile: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    applicability_mode: str
    feature_coverage: float = Field(ge=0.0, le=1.0)
    overall_oof_spearman: Optional[float] = None
    fold_oof_spearman: Optional[float] = None
    domain_oof_spearman: Optional[float] = None
    drift_flags: List[str] = Field(default_factory=list)
    limited: bool = False


class ContextSpan(ReviewModel):
    evidence_key: str
    span_id: str
    page: int = Field(ge=1)
    section_path: List[str] = Field(default_factory=list)
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
    claims: List[ContextClaim]
    spans: List[ContextSpan]
    graph: GraphReviewContext


class ReviewPointState(ReviewModel):
    point_id: str
    status: PointValidationStatus = PointValidationStatus.PENDING
    retained: bool = True
    evidence_keys: List[str] = Field(default_factory=list)
    relation_evidence_keys: List[str] = Field(default_factory=list)
    validation_notes: List[str] = Field(default_factory=list)


class ReviewState(ReviewModel):
    contract: Literal["review_review_state"] = "review_review_state"
    state_id: str
    paper_id: str
    paper_sha256: str
    cutoff_date: date
    draft_review: StructuredReview
    graph_context: GraphReviewContext
    point_states: Dict[str, ReviewPointState]
    retrieved_work_evidence_keys: List[str] = Field(default_factory=list)
    relation_evidence_keys: List[str] = Field(default_factory=list)
    graph_text_tension_point_ids: List[str] = Field(default_factory=list)
    failure_ledger: List[FailureRecord] = Field(default_factory=list)
    finalized: bool = False

    @model_validator(mode="after")
    def validate_point_links(self) -> "ReviewState":
        point_ids = {point.point_id for point in self.draft_review.all_points()}
        if set(self.point_states) != point_ids:
            raise ValueError("ReviewState point keys differ from draft review")
        if any(key != value.point_id for key, value in self.point_states.items()):
            raise ValueError("ReviewState point-state identity mismatch")
        return self


class CriticRunMetadata(ReviewModel):
    critic_source: CriticSource
    model_id: str


class VerificationIssue(ReviewModel):
    issue_id: str
    code: str
    message: str
    point_id: Optional[str] = None
    repairable: bool = False


class VerificationReport(ReviewModel):
    passed: bool
    limited: bool = False
    issues: List[VerificationIssue] = Field(default_factory=list)
    semantic_verification_available: bool = False
    graph_semantic_violation_count: int = Field(default=0, ge=0)
    unsupported_major_count: int = Field(default=0, ge=0)


class ReviewBundle(ReviewModel):
    contract: Literal["aspr_gear_review_bundle"] = "aspr_gear_review_bundle"
    status: ReviewStatus
    paper_ir: PaperIR
    calibration: Union[CalibrationPacketV3, SubmissionCalibrationPacketV1]
    graph_context: GraphReviewContext
    critic: CriticRunMetadata
    state: ReviewState
    structured_review: StructuredReview
    review_markdown: str
    verification: VerificationReport
    output_files: Dict[str, str] = Field(default_factory=dict)


def _validate_evidence_keys(value: List[str]) -> List[str]:
    if len(value) != len(set(value)):
        raise ValueError("evidence_keys must be unique")
    for key in value:
        if not re.fullmatch(r"(?:P:S-[A-Za-z0-9_-]+|R:[A-Za-z0-9:_-]+)", key):
            raise ValueError(
                "review evidence keys must reference paper spans or validated relations"
            )
    return value


__all__ = [
    "ContextClaim",
    "ContextSpan",
    "CriticRunMetadata",
    "CriticSource",
    "GraphReviewContext",
    "NoveltyAssessment",
    "NoveltyJudgment",
    "PointSeverity",
    "PointValidationStatus",
    "ReviewAspect",
    "ReviewBundle",
    "ReviewContextPack",
    "ReviewPointState",
    "ReviewPoint",
    "ReviewState",
    "ReviewSummary",
    "StructuredReview",
    "SCHEMA_VERSION",
    "VerificationIssue",
    "VerificationReport",
    "infer_novelty_judgment",
    "word_count",
]
