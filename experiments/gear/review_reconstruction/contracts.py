"""Strict contracts for isolated, no-API reconstruction sessions."""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import Field, field_validator, model_validator

from gear.review_contracts import (
    ContextClaim,
    ContextSpan,
    StructuredReview,
    ReviewModel,
)


class ReviewSourceRole(str, Enum):
    REVIEWER_REPORT = "reviewer_report"
    AUTHOR_RESPONSE = "author_response"


class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    PERSISTS = "persists"
    UNVERIFIABLE = "unverifiable"


class ReviewSourceExcerpt(ReviewModel):
    source_key: str
    source_role: ReviewSourceRole
    reviewer_id_hash: Optional[str] = None
    round_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text: str
    text_sha256: str


class ReconstructionPaperContext(ReviewModel):
    contract: Literal["reconstruction_paper_context"] = (
        "reconstruction_paper_context"
    )
    paper_id: str
    paper_sha256: str
    claims: List[ContextClaim]
    spans: List[ContextSpan]


class ReferenceTrace(ReviewModel):
    trace_id: str
    paper_id: str
    point_id: Optional[str] = None
    reviewer_quote_keys: List[str] = Field(min_length=1)
    author_response_keys: List[str] = Field(default_factory=list)
    round_ids: List[str] = Field(min_length=1)
    reviewer_id_hashes: List[str] = Field(min_length=1)
    resolution_status: ResolutionStatus
    rationale: str = ""

    @field_validator(
        "reviewer_quote_keys",
        "author_response_keys",
        "round_ids",
        "reviewer_id_hashes",
    )
    @classmethod
    def unique_refs(cls, value: List[str]) -> List[str]:
        if len(value) != len(set(value)):
            raise ValueError("ReferenceTrace reference lists must be unique")
        return value

    @model_validator(mode="after")
    def resolution_target_rule(self) -> "ReferenceTrace":
        excluded = self.resolution_status in {
            ResolutionStatus.RESOLVED,
            ResolutionStatus.UNVERIFIABLE,
        }
        if excluded and self.point_id is not None:
            raise ValueError("resolved/unverifiable traces cannot target an SFT point")
        if not excluded and self.point_id is None:
            raise ValueError("persisting traces must target a retained review point")
        return self


class RevisionLedgerEntry(ReviewModel):
    ledger_id: str
    paper_id: str
    reviewer_quote_keys: List[str] = Field(min_length=1)
    author_response_keys: List[str] = Field(default_factory=list)
    resolution_status: ResolutionStatus
    final_paper_evidence_keys: List[str] = Field(default_factory=list)
    residual_summary: str = ""


class ReconstructionSessionPackage(ReviewModel):
    contract: Literal["reconstruction_session_package"] = (
        "reconstruction_session_package"
    )
    package_id: str
    session_kind: Literal["reconstruction"] = "reconstruction"
    paper_id: str
    paper_sha256: str
    review_source_sha256: str
    prompt_sha256: str
    schema_sha256: str
    input_sha256: str
    paper_context: ReconstructionPaperContext
    reviewer_spans: List[ReviewSourceExcerpt]
    author_response_spans: List[ReviewSourceExcerpt]
    instructions: str

    @model_validator(mode="after")
    def identity_links(self) -> "ReconstructionSessionPackage":
        if self.paper_context.paper_id != self.paper_id:
            raise ValueError("session paper context identity mismatch")
        if self.paper_context.paper_sha256 != self.paper_sha256:
            raise ValueError("session paper hash mismatch")
        if any(
            span.source_role != ReviewSourceRole.REVIEWER_REPORT
            for span in self.reviewer_spans
        ):
            raise ValueError("reviewer_spans contains a non-reviewer role")
        if any(
            span.source_role != ReviewSourceRole.AUTHOR_RESPONSE
            for span in self.author_response_spans
        ):
            raise ValueError("author_response_spans contains a non-author role")
        source_keys = [
            span.source_key
            for span in [*self.reviewer_spans, *self.author_response_spans]
        ]
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("session source keys must be unique")
        paper_keys = [span.evidence_key for span in self.paper_context.spans]
        if len(paper_keys) != len(set(paper_keys)):
            raise ValueError("paper evidence keys must be unique")
        return self


class ReconstructionSessionResponse(ReviewModel):
    contract: Literal["reconstruction_session_response"] = (
        "reconstruction_session_response"
    )
    package_id: str
    session_kind: Literal["reconstruction"] = "reconstruction"
    paper_id: str
    model_id: str = Field(min_length=1)
    conversation_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    prompt_sha256: str
    schema_sha256: str
    input_sha256: str
    review: StructuredReview
    reference_traces: List[ReferenceTrace]
    revision_ledger: List[RevisionLedgerEntry] = Field(default_factory=list)
    output_sha256: str

    @model_validator(mode="after")
    def response_links(self) -> "ReconstructionSessionResponse":
        if self.review.paper_id != self.paper_id:
            raise ValueError("response review paper_id mismatch")
        point_ids = {point.point_id for point in self.review.all_points()}
        trace_point_ids = {
            trace.point_id for trace in self.reference_traces if trace.point_id
        }
        if not trace_point_ids.issubset(point_ids):
            raise ValueError("ReferenceTrace targets an unknown review point")
        return self

    def hash_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract,
            "package_id": self.package_id,
            "session_kind": self.session_kind,
            "paper_id": self.paper_id,
            "model_id": self.model_id,
            "conversation_hash": self.conversation_hash,
            "prompt_sha256": self.prompt_sha256,
            "schema_sha256": self.schema_sha256,
            "input_sha256": self.input_sha256,
            "review": self.review,
            "reference_traces": self.reference_traces,
            "revision_ledger": self.revision_ledger,
        }


__all__ = [
    "ReconstructionPaperContext",
    "ReconstructionSessionPackage",
    "ReconstructionSessionResponse",
    "ReferenceTrace",
    "ResolutionStatus",
    "ReviewSourceExcerpt",
    "ReviewSourceRole",
    "RevisionLedgerEntry",
]
