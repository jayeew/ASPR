"""Strict disk contracts for the current Graph + GEAR innovation system."""

from __future__ import annotations

from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from gear.claim_graph.contracts import InnovationClaimType
from gear.contracts import StrictModel


class BranchStatus(str, Enum):
    COMPLETE = "complete"
    LIMITED = "limited"
    FAILED = "failed"


class InternalSupportStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "internally_unsupported"


class GearEvidenceStatus(str, Enum):
    INTERNALLY_UNSUPPORTED = "internally_unsupported"
    ANTECEDENT_FOUND = "antecedent_found"
    RESIDUAL_EXTENSION = "residual_extension"
    BOUNDED_NO_ANTECEDENT = "bounded_no_antecedent"
    INCONCLUSIVE = "inconclusive"


class ReviewerStance(str, Enum):
    RECOGNIZED = "recognized"
    INCREMENTAL_OR_LIMITED = "incremental_or_limited"
    CHALLENGED = "challenged"
    UNRESOLVED = "unresolved"


class InnovationPaperInput(StrictModel):
    paper_id: str
    paper_path: Path
    title: str
    doi: str | None = None
    venue: str | None = None
    publication_date: date
    cutoff_date: date
    abstract_text: str
    abstract_source: str
    openalex_work_id: str | None = None
    reference_work_ids: list[str] = Field(default_factory=list)


class NumberedSentence(StrictModel):
    sentence_id: str
    text: str


class GraphClaim(StrictModel):
    claim_id: str
    paper_id: str
    claim_type: InnovationClaimType
    claim_text: str
    source_sentence_ids: list[str]
    source_sentence_texts: list[str]


class GraphNeighbor(StrictModel):
    claim_id: str
    parent_paper_id: str
    parent_openalex_work_id: str | None = None
    claim_type: InnovationClaimType
    claim_text: str
    publication_date: date
    cosine_similarity: float
    semantic_rank: int
    community_id: int | None = None
    direct_citation: bool = False
    two_hop_path_count: int = 0
    shared_reference_count: int = 0
    shared_reference_salton: float = 0.0


class MetricFact(StrictModel):
    name: str
    value: float | int | None
    global_percentile: float | None = None
    claim_type_percentile: float | None = None
    direction: str | None = None


class GraphFactCard(StrictModel):
    claim: GraphClaim
    neighbors: list[GraphNeighbor]
    metrics: list[MetricFact]
    community_ids: list[int] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class GraphBranchResult(StrictModel):
    paper_id: str
    status: BranchStatus
    claims: list[GraphClaim] = Field(default_factory=list)
    fact_cards: list[GraphFactCard] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    output_files: dict[str, str] = Field(default_factory=dict)


class ClaimCandidate(StrictModel):
    candidate_id: str
    claim_type: InnovationClaimType
    author_claim_text: str
    source_span_ids: list[str]


class GearClaim(StrictModel):
    claim_id: str
    claim_type: InnovationClaimType
    author_claim_text: str
    normalized_claim_text: str
    source_span_ids: list[str]
    support_span_ids: list[str]
    internal_support: InternalSupportStatus
    narrowing_reason: str


class SupervisorAction(StrictModel):
    step: int
    action: str
    reason: str
    input_ids: list[str] = Field(default_factory=list)
    output_ids: list[str] = Field(default_factory=list)


class GearClaimCard(StrictModel):
    claim: GearClaim
    status: GearEvidenceStatus
    summary: str
    strongest_relation: str | None = None
    antecedent_work_ids: list[str] = Field(default_factory=list)
    residual_contribution: str | None = None
    evidence_keys: list[str] = Field(default_factory=list)
    assessed_work_ids: list[str] = Field(default_factory=list)
    actions: list[SupervisorAction] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class GearBranchResult(StrictModel):
    paper_id: str
    status: BranchStatus
    claims: list[GearClaim] = Field(default_factory=list)
    claim_cards: list[GearClaimCard] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    output_files: dict[str, str] = Field(default_factory=dict)


class AlignmentLink(StrictModel):
    graph_claim_id: str
    gear_claim_id: str
    relation: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class JointInnovationClaimCard(StrictModel):
    joint_claim_id: str
    graph_claim_ids: list[str]
    gear_claim_ids: list[str]
    statement: str
    evidence_status: GearEvidenceStatus | None = None
    graph_facts: list[MetricFact] = Field(default_factory=list)
    graph_neighbor_ids: list[str] = Field(default_factory=list)
    evidence_keys: list[str] = Field(default_factory=list)
    interpretation: str
    limitations: list[str] = Field(default_factory=list)


class FusionResult(StrictModel):
    paper_id: str
    mode: str
    status: BranchStatus
    alignments: list[AlignmentLink] = Field(default_factory=list)
    joint_claim_cards: list[JointInnovationClaimCard] = Field(default_factory=list)
    recovered_claim_ids: list[str] = Field(default_factory=list)
    graph_triggered_rechecks: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    report_path: str | None = None


class ReviewerClaim(StrictModel):
    reviewer_claim_id: str
    paper_id: str
    reviewer_id: str
    round_number: int
    target_claim_text: str
    stance: ReviewerStance
    source_block_id: str
    source_quote: str


class ReviewerView(StrictModel):
    paper_id: str
    reviewer_id: str
    claims: list[ReviewerClaim]


class DiscussionResolvedReference(StrictModel):
    paper_id: str
    claims: list[ReviewerClaim]
    resolution_notes: dict[str, str] = Field(default_factory=dict)


class EvaluationCase(StrictModel):
    paper_id: str
    system_name: str
    predicted_claim_id: str
    reference_claim_id: str | None = None
    semantic_similarity: float | None = None
    judge_match: bool | None = None
    predicted_stance: ReviewerStance | None = None
    reference_stance: ReviewerStance | None = None


class EvaluationSummary(StrictModel):
    system_name: str
    paper_count: int
    predicted_claim_count: int
    reference_claim_count: int
    claim_precision: float | None = None
    claim_recall: float | None = None
    claim_f1: float | None = None
    stance_macro_f1: float | None = None
    details_path: str


class ClaimList(StrictModel):
    claims: list[dict[str, Any]]

    @model_validator(mode="after")
    def nonempty_claims(self) -> "ClaimList":
        if not self.claims:
            raise ValueError("模型没有返回任何 Claim")
        return self
