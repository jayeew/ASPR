"""Strict v2 artifacts; legacy result contracts remain readable separately."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from gear.contracts import StrictModel
from gear.review_contracts import BranchStatus, GearClaim, ReviewerStance


class ClaimSet(StrictModel):
    schema_version: Literal["innovation_v2"] = "innovation_v2"
    paper_id: str
    input_fingerprint: str
    claims: list[GearClaim]
    candidate_records: list[dict[str, object]] = Field(default_factory=list)
    consolidation_records: list[dict[str, object]] = Field(default_factory=list)


class Finding(StrictModel):
    dimension: Literal[
        "internal_support",
        "historical_basis",
        "increment",
        "knowledge_relation",
        "combination",
        "local_structure",
        "novelty",
    ]
    text: str
    evidence_keys: list[str]
    stance: ReviewerStance | None = None


class Assessment(StrictModel):
    claim_id: str
    claim_text: str
    supported_scope: str
    findings: list[Finding]
    overall_stance: ReviewerStance
    overall_reason: str
    limitations: list[str]


class AnalysisResult(StrictModel):
    schema_version: Literal["innovation_v2"] = "innovation_v2"
    paper_id: str
    system: str
    claim_fingerprint: str
    status: BranchStatus
    assessments: list[Assessment] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class HumanPoint(StrictModel):
    contribution_id: str = ""
    reference_id: str
    paper_id: str
    reviewer_id: str
    round_number: int
    source_block_id: str
    source_quote: str
    target_text: str
    dimension: Literal[
        "identification", "firstness", "increment", "conceptual", "knowledge_relation"
    ]
    stance: ReviewerStance | None
    reasons: list[str]
    tier: Literal["A", "B", "excluded"]
    version_status: Literal["applicable", "uncertain", "conflict"]
    exclusion_reason: str


class ReferenceSet(StrictModel):
    schema_version: Literal["innovation_v2"] = "innovation_v2"
    paper_id: str
    source_fingerprint: str
    status: BranchStatus
    points: list[HumanPoint]
    retained_ids: list[str]
    limitations: list[str]
