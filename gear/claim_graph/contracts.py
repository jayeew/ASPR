"""Strict contracts shared by Claim Graph offline stages and runtime insertion."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ClaimGraphModel(BaseModel):
    """Base model for Claim Graph records."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class InnovationClaimType(str, Enum):
    """Contribution role attached to a Claim node, not a graph partition."""

    METHOD = "METHOD"
    FINDING = "FINDING"
    MECHANISM = "MECHANISM"
    RESOURCE = "RESOURCE"
    THEORY = "THEORY"


class InnovationClaim(ClaimGraphModel):
    claim_id: str = Field(min_length=1)
    parent_paper_id: str = Field(min_length=1)
    claim_type: InnovationClaimType
    claim_text: str = Field(min_length=1)
    source_sentence_ids: list[str] = Field(min_length=1)
    source_sentence_texts: list[str] = Field(min_length=1)
    source_fragments: list[str] = Field(min_length=1)
    title: str = Field(min_length=1)
    abstract_text: str = Field(min_length=1)
    publication_date: date

    @model_validator(mode="after")
    def validate_source_binding_lengths(self) -> InnovationClaim:
        """Require each fragment to be paired with a source sentence."""
        if len(self.source_sentence_ids) != len(self.source_sentence_texts):
            raise ValueError("source_sentence_ids 与 source_sentence_texts 长度必须一致")
        if len(self.source_fragments) > len(self.source_sentence_ids):
            raise ValueError("source_fragments 数量不能超过绑定句数量")
        return self


class InnovationClaimInventory(ClaimGraphModel):
    parent_paper_id: str = Field(min_length=1)
    claims: list[InnovationClaim] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def validate_parent_papers(self) -> InnovationClaimInventory:
        """Keep one inventory strictly scoped to its parent paper."""
        if any(claim.parent_paper_id != self.parent_paper_id for claim in self.claims):
            raise ValueError("inventory 内所有 Claim 必须属于同一 parent_paper_id")
        return self


class ClaimCandidateEdge(ClaimGraphModel):
    earlier_claim_id: str = Field(min_length=1)
    later_claim_id: str = Field(min_length=1)
    earlier_paper_id: str = Field(min_length=1)
    later_paper_id: str = Field(min_length=1)
    earlier_claim_type: InnovationClaimType
    later_claim_type: InnovationClaimType
    earlier_publication_date: date
    later_publication_date: date
    cosine_similarity: float | None = None
    semantic_rank: int | None = Field(default=None, ge=1)
    from_semantic: bool = False
    from_paper_path: bool = False

    @model_validator(mode="after")
    def validate_edge(self) -> ClaimCandidateEdge:
        """Enforce the two non-negotiable Claim Graph edge constraints."""
        if self.earlier_paper_id == self.later_paper_id:
            raise ValueError("同一父论文的 Claim 不允许连边")
        if self.earlier_publication_date >= self.later_publication_date:
            raise ValueError("Claim 边必须严格从较早日期指向较晚日期")
        if not self.from_semantic and not self.from_paper_path:
            raise ValueError("Claim 边至少需要一个候选来源")
        return self

    @property
    def is_cross_type(self) -> bool:
        """Whether the edge spans two contribution roles."""
        return self.earlier_claim_type != self.later_claim_type


class ClaimCommunity(ClaimGraphModel):
    claim_id: str = Field(min_length=1)
    claim_type: InnovationClaimType
    community_id: str = Field(min_length=1)
    community_size: int = Field(ge=1)


class ClaimInsertionProfile(ClaimGraphModel):
    claim_id: str = Field(min_length=1)
    parent_paper_id: str = Field(min_length=1)
    claim_type: InnovationClaimType
    neighbor_count: int = Field(ge=0)
    cross_type_neighbor_count: int = Field(ge=0)
    nearest_prior_claim_id: str | None = None
    nearest_prior_similarity: float | None = None
    effective_community_count: float | None = Field(default=None, ge=1.0)
    community_rao_stirling: float | None = Field(default=None, ge=0.0)
    first_observed_recent_nature_pair_share: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    neighbor_induced_density: float | None = Field(default=None, ge=0.0, le=1.0)
    component_merge_count: int | None = Field(default=None, ge=0)
    cross_boundary_weight_share: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_neighbor_counts(self) -> ClaimInsertionProfile:
        """Prevent impossible insertion counts without inventing missing values."""
        if self.cross_type_neighbor_count > self.neighbor_count:
            raise ValueError("cross_type_neighbor_count 不能大于 neighbor_count")
        return self
