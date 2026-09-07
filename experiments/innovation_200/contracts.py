"""Plain intermediate contracts shared by the independent experiment scripts."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from gear.contracts import StrictModel
from gear.review_contracts import ReviewerStance

SystemName = Literal[
    "direct_llm",
    "gear",
    "graph",
    "fusion",
    "fusion_no_joint",
    "fusion_no_metrics",
    "fusion_no_citation_paths",
    "fusion_text_only",
]

SYSTEMS: tuple[str, ...] = (
    "direct_llm",
    "gear",
    "graph",
    "fusion",
    "fusion_no_joint",
    "fusion_no_metrics",
    "fusion_no_citation_paths",
    "fusion_text_only",
)


class PaperRow(StrictModel):
    paper_id: str
    title: str
    doi: str
    journal_id: str
    journal_name: str
    publication_date: str
    paper_path: str
    review_path: str
    field_name: str
    field_source: Literal["historical_metadata", "openalex", "model"]
    abstract_text: str
    openalex_work_id: str | None = None
    reference_work_ids: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)


class HumanReference(StrictModel):
    reference_id: str
    paper_id: str
    reviewer_id: str
    round_number: int
    contribution_id: str = ""
    contribution_text: str
    dimension: Literal[
        "identification", "firstness", "increment", "conceptual", "knowledge_relation"
    ]
    stance: ReviewerStance | None = None
    reasons: list[str] = Field(default_factory=list)
    source_quote: str
    source_context: str
    tier: Literal["A", "B"]
    use_in_main: bool = True


class HumanReferenceSet(StrictModel):
    paper_id: str
    references: list[HumanReference]
    limitations: list[str] = Field(default_factory=list)


class ReportSource(StrictModel):
    source_id: str
    passage_id: str
    source_type: Literal[
        "manuscript", "fulltext", "abstract", "historical_claim", "reference_only"
    ]
    title: str = ""
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    passage: str


class ReportDraft(StrictModel):
    body: str = Field(min_length=1000, max_length=4000)
    cited_source_ids: list[str] = Field(default_factory=list)


class ReportBundle(StrictModel):
    paper_id: str
    system: SystemName
    body: str = Field(min_length=1000, max_length=4000)
    references: list[ReportSource] = Field(default_factory=list)


class HumanMatch(StrictModel):
    reference_id: str
    scope: Literal["same", "partial", "none"]
    predicted_stance: ReviewerStance | None = None
    reason_coverage: Literal["complete", "partial", "missing"]
    contradiction: bool
    rationale: str


class HumanEvaluation(StrictModel):
    paper_id: str
    system: SystemName
    matches: list[HumanMatch]


Preference = Literal["A", "B", "tie"]


class PairwiseEvaluation(StrictModel):
    paper_id: str
    baseline_system: SystemName
    fusion_position: Literal["A", "B"]
    overall: Preference
    contribution_accuracy: Preference
    historical_comparison: Preference
    knowledge_explanation: Preference
    evidence_uncertainty: Preference
    practical_value: Preference
    rationale: str
