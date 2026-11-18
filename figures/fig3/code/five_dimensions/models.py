from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import DIMENSIONS

Dimension = Literal['historical_originality', 'understanding_capability',
                    'knowledge_organisation', 'evidential_basis', 'scientific_value']


class Record(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Evidence(Record):
    source_id: str
    quote: str
    explanation: str


class DimensionReport(Record):
    dimension: Dimension
    judgment: str
    rationale: str
    evidence: list[Evidence]
    scope: str
    status: Literal['supported', 'mixed', 'unresolved']


class Report(Record):
    contribution_overview: str
    dimensions: list[DimensionReport]
    synthesis: str

    @model_validator(mode='after')
    def all_dimensions(self) -> Report:
        assert_dimensions([row.dimension for row in self.dimensions])
        return self


class DimensionChecklist(Record):
    dimension: Dimension
    applicability: Literal['applicable', 'not_applicable', 'unresolved']
    criteria: list[str]
    evidence: list[Evidence]
    scope: str
    reason: str


class Checklist(Record):
    dimensions: list[DimensionChecklist]

    @model_validator(mode='after')
    def all_dimensions(self) -> Checklist:
        assert_dimensions([row.dimension for row in self.dimensions])
        return self


class AssertionCheck(Record):
    report_quote: str
    support: Literal['supported', 'partly_supported', 'not_verifiable', 'contradicted']
    scope_correct: bool | None
    evidence: list[Evidence]
    reason: str


class DimensionEvaluation(Record):
    dimension: Dimension
    status: Literal['assessed', 'not_applicable', 'unresolved']
    score: int | None = Field(ge=0, le=3)
    checks: list[AssertionCheck]
    report_quotes: list[str]
    reason: str

    @model_validator(mode='after')
    def valid_score(self) -> DimensionEvaluation:
        if (self.status == 'assessed') != (self.score is not None):
            raise ValueError('Only assessed dimensions have a numeric score')
        return self


class TechnicalFailureOrigin(Record):
    stage: Literal['reference_citation_validation']
    paper_id: str
    dimension: Dimension
    audit_path: str
    error_count: int = Field(ge=1)


class TechnicalDimensionEvaluation(Record):
    dimension: Dimension
    status: Literal['technical_failure']
    score: None
    checks: list[AssertionCheck] = Field(max_length=0)
    report_quotes: list[str] = Field(max_length=0)
    reason: str
    origin: TechnicalFailureOrigin


class Evaluation(Record):
    dimensions: list[DimensionEvaluation | TechnicalDimensionEvaluation]

    @model_validator(mode='after')
    def all_dimensions(self) -> Evaluation:
        assert_dimensions([row.dimension for row in self.dimensions])
        return self


class Reading(Record):
    facts: list[Evidence]
    scope_notes: list[str]


def assert_dimensions(values: list[str]) -> None:
    if len(values) != len(DIMENSIONS) or set(values) != set(DIMENSIONS):
        raise ValueError('Exactly one entry for each of the five dimensions is required')
