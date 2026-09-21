"""Structured scientific outputs; no simulated measurements or missing-to-zero conversion."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OutputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Report(OutputModel):
    body: str = Field(min_length=1000, max_length=5000)
    cited_source_ids: list[str]


class Unit(OutputModel):
    unit_id: str
    quote: str
    claim_ids: list[str]
    substantive: bool
    paraphrase: bool
    needs_verification: bool
    primary_type: Literal["supported_scope", "historical_increment", "knowledge_relation", "appropriate_limitation"]
    insight_type: Literal["none", "historical_increment", "cross_work_relation", "scope_correction", "cross_contribution"]
    cluster_id: str
    cited_source_ids: list[str]
    original_locator: str


class Mention(OutputModel):
    claim_id: str
    quotes: list[str]


class Extracted(OutputModel):
    units: list[Unit]
    mentions: list[Mention]
    additional_contributions: list[str]


class Support(OutputModel):
    unit_id: str
    support: Literal["supported", "partly_supported", "not_verifiable", "contradicted"]
    scope_correct: bool
    source_ids: list[str]
    source_quotes: list[str]
    originally_substantiated: bool
    original_locator: str
    nonparaphrase_insight: bool
    errors: list[Literal["unsupported_definitive", "false_antecedence", "semantic_causal", "omitted_scope"]]
    reason: str


class Supported(OutputModel):
    units: list[Support]


class QualityItem(OutputModel):
    dimension: Literal["contribution_fidelity", "historical_increment", "knowledge_relation", "scope_calibration", "whole_paper_synthesis"]
    score: int = Field(ge=0, le=3)
    applicable: bool
    reason_code: Literal["supported", "missing", "error", "insufficient_material", "partial", "not_applicable"]
    reason: str
    report_quote: str
    source_ids: list[str]


class Quality(OutputModel):
    dimensions: list[QualityItem]


class Preference(OutputModel):
    status: Literal["judged", "cannot_judge"]
    overall: Literal["A", "B", "tie", "cannot_judge"]
    clarity_increment: Literal["A", "B", "tie", "cannot_judge"]
    evidence_traceability: Literal["A", "B", "tie", "cannot_judge"]
    knowledge_usefulness: Literal["A", "B", "tie", "cannot_judge"]
    appropriate_limitations: Literal["A", "B", "tie", "cannot_judge"]
    reason: str
    quote_a: str
    quote_b: str


class InsightCluster(OutputModel):
    cluster_id: str
    unit_keys: list[str]
    primary_type: Literal["historical_increment", "cross_work_relation", "scope_correction", "cross_contribution"]
    summary: str


class InsightClusters(OutputModel):
    clusters: list[InsightCluster]
