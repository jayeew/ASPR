"""Standalone multi-source case contracts, not paper-review input objects."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Claim(Strict):
    claim_id: str
    group_id: str
    text_zh: str
    text_en: str
    claim_type: Literal["METHOD", "FINDING", "MECHANISM"]
    historical_period: str
    cutoff_exclusive: date
    evidence_ids: list[str]
    source_ids: list[str]
    evidence_status: Literal["located", "pending"]
    limitations: list[str]


class Evidence(Strict):
    evidence_id: str
    source_id: str
    claim_ids: list[str]
    text: str
    material_type: Literal["abstract", "fulltext_excerpt"]
    locator: str
    raw_file: str
    raw_sha256: str
    role: Literal["original", "followup"]
    support_scope: str
    relationship: Literal["core_support", "development_context"]


class Source(Strict):
    source_id: str
    claim_ids: list[str]
    role: Literal["original", "followup"]
    doi: str
    url: str
    title: str
    date_lower: date | None
    date_upper: date | None
    date_precision: Literal["day", "month", "year", "unknown"]
    date_basis: str
    acquired_at: str
    authors: list[dict]
    content_version_date: date | None = None
    material_status: Literal["fulltext", "abstract", "reference_only"]
    raw_directory: str


class Group(Strict):
    group_id: str
    topic: str
    claim_ids: list[str] = Field(min_length=1)
    cutoff_exclusive: date
    graph_root: str
    graph_coverage_end: date
    graph_limitations: list[str]
