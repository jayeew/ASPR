from __future__ import annotations

from pathlib import Path
from typing import Literal

from figure_pipeline.fig3_revision.config import ROOT
from figure_pipeline.fig3_revision.models import Record
from figure_pipeline.fig5_robustness.models import ConditionalPart, RobustPart
from figure_pipeline.fig5_robustness.models import Config as OldConfig


class Config(OldConfig):
    output: Path = ROOT / 'outputs/fig5_revision'
    old: Path = ROOT / 'outputs/fig5_robustness'
    ordinary_limit: int = 3600
    additional_limit: int = 400
    call_limit: int = 4000
    seed: int = 20261003


class Reference(Record):
    parts: list[ConditionalPart]


class SupportTarget(Record):
    question_id: str
    part_id: str
    support_groups: list[list[str]]
    support_block_ids: list[str]
    alternative_support_audit: str
    critical_remove: list[str]
    noncritical_remove: list[str]
    necessity_reason: str


class Supports(Record):
    targets: list[SupportTarget]
    ineligible_reason: str


class Part(RobustPart):
    response_type: Literal['substantive_answer', 'target_abstention', 'omission', 'unresolved']
    conflicting_material_error: bool | None


class Question(Record):
    question_id: str
    parts: list[Part]


class Candidate(Record):
    report_id: str
    questions: list[Question]


class Evaluation(Record):
    candidates: list[Candidate]
