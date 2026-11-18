from __future__ import annotations

from pathlib import Path
from typing import Literal

from figure_pipeline.fig3_revision.config import ROOT
from figure_pipeline.fig3_revision.models import Record
from figure_pipeline.fig4_explanation_100.models import Config as PreviousConfig
from figure_pipeline.fig4_explanation_study.models import Evidence, PartJudgment


class Config(PreviousConfig):
    source: Path = ROOT / 'outputs/fig4_reference/experiment'
    output: Path = ROOT / 'outputs/fig5_robustness'
    seed: int = 20261002
    workers: int = 32
    cli_limit: int = 32
    cli_initial: int = 32
    ordinary_limit: int = 295
    additional_limit: int = 30
    call_limit: int = 325


CONDITIONS = ('F', 'E50', 'K5', 'LUNA', 'F_REPEAT')
VIEWS = ('F', 'E50', 'K5')


class ConditionalPart(Record):
    question_id: str
    part_id: str
    answerability: Literal['answerable', 'unanswerable', 'unresolved']
    expected_content: str
    acceptable_variants: str
    evidence: list[Evidence]
    reason: str


class ReferenceView(Record):
    view_id: Literal['F', 'E50', 'K5']
    parts: list[ConditionalPart]


class ConditionalReference(Record):
    views: list[ReferenceView]


class RobustPart(PartJudgment):
    explicitly_abstains: bool | None
    abstention_segment_ids: list[str]
    abstention_appropriateness: Literal['reasonable', 'unnecessary', 'unresolved', 'not_applicable']
    unsupported_definitive: bool | None
    unsupported_segment_ids: list[str]
    support_reason: str


class RobustQuestion(Record):
    question_id: str
    parts: list[RobustPart]


class RobustCandidate(Record):
    report_id: str
    questions: list[RobustQuestion]


class Evaluation(Record):
    candidates: list[RobustCandidate]
