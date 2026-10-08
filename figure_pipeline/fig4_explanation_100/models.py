from __future__ import annotations

from pathlib import Path

from figure_pipeline.fig3_revision.config import ROOT
from figure_pipeline.fig4_explanation_study.models import (
    ASPECTS, Analysis, Judgments, Question, Reference as PreviousReference,
)
from figure_pipeline.fig4_rerun.config import CONDITIONS, NAMES, Report
from figure_pipeline.fig4_rerun.config import Config as PreviousConfig


class Config(PreviousConfig):
    model: str = 'gpt-6.1-sol'
    source: Path = ROOT / 'outputs/fig4_rerun'
    output: Path = ROOT / 'outputs/fig4_explanation_100'
    executable: str = str(ROOT / 'figure_pipeline/fig4_explanation_study/codex_session')
    workers: int = 64
    cli_limit: int = 64
    cli_initial: int = 32
    timeout_seconds: int = 1800
    synthesis_timeout_seconds: int = 1800
    generation_effort: str = 'medium'
    evaluation_effort: str = 'high'
    ordinary_limit: int = 1800
    additional_limit: int = 180
    call_limit: int = 1980
    ledger_root: Path | None = None
    supplemental_round: bool = False
    focused_refinement: bool = False
    review_joint: bool = False
    expanded_protocol: bool = False


class PublicQuestion(Question):
    manuscript_anchor: str


class Reference(PreviousReference):
    questions: list[PublicQuestion]


QUESTION_IDS = ('H1', 'H2', 'K1', 'K2', 'J1', 'J2', 'S1', 'S2', 'P1', 'P2')
QUESTION_ASPECTS = dict(zip(QUESTION_IDS, [a for a in ASPECTS for _ in range(2)]))
