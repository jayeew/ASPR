from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from figure_pipeline.fig3_revision.config import ROOT
from figure_pipeline.fig3_revision.models import Record
from figure_pipeline.fig4_rerun.config import Config as RerunConfig, CoreJudgment, SourceEvidence

DIMENSIONS = {
    'scientific_increment': '科学增量解释',
    'knowledge_relations': '知识关系刻画',
    'joint_explanation': '联合贡献解释',
    'structure_explanation': '结构刻画',
    'citation_explanation': '引用联系辨别',
}


class Config(RerunConfig):
    source: Path = ROOT / 'outputs/fig4_rerun'
    output: Path = ROOT / 'outputs/fig4_capabilities_pilot'
    evaluation_round: str = 'capabilities_pilot_v1'


class DimensionJudgment(Record):
    dimension: Literal['scientific_increment', 'knowledge_relations', 'joint_explanation',
                       'structure_explanation', 'citation_explanation']
    status: Literal['assessed', 'not_applicable', 'unresolved']
    score: int | None = Field(ge=0, le=3)
    report_quotes: list[str]
    evidence: list[SourceEvidence]
    graph_fact_fields: list[str]
    reason: str
    unresolved_reason: str


class Evaluation(Record):
    items: list[CoreJudgment]
    dimensions: list[DimensionJudgment]
