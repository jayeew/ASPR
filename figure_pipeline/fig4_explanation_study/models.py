from __future__ import annotations

from pathlib import Path
from typing import Literal

from figure_pipeline.fig3_revision.config import ROOT
from figure_pipeline.fig3_revision.models import Record
from figure_pipeline.fig4_rerun.config import Config as ExistingConfig

ASPECTS = {
    'historical_verification': '独立历史核对与剩余增量',
    'knowledge_position': '知识邻域与关系类型',
    'joint_contribution': '多贡献的共同知识结构',
    'structural_resolution': '知识结构的定量辨别',
    'citation_contact': '具体引用接触辨别',
}


class Config(ExistingConfig):
    model: str = 'gpt-6-astra'
    source: Path = ROOT / 'outputs/fig4_capabilities_pilot'
    output: Path = ROOT / 'outputs/fig4_explanation_study'
    executable: str = str(ROOT / 'figure_pipeline/fig4_explanation_study/codex_session')
    timeout_seconds: int = 1800
    synthesis_timeout_seconds: int = 1800
    workers: int = 20
    cli_limit: int = 20
    paper_ids: list[str] | None = None
    conditions: list[str] | None = None
    retry_failed: bool = False


class Evidence(Record):
    kind: Literal['manuscript', 'history', 'graph']
    source_id: str
    location: str
    quote_or_value: str


class GraphUse(Record):
    claim_id: str
    historical_source_ids: list[str]
    useful_relation: str
    limitation: str


class AnswerPart(Record):
    part_id: str
    expected_content: str
    evidence: list[Evidence]
    acceptable_variants: str
    insufficient_answer: str


class Question(Record):
    question_id: str
    aspect: Literal['historical_verification', 'knowledge_position', 'joint_contribution',
                    'structural_resolution', 'citation_contact']
    applicable: bool
    applicability_reason: str
    claim_ids: list[str]
    question: str
    scientific_use: str
    answer_parts: list[AnswerPart]
    manuscript_already_answers: bool
    manuscript_overlap_note: str


class Reference(Record):
    graph_useful: bool
    eligibility_reason: str
    graph_examples: list[GraphUse]
    questions: list[Question]
    limitations: list[str]


class Finding(Record):
    aspect: Literal['historical_verification', 'knowledge_position', 'joint_contribution',
                    'structural_resolution', 'citation_contact']
    claim_ids: list[str]
    observation: str
    interpretation: str
    evidence: list[Evidence]
    limitation: str


class Analysis(Record):
    findings: list[Finding]
    limitations: list[str]


class PartJudgment(Record):
    part_id: str
    status: Literal['correct', 'incorrect', 'not_written', 'unresolved']
    report_segment_ids: list[str]
    scope_correct: bool | None
    grounding: Literal['independent_history', 'native_graph', 'derived_visible_graph',
                       'manuscript_only', 'unsupported', 'unclear']
    evidence_source_ids: list[str]
    reason: str


class QuestionJudgment(Record):
    question_id: str
    parts: list[PartJudgment]


class CandidateJudgment(Record):
    report_id: str
    questions: list[QuestionJudgment]


class Judgments(Record):
    candidates: list[CandidateJudgment]
