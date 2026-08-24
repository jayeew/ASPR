"""Strict contracts for GEAR evaluation inputs and results."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from gear.config import CodexCliEndpoint, OpenAICompatibleEndpoint
from gear.contracts import ReviewStatus, StrictModel
from gear.graph_prior import graph_result_v4
from gear.graph_prior_contracts import GraphResultV4
from gear.review_contracts import PointSeverity, ReviewAspect

EvaluationTrack = Literal[
    "review_quality",
    "novelty",
    "evidence_support",
    "revision",
    "reliability",
    "graph_ablation",
    "efficiency",
]
GraphVariantName = Literal["full", "neutral", "score_only", "guidance_only", "shuffled"]


class EvaluationCase(StrictModel):
    case_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    manuscript_path: Path
    metadata_path: Path
    cutoff_date: date
    graph_result: GraphResultV4
    clean_run_dir: Path | None = None
    prior_art_gold_path: Path | None = None

    @field_validator("graph_result", mode="before")
    @classmethod
    def migrate_graph_result(cls, value: object) -> GraphResultV4:
        return graph_result_v4(value)  # type: ignore[arg-type]

    @model_validator(mode="after")
    def graph_identity(self) -> EvaluationCase:
        if self.graph_result.paper_id != self.paper_id:
            raise ValueError("evaluation case Graph paper_id mismatch")
        return self


class EvaluationManifestV1(StrictModel):
    contract: Literal["gear_evaluation_manifest_v1"] = "gear_evaluation_manifest_v1"
    dataset_id: str = Field(min_length=1)
    development_non_confirmatory: bool = True
    human_release_dir: Path
    cases: list[EvaluationCase] = Field(min_length=1)
    tracks: list[EvaluationTrack] = Field(min_length=1)
    bootstrap_samples: int = Field(default=5000, ge=1)
    seed: int = 20260821

    @model_validator(mode="after")
    def unique_cases_and_tracks(self) -> EvaluationManifestV1:
        case_ids = [case.case_id for case in self.cases]
        paper_ids = [case.paper_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("evaluation case_id values must be unique")
        if len(paper_ids) != len(set(paper_ids)):
            raise ValueError("evaluation paper_id values must be unique")
        if len(self.tracks) != len(set(self.tracks)):
            raise ValueError("evaluation tracks must be unique")
        return self


class EvaluatorConfigV1(StrictModel):
    contract: Literal["gear_evaluator_config_v1"] = "gear_evaluator_config_v1"
    backend: Literal["codex_cli", "openai_compatible"]
    codex_cli: CodexCliEndpoint | None = None
    openai_compatible: OpenAICompatibleEndpoint | None = None
    cache_dir: Path

    @model_validator(mode="after")
    def selected_backend_is_configured(self) -> EvaluatorConfigV1:
        endpoint = (
            self.codex_cli if self.backend == "codex_cli" else self.openai_compatible
        )
        if endpoint is None:
            raise ValueError(f"{self.backend} evaluator endpoint is required")
        return self


class RevisionIssueLabel(StrictModel):
    contract: Literal["gear_revision_issue_label_v1"] = "gear_revision_issue_label_v1"
    paper_id: str
    issue_id: str
    text: str = Field(min_length=1)
    section: Literal["novelty", "weaknesses", "questions"]
    aspect: ReviewAspect
    severity: PointSeverity
    status: Literal["persists", "partially_resolved", "resolved", "unverifiable"]
    paper_evidence_keys: list[str] = Field(default_factory=list)


class RevisionIssueDraft(StrictModel):
    text: str = Field(min_length=1)
    section: Literal["novelty", "weaknesses", "questions"]
    aspect: ReviewAspect
    severity: PointSeverity
    status: Literal["persists", "partially_resolved", "resolved", "unverifiable"]
    paper_evidence_keys: list[str] = Field(default_factory=list)


class RevisionIssueMatchDecision(StrictModel):
    issue_id: str
    candidate_point_id: str
    label: Literal["SAME_POINT", "PARTIAL_POINT", "CONTRADICTORY", "NO_MATCH"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class RevisionIssueMatchResponseV1(StrictModel):
    contract: Literal["gear_revision_issue_match_response_v1"] = (
        "gear_revision_issue_match_response_v1"
    )
    paper_id: str
    decisions: list[RevisionIssueMatchDecision]

    @model_validator(mode="after")
    def unique_pairs(self) -> RevisionIssueMatchResponseV1:
        pairs = [(row.issue_id, row.candidate_point_id) for row in self.decisions]
        if len(pairs) != len(set(pairs)):
            raise ValueError("revision match response contains duplicate pairs")
        return self


class PointSupportDecisionV1(StrictModel):
    contract: Literal["gear_point_support_decision_v1"] = (
        "gear_point_support_decision_v1"
    )
    paper_id: str
    point_id: str
    label: Literal[
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "UNSUPPORTED",
        "UNVERIFIABLE",
    ]
    supported_evidence_keys: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class PointSupportResponseV1(StrictModel):
    contract: Literal["gear_point_support_response_v1"] = (
        "gear_point_support_response_v1"
    )
    paper_id: str
    decisions: list[PointSupportDecisionV1] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_points(self) -> PointSupportResponseV1:
        point_ids = [row.point_id for row in self.decisions]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("point support decisions contain duplicate point IDs")
        if any(row.paper_id != self.paper_id for row in self.decisions):
            raise ValueError("point support decision paper_id mismatch")
        return self


class RubricDefinition(StrictModel):
    title: str
    description: str
    polarity: Literal["positive", "risk"]


class RubricSetV1(StrictModel):
    contract: Literal["gear_reviewbench_rubric_set_v1"] = (
        "gear_reviewbench_rubric_set_v1"
    )
    paper_id: str
    rubrics: list[RubricDefinition] = Field(min_length=8, max_length=8)


class RubricDecision(StrictModel):
    title: str
    score: int = Field(ge=-2, le=2)
    rationale: str
    paper_excerpt: str = ""
    unverifiable: bool = False


class RubricScoreResponseV1(StrictModel):
    contract: Literal["gear_reviewbench_score_v1"] = "gear_reviewbench_score_v1"
    paper_id: str
    scores: list[RubricDecision] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def unique_titles(self) -> RubricScoreResponseV1:
        titles = [row.title for row in self.scores]
        if len(titles) != len(set(titles)):
            raise ValueError("rubric response contains duplicate titles")
        return self


class FaultScenarioV1(StrictModel):
    contract: Literal["gear_fault_scenario_v1"] = "gear_fault_scenario_v1"
    scenario_id: str
    kind: Literal[
        "graph_exception",
        "graph_id_mismatch",
        "graph_invalid",
        "semantic_exception",
        "semantic_invalid",
        "retrieval_exception",
        "retrieval_empty",
        "agent_invalid",
        "qwen_required_missing",
        "trace_corruption",
        "wrong_paper_relation",
        "prompt_injection",
        "section_reorder",
        "scattered_information",
        "ocr_noise",
        "unrelated_references",
        "wrong_paper_retrieval",
    ]
    injection_stage: str
    expected_status: ReviewStatus
    required_reason_codes: list[str] = Field(default_factory=list)
    forbidden_output_patterns: list[str] = Field(default_factory=list)
    perturbation_parameters: dict[str, float | int | str | bool] = Field(
        default_factory=dict
    )


class GraphAblationVariant(StrictModel):
    name: GraphVariantName
    result: GraphResultV4

    @field_validator("result", mode="before")
    @classmethod
    def migrate_graph_result(cls, value: object) -> GraphResultV4:
        return graph_result_v4(value)  # type: ignore[arg-type]


class EvaluationContextPack(StrictModel):
    contract: Literal["gear_evaluation_context_v1"] = "gear_evaluation_context_v1"
    paper_id: str
    spans: list[dict[str, object]]
    omitted_span_count: int = Field(default=0, ge=0)
    content_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class NotMeasuredRecord(StrictModel):
    paper_id: str | None = None
    metric: str
    reason: str


__all__ = [
    "EvaluationCase",
    "EvaluationContextPack",
    "EvaluationManifestV1",
    "EvaluationTrack",
    "EvaluatorConfigV1",
    "FaultScenarioV1",
    "GraphAblationVariant",
    "GraphVariantName",
    "NotMeasuredRecord",
    "PointSupportDecisionV1",
    "PointSupportResponseV1",
    "RevisionIssueDraft",
    "RevisionIssueLabel",
    "RevisionIssueMatchDecision",
    "RevisionIssueMatchResponseV1",
    "RubricDecision",
    "RubricDefinition",
    "RubricScoreResponseV1",
    "RubricSetV1",
]
