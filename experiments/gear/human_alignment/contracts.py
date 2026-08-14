"""Strict data contracts separating submission-time and revision-aware tasks."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import List, Literal, Optional

from pydantic import Field, model_validator

from gear.contracts import StrictModel
from gear.review_contracts import NoveltyJudgment, PointSeverity, ReviewAspect


class EvaluationTask(str, Enum):
    SUBMISSION_TIME = "submission_time"
    REVISION_AWARE_AUDIT = "revision_aware_audit"


class HumanReviewPoint(StrictModel):
    point_id: str
    reviewer_id: str
    aspect: ReviewAspect
    severity: PointSeverity
    proposition: str
    paper_evidence_keys: List[str] = Field(default_factory=list)
    prior_work_ids: List[str] = Field(default_factory=list)


class PaperAlignmentCase(StrictModel):
    paper_id: str
    task: EvaluationTask
    manuscript_version: str
    cutoff_date: date
    human_points: List[HumanReviewPoint] = Field(default_factory=list)
    agent_point_ids: List[str] = Field(default_factory=list)
    human_novelty: Optional[NoveltyJudgment] = None
    agent_novelty: Optional[NoveltyJudgment] = None
    agent_available: bool
    revision_material_visible_to_agent: bool = False
    split: Literal["train", "dev_mini", "dev", "test"] = "dev"

    @model_validator(mode="after")
    def prevent_temporal_leakage(self) -> "PaperAlignmentCase":
        if (
            self.task == EvaluationTask.SUBMISSION_TIME
            and self.revision_material_visible_to_agent
        ):
            raise ValueError(
                "revision/rebuttal material cannot enter submission-time evaluation"
            )
        return self


class PointMatch(StrictModel):
    paper_id: str
    human_point_id: Optional[str] = None
    agent_point_id: Optional[str] = None
    label: Literal["SAME", "PARTIAL", "CONTRADICTORY", "NO_MATCH"]


__all__ = [
    "EvaluationTask",
    "HumanReviewPoint",
    "PaperAlignmentCase",
    "PointMatch",
]
