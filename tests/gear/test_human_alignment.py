from __future__ import annotations

import pytest

from experiments.gear.human_alignment.contracts import (
    EvaluationTask,
    PaperAlignmentCase,
)
from experiments.gear.human_alignment.evidence_metrics import atomic_match_metrics


def test_empty_empty_review_does_not_score_as_true_match() -> None:
    metrics = atomic_match_metrics([], reference_count=0, candidate_count=0)
    assert metrics["both_empty"] is True
    assert metrics["f1"] is None


def test_revision_aware_data_cannot_enter_submission_time_eval() -> None:
    with pytest.raises(ValueError, match="revision/rebuttal"):
        PaperAlignmentCase(
            paper_id="paper",
            task=EvaluationTask.SUBMISSION_TIME,
            manuscript_version="submission",
            cutoff_date="2025-01-01",
            agent_available=True,
            revision_material_visible_to_agent=True,
        )
