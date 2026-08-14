"""Validation helpers for a leakage-free submission-time benchmark."""

from __future__ import annotations

from typing import Iterable, List

from .contracts import EvaluationTask, PaperAlignmentCase


def validate_submission_time_cases(
    cases: Iterable[PaperAlignmentCase],
) -> List[PaperAlignmentCase]:
    validated = list(cases)
    paper_splits: dict[str, str] = {}
    for case in validated:
        if case.task != EvaluationTask.SUBMISSION_TIME:
            raise ValueError(
                f"revision-aware case cannot enter submission-time dataset: {case.paper_id}"
            )
        previous = paper_splits.setdefault(case.paper_id, case.split)
        if previous != case.split:
            raise ValueError(f"paper leaks across dataset splits: {case.paper_id}")
    return validated


__all__ = ["validate_submission_time_cases"]
