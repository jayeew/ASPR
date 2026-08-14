"""Deterministically compile validated current point state into five sections."""

from __future__ import annotations

from typing import Iterable, List

from .review_contracts import (
    NoveltyAssessment,
    ReviewPoint,
    ReviewState,
    StructuredReview,
    infer_novelty_judgment,
)


def calibration_evidence_key(calibration: object, part: str) -> str:
    """Return the immutable evidence key for a calibration-packet component."""
    contract = str(getattr(calibration, "contract", ""))
    prefix = (
        "G:SCP" if contract == "aspr_submission_calibration_packet_v1" else "G:CP"
    )
    return f"{prefix}:{part}"


class ReviewCompiler:
    """Filter rejected points; never add graph or recommendation prose."""

    def compile(self, state: ReviewState) -> StructuredReview:
        draft = state.draft_review
        supporting = self._retained(draft.novelty.supporting_points, state)
        limiting = self._retained(draft.novelty.limiting_points, state)
        return StructuredReview(
            paper_id=draft.paper_id,
            summary=draft.summary,
            novelty=NoveltyAssessment(
                judgment=infer_novelty_judgment(supporting, limiting),
                supporting_points=supporting,
                limiting_points=limiting,
            ),
            strengths=self._retained(draft.strengths, state),
            weaknesses=self._retained(draft.weaknesses, state),
            questions=self._retained(draft.questions, state),
        )

    @staticmethod
    def _retained(
        points: Iterable[ReviewPoint], state: ReviewState
    ) -> List[ReviewPoint]:
        output: List[ReviewPoint] = []
        for point in points:
            point_state = state.point_states[point.point_id]
            if not point_state.retained:
                continue
            evidence_keys = list(
                dict.fromkeys(
                    [*point_state.evidence_keys, *point_state.relation_evidence_keys]
                )
            )
            output.append(point.model_copy(update={"evidence_keys": evidence_keys}))
        return output


def render_markdown(review: StructuredReview) -> str:
    lines: List[str] = [f"# ASPR-GEAR Peer Review: {review.paper_id}", ""]
    lines.extend(["## Contribution summary", "", review.summary.text, ""])
    summary_refs = " ".join(f"[{key}]" for key in review.summary.evidence_keys)
    if summary_refs:
        lines.extend([summary_refs, ""])
    lines.extend(["## Novelty", "", f"Judgment: `{review.novelty.judgment.value}`", ""])
    lines.extend(_render_points("Supporting points", review.novelty.supporting_points))
    lines.extend(_render_points("Limiting points", review.novelty.limiting_points))
    lines.extend(["## Strengths", ""])
    lines.extend(_render_point_list(review.strengths))
    lines.extend(["## Weaknesses", ""])
    lines.extend(_render_point_list(review.weaknesses))
    lines.extend(["## Questions", ""])
    lines.extend(_render_point_list(review.questions))
    return "\n".join(lines).rstrip() + "\n"


def _render_points(title: str, points: List[ReviewPoint]) -> List[str]:
    return [f"### {title}", "", *_render_point_list(points)]


def _render_point_list(points: List[ReviewPoint]) -> List[str]:
    if not points:
        return ["No evidence-backed point was retained.", ""]
    lines: List[str] = []
    for point in points:
        refs = " ".join(f"[{key}]" for key in point.evidence_keys)
        action = (
            f" Suggested action: {point.suggested_action}"
            if point.suggested_action
            else ""
        )
        lines.extend([f"- {point.text}{action} {refs}".rstrip(), ""])
    return lines


__all__ = ["ReviewCompiler", "calibration_evidence_key", "render_markdown"]
