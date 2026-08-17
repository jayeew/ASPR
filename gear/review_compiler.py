"""Deterministically compile validated current point state into five sections."""

from __future__ import annotations

from typing import Dict, Iterable, List

from .review_contracts import (
    NoveltyAssessment,
    PointValidationStatus,
    ReviewAspect,
    ReviewPhase,
    ReviewPoint,
    ReviewSource,
    ReviewState,
    ReviewStateV2,
    ReviewSummary,
    StructuredReview,
    infer_novelty_judgment,
)


def calibration_evidence_key(calibration: object, part: str) -> str:
    """Return the immutable evidence key for a calibration-packet component."""
    contract = str(getattr(calibration, "contract", ""))
    prefix = "G:SCP" if contract == "aspr_submission_calibration_packet_v1" else "G:CP"
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

    def compile_v2(self, state: ReviewStateV2) -> StructuredReview:
        if state.phase not in {
            ReviewPhase.EVIDENCE_FINALIZED,
            ReviewPhase.VERIFIED,
            ReviewPhase.COMPILED,
        }:
            raise ValueError("V2 compiler requires evidence-finalized state")
        points = [
            point
            for point in state.canonical_points.values()
            if point.retained
            and point.validation_status == PointValidationStatus.VALIDATED
        ]
        compiled: Dict[str, List[ReviewPoint]] = {
            "novelty_support": [],
            "novelty_limit": [],
            "strengths": [],
            "weaknesses": [],
            "questions": [],
        }
        for point in points:
            compiled[point.section].append(
                ReviewPoint(
                    point_id=point.point_id,
                    aspect=point.aspect,
                    text=point.resolved_proposition or point.proposition,
                    severity=point.severity,
                    suggested_action=point.suggested_action or "",
                    evidence_keys=list(
                        dict.fromkeys(
                            [
                                *point.paper_evidence_keys,
                                *point.relation_evidence_keys,
                                *point.coverage_evidence_keys,
                            ]
                        )
                    ),
                    external_verification_required=point.requires_external_evidence,
                )
            )
        summary = _rebuild_summary(state, points)
        supporting = compiled["novelty_support"]
        limiting = compiled["novelty_limit"]
        return StructuredReview(
            paper_id=state.paper_id,
            summary=summary,
            novelty=NoveltyAssessment(
                judgment=infer_novelty_judgment(supporting, limiting),
                supporting_points=supporting,
                limiting_points=limiting,
            ),
            strengths=compiled["strengths"],
            weaknesses=compiled["weaknesses"],
            questions=compiled["questions"],
        )

    def compile_verified(self, state: ReviewStateV2) -> StructuredReview:
        if state.phase != ReviewPhase.VERIFIED:
            raise ValueError("final compiler reads VERIFIED state only")
        review = self.compile_v2(state)
        state.phase = ReviewPhase.COMPILED
        return review


def _rebuild_summary(state: ReviewStateV2, points: list) -> ReviewSummary:
    contribution = next(
        (
            point
            for point in points
            if point.aspect == ReviewAspect.CONTRIBUTION
            and point.section in {"strengths", "novelty_support"}
        ),
        None,
    )
    if contribution is not None:
        text = (
            "The manuscript's evidence-backed contribution is: "
            f"{contribution.proposition}"
        )
        keys = list(contribution.paper_evidence_keys)
    else:
        text = (
            "The manuscript was assessed from its stable paper spans; the summary "
            f"reflects {len(points)} retained evidence-backed review point(s)."
        )
        agent = state.branch_reviews.get(ReviewSource.AGENT)
        keys = list(agent.summary.evidence_keys) if agent is not None else []
    if not keys:
        raise ValueError("compiled summary requires at least one paper evidence key")
    return ReviewSummary(text=text, evidence_keys=keys)


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
