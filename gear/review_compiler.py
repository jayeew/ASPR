"""Deterministically compile validated current point state into five sections."""

from __future__ import annotations

from collections.abc import Iterable

from .review_contracts import (
    CanonicalReviewPoint,
    NoveltyAssessment,
    NoveltyJudgment,
    NoveltyVerificationStatus,
    PointValidationStatus,
    ReviewAspect,
    ReviewPhase,
    ReviewPoint,
    ReviewSource,
    ReviewState,
    ReviewStateV3,
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
        uncertain = self._retained(draft.novelty.uncertain_points, state)
        return StructuredReview(
            paper_id=draft.paper_id,
            summary=draft.summary,
            novelty=NoveltyAssessment(
                judgment=draft.novelty.judgment,
                verification_status=draft.novelty.verification_status,
                confidence=draft.novelty.confidence,
                supporting_points=supporting,
                limiting_points=limiting,
                uncertain_points=uncertain,
            ),
            strengths=self._retained(draft.strengths, state),
            weaknesses=self._retained(draft.weaknesses, state),
            questions=self._retained(draft.questions, state),
        )

    @staticmethod
    def _retained(
        points: Iterable[ReviewPoint], state: ReviewState
    ) -> list[ReviewPoint]:
        output: list[ReviewPoint] = []
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

    def compile_v3(self, state: ReviewStateV3) -> StructuredReview:
        if state.phase not in {
            ReviewPhase.EVIDENCE_FINALIZED,
            ReviewPhase.VERIFIED,
            ReviewPhase.COMPILED,
        }:
            raise ValueError("V3 compiler requires evidence-finalized state")
        points = [
            point
            for point in state.canonical_points.values()
            if point.retained
            and point.validation_status == PointValidationStatus.VALIDATED
        ]
        compiled: dict[str, list[ReviewPoint]] = {
            "novelty_support": [],
            "novelty_limit": [],
            "novelty_uncertain": [],
            "strengths": [],
            "weaknesses": [],
            "questions": [],
        }
        for point in points:
            section = (
                "novelty_uncertain"
                if point.section == "questions" and point.requires_external_evidence
                else point.section
            )
            compiled[section].append(
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
                    confidence=point.novelty_confidence,
                )
            )
        summary = _rebuild_summary(state, points)
        supporting = compiled["novelty_support"]
        limiting = compiled["novelty_limit"]
        uncertain = compiled["novelty_uncertain"]
        direction = _preserved_novelty_direction(state, supporting, limiting, uncertain)
        verification_status = _novelty_verification_status(state)
        confidence = _novelty_direction_confidence(state, verification_status)
        state.novelty_direction = direction
        state.novelty_verification_status = verification_status
        state.novelty_direction_confidence = confidence
        return StructuredReview(
            paper_id=state.paper_id,
            summary=summary,
            novelty=NoveltyAssessment(
                judgment=direction,
                verification_status=verification_status,
                confidence=confidence,
                supporting_points=supporting,
                limiting_points=limiting,
                uncertain_points=uncertain,
            ),
            strengths=compiled["strengths"],
            weaknesses=compiled["weaknesses"],
            questions=compiled["questions"],
        )

    def compile_verified(self, state: ReviewStateV3) -> StructuredReview:
        if state.phase != ReviewPhase.VERIFIED:
            raise ValueError("final compiler reads VERIFIED state only")
        review = self.compile_v3(state)
        state.phase = ReviewPhase.COMPILED
        return review


def _rebuild_summary(state: ReviewStateV3, points: list) -> ReviewSummary:
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


def _preserved_novelty_direction(
    state: ReviewStateV3,
    supporting: list[ReviewPoint],
    limiting: list[ReviewPoint],
    uncertain: list[ReviewPoint],
) -> NoveltyJudgment:
    """Keep the graph-blind direction independent of evidence availability."""
    if state.novelty_direction is not None:
        return state.novelty_direction
    agent = state.branch_reviews.get(ReviewSource.AGENT)
    if agent is not None:
        return agent.novelty.judgment
    return infer_novelty_judgment(supporting, limiting, uncertain)


def _novelty_verification_status(
    state: ReviewStateV3,
) -> NoveltyVerificationStatus:
    novelty_points = [
        point
        for point in state.canonical_points.values()
        if _is_novelty_point(state, point)
    ]
    if not novelty_points:
        return NoveltyVerificationStatus.NOT_ASSESSED
    insufficient_markers = (
        "budget_exhausted",
        "finalized_unresolved",
        "insufficient_coverage",
        "retrieval_gap",
        "not_fully_verified",
        "single_antecedent_downgraded",
    )
    verified = 0
    for point in novelty_points:
        incomplete = not point.retained or any(
            marker in note
            for note in point.validation_notes
            for marker in insufficient_markers
        )
        if (
            not incomplete
            and point.validation_status == PointValidationStatus.VALIDATED
        ):
            verified += 1
    if verified == len(novelty_points):
        return NoveltyVerificationStatus.VERIFIED
    if verified:
        return NoveltyVerificationStatus.PARTIALLY_VERIFIED
    return NoveltyVerificationStatus.INSUFFICIENT_COVERAGE


def _novelty_direction_confidence(
    state: ReviewStateV3,
    status: NoveltyVerificationStatus,
) -> float | None:
    direction = state.novelty_direction
    if direction in {NoveltyJudgment.NOT_DISCUSSED, None}:
        return None
    base = state.novelty_direction_confidence
    point_confidences = [
        point.novelty_confidence
        for point in state.canonical_points.values()
        if _is_novelty_point(state, point) and point.novelty_confidence is not None
    ]
    if base is None and point_confidences:
        base = sum(point_confidences) / len(point_confidences)
    ceiling = {
        NoveltyVerificationStatus.VERIFIED: 0.85,
        NoveltyVerificationStatus.PARTIALLY_VERIFIED: 0.65,
        NoveltyVerificationStatus.INSUFFICIENT_COVERAGE: 0.40,
        NoveltyVerificationStatus.NOT_ASSESSED: 0.50,
    }[status]
    return ceiling if base is None else min(base, ceiling)


def _is_novelty_point(state: ReviewStateV3, point: CanonicalReviewPoint) -> bool:
    section = point.initial_section or point.section
    if section.startswith("novelty_"):
        return True
    for source, branch in state.branch_reviews.items():
        branch_novelty_ids = {
            review_point.point_id
            for review_point in [
                *branch.novelty.supporting_points,
                *branch.novelty.limiting_points,
                *branch.novelty.uncertain_points,
            ]
        }
        if branch_novelty_ids.intersection(point.source_point_ids.get(source, [])):
            return True
    return False


def render_markdown(review: StructuredReview) -> str:
    lines: list[str] = [f"# ASPR-GEAR Peer Review: {review.paper_id}", ""]
    lines.extend(["## Contribution summary", "", review.summary.text, ""])
    summary_refs = " ".join(f"[{key}]" for key in review.summary.evidence_keys)
    if summary_refs:
        lines.extend([summary_refs, ""])
    lines.extend(["## Novelty", "", f"Judgment: `{review.novelty.judgment.value}`", ""])
    lines.extend(
        [
            f"Verification: `{review.novelty.verification_status.value}`",
            "",
        ]
    )
    if review.novelty.confidence is not None:
        lines.extend([f"Direction confidence: `{review.novelty.confidence:.2f}`", ""])
    lines.extend(_render_points("Supporting points", review.novelty.supporting_points))
    lines.extend(_render_points("Limiting points", review.novelty.limiting_points))
    lines.extend(_render_points("Uncertain points", review.novelty.uncertain_points))
    lines.extend(["## Strengths", ""])
    lines.extend(_render_point_list(review.strengths))
    lines.extend(["## Weaknesses", ""])
    lines.extend(_render_point_list(review.weaknesses))
    lines.extend(["## Questions", ""])
    lines.extend(_render_point_list(review.questions))
    return "\n".join(lines).rstrip() + "\n"


def _render_points(title: str, points: list[ReviewPoint]) -> list[str]:
    return [f"### {title}", "", *_render_point_list(points)]


def _render_point_list(points: list[ReviewPoint]) -> list[str]:
    if not points:
        return ["No evidence-backed point was retained.", ""]
    lines: list[str] = []
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
