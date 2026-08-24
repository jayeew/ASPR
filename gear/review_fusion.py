"""Fuse independent text branches while keeping Graph semantically bounded."""

from __future__ import annotations

import hashlib
from typing import Literal

from .point_matcher import PointMatcher, branch_sections
from .review_contracts import (
    BranchReview,
    CanonicalReviewPoint,
    FusionReport,
    NoveltyJudgment,
    PointValidationStatus,
    ReviewAspect,
    ReviewPhase,
    ReviewPoint,
    ReviewSource,
    ReviewStateV3,
)


class ReviewFusion:
    def __init__(self, *, matcher: PointMatcher | None = None) -> None:
        self.matcher = matcher or PointMatcher()

    def fuse(
        self,
        state: ReviewStateV3,
        agent: BranchReview,
        qwen: BranchReview | None = None,
    ) -> tuple[ReviewStateV3, FusionReport]:
        if agent.source != ReviewSource.AGENT:
            raise ValueError("fusion requires Agent Reviewer as the primary branch")
        if agent.paper_id != state.paper_id:
            raise ValueError("Agent branch paper_id mismatch")
        branches = {ReviewSource.AGENT: agent}
        if qwen is not None:
            if qwen.source != ReviewSource.ASPR_QWEN or qwen.paper_id != state.paper_id:
                raise ValueError("invalid ASPR-Qwen branch")
            branches[ReviewSource.ASPR_QWEN] = qwen
        canonical = self._agent_candidates(agent)
        matches = []
        if qwen is not None:
            matches = self.matcher.match(agent, qwen)
            self._merge_qwen(canonical, agent, qwen, matches)
        tension_scores, focus_weights, triggered = self._apply_graph_calibration(
            canonical, agent, state
        )
        failures = [*agent.failures, *(qwen.failures if qwen is not None else [])]
        report = FusionReport(
            paper_id=state.paper_id,
            matches=matches,
            canonical_point_ids=list(canonical),
            graph_tension_scores=tension_scores,
            graph_focus_weights=focus_weights,
            graph_triggered_actions=triggered,
            graph_guided_point_ids=[
                point_id
                for point_id, point in canonical.items()
                if state.graph_result is not None
                and (
                    state.graph_result.seed_work_ids or state.graph_result.search_terms
                )
                and point.section.startswith("novelty_")
            ],
            graph_query_replacements={
                point_id: "author_terminology->graph_focus"
                for point_id, point in canonical.items()
                if state.graph_result is not None
                and state.graph_result.search_terms
                and point.section.startswith("novelty_")
            },
            failures=failures,
        )
        updated = state.model_copy(
            update={
                "phase": ReviewPhase.FUSED,
                "branch_reviews": branches,
                "novelty_direction": agent.novelty.judgment,
                "novelty_direction_confidence": agent.novelty.confidence,
                "canonical_points": canonical,
                "unresolved_target_ids": list(canonical),
                "process_features": state.process_features.model_copy(
                    update={
                        "agent_review_available": not agent.failures,
                        "qwen_review_available": qwen is not None and not qwen.failures,
                        "graph_text_tension": any(
                            point.graph_tension for point in canonical.values()
                        ),
                        "failure_count": len(failures),
                    }
                ),
            }
        )
        return updated, report

    @staticmethod
    def _agent_candidates(agent: BranchReview) -> dict[str, CanonicalReviewPoint]:
        sections = branch_sections(agent)
        output: dict[str, CanonicalReviewPoint] = {}
        for point in agent.all_points():
            canonical_id = _canonical_id(point)
            output[canonical_id] = _canonical_point(
                canonical_id, point, sections[point.point_id], ReviewSource.AGENT
            )
        return output

    @staticmethod
    def _merge_qwen(
        canonical: dict[str, CanonicalReviewPoint],
        agent: BranchReview,
        qwen: BranchReview,
        matches: list,
    ) -> None:
        agent_to_canonical = {
            source_id: point_id
            for point_id, point in canonical.items()
            for source_id in point.source_point_ids.get(ReviewSource.AGENT, [])
        }
        qwen_map = {point.point_id: point for point in qwen.all_points()}
        qwen_sections = branch_sections(qwen)
        for match in matches:
            if match.qwen_point_id is None:
                continue
            qwen_point = qwen_map[match.qwen_point_id]
            if match.agent_point_id is None:
                canonical_id = _canonical_id(qwen_point)
                canonical[canonical_id] = _canonical_point(
                    canonical_id,
                    qwen_point,
                    qwen_sections[qwen_point.point_id],
                    ReviewSource.ASPR_QWEN,
                )
                continue
            canonical_point = canonical[agent_to_canonical[match.agent_point_id]]
            canonical_point.source_point_ids[ReviewSource.ASPR_QWEN] = [
                qwen_point.point_id
            ]
            canonical_point.qwen_support = match.relation in {
                "SAME_POINT",
                "PARTIAL",
            }
            canonical_point.qwen_conflict = match.relation == "CONTRADICTORY"
            canonical_point.paper_evidence_keys = list(
                dict.fromkeys(
                    [*canonical_point.paper_evidence_keys, *qwen_point.evidence_keys]
                )
            )

    @staticmethod
    def _apply_graph_calibration(
        canonical: dict[str, CanonicalReviewPoint],
        agent: BranchReview,
        state: ReviewStateV3,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, list[str]]]:
        graph = state.graph_result
        if graph is None:
            return {}, {}, {}
        direction = (graph.score_0_100 - 50.0) / 50.0
        axis = {
            NoveltyJudgment.POSITIVE: 1.0,
            NoveltyJudgment.NEGATIVE: -1.0,
            NoveltyJudgment.MIXED: 0.0,
            NoveltyJudgment.UNCERTAIN: 0.0,
            NoveltyJudgment.NOT_DISCUSSED: 0.0,
        }[agent.novelty.judgment]
        tension = min(
            1.0,
            graph.feature_coverage * abs(direction) * abs(direction - axis) / 2.0,
        )
        expected_diffusion = (
            graph.feature_coverage * graph.p_uptake * graph.conditional_diffusion
        )
        diffusion_weight = graph.feature_coverage * graph.conditional_diffusion
        tension_scores: dict[str, float] = {}
        focus_weights: dict[str, float] = {}
        triggered: dict[str, list[str]] = {}
        for point_id, point in canonical.items():
            point.graph_focus_weight = (
                expected_diffusion
                if point.aspect
                in {
                    ReviewAspect.CONTRIBUTION,
                    ReviewAspect.METHOD,
                    ReviewAspect.OTHER,
                }
                else diffusion_weight
            )
            focus_weights[point_id] = point.graph_focus_weight
            if point.section.startswith("novelty_"):
                point.graph_tension_score = tension
                point.graph_extra_counterfactual_actions = 0
                point.graph_tension = tension > 0
                tension_scores[point_id] = tension
                if graph.seed_work_ids or graph.search_terms:
                    triggered[point_id] = ["graph_guided_query_replacement"]
        return tension_scores, focus_weights, triggered


def _canonical_id(point: ReviewPoint) -> str:
    identity = (
        f"{point.aspect.value}|{point.text.strip()}|{'|'.join(point.evidence_keys)}"
    )
    return "CP-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:18]


def _canonical_point(
    canonical_id: str,
    point: ReviewPoint,
    section: Literal[
        "novelty_support", "novelty_limit", "strengths", "weaknesses", "questions"
    ],
    source: ReviewSource,
) -> CanonicalReviewPoint:
    agent = source == ReviewSource.AGENT
    return CanonicalReviewPoint(
        point_id=canonical_id,
        section=section,
        initial_section=section,
        aspect=point.aspect,
        severity=point.severity,
        proposition=point.text,
        novelty_confidence=point.confidence,
        suggested_action=point.suggested_action or None,
        source_point_ids={source: [point.point_id]},
        paper_evidence_keys=[
            key for key in point.evidence_keys if key.startswith("P:")
        ],
        agent_support=agent,
        qwen_support=None if agent else True,
        requires_external_evidence=(
            point.external_verification_required
            or section.startswith("novelty_")
            or point.aspect == ReviewAspect.NOVELTY_PRIOR_ART
        ),
        validation_status=PointValidationStatus.PENDING,
        stability_status=(
            "pending"
            if point.severity.value == "major" or section.startswith("novelty_")
            else "not_required"
        ),
    )


__all__ = ["ReviewFusion"]
