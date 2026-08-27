"""Deterministic post-fusion Graph guidance under matched resource caps."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Literal, TypeVar

from .contracts import ScientificSearchFrame
from .graph_calibration import ForecastAnalogIndex
from .graph_prior_contracts import (
    AnalogSeed,
    CalibrationTension,
    ClaimGuidance,
    GraphResourceCaps,
    GraphRuntimePacket,
    InfluenceForecast,
    RetrievalGuidancePlan,
    RetrievalMission,
    TopologySeed,
)
from .review_contracts import CanonicalReviewPoint, ReviewAspect, ReviewState

GRAPH_GUIDANCE_POLICY_VERSION = "primary16_forecast_calibration_v1"
MAX_TOPOLOGY_GUIDED_CLAIMS = 2
GENERIC_SCIENTIFIC_TOKENS = {
    "analysis",
    "approach",
    "based",
    "contribution",
    "data",
    "effect",
    "evidence",
    "method",
    "mechanism",
    "model",
    "novel",
    "paper",
    "result",
    "study",
    "system",
    "work",
}
PRIOR_ART_DIRECTION_RE = re.compile(
    r"\b(?:antecedents?|predecessors?|earlier work|existing work|"
    r"closest (?:external )?(?:antecedents?|literature|predecessors?|prior work)|"
    r"first(?:-ever)?|literature|"
    r"nearest prior|novel(?:ty)?|precedent|previous(?:ly)?|prior(?:-| )art|"
    r"prior (?:approaches?|catalysts?|literature|methods?|studies|systems?)|"
    r"prior work|state(?:-| )of(?:-| )the(?:-| )art|unprecedented)\b|"
    r"先例|先前|已有|首次|文献",
    re.IGNORECASE,
)
ABSOLUTE_PRIORITY_RE = re.compile(
    r"\b(?:first(?:-ever)?|first comprehensive|never before|unprecedented|"
    r"previously unreported|no prior work)\b|首次|前所未有|未曾报道",
    re.IGNORECASE,
)
MissionType = Literal[
    "local_nearest_antecedent",
    "remote_mechanism_analogue",
    "topology_seed",
]
MissionOrigin = Literal["score", "topology", "calibration"]
MissionOrientation = Literal["neutral"]
Traversal = Literal["none", "references", "citations"]
CandidateT = TypeVar("CandidateT")


def routing_weights(
    forecast: InfluenceForecast | None,
    *,
    use_score: bool,
) -> tuple[float, float, float]:
    """Return frozen equal-budget routing weights.

    Coverage shrinks the forecast toward the neutral midpoint.  An unavailable
    forecast and the neutral arm are identical by construction.
    """

    if (
        not use_score
        or forecast is None
        or forecast.status != "available"
        or forecast.prospective_5y_diffusion_percentile is None
    ):
        q_effective = 0.5
    else:
        q = forecast.prospective_5y_diffusion_percentile / 100.0
        q_effective = 0.5 + forecast.feature_coverage * (q - 0.5)
    remote_weight = 0.25 + 0.5 * q_effective
    return 1.0 - remote_weight, remote_weight, q_effective


def weighted_interleave(
    local: Sequence[CandidateT],
    remote: Sequence[CandidateT],
    *,
    local_weight: float,
    remote_weight: float,
    limit: int,
) -> list[CandidateT]:
    """Deterministically reorder two already-ranked pools without changing them."""

    if limit < 0 or local_weight < 0.0 or remote_weight < 0.0:
        raise ValueError("interleaving limit and weights must be non-negative")
    if local_weight + remote_weight <= 0.0:
        raise ValueError("at least one interleaving weight must be positive")
    output: list[CandidateT] = []
    local_index = 0
    remote_index = 0
    local_credit = 0.0
    remote_credit = 0.0
    while len(output) < limit and (
        local_index < len(local) or remote_index < len(remote)
    ):
        local_credit += local_weight
        remote_credit += remote_weight
        choose_remote = remote_credit > local_credit
        if local_index >= len(local):
            choose_remote = True
        elif remote_index >= len(remote):
            choose_remote = False
        if choose_remote:
            output.append(remote[remote_index])
            remote_index += 1
            remote_credit -= 1.0
        else:
            output.append(local[local_index])
            local_index += 1
            local_credit -= 1.0
    return output


def score_controller(
    packet: GraphRuntimePacket,
    query_slots: int = 8,
    *,
    enabled: bool = False,
) -> tuple[int, int, float]:
    """Return matched local/remote geometry; forecast routing is opt-in only."""
    _, _, q_effective = routing_weights(packet.forecast, use_score=enabled)
    local = query_slots // 2
    return local, query_slots - local, q_effective


def is_graph_guidance_target(point: CanonicalReviewPoint) -> bool:
    return bool(
        point.retained
        and (
            point.section.startswith("novelty_")
            or (
                point.section == "questions"
                and point.aspect
                in {ReviewAspect.NOVELTY_PRIOR_ART, ReviewAspect.CONTRIBUTION}
                and point.requires_external_evidence
            )
        )
    )


class GraphGuidancePlanner:
    """Build claim-specific missions after graph-blind text fusion."""

    def __init__(
        self,
        *,
        resource_caps: GraphResourceCaps | None = None,
        policy_version: str = GRAPH_GUIDANCE_POLICY_VERSION,
        analog_index: ForecastAnalogIndex | None = None,
    ) -> None:
        self.resource_caps = resource_caps or GraphResourceCaps()
        self.policy_version = policy_version
        self.analog_index = analog_index

    def plan(
        self,
        state: ReviewState,
        *,
        search_frames: Mapping[str, ScientificSearchFrame] | None = None,
        enable_score_routing: bool = False,
        enable_topology: bool = True,
        calibration_variant: Literal[
            "neutral",
            "topology_only",
            "scalar_score",
            "hgb_analog",
            "full_calibrated",
            "shuffled_hgb",
        ] = "topology_only",
    ) -> RetrievalGuidancePlan:
        packet = state.graph_result
        if packet is None:
            raise ValueError("Graph guidance requires a runtime packet")
        points = sorted(
            (
                point
                for point in state.canonical_points.values()
                if is_graph_guidance_target(point)
            ),
            key=_guidance_sort_key,
        )
        # The old planner always apportioned all eight paper-level slots even
        # though the runtime could execute only two normal queries per claim.
        # That made part of the Score policy declarative rather than causal.
        # Three executable slots per claim are enough to distinguish the local
        # and remote geometries while retaining a strict paper-level cap.
        executable_slots = min(
            self.resource_caps.provider_searches,
            max(3, 3 * len(points)),
        )
        scalar_score = enable_score_routing or calibration_variant == "scalar_score"
        topology_enabled = enable_topology and calibration_variant in {
            "topology_only",
            "hgb_analog",
            "full_calibrated",
            "shuffled_hgb",
        }
        analog_enabled = calibration_variant in {
            "hgb_analog",
            "full_calibrated",
            "shuffled_hgb",
        }
        full_calibration = calibration_variant == "full_calibrated"
        local, remote, q_effective = score_controller(
            packet,
            executable_slots,
            enabled=scalar_score,
        )
        weights = [_guidance_weight(point) for point in points]
        local_alloc, remote_alloc = _allocate_geometry(local, remote, weights)
        frames = search_frames or {}
        seed_usage: dict[str, int] = {}
        guidance = []
        topology_claims_used = 0
        for index, point in enumerate(points):
            frame = frames.get(point.point_id)
            analog_seeds = (
                self.analog_index.select(
                    packet.forecast_anatomy,
                    claim_id=point.novelty_claim_id or point.point_id,
                    terms=_frame_terms(frame),
                    cutoff_date=state.cutoff_date,
                    target_field=(
                        packet.forecast_anatomy.target_field
                        if packet.forecast_anatomy is not None
                        else None
                    ),
                    shuffled=calibration_variant == "shuffled_hgb",
                )
                if analog_enabled and self.analog_index is not None
                else []
            )
            tension = (
                next(
                    (item for item in packet.calibration_tensions if item.active), None
                )
                if full_calibration
                else None
            )
            if tension is not None:
                point.graph_tension = True
                point.graph_tension_score = tension.score
                point.validation_notes.append(
                    f"calibration_tension:{tension.kind}:{tension.review_effect}"
                )
            claim_guidance = self._claim_guidance(
                point,
                packet,
                frame,
                local_alloc[index],
                remote_alloc[index],
                q_effective,
                topology_enabled and topology_claims_used < MAX_TOPOLOGY_GUIDED_CLAIMS,
                seed_usage,
                analog_seeds=analog_seeds,
                tension=tension,
            )
            if any(mission.origin == "topology" for mission in claim_guidance.missions):
                topology_claims_used += 1
            guidance.append(claim_guidance)
        return RetrievalGuidancePlan(
            paper_id=state.paper_id,
            policy_version=self.policy_version,
            source_packet_evidence_key=state.graph_result_evidence_key or "G:RESULT",
            controller_state={
                "score_0_100": packet.score_0_100,
                "score_semantics": "prospective_5y_diffusion_percentile",
                "score_routing_active": scalar_score,
                "calibration_variant": calibration_variant,
                "feature_reliability": packet.feature_coverage,
                "q_effective": q_effective,
                "executable_query_slots": executable_slots,
                "local_slots": local,
                "remote_slots": remote,
            },
            resource_caps=self.resource_caps,
            claim_guidance=guidance,
            no_effect_reason=None if points else "no_retained_novelty_claim",
        )

    def _claim_guidance(
        self,
        point: CanonicalReviewPoint,
        packet: GraphRuntimePacket,
        frame: ScientificSearchFrame | None,
        local_slots: int,
        remote_slots: int,
        q_effective: float,
        enable_topology: bool,
        seed_usage: dict[str, int],
        *,
        analog_seeds: Sequence[AnalogSeed],
        tension: CalibrationTension | None,
    ) -> ClaimGuidance:
        claim_id = point.novelty_claim_id or point.point_id
        missions = _score_missions(
            claim_id,
            local_slots,
            remote_slots,
            q_effective=q_effective,
            section=point.section,
        )
        assigned, relevance = (
            _assign_seeds(packet.topology_seeds, point, frame, seed_usage)
            if enable_topology and is_prior_art_direction_claim(point)
            else ([], 0.0)
        )
        for seed in assigned:
            seed_usage[seed.work_id] = seed_usage.get(seed.work_id, 0) + 1
        claim_citations_available = bool(frame and frame.citation_seed_ids)
        if (
            enable_topology
            and claim_citations_available
            and is_prior_art_direction_claim(point)
        ):
            missions.append(_citation_topology_mission(claim_id))
            relevance = max(relevance, 1.0)
        if assigned:
            missions.extend(
                _topology_missions(
                    claim_id,
                    assigned,
                    section=point.section,
                    q_effective=q_effective,
                )
            )
        if analog_seeds:
            missions.extend(_analog_missions(claim_id, analog_seeds))
        if tension is not None:
            missions.append(_tension_mission(claim_id, tension))
        return ClaimGuidance(
            review_point_id=point.point_id,
            claim_id=claim_id,
            claim_relevance=relevance,
            allocated_local_query_slots=local_slots,
            allocated_remote_query_slots=remote_slots,
            missions=_deduplicate_missions(missions),
            analog_seeds=list(analog_seeds),
        )


def _score_missions(
    claim_id: str,
    local: int,
    remote: int,
    *,
    q_effective: float,
    section: str,
) -> list[RetrievalMission]:
    local_mission: RetrievalMission | None = None
    remote_mission: RetrievalMission | None = None
    if local:
        local_mission = _mission(
            claim_id,
            "local_nearest_antecedent",
            "score",
            "neutral",
            ["author_terminology", "object_problem"],
            "none",
            "local_slots_exhausted_or_direct_antecedent",
        )
    if remote:
        remote_mission = _mission(
            claim_id,
            "remote_mechanism_analogue",
            "score",
            "neutral",
            ["mechanism_outcome", "purpose_semantic"],
            "none",
            "remote_slots_exhausted_or_comparable_relation",
        )
    ordered = (
        [remote_mission, local_mission]
        if remote > local
        else [local_mission, remote_mission]
    )
    # The legacy contrastive query is intentionally absent.  In the dev10
    # audit it consumed 21 matched-budget slots and yielded no verified
    # relation.  Local/remote scientific query families already provide the
    # terminology-free comparison geometry without a separate request.
    return [mission for mission in ordered if mission is not None]


def _assign_seeds(
    seeds: Sequence[TopologySeed],
    point: CanonicalReviewPoint,
    frame: ScientificSearchFrame | None,
    seed_usage: Mapping[str, int],
) -> tuple[list[TopologySeed], float]:
    groups = _frame_groups(frame)
    point_tokens = _tokens(point.proposition)
    scientific_tokens = set().union(*groups) if groups else point_tokens
    claim_citation_ids = set(frame.citation_seed_ids) if frame is not None else set()
    ranked: list[tuple[float, int, str, TopologySeed]] = []
    for seed in seeds:
        if seed_usage.get(seed.work_id, 0) >= 1:
            continue
        seed_tokens = _tokens(seed.title)
        distinctive_overlap = seed_tokens & scientific_tokens
        overlap = len(distinctive_overlap)
        matched_groups = sum(len(seed_tokens & group) >= 1 for group in groups if group)
        group_ratio = matched_groups / len(groups) if groups else 0.0
        citation_overlap = len(claim_citation_ids & set(seed.shared_reference_ids))
        citation_anchor = min(1.0, citation_overlap / 2.0)
        relevance = min(
            1.0,
            0.55 * min(1.0, overlap / 4.0)
            + 0.25 * group_ratio
            + 0.20 * citation_anchor,
        )
        # Topology is only an entrance.  A seed must align with at least two
        # independent claim-frame groups and three distinctive title tokens
        # before it is allowed to consume even a direct-fetch opportunity.
        claim_linked = (
            citation_overlap >= 1 and overlap >= 2 and matched_groups >= 1
            if claim_citation_ids
            else overlap >= 3 and matched_groups >= 2
        )
        if claim_linked and seed.shared_reference_count >= 2:
            ranked.append((relevance, seed.shared_reference_count, seed.work_id, seed))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    selected = [item[3] for item in ranked[:1]]
    return selected, (ranked[0][0] if ranked else 0.0)


def _topology_missions(
    claim_id: str,
    seeds: Sequence[TopologySeed],
    *,
    section: str,
    q_effective: float,
) -> list[RetrievalMission]:
    if not seeds:
        return []
    # A title-directed provider search explores the seed's local literature
    # neighborhood inside the same logical request.  The dev10 audit found no
    # verified-relation yield from an additional one-hop traversal, so the
    # default policy spends that matched slot on claim-level search breadth.
    traversal: Traversal = "none"
    orientation: MissionOrientation = "neutral"
    mission = _mission(
        claim_id,
        "topology_seed",
        "topology",
        orientation,
        ["graph_seed"],
        traversal,
        "two_seeds_or_one_verified_relation",
    )
    return [
        mission.model_copy(update={"seed_work_ids": [seed.work_id for seed in seeds]})
    ]


def _citation_topology_mission(claim_id: str) -> RetrievalMission:
    """Use exact manuscript citations as the first claim-verification entrance."""

    return _mission(
        claim_id,
        "topology_seed",
        "topology",
        "neutral",
        ["author_citation"],
        "none",
        "one_semantically_admitted_claim_citation_or_citation_pool_exhausted",
    )


def _analog_missions(
    claim_id: str, seeds: Sequence[AnalogSeed]
) -> list[RetrievalMission]:
    return [
        _mission(
            claim_id,
            "topology_seed",
            "calibration",
            "neutral",
            ["graph_seed"],
            "references",
            "one_cutoff_safe_hgb_analog_or_analog_pool_exhausted",
        ).model_copy(update={"seed_work_ids": [seed.work_id for seed in seeds]})
    ]


def _tension_mission(claim_id: str, tension: CalibrationTension) -> RetrievalMission:
    effect = tension.review_effect
    role = (
        "object_problem"
        if effect == "antecedent_attribution_check"
        else "mechanism_outcome"
    )
    return _mission(
        claim_id,
        (
            "local_nearest_antecedent"
            if role == "object_problem"
            else "remote_mechanism_analogue"
        ),
        "calibration",
        "neutral",
        [role],
        "none",
        "calibration_tension_checked",
    )


def _frame_terms(frame: ScientificSearchFrame | None) -> list[str]:
    if frame is None:
        return []
    values = [
        *frame.target_object,
        *frame.task_problem,
        *frame.mechanism,
        *frame.outcome_observable,
        *frame.author_terms,
    ]
    return [token for value in values for token in _tokens(value)][:24]


def _mission(
    claim_id: str,
    mission_type: MissionType,
    origin: MissionOrigin,
    orientation: MissionOrientation,
    query_roles: list[str],
    traversal: Traversal,
    stop_rule: str,
) -> RetrievalMission:
    identity = f"{claim_id}|{mission_type}|{origin}|{','.join(query_roles)}|{traversal}"
    mission_id = "GM-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:18]
    return RetrievalMission(
        mission_id=mission_id,
        mission_type=mission_type,
        origin=origin,
        target_claim_id=claim_id,
        orientation=orientation,
        query_roles=query_roles,
        traversal=traversal,
        expected_relation_types=[
            "DIRECT_ANTECEDENT",
            "PARTIAL_ANTECEDENT",
            "EXTENSION",
            "PARALLEL",
            "SUPPORT",
        ],
        stop_rule=stop_rule,
    )


def _weighted_claim_slots(total: int, weights: Sequence[int]) -> list[int]:
    """Allocate at most three slots per claim without point-id lottery."""
    count = len(weights)
    if count == 0:
        return []
    if total <= count:
        return [int(index < total) for index in range(count)]
    slots = [1] * count
    remaining = total - count
    weight_sum = max(sum(max(1, weight) for weight in weights), 1)
    ideals = [remaining * max(1, weight) / weight_sum for weight in weights]
    extras = [min(2, int(value)) for value in ideals]
    slots = [slot + extra for slot, extra in zip(slots, extras)]
    missing = total - sum(slots)
    order = sorted(
        range(count),
        key=lambda index: (
            -(ideals[index] - extras[index]),
            -max(1, weights[index]),
            index,
        ),
    )
    while missing > 0:
        progressed = False
        for index in order:
            if slots[index] >= 3:
                continue
            slots[index] += 1
            missing -= 1
            progressed = True
            if missing == 0:
                break
        if not progressed:
            break
    return slots


def _allocate_geometry(
    local: int, remote: int, weights: Sequence[int]
) -> tuple[list[int], list[int]]:
    """Apportion exact executable slots without favoring the first claim."""
    count = len(weights)
    if count == 0:
        return [], []
    total = local + remote
    claim_totals = _weighted_claim_slots(total, weights)
    ideals = [slots * remote / total for slots in claim_totals]
    remote_alloc = [
        min(slots, int(ideal)) for slots, ideal in zip(claim_totals, ideals)
    ]
    missing = remote - sum(remote_alloc)
    order = sorted(
        range(count),
        key=lambda index: (-(ideals[index] - remote_alloc[index]), index),
    )
    while missing > 0:
        progressed = False
        for index in order:
            if remote_alloc[index] >= claim_totals[index]:
                continue
            remote_alloc[index] += 1
            missing -= 1
            progressed = True
            if missing == 0:
                break
        if not progressed:
            break
    local_alloc = [
        slots - remote_slots for slots, remote_slots in zip(claim_totals, remote_alloc)
    ]
    return local_alloc, remote_alloc


def is_prior_art_direction_claim(point: CanonicalReviewPoint) -> bool:
    """Return whether external antecedence may legitimately change direction."""
    if point.section != "novelty_limit":
        return True
    return bool(PRIOR_ART_DIRECTION_RE.search(point.proposition))


def is_absolute_priority_claim(point: CanonicalReviewPoint) -> bool:
    """Return whether one verified counterexample can falsify the wording."""
    return bool(ABSOLUTE_PRIORITY_RE.search(point.proposition))


def _guidance_weight(point: CanonicalReviewPoint) -> int:
    if point.section == "novelty_limit" and is_prior_art_direction_claim(point):
        return 3
    if point.section == "novelty_support":
        return 2
    return 1


def _guidance_sort_key(point: CanonicalReviewPoint) -> tuple[int, int, str]:
    severity = {"critical": 3, "major": 2, "minor": 1, "none": 0}
    return (
        -_guidance_weight(point),
        -severity.get(point.severity.value, 0),
        point.point_id,
    )


def _frame_groups(frame: ScientificSearchFrame | None) -> list[set[str]]:
    if frame is None:
        return []
    names = (
        "target_object",
        "task_problem",
        "mechanism",
        "population_input",
        "outcome_observable",
        "comparator",
    )
    return [
        tokens for name in names if (tokens := _tokens(" ".join(getattr(frame, name))))
    ]


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9-]+", value.casefold())
        if len(token) > 2 and token not in GENERIC_SCIENTIFIC_TOKENS
    }


def _deduplicate_missions(
    missions: Sequence[RetrievalMission],
) -> list[RetrievalMission]:
    return list({mission.mission_id: mission for mission in missions}.values())


__all__ = [
    "GRAPH_GUIDANCE_POLICY_VERSION",
    "GraphGuidancePlanner",
    "is_absolute_priority_claim",
    "is_graph_guidance_target",
    "is_prior_art_direction_claim",
    "score_controller",
]
