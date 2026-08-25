"""Deterministic post-fusion Graph guidance under matched resource caps."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Literal

from .contracts import ScientificSearchFrame
from .graph_prior_contracts import (
    GraphClaimGuidanceV1,
    GraphGuidancePlanV1,
    GraphMissionV1,
    GraphResourceCapsV1,
    GraphRuntimePacketV1,
    GraphTopologySeedV1,
)
from .review_contracts import CanonicalReviewPoint, ReviewAspect, ReviewStateV3

STRUCTURAL_DIVERSITY = {"EF0017", "EF0309", "EF0312", "EF0315", "EF0318"}
HISTORICAL_DEPTH = "EF0052"
TERMINOLOGY_EMERGENCE = "EF0240"
GRAPH_GUIDANCE_POLICY_VERSION = "score_profile_topology_v22_claim_aligned"
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
    "terminology_free_counterfactual",
    "reference_structure_diversity",
    "historical_lineage",
    "recent_direct_predecessor",
    "terminology_lineage",
    "topology_seed",
    "remote_rescue",
]
MissionOrigin = Literal["score", "profile", "topology", "rescue"]
MissionOrientation = Literal["falsification", "rescue", "neutral"]
Traversal = Literal["none", "references", "citations"]


def score_controller(
    packet: GraphRuntimePacketV1, query_slots: int = 8
) -> tuple[int, int, float]:
    """Map ASPR continuously to local/remote geometry without a novelty axis."""
    reliability = 1.0 - len(set(packet.missing_feature_ids)) / 16.0
    reliability = min(1.0, max(0.0, reliability))
    q = packet.score_0_100 / 100.0
    q_effective = 0.5 + reliability * (q - 0.5)
    remote = round(query_slots * (0.25 + 0.50 * q_effective))
    remote = min(query_slots - 1, max(1, remote))
    return query_slots - remote, remote, q_effective


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
        resource_caps: GraphResourceCapsV1 | None = None,
        policy_version: str = GRAPH_GUIDANCE_POLICY_VERSION,
    ) -> None:
        self.resource_caps = resource_caps or GraphResourceCapsV1()
        self.policy_version = policy_version

    def plan(
        self,
        state: ReviewStateV3,
        *,
        search_frames: Mapping[str, ScientificSearchFrame] | None = None,
        enable_profile: bool = True,
        enable_topology: bool = True,
    ) -> GraphGuidancePlanV1:
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
        local, remote, q_effective = score_controller(packet, executable_slots)
        weights = [_guidance_weight(point) for point in points]
        local_alloc, remote_alloc = _allocate_geometry(local, remote, weights)
        frames = search_frames or {}
        seed_usage: dict[str, int] = {}
        profile_count = len(_profile_missions(packet, "PROFILE-PROBE"))
        profile_assignment_count = min(
            profile_count,
            (1 if len(points) == 1 else max(0, len(points) - 1)),
        )
        guidance = []
        topology_claims_used = 0
        for index, point in enumerate(points):
            claim_guidance = self._claim_guidance(
                point,
                packet,
                frames.get(point.point_id),
                local_alloc[index],
                remote_alloc[index],
                q_effective,
                enable_profile,
                enable_topology and topology_claims_used < MAX_TOPOLOGY_GUIDED_CLAIMS,
                seed_usage,
                profile_mission_index=(
                    index if index < profile_assignment_count else None
                ),
            )
            if any(mission.origin == "topology" for mission in claim_guidance.missions):
                topology_claims_used += 1
            guidance.append(claim_guidance)
        return GraphGuidancePlanV1(
            paper_id=state.paper_id,
            policy_version=self.policy_version,
            source_packet_evidence_key=state.graph_result_evidence_key or "G:RESULT",
            controller_state={
                "score_0_100": packet.score_0_100,
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
        packet: GraphRuntimePacketV1,
        frame: ScientificSearchFrame | None,
        local_slots: int,
        remote_slots: int,
        q_effective: float,
        enable_profile: bool,
        enable_topology: bool,
        seed_usage: dict[str, int],
        profile_mission_index: int | None,
    ) -> GraphClaimGuidanceV1:
        claim_id = point.novelty_claim_id or point.point_id
        missions = _score_missions(
            claim_id,
            local_slots,
            remote_slots,
            q_effective=q_effective,
            section=point.section,
        )
        if enable_profile and profile_mission_index is not None:
            profile_missions = _profile_missions(packet, claim_id)
            if profile_mission_index < len(profile_missions):
                missions.append(profile_missions[profile_mission_index])
        assigned, relevance = (
            _assign_seeds(packet.topology_seeds, point, frame, seed_usage)
            if enable_topology and is_prior_art_direction_claim(point)
            else ([], 0.0)
        )
        for seed in assigned:
            seed_usage[seed.work_id] = seed_usage.get(seed.work_id, 0) + 1
        if assigned:
            missions.extend(
                _topology_missions(
                    claim_id,
                    assigned,
                    section=point.section,
                    q_effective=q_effective,
                )
            )
        return GraphClaimGuidanceV1(
            review_point_id=point.point_id,
            claim_id=claim_id,
            claim_relevance=relevance,
            allocated_local_query_slots=local_slots,
            allocated_remote_query_slots=remote_slots,
            missions=_deduplicate_missions(missions),
        )


def _score_missions(
    claim_id: str,
    local: int,
    remote: int,
    *,
    q_effective: float,
    section: str,
) -> list[GraphMissionV1]:
    local_mission: GraphMissionV1 | None = None
    remote_mission: GraphMissionV1 | None = None
    if local:
        local_mission = _mission(
            claim_id,
            "local_nearest_antecedent",
            "score",
            "falsification",
            ["author_terminology", "object_problem"],
            "none",
            "local_slots_exhausted_or_direct_antecedent",
        )
    if remote:
        rescue = q_effective >= 0.5 and section == "novelty_support"
        remote_mission = _mission(
            claim_id,
            "remote_rescue" if rescue else "remote_mechanism_analogue",
            "rescue" if rescue else "score",
            "rescue" if rescue else "neutral",
            (
                ["purpose_semantic", "mechanism_outcome"]
                if rescue
                else ["mechanism_outcome", "purpose_semantic"]
            ),
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


def _profile_missions(
    packet: GraphRuntimePacketV1, claim_id: str
) -> list[GraphMissionV1]:
    bands = packet.historical_bands
    missions: list[GraphMissionV1] = []
    diversity_bands = {bands.get(name, "").casefold() for name in STRUCTURAL_DIVERSITY}
    if diversity_bands & {"high", "upper", "above", "high_band", "high_extreme"}:
        missions.append(
            _mission(
                claim_id,
                "reference_structure_diversity",
                "profile",
                "falsification",
                ["mechanism_outcome"],
                "none",
                "one_comparable_relation",
            )
        )
    elif diversity_bands & {"low", "lower", "below", "low_band", "low_extreme"}:
        missions.append(
            _mission(
                claim_id,
                "reference_structure_diversity",
                "profile",
                "neutral",
                ["object_problem"],
                "none",
                "one_local_predecessor",
            )
        )
    depth = bands.get(HISTORICAL_DEPTH, "").casefold()
    if depth in {"high", "upper", "above", "high_band", "high_extreme"}:
        missions.append(
            _mission(
                claim_id,
                "historical_lineage",
                "profile",
                "neutral",
                ["author_citation"],
                "references",
                "one_historical_lineage",
            )
        )
    elif depth in {"low", "lower", "below", "low_band", "low_extreme"}:
        missions.append(
            _mission(
                claim_id,
                "recent_direct_predecessor",
                "profile",
                "falsification",
                ["object_problem"],
                "none",
                "one_recent_predecessor",
            )
        )
    emergence = bands.get(TERMINOLOGY_EMERGENCE, "").casefold()
    if emergence in {"high", "upper", "above", "high_band", "high_extreme"}:
        missions.append(
            _mission(
                claim_id,
                "terminology_lineage",
                "profile",
                "neutral",
                ["author_terminology"],
                "references",
                "one_terminology_lineage",
            )
        )
    return missions


def _assign_seeds(
    seeds: Sequence[GraphTopologySeedV1],
    point: CanonicalReviewPoint,
    frame: ScientificSearchFrame | None,
    seed_usage: Mapping[str, int],
) -> tuple[list[GraphTopologySeedV1], float]:
    groups = _frame_groups(frame)
    point_tokens = _tokens(point.proposition)
    scientific_tokens = set().union(*groups) if groups else point_tokens
    ranked: list[tuple[float, int, str, GraphTopologySeedV1]] = []
    for seed in seeds:
        if seed_usage.get(seed.work_id, 0) >= 1:
            continue
        seed_tokens = _tokens(seed.title)
        distinctive_overlap = seed_tokens & scientific_tokens
        overlap = len(distinctive_overlap)
        matched_groups = sum(len(seed_tokens & group) >= 1 for group in groups if group)
        group_ratio = matched_groups / len(groups) if groups else 0.0
        relevance = min(1.0, 0.75 * min(1.0, overlap / 4.0) + 0.25 * group_ratio)
        # One generic scientific word was enough to saturate the old score and
        # routed unrelated graph anchors into every claim.  Direct fetches now
        # require two claim-specific lexical anchors; semantic provider queries
        # remain the fallback for terminology mismatch.
        if overlap >= 2:
            ranked.append((relevance, seed.shared_reference_count, seed.work_id, seed))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    selected = [item[3] for item in ranked[:1]]
    return selected, (ranked[0][0] if ranked else 0.0)


def _topology_missions(
    claim_id: str,
    seeds: Sequence[GraphTopologySeedV1],
    *,
    section: str,
    q_effective: float,
) -> list[GraphMissionV1]:
    if not seeds:
        return []
    # Falsification follows an anchor's antecedent chain; only a supporting
    # claim with sufficiently high prospective diffusion follows forward
    # citations for rescue.  Score changes search geometry, never polarity.
    # A title-directed provider search explores the seed's local literature
    # neighborhood inside the same logical request.  The dev10 audit found no
    # verified-relation yield from an additional one-hop traversal, so the
    # default policy spends that matched slot on claim-level search breadth.
    traversal: Traversal = "none"
    orientation: MissionOrientation = "neutral"
    if section == "novelty_support" and q_effective >= 0.5:
        orientation = "rescue"
    elif section == "novelty_limit":
        orientation = "falsification"
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


def _mission(
    claim_id: str,
    mission_type: MissionType,
    origin: MissionOrigin,
    orientation: MissionOrientation,
    query_roles: list[str],
    traversal: Traversal,
    stop_rule: str,
) -> GraphMissionV1:
    identity = f"{claim_id}|{mission_type}|{origin}|{','.join(query_roles)}|{traversal}"
    mission_id = "GM-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:18]
    return GraphMissionV1(
        mission_id=mission_id,
        mission_type=mission_type,
        origin=origin,
        target_claim_id=claim_id,
        orientation=orientation,
        query_roles=query_roles,
        traversal=traversal,
        expected_relation_types=(
            ["EXTENSION", "PARALLEL", "SUPPORT"]
            if orientation == "rescue"
            else (
                ["DIRECT_ANTECEDENT", "PARTIAL_ANTECEDENT", "BUILDING_BLOCK"]
                if orientation == "falsification"
                else [
                    "DIRECT_ANTECEDENT",
                    "PARTIAL_ANTECEDENT",
                    "EXTENSION",
                    "PARALLEL",
                    "SUPPORT",
                ]
            )
        ),
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


def _deduplicate_missions(missions: Sequence[GraphMissionV1]) -> list[GraphMissionV1]:
    return list({mission.mission_id: mission for mission in missions}.values())


__all__ = [
    "GRAPH_GUIDANCE_POLICY_VERSION",
    "GraphGuidancePlanner",
    "is_absolute_priority_claim",
    "is_graph_guidance_target",
    "is_prior_art_direction_claim",
    "score_controller",
]
