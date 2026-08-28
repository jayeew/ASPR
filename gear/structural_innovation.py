"""Evidence-gated structural innovation contracts and deterministic fusion."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from .claim_attribution import (
    deterministic_t0_attribution,
    learned_t0_attribution,
)
from .contracts import ClaimStrength, ClaimType, PaperIR, RelationLabel
from .graph_prior_contracts import (
    ClaimAttributionAudit,
    ClaimGraphPrior,
    ClaimInventoryEntry,
    GraphRuntimePacket,
    GraphSignalBundle,
    ImpactPathwayCard,
    StructuralInnovationCard,
)
from .review_contracts import CanonicalReviewPoint, PointValidationStatus, ReviewState
from .trace import EvidenceStore

_CENTRALITY = {
    ClaimStrength.WEAK: 0.35,
    ClaimStrength.MODERATE: 0.65,
    ClaimStrength.STRONG: 1.0,
}
_TYPE_WEIGHT = {
    ClaimType.NOVELTY: 1.0,
    ClaimType.METHOD: 0.95,
    ClaimType.CAUSAL: 0.9,
    ClaimType.RESULT: 0.8,
    ClaimType.SIGNIFICANCE: 0.75,
    ClaimType.SCOPE: 0.55,
}
ForecastRole = Literal[
    "substantive_innovation", "t0_potential", "opportunity", "context", "unknown"
]
PathwayType = Literal[
    "local_method_adoption",
    "cross_field_bridge",
    "reusable_resource",
    "platform_scaling",
    "mechanism_transfer",
    "unspecified",
]
ProfileClass = Literal[
    "transformative_broad",
    "niche_or_delayed_foundational",
    "scalable_incremental",
    "limited_structural_innovation",
    "insufficient_evidence",
]


def build_claim_inventory(paper_ir: PaperIR) -> list[ClaimInventoryEntry]:
    """Freeze the extractor's graph-blind claims and their manuscript spans."""
    inventory: list[ClaimInventoryEntry] = []
    span_ids = set(paper_ir.span_map())
    for claim in paper_ir.claims:
        if claim.span_id not in span_ids:
            continue
        centrality = _CENTRALITY[claim.strength] * _TYPE_WEIGHT[claim.claim_type]
        inventory.append(
            ClaimInventoryEntry(
                claim_id=claim.claim_id,
                claim_type=claim.claim_type.value,
                text=claim.text,
                manuscript_evidence_keys=[f"P:{claim.span_id}"],
                centrality=min(1.0, centrality),
            )
        )
    return inventory


def build_graph_signal_bundle(packet: GraphRuntimePacket) -> GraphSignalBundle:
    """Reliability-shrink a forecast without interpreting percentile as probability."""
    forecast = packet.forecast
    expected = forecast.expected_diffusion or 0.0
    structural_available = forecast.structural_heads_status == "available"
    diffusion_signal = (
        forecast.excess_diffusion
        if structural_available and forecast.excess_diffusion is not None
        else expected
    )
    diagnostics = list(forecast.diagnostics)
    if forecast.field_year_base is None:
        base = 0.0
        diagnostics.append("field_year_base_unavailable:conservative_zero_used")
    else:
        base = forecast.field_year_base
    reliability = (
        forecast.feature_coverage
        * forecast.ood_reliability
        * forecast.calibration_reliability
    )
    if forecast.status != "available":
        reliability = 0.0
        diagnostics.append("forecast_unavailable:shrunk_to_base")
    if forecast.prediction_interval_width is not None:
        reliability *= 1.0 - forecast.prediction_interval_width
    shrunk = base + reliability * (diffusion_signal - base)
    structural_share, opportunity_share = _anatomy_shares(packet)
    return GraphSignalBundle(
        paper_id=packet.paper_id,
        expected_diffusion=expected,
        uptake_probability=forecast.uptake_probability or 0.0,
        conditional_diffusion=forecast.conditional_diffusion or 0.0,
        excess_diffusion=(forecast.excess_diffusion if structural_available else None),
        field_year_base=base,
        reliability=reliability,
        shrunk_diffusion=min(1.0, max(0.0, shrunk)),
        perturbation_potential=(
            forecast.perturbation_potential if structural_available else None
        ),
        perturbation_components=(
            forecast.perturbation_components if structural_available else {}
        ),
        percentile_display=forecast.prospective_5y_diffusion_percentile,
        structural_contribution_share=structural_share,
        opportunity_context_share=opportunity_share,
        structural_heads_status=forecast.structural_heads_status,
        limited=(
            forecast.status != "available"
            or forecast.structural_heads_status != "available"
            or reliability < 0.5
        ),
        diagnostics=diagnostics,
    )


def attribute_graph_signal(
    inventory: list[ClaimInventoryEntry],
    bundle: GraphSignalBundle,
    packet: GraphRuntimePacket,
) -> list[ClaimGraphPrior]:
    """Run the phase-one deterministic T0 baseline for API compatibility."""
    priors, _ = deterministic_t0_attribution(inventory, bundle, packet)
    return priors


def attribute_graph_signal_with_audit(
    inventory: list[ClaimInventoryEntry],
    bundle: GraphSignalBundle,
    packet: GraphRuntimePacket,
    *,
    mode: Literal["deterministic_t0", "learned_t0"] = "deterministic_t0",
    learned_manifest: Path | None = None,
) -> tuple[list[ClaimGraphPrior], ClaimAttributionAudit]:
    """Allocate Graph signal using an explicitly identified T0 contract."""
    if mode == "learned_t0":
        return learned_t0_attribution(inventory, bundle, packet, learned_manifest)
    return deterministic_t0_attribution(inventory, bundle, packet)


def build_impact_pathways(
    state: ReviewState, store: EvidenceStore
) -> list[ImpactPathwayCard]:
    """Verify pathway hypotheses only when manuscript and prior-work traces exist."""
    inventory = {item.claim_id: item for item in state.claim_inventory}
    cards: list[ImpactPathwayCard] = []
    for prior in state.claim_graph_priors:
        item = inventory[prior.claim_id]
        points = _points_for_claim(state, item)
        prior_keys = list(
            dict.fromkeys(
                key for point in points for key in point.relation_evidence_keys
            )
        )
        valid_prior = [key for key in prior_keys if store.get(key) is not None]
        manuscript = [
            key for key in item.manuscript_evidence_keys if store.get(key) is not None
        ]
        verified = bool(
            manuscript and valid_prior and prior.pathway_hypothesis != "unspecified"
        )
        plausibility = prior.confidence if verified else 0.0
        identity = (
            f"{prior.claim_id}|{prior.pathway_hypothesis}|{'|'.join(valid_prior)}"
        )
        cards.append(
            ImpactPathwayCard(
                pathway_id="IP-" + hashlib.sha256(identity.encode()).hexdigest()[:18],
                claim_id=prior.claim_id,
                pathway_type=prior.pathway_hypothesis,
                manuscript_evidence_ids=manuscript,
                prior_work_evidence_ids=valid_prior,
                pathway_plausibility=plausibility,
                attribution_confidence=prior.confidence,
                verified=verified,
            )
        )
    return cards


def fuse_structural_innovation(
    state: ReviewState,
    store: EvidenceStore,
    *,
    epsilon: float = 0.1,
    alpha: float = 1.0,
    beta: float = 1.0,
    eta: float = 1.0,
    gamma: float = 0.5,
) -> ReviewState:
    """Apply the registered non-compensatory, monotone claim-level fusion."""
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")
    if min(alpha, beta, eta, gamma) < 0.0 or beta <= 0.0:
        raise ValueError("fusion exponents violate the monotonicity contract")
    if state.graph_signal_bundle is None:
        return state
    inventory = {item.claim_id: item for item in state.claim_inventory}
    pathways = {card.claim_id: card for card in state.impact_pathway_cards}
    cards: list[StructuralInnovationCard] = []
    for prior in state.claim_graph_priors:
        item = inventory[prior.claim_id]
        pathway = pathways.get(prior.claim_id)
        variables = _evidence_variables(state, item, store, pathway)
        gate = (
            variables["validity"]
            * variables["coverage"]
            * (1.0 - variables["antecedent"])
            * variables["residual"]
        )
        diffusion = prior.diffusion_prior
        perturbation = prior.perturbation_prior
        score = gate**alpha
        score *= (epsilon + (1.0 - epsilon) * diffusion) ** beta
        if perturbation is not None:
            score *= (epsilon + (1.0 - epsilon) * perturbation) ** eta
        score *= variables["mechanism"] ** gamma
        score = min(1.0, max(0.0, score))
        profile = _profile(gate, diffusion, variables["coverage"])
        evidence_keys = list(
            dict.fromkeys(
                [
                    *item.manuscript_evidence_keys,
                    *(
                        key
                        for point in variables["points"]
                        for key in point.relation_evidence_keys
                    ),
                    *(
                        key
                        for point in variables["points"]
                        for key in point.coverage_evidence_keys
                    ),
                ]
            )
        )
        cards.append(
            StructuralInnovationCard(
                claim_id=item.claim_id,
                manuscript_validity=variables["validity"],
                antecedent_risk=variables["antecedent"],
                residual_novelty=variables["residual"],
                evidence_coverage=variables["coverage"],
                mechanism_validity=variables["mechanism"],
                evidence_gate=gate,
                diffusion_potential=diffusion,
                perturbation_potential=perturbation,
                structural_innovation_score=score,
                uncertainty=1.0
                - variables["validity"]
                * variables["coverage"]
                * state.graph_signal_bundle.reliability,
                profile_class=profile,
                evidence_keys=evidence_keys,
            )
        )
    paper_score = _noisy_or(cards, state.claim_inventory)
    paper_profile = _paper_profile(cards, paper_score)
    return state.model_copy(
        update={
            "structural_innovation_cards": cards,
            "paper_structural_innovation_score": paper_score,
            "structural_innovation_profile": paper_profile,
        }
    )


def _anatomy_shares(packet: GraphRuntimePacket) -> tuple[float | None, float | None]:
    anatomy = packet.forecast_anatomy
    if anatomy is None:
        return None, None
    totals = {
        role: 0.0
        for role in ("substantive_innovation", "t0_potential", "opportunity", "context")
    }
    for contributions in (
        anatomy.uptake_role_contributions,
        anatomy.conditional_role_contributions,
    ):
        for role in totals:
            totals[role] += abs(float(contributions.get(role, 0.0)))
    denominator = sum(totals.values())
    if denominator <= 0.0:
        return 0.0, 0.0
    structural = (
        totals["substantive_innovation"] + totals["t0_potential"]
    ) / denominator
    return structural, 1.0 - structural


def _dominant_role(packet: GraphRuntimePacket) -> ForecastRole:
    anatomy = packet.forecast_anatomy
    if anatomy is None:
        return "unknown"
    scores: dict[ForecastRole, float] = {}
    roles: tuple[ForecastRole, ...] = (
        "substantive_innovation",
        "t0_potential",
        "opportunity",
        "context",
    )
    for role in roles:
        scores[role] = abs(anatomy.uptake_role_contributions.get(role, 0.0)) + abs(
            anatomy.conditional_role_contributions.get(role, 0.0)
        )
    return (
        max(scores.items(), key=lambda item: item[1])[0]
        if any(scores.values())
        else "unknown"
    )


def _pathway_hypothesis(item: ClaimInventoryEntry) -> PathwayType:
    text = item.text.casefold()
    if any(token in text for token in ("dataset", "resource", "benchmark", "database")):
        return "reusable_resource"
    if any(token in text for token in ("platform", "framework", "pipeline", "system")):
        return "platform_scaling"
    if any(
        token in text for token in ("across", "transfer", "generaliz", "cross-field")
    ):
        return "cross_field_bridge"
    if item.claim_type in {ClaimType.METHOD.value, ClaimType.NOVELTY.value}:
        return "local_method_adoption"
    if item.claim_type == ClaimType.CAUSAL.value:
        return "mechanism_transfer"
    return "unspecified"


def _points_for_claim(
    state: ReviewState, item: ClaimInventoryEntry
) -> list[CanonicalReviewPoint]:
    keys = set(item.manuscript_evidence_keys)
    return [
        point
        for point in state.canonical_points.values()
        if point.retained
        and (
            point.novelty_claim_id == item.claim_id
            or point.contribution_id == item.claim_id
            or bool(keys.intersection(point.paper_evidence_keys))
        )
    ]


def _evidence_variables(
    state: ReviewState,
    item: ClaimInventoryEntry,
    store: EvidenceStore,
    pathway: ImpactPathwayCard | None,
) -> dict[str, Any]:
    points = _points_for_claim(state, item)
    manuscript_validity = float(
        bool(item.manuscript_evidence_keys)
        and all(store.get(key) is not None for key in item.manuscript_evidence_keys)
    )
    coverage = max((_point_coverage(point, store) for point in points), default=0.0)
    antecedent = max(
        (_point_antecedent_risk(point, store) for point in points), default=0.0
    )
    residual = _residual_novelty(points, antecedent, coverage)
    validated = any(
        point.validation_status
        in {PointValidationStatus.VALIDATED, PointValidationStatus.EXTERNALLY_VALIDATED}
        for point in points
    )
    mechanism = (
        1.0 if pathway is not None and pathway.verified else (0.7 if validated else 0.5)
    )
    return {
        "validity": manuscript_validity,
        "coverage": coverage,
        "antecedent": antecedent,
        "residual": residual,
        "mechanism": mechanism,
        "points": points,
    }


def _point_coverage(point: CanonicalReviewPoint, store: EvidenceStore) -> float:
    values: list[float] = []
    for key in point.coverage_evidence_keys:
        record = store.get(key)
        payload = record.payload if record is not None else {}
        if payload.get("service_failed"):
            values.append(0.0)
            continue
        required = len(payload.get("required_query_roles") or [])
        completed = len(payload.get("completed_query_roles") or [])
        ratio = completed / required if required else 0.0
        if payload.get("coverage_sufficient"):
            ratio = 1.0
        values.append(min(1.0, ratio))
    return max(values, default=0.0)


def _point_antecedent_risk(point: CanonicalReviewPoint, store: EvidenceStore) -> float:
    risk = 0.0
    for key in point.relation_evidence_keys:
        record = store.get(key)
        payload = record.payload if record is not None else {}
        if not payload.get("temporal_valid") or not payload.get(
            "independent_verification_passed"
        ):
            continue
        coverage = float(payload.get("essential_facet_coverage") or 0.0)
        label = payload.get("relation_label")
        if label == RelationLabel.DIRECT_ANTECEDENT.value:
            risk = max(risk, coverage)
        elif label in {
            RelationLabel.PARTIAL_ANTECEDENT.value,
            RelationLabel.BUILDING_BLOCK.value,
        }:
            risk = max(risk, 0.5 * coverage)
    return min(1.0, risk)


def _residual_novelty(
    points: Iterable[CanonicalReviewPoint], antecedent: float, coverage: float
) -> float:
    resolutions = {point.novelty_resolution for point in points}
    if "antecedent_found" in resolutions and antecedent >= 0.95:
        return 0.0
    if "incremental_or_parallel" in resolutions:
        return min(0.65, 1.0 - antecedent)
    if "bounded_no_antecedent" in resolutions and coverage > 0.0:
        return 1.0
    if coverage <= 0.0:
        return 0.0
    return max(0.0, 1.0 - antecedent)


def _profile(gate: float, diffusion: float, coverage: float) -> ProfileClass:
    if coverage <= 0.0:
        return "insufficient_evidence"
    novelty_high = gate >= 0.5
    diffusion_high = diffusion >= 0.5
    if novelty_high and diffusion_high:
        return "transformative_broad"
    if novelty_high:
        return "niche_or_delayed_foundational"
    if diffusion_high:
        return "scalable_incremental"
    return "limited_structural_innovation"


def _noisy_or(
    cards: list[StructuralInnovationCard], inventory: list[ClaimInventoryEntry]
) -> float:
    centrality = {item.claim_id: item.centrality for item in inventory}
    selected = sorted(
        cards,
        key=lambda card: (-centrality.get(card.claim_id, 0.0), card.claim_id),
    )[:3]
    total = sum(centrality.get(card.claim_id, 0.0) for card in selected)
    if not selected or total <= 0.0:
        return 0.0
    survival = 1.0
    for card in selected:
        weight = centrality.get(card.claim_id, 0.0) / total
        survival *= 1.0 - weight * card.structural_innovation_score
    return min(1.0, max(0.0, 1.0 - survival))


def _paper_profile(
    cards: list[StructuralInnovationCard], paper_score: float
) -> str | None:
    if not cards:
        return None
    if all(card.profile_class == "insufficient_evidence" for card in cards):
        return "insufficient_evidence"
    representative = max(cards, key=lambda card: card.structural_innovation_score)
    if paper_score <= 0.0 and representative.evidence_coverage <= 0.0:
        return "insufficient_evidence"
    return representative.profile_class


__all__ = [
    "attribute_graph_signal",
    "attribute_graph_signal_with_audit",
    "build_claim_inventory",
    "build_graph_signal_bundle",
    "build_impact_pathways",
    "fuse_structural_innovation",
]
