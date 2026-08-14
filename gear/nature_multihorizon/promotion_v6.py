"""Fail-closed runtime promotion for v6 innovation evidence definitions.

The literature registry establishes construct and formula eligibility.  It
does not by itself make an indicator confirmatory.  This module records the
separate empirical and implementation evidence required for a frozen release
to promote a definition-stage candidate.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Mapping, Tuple

from pydantic import Field, model_validator

from .contracts_v6 import FrozenV6Contract, RegistryStatus, V6_SCHEMA_VERSION
from .evidence_registry import (
    PROMOTION_GATE_IDS,
    EvidenceRegistry,
    registry_sha256,
)


class PromotionStatus(str, Enum):
    """Runtime disposition of one registered entity."""

    PROMOTED = "promoted"
    HELD = "held"
    NOT_ELIGIBLE = "not_eligible"


class PromotionEntityType(str, Enum):
    """Entity type covered by a promotion decision."""

    DIMENSION = "dimension"
    METRIC = "metric"


class PromotionGateEvidence(FrozenV6Contract):
    """Auditable evidence for one P1-P8 runtime gate."""

    gate_id: str = Field(pattern=r"^P[1-8]$")
    passed: bool
    evidence_artifact_ids: Tuple[str, ...] = ()
    detail: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_evidence_for_pass(self) -> "PromotionGateEvidence":
        """A bare Boolean cannot promote an entity."""
        if self.passed and not self.evidence_artifact_ids:
            raise ValueError("a passing promotion gate requires an evidence artifact")
        return self


class EntityPromotion(FrozenV6Contract):
    """One fail-closed metric or dimension promotion decision."""

    entity_id: str = Field(min_length=1)
    entity_type: PromotionEntityType
    definition_status: RegistryStatus
    promotion_status: PromotionStatus
    gates: Dict[str, PromotionGateEvidence]

    @model_validator(mode="after")
    def validate_decision(self) -> "EntityPromotion":
        """Derive the only valid disposition from eligibility and gates."""
        if set(self.gates) != set(PROMOTION_GATE_IDS):
            raise ValueError("promotion evidence must contain P1-P8 exactly")
        for gate_id, gate in self.gates.items():
            if gate_id != gate.gate_id:
                raise ValueError("promotion gate mapping key must equal gate_id")
        all_pass = all(gate.passed for gate in self.gates.values())
        eligible = self.definition_status is RegistryStatus.CANDIDATE_CONFIRMATORY
        expected = (
            PromotionStatus.PROMOTED
            if eligible and all_pass
            else (
                PromotionStatus.HELD
                if eligible
                else PromotionStatus.NOT_ELIGIBLE
            )
        )
        if self.promotion_status is not expected:
            raise ValueError(
                f"{self.entity_id} promotion_status must be {expected.value}"
            )
        return self


class PromotionReport(FrozenV6Contract):
    """Frozen release evidence separating eligibility from confirmation."""

    schema_version: str = V6_SCHEMA_VERSION
    report_id: str = Field(min_length=1)
    registry_version: str = Field(min_length=1)
    registry_sha256: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    evaluated_artifact_id: str = Field(min_length=1)
    sealed_holdout_inspected: bool
    decisions: Dict[str, EntityPromotion]

    @model_validator(mode="after")
    def validate_report(self) -> "PromotionReport":
        """Prevent duplicate identities and post-holdout promotion."""
        if not self.decisions:
            raise ValueError("promotion report cannot be empty")
        for entity_id, decision in self.decisions.items():
            if entity_id != decision.entity_id:
                raise ValueError("decision mapping key must equal entity_id")
        if self.sealed_holdout_inspected and any(
            decision.promotion_status is PromotionStatus.PROMOTED
            for decision in self.decisions.values()
        ):
            raise ValueError(
                "definition promotion must be frozen before sealed-holdout inspection"
            )
        return self

    @property
    def promoted_entity_ids(self) -> Tuple[str, ...]:
        """Return release-confirmatory entities in registry order."""
        return tuple(
            entity_id
            for entity_id, decision in self.decisions.items()
            if decision.promotion_status is PromotionStatus.PROMOTED
        )


def _missing_gate(gate_id: str) -> PromotionGateEvidence:
    return PromotionGateEvidence(
        gate_id=gate_id,
        passed=False,
        detail="not evaluated; fail-closed",
    )


def _normalize_gates(
    supplied: Mapping[str, PromotionGateEvidence] | None,
) -> Dict[str, PromotionGateEvidence]:
    evidence = supplied or {}
    unknown = sorted(set(evidence) - set(PROMOTION_GATE_IDS))
    if unknown:
        raise ValueError(f"unknown promotion gates: {unknown}")
    return {
        gate_id: evidence.get(gate_id, _missing_gate(gate_id))
        for gate_id in PROMOTION_GATE_IDS
    }


def build_promotion_report(
    registry: EvidenceRegistry,
    evidence_by_entity: Mapping[
        str, Mapping[str, PromotionGateEvidence]
    ],
    *,
    report_id: str,
    evaluated_artifact_id: str,
    sealed_holdout_inspected: bool = False,
) -> PromotionReport:
    """Evaluate every profile definition; absent evidence is an explicit fail."""
    known_entities = {
        **registry.dimensions,
        **{
            metric_id: metric
            for metric_id, metric in registry.metrics.items()
            if metric.dimension_id is not None
        },
    }
    unknown = sorted(set(evidence_by_entity) - set(known_entities))
    if unknown:
        raise ValueError(f"promotion evidence has unknown entities: {unknown}")

    decisions: Dict[str, EntityPromotion] = {}
    for entity_id, entity in known_entities.items():
        gates = _normalize_gates(evidence_by_entity.get(entity_id))
        eligible = entity.status is RegistryStatus.CANDIDATE_CONFIRMATORY
        all_pass = all(gate.passed for gate in gates.values())
        status = (
            PromotionStatus.PROMOTED
            if eligible and all_pass
            else (
                PromotionStatus.HELD
                if eligible
                else PromotionStatus.NOT_ELIGIBLE
            )
        )
        decisions[entity_id] = EntityPromotion(
            entity_id=entity_id,
            entity_type=(
                PromotionEntityType.DIMENSION
                if entity_id in registry.dimensions
                else PromotionEntityType.METRIC
            ),
            definition_status=entity.status,
            promotion_status=status,
            gates=gates,
        )
    return PromotionReport(
        report_id=report_id,
        registry_version=registry.registry_version,
        registry_sha256=registry_sha256(registry),
        evaluated_artifact_id=evaluated_artifact_id,
        sealed_holdout_inspected=sealed_holdout_inspected,
        decisions=decisions,
    )


__all__ = [
    "EntityPromotion",
    "PromotionEntityType",
    "PromotionGateEvidence",
    "PromotionReport",
    "PromotionStatus",
    "build_promotion_report",
]
