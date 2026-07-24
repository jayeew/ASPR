"""Audit the targeted evidence map behind both v6 registries."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from pydantic import Field, model_validator

from .contracts_v6 import FrozenV6Contract, V6_SCHEMA_VERSION
from .evidence_registry import EvidenceRegistry
from .prediction_registry_v6 import PredictionRegistry


SOURCE_SELECTION_GATE_IDS = tuple(f"S{index}" for index in range(1, 9))


class SelectionDecision(str, Enum):
    """Disposition in the evidence map."""

    INCLUDED = "included"
    CONDITIONAL_SUPPORT = "conditional_support"
    EXCLUDED = "excluded"


class SourceTier(str, Enum):
    """Role of a source in construct and indicator justification."""

    FOUNDATIONAL = "foundational"
    OPERATIONALIZATION = "operationalization"
    INDEPENDENT_VALIDITY = "independent_validity"
    CRITIQUE = "critique"
    STATISTICAL_METHOD = "statistical_method"
    PREDICTION_EVIDENCE = "prediction_evidence"


class EvidenceSelectionRecord(FrozenV6Contract):
    """One included, conditional, or excluded source decision."""

    record_id: str = Field(min_length=1)
    source_id: Optional[str] = None
    citation: str = Field(min_length=1)
    doi: Optional[str] = None
    url: Optional[str] = None
    decision: SelectionDecision
    source_tiers: Tuple[SourceTier, ...] = Field(min_length=1)
    registry_roles: Tuple[str, ...] = ()
    construct_tags: Tuple[str, ...] = Field(min_length=1)
    decision_reason: str = Field(min_length=1)
    limitations: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_record(self) -> "EvidenceSelectionRecord":
        if not self.doi and not self.url:
            raise ValueError("selection records require a DOI or stable URL")
        if self.decision is not SelectionDecision.EXCLUDED and not self.source_id:
            raise ValueError("admitted records require a registry source_id")
        if self.decision is SelectionDecision.EXCLUDED and self.registry_roles:
            raise ValueError("excluded records cannot claim registry roles")
        return self


class EvidenceSelectionProtocol(FrozenV6Contract):
    """Reproducible targeted evidence map, explicitly not a meta-analysis."""

    schema_version: str = V6_SCHEMA_VERSION
    protocol_id: str = Field(min_length=1)
    review_type: str = Field(
        pattern=r"^targeted_scoping_evidence_map_not_systematic_review$"
    )
    execution_status: str = Field(pattern=r"^completed_for_registered_sources$")
    search_cutoff: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    objective: str = Field(min_length=1)
    discovery_channels: Tuple[str, ...] = Field(min_length=1)
    query_blocks: Tuple[str, ...] = Field(min_length=1)
    source_selection_rules: Dict[str, str]
    exclusion_rules: Tuple[str, ...] = Field(min_length=1)
    limitations: Tuple[str, ...] = Field(min_length=1)
    records: Dict[str, EvidenceSelectionRecord]

    @model_validator(mode="after")
    def validate_protocol(self) -> "EvidenceSelectionProtocol":
        if set(self.source_selection_rules) != set(SOURCE_SELECTION_GATE_IDS):
            raise ValueError("source selection protocol must define S1-S8")
        for record_id, record in self.records.items():
            if record_id != record.record_id:
                raise ValueError("record mapping key must equal record_id")
        admitted_source_ids = [
            record.source_id
            for record in self.records.values()
            if record.decision is not SelectionDecision.EXCLUDED
        ]
        if len(admitted_source_ids) != len(set(admitted_source_ids)):
            raise ValueError("admitted source_id values must be unique")
        return self


def load_evidence_selection_protocol(path: Path) -> EvidenceSelectionProtocol:
    """Load and validate the v6 source-selection audit."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evidence selection protocol must be a JSON object")
    return EvidenceSelectionProtocol.model_validate(payload)


def audit_registry_source_selection(
    protocol: EvidenceSelectionProtocol,
    innovation_registry: EvidenceRegistry,
    prediction_registry: PredictionRegistry,
) -> Dict[str, object]:
    """Ensure every registry source has an admitted, locator-consistent record."""
    records = {
        str(record.source_id): record
        for record in protocol.records.values()
        if record.source_id is not None
    }
    sources = {
        **innovation_registry.sources,
        **prediction_registry.sources,
    }
    rows = []
    for source_id, source in sources.items():
        record = records.get(source_id)
        missing = record is None
        excluded = bool(
            record and record.decision is SelectionDecision.EXCLUDED
        )
        locator_match = bool(
            record
            and (
                (source.doi and record.doi == source.doi)
                or (source.url and record.url == source.url)
            )
        )
        rows.append(
            {
                "source_id": source_id,
                "record_present": not missing,
                "admitted": not missing and not excluded,
                "locator_match": locator_match,
            }
        )
    failures = [
        row
        for row in rows
        if not (
            row["record_present"]
            and row["admitted"]
            and row["locator_match"]
        )
    ]
    return {
        "overall_pass": not failures,
        "n_registry_sources": len(rows),
        "n_failures": len(failures),
        "rows": rows,
    }


def evidence_selection_sha256(
    protocol: EvidenceSelectionProtocol,
) -> str:
    """Return a canonical hash for registry lineage."""
    payload = protocol.model_dump(mode="json")
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


__all__ = [
    "EvidenceSelectionProtocol",
    "EvidenceSelectionRecord",
    "SOURCE_SELECTION_GATE_IDS",
    "SelectionDecision",
    "SourceTier",
    "audit_registry_source_selection",
    "evidence_selection_sha256",
    "load_evidence_selection_protocol",
]
