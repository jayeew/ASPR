"""Load and audit the machine-readable v6 innovation evidence registry."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import pandas as pd
from pydantic import Field, model_validator

from .contracts_v6 import (
    DimensionDefinition,
    FrozenV6Contract,
    MetricDefinition,
    MetricRole,
    ModelUse,
    RegistryStatus,
    SourceReference,
    V6_SCHEMA_VERSION,
)


DIMENSION_GATE_IDS = tuple(f"D{index}" for index in range(1, 8))
INDICATOR_GATE_IDS = tuple(f"I{index}" for index in range(1, 11))
PROMOTION_GATE_IDS = tuple(f"P{index}" for index in range(1, 9))


class EvidenceRegistry(FrozenV6Contract):
    """Complete source, construct, metric, and selection protocol."""

    schema_version: str = V6_SCHEMA_VERSION
    registry_version: str = Field(min_length=1)
    registry_stage: str = Field(pattern=r"^definition_preregistered$")
    literature_search_cutoff: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    network_policy: str = Field(pattern=r"^forbidden$")
    raw_data_policy: str = Field(pattern=r"^local_frozen_only$")
    target_scope: str = Field(min_length=1)
    dimension_inclusion_rules: Dict[str, str]
    indicator_inclusion_rules: Dict[str, str]
    runtime_promotion_rules: Dict[str, str]
    dimension_exclusion_rules: Tuple[str, ...] = Field(min_length=1)
    indicator_exclusion_rules: Tuple[str, ...] = Field(min_length=1)
    sources: Dict[str, SourceReference]
    dimensions: Dict[str, DimensionDefinition]
    metrics: Dict[str, MetricDefinition]

    @model_validator(mode="after")
    def validate_registry(self) -> "EvidenceRegistry":
        """Reject dangling sources, semantic laundering, and duplicate inputs."""
        if set(self.dimension_inclusion_rules) != set(DIMENSION_GATE_IDS):
            raise ValueError("dimension inclusion protocol must define D1-D7")
        if set(self.indicator_inclusion_rules) != set(INDICATOR_GATE_IDS):
            raise ValueError("indicator inclusion protocol must define I1-I10")
        if set(self.runtime_promotion_rules) != set(PROMOTION_GATE_IDS):
            raise ValueError("runtime promotion protocol must define P1-P8")
        self._validate_mapping_keys()
        self._validate_references()
        self._validate_dimensions_and_roles()
        self._validate_model_families()
        self._forbid_premature_confirmatory_labels()
        return self

    def _validate_mapping_keys(self) -> None:
        for source_id, source in self.sources.items():
            if source_id != source.source_id:
                raise ValueError("source mapping key must equal source_id")
        for dimension_id, dimension in self.dimensions.items():
            if dimension_id != dimension.dimension_id:
                raise ValueError("dimension mapping key must equal dimension_id")
        for metric_id, metric in self.metrics.items():
            if metric_id != metric.metric_id:
                raise ValueError("metric mapping key must equal metric_id")
        code_names = [metric.code_name for metric in self.metrics.values()]
        if len(code_names) != len(set(code_names)):
            raise ValueError("metric code_name values must be unique")

    def _validate_references(self) -> None:
        known_sources = set(self.sources)
        for dimension in self.dimensions.values():
            used = set(dimension.foundational_source_ids) | set(
                dimension.paper_level_source_ids
            )
            missing = used - known_sources
            if missing:
                raise ValueError(
                    f"dimension {dimension.dimension_id} has unknown sources: {sorted(missing)}"
                )
        for metric in self.metrics.values():
            missing = set(metric.source_ids) - known_sources
            if missing:
                raise ValueError(
                    f"metric {metric.metric_id} has unknown sources: {sorted(missing)}"
                )

    def _validate_dimensions_and_roles(self) -> None:
        for metric in self.metrics.values():
            if metric.dimension_id:
                if metric.dimension_id not in self.dimensions:
                    raise ValueError(
                        f"metric {metric.metric_id} has unknown dimension "
                        f"{metric.dimension_id}"
                    )
                dimension = self.dimensions[metric.dimension_id]
                if metric.role is not dimension.role:
                    raise ValueError(
                        f"metric {metric.metric_id} role disagrees with its dimension"
                    )
                if (
                    metric.status
                    in {
                        RegistryStatus.CONFIRMATORY,
                        RegistryStatus.CANDIDATE_CONFIRMATORY,
                    }
                    and dimension.status
                    not in {
                        RegistryStatus.CONFIRMATORY,
                        RegistryStatus.CANDIDATE_CONFIRMATORY,
                    }
                ):
                    raise ValueError(
                        "a confirmatory-candidate metric cannot belong to an "
                        "ineligible dimension"
                    )
            elif metric.role not in {
                MetricRole.CONTROL,
                MetricRole.OUTCOME,
                MetricRole.EXCLUDED,
                MetricRole.PREDICTION_ONLY,
            }:
                raise ValueError(
                    f"metric {metric.metric_id} requires a registered dimension"
                )
            if metric.model_use is ModelUse.PRIMARY:
                if not metric.implementation_name or not metric.unit_test_ids:
                    raise ValueError(
                        f"primary metric {metric.metric_id} lacks implementation tests"
                    )

        for dimension in self.dimensions.values():
            confirmatory_candidates = [
                metric
                for metric in self.metrics.values()
                if metric.dimension_id == dimension.dimension_id
                and metric.status
                in {
                    RegistryStatus.CONFIRMATORY,
                    RegistryStatus.CANDIDATE_CONFIRMATORY,
                }
            ]
            if dimension.status in {
                RegistryStatus.CONFIRMATORY,
                RegistryStatus.CANDIDATE_CONFIRMATORY,
            }:
                families = {
                    metric.algebraic_family
                    for metric in confirmatory_candidates
                }
                if len(families) < 2:
                    raise ValueError(
                        f"confirmatory-candidate dimension "
                        f"{dimension.dimension_id} needs two eligible metric families"
                    )

    def _validate_model_families(self) -> None:
        primary_keys = [
            (metric.dimension_id, metric.algebraic_family)
            for metric in self.metrics.values()
            if metric.model_use is ModelUse.PRIMARY
        ]
        duplicates = sorted(
            key for key in set(primary_keys) if primary_keys.count(key) > 1
        )
        if duplicates:
            raise ValueError(
                "only one primary model input is allowed per algebraic family: "
                f"{duplicates}"
            )

    def _forbid_premature_confirmatory_labels(self) -> None:
        """Keep literature eligibility separate from empirical promotion."""
        prematurely_confirmatory = [
            entity_id
            for entity_id, entity in (
                list(self.dimensions.items()) + list(self.metrics.items())
            )
            if entity.status is RegistryStatus.CONFIRMATORY
        ]
        if prematurely_confirmatory:
            raise ValueError(
                "definition_preregistered registries must use "
                "candidate_confirmatory; confirmatory is assigned only by a "
                f"runtime promotion report: {prematurely_confirmatory}"
            )

    @property
    def primary_feature_names(self) -> Tuple[str, ...]:
        """Return publication-time model inputs in registry order."""
        return tuple(
            metric.code_name
            for metric in self.metrics.values()
            if metric.model_use is ModelUse.PRIMARY
            and metric.role is not MetricRole.OUTCOME
        )

    @property
    def profile_metric_ids(self) -> Tuple[str, ...]:
        """Return metrics eligible for the public evidence profile."""
        allowed_roles = {
            MetricRole.DIRECT_INNOVATION,
            MetricRole.SUPPORTING_CONTEXT,
            MetricRole.STRUCTURAL_LEADING,
        }
        return tuple(
            metric.metric_id
            for metric in self.metrics.values()
            if metric.role in allowed_roles
            and metric.status
            in {
                RegistryStatus.CONFIRMATORY,
                RegistryStatus.CANDIDATE_CONFIRMATORY,
                RegistryStatus.CONDITIONAL,
                RegistryStatus.EXPLORATORY,
            }
        )


def load_evidence_registry(path: Path) -> EvidenceRegistry:
    """Read and validate a JSON evidence registry."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evidence registry must be a JSON object")
    return EvidenceRegistry.model_validate(payload)


def registry_sha256(registry: EvidenceRegistry) -> str:
    """Return a deterministic content hash for registry lineage."""
    payload = registry.model_dump(mode="json", by_alias=True)
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def resolve_implementation(dotted_name: str) -> Any:
    """Resolve a registered dotted implementation name without executing it."""
    module_name, separator, attribute_name = dotted_name.rpartition(".")
    if not separator:
        raise ValueError(f"implementation name is not dotted: {dotted_name}")
    module = importlib.import_module(module_name)
    implementation = getattr(module, attribute_name, None)
    if implementation is None or not callable(implementation):
        raise ValueError(f"registered implementation is unavailable: {dotted_name}")
    return implementation


def audit_registry_implementations(registry: EvidenceRegistry) -> Dict[str, Any]:
    """Check that every model-active implementation is importable."""
    rows = []
    for metric in registry.metrics.values():
        required = metric.model_use is ModelUse.PRIMARY
        resolved = False
        error = ""
        if metric.implementation_name:
            try:
                resolve_implementation(metric.implementation_name)
                resolved = True
            except (ImportError, AttributeError, ValueError) as exc:
                error = str(exc)
        rows.append(
            {
                "metric_id": metric.metric_id,
                "code_name": metric.code_name,
                "required": required,
                "resolved": resolved,
                "error": error,
            }
        )
    failures = [row for row in rows if row["required"] and not row["resolved"]]
    return {
        "overall_pass": not failures,
        "n_metrics": len(rows),
        "n_required": sum(int(row["required"]) for row in rows),
        "n_required_failures": len(failures),
        "rows": rows,
    }


def audit_publication_time_frame(
    frame: pd.DataFrame,
    registry: EvidenceRegistry,
    *,
    publication_year_column: str = "publication_year",
    source_max_year_column: str = "source_max_year",
) -> Dict[str, Any]:
    """Audit feature presence, finiteness, and strict publication-time cutoff."""
    required = set(registry.primary_feature_names) | {
        publication_year_column,
        source_max_year_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        return {
            "overall_pass": False,
            "missing_columns": missing,
            "strict_prior_violations": None,
            "nonfinite_counts": {},
        }
    publication_year = pd.to_numeric(
        frame[publication_year_column], errors="coerce"
    )
    source_max_year = pd.to_numeric(frame[source_max_year_column], errors="coerce")
    strict_prior = source_max_year.lt(publication_year)
    nonfinite_counts = {
        name: int(pd.to_numeric(frame[name], errors="coerce").isna().sum())
        for name in registry.primary_feature_names
    }
    violations = int((~strict_prior).sum())
    return {
        "overall_pass": violations == 0 and not any(nonfinite_counts.values()),
        "missing_columns": [],
        "strict_prior_violations": violations,
        "nonfinite_counts": nonfinite_counts,
    }


def write_registry_snapshot(
    registry: EvidenceRegistry, output_path: Path
) -> Mapping[str, str]:
    """Write a canonical derived snapshot and return its lineage hash."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            registry.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"path": str(path), "sha256": registry_sha256(registry)}


__all__ = [
    "DIMENSION_GATE_IDS",
    "EvidenceRegistry",
    "INDICATOR_GATE_IDS",
    "PROMOTION_GATE_IDS",
    "audit_publication_time_frame",
    "audit_registry_implementations",
    "load_evidence_registry",
    "registry_sha256",
    "resolve_implementation",
    "write_registry_snapshot",
]
