"""Source-backed registry for v6 influence outcomes and prediction covariates.

Innovation evidence and future-influence prediction intentionally use
different registries.  Future observations may define outcomes, but they can
never be relabelled as publication-time innovation evidence.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

from pydantic import Field, model_validator

from .contracts_v6 import (
    EvidenceClass,
    FrozenV6Contract,
    QualityRule,
    RegistryStatus,
    SourceReference,
    V6_SCHEMA_VERSION,
)


CATEGORY_GATE_IDS = tuple(f"K{index}" for index in range(1, 8))
VARIABLE_GATE_IDS = tuple(f"V{index}" for index in range(1, 11))
EVALUATION_RULE_IDS = tuple(f"E{index}" for index in range(1, 8))


class PredictionRole(str, Enum):
    """Scientific role of a prediction-registry entity."""

    OUTCOME = "outcome"
    CONTROL = "control"
    OPPORTUNITY = "opportunity"


class DefinitionOrigin(str, Enum):
    """Whether the exact operational definition comes from a source."""

    SOURCE_DEFINED = "source_defined"
    SOURCE_ADAPTED = "source_adapted"
    PROJECT_DEFINED = "project_defined"


class PredictionUse(str, Enum):
    """Frozen use of one variable in the influence workflow."""

    PRIMARY_UPTAKE_OUTCOME = "primary_uptake_outcome"
    PRIMARY_CONDITIONAL_OUTCOME = "primary_conditional_outcome"
    DESCRIPTIVE_OUTCOME = "descriptive_outcome"
    SECONDARY_OUTCOME = "secondary_outcome"
    STRONG_CONTROL_BASELINE = "strong_control_baseline"
    OPPORTUNITY_PREDICTOR = "opportunity_predictor"
    SENSITIVITY_ONLY = "sensitivity_only"
    NOT_USED = "not_used"


class PredictionCategory(FrozenV6Contract):
    """Source-backed outcome, control, or opportunity category."""

    category_id: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    label: str = Field(min_length=1)
    construct_definition: str = Field(
        min_length=1,
        validation_alias="construct",
        serialization_alias="construct",
    )
    role: PredictionRole
    status: RegistryStatus
    source_ids: Tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    boundary: str = Field(min_length=1)
    admission_checks: Dict[str, bool]

    @model_validator(mode="after")
    def validate_category(self) -> "PredictionCategory":
        if set(self.admission_checks) != set(CATEGORY_GATE_IDS):
            raise ValueError("prediction category checks must contain K1-K7")
        if self.status is RegistryStatus.REGISTERED and not all(
            self.admission_checks.values()
        ):
            raise ValueError("registered prediction categories must pass K1-K7")
        return self


class PredictionVariable(FrozenV6Contract):
    """Fully specified outcome, control, or opportunity variable."""

    variable_id: str = Field(pattern=r"^[A-Z][A-Z0-9_.-]*$")
    code_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    category_id: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    role: PredictionRole
    status: RegistryStatus
    evidence_class: EvidenceClass
    source_ids: Tuple[str, ...] = Field(min_length=1)
    definition_origin: DefinitionOrigin
    formula: str = Field(min_length=1)
    algebraic_family: str = Field(min_length=1)
    prediction_use: PredictionUse
    horizon_rule: str = Field(min_length=1)
    information_time: str = Field(min_length=1)
    cutoff_rule: str = Field(min_length=1)
    zero_rule: str = Field(min_length=1)
    missingness_rule: str = Field(min_length=1)
    fold_rule: str = Field(min_length=1)
    quality_rules: Tuple[QualityRule, ...] = Field(min_length=1)
    implementation_name: str = Field(min_length=1)
    admission_checks: Dict[str, bool]
    disposition_reason: str = Field(min_length=1)
    future_information_allowed: bool = False

    @model_validator(mode="after")
    def validate_variable(self) -> "PredictionVariable":
        if set(self.admission_checks) != set(VARIABLE_GATE_IDS):
            raise ValueError("prediction variable checks must contain V1-V10")
        if self.status is RegistryStatus.REGISTERED and not all(
            self.admission_checks.values()
        ):
            raise ValueError("registered prediction variables must pass V1-V10")
        if self.role is PredictionRole.OUTCOME:
            if not self.future_information_allowed:
                raise ValueError("registered outcomes must declare future information")
        elif self.future_information_allowed:
            raise ValueError("controls and opportunity predictors cannot use the future")
        outcome_uses = {
            PredictionUse.PRIMARY_UPTAKE_OUTCOME,
            PredictionUse.PRIMARY_CONDITIONAL_OUTCOME,
            PredictionUse.DESCRIPTIVE_OUTCOME,
            PredictionUse.SECONDARY_OUTCOME,
        }
        if (self.prediction_use in outcome_uses) != (
            self.role is PredictionRole.OUTCOME
        ):
            raise ValueError("prediction_use and variable role disagree")
        if (
            self.prediction_use is PredictionUse.STRONG_CONTROL_BASELINE
            and self.role is not PredictionRole.CONTROL
        ):
            raise ValueError("strong baselines must be registered controls")
        if (
            self.prediction_use is PredictionUse.OPPORTUNITY_PREDICTOR
            and self.role is not PredictionRole.OPPORTUNITY
        ):
            raise ValueError("opportunity use requires the opportunity role")
        return self


class PredictionRegistry(FrozenV6Contract):
    """Complete v6 target, baseline, opportunity, and evaluation protocol."""

    schema_version: str = V6_SCHEMA_VERSION
    registry_version: str = Field(min_length=1)
    registry_stage: str = Field(pattern=r"^definition_preregistered$")
    literature_search_cutoff: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    network_policy: str = Field(pattern=r"^forbidden$")
    raw_data_policy: str = Field(pattern=r"^local_frozen_only$")
    target_scope: str = Field(min_length=1)
    category_inclusion_rules: Dict[str, str]
    variable_inclusion_rules: Dict[str, str]
    evaluation_rules: Dict[str, str]
    category_exclusion_rules: Tuple[str, ...] = Field(min_length=1)
    variable_exclusion_rules: Tuple[str, ...] = Field(min_length=1)
    evaluation_source_ids: Tuple[str, ...] = Field(min_length=1)
    sources: Dict[str, SourceReference]
    categories: Dict[str, PredictionCategory]
    variables: Dict[str, PredictionVariable]

    @model_validator(mode="after")
    def validate_registry(self) -> "PredictionRegistry":
        if set(self.category_inclusion_rules) != set(CATEGORY_GATE_IDS):
            raise ValueError("category protocol must define K1-K7")
        if set(self.variable_inclusion_rules) != set(VARIABLE_GATE_IDS):
            raise ValueError("variable protocol must define V1-V10")
        if set(self.evaluation_rules) != set(EVALUATION_RULE_IDS):
            raise ValueError("evaluation protocol must define E1-E7")
        self._validate_keys_and_references()
        self._validate_roles_and_uses()
        return self

    def _validate_keys_and_references(self) -> None:
        for source_id, source in self.sources.items():
            if source_id != source.source_id:
                raise ValueError("source mapping key must equal source_id")
        for category_id, category in self.categories.items():
            if category_id != category.category_id:
                raise ValueError("category mapping key must equal category_id")
        for variable_id, variable in self.variables.items():
            if variable_id != variable.variable_id:
                raise ValueError("variable mapping key must equal variable_id")
        code_names = [variable.code_name for variable in self.variables.values()]
        if len(code_names) != len(set(code_names)):
            raise ValueError("prediction variable code_name values must be unique")
        known_sources = set(self.sources)
        referenced = set(self.evaluation_source_ids)
        for category in self.categories.values():
            referenced.update(category.source_ids)
        for variable in self.variables.values():
            referenced.update(variable.source_ids)
        missing = sorted(referenced - known_sources)
        if missing:
            raise ValueError(f"prediction registry has unknown sources: {missing}")

    def _validate_roles_and_uses(self) -> None:
        for variable in self.variables.values():
            if variable.category_id not in self.categories:
                raise ValueError(
                    f"{variable.variable_id} has unknown category "
                    f"{variable.category_id}"
                )
            category = self.categories[variable.category_id]
            if category.role is not variable.role:
                raise ValueError(
                    f"{variable.variable_id} role disagrees with its category"
                )
        required_uses = {
            PredictionUse.PRIMARY_UPTAKE_OUTCOME,
            PredictionUse.PRIMARY_CONDITIONAL_OUTCOME,
            PredictionUse.STRONG_CONTROL_BASELINE,
            PredictionUse.OPPORTUNITY_PREDICTOR,
        }
        present = {
            variable.prediction_use for variable in self.variables.values()
        }
        missing = sorted(item.value for item in required_uses - present)
        if missing:
            raise ValueError(
                f"prediction registry lacks required design roles: {missing}"
            )

    @property
    def strong_control_names(self) -> Tuple[str, ...]:
        return tuple(
            variable.code_name
            for variable in self.variables.values()
            if variable.prediction_use
            is PredictionUse.STRONG_CONTROL_BASELINE
        )

    @property
    def opportunity_feature_names(self) -> Tuple[str, ...]:
        return tuple(
            variable.code_name
            for variable in self.variables.values()
            if variable.prediction_use is PredictionUse.OPPORTUNITY_PREDICTOR
        )

    @property
    def primary_outcome_names(self) -> Tuple[str, ...]:
        primary = {
            PredictionUse.PRIMARY_UPTAKE_OUTCOME,
            PredictionUse.PRIMARY_CONDITIONAL_OUTCOME,
        }
        return tuple(
            variable.code_name
            for variable in self.variables.values()
            if variable.prediction_use in primary
        )


def load_prediction_registry(path: Path) -> PredictionRegistry:
    """Read and fully validate a v6 prediction registry."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("prediction registry must be a JSON object")
    return PredictionRegistry.model_validate(payload)


def prediction_registry_sha256(registry: PredictionRegistry) -> str:
    """Return the canonical prediction-registry lineage hash."""
    payload = registry.model_dump(mode="json", by_alias=True)
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def audit_prediction_registry_implementations(
    registry: PredictionRegistry,
) -> Dict[str, object]:
    """Resolve every registered materializer without executing data access."""
    rows = []
    for variable in registry.variables.values():
        module_name, separator, attribute_name = variable.implementation_name.rpartition(
            "."
        )
        resolved = False
        error = ""
        if not separator:
            error = "implementation name is not dotted"
        else:
            try:
                module = importlib.import_module(module_name)
                implementation = getattr(module, attribute_name)
                if not callable(implementation):
                    raise TypeError("registered implementation is not callable")
                resolved = True
            except (ImportError, AttributeError, TypeError) as exc:
                error = str(exc)
        rows.append(
            {
                "variable_id": variable.variable_id,
                "implementation_name": variable.implementation_name,
                "resolved": resolved,
                "error": error,
            }
        )
    failures = [row for row in rows if not row["resolved"]]
    return {
        "overall_pass": not failures,
        "n_variables": len(rows),
        "n_failures": len(failures),
        "rows": rows,
    }


__all__ = [
    "CATEGORY_GATE_IDS",
    "DefinitionOrigin",
    "EVALUATION_RULE_IDS",
    "PredictionCategory",
    "PredictionRegistry",
    "PredictionRole",
    "PredictionUse",
    "PredictionVariable",
    "VARIABLE_GATE_IDS",
    "audit_prediction_registry_implementations",
    "load_prediction_registry",
    "prediction_registry_sha256",
]
