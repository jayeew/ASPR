"""Declared module boundaries for the shared ASPR artifact exchange."""

from __future__ import annotations

from collections.abc import Mapping

MODULE_DEPENDENCIES: Mapping[str, frozenset[str]] = {
    "datasets": frozenset(),
    "indicator_definition": frozenset({"datasets"}),
    "aspr_scoring": frozenset({"datasets", "indicator_definition"}),
    "review_reconstruction": frozenset({"datasets"}),
    "gear_agent": frozenset({"datasets", "indicator_definition", "aspr_scoring"}),
    "figures": frozenset({"datasets"}),
    "consistency_evaluation": frozenset({"review_reconstruction", "gear_agent"}),
    "review_evaluation": frozenset({"review_reconstruction", "gear_agent"}),
}


def validate_dependency(producer: str, dependency_producer: str) -> None:
    """Reject an undeclared cross-module dependency."""
    allowed = MODULE_DEPENDENCIES.get(producer)
    if allowed is None:
        raise ValueError(f"unknown artifact producer: {producer}")
    if dependency_producer not in allowed:
        raise ValueError(
            f"{producer} cannot consume artifacts from {dependency_producer}; "
            f"allowed={sorted(allowed)}"
        )


__all__ = ["MODULE_DEPENDENCIES", "validate_dependency"]
