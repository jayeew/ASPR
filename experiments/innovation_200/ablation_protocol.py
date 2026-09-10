"""Fail closed on known confounded ablations; expose the pending run protocol."""

from __future__ import annotations

from typing import Any

from experiments.innovation_200.contracts import SYSTEMS

MIXED_PATH_VARIANTS = frozenset(
    {
        "fusion_no_metrics",
        "fusion_no_citation_paths",
        "fusion_text_only",
    }
)
SUPPORTED_REPORT_SYSTEMS = tuple(
    system for system in SYSTEMS if system not in MIXED_PATH_VARIANTS
)


class UncontrolledAblationError(ValueError):
    """A requested condition changes both information and interpretation stages."""


def require_supported_generation(system: str) -> None:
    if system in MIXED_PATH_VARIANTS:
        raise UncontrolledAblationError(
            f"{system} is blocked: the saved implementation mixes interpreted "
            "full-system inputs with raw-fact variant inputs. Regenerate masked "
            "branch/joint interpretations through the same stages and audit the "
            "source catalog before running a controlled ablation. Legacy views "
            "remain inspectable with inspect_legacy_variant=True."
        )


def ablation_plan(system: str) -> dict[str, Any]:
    return {
        "system": system,
        "protocol_version": "controlled_ablation_pending_v1",
        "generation_blocked": system in MIXED_PATH_VARIANTS,
        "causal_effect_ready": False,
        "comparison_label": "pipeline variant comparison until controls are verified",
        "fixed_inputs": [
            "paper cohort",
            "grounded claims",
            "historical text union",
            "time cutoff",
            "model and role routing",
            "writer task",
        ],
        "required_before_run": [
            "Mask facts before branch interpretation; regenerate dependent analyses and caches.",
            "Regenerate joint facts and interpretation under the same information condition.",
            "Audit all exposed analysis fields and the source catalog; save exposure hashes.",
            "Match reasoning stages and record input/output tokens and resource budgets.",
            "For joint, keep the same historical text union and whole-paper writer opportunity.",
        ],
    }
