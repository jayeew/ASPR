"""Report conditions sharing masked Graph interpretation stages."""

from __future__ import annotations

from typing import Any

from experiments.innovation_200.contracts import SYSTEMS

MASKED_GRAPH_VARIANTS = frozenset(
    {
        "fusion_no_metrics",
        "fusion_no_citation_paths",
        "fusion_text_only",
    }
)
SUPPORTED_REPORT_SYSTEMS = SYSTEMS


def require_supported_generation(system: str) -> None:
    if system not in SYSTEMS:
        raise ValueError(f"Unknown report system: {system}")


def ablation_plan(system: str) -> dict[str, Any]:
    return {
        "system": system,
        "interpretation": "masked_graph_and_joint",
        "generation_blocked": False,
        "causal_effect_ready": False,
        "comparison_label": "matched interpretation stages; budget comparability reported separately",
        "fixed_inputs": [
            "paper cohort",
            "grounded claims",
            "historical text union",
            "time cutoff",
            "model and role routing",
            "writer task",
        ],
        "generation_stages": [
            "Mask graph facts before interpretation.",
            "Interpret each eligible claim using the normal graph role and prompt.",
            "Interpret the joint masked neighborhood using the normal joint role and prompt.",
            "Write the report with the unchanged historical text catalog and GEAR analysis.",
        ],
    }
