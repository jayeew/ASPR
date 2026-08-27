"""Single Graph runtime boundary for future diffusion and topology entrances."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Protocol

from .config import GearConfig, load_config
from .contracts import PaperIR
from .diffusion_forecast import DiffusionForecastService
from .graph_prior_contracts import GraphRuntimePacket


class GraphUnavailableError(RuntimeError):
    """Raised only when an injected scorer cannot return a limited packet."""


class GraphScorer(Protocol):
    def score(self, paper_ir: PaperIR, cutoff_date: date) -> GraphRuntimePacket: ...


class GraphService:
    """Lazy facade over the sole frozen D5 release."""

    def __init__(self, config: GearConfig | None = None) -> None:
        resolved = config or load_config()
        self._service = DiffusionForecastService(
            resolved.resolved_forecast_release_manifest(),
            resolved.resolved_forecast_runtime_manifest(),
        )

    def score(self, paper_ir: PaperIR, cutoff_date: date) -> GraphRuntimePacket:
        return self._service.score(paper_ir, cutoff_date)


def graph_runtime_packet(value: Any) -> GraphRuntimePacket:
    """Validate the sole runtime contract; legacy packets are rejected."""

    return GraphRuntimePacket.model_validate(value)


def cutoff_safe_runtime_packet(
    packet: GraphRuntimePacket, cutoff_date: date
) -> GraphRuntimePacket:
    """Revalidate a packet against the requested cutoff without silent filtering."""

    if packet.cutoff_date != cutoff_date:
        raise ValueError("Graph packet cutoff_date mismatch")
    return GraphRuntimePacket.model_validate(packet)


def normalize_openalex_id(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    suffix = text.rsplit("/", 1)[-1]
    if not re.fullmatch(r"W\d+", suffix, re.IGNORECASE):
        return None
    return f"https://openalex.org/{suffix.upper()}"


GraphPriorService = GraphService

__all__ = [
    "GraphPriorService",
    "GraphScorer",
    "GraphService",
    "GraphUnavailableError",
    "cutoff_safe_runtime_packet",
    "graph_runtime_packet",
    "normalize_openalex_id",
]
