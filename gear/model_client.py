"""Backend-neutral, stateless structured model client selection."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .config import GearConfig


class ModelClientUnavailableError(RuntimeError):
    """Raised when the configured structured model backend cannot respond."""


class JsonModelClient(Protocol):
    """Generate one JSON object without retaining conversation state."""

    @property
    def model_name(self) -> str: ...

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        response_schema: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...


def build_json_model_client(config: GearConfig) -> JsonModelClient:
    """Build the configured stateless structured-output backend lazily."""
    if config.model_backend == "codex_cli":
        from .codex_cli import CodexCliJsonClient

        return CodexCliJsonClient(
            config.codex_cli,
            cache_dir=(
                config.resolve_path(config.cache_dir) / "model_responses"
                if config.model_cache_enabled
                else None
            ),
        )
    if config.openai_compatible is None:
        raise ModelClientUnavailableError(
            "openai_compatible configuration is missing for API mode"
        )
    from .openai_compatible_api import OpenAICompatibleJsonClient

    return OpenAICompatibleJsonClient(config.openai_compatible)


__all__ = [
    "JsonModelClient",
    "LazyRoleClient",
    "ModelClientUnavailableError",
    "build_json_model_client",
]


ROLE_MODELS = {
    "field_classifier": ("gpt-5.6-luna", "low"),
    "graph_analysis": ("gpt-5.6-luna", "high"),
    "reference_extract": ("gpt-5.6-luna", "low"),
    "reference_check": ("gpt-5.6-luna", "low"),
    "graph_claim": ("gpt-5.6-luna", "medium"),
    "claim_miner": ("gpt-5.6-luna", "medium"),
    "supervisor_planner": ("gpt-5.6-luna", "medium"),
    "claim_consolidator": ("gpt-5.6-terra", "high"),
    "internal_verifier": ("gpt-5.6-terra", "high"),
    "relation_fusion": ("gpt-5.6-terra", "high"),
    "evaluation_judge": ("gpt-5.6-sol", "high"),
    "report_writer": ("gpt-5.6-luna", "high"),
    "pairwise_judge": ("gpt-5.6-luna", "high"),
}


def resolve_role_model(config: GearConfig, role: str) -> tuple[str, str]:
    model, effort = ROLE_MODELS[role]
    return (
        config.role_model_override or model,
        config.role_effort_overrides.get(role, effort),
    )


@dataclass
class LazyRoleClient:
    """Build one role-specific model client only on its first request."""

    config: GearConfig
    role: str
    _client: JsonModelClient | None = None

    def _get(self) -> JsonModelClient:
        if self._client is None:
            model, effort = resolve_role_model(self.config, self.role)
            endpoint = self.config.codex_cli.model_copy(
                update={"model": model, "reasoning_effort": effort}
            )
            self._client = build_json_model_client(
                self.config.model_copy(update={"codex_cli": endpoint})
            )
        return self._client

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        response_schema: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        from .innovation.usage import progress_scope

        with progress_scope(f"role={self.role}"):
            retries = max(0, int(os.environ.get("GEAR_MODEL_RETRIES", "0")))
            for attempt in range(retries + 1):
                try:
                    return self._get().generate_json(
                        system=system,
                        user=user,
                        response_schema=response_schema,
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    blocked = "limited access to this content" in str(exc).casefold()
                    if blocked or attempt == retries:
                        raise
                    time.sleep(attempt + 1)
        raise RuntimeError("unreachable")
