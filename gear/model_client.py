"""Backend-neutral, stateless structured model client selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

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
            cache_dir=config.resolve_path(config.cache_dir) / "model_responses",
        )
    if config.openai_compatible is None:
        raise ModelClientUnavailableError(
            "openai_compatible configuration is missing for API mode"
        )
    from .openai_compatible_api import OpenAICompatibleJsonClient

    return OpenAICompatibleJsonClient(config.openai_compatible)


__all__ = [
    "JsonModelClient",
    "ModelClientUnavailableError",
    "build_json_model_client",
    "LazyRoleClient",
]


ROLE_MODELS = {
    "graph_claim": ("gpt-5.6-luna", "medium"),
    "claim_miner": ("gpt-5.6-luna", "medium"),
    "supervisor_planner": ("gpt-5.6-luna", "medium"),
    "claim_consolidator": ("gpt-5.6-terra", "high"),
    "internal_verifier": ("gpt-5.6-terra", "high"),
    "relation_fusion": ("gpt-5.6-terra", "high"),
    "evaluation_judge": ("gpt-5.6-sol", "high"),
}


@dataclass
class LazyRoleClient:
    """Build one role-specific model client only on its first request."""

    config: GearConfig
    role: str
    _client: JsonModelClient | None = None

    def _get(self) -> JsonModelClient:
        if self._client is None:
            model, effort = ROLE_MODELS[self.role]
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
        return self._get().generate_json(
            system=system,
            user=user,
            response_schema=response_schema,
        )
