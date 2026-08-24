"""Backend-neutral, stateless structured model client selection."""

from __future__ import annotations

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
]
