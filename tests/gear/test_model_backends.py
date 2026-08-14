from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request

import pytest

from gear.config import OpenAICompatibleEndpoint, load_config
from gear.model_client import ModelClientUnavailableError, build_json_model_client
from gear.openai_compatible_api import OpenAICompatibleJsonClient


def test_config_defaults_to_codex_cli() -> None:
    config = load_config()
    assert config.model_backend == "codex_cli"
    assert build_json_model_client(config).model_name == "gpt-5.6-terra"


def test_openai_compatible_client_is_stateless_and_schema_prompted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "test-secret")
    seen: dict[str, object] = {}

    def requester(request: Request, timeout: int) -> bytes:
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["authorization"] = request.get_header("Authorization")
        seen["payload"] = json.loads((request.data or b"").decode("utf-8"))
        return b'{"choices":[{"message":{"content":"{\\"answer\\":\\"ok\\"}"}}]}'

    client = OpenAICompatibleJsonClient(
        OpenAICompatibleEndpoint(
            base_url="https://api.example.test/v1/",
            model="deepseek-reasoner",
            api_key_env="TEST_DEEPSEEK_KEY",
            timeout_seconds=42,
        ),
        requester=requester,
    )
    result = client.generate_json(
        system="Return JSON.",
        user="Review this paper.",
        response_schema={"type": "object"},
    )

    assert result == {"answer": "ok"}
    assert seen["url"] == "https://api.example.test/v1/chat/completions"
    assert seen["timeout"] == 42
    assert seen["authorization"] == "Bearer test-secret"
    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert payload["response_format"] == {"type": "json_object"}
    assert "OUTPUT JSON SCHEMA" in payload["messages"][0]["content"]


def test_openai_compatible_client_fails_closed_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISSING_PROVIDER_KEY", raising=False)
    client = OpenAICompatibleJsonClient(
        OpenAICompatibleEndpoint(
            base_url="https://api.example.test/v1",
            model="provider-model",
            api_key_env="MISSING_PROVIDER_KEY",
        )
    )
    with pytest.raises(ModelClientUnavailableError, match="MISSING_PROVIDER_KEY"):
        client.generate_json(system="system", user="user")


def test_deepseek_config_merges_with_default_assets() -> None:
    config = load_config(Path("configs/gear/deepseek.example.json"))
    assert config.model_backend == "openai_compatible"
    assert config.openai_compatible is not None
    assert config.openai_compatible.model == "deepseek-reasoner"


def test_environment_can_select_openai_compatible_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASPR_GEAR_MODEL_BACKEND", "openai_compatible")
    monkeypatch.setenv("ASPR_GEAR_API_BASE_URL", "https://api.example.test/v1")
    monkeypatch.setenv("ASPR_GEAR_API_MODEL", "provider-model")
    config = load_config()
    assert config.model_backend == "openai_compatible"
    assert config.openai_compatible is not None
    assert config.openai_compatible.api_key_env == "ASPR_GEAR_API_KEY"
