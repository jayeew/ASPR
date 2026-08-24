from __future__ import annotations

import json
from pathlib import Path

from gear.codex_cli import CodexCliJsonClient
from gear.config import CodexCliEndpoint


def test_codex_cli_client_builds_ephemeral_schema_constrained_command() -> None:
    seen: dict[str, object] = {}

    def runner(command: list[str], prompt: str, root: Path) -> str:
        seen["command"] = command
        seen["prompt"] = prompt
        seen["root"] = root
        seen["schema"] = json.loads((root / "response_schema.json").read_text())
        return '{"answer":"ok"}'

    client = CodexCliJsonClient(
        CodexCliEndpoint(
            executable="codex",
            model="gpt-5.6-terra",
            reasoning_effort="high",
        ),
        runner=runner,
    )
    result = client.generate_json(
        system="Return JSON.",
        user="Review this paper.",
        response_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )

    command = list(seen["command"])
    assert result == {"answer": "ok"}
    assert "--ephemeral" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--model") + 1] == "gpt-5.6-terra"
    assert "--output-schema" in command
    assert 'model_reasoning_effort="high"' in command
    assert "Return only the requested JSON object." in str(seen["prompt"])
    schema = seen["schema"]
    assert isinstance(schema, dict)
    assert schema["required"] == ["answer"]
    assert schema["additionalProperties"] is False


def test_codex_cli_client_makes_optional_fields_explicitly_required() -> None:
    seen: dict[str, object] = {}

    def runner(command: list[str], prompt: str, root: Path) -> str:
        seen["schema"] = json.loads((root / "response_schema.json").read_text())
        return '{"answer":"ok","optional":null}'

    client = CodexCliJsonClient(CodexCliEndpoint(executable="codex"), runner=runner)
    client.generate_json(
        system="Return JSON.",
        user="Review this paper.",
        response_schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "optional": {
                    "type": ["string", "null"],
                    "default": None,
                },
            },
            "required": ["answer"],
        },
    )

    schema = seen["schema"]
    assert isinstance(schema, dict)
    assert schema["required"] == ["answer", "optional"]
    assert schema["additionalProperties"] is False
    assert "default" not in schema["properties"]["optional"]


def test_codex_cli_client_reuses_successful_response_cache(
    tmp_path, monkeypatch
) -> None:
    client = CodexCliJsonClient(
        CodexCliEndpoint(executable="codex", model="gpt-5.6-terra"),
        cache_dir=tmp_path / "responses",
    )
    calls = 0

    def run(prompt, response_schema):
        nonlocal calls
        calls += 1
        return '{"answer":"cached"}'

    monkeypatch.setattr(client, "_run", run)

    first = client.generate_json(system="system", user="same request")
    second = client.generate_json(system="system", user="same request")

    assert first == second == {"answer": "cached"}
    assert calls == 1
    assert client.last_cache_hit is True
