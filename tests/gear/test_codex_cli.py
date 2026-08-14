from __future__ import annotations

from pathlib import Path

from gear.codex_cli import CodexCliJsonClient
from gear.config import CodexCliEndpoint


def test_codex_cli_client_builds_ephemeral_schema_constrained_command() -> None:
    seen: dict[str, object] = {}

    def runner(command: list[str], prompt: str, root: Path) -> str:
        seen["command"] = command
        seen["prompt"] = prompt
        seen["root"] = root
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
    assert "model_reasoning_effort=\"high\"" in command
    assert "Return only the requested JSON object." in str(seen["prompt"])
