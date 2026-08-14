"""Isolated, schema-constrained Codex CLI adapter for GEAR."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import CodexCliEndpoint
from .model_client import ModelClientUnavailableError


class CodexCLIUnavailableError(ModelClientUnavailableError):
    """Raised when a Codex CLI session cannot produce a valid JSON response."""


CodexRunner = Callable[[Sequence[str], str, Path], str]


class CodexCliJsonClient:
    """Run one ephemeral, read-only Codex CLI session per structured request."""

    def __init__(
        self,
        endpoint: CodexCliEndpoint,
        *,
        runner: CodexRunner | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.runner = runner

    @property
    def model_name(self) -> str:
        return self.endpoint.model

    def generate_json(
        self,
        *,
        system: str,
        user: str,
        response_schema: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        prompt = self._prompt(system, user)
        try:
            raw = self._run(prompt, response_schema)
            return _extract_json_object(raw)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise CodexCLIUnavailableError(
                f"Codex CLI session failed for {self.endpoint.model}: {exc}"
            ) from exc

    def _run(
        self,
        prompt: str,
        response_schema: Mapping[str, Any] | None,
    ) -> str:
        with tempfile.TemporaryDirectory(prefix="gear-codex-") as directory:
            root = Path(directory)
            command = self._command(root, response_schema)
            if self.runner is not None:
                return self.runner(command, prompt, root)
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                check=False,
                cwd=root,
                timeout=self.endpoint.timeout_seconds,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise CodexCLIUnavailableError(
                    f"codex exec exited {completed.returncode}: {detail[:1000]}"
                )
            output_path = root / "response.json"
            if output_path.is_file():
                return output_path.read_text(encoding="utf-8")
            if completed.stdout.strip():
                return completed.stdout
            raise CodexCLIUnavailableError("codex exec returned no final response")

    def _command(
        self,
        root: Path,
        response_schema: Mapping[str, Any] | None,
    ) -> list[str]:
        command = [
            self.endpoint.executable,
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            self.endpoint.sandbox,
            "--model",
            self.endpoint.model,
            "--config",
            f'model_reasoning_effort="{self.endpoint.reasoning_effort}"',
            "--output-last-message",
            str(root / "response.json"),
        ]
        if response_schema is not None:
            schema_path = root / "response_schema.json"
            schema_path.write_text(
                json.dumps(
                    _strict_response_schema(response_schema), ensure_ascii=False
                ),
                encoding="utf-8",
            )
            command.extend(["--output-schema", str(schema_path)])
        command.append("-")
        return command

    @staticmethod
    def _prompt(system: str, user: str) -> str:
        return (
            "You are running in an isolated GEAR review worker. Follow the system "
            "instructions exactly. Do not access files, run tools, or add commentary. "
            "Return only the requested JSON object.\n\nSYSTEM INSTRUCTIONS:\n"
            f"{system}\n\nTASK INPUT:\n{user}"
        )


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object from the CLI final message."""
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("CLI response does not contain a JSON object") from exc
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise TypeError("CLI response must be a JSON object")
    return payload


def _strict_response_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Make a Pydantic-style schema acceptable to Codex strict JSON mode.

    Codex requires every object property to appear in ``required``. Optional
    contract fields remain nullable where the original schema permits it, but
    must be explicitly emitted as ``null`` rather than omitted.
    """
    normalized = deepcopy(dict(schema))

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("default", None)
            properties = value.get("properties")
            if isinstance(properties, dict):
                value["required"] = list(properties)
                value["additionalProperties"] = False
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(normalized)
    return normalized


__all__ = ["CodexCLIUnavailableError", "CodexCliJsonClient", "CodexRunner"]
