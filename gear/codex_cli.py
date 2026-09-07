"""Isolated, schema-constrained Codex CLI adapter for GEAR."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import CodexCliEndpoint
from .env import getenv
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
        cache_dir: Path | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.runner = runner
        self.cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
        self.last_cache_hit = False

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
        started = time.monotonic()
        success = False
        from .innovation.usage import log_progress

        log_progress(
            "[模型调用开始] model=%s，effort=%s，service_tier=%s",
            self.endpoint.model,
            self.endpoint.reasoning_effort,
            _codex_service_tier() or "configured_default",
        )
        prompt = self._prompt(system, user)
        cache_path = self._cache_path(prompt, response_schema)
        self.last_cache_hit = False
        try:
            if cache_path is not None and cache_path.is_file():
                cached = _extract_json_object(cache_path.read_text(encoding="utf-8"))
                self.last_cache_hit = True
                success = True
                return cached
            raw = self._run(prompt, response_schema)
            payload = _extract_json_object(raw)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(cache_path)
            success = True
            return payload
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise CodexCLIUnavailableError(
                f"Codex CLI session failed for {self.endpoint.model}: {exc}"
            ) from exc

        finally:
            from .innovation.usage import record_call

            record_call(
                self.endpoint.model,
                time.monotonic() - started,
                self.last_cache_hit,
                success,
            )

    def _cache_path(
        self,
        prompt: str,
        response_schema: Mapping[str, Any] | None,
    ) -> Path | None:
        if self.cache_dir is None or self.runner is not None:
            return None
        identity = json.dumps(
            {
                "contract": "gear_codex_response_cache_v1",
                "model": self.endpoint.model,
                "reasoning_effort": self.endpoint.reasoning_effort,
                "service_tier": _codex_service_tier(),
                "prompt": prompt,
                "response_schema": response_schema,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

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
            from .cli_limiter import codex_cli_lease

            with codex_cli_lease():
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
                    f"codex exec exited {completed.returncode}: {detail[-2000:]}"
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
        ]
        service_tier = _codex_service_tier()
        if service_tier:
            command.extend(["--config", f'service_tier="{service_tier}"'])
        command.extend(["--output-last-message", str(root / "response.json")])
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


def _codex_service_tier() -> str:
    value = getenv("GEAR_CODEX_SERVICE_TIER").strip().casefold()
    if value not in {"", "auto", "default", "flex", "fast", "priority"}:
        raise ValueError(f"Unsupported Codex service tier: {value}")
    return value


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
