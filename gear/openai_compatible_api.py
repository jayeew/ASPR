"""Stateless OpenAI-compatible chat-completions JSON client.

This adapter supports providers such as DeepSeek that expose the standard
``/chat/completions`` endpoint. Credentials are read only from the configured
environment variable and are never persisted in review artifacts or logs.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import OpenAICompatibleEndpoint
from .env import getenv
from .model_client import ModelClientUnavailableError

HttpRequester = Callable[[Request, int], bytes]


class OpenAICompatibleJsonClient:
    """Run one independent JSON-only chat completion per request."""

    def __init__(
        self,
        endpoint: OpenAICompatibleEndpoint,
        *,
        requester: HttpRequester | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.requester = requester or _request_bytes

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
        api_key = getenv(self.endpoint.api_key_env)
        if not api_key:
            raise ModelClientUnavailableError(
                f"API key environment variable is unset: {self.endpoint.api_key_env}"
            )
        payload = {
            "model": self.endpoint.model,
            "messages": [
                {
                    "role": "system",
                    "content": _system_prompt(system, response_schema),
                },
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        request = Request(
            url=f"{self.endpoint.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            raw = self.requester(request, self.endpoint.timeout_seconds)
            response = json.loads(raw.decode("utf-8"))
            content = response["choices"][0]["message"]["content"]
            return _json_object(content)
        except (
            HTTPError,
            URLError,
            KeyError,
            TimeoutError,
            TypeError,
            UnicodeDecodeError,
            ValueError,
        ) as exc:
            raise ModelClientUnavailableError(
                f"OpenAI-compatible API request failed for {self.endpoint.model}: {exc}"
            ) from exc


def _request_bytes(request: Request, timeout_seconds: int) -> bytes:
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


def _system_prompt(
    system: str,
    response_schema: Mapping[str, Any] | None,
) -> str:
    prompt = (
        "You are running as a stateless GEAR model worker. Follow the system "
        "instructions exactly. Return only one JSON object; do not add markdown or commentary.\n\n"
        f"SYSTEM INSTRUCTIONS:\n{system}"
    )
    if response_schema is not None:
        prompt += "\n\nOUTPUT JSON SCHEMA:\n" + json.dumps(
            response_schema, ensure_ascii=False, sort_keys=True
        )
    return prompt


def _json_object(content: Any) -> dict[str, Any]:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3].rstrip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("API response content is not a JSON object")
    return value


__all__ = ["OpenAICompatibleJsonClient"]
