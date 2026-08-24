"""Backend-neutral, content-addressed evaluator client."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from gear.codex_cli import CodexCliJsonClient
from gear.model_client import JsonModelClient
from gear.openai_compatible_api import OpenAICompatibleJsonClient
from gear.trace import canonical_json, sha256_value

from .contracts import EvaluatorConfigV1

ModelT = TypeVar("ModelT", bound=BaseModel)


class EvaluationJudgeError(RuntimeError):
    """Raised after a judge cannot produce a valid typed response."""


class CachedEvaluatorClient:
    def __init__(self, config: EvaluatorConfigV1) -> None:
        self.config = config
        self.client: JsonModelClient = self._client(config)
        self.last_cache_hit = False

    @property
    def model_name(self) -> str:
        return self.client.model_name

    def generate_model(
        self,
        *,
        system: str,
        user: str,
        response_model: type[ModelT],
        attempts: int = 2,
    ) -> ModelT:
        schema = response_model.model_json_schema()
        path = self._cache_path(system, user, schema)
        self.last_cache_hit = False
        if path.is_file():
            try:
                value = response_model.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                self.last_cache_hit = True
                return value
            except (OSError, ValidationError, ValueError):
                path.unlink(missing_ok=True)
        last_error: Exception | None = None
        for attempt in range(max(1, attempts)):
            repair = ""
            if attempt:
                repair = (
                    "\nThe prior response was invalid. Return exactly one object "
                    "matching the supplied schema, with no omitted items."
                )
            try:
                payload = self.client.generate_json(
                    system=system + repair,
                    user=user,
                    response_schema=schema,
                )
                value = response_model.model_validate(payload)
                self._write_cache(path, value)
                return value
            except (
                OSError,
                RuntimeError,
                TypeError,
                ValidationError,
                ValueError,
            ) as exc:
                last_error = exc
        raise EvaluationJudgeError(
            f"judge_failed:{self.model_name}:{type(last_error).__name__}:{last_error}"
        )

    def fingerprint(self) -> dict[str, Any]:
        endpoint: Any = (
            self.config.codex_cli
            if self.config.backend == "codex_cli"
            else self.config.openai_compatible
        )
        payload = endpoint.model_dump(mode="json") if endpoint is not None else {}
        payload.pop("api_key_env", None)
        return {
            "backend": self.config.backend,
            "endpoint": payload,
            "model": self.model_name,
        }

    def _cache_path(self, system: str, user: str, schema: dict[str, Any]) -> Path:
        identity = {
            "contract": "gear_evaluator_cache_v1",
            "client": self.fingerprint(),
            "system_sha256": sha256_value(system),
            "user_sha256": sha256_value(user),
            "schema_sha256": sha256_value(schema),
        }
        digest = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
        return self.config.cache_dir.resolve() / "judge_responses" / f"{digest}.json"

    @staticmethod
    def _write_cache(path: Path, value: BaseModel) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(value.model_dump_json(), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _client(config: EvaluatorConfigV1) -> JsonModelClient:
        if config.backend == "codex_cli":
            if config.codex_cli is None:
                raise ValueError("codex_cli evaluator endpoint missing")
            return CodexCliJsonClient(config.codex_cli, cache_dir=None)
        if config.openai_compatible is None:
            raise ValueError("openai-compatible evaluator endpoint missing")
        return OpenAICompatibleJsonClient(config.openai_compatible)


def load_evaluator_config(path: Path) -> EvaluatorConfigV1:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    base = Path(path).resolve().parent
    cache = Path(raw["cache_dir"])
    raw["cache_dir"] = str(cache if cache.is_absolute() else (base / cache).resolve())
    return EvaluatorConfigV1.model_validate(raw)


__all__ = [
    "CachedEvaluatorClient",
    "EvaluationJudgeError",
    "load_evaluator_config",
]
