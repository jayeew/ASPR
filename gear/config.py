"""Runtime configuration for Claim Graph + full-text GEAR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from .contracts import StrictModel
from .env import getenv, getenv_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gear" / "default.json"


class CodexCliEndpoint(StrictModel):
    executable: str = "codex"
    model: str = "gpt-5.6-terra"
    reasoning_effort: Literal["low", "medium", "high", "xhigh"] = "high"
    timeout_seconds: int = 1800
    sandbox: Literal["read-only"] = "read-only"


class OpenAICompatibleEndpoint(StrictModel):
    base_url: str
    model: str
    api_key_env: str = "ASPR_GEAR_API_KEY"
    timeout_seconds: int = 1800


class RetrievalLimits(StrictModel):
    normal_max: int = Field(default=4, ge=0)
    contrastive_max: int = Field(default=1, ge=0)
    citation_expansion_max: int = Field(default=1, ge=0)
    fulltext_max: int = Field(default=12, ge=0)
    provider_limit: int = Field(default=50, ge=1)
    relation_cards_max: int = Field(default=24, ge=1)
    total_actions_max: int = Field(default=48, ge=1)
    scientific_query_enabled: bool = True
    query_reasoning_effort: Literal["low", "medium", "high", "xhigh"] = "medium"
    lexical_candidate_limit: int = Field(default=20, ge=1, le=100)
    semantic_candidate_limit: int = Field(default=30, ge=1, le=50)
    candidate_union_limit: int = Field(default=120, ge=20, le=500)
    embedding_candidate_limit: int = Field(default=100, ge=10, le=200)
    rerank_candidate_limit: int = Field(default=24, ge=1, le=100)
    dual_rerank_top_k: int = Field(default=15, ge=1, le=50)
    retained_candidates_per_claim: int = Field(default=10, ge=1, le=24)
    per_family_retained_max: int = Field(default=3, ge=1, le=12)
    minimum_comparable_candidates: int = Field(default=10, ge=1, le=24)
    minimum_unique_candidates: int = Field(default=20, ge=1, le=120)
    local_recall_enabled: bool = True
    local_reranker_enabled: bool = True
    recall_model_path: Path = Path("data/models/Qwen3-Embedding-4B")
    reranker_model_path: Path = Path("/home/jayee/models/OpenScholar_Reranker")
    ranking_algorithm_fingerprint: str = "qwen3_openscholar_claim_prior_v1"
    openalex_pdf_enabled: bool = False
    openalex_pdf_max_downloads: int = Field(default=3, ge=0, le=12)
    openalex_pdf_max_bytes: int = Field(default=25_000_000, ge=1_000_000)
    openalex_pdf_max_pages: int = Field(default=100, ge=1)
    openalex_pdf_max_characters: int = Field(default=30_000, ge=1_000)


class GearConfig(StrictModel):
    model_backend: Literal["codex_cli", "openai_compatible"] = "codex_cli"
    codex_cli: CodexCliEndpoint = Field(default_factory=CodexCliEndpoint)
    openai_compatible: OpenAICompatibleEndpoint | None = None
    retrieval: RetrievalLimits = Field(default_factory=RetrievalLimits)
    allow_external_retrieval: bool = True
    cache_dir: Path = Path("data/gear_cache")
    output_root: Path = Path("outputs/gear/runs")
    minimum_pdf_characters: int = 1500
    minimum_nonempty_page_ratio: float = 0.5
    max_claims: int = 12

    @model_validator(mode="after")
    def endpoint_available(self) -> "GearConfig":
        if self.model_backend == "openai_compatible" and self.openai_compatible is None:
            raise ValueError("openai_compatible endpoint is required")
        return self

    def resolve_path(self, value: Path | str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    output = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _merge(output[key], value)
        else:
            output[key] = value
    return output


def load_config(
    path: Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    validate_assets: bool = False,
) -> GearConfig:
    """Load model and retrieval settings; no release or asset validation exists."""
    del validate_assets
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path is not None:
        payload = _merge(payload, json.loads(path.read_text(encoding="utf-8")))
    if overrides:
        payload = _merge(payload, overrides)
    backend = getenv_runtime("ASPR_GEAR_MODEL_BACKEND")
    if backend:
        payload["model_backend"] = backend
    codex = payload.setdefault("codex_cli", {})
    for field, variable in {
        "executable": "ASPR_GEAR_CODEX_EXECUTABLE",
        "model": "ASPR_GEAR_CODEX_MODEL",
        "reasoning_effort": "ASPR_GEAR_CODEX_REASONING_EFFORT",
        "timeout_seconds": "ASPR_GEAR_CODEX_TIMEOUT_SECONDS",
    }.items():
        value = getenv_runtime(variable)
        if value:
            codex[field] = int(value) if field == "timeout_seconds" else value
    api_url = getenv_runtime("ASPR_GEAR_API_BASE_URL")
    api_model = getenv_runtime("ASPR_GEAR_API_MODEL")
    if api_url or api_model:
        endpoint = payload.setdefault("openai_compatible", {})
        if api_url:
            endpoint["base_url"] = api_url
        if api_model:
            endpoint["model"] = api_model
    for field, variable in {
        "recall_model_path": "ASPR_RECALL_MODEL_PATH",
        "reranker_model_path": "ASPR_RERANKER_MODEL_PATH",
    }.items():
        value = getenv(variable)
        if value:
            payload.setdefault("retrieval", {})[field] = value
    return GearConfig.model_validate(payload)


__all__ = [
    "CodexCliEndpoint",
    "GearConfig",
    "OpenAICompatibleEndpoint",
    "RetrievalLimits",
    "load_config",
]
