"""Configuration and fail-closed evidence policy for ASPR-GEAR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .contracts import StrictModel
from .env import getenv, getenv_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "gear" / "default.json"


class CodexCliEndpoint(StrictModel):
    """Configuration for one isolated Codex CLI review session."""

    executable: str = "codex"
    model: str = "gpt-5.6-terra"
    reasoning_effort: Literal["low", "medium", "high", "xhigh"] = "high"
    timeout_seconds: int = 1800
    sandbox: Literal["read-only"] = "read-only"


class OpenAICompatibleEndpoint(StrictModel):
    """Stateless JSON endpoint compatible with the chat-completions API."""

    base_url: str
    model: str
    api_key_env: str = "ASPR_GEAR_API_KEY"
    timeout_seconds: int = 1800


class ASPRQwenConfig(StrictModel):
    """Optional graph-blind auxiliary reviewer endpoint."""

    enabled: bool = False
    backend: Literal["openai_compatible"] = "openai_compatible"
    model: str = "aspr-qwen-reviewer"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_env: str = "ASPR_QWEN_API_KEY"
    timeout_seconds: int = 1800
    required: bool = False


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
    recall_model_path: Path = Path("/home/jayee/models/bge-m3")
    reranker_model_path: Path = Path("/home/jayee/models/OpenScholar_Reranker")
    ranking_algorithm_fingerprint: str = "dual_view_bge_m3_openscholar_v1"
    openalex_pdf_enabled: bool = False
    openalex_pdf_max_downloads: int = Field(default=3, ge=0, le=12)
    openalex_pdf_max_bytes: int = Field(default=25_000_000, ge=1_000_000)
    openalex_pdf_max_pages: int = Field(default=100, ge=1)
    openalex_pdf_max_characters: int = Field(default=30_000, ge=1_000)


class GraphGuidanceConfig(StrictModel):
    policy_version: str = "primary16_forecast_calibration_v1"
    action_policy_enabled: bool = False
    shadow: bool = False
    score_routing_enabled: bool = False
    topology_enabled: bool = True
    calibration_enabled: bool = False
    calibration_variant: Literal[
        "neutral",
        "topology_only",
        "scalar_score",
        "hgb_analog",
        "full_calibrated",
        "shuffled_hgb",
    ] = "topology_only"
    provider_searches: int = Field(default=8, ge=0)
    direct_fetches: int = Field(default=8, ge=0)
    neighbor_expansions: int = Field(default=2, ge=0)
    fulltext_candidates: int = Field(default=12, ge=0)
    relation_classifications: int = Field(default=12, ge=0)


class ClaimAttributionConfig(StrictModel):
    """Choose the declared T0 baseline or a promoted learned head."""

    mode: Literal["deterministic_t0", "learned_t0"] = "deterministic_t0"
    learned_manifest: Path | None = None


class GearConfig(StrictModel):
    config_version: str
    evidence_policy: str
    current_fig1_3_only: bool = True
    deprecated_fig4_to_fig10_used: bool = False
    forecast_release_manifest: Path
    forecast_runtime_manifest: Path | None = None
    forecast_anatomy_manifest: Path | None = None
    structural_head_manifest: Path | None = None
    graph_action_policy_manifest: Path | None = None
    graph_forecast_enabled: bool = True
    graph_results_path: Path | None = None
    allowed_asset_roots: list[Path]
    denied_path_fragments: list[str]
    model_backend: Literal["codex_cli", "openai_compatible"] = "codex_cli"
    codex_cli: CodexCliEndpoint
    openai_compatible: OpenAICompatibleEndpoint | None = None
    aspr_qwen: ASPRQwenConfig = Field(default_factory=ASPRQwenConfig)
    retrieval: RetrievalLimits = Field(default_factory=RetrievalLimits)
    graph_guidance: GraphGuidanceConfig = Field(default_factory=GraphGuidanceConfig)
    claim_attribution: ClaimAttributionConfig = Field(
        default_factory=ClaimAttributionConfig
    )
    allow_external_retrieval: bool = True
    cache_dir: Path
    output_root: Path
    minimum_pdf_characters: int = 1500
    minimum_nonempty_page_ratio: float = 0.5
    max_claims: int = 12

    @model_validator(mode="after")
    def selected_model_backend_is_configured(self) -> GearConfig:
        if self.model_backend == "openai_compatible" and self.openai_compatible is None:
            raise ValueError("openai_compatible configuration is required for API mode")
        return self

    @field_validator("evidence_policy")
    @classmethod
    def current_policy_only(cls, value: str) -> str:
        if value != "fig1_fig2_fig3_current_only":
            raise ValueError("GEAR only accepts current Fig.1-Fig.3 evidence")
        return value

    @field_validator("current_fig1_3_only")
    @classmethod
    def current_assets_only(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("current_fig1_3_only cannot be disabled")
        return value

    def resolve_path(self, value: Path | str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def resolved_forecast_release_manifest(self) -> Path:
        return self.validate_asset_path(
            self.resolve_path(self.forecast_release_manifest)
        )

    def resolved_forecast_runtime_manifest(self) -> Path | None:
        if self.forecast_runtime_manifest is None:
            return None
        return self.validate_asset_path(
            self.resolve_path(self.forecast_runtime_manifest)
        )

    def resolved_forecast_anatomy_manifest(self) -> Path | None:
        if self.forecast_anatomy_manifest is None:
            return None
        return self.validate_asset_path(
            self.resolve_path(self.forecast_anatomy_manifest)
        )

    def resolved_structural_head_manifest(self) -> Path | None:
        if self.structural_head_manifest is None:
            return None
        return self.validate_asset_path(
            self.resolve_path(self.structural_head_manifest)
        )

    def resolved_claim_attribution_manifest(self) -> Path | None:
        if self.claim_attribution.learned_manifest is None:
            return None
        return self.validate_asset_path(
            self.resolve_path(self.claim_attribution.learned_manifest)
        )

    def resolved_graph_action_policy_manifest(self) -> Path | None:
        if self.graph_action_policy_manifest is None:
            return None
        return self.validate_asset_path(
            self.resolve_path(self.graph_action_policy_manifest)
        )

    def validate_asset_path(self, value: Path) -> Path:
        path = self.resolve_path(value)
        normalized = path.as_posix().casefold()
        for fragment in self.denied_path_fragments:
            if fragment.casefold() in normalized:
                raise ValueError(f"deprecated evidence path is forbidden: {path}")
        allowed = [self.resolve_path(root) for root in self.allowed_asset_roots]
        if not any(_is_relative_to(path, root) for root in allowed):
            raise ValueError(
                f"asset path is outside current evidence allowlist: {path}"
            )
        return path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _recursive_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _recursive_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    path: Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    validate_assets: bool = False,
) -> GearConfig:
    """Load configuration without touching heavyweight Graph assets by default."""
    payload = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        override_payload = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
        payload = _recursive_update(payload, override_payload)
    if overrides:
        payload = _recursive_update(payload, overrides)
    codex_env = {
        "executable": "ASPR_GEAR_CODEX_EXECUTABLE",
        "model": "ASPR_GEAR_CODEX_MODEL",
        "reasoning_effort": "ASPR_GEAR_CODEX_REASONING_EFFORT",
        "timeout_seconds": "ASPR_GEAR_CODEX_TIMEOUT_SECONDS",
    }
    for field_name, environment_name in codex_env.items():
        value = getenv_runtime(environment_name)
        if value:
            payload["codex_cli"][field_name] = (
                int(value) if field_name == "timeout_seconds" else value
            )
    backend = getenv_runtime("ASPR_GEAR_MODEL_BACKEND")
    explicit_api_endpoint = getenv_runtime("ASPR_GEAR_API_BASE_URL")
    if backend or explicit_api_endpoint:
        payload["model_backend"] = backend or "openai_compatible"
    api_env = {
        "base_url": "ASPR_GEAR_API_BASE_URL",
        "model": "ASPR_GEAR_API_MODEL",
        "api_key_env": "ASPR_GEAR_API_KEY_ENV",
        "timeout_seconds": "ASPR_GEAR_API_TIMEOUT_SECONDS",
    }
    for field_name, environment_name in api_env.items():
        value = getenv_runtime(environment_name)
        if value:
            api_payload = payload.get("openai_compatible")
            if not isinstance(api_payload, dict):
                api_payload = {}
                payload["openai_compatible"] = api_payload
            api_payload[field_name] = (
                int(value) if field_name == "timeout_seconds" else value
            )
    retrieval_value = getenv_runtime("ASPR_GEAR_ALLOW_EXTERNAL_RETRIEVAL")
    if retrieval_value:
        payload["allow_external_retrieval"] = retrieval_value.strip().casefold() in {
            "1",
            "true",
            "yes",
            "on",
        }
    for field_name, environment_name in (
        ("recall_model_path", "ASPR_RECALL_MODEL_PATH"),
        ("reranker_model_path", "ASPR_RERANKER_MODEL_PATH"),
    ):
        value = getenv(environment_name)
        if value:
            payload["retrieval"][field_name] = value
    for field_name, environment_name in (
        ("cache_dir", "ASPR_GEAR_CACHE_DIR"),
        ("output_root", "ASPR_GEAR_OUTPUT_ROOT"),
        ("forecast_release_manifest", "ASPR_GEAR_FORECAST_RELEASE_MANIFEST"),
        ("forecast_runtime_manifest", "ASPR_GEAR_FORECAST_RUNTIME_MANIFEST"),
        (
            "graph_action_policy_manifest",
            "ASPR_GEAR_GRAPH_ACTION_POLICY_MANIFEST",
        ),
    ):
        value = getenv_runtime(environment_name)
        if value:
            payload[field_name] = value
    graph_results_path = getenv_runtime("ASPR_GEAR_GRAPH_RESULTS_PATH")
    if graph_results_path:
        payload["graph_results_path"] = graph_results_path
    attribution_mode = getenv_runtime("ASPR_GEAR_CLAIM_ATTRIBUTION_MODE")
    if attribution_mode:
        payload.setdefault("claim_attribution", {})["mode"] = attribution_mode
    attribution_manifest = getenv_runtime("ASPR_GEAR_CLAIM_ATTRIBUTION_MANIFEST")
    if attribution_manifest:
        payload.setdefault("claim_attribution", {})[
            "learned_manifest"
        ] = attribution_manifest
    qwen_env = {
        "model": "ASPR_QWEN_MODEL",
        "base_url": "ASPR_QWEN_BASE_URL",
        "api_key_env": "ASPR_QWEN_API_KEY_ENV",
        "timeout_seconds": "ASPR_QWEN_TIMEOUT_SECONDS",
    }
    for field_name, environment_name in qwen_env.items():
        value = getenv_runtime(environment_name)
        if value:
            payload.setdefault("aspr_qwen", {})[field_name] = (
                int(value) if field_name == "timeout_seconds" else value
            )
    qwen_enabled = getenv_runtime("ASPR_QWEN_ENABLED")
    if qwen_enabled:
        payload.setdefault("aspr_qwen", {})[
            "enabled"
        ] = qwen_enabled.strip().casefold() in {"1", "true", "yes", "on"}
    for field_name, environment_name in (
        ("normal_max", "ASPR_GEAR_RETRIEVAL_NORMAL_MAX"),
        ("contrastive_max", "ASPR_GEAR_RETRIEVAL_CONTRASTIVE_MAX"),
        ("citation_expansion_max", "ASPR_GEAR_RETRIEVAL_CITATION_MAX"),
        ("fulltext_max", "ASPR_GEAR_RETRIEVAL_EVIDENCE_MAX"),
        ("lexical_candidate_limit", "ASPR_GEAR_RETRIEVAL_LEXICAL_CANDIDATES"),
        ("semantic_candidate_limit", "ASPR_GEAR_RETRIEVAL_SEMANTIC_CANDIDATES"),
        ("rerank_candidate_limit", "ASPR_GEAR_RETRIEVAL_RERANK_CANDIDATES"),
        ("retained_candidates_per_claim", "ASPR_GEAR_RETRIEVAL_RETAINED_PER_CLAIM"),
    ):
        value = getenv_runtime(environment_name)
        if value:
            payload["retrieval"][field_name] = int(value)
    openalex_pdf_enabled = getenv_runtime("ASPR_GEAR_OPENALEX_PDF_ENABLED")
    if openalex_pdf_enabled:
        payload["retrieval"][
            "openalex_pdf_enabled"
        ] = openalex_pdf_enabled.strip().casefold() in {"1", "true", "yes", "on"}
    for field_name, environment_name in (
        ("openalex_pdf_max_downloads", "ASPR_GEAR_OPENALEX_PDF_MAX_DOWNLOADS"),
        ("openalex_pdf_max_bytes", "ASPR_GEAR_OPENALEX_PDF_MAX_BYTES"),
        ("openalex_pdf_max_pages", "ASPR_GEAR_OPENALEX_PDF_MAX_PAGES"),
        (
            "openalex_pdf_max_characters",
            "ASPR_GEAR_OPENALEX_PDF_MAX_CHARACTERS",
        ),
    ):
        value = getenv_runtime(environment_name)
        if value:
            payload["retrieval"][field_name] = int(value)
    config = GearConfig.model_validate(payload)
    if config.deprecated_fig4_to_fig10_used:
        raise ValueError("deprecated Fig.4-Fig.10 evidence cannot be enabled")
    if validate_assets:
        from .diffusion_forecast import (
            ForecastRelease,
            RuntimeFeatureRelease,
            StructuralHeadRelease,
        )
        from .graph_calibration import load_forecast_analog_index

        release = ForecastRelease(config.resolved_forecast_release_manifest())
        release.verify()
        runtime_manifest = config.resolved_forecast_runtime_manifest()
        runtime_release = None
        if runtime_manifest is not None:
            runtime_release = RuntimeFeatureRelease(runtime_manifest)
            runtime_release.verify(release)
        anatomy_manifest = config.resolved_forecast_anatomy_manifest()
        if anatomy_manifest is not None:
            load_forecast_analog_index(anatomy_manifest)
        structural_manifest = config.resolved_structural_head_manifest()
        if structural_manifest is not None:
            StructuralHeadRelease(structural_manifest).verify(release, runtime_release)
        resolved_attribution_manifest = config.resolved_claim_attribution_manifest()
        if config.claim_attribution.mode == "learned_t0":
            if resolved_attribution_manifest is None:
                raise ValueError(
                    "learned claim attribution requires a promoted manifest"
                )
            from .claim_attribution import load_claim_attribution_release

            load_claim_attribution_release(resolved_attribution_manifest)
        action_policy_manifest = config.resolved_graph_action_policy_manifest()
        if action_policy_manifest is not None:
            from .graph_action_policy import load_graph_action_policy_release

            load_graph_action_policy_release(action_policy_manifest)
        elif config.graph_guidance.action_policy_enabled:
            raise ValueError("enabled Graph action policy requires a promoted manifest")
    return config


__all__ = [
    "ASPRQwenConfig",
    "ClaimAttributionConfig",
    "CodexCliEndpoint",
    "GearConfig",
    "OpenAICompatibleEndpoint",
    "load_config",
]
