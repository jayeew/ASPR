"""Configuration and fail-closed evidence policy for ASPR-GEAR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import Field, field_validator, model_validator

from .contracts import StrictModel
from .env import getenv

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


class AssetPaths(StrictModel):
    official_run_manifest: Path
    official_model_json: Path
    official_model_joblib: Path
    official_score_table: Path
    feature_matrix_16: Path
    matrix_manifest: Path
    matrix_input_snapshot: Path
    paper_metadata: Path
    oof_metrics: Path
    oof_fold_metrics: Path
    oof_domain_metrics: Path


class RetrievalLimits(StrictModel):
    normal_max: int = Field(default=2, ge=0)
    contrastive_max: int = Field(default=1, ge=0)
    citation_expansion_max: int = Field(default=1, ge=0)
    fulltext_max: int = Field(default=12, ge=0)
    provider_limit: int = Field(default=50, ge=1)


class GearConfig(StrictModel):
    config_version: str
    evidence_policy: str
    current_fig1_3_only: bool = True
    deprecated_fig4_to_fig10_used: bool = False
    calibration_registry: Path
    calibration_release: str
    runtime_replay_manifest: Path
    runtime_replay_manifest_sha256: str
    allowed_asset_roots: List[Path]
    denied_path_fragments: List[str]
    model_backend: Literal["codex_cli", "openai_compatible"] = "codex_cli"
    codex_cli: CodexCliEndpoint
    openai_compatible: Optional[OpenAICompatibleEndpoint] = None
    retrieval: RetrievalLimits = Field(default_factory=RetrievalLimits)
    allow_external_retrieval: bool = True
    cache_dir: Path
    output_root: Path
    minimum_pdf_characters: int = 1500
    minimum_nonempty_page_ratio: float = 0.5
    high_aspr_threshold: float = 90.0
    profile_low_quantile: float = 0.1
    profile_high_quantile: float = 0.9
    max_claims: int = 12

    @model_validator(mode="after")
    def selected_model_backend_is_configured(self) -> "GearConfig":
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

    def resolve_path(self, value: Union[Path, str]) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def resolved_assets(self) -> AssetPaths:
        from .calibration_assets import load_calibration_release

        release = load_calibration_release(
            self.calibration_release,
            registry_path=self.resolve_path(self.calibration_registry),
        )
        return AssetPaths.model_validate(release.core_paths())

    def resolved_calibration_release(self) -> Any:
        """Return the public release object for experiments needing extra assets."""
        from .calibration_assets import load_calibration_release

        return load_calibration_release(
            self.calibration_release,
            registry_path=self.resolve_path(self.calibration_registry),
        )

    def resolved_runtime_replay_manifest(self) -> Path:
        """Resolve the separately frozen online-materialization replay gate."""
        return self.validate_asset_path(self.resolve_path(self.runtime_replay_manifest))

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


def _recursive_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _recursive_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    path: Optional[Path] = None,
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> GearConfig:
    """Load deterministic defaults and non-secret environment overrides."""
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
        value = getenv(environment_name)
        if value:
            payload["codex_cli"][field_name] = (
                int(value) if field_name == "timeout_seconds" else value
            )
    backend = getenv("ASPR_GEAR_MODEL_BACKEND")
    if backend:
        payload["model_backend"] = backend
    api_env = {
        "base_url": "ASPR_GEAR_API_BASE_URL",
        "model": "ASPR_GEAR_API_MODEL",
        "api_key_env": "ASPR_GEAR_API_KEY_ENV",
        "timeout_seconds": "ASPR_GEAR_API_TIMEOUT_SECONDS",
    }
    for field_name, environment_name in api_env.items():
        value = getenv(environment_name)
        if value:
            api_payload = payload.get("openai_compatible")
            if not isinstance(api_payload, dict):
                api_payload = {}
                payload["openai_compatible"] = api_payload
            api_payload[field_name] = (
                int(value) if field_name == "timeout_seconds" else value
            )
    retrieval_value = getenv("ASPR_GEAR_ALLOW_EXTERNAL_RETRIEVAL")
    if retrieval_value:
        payload["allow_external_retrieval"] = retrieval_value.strip().casefold() in {
            "1",
            "true",
            "yes",
            "on",
        }
    for field_name, environment_name in (
        ("cache_dir", "ASPR_GEAR_CACHE_DIR"),
        ("output_root", "ASPR_GEAR_OUTPUT_ROOT"),
        ("calibration_registry", "ASPR_GEAR_CALIBRATION_REGISTRY"),
        ("runtime_replay_manifest", "ASPR_GEAR_RUNTIME_REPLAY_MANIFEST"),
    ):
        value = getenv(environment_name)
        if value:
            payload[field_name] = value
    calibration_release = getenv("ASPR_GEAR_CALIBRATION_RELEASE")
    if calibration_release:
        payload["calibration_release"] = calibration_release
    for field_name, environment_name in (
        ("normal_max", "ASPR_GEAR_RETRIEVAL_NORMAL_MAX"),
        ("contrastive_max", "ASPR_GEAR_RETRIEVAL_CONTRASTIVE_MAX"),
        ("citation_expansion_max", "ASPR_GEAR_RETRIEVAL_CITATION_MAX"),
        ("fulltext_max", "ASPR_GEAR_RETRIEVAL_EVIDENCE_MAX"),
    ):
        value = getenv(environment_name)
        if value:
            payload["retrieval"][field_name] = int(value)
    config = GearConfig.model_validate(payload)
    if config.deprecated_fig4_to_fig10_used:
        raise ValueError("deprecated Fig.4-Fig.10 evidence cannot be enabled")
    for asset in config.resolved_assets().model_dump().values():
        config.validate_asset_path(Path(asset))
    return config


__all__ = [
    "CodexCliEndpoint",
    "GearConfig",
    "OpenAICompatibleEndpoint",
    "load_config",
]
