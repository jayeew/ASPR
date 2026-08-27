"""Lazy, fail-closed access to the frozen Fig.2/3 Primary16 D5 forecast."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import Field

from .contracts import PaperIR, StrictModel
from .graph_prior_contracts import GraphRuntimePacket, InfluenceForecast


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


class ForecastAsset(StrictModel):
    file: str
    sha256: str
    size_bytes: int = Field(ge=0)


class ForecastReleaseManifest(StrictModel):
    contract: str
    alias: str
    status: str
    runtime_status: str
    protocol_version: str
    model_id: str
    horizon_years: int
    score_semantics: str
    training_row_count: int = Field(ge=1)
    oof_row_count: int = Field(ge=1)
    assets: dict[str, ForecastAsset]
    release_id: str
    created_at_utc: str
    source_release: str | None = None


class RuntimeFeatureAsset(StrictModel):
    file: str
    sha256: str


class RuntimeFeatureManifest(StrictModel):
    contract: str
    release_id: str
    feature_protocol_version: str
    target_count: int = Field(ge=1)
    feature_names: list[str]
    source_max_year: int = Field(ge=0)
    feature_time_basis: str
    classification_source: str
    frozen_context_max_year: int = Field(ge=0)
    post_context_target_policy: Literal["recompute_primary16_then_frozen_hgb"]
    target_features_recomputed: bool
    model_refit: bool
    historical_context_limited_for: list[str] = Field(default_factory=list)
    future_citation_counts_used: bool
    network_used_for_target_freeze: bool
    created_at_utc: str
    assets: dict[str, RuntimeFeatureAsset]


class ForecastRelease:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.manifest = ForecastReleaseManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )

    def path(self, name: str, *, verify: bool = True) -> Path:
        asset = self.manifest.assets[name]
        path = (self.root / asset.file).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"unsafe forecast asset path: {path}") from exc
        if not path.is_file() or path.stat().st_size != asset.size_bytes:
            raise FileNotFoundError(path)
        if verify and sha256_file(path) != asset.sha256:
            raise ValueError(f"forecast asset hash mismatch: {name}")
        return path

    def verify(self) -> dict[str, Any]:
        required = {
            "model",
            "feature_registry",
            "training_snapshot",
            "score_table",
            "percentile_reference",
            "runtime_replay_matrix",
            "oof_predictions",
            "oof_metrics",
            "oof_fold_metrics",
            "oof_domain_metrics",
            "temporal_folds",
            "oof_run_manifest",
            "registry_freeze_manifest",
        }
        missing = sorted(required - set(self.manifest.assets))
        if missing:
            raise ValueError(f"forecast release lacks assets: {missing}")
        for name in self.manifest.assets:
            self.path(name)
        reference = pd.read_parquet(self.path("percentile_reference"))[
            "expected_diffusion_score"
        ].to_numpy(dtype=float)
        if not len(reference) or np.any(np.diff(reference) < 0):
            raise ValueError("percentile reference is empty or non-monotone")
        folds = pd.read_csv(self.path("temporal_folds"))
        if not (folds["train_year_max"] < folds["test_year_min"]).all():
            raise ValueError("OOF temporal-fold leakage detected")
        fold_count = int(folds["fold_id"].nunique())
        if fold_count < 2:
            raise ValueError("forecast release must contain temporal folds")
        return {
            "passed": True,
            "release_id": self.manifest.release_id,
            "assets_verified": len(self.manifest.assets),
            "training_rows": self.manifest.training_row_count,
            "oof_rows": self.manifest.oof_row_count,
            "temporal_folds": fold_count,
        }


class RuntimeFeatureRelease:
    """Target-only feature inference tied to one immutable HGB release."""

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.manifest = RuntimeFeatureManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )

    def path(self, name: str) -> Path:
        asset = self.manifest.assets[name]
        path = (self.root / asset.file).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"unsafe runtime feature asset path: {path}") from exc
        if not path.is_file() or sha256_file(path) != asset.sha256:
            raise ValueError(f"runtime feature asset mismatch: {name}")
        return path

    def verify(self, release: ForecastRelease) -> dict[str, Any]:
        if self.manifest.release_id != release.manifest.release_id:
            raise ValueError("runtime features target a different HGB release")
        if self.manifest.feature_protocol_version != release.manifest.protocol_version:
            raise ValueError("runtime feature protocol mismatch")
        if not self.manifest.target_features_recomputed or self.manifest.model_refit:
            raise ValueError("runtime must recompute Primary16 without refitting HGB")
        required = {"runtime_score_table", "runtime_feature_table"}
        if not required.issubset(self.manifest.assets):
            raise ValueError("runtime feature release lacks score/feature table")
        for name in self.manifest.assets:
            self.path(name)
        scores = pd.read_parquet(self.path("runtime_score_table"))
        if len(scores) != self.manifest.target_count:
            raise ValueError("runtime target count mismatch")
        return {"passed": True, "runtime_targets": len(scores)}


class DiffusionForecastService:
    """Exact release lookup with an explicit unavailable result for every failure."""

    def __init__(
        self,
        manifest_path: Path,
        runtime_manifest_path: Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.runtime_manifest_path = (
            Path(runtime_manifest_path).resolve()
            if runtime_manifest_path is not None
            else None
        )
        self._release: ForecastRelease | None = None
        self._scores: pd.DataFrame | None = None

    def _load(self) -> tuple[ForecastRelease, pd.DataFrame]:
        if self._release is None:
            self._release = ForecastRelease(self.manifest_path)
            self._release.verify()
            self._scores = pd.read_parquet(self._release.path("score_table"))
            self._scores["feature_source"] = "release_score_table"
            if self.runtime_manifest_path is not None:
                runtime = RuntimeFeatureRelease(self.runtime_manifest_path)
                runtime.verify(self._release)
                runtime_scores = pd.read_parquet(runtime.path("runtime_score_table"))
                self._scores = pd.concat(
                    [self._scores, runtime_scores], ignore_index=True, sort=False
                )
        assert self._scores is not None
        return self._release, self._scores

    def score(self, paper_ir: PaperIR, cutoff_date: date) -> GraphRuntimePacket:
        try:
            release, scores = self._load()
            identifiers = _identifiers(paper_ir)
            matches = scores.loc[
                scores["paper_id"].astype(str).map(_normalize).isin(identifiers)
            ]
            if len(matches) != 1:
                reason = (
                    "forecast_exact_lookup_missing"
                    if matches.empty
                    else "forecast_identity_ambiguous"
                )
                return unavailable_packet(paper_ir.paper_id, cutoff_date, reason)
            row = matches.iloc[0]
            runtime_inference = str(row.get("feature_source", "")).startswith(
                "runtime_recomputed_"
            )
            target_date = (
                paper_ir.metadata.submission_date
                or paper_ir.metadata.publication_date
                or cutoff_date
            )
            if target_date.year > 2022 and not runtime_inference:
                return unavailable_packet(
                    paper_ir.paper_id,
                    cutoff_date,
                    "post_2022_requires_runtime_primary16_recompute",
                )
            if (
                runtime_inference
                and str(row.get("as_of_date"))[:10] != cutoff_date.isoformat()
            ):
                return unavailable_packet(
                    paper_ir.paper_id, cutoff_date, "runtime_feature_cutoff_mismatch"
                )
            if (
                not runtime_inference
                and int(row["publication_year"]) >= cutoff_date.year
            ):
                return unavailable_packet(
                    paper_ir.paper_id, cutoff_date, "forecast_target_not_pre_cutoff"
                )
            manifest = release.manifest
            coverage = float(row.get("feature_coverage") or 1.0)
            diagnostics = (
                [
                    "runtime_feature_inference",
                    "target_primary16_recomputed",
                    "frozen_primary16_hgb_reused",
                ]
                if runtime_inference
                else []
            )
            source_max_year = row.get("source_max_year")
            if runtime_inference and pd.notna(source_max_year):
                diagnostics.append(
                    f"historical_context_source_max_year:{int(source_max_year)}"
                )
                if int(source_max_year) < cutoff_date.year - 1:
                    diagnostics.append("historical_context_limited")
            return GraphRuntimePacket(
                paper_id=paper_ir.paper_id,
                cutoff_date=cutoff_date,
                forecast=InfluenceForecast(
                    status="available",
                    prospective_5y_diffusion_percentile=float(
                        row["prospective_5y_diffusion_percentile"]
                    ),
                    uptake_probability=float(row["uptake_probability_calibrated"]),
                    conditional_diffusion=float(
                        row["conditional_diffusion_calibrated"]
                    ),
                    expected_diffusion=float(row["expected_diffusion_score"]),
                    feature_coverage=coverage,
                    release_id=manifest.release_id,
                    model_sha256=manifest.assets["model"].sha256,
                    feature_registry_sha256=manifest.assets["feature_registry"].sha256,
                    training_snapshot_sha256=manifest.assets[
                        "training_snapshot"
                    ].sha256,
                    percentile_reference_sha256=manifest.assets[
                        "percentile_reference"
                    ].sha256,
                    diagnostics=diagnostics,
                ),
                diagnostics=diagnostics,
            )
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
            return unavailable_packet(
                paper_ir.paper_id,
                cutoff_date,
                f"forecast_release_unavailable:{type(exc).__name__}",
            )


def unavailable_packet(
    paper_id: str, cutoff_date: date, reason: str
) -> GraphRuntimePacket:
    return GraphRuntimePacket(
        paper_id=paper_id,
        cutoff_date=cutoff_date,
        forecast=InfluenceForecast(status="unavailable", diagnostics=[reason]),
        diagnostics=["graph_limited", reason],
    )


def validate_runtime_replay(manifest_path: Path) -> dict[str, Any]:
    release = ForecastRelease(manifest_path)
    report = release.verify()
    model = joblib.load(release.path("model"))
    snapshot = pd.read_parquet(release.path("training_snapshot"))
    replay = pd.read_parquet(release.path("runtime_replay_matrix"))
    selected = snapshot.loc[
        snapshot["paper_id"].astype(str).isin(set(replay["paper_id"].astype(str)))
    ].sort_values("paper_id", kind="stable")
    expected = replay.sort_values("paper_id", kind="stable")
    observed = _predict_model(model, selected)
    columns = (
        "uptake_probability_calibrated",
        "conditional_diffusion_calibrated",
        "expected_diffusion_score",
    )
    max_error = max(
        float(
            np.max(
                np.abs(
                    observed[column].to_numpy(dtype=float)
                    - expected[column].to_numpy(dtype=float)
                )
            )
        )
        for column in columns
    )
    if max_error > 1e-12:
        raise ValueError(f"forecast runtime replay mismatch: {max_error}")
    return {**report, "runtime_replay_rows": len(replay), "max_abs_error": max_error}


def _predict_model(model: Any, frame: pd.DataFrame) -> pd.DataFrame:
    """Predict with the sole canonical Primary16 two-part bundle."""
    if not isinstance(model, dict):
        raise TypeError("GEAR accepts only the canonical Primary16 model bundle")
    required = {"model", "uptake_calibrator", "conditional_calibrator"}
    if not required.issubset(model):
        raise ValueError("Primary16 model bundle is incomplete")
    uptake_raw, conditional_raw = model["model"].predict_raw(frame)
    uptake = model["uptake_calibrator"].predict(uptake_raw)
    conditional = model["conditional_calibrator"].predict(conditional_raw)
    return pd.DataFrame(
        {
            "uptake_probability_calibrated": uptake,
            "conditional_diffusion_calibrated": conditional,
            "expected_diffusion_score": uptake * conditional,
        },
        index=frame.index,
    )


def _normalize(value: object) -> str:
    text = str(value or "").strip().casefold()
    if text.startswith("https://doi.org/"):
        return text.removeprefix("https://doi.org/")
    if text.startswith("doi:"):
        return text.removeprefix("doi:")
    return text


def _identifiers(paper_ir: PaperIR) -> set[str]:
    return {
        _normalize(value)
        for value in (
            paper_ir.paper_id,
            paper_ir.metadata.openalex_id,
            paper_ir.metadata.doi,
        )
        if value
    }


__all__ = [
    "DiffusionForecastService",
    "ForecastRelease",
    "RuntimeFeatureRelease",
    "unavailable_packet",
    "validate_runtime_replay",
]
