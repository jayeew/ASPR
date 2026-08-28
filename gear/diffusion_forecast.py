"""Lazy, fail-closed access to the frozen Fig.2/3 Primary16 D5 forecast."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, ClassVar, Literal, NamedTuple, cast

import joblib
import numpy as np
import pandas as pd
from pydantic import Field

from .contracts import PaperIR, StrictModel
from .graph_calibration import (
    anatomy_from_row,
    calibration_tensions,
    load_forecast_analog_index,
)
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
    anatomy_release_manifest: str | None = None


class StructuralHeadReleaseManifest(StrictModel):
    """Frozen D-excess/P sidecar tied to the canonical Primary16 release."""

    contract: Literal["gear_structural_head_release_v1"]
    release_id: str
    parent_forecast_release_id: str
    parent_feature_registry_sha256: str
    feature_protocol_version: str
    horizon_years: Literal[5]
    status: Literal["promoted"]
    feature_time_basis: Literal["T0_only"]
    uses_future_features: Literal[False]
    historical_prediction_policy: Literal["strict_oof_only"]
    runtime_prediction_policy: Literal["frozen_model_t0_only"]
    excess_target_fit_scope: Literal["outer_training_fold_only"]
    perturbation_target_fit_scope: Literal["outer_training_fold_only"]
    feature_names: list[str] = Field(min_length=1)
    training_row_count: int = Field(ge=20)
    runtime_feature_release_id: str | None = None
    runtime_feature_table_sha256: str | None = None
    runtime_score_table_sha256: str | None = None
    runtime_anatomy_table_sha256: str | None = None
    runtime_prediction_row_count: int = Field(default=0, ge=0)
    parent_oof_predictions_sha256: str | None = None
    parent_training_snapshot_sha256: str | None = None
    parent_oof_inference_row_count: int = Field(default=0, ge=0)
    created_at_utc: str
    assets: dict[str, ForecastAsset]


class StructuralHeadValues(NamedTuple):
    status: Literal["available", "limited", "unavailable"]
    excess_diffusion: float | None
    field_year_base: float | None
    perturbation_potential: float | None
    perturbation_components: dict[str, float]
    prediction_interval_width: float | None
    ood_reliability: float
    calibration_reliability: float
    release_id: str | None
    model_sha256: str | None
    training_reference_sha256: str | None
    prediction_protocol: (
        Literal["strict_oof", "frozen_t0_runtime", "frozen_t0_out_of_training"] | None
    )
    diagnostics: list[str]


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


class StructuralHeadRelease:
    """Immutable T0-only D-excess/P predictions with strict row provenance."""

    REQUIRED_ASSETS: ClassVar[set[str]] = {
        "model",
        "feature_registry",
        "training_reference",
        "prediction_table",
        "validation_report",
        "runtime_replay",
        "coverage_audit",
    }
    REQUIRED_COLUMNS: ClassVar[set[str]] = {
        "paper_id",
        "prediction_protocol",
        "as_of_date",
        "target_publication_date",
        "feature_source_max_date",
        "outer_fold_id",
        "excess_diffusion_head_d",
        "field_year_base",
        "perturbation_head_p",
        "prediction_interval_width",
        "ood_reliability",
        "calibration_reliability",
        "feature_source_sha256",
    }

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.manifest = StructuralHeadReleaseManifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )

    def path(self, name: str) -> Path:
        asset = self.manifest.assets[name]
        path = (self.root / asset.file).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"unsafe structural-head asset path: {path}") from exc
        if not path.is_file() or path.stat().st_size != asset.size_bytes:
            raise FileNotFoundError(path)
        if sha256_file(path) != asset.sha256:
            raise ValueError(f"structural-head asset hash mismatch: {name}")
        return path

    def verify(
        self,
        forecast_release: ForecastRelease,
        runtime_release: RuntimeFeatureRelease | None = None,
    ) -> dict[str, Any]:
        manifest = self.manifest
        if manifest.parent_forecast_release_id != forecast_release.manifest.release_id:
            raise ValueError("structural heads target a different forecast release")
        parent_registry = forecast_release.manifest.assets["feature_registry"].sha256
        if manifest.parent_feature_registry_sha256 != parent_registry:
            raise ValueError("structural-head parent feature registry mismatch")
        if (
            manifest.feature_protocol_version
            != forecast_release.manifest.protocol_version
        ):
            raise ValueError("structural-head feature protocol mismatch")
        missing = sorted(self.REQUIRED_ASSETS - set(manifest.assets))
        if missing:
            raise ValueError(f"structural-head release lacks assets: {missing}")
        for name in manifest.assets:
            self.path(name)
        registry = json.loads(self.path("feature_registry").read_text(encoding="utf-8"))
        parent_registry_payload = json.loads(
            forecast_release.path("feature_registry").read_text(encoding="utf-8")
        )
        if manifest.feature_names != parent_registry_payload.get("feature_names"):
            raise ValueError(
                "structural heads do not use the frozen Primary16 registry"
            )
        if registry.get("feature_names") != manifest.feature_names:
            raise ValueError("structural-head feature registry is inconsistent")
        if registry.get("parent_feature_registry_sha256") != parent_registry:
            raise ValueError("structural-head registry parent hash mismatch")
        if registry.get("feature_time_basis") != "T0_only":
            raise ValueError("structural-head registry is not T0-only")
        if registry.get("uses_future_features") is not False:
            raise ValueError("structural-head registry permits future features")
        if registry.get("label_only_columns_excluded_from_models") is not True:
            raise ValueError("structural-head registry does not exclude future labels")
        report = json.loads(self.path("validation_report").read_text(encoding="utf-8"))
        if (
            report.get("contract") != "gear_structural_head_validation_v1"
            or report.get("status") != "supported"
            or report.get("promotion_passed") is not True
        ):
            raise ValueError("structural-head scientific promotion is unsupported")
        gates = report.get("promotion_gates")
        if (
            not isinstance(gates, dict)
            or not gates
            or not all(value is True for value in gates.values())
        ):
            raise ValueError("structural-head promotion gates are incomplete")
        coverage = json.loads(self.path("coverage_audit").read_text(encoding="utf-8"))
        if (
            coverage.get("contract") != "gear_structural_head_coverage_audit_v1"
            or coverage.get("passed") is not True
        ):
            raise ValueError("structural-head cohort coverage is incomplete")
        training = pd.read_parquet(self.path("training_reference"))
        if (
            len(training) != manifest.training_row_count
            or "paper_id" not in training
            or training["paper_id"].astype(str).duplicated().any()
        ):
            raise ValueError("structural-head training reference is inconsistent")
        if set(training.get("feature_time_basis", [])) != {"T0_only"}:
            raise ValueError("structural-head training reference is not T0-only")
        if set(training.get("future_columns_role", [])) != {
            "label_construction_only_not_inference"
        }:
            raise ValueError("structural-head future labels lack a safe role")
        table = pd.read_parquet(self.path("prediction_table"))
        missing_columns = sorted(self.REQUIRED_COLUMNS - set(table))
        if missing_columns:
            raise ValueError(f"structural-head table lacks columns: {missing_columns}")
        _validate_structural_prediction_table(table)
        historical = table.loc[table["prediction_protocol"].eq("strict_oof")]
        fold_reference = training[["paper_id", "outer_fold_id"]].copy()
        fold_reference["paper_id"] = (
            fold_reference["paper_id"].astype(str).map(_normalize)
        )
        fold_reference["outer_fold_id"] = fold_reference["outer_fold_id"].astype(str)
        observed = historical[["paper_id", "outer_fold_id"]].copy()
        observed["paper_id"] = observed["paper_id"].astype(str).map(_normalize)
        observed["outer_fold_id"] = observed["outer_fold_id"].astype(str)
        folds = observed.merge(
            fold_reference,
            on="paper_id",
            how="left",
            suffixes=("_prediction", "_training"),
            indicator=True,
        )
        if (
            not folds["_merge"].eq("both").all()
            or not folds["outer_fold_id_prediction"]
            .eq(folds["outer_fold_id_training"])
            .all()
        ):
            raise ValueError(
                "structural-head OOF folds do not match training reference"
            )
        runtime = table.loc[table["prediction_protocol"].eq("frozen_t0_runtime")]
        _validate_structural_runtime_binding(manifest, runtime, runtime_release)
        inference = table.loc[
            table["prediction_protocol"].eq("frozen_t0_out_of_training")
        ]
        _validate_structural_parent_inference(
            manifest, historical, inference, training, forecast_release
        )
        return {
            "passed": True,
            "release_id": manifest.release_id,
            "prediction_rows": len(table),
            "assets_verified": len(manifest.assets),
        }

    def predictions(
        self,
        forecast_release: ForecastRelease,
        runtime_release: RuntimeFeatureRelease | None = None,
    ) -> pd.DataFrame:
        self.verify(forecast_release, runtime_release)
        return pd.read_parquet(self.path("prediction_table"))


class DiffusionForecastService:
    """Exact release lookup with an explicit unavailable result for every failure."""

    def __init__(
        self,
        manifest_path: Path,
        runtime_manifest_path: Path | None = None,
        anatomy_manifest_path: Path | None = None,
        structural_head_manifest_path: Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.runtime_manifest_path = (
            Path(runtime_manifest_path).resolve()
            if runtime_manifest_path is not None
            else None
        )
        self.anatomy_manifest_path = (
            Path(anatomy_manifest_path).resolve()
            if anatomy_manifest_path is not None
            else None
        )
        self.structural_head_manifest_path = (
            Path(structural_head_manifest_path).resolve()
            if structural_head_manifest_path is not None
            else None
        )
        self._release: ForecastRelease | None = None
        self._scores: pd.DataFrame | None = None
        self._runtime_anatomy: pd.DataFrame | None = None
        self._release_anatomy: pd.DataFrame | None = None
        self._anatomy_load_failure: str | None = None
        self._runtime_release: RuntimeFeatureRelease | None = None
        self._structural_release: StructuralHeadRelease | None = None
        self._structural_predictions: pd.DataFrame | None = None
        self._structural_load_failure: str | None = None

    def _load(self) -> tuple[ForecastRelease, pd.DataFrame]:
        if self._release is None:
            self._release = ForecastRelease(self.manifest_path)
            self._release.verify()
            self._scores = pd.read_parquet(self._release.path("score_table"))
            self._scores["feature_source"] = "release_score_table"
            if self.anatomy_manifest_path is not None:
                try:
                    self._release_anatomy = load_forecast_analog_index(
                        self.anatomy_manifest_path
                    ).table()
                except (KeyError, OSError, TypeError, ValueError) as exc:
                    self._anatomy_load_failure = type(exc).__name__
            if self.runtime_manifest_path is not None:
                self._runtime_release = RuntimeFeatureRelease(
                    self.runtime_manifest_path
                )
                self._runtime_release.verify(self._release)
                runtime_scores = pd.read_parquet(
                    self._runtime_release.path("runtime_score_table")
                )
                self._scores = pd.concat(
                    [self._scores, runtime_scores], ignore_index=True, sort=False
                )
                if "runtime_anatomy_table" in self._runtime_release.manifest.assets:
                    self._runtime_anatomy = pd.read_parquet(
                        self._runtime_release.path("runtime_anatomy_table")
                    )
            if self.structural_head_manifest_path is not None:
                try:
                    self._structural_release = StructuralHeadRelease(
                        self.structural_head_manifest_path
                    )
                    self._structural_predictions = self._structural_release.predictions(
                        self._release, self._runtime_release
                    )
                except (OSError, TypeError, ValueError) as exc:
                    self._structural_load_failure = type(exc).__name__
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
            publication_year = (
                None if runtime_inference else int(row["publication_year"])
            )
            exact_publication_t0 = (
                not runtime_inference
                and paper_ir.metadata.publication_date == cutoff_date
                and publication_year == cutoff_date.year
            )
            if not runtime_inference and (
                publication_year is None
                or publication_year > cutoff_date.year
                or (publication_year == cutoff_date.year and not exact_publication_t0)
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
            if exact_publication_t0 and not runtime_inference:
                diagnostics.append("frozen_publication_t0_oof_lookup")
            source_max_year = row.get("source_max_year")
            if runtime_inference and pd.notna(source_max_year):
                diagnostics.append(
                    f"historical_context_source_max_year:{int(source_max_year)}"
                )
                if int(source_max_year) < cutoff_date.year - 1:
                    diagnostics.append("historical_context_limited")
            anatomy = None
            tensions = []
            anatomy_table = (
                self._runtime_anatomy if runtime_inference else self._release_anatomy
            )
            if anatomy_table is not None:
                anatomy_rows = anatomy_table.loc[
                    anatomy_table["paper_id"]
                    .astype(str)
                    .map(_normalize)
                    .isin(identifiers)
                ]
                if len(anatomy_rows) == 1:
                    anatomy = anatomy_from_row(anatomy_rows.iloc[0].to_dict())
                    tensions = calibration_tensions(anatomy)
                    diagnostics.append("forecast_anatomy_available")
                    if anatomy.limited:
                        diagnostics.append("forecast_anatomy_limited")
            elif self.anatomy_manifest_path is not None:
                diagnostics.append(
                    "forecast_anatomy_unavailable:"
                    + (self._anatomy_load_failure or "target_missing")
                )
            structural = self._structural_values(
                identifiers=identifiers,
                cutoff_date=cutoff_date,
                runtime_inference=runtime_inference,
            )
            diagnostics.extend(structural.diagnostics)
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
                    excess_diffusion=structural.excess_diffusion,
                    field_year_base=structural.field_year_base,
                    perturbation_potential=structural.perturbation_potential,
                    perturbation_components=structural.perturbation_components,
                    prediction_interval_width=structural.prediction_interval_width,
                    ood_reliability=structural.ood_reliability,
                    calibration_reliability=structural.calibration_reliability,
                    structural_heads_status=structural.status,
                    structural_head_release_id=structural.release_id,
                    structural_head_model_sha256=structural.model_sha256,
                    structural_head_training_reference_sha256=(
                        structural.training_reference_sha256
                    ),
                    structural_head_prediction_protocol=(
                        structural.prediction_protocol
                    ),
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
                forecast_anatomy=anatomy,
                calibration_tensions=tensions,
                diagnostics=diagnostics,
            )
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
            return unavailable_packet(
                paper_ir.paper_id,
                cutoff_date,
                f"forecast_release_unavailable:{type(exc).__name__}",
            )

    def _structural_values(
        self,
        *,
        identifiers: set[str],
        cutoff_date: date,
        runtime_inference: bool,
    ) -> StructuralHeadValues:
        if self.structural_head_manifest_path is None:
            return _unavailable_structural_heads("manifest_not_configured")
        if self._structural_predictions is None or self._structural_release is None:
            reason = self._structural_load_failure or "asset_unavailable"
            return _unavailable_structural_heads(f"release_unavailable:{reason}")
        matches = self._structural_predictions.loc[
            self._structural_predictions["paper_id"]
            .astype(str)
            .map(_normalize)
            .isin(identifiers)
        ]
        if len(matches) != 1:
            reason = "exact_lookup_missing" if matches.empty else "identity_ambiguous"
            return _unavailable_structural_heads(reason)
        row = matches.iloc[0]
        protocol = str(row["prediction_protocol"])
        if runtime_inference and protocol != "frozen_t0_runtime":
            return _unavailable_structural_heads("prediction_protocol_mismatch")
        if not runtime_inference and protocol not in {
            "strict_oof",
            "frozen_t0_out_of_training",
        }:
            return _unavailable_structural_heads("prediction_protocol_mismatch")
        if (
            protocol == "frozen_t0_runtime"
            and str(row["as_of_date"])[:10] != cutoff_date.isoformat()
        ):
            return _unavailable_structural_heads("runtime_cutoff_mismatch")
        if protocol == "strict_oof" and pd.isna(row["outer_fold_id"]):
            return _unavailable_structural_heads("historical_outer_fold_missing")
        if protocol != "strict_oof" and pd.notna(row["outer_fold_id"]):
            return _unavailable_structural_heads("frozen_inference_claims_oof_fold")
        if (
            protocol == "frozen_t0_out_of_training"
            and str(row["as_of_date"])[:10] > cutoff_date.isoformat()
        ):
            return _unavailable_structural_heads("frozen_inference_post_cutoff")
        try:
            components = _structural_components(row)
            values = [
                float(row["excess_diffusion_head_d"]),
                float(row["field_year_base"]),
                float(row["perturbation_head_p"]),
                float(row["prediction_interval_width"]),
                float(row["ood_reliability"]),
                float(row["calibration_reliability"]),
                *components.values(),
            ]
        except (TypeError, ValueError):
            return _unavailable_structural_heads("prediction_values_invalid")
        if not all(np.isfinite(values)) or not all(
            0.0 <= value <= 1.0 for value in values
        ):
            return _unavailable_structural_heads("prediction_values_out_of_range")
        manifest = self._structural_release.manifest
        assets = manifest.assets
        return StructuralHeadValues(
            status="available",
            excess_diffusion=values[0],
            field_year_base=values[1],
            perturbation_potential=values[2],
            perturbation_components=components,
            prediction_interval_width=values[3],
            ood_reliability=values[4],
            calibration_reliability=values[5],
            release_id=manifest.release_id,
            model_sha256=assets["model"].sha256,
            training_reference_sha256=assets["training_reference"].sha256,
            prediction_protocol=cast(
                Literal[
                    "strict_oof",
                    "frozen_t0_runtime",
                    "frozen_t0_out_of_training",
                ],
                protocol,
            ),
            diagnostics=[
                "structural_heads_available",
                f"structural_head_protocol:{protocol}",
                "structural_head_features_t0_only",
            ],
        )


def _validate_structural_prediction_table(table: pd.DataFrame) -> None:
    normalized = table["paper_id"].astype(str).map(_normalize)
    if normalized.eq("").any() or normalized.duplicated().any():
        raise ValueError("structural-head paper identities must be unique and nonempty")
    protocols = set(table["prediction_protocol"].astype(str))
    if not protocols.issubset(
        {"strict_oof", "frozen_t0_runtime", "frozen_t0_out_of_training"}
    ):
        raise ValueError("structural-head prediction protocol is not registered")
    historical = table["prediction_protocol"].astype(str).eq("strict_oof")
    runtime = table["prediction_protocol"].astype(str).eq("frozen_t0_runtime")
    out_of_training = (
        table["prediction_protocol"].astype(str).eq("frozen_t0_out_of_training")
    )
    if table.loc[historical, "outer_fold_id"].isna().any():
        raise ValueError("historical structural predictions require an OOF fold")
    if table.loc[runtime, "outer_fold_id"].notna().any():
        raise ValueError("runtime structural predictions cannot claim an OOF fold")
    if table.loc[out_of_training, "outer_fold_id"].notna().any():
        raise ValueError("out-of-training predictions cannot claim an OOF fold")
    source_hash = table["feature_source_sha256"].astype(str)
    if not source_hash.str.fullmatch(r"sha256:[0-9a-f]{64}").all():
        raise ValueError("structural-head feature source hash is invalid")
    as_of = pd.to_datetime(table["as_of_date"], errors="coerce", utc=True)
    publication = pd.to_datetime(
        table["target_publication_date"], errors="coerce", utc=True
    )
    source_max = pd.to_datetime(
        table["feature_source_max_date"], errors="coerce", utc=True
    )
    if as_of.isna().any() or publication.isna().any() or source_max.isna().any():
        raise ValueError("structural-head cutoff provenance contains invalid dates")
    if (source_max > as_of).any() or (publication > as_of).any():
        raise ValueError("structural-head features or target are post-cutoff")
    if not as_of.loc[historical].equals(publication.loc[historical]):
        raise ValueError("historical structural predictions must use publication T0")
    value_columns = [
        "excess_diffusion_head_d",
        "field_year_base",
        "perturbation_head_p",
        "prediction_interval_width",
        "ood_reliability",
        "calibration_reliability",
    ]
    component_columns = [
        column for column in table if column.startswith("perturbation_component_")
    ]
    values = table[[*value_columns, *component_columns]].apply(
        pd.to_numeric, errors="coerce"
    )
    if values.isna().any().any() or not (
        values.ge(0.0).all().all() and values.le(1.0).all().all()
    ):
        raise ValueError("structural-head values must be finite probabilities")


def _validate_structural_runtime_binding(
    manifest: StructuralHeadReleaseManifest,
    runtime_rows: pd.DataFrame,
    runtime_release: RuntimeFeatureRelease | None,
) -> None:
    bindings = (
        manifest.runtime_feature_release_id,
        manifest.runtime_feature_table_sha256,
        manifest.runtime_score_table_sha256,
        manifest.runtime_anatomy_table_sha256,
    )
    if manifest.runtime_prediction_row_count == 0:
        if len(runtime_rows) or any(value is not None for value in bindings):
            raise ValueError("structural-head runtime binding is inconsistent")
        return
    if runtime_release is None or any(value is None for value in bindings):
        raise ValueError("structural-head runtime release is not configured")
    if manifest.runtime_feature_release_id != runtime_release.manifest.release_id:
        raise ValueError("structural-head runtime release id mismatch")
    expected_hashes = {
        "runtime_feature_table": manifest.runtime_feature_table_sha256,
        "runtime_score_table": manifest.runtime_score_table_sha256,
        "runtime_anatomy_table": manifest.runtime_anatomy_table_sha256,
    }
    for name, expected in expected_hashes.items():
        if name not in runtime_release.manifest.assets:
            raise ValueError(f"structural-head runtime asset missing: {name}")
        if runtime_release.manifest.assets[name].sha256 != expected:
            raise ValueError(f"structural-head runtime asset hash mismatch: {name}")
    scores = pd.read_parquet(runtime_release.path("runtime_score_table"))
    if (
        len(runtime_rows) != manifest.runtime_prediction_row_count
        or len(runtime_rows) != runtime_release.manifest.target_count
    ):
        raise ValueError("structural-head runtime prediction coverage mismatch")
    expected = scores[["paper_id", "as_of_date"]].copy()
    expected["paper_id"] = expected["paper_id"].astype(str).map(_normalize)
    expected["as_of_date"] = expected["as_of_date"].astype(str).str[:10]
    observed = runtime_rows[["paper_id", "as_of_date"]].copy()
    observed["paper_id"] = observed["paper_id"].astype(str).map(_normalize)
    observed["as_of_date"] = observed["as_of_date"].astype(str).str[:10]
    coverage = observed.merge(
        expected,
        on=["paper_id", "as_of_date"],
        how="outer",
        indicator=True,
    )
    if not coverage["_merge"].eq("both").all():
        raise ValueError("structural-head runtime ids or cutoffs differ")


def _validate_structural_parent_inference(
    manifest: StructuralHeadReleaseManifest,
    historical: pd.DataFrame,
    inference: pd.DataFrame,
    training: pd.DataFrame,
    forecast_release: ForecastRelease,
) -> None:
    bindings = (
        manifest.parent_oof_predictions_sha256,
        manifest.parent_training_snapshot_sha256,
    )
    if manifest.parent_oof_inference_row_count == 0:
        if len(inference) or any(value is not None for value in bindings):
            raise ValueError("structural-head parent inference binding is inconsistent")
        return
    if any(value is None for value in bindings):
        raise ValueError("structural-head parent inference hashes are missing")
    if (
        manifest.parent_oof_predictions_sha256
        != forecast_release.manifest.assets["oof_predictions"].sha256
    ):
        raise ValueError("structural-head parent OOF hash mismatch")
    if (
        manifest.parent_training_snapshot_sha256
        != forecast_release.manifest.assets["training_snapshot"].sha256
    ):
        raise ValueError("structural-head parent feature hash mismatch")
    parent = pd.read_parquet(
        forecast_release.path("oof_predictions"), columns=["paper_id"]
    )
    parent_ids = set(parent["paper_id"].astype(str).map(_normalize))
    training_ids = set(training["paper_id"].astype(str).map(_normalize))
    historical_ids = set(historical["paper_id"].astype(str).map(_normalize))
    inference_ids = set(inference["paper_id"].astype(str).map(_normalize))
    if historical_ids != training_ids:
        raise ValueError("structural-head training papers are not exactly strict OOF")
    if inference_ids & training_ids:
        raise ValueError("structural-head frozen inference overlaps training papers")
    if inference_ids != parent_ids - training_ids:
        raise ValueError("structural-head parent OOF inference coverage is incomplete")
    if len(inference) != manifest.parent_oof_inference_row_count:
        raise ValueError("structural-head parent OOF inference row count mismatch")


def _structural_components(row: pd.Series) -> dict[str, float]:
    prefix = "perturbation_component_"
    return {
        str(column).removeprefix(prefix): float(value)
        for column, value in row.items()
        if str(column).startswith(prefix)
    }


def _unavailable_structural_heads(reason: str) -> StructuralHeadValues:
    return StructuralHeadValues(
        status="unavailable",
        excess_diffusion=None,
        field_year_base=None,
        perturbation_potential=None,
        perturbation_components={},
        prediction_interval_width=None,
        ood_reliability=1.0,
        calibration_reliability=1.0,
        release_id=None,
        model_sha256=None,
        training_reference_sha256=None,
        prediction_protocol=None,
        diagnostics=["structural_heads_limited", f"structural_heads_{reason}"],
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


def validate_structural_head_replay(
    manifest_path: Path,
    forecast_manifest_path: Path,
    runtime_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Replay the frozen T0 structural bundle against its hash-bound matrix."""
    forecast = ForecastRelease(forecast_manifest_path)
    runtime = (
        RuntimeFeatureRelease(runtime_manifest_path)
        if runtime_manifest_path is not None
        else None
    )
    if runtime is not None:
        runtime.verify(forecast)
    release = StructuralHeadRelease(manifest_path)
    verification = release.verify(forecast, runtime)
    model = joblib.load(release.path("model"))
    replay = pd.read_parquet(release.path("runtime_replay"))
    observed = _predict_structural_bundle(model, replay)
    expected_columns = [
        column.removeprefix("expected_")
        for column in replay
        if column.startswith("expected_")
    ]
    if not expected_columns:
        raise ValueError("structural-head replay has no expected outputs")
    errors = []
    for column in expected_columns:
        if column not in observed:
            raise ValueError(f"structural-head replay output missing: {column}")
        expected = pd.to_numeric(replay[f"expected_{column}"], errors="coerce")
        actual = pd.to_numeric(observed[column], errors="coerce")
        errors.append(float(np.max(np.abs(expected - actual))))
    max_error = max(errors)
    if max_error > 1e-12:
        raise ValueError(f"structural-head runtime replay mismatch: {max_error}")
    return {
        **verification,
        "runtime_replay_rows": len(replay),
        "runtime_replay_outputs": len(expected_columns),
        "max_abs_error": max_error,
    }


def _predict_structural_bundle(model: Any, frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(model, dict) or model.get("contract") != (
        "gear_frozen_structural_head_bundle_v1"
    ):
        raise TypeError("unregistered structural-head model bundle")
    feature_names = [str(value) for value in model["feature_names"]]
    required = {*feature_names, "domain12", "publication_year"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"structural-head replay lacks inputs: {missing}")
    output = pd.DataFrame(index=frame.index)
    for name, head in model["heads"].items():
        output[str(name)] = np.clip(head.predict(frame[feature_names]), 0.0, 1.0)
    output["field_year_base"] = np.clip(
        model["field_year_model"].predict(frame[["domain12", "publication_year"]]),
        0.0,
        1.0,
    )
    categorical = set(model["categorical_feature_names"])
    numeric = [value for value in feature_names if value not in categorical]
    values = frame[numeric].apply(pd.to_numeric, errors="coerce")
    lower = pd.Series(model["numeric_support_lower"])
    upper = pd.Series(model["numeric_support_upper"])
    support = (values.ge(lower, axis=1) & values.le(upper, axis=1)).mean(axis=1)
    category_support = pd.Series(1.0, index=frame.index)
    for column, allowed in model["categorical_support"].items():
        category_support *= frame[column].astype(str).isin(set(allowed)).astype(float)
    coverage = frame[feature_names].notna().mean(axis=1)
    output["ood_reliability"] = np.clip(
        coverage * (0.8 * support + 0.2 * category_support), 0.0, 1.0
    )
    output["prediction_interval_width"] = float(model["prediction_interval_width"])
    output["calibration_reliability"] = float(model["calibration_reliability"])
    return output


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
    "StructuralHeadRelease",
    "unavailable_packet",
    "validate_runtime_replay",
    "validate_structural_head_replay",
]
