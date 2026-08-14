"""Fail-closed adapter for the current Fig.3 D5 Full-text-16 HGB model."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
import requests

from gear.nature_multihorizon.modeling_v6 import FittedTwoPartModel
from gear.nature_multihorizon.t0_runtime_v3 import (
    ContextSnapshot,
    MaterializationReplayReport,
    TargetT0Record,
    coerce_fulltext16_storage_schema,
    materialize_fulltext16,
    validate_materialization_replay,
)

from .config import AssetPaths, GearConfig, load_config
from .contracts import (
    CalibrationCutoff,
    CalibrationForecast,
    CalibrationInterpretation,
    CalibrationMeasurement,
    CalibrationMode,
    CalibrationPacketV3,
    CalibrationProvenance,
    CalibrationReliability,
    PaperIR,
)


def _load_official_joblib(path: Path) -> Dict[str, Any]:
    """Load the frozen model through temporary historical module aliases.

    The official Fig.3 joblib was serialized before the package was renamed.
    This is an in-memory deserialization alias, not a public ``aspr`` package or
    runtime fallback.
    """
    from gear import nature_multihorizon
    from gear.nature_multihorizon import modeling_v6

    names = (
        "aspr",
        "aspr.nature_multihorizon",
        "aspr.nature_multihorizon.modeling_v6",
    )
    previous = {name: sys.modules.get(name) for name in names}
    legacy_root = previous["aspr"] or types.ModuleType("aspr")
    legacy_root.__path__ = []
    try:
        sys.modules["aspr"] = legacy_root
        sys.modules["aspr.nature_multihorizon"] = nature_multihorizon
        sys.modules["aspr.nature_multihorizon.modeling_v6"] = modeling_v6
        bundle = joblib.load(path)
    finally:
        for name in reversed(names):
            prior = previous[name]
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
    if not isinstance(bundle, dict):
        raise ValueError("official HGB bundle must be a mapping")
    return bundle


FEATURE_ROLES: Mapping[str, Tuple[str, ...]] = {
    "substantive_innovation": ("EF0017", "EF0052", "EF0240"),
    "t0_potential": ("EF0309", "EF0312", "EF0315", "EF0318"),
    "opportunity": ("EF0083", "EF0186", "EF0188", "EF0238", "EF0319"),
    "context_control": ("EF0038", "EF0197", "EF0307", "EF0314"),
}
FEATURE_NAMES: Tuple[str, ...] = tuple(
    feature for role in FEATURE_ROLES.values() for feature in role
)
SURROGATES = ("EF0017", "EF0083", "EF0240", "EF0319")
_RUNTIME_CONTEXT_CACHE: Dict[
    str, Tuple[ContextSnapshot, MaterializationReplayReport, Dict[str, Any]]
] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def normalize_openalex_id(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.casefold().startswith("https://openalex.org/"):
        suffix = text.rsplit("/", 1)[-1]
    else:
        suffix = text
    if not suffix.upper().startswith("W"):
        return None
    return f"https://openalex.org/{suffix.upper()}"


def _safe_scalar(value: Any) -> Any:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


class CalibrationService:
    """Load trusted assets lazily and expose exact lookup without legacy fallback."""

    def __init__(
        self,
        config: Optional[GearConfig] = None,
        *,
        doi_resolver: Optional[Callable[[str], Optional[str]]] = None,
        asset_paths: Optional[AssetPaths] = None,
        runtime_target_builder: Optional[
            Callable[[PaperIR, Optional[date]], TargetT0Record]
        ] = None,
    ) -> None:
        self.config = config or load_config()
        self.paths = asset_paths or self.config.resolved_assets()
        self.release_id = (
            None
            if asset_paths is not None
            else self.config.resolved_calibration_release().release_id
        )
        self.doi_resolver = doi_resolver or self._resolve_doi_via_openalex
        self.runtime_target_builder = runtime_target_builder
        self._loaded = False
        self._model_bundle: Dict[str, Any] = {}
        self._matrix = pd.DataFrame()
        self._scores = pd.DataFrame()
        self._metadata = pd.DataFrame()
        self._fold_metrics = pd.DataFrame()
        self._domain_metrics = pd.DataFrame()
        self._overall_metrics = pd.DataFrame()
        self._official: Dict[str, Any] = {}
        self._run_manifest: Dict[str, Any] = {}
        self._matrix_manifest: Dict[str, Any] = {}
        self._input_snapshot: Dict[str, Any] = {}
        self._approved_runtime_replay: Optional[MaterializationReplayReport] = None
        self._runtime_context: Optional[ContextSnapshot] = None
        self._runtime_manifest: Dict[str, Any] = {}

    def _load(self) -> None:
        if self._loaded:
            return
        for value in self.paths.model_dump().values():
            path = self.config.validate_asset_path(Path(value))
            if not path.is_file():
                raise FileNotFoundError(path)
        self._official = json.loads(
            self.paths.official_model_json.read_text(encoding="utf-8")
        )
        self._run_manifest = json.loads(
            self.paths.official_run_manifest.read_text(encoding="utf-8")
        )
        self._matrix_manifest = json.loads(
            self.paths.matrix_manifest.read_text(encoding="utf-8")
        )
        self._input_snapshot = json.loads(
            self.paths.matrix_input_snapshot.read_text(encoding="utf-8")
        )
        expected_model = self._official["production"]["model_sha256"]
        expected_score = self._official["production"]["score_sha256"]
        expected_matrix = self._matrix_manifest["outputs"]["matrix_16"]["sha256"]
        checks = {
            self.paths.official_model_joblib: expected_model,
            self.paths.official_score_table: expected_score,
            self.paths.feature_matrix_16: expected_matrix,
            self.paths.matrix_input_snapshot: self._matrix_manifest["outputs"][
                "input_snapshot"
            ]["sha256"],
            self.paths.paper_metadata: self._input_snapshot["papers"]["sha256"],
            self.paths.official_model_json: self._run_manifest["outputs"][
                "official_model.json"
            ]["sha256"],
            self.paths.oof_metrics: self._run_manifest["outputs"]["oof_metrics.csv"][
                "sha256"
            ],
            self.paths.oof_fold_metrics: self._run_manifest["outputs"][
                "oof_fold_metrics.csv"
            ]["sha256"],
            self.paths.oof_domain_metrics: self._run_manifest["outputs"][
                "oof_domain_metrics.csv"
            ]["sha256"],
        }
        for path, expected in checks.items():
            observed = sha256_file(path)
            if observed != expected:
                raise ValueError(f"frozen graph asset hash mismatch: {path}")
        if (
            self._official.get("feature_set") != "fulltext_16"
            or int(self._official.get("horizon", -1)) != 5
        ):
            raise ValueError("official graph contract is not D5 Full-text 16")
        self._model_bundle = _load_official_joblib(self.paths.official_model_joblib)
        required_model_keys = {"model", "uptake_calibrator", "conditional_calibrator"}
        if set(self._model_bundle) != required_model_keys:
            raise ValueError("unexpected official HGB bundle keys")
        fitted_model = self._model_bundle["model"]
        if not isinstance(fitted_model, FittedTwoPartModel):
            raise ValueError("official HGB bundle has an unexpected model type")
        if tuple(fitted_model.feature_names) != tuple(sorted(FEATURE_NAMES)):
            raise ValueError(
                "official HGB model feature order differs from Full-text 16"
            )
        if tuple(fitted_model.categorical_names) != ("EF0197",):
            raise ValueError("official HGB categorical contract differs from Fig.3")
        self._matrix = pd.read_parquet(self.paths.feature_matrix_16).set_index(
            "paper_id", drop=False
        )
        self._scores = pd.read_parquet(self.paths.official_score_table).set_index(
            "paper_id", drop=False
        )
        self._metadata = pd.read_parquet(
            self.paths.paper_metadata,
            columns=[
                "paper_id",
                "publication_year",
                "domain12",
                "source_id",
                "source_display_name",
            ],
        ).set_index("paper_id", drop=False)
        if set(FEATURE_NAMES) != set(self._matrix.columns) - {"paper_id"}:
            raise ValueError("official matrix does not contain the frozen 16 features")
        self._fold_metrics = pd.read_csv(self.paths.oof_fold_metrics)
        self._domain_metrics = pd.read_csv(self.paths.oof_domain_metrics)
        self._overall_metrics = pd.read_csv(self.paths.oof_metrics)
        if int(self._matrix_manifest.get("row_count", -1)) != 411_490:
            raise ValueError("official Full-text 16 manifest row count is not frozen")
        for label, frame in (
            ("matrix", self._matrix),
            ("score", self._scores),
            ("metadata", self._metadata),
        ):
            if len(frame) != 411_490 or frame.index.duplicated().any():
                raise ValueError(f"official {label} table violates the frozen cohort")
        if set(self._matrix.index) != set(self._scores.index) or set(
            self._matrix.index
        ) != set(self._metadata.index):
            raise ValueError("official graph tables contain different paper cohorts")
        overall_selector = (
            self._overall_metrics["horizon"].eq(5)
            & self._overall_metrics["model_family"].eq("hgb")
            & self._overall_metrics["model_id"].eq("fulltext_16")
        )
        if int(overall_selector.sum()) != 1:
            raise ValueError("official overall OOF metric is not unique")
        self._loaded = True

    def build_packet(
        self,
        paper_ir: PaperIR,
        *,
        cutoff_date: Optional[date] = None,
    ) -> CalibrationPacketV3:
        if (
            cutoff_date is not None
            and paper_ir.metadata.publication_date is not None
            and cutoff_date < paper_ir.metadata.publication_date
        ):
            return self._unavailable_packet(
                paper_ir.paper_id,
                paper_ir.metadata.publication_date,
                "review_cutoff_precedes_publication_date",
            )
        work_id = normalize_openalex_id(paper_ir.metadata.openalex_id)
        if work_id is None and paper_ir.metadata.doi:
            try:
                work_id = normalize_openalex_id(
                    self.doi_resolver(paper_ir.metadata.doi)
                )
            except Exception:
                work_id = None
        if work_id is not None:
            try:
                return self.exact_lookup(work_id)
            except KeyError:
                pass
        try:
            context, replay_report = self.load_approved_runtime_context()
            target = self._build_runtime_target(paper_ir, cutoff_date)
            return self.eligible_inference(
                target,
                context,
                replay_report=replay_report,
            )
        except (
            OSError,
            TypeError,
            ValueError,
            KeyError,
            requests.RequestException,
        ) as exc:
            return self._unavailable_packet(
                work_id or paper_ir.paper_id,
                paper_ir.metadata.publication_date,
                f"online_fulltext16_unavailable:{type(exc).__name__}",
            )

    def _build_runtime_target(
        self,
        paper_ir: PaperIR,
        cutoff_date: Optional[date],
    ) -> TargetT0Record:
        if self.runtime_target_builder is not None:
            return self.runtime_target_builder(paper_ir, cutoff_date)
        if not self.config.allow_external_retrieval:
            raise ValueError("OpenAlex T0 enrichment is disabled")
        from .t0_enrichment import OpenAlexT0Enricher

        return OpenAlexT0Enricher().build_target(
            paper_ir,
            evidence_date=cutoff_date,
        )

    def unavailable_packet(
        self,
        paper_ir: PaperIR,
        reason: str,
        *,
        profile_only: bool = False,
    ) -> CalibrationPacketV3:
        """Create an explicit fail-closed packet without loading graph assets."""
        return self._unavailable_packet(
            paper_ir.paper_id,
            paper_ir.metadata.publication_date,
            reason,
            profile_only=profile_only,
        )

    def exact_lookup(self, openalex_id: str) -> CalibrationPacketV3:
        self._load()
        paper_id = normalize_openalex_id(openalex_id)
        if paper_id is None or paper_id not in self._matrix.index:
            raise KeyError(openalex_id)
        row = self._matrix.loc[[paper_id], [*FEATURE_NAMES]].copy()
        uptake_raw, conditional_raw = self._model_bundle["model"].predict_raw(row)
        p_uptake = float(self._model_bundle["uptake_calibrator"].predict(uptake_raw)[0])
        conditional = float(
            self._model_bundle["conditional_calibrator"].predict(conditional_raw)[0]
        )
        raw = p_uptake * conditional
        score_row = self._scores.loc[paper_id]
        expected_raw = float(score_row["raw_prediction_score"])
        if not np.isclose(raw, expected_raw, rtol=0.0, atol=1e-10):
            raise ValueError(
                f"official raw prediction mismatch for {paper_id}: "
                f"{raw} != {expected_raw}"
            )
        metadata = self._metadata.loc[paper_id]
        publication_year = int(metadata["publication_year"])
        domain = str(metadata["domain12"])
        feature_values = {
            feature: _safe_scalar(row.iloc[0][feature]) for feature in FEATURE_NAMES
        }
        missing = [name for name, value in feature_values.items() if value is None]
        domain_metrics = self._domain_metrics[
            (self._domain_metrics["horizon"] == 5)
            & (self._domain_metrics["model_family"] == "hgb")
            & (self._domain_metrics["model_id"] == "fulltext_16")
            & (self._domain_metrics["domain12"].astype(str) == domain)
        ]
        domain_support = (
            int(domain_metrics.iloc[0]["n_oof"]) if not domain_metrics.empty else None
        )
        domain_spearman = (
            float(domain_metrics.iloc[0]["spearman"])
            if not domain_metrics.empty
            else None
        )
        overall_metrics = self._overall_metrics[
            (self._overall_metrics["horizon"] == 5)
            & (self._overall_metrics["model_family"] == "hgb")
            & (self._overall_metrics["model_id"] == "fulltext_16")
        ]
        overall_spearman = (
            float(overall_metrics.iloc[0]["spearman"])
            if not overall_metrics.empty
            else None
        )
        temporal_block = self._temporal_block(publication_year)
        fold_spearman = self._fold_spearman(publication_year)
        drift_flags = []
        if publication_year > 2020:
            drift_flags.append("immature_d5_publication_cohort")
        bands = self._historical_bands(
            paper_id, publication_year, domain, feature_values
        )
        measurement = CalibrationMeasurement(
            substantive_innovation={
                name: feature_values[name]
                for name in FEATURE_ROLES["substantive_innovation"]
            },
            t0_potential={
                name: feature_values[name] for name in FEATURE_ROLES["t0_potential"]
            },
            opportunity={
                name: feature_values[name] for name in FEATURE_ROLES["opportunity"]
            },
            context_control={
                name: feature_values[name] for name in FEATURE_ROLES["context_control"]
            },
            local_surrogates=list(SURROGATES),
            historical_bands=bands,
        )
        return CalibrationPacketV3(
            paper_id=paper_id,
            cutoff=CalibrationCutoff(
                publication_year=publication_year,
                source_max_year=publication_year - 1,
                granularity="year",
            ),
            measurement=measurement,
            forecast=CalibrationForecast(
                p_uptake=p_uptake,
                conditional_diffusion=conditional,
                raw_expected_diffusion=raw,
                aspr_score_0_100=float(score_row["aspr_score"]),
            ),
            reliability=CalibrationReliability(
                mode=CalibrationMode.EXACT_LOOKUP,
                domain=domain,
                domain_support_n=domain_support,
                temporal_block=temporal_block,
                overall_oof_spearman=overall_spearman,
                fold_oof_spearman=fold_spearman,
                domain_oof_spearman=domain_spearman,
                feature_coverage=(len(FEATURE_NAMES) - len(missing))
                / len(FEATURE_NAMES),
                missing_features=missing,
                drift_flags=drift_flags,
            ),
            provenance=CalibrationProvenance(
                calibration_release_id=self.release_id,
                model_sha256=self._official["production"]["model_sha256"],
                score_table_sha256=self._official["production"]["score_sha256"],
                feature_matrix_sha256=self._matrix_manifest["outputs"]["matrix_16"][
                    "sha256"
                ],
                reference_corpus_sha256=sha256_file(self.paths.paper_metadata),
                oof_metrics_sha256=sha256_file(self.paths.oof_metrics),
                oof_fold_metrics_sha256=sha256_file(self.paths.oof_fold_metrics),
                oof_domain_metrics_sha256=sha256_file(self.paths.oof_domain_metrics),
            ),
        )

    def validate_official_replay(self, *, batch_size: int = 50_000) -> Dict[str, Any]:
        """Replay every frozen matrix row through the official production bundle.

        This is intentionally separate from normal review execution because a
        complete 411,490-row replay is an asset-release check, not per-paper
        inference work.
        """
        self._load()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        score_rows = self._scores.reindex(self._matrix.index)
        if score_rows["raw_prediction_score"].isna().any():
            raise ValueError("official score table is missing frozen matrix rows")
        observed_parts = []
        uptake_parts = []
        conditional_parts = []
        for start in range(0, len(self._matrix), int(batch_size)):
            features = self._matrix.iloc[start : start + int(batch_size)][
                [*FEATURE_NAMES]
            ]
            uptake_raw, conditional_raw = self._model_bundle["model"].predict_raw(
                features
            )
            uptake = self._model_bundle["uptake_calibrator"].predict(uptake_raw)
            conditional = self._model_bundle["conditional_calibrator"].predict(
                conditional_raw
            )
            uptake_parts.append(np.asarray(uptake, dtype=float))
            conditional_parts.append(np.asarray(conditional, dtype=float))
            observed_parts.append(
                np.asarray(uptake, dtype=float) * np.asarray(conditional, dtype=float)
            )
        uptake_values = np.concatenate(uptake_parts)
        conditional_values = np.concatenate(conditional_parts)
        observed = np.concatenate(observed_parts)
        expected = score_rows["raw_prediction_score"].to_numpy(dtype=float)
        errors = np.abs(observed - expected)
        finite = np.isfinite(errors)
        maximum = float(errors[finite].max()) if finite.any() else float("inf")
        product_error = np.abs(observed - uptake_values * conditional_values)
        product_maximum = float(product_error.max()) if len(product_error) else 0.0
        passed = (
            len(observed) == 411_490
            and finite.all()
            and maximum <= 1e-10
            and product_maximum <= np.finfo(float).eps
        )
        return {
            "row_count": len(observed),
            "maximum_absolute_raw_error": maximum,
            "maximum_product_identity_error": product_maximum,
            "tolerance": 1e-10,
            "passed": bool(passed),
        }

    def validate_runtime_replay(
        self,
        runtime_matrix: pd.DataFrame,
    ) -> MaterializationReplayReport:
        """Compare a complete runtime rematerialization to the frozen matrix."""
        self._load()

        def prediction(frame: pd.DataFrame) -> np.ndarray:
            uptake_raw, conditional_raw = self._model_bundle["model"].predict_raw(frame)
            uptake = self._model_bundle["uptake_calibrator"].predict(uptake_raw)
            conditional = self._model_bundle["conditional_calibrator"].predict(
                conditional_raw
            )
            return np.asarray(uptake, dtype=float) * np.asarray(
                conditional, dtype=float
            )

        report = validate_materialization_replay(
            runtime_matrix,
            self._matrix.reset_index(drop=True),
            prediction_func=prediction,
            rtol=1e-7,
            atol=1e-9,
        )
        self._approved_runtime_replay = (
            report
            if report.eligible_inference and report.row_count == len(self._matrix)
            else None
        )
        return report

    def load_approved_runtime_context(
        self,
    ) -> Tuple[ContextSnapshot, MaterializationReplayReport]:
        """Load the hash-pinned context only when the full replay gate passed."""
        self._load()
        if (
            self._runtime_context is not None
            and self._approved_runtime_replay is not None
        ):
            return self._runtime_context, self._approved_runtime_replay
        cache_key = self.config.runtime_replay_manifest_sha256
        cached = _RUNTIME_CONTEXT_CACHE.get(cache_key)
        if cached is not None:
            context, report, manifest = cached
            self._runtime_context = context
            self._approved_runtime_replay = report
            self._runtime_manifest = manifest
            return context, report
        manifest_path = self.config.resolved_runtime_replay_manifest()
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        if sha256_file(manifest_path) != self.config.runtime_replay_manifest_sha256:
            raise ValueError("runtime replay manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract") != "aspr_fulltext16_runtime_replay_release_v1":
            raise ValueError("unexpected runtime replay manifest contract")
        official_hash = self._matrix_manifest["outputs"]["matrix_16"]["sha256"]
        if manifest.get("official_matrix_sha256") != official_hash:
            raise ValueError("runtime replay targets a different official matrix")
        report = MaterializationReplayReport(**manifest.get("replay_report", {}))
        if not report.eligible_inference or report.row_count != len(self._matrix):
            raise ValueError("runtime replay did not pass the complete promotion gate")
        runtime_matrix_path = self.config.validate_asset_path(
            manifest_path.parent / "runtime_fulltext16.parquet"
        )
        context_path = self.config.validate_asset_path(
            manifest_path.parent / "context_snapshot.joblib"
        )
        if sha256_file(runtime_matrix_path) != manifest.get("runtime_matrix_sha256"):
            raise ValueError("runtime replay matrix hash mismatch")
        if sha256_file(context_path) != manifest.get("context_snapshot_sha256"):
            raise ValueError("runtime context snapshot hash mismatch")
        context = joblib.load(context_path)
        if not isinstance(context, ContextSnapshot):
            raise TypeError("runtime context snapshot has an unexpected type")
        online_year = int(
            manifest.get("input_manifest", {}).get("online_context_year", 0)
        )
        if online_year <= 0 or context.source_max_year != online_year - 1:
            raise ValueError("runtime context cutoff differs from replay manifest")
        self._approved_runtime_replay = report
        self._runtime_context = context
        self._runtime_manifest = manifest
        _RUNTIME_CONTEXT_CACHE[cache_key] = (context, report, manifest)
        return context, report

    def eligible_inference(
        self,
        target: TargetT0Record,
        context: ContextSnapshot,
        *,
        replay_report: Optional[MaterializationReplayReport] = None,
    ) -> CalibrationPacketV3:
        """Run online D5 inference only after the full frozen replay gate."""
        _, approved_report = self.load_approved_runtime_context()
        if (
            self._approved_runtime_replay is None
            or (replay_report is not None and replay_report != approved_report)
            or not approved_report.eligible_inference
            or approved_report.row_count != len(self._matrix)
        ):
            raise ValueError(
                "online Full-text 16 inference has not passed frozen replay"
            )
        values = materialize_fulltext16(target, context)
        row = coerce_fulltext16_storage_schema(
            pd.DataFrame([{name: values.get(name) for name in FEATURE_NAMES}])
        )
        uptake_raw, conditional_raw = self._model_bundle["model"].predict_raw(row)
        uptake = float(self._model_bundle["uptake_calibrator"].predict(uptake_raw)[0])
        conditional = float(
            self._model_bundle["conditional_calibrator"].predict(conditional_raw)[0]
        )
        raw = uptake * conditional
        d5_folds = self._fold_metrics[
            self._fold_metrics["horizon"].eq(5)
            & self._fold_metrics["model_family"].eq("hgb")
            & self._fold_metrics["model_id"].eq("fulltext_16")
        ]
        if d5_folds.empty:
            raise ValueError("official mature D5 year boundary is unavailable")
        mature_year_max = int(d5_folds["test_year_max"].max())
        mature_ids = self._metadata.index[
            pd.to_numeric(self._metadata["publication_year"], errors="coerce").le(
                mature_year_max
            )
        ]
        reference = np.sort(
            pd.to_numeric(
                self._scores.reindex(mature_ids)["raw_prediction_score"],
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
        )
        if not len(reference):
            raise ValueError("official mature D5 score reference is unavailable")
        score = (
            100.0
            * float(np.searchsorted(reference, raw, side="right"))
            / len(reference)
        )
        feature_values = {
            name: _safe_scalar(row.iloc[0][name]) for name in FEATURE_NAMES
        }
        missing = [name for name, value in feature_values.items() if value is None]
        bands = self._historical_bands(
            target.paper_id,
            int(target.publication_year),
            "",
            feature_values,
        )
        maximum_frozen_year = int(
            pd.to_numeric(self._metadata["publication_year"], errors="coerce").max()
        )
        drift_flags = []
        if int(target.publication_year) > maximum_frozen_year:
            drift_flags.append("post_frozen_cohort_year")
        if int(context.source_max_year) < int(target.publication_year) - 1:
            drift_flags.append("historical_context_lag")
        return CalibrationPacketV3(
            paper_id=target.paper_id,
            cutoff=CalibrationCutoff(
                publication_year=int(target.publication_year),
                source_max_year=int(context.source_max_year),
                granularity="year",
            ),
            measurement=CalibrationMeasurement(
                substantive_innovation={
                    name: feature_values[name]
                    for name in FEATURE_ROLES["substantive_innovation"]
                },
                t0_potential={
                    name: feature_values[name] for name in FEATURE_ROLES["t0_potential"]
                },
                opportunity={
                    name: feature_values[name] for name in FEATURE_ROLES["opportunity"]
                },
                context_control={
                    name: feature_values[name]
                    for name in FEATURE_ROLES["context_control"]
                },
                local_surrogates=list(SURROGATES),
                historical_bands=bands,
            ),
            forecast=CalibrationForecast(
                p_uptake=uptake,
                conditional_diffusion=conditional,
                raw_expected_diffusion=raw,
                aspr_score_0_100=score,
            ),
            reliability=CalibrationReliability(
                mode=CalibrationMode.ELIGIBLE_INFERENCE,
                temporal_block=None,
                feature_coverage=(len(FEATURE_NAMES) - len(missing))
                / len(FEATURE_NAMES),
                missing_features=missing,
                drift_flags=drift_flags,
                quality_flags=["full_frozen_matrix_replay_passed"],
            ),
            provenance=CalibrationProvenance(
                calibration_release_id=self.release_id,
                model_sha256=self._official["production"]["model_sha256"],
                score_table_sha256=self._official["production"]["score_sha256"],
                feature_matrix_sha256=self._matrix_manifest["outputs"]["matrix_16"][
                    "sha256"
                ],
                reference_corpus_sha256=sha256_file(self.paths.paper_metadata),
                runtime_replay_manifest_sha256=self.config.runtime_replay_manifest_sha256,
                runtime_matrix_sha256=self._runtime_manifest.get(
                    "runtime_matrix_sha256"
                ),
                context_snapshot_sha256=self._runtime_manifest.get(
                    "context_snapshot_sha256"
                ),
            ),
        )

    def _historical_bands(
        self,
        paper_id: str,
        publication_year: int,
        domain: str,
        values: Mapping[str, Any],
    ) -> Dict[str, str]:
        metadata = self._metadata[["paper_id", "publication_year", "domain12"]]
        eligible_ids = metadata.index[
            (
                pd.to_numeric(metadata["publication_year"], errors="coerce")
                < publication_year
            )
            & (metadata["domain12"].astype(str) == domain)
        ]
        if len(eligible_ids) < 100:
            eligible_ids = metadata.index[
                pd.to_numeric(metadata["publication_year"], errors="coerce")
                < publication_year
            ]
        bands: Dict[str, str] = {}
        for feature in (
            *FEATURE_ROLES["substantive_innovation"],
            *FEATURE_ROLES["t0_potential"],
        ):
            value = values.get(feature)
            if value is None or isinstance(value, str) or not len(eligible_ids):
                bands[feature] = "unavailable"
                continue
            history = pd.to_numeric(
                self._matrix.loc[
                    self._matrix.index.intersection(eligible_ids), feature
                ],
                errors="coerce",
            ).dropna()
            if len(history) < 30:
                bands[feature] = "unavailable"
                continue
            low = float(history.quantile(self.config.profile_low_quantile))
            high = float(history.quantile(self.config.profile_high_quantile))
            numeric = float(value)
            bands[feature] = (
                "low_extreme"
                if numeric <= low
                else ("high_extreme" if numeric >= high else "typical")
            )
        return bands

    def _temporal_block(self, year: int) -> Optional[str]:
        rows = self._fold_metrics[
            (self._fold_metrics["horizon"] == 5)
            & (self._fold_metrics["model_family"] == "hgb")
            & (self._fold_metrics["model_id"] == "fulltext_16")
            & (self._fold_metrics["test_year_min"] <= year)
            & (self._fold_metrics["test_year_max"] >= year)
        ]
        if rows.empty:
            return None
        row = rows.iloc[0]
        fold_id = int(row["outer_fold_id"])
        year_min = int(row["test_year_min"])
        year_max = int(row["test_year_max"])
        return f"fold_{fold_id}:{year_min}-{year_max}"

    def _fold_spearman(self, year: int) -> Optional[float]:
        rows = self._fold_metrics[
            (self._fold_metrics["horizon"] == 5)
            & (self._fold_metrics["model_family"] == "hgb")
            & (self._fold_metrics["model_id"] == "fulltext_16")
            & (self._fold_metrics["test_year_min"] <= year)
            & (self._fold_metrics["test_year_max"] >= year)
        ]
        return float(rows.iloc[0]["spearman"]) if not rows.empty else None

    def _unavailable_packet(
        self,
        paper_id: str,
        publication_date: Optional[date],
        reason: str,
        *,
        profile_only: bool = False,
    ) -> CalibrationPacketV3:
        year = publication_date.year if publication_date else None
        return CalibrationPacketV3(
            paper_id=paper_id,
            cutoff=CalibrationCutoff(
                publication_date=publication_date,
                publication_year=year,
                source_max_year=(year - 1) if year else None,
                granularity="year" if year else "unknown",
            ),
            measurement=CalibrationMeasurement(),
            forecast=CalibrationForecast(),
            reliability=CalibrationReliability(
                mode=(
                    CalibrationMode.PROFILE_ONLY
                    if profile_only
                    else CalibrationMode.UNAVAILABLE
                ),
                quality_flags=[reason],
            ),
            interpretation=CalibrationInterpretation(),
            provenance=CalibrationProvenance(
                calibration_release_id=self.release_id,
            ),
        )

    def _resolve_doi_via_openalex(self, doi: str) -> Optional[str]:
        if not self.config.allow_external_retrieval:
            return None
        url = "https://api.openalex.org/works/https://doi.org/" + quote(doi, safe="")
        params: Dict[str, str] = {}
        api_key = os.getenv("OPENALEX_API_KEY", "").strip()
        if api_key:
            params["api_key"] = api_key
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            return None
        return str(response.json().get("id") or "") or None


__all__ = [
    "CalibrationService",
    "FEATURE_NAMES",
    "FEATURE_ROLES",
    "SURROGATES",
    "normalize_openalex_id",
]
