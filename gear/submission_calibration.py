"""Journal-free submission-time calibration with an explicit promotion gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from gear.nature_multihorizon.modeling_v6 import FittedTwoPartModel
from gear.nature_multihorizon.t0_runtime_v3 import (
    ContextSnapshot,
    TargetT0Record,
    materialize_fulltext16,
)

from .calibration import FEATURE_NAMES, FEATURE_ROLES, SURROGATES, sha256_file
from .config import GearConfig, load_config
from .contracts import (
    CalibrationCutoff,
    CalibrationForecast,
    CalibrationInterpretation,
    CalibrationMeasurement,
    CalibrationMode,
    CalibrationProvenance,
    CalibrationReliability,
    PaperIR,
    SubmissionCalibrationPacketV1,
)

SUBMISSION_FEATURES = tuple(sorted(name for name in FEATURE_NAMES if name != "EF0197"))


class SubmissionCalibrationService:
    """Load a promoted submission model or return a transparent profile only."""

    def __init__(
        self,
        config: Optional[GearConfig] = None,
        *,
        asset_dir: Optional[Path] = None,
    ) -> None:
        self.config = config or load_config()
        self.asset_dir = Path(
            asset_dir or self.config.resolve_path("outputs/gear/submission_calibration")
        ).resolve()
        self.manifest_path = self.asset_dir / "submission_model_manifest.json"
        self._bundle: Optional[Dict[str, Any]] = None
        self._reference: Optional[np.ndarray] = None
        self._manifest: Dict[str, Any] = {}

    def build_packet(
        self,
        target: TargetT0Record,
        context: ContextSnapshot,
    ) -> SubmissionCalibrationPacketV1:
        values = materialize_fulltext16(
            target,
            context,
            include_journal_identity=False,
        )
        values.pop("EF0197", None)
        measurement = self._measurement(values)
        if not self._load_promoted():
            return SubmissionCalibrationPacketV1(
                paper_id=target.paper_id,
                cutoff=CalibrationCutoff(
                    publication_year=int(target.publication_year),
                    source_max_year=int(context.source_max_year),
                    granularity="year",
                ),
                measurement=measurement,
                forecast=CalibrationForecast(reference_corpus="submission-safe-d5"),
                reliability=CalibrationReliability(
                    mode=CalibrationMode.PROFILE_ONLY,
                    feature_coverage=self._coverage(values),
                    missing_features=[
                        name for name in SUBMISSION_FEATURES if values.get(name) is None
                    ],
                    quality_flags=["submission_model_not_promoted"],
                ),
                interpretation=CalibrationInterpretation(),
                provenance=CalibrationProvenance(),
            )
        assert self._bundle is not None and self._reference is not None
        row = pd.DataFrame([{name: values.get(name) for name in SUBMISSION_FEATURES}])
        uptake_raw, conditional_raw = self._bundle["model"].predict_raw(row)
        uptake = float(self._bundle["uptake_calibrator"].predict(uptake_raw)[0])
        conditional = float(
            self._bundle["conditional_calibrator"].predict(conditional_raw)[0]
        )
        raw = uptake * conditional
        score = (
            100.0
            * float(np.searchsorted(self._reference, raw, side="right"))
            / len(self._reference)
        )
        production = self._manifest["production"]
        return SubmissionCalibrationPacketV1(
            paper_id=target.paper_id,
            cutoff=CalibrationCutoff(
                publication_year=int(target.publication_year),
                source_max_year=int(context.source_max_year),
                granularity="year",
            ),
            measurement=measurement,
            forecast=CalibrationForecast(
                p_uptake=uptake,
                conditional_diffusion=conditional,
                raw_expected_diffusion=raw,
                aspr_score_0_100=score,
                reference_corpus="submission-safe-d5",
            ),
            reliability=CalibrationReliability(
                mode=CalibrationMode.ELIGIBLE_INFERENCE,
                feature_coverage=self._coverage(values),
                missing_features=[
                    name for name in SUBMISSION_FEATURES if values.get(name) is None
                ],
                quality_flags=["submission_model_preregistered_gates_passed"],
            ),
            interpretation=CalibrationInterpretation(),
            provenance=CalibrationProvenance(
                model_sha256=production["model_sha256"],
                reference_corpus_sha256=production["reference_sha256"],
            ),
        )

    def build_from_paper_ir(
        self,
        paper_ir: PaperIR,
        *,
        reason: str = "submission_context_snapshot_unavailable",
    ) -> SubmissionCalibrationPacketV1:
        """Return a journal-free manuscript profile when T0 context is absent."""
        publication_date = (
            paper_ir.metadata.submission_date or paper_ir.metadata.publication_date
        )
        year = publication_date.year if publication_date is not None else None
        values: Dict[str, Any] = {name: None for name in SUBMISSION_FEATURES}
        values["EF0038"] = (
            float(len(paper_ir.metadata.authors)) if paper_ir.metadata.authors else None
        )
        values["EF0307"] = float(year) if year is not None else None
        values["EF0314"] = float(len(paper_ir.references))
        missing = [name for name in SUBMISSION_FEATURES if values.get(name) is None]
        return SubmissionCalibrationPacketV1(
            paper_id=paper_ir.paper_id,
            cutoff=CalibrationCutoff(
                publication_date=publication_date,
                publication_year=year,
                source_max_year=(year - 1) if year is not None else None,
                granularity="year" if year is not None else "unknown",
            ),
            measurement=self._measurement(values),
            forecast=CalibrationForecast(reference_corpus="submission-safe-d5"),
            reliability=CalibrationReliability(
                mode=CalibrationMode.PROFILE_ONLY,
                feature_coverage=self._coverage(values),
                missing_features=missing,
                quality_flags=[reason],
            ),
            interpretation=CalibrationInterpretation(),
            provenance=CalibrationProvenance(),
        )

    def _load_promoted(self) -> bool:
        if self._bundle is not None and self._reference is not None:
            return True
        if not self.manifest_path.is_file():
            return False
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("contract") != "aspr_submission_calibration_training_v1"
            or manifest.get("feature_set") != "submission_safe_15"
            or int(manifest.get("horizon", -1)) != 5
        ):
            raise ValueError("unsupported submission model manifest contract")
        if manifest.get("promotion_status") != "passed":
            return False
        if not bool((manifest.get("gates") or {}).get("passed")):
            raise ValueError("submission model claims promotion without passing gates")
        if tuple(manifest.get("features") or ()) != SUBMISSION_FEATURES:
            raise ValueError(
                "promoted submission feature contract differs from runtime"
            )
        if "EF0197" in set(manifest.get("features") or ()):
            raise ValueError("journal identity is forbidden in submission calibration")
        production = manifest.get("production") or {}
        model_path = Path(production.get("model_path") or "").resolve()
        reference_path = Path(production.get("reference_path") or "").resolve()
        for path in (model_path, reference_path):
            try:
                path.relative_to(self.asset_dir)
            except ValueError as exc:
                raise ValueError(
                    "submission artifact is outside its promoted asset directory"
                ) from exc
        if not model_path.is_file() or not reference_path.is_file():
            raise ValueError("promoted submission artifacts are missing")
        if sha256_file(model_path) != production.get("model_sha256"):
            raise ValueError("submission model hash mismatch")
        if sha256_file(reference_path) != production.get("reference_sha256"):
            raise ValueError("submission score reference hash mismatch")
        bundle = joblib.load(model_path)
        if set(bundle.get("feature_names") or ()) != set(SUBMISSION_FEATURES):
            raise ValueError("submission model bundle has unexpected features")
        if not isinstance(bundle.get("model"), FittedTwoPartModel):
            raise ValueError("submission model bundle has an unexpected model type")
        if tuple(bundle["model"].feature_names) != SUBMISSION_FEATURES:
            raise ValueError("submission model feature order differs from runtime")
        if tuple(bundle["model"].categorical_names):
            raise ValueError("submission model cannot contain categorical features")
        reference = np.load(reference_path, allow_pickle=False)
        if (
            reference.ndim != 1
            or not len(reference)
            or not np.isfinite(reference).all()
        ):
            raise ValueError("submission score reference is invalid")
        self._manifest = manifest
        self._bundle = bundle
        self._reference = np.sort(np.asarray(reference, dtype=float))
        return True

    @staticmethod
    def _coverage(values: Dict[str, Any]) -> float:
        present = sum(values.get(name) is not None for name in SUBMISSION_FEATURES)
        return present / len(SUBMISSION_FEATURES)

    @staticmethod
    def _measurement(values: Dict[str, Any]) -> CalibrationMeasurement:
        return CalibrationMeasurement(
            feature_set="submission_safe_15",
            feature_version="submission_v1",
            substantive_innovation={
                name: values.get(name)
                for name in FEATURE_ROLES["substantive_innovation"]
            },
            t0_potential={
                name: values.get(name) for name in FEATURE_ROLES["t0_potential"]
            },
            opportunity={
                name: values.get(name)
                for name in FEATURE_ROLES["opportunity"]
                if name != "EF0197"
            },
            context_control={
                name: values.get(name)
                for name in FEATURE_ROLES["context_control"]
                if name != "EF0197"
            },
            local_surrogates=list(SURROGATES),
        )


__all__ = ["SUBMISSION_FEATURES", "SubmissionCalibrationService"]
