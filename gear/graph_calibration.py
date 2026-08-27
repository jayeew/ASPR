"""Frozen Primary16 forecast anatomy and cutoff-safe analog selection."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .graph_prior_contracts import AnalogSeed, CalibrationTension, ForecastAnatomy

ROLE_FEATURES: dict[str, tuple[str, ...]] = {
    "substantive_innovation": ("EF0017", "EF0052", "EF0240"),
    "t0_potential": ("EF0309", "EF0312", "EF0315", "EF0318"),
    "opportunity": ("EF0083", "EF0186", "EF0188", "EF0238", "EF0319"),
    "context": ("EF0038", "EF0197", "EF0307", "EF0314"),
}
ROLE_NAMES = tuple(ROLE_FEATURES)
ANATOMY_MIN_COVERAGE = 0.5


def sha256_payload(value: object) -> str:
    """Hash a JSON-safe payload deterministically, including missing values."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map values to a frozen empirical percentile reference."""

    return 100.0 * np.searchsorted(reference, values, side="right") / len(reference)


def _predict_model(model: Any, frame: pd.DataFrame) -> pd.DataFrame:
    """Predict with the frozen canonical two-part Primary16 HGB bundle."""

    if not isinstance(model, dict):
        raise TypeError("Primary16 HGB bundle must be a dictionary")
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


def role_coverage(features: pd.DataFrame) -> pd.DataFrame:
    """Return observed-feature coverage separately for each frozen role."""

    return pd.DataFrame(
        {
            role: features.loc[:, names].notna().mean(axis=1)
            for role, names in ROLE_FEATURES.items()
        },
        index=features.index,
    )


def _coalition_weight(size: int) -> float:
    n_roles = len(ROLE_NAMES)
    return (
        math.factorial(size)
        * math.factorial(n_roles - size - 1)
        / math.factorial(n_roles)
    )


def group_shapley(
    model: dict[str, Any],
    target: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    feature_names: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Exact four-group Shapley values for uptake and conditional diffusion."""

    if len(target) != len(baseline):
        raise ValueError("target and baseline must have the same row count")
    subsets = [
        set(items)
        for size in range(5)
        for items in itertools.combinations(ROLE_NAMES, size)
    ]
    predictions: dict[frozenset[str], pd.DataFrame] = {}
    for subset in subsets:
        frame = baseline.loc[:, feature_names].copy()
        for role in subset:
            names = ROLE_FEATURES[role]
            frame.loc[:, names] = target.loc[:, names].to_numpy()
        predictions[frozenset(subset)] = _predict_model(model, frame)
    values = {
        "uptake": pd.DataFrame(0.0, index=target.index, columns=ROLE_NAMES),
        "conditional": pd.DataFrame(0.0, index=target.index, columns=ROLE_NAMES),
    }
    for role in ROLE_NAMES:
        for subset in subsets:
            if role in subset:
                continue
            before = frozenset(subset)
            after = frozenset({*subset, role})
            weight = _coalition_weight(len(subset))
            values["uptake"].loc[:, role] += weight * (
                predictions[after]["uptake_probability_calibrated"].to_numpy()
                - predictions[before]["uptake_probability_calibrated"].to_numpy()
            )
            values["conditional"].loc[:, role] += weight * (
                predictions[after]["conditional_diffusion_calibrated"].to_numpy()
                - predictions[before]["conditional_diffusion_calibrated"].to_numpy()
            )
    return values


def compute_anatomy(
    model: dict[str, Any],
    target: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    uptake_reference: np.ndarray,
    conditional_reference: np.ndarray,
    expected_reference: np.ndarray,
    baseline_ids: Sequence[str],
    release_id: str,
    target_fields: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute a batch of exact frozen-HGB anatomy rows without outcome access."""

    if len(target) != len(baseline) or len(target) != len(baseline_ids):
        raise ValueError("anatomy batch inputs have inconsistent lengths")
    prediction = _predict_model(model, target.loc[:, feature_names])
    values = group_shapley(model, target, baseline, feature_names=feature_names)
    coverage = role_coverage(target)
    output = pd.DataFrame(index=target.index)
    output["paper_id"] = target["paper_id"].astype(str)
    if target_fields is not None:
        if len(target_fields) != len(target):
            raise ValueError("anatomy target fields do not match target rows")
        output["target_field"] = list(target_fields)
    output["uptake_percentile"] = percentile(
        prediction["uptake_probability_calibrated"].to_numpy(float), uptake_reference
    )
    output["conditional_diffusion_percentile"] = percentile(
        prediction["conditional_diffusion_calibrated"].to_numpy(float),
        conditional_reference,
    )
    output["expected_diffusion_percentile"] = percentile(
        prediction["expected_diffusion_score"].to_numpy(float), expected_reference
    )
    for role in ROLE_NAMES:
        output[f"uptake_contribution__{role}"] = values["uptake"][role]
        output[f"conditional_contribution__{role}"] = values["conditional"][role]
        output[f"coverage__{role}"] = coverage[role]
    output["baseline_id"] = list(baseline_ids)
    output["feature_input_sha256"] = [
        sha256_payload(row)
        for row in target.loc[:, feature_names]
        .replace({np.nan: None})
        .to_dict("records")
    ]
    output["anatomy_release_id"] = release_id
    output["anatomy_limited"] = coverage.min(axis=1) < ANATOMY_MIN_COVERAGE
    return output.reset_index(drop=True)


def anatomy_from_row(row: Mapping[str, Any]) -> ForecastAnatomy:
    """Convert one stored anatomy row to the strict public packet projection."""

    limited = bool(row.get("anatomy_limited", True))
    uptake = {role: float(row[f"uptake_contribution__{role}"]) for role in ROLE_NAMES}
    conditional = {
        role: float(row[f"conditional_contribution__{role}"]) for role in ROLE_NAMES
    }
    coverage = {role: float(row[f"coverage__{role}"]) for role in ROLE_NAMES}
    return ForecastAnatomy(
        paper_id=str(row["paper_id"]),
        target_field=(
            str(row["target_field"]) if row.get("target_field") is not None else None
        ),
        uptake_percentile=float(row["uptake_percentile"]),
        conditional_diffusion_percentile=float(row["conditional_diffusion_percentile"]),
        expected_diffusion_percentile=float(row["expected_diffusion_percentile"]),
        uptake_role_contributions={} if limited else uptake,
        conditional_role_contributions={} if limited else conditional,
        role_coverage={} if limited else coverage,
        baseline_id=str(row.get("baseline_id") or ""),
        feature_input_sha256=str(row.get("feature_input_sha256") or ""),
        anatomy_release_id=str(row.get("anatomy_release_id") or ""),
        limited=limited,
    )


def calibration_tensions(anatomy: ForecastAnatomy | None) -> list[CalibrationTension]:
    """Derive frozen process-only tension checks from the HGB anatomy."""

    if anatomy is None or anatomy.limited:
        return []
    positive = {
        role: max(0.0, anatomy.uptake_role_contributions[role])
        + max(0.0, anatomy.conditional_role_contributions[role])
        for role in ROLE_NAMES
    }
    total = sum(positive.values())
    if total <= 0.0:
        return []
    opportunity_share = (positive["opportunity"] + positive["context"]) / total
    structural_share = (
        positive["substantive_innovation"] + positive["t0_potential"]
    ) / total
    high = float(anatomy.expected_diffusion_percentile or 0.0) >= 67.0
    cross_field = float(anatomy.conditional_diffusion_percentile or 0.0) >= 67.0
    tensions = []
    if high and opportunity_share >= 0.60:
        tensions.append(
            CalibrationTension(
                kind="opportunity_dominant",
                active=True,
                score=float(opportunity_share),
                review_effect="antecedent_attribution_check",
            )
        )
    if cross_field and structural_share >= 0.60:
        tensions.append(
            CalibrationTension(
                kind="integration_dominant",
                active=True,
                score=float(structural_share),
                review_effect="cross_field_bridge_check",
            )
        )
    return tensions


@dataclass(frozen=True)
class AnalogCandidate:
    work_id: str
    title: str
    publication_year: int
    field: str
    semantic_score: float
    anatomy_score: float
    combined_score: float


class ForecastAnalogIndex:
    """Frozen semantic-gated HGB anatomy index; it never reads outcomes."""

    def __init__(self, path: Path, manifest: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.manifest = dict(manifest)
        self._frame: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._frame is None:
            self._frame = pd.read_parquet(self.path)
        return self._frame

    def table(self) -> pd.DataFrame:
        """Return the verified frozen index for exact target anatomy lookup."""

        return self._load()

    @staticmethod
    def _semantic_score(title: str, terms: Iterable[str]) -> float:
        tokens = {value.casefold() for value in re_tokens(title)}
        query = {value.casefold() for value in terms}
        return len(tokens & query) / max(1, len(query))

    def select(
        self,
        anatomy: ForecastAnatomy | None,
        *,
        claim_id: str,
        terms: Sequence[str],
        cutoff_date: date,
        target_field: str | None,
        shuffled: bool = False,
    ) -> list[AnalogSeed]:
        if anatomy is None or anatomy.limited:
            return []
        frame = self._load()
        eligible = frame.loc[
            frame["publication_year"].astype(int) < cutoff_date.year
        ].copy()
        eligible["semantic_score"] = eligible["title"].map(
            lambda value: self._semantic_score(str(value), terms)
        )
        semantic = eligible.nlargest(200, "semantic_score")
        semantic = semantic.loc[semantic["semantic_score"] > 0.0].copy()
        if semantic.empty:
            return []
        target_vector = np.array(
            [anatomy.uptake_role_contributions[role] for role in ROLE_NAMES]
            + [anatomy.conditional_role_contributions[role] for role in ROLE_NAMES],
            dtype=float,
        )
        if shuffled:
            shift = int(hashlib.sha256(claim_id.encode("utf-8")).hexdigest()[:8], 16)
            target_vector = np.roll(target_vector, 1 + shift % (len(target_vector) - 1))
        weights = np.array(
            [anatomy.role_coverage[role] for role in ROLE_NAMES] * 2, dtype=float
        )
        columns = [
            *(f"uptake_contribution__{role}" for role in ROLE_NAMES),
            *(f"conditional_contribution__{role}" for role in ROLE_NAMES),
        ]
        matrix = semantic.loc[:, columns].to_numpy(dtype=float)
        distance = np.sqrt(
            ((matrix - target_vector) ** 2 * weights).sum(axis=1) / weights.sum()
        )
        semantic["anatomy_score"] = 1.0 / (1.0 + distance)
        semantic["combined_score"] = (
            0.65 * semantic["semantic_score"] + 0.35 * semantic["anatomy_score"]
        )
        selected: list[tuple[str, pd.Series]] = []
        local = (
            semantic
            if not target_field
            else semantic.loc[semantic["field"] == target_field]
        )
        remote = (
            semantic
            if not target_field
            else semantic.loc[semantic["field"] != target_field]
        )
        for lane, candidates in (
            ("local_adoption", local),
            ("cross_field_bridge", remote),
        ):
            if not candidates.empty:
                selected.append(
                    (lane, candidates.nlargest(1, "combined_score").iloc[0])
                )
        return [
            AnalogSeed(
                claim_id=claim_id,
                work_id=str(row["paper_id"]),
                title=str(row["title"]),
                lane=lane,  # type: ignore[arg-type]
                semantic_score=float(row["semantic_score"]),
                anatomy_score=float(row["anatomy_score"]),
                combined_score=float(row["combined_score"]),
                publication_year=int(row["publication_year"]),
                cutoff_date=cutoff_date,
                source_snapshot_id=str(self.manifest["source_snapshot_id"]),
                source_snapshot_sha256=str(self.manifest["source_snapshot_sha256"]),
                text_sha256=str(row.get("title_sha256") or "") or None,
                text_version="frozen_metadata_title_v1",
                cutoff_valid=True,
            )
            for lane, row in selected
        ]


def re_tokens(text: str) -> list[str]:
    """Tokenize immutable title text without importing retrieval infrastructure."""

    return [token for token in re.findall(r"[A-Za-z0-9]+", text) if len(token) > 2]


def load_forecast_analog_index(manifest_path: Path) -> ForecastAnalogIndex:
    """Load and integrity-check the one immutable anatomy candidate index."""

    manifest_file = Path(manifest_path).resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("contract") != "gear_primary16_forecast_anatomy_release_v1":
        raise ValueError("unsupported forecast anatomy release")
    if manifest.get("uses_review_or_relation_labels") is not False:
        raise ValueError("forecast anatomy release may not use review labels")
    if manifest.get("uses_future_citation_outcomes") is not False:
        raise ValueError("forecast anatomy release may not use future outcomes")
    asset = manifest.get("assets", {}).get("anatomy_index")
    if not isinstance(asset, dict):
        raise TypeError("forecast anatomy index asset is missing")
    path = (manifest_file.parent / str(asset.get("file", ""))).resolve()
    if not path.is_file() or sha256_file(path) != asset.get("sha256"):
        raise ValueError("forecast anatomy index hash mismatch")
    return ForecastAnalogIndex(path, manifest)


def sha256_file(path: Path) -> str:
    """Return a stable asset hash for release-level provenance validation."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "ANATOMY_MIN_COVERAGE",
    "ROLE_FEATURES",
    "ROLE_NAMES",
    "ForecastAnalogIndex",
    "anatomy_from_row",
    "calibration_tensions",
    "compute_anatomy",
    "group_shapley",
    "load_forecast_analog_index",
    "role_coverage",
]
