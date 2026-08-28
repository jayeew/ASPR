"""Point-in-time-safe claim attribution with frozen learned-head promotion."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, model_validator

from .contracts import ClaimType, StrictModel
from .graph_prior_contracts import (
    ClaimAttributionAudit,
    ClaimGraphPrior,
    ClaimInventoryEntry,
    GraphRuntimePacket,
    GraphSignalBundle,
)

FEATURE_SCHEMA_VERSION: Literal["gear_claim_attribution_t0_features_v1"] = (
    "gear_claim_attribution_t0_features_v1"
)
ForecastRole = Literal[
    "substantive_innovation", "t0_potential", "opportunity", "context", "unknown"
]
PathwayType = Literal[
    "local_method_adoption",
    "cross_field_bridge",
    "reusable_resource",
    "platform_scaling",
    "mechanism_transfer",
    "unspecified",
]
CLAIM_TYPES = tuple(item.value for item in ClaimType)
FORECAST_ROLES = (
    "substantive_innovation",
    "t0_potential",
    "opportunity",
    "context",
)
PATHWAYS = (
    "local_method_adoption",
    "cross_field_bridge",
    "reusable_resource",
    "platform_scaling",
    "mechanism_transfer",
    "unspecified",
)
T0_FEATURE_NAMES = (
    "claim_centrality",
    *(f"claim_type__{value}" for value in CLAIM_TYPES),
    *(f"anatomy_role__{value}" for value in FORECAST_ROLES),
    *(f"pathway__{value}" for value in PATHWAYS),
)


class ClaimAttributionLinearHead(StrictModel):
    """Portable numeric model; loading never executes serialized Python code."""

    contract: Literal["gear_claim_attribution_linear_head_v1"] = (
        "gear_claim_attribution_linear_head_v1"
    )
    feature_names: list[str]
    coefficients: list[float]
    intercept: float = 0.0

    @model_validator(mode="after")
    def exact_feature_schema(self) -> ClaimAttributionLinearHead:
        if tuple(self.feature_names) != T0_FEATURE_NAMES:
            raise ValueError(
                "claim-attribution head feature schema is not frozen T0 v1"
            )
        if len(self.coefficients) != len(self.feature_names):
            raise ValueError("claim-attribution coefficient count mismatch")
        if not all(
            math.isfinite(value) for value in [self.intercept, *self.coefficients]
        ):
            raise ValueError("claim-attribution model contains non-finite coefficients")
        return self

    def predict(self, rows: Sequence[Sequence[float]]) -> list[float]:
        return [
            max(
                0.0,
                self.intercept
                + sum(
                    coefficient * float(value)
                    for coefficient, value in zip(self.coefficients, row, strict=True)
                ),
            )
            for row in rows
        ]


class ClaimAttributionRelease(StrictModel):
    """Hash-bound promotion sidecar for a development-only learned head."""

    contract: Literal["gear_claim_attribution_release_v1"] = (
        "gear_claim_attribution_release_v1"
    )
    release_id: str
    status: Literal["promoted"]
    feature_schema_version: Literal["gear_claim_attribution_t0_features_v1"]
    feature_names: list[str]
    model_path: str
    model_sha256: str
    replay_path: str
    replay_sha256: str
    gate1_report_paths: list[str] = Field(min_length=2)
    gate1_report_sha256: list[str] = Field(min_length=2)
    training_target: Literal["future_claim_adoption_share_given_any_adoption"]
    training_features_t0_only: Literal[True]
    development_only: Literal[True]
    sealed_holdout_labels_used: Literal[False]
    future_contexts_used_at_inference: Literal[False]

    @model_validator(mode="after")
    def frozen_schema_and_references(self) -> ClaimAttributionRelease:
        if tuple(self.feature_names) != T0_FEATURE_NAMES:
            raise ValueError("release does not use the frozen T0 feature schema")
        if len(self.gate1_report_paths) != len(self.gate1_report_sha256):
            raise ValueError("Gate-1 report paths and hashes differ in length")
        if len(set(self.gate1_report_paths)) != len(self.gate1_report_paths):
            raise ValueError("Gate-1 promotion reports must be distinct")
        return self


def pathway_hypothesis(item: ClaimInventoryEntry) -> PathwayType:
    """Derive the predeclared pathway only from manuscript-time claim fields."""
    text = item.text.casefold()
    if any(token in text for token in ("dataset", "resource", "benchmark", "database")):
        return "reusable_resource"
    if any(token in text for token in ("platform", "framework", "pipeline", "system")):
        return "platform_scaling"
    if any(
        token in text for token in ("across", "transfer", "generaliz", "cross-field")
    ):
        return "cross_field_bridge"
    if item.claim_type in {ClaimType.METHOD.value, ClaimType.NOVELTY.value}:
        return "local_method_adoption"
    if item.claim_type == ClaimType.CAUSAL.value:
        return "mechanism_transfer"
    return "unspecified"


def t0_feature_row(
    item: ClaimInventoryEntry, packet: GraphRuntimePacket
) -> tuple[list[float], PathwayType]:
    """Materialize only claim/manuscript and point-in-time forecast anatomy."""
    pathway = pathway_hypothesis(item)
    anatomy = _normalized_anatomy(packet)
    row = [float(item.centrality)]
    row.extend(float(item.claim_type == value) for value in CLAIM_TYPES)
    row.extend(anatomy[value] for value in FORECAST_ROLES)
    row.extend(float(pathway == value) for value in PATHWAYS)
    return row, pathway


def deterministic_t0_attribution(
    inventory: Sequence[ClaimInventoryEntry],
    bundle: GraphSignalBundle,
    packet: GraphRuntimePacket,
) -> tuple[list[ClaimGraphPrior], ClaimAttributionAudit]:
    """Run the declared phase-one baseline; this is not a learned model."""
    rows = [t0_feature_row(item, packet) for item in inventory]
    scores = [
        _deterministic_score(item, row, pathway)
        for item, (row, pathway) in zip(inventory, rows, strict=True)
    ]
    diagnostics = (
        [] if packet.forecast_anatomy is not None else ["forecast_anatomy_unavailable"]
    )
    priors = _build_priors(
        inventory,
        bundle,
        packet,
        scores,
        [pathway for _, pathway in rows],
        method="deterministic_t0",
        release_id=None,
        diagnostics=diagnostics,
    )
    return priors, ClaimAttributionAudit(
        requested_mode="deterministic_t0",
        applied_mode="deterministic_t0",
        status="available",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=list(T0_FEATURE_NAMES),
        diagnostics=diagnostics,
    )


def learned_t0_attribution(
    inventory: Sequence[ClaimInventoryEntry],
    bundle: GraphSignalBundle,
    packet: GraphRuntimePacket,
    manifest_path: Path | None,
) -> tuple[list[ClaimGraphPrior], ClaimAttributionAudit]:
    """Run a verified learned head or return an explicit limited result."""
    try:
        if manifest_path is None:
            raise ValueError("learned_manifest_unconfigured")
        release, model = load_claim_attribution_release(manifest_path)
        rows = [t0_feature_row(item, packet) for item in inventory]
        scores = model.predict([row for row, _ in rows])
        priors = _build_priors(
            inventory,
            bundle,
            packet,
            scores,
            [pathway for _, pathway in rows],
            method="learned_t0",
            release_id=release.release_id,
            diagnostics=[],
        )
        return priors, ClaimAttributionAudit(
            requested_mode="learned_t0",
            applied_mode="learned_t0",
            status="available",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_names=list(T0_FEATURE_NAMES),
            release_id=release.release_id,
            manifest_sha256=_sha256(manifest_path),
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        reason = f"learned_head_unavailable:{type(exc).__name__}:{exc}"
        return [], ClaimAttributionAudit(
            requested_mode="learned_t0",
            applied_mode="unavailable",
            status="limited",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            feature_names=list(T0_FEATURE_NAMES),
            diagnostics=[reason],
        )


def load_claim_attribution_release(
    manifest_path: Path,
) -> tuple[ClaimAttributionRelease, ClaimAttributionLinearHead]:
    """Verify every promoted artifact before parsing the non-executable model."""
    path = Path(manifest_path).resolve()
    release = ClaimAttributionRelease.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    model_path = _bound_path(path, release.model_path)
    replay_path = _bound_path(path, release.replay_path)
    _require_hash(model_path, release.model_sha256)
    _require_hash(replay_path, release.replay_sha256)
    axes = set()
    for report_name, expected_hash in zip(
        release.gate1_report_paths, release.gate1_report_sha256, strict=True
    ):
        report_path = _bound_path(path, report_name)
        _require_hash(report_path, expected_hash)
        axes.add(_verify_gate1_report(report_path, release.model_sha256))
    if not {"temporal", "domain"}.issubset(axes):
        raise ValueError("promotion lacks distinct temporal and domain Gate-1 passes")
    model = ClaimAttributionLinearHead.model_validate_json(
        model_path.read_text(encoding="utf-8")
    )
    validate_claim_attribution_replay(path, release=release, model=model)
    return release, model


def validate_claim_attribution_replay(
    manifest_path: Path,
    *,
    release: ClaimAttributionRelease | None = None,
    model: ClaimAttributionLinearHead | None = None,
) -> dict[str, Any]:
    """Require exact replay of frozen T0 rows and normalized paper weights."""
    path = Path(manifest_path).resolve()
    if release is None or model is None:
        payload = ClaimAttributionRelease.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        model_path = _bound_path(path, payload.model_path)
        _require_hash(model_path, payload.model_sha256)
        release, model = payload, ClaimAttributionLinearHead.model_validate_json(
            model_path.read_text(encoding="utf-8")
        )
    replay_path = _bound_path(path, release.replay_path)
    _require_hash(replay_path, release.replay_sha256)
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("claim-attribution replay is empty")
    features = [row.get("features") for row in rows]
    if any(
        not isinstance(row, list) or len(row) != len(T0_FEATURE_NAMES)
        for row in features
    ):
        raise ValueError("claim-attribution replay feature shape mismatch")
    if any(
        not all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in row
        )
        for row in features
    ):
        raise ValueError("claim-attribution replay contains non-finite features")
    raw = model.predict(features)
    expected_raw = [float(row["expected_raw_score"]) for row in rows]
    max_raw_error = max(
        abs(left - right) for left, right in zip(raw, expected_raw, strict=True)
    )
    weights = _normalize_by_paper([str(row["paper_id"]) for row in rows], raw)
    expected_weights = [float(row["expected_attribution_weight"]) for row in rows]
    max_weight_error = max(
        abs(left - right) for left, right in zip(weights, expected_weights, strict=True)
    )
    if max(max_raw_error, max_weight_error) > 1e-12:
        raise ValueError("claim-attribution runtime replay mismatch")
    return {
        "contract": "gear_claim_attribution_replay_validation_v1",
        "passed": True,
        "release_id": release.release_id,
        "rows": len(rows),
        "max_raw_error": max_raw_error,
        "max_weight_error": max_weight_error,
    }


def promote_claim_attribution_release(
    *,
    model_path: Path,
    replay_path: Path,
    gate1_report_paths: Sequence[Path],
    output_path: Path,
    release_id: str,
) -> ClaimAttributionRelease:
    """Promote only a replayable model evaluated by two hash-bound Gate-1 runs."""
    model = ClaimAttributionLinearHead.model_validate_json(
        model_path.read_text(encoding="utf-8")
    )
    model_hash = _sha256(model_path)
    if len(gate1_report_paths) < 2:
        raise ValueError("promotion requires temporal and domain Gate-1 reports")
    axes = {
        _verify_gate1_report(report_path, model_hash)
        for report_path in gate1_report_paths
    }
    if not {"temporal", "domain"}.issubset(axes):
        raise ValueError("promotion lacks distinct temporal and domain Gate-1 passes")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    release = ClaimAttributionRelease(
        release_id=release_id,
        status="promoted",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=list(T0_FEATURE_NAMES),
        model_path=_relative_reference(output_path, model_path),
        model_sha256=model_hash,
        replay_path=_relative_reference(output_path, replay_path),
        replay_sha256=_sha256(replay_path),
        gate1_report_paths=[
            _relative_reference(output_path, value) for value in gate1_report_paths
        ],
        gate1_report_sha256=[_sha256(value) for value in gate1_report_paths],
        training_target="future_claim_adoption_share_given_any_adoption",
        training_features_t0_only=True,
        development_only=True,
        sealed_holdout_labels_used=False,
        future_contexts_used_at_inference=False,
    )
    validate_claim_attribution_replay(output_path, release=release, model=model)
    output_path.write_text(release.model_dump_json(indent=2) + "\n", encoding="utf-8")
    load_claim_attribution_release(output_path)
    return release


def _deterministic_score(
    item: ClaimInventoryEntry, row: Sequence[float], pathway: str
) -> float:
    type_signal = {
        ClaimType.NOVELTY.value: 1.0,
        ClaimType.METHOD.value: 0.95,
        ClaimType.CAUSAL.value: 0.9,
        ClaimType.RESULT.value: 0.8,
        ClaimType.SIGNIFICANCE.value: 0.75,
        ClaimType.SCOPE.value: 0.55,
    }.get(item.claim_type, 0.5)
    anatomy_offset = 1 + len(CLAIM_TYPES)
    anatomy = dict(
        zip(
            FORECAST_ROLES,
            row[anatomy_offset : anatomy_offset + len(FORECAST_ROLES)],
            strict=True,
        )
    )
    structural = anatomy["substantive_innovation"] + anatomy["t0_potential"]
    pathway_signal = 0.5 if pathway == "unspecified" else 1.0
    return max(
        1e-9,
        0.55 * item.centrality
        + 0.2 * type_signal
        + 0.15 * structural
        + 0.1 * pathway_signal,
    )


def _build_priors(
    inventory: Sequence[ClaimInventoryEntry],
    bundle: GraphSignalBundle,
    packet: GraphRuntimePacket,
    scores: Sequence[float],
    pathways: Sequence[PathwayType],
    *,
    method: Literal["deterministic_t0", "learned_t0"],
    release_id: str | None,
    diagnostics: list[str],
) -> list[ClaimGraphPrior]:
    weights = _normalize_by_paper([bundle.paper_id] * len(scores), scores)
    dominant = _dominant_role(packet)
    return [
        ClaimGraphPrior(
            claim_id=item.claim_id,
            attribution_weight=weight,
            diffusion_prior=bundle.shrunk_diffusion * weight,
            perturbation_prior=(
                None
                if bundle.perturbation_potential is None
                else bundle.perturbation_potential * weight
            ),
            dominant_forecast_role=dominant,
            pathway_hypothesis=pathway,
            confidence=bundle.reliability * (0.5 + 0.5 * item.centrality),
            attribution_method=method,
            attribution_release_id=release_id,
            diagnostics=diagnostics,
        )
        for item, weight, pathway in zip(inventory, weights, pathways, strict=True)
    ]


def _normalized_anatomy(packet: GraphRuntimePacket) -> dict[str, float]:
    anatomy = packet.forecast_anatomy
    if anatomy is None:
        return dict.fromkeys(FORECAST_ROLES, 0.0)
    totals = {
        role: abs(float(anatomy.uptake_role_contributions.get(role, 0.0)))
        + abs(float(anatomy.conditional_role_contributions.get(role, 0.0)))
        for role in FORECAST_ROLES
    }
    denominator = sum(totals.values())
    return (
        {key: value / denominator for key, value in totals.items()}
        if denominator > 0.0
        else dict.fromkeys(FORECAST_ROLES, 0.0)
    )


def _dominant_role(packet: GraphRuntimePacket) -> ForecastRole:
    shares = _normalized_anatomy(packet)
    if not any(shares.values()):
        return "unknown"
    return cast(ForecastRole, max(shares, key=lambda role: shares[role]))


def _normalize_by_paper(
    paper_ids: Sequence[str], scores: Sequence[float]
) -> list[float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for paper_id, score in zip(paper_ids, scores, strict=True):
        totals[paper_id] = totals.get(paper_id, 0.0) + max(0.0, float(score))
        counts[paper_id] = counts.get(paper_id, 0) + 1
    return [
        (
            max(0.0, float(score)) / totals[paper_id]
            if totals[paper_id] > 0.0
            else 1.0 / counts[paper_id]
        )
        for paper_id, score in zip(paper_ids, scores, strict=True)
    ]


def _verify_gate1_report(path: Path, model_sha256: str) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or payload.get("claim_allowed") is not True:
        raise ValueError(f"Gate-1 report did not pass: {path}")
    binding = payload.get("claim_attribution_runtime_candidate") or {}
    if binding.get("model_sha256") != model_sha256:
        raise ValueError(f"Gate-1 report is not bound to candidate model: {path}")
    if binding.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"Gate-1 report has the wrong T0 feature schema: {path}")
    if binding.get("future_contexts_used_at_inference") is not False:
        raise ValueError(f"Gate-1 report permits future context at inference: {path}")
    if binding.get("development_only") is not True:
        raise ValueError(f"Gate-1 report is not development-only: {path}")
    if binding.get("training_split") != "development":
        raise ValueError(f"Gate-1 report has an invalid training split: {path}")
    if binding.get("sealed_holdout_labels_used") is not False:
        raise ValueError(f"Gate-1 report used sealed holdout labels: {path}")
    if binding.get("holdout_labels_used_for_model_selection") is not False:
        raise ValueError(
            f"Gate-1 report used holdout labels for model selection: {path}"
        )
    if binding.get("fold_local_target_fit") is not True:
        raise ValueError(f"Gate-1 report lacks fold-local target fitting: {path}")
    if binding.get("future_features_used") is not False:
        raise ValueError(f"Gate-1 report used future features: {path}")
    axis = str(binding.get("evaluation_axis", ""))
    if axis not in {"temporal", "domain"}:
        raise ValueError(f"Gate-1 report lacks a valid evaluation axis: {path}")
    return axis


def _bound_path(manifest_path: Path, reference: str) -> Path:
    candidate = (manifest_path.parent / reference).resolve()
    if not candidate.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError(
            "claim-attribution release reference escapes release directory"
        )
    return candidate


def _relative_reference(output_path: Path, target: Path) -> str:
    try:
        return target.resolve().relative_to(output_path.parent.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            "promotion inputs must be inside the release directory"
        ) from exc


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    if _sha256(path) != expected:
        raise ValueError(f"claim-attribution artifact hash mismatch: {path}")


__all__ = [
    "CLAIM_TYPES",
    "FEATURE_SCHEMA_VERSION",
    "FORECAST_ROLES",
    "PATHWAYS",
    "T0_FEATURE_NAMES",
    "ClaimAttributionLinearHead",
    "ClaimAttributionRelease",
    "deterministic_t0_attribution",
    "learned_t0_attribution",
    "load_claim_attribution_release",
    "pathway_hypothesis",
    "promote_claim_attribution_release",
    "t0_feature_row",
    "validate_claim_attribution_replay",
]
