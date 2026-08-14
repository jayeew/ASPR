"""Auditable development-only execution for the ASPR v6 forecast protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .evidence_registry import load_evidence_registry, registry_sha256
from .modeling_v6 import (
    MODEL_PARAMETER_GRID,
    assemble_development_frame,
    build_feature_sets,
    evaluate_development_oof,
    evaluate_temporal_folds,
    run_nested_development_oof,
    write_development_run,
)
from .prediction_registry_v6 import (
    load_prediction_registry,
    prediction_registry_sha256,
)
from .source_audit_v6 import sha256_file


DEVELOPMENT_PROTOCOL_VERSION = "aspr-v6-development-runner-1"


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _resolve(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else project_root / path


def audit_required_feature_sets(
    feature_sets: Mapping[str, Sequence[str]],
) -> Mapping[str, Any]:
    """Fail if a registered v6 comparison family is absent or malformed."""
    required_exact = {
        "controls_only",
        "innovation_plus_controls",
        "opportunity_only_plus_controls",
        "innovation_plus_opportunity_plus_controls",
    }
    missing_exact = sorted(required_exact - set(feature_sets))
    dimension_models = sorted(
        name
        for name in feature_sets
        if name.endswith("_plus_controls")
        and name
        not in {
            "innovation_plus_controls",
            "opportunity_only_plus_controls",
            "innovation_plus_opportunity_plus_controls",
        }
    )
    leave_out_models = sorted(
        name for name in feature_sets if name.startswith("leave_out_")
    )
    controls = tuple(feature_sets.get("controls_only", ()))
    innovation_full = tuple(
        feature_sets.get("innovation_plus_controls", ())
    )
    innovation_features = tuple(
        feature
        for feature in innovation_full
        if feature not in set(controls)
    )
    by_features = {
        tuple(values): name for name, values in feature_sets.items()
    }
    leave_out_aliases = {}
    for dimension_model in dimension_models:
        dimension_features = {
            feature
            for feature in feature_sets[dimension_model]
            if feature not in set(controls)
        }
        complement = tuple(
            (*controls,)
            + tuple(
                feature
                for feature in innovation_features
                if feature not in dimension_features
            )
        )
        dimension_name = dimension_model.removesuffix("_plus_controls")
        explicit_name = f"leave_out_{dimension_name}"
        represented_by = (
            explicit_name
            if explicit_name in feature_sets
            else by_features.get(complement)
        )
        if represented_by:
            leave_out_aliases[explicit_name] = represented_by
    duplicate_features = {
        name: sorted(
            {
                feature
                for feature in values
                if tuple(values).count(feature) > 1
            }
        )
        for name, values in feature_sets.items()
        if len(tuple(values)) != len(set(values))
    }
    checks = {
        "required_exact_present": not missing_exact,
        "dimension_models_present": len(dimension_models) >= 2,
        "leave_out_comparisons_represented": (
            len(leave_out_aliases) == len(dimension_models)
        ),
        "no_duplicate_features": not duplicate_features,
    }
    return {
        "overall_pass": all(checks.values()),
        "checks": checks,
        "missing_exact": missing_exact,
        "dimension_models": dimension_models,
        "leave_out_models": leave_out_models,
        "leave_out_aliases": leave_out_aliases,
        "duplicate_features": duplicate_features,
    }


def _gate(
    gate_id: str,
    description: str,
    observed: float,
    comparator: str,
    threshold: float,
    passed: bool,
    *,
    blocking: bool = True,
) -> Dict[str, Any]:
    return {
        "gate_id": gate_id,
        "description": description,
        "observed": float(observed) if np.isfinite(observed) else np.nan,
        "comparator": comparator,
        "threshold": float(threshold),
        "passed": int(bool(passed)),
        "blocking": int(bool(blocking)),
        "scope": "development_only_sealed_holdout_not_accessed",
    }


def evaluate_development_gates(
    metrics: pd.DataFrame,
    temporal_fold_metrics: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    horizon: int,
) -> pd.DataFrame:
    """Apply preregistered gates without changing them from observed OOF."""
    acceptance = config["acceptance_gates"]
    primary_model_id = str(
        config["outcome_protocol"]["primary_feature_model_id"]
    )
    selected = metrics[metrics["model_id"].eq(primary_model_id)]
    if len(selected) != 1:
        raise ValueError("primary feature model is missing or duplicated")
    row = selected.iloc[0]
    temporal = temporal_fold_metrics[
        temporal_fold_metrics["model_id"].eq(primary_model_id)
        & temporal_fold_metrics["latest_development_fold"].eq(1)
    ]
    if len(temporal) != 1:
        raise ValueError("latest development temporal fold is missing")
    latest = temporal.iloc[0]
    expected_domains = 12
    rows = [
        _gate(
            "G1_NESTED_OOF_RANK",
            "D5 nested expanding-year OOF Spearman",
            float(row["spearman_expected"]),
            ">=",
            float(acceptance["nested_oof_spearman_min"]),
            float(row["spearman_expected"])
            >= float(acceptance["nested_oof_spearman_min"]),
        ),
        _gate(
            "G2_POSITIVE_RANK_CI",
            "Nested OOF Spearman 95% bootstrap lower bound",
            float(row["spearman_ci_low"]),
            ">",
            0.0,
            float(row["spearman_ci_low"]) > 0.0,
        ),
        _gate(
            "G3_DOMAIN_MACRO",
            "Twelve-domain macro-average OOF Spearman",
            float(row["domain_macro_spearman"]),
            ">=",
            float(acceptance["domain_macro_spearman_min"]),
            float(row["domain_macro_spearman"])
            >= float(acceptance["domain_macro_spearman_min"]),
        ),
        _gate(
            "G4_DOMAIN_COVERAGE",
            "Number of reportable natural-science domains",
            float(row["n_reportable_domains"]),
            ">=",
            float(expected_domains),
            int(row["n_reportable_domains"]) >= expected_domains,
        ),
        _gate(
            "G5_GAIN_OVER_CONTROLS",
            "Incremental OOF Spearman over strong controls",
            float(row["gain_over_controls"]),
            ">=",
            float(acceptance["gain_over_strong_controls_min"]),
            float(row["gain_over_controls"])
            >= float(acceptance["gain_over_strong_controls_min"]),
        ),
        _gate(
            "G6_POSITIVE_GAIN_CI",
            "Paired-bootstrap gain lower bound",
            float(row["gain_over_controls_ci_low"]),
            ">",
            0.0,
            float(row["gain_over_controls_ci_low"]) > 0.0,
        ),
        _gate(
            "G7_LATEST_DEVELOPMENT_TIME",
            "Most recent development-period temporal-fold Spearman",
            float(latest["spearman_expected"]),
            ">=",
            float(acceptance["temporal_holdout_spearman_min"]),
            float(latest["spearman_expected"])
            >= float(acceptance["temporal_holdout_spearman_min"]),
        ),
        _gate(
            "G8_UPTAKE_ECE",
            "Uptake expected calibration error, ten bins",
            float(row["uptake_ece_10"]),
            "<=",
            float(acceptance["uptake_ece_10_max"]),
            float(row["uptake_ece_10"])
            <= float(acceptance["uptake_ece_10_max"]),
        ),
        _gate(
            "G9_UPTAKE_BRIER_SKILL",
            "Uptake Brier skill versus fold-local training prevalence",
            float(row["uptake_brier_skill_score"]),
            ">",
            float(acceptance["uptake_brier_skill_score_min"]),
            float(row["uptake_brier_skill_score"])
            > float(acceptance["uptake_brier_skill_score_min"]),
        ),
        _gate(
            "G10_REALIZED_INTERVAL_COVERAGE",
            "Marginal realized-score interval coverage at nominal 90%",
            float(row["realized_interval_coverage_90"]),
            ">=",
            float(acceptance["realized_interval_coverage_90_min"]),
            float(row["realized_interval_coverage_90"])
            >= float(acceptance["realized_interval_coverage_90_min"]),
        ),
        _gate(
            "G11_REALIZED_INTERVAL_WIDTH",
            "Mean marginal realized-score interval width",
            float(row["realized_interval_mean_width"]),
            "<=",
            float(acceptance["realized_interval_mean_width_max"]),
            float(row["realized_interval_mean_width"])
            <= float(acceptance["realized_interval_mean_width_max"]),
        ),
    ]
    output = pd.DataFrame(rows)
    output["horizon"] = int(horizon)
    output["primary_model_id"] = primary_model_id
    return output


def evaluate_directional_horizon_gates(
    metrics: pd.DataFrame,
    temporal_fold_metrics: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    horizon: int,
) -> pd.DataFrame:
    """Require D3/D8 to support the same positive direction and diagnostics."""
    acceptance = config["acceptance_gates"]
    primary_model_id = str(
        config["outcome_protocol"]["primary_feature_model_id"]
    )
    selected = metrics[metrics["model_id"].eq(primary_model_id)]
    if len(selected) != 1:
        raise ValueError("primary feature model is missing or duplicated")
    row = selected.iloc[0]
    temporal = temporal_fold_metrics[
        temporal_fold_metrics["model_id"].eq(primary_model_id)
        & temporal_fold_metrics["latest_development_fold"].eq(1)
    ]
    if len(temporal) != 1:
        raise ValueError("latest development temporal fold is missing")
    latest = temporal.iloc[0]
    rows = [
        _gate(
            "H1_POSITIVE_NESTED_OOF",
            f"D{int(horizon)} nested OOF Spearman direction",
            float(row["spearman_expected"]),
            ">",
            0.0,
            float(row["spearman_expected"]) > 0.0,
        ),
        _gate(
            "H2_POSITIVE_RANK_CI",
            f"D{int(horizon)} OOF Spearman lower bound",
            float(row["spearman_ci_low"]),
            ">",
            0.0,
            float(row["spearman_ci_low"]) > 0.0,
        ),
        _gate(
            "H3_POSITIVE_GAIN",
            f"D{int(horizon)} incremental Spearman over controls",
            float(row["gain_over_controls"]),
            ">",
            0.0,
            float(row["gain_over_controls"]) > 0.0,
        ),
        _gate(
            "H4_POSITIVE_GAIN_CI",
            f"D{int(horizon)} paired gain lower bound",
            float(row["gain_over_controls_ci_low"]),
            ">",
            0.0,
            float(row["gain_over_controls_ci_low"]) > 0.0,
        ),
        _gate(
            "H5_POSITIVE_DOMAIN_MACRO",
            f"D{int(horizon)} domain-macro Spearman direction",
            float(row["domain_macro_spearman"]),
            ">",
            0.0,
            float(row["domain_macro_spearman"]) > 0.0,
        ),
        _gate(
            "H6_LATEST_DEVELOPMENT_TIME",
            f"D{int(horizon)} latest development-fold direction",
            float(latest["spearman_expected"]),
            ">",
            0.0,
            float(latest["spearman_expected"]) > 0.0,
        ),
        _gate(
            "H7_UPTAKE_ECE",
            f"D{int(horizon)} uptake expected calibration error",
            float(row["uptake_ece_10"]),
            "<=",
            float(acceptance["uptake_ece_10_max"]),
            float(row["uptake_ece_10"])
            <= float(acceptance["uptake_ece_10_max"]),
        ),
        _gate(
            "H8_UPTAKE_BRIER_SKILL",
            f"D{int(horizon)} uptake Brier skill",
            float(row["uptake_brier_skill_score"]),
            ">",
            float(acceptance["uptake_brier_skill_score_min"]),
            float(row["uptake_brier_skill_score"])
            > float(acceptance["uptake_brier_skill_score_min"]),
        ),
        _gate(
            "H9_REALIZED_INTERVAL_COVERAGE",
            f"D{int(horizon)} realized-score interval coverage",
            float(row["realized_interval_coverage_90"]),
            ">=",
            float(acceptance["realized_interval_coverage_90_min"]),
            float(row["realized_interval_coverage_90"])
            >= float(acceptance["realized_interval_coverage_90_min"]),
        ),
        _gate(
            "H10_REALIZED_INTERVAL_WIDTH",
            f"D{int(horizon)} realized-score interval width",
            float(row["realized_interval_mean_width"]),
            "<=",
            float(acceptance["realized_interval_mean_width_max"]),
            float(row["realized_interval_mean_width"])
            <= float(acceptance["realized_interval_mean_width_max"]),
        ),
    ]
    output = pd.DataFrame(rows)
    output["horizon"] = int(horizon)
    output["primary_model_id"] = primary_model_id
    return output


def _lineage(
    *,
    project_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    dataset_dir: Path,
    feature_sets: Mapping[str, Sequence[str]],
    horizon: int,
    bootstrap_iterations: int,
    parameter_grid: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    innovation_path = _resolve(
        project_root, str(config["evidence_registry_path"])
    )
    prediction_path = _resolve(
        project_root, str(config["prediction_registry_path"])
    )
    selection_path = _resolve(
        project_root, str(config["evidence_selection_protocol_path"])
    )
    innovation = load_evidence_registry(innovation_path)
    prediction = load_prediction_registry(prediction_path)
    module_root = Path(__file__).resolve().parent
    code_files = (
        Path(__file__).resolve(),
        module_root / "modeling_v6.py",
        module_root / "targets_v6.py",
        module_root / "evidence_registry.py",
        module_root / "prediction_registry_v6.py",
    )
    manifests = {}
    for name in (
        "input_views_manifest.json",
        "field_events_manifest.json",
        "publication_features_manifest.json",
        "opportunity_features_manifest.json",
        "targets_cohort_manifest.json",
    ):
        path = dataset_dir / name
        payload = _load_json(path)
        manifests[name] = {
            "artifact_id": payload["artifact_id"],
            "manifest_sha256": sha256_file(path),
        }
    publication_manifest = _load_json(
        dataset_dir / "publication_features_manifest.json"
    )
    if (
        publication_manifest["inputs"]["innovation_registry_sha256"]
        != registry_sha256(innovation)
    ):
        raise ValueError("innovation features do not match current registry")
    opportunity_manifest = _load_json(
        dataset_dir / "opportunity_features_manifest.json"
    )
    if (
        opportunity_manifest["inputs"]["prediction_registry_sha256"]
        != prediction_registry_sha256(prediction)
    ):
        raise ValueError("opportunity features do not match current registry")
    return {
        "development_protocol_version": DEVELOPMENT_PROTOCOL_VERSION,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "innovation_registry_sha256": registry_sha256(innovation),
        "prediction_registry_sha256": prediction_registry_sha256(prediction),
        "evidence_selection_protocol_sha256": sha256_file(selection_path),
        "dataset_dir": str(dataset_dir),
        "dataset_manifests": manifests,
        "horizon": int(horizon),
        "bootstrap_iterations": int(bootstrap_iterations),
        "parameter_grid": [dict(item) for item in parameter_grid],
        "feature_sets": {
            name: list(values) for name, values in feature_sets.items()
        },
        "code_sha256": {
            path.name: sha256_file(path) for path in code_files
        },
        "sealed_holdout_accessed": False,
        "network_policy": "forbidden",
    }


def run_development_protocol(
    *,
    project_root: Path,
    config_path: Path,
    dataset_dir: Path,
    output_root: Path,
    horizon: int,
    bootstrap_iterations: int | None = None,
    parameter_grid: Sequence[Mapping[str, Any]] = MODEL_PARAMETER_GRID,
    model_scope: str = "full",
) -> Tuple[Mapping[str, Any], Path]:
    """Execute and persist one immutable, development-only nested OOF run."""
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    output_root = Path(output_root).resolve()
    config = _load_json(config_path)
    if config.get("network_policy") != "forbidden":
        raise ValueError("v6 development requires a forbidden network policy")
    horizon_spec = next(
        (
            item
            for item in config["horizons"]
            if int(item["tau"]) == int(horizon)
        ),
        None,
    )
    if horizon_spec is None:
        raise ValueError(f"unregistered horizon: {horizon}")
    innovation = load_evidence_registry(
        _resolve(project_root, str(config["evidence_registry_path"]))
    )
    prediction = load_prediction_registry(
        _resolve(project_root, str(config["prediction_registry_path"]))
    )
    all_feature_sets = build_feature_sets(innovation, prediction)
    feature_set_audit = audit_required_feature_sets(all_feature_sets)
    if not feature_set_audit["overall_pass"]:
        raise ValueError(
            f"required feature-set audit failed: {feature_set_audit}"
        )
    if model_scope == "full":
        feature_sets = all_feature_sets
    elif model_scope == "directional":
        primary_model_id = str(
            config["outcome_protocol"]["primary_feature_model_id"]
        )
        feature_sets = {
            name: all_feature_sets[name]
            for name in ("controls_only", primary_model_id)
        }
    else:
        raise ValueError("model_scope must be full or directional")
    iterations = int(
        bootstrap_iterations
        if bootstrap_iterations is not None
        else config["validation_protocol"]["bootstrap_iterations"]
    )
    lineage = dict(
        _lineage(
            project_root=project_root,
            config_path=config_path,
            config=config,
            dataset_dir=dataset_dir,
            feature_sets=feature_sets,
            horizon=int(horizon),
            bootstrap_iterations=iterations,
            parameter_grid=parameter_grid,
        )
    )
    lineage["feature_set_audit"] = feature_set_audit
    lineage["model_scope"] = model_scope
    run_spec_hash = _canonical_hash(lineage)
    lineage["run_spec_hash"] = run_spec_hash
    output_dir = output_root / (
        f"development_D{int(horizon)}_{run_spec_hash.removeprefix('sha256:')[:12]}"
    )
    existing_manifest = output_dir / "development_run_manifest.json"
    if existing_manifest.is_file():
        payload = _load_json(existing_manifest)
        if payload.get("lineage", {}).get("run_spec_hash") != run_spec_hash:
            raise ValueError("existing development run has different lineage")
        return payload, output_dir
    frame = assemble_development_frame(
        dataset_dir,
        horizon=int(horizon),
        development_end_year=int(horizon_spec["development_end_year"]),
    )
    lineage["development_frame"] = {
        "n_rows": len(frame),
        "publication_year_min": int(frame["publication_year"].min()),
        "publication_year_max": int(frame["publication_year"].max()),
        "n_observed_zero": int(frame["future_uptake"].eq(0).sum()),
        "n_conditional_diffusion": int(
            frame["conditional_diffusion_member"].eq(1).sum()
        ),
        "n_domains": int(frame["domain12"].nunique()),
    }
    predictions, ledger, folds = run_nested_development_oof(
        frame,
        feature_sets=feature_sets,
        horizon=int(horizon),
        n_outer=int(config["validation_protocol"]["outer_folds"]),
        n_inner=int(config["validation_protocol"]["inner_folds"]),
        parameter_grid=parameter_grid,
        seed=int(config["validation_protocol"]["seed"]),
    )
    metrics, domain_metrics = evaluate_development_oof(
        predictions,
        bootstrap_iterations=iterations,
        min_domain_rows=int(
            config["quality_protocol"]["min_reportable_domain_rows"]
        ),
        seed=int(config["validation_protocol"]["seed"]),
    )
    temporal_metrics = evaluate_temporal_folds(predictions)
    if int(horizon) == int(config["acceptance_gates"]["primary_horizon"]):
        gates = evaluate_development_gates(
            metrics,
            temporal_metrics,
            config=config,
            horizon=int(horizon),
        )
    else:
        gates = evaluate_directional_horizon_gates(
            metrics,
            temporal_metrics,
            config=config,
            horizon=int(horizon),
        )
    lineage["development_gate_pass"] = bool(
        gates.loc[gates["blocking"].eq(1), "passed"].eq(1).all()
    )
    manifest = write_development_run(
        output_dir,
        predictions=predictions,
        model_ledger=ledger,
        folds=folds,
        metrics=metrics,
        domain_metrics=domain_metrics,
        temporal_fold_metrics=temporal_metrics,
        acceptance_gates=gates,
        lineage=lineage,
    )
    return manifest, output_dir


__all__ = [
    "DEVELOPMENT_PROTOCOL_VERSION",
    "audit_required_feature_sets",
    "evaluate_development_gates",
    "evaluate_directional_horizon_gates",
    "run_development_protocol",
]
