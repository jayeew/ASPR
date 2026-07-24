"""Single-unlock sealed evaluation for the frozen ASPR v6 D5 candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import joblib
import numpy as np
import pandas as pd

from .evidence_registry import load_evidence_registry, registry_sha256
from .modeling_v6 import (
    MODEL_PARAMETER_GRID,
    assemble_development_frame,
    build_feature_sets,
    evaluate_development_oof,
    fit_calibrated_final_model,
    realized_diffusion_target,
    select_final_parameters,
)
from .prediction_registry_v6 import load_prediction_registry
from .release_v6 import _current_development_runs
from .source_audit_v6 import sha256_file


SEALED_PROTOCOL_VERSION = "aspr-v6-single-unlock-sealed-1"
SEALED_OUTCOME_COLUMNS = {
    "future_uptake",
    "future_field_reach",
    "future_subfield_reach",
    "future_topic_reach",
    "future_field_simpson",
    "future_topic_simpson",
    "conditional_diffusion_member",
    "realized_diffusion_target",
}


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


def _find_release_candidate(
    output_root: Path,
    *,
    config_sha256: str,
    registry_sha256_value: str,
) -> Tuple[Mapping[str, Any], Path]:
    matches = []
    current_release_code_hash = sha256_file(
        Path(__file__).resolve().parent / "release_v6.py"
    )
    for path in Path(output_root).glob(
        "release_candidate_*/release_candidate_manifest.json"
    ):
        manifest = _load_json(path)
        lineage = manifest.get("lineage", {})
        if (
            lineage.get("config_sha256") == config_sha256
            and lineage.get("innovation_registry_sha256")
            == registry_sha256_value
            and lineage.get("code_sha256")
            == current_release_code_hash
            and manifest.get("summary", {}).get(
                "release_candidate_ready_before_sealed"
            )
            is True
            and manifest.get("summary", {}).get(
                "sealed_holdout_accessed", True
            )
            is False
        ):
            matches.append((manifest, path.parent))
    if len(matches) != 1:
        raise ValueError(
            f"expected one current release candidate, found {len(matches)}"
        )
    return matches[0]


def assemble_locked_sealed_features(
    dataset_dir: Path,
    *,
    horizon: int,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Assemble the fixed sample and T0 views without reading outcome columns."""
    root = Path(dataset_dir)
    membership_columns = [
        "paper_id",
        "horizon",
        "publication_year",
        "domain12",
        "venue_family",
        "cohort_member",
    ]
    membership = pd.read_parquet(
        root / "cohort_membership.parquet",
        columns=membership_columns,
        filters=[
            ("horizon", "=", int(horizon)),
            ("publication_year", ">=", int(start_year)),
            ("publication_year", "<=", int(end_year)),
        ],
    )
    frame = membership[membership["cohort_member"].eq(1)].drop(
        columns=["cohort_member"]
    )
    for name in (
        "innovation_features.parquet",
        "control_features.parquet",
        "opportunity_features.parquet",
    ):
        view = pd.read_parquet(root / name)
        duplicate = set(view.columns) & set(frame.columns) - {"paper_id"}
        view = view.drop(columns=sorted(duplicate), errors="ignore")
        frame = frame.merge(
            view,
            on="paper_id",
            how="left",
            validate="one_to_one",
        )
    if set(frame.columns) & SEALED_OUTCOME_COLUMNS:
        raise ValueError("sealed feature lock contains outcome columns")
    if frame.duplicated("paper_id").any():
        raise ValueError("sealed sample lock contains duplicate papers")
    if not frame["publication_year"].between(start_year, end_year).all():
        raise ValueError("sealed sample lock contains an out-of-window paper")
    return frame.sort_values(["publication_year", "paper_id"]).reset_index(
        drop=True
    )


def _find_sealed_freeze(
    output_root: Path,
    *,
    config_sha256: str,
    registry_sha256_value: str,
) -> Tuple[Mapping[str, Any], Path]:
    matches = []
    for path in Path(output_root).glob(
        "sealed_D5_*/sealed_model_freeze_manifest.json"
    ):
        manifest = _load_json(path)
        lineage = manifest.get("lineage", {})
        if (
            lineage.get("config_sha256") == config_sha256
            and lineage.get("innovation_registry_sha256")
            == registry_sha256_value
            and manifest.get("sealed_holdout_labels_accessed") is False
        ):
            matches.append((manifest, path.parent))
    if len(matches) != 1:
        raise ValueError(
            f"expected one current sealed freeze, found {len(matches)}"
        )
    return matches[0]


def freeze_sealed_candidate(
    *,
    project_root: Path,
    config_path: Path,
    dataset_dir: Path,
    output_root: Path,
) -> Tuple[Mapping[str, Any], Path]:
    """Fit final development models and lock prelabel sealed predictions."""
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    output_root = Path(output_root).resolve()
    config = _load_json(config_path)
    registry = load_evidence_registry(
        project_root / str(config["evidence_registry_path"])
    )
    prediction_registry = load_prediction_registry(
        project_root / str(config["prediction_registry_path"])
    )
    config_hash = sha256_file(config_path)
    registry_hash = registry_sha256(registry)
    release, release_dir = _find_release_candidate(
        output_root,
        config_sha256=config_hash,
        registry_sha256_value=registry_hash,
    )
    development = _current_development_runs(
        output_root,
        config_sha256=config_hash,
        registry_sha256_value=registry_hash,
    )
    d5_manifest, d5_dir = development[5]
    model_ids = tuple(config["release_protocol"]["sealed_models"])
    feature_sets = build_feature_sets(registry, prediction_registry)
    missing_models = sorted(set(model_ids) - set(feature_sets))
    if missing_models:
        raise ValueError(f"sealed model ids are unavailable: {missing_models}")
    module_root = Path(__file__).resolve().parent
    pre_spec = {
        "sealed_protocol_version": SEALED_PROTOCOL_VERSION,
        "config_sha256": config_hash,
        "innovation_registry_sha256": registry_hash,
        "release_candidate_artifact_id": release["artifact_id"],
        "development_D5_artifact_id": d5_manifest["artifact_id"],
        "model_ids": list(model_ids),
        "code_sha256": {
            path.name: sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                module_root / "modeling_v6.py",
                module_root / "targets_v6.py",
            )
        },
        "sealed_holdout_labels_accessed": False,
    }
    run_hash = _canonical_hash(pre_spec)
    output_dir = output_root / (
        f"sealed_D5_{run_hash.removeprefix('sha256:')[:12]}"
    )
    manifest_path = output_dir / "sealed_model_freeze_manifest.json"
    if manifest_path.is_file():
        return _load_json(manifest_path), output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    if (output_dir / "sealed_unlock_receipt.json").exists():
        raise RuntimeError("sealed unlock already exists before model freeze")
    horizon_spec = next(
        item for item in config["horizons"] if int(item["tau"]) == 5
    )
    development_frame = assemble_development_frame(
        dataset_dir,
        horizon=5,
        development_end_year=int(horizon_spec["development_end_year"]),
    )
    sealed_features = assemble_locked_sealed_features(
        dataset_dir,
        horizon=5,
        start_year=int(horizon_spec["sealed_test_start_year"]),
        end_year=int(horizon_spec["sealed_test_end_year"]),
    )
    predictions = pd.read_parquet(
        d5_dir / "development_oof_predictions.parquet"
    )
    ledger = pd.read_parquet(d5_dir / "development_model_ledger.parquet")
    locked_rows: List[pd.DataFrame] = []
    model_outputs = {}
    selected_parameters = {}
    training_prevalence = float(
        development_frame["future_uptake"].mean()
    )
    for model_index, model_id in enumerate(model_ids):
        parameters = select_final_parameters(
            ledger,
            model_id=model_id,
            parameter_grid=MODEL_PARAMETER_GRID,
        )
        selected_parameters[model_id] = str(parameters["parameter_id"])
        oof = predictions[predictions["model_id"].eq(model_id)].copy()
        bundle = fit_calibrated_final_model(
            development_frame,
            oof,
            feature_names=feature_sets[model_id],
            parameters=parameters,
            horizon=5,
            seed=int(config["validation_protocol"]["seed"])
            + 50_000
            + model_index,
        )
        model_path = output_dir / f"{model_id}.joblib"
        joblib.dump(bundle, model_path, compress=3)
        model_outputs[model_id] = {
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size,
            "sha256": sha256_file(model_path),
        }
        scored = bundle.predict(sealed_features)
        scored.insert(
            0,
            "paper_id",
            sealed_features["paper_id"].astype(str).to_numpy(),
        )
        scored["publication_year"] = sealed_features[
            "publication_year"
        ].to_numpy(dtype=int)
        scored["domain12"] = sealed_features["domain12"].astype(str).to_numpy()
        scored["horizon"] = 5
        scored["model_id"] = model_id
        scored["outer_fold_id"] = 0
        scored["training_uptake_prevalence"] = training_prevalence
        scored["selected_parameter_id"] = str(parameters["parameter_id"])
        scored["scope"] = "sealed_predictions_locked_prelabel"
        locked_rows.append(scored)
    sample_path = output_dir / "sealed_sample_lock.parquet"
    prediction_path = output_dir / "sealed_predictions_locked.parquet"
    sealed_features[
        ["paper_id", "publication_year", "domain12", "horizon"]
    ].to_parquet(sample_path, index=False)
    locked = pd.concat(locked_rows, ignore_index=True)
    locked.to_parquet(prediction_path, index=False)
    if set(locked.columns) & SEALED_OUTCOME_COLUMNS:
        raise ValueError("locked sealed predictions contain outcome columns")
    if locked.duplicated(["paper_id", "model_id"]).any():
        raise ValueError("locked sealed predictions contain duplicate keys")
    lineage = {
        **pre_spec,
        "release_candidate_dir": str(release_dir),
        "development_D5_dir": str(d5_dir),
        "dataset_manifests": {
            name: _load_json(dataset_dir / name)["artifact_id"]
            for name in (
                "input_views_manifest.json",
                "publication_features_manifest.json",
                "opportunity_features_manifest.json",
                "targets_cohort_manifest.json",
            )
        },
    }
    manifest = {
        "artifact_kind": "aspr_v6_sealed_model_freeze",
        "lineage": lineage,
        "sealed_holdout_labels_accessed": False,
        "sealed_unlock_count_allowed": 1,
        "sealed_unlock_count_used": 0,
        "sample": {
            "n_papers": int(len(sealed_features)),
            "publication_year_min": int(
                sealed_features["publication_year"].min()
            ),
            "publication_year_max": int(
                sealed_features["publication_year"].max()
            ),
            "n_domains": int(sealed_features["domain12"].nunique()),
            "columns_read_from_membership": [
                "paper_id",
                "horizon",
                "publication_year",
                "domain12",
                "venue_family",
                "cohort_member",
            ],
            "outcome_columns_present": [],
        },
        "selected_parameters": selected_parameters,
        "models": model_outputs,
        "outputs": {
            "sample_lock": {
                "path": str(sample_path),
                "size_bytes": sample_path.stat().st_size,
                "sha256": sha256_file(sample_path),
            },
            "predictions_lock": {
                "path": str(prediction_path),
                "size_bytes": prediction_path.stat().st_size,
                "sha256": sha256_file(prediction_path),
            },
        },
    }
    manifest["artifact_id"] = _canonical_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest, output_dir


def _sealed_gates(
    metrics: pd.DataFrame,
    domain_metrics: pd.DataFrame,
    *,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    acceptance = config["acceptance_gates"]
    model_id = str(config["release_protocol"]["primary_model_id"])
    selected = metrics[metrics["model_id"].eq(model_id)]
    if len(selected) != 1:
        raise ValueError("sealed primary model metric is missing")
    row = selected.iloc[0]
    domains = domain_metrics[
        domain_metrics["model_id"].eq(model_id)
        & domain_metrics["spearman_expected"].notna()
    ]
    all_domain_macro = float(domains["spearman_expected"].mean())
    gate_rows = [
        (
            "S1_SEALED_TEMPORAL_RANK",
            float(row["spearman_expected"]),
            ">=",
            float(acceptance["temporal_holdout_spearman_min"]),
        ),
        (
            "S2_SEALED_POSITIVE_RANK_CI",
            float(row["spearman_ci_low"]),
            ">",
            0.0,
        ),
        (
            "S3_SEALED_12_DOMAIN_MACRO",
            all_domain_macro,
            ">=",
            float(acceptance["domain_macro_spearman_min"]),
        ),
        (
            "S4_SEALED_POSITIVE_GAIN",
            float(row["gain_over_controls"]),
            ">",
            0.0,
        ),
        (
            "S5_SEALED_POSITIVE_GAIN_CI",
            float(row["gain_over_controls_ci_low"]),
            ">",
            0.0,
        ),
        (
            "S6_SEALED_UPTAKE_ECE",
            float(row["uptake_ece_10"]),
            "<=",
            float(acceptance["uptake_ece_10_max"]),
        ),
        (
            "S7_SEALED_BRIER_SKILL",
            float(row["uptake_brier_skill_score"]),
            ">",
            float(acceptance["uptake_brier_skill_score_min"]),
        ),
        (
            "S8_SEALED_INTERVAL_COVERAGE",
            float(row["realized_interval_coverage_90"]),
            ">=",
            float(acceptance["realized_interval_coverage_90_min"]),
        ),
        (
            "S9_SEALED_INTERVAL_WIDTH",
            float(row["realized_interval_mean_width"]),
            "<=",
            float(acceptance["realized_interval_mean_width_max"]),
        ),
        (
            "S10_SEALED_DOMAIN_PRESENCE",
            float(len(domains)),
            ">=",
            12.0,
        ),
    ]
    rows = []
    for gate_id, observed, comparator, threshold in gate_rows:
        passed = (
            observed >= threshold
            if comparator == ">="
            else (
                observed <= threshold
                if comparator == "<="
                else observed > threshold
            )
        )
        rows.append(
            {
                "gate_id": gate_id,
                "observed": observed,
                "comparator": comparator,
                "threshold": threshold,
                "passed": int(bool(passed)),
                "scope": "single_unlock_sealed_2014_2017",
            }
        )
    return pd.DataFrame(rows)


def run_single_unlock_sealed_evaluation(
    *,
    project_root: Path,
    config_path: Path,
    dataset_dir: Path,
    output_root: Path,
) -> Tuple[Mapping[str, Any], Path]:
    """Consume the single unlock and evaluate without permitting a rerun."""
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    output_root = Path(output_root).resolve()
    config = _load_json(config_path)
    registry = load_evidence_registry(
        project_root / str(config["evidence_registry_path"])
    )
    config_hash = sha256_file(config_path)
    registry_hash = registry_sha256(registry)
    freeze, output_dir = _find_sealed_freeze(
        output_root,
        config_sha256=config_hash,
        registry_sha256_value=registry_hash,
    )
    final_manifest_path = output_dir / "sealed_evaluation_manifest.json"
    if final_manifest_path.is_file():
        return _load_json(final_manifest_path), output_dir
    receipt_path = output_dir / "sealed_unlock_receipt.json"
    if receipt_path.exists():
        raise RuntimeError(
            "the single sealed unlock was already consumed; rerun forbidden"
        )
    receipt = {
        "artifact_kind": "aspr_v6_sealed_unlock_receipt",
        "freeze_artifact_id": freeze["artifact_id"],
        "unlock_number": 1,
        "maximum_unlocks": 1,
        "purpose": "single final evaluation only; no tuning or re-selection",
        "config_sha256": config_hash,
        "innovation_registry_sha256": registry_hash,
    }
    receipt["receipt_id"] = _canonical_hash(receipt)
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    sample = pd.read_parquet(output_dir / "sealed_sample_lock.parquet")
    locked = pd.read_parquet(
        output_dir / "sealed_predictions_locked.parquet"
    )
    target_columns = [
        "paper_id",
        "horizon",
        "publication_year",
        "future_uptake",
        "future_field_reach",
        "future_subfield_reach",
        "future_topic_reach",
        "future_field_simpson",
        "future_topic_simpson",
    ]
    targets = pd.read_parquet(
        dataset_dir / "targets_zero_inclusive.parquet",
        columns=target_columns,
        filters=[
            ("horizon", "=", 5),
            ("publication_year", ">=", 2014),
            ("publication_year", "<=", 2017),
        ],
    )
    membership = pd.read_parquet(
        dataset_dir / "cohort_membership.parquet",
        columns=[
            "paper_id",
            "horizon",
            "publication_year",
            "conditional_diffusion_member",
            "cap_hit",
        ],
        filters=[
            ("horizon", "=", 5),
            ("publication_year", ">=", 2014),
            ("publication_year", "<=", 2017),
        ],
    )
    evaluation_targets = (
        sample.merge(
            targets,
            on=["paper_id", "horizon", "publication_year"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            membership,
            on=["paper_id", "horizon", "publication_year"],
            how="left",
            validate="one_to_one",
        )
    )
    if evaluation_targets["future_uptake"].isna().any():
        raise RuntimeError(
            "sealed labels are incomplete after the single unlock"
        )
    model_rows = []
    for model_id, model_info in freeze["models"].items():
        bundle = joblib.load(model_info["path"])
        conditional, realized = realized_diffusion_target(
            evaluation_targets,
            bundle.fitted_model.target_transformer,
        )
        scored_targets = evaluation_targets[
            [
                "paper_id",
                "future_uptake",
                "conditional_diffusion_member",
                "cap_hit",
            ]
        ].copy()
        scored_targets["conditional_diffusion_target"] = conditional
        scored_targets["realized_diffusion_target"] = realized
        predicted = locked[locked["model_id"].eq(model_id)].merge(
            scored_targets,
            on="paper_id",
            how="inner",
            validate="one_to_one",
        )
        predicted["scope"] = "single_unlock_sealed_2014_2017"
        model_rows.append(predicted)
    predictions = pd.concat(model_rows, ignore_index=True)
    metrics, domains = evaluate_development_oof(
        predictions,
        bootstrap_iterations=int(
            config["validation_protocol"]["bootstrap_iterations"]
        ),
        min_domain_rows=int(
            config["quality_protocol"]["min_reportable_domain_rows"]
        ),
        seed=int(config["validation_protocol"]["seed"]) + 90_000,
    )
    gates = _sealed_gates(metrics, domains, config=config)
    predictions_path = output_dir / "sealed_evaluation_predictions.parquet"
    metrics_path = output_dir / "sealed_evaluation_metrics.csv"
    domains_path = output_dir / "sealed_evaluation_domain_metrics.csv"
    gates_path = output_dir / "sealed_evaluation_gates.csv"
    predictions.to_parquet(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    domains.to_csv(domains_path, index=False)
    gates.to_csv(gates_path, index=False)
    primary = metrics[
        metrics["model_id"].eq(
            config["release_protocol"]["primary_model_id"]
        )
    ].iloc[0]
    summary = {
        "final_release_pass": bool(gates["passed"].eq(1).all()),
        "n_sealed_papers": int(len(sample)),
        "n_observed_zero": int(
            predictions.loc[
                predictions["model_id"].eq(
                    config["release_protocol"]["primary_model_id"]
                ),
                "future_uptake",
            ].eq(0).sum()
        ),
        "primary_spearman": float(primary["spearman_expected"]),
        "primary_spearman_ci_low": float(primary["spearman_ci_low"]),
        "gain_over_controls": float(primary["gain_over_controls"]),
        "gain_over_controls_ci_low": float(
            primary["gain_over_controls_ci_low"]
        ),
        "domain_macro_reportable": float(
            primary["domain_macro_spearman"]
        ),
        "uptake_ece_10": float(primary["uptake_ece_10"]),
        "realized_interval_coverage_90": float(
            primary["realized_interval_coverage_90"]
        ),
        "sealed_unlocks_used": 1,
        "sealed_reunlock_forbidden": True,
    }
    outputs = {
        "unlock_receipt": receipt_path,
        "predictions": predictions_path,
        "metrics": metrics_path,
        "domains": domains_path,
        "gates": gates_path,
    }
    manifest = {
        "artifact_kind": "aspr_v6_single_unlock_sealed_evaluation",
        "sealed_protocol_version": SEALED_PROTOCOL_VERSION,
        "freeze_artifact_id": freeze["artifact_id"],
        "unlock_receipt_id": receipt["receipt_id"],
        "summary": summary,
        "outputs": {
            name: {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in outputs.items()
        },
    }
    manifest["artifact_id"] = _canonical_hash(manifest)
    final_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest, output_dir


__all__ = [
    "SEALED_PROTOCOL_VERSION",
    "assemble_locked_sealed_features",
    "freeze_sealed_candidate",
    "run_single_unlock_sealed_evaluation",
]
