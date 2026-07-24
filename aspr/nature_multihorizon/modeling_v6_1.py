"""Fixed-medium six-fold all-period OOF protocol for ASPR v6.1."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .candidate_registry_v6_1 import (
    CandidateRegistryV61,
    candidate_registry_sha256,
    load_candidate_registry_v6_1,
    verify_search_log,
)
from .materialize_v6_1 import B0_INNOVATION_FEATURES
from .modeling_v6 import (
    _fit_calibrators,
    _fit_two_part,
    _inner_oof_for_parameters,
    _realized_diffusion,
    safe_spearman,
)
from .source_audit_v6 import sha256_file


MODEL_PROTOCOL_VERSION_V6_1 = "aspr-v6.1-fixed-medium-six-fold-oof-1"
FROZEN_REGISTRY_STAGE = "posthoc_versioned_extension_frozen_before_oof"


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(project_root) / path


def load_simple_config(path: Path) -> Dict[str, Any]:
    """Load and minimally validate the v6.1 simple protocol."""
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "6.1.0":
        raise ValueError("v6.1 simple config has an unexpected schema version")
    if payload.get("model", {}).get("parameter_id") != "medium":
        raise ValueError("v6.1 OOF must use the fixed medium model")
    expected_model = {
        "parameter_id": "medium",
        "max_leaf_nodes": 31,
        "max_depth": 4,
        "min_samples_leaf": 50,
        "learning_rate": 0.05,
        "max_iter": 200,
        "l2_regularization": 10.0,
        "inner_temporal_folds": 4,
        "seed": 20260724,
    }
    if payload.get("model") != expected_model:
        raise ValueError("v6.1 medium model parameters differ from protocol")
    if int(payload.get("main_horizon", -1)) != 5 or tuple(
        int(item) for item in payload.get("supplementary_horizons", [])
    ) != (3, 8):
        raise ValueError("v6.1 must report D5 with D3/D8 supplementary")
    if payload.get("network_policy_for_experiment") != "forbidden":
        raise ValueError("v6.1 experiment cannot use network data")
    if payload.get("raw_data_policy") != "local_frozen_only":
        raise ValueError("v6.1 must use only locally frozen raw data")
    if payload.get("evaluation", {}).get(
        "conditional_spearman_reported"
    ) is not False:
        raise ValueError("v6.1 does not report conditional Spearman")
    folds = payload.get("temporal_folds") or []
    if len(folds) != 6:
        raise ValueError("v6.1 OOF requires exactly six temporal folds")
    expected = (
        (1985, 1986, 1999),
        (1999, 2000, 2004),
        (2004, 2005, 2009),
        (2009, 2010, 2012),
        (2012, 2013, 2013),
        (2013, 2014, 2017),
    )
    observed = tuple(
        (
            int(item["train_year_max"]),
            int(item["test_year_min"]),
            int(item["test_year_max"]),
        )
        for item in folds
    )
    if observed != expected:
        raise ValueError("v6.1 temporal folds differ from the frozen protocol")
    control_registry_value = payload.get("paths", {}).get(
        "control_registry"
    )
    if not control_registry_value:
        raise ValueError("v6.1 config must register control-feature evidence")
    control_registry_path = Path(control_registry_value)
    if not control_registry_path.is_absolute():
        control_registry_path = (
            config_path.parents[2] / control_registry_path
        )
    control_registry = json.loads(
        control_registry_path.read_text(encoding="utf-8")
    )
    if control_registry.get("schema_version") != "6.1.0":
        raise ValueError("control-feature registry schema is not v6.1")
    registered_features = set(
        (control_registry.get("features") or {}).keys()
    )
    expected_controls = set(payload.get("k1_controls") or ()) | set(
        payload.get("k2_additional_controls") or ()
    )
    if registered_features != expected_controls:
        raise ValueError(
            "control-feature registry differs from K1/K2 config: "
            f"missing={sorted(expected_controls - registered_features)}, "
            f"extra={sorted(registered_features - expected_controls)}"
        )
    source_ids = set((control_registry.get("sources") or {}).keys())
    for feature_name, definition in (
        control_registry.get("features") or {}
    ).items():
        missing_sources = set(definition.get("source_ids") or ()) - source_ids
        if missing_sources:
            raise ValueError(
                f"{feature_name} has unknown control sources: "
                f"{sorted(missing_sources)}"
            )
    return payload


def _find_screening_manifest(
    analysis_root: Path,
    artifact_id: str,
) -> Tuple[Path, Mapping[str, Any]]:
    matches = []
    for path in sorted(Path(analysis_root).glob("screening_*/screening_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("artifact_id") == artifact_id:
            matches.append((path, payload))
    if len(matches) != 1:
        raise ValueError(
            "frozen registry must resolve to exactly one screening manifest; "
            f"found {len(matches)}"
        )
    return matches[0]


def freeze_registry_before_oof(
    project_root: Path,
    config_path: Path,
) -> Mapping[str, Any]:
    """Verify and record the outcome-blind registry freeze before labels load."""
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    config = load_simple_config(config_path)
    registry_path = _resolve(
        project_root, config["paths"]["candidate_registry"]
    ).resolve()
    registry = load_candidate_registry_v6_1(registry_path)
    if registry.registry_stage != FROZEN_REGISTRY_STAGE:
        raise ValueError("candidate registry is not frozen; OOF is forbidden")
    verify_search_log(registry, project_root)
    primary = [
        item
        for item in registry.candidates.values()
        if item.final_role == "primary"
    ]
    screening_ids = {
        item.empirical_screen.screening_artifact_id for item in primary
    }
    if None in screening_ids or len(screening_ids) != 1:
        raise ValueError("primary metrics do not share one screening artifact")
    analysis_root = _resolve(
        project_root, config["paths"]["v6_1_analysis"]
    ).resolve()
    analysis_root.mkdir(parents=True, exist_ok=True)
    screening_path, screening = _find_screening_manifest(
        analysis_root, str(next(iter(screening_ids)))
    )
    for output_name, output_record in screening["outputs"].items():
        output_path = Path(output_record["path"]).resolve()
        if not output_path.is_file():
            raise ValueError(
                f"screening output is missing: {output_name}={output_path}"
            )
        if sha256_file(output_path) != output_record["sha256"]:
            raise ValueError(
                f"screening output changed after screening: {output_name}"
            )
    lineage_files = {
        "candidate_catalog_sha256": _resolve(
            project_root, config["paths"]["candidate_catalog"]
        ).resolve(),
        "search_log_sha256": _resolve(
            project_root, registry.search_log_path
        ).resolve(),
        "screening_implementation_sha256": project_root
        / "aspr"
        / "nature_multihorizon"
        / "screening_v6_1.py",
        "feature_formula_implementation_sha256": project_root
        / "aspr"
        / "nature_multihorizon"
        / "features_v6_1.py",
        "legacy_formula_implementation_sha256": project_root
        / "aspr"
        / "nature_multihorizon"
        / "features_v6.py",
        "materialization_implementation_sha256": project_root
        / "aspr"
        / "nature_multihorizon"
        / "materialize_v6_1.py",
    }
    for lineage_key, source_path in lineage_files.items():
        actual_hash = (
            candidate_registry_sha256(
                load_candidate_registry_v6_1(source_path)
            )
            if lineage_key == "candidate_catalog_sha256"
            else sha256_file(source_path)
        )
        if actual_hash != screening["lineage"][lineage_key]:
            raise ValueError(
                f"screening lineage source changed: {lineage_key}"
            )
    dataset_root = _resolve(
        project_root, config["paths"]["v6_1_dataset"]
    ).resolve()
    candidate_path = dataset_root / "innovation_candidate_features.parquet"
    expected_candidate_hash = screening["lineage"][
        "candidate_features_sha256"
    ]
    if sha256_file(candidate_path) != expected_candidate_hash:
        raise ValueError("candidate feature artifact changed after screening")
    historical_references_path = (
        dataset_root / "historical_paper_references.parquet"
    )
    expected_history_hash = screening["lineage"][
        "historical_paper_references_sha256"
    ]
    if (
        not historical_references_path.is_file()
        or sha256_file(historical_references_path)
        != expected_history_hash
    ):
        raise ValueError(
            "reference-overlap history changed after screening"
        )
    identity = {
        "artifact_kind": "aspr_v6_1_registry_freeze",
        "protocol_version": MODEL_PROTOCOL_VERSION_V6_1,
        "labels_read_before_freeze": False,
        "registry_path": str(registry_path),
        "registry_file_sha256": sha256_file(registry_path),
        "registry_canonical_sha256": candidate_registry_sha256(registry),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "control_registry_sha256": sha256_file(
            _resolve(
                project_root, config["paths"]["control_registry"]
            ).resolve()
        ),
        "screening_manifest_path": str(screening_path),
        "screening_artifact_id": screening["artifact_id"],
        "candidate_features_sha256": expected_candidate_hash,
        "historical_paper_references_sha256": expected_history_hash,
    }
    payload = {**identity, "frozen_at_utc": _utc_now()}
    payload["artifact_id"] = _canonical_hash(identity)
    freeze_path = analysis_root / "registry_freeze_manifest.json"
    if freeze_path.is_file():
        existing = json.loads(freeze_path.read_text(encoding="utf-8"))
        invariant_keys = (
            "registry_file_sha256",
            "registry_canonical_sha256",
            "config_sha256",
            "control_registry_sha256",
            "screening_artifact_id",
            "candidate_features_sha256",
            "historical_paper_references_sha256",
        )
        if any(existing.get(key) != payload.get(key) for key in invariant_keys):
            raise ValueError("existing registry freeze differs from current inputs")
        return existing
    freeze_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return payload


def _merge_publication_view(
    frame: pd.DataFrame,
    view: pd.DataFrame,
) -> pd.DataFrame:
    duplicate = set(view.columns) & set(frame.columns) - {"paper_id"}
    trimmed = view.drop(columns=sorted(duplicate), errors="ignore")
    return frame.merge(
        trimmed,
        on="paper_id",
        how="left",
        validate="one_to_one",
    )


def _validate_publication_time_view(
    view: pd.DataFrame,
    *,
    name: str,
) -> None:
    if {"source_max_year", "publication_year"}.issubset(view.columns):
        source_year = pd.to_numeric(view["source_max_year"], errors="coerce")
        publication_year = pd.to_numeric(
            view["publication_year"], errors="coerce"
        )
        leaked = source_year.notna() & source_year.ge(publication_year)
        if leaked.any():
            raise ValueError(f"{name} contains publication-time leakage")


def assemble_all_period_frame(
    dataset_dir: Path,
    *,
    horizon: int,
) -> pd.DataFrame:
    """Assemble one common 1980-2017 cohort after registry freeze."""
    root = Path(dataset_dir)
    targets = pd.read_parquet(
        root / "targets_zero_inclusive.parquet",
        filters=[("horizon", "=", int(horizon))],
    )
    membership = pd.read_parquet(
        root / "cohort_membership.parquet",
        filters=[("horizon", "=", int(horizon))],
    )
    membership = membership[membership["cohort_member"].eq(1)].copy()
    membership_columns = [
        "paper_id",
        "publication_year",
        "domain12",
        "venue_family",
        "conditional_diffusion_member",
        "reference_evidence_eligible",
        "cap_hit",
    ]
    target_columns = [
        "paper_id",
        "future_uptake",
        "future_field_reach",
        "future_subfield_reach",
        "future_topic_reach",
        "future_field_simpson",
        "future_topic_simpson",
    ]
    frame = membership[membership_columns].merge(
        targets[target_columns],
        on="paper_id",
        how="inner",
        validate="one_to_one",
    )
    paths = (
        "innovation_candidate_features.parquet",
        "control_features_v6_1.parquet",
        "opportunity_features.parquet",
    )
    for name in paths:
        view = pd.read_parquet(root / name)
        _validate_publication_time_view(view, name=name)
        frame = _merge_publication_view(frame, view)
    frame["horizon"] = int(horizon)
    if frame["future_uptake"].isna().any():
        raise ValueError("all-period uptake label is incomplete")
    years = set(frame["publication_year"].astype(int).unique())
    if min(years) != 1980 or max(years) != 2017:
        raise ValueError("all-period cohort must span 1980-2017")
    return frame.sort_values(
        ["publication_year", "paper_id"], kind="stable"
    ).reset_index(drop=True)


def _unique(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def build_v6_1_feature_sets(
    registry: CandidateRegistryV61,
    config: Mapping[str, Any],
) -> Dict[str, Tuple[str, ...]]:
    """Build all model sets from the frozen registry instead of hardcoding it."""
    if registry.registry_stage != FROZEN_REGISTRY_STAGE:
        raise ValueError("feature sets require a frozen candidate registry")
    k0 = tuple(config["k0_controls"])
    k1 = tuple(config["k1_controls"])
    k2 = _unique((*k1, *config["k2_additional_controls"]))
    b0 = tuple(f"b0_{name}" for name in B0_INNOVATION_FEATURES)
    final = registry.primary_feature_names
    return {
        "k0_controls": k0,
        "k1_controls": k1,
        "b0_v6_primary_plus_k0": _unique((*k0, *b0)),
        "provisional_core8_plus_k1": _unique(
            (*k1, *registry.provisional_core8_names)
        ),
        "final_innovation_plus_k1": _unique((*k1, *final)),
        "k2_controls": k2,
        "final_innovation_plus_k2": _unique((*k2, *final)),
    }


def explicit_temporal_splits(
    frame: pd.DataFrame,
    fold_config: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    """Resolve the six registered expanding-time folds to row positions."""
    years = pd.to_numeric(frame["publication_year"], errors="coerce")
    if years.isna().any():
        raise ValueError("publication_year is required for temporal OOF")
    rows = []
    seen_test: set[int] = set()
    for item in fold_config:
        train = np.flatnonzero(years.le(int(item["train_year_max"])))
        test = np.flatnonzero(
            years.between(
                int(item["test_year_min"]),
                int(item["test_year_max"]),
                inclusive="both",
            )
        )
        if not len(train) or not len(test):
            raise ValueError(f"empty registered fold {item['fold_id']}")
        if int(years.iloc[train].max()) >= int(years.iloc[test].min()):
            raise ValueError("temporal leakage in registered folds")
        overlap = seen_test & set(test.tolist())
        if overlap:
            raise ValueError("registered tests overlap")
        seen_test.update(test.tolist())
        rows.append(
            {
                **dict(item),
                "train_index": train,
                "test_index": test,
                "n_train": len(train),
                "n_test": len(test),
            }
        )
    expected = set(np.flatnonzero(years.between(1986, 2017)).tolist())
    if seen_test != expected:
        raise ValueError("registered folds do not cover every 1986-2017 paper")
    return tuple(rows)


def _run_fold_model(
    training: pd.DataFrame,
    testing: pd.DataFrame,
    *,
    model_id: str,
    feature_names: Sequence[str],
    categorical_names: Sequence[str],
    parameters: Mapping[str, Any],
    inner_folds: int,
    horizon: int,
    fold_id: int,
    seed: int,
) -> pd.DataFrame:
    inner = _inner_oof_for_parameters(
        training,
        feature_names=feature_names,
        categorical_names=categorical_names,
        parameters=parameters,
        n_inner=int(inner_folds),
        seed=int(seed + fold_id * 100),
    )
    (
        uptake_calibrator,
        conditional_calibrator,
        conditional_residual_quantile,
        realized_residual_quantile,
    ) = _fit_calibrators(inner)
    model = _fit_two_part(
        training,
        feature_names=feature_names,
        categorical_names=categorical_names,
        parameters=parameters,
        seed=int(seed + fold_id * 1000),
    )
    uptake_raw, conditional_raw = model.predict_raw(testing)
    uptake = uptake_calibrator.predict(uptake_raw)
    conditional = conditional_calibrator.predict(conditional_raw)
    conditional_target, realized = _realized_diffusion(
        testing, model.target_transformer
    )
    expected = uptake * conditional
    return pd.DataFrame(
        {
            "paper_id": testing["paper_id"].astype(str).to_numpy(),
            "publication_year": testing["publication_year"].to_numpy(
                dtype=int
            ),
            "domain12": testing["domain12"].astype(str).to_numpy(),
            "horizon": int(horizon),
            "model_id": model_id,
            "outer_fold_id": int(fold_id),
            "future_uptake": testing["future_uptake"].to_numpy(dtype=float),
            "conditional_diffusion_member": testing[
                "conditional_diffusion_member"
            ].to_numpy(dtype=int),
            "conditional_diffusion_target": conditional_target,
            "realized_diffusion_target": realized,
            "uptake_probability": uptake,
            "conditional_diffusion_prediction": conditional,
            "expected_diffusion_score": expected,
            "conditional_residual_quantile_90": (
                conditional_residual_quantile
            ),
            "realized_residual_quantile_90": realized_residual_quantile,
            "selected_parameter_id": "medium",
            "scope": "all_period_fixed_temporal_oof",
        }
    )


def _checkpoint_valid(
    checkpoint: pd.DataFrame,
    testing: pd.DataFrame,
    *,
    model_id: str,
    fold_id: int,
) -> bool:
    required = {
        "paper_id",
        "model_id",
        "outer_fold_id",
        "expected_diffusion_score",
        "realized_diffusion_target",
    }
    return (
        required.issubset(checkpoint.columns)
        and len(checkpoint) == len(testing)
        and checkpoint["model_id"].eq(model_id).all()
        and checkpoint["outer_fold_id"].eq(int(fold_id)).all()
        and set(checkpoint["paper_id"].astype(str))
        == set(testing["paper_id"].astype(str))
    )


def run_fixed_medium_oof(
    frame: pd.DataFrame,
    *,
    feature_sets: Mapping[str, Sequence[str]],
    model_ids: Sequence[str],
    fold_config: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any],
    categorical_features: Sequence[str],
    inner_folds: int,
    horizon: int,
    checkpoint_root: Path,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run or resume all requested fixed-medium OOF fits."""
    folds = explicit_temporal_splits(frame, fold_config)
    checkpoint_root = Path(checkpoint_root)
    prediction_rows: List[pd.DataFrame] = []
    fold_rows: List[Dict[str, Any]] = []
    for fold in folds:
        training = frame.iloc[fold["train_index"]].copy()
        testing = frame.iloc[fold["test_index"]].copy()
        fold_rows.append(
            {
                key: value
                for key, value in fold.items()
                if key not in {"train_index", "test_index"}
            }
        )
        for model_id in model_ids:
            feature_names = tuple(feature_sets[model_id])
            missing = sorted(set(feature_names) - set(frame.columns))
            if missing:
                raise ValueError(f"{model_id} is missing features: {missing}")
            categorical = tuple(
                name
                for name in categorical_features
                if name in feature_names
            )
            checkpoint_path = (
                checkpoint_root
                / f"D{int(horizon)}"
                / model_id
                / f"fold_{int(fold['fold_id'])}.parquet"
            )
            if checkpoint_path.is_file():
                checkpoint = pd.read_parquet(checkpoint_path)
                if not _checkpoint_valid(
                    checkpoint,
                    testing,
                    model_id=model_id,
                    fold_id=int(fold["fold_id"]),
                ):
                    raise ValueError(f"invalid OOF checkpoint: {checkpoint_path}")
            else:
                checkpoint = _run_fold_model(
                    training,
                    testing,
                    model_id=model_id,
                    feature_names=feature_names,
                    categorical_names=categorical,
                    parameters=parameters,
                    inner_folds=int(inner_folds),
                    horizon=int(horizon),
                    fold_id=int(fold["fold_id"]),
                    seed=int(seed),
                )
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.to_parquet(checkpoint_path, index=False)
            prediction_rows.append(checkpoint)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    if predictions.duplicated(["paper_id", "model_id"]).any():
        raise ValueError("OOF created duplicate paper/model predictions")
    expected_rows = len(
        frame[frame["publication_year"].between(1986, 2017)]
    )
    counts = predictions.groupby("model_id")["paper_id"].nunique()
    if not counts.eq(expected_rows).all():
        raise ValueError("models do not share the complete OOF paper set")
    return predictions, pd.DataFrame(fold_rows)


def _rank_group_ids(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    changes = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    sorted_groups = np.cumsum(changes) - 1
    groups = np.empty(len(values), dtype=np.int64)
    groups[order] = sorted_groups
    return groups


def _weighted_midranks(
    groups: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    group_weights = np.bincount(groups, weights=counts)
    cumulative = np.cumsum(group_weights)
    midranks = cumulative - (group_weights - 1.0) / 2.0
    return midranks[groups]


def _weighted_rank_correlation(
    left_groups: np.ndarray,
    right_groups: np.ndarray,
    counts: np.ndarray,
) -> float:
    left = _weighted_midranks(left_groups, counts)
    right = _weighted_midranks(right_groups, counts)
    total = float(counts.sum())
    left_mean = float(np.dot(counts, left) / total)
    right_mean = float(np.dot(counts, right) / total)
    left_centered = left - left_mean
    right_centered = right - right_mean
    covariance = float(np.dot(counts, left_centered * right_centered))
    denominator = math.sqrt(
        float(np.dot(counts, left_centered**2))
        * float(np.dot(counts, right_centered**2))
    )
    return covariance / denominator if denominator > 0.0 else np.nan


def paired_bootstrap_gain_intervals(
    truth: Sequence[float],
    candidate: Sequence[float],
    baselines: Mapping[str, Sequence[float]],
    *,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Exact paper-resampling CIs using weighted midranks for speed."""
    truth_array = np.asarray(truth, dtype=float)
    candidate_array = np.asarray(candidate, dtype=float)
    baseline_arrays = {
        name: np.asarray(values, dtype=float)
        for name, values in baselines.items()
    }
    valid = np.isfinite(truth_array) & np.isfinite(candidate_array)
    for values in baseline_arrays.values():
        valid &= np.isfinite(values)
    truth_array = truth_array[valid]
    candidate_array = candidate_array[valid]
    baseline_arrays = {
        name: values[valid] for name, values in baseline_arrays.items()
    }
    if len(truth_array) < 3:
        raise ValueError("paired bootstrap requires at least three papers")
    groups = {
        "truth": _rank_group_ids(truth_array),
        "candidate": _rank_group_ids(candidate_array),
        **{
            name: _rank_group_ids(values)
            for name, values in baseline_arrays.items()
        },
    }
    estimates = {
        name: np.empty(int(iterations), dtype=float)
        for name in baseline_arrays
    }
    rng = np.random.default_rng(int(seed))
    for iteration in range(int(iterations)):
        sampled = rng.integers(0, len(truth_array), size=len(truth_array))
        counts = np.bincount(
            sampled, minlength=len(truth_array)
        ).astype(float)
        candidate_rho = _weighted_rank_correlation(
            groups["truth"], groups["candidate"], counts
        )
        for name in baseline_arrays:
            estimates[name][iteration] = candidate_rho - (
                _weighted_rank_correlation(
                    groups["truth"], groups[name], counts
                )
            )
    rows = []
    candidate_rho = safe_spearman(truth_array, candidate_array)
    for name, values in baseline_arrays.items():
        finite = estimates[name][np.isfinite(estimates[name])]
        rows.append(
            {
                "candidate_model_id": "final_innovation_plus_k1",
                "baseline_model_id": name,
                "n_papers": len(truth_array),
                "candidate_spearman": candidate_rho,
                "baseline_spearman": safe_spearman(truth_array, values),
                "spearman_gain": candidate_rho
                - safe_spearman(truth_array, values),
                "gain_ci_low": float(np.quantile(finite, 0.025)),
                "gain_ci_high": float(np.quantile(finite, 0.975)),
                "bootstrap_iterations": int(iterations),
                "bootstrap_unit": "paper_id",
            }
        )
    return pd.DataFrame(rows)


def evaluate_oof_points(
    predictions: pd.DataFrame,
    *,
    minimum_domain_rows: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute headline, fold, and 12-domain rank results."""
    overall_rows = []
    fold_rows = []
    domain_rows = []
    grouping = ["horizon", "model_id"]
    for (horizon, model_id), group in predictions.groupby(
        grouping, sort=False
    ):
        overall_rows.append(
            {
                "horizon": int(horizon),
                "model_id": str(model_id),
                "n_oof": len(group),
                "n_rank_valid": int(
                    group[
                        [
                            "realized_diffusion_target",
                            "expected_diffusion_score",
                        ]
                    ]
                    .dropna()
                    .shape[0]
                ),
                "spearman_expected": safe_spearman(
                    group["realized_diffusion_target"],
                    group["expected_diffusion_score"],
                ),
            }
        )
        for fold_id, fold in group.groupby("outer_fold_id", sort=True):
            fold_rows.append(
                {
                    "horizon": int(horizon),
                    "model_id": str(model_id),
                    "outer_fold_id": int(fold_id),
                    "test_year_min": int(fold["publication_year"].min()),
                    "test_year_max": int(fold["publication_year"].max()),
                    "n_oof": len(fold),
                    "spearman_expected": safe_spearman(
                        fold["realized_diffusion_target"],
                        fold["expected_diffusion_score"],
                    ),
                }
            )
        for domain, subset in group.groupby("domain12", sort=True):
            domain_rows.append(
                {
                    "horizon": int(horizon),
                    "model_id": str(model_id),
                    "domain12": str(domain),
                    "n_oof": len(subset),
                    "reportable": int(
                        len(subset) >= int(minimum_domain_rows)
                    ),
                    "spearman_expected": safe_spearman(
                        subset["realized_diffusion_target"],
                        subset["expected_diffusion_score"],
                    ),
                }
            )
    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(domain_rows),
    )


def _wide_predictions(
    predictions: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    selected = predictions[predictions["horizon"].eq(int(horizon))]
    truth = selected[
        ["paper_id", "model_id", "realized_diffusion_target"]
    ].pivot(index="paper_id", columns="model_id", values="realized_diffusion_target")
    if truth.max(axis=1, skipna=True).sub(
        truth.min(axis=1, skipna=True)
    ).fillna(0.0).abs().gt(1e-12).any():
        raise ValueError("models were evaluated against different labels")
    scores = selected.pivot(
        index="paper_id",
        columns="model_id",
        values="expected_diffusion_score",
    )
    output = pd.DataFrame(
        {"realized_diffusion_target": truth.bfill(axis=1).iloc[:, 0]}
    )
    return output.join(scores, how="inner")


def _acceptance_results(
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    acceptance = config["acceptance"]
    metric = metrics.set_index(["horizon", "model_id"])[
        "spearman_expected"
    ].to_dict()
    d5 = float(metric[(5, "final_innovation_plus_k1")])
    comparison = comparisons.set_index("baseline_model_id").to_dict("index")
    d3_gain = float(
        metric[(3, "final_innovation_plus_k1")]
        - metric[(3, "k1_controls")]
    )
    d8_gain = float(
        metric[(8, "final_innovation_plus_k1")]
        - metric[(8, "k1_controls")]
    )
    gates = {
        "d5_target_0_75": d5 >= float(acceptance["d5_spearman_target"]),
        "d5_hard_minimum_0_74": d5
        >= float(acceptance["d5_spearman_hard_minimum"]),
        "noninferior_to_b0": float(
            comparison["b0_v6_primary_plus_k0"]["gain_ci_low"]
        )
        >= float(acceptance["paired_noninferiority_margin_vs_b0"]),
        "positive_increment_over_k1": float(
            comparison["k1_controls"]["gain_ci_low"]
        )
        > float(acceptance["gain_vs_k1_ci_lower_must_exceed"]),
        "d3_direction_positive": d3_gain > 0.0,
        "d8_direction_positive": d8_gain > 0.0,
    }
    return {
        "headline_d5_spearman": d5,
        "d3_gain_over_k1": d3_gain,
        "d8_gain_over_k1": d8_gain,
        "gates": gates,
        "all_required_gates_pass": bool(all(gates.values())),
    }


def _reflection(
    acceptance: Mapping[str, Any],
    frame_d5: pd.DataFrame,
    final_features: Sequence[str],
) -> Mapping[str, Any]:
    missingness = {
        name: float(frame_d5[name].isna().mean()) for name in final_features
    }
    failed = [
        key for key, passed in acceptance["gates"].items() if not passed
    ]
    return {
        "triggered": bool(failed),
        "failed_gates": failed,
        "diagnostic_order": [
            "formula_fidelity",
            "coverage_and_missingness",
            "temporal_drift",
            "target_noise_and_zero_process",
            "model_capacity_and_calibration",
        ],
        "final_feature_missingness": missingness,
        "forbidden_response": (
            "Do not remove an admitted indicator, paper, year, or domain "
            "because it lowers OOF."
        ),
        "next_revision_rule": (
            "Any adjustment must receive a new version, be justified by "
            "formula/data diagnostics rather than OOF feature selection, and "
            "rerun every registered fold."
        ),
    }


def run_v6_1_experiment(
    project_root: Path,
    config_path: Path,
) -> Tuple[Mapping[str, Any], Path]:
    """Run the complete D5 main and D3/D8 directional v6.1 experiment."""
    project_root = Path(project_root).resolve()
    config_path = Path(config_path).resolve()
    freeze = freeze_registry_before_oof(project_root, config_path)
    config = load_simple_config(config_path)
    registry_path = _resolve(
        project_root, config["paths"]["candidate_registry"]
    )
    registry = load_candidate_registry_v6_1(registry_path)
    dataset_dir = _resolve(
        project_root, config["paths"]["v6_1_dataset"]
    ).resolve()
    analysis_root = _resolve(
        project_root, config["paths"]["v6_1_analysis"]
    ).resolve()
    lineage = {
        "protocol_version": MODEL_PROTOCOL_VERSION_V6_1,
        "modeling_implementation_sha256": sha256_file(
            Path(__file__).resolve()
        ),
        "shared_modeling_implementation_sha256": sha256_file(
            project_root
            / "aspr"
            / "nature_multihorizon"
            / "modeling_v6.py"
        ),
        "config_sha256": sha256_file(config_path),
        "registry_freeze_artifact_id": freeze["artifact_id"],
        "registry_canonical_sha256": candidate_registry_sha256(registry),
        "innovation_features_sha256": sha256_file(
            dataset_dir / "innovation_candidate_features.parquet"
        ),
        "controls_sha256": sha256_file(
            dataset_dir / "control_features_v6_1.parquet"
        ),
        "control_registry_sha256": sha256_file(
            _resolve(
                project_root, config["paths"]["control_registry"]
            ).resolve()
        ),
        "targets_sha256": sha256_file(
            dataset_dir / "targets_zero_inclusive.parquet"
        ),
        "cohort_sha256": sha256_file(
            dataset_dir / "cohort_membership.parquet"
        ),
    }
    run_hash = _canonical_hash(lineage)
    run_dir = analysis_root / (
        f"oof_{run_hash.removeprefix('sha256:')[:12]}"
    )
    manifest_path = run_dir / "oof_run_manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8")), run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    feature_sets = build_v6_1_feature_sets(registry, config)
    all_predictions = []
    all_folds = []
    frames: Dict[int, pd.DataFrame] = {}
    for horizon in (
        int(config["main_horizon"]),
        *[int(item) for item in config["supplementary_horizons"]],
    ):
        frame = assemble_all_period_frame(dataset_dir, horizon=horizon)
        frames[horizon] = frame
        if horizon == int(config["main_horizon"]):
            model_ids = _unique(
                (
                    *config["main_model_ids"],
                    *config["sensitivity_model_ids"],
                )
            )
        else:
            model_ids = tuple(config["supplementary_model_ids"])
        predictions, folds = run_fixed_medium_oof(
            frame,
            feature_sets=feature_sets,
            model_ids=model_ids,
            fold_config=config["temporal_folds"],
            parameters=config["model"],
            categorical_features=config["categorical_features"],
            inner_folds=int(config["model"]["inner_temporal_folds"]),
            horizon=horizon,
            checkpoint_root=run_dir / "checkpoints",
            seed=int(config["model"]["seed"]),
        )
        all_predictions.append(predictions)
        folds["horizon"] = horizon
        all_folds.append(folds)
    predictions = pd.concat(all_predictions, ignore_index=True)
    folds = pd.concat(all_folds, ignore_index=True)
    metrics, fold_metrics, domain_metrics = evaluate_oof_points(
        predictions,
        minimum_domain_rows=int(
            config["evaluation"]["minimum_domain_rows"]
        ),
    )
    wide = _wide_predictions(predictions, horizon=5).dropna()
    comparisons = paired_bootstrap_gain_intervals(
        wide["realized_diffusion_target"],
        wide["final_innovation_plus_k1"],
        {
            "k1_controls": wide["k1_controls"],
            "b0_v6_primary_plus_k0": wide["b0_v6_primary_plus_k0"],
        },
        iterations=int(config["evaluation"]["bootstrap_iterations"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
    )
    comparisons.insert(0, "horizon", 5)
    acceptance = _acceptance_results(metrics, comparisons, config)
    reflection = _reflection(
        acceptance,
        frames[5],
        registry.primary_feature_names,
    )
    outputs = {
        "predictions": run_dir / "oof_predictions.parquet",
        "folds": run_dir / "temporal_folds.csv",
        "metrics": run_dir / "oof_metrics.csv",
        "fold_metrics": run_dir / "oof_fold_metrics.csv",
        "domain_metrics": run_dir / "oof_domain_metrics.csv",
        "comparisons": run_dir / "paired_bootstrap_comparisons.csv",
        "acceptance": run_dir / "acceptance_results.json",
        "reflection": run_dir / "reflection_report.json",
    }
    predictions.to_parquet(outputs["predictions"], index=False)
    folds.to_csv(outputs["folds"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    fold_metrics.to_csv(outputs["fold_metrics"], index=False)
    domain_metrics.to_csv(outputs["domain_metrics"], index=False)
    comparisons.to_csv(outputs["comparisons"], index=False)
    outputs["acceptance"].write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    outputs["reflection"].write_text(
        json.dumps(reflection, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "artifact_kind": "aspr_v6_1_fixed_medium_all_period_oof",
        "lineage": lineage,
        "run_hash": run_hash,
        "headline_metric": config["evaluation"]["headline_metric"],
        "conditional_spearman_reported": False,
        "same_papers_folds_and_labels_across_models": True,
        "parameter_selection_from_oof": False,
        "fixed_parameter_id": "medium",
        "feature_selection_from_oof": False,
        "acceptance": acceptance,
        "outputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in outputs.items()
        },
    }
    manifest["artifact_id"] = _canonical_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest, run_dir


__all__ = [
    "MODEL_PROTOCOL_VERSION_V6_1",
    "assemble_all_period_frame",
    "build_v6_1_feature_sets",
    "explicit_temporal_splits",
    "freeze_registry_before_oof",
    "load_simple_config",
    "paired_bootstrap_gain_intervals",
    "run_fixed_medium_oof",
    "run_v6_1_experiment",
]
