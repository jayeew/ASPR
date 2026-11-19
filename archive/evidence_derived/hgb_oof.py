"""Canonical four-set HGB OOF adapter for the simplified evidence protocol."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from gear.nature_multihorizon.active_dataset import load_active_dataset
from gear.nature_multihorizon.modeling_v6 import safe_spearman
from gear.nature_multihorizon.modeling_v6_1 import (
    assemble_all_period_frame,
    run_fixed_medium_oof,
)

from .core import canonical_json, sha256_text

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DEFAULT_MATRIX_MANIFEST = ROOT / "outputs/training_matrix_manifest.json"
DEFAULT_FROZEN_MANIFEST = ROOT / "outputs/final_feature_sets.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs/nature_multihorizon/hgb_uncapped_v2.json"
DEFAULT_OUTPUT = ROOT / "outputs/hgb_oof_working"
MODEL_FAMILY = "hgb"
FORBIDDEN_OUTCOME_COLUMNS = {
    "conditional_diffusion_target",
    "expected_diffusion_score",
    "future_field_reach",
    "future_subfield_reach",
    "future_topic_reach",
    "future_field_simpson",
    "future_topic_simpson",
    "future_uptake",
    "horizon",
    "outer_fold_id",
    "realized_diffusion_target",
    "uptake_probability",
}


class HgbOofContractError(RuntimeError):
    """Raised when frozen inputs or OOF artifacts fail closed."""


@dataclass(frozen=True)
class TrainingBundle:
    """Validated canonical feature matrices and their frozen lineage."""

    set_names: tuple[str, ...]
    feature_sets: dict[str, tuple[str, ...]]
    canonical_feature_sets: dict[str, tuple[str, ...]]
    feature_aliases: dict[str, str]
    categorical_features: tuple[str, ...]
    matrix: pd.DataFrame
    freeze_hash: str
    protocol_hash: str
    matrix_manifest_path: Path
    frozen_manifest_path: Path
    matrix_hashes: dict[str, str]


OofRunner = Callable[..., tuple[pd.DataFrame, pd.DataFrame]]


def sha256_file(path: Path) -> str:
    """Return an unprefixed SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve(source: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else source.parent / path


def _reconstructed_freeze_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    sets = dict(payload.get("sets") or {})
    canonical = payload.get("canonical_sets") or {}
    sets.update(
        {
            "all": canonical.get("F_all", []),
            "model": canonical.get("F_model", []),
            "strict": canonical.get("F_strict", []),
        }
    )
    return sets


def _validate_frozen_lineage(
    matrix_payload: Mapping[str, Any], frozen_payload: Mapping[str, Any]
) -> tuple[str, str]:
    if matrix_payload.get("contract") != "evidence_derived_training_matrices_v1":
        raise HgbOofContractError("unsupported training matrix manifest")
    if frozen_payload.get("frozen_before_model_training") is not True:
        raise HgbOofContractError("feature sets were not frozen before model training")
    if matrix_payload.get("frozen_before_model_training") is not True:
        raise HgbOofContractError("matrix manifest is not frozen")
    if frozen_payload.get("outcome_columns_used") is not False:
        raise HgbOofContractError("feature selection did not pass outcome blindness")
    if matrix_payload.get("outcome_columns_used") is not False:
        raise HgbOofContractError("matrix materialization used outcome columns")
    reconstructed = sha256_text(
        canonical_json(_reconstructed_freeze_payload(frozen_payload))
    )
    frozen_hash = str(frozen_payload.get("freeze_hash") or "")
    matrix_hash = str(matrix_payload.get("feature_set_freeze_hash") or "")
    if not frozen_hash or reconstructed != frozen_hash or matrix_hash != frozen_hash:
        raise HgbOofContractError("feature-set freeze hash mismatch")
    protocol_hash = str(frozen_payload.get("protocol_hash") or "")
    if not protocol_hash or matrix_payload.get("protocol_hash") != protocol_hash:
        raise HgbOofContractError("protocol hash mismatch")
    return frozen_hash, protocol_hash


def _load_set_matrix(
    manifest_path: Path,
    definition: Mapping[str, Any],
    expected_ids: Sequence[str],
) -> tuple[pd.DataFrame, str]:
    path = _resolve(manifest_path, str(definition.get("path") or ""))
    if not path.is_file():
        raise HgbOofContractError(f"training matrix is missing: {path}")
    digest = sha256_file(path)
    if digest != definition.get("sha256"):
        raise HgbOofContractError(f"training matrix hash mismatch: {path.name}")
    feature_names = tuple(str(item) for item in definition.get("feature_names") or [])
    schema = pq.ParquetFile(path).schema_arrow.names
    if schema != ["paper_id", *feature_names]:
        raise HgbOofContractError(f"training matrix schema mismatch: {path.name}")
    frame = pd.read_parquet(path)
    if len(frame) != int(definition.get("row_count", -1)):
        raise HgbOofContractError(f"training matrix row count mismatch: {path.name}")
    paper_ids = frame["paper_id"].astype(str)
    if frame["paper_id"].isna().any() or paper_ids.duplicated().any():
        raise HgbOofContractError(f"invalid paper_id grain: {path.name}")
    expected = [str(item) for item in expected_ids]
    if len(paper_ids) != len(expected) or set(paper_ids) != set(expected):
        raise HgbOofContractError(f"paper_id membership mismatch: {path.name}")
    if paper_ids.tolist() != expected:
        order = pd.Series(range(len(expected)), index=expected)
        frame = (
            frame.assign(paper_id=paper_ids, __paper_order=paper_ids.map(order))
            .sort_values("__paper_order", kind="stable")
            .drop(columns="__paper_order")
            .reset_index(drop=True)
        )
    return frame, digest


def _categorical_columns(
    frame: pd.DataFrame, features: Sequence[str]
) -> tuple[str, ...]:
    categorical: list[str] = []
    for name in features:
        dtype = frame[name].dtype
        if pd.api.types.is_datetime64_any_dtype(dtype):
            raise HgbOofContractError(f"datetime feature is unsupported: {name}")
        if not pd.api.types.is_numeric_dtype(dtype):
            categorical.append(name)
    return tuple(categorical)


def load_training_bundle(
    matrix_manifest_path: Path,
    frozen_manifest_path: Path,
    expected_paper_ids: Sequence[str],
) -> TrainingBundle:
    """Load four frozen canonical matrices and enforce their full contract."""
    matrix_path = Path(matrix_manifest_path).resolve()
    frozen_path = Path(frozen_manifest_path).resolve()
    matrix_payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    frozen_payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    freeze_hash, protocol_hash = _validate_frozen_lineage(
        matrix_payload, frozen_payload
    )
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    set_names = tuple(str(item) for item in protocol["feature_sets"])
    definitions = matrix_payload.get("sets") or {}
    if set(definitions) != set(set_names):
        raise HgbOofContractError("matrix set names differ from frozen protocol")
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    canonical_feature_sets: dict[str, tuple[str, ...]] = {}
    for set_name in set_names:
        definition = definitions[set_name]
        source_set = str(definition.get("source_set") or "")
        frozen_ids = tuple((frozen_payload.get("sets") or {}).get(source_set) or [])
        if tuple(definition.get("indicator_ids") or []) != frozen_ids:
            raise HgbOofContractError(f"frozen membership mismatch: {set_name}")
        frame, digest = _load_set_matrix(matrix_path, definition, expected_paper_ids)
        frames[set_name] = frame
        hashes[set_name] = digest
        canonical_feature_sets[set_name] = tuple(definition["feature_names"])
    for left, right in pairwise(set_names):
        if not set(canonical_feature_sets[left]).issubset(
            canonical_feature_sets[right]
        ):
            raise HgbOofContractError(f"canonical sets are not nested: {left}/{right}")
    superset = frames[set_names[-1]]
    forbidden = FORBIDDEN_OUTCOME_COLUMNS & set(superset.columns)
    forbidden.update(name for name in superset.columns if name.startswith("future_"))
    if forbidden:
        raise HgbOofContractError(f"outcome leakage columns: {sorted(forbidden)}")
    canonical_superset = canonical_feature_sets[set_names[-1]]
    canonical_categorical = _categorical_columns(superset, canonical_superset)
    aliases = {
        name: f"canonical_feature_{index:04d}"
        for index, name in enumerate(canonical_superset, start=1)
    }
    feature_sets = {
        set_name: tuple(aliases[name] for name in canonical_feature_sets[set_name])
        for set_name in set_names
    }
    matrix = superset.rename(columns=aliases)
    categorical = tuple(aliases[name] for name in canonical_categorical)
    return TrainingBundle(
        set_names=set_names,
        feature_sets=feature_sets,
        canonical_feature_sets=canonical_feature_sets,
        feature_aliases=aliases,
        categorical_features=categorical,
        matrix=matrix,
        freeze_hash=freeze_hash,
        protocol_hash=protocol_hash,
        matrix_manifest_path=matrix_path,
        frozen_manifest_path=frozen_path,
        matrix_hashes=hashes,
    )


def assemble_frames(
    bundle: TrainingBundle, dataset_dir: Path
) -> dict[int, pd.DataFrame]:
    """Join the frozen canonical superset to each active mature cohort."""
    frames: dict[int, pd.DataFrame] = {}
    for horizon in (3, 5, 8):
        frame = assemble_all_period_frame(dataset_dir, horizon=horizon)
        rows = len(frame)
        frame = frame.merge(
            bundle.matrix, on="paper_id", how="left", validate="one_to_one"
        )
        if len(frame) != rows:
            raise HgbOofContractError(f"D{horizon} feature merge changed row count")
        frames[horizon] = frame
    return frames


def evaluate_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute overall, fold, and domain OOF Spearman metrics."""
    keys = ["horizon", "model_family", "model_id"]
    overall: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    domains: list[dict[str, Any]] = []
    for key, group in predictions.groupby(keys, sort=False):
        base = dict(zip(keys, key, strict=True))
        overall.append(
            {
                **base,
                "n_oof": len(group),
                "spearman": safe_spearman(
                    group["realized_diffusion_target"],
                    group["expected_diffusion_score"],
                ),
            }
        )
        for fold_id, subset in group.groupby("outer_fold_id", sort=True):
            folds.append(
                {
                    **base,
                    "outer_fold_id": int(fold_id),
                    "test_year_min": int(subset["publication_year"].min()),
                    "test_year_max": int(subset["publication_year"].max()),
                    "n_oof": len(subset),
                    "spearman": safe_spearman(
                        subset["realized_diffusion_target"],
                        subset["expected_diffusion_score"],
                    ),
                }
            )
        for domain, subset in group.groupby("domain12", sort=True):
            domains.append(
                {
                    **base,
                    "domain12": str(domain),
                    "n_oof": len(subset),
                    "spearman": safe_spearman(
                        subset["realized_diffusion_target"],
                        subset["expected_diffusion_score"],
                    ),
                }
            )
    return pd.DataFrame(overall), pd.DataFrame(folds), pd.DataFrame(domains)


def build_model_comparison(
    metrics: pd.DataFrame, set_names: Sequence[str]
) -> pd.DataFrame:
    """Rank frozen sets without feeding performance back into selection."""
    baseline = str(set_names[0])
    rows: list[pd.DataFrame] = []
    for horizon, group in metrics.groupby("horizon", sort=True):
        ranked = group.copy()
        base = ranked.loc[ranked["model_id"].eq(baseline), "spearman"]
        if len(base) != 1:
            raise HgbOofContractError(f"D{horizon} lacks baseline {baseline}")
        ranked["rank"] = (
            ranked["spearman"].rank(method="min", ascending=False).astype(int)
        )
        ranked["baseline_model_id"] = baseline
        ranked["delta_vs_baseline"] = ranked["spearman"] - float(base.iloc[0])
        ranked["selection_feedback_used"] = False
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True)


def build_paper_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create fold-valid long-form OOF paper scores for every frozen set."""
    columns = [
        "paper_id",
        "publication_year",
        "domain12",
        "horizon",
        "model_family",
        "model_id",
        "outer_fold_id",
        "expected_diffusion_score",
        "realized_diffusion_target",
    ]
    scores = predictions[columns].copy()
    scores["oof_percentile_score"] = 100.0 * scores.groupby(
        ["horizon", "model_id"], sort=False
    )["expected_diffusion_score"].rank(method="average", pct=True)
    scores["score_scope"] = "fold_valid_oof_only"
    return scores


def _write_outputs(
    output_dir: Path,
    predictions: pd.DataFrame,
    set_names: Sequence[str],
) -> dict[str, Path]:
    metrics, folds, domains = evaluate_predictions(predictions)
    comparison = build_model_comparison(metrics, set_names)
    scores = build_paper_scores(predictions)
    outputs = {
        "oof_predictions": output_dir / "oof_predictions.parquet",
        "oof_metrics": output_dir / "oof_metrics.csv",
        "oof_fold_metrics": output_dir / "oof_fold_metrics.csv",
        "oof_domain_metrics": output_dir / "oof_domain_metrics.csv",
        "model_comparison": output_dir / "model_comparison.csv",
        "paper_scores": output_dir / "paper_scores.parquet",
    }
    predictions.to_parquet(outputs["oof_predictions"], index=False)
    metrics.to_csv(outputs["oof_metrics"], index=False)
    folds.to_csv(outputs["oof_fold_metrics"], index=False)
    domains.to_csv(outputs["oof_domain_metrics"], index=False)
    comparison.to_csv(outputs["model_comparison"], index=False)
    scores.to_parquet(outputs["paper_scores"], index=False)
    return outputs


def run_with_frames(
    bundle: TrainingBundle,
    frames: Mapping[int, pd.DataFrame],
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    oof_runner: OofRunner = run_fixed_medium_oof,
    dataset_version: str = "synthetic",
) -> dict[str, Any]:
    """Run or resume OOF from validated matrices and assembled horizon frames."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions: list[pd.DataFrame] = []
    for horizon in (3, 5, 8):
        result, _ = oof_runner(
            frames[horizon],
            feature_sets=bundle.feature_sets,
            model_ids=bundle.set_names,
            fold_config=config["horizon_folds"][str(horizon)],
            parameters=config["hgb"],
            categorical_features=bundle.categorical_features,
            inner_folds=int(config["hgb"]["inner_temporal_folds"]),
            horizon=horizon,
            checkpoint_root=output / "checkpoints/hgb",
            seed=int(config["hgb"]["seed"]),
        )
        result["model_family"] = MODEL_FAMILY
        predictions.append(result)
    combined = pd.concat(predictions, ignore_index=True)
    paths = _write_outputs(output, combined, bundle.set_names)
    checkpoint_count = len(
        list((output / "checkpoints/hgb").glob("D*/*/fold_*.parquet"))
    )
    manifest = {
        "contract": "evidence_derived_canonical_hgb_oof_v1",
        "result_scope": "new_protocol_canonical_four_set_oof",
        "dataset_version": dataset_version,
        "horizons": [3, 5, 8],
        "set_names": list(bundle.set_names),
        "feature_counts": {
            name: len(bundle.feature_sets[name]) for name in bundle.set_names
        },
        "canonical_feature_sets": {
            name: list(bundle.canonical_feature_sets[name]) for name in bundle.set_names
        },
        "model_feature_aliases": bundle.feature_aliases,
        "feature_set_freeze_hash": bundle.freeze_hash,
        "protocol_hash": bundle.protocol_hash,
        "matrix_manifest": {
            "path": str(bundle.matrix_manifest_path),
            "sha256": sha256_file(bundle.matrix_manifest_path),
        },
        "frozen_manifest": {
            "path": str(bundle.frozen_manifest_path),
            "sha256": sha256_file(bundle.frozen_manifest_path),
        },
        "matrix_hashes": bundle.matrix_hashes,
        "checkpoint_count": checkpoint_count,
        "selection_feedback_used": False,
        "network_used": False,
        "outputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
    }
    write_json(output / "run_manifest.json", manifest)
    return manifest


def _active_paper_ids(dataset_dir: Path) -> list[str]:
    papers = pd.read_parquet(
        dataset_dir / "papers_primary_articles.parquet", columns=["paper_id"]
    )
    return papers["paper_id"].astype(str).tolist()


def _require_complete_protocol_audit(path: Path) -> None:
    if not path.is_file():
        raise HgbOofContractError("protocol audit is not COMPLETE")
    status_lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("Status:")
    ]
    if status_lines != ["Status: **COMPLETE**"]:
        raise HgbOofContractError("protocol audit is not COMPLETE")


def run(
    matrix_manifest_path: Path,
    frozen_manifest_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the canonical experiment against the active registered dataset."""
    active = load_active_dataset(PROJECT_ROOT)
    _require_complete_protocol_audit(
        Path(matrix_manifest_path).parent / "audit_report.md"
    )
    dataset_dir = Path(active["feature_dataset_dir"])
    bundle = load_training_bundle(
        matrix_manifest_path,
        frozen_manifest_path,
        _active_paper_ids(dataset_dir),
    )
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("dataset_version") != active["active_dataset_version"]:
        raise HgbOofContractError("model config and active dataset differ")
    frames = assemble_frames(bundle, dataset_dir)
    return run_with_frames(
        bundle,
        frames,
        config,
        output_dir,
        dataset_version=str(active["active_dataset_version"]),
    )


def _metrics_match(predictions: pd.DataFrame, metrics: pd.DataFrame) -> bool:
    recomputed, _, _ = evaluate_predictions(predictions)
    keys = ["horizon", "model_family", "model_id"]
    left = metrics.sort_values(keys).reset_index(drop=True)
    right = recomputed.sort_values(keys).reset_index(drop=True)
    if left[keys + ["n_oof"]].equals(right[keys + ["n_oof"]]) is False:
        return False
    for observed, expected in zip(left["spearman"], right["spearman"], strict=True):
        if pd.isna(observed) and pd.isna(expected):
            continue
        if not math.isclose(float(observed), float(expected), abs_tol=1e-12):
            return False
    return True


def _sets_share_alignment(predictions: pd.DataFrame, set_names: Sequence[str]) -> bool:
    columns = [
        "paper_id",
        "publication_year",
        "outer_fold_id",
        "realized_diffusion_target",
    ]
    for _, group in predictions.groupby("horizon", sort=True):
        baseline = (
            group[group["model_id"].eq(set_names[0])][columns]
            .sort_values("paper_id")
            .reset_index(drop=True)
        )
        for set_name in set_names[1:]:
            candidate = (
                group[group["model_id"].eq(set_name)][columns]
                .sort_values("paper_id")
                .reset_index(drop=True)
            )
            if not baseline.equals(candidate):
                return False
    return True


def _artifact_hashes_match(manifest: Mapping[str, Any]) -> bool:
    try:
        return all(
            sha256_file(Path(item["path"])) == item["sha256"]
            for item in manifest["outputs"].values()
        )
    except (KeyError, OSError, TypeError):
        return False


def _checkpoint_layout_matches(
    output: Path, set_names: Sequence[str], config: Mapping[str, Any]
) -> tuple[bool, int]:
    root = output / "checkpoints/hgb"
    observed = {path.relative_to(root) for path in root.glob("D*/*/fold_*.parquet")}
    expected = {
        Path(f"D{horizon}/{set_name}/fold_{int(fold['fold_id'])}.parquet")
        for horizon in (3, 5, 8)
        for set_name in set_names
        for fold in config["horizon_folds"][str(horizon)]
    }
    return observed == expected, len(observed)


def _model_comparison_valid(path: Path, expected_rows: int) -> bool:
    comparison = pd.read_csv(path)
    return bool(
        len(comparison) == expected_rows
        and comparison["selection_feedback_used"]
        .astype(str)
        .str.casefold()
        .eq("false")
        .all()
        and comparison["rank"].ge(1).all()
    )


def _paper_scores_match(scores: pd.DataFrame, predictions: pd.DataFrame) -> bool:
    columns = [
        "paper_id",
        "publication_year",
        "domain12",
        "horizon",
        "model_family",
        "model_id",
        "outer_fold_id",
        "expected_diffusion_score",
        "realized_diffusion_target",
    ]
    if len(scores) != len(predictions) or not scores[columns].equals(
        predictions[columns]
    ):
        return False
    expected = 100.0 * predictions.groupby(["horizon", "model_id"], sort=False)[
        "expected_diffusion_score"
    ].rank(method="average", pct=True)
    difference = (scores["oof_percentile_score"] - expected).abs()
    return bool(
        scores["score_scope"].eq("fold_valid_oof_only").all()
        and difference.le(1e-12).all()
        and scores["oof_percentile_score"].between(0, 100).all()
    )


def validate_outputs(
    output_dir: Path,
    bundle: TrainingBundle,
    frames: Mapping[int, pd.DataFrame],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate canonical OOF artifacts against frozen inputs and folds."""
    output = Path(output_dir).resolve()
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(output / "oof_predictions.parquet")
    metrics = pd.read_csv(output / "oof_metrics.csv")
    scores = pd.read_parquet(output / "paper_scores.parquet")
    checks: dict[str, bool] = {}
    checks["canonical_contract"] = (
        manifest.get("contract") == "evidence_derived_canonical_hgb_oof_v1"
    )
    checks["frozen_lineage"] = (
        manifest.get("feature_set_freeze_hash") == bundle.freeze_hash
        and manifest.get("protocol_hash") == bundle.protocol_hash
        and manifest.get("set_names") == list(bundle.set_names)
        and manifest.get("matrix_manifest", {}).get("sha256")
        == sha256_file(bundle.matrix_manifest_path)
        and manifest.get("frozen_manifest", {}).get("sha256")
        == sha256_file(bundle.frozen_manifest_path)
    )
    checks["only_hgb"] = set(predictions["model_family"]) == {MODEL_FAMILY}
    checks["unique_oof_keys"] = not predictions.duplicated(
        ["paper_id", "horizon", "model_id", "outer_fold_id"]
    ).any()
    expected_metric_rows = 3 * len(bundle.set_names)
    checks["metric_coverage"] = (
        len(metrics) == expected_metric_rows
        and set(metrics["model_id"]) == set(bundle.set_names)
        and set(metrics["horizon"]) == {3, 5, 8}
    )
    checks["metric_values_recomputed"] = _metrics_match(predictions, metrics)
    checks["sets_share_papers_folds_labels"] = _sets_share_alignment(
        predictions, bundle.set_names
    )
    checks["paper_score_alignment"] = _paper_scores_match(scores, predictions)
    checks["fold_and_row_coverage"] = _validate_fold_coverage(
        predictions, bundle, frames, config
    )
    checkpoint_layout, checkpoint_count = _checkpoint_layout_matches(
        output, bundle.set_names, config
    )
    checks["checkpoint_layout"] = checkpoint_layout
    checks["model_comparison"] = _model_comparison_valid(
        output / "model_comparison.csv", expected_metric_rows
    )
    checks["artifact_hashes"] = _artifact_hashes_match(manifest)
    normalized_checks = {name: bool(value) for name, value in checks.items()}
    report = {
        "contract": "evidence_derived_canonical_hgb_oof_validation_v1",
        "passed": all(normalized_checks.values()),
        "checks": normalized_checks,
        "prediction_rows": len(predictions),
        "metric_rows": len(metrics),
        "checkpoint_count": checkpoint_count,
    }
    write_json(output / "validation_report.json", report)
    return report


def _validate_fold_coverage(
    predictions: pd.DataFrame,
    bundle: TrainingBundle,
    frames: Mapping[int, pd.DataFrame],
    config: Mapping[str, Any],
) -> bool:
    for horizon in (3, 5, 8):
        group = predictions[predictions["horizon"].eq(horizon)]
        folds = config["horizon_folds"][str(horizon)]
        expected_ids = set(
            frames[horizon]
            .loc[
                frames[horizon]["publication_year"].between(
                    min(int(item["test_year_min"]) for item in folds),
                    max(int(item["test_year_max"]) for item in folds),
                ),
                "paper_id",
            ]
            .astype(str)
        )
        for set_name in bundle.set_names:
            subset = group[group["model_id"].eq(set_name)]
            if set(subset["paper_id"].astype(str)) != expected_ids:
                return False
        for item in folds:
            subset = group[group["outer_fold_id"].eq(int(item["fold_id"]))]
            if subset.empty or int(item["train_year_max"]) >= int(
                subset["publication_year"].min()
            ):
                return False
            if int(subset["publication_year"].min()) != int(item["test_year_min"]):
                return False
            if int(subset["publication_year"].max()) != int(item["test_year_max"]):
                return False
    return True


def validate(
    matrix_manifest_path: Path,
    frozen_manifest_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate a real active-dataset canonical OOF run."""
    active = load_active_dataset(PROJECT_ROOT)
    dataset_dir = Path(active["feature_dataset_dir"])
    bundle = load_training_bundle(
        matrix_manifest_path,
        frozen_manifest_path,
        _active_paper_ids(dataset_dir),
    )
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return validate_outputs(
        output_dir, bundle, assemble_frames(bundle, dataset_dir), config
    )
