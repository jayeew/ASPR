"""Run fixed-medium D5 temporal OOF for four evidence-v3 feature sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.nature_multihorizon.modeling_v6 import safe_spearman
from aspr.nature_multihorizon.modeling_v6_1 import (
    assemble_all_period_frame,
    evaluate_oof_points,
    run_fixed_medium_oof,
)


DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "knowledge_corpus"
    / "nature_multihorizon_v6_1_local"
)
DEFAULT_OUTPUT = HERE / "outputs"
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "nature_multihorizon" / "v6_1_simple.json"
)
MODEL_IDS = (
    "strict_7",
    "fulltext_16",
    "source_154",
    "ultrarelaxed_221",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def load_inputs(
    output_dir: Path,
) -> tuple[pd.DataFrame, Dict[str, Sequence[str]], Mapping[str, Any]]:
    matrix_path = output_dir / "indicator_matrix_221.parquet"
    sets_path = output_dir / "feature_sets.json"
    manifest_path = output_dir / "matrix_manifest.json"
    for path in (matrix_path, sets_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    matrix_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = matrix_manifest["outputs"]["matrix"]["sha256"]
    if sha256_file(matrix_path) != expected_hash:
        raise ValueError("indicator matrix changed after materialization")
    sets_payload = json.loads(sets_path.read_text(encoding="utf-8"))
    feature_sets = {
        model_id: tuple(sets_payload["sets"][model_id]["feature_ids"])
        for model_id in MODEL_IDS
    }
    matrix = pd.read_parquet(matrix_path)
    return matrix, feature_sets, matrix_manifest


def paired_year_block_bootstrap(
    predictions: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Compare Spearman differences by resampling publication years."""
    truth = predictions[
        ["paper_id", "publication_year", "model_id", "realized_diffusion_target"]
    ].pivot(
        index=["paper_id", "publication_year"],
        columns="model_id",
        values="realized_diffusion_target",
    )
    scores = predictions.pivot(
        index=["paper_id", "publication_year"],
        columns="model_id",
        values="expected_diffusion_score",
    )
    labels = truth.bfill(axis=1).iloc[:, 0]
    if truth.max(axis=1).sub(truth.min(axis=1)).fillna(0).abs().gt(1e-12).any():
        raise ValueError("models do not share identical D5 labels")
    years = np.asarray(sorted(scores.index.get_level_values(1).unique()))
    rng = np.random.default_rng(int(seed))
    estimates: Dict[tuple[str, str], list[float]] = {}
    pairs = [
        (left, right)
        for left_index, left in enumerate(MODEL_IDS)
        for right in MODEL_IDS[left_index + 1 :]
    ]
    for pair in pairs:
        estimates[pair] = []
    label_values = labels.to_numpy(dtype=float)
    score_values = {
        model_id: scores[model_id].to_numpy(dtype=float)
        for model_id in MODEL_IDS
    }
    valid = np.isfinite(label_values)
    for values in score_values.values():
        valid &= np.isfinite(values)
    label_values = label_values[valid]
    score_values = {
        model_id: values[valid] for model_id, values in score_values.items()
    }
    row_years = scores.index.get_level_values(1).to_numpy()[valid]
    year_lookup = {int(year): index for index, year in enumerate(years)}
    row_year_codes = np.asarray(
        [year_lookup[int(year)] for year in row_years], dtype=np.int64
    )
    rank_groups = {
        "truth": _rank_group_ids(label_values),
        **{
            model_id: _rank_group_ids(values)
            for model_id, values in score_values.items()
        },
    }
    for _ in range(int(iterations)):
        sampled_year_codes = rng.integers(
            0, len(years), size=len(years)
        )
        year_counts = np.bincount(
            sampled_year_codes, minlength=len(years)
        ).astype(float)
        row_counts = year_counts[row_year_codes]
        correlations = {
            model_id: _weighted_rank_correlation(
                rank_groups["truth"],
                rank_groups[model_id],
                row_counts,
            )
            for model_id in MODEL_IDS
        }
        for left, right in pairs:
            estimates[(left, right)].append(
                correlations[right] - correlations[left]
            )
    rows = []
    point = {
        model_id: safe_spearman(label_values, values)
        for model_id, values in score_values.items()
    }
    for left, right in pairs:
        values = np.asarray(estimates[(left, right)], dtype=float)
        rows.append(
            {
                "left_model_id": left,
                "right_model_id": right,
                "spearman_left": point[left],
                "spearman_right": point[right],
                "right_minus_left": point[right] - point[left],
                "ci_low": float(np.nanquantile(values, 0.025)),
                "ci_high": float(np.nanquantile(values, 0.975)),
                "bootstrap_unit": "publication_year",
                "bootstrap_iterations": int(iterations),
            }
        )
    return pd.DataFrame(rows)


def _rank_group_ids(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    changes = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    sorted_groups = np.cumsum(changes) - 1
    groups = np.empty(len(values), dtype=np.int64)
    groups[order] = sorted_groups
    return groups


def _weighted_rank_correlation(
    left_groups: np.ndarray,
    right_groups: np.ndarray,
    counts: np.ndarray,
) -> float:
    left = _weighted_midranks(left_groups, counts)
    right = _weighted_midranks(right_groups, counts)
    total = float(counts.sum())
    left_centered = left - float(np.dot(counts, left) / total)
    right_centered = right - float(np.dot(counts, right) / total)
    covariance = float(np.dot(counts, left_centered * right_centered))
    denominator = np.sqrt(
        float(np.dot(counts, left_centered**2))
        * float(np.dot(counts, right_centered**2))
    )
    return covariance / denominator if denominator > 0 else np.nan


def _weighted_midranks(
    groups: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    weights = np.bincount(groups, weights=counts)
    cumulative = np.cumsum(weights)
    midranks = cumulative - (weights - 1.0) / 2.0
    return midranks[groups]


def run(output_dir: Path) -> Mapping[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix, feature_sets, matrix_manifest = load_inputs(output_dir)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frame = assemble_all_period_frame(DATA_ROOT, horizon=5)
    cohort_rows = len(frame)
    if not set(frame["paper_id"]).issubset(set(matrix["paper_id"])):
        raise ValueError("D5 cohort contains papers absent from the indicator matrix")
    frame = frame.merge(
        matrix, on="paper_id", how="left", validate="one_to_one"
    )
    if len(frame) != cohort_rows:
        raise ValueError("D5 frame and indicator matrix do not align")
    predictions, folds = run_fixed_medium_oof(
        frame,
        feature_sets=feature_sets,
        model_ids=MODEL_IDS,
        fold_config=config["temporal_folds"],
        parameters=config["model"],
        categorical_features=("EF0197",),
        inner_folds=int(config["model"]["inner_temporal_folds"]),
        horizon=5,
        checkpoint_root=output_dir / "checkpoints",
        seed=int(config["model"]["seed"]),
    )
    metrics, fold_metrics, domain_metrics = evaluate_oof_points(
        predictions,
        minimum_domain_rows=int(
            config["evaluation"]["minimum_domain_rows"]
        ),
    )
    comparisons = paired_year_block_bootstrap(
        predictions,
        iterations=2_000,
        seed=int(config["evaluation"]["bootstrap_seed"]),
    )
    metrics = metrics.sort_values(
        "spearman_expected", ascending=False, kind="stable"
    ).reset_index(drop=True)
    metrics.insert(0, "rank", np.arange(1, len(metrics) + 1))
    outputs = {
        "predictions": output_dir / "oof_predictions.parquet",
        "folds": output_dir / "temporal_folds.csv",
        "metrics": output_dir / "oof_metrics.csv",
        "fold_metrics": output_dir / "oof_fold_metrics.csv",
        "domain_metrics": output_dir / "oof_domain_metrics.csv",
        "pairwise_comparisons": output_dir
        / "paired_year_bootstrap_comparisons.csv",
    }
    predictions.to_parquet(outputs["predictions"], index=False)
    folds.to_csv(outputs["folds"], index=False)
    metrics.to_csv(outputs["metrics"], index=False)
    fold_metrics.to_csv(outputs["fold_metrics"], index=False)
    domain_metrics.to_csv(outputs["domain_metrics"], index=False)
    comparisons.to_csv(outputs["pairwise_comparisons"], index=False)
    manifest = {
        "artifact_kind": "evidence_v3_four_set_fixed_medium_d5_oof",
        "headline_metric": "D5 all-period temporal OOF Spearman",
        "model_ids": list(MODEL_IDS),
        "same_papers_labels_folds_parameters": True,
        "feature_selection_from_oof": False,
        "parameter_selection_from_oof": False,
        "fixed_parameter_id": config["model"]["parameter_id"],
        "fixed_seed": config["model"]["seed"],
        "categorical_features": ["EF0197"],
        "matrix_manifest_sha256": sha256_file(
            output_dir / "matrix_manifest.json"
        ),
        "matrix_manifest": matrix_manifest,
        "best_model_id": str(metrics.iloc[0]["model_id"]),
        "best_oof_spearman": float(metrics.iloc[0]["spearman_expected"]),
        "outputs": {
            key: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for key, path in outputs.items()
        },
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_json(output_dir / "oof_run_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args().output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
