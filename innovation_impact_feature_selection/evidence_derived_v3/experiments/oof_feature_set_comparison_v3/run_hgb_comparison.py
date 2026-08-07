"""Train the final HGB ASPR model across all horizons and feature sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.nature_multihorizon.active_dataset import load_active_dataset
from aspr.nature_multihorizon.modeling_v6 import (
    _fit_calibrators,
    _fit_two_part,
    _inner_oof_for_parameters,
    safe_spearman,
)
from aspr.nature_multihorizon.modeling_v6_1 import (
    assemble_all_period_frame,
    run_fixed_medium_oof,
)

ACTIVE_DATASET = load_active_dataset(PROJECT_ROOT)
DATA_ROOT = Path(ACTIVE_DATASET["feature_dataset_dir"])
MATRIX_ROOT = Path(ACTIVE_DATASET["indicator_matrix_dir"])
CONFIG_PATH = PROJECT_ROOT / "configs/nature_multihorizon/hgb_uncapped_v2.json"
DEFAULT_OUTPUT = HERE / "outputs/hgb_uncapped_v2"
MODEL_IDS = ("strict_7", "fulltext_16", "source_154", "ultrarelaxed_221")


def sha256_file(path: Path) -> str:
    """Return a prefixed SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable human-readable JSON."""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_feature_sets() -> dict[str, tuple[str, ...]]:
    """Load the four frozen feature sets."""
    payload = json.loads((MATRIX_ROOT / "feature_sets.json").read_text("utf-8"))
    return {
        model_id: tuple(payload["sets"][model_id]["feature_ids"])
        for model_id in MODEL_IDS
    }


def load_horizon_frame(horizon: int) -> pd.DataFrame:
    """Join one mature outcome cohort to the common T0 indicator matrix."""
    frame = assemble_all_period_frame(DATA_ROOT, horizon=horizon)
    matrix = pd.read_parquet(MATRIX_ROOT / "indicator_matrix_221.parquet")
    rows = len(frame)
    frame = frame.merge(matrix, on="paper_id", how="left", validate="one_to_one")
    if len(frame) != rows:
        raise ValueError("feature merge changed the horizon cohort")
    return frame


def evaluate_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute overall, temporal-fold, and domain OOF Spearman metrics."""
    keys = ["horizon", "model_family", "model_id"]
    overall: list[Mapping[str, Any]] = []
    folds: list[Mapping[str, Any]] = []
    domains: list[Mapping[str, Any]] = []
    for key, group in predictions.groupby(keys, sort=False):
        base = dict(zip(keys, key))
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


def fit_official_model(
    frame: pd.DataFrame,
    all_papers: pd.DataFrame,
    feature_names: Sequence[str],
    config: Mapping[str, Any],
    output_dir: Path,
) -> Mapping[str, Any]:
    """Fit D5 Full-text 16 HGB and generate the two official score fields."""
    parameters = config["hgb"]
    categorical = tuple(name for name in ("EF0197",) if name in feature_names)
    inner = _inner_oof_for_parameters(
        frame,
        feature_names=feature_names,
        categorical_names=categorical,
        parameters=parameters,
        n_inner=int(parameters["inner_temporal_folds"]),
        seed=int(parameters["seed"]),
    )
    uptake_calibrator, conditional_calibrator, _, _ = _fit_calibrators(inner)
    model = _fit_two_part(
        frame,
        feature_names=feature_names,
        categorical_names=categorical,
        parameters=parameters,
        seed=int(parameters["seed"]),
    )
    uptake_raw, conditional_raw = model.predict_raw(all_papers)
    raw = uptake_calibrator.predict(uptake_raw) * conditional_calibrator.predict(
        conditional_raw
    )
    model_path = output_dir / "official_hgb_model.joblib"
    joblib.dump(
        {
            "model": model,
            "uptake_calibrator": uptake_calibrator,
            "conditional_calibrator": conditional_calibrator,
        },
        model_path,
    )
    mature_ids = set(frame["paper_id"].astype(str))
    mature = all_papers["paper_id"].astype(str).isin(mature_ids).to_numpy()
    reference = np.sort(raw[mature & np.isfinite(raw)])
    score = 100.0 * np.searchsorted(reference, raw, side="right") / len(reference)
    score_path = output_dir / "official_aspr_scores.parquet"
    pd.DataFrame(
        {
            "paper_id": all_papers["paper_id"].astype(str),
            "official_model_family": "hgb",
            "official_feature_set": "fulltext_16",
            "raw_prediction_score": raw,
            "aspr_score": score,
        }
    ).to_parquet(score_path, index=False)
    return {
        "model_path": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "score_path": str(score_path.resolve()),
        "score_sha256": sha256_file(score_path),
        "score_definition": "100 times the empirical CDF of the raw D5 HGB prediction in the mature D5 cohort",
    }


def run(output_dir: Path, config_path: Path = CONFIG_PATH) -> Mapping[str, Any]:
    """Train all HGB OOF models and materialize the official ASPR score."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text("utf-8"))
    if config["dataset_version"] != ACTIVE_DATASET["active_dataset_version"]:
        raise ValueError("model config does not match the active dataset")
    feature_sets = load_feature_sets()
    predictions: list[pd.DataFrame] = []
    frames: dict[int, pd.DataFrame] = {}
    for horizon in (3, 5, 8):
        frame = load_horizon_frame(horizon)
        frames[horizon] = frame
        result, _ = run_fixed_medium_oof(
            frame,
            feature_sets=feature_sets,
            model_ids=MODEL_IDS,
            fold_config=config["horizon_folds"][str(horizon)],
            parameters=config["hgb"],
            categorical_features=("EF0197",),
            inner_folds=int(config["hgb"]["inner_temporal_folds"]),
            horizon=horizon,
            checkpoint_root=output_dir / "checkpoints/hgb",
            seed=int(config["hgb"]["seed"]),
        )
        result["model_family"] = "hgb"
        predictions.append(result)
    combined = pd.concat(predictions, ignore_index=True)
    metrics, fold_metrics, domain_metrics = evaluate_predictions(combined)
    outputs = {
        "oof_predictions": output_dir / "oof_predictions.parquet",
        "oof_metrics": output_dir / "oof_metrics.csv",
        "fold_metrics": output_dir / "oof_fold_metrics.csv",
        "domain_metrics": output_dir / "oof_domain_metrics.csv",
    }
    combined.to_parquet(outputs["oof_predictions"], index=False)
    metrics.to_csv(outputs["oof_metrics"], index=False)
    fold_metrics.to_csv(outputs["fold_metrics"], index=False)
    domain_metrics.to_csv(outputs["domain_metrics"], index=False)
    all_papers = pd.read_parquet(MATRIX_ROOT / "indicator_matrix_221.parquet")
    production = fit_official_model(
        frames[5], all_papers, feature_sets["fulltext_16"], config, output_dir
    )
    official = {
        "horizon": 5,
        "feature_set": "fulltext_16",
        "model_family": "hgb",
        "selection_basis": "predeclared five-year target and best D5 OOF performance among the four frozen feature sets",
        "production": production,
    }
    write_json(output_dir / "official_model.json", official)
    manifest = {
        "artifact_kind": "nature-multihorizon-uncapped-v2-final-hgb-oof",
        "dataset_version": config["dataset_version"],
        "horizons": [3, 5, 8],
        "feature_sets": {name: len(values) for name, values in feature_sets.items()},
        "official_model": official,
        "network_used": False,
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in outputs.items()
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments.output_dir.resolve(), arguments.config.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
