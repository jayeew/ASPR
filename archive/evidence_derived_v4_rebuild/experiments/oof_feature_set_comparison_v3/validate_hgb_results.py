"""Validate the final HGB-only ASPR OOF and production-score artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
DEFAULT_OUTPUT = HERE / "outputs/hgb_uncapped_v2"
CONFIG_PATH = PROJECT_ROOT / "configs/nature_multihorizon/hgb_uncapped_v2.json"
MODEL_IDS = {"strict_7", "fulltext_16", "source_154", "ultrarelaxed_221"}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable human-readable JSON."""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate(output_dir: Path, config_path: Path) -> Mapping[str, Any]:
    """Run structural, temporal, alignment, and score-definition checks."""
    config = json.loads(config_path.read_text("utf-8"))
    predictions = pd.read_parquet(output_dir / "oof_predictions.parquet")
    metrics = pd.read_csv(output_dir / "oof_metrics.csv")
    scores = pd.read_parquet(output_dir / "official_aspr_scores.parquet")
    checks: dict[str, bool] = {}
    checks["only_hgb_predictions"] = set(predictions["model_family"]) == {"hgb"}
    checks["only_hgb_metrics"] = set(metrics["model_family"]) == {"hgb"}
    checks["twelve_results"] = (
        len(metrics) == 12
        and set(metrics["horizon"]) == {3, 5, 8}
        and set(metrics["model_id"]) == MODEL_IDS
    )
    keys = ["paper_id", "horizon", "model_id", "outer_fold_id"]
    checks["unique_oof_keys"] = not predictions.duplicated(keys).any()
    aligned = True
    compare = [
        "paper_id",
        "publication_year",
        "outer_fold_id",
        "realized_diffusion_target",
    ]
    for _, group in predictions.groupby("horizon"):
        baseline = (
            group[group["model_id"].eq("strict_7")][compare]
            .sort_values("paper_id")
            .reset_index(drop=True)
        )
        for model_id in MODEL_IDS - {"strict_7"}:
            candidate = (
                group[group["model_id"].eq(model_id)][compare]
                .sort_values("paper_id")
                .reset_index(drop=True)
            )
            aligned &= baseline.equals(candidate)
    checks["sets_share_papers_folds_labels"] = bool(aligned)
    fold_lookup = {
        (int(horizon), int(fold["fold_id"])): fold
        for horizon, folds in config["horizon_folds"].items()
        for fold in folds
    }
    temporal = True
    for (horizon, fold_id), group in predictions.groupby(["horizon", "outer_fold_id"]):
        fold = fold_lookup[(int(horizon), int(fold_id))]
        temporal &= int(fold["train_year_max"]) < int(group["publication_year"].min())
        temporal &= int(group["publication_year"].min()) == int(fold["test_year_min"])
        temporal &= int(group["publication_year"].max()) == int(fold["test_year_max"])
    checks["strict_forward_temporal_folds"] = bool(temporal)
    checks["official_score_fields"] = {
        "paper_id",
        "raw_prediction_score",
        "aspr_score",
    }.issubset(scores.columns)
    checks["official_identity"] = set(scores["official_model_family"]) == {
        "hgb"
    } and set(scores["official_feature_set"]) == {"fulltext_16"}
    ordered = scores.sort_values("raw_prediction_score")
    checks["score_is_monotone_percentile"] = bool(
        ordered["aspr_score"].diff().fillna(0).ge(-1e-12).all()
        and np.isfinite(scores["aspr_score"]).all()
        and scores["aspr_score"].between(0, 100).all()
    )
    checks["production_model_present"] = (
        output_dir / "official_hgb_model.joblib"
    ).is_file()
    result = {
        "artifact_kind": "nature-multihorizon-uncapped-v2-final-hgb-validation",
        "passed": all(checks.values()),
        "checks": checks,
        "prediction_rows": len(predictions),
        "score_rows": len(scores),
    }
    write_json(output_dir / "validation_report.json", result)
    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    report = validate(arguments.output_dir.resolve(), arguments.config.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
