"""Train and gate the journal-free ASPR-GEAR submission calibration model.

The script never promotes a model merely because training completed.  It
writes ``promotion_status=passed`` only when the preregistered overall,
late-fold, and top-decile gates all pass.  Runtime code must check that field
before loading any produced estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.nature_multihorizon.evaluation import top_decile_enrichment  # noqa: E402
from gear.nature_multihorizon.modeling_v6 import (  # noqa: E402
    _fit_calibrators,
    _fit_two_part,
    _inner_oof_for_parameters,
    safe_spearman,
)
from gear.nature_multihorizon.modeling_v6_1 import run_fixed_medium_oof  # noqa: E402
from innovation_impact_feature_selection.evidence_derived_v3.experiments.oof_feature_set_comparison_v3 import (  # noqa: E402
    run_hgb_comparison,
)

FEATURE_SET_ID = "submission_safe_15"
FORBIDDEN_FEATURES = {"EF0197"}
DEFAULT_CONFIG = PROJECT_ROOT / "configs/nature_multihorizon/hgb_uncapped_v2.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/gear/submission_calibration"
OFFICIAL_OUTPUT = (
    PROJECT_ROOT
    / "innovation_impact_feature_selection/evidence_derived_v3/experiments/"
    "oof_feature_set_comparison_v3/outputs/hgb_uncapped_v2"
)


def sha256_file(path: Path) -> str:
    """Return a prefixed SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a deterministic UTF-8 JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def submission_feature_names() -> tuple[str, ...]:
    """Return the frozen Full-text 16 set without journal identity."""
    fulltext = run_hgb_comparison.load_feature_sets()["fulltext_16"]
    features = tuple(name for name in fulltext if name not in FORBIDDEN_FEATURES)
    if len(features) != 15 or set(features) & FORBIDDEN_FEATURES:
        raise ValueError(
            "submission-safe feature contract must contain exactly 15 fields"
        )
    return features


def _metric_rows(predictions: pd.DataFrame) -> tuple[float, Dict[int, float]]:
    overall = safe_spearman(
        predictions["realized_diffusion_target"],
        predictions["expected_diffusion_score"],
    )
    folds = {
        int(fold_id): safe_spearman(
            group["realized_diffusion_target"],
            group["expected_diffusion_score"],
        )
        for fold_id, group in predictions.groupby("outer_fold_id", sort=True)
    }
    return float(overall), folds


def _official_metrics() -> tuple[float, Dict[int, float]]:
    overall = pd.read_csv(OFFICIAL_OUTPUT / "oof_metrics.csv")
    folds = pd.read_csv(OFFICIAL_OUTPUT / "oof_fold_metrics.csv")
    selector = (
        overall["horizon"].eq(5)
        & overall["model_family"].eq("hgb")
        & overall["model_id"].eq("fulltext_16")
    )
    fold_selector = (
        folds["horizon"].eq(5)
        & folds["model_family"].eq("hgb")
        & folds["model_id"].eq("fulltext_16")
    )
    if selector.sum() != 1 or not fold_selector.any():
        raise ValueError("official Full-text 16 OOF metrics are incomplete")
    return (
        float(overall.loc[selector, "spearman"].iloc[0]),
        {
            int(row.outer_fold_id): float(row.spearman)
            for row in folds.loc[fold_selector].itertuples(index=False)
        },
    )


def evaluate_gates(predictions: pd.DataFrame) -> Dict[str, Any]:
    """Apply the preregistered model-promotion gates."""
    observed, observed_folds = _metric_rows(predictions)
    official, official_folds = _official_metrics()
    late_fold_ids = sorted(set(observed_folds) & set(official_folds))[-2:]
    late_checks = {
        str(fold_id): {
            "observed": observed_folds[fold_id],
            "official": official_folds[fold_id],
            "minimum": official_folds[fold_id] - 0.03,
            "passed": observed_folds[fold_id] >= official_folds[fold_id] - 0.03,
        }
        for fold_id in late_fold_ids
    }
    enrichment = float(
        top_decile_enrichment(
            predictions["expected_diffusion_score"],
            predictions["realized_diffusion_target"],
        )
    )
    checks: Dict[str, Dict[str, Any]] = {
        "overall_spearman": {
            "observed": observed,
            "official": official,
            "minimum": official - 0.02,
            "passed": observed >= official - 0.02,
        },
        "late_temporal_folds": {
            "folds": late_checks,
            "passed": len(late_checks) == 2
            and all(item["passed"] for item in late_checks.values()),
        },
        "top_decile_enrichment": {
            "observed": enrichment,
            "minimum": 3.5,
            "passed": enrichment >= 3.5,
        },
    }
    return {
        "checks": checks,
        "passed": all(item["passed"] for item in checks.values()),
    }


def fit_promoted_model(
    frame: pd.DataFrame,
    all_papers: pd.DataFrame,
    features: Sequence[str],
    parameters: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Fit the production bundle using only the journal-free fields."""
    seed = int(parameters["seed"])
    inner = _inner_oof_for_parameters(
        frame,
        feature_names=features,
        categorical_names=(),
        parameters=parameters,
        n_inner=int(parameters["inner_temporal_folds"]),
        seed=seed,
    )
    uptake_calibrator, conditional_calibrator, _, _ = _fit_calibrators(inner)
    model = _fit_two_part(
        frame,
        feature_names=features,
        categorical_names=(),
        parameters=parameters,
        seed=seed,
    )
    uptake_raw, conditional_raw = model.predict_raw(all_papers)
    raw = uptake_calibrator.predict(uptake_raw) * conditional_calibrator.predict(
        conditional_raw
    )
    mature_ids = set(frame["paper_id"].astype(str))
    mature = all_papers["paper_id"].astype(str).isin(mature_ids).to_numpy()
    reference = np.sort(raw[mature & np.isfinite(raw)])
    if not len(reference):
        raise ValueError("submission model has no finite mature reference predictions")
    model_path = output_dir / "submission_hgb_model.joblib"
    reference_path = output_dir / "submission_score_reference.npy"
    joblib.dump(
        {
            "model": model,
            "uptake_calibrator": uptake_calibrator,
            "conditional_calibrator": conditional_calibrator,
            "feature_names": tuple(features),
        },
        model_path,
    )
    np.save(reference_path, reference, allow_pickle=False)
    return {
        "model_path": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "reference_path": str(reference_path.resolve()),
        "reference_sha256": sha256_file(reference_path),
        "reference_n": len(reference),
    }


def run(output_dir: Path, config_path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Run D5 OOF, gate promotion, and fit only when every gate passes."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    features = submission_feature_names()
    frame = run_hgb_comparison.load_horizon_frame(5)
    predictions, folds = run_fixed_medium_oof(
        frame,
        feature_sets={FEATURE_SET_ID: features},
        model_ids=(FEATURE_SET_ID,),
        fold_config=config["horizon_folds"]["5"],
        parameters=config["hgb"],
        categorical_features=(),
        inner_folds=int(config["hgb"]["inner_temporal_folds"]),
        horizon=5,
        checkpoint_root=output_dir / "checkpoints",
        seed=int(config["hgb"]["seed"]),
    )
    predictions_path = output_dir / "submission_oof_predictions.parquet"
    fold_path = output_dir / "submission_oof_folds.csv"
    predictions.to_parquet(predictions_path, index=False)
    folds.to_csv(fold_path, index=False)
    gates = evaluate_gates(predictions)
    production: Dict[str, Any] = {}
    if gates["passed"]:
        all_papers = pd.read_parquet(
            run_hgb_comparison.MATRIX_ROOT / "indicator_matrix_221.parquet"
        )
        production = fit_promoted_model(
            frame,
            all_papers,
            features,
            config["hgb"],
            output_dir,
        )
    manifest: Dict[str, Any] = {
        "contract": "aspr_submission_calibration_training_v1",
        "promotion_status": "passed" if gates["passed"] else "failed_profile_only",
        "horizon": 5,
        "feature_set": FEATURE_SET_ID,
        "features": list(features),
        "forbidden_features": sorted(FORBIDDEN_FEATURES),
        "categorical_features": [],
        "gates": gates,
        "production": production,
        "inputs": {
            "config_path": str(Path(config_path).resolve()),
            "config_sha256": sha256_file(Path(config_path)),
            "matrix_manifest": str(
                (run_hgb_comparison.MATRIX_ROOT / "matrix_manifest.json").resolve()
            ),
            "matrix_manifest_sha256": sha256_file(
                run_hgb_comparison.MATRIX_ROOT / "matrix_manifest.json"
            ),
        },
        "outputs": {
            "predictions": {
                "path": str(predictions_path.resolve()),
                "sha256": sha256_file(predictions_path),
            },
            "folds": {
                "path": str(fold_path.resolve()),
                "sha256": sha256_file(fold_path),
            },
        },
    }
    write_json(output_dir / "submission_model_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments.output_dir, arguments.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
