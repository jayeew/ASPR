"""Run the supplemental D5 OOF model using only frozen innovation metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.nature_multihorizon.candidate_registry_v6_1 import (
    candidate_registry_sha256,
    load_candidate_registry_v6_1,
)
from aspr.nature_multihorizon.modeling_v6 import safe_spearman
from aspr.nature_multihorizon.modeling_v6_1 import (
    assemble_all_period_frame,
    freeze_registry_before_oof,
    load_simple_config,
    run_fixed_medium_oof,
)
from aspr.nature_multihorizon.source_audit_v6 import sha256_file


DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "nature_multihorizon" / "v6_1_simple.json"
)
MODEL_ID = "innovation_only"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _single_main_manifest(analysis_root: Path) -> Tuple[Path, Dict[str, Any]]:
    matches = sorted(analysis_root.glob("oof_*/oof_run_manifest.json"))
    if len(matches) != 1:
        raise ValueError(
            "innovation-only supplement requires exactly one main OOF manifest"
        )
    path = matches[0].resolve()
    return path, json.loads(path.read_text(encoding="utf-8"))


def _validate_same_test_labels(
    supplement: pd.DataFrame,
    main_manifest: Mapping[str, Any],
) -> None:
    main = pd.read_parquet(
        main_manifest["outputs"]["predictions"]["path"],
        filters=[
            ("horizon", "=", 5),
            ("model_id", "=", "final_innovation_plus_k1"),
        ],
    )
    columns = [
        "paper_id",
        "future_uptake",
        "conditional_diffusion_member",
        "conditional_diffusion_target",
        "realized_diffusion_target",
    ]
    left = supplement[columns].sort_values("paper_id").reset_index(drop=True)
    right = main[columns].sort_values("paper_id").reset_index(drop=True)
    if not left["paper_id"].equals(right["paper_id"]):
        raise ValueError("innovation-only model uses different OOF papers")
    for column in columns[1:]:
        if not np.allclose(
            left[column].to_numpy(),
            right[column].to_numpy(),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            raise ValueError(
                f"innovation-only model uses a different label: {column}"
            )


def _metrics(predictions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame(
        [
            {
                "horizon": 5,
                "model_id": MODEL_ID,
                "n_oof": len(predictions),
                "n_rank_valid": int(
                    predictions[
                        [
                            "realized_diffusion_target",
                            "expected_diffusion_score",
                        ]
                    ]
                    .dropna()
                    .shape[0]
                ),
                "spearman_expected": safe_spearman(
                    predictions["realized_diffusion_target"],
                    predictions["expected_diffusion_score"],
                ),
            }
        ]
    )
    fold_rows = []
    for fold_id, fold in predictions.groupby("outer_fold_id", sort=True):
        fold_rows.append(
            {
                "horizon": 5,
                "model_id": MODEL_ID,
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
    return overall, pd.DataFrame(fold_rows)


def run(config_path: Path) -> Tuple[Mapping[str, Any], Path]:
    """Run or resume the frozen innovation-only D5 supplement."""
    config_path = config_path.resolve()
    config = load_simple_config(config_path)
    freeze = freeze_registry_before_oof(PROJECT_ROOT, config_path)
    registry_path = _resolve(config["paths"]["candidate_registry"])
    registry = load_candidate_registry_v6_1(registry_path)
    feature_names = registry.primary_feature_names
    controls = set(config["k1_controls"]) | set(
        config["k2_additional_controls"]
    )
    if set(feature_names) & controls:
        raise ValueError("innovation-only features overlap registered controls")
    dataset_root = _resolve(config["paths"]["v6_1_dataset"])
    analysis_root = _resolve(config["paths"]["v6_1_analysis"])
    main_manifest_path, main_manifest = _single_main_manifest(analysis_root)
    lineage = {
        "artifact_kind": "aspr_v6_1_innovation_only_d5_supplement",
        "config_sha256": sha256_file(config_path),
        "registry_file_sha256": sha256_file(registry_path),
        "registry_canonical_sha256": candidate_registry_sha256(registry),
        "registry_freeze_artifact_id": freeze["artifact_id"],
        "innovation_features_sha256": sha256_file(
            dataset_root / "innovation_candidate_features.parquet"
        ),
        "targets_sha256": sha256_file(
            dataset_root / "targets_zero_inclusive.parquet"
        ),
        "cohort_sha256": sha256_file(
            dataset_root / "cohort_membership.parquet"
        ),
        "main_oof_artifact_id": main_manifest["artifact_id"],
        "main_oof_manifest_sha256": sha256_file(main_manifest_path),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "feature_names": list(feature_names),
        "model_parameter_id": config["model"]["parameter_id"],
        "seed": int(config["model"]["seed"]),
    }
    run_hash = _canonical_hash(lineage)
    output = analysis_root / (
        "supplement_innovation_only_"
        + run_hash.removeprefix("sha256:")[:12]
    )
    manifest_path = output / "innovation_only_manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8")), output

    output.mkdir(parents=True, exist_ok=True)
    frame = assemble_all_period_frame(dataset_root, horizon=5)
    predictions, folds = run_fixed_medium_oof(
        frame,
        feature_sets={MODEL_ID: feature_names},
        model_ids=(MODEL_ID,),
        fold_config=config["temporal_folds"],
        parameters=config["model"],
        categorical_features=config["categorical_features"],
        inner_folds=int(config["model"]["inner_temporal_folds"]),
        horizon=5,
        checkpoint_root=output / "checkpoints",
        seed=int(config["model"]["seed"]),
    )
    _validate_same_test_labels(predictions, main_manifest)
    metrics, fold_metrics = _metrics(predictions)
    paths = {
        "predictions": output / "innovation_only_oof_predictions.parquet",
        "metrics": output / "innovation_only_oof_metrics.csv",
        "fold_metrics": output / "innovation_only_fold_metrics.csv",
        "folds": output / "innovation_only_temporal_folds.csv",
    }
    predictions.to_parquet(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    fold_metrics.to_csv(paths["fold_metrics"], index=False)
    folds.to_csv(paths["folds"], index=False)
    manifest: Dict[str, Any] = {
        "artifact_kind": "aspr_v6_1_innovation_only_d5_supplement",
        "lineage": lineage,
        "run_hash": run_hash,
        "horizon": 5,
        "model_id": MODEL_ID,
        "feature_names": list(feature_names),
        "n_features": len(feature_names),
        "controls_included": [],
        "same_oof_papers_and_labels_as_main": True,
        "spearman_expected": float(metrics.loc[0, "spearman_expected"]),
        "n_oof": int(metrics.loc[0, "n_oof"]),
        "n_rank_valid": int(metrics.loc[0, "n_rank_valid"]),
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
    }
    manifest["artifact_id"] = _canonical_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest, output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    manifest, output = run(args.config)
    print(
        json.dumps(
            {"manifest": manifest, "output_dir": str(output)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
