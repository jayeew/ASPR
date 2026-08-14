"""Verify ASPR v6.1 output integrity and deterministic OOF replay."""

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

from gear.nature_multihorizon.candidate_registry_v6_1 import (
    load_candidate_registry_v6_1,
)
from gear.nature_multihorizon.modeling_v6_1 import (
    assemble_all_period_frame,
    build_v6_1_feature_sets,
    load_simple_config,
    run_fixed_medium_oof,
    run_v6_1_experiment,
)
from gear.nature_multihorizon.source_audit_v6 import sha256_file


DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "nature_multihorizon" / "v6_1_simple.json"
)
SORT_COLUMNS = ["horizon", "model_id", "outer_fold_id", "paper_id"]


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


def _single_oof_manifest(analysis_root: Path) -> Tuple[Path, Dict[str, Any]]:
    matches = sorted(analysis_root.glob("oof_*/oof_run_manifest.json"))
    if len(matches) != 1:
        raise ValueError(
            "reproducibility verification requires exactly one OOF manifest; "
            f"found {len(matches)}"
        )
    path = matches[0].resolve()
    return path, json.loads(path.read_text(encoding="utf-8"))


def _verify_manifest_outputs(manifest: Mapping[str, Any]) -> int:
    verified = 0
    for name, item in manifest["outputs"].items():
        path = Path(item["path"]).resolve()
        if not path.is_file():
            raise ValueError(f"manifest output is missing: {name}: {path}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"manifest output hash mismatch: {name}")
        if path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"manifest output size mismatch: {name}")
        verified += 1
    return verified


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(SORT_COLUMNS).reset_index(drop=True)


def _assert_same_frame(left: pd.DataFrame, right: pd.DataFrame) -> None:
    shared = list(left.columns)
    if set(shared) != set(right.columns):
        raise ValueError("prediction schemas differ")
    pd.testing.assert_frame_equal(
        _sorted(left),
        _sorted(right[shared]),
        check_dtype=True,
        check_exact=True,
    )


def _verify_same_test_sets_and_labels(predictions: pd.DataFrame) -> int:
    groups_verified = 0
    label_columns = [
        "paper_id",
        "future_uptake",
        "conditional_diffusion_member",
        "conditional_diffusion_target",
        "realized_diffusion_target",
    ]
    for (_, _), group in predictions.groupby(
        ["horizon", "outer_fold_id"], sort=True
    ):
        baseline: pd.DataFrame | None = None
        for _, model in group.groupby("model_id", sort=True):
            current = model[label_columns].sort_values("paper_id").reset_index(
                drop=True
            )
            if baseline is None:
                baseline = current
                continue
            if not baseline["paper_id"].equals(current["paper_id"]):
                raise ValueError("models do not share the same test papers")
            for column in label_columns[1:]:
                left = baseline[column].to_numpy()
                right = current[column].to_numpy()
                if not np.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True):
                    raise ValueError(
                        f"models do not share the same label: {column}"
                    )
        groups_verified += 1
    return groups_verified


def _verify_checkpoints(
    run_dir: Path,
    predictions: pd.DataFrame,
) -> Tuple[int, str]:
    paths = sorted((run_dir / "checkpoints").glob("D*/*/fold_*.parquet"))
    expected_cells = {
        (int(horizon), str(model), int(fold))
        for horizon, model, fold in predictions[
            ["horizon", "model_id", "outer_fold_id"]
        ].drop_duplicates().itertuples(index=False, name=None)
    }
    if len(paths) != len(expected_cells):
        raise ValueError(
            f"checkpoint count mismatch: {len(paths)} != {len(expected_cells)}"
        )
    rows = []
    file_records = []
    observed_cells = set()
    for path in paths:
        checkpoint = pd.read_parquet(path)
        if checkpoint["paper_id"].duplicated().any():
            raise ValueError(f"duplicate paper in checkpoint: {path}")
        horizon = int(path.parents[1].name.removeprefix("D"))
        model_id = path.parent.name
        fold_id = int(path.stem.removeprefix("fold_"))
        cell = (horizon, model_id, fold_id)
        if cell in observed_cells:
            raise ValueError(f"duplicate checkpoint cell: {cell}")
        if (
            not checkpoint["horizon"].eq(horizon).all()
            or not checkpoint["model_id"].eq(model_id).all()
            or not checkpoint["outer_fold_id"].eq(fold_id).all()
        ):
            raise ValueError(f"checkpoint path/content mismatch: {path}")
        observed_cells.add(cell)
        rows.append(checkpoint)
        file_records.append(
            {
                "path": str(path.relative_to(run_dir)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if observed_cells != expected_cells:
        raise ValueError("checkpoint cells differ from final predictions")
    combined = pd.concat(rows, ignore_index=True)
    if combined.duplicated(SORT_COLUMNS).any():
        raise ValueError("duplicate OOF prediction across checkpoints")
    _assert_same_frame(predictions, combined)
    return len(paths), _canonical_hash({"files": file_records})


def _model_ids(config: Mapping[str, Any], horizon: int) -> Tuple[str, ...]:
    if int(horizon) == int(config["main_horizon"]):
        values = (
            *config["main_model_ids"],
            *config["sensitivity_model_ids"],
        )
    else:
        values = tuple(config["supplementary_model_ids"])
    return tuple(dict.fromkeys(str(item) for item in values))


def _full_replay(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    original: pd.DataFrame,
    replay_root: Path,
) -> Tuple[bool, int, str]:
    registry = load_candidate_registry_v6_1(
        _resolve(config["paths"]["candidate_registry"])
    )
    feature_sets = build_v6_1_feature_sets(registry, config)
    dataset_dir = _resolve(config["paths"]["v6_1_dataset"])
    preexisting = replay_root.exists()
    replay_rows = []
    for horizon in (
        int(config["main_horizon"]),
        *[int(item) for item in config["supplementary_horizons"]],
    ):
        frame = assemble_all_period_frame(dataset_dir, horizon=horizon)
        predictions, _ = run_fixed_medium_oof(
            frame,
            feature_sets=feature_sets,
            model_ids=_model_ids(config, horizon),
            fold_config=config["temporal_folds"],
            parameters=config["model"],
            categorical_features=config["categorical_features"],
            inner_folds=int(config["model"]["inner_temporal_folds"]),
            horizon=horizon,
            checkpoint_root=replay_root,
            seed=int(config["model"]["seed"]),
        )
        replay_rows.append(predictions)
    replay = pd.concat(replay_rows, ignore_index=True)
    _assert_same_frame(original, replay)
    paths = sorted(replay_root.glob("D*/*/fold_*.parquet"))
    tree = [
        {
            "path": str(path.relative_to(replay_root)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    expected = int(
        original[
            ["horizon", "model_id", "outer_fold_id"]
        ].drop_duplicates().shape[0]
    )
    if len(paths) != expected:
        raise ValueError("full replay did not produce every checkpoint")
    return preexisting, len(paths), _canonical_hash({"files": tree})


def verify(config_path: Path, *, full_replay: bool) -> Path:
    """Verify lineage, outputs, checkpoints, and optional full model replay."""
    config_path = config_path.resolve()
    config = load_simple_config(config_path)
    analysis_root = _resolve(config["paths"]["v6_1_analysis"]).resolve()
    manifest_path, manifest = _single_oof_manifest(analysis_root)
    manifest_hash_before = sha256_file(manifest_path)
    output_count = _verify_manifest_outputs(manifest)
    predictions_path = Path(manifest["outputs"]["predictions"]["path"])
    predictions = pd.read_parquet(predictions_path)
    checkpoint_count, checkpoint_tree_hash = _verify_checkpoints(
        manifest_path.parent, predictions
    )
    test_set_groups = _verify_same_test_sets_and_labels(predictions)

    returned_manifest, returned_dir = run_v6_1_experiment(
        PROJECT_ROOT, config_path
    )
    manifest_hash_after = sha256_file(manifest_path)
    idempotent_match = bool(
        returned_manifest == manifest
        and returned_dir.resolve() == manifest_path.parent.resolve()
        and manifest_hash_after == manifest_hash_before
    )
    if not idempotent_match:
        raise ValueError("OOF runner was not idempotent")

    replay_preexisting = None
    replay_count = 0
    replay_tree_hash = None
    replay_exact_match = None
    if full_replay:
        replay_root = (
            analysis_root
            / f"deterministic_replay_{manifest['run_hash'].split(':')[-1][:12]}"
            / "checkpoints"
        )
        (
            replay_preexisting,
            replay_count,
            replay_tree_hash,
        ) = _full_replay(config, manifest, predictions, replay_root)
        replay_exact_match = True

    dataset_dir = _resolve(config["paths"]["v6_1_dataset"])
    report: Dict[str, Any] = {
        "artifact_kind": "aspr_v6_1_reproducibility_verification",
        "assessment": "pass",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "oof_manifest_path": str(manifest_path),
        "oof_manifest_sha256_before": manifest_hash_before,
        "oof_manifest_sha256_after": manifest_hash_after,
        "oof_artifact_id": manifest["artifact_id"],
        "n_manifest_outputs_verified": output_count,
        "n_checkpoints_verified": checkpoint_count,
        "checkpoint_tree_hash": checkpoint_tree_hash,
        "n_horizon_fold_test_set_groups_verified": test_set_groups,
        "same_test_papers_and_labels_across_models": True,
        "idempotent_manifest_match": idempotent_match,
        "full_replay_requested": bool(full_replay),
        "full_replay_checkpoint_root_preexisting": replay_preexisting,
        "n_full_replay_checkpoints": replay_count,
        "full_replay_exact_prediction_match": replay_exact_match,
        "full_replay_checkpoint_tree_hash": replay_tree_hash,
        "determinism_scope": (
            "Exact-value comparison of every OOF prediction generated from "
            "the same frozen feature views, registry, folds, parameters, and "
            "seed; output hashes and checkpoint partitions are also verified."
            if full_replay
            else "Integrity and idempotent-resume verification only."
        ),
        "frozen_input_hashes": {
            "innovation_candidate_features": sha256_file(
                dataset_dir / "innovation_candidate_features.parquet"
            ),
            "control_features_v6_1": sha256_file(
                dataset_dir / "control_features_v6_1.parquet"
            ),
            "targets_zero_inclusive": sha256_file(
                dataset_dir / "targets_zero_inclusive.parquet"
            ),
            "cohort_membership": sha256_file(
                dataset_dir / "cohort_membership.parquet"
            ),
        },
    }
    report["artifact_id"] = _canonical_hash(report)
    output = analysis_root / "reproducibility_report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--full-replay",
        action="store_true",
        help="refit all 66 fold-model cells and compare every prediction",
    )
    args = parser.parse_args(argv)
    print(verify(args.config, full_replay=bool(args.full_replay)))


if __name__ == "__main__":
    main()
