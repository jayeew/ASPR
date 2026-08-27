#!/usr/bin/env python3
"""Fit and freeze the final D5 Primary16 model from the tuned OOF protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from gear.nature_multihorizon.active_dataset import load_active_dataset
from gear.nature_multihorizon.modeling_v6 import (
    _fit_calibrators,
    _fit_two_part,
    _inner_oof_for_parameters,
)
from gear.nature_multihorizon.modeling_v6_1 import assemble_all_period_frame

from .release_registry import load_current_release

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
DEFAULT_OUTPUT = ROOT / "production_releases/d5_primary16_tuned_20260827"


def sha256_file(path: Path) -> str:
    """Return an unprefixed SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON."""
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _selected_parameters(release: Path) -> dict[str, Any]:
    """Choose the modal D5 inner-selected configuration without outer labels."""
    selected = pd.read_csv(release / "selected_parameters_by_outer_fold.csv")
    d5 = selected[selected["horizon"].eq(5) & selected["selected"].astype(bool)]
    counts = d5["parameter_id"].value_counts()
    parameter_id = sorted(counts[counts.eq(counts.max())].index.astype(str))[0]
    search = json.loads((release / "search_space.json").read_text(encoding="utf-8"))
    parameters = {
        str(item["parameter_id"]): dict(item) for item in search["parameters"]
    }[parameter_id]
    parameters["seed"] = 20260806
    parameters["inner_temporal_folds"] = int(search["inner_temporal_folds"])
    return parameters


def _primary_matrix(release: Path) -> tuple[pd.DataFrame, tuple[str, ...], Path]:
    manifest_path = release / "frozen_input_matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    definition = manifest["sets"]["primary"]
    path = Path(definition["path"])
    if sha256_file(path) != str(definition["sha256"]):
        raise ValueError("Primary16 matrix hash mismatch")
    features = tuple(str(value) for value in definition["feature_names"])
    matrix = pd.read_parquet(path, columns=["paper_id", *features])
    if len(matrix) != int(definition["row_count"]):
        raise ValueError("Primary16 row count mismatch")
    return matrix, features, path


def fit(output: Path) -> dict[str, Any]:
    """Train, score, replay, and freeze one production model release."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite production release: {output}")
    output.mkdir(parents=True)
    current = load_current_release()
    tuned = Path(current["resolved_release_path"])
    matrix, features, matrix_path = _primary_matrix(tuned)
    active = load_active_dataset(PROJECT_ROOT)
    training = assemble_all_period_frame(Path(active["feature_dataset_dir"]), horizon=5)
    training = training.merge(matrix, on="paper_id", how="left", validate="one_to_one")
    parameters = _selected_parameters(tuned)
    categorical = tuple(name for name in ("EF0197",) if name in features)
    inner = _inner_oof_for_parameters(
        training,
        feature_names=features,
        categorical_names=categorical,
        parameters=parameters,
        n_inner=int(parameters["inner_temporal_folds"]),
        seed=int(parameters["seed"]),
    )
    uptake_calibrator, conditional_calibrator, _, _ = _fit_calibrators(inner)
    model = _fit_two_part(
        training,
        feature_names=features,
        categorical_names=categorical,
        parameters=parameters,
        seed=int(parameters["seed"]),
    )
    model_path = output / "primary16_d5_model.joblib"
    joblib.dump(
        {
            "model": model,
            "uptake_calibrator": uptake_calibrator,
            "conditional_calibrator": conditional_calibrator,
            "feature_names": features,
            "categorical_names": categorical,
            "parameters": parameters,
        },
        model_path,
    )
    uptake_raw, conditional_raw = model.predict_raw(matrix)
    uptake = uptake_calibrator.predict(uptake_raw)
    conditional = conditional_calibrator.predict(conditional_raw)
    expected = uptake * conditional
    oof = pd.read_parquet(tuned / "oof_predictions.parquet")
    reference = np.sort(
        oof[
            oof["horizon"].eq(5) & oof["model_id"].eq("primary")
        ]["expected_diffusion_score"].to_numpy(dtype=float)
    )
    percentile = 100.0 * np.searchsorted(reference, expected, side="right") / len(reference)
    score_path = output / "prospective_5y_diffusion_scores.parquet"
    pd.DataFrame(
        {
            "paper_id": matrix["paper_id"].astype(str),
            "model_id": "primary",
            "horizon": 5,
            "expected_diffusion_score": expected,
            "prospective_5y_diffusion_percentile": percentile,
            "score_scope": "final_model_mapped_to_tuned_d5_primary_oof_reference",
        }
    ).to_parquet(score_path, index=False)
    reference_path = output / "d5_primary_oof_percentile_reference.npy"
    np.save(reference_path, reference)
    replay_path = output / "runtime_replay_100.parquet"
    pd.DataFrame(
        {
            "paper_id": matrix["paper_id"].astype(str).iloc[:100],
            "expected_diffusion_score": expected[:100],
            "prospective_5y_diffusion_percentile": percentile[:100],
        }
    ).to_parquet(replay_path, index=False)
    selection = {
        "selection_rule": "modal parameter_id across D5 outer-fold inner selections; outer-test labels excluded",
        "selected_parameter_id": parameters["parameter_id"],
        "parameters": parameters,
    }
    write_json(output / "parameter_selection.json", selection)
    artifacts = [model_path, score_path, reference_path, replay_path, output / "parameter_selection.json"]
    manifest = {
        "contract": "evidence_derived_tuned_primary16_d5_production_v1",
        "source_tuned_release": current["release_id"],
        "source_tuned_release_path": str(tuned),
        "matrix_path": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "training_rows": len(training),
        "scored_rows": len(matrix),
        "feature_names": list(features),
        "feature_count": len(features),
        "horizon": 5,
        "score_name": "prospective_5y_diffusion_percentile",
        "score_is_probability": False,
        "score_is_novelty": False,
        "oof_reference_rows": len(reference),
        "parameter_selection": selection,
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in artifacts
        },
    }
    write_json(output / "release_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = fit(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "training_rows": manifest["training_rows"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
