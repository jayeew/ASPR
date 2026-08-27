"""Build the sole GEAR D5 release from the Fig.2/3 Primary16 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from gear.diffusion_forecast import _predict_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = PROJECT_ROOT / (
    "innovation_impact_feature_selection/evidence_derived/production_releases/"
    "d5_primary16_tuned_20260827"
)
TUNED = PROJECT_ROOT / (
    "innovation_impact_feature_selection/evidence_derived/frozen_releases/"
    "hgb_nested_tuned_7_16_153_219_20260820_b48936af"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data/calibration/releases/gear-d5-primary16-current"
ALIAS = "fig2_fig3_primary16:d5_diffusion"
PROTOCOL = "evidence-derived-primary16-d5-tuned-1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _asset(path: Path, root: Path) -> dict[str, Any]:
    return {
        "file": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _temporal_folds(source: pd.DataFrame) -> pd.DataFrame:
    folds = source.loc[
        source["horizon"].eq(5) & source["model_id"].eq("primary")
    ].copy()
    folds = folds.rename(columns={"outer_fold_id": "fold_id"})
    folds["train_year_max"] = folds["test_year_min"].astype(int) - 1
    return folds[
        ["fold_id", "train_year_max", "test_year_min", "test_year_max", "n_oof"]
    ]


def build(output: Path) -> Path:
    """Package the already-trained Primary16 model without refitting it."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    production_manifest = json.loads(
        (PRODUCTION / "release_manifest.json").read_text(encoding="utf-8")
    )
    feature_names = tuple(production_manifest["feature_names"])
    matrix_path = Path(production_manifest["matrix_path"])
    matrix = pd.read_parquet(matrix_path, columns=["paper_id", *feature_names])
    model_source = PRODUCTION / "primary16_d5_model.joblib"
    model = joblib.load(model_source)
    prediction = _predict_model(model, matrix)
    reference = np.load(PRODUCTION / "d5_primary_oof_percentile_reference.npy")
    percentile = (
        100.0
        * np.searchsorted(
            reference, prediction["expected_diffusion_score"], side="right"
        )
        / len(reference)
    )
    scores = prediction.copy()
    scores.insert(0, "paper_id", matrix["paper_id"].astype(str).to_numpy())
    scores["publication_year"] = matrix["EF0307"].astype(int).to_numpy()
    scores["prospective_5y_diffusion_percentile"] = percentile
    scores["feature_coverage"] = matrix[list(feature_names)].notna().mean(axis=1)

    production_scores = pd.read_parquet(
        PRODUCTION / "prospective_5y_diffusion_scores.parquet"
    ).sort_values("paper_id", kind="stable")
    comparison = scores.sort_values("paper_id", kind="stable")
    error = np.max(
        np.abs(
            comparison["prospective_5y_diffusion_percentile"].to_numpy()
            - production_scores["prospective_5y_diffusion_percentile"].to_numpy()
        )
    )
    if float(error) > 1e-12:
        raise ValueError(f"Primary16 production replay mismatch: {error}")

    model_path = output / "model.joblib"
    shutil.copy2(model_source, model_path)
    snapshot_path = output / "training_snapshot.parquet"
    matrix.assign(publication_year=matrix["EF0307"].astype(int)).to_parquet(
        snapshot_path, index=False
    )
    score_path = output / "score_table.parquet"
    scores.to_parquet(score_path, index=False)
    reference_path = output / "percentile_reference.parquet"
    pd.DataFrame({"expected_diffusion_score": reference}).to_parquet(
        reference_path, index=False
    )
    replay_path = output / "runtime_replay_matrix.parquet"
    scores.sample(n=4096, random_state=20260806).sort_values(
        "paper_id", kind="stable"
    ).to_parquet(replay_path, index=False)

    feature_path = output / "feature_registry.json"
    _write_json(
        feature_path,
        {
            "contract": "gear_d5_primary16_feature_registry",
            "protocol_version": PROTOCOL,
            "model_id": "primary16",
            "horizon_years": 5,
            "outcome": "future_5y_scholarly_diffusion",
            "feature_names": list(feature_names),
            "categorical_feature_names": list(model["categorical_names"]),
            "source_fig2_sets": {
                "strict": 7,
                "primary": 16,
                "expanded": 153,
                "broad_t0": 219,
            },
            "source_release": production_manifest["source_tuned_release"],
        },
    )

    oof = pd.read_parquet(TUNED / "oof_predictions.parquet")
    oof = oof.loc[oof["horizon"].eq(5) & oof["model_id"].eq("primary")]
    oof.to_parquet(output / "oof_predictions.parquet", index=False)
    for name in ("oof_metrics", "oof_fold_metrics", "oof_domain_metrics"):
        frame = pd.read_csv(TUNED / f"{name}.csv")
        frame = frame.loc[frame["horizon"].eq(5) & frame["model_id"].eq("primary")]
        frame.to_csv(output / f"{name}.csv", index=False)
    fold_source = pd.read_csv(TUNED / "oof_fold_metrics.csv")
    _temporal_folds(fold_source).to_csv(output / "temporal_folds.csv", index=False)
    shutil.copy2(TUNED / "run_manifest.json", output / "oof_run_manifest.json")
    shutil.copy2(
        TUNED / "frozen_input_matrix_manifest.json",
        output / "registry_freeze_manifest.json",
    )

    assets = {
        path.stem: _asset(path, output)
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    identity = {
        "contract": "gear_diffusion_forecast_release",
        "alias": ALIAS,
        "status": "frozen",
        "runtime_status": "available",
        "protocol_version": PROTOCOL,
        "model_id": "primary16",
        "horizon_years": 5,
        "score_semantics": "prospective_5y_diffusion_percentile",
        "training_row_count": int(production_manifest["training_rows"]),
        "oof_row_count": int(production_manifest["oof_reference_rows"]),
        "source_release": production_manifest["source_tuned_release"],
        "assets": assets,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    identity["release_id"] = (
        "gear-d5-primary16-" + hashlib.sha256(canonical).hexdigest()[:12]
    )
    identity["created_at_utc"] = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )
    manifest_path = output / "release_manifest.json"
    _write_json(manifest_path, identity)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
