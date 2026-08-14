from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from gear.calibration import CalibrationService, sha256_file
from gear.config import PROJECT_ROOT, load_config
from gear.nature_multihorizon.runtime_replay_v3 import (
    build_runtime_fulltext16_matrix,
)
from gear.nature_multihorizon.t0_runtime_v3 import validate_materialization_replay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay GEAR Full-text-16")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, action="append")
    parser.add_argument("--limit-per-year", type=int)
    parser.add_argument("--config", type=Path)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    service = CalibrationService(config)
    paths = config.resolved_assets()
    matrix, context, input_manifest = build_runtime_fulltext16_matrix(
        project_root=PROJECT_ROOT,
        official_matrix_path=paths.feature_matrix_16,
        years=args.year,
        limit_per_year=args.limit_per_year,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = output / "runtime_fulltext16.parquet"
    context_path = output / "context_snapshot.joblib"
    matrix.to_parquet(matrix_path, index=False)
    joblib.dump(context, context_path, compress=3)
    if args.year or args.limit_per_year is not None:
        service._load()
        selected = pd.read_parquet(paths.feature_matrix_16)
        selected = selected[selected["paper_id"].isin(set(matrix["paper_id"]))]

        def prediction(frame: pd.DataFrame) -> np.ndarray:
            features = frame.drop(columns=["paper_id"], errors="ignore")
            uptake_raw, conditional_raw = service._model_bundle["model"].predict_raw(
                features
            )
            uptake = service._model_bundle["uptake_calibrator"].predict(uptake_raw)
            conditional = service._model_bundle["conditional_calibrator"].predict(
                conditional_raw
            )
            return np.asarray(uptake, dtype=float) * np.asarray(
                conditional, dtype=float
            )

        report = validate_materialization_replay(
            matrix,
            selected,
            prediction_func=prediction,
            rtol=1e-7,
            atol=1e-9,
        )
    else:
        report = service.validate_runtime_replay(matrix)
    payload = {
        "contract": "aspr_fulltext16_runtime_replay_release_v1",
        "input_manifest": input_manifest,
        "runtime_matrix_sha256": sha256_file(matrix_path),
        "context_snapshot_sha256": sha256_file(context_path),
        "official_matrix_sha256": sha256_file(paths.feature_matrix_16),
        "replay_report": report.__dict__,
    }
    (output / "runtime_replay_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.eligible_inference else 1


if __name__ == "__main__":
    raise SystemExit(main())
