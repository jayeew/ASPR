"""Train leakage-safe cross-fitted HGB U/D/P/R heads from prepared targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .structural_head_training import run_cross_fitted_structural_heads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--feature", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-column", default="outer_fold_id")
    parser.add_argument("--joint-lambda", type=float, default=0.5)
    args = parser.parse_args()
    frame = (
        pd.read_csv(args.input)
        if args.input.suffix.casefold() == ".csv"
        else pd.read_parquet(args.input)
    )
    predictions, manifest = run_cross_fitted_structural_heads(
        frame,
        feature_columns=args.feature,
        fold_column=args.fold_column,
        joint_lambda=args.joint_lambda,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(
        args.output_dir / "structural_heads_oof.parquet", index=False
    )
    (args.output_dir / "structural_heads_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
