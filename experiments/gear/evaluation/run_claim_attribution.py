"""Train the claim-attribution head from aligned citing-context labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .claim_attribution_training import run_claim_attribution_oof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--feature", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold-column", default="outer_fold_id")
    args = parser.parse_args()
    frame = (
        pd.read_csv(args.input)
        if args.input.suffix.casefold() == ".csv"
        else pd.read_parquet(args.input)
    )
    predictions, manifest = run_claim_attribution_oof(
        frame, feature_columns=args.feature, fold_column=args.fold_column
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(
        args.output_dir / "claim_attribution_oof.parquet", index=False
    )
    (args.output_dir / "claim_attribution_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
