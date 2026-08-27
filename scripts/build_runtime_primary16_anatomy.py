#!/usr/bin/env python3
"""Attach frozen-HGB anatomy to a recomputed Primary16 runtime release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gear.graph_calibration import compute_anatomy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = (
    PROJECT_ROOT / "data/calibration/runtime_features/gear-d5-primary16-dev10-v1"
)
RELEASE_ROOT = PROJECT_ROOT / "data/calibration/releases/gear-d5-primary16-current"
ANATOMY_ROOT = (
    PROJECT_ROOT / "data/calibration/graph_calibration/primary16_forecast_anatomy_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def build(runtime_root: Path, anatomy_root: Path) -> Path:
    """Create one runtime anatomy table and update its local asset manifest."""

    manifest_path = runtime_root / "runtime_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    feature_path = runtime_root / manifest["assets"]["runtime_feature_table"]["file"]
    target = pd.read_parquet(feature_path)
    registry = json.loads((RELEASE_ROOT / "feature_registry.json").read_text())
    names = [str(value) for value in registry["feature_names"]]
    snapshot = json.loads((runtime_root / "target_openalex_snapshot.json").read_text())
    fields = {}
    for row in snapshot["records"]:
        topic = row.get("primary_topic") or {}
        field = topic.get("field") or {}
        fields[str(row["id"])] = str(field.get("display_name") or "__GLOBAL__")
    target["field"] = target["paper_id"].astype(str).map(fields).fillna("__GLOBAL__")
    baselines = pd.read_parquet(anatomy_root / "field_baselines.parquet")
    baseline = target[["paper_id", "field"]].merge(
        baselines, on="field", how="left", validate="many_to_one"
    )
    global_row = baselines.loc[
        baselines["field"] == "__GLOBAL__", names + ["baseline_id"]
    ].iloc[0]
    missing = baseline[names].isna().all(axis=1)
    for name in [*names, "baseline_id"]:
        baseline.loc[missing, name] = global_row[name]
    release_manifest = json.loads((RELEASE_ROOT / "release_manifest.json").read_text())
    model = joblib.load(RELEASE_ROOT / release_manifest["assets"]["model"]["file"])
    scores = pd.read_parquet(
        RELEASE_ROOT / release_manifest["assets"]["score_table"]["file"]
    )
    anatomy = compute_anatomy(
        model,
        target[["paper_id", *names]],
        baseline[names],
        feature_names=names,
        uptake_reference=np.sort(
            scores["uptake_probability_calibrated"].to_numpy(float)
        ),
        conditional_reference=np.sort(
            scores["conditional_diffusion_calibrated"].to_numpy(float)
        ),
        expected_reference=np.sort(scores["expected_diffusion_score"].to_numpy(float)),
        baseline_ids=baseline["baseline_id"].astype(str).tolist(),
        release_id=release_manifest["release_id"],
        target_fields=target["field"].astype(str).tolist(),
    )
    output = runtime_root / "runtime_anatomy_table.parquet"
    anatomy.to_parquet(output, index=False)
    manifest["assets"]["runtime_anatomy_table"] = {
        "file": output.name,
        "sha256": _sha256(output),
    }
    manifest["anatomy_release_manifest"] = str(anatomy_root / "manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    parser.add_argument("--anatomy-root", type=Path, default=ANATOMY_ROOT)
    args = parser.parse_args()
    print(build(args.runtime_root.resolve(), args.anatomy_root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
