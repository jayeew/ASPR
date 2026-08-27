#!/usr/bin/env python3
"""Build the frozen Primary16 HGB anatomy index without review-label access."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from gear.graph_calibration import ROLE_FEATURES, compute_anatomy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = PROJECT_ROOT / "data/calibration/releases/gear-d5-primary16-current"
TARGET_WORKS = Path(
    "/mnt/d/aspr_nature_portfolio_v5/openalex_outputs/uncapped_aspr_v2/"
    "nature_target_works.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/calibration/graph_calibration/primary16_forecast_anatomy_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _mode(series: pd.Series) -> str:
    nonempty = series.dropna().astype(str)
    return nonempty.mode().iloc[0] if not nonempty.empty else "__MISSING__"


def _metadata() -> pd.DataFrame:
    rows = pd.read_csv(
        TARGET_WORKS,
        usecols=["id", "title", "year", "openalex_primary_field"],
        low_memory=False,
    ).rename(
        columns={
            "id": "paper_id",
            "year": "publication_year_metadata",
            "openalex_primary_field": "field",
        }
    )
    rows["paper_id"] = rows["paper_id"].astype(str)
    rows["field"] = rows["field"].fillna("__UNKNOWN__").astype(str)
    rows["title"] = rows["title"].fillna("").astype(str)
    return rows.drop_duplicates("paper_id", keep="last")


def _baselines(
    frame: pd.DataFrame, names: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = [name for name in names if name != "EF0197"]
    grouped = frame.groupby("field", dropna=False)
    values = grouped[numeric].median().reset_index()
    values["EF0197"] = grouped["EF0197"].agg(_mode).to_numpy()
    global_row: dict[str, Any] = {name: frame[name].median() for name in numeric}
    global_row["EF0197"] = _mode(frame["EF0197"])
    global_row["field"] = "__GLOBAL__"
    values = pd.concat([values, pd.DataFrame([global_row])], ignore_index=True)
    values["baseline_id"] = "field:" + values["field"].astype(str)
    merged = frame[["paper_id", "field"]].merge(
        values, on="field", how="left", validate="many_to_one"
    )
    if merged[names].isna().all(axis=1).any():
        fallback = values.loc[
            values["field"] == "__GLOBAL__", names + ["baseline_id"]
        ].iloc[0]
        missing = merged[names].isna().all(axis=1)
        for name in [*names, "baseline_id"]:
            merged.loc[missing, name] = fallback[name]
    return values, merged[["paper_id", *names, "baseline_id"]]


def build(output: Path, *, batch_size: int = 10_000) -> Path:
    """Materialize an immutable, model-derived anatomy index."""

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = json.loads((RELEASE_ROOT / "release_manifest.json").read_text())
    registry = json.loads((RELEASE_ROOT / "feature_registry.json").read_text())
    names = [str(value) for value in registry["feature_names"]]
    training_path = RELEASE_ROOT / manifest["assets"]["training_snapshot"]["file"]
    score_path = RELEASE_ROOT / manifest["assets"]["score_table"]["file"]
    training = pd.read_parquet(training_path)
    training["paper_id"] = training["paper_id"].astype(str)
    metadata = _metadata()
    training = training.merge(
        metadata, on="paper_id", how="left", validate="one_to_one"
    )
    training["field"] = training["field"].fillna("__UNKNOWN__").astype(str)
    training["title"] = training["title"].fillna("").astype(str)
    baseline_table, baseline_values = _baselines(training, names)
    baseline = training[["paper_id"]].merge(
        baseline_values, on="paper_id", how="left", validate="one_to_one"
    )
    model = joblib.load(RELEASE_ROOT / manifest["assets"]["model"]["file"])
    scores = pd.read_parquet(score_path)
    uptake_reference = np.sort(scores["uptake_probability_calibrated"].to_numpy(float))
    conditional_reference = np.sort(
        scores["conditional_diffusion_calibrated"].to_numpy(float)
    )
    expected_reference = np.sort(scores["expected_diffusion_score"].to_numpy(float))
    parts = []
    for start in range(0, len(training), batch_size):
        target = training.iloc[start : start + batch_size][["paper_id", *names]].copy()
        base = baseline.iloc[start : start + batch_size][
            ["paper_id", *names, "baseline_id"]
        ].copy()
        part = compute_anatomy(
            model,
            target,
            base[names],
            feature_names=names,
            uptake_reference=uptake_reference,
            conditional_reference=conditional_reference,
            expected_reference=expected_reference,
            baseline_ids=base["baseline_id"].astype(str).tolist(),
            release_id=manifest["release_id"],
            target_fields=training.iloc[start : start + batch_size]["field"]
            .astype(str)
            .tolist(),
        )
        parts.append(part)
    anatomy = pd.concat(parts, ignore_index=True)
    index = training[["paper_id", "title", "publication_year", "field"]].merge(
        anatomy, on="paper_id", how="inner", validate="one_to_one"
    )
    index["title_sha256"] = index["title"].map(
        lambda value: "sha256:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    )
    index_path = output / "anatomy_index.parquet"
    baselines_path = output / "field_baselines.parquet"
    anatomy.to_parquet(output / "training_anatomy.parquet", index=False)
    index.to_parquet(index_path, index=False)
    baseline_table.to_parquet(baselines_path, index=False)
    shutil.copy2(
        RELEASE_ROOT / "feature_registry.json", output / "feature_registry.json"
    )
    source_hash = _sha256(training_path)
    metadata_hash = _sha256(TARGET_WORKS)
    combined_source_hash = (
        "sha256:"
        + hashlib.sha256(f"{source_hash}|{metadata_hash}".encode()).hexdigest()
    )
    payload = {
        "contract": "gear_primary16_forecast_anatomy_release_v1",
        "release_id": manifest["release_id"],
        "feature_protocol_version": manifest["protocol_version"],
        "roles": ROLE_FEATURES,
        "source_snapshot_id": "primary16_training_features_plus_frozen_metadata_v1",
        "source_snapshot_sha256": combined_source_hash,
        "feature_snapshot_sha256": source_hash,
        "metadata_snapshot_sha256": metadata_hash,
        "row_count": len(index),
        "baseline_policy": "frozen_same_openalex_field_median_mode_global_fallback",
        "uses_review_or_relation_labels": False,
        "uses_future_citation_outcomes": False,
        "assets": {
            name: {"file": path.name, "sha256": _sha256(path)}
            for name, path in {
                "anatomy_index": index_path,
                "field_baselines": baselines_path,
                "training_anatomy": output / "training_anatomy.parquet",
                "feature_registry": output / "feature_registry.json",
            }.items()
        },
    }
    (output / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return output / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args()
    print(build(args.output.resolve(), batch_size=args.batch_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
