#!/usr/bin/env python3
"""Recompute the Fig.2/3 Primary16 D5 forecast for recent GEAR papers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from gear.diffusion_forecast import _predict_model
from gear.nature_multihorizon.runtime_replay_v3 import (
    build_runtime_context_for_year,
)
from gear.nature_multihorizon.t0_runtime_v3 import (
    ReferenceT0,
    TargetT0Record,
    coerce_fulltext16_storage_schema,
    materialize_fulltext16,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE = PROJECT_ROOT / "data/calibration/releases/gear-d5-primary16-current"
PRODUCTION = PROJECT_ROOT / (
    "innovation_impact_feature_selection/evidence_derived/production_releases/"
    "d5_primary16_tuned_20260827"
)
DEFAULT_CASES = PROJECT_ROOT / "outputs/gear/dev10_claim_graph/selection.jsonl"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/calibration/runtime_features/gear-d5-primary16-dev10-v1"
)
DEFAULT_SNAPSHOTS = DEFAULT_OUTPUT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _snapshot(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): dict(row) for row in payload["records"]}


def _field(record: Mapping[str, Any]) -> str | None:
    topic = record.get("primary_topic")
    field = topic.get("field") if isinstance(topic, dict) else None
    value = field.get("display_name") if isinstance(field, dict) else None
    return str(value) if value else None


def _source(record: Mapping[str, Any]) -> str | None:
    location = record.get("primary_location")
    source = location.get("source") if isinstance(location, dict) else None
    value = source.get("id") if isinstance(source, dict) else None
    return str(value) if value else None


def _target(
    case: Mapping[str, Any],
    record: Mapping[str, Any],
    references: Mapping[str, Mapping[str, Any]],
) -> TargetT0Record:
    cutoff_year = int(str(case["submission_date"])[:4])
    authorships = [
        row for row in record.get("authorships") or [] if isinstance(row, dict)
    ]
    author_ids = tuple(
        str(author["id"])
        for row in authorships
        if isinstance((author := row.get("author")), dict) and author.get("id")
    )
    reference_rows = []
    for reference_id in record.get("referenced_works") or []:
        metadata = references.get(str(reference_id), {})
        year = metadata.get("publication_year")
        reference_rows.append(
            ReferenceT0(
                reference_id=str(reference_id),
                publication_year=int(year) if year is not None else None,
                field_id=_field(metadata),
            )
        )
    country_count = int(record.get("countries_distinct_count") or 0)
    return TargetT0Record(
        paper_id=str(case["paper_id"]),
        publication_year=cutoff_year,
        title=str(record.get("display_name") or ""),
        author_ids=author_ids,
        author_count=len(authorships) or None,
        country_codes=tuple(f"COUNTRY_{index}" for index in range(country_count)),
        metadata_observed=True,
        source_id=_source(record),
        references=tuple(reference_rows),
    )


def build(cases_path: Path, snapshots: Path, output: Path) -> Path:
    """Recompute target Primary16 fields and apply the frozen HGB unchanged."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cases = _jsonl(cases_path)
    target_snapshot = snapshots / "target_openalex_snapshot.json"
    reference_snapshot = snapshots / "reference_openalex_snapshot.json"
    targets = _snapshot(target_snapshot)
    references = _snapshot(reference_snapshot)
    missing = sorted({str(row["paper_id"]) for row in cases} - set(targets))
    if missing:
        raise ValueError(f"target snapshot misses papers: {missing}")

    production = json.loads(
        (PRODUCTION / "release_manifest.json").read_text(encoding="utf-8")
    )
    official_matrix = Path(production["matrix_path"])
    rows = []
    context_manifests = []
    source_year_by_target: dict[int, int] = {}
    cases_by_year: dict[int, list[dict[str, Any]]] = {}
    for case in cases:
        cases_by_year.setdefault(int(str(case["submission_date"])[:4]), []).append(case)
    for year, year_cases in sorted(cases_by_year.items()):
        context, context_manifest = build_runtime_context_for_year(
            project_root=PROJECT_ROOT,
            official_matrix_path=official_matrix,
            target_year=year,
        )
        context_manifests.append(context_manifest)
        source_year_by_target[year] = int(context.source_max_year)
        for case in year_cases:
            target = _target(case, targets[str(case["paper_id"])], references)
            rows.append(
                {
                    "paper_id": target.paper_id,
                    **materialize_fulltext16(target, context),
                }
            )
        del context
        gc.collect()
    features = coerce_fulltext16_storage_schema(pd.DataFrame(rows))

    release_manifest = json.loads(
        (RELEASE / "release_manifest.json").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (RELEASE / "feature_registry.json").read_text(encoding="utf-8")
    )
    names = [str(value) for value in registry["feature_names"]]
    model = joblib.load(RELEASE / "model.joblib")
    scored = _predict_model(model, features[names])
    reference = pd.read_parquet(RELEASE / "percentile_reference.parquet")[
        "expected_diffusion_score"
    ].to_numpy(dtype=float)
    scored.insert(0, "paper_id", features["paper_id"].astype(str).to_numpy())
    scored["prospective_5y_diffusion_percentile"] = (
        100.0
        * np.searchsorted(reference, scored["expected_diffusion_score"], side="right")
        / len(reference)
    )
    case_map = {str(row["paper_id"]): row for row in cases}
    scored["as_of_date"] = scored["paper_id"].map(
        lambda value: str(case_map[str(value)]["submission_date"])
    )
    scored["feature_source"] = "runtime_recomputed_primary16"
    scored["source_max_year"] = (
        scored["as_of_date"].str[:4].astype(int).map(source_year_by_target)
    )
    completeness = features[names].notna().mean(axis=1).to_numpy(dtype=float)
    cutoff_year = scored["as_of_date"].str[:4].astype(int)
    history_start = cutoff_year - 5
    history_years = np.maximum(
        0, np.minimum(cutoff_year - 1, scored["source_max_year"]) - history_start + 1
    )
    scored["feature_completeness"] = completeness
    scored["historical_context_coverage"] = history_years / 5.0
    scored["feature_coverage"] = np.minimum(
        scored["feature_completeness"], scored["historical_context_coverage"]
    )

    score_path = output / "runtime_score_table.parquet"
    feature_path = output / "runtime_feature_table.parquet"
    scored.to_parquet(score_path, index=False)
    features[["paper_id", *names]].to_parquet(feature_path, index=False)
    shutil.copy2(target_snapshot, output / target_snapshot.name)
    shutil.copy2(reference_snapshot, output / reference_snapshot.name)
    context_path = output / "context_manifests.json"
    context_path.write_text(
        json.dumps(context_manifests, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "contract": "gear_d5_runtime_feature_release",
        "release_id": release_manifest["release_id"],
        "feature_protocol_version": release_manifest["protocol_version"],
        "target_count": len(scored),
        "feature_names": names,
        "source_max_year": int(scored["source_max_year"].max()),
        "feature_time_basis": "review_cutoff_primary16_t0",
        "classification_source": "evidence_derived_primary16_runtime_v3",
        "frozen_context_max_year": 2022,
        "post_context_target_policy": "recompute_primary16_then_frozen_hgb",
        "target_features_recomputed": True,
        "model_refit": False,
        "historical_context_limited_for": scored.loc[
            scored["source_max_year"] < cutoff_year - 1, "paper_id"
        ]
        .astype(str)
        .tolist(),
        "future_citation_counts_used": False,
        "network_used_for_target_freeze": True,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "assets": {
            "runtime_score_table": {
                "file": score_path.name,
                "sha256": _sha256(score_path),
            },
            "runtime_feature_table": {
                "file": feature_path.name,
                "sha256": _sha256(feature_path),
            },
            "target_snapshot": {
                "file": target_snapshot.name,
                "sha256": _sha256(output / target_snapshot.name),
            },
            "reference_snapshot": {
                "file": reference_snapshot.name,
                "sha256": _sha256(output / reference_snapshot.name),
            },
            "context_manifests": {
                "file": context_path.name,
                "sha256": _sha256(context_path),
            },
        },
    }
    manifest_path = output / "runtime_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build(args.cases.resolve(), args.snapshots.resolve(), args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
