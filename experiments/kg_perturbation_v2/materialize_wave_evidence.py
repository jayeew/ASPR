#!/usr/bin/env python3
"""Materialize provenance-bound Wave-B/C evidence tables for a V2 release."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aspr.nature_multihorizon.artifact_store import hash_file  # noqa: E402
from aspr.nature_multihorizon.figure_views import (  # noqa: E402
    CORE_FEATURES,
    MECHANISM_COLUMNS,
)
from aspr.nature_multihorizon.features import build_feature_table  # noqa: E402
from aspr.nature_multihorizon.scoring import FrozenReleaseScorer  # noqa: E402
from scripts.run_nature_multihorizon import _figure_evidence_sources  # noqa: E402


ROBUSTNESS_IDS = (
    "horizon_3_5_8",
    "citation_threshold_sensitivity",
    "graph_snapshot_frequency",
    "community_algorithm",
    *(f"remove_{name}" for name in MECHANISM_COLUMNS),
    "remove_all_auxiliary",
    "remove_calibration",
    "model_family_comparison",
    "seed_stability",
    "fold_stability",
)
ABLATION_IDS = (
    *(f"remove_{name}" for name in MECHANISM_COLUMNS),
    "remove_all_auxiliary",
    "no_calibration",
    "model_family_comparison",
    "no_graph_agent",
    "no_qwen",
    "no_fusion_verifier",
)
REQUIRED_IDS: Dict[str, tuple[str, ...]] = {
    "fig04_peer_review_validation": (
        "peer_review_resample_v2",
        "new_score_external_validity",
    ),
    "fig05_frontier_backtest": (
        "ai_frontier_tau5_join",
        "forecast_backtest_v2",
    ),
    "fig06_registered_robustness": ROBUSTNESS_IDS,
    "fig07_venue_family_inference": (
        "venue_family_diffusion_enrichment_mechanism_time_panels",
    ),
    "fig09_case_evidence": ("graph_qwen_fusion_rerun",),
    "fig10_registered_ablations": ABLATION_IDS,
}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError("Input table must be CSV or Parquet")


def _prepare_assets(
    output_dir: Path,
    sources: Sequence[Path],
    protocol: Path,
) -> tuple[set[str], str]:
    assets = output_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    inputs = [Path(path).expanduser().resolve() for path in (*sources, protocol)]
    names = [path.name for path in inputs]
    if len(names) != len(set(names)):
        raise ValueError("Evidence asset basenames must be unique")
    hashes = set()
    protocol_hash = ""
    for source in inputs:
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Evidence asset must be a regular file: {source}")
        destination = assets / source.name
        if destination.exists():
            if hash_file(destination) != hash_file(source):
                raise FileExistsError(
                    f"Evidence asset exists with different content: {destination}"
                )
        else:
            shutil.copyfile(source, destination)
        value = hash_file(destination)
        hashes.add(value)
        if source == Path(protocol).expanduser().resolve():
            protocol_hash = value
    return hashes, protocol_hash


def package_table(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _read_table(args.input.expanduser().resolve())
    if frame.empty:
        raise ValueError("Evidence table is empty")
    required_ids = set(REQUIRED_IDS[args.artifact])
    observed_ids = set(
        frame.get("evidence_id", pd.Series(dtype=str)).dropna().astype(str)
    )
    if observed_ids != required_ids:
        raise ValueError(
            "Evidence IDs must exactly match the locked set; "
            f"missing={sorted(required_ids - observed_ids)}, "
            f"unexpected={sorted(observed_ids - required_ids)}"
        )
    asset_hashes, protocol_hash = _prepare_assets(
        output_dir, args.source, args.protocol
    )
    if "protocol_hash" not in frame:
        frame["protocol_hash"] = protocol_hash
    if "source_artifact_sha256" not in frame:
        source_hashes = asset_hashes - {protocol_hash}
        if len(source_hashes) != 1:
            raise ValueError(
                "Multiple source assets require source_artifact_sha256 per row"
            )
        frame["source_artifact_sha256"] = next(iter(source_hashes))
    for column in ("source_artifact_sha256", "protocol_hash"):
        declared = set(frame[column].dropna().astype(str))
        if not declared.issubset(asset_hashes):
            raise ValueError(f"{column} contains a hash absent from copied assets")
    required_columns = {"metric", "value", "n", "source_artifact_sha256", "protocol_hash"}
    if args.artifact != "fig09_case_evidence":
        required_columns.update({"ci_low", "ci_high"})
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Evidence table is missing columns: {missing}")
    if args.artifact in {
        "fig06_registered_robustness",
        "fig10_registered_ablations",
    }:
        metric = frame["metric"].fillna("").astype(str)
        visible = (
            metric.str.startswith("rho_")
            | metric.eq("top_decile_enrichment")
            | metric.str.contains("ratio|share", regex=True)
            | metric.str.startswith("n_")
        )
        visible_ids = set(frame.loc[visible, "evidence_id"].astype(str))
        if not required_ids.issubset(visible_ids):
            raise ValueError("Some registered rows would be invisible in Fig.6/Fig.10")
    output = output_dir / f"{args.artifact}.parquet"
    if output.exists():
        raise FileExistsError(f"Evidence table is immutable: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Stale evidence-table temporary file: {temporary}")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    _figure_evidence_sources(
        SimpleNamespace(figure_evidence_dir=output_dir)
    )
    return {
        "artifact": args.artifact,
        "path": str(output),
        "sha256": hash_file(output),
        "rows": int(len(frame)),
    }


def build_case_features(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the same locked 18 features for one fixed external case."""

    papers = _read_table(args.paper.expanduser().resolve())
    references = _read_table(args.references.expanduser().resolve())
    reference_works = _read_table(args.reference_works.expanduser().resolve())
    if len(papers) != 1:
        raise ValueError("build-case-features requires exactly one focal paper")
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".parquet":
        raise ValueError("Case feature output must use .parquet")
    if output.exists():
        raise FileExistsError(f"Case feature table is immutable: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Stale case-feature temporary file: {temporary}")
    try:
        frame = build_feature_table(
            papers,
            references,
            reference_works,
            args.graph_snapshots.expanduser().resolve(),
        )
        for column in ("case_id", "doi", "title"):
            if column in papers:
                frame[column] = papers.iloc[0][column]
        frame.to_parquet(temporary, index=False)
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    if len(frame) != 1:
        raise RuntimeError("Case feature builder did not emit exactly one row")
    return {"path": str(output), "sha256": hash_file(output), "rows": 1}


def score_case(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = args.features.expanduser().resolve()
    release_path = args.release.expanduser().resolve()
    if release_path.is_dir():
        release_path = release_path / "release.json"
    if not release_path.is_file():
        raise FileNotFoundError(release_path)
    features = _read_table(features_path)
    if len(features) != 1:
        raise ValueError("score-case requires exactly one publication-time feature row")
    scorer = FrozenReleaseScorer(release_path, horizon=5)
    registry_path = scorer.release.artifact("case_registry")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    cases = registry.get("cases", []) if isinstance(registry, dict) else []
    matched = [
        row
        for row in cases
        if isinstance(row, dict) and str(row.get("case_id") or "") == args.case_id
    ]
    if len(matched) != 1:
        raise ValueError(
            "--case-id must identify exactly one pre-locked case_registry row"
        )
    locked_case = matched[0]

    def normalize_doi(value: Any) -> str:
        return str(value or "").lower().replace("https://doi.org/", "").strip()

    feature_row = features.iloc[0]
    locked_doi = normalize_doi(locked_case.get("doi"))
    feature_doi = normalize_doi(feature_row.get("doi"))
    locked_paper_id = str(locked_case.get("paper_id") or "").strip()
    feature_paper_id = str(feature_row.get("paper_id") or "").strip()
    feature_case_id = str(feature_row.get("case_id") or "").strip()
    if feature_case_id and feature_case_id != args.case_id:
        raise ValueError("Feature row case_id does not match the locked case")
    identity_matches = bool(
        (locked_doi and feature_doi == locked_doi)
        or (locked_paper_id and feature_paper_id == locked_paper_id)
    )
    if not identity_matches:
        raise ValueError(
            "Feature row DOI/paper_id does not match the pre-locked case registry"
        )
    valid_references = pd.to_numeric(
        pd.Series([feature_row.get("valid_reference_count")]), errors="coerce"
    ).iloc[0]
    metadata_coverage = pd.to_numeric(
        pd.Series([feature_row.get("reference_metadata_coverage")]),
        errors="coerce",
    ).iloc[0]
    if not (
        np.isfinite(valid_references)
        and valid_references >= 10
        and np.isfinite(metadata_coverage)
        and metadata_coverage >= 0.60
    ):
        raise ValueError(
            "Fixed case requires >=10 valid references and >=60% metadata coverage"
        )
    core_values = features[list(CORE_FEATURES)].apply(
        pd.to_numeric, errors="coerce"
    ) if set(CORE_FEATURES).issubset(features.columns) else pd.DataFrame()
    if core_values.empty or not np.isfinite(core_values.to_numpy(float)).all():
        raise ValueError("Fixed case requires finite values for all eight core indicators")
    protocol_source = output_dir / "case_score_protocol.json"
    if protocol_source.exists():
        raise FileExistsError(f"Stale case-score protocol exists: {protocol_source}")
    protocol_source.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "operation": "frozen_release_score_case",
                "release": str(release_path),
                "release_sha256": hash_file(release_path),
                "horizon": 5,
                "case_id": args.case_id,
                "locked_case": locked_case,
                "case_registry_sha256": hash_file(registry_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        asset_hashes, protocol_hash = _prepare_assets(
            output_dir, [features_path], protocol_source
        )
    finally:
        protocol_source.unlink(missing_ok=True)
    source_hashes = asset_hashes - {protocol_hash}
    source_hash = next(iter(source_hashes))
    scored, _ = scorer.score_features(features)
    row = scored.iloc[0].to_dict()
    value = float(row["score_performance_percentile"])
    if not np.isfinite(value):
        raise ValueError("Case score is not finite")
    row.update(
        {
            "case_id": args.case_id,
            "doi": locked_doi,
            "case_status": "scored",
            "evidence_id": "fixed_case_score",
            "metric": "score_performance_percentile",
            "value": value,
            "n": 1,
            "source_artifact_sha256": source_hash,
            "protocol_hash": protocol_hash,
            "score_scope": "out_of_cohort_extrapolation",
            "outcome_observable": "unknown",
            "valid_reference_count": int(valid_references),
            "reference_metadata_coverage": float(metadata_coverage),
        }
    )
    output = output_dir / "fig09_case_profile.parquet"
    if output.exists():
        raise FileExistsError(f"Evidence table is immutable: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Stale case-profile temporary file: {temporary}")
    try:
        pd.DataFrame([row]).to_parquet(temporary, index=False)
        temporary.rename(output)
    finally:
        temporary.unlink(missing_ok=True)
    _figure_evidence_sources(
        SimpleNamespace(figure_evidence_dir=output_dir)
    )
    return {"path": str(output), "sha256": hash_file(output), "rows": 1}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    package = subparsers.add_parser("package-table")
    package.add_argument("--artifact", choices=sorted(REQUIRED_IDS), required=True)
    package.add_argument("--input", type=Path, required=True)
    package.add_argument("--source", type=Path, nargs="+", required=True)
    package.add_argument("--protocol", type=Path, required=True)
    package.add_argument("--output-dir", type=Path, required=True)
    case = subparsers.add_parser("score-case")
    case.add_argument("--release", type=Path, required=True)
    case.add_argument("--features", type=Path, required=True)
    case.add_argument("--case-id", required=True)
    case.add_argument("--output-dir", type=Path, required=True)
    feature = subparsers.add_parser("build-case-features")
    feature.add_argument("--paper", type=Path, required=True)
    feature.add_argument("--references", type=Path, required=True)
    feature.add_argument("--reference-works", type=Path, required=True)
    feature.add_argument("--graph-snapshots", type=Path, required=True)
    feature.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "score-case":
        result = score_case(args)
    elif args.command == "build-case-features":
        result = build_case_features(args)
    else:
        result = package_table(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
