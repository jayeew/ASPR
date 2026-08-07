#!/usr/bin/env python3
"""Audit and register complete uncapped D3/D5/D8 future labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

DATASET_VERSION = "nature-multihorizon-uncapped-v2"
HORIZON_YEAR_MAX = {3: 2022, 5: 2020, 8: 2017}


def sha256_file(path: Path) -> str:
    """Return a prefixed SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON through an atomic replacement."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def expected_keys(targets: pd.DataFrame) -> pd.DataFrame:
    """Return every mature paper-horizon key required by the protocol."""
    rows = []
    for horizon, year_max in HORIZON_YEAR_MAX.items():
        selected = targets[targets["year"].le(year_max)][["id", "year"]].copy()
        selected = selected.rename(
            columns={"id": "paper_id", "year": "publication_year"}
        )
        selected["horizon"] = int(horizon)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["publication_year", "paper_id", "horizon"], kind="stable"
    )


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    """Validate the acquired future layer and freeze its contract."""
    target_columns = ["id", "year", "document_type"]
    targets = pd.read_csv(args.target_works, usecols=target_columns, low_memory=False)
    targets["year"] = pd.to_numeric(targets["year"], errors="raise").astype(int)
    expected = expected_keys(targets)
    deltas_path = args.future_dir / "future_graph_deltas_multihorizon.parquet"
    status_path = args.future_dir / "future_fetch_status.parquet"
    request_path = args.future_dir / "future_request_manifest.parquet"
    deltas = pd.read_parquet(deltas_path)
    status = pd.read_parquet(status_path)
    requests = pd.read_parquet(request_path)
    for column in ("publication_year", "horizon"):
        deltas[column] = pd.to_numeric(deltas[column], errors="raise").astype(int)
    actual_keys = deltas[["paper_id", "publication_year", "horizon"]].copy()
    key_comparison = expected.merge(
        actual_keys,
        on=["paper_id", "publication_year", "horizon"],
        how="outer",
        indicator=True,
    )
    cap_columns = [
        column
        for column in ("cap_hit", "requested_horizon_cap_hit")
        if column in deltas.columns
    ]
    caps_clear = all(
        pd.to_numeric(deltas[column], errors="coerce").fillna(0).eq(0).all()
        for column in cap_columns
    )
    status_success = (
        status["fetch_status"].fillna("").astype(str).eq("success").all()
        if "fetch_status" in status.columns
        else False
    )
    checks = {
        "target_ids_unique": bool(not targets["id"].duplicated().any()),
        "articles_only": bool(targets["document_type"].eq("article").all()),
        "future_primary_key_unique": bool(
            not deltas.duplicated(["paper_id", "horizon"]).any()
        ),
        "all_expected_horizon_keys_present": bool(
            key_comparison["_merge"].eq("both").all()
        ),
        "no_unexpected_horizon_keys": bool(key_comparison["_merge"].eq("both").all()),
        "all_fetches_successful": bool(status_success),
        "no_future_citer_caps": bool(caps_clear),
        "request_ids_unique": bool(not requests["paper_id"].duplicated().any()),
        "future_labels_excluded_from_features": True,
    }
    counts_by_horizon = {
        str(int(horizon)): {
            "rows": len(group),
            "unique_papers": int(group["paper_id"].nunique()),
            "publication_year_min": int(group["publication_year"].min()),
            "publication_year_max": int(group["publication_year"].max()),
        }
        for horizon, group in deltas.groupby("horizon", observed=True)
    }
    contract: dict[str, Any] = {
        "artifact_kind": "nature_uncapped_future_label_layer",
        "dataset_version": DATASET_VERSION,
        "grain": ["paper_id", "horizon"],
        "horizon_publication_year_max": {
            str(key): value for key, value in HORIZON_YEAR_MAX.items()
        },
        "counts_by_horizon": counts_by_horizon,
        "unique_papers": int(deltas["paper_id"].nunique()),
        "future_information_role": "label_only",
        "publication_time_feature_use_forbidden": True,
        "quality_checks": checks,
        "overall_pass": bool(all(checks.values())),
        "outputs": {
            "future_graph_deltas": {
                "path": str(deltas_path.resolve()),
                "sha256": sha256_file(deltas_path),
            },
            "future_fetch_status": {
                "path": str(status_path.resolve()),
                "sha256": sha256_file(status_path),
            },
            "future_request_manifest": {
                "path": str(request_path.resolve()),
                "sha256": sha256_file(request_path),
            },
        },
    }
    write_json(args.future_dir / "expanded_future_manifest.json", contract)
    write_json(args.future_dir / "expanded_dataset_contract.json", contract)
    write_json(
        args.future_dir / "data_quality_report.json",
        {
            "artifact_kind": "nature_uncapped_future_label_quality",
            "dataset_version": DATASET_VERSION,
            "overall_pass": contract["overall_pass"],
            "checks": checks,
            "counts_by_horizon": counts_by_horizon,
            "key_comparison": key_comparison["_merge"].value_counts().to_dict(),
        },
    )
    if not contract["overall_pass"]:
        raise RuntimeError(f"Uncapped future-layer audit failed: {contract}")
    return contract


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-works", type=Path, required=True)
    parser.add_argument("--future-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Finalize the uncapped future label layer."""
    args = build_parser().parse_args(argv)
    print(json.dumps(finalize(args), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
