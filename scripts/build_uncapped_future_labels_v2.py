#!/usr/bin/env python3
"""Prepare new targets and merge audited future labels for uncapped v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

HORIZON_YEAR_MAX = {3: 2022, 5: 2020, 8: 2017}


def sha256_file(path: Path) -> str:
    """Return a prefixed SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def expected_keys(targets: pd.DataFrame) -> pd.DataFrame:
    """Build the protocol-required paper-horizon key frame."""
    rows = []
    for horizon, year_max in HORIZON_YEAR_MAX.items():
        selected = targets.loc[targets["year"].le(year_max), ["id", "year"]].copy()
        selected.columns = ["paper_id", "publication_year"]
        selected["horizon"] = int(horizon)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True).drop_duplicates(["paper_id", "horizon"])


def read_targets(path: Path) -> pd.DataFrame:
    """Read and validate the article target table."""
    targets = pd.read_csv(path, low_memory=False)
    required = {"id", "year", "document_type"}
    missing = required - set(targets.columns)
    if missing:
        raise ValueError(f"Target table is missing columns: {sorted(missing)}")
    targets["year"] = pd.to_numeric(targets["year"], errors="raise").astype(int)
    if targets["id"].duplicated().any():
        raise ValueError("Target paper IDs must be unique")
    if not targets["document_type"].eq("article").all():
        raise ValueError("Uncapped target table must contain articles only")
    return targets


def prepare_added_targets(args: argparse.Namespace) -> dict[str, Any]:
    """Write only papers whose mature horizon keys are absent from the seed."""
    targets = read_targets(args.target_works)
    expected = expected_keys(targets)
    seed = pd.read_parquet(
        args.seed_future_dir / "future_graph_deltas_multihorizon.parquet",
        columns=["paper_id", "horizon"],
    ).drop_duplicates()
    comparison = expected.merge(
        seed,
        on=["paper_id", "horizon"],
        how="left",
        indicator=True,
    )
    missing = comparison.loc[comparison["_merge"].eq("left_only")].copy()
    added_ids = set(missing["paper_id"])
    selected = targets[targets["id"].isin(added_ids)].copy()
    args.output_target_works.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_target_works, index=False)
    partial_seed_papers = int(
        comparison.groupby("paper_id")["_merge"].nunique().gt(1).sum()
    )
    manifest = {
        "artifact_kind": "nature_uncapped_v2_added_future_targets",
        "n_all_targets": len(targets),
        "n_expected_keys": len(expected),
        "n_seed_keys_reused": int(comparison["_merge"].eq("both").sum()),
        "n_missing_keys_to_fetch": len(missing),
        "n_added_targets_to_fetch": len(selected),
        "n_partial_seed_papers": partial_seed_papers,
        "output_target_works": str(args.output_target_works.resolve()),
        "output_sha256": sha256_file(args.output_target_works),
        "overall_pass": partial_seed_papers == 0,
    }
    write_json(args.output_target_works.with_suffix(".manifest.json"), manifest)
    if not manifest["overall_pass"]:
        raise RuntimeError(f"Seed labels partially cover some papers: {manifest}")
    return manifest


def bad_seed_paper_ids(seed_future_dir: Path) -> set[str]:
    """Return seed papers with failed or capped future-label requests."""

    status = pd.read_parquet(seed_future_dir / "future_fetch_status.parquet")
    bad = status["fetch_status"].astype(str).ne("success")
    for column in ("cap_hit", "requested_horizon_cap_hit"):
        if column in status:
            bad |= pd.to_numeric(status[column], errors="coerce").fillna(0).ne(0)
    return set(status.loc[bad, "paper_id"].astype(str))


def prepare_repair_targets(args: argparse.Namespace) -> dict[str, Any]:
    """Write seed papers whose prior future labels failed or were capped."""

    targets = read_targets(args.target_works)
    all_repair_ids = bad_seed_paper_ids(args.seed_future_dir)
    target_ids = set(targets["id"].astype(str))
    repair_ids = all_repair_ids & target_ids
    selected = targets[targets["id"].astype(str).isin(repair_ids)].copy()
    if set(selected["id"].astype(str)) != repair_ids:
        missing = repair_ids - set(selected["id"].astype(str))
        raise RuntimeError(
            f"Repair targets are absent from target works: {len(missing)}"
        )
    args.output_target_works.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_target_works, index=False)
    manifest = {
        "artifact_kind": "nature_uncapped_v2_seed_future_repair_targets",
        "n_repair_targets": len(selected),
        "n_out_of_scope_seed_repair_ids": len(all_repair_ids - target_ids),
        "publication_year_min": int(selected["year"].min()),
        "publication_year_max": int(selected["year"].max()),
        "output_target_works": str(args.output_target_works.resolve()),
        "output_sha256": sha256_file(args.output_target_works),
        "repair_reason": "failed fetch or nonzero citer-cap flag in frozen seed",
        "overall_pass": len(selected) == len(repair_ids),
    }
    write_json(args.output_target_works.with_suffix(".manifest.json"), manifest)
    return manifest


def _read_delta_layer(root: Path, expected: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_parquet(root / "future_graph_deltas_multihorizon.parquet")
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="raise").astype(int)
    return frame.merge(
        expected[["paper_id", "horizon"]],
        on=["paper_id", "horizon"],
        how="inner",
        validate="one_to_one",
    )


def _read_status_layer(root: Path, expected: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_parquet(root / "future_fetch_status.parquet")
    frame["requested_horizon"] = pd.to_numeric(
        frame["requested_horizon"], errors="raise"
    ).astype(int)
    keys = expected[["paper_id", "horizon"]].rename(
        columns={"horizon": "requested_horizon"}
    )
    return frame.merge(
        keys,
        on=["paper_id", "requested_horizon"],
        how="inner",
        validate="one_to_one",
    )


def merge_layers(args: argparse.Namespace) -> dict[str, Any]:
    """Merge reused and newly fetched labels and enforce exact key coverage."""
    targets = read_targets(args.target_works)
    expected = expected_keys(targets)
    seed_deltas = _read_delta_layer(args.seed_future_dir, expected)
    seed_statuses = _read_status_layer(args.seed_future_dir, expected)
    repair_ids = bad_seed_paper_ids(args.seed_future_dir) & set(
        expected["paper_id"].astype(str)
    )
    repair_deltas = pd.DataFrame(columns=seed_deltas.columns)
    repair_statuses = pd.DataFrame(columns=seed_statuses.columns)
    if args.repair_future_dir is not None:
        repair_deltas = _read_delta_layer(args.repair_future_dir, expected)
        repair_statuses = _read_status_layer(args.repair_future_dir, expected)
        repair_papers = set(repair_deltas["paper_id"].astype(str))
        if repair_papers != repair_ids:
            raise RuntimeError(
                "Repair layer must replace exactly the failed/capped seed papers: "
                f"expected={len(repair_ids)}, observed={len(repair_papers)}"
            )
        seed_deltas = seed_deltas[
            ~seed_deltas["paper_id"].astype(str).isin(repair_ids)
        ].copy()
        seed_statuses = seed_statuses[
            ~seed_statuses["paper_id"].astype(str).isin(repair_ids)
        ].copy()
    elif repair_ids:
        raise RuntimeError(
            f"Seed layer has {len(repair_ids)} failed/capped papers; "
            "provide --repair-future-dir"
        )
    added_deltas = _read_delta_layer(args.added_future_dir, expected)
    replacement_deltas = pd.concat(
        [seed_deltas, repair_deltas], ignore_index=True, sort=False
    )
    overlap = set(replacement_deltas["paper_id"]) & set(added_deltas["paper_id"])
    if overlap:
        raise ValueError(
            f"Seed and added delta layers overlap for {len(overlap)} papers"
        )
    deltas = pd.concat(
        [replacement_deltas, added_deltas], ignore_index=True, sort=False
    )
    statuses = pd.concat(
        [
            seed_statuses,
            repair_statuses,
            _read_status_layer(args.added_future_dir, expected),
        ],
        ignore_index=True,
        sort=False,
    )
    comparison = expected.merge(
        deltas[["paper_id", "horizon"]],
        on=["paper_id", "horizon"],
        how="outer",
        indicator=True,
    )
    status_success = statuses["fetch_status"].astype(str).eq("success").all()
    cap_columns = [
        name
        for name in ("cap_hit", "requested_horizon_cap_hit")
        if name in statuses.columns
    ]
    caps_clear = all(
        pd.to_numeric(statuses[name], errors="coerce").fillna(0).eq(0).all()
        for name in cap_columns
    )
    checks = {
        "seed_added_paper_sets_disjoint": not overlap,
        "delta_primary_key_unique": bool(
            not deltas.duplicated(["paper_id", "horizon"]).any()
        ),
        "status_primary_key_unique": bool(
            not statuses.duplicated(["paper_id", "requested_horizon"]).any()
        ),
        "all_expected_keys_present": bool(comparison["_merge"].eq("both").all()),
        "no_unexpected_keys": bool(comparison["_merge"].eq("both").all()),
        "all_fetches_successful": bool(status_success),
        "no_citer_caps": bool(caps_clear),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Merged future-layer audit failed: {checks}")
    requests = pd.concat(
        [
            pd.read_parquet(args.seed_future_dir / "future_request_manifest.parquet"),
            *(
                [
                    pd.read_parquet(
                        args.repair_future_dir / "future_request_manifest.parquet"
                    )
                ]
                if args.repair_future_dir is not None
                else []
            ),
            pd.read_parquet(args.added_future_dir / "future_request_manifest.parquet"),
        ],
        ignore_index=True,
    )
    requests = requests[requests["paper_id"].isin(set(expected["paper_id"]))]
    requests = requests.drop_duplicates("paper_id", keep="last")
    if set(requests["paper_id"]) != set(expected["paper_id"]):
        raise RuntimeError("Request manifest does not cover every labeled paper")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "future_graph_deltas_multihorizon.parquet": deltas,
        "future_fetch_status.parquet": statuses,
        "future_request_manifest.parquet": requests,
    }
    for name, frame in outputs.items():
        frame.to_parquet(args.output_dir / name, index=False, compression="zstd")
    manifest = {
        "artifact_kind": "nature_uncapped_v2_merged_future_labels",
        "n_expected_keys": len(expected),
        "n_seed_keys_reused": len(seed_deltas),
        "n_seed_papers_repaired": len(repair_ids),
        "n_repair_keys_fetched": len(repair_deltas),
        "n_added_keys_fetched": len(added_deltas),
        "n_unique_papers": int(deltas["paper_id"].nunique()),
        "checks": checks,
        "overall_pass": True,
        "lineage": {
            "seed_future_dir": str(args.seed_future_dir.resolve()),
            "repair_future_dir": (
                str(args.repair_future_dir.resolve())
                if args.repair_future_dir is not None
                else None
            ),
            "added_future_dir": str(args.added_future_dir.resolve()),
        },
        "outputs": {
            name: {
                "path": str((args.output_dir / name).resolve()),
                "sha256": sha256_file(args.output_dir / name),
            }
            for name in outputs
        },
    }
    write_json(args.output_dir / "future_label_merge_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-added-targets")
    prepare.add_argument("--target-works", type=Path, required=True)
    prepare.add_argument("--seed-future-dir", type=Path, required=True)
    prepare.add_argument("--output-target-works", type=Path, required=True)
    repair = subparsers.add_parser("prepare-repair-targets")
    repair.add_argument("--target-works", type=Path, required=True)
    repair.add_argument("--seed-future-dir", type=Path, required=True)
    repair.add_argument("--output-target-works", type=Path, required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--target-works", type=Path, required=True)
    merge.add_argument("--seed-future-dir", type=Path, required=True)
    merge.add_argument("--added-future-dir", type=Path, required=True)
    merge.add_argument("--repair-future-dir", type=Path)
    merge.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one stage and print its manifest."""
    args = build_parser().parse_args(argv)
    if args.command == "prepare-added-targets":
        manifest = prepare_added_targets(args)
    elif args.command == "prepare-repair-targets":
        manifest = prepare_repair_targets(args)
    else:
        manifest = merge_layers(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
