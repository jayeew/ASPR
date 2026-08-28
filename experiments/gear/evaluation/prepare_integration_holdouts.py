"""Freeze outcome-blind development, temporal, and domain integration splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def prepare_holdouts(
    benchmark_manifest: Path,
    output_path: Path,
    *,
    temporal_fraction: float = 0.25,
    heldout_domain_count: int = 2,
    seed: int = 20260828,
    development_only_manifest: Path | None = None,
) -> dict[str, Any]:
    """Assign splits using publication time/domain only, never future outcomes."""
    if not 0.0 < temporal_fraction < 0.5:
        raise ValueError("temporal_fraction must be in (0, 0.5)")
    payload = json.loads(benchmark_manifest.read_text(encoding="utf-8"))
    cases = list(payload.get("cases", []))
    frame = pd.DataFrame(
        {
            "paper_id": [str(case["paper_id"]) for case in cases],
            "publication_year": [_publication_year(case) for case in cases],
            "domain12": [_domain(case) for case in cases],
        }
    )
    if frame["paper_id"].duplicated().any():
        raise ValueError("benchmark manifest paper IDs must be unique")
    development_only_ids = (
        _manifest_ids(development_only_manifest)
        if development_only_manifest is not None
        else set()
    )
    eligible_holdout = frame[~frame["paper_id"].isin(development_only_ids)].copy()
    domains = eligible_holdout["domain12"].value_counts()
    if heldout_domain_count < 1 or heldout_domain_count >= len(domains):
        raise ValueError("heldout_domain_count must leave development domains")
    ranked_domains = sorted(
        domains.index,
        key=lambda domain: (
            -int(domains[domain]),
            _key(str(domain), seed),
        ),
    )
    heldout_domains = set(ranked_domains[:heldout_domain_count])
    eligible_holdout["split_key"] = eligible_holdout["paper_id"].map(
        lambda value: _key(str(value), seed)
    )
    temporal_count = max(1, math.ceil(len(eligible_holdout) * temporal_fraction))
    temporal_rows = eligible_holdout.sort_values(
        ["publication_year", "split_key"], ascending=[False, True]
    ).head(temporal_count)
    temporal_ids = set(temporal_rows["paper_id"].astype(str))
    temporal_start_year = int(temporal_rows["publication_year"].min())
    split_by_id: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        if str(row.paper_id) in development_only_ids:
            split_by_id[str(row.paper_id)] = "development"
            continue
        temporal = str(row.paper_id) in temporal_ids
        domain = str(row.domain12) in heldout_domains
        split = (
            "joint_time_domain_holdout"
            if temporal and domain
            else (
                "temporal_holdout"
                if temporal
                else "domain_holdout" if domain else "development"
            )
        )
        split_by_id[str(row.paper_id)] = split
    output_cases = [
        {**case, "integration_split": split_by_id[str(case["paper_id"])]}
        for case in cases
    ]
    output = {
        "contract": "gear_integration_holdout_manifest_v1",
        "selection_uses_future_outcomes": False,
        "selection_variables": [
            "paper_id",
            "publication_year",
            "domain12",
            "development_only_membership",
        ],
        "freeze_claim": "before_integration_evaluation_not_raw_data_acquisition",
        "seed": seed,
        "temporal_start_year": temporal_start_year,
        "heldout_domains": sorted(heldout_domains),
        "development_only_papers": len(development_only_ids & set(frame["paper_id"])),
        "cases": output_cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    counts = pd.Series(split_by_id).value_counts().sort_index().to_dict()
    return {
        "papers": len(output_cases),
        "temporal_start_year": temporal_start_year,
        "heldout_domains": sorted(heldout_domains),
        "development_only_papers": len(development_only_ids & set(frame["paper_id"])),
        "split_counts": {str(key): int(value) for key, value in counts.items()},
        "output": str(output_path.resolve()),
    }


def _publication_year(case: dict[str, Any]) -> int:
    value = case.get("publication_year") or str(case.get("cutoff", ""))[:4]
    return int(value)


def _domain(case: dict[str, Any]) -> str:
    return str(case.get("domain12") or (case.get("metadata") or {}).get("domain"))


def _key(value: str, seed: int) -> str:
    return hashlib.sha256(f"gear-integration|{seed}|{value}".encode()).hexdigest()


def _manifest_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(case["paper_id"]) for case in payload.get("cases", [])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporal-fraction", type=float, default=0.25)
    parser.add_argument("--heldout-domain-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--development-only-manifest", type=Path)
    args = parser.parse_args()
    result = prepare_holdouts(
        args.benchmark_manifest,
        args.output,
        temporal_fraction=args.temporal_fraction,
        heldout_domain_count=args.heldout_domain_count,
        seed=args.seed,
        development_only_manifest=args.development_only_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["prepare_holdouts"]
