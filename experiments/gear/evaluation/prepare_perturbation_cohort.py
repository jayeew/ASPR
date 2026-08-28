"""Prepare a score-balanced cohort with observable future-graph records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def prepare_cohort(
    cohort_path: Path,
    future_summary_path: Path,
    output_path: Path,
    *,
    horizon: int = 5,
    limit: int = 800,
) -> dict[str, Any]:
    """Select by availability and frozen strata without reading future values."""
    cohort = pd.read_csv(cohort_path)
    future = pd.read_parquet(
        future_summary_path,
        columns=["paper_id", "horizon", "fetch_valid"],
    )
    future = future[future["horizon"].eq(horizon) & future["fetch_valid"].eq(1)][
        ["paper_id"]
    ].drop_duplicates()
    eligible = cohort.merge(future, on="paper_id", how="inner", validate="one_to_one")
    eligible["blind_key"] = eligible["paper_id"].map(
        lambda value: hashlib.sha256(f"gear-perturbation|{value}".encode()).hexdigest()
    )
    eligible["cell_rank"] = eligible.groupby(
        ["score_decile", "domain12"], observed=True
    )["blind_key"].rank(method="first")
    selected = eligible.sort_values(
        ["cell_rank", "score_decile", "domain12", "blind_key"]
    ).head(limit)
    payload = {
        "contract": "gear_real_perturbation_cohort_v1",
        "selection_uses_future_values": False,
        "selection_requires_future_graph_availability": True,
        "horizon": horizon,
        "cases": [
            {
                "paper_id": row.paper_id,
                "score_decile": int(row.score_decile),
                "domain12": row.domain12,
                "publication_year": int(row.publication_year),
            }
            for row in selected.itertuples(index=False)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    cohort_csv = output_path.with_suffix(".csv")
    selected.drop(columns=["blind_key", "cell_rank"]).to_csv(cohort_csv, index=False)
    return {
        "eligible": len(eligible),
        "selected": len(selected),
        "score_deciles": int(selected["score_decile"].nunique()),
        "domains": int(selected["domain12"].nunique()),
        "output": str(output_path.resolve()),
        "cohort_csv": str(cohort_csv.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--future-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--limit", type=int, default=800)
    args = parser.parse_args()
    summary = prepare_cohort(
        args.cohort,
        args.future_summary,
        args.output,
        horizon=args.horizon,
        limit=args.limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["prepare_cohort"]
