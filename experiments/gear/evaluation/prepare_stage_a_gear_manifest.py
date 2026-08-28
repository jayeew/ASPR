"""Build a blinded GEAR benchmark manifest from acquired OOF manuscripts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

EXCLUDED_TITLE_TOKENS = (
    "correction:",
    "corrigendum",
    "erratum",
    "retraction",
    "publisher correction",
)


def prepare_manifest(
    cohort_path: Path,
    acquisition_path: Path,
    output_path: Path,
    *,
    limit: int = 120,
) -> dict[str, Any]:
    """Select a deterministic, score-stratified set without reading outcomes."""
    cohort = pd.read_csv(cohort_path)
    acquired = pd.read_csv(acquisition_path)
    _validate_inputs(cohort, acquired)
    acquired = acquired[acquired["download_status"].eq("downloaded")].copy()
    title = acquired.get("title", pd.Series("", index=acquired.index)).fillna("")
    excluded = title.str.casefold().map(
        lambda value: any(token in value for token in EXCLUDED_TITLE_TOKENS)
    )
    acquired = acquired[~excluded]
    cohort_fields = ["paper_id", "score_decile", "domain12"]
    frame = cohort[cohort_fields].merge(
        acquired, on="paper_id", how="inner", validate="one_to_one"
    )
    frame = frame.dropna(subset=["publication_date_resolved", "manuscript_path"])
    selected = _balanced_selection(frame, limit)
    payload = {
        "contract": "gear_stage_a_blinded_benchmark_v1",
        "selection_uses_future_outcomes": False,
        "cases": [_case(row) for _, row in selected.iterrows()],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pilot = _pilot(selected)
    pilot_path = output_path.with_name(output_path.stem + "_pilot" + output_path.suffix)
    pilot_payload = {**payload, "cases": [_case(row) for _, row in pilot.iterrows()]}
    pilot_path.write_text(
        json.dumps(pilot_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "eligible": len(frame),
        "selected": len(selected),
        "score_deciles": sorted(
            int(value) for value in selected["score_decile"].unique()
        ),
        "domains": int(selected["domain12"].nunique()),
        "pilot_cases": len(pilot),
        "manifest": str(output_path.resolve()),
        "pilot_manifest": str(pilot_path.resolve()),
    }


def _validate_inputs(cohort: pd.DataFrame, acquired: pd.DataFrame) -> None:
    cohort_required = {"paper_id", "score_decile", "domain12"}
    acquired_required = {
        "paper_id",
        "download_status",
        "publication_date_resolved",
        "manuscript_path",
    }
    missing = (cohort_required - set(cohort)) | (acquired_required - set(acquired))
    if missing:
        raise ValueError(f"manifest inputs missing columns: {sorted(missing)}")
    if cohort["paper_id"].duplicated().any() or acquired["paper_id"].duplicated().any():
        raise ValueError("manifest inputs must be unique at paper grain")


def _balanced_selection(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit < 1:
        raise ValueError("limit must be positive")
    ranked = frame.copy()
    ranked["blind_key"] = ranked["paper_id"].map(
        lambda value: hashlib.sha256(f"gear-stage-a|{value}".encode()).hexdigest()
    )
    ranked["cell_rank"] = ranked.groupby(["score_decile", "domain12"], observed=True)[
        "blind_key"
    ].rank(method="first")
    return (
        ranked.sort_values(["cell_rank", "score_decile", "domain12", "blind_key"])
        .head(limit)
        .reset_index(drop=True)
    )


def _case(row: pd.Series) -> dict[str, Any]:
    paper_id = str(row["paper_id"])
    short_id = paper_id.rstrip("/").rsplit("/", 1)[-1]
    return {
        "case_id": short_id,
        "paper_id": paper_id,
        "paper_path": str(Path(str(row["manuscript_path"])).resolve()),
        "cutoff": str(row["publication_date_resolved"]),
        "score_decile": int(row["score_decile"]),
        "domain12": str(row["domain12"]),
        "publication_year": int(str(row["publication_date_resolved"])[:4]),
        "metadata": {
            "title": _optional_string(row.get("title")) or "",
            "doi": _optional_string(row.get("doi")),
            "openalex_id": paper_id,
            "publication_date": str(row["publication_date_resolved"]),
            "domain": str(row["domain12"]),
        },
    }


def _optional_string(value: Any) -> str | None:
    return None if pd.isna(value) or not str(value).strip() else str(value)


def _pilot(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected
    deciles = sorted(int(value) for value in selected["score_decile"].unique())
    targets = [deciles[0], deciles[len(deciles) // 2], deciles[-1]]
    return pd.concat(
        [selected[selected["score_decile"].eq(target)].head(1) for target in targets],
        ignore_index=True,
    ).drop_duplicates("paper_id")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()
    summary = prepare_manifest(
        args.cohort, args.acquisition, args.output, limit=args.limit
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["prepare_manifest"]
