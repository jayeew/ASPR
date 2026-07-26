from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd


def parse_reference_list(value: object) -> List[str]:
    """Parse OpenAlex referenced_works stored as JSON, semicolon, comma, or pipe text."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if text.startswith("["):
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                return [str(item).strip() for item in payload if str(item).strip()]
        except json.JSONDecodeError:
            pass
    for delimiter in (";", "|", ","):
        if delimiter in text:
            return [part.strip() for part in text.split(delimiter) if part.strip()]
    return [text]


def _paper_id(row: pd.Series) -> str:
    for column in ("paper_id", "id", "work_id", "openalex_id"):
        if column in row and str(row[column]).strip():
            return str(row[column])
    return str(row.name)


def build_reference_closure(works: pd.DataFrame, coverage_target: float = 0.80) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize cached OpenAlex references and report per-domain closure coverage."""
    if "domain" not in works.columns:
        raise ValueError("works table must include a domain column")
    if "referenced_works" not in works.columns:
        raise ValueError("works table must include a referenced_works column")

    rows: List[dict[str, object]] = []
    reports: List[dict[str, object]] = []
    work = works.copy()
    work["domain"] = work["domain"].astype(str)
    for domain, group in work.groupby("domain", sort=True):
        total = 0
        materialized = 0
        unique_refs: set[str] = set()
        for _, row in group.iterrows():
            refs = parse_reference_list(row.get("referenced_works"))
            total += len(refs)
            pid = _paper_id(row)
            for ref in refs:
                rows.append(
                    {
                        "domain": domain,
                        "paper_id": pid,
                        "referenced_work_id": ref,
                        "reference_source": "cached_openalex_referenced_works",
                    }
                )
                unique_refs.add(ref)
                materialized += 1
        coverage = float(materialized / total) if total else 0.0
        measured = int(total > 0)
        reports.append(
            {
                "domain": domain,
                "reference_closure_mode": "cached_openalex_referenced_works",
                "online_expand": 0,
                "eligible_papers": int(len(group)),
                "referenced_works_count": int(total),
                "materialized_reference_count": int(materialized),
                "external_unique_references": int(len(unique_refs)),
                "coverage_materialized": coverage,
                "coverage_measured": measured,
                "coverage_status": "measured_cached_references" if measured else "not_measured_no_references",
                "coverage_target": float(coverage_target),
                "quality_gate_pass": int(measured == 1 and coverage >= float(coverage_target)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(reports)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize Fig.2 reference closure from cached OpenAlex references.")
    parser.add_argument("--works", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--coverage-target", type=float, default=0.80)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    works = pd.read_csv(args.works)
    closure, report = build_reference_closure(works, coverage_target=float(args.coverage_target))
    closure.to_csv(args.out_dir / "reference_closure_table.csv", index=False)
    report.to_csv(args.out_dir / "fig2_reference_closure_report.csv", index=False)


if __name__ == "__main__":
    main()
