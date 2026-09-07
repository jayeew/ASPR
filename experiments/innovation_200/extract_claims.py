#!/usr/bin/env python3
"""Compile manuscripts and extract one shared set of at most eight claims."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.innovation_200.common import (
    configure_limits,
    experiment_config,
    read_jsonl,
    run_stage,
    setup_stage_logging,
)
from experiments.innovation_200.contracts import PaperRow
from gear.innovation.shared import prepare_shared
from gear.review_contracts import InnovationPaperInput


def paper_input(raw: dict) -> InnovationPaperInput:
    row = PaperRow.model_validate(raw)
    published = date.fromisoformat(row.publication_date)
    return InnovationPaperInput(
        paper_id=row.paper_id,
        paper_path=Path(row.paper_path),
        title=row.title,
        doi=row.doi,
        venue=row.journal_name,
        publication_date=published,
        cutoff_date=published,
        abstract_text=row.abstract_text,
        abstract_source=row.field_source,
        openalex_work_id=row.openalex_work_id,
        reference_work_ids=row.reference_work_ids,
        authors=row.authors,
    )


def extract(raw: dict, study: Path, overwrite: bool) -> dict[str, object]:
    item = paper_input(raw)
    root = study / "papers" / item.paper_id
    target = root / "shared/claims.json"
    if target.exists() and not overwrite:
        from gear.artifacts import read_model
        from gear.innovation.contracts import ClaimSet

        saved = read_model(target, ClaimSet)
        return {"skipped": True, "claims": len(saved.claims)}
    if overwrite and target.exists():
        target.unlink()
        paper_ir = root / "shared/paper_ir.json"
        if paper_ir.exists():
            paper_ir.unlink()
    _, claims = prepare_shared(item, root, experiment_config())
    return {"skipped": False, "claims": len(claims.claims)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cli-limit", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_stage_logging(args.study, "extract_claims", args.verbose)
    logger.info(
        "[配置] study=%s，workers=%d，cli_limit=%d，overwrite=%s",
        args.study,
        args.workers,
        args.cli_limit,
        args.overwrite,
    )
    configure_limits(args.cli_limit)
    run_stage(
        read_jsonl(args.study / "papers.jsonl"),
        lambda row: extract(row, args.study, args.overwrite),
        workers=args.workers,
        status_path=args.study / "status/extract_claims.json",
        usage_dir=args.study / "status/usage/extract_claims",
        logger=logger,
    )


if __name__ == "__main__":
    main()
