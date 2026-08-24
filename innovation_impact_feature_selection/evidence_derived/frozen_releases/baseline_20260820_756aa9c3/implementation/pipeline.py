#!/usr/bin/env python3
"""Eight-command CLI for the simplified evidence-derived protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from .core import (
        EvidenceProtocol,
        ProtocolError,
        file_hash,
        insert_rows,
        read_json_rows,
    )
except ImportError:  # Direct execution from this directory.
    from core import (  # type: ignore[no-redef]
        EvidenceProtocol,
        ProtocolError,
        file_hash,
        insert_rows,
        read_json_rows,
    )

IMPORT_TABLES = {
    "works",
    "citations",
    "search_runs",
    "terms",
    "term_families",
    "search_domains",
    "logical_queries",
    "physical_queries",
    "seed_recall",
    "screening_decisions",
    "screening_final",
    "construct_mentions",
    "candidate_dimensions",
    "indicator_mentions",
    "indicator_families",
    "indicator_evidence",
    "indicator_data_mapping",
    "hard_gate_decisions",
    "evidence_tiers",
    "review_sessions",
}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--database", type=Path, default=Path("outputs/evidence_derived.sqlite3")
    )
    result.add_argument("--output-dir", type=Path, default=Path("outputs"))
    commands = result.add_subparsers(dest="command", required=True)

    bootstrap = commands.add_parser("bootstrap")
    bootstrap.add_argument("--legacy-input", type=Path, action="append", default=[])
    bootstrap.add_argument("--input", type=Path)
    bootstrap.add_argument("--table", choices=sorted(IMPORT_TABLES))

    saturate = commands.add_parser("saturate")
    saturate.add_argument("--round", type=int, required=True)
    saturate.add_argument("--new-terms", type=int, required=True)
    saturate.add_argument("--new-indicators", type=int, required=True)
    saturate.add_argument("--review-artifact", type=Path, required=True)
    saturate.add_argument("--fully-reviewed", action="store_true")

    freeze = commands.add_parser("freeze-search")
    freeze.add_argument("--input", type=Path)
    freeze.add_argument(
        "--table",
        choices=[
            "search_domains",
            "logical_queries",
            "physical_queries",
            "seed_recall",
            "review_sessions",
        ],
    )

    screen = commands.add_parser("screen")
    screen.add_argument("--input", type=Path)
    screen.add_argument(
        "--table",
        choices=[
            "works",
            "citations",
            "screening_decisions",
            "screening_final",
            "review_sessions",
        ],
    )

    dimensions = commands.add_parser("derive-dimensions")
    dimensions.add_argument("--input", type=Path)
    dimensions.add_argument(
        "--table",
        choices=["construct_mentions", "candidate_dimensions", "review_sessions"],
    )

    census = commands.add_parser("census-indicators")
    census.add_argument("--input", type=Path)
    census.add_argument(
        "--table",
        choices=[
            "indicator_mentions",
            "indicator_families",
            "indicator_evidence",
            "indicator_data_mapping",
            "review_sessions",
        ],
    )

    select = commands.add_parser("select-features")
    select.add_argument("--input", type=Path)
    select.add_argument(
        "--table",
        choices=[
            "hard_gate_decisions",
            "evidence_tiers",
            "indicator_data_mapping",
            "review_sessions",
        ],
    )
    select.add_argument("--training-source", type=Path)
    select.add_argument("--id-column", default="paper_id")

    commands.add_parser("audit")
    return result


def import_optional(
    engine: EvidenceProtocol, table: str | None, path: Path | None
) -> int:
    if bool(table) != bool(path):
        raise ProtocolError("--table and --input must be supplied together")
    if not table or not path:
        return 0
    rows = read_json_rows(path)
    return insert_rows(engine.connection, table, rows, IMPORT_TABLES)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    with EvidenceProtocol(args.database, args.output_dir) as engine:
        engine.initialize()
        if args.command == "bootstrap":
            imported = import_optional(engine, args.table, args.input)
            inventory = engine.register_legacy_inventory(args.legacy_input)
            return {
                "stage": "bootstrap",
                "legacy_inventory": inventory,
                "rows_imported": imported,
            }
        if args.command == "saturate":
            basis = engine.record_saturation_round(
                args.round,
                args.new_terms,
                args.new_indicators,
                args.fully_reviewed,
                file_hash(args.review_artifact),
            )
            return {"stage": "saturate", "round": args.round, "stop_basis": basis}
        imported = import_optional(
            engine, getattr(args, "table", None), getattr(args, "input", None)
        )
        if args.command == "freeze-search":
            return {
                "stage": "freeze-search",
                "rows_imported": imported,
                **engine.freeze_search(),
            }
        if args.command == "screen":
            return {
                "stage": "screen",
                "rows_imported": imported,
                "final_rows": engine.finalize_screening(),
            }
        if args.command == "derive-dimensions":
            return {
                "stage": "derive-dimensions",
                "rows_imported": imported,
                "M": engine.validate_dimensions(),
            }
        if args.command == "census-indicators":
            return {
                "stage": "census-indicators",
                "rows_imported": imported,
                "F_all": engine.validate_indicator_census(),
            }
        if args.command == "select-features":
            sets = engine.select_features()
            materialized = None
            if args.training_source:
                materialized = engine.materialize_training_sets(
                    args.training_source, args.id_column
                )
            return {
                "stage": "select-features",
                "rows_imported": imported,
                "counts": {key: len(value) for key, value in sets.items()},
                "materialized": materialized,
            }
        if args.command == "audit":
            return engine.audit()
    raise ProtocolError(f"Unsupported command: {args.command}")


def main() -> int:
    try:
        result = execute(parser().parse_args())
    except (ProtocolError, OSError, json.JSONDecodeError) as error:
        print(
            json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
