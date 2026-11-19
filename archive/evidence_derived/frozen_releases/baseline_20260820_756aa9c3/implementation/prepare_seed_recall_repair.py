#!/usr/bin/env python3
"""Propose evidence-sourced query repairs for SEARCH_TERM_MISSING seeds."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    from .core import canonical_json, file_hash
except ImportError:
    from core import canonical_json, file_hash  # type: ignore[no-redef]

csv.field_size_limit(sys.maxsize)

REPAIRS = {
    "FLQ01": {
        "old": '("citation impact" OR "citation count" OR "citation counts" OR bibliometric OR bibliometrics OR "scientific influence" OR "scholarly influence")',
        "new": '("citation impact" OR "citation count" OR "citation counts" OR citations OR bibliometric OR bibliometrics OR "citation-based impact" OR "impact measure" OR "impact measures" OR "scientific impact" OR "research impact" OR "long-term scientific impact" OR "scientific influence" OR "scholarly influence")',
        "method": (" OR quantify OR quantifying OR dynamics OR evolution"),
        "seeds": [
            "DEV_18b31deb0a35fcaa",
            "DEV_d62f25d979a1b245",
            "H2_HIDDEN_VALIDATION_001",
        ],
    },
    "FLQ03": {
        "old": "(novelty OR innovation OR disruption OR breakthrough OR breakthroughs OR originality)",
        "new": '(novelty OR innovation OR innovativeness OR disruption OR disrupt OR breakthrough OR breakthroughs OR originality OR "atypical combination" OR "atypical combinations" OR "unusual combination" OR "unusual combinations" OR diversity OR "knowledge diversity" OR "research portfolio" OR "topic prominence")',
        "method": " OR analysis OR framework OR quantify OR quantitative OR link",
        "seeds": [
            "DEV_38c1e27ec30965e2",
            "DEV_5bbf0dfb38f65ed4",
            "DEV_8a06bd1070c05b83",
            "H2_HIDDEN_VALIDATION_003",
        ],
    },
    "FLQ08": {
        "old": '("open access" OR "open data" OR preregistration OR preregistered OR "open science" OR "research dissemination")',
        "new": '("open access" OR "open data" OR "data reuse" OR preregistration OR preregistered OR "open science" OR "research dissemination")',
        "method": " OR trial OR advantage OR benefit OR diffusion",
        "target_old": '(innovation OR novelty OR "research quality" OR "scholarly impact" OR "citation impact" OR "scientific influence")',
        "target_new": '(innovation OR novelty OR "research quality" OR "scholarly impact" OR "citation impact" OR citation OR citations OR "citation advantage" OR readership OR downloads OR "scientific influence")',
        "seeds": ["DEV_bb6e89e5630da003", "DEV_d2e6d23d27bf7886"],
    },
    "FLQ10": {
        "old": '(authorship OR collaboration OR coauthorship OR "team composition" OR "research team" OR "research funding")',
        "new": '(authorship OR collaboration OR coauthorship OR team OR teams OR "team size" OR "team composition" OR "research team" OR "knowledge network" OR "knowledge networks" OR "research funding")',
        "method": " OR analysis OR advantage OR dominance OR produce OR production",
        "target_old": '(innovation OR novelty OR "research quality" OR "scholarly impact" OR "citation impact" OR "scientific influence")',
        "target_new": '(innovation OR novelty OR disrupt OR disruption OR "research quality" OR "scholarly impact" OR "citation impact" OR citation OR citations OR "scientific impact" OR "high-impact" OR "scientific influence" OR "production of knowledge")',
        "seeds": [
            "DEV_231292517512442d",
            "DEV_407ed308a884f9e6",
            "DEV_a066409035d11dd8",
        ],
    },
    "FLQ12": {
        "old": '("article title" OR "paper title" OR "article abstract" OR "paper abstract" OR "textual feature" OR "textual features" OR "reference list" OR "reference lists" OR "citation context")',
        "new": '("article title" OR "paper title" OR "short title" OR "title length" OR "article abstract" OR "paper abstract" OR "textual feature" OR "textual features" OR "reference list" OR "reference lists" OR "citation context")',
        "method": " OR investigate OR analysis OR metric OR advantage",
        "target_old": '(innovation OR novelty OR "research quality" OR "scholarly impact" OR "citation impact" OR "scientific influence")',
        "target_new": '(innovation OR novelty OR "research quality" OR "scholarly impact" OR "citation impact" OR citation OR citations OR attention OR "scientific influence")',
        "seeds": ["DEV_ac605b1c190125fd"],
    },
}

METHOD_TAIL = (
    "predict OR prediction OR predictive OR model OR modeling OR association OR "
    "relationship OR determinant OR effect)"
)


def repair_expression(query_id: str, expression: str) -> str:
    repair = REPAIRS.get(query_id)
    if not repair:
        return expression
    revised = expression.replace(str(repair["old"]), str(repair["new"]))
    if "target_old" in repair:
        revised = revised.replace(str(repair["target_old"]), str(repair["target_new"]))
    revised = revised.replace(
        METHOD_TAIL, METHOD_TAIL[:-1] + str(repair["method"]) + ")"
    )
    return revised


def prepare(database: Path, queries: Path, output_dir: Path) -> dict[str, object]:
    import sqlite3

    with queries.open(encoding="utf-8", newline="") as handle:
        query_rows = list(csv.DictReader(handle))
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    evidence: dict[str, dict[str, str]] = {}
    for repair in REPAIRS.values():
        for seed_id in repair["seeds"]:
            row = connection.execute(
                "SELECT s.seed_id,s.cohort,i.doi,w.title,w.abstract FROM seed_recall s "
                "JOIN seed_inputs i USING(seed_id) JOIN works w USING(work_id) "
                "WHERE s.seed_id=? AND s.reason_code='SEARCH_TERM_MISSING'",
                (seed_id,),
            ).fetchone()
            if not row:
                raise RuntimeError(
                    f"Repair seed is not an active SEARCH_TERM_MISSING case: {seed_id}"
                )
            evidence[seed_id] = dict(row)
    connection.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    repaired_path = output_dir / "final_search_queries_recall_repaired_primary.csv"
    with repaired_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(query_rows[0]))
        writer.writeheader()
        for row in query_rows:
            row["expression"] = repair_expression(row["query_id"], row["expression"])
            writer.writerow(row)
    repair_path = output_dir / "search_frame_seed_recall_repair_primary.csv"
    fields = [
        "query_id",
        "seed_id",
        "cohort",
        "doi",
        "source_title",
        "source_abstract",
        "reason_code",
        "repair_reason",
    ]
    with repair_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for query_id, repair in sorted(REPAIRS.items()):
            for seed_id in repair["seeds"]:
                row = evidence[seed_id]
                writer.writerow(
                    {
                        "query_id": query_id,
                        "seed_id": seed_id,
                        "cohort": row["cohort"],
                        "doi": row["doi"],
                        "source_title": row["title"],
                        "source_abstract": row["abstract"],
                        "reason_code": "SEARCH_TERM_MISSING",
                        "repair_reason": "English source wording added without consulting model outcomes",
                    }
                )
    manifest = {
        "artifact": "primary_seed_recall_query_repair",
        "database": str(database.resolve()),
        "input_queries_sha256": file_hash(queries),
        "repaired_queries_sha256": file_hash(repaired_path),
        "repair_evidence_sha256": file_hash(repair_path),
        "changed_query_ids": sorted(REPAIRS),
        "query_count": len(query_rows),
        "evidence_seed_count": len(evidence),
        "model_outcomes_consulted": False,
    }
    manifest_path = output_dir / "search_frame_seed_recall_repair_primary.manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_json(prepare(args.database, args.queries, args.output_dir)))


if __name__ == "__main__":
    main()
