#!/usr/bin/env python3
"""Import, validate, and seed-test the adjudicated OpenAlex search frame."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .core import (
        EvidenceProtocol,
        ProtocolError,
        canonical_json,
        file_hash,
        sha256_text,
        stable_id,
        utc_now,
    )
    from .providers import OpenAlexClient
except ImportError:
    from core import (  # type: ignore[no-redef]
        EvidenceProtocol,
        ProtocolError,
        canonical_json,
        file_hash,
        sha256_text,
        stable_id,
        utc_now,
    )
    from providers import OpenAlexClient  # type: ignore[no-redef]

csv.field_size_limit(sys.maxsize)

WILDCARD_EXPANSIONS = {
    "altmetric": ("altmetric", "altmetrics"),
    "bibliometric": ("bibliometric", "bibliometrics"),
    "breakthrough": ("breakthrough", "breakthroughs"),
    "download": ("download", "downloads", "downloaded", "downloading"),
    "evaluat": (
        "evaluate",
        "evaluates",
        "evaluated",
        "evaluating",
        "evaluation",
        "evaluations",
        "evaluative",
    ),
    "indicator": ("indicator", "indicators"),
    "measure": (
        "measure",
        "measures",
        "measured",
        "measuring",
        "measurement",
        "measurements",
    ),
    "patent": ("patent", "patents", "patented"),
    "predict": (
        "predict",
        "predicts",
        "predicted",
        "predicting",
        "prediction",
        "predictions",
        "predictor",
        "predictors",
        "predictive",
    ),
    "preregistr": (
        "preregister",
        "preregistered",
        "preregistration",
        "preregistrations",
    ),
    "reference": ("reference", "references", "referencing"),
    "replicat": (
        "replicate",
        "replicates",
        "replicated",
        "replicating",
        "replication",
        "replications",
    ),
    "reproducib": ("reproducible", "reproducibility"),
    "reviewer": ("reviewer", "reviewers"),
    "validat": (
        "validate",
        "validates",
        "validated",
        "validating",
        "validation",
        "validations",
        "validity",
    ),
}
PAPER_OBJECT_BLOCK = (
    '(paper OR article OR publication OR manuscript OR "research study") AND '
)
OPENALEX_ELIGIBILITY_FILTER = (
    "to_publication_date:2026-07-28,language:en," "type:article|conference-paper|review"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def expand_wildcards(expression: str) -> str:
    """Translate registered wildcard stems to explicit OpenAlex search terms."""

    def replace(match: re.Match[str]) -> str:
        stem = match.group(1)
        values = WILDCARD_EXPANSIONS.get(stem.casefold())
        if not values:
            raise ProtocolError(f"Unregistered OpenAlex wildcard stem: {stem}*")
        return "(" + " OR ".join(values) + ")"

    expanded = re.sub(r"\b([A-Za-z][A-Za-z-]*)\*", replace, expression)
    if "*" in expanded or "?" in expanded:
        raise ProtocolError(
            "OpenAlex physical query retains unsupported wildcard syntax"
        )
    return expanded


def physical_openalex_expression(expression: str) -> str:
    """Move the paper-object clause to the OpenAlex type filter."""
    translated = expression.strip()
    translated = translated.removeprefix(PAPER_OBJECT_BLOCK)
    return expand_wildcards(translated)


def physical_split_expression(split: object) -> str:
    if isinstance(split, str):
        return physical_openalex_expression(split)
    if not isinstance(split, dict) or not isinstance(split.get("search"), str):
        raise ProtocolError("PRESS physical split requires a search expression")
    if split.get("filter") != OPENALEX_ELIGIBILITY_FILTER:
        raise ProtocolError(
            "PRESS physical split conflicts with protocol eligibility filter"
        )
    return physical_openalex_expression(split["search"])


def _review_session(
    engine: EvidenceProtocol,
    role: str,
    input_path: Path,
    output_path: Path,
    reason: str,
) -> None:
    session_id = stable_id("RS", role, file_hash(input_path), file_hash(output_path))
    engine.connection.execute(
        "INSERT OR REPLACE INTO review_sessions VALUES(?,?,?,?,?,?,?,?,?)",
        (
            session_id,
            stable_id("RUN", session_id),
            role,
            file_hash(input_path),
            file_hash(output_path),
            (
                "Codex independent session"
                if role != "Primary AI"
                else "Codex primary session"
            ),
            str(output_path.resolve()),
            reason,
            utc_now(),
        ),
    )


def import_search_frame(
    database: Path,
    review_dir: Path,
    query_file: Path,
    press_file: Path,
) -> dict[str, Any]:
    adjudicated_path = review_dir / "search_frame_adjudicated_coded.csv"
    domain_path = review_dir / "final_search_domains.csv"
    coding_path = review_dir / "search_frame_coding_blind.csv"
    adjudicated = read_rows(adjudicated_path)
    domains = read_rows(domain_path)
    queries = read_rows(query_file)
    press = read_rows(press_file)
    if len(adjudicated) != 596 or len(domains) != 13 or len(queries) != 13:
        raise ProtocolError("Search-frame artifact cardinality changed unexpectedly")
    press_by_query = {row["query_id"]: row for row in press}
    if set(press_by_query) != {row["query_id"] for row in queries}:
        raise ProtocolError(
            "Final PRESS rows do not exactly cover final logical queries"
        )
    if any(
        (row.get("post_revision_status") or row["status"]) != "pass" for row in press
    ):
        raise ProtocolError("Final PRESS contains unresolved queries")

    active = [
        row for row in adjudicated if row["adjudicated_disposition"] != "exclude_noise"
    ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in active:
        grouped[row["adjudicated_canonical_group"]].append(row)
    family_ids = {group: stable_id("TF", group) for group in sorted(grouped)}
    with EvidenceProtocol(database, database.parent) as engine:
        engine.initialize()
        if engine.get_metadata("search_frame"):
            raise ProtocolError("Frozen search frame is immutable")
        for table in (
            "physical_queries",
            "logical_queries",
            "search_domains",
            "term_families",
        ):
            engine.connection.execute(f"DELETE FROM {table}")
        for group, rows in sorted(grouped.items()):
            roles = sorted({row["adjudicated_term_role"] for row in rows})
            engine.connection.execute(
                "INSERT INTO term_families VALUES(?,?,?,?,?,?)",
                (
                    family_ids[group],
                    group,
                    roles[0] if len(roles) == 1 else "multi_role",
                    canonical_json(sorted(row["family_id"] for row in rows)),
                    "adjudicated_active",
                    "; ".join(sorted({row["adjudication_reason"] for row in rows})),
                ),
            )
        domain_groups = {
            row["domain_id"]: json.loads(row["canonical_groups_json"])
            for row in domains
        }
        active_by_domain: dict[str, set[str]] = defaultdict(set)
        source_by_domain: dict[str, set[str]] = defaultdict(set)
        for row in active:
            active_by_domain[row["adjudicated_domain"]].add(
                family_ids[row["adjudicated_canonical_group"]]
            )
            source_by_domain[row["adjudicated_domain"]].update(
                json.loads(row["source_work_ids_json"])
            )
        for row in domains:
            expected = {family_ids[group] for group in domain_groups[row["domain_id"]]}
            if expected != active_by_domain[row["domain_id"]]:
                raise ProtocolError(f"Domain family closure failed: {row['domain_id']}")
            engine.connection.execute(
                "INSERT INTO search_domains VALUES(?,?,?,?,?,?,?,?)",
                (
                    row["domain_id"],
                    row["label"],
                    row["definition"],
                    canonical_json(sorted(expected)),
                    canonical_json(sorted(source_by_domain[row["domain_id"]])),
                    "reviewed",
                    "reviewed",
                    "active",
                ),
            )
        physical_count = 0
        for query in queries:
            expression = query.get("revised_expression") or query["expression"]
            engine.connection.execute(
                "INSERT INTO logical_queries VALUES(?,?,?,?,?,?,?,?)",
                (
                    query["query_id"],
                    query["domain_id"],
                    expression,
                    query["semantic_expression"],
                    query["evidence_family_ids_json"],
                    "pass",
                    "active",
                    0,
                ),
            )
            reviewed_splits = json.loads(
                press_by_query[query["query_id"]]["physical_split_json"]
            )
            if not reviewed_splits:
                raise ProtocolError("Final PRESS lacks a physical-plan review")
            physical_expression = physical_openalex_expression(expression)
            if len(physical_expression.encode("utf-8")) > 3500:
                raise ProtocolError(
                    f"Logical query requires a proven equivalent split: {query['query_id']}"
                )
            splits: list[object] = [physical_expression]
            for number, split in enumerate(splits, start=1):
                physical_count += 1
                physical_id = f"{query['query_id']}_OA_{number:02d}"
                engine.connection.execute(
                    "INSERT INTO physical_queries VALUES(?,?,?,?,?,?)",
                    (
                        physical_id,
                        query["query_id"],
                        "OpenAlex",
                        physical_split_expression(split),
                        "Deterministic unsplit translation: full logical expression under byte limit; paper object enforced by protocol work-type filter",
                        1,
                    ),
                )
        _review_session(
            engine,
            "Primary AI",
            coding_path,
            review_dir / "search_frame_primary_coded.csv",
            "Blind primary terminology coding",
        )
        _review_session(
            engine,
            "Independent Reviewer AI",
            coding_path,
            review_dir / "search_frame_independent_coded.csv",
            "Blind independent terminology coding",
        )
        _review_session(
            engine,
            "Adjudicator AI",
            adjudicated_path,
            query_file,
            "Disposition, grouping, domain, and query adjudication",
        )
        _review_session(
            engine,
            "Independent PRESS Reviewer AI",
            query_file,
            press_file,
            "Final PRESS review with no seed or model outcomes",
        )
        engine.set_metadata("stage", "freeze-search-validation")
        engine.connection.commit()
        return {
            "term_families": len(grouped),
            "K": len(domains),
            "Q": len(queries),
            "P": physical_count,
            "query_file_sha256": file_hash(query_file),
            "press_file_sha256": file_hash(press_file),
        }


def validate_seed_recall(database: Path, batch_size: int = 40) -> dict[str, Any]:
    client = OpenAlexClient()
    if not client.configured_slots:
        raise ProtocolError(
            "Seed recall validation requires configured OpenAlex key slots"
        )
    with EvidenceProtocol(database, database.parent) as engine:
        engine.initialize()
        seed_rows = list(
            engine.connection.execute(
                "SELECT s.seed_id,s.cohort,s.work_id,"
                "COALESCE(NULLIF(w.openalex_id,''),p.provider_id) AS openalex_id "
                "FROM seed_recall s JOIN seed_inputs i USING(seed_id) "
                "JOIN works w USING(work_id) LEFT JOIN provider_cache_records p "
                "ON p.provider='OpenAlex' AND p.record_key='doi:'||i.doi "
                "WHERE s.indexability='indexable' ORDER BY s.seed_id"
            )
        )
        if not seed_rows:
            raise ProtocolError("No indexable seeds available for recall validation")
        id_to_seeds: dict[str, set[str]] = defaultdict(set)
        for row in seed_rows:
            openalex_id = str(row["openalex_id"]).upper().rsplit("/", 1)[-1]
            if not re.fullmatch(r"W\d+", openalex_id):
                raise ProtocolError(
                    "Every indexable seed requires one valid OpenAlex work ID"
                )
            id_to_seeds[openalex_id].add(str(row["seed_id"]))
        matched: dict[str, set[str]] = defaultdict(set)
        completed_runs = 0
        for physical in engine.connection.execute(
            "SELECT * FROM physical_queries WHERE active=1 ORDER BY physical_query_id"
        ):
            ids = sorted(id_to_seeds)
            for batch_no, start in enumerate(range(0, len(ids), batch_size), start=1):
                batch = ids[start : start + batch_size]
                run_id = stable_id("SEEDRUN", physical["physical_query_id"], batch_no)
                started = utc_now()
                filter_expression = (
                    "openalex_id:" + "|".join(batch) + "," + OPENALEX_ELIGIBILITY_FILTER
                )
                try:
                    page = client.fetch_search_page(
                        str(physical["request_expression"]), filter_expression
                    )
                except ProtocolError as error:
                    engine.connection.execute(
                        "INSERT OR REPLACE INTO search_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            run_id,
                            "seed-recall",
                            physical["query_id"],
                            "OpenAlex",
                            "",
                            "",
                            "blocked",
                            1,
                            "",
                            type(error).__name__,
                            started,
                            utc_now(),
                        ),
                    )
                    engine.connection.commit()
                    raise
                response_hash = sha256_text(canonical_json(page.response_hash_source))
                engine.connection.execute(
                    "INSERT OR REPLACE INTO search_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        "seed-recall",
                        physical["query_id"],
                        "OpenAlex",
                        page.key_slot,
                        page.next_cursor,
                        "complete",
                        1,
                        response_hash,
                        "",
                        started,
                        utc_now(),
                    ),
                )
                completed_runs += 1
                for record in page.records:
                    openalex_id = str(record.get("id") or "").upper().rsplit("/", 1)[-1]
                    for seed_id in id_to_seeds.get(openalex_id, set()):
                        matched[seed_id].add(str(physical["query_id"]))
        cohort_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "recalled": 0}
        )
        missing: list[str] = []
        for row in seed_rows:
            links = sorted(matched[row["seed_id"]])
            recalled = bool(links)
            cohort_counts[row["cohort"]]["total"] += 1
            cohort_counts[row["cohort"]]["recalled"] += int(recalled)
            if not recalled:
                missing.append(str(row["seed_id"]))
            engine.connection.execute(
                "UPDATE seed_recall SET recall_status=?,reason_code=?,matched_query_ids_json=? WHERE seed_id=?",
                (
                    "recalled" if recalled else "missed",
                    (
                        "ONLINE_OPENALEX_QUERY_MATCH"
                        if recalled
                        else "SEARCH_TERM_MISSING"
                    ),
                    canonical_json(links),
                    row["seed_id"],
                ),
            )
        engine.connection.commit()
        return {
            "configured_slots": client.configured_slots,
            "secret_material_persisted": False,
            "completed_requests": completed_runs,
            "cohorts": dict(sorted(cohort_counts.items())),
            "missing_seed_ids": missing,
            "all_recalled": not missing,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import")
    importer.add_argument("--review-dir", type=Path, required=True)
    importer.add_argument("--query-file", type=Path, required=True)
    importer.add_argument("--press-file", type=Path, required=True)
    recall = commands.add_parser("seed-recall")
    recall.add_argument("--batch-size", type=int, default=40)
    args = parser.parse_args()
    if args.command == "import":
        result = import_search_frame(
            args.database, args.review_dir, args.query_file, args.press_file
        )
    else:
        result = validate_seed_recall(args.database, args.batch_size)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
