#!/usr/bin/env python3
"""Build the post-freeze formal screening pool from current evidence routes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Mapping
from datetime import date
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
    from .resolve_seed_indexability import _abstract
    from .search_frame import OPENALEX_ELIGIBILITY_FILTER
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
    from resolve_seed_indexability import _abstract  # type: ignore[no-redef]
    from search_frame import OPENALEX_ELIGIBILITY_FILTER  # type: ignore[no-redef]


def _normalized_doi(record: Mapping[str, Any]) -> str:
    return str(record.get("doi") or "").lower().removeprefix("https://doi.org/")


def _work(record: Mapping[str, Any], route: str) -> dict[str, Any]:
    return {
        "doi": _normalized_doi(record),
        "openalex_id": str(record.get("id") or ""),
        "title": str(record.get("display_name") or record.get("title") or ""),
        "publication_year": record.get("publication_year"),
        "language": str(record.get("language") or ""),
        "work_type": str(record.get("type") or ""),
        "abstract": _abstract(record),
        "source_route": route,
    }


def _eligible(record: Mapping[str, Any], cutoff: date) -> bool:
    if str(record.get("language") or "") != "en":
        return False
    if str(record.get("type") or "") not in {"article", "conference-paper", "review"}:
        return False
    publication_date = str(record.get("publication_date") or "")
    if publication_date:
        try:
            return date.fromisoformat(publication_date) <= cutoff
        except ValueError:
            return False
    year = record.get("publication_year")
    return year is not None and int(year) < cutoff.year


def _raw_cache_record(engine: EvidenceProtocol, work_id: str) -> dict[str, Any] | None:
    row = engine.connection.execute(
        "SELECT raw_json FROM provider_cache_records WHERE provider='OpenAlex' "
        "AND (record_key=? OR doi=(SELECT doi FROM provider_cache_records "
        "WHERE provider='OpenAlex' AND record_key=? LIMIT 1)) ORDER BY record_key LIMIT 1",
        (work_id, work_id),
    ).fetchone()
    if not row:
        return None
    raw = json.loads(row[0])
    return raw if isinstance(raw, dict) else None


def build_formal_pool(database: Path, output_dir: Path) -> dict[str, Any]:
    client = OpenAlexClient()
    if not client.configured_slots:
        raise ProtocolError("Formal online increment requires OpenAlex key slots")
    with EvidenceProtocol(database, output_dir) as engine:
        engine.initialize()
        frame = engine.get_metadata("search_frame", {})
        if not frame or frame.get("hash") != engine._search_frame_hash():
            raise ProtocolError("Formal pool requires an intact frozen search frame")
        cutoff = date.fromisoformat(engine.get_metadata("protocol")["cutoff_date"])
        routes: dict[str, set[str]] = defaultdict(set)
        query_ids: dict[str, set[str]] = defaultdict(set)
        payloads: dict[str, str] = {}
        raw_by_work: dict[str, dict[str, Any]] = {}

        discovery_ids = [
            row[0]
            for row in engine.connection.execute(
                "SELECT work_id FROM discovery_final WHERE decision IN ('include','uncertain') ORDER BY work_id"
            )
        ]
        for discovery_id in discovery_ids:
            raw = _raw_cache_record(engine, discovery_id)
            if not raw or not _eligible(raw, cutoff):
                continue
            work_id, _ = engine.ingest_work(_work(raw, "current_saturation_evidence"))
            routes[work_id].add("current_saturation_evidence")
            payloads[work_id] = sha256_text(canonical_json(raw))
            raw_by_work[work_id] = raw

        for seed in engine.connection.execute(
            "SELECT DISTINCT work_id FROM seed_recall WHERE indexability='indexable' ORDER BY work_id"
        ):
            routes[str(seed[0])].add("recalled_seed")
            row = engine.connection.execute(
                "SELECT payload_hash FROM works WHERE work_id=?", (seed[0],)
            ).fetchone()
            payloads[str(seed[0])] = str(row[0]) if row else ""

        for physical in engine.connection.execute(
            "SELECT * FROM physical_queries WHERE active=1 ORDER BY physical_query_id"
        ):
            started = utc_now()
            run_id = stable_id(
                "FORMALRUN", physical["physical_query_id"], frame["hash"]
            )
            try:
                page = client.fetch_search_page(
                    str(physical["request_expression"]),
                    OPENALEX_ELIGIBILITY_FILTER,
                    per_page=200,
                )
            except ProtocolError as error:
                engine.connection.execute(
                    "INSERT OR REPLACE INTO search_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        "formal-online-increment",
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
                    "formal-online-increment",
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
            for record in page.records:
                if not _eligible(record, cutoff):
                    continue
                work_id, _ = engine.ingest_work(
                    _work(record, "frozen_query_online_increment")
                )
                routes[work_id].add("frozen_query_online_increment")
                query_ids[work_id].add(str(physical["query_id"]))
                payloads[work_id] = sha256_text(canonical_json(record))
                raw_by_work[work_id] = dict(record)

        engine.connection.execute("DELETE FROM formal_pool_records")
        for work_id in sorted(routes):
            engine.connection.execute(
                "INSERT INTO formal_pool_records VALUES(?,?,?,?,?)",
                (
                    work_id,
                    canonical_json(sorted(routes[work_id])),
                    canonical_json(sorted(query_ids[work_id])),
                    sha256_text(f"{frame['hash']}|{work_id}"),
                    payloads[work_id],
                ),
            )
        openalex_to_work = {
            row[0].upper(): row[1]
            for row in engine.connection.execute(
                "SELECT openalex_id,work_id FROM works WHERE openalex_id<>''"
            )
        }
        for citing_id, raw in raw_by_work.items():
            for cited in raw.get("referenced_works") or []:
                cited_id = openalex_to_work.get(str(cited).upper())
                if cited_id:
                    engine.connection.execute(
                        "INSERT OR IGNORE INTO citations VALUES(?,?,?)",
                        (citing_id, cited_id, "formal_pool_reference"),
                    )
        engine.connection.commit()
        output_dir.mkdir(parents=True, exist_ok=True)
        screening_path = output_dir / "formal_screening_blind.csv"
        rows = list(
            engine.connection.execute(
                "SELECT p.*,w.doi,w.openalex_id,w.title,w.publication_year,w.language,"
                "w.work_type,w.abstract,w.source_route FROM formal_pool_records p "
                "JOIN works w USING(work_id) ORDER BY p.stable_rank"
            )
        )
        with screening_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(rows[0].keys() if rows else [])
            writer.writerows(tuple(row) for row in rows)
        manifest = {
            "artifact": "formal_screening_pool",
            "search_frame_hash": frame["hash"],
            "pool_count": len(rows),
            "current_saturation_candidate_count": len(discovery_ids),
            "online_query_count": 13,
            "online_page_size": 200,
            "online_route_is_increment_not_complete_hit_download": True,
            "configured_slots": client.configured_slots,
            "secret_material_persisted": False,
            "screening_sha256": file_hash(screening_path),
        }
        manifest_path = output_dir / "formal_screening_blind.manifest.json"
        manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
        engine.set_metadata("formal_pool", manifest)
        engine.set_metadata("stage", "formal-pool")
        engine.connection.commit()
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(canonical_json(build_formal_pool(args.database, args.output_dir)))


if __name__ == "__main__":
    main()
