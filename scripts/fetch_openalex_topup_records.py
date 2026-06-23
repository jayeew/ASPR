from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from publication_corpus_v2 import (
        DEFAULT_COMPLETE_END_YEAR,
        fetch_openalex_works_for_query,
        normalize_openalex_id,
        slugify,
        utc_now,
        write_json,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised when imported as a package in tests.
    from scripts.publication_corpus_v2 import (
        DEFAULT_COMPLETE_END_YEAR,
        fetch_openalex_works_for_query,
        normalize_openalex_id,
        slugify,
        utc_now,
        write_json,
    )


def unique_query_records(
    domain: str,
    queries: Sequence[str],
    max_records_per_query: int,
    timeout_seconds: int,
    year_min: Optional[int] = None,
    year_max: int = DEFAULT_COMPLETE_END_YEAR,
) -> List[Dict[str, Any]]:
    """Fetch unique OpenAlex records for manual top-up queries."""
    records: List[Dict[str, Any]] = []
    seen: set[str] = set()
    domain_slug = slugify(domain)
    for query in queries:
        clean_query = str(query).strip()
        if not clean_query:
            continue
        works = fetch_openalex_works_for_query(
            clean_query,
            max_records=max_records_per_query,
            timeout_seconds=timeout_seconds,
        )
        for work in works:
            work_id = normalize_openalex_id(work.get("id"))
            if not work_id or work_id in seen:
                continue
            year = work.get("publication_year")
            try:
                year_int = int(year)
            except (TypeError, ValueError):
                continue
            if year_min is not None and year_int < int(year_min):
                continue
            if year_int > int(year_max):
                continue
            records.append({"domain": domain_slug, "query": clean_query, "work": work})
            seen.add(work_id)
    return records


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    """Write one JSON object per line and return the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch auditable OpenAlex top-up records for one domain.")
    parser.add_argument("--domain", required=True, help="Target corpus domain slug.")
    parser.add_argument("--query", action="append", required=True, help="OpenAlex search query; may be repeated.")
    parser.add_argument("--out-jsonl", type=Path, required=True, help="Output JSONL consumed by topup-openalex-works.")
    parser.add_argument("--manifest-path", type=Path, default=None, help="Optional fetch manifest path.")
    parser.add_argument("--max-records-per-query", type=int, default=500)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--year-min", type=int, default=None)
    parser.add_argument("--year-max", type=int, default=DEFAULT_COMPLETE_END_YEAR)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    records = unique_query_records(
        domain=args.domain,
        queries=args.query,
        max_records_per_query=args.max_records_per_query,
        timeout_seconds=args.timeout_seconds,
        year_min=args.year_min,
        year_max=args.year_max,
    )
    n_records = write_jsonl(args.out_jsonl, records)
    manifest_path = args.manifest_path or args.out_jsonl.with_suffix(".manifest.json")
    write_json(
        manifest_path,
        {
            "artifact_kind": "openalex_manual_topup_records",
            "created_at": utc_now(),
            "domain": slugify(args.domain),
            "queries": list(args.query),
            "max_records_per_query": int(args.max_records_per_query),
            "timeout_seconds": int(args.timeout_seconds),
            "year_min": args.year_min,
            "year_max": int(args.year_max),
            "out_jsonl": str(args.out_jsonl),
            "n_records": int(n_records),
        },
    )
    print(f"[openalex-topup-records] wrote {n_records} records to {args.out_jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
