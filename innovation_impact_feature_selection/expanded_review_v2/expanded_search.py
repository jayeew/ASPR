from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sqlite3
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent.parent
OUTPUT_DIR = ROOT / "outputs"
QUERY_PATH = ROOT / "search_queries_v2.json"
DEFAULT_DATABASE = OUTPUT_DIR / "expanded_search.sqlite3"
USER_AGENT = "ASPR-innovation-impact-systematic-census/2.0"


def utc_now() -> str:
    """Return a stable ISO UTC timestamp."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    """Read one JSON object.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON object.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write deterministic UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the streaming SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_environment_value(name: str) -> str:
    """Read one approved value from the process or a non-executable .env.

    Args:
        name: Exact environment variable name.

    Returns:
        The value or an empty string. Values are never logged.
    """
    process_value = os.environ.get(name, "").strip()
    if process_value:
        return process_value
    env_path = WORKSPACE_ROOT / ".env"
    if not env_path.exists():
        return ""
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        raw_name, raw_value = line.split("=", maxsplit=1)
        variable_name = raw_name.strip().removeprefix("export ").strip()
        if variable_name != name:
            continue
        value = raw_value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        return value.strip()
    return ""


def quote_search_term(term: str) -> str:
    """Quote a multiword database-search term."""
    escaped = term.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"' if " " in escaped else escaped


def compile_queries(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Compile the frozen concept matrix into exact query records."""
    queries: List[Dict[str, Any]] = []
    for domain in config["domains"]:
        context_terms = [str(value) for value in domain["context_terms"]]
        context_expression = " OR ".join(
            quote_search_term(value) for value in context_terms
        )
        for index, concept in enumerate(domain["concepts"], start=1):
            concept_terms = [str(value) for value in concept["terms"]]
            concept_expression = " OR ".join(
                quote_search_term(value) for value in concept_terms
            )
            expression = (
                f"({concept_expression}) AND ({context_expression})"
            )
            query_id = (
                f"{domain['domain_id']}__{index:02d}_{concept['concept_id']}"
            )
            payload = {
                "query_id": query_id,
                "domain_id": domain["domain_id"],
                "dimension_ids": list(domain["dimension_ids"]),
                "tier": domain["tier"],
                "concept_id": concept["concept_id"],
                "concept_terms": concept_terms,
                "context_terms": context_terms,
                "expression": expression,
            }
            payload["query_hash"] = sha256_bytes(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            )
            queries.append(payload)
    expected = int(config["query_compilation"]["expected_compiled_queries"])
    if len(queries) != expected:
        raise ValueError(
            f"Compiled {len(queries)} queries, but configuration expects "
            f"{expected}"
        )
    identifiers = [str(query["query_id"]) for query in queries]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Compiled query identifiers are not unique")
    return queries


def connect_database(path: Path) -> sqlite3.Connection:
    """Open the scalable retrieval database and initialize its schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=90)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 90000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS works (
            provider TEXT NOT NULL,
            record_key TEXT NOT NULL,
            provider_id TEXT NOT NULL,
            doi TEXT NOT NULL,
            title TEXT NOT NULL,
            publication_year INTEGER,
            work_type TEXT NOT NULL,
            abstract TEXT NOT NULL,
            source_url TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            PRIMARY KEY (provider, record_key)
        );
        CREATE TABLE IF NOT EXISTS query_hits (
            provider TEXT NOT NULL,
            query_id TEXT NOT NULL,
            record_key TEXT NOT NULL,
            rank INTEGER NOT NULL,
            PRIMARY KEY (provider, query_id, record_key),
            FOREIGN KEY (provider, record_key)
                REFERENCES works(provider, record_key)
        );
        CREATE TABLE IF NOT EXISTS query_runs (
            provider TEXT NOT NULL,
            query_id TEXT NOT NULL,
            query_hash TEXT NOT NULL,
            expression TEXT NOT NULL,
            cutoff_date TEXT NOT NULL,
            reported_total INTEGER,
            retrieved_rows INTEGER NOT NULL DEFAULT 0,
            unique_hits INTEGER NOT NULL DEFAULT 0,
            pages INTEGER NOT NULL DEFAULT 0,
            next_cursor TEXT NOT NULL DEFAULT '*',
            complete INTEGER NOT NULL DEFAULT 0,
            stopped_reason TEXT NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (provider, query_id)
        );
        CREATE TABLE IF NOT EXISTS snapshot_files (
            provider TEXT NOT NULL,
            path TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            modified_ns INTEGER NOT NULL,
            records_scanned INTEGER NOT NULL,
            matches_found INTEGER NOT NULL,
            complete INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (provider, path)
        );
        CREATE TABLE IF NOT EXISTS manual_screening (
            provider TEXT NOT NULL,
            record_key TEXT NOT NULL,
            reviewer_id TEXT NOT NULL,
            title_abstract_decision TEXT NOT NULL DEFAULT '',
            full_text_decision TEXT NOT NULL DEFAULT '',
            exclusion_reason TEXT NOT NULL DEFAULT '',
            indicator_terms TEXT NOT NULL DEFAULT '',
            dimension_ids TEXT NOT NULL DEFAULT '',
            formula_location TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            decided_at TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (provider, record_key, reviewer_id),
            FOREIGN KEY (provider, record_key)
                REFERENCES works(provider, record_key)
        );
        CREATE INDEX IF NOT EXISTS idx_works_doi ON works(doi);
        CREATE INDEX IF NOT EXISTS idx_hits_record
            ON query_hits(provider, record_key);
        CREATE INDEX IF NOT EXISTS idx_hits_query
            ON query_hits(provider, query_id);
        """
    )
    return connection


def connect_existing_database(path: Path) -> sqlite3.Connection:
    """Open an initialized database for one concurrent query worker."""
    connection = sqlite3.connect(path, timeout=90)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 90000")
    return connection


def fetch_json(
    url: str,
    retries: int = 5,
    timeout_seconds: int = 45,
) -> Dict[str, Any]:
    """Fetch a JSON object with bounded retry and Retry-After support."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("API response is not a JSON object")
            return payload
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            last_error = error
            if attempt + 1 >= retries:
                break
            retry_after = 0.0
            if isinstance(error, urllib.error.HTTPError):
                raw_retry_after = error.headers.get("Retry-After", "")
                try:
                    retry_after = float(raw_retry_after)
                except ValueError:
                    retry_after = 0.0
            delay = min(max(retry_after, float(2**attempt)), 45.0)
            time.sleep(delay)
    raise RuntimeError(f"Request failed after {retries} attempts: {last_error}")


def openalex_url(
    config: Mapping[str, Any],
    query: Mapping[str, Any],
    cursor: str,
    api_key: str,
    per_page: int,
) -> str:
    """Build one OpenAlex cursor request URL."""
    types = "|".join(str(value) for value in config["work_types"])
    parameters: Dict[str, Any] = {
        "search": query["expression"],
        "filter": (
            f"from_publication_date:{config['from_date']},"
            f"to_publication_date:{config['cutoff_date']},"
            f"type:{types}"
        ),
        "per-page": per_page,
        "cursor": cursor,
        "select": (
            "id,doi,display_name,publication_year,type,"
            "abstract_inverted_index,primary_location"
        ),
    }
    if api_key:
        parameters["api_key"] = api_key
    return "https://api.openalex.org/works?" + urllib.parse.urlencode(
        parameters
    )


def reconstruct_abstract(inverted_index: Any) -> str:
    """Reconstruct an OpenAlex abstract from its inverted index."""
    if not isinstance(inverted_index, dict):
        return ""
    positioned: List[tuple[int, str]] = []
    for token, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                positioned.append((position, str(token)))
    return " ".join(token for _, token in sorted(positioned))


def normalize_doi(value: Any) -> str:
    """Normalize a DOI or DOI URL."""
    doi = str(value or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
    return doi


def normalize_text(value: Any) -> str:
    """Normalize text for deterministic identity and phrase matching."""
    text = str(value or "").casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def record_key(
    doi: str,
    provider_id: str,
    title: str,
    year: Any,
) -> str:
    """Build a stable within-provider record identity."""
    if doi:
        return f"doi:{doi}"
    if provider_id:
        return f"id:{provider_id.casefold()}"
    fallback = f"{normalize_text(title)}|{year or ''}".encode("utf-8")
    return f"title_year:{sha256_bytes(fallback)}"


def openalex_record(item: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize one OpenAlex work."""
    doi = normalize_doi(item.get("doi"))
    provider_id = str(item.get("id") or "")
    title = str(item.get("display_name") or "")
    year = item.get("publication_year")
    location = item.get("primary_location")
    if not isinstance(location, dict):
        location = {}
    key = record_key(doi, provider_id, title, year)
    return {
        "provider": "OpenAlex",
        "record_key": key,
        "provider_id": provider_id,
        "doi": doi,
        "title": title,
        "publication_year": year if isinstance(year, int) else None,
        "work_type": str(item.get("type") or ""),
        "abstract": reconstruct_abstract(item.get("abstract_inverted_index")),
        "source_url": str(
            location.get("landing_page_url") or provider_id
        ),
        "raw_json": json.dumps(
            item,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "first_seen_at": utc_now(),
    }


def insert_work(
    connection: sqlite3.Connection,
    record: Mapping[str, Any],
) -> None:
    """Insert one normalized record without overwriting a frozen first copy."""
    connection.execute(
        """
        INSERT OR IGNORE INTO works (
            provider, record_key, provider_id, doi, title,
            publication_year, work_type, abstract, source_url,
            raw_json, first_seen_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["provider"],
            record["record_key"],
            record["provider_id"],
            record["doi"],
            record["title"],
            record["publication_year"],
            record["work_type"],
            record["abstract"],
            record["source_url"],
            record["raw_json"],
            record["first_seen_at"],
        ),
    )


def existing_run(
    connection: sqlite3.Connection,
    provider: str,
    query: Mapping[str, Any],
) -> sqlite3.Row | None:
    """Return an existing query-run row after verifying its query hash."""
    connection.row_factory = sqlite3.Row
    row = connection.execute(
        """
        SELECT * FROM query_runs
        WHERE provider = ? AND query_id = ?
        """,
        (provider, query["query_id"]),
    ).fetchone()
    if row and row["query_hash"] != query["query_hash"]:
        raise ValueError(
            f"{provider}/{query['query_id']} changed after retrieval began; "
            "use a new output database to preserve provenance"
        )
    return row


def upsert_query_run(
    connection: sqlite3.Connection,
    values: Mapping[str, Any],
) -> None:
    """Insert or update one query checkpoint."""
    connection.execute(
        """
        INSERT INTO query_runs (
            provider, query_id, query_hash, expression, cutoff_date,
            reported_total, retrieved_rows, unique_hits, pages,
            next_cursor, complete, stopped_reason, error, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(provider, query_id) DO UPDATE SET
            reported_total = excluded.reported_total,
            retrieved_rows = excluded.retrieved_rows,
            unique_hits = excluded.unique_hits,
            pages = excluded.pages,
            next_cursor = excluded.next_cursor,
            complete = excluded.complete,
            stopped_reason = excluded.stopped_reason,
            error = excluded.error,
            updated_at = excluded.updated_at
        """,
        (
            values["provider"],
            values["query_id"],
            values["query_hash"],
            values["expression"],
            values["cutoff_date"],
            values.get("reported_total"),
            values.get("retrieved_rows", 0),
            values.get("unique_hits", 0),
            values.get("pages", 0),
            values.get("next_cursor", "*"),
            int(bool(values.get("complete", False))),
            values.get("stopped_reason", ""),
            values.get("error", ""),
            utc_now(),
        ),
    )


def inventory_openalex(
    config: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
    api_key: str,
    delay_seconds: float,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    """Fetch result counts with an incremental resumable checkpoint."""
    config_digest = sha256_file(QUERY_PATH)
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    if checkpoint_path.exists():
        prior = read_json(checkpoint_path)
        if prior.get("query_config_sha256") != config_digest:
            raise ValueError(
                "Inventory query configuration changed; use a new manifest "
                "path to preserve provenance"
            )
        rows_by_id = {
            str(row["query_id"]): dict(row)
            for row in prior.get("queries", [])
            if not row.get("error")
        }

    def current_manifest() -> Dict[str, Any]:
        ordered_rows = [
            rows_by_id[str(query["query_id"])]
            for query in queries
            if str(query["query_id"]) in rows_by_id
        ]
        return {
            "schema_version": "2.0.0",
            "provider": "OpenAlex",
            "inventory_only": True,
            "query_count_expected": len(queries),
            "query_count_attempted": len(ordered_rows),
            "query_count_successful": sum(
                not row["error"] for row in ordered_rows
            ),
            "all_queries_inventory_complete": (
                len(ordered_rows) == len(queries)
                and all(not row["error"] for row in ordered_rows)
            ),
            "reported_total_before_cross_query_deduplication": sum(
                int(row["reported_total"])
                for row in ordered_rows
                if not row["error"]
            ),
            "queries_with_errors": sum(
                bool(row["error"]) for row in ordered_rows
            ),
            "retrieved_at": utc_now(),
            "query_config_sha256": config_digest,
            "queries": ordered_rows,
        }

    for query in queries:
        query_id = str(query["query_id"])
        if query_id in rows_by_id:
            continue
        url = openalex_url(config, query, "*", api_key, 1)
        try:
            payload = fetch_json(
                url,
                retries=2,
                timeout_seconds=15,
            )
            count = int(payload.get("meta", {}).get("count", 0))
            error = ""
        except RuntimeError as exc:
            count = 0
            error = str(exc)
        rows_by_id[query_id] = {
            **dict(query),
            "reported_total": count,
            "error": error,
        }
        write_json(checkpoint_path, current_manifest())
        time.sleep(delay_seconds)
    return current_manifest()


def retrieve_openalex_query(
    connection: sqlite3.Connection,
    config: Mapping[str, Any],
    query: Mapping[str, Any],
    api_key: str,
    delay_seconds: float,
    max_records: int | None,
) -> None:
    """Retrieve all cursor pages for one frozen OpenAlex query."""
    prior = existing_run(connection, "OpenAlex", query)
    if prior and bool(prior["complete"]):
        return
    cursor = str(prior["next_cursor"]) if prior else "*"
    retrieved = int(prior["retrieved_rows"]) if prior else 0
    pages = int(prior["pages"]) if prior else 0
    reported_total = prior["reported_total"] if prior else None
    while True:
        if max_records is not None and retrieved >= max_records:
            unique_hits = connection.execute(
                """
                SELECT COUNT(*) FROM query_hits
                WHERE provider = 'OpenAlex' AND query_id = ?
                """,
                (query["query_id"],),
            ).fetchone()[0]
            upsert_query_run(
                connection,
                {
                    "provider": "OpenAlex",
                    **dict(query),
                    "cutoff_date": config["cutoff_date"],
                    "reported_total": reported_total,
                    "retrieved_rows": retrieved,
                    "unique_hits": unique_hits,
                    "pages": pages,
                    "next_cursor": cursor,
                    "complete": False,
                    "stopped_reason": "operator_test_cap",
                },
            )
            connection.commit()
            return
        request_page_size = 100
        if max_records is not None:
            request_page_size = min(
                request_page_size,
                max(max_records - retrieved, 1),
            )
        url = openalex_url(
            config,
            query,
            cursor,
            api_key,
            request_page_size,
        )
        try:
            payload = fetch_json(url)
        except RuntimeError as exc:
            upsert_query_run(
                connection,
                {
                    "provider": "OpenAlex",
                    **dict(query),
                    "cutoff_date": config["cutoff_date"],
                    "reported_total": reported_total,
                    "retrieved_rows": retrieved,
                    "unique_hits": connection.execute(
                        """
                        SELECT COUNT(*) FROM query_hits
                        WHERE provider = 'OpenAlex' AND query_id = ?
                        """,
                        (query["query_id"],),
                    ).fetchone()[0],
                    "pages": pages,
                    "next_cursor": cursor,
                    "complete": False,
                    "stopped_reason": "request_error",
                    "error": str(exc),
                },
            )
            connection.commit()
            return
        meta = payload.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        if reported_total is None:
            reported_total = int(meta.get("count", 0))
        results = payload.get("results")
        if not isinstance(results, list):
            results = []
        next_cursor = str(meta.get("next_cursor") or "")
        for offset, raw_item in enumerate(results, start=1):
            if not isinstance(raw_item, dict):
                continue
            record = openalex_record(raw_item)
            insert_work(connection, record)
            connection.execute(
                """
                INSERT OR IGNORE INTO query_hits (
                    provider, query_id, record_key, rank
                ) VALUES ('OpenAlex', ?, ?, ?)
                """,
                (
                    query["query_id"],
                    record["record_key"],
                    retrieved + offset,
                ),
            )
        retrieved += len(results)
        pages += 1
        unique_hits = connection.execute(
            """
            SELECT COUNT(*) FROM query_hits
            WHERE provider = 'OpenAlex' AND query_id = ?
            """,
            (query["query_id"],),
        ).fetchone()[0]
        complete = not results or not next_cursor
        upsert_query_run(
            connection,
            {
                "provider": "OpenAlex",
                **dict(query),
                "cutoff_date": config["cutoff_date"],
                "reported_total": reported_total,
                "retrieved_rows": retrieved,
                "unique_hits": unique_hits,
                "pages": pages,
                "next_cursor": next_cursor or cursor,
                "complete": complete,
                "stopped_reason": "cursor_exhausted" if complete else "",
            },
        )
        connection.commit()
        if complete:
            return
        if next_cursor == cursor:
            raise RuntimeError(
                f"OpenAlex returned an unchanged cursor for "
                f"{query['query_id']}"
            )
        cursor = next_cursor
        time.sleep(delay_seconds)


def retrieve_openalex_worker(
    database_path: Path,
    config: Mapping[str, Any],
    query: Mapping[str, Any],
    api_key: str,
    delay_seconds: float,
    max_records: int | None,
) -> Dict[str, Any]:
    """Run one cursor chain in an isolated SQLite connection."""
    connection = connect_existing_database(database_path)
    try:
        retrieve_openalex_query(
            connection,
            config,
            query,
            api_key,
            delay_seconds,
            max_records,
        )
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT query_id, reported_total, retrieved_rows, unique_hits,
                   pages, complete, stopped_reason, error
            FROM query_runs
            WHERE provider = 'OpenAlex' AND query_id = ?
            """,
            (query["query_id"],),
        ).fetchone()
        if row is None:
            raise RuntimeError(
                f"No checkpoint created for {query['query_id']}"
            )
        return dict(row)
    finally:
        connection.close()


def database_manifest(
    database_path: Path,
    provider: str,
    config: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Summarize retrieval completion from a closed SQLite database."""
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    run_rows = [
        dict(row)
        for row in connection.execute(
            """
            SELECT * FROM query_runs
            WHERE provider = ?
            ORDER BY query_id
            """,
            (provider,),
        )
    ]
    unique_records = int(
        connection.execute(
            "SELECT COUNT(*) FROM works WHERE provider = ?",
            (provider,),
        ).fetchone()[0]
    )
    query_hits = int(
        connection.execute(
            "SELECT COUNT(*) FROM query_hits WHERE provider = ?",
            (provider,),
        ).fetchone()[0]
    )
    connection.close()
    return {
        "schema_version": "2.0.0",
        "provider": provider,
        "query_count_expected": len(queries),
        "query_count_started": len(run_rows),
        "query_count_complete": sum(bool(row["complete"]) for row in run_rows),
        "all_queries_complete": (
            len(run_rows) == len(queries)
            and all(bool(row["complete"]) for row in run_rows)
        ),
        "unique_records": unique_records,
        "query_record_links": query_hits,
        "database_path": str(database_path.resolve()),
        "database_sha256": sha256_file(database_path),
        "query_config_sha256": sha256_file(QUERY_PATH),
        "cutoff_date": config["cutoff_date"],
        "generated_at": utc_now(),
        "queries": run_rows,
    }


def run_openalex(args: argparse.Namespace) -> None:
    """Run OpenAlex inventory or complete cursor retrieval."""
    config = read_json(QUERY_PATH)
    queries = compile_queries(config)
    api_key = local_environment_value("OPENALEX_API_KEY")
    if args.inventory_only:
        manifest = inventory_openalex(
            config,
            queries,
            api_key,
            args.delay_seconds,
            args.manifest,
        )
        write_json(args.manifest, manifest)
        print(
            f"OpenAlex inventory: {len(queries)} queries, "
            f"{manifest['reported_total_before_cross_query_deduplication']} "
            f"reported hits before deduplication."
        )
        return
    if not api_key:
        raise RuntimeError(
            "Full OpenAlex retrieval requires OPENALEX_API_KEY in the "
            "environment. Inventory-only mode can be used for a probe."
        )
    initializer = connect_database(args.database)
    initializer.close()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                retrieve_openalex_worker,
                args.database,
                config,
                query,
                api_key,
                args.delay_seconds,
                args.max_records_per_query,
            ): query
            for query in queries
        }
        for future in as_completed(futures):
            query = futures[future]
            result = future.result()
            state = "complete" if result["complete"] else "incomplete"
            print(
                f"{query['query_id']}: {state}, "
                f"{result['unique_hits']} unique hits",
                flush=True,
            )
    finalizer = connect_database(args.database)
    finalizer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finalizer.close()
    manifest = database_manifest(
        args.database,
        "OpenAlex",
        config,
        queries,
    )
    write_json(args.manifest, manifest)
    print(
        f"OpenAlex retrieval: {manifest['query_count_complete']}/"
        f"{manifest['query_count_expected']} queries complete; "
        f"{manifest['unique_records']} unique records."
    )


def run_compile(args: argparse.Namespace) -> None:
    """Freeze the fully expanded query text before database access."""
    config = read_json(QUERY_PATH)
    queries = compile_queries(config)
    payload = {
        "schema_version": "2.0.0",
        "cutoff_date": config["cutoff_date"],
        "from_date": config["from_date"],
        "work_types": config["work_types"],
        "crossref_types": config["crossref_types"],
        "query_count": len(queries),
        "query_config_sha256": sha256_file(QUERY_PATH),
        "queries": queries,
    }
    write_json(args.output, payload)
    print(f"Frozen {len(queries)} compiled queries at {args.output}.")


def json_items(value: Any) -> Iterator[Mapping[str, Any]]:
    """Yield Crossref works from common snapshot JSON containers."""
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(value, dict):
        return
    if isinstance(value.get("items"), list):
        yield from json_items(value["items"])
        return
    message = value.get("message")
    if isinstance(message, dict) and isinstance(message.get("items"), list):
        yield from json_items(message["items"])
        return
    if "DOI" in value or "title" in value:
        yield value


def stream_json_lines(handle: Iterable[bytes]) -> Iterator[Mapping[str, Any]]:
    """Yield records from a byte-oriented JSONL stream."""
    for raw_line in handle:
        line = raw_line.decode("utf-8").strip()
        if not line:
            continue
        yield from json_items(json.loads(line))


def iter_crossref_file(path: Path) -> Iterator[Mapping[str, Any]]:
    """Yield Crossref records from supported snapshot shard formats."""
    lower_name = path.name.casefold()
    if lower_name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path, mode="r:*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                member_name = member.name.casefold()
                if member_name.endswith((".jsonl", ".ndjson")):
                    yield from stream_json_lines(extracted)
                elif member_name.endswith(".json"):
                    yield from json_items(json.load(extracted))
        return
    if lower_name.endswith((".jsonl.gz", ".ndjson.gz")):
        with gzip.open(path, "rb") as handle:
            yield from stream_json_lines(handle)
        return
    if lower_name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield from json_items(json.load(handle))
        return
    if lower_name.endswith((".jsonl", ".ndjson")):
        with path.open("rb") as handle:
            yield from stream_json_lines(handle)
        return
    if lower_name.endswith(".json"):
        with path.open("r", encoding="utf-8") as handle:
            yield from json_items(json.load(handle))
        return
    raise ValueError(f"Unsupported snapshot shard: {path}")


def snapshot_paths(root: Path) -> List[Path]:
    """List supported snapshot shards deterministically."""
    if root.is_file():
        return [root]
    suffixes = (
        ".json",
        ".jsonl",
        ".ndjson",
        ".json.gz",
        ".jsonl.gz",
        ".ndjson.gz",
        ".tar",
        ".tar.gz",
        ".tgz",
    )
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and any(path.name.casefold().endswith(value) for value in suffixes)
    )


def crossref_year(item: Mapping[str, Any]) -> int | None:
    """Extract a Crossref publication year."""
    for field in ("published", "published-print", "published-online", "issued"):
        value = item.get(field)
        if not isinstance(value, dict):
            continue
        parts = value.get("date-parts")
        try:
            year = int(parts[0][0])
        except (IndexError, TypeError, ValueError):
            continue
        return year
    return None


def first_text(value: Any) -> str:
    """Return the first text value from a Crossref scalar/list field."""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def matching_query_ids(
    item: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
) -> List[str]:
    """Apply the frozen exact phrase clauses to Crossref title/abstract text."""
    title = first_text(item.get("title"))
    abstract = str(item.get("abstract") or "")
    text = normalize_text(f"{title} {abstract}")
    matches: List[str] = []
    for query in queries:
        concept_match = any(
            normalize_text(term) in text for term in query["concept_terms"]
        )
        context_match = any(
            normalize_text(term) in text for term in query["context_terms"]
        )
        if concept_match and context_match:
            matches.append(str(query["query_id"]))
    return matches


def crossref_record(item: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize one Crossref snapshot work."""
    doi = normalize_doi(item.get("DOI"))
    title = first_text(item.get("title"))
    year = crossref_year(item)
    provider_id = doi
    key = record_key(doi, provider_id, title, year)
    return {
        "provider": "CrossrefSnapshot",
        "record_key": key,
        "provider_id": provider_id,
        "doi": doi,
        "title": title,
        "publication_year": year,
        "work_type": str(item.get("type") or ""),
        "abstract": str(item.get("abstract") or ""),
        "source_url": str(item.get("URL") or ""),
        "raw_json": json.dumps(
            item,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "first_seen_at": utc_now(),
    }


def crossref_in_scope(
    item: Mapping[str, Any],
    config: Mapping[str, Any],
) -> bool:
    """Apply the frozen Crossref type and publication-date boundary."""
    if str(item.get("type") or "") not in set(config["crossref_types"]):
        return False
    year = crossref_year(item)
    if year is None:
        return False
    minimum_year = int(str(config["from_date"])[:4])
    maximum_year = int(str(config["cutoff_date"])[:4])
    return minimum_year <= year <= maximum_year


def snapshot_file_complete(
    connection: sqlite3.Connection,
    path: Path,
) -> bool:
    """Check whether an unchanged snapshot shard has already completed."""
    stat = path.stat()
    row = connection.execute(
        """
        SELECT size_bytes, modified_ns, complete
        FROM snapshot_files
        WHERE provider = 'CrossrefSnapshot' AND path = ?
        """,
        (str(path.resolve()),),
    ).fetchone()
    return bool(
        row
        and int(row[0]) == stat.st_size
        and int(row[1]) == stat.st_mtime_ns
        and bool(row[2])
    )


def scan_crossref_shard(
    connection: sqlite3.Connection,
    path: Path,
    config: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    """Scan one full Crossref snapshot shard transactionally."""
    stat = path.stat()
    scanned = 0
    matches = 0
    rank_by_query: Dict[str, int] = {}
    connection.execute("BEGIN")
    try:
        for item in iter_crossref_file(path):
            scanned += 1
            if not crossref_in_scope(item, config):
                continue
            query_ids = matching_query_ids(item, queries)
            if not query_ids:
                continue
            record = crossref_record(item)
            insert_work(connection, record)
            for query_id in query_ids:
                rank_by_query[query_id] = rank_by_query.get(query_id, 0) + 1
                connection.execute(
                    """
                    INSERT OR IGNORE INTO query_hits (
                        provider, query_id, record_key, rank
                    ) VALUES ('CrossrefSnapshot', ?, ?, ?)
                    """,
                    (
                        query_id,
                        record["record_key"],
                        rank_by_query[query_id],
                    ),
                )
                matches += 1
        connection.execute(
            """
            INSERT INTO snapshot_files (
                provider, path, size_bytes, modified_ns, records_scanned,
                matches_found, complete, updated_at
            ) VALUES ('CrossrefSnapshot', ?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(provider, path) DO UPDATE SET
                size_bytes = excluded.size_bytes,
                modified_ns = excluded.modified_ns,
                records_scanned = excluded.records_scanned,
                matches_found = excluded.matches_found,
                complete = 1,
                updated_at = excluded.updated_at
            """,
            (
                str(path.resolve()),
                stat.st_size,
                stat.st_mtime_ns,
                scanned,
                matches,
                utc_now(),
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return scanned, matches


def finalize_crossref_runs(
    connection: sqlite3.Connection,
    config: Mapping[str, Any],
    queries: Sequence[Mapping[str, Any]],
) -> None:
    """Mark every local exact-match Crossref query complete after full scan."""
    for query in queries:
        count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM query_hits
                WHERE provider = 'CrossrefSnapshot' AND query_id = ?
                """,
                (query["query_id"],),
            ).fetchone()[0]
        )
        upsert_query_run(
            connection,
            {
                "provider": "CrossrefSnapshot",
                **dict(query),
                "cutoff_date": config["cutoff_date"],
                "reported_total": count,
                "retrieved_rows": count,
                "unique_hits": count,
                "pages": 0,
                "next_cursor": "",
                "complete": True,
                "stopped_reason": "local_snapshot_exhausted",
            },
        )
    connection.commit()


def run_crossref_snapshot(args: argparse.Namespace) -> None:
    """Scan every record in a local Crossref snapshot."""
    config = read_json(QUERY_PATH)
    queries = compile_queries(config)
    paths = snapshot_paths(args.snapshot)
    if not paths:
        raise ValueError(f"No supported snapshot files found under {args.snapshot}")
    connection = connect_database(args.database)
    try:
        for path in paths:
            if snapshot_file_complete(connection, path):
                continue
            scanned, matches = scan_crossref_shard(
                connection,
                path,
                config,
                queries,
            )
            print(f"Scanned {path.name}: {scanned} records, {matches} links.")
        finalize_crossref_runs(connection, config, queries)
    finally:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.close()
    manifest = database_manifest(
        args.database,
        "CrossrefSnapshot",
        config,
        queries,
    )
    manifest["snapshot_root"] = str(args.snapshot.resolve())
    manifest["snapshot_files"] = len(paths)
    write_json(args.manifest, manifest)
    print(
        f"Crossref snapshot: {manifest['unique_records']} unique matched "
        f"records from {len(paths)} fully scanned shard(s)."
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Exhaustive, resumable evidence retrieval for publication-time "
            "innovation and potential-impact indicators."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "frozen_queries_v2.json",
    )
    openalex = subparsers.add_parser("openalex")
    openalex.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
    )
    openalex.add_argument(
        "--manifest",
        type=Path,
        default=OUTPUT_DIR / "openalex_manifest.json",
    )
    openalex.add_argument("--inventory-only", action="store_true")
    openalex.add_argument(
        "--delay-seconds",
        type=float,
        default=0.15,
    )
    openalex.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Concurrent query chains; each cursor chain remains sequential.",
    )
    openalex.add_argument(
        "--max-records-per-query",
        type=int,
        default=None,
        help=(
            "Test-only cap. Any capped query is explicitly marked incomplete."
        ),
    )
    crossref = subparsers.add_parser("crossref-snapshot")
    crossref.add_argument("--snapshot", type=Path, required=True)
    crossref.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
    )
    crossref.add_argument(
        "--manifest",
        type=Path,
        default=OUTPUT_DIR / "crossref_snapshot_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    """Run the requested provider-specific retrieval workflow."""
    args = parse_args()
    if args.command == "compile":
        run_compile(args)
        return
    if args.command == "openalex":
        if args.delay_seconds < 0:
            raise ValueError("--delay-seconds cannot be negative")
        if args.workers < 1 or args.workers > 6:
            raise ValueError("--workers must be between 1 and 6")
        if (
            args.max_records_per_query is not None
            and args.max_records_per_query < 1
        ):
            raise ValueError("--max-records-per-query must be positive")
        run_openalex(args)
        return
    if args.command == "crossref-snapshot":
        run_crossref_snapshot(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
