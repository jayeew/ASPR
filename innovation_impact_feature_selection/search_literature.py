from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
USER_AGENT = "ASPR-innovation-impact-evidence-census/1.0"
PROVIDERS = ("Crossref", "OpenAlex")


def fetch_json(url: str, retries: int = 3) -> Dict[str, Any]:
    """Fetch a public JSON API with bounded exponential retries.

    Args:
        url: Public API URL.
        retries: Maximum request attempts.

    Returns:
        Parsed JSON object.
    """
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("API response is not a JSON object")
            return payload
        except (OSError, ValueError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(float(2**attempt))
    raise RuntimeError(f"request failed after {retries} attempts: {last_error}")


def first_year(date_parts: Any) -> str:
    """Extract a year from a Crossref date-parts field."""
    try:
        return str(date_parts[0][0])
    except (IndexError, KeyError, TypeError):
        return ""


def crossref_rows(
    query: Mapping[str, Any],
    cutoff_date: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """Retrieve one ranked Crossref page for a frozen query."""
    parameters = urllib.parse.urlencode(
        {
            "query.bibliographic": query["query"],
            "filter": f"until-pub-date:{cutoff_date}",
            "rows": limit,
            "select": "DOI,title,published,type,URL",
        }
    )
    payload = fetch_json(f"https://api.crossref.org/works?{parameters}")
    items = payload.get("message", {}).get("items", [])
    rows: List[Dict[str, Any]] = []
    for rank, item in enumerate(items, start=1):
        titles = item.get("title") or [""]
        rows.append(
            {
                "provider": "Crossref",
                "query_id": query["query_id"],
                "theme": query["theme"],
                "query": query["query"],
                "rank": rank,
                "work_id": item.get("DOI", ""),
                "doi": item.get("DOI", ""),
                "title": titles[0],
                "year": first_year(item.get("published", {}).get("date-parts")),
                "work_type": item.get("type", ""),
                "url": item.get("URL", ""),
            }
        )
    return rows


def openalex_rows(
    query: Mapping[str, Any],
    cutoff_date: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """Retrieve one ranked OpenAlex page for a frozen query."""
    parameters = urllib.parse.urlencode(
        {
            "search": query["query"],
            "filter": (
                "from_publication_date:1900-01-01,"
                f"to_publication_date:{cutoff_date}"
            ),
            "per-page": limit,
        }
    )
    payload = fetch_json(f"https://api.openalex.org/works?{parameters}")
    rows: List[Dict[str, Any]] = []
    for rank, item in enumerate(payload.get("results", []), start=1):
        identifiers = item.get("ids") or {}
        doi_url = identifiers.get("doi") or ""
        primary_location = item.get("primary_location") or {}
        rows.append(
            {
                "provider": "OpenAlex",
                "query_id": query["query_id"],
                "theme": query["theme"],
                "query": query["query"],
                "rank": rank,
                "work_id": item.get("id", ""),
                "doi": doi_url.removeprefix("https://doi.org/"),
                "title": item.get("display_name", ""),
                "year": item.get("publication_year", ""),
                "work_type": item.get("type", ""),
                "url": (
                    primary_location.get("landing_page_url")
                    or item.get("id", "")
                ),
            }
        )
    return rows


def run_provider(
    provider: str,
    query: Mapping[str, Any],
    cutoff_date: str,
    limit: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    """Run one provider-query pair and retain a machine-readable failure."""
    try:
        if provider == "Crossref":
            return crossref_rows(query, cutoff_date, limit), None
        if provider == "OpenAlex":
            return openalex_rows(query, cutoff_date, limit), None
        raise ValueError(f"Unknown provider: {provider}")
    except (RuntimeError, ValueError) as error:
        return [], {
            "provider": provider,
            "query_id": query["query_id"],
            "error_type": type(error).__name__,
            "error": str(error),
        }


def deduplicate(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate within provider-query rankings."""
    seen: set[tuple[str, str, str]] = set()
    output: List[Dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        identity = str(row["doi"]).lower() or str(row["work_id"]).lower()
        key = (str(row["provider"]), str(row["query_id"]), identity)
        if key not in seen:
            output.append(row)
            seen.add(key)
    return output


def write_snapshot(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write the deterministic ranked search snapshot."""
    fields = (
        "provider",
        "query_id",
        "theme",
        "query",
        "rank",
        "work_id",
        "doi",
        "title",
        "year",
        "work_type",
        "url",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_errors(
    path: Path,
    cutoff_date: str,
    limit: int,
    errors: Sequence[Mapping[str, Any]],
) -> None:
    """Write provider failures without changing the successful snapshot."""
    payload = {
        "schema_version": "1.0.0",
        "cutoff_date": cutoff_date,
        "limit_per_provider": limit,
        "errors": list(errors),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Freeze Crossref and OpenAlex rankings for the publication-time "
            "innovation and potential-impact evidence census."
        )
    )
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "search_snapshot.csv",
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=ROOT / "search_errors.json",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
    )
    return parser.parse_args()


def main() -> None:
    """Run all frozen queries and save rankings plus explicit failures."""
    args = parse_args()
    if args.limit < 1 or args.limit > 200:
        raise ValueError("--limit must be between 1 and 200")
    search_config = json.loads(
        (ROOT / "search_queries.json").read_text(encoding="utf-8")
    )
    cutoff_date = str(search_config["cutoff_date"])
    queries = search_config["queries"]
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    futures = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        for query in queries:
            for provider in args.providers:
                futures.append(
                    executor.submit(
                        run_provider,
                        provider,
                        query,
                        cutoff_date,
                        args.limit,
                    )
                )
        for future in as_completed(futures):
            result_rows, error = future.result()
            rows.extend(result_rows)
            if error is not None:
                errors.append(error)
    ordered_rows = sorted(
        deduplicate(rows),
        key=lambda row: (
            row["query_id"],
            row["provider"],
            int(row["rank"]),
            row["work_id"],
        ),
    )
    ordered_errors = sorted(
        errors,
        key=lambda row: (row["query_id"], row["provider"]),
    )
    write_snapshot(args.output, ordered_rows)
    write_errors(args.errors, cutoff_date, args.limit, ordered_errors)
    successful_pairs = len(queries) * len(args.providers) - len(ordered_errors)
    print(
        f"Wrote {len(ordered_rows)} ranked records from "
        f"{successful_pairs}/{len(queries) * len(args.providers)} "
        f"provider-query pairs to {args.output}"
    )
    if ordered_errors:
        print(f"Recorded {len(ordered_errors)} provider failures in {args.errors}")


if __name__ == "__main__":
    main()
