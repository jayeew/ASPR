from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


ROOT = Path(__file__).resolve().parent
CUTOFF_DATE = "2026-07-28"
USER_AGENT = "ASPR-innovation-evidence-census/1.0 (reproducible academic search)"


def fetch_json(url: str, retries: int = 2) -> Dict[str, Any]:
    """Fetch a JSON object with bounded retries.

    Args:
        url: Public API URL.
        retries: Maximum attempts.

    Returns:
        Parsed JSON response.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                value = json.loads(response.read().decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
            return value
        except (OSError, ValueError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def first_year(parts: Any) -> str:
    """Extract a publication year from Crossref date-parts."""
    try:
        return str(parts[0][0])
    except (IndexError, KeyError, TypeError):
        return ""


def crossref_rows(query: Mapping[str, Any], limit: int) -> List[Dict[str, Any]]:
    """Fetch one ranked Crossref result page."""
    params = urllib.parse.urlencode(
        {
            "query.bibliographic": query["query"],
            "filter": f"until-pub-date:{CUTOFF_DATE}",
            "rows": limit,
            "select": "DOI,title,published,type,URL",
        }
    )
    payload = fetch_json(f"https://api.crossref.org/works?{params}")
    items = payload.get("message", {}).get("items", [])
    rows: List[Dict[str, Any]] = []
    for rank, item in enumerate(items, start=1):
        title_value = item.get("title") or [""]
        rows.append(
            {
                "provider": "Crossref",
                "query_id": query["query_id"],
                "query": query["query"],
                "rank": rank,
                "work_id": item.get("DOI", ""),
                "doi": item.get("DOI", ""),
                "title": title_value[0],
                "year": first_year(item.get("published", {}).get("date-parts")),
                "work_type": item.get("type", ""),
                "url": item.get("URL", ""),
            }
        )
    return rows


def openalex_rows(query: Mapping[str, Any], limit: int) -> List[Dict[str, Any]]:
    """Fetch one ranked OpenAlex result page."""
    params = urllib.parse.urlencode(
        {
            "search": query["query"],
            "filter": (
                "from_publication_date:1900-01-01,"
                f"to_publication_date:{CUTOFF_DATE}"
            ),
            "per-page": limit,
        }
    )
    payload = fetch_json(f"https://api.openalex.org/works?{params}")
    rows: List[Dict[str, Any]] = []
    for rank, item in enumerate(payload.get("results", []), start=1):
        ids = item.get("ids") or {}
        doi_url = ids.get("doi") or ""
        doi = doi_url.removeprefix("https://doi.org/")
        primary_location = item.get("primary_location") or {}
        rows.append(
            {
                "provider": "OpenAlex",
                "query_id": query["query_id"],
                "query": query["query"],
                "rank": rank,
                "work_id": item.get("id", ""),
                "doi": doi,
                "title": item.get("display_name", ""),
                "year": item.get("publication_year", ""),
                "work_type": item.get("type", ""),
                "url": primary_location.get("landing_page_url") or item.get("id", ""),
            }
        )
    return rows


def deduplicate(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate within each query while retaining both provider rankings."""
    seen: set[tuple[str, str, str]] = set()
    output: List[Dict[str, Any]] = []
    for raw_row in rows:
        row = dict(raw_row)
        identity = str(row["doi"]).lower() or str(row["work_id"]).lower()
        key = (str(row["provider"]), str(row["query_id"]), identity)
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def write_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write a frozen ranked search snapshot."""
    fields = (
        "provider",
        "query_id",
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Re-run the Crossref/OpenAlex novelty-measure search."
    )
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "search_snapshot.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Run all frozen queries and save the ranked result snapshot."""
    args = parse_args()
    if args.limit < 1 or args.limit > 200:
        raise ValueError("--limit must be between 1 and 200")
    queries = json.loads(
        (ROOT / "search_queries.json").read_text(encoding="utf-8")
    )["queries"]
    rows: List[Dict[str, Any]] = []
    jobs = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for query in queries:
            jobs.append(executor.submit(crossref_rows, query, args.limit))
            jobs.append(executor.submit(openalex_rows, query, args.limit))
        for job in as_completed(jobs):
            rows.extend(job.result())
    result = sorted(
        deduplicate(rows),
        key=lambda row: (
            row["query_id"],
            row["provider"],
            int(row["rank"]),
            row["work_id"],
        ),
    )
    write_rows(args.output, result)
    print(f"Wrote {len(result)} ranked records to {args.output}")


if __name__ == "__main__":
    main()
