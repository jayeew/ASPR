"""Acquire post-cutoff citation contexts as observable claim-adoption evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

S2_API = "https://api.semanticscholar.org/graph/v1/paper"
CITATION_FIELDS = (
    "contexts,intents,isInfluential,citingPaper.paperId,citingPaper.title,"
    "citingPaper.year,citingPaper.publicationDate,citingPaper.fieldsOfStudy,"
    "citingPaper.s2FieldsOfStudy,citingPaper.externalIds"
)


def acquire_contexts(
    benchmark_manifest: Path,
    output_dir: Path,
    *,
    horizon_years: int = 5,
    max_citations_per_paper: int = 1000,
) -> dict[str, Any]:
    """Fetch real future citation contexts; no semantic label is inferred here."""
    payload = json.loads(benchmark_manifest.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(output_dir / "citation_contexts.jsonl")
    rows = _filter_existing_rows(rows, cases, horizon_years=horizon_years)
    paper_rows = _read_jsonl(output_dir / "citation_context_papers.jsonl")
    resolved = {
        str(row["paper_id"])
        for row in paper_rows
        if str(row.get("fetch_status", "")).startswith("resolved")
    }
    session = requests.Session()
    session.headers["User-Agent"] = "ASPR-GEAR-research-validation/1.0"
    for case in cases:
        if str(case.get("paper_id")) in resolved:
            continue
        contexts, status = _fetch_case(
            session,
            case,
            horizon_years=horizon_years,
            max_citations=max_citations_per_paper,
        )
        existing_contexts = {str(row["context_id"]) for row in rows}
        rows.extend(
            row for row in contexts if str(row["context_id"]) not in existing_contexts
        )
        paper_id = str(case.get("paper_id"))
        paper_rows = [row for row in paper_rows if str(row["paper_id"]) != paper_id]
        paper_rows.append(
            {
                "paper_id": paper_id,
                "fetch_status": status,
                "context_rows": len(contexts),
            }
        )
        _write_jsonl(output_dir / "citation_contexts.jsonl", rows)
        _write_jsonl(output_dir / "citation_context_papers.jsonl", paper_rows)
        time.sleep(0.25)
    paper_rows = _refresh_context_counts(paper_rows, rows)
    contexts_path = output_dir / "citation_contexts.jsonl"
    papers_path = output_dir / "citation_context_papers.jsonl"
    _write_jsonl(contexts_path, rows)
    _write_jsonl(papers_path, paper_rows)
    summary = {
        **_summary(cases, rows, paper_rows, horizon_years),
        "citation_contexts_sha256": "sha256:"
        + hashlib.sha256(contexts_path.read_bytes()).hexdigest(),
        "citation_context_papers_sha256": "sha256:"
        + hashlib.sha256(papers_path.read_bytes()).hexdigest(),
    }
    (output_dir / "citation_context_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _fetch_case(
    session: requests.Session,
    case: dict[str, Any],
    *,
    horizon_years: int,
    max_citations: int,
) -> tuple[list[dict[str, Any]], str]:
    doi = _normalize_doi((case.get("metadata") or {}).get("doi"))
    identifiers = [f"DOI:{doi}"] if doi else []
    mag_id = _openalex_mag_id(case.get("paper_id"))
    if mag_id is not None:
        identifiers.append(f"MAG:{mag_id}")
    if not identifiers:
        return [], "missing_doi"
    cutoff = date.fromisoformat(str(case.get("cutoff") or ""))
    last_status = "error:identifier_resolution_failed"
    for identifier in identifiers:
        endpoint = f"{S2_API}/{quote(identifier, safe=':')}/citations"
        rows, status = _fetch_endpoint(
            session,
            endpoint,
            case,
            cutoff=cutoff,
            horizon_years=horizon_years,
            max_citations=max_citations,
        )
        if status.startswith("resolved"):
            suffix = "_via_mag" if identifier.startswith("MAG:") else ""
            return rows, status + suffix
        last_status = status
    return [], last_status


def _fetch_endpoint(
    session: requests.Session,
    endpoint: str,
    case: dict[str, Any],
    *,
    cutoff: date,
    horizon_years: int,
    max_citations: int,
) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while offset < max_citations:
        result, status = _request_page(session, endpoint, offset)
        if result is None:
            return rows, status
        for citation in result.get("data", []):
            rows.extend(
                _context_rows(
                    case,
                    citation,
                    cutoff=cutoff,
                    horizon_years=horizon_years,
                )
            )
        next_offset = result.get("next")
        if next_offset is None or int(next_offset) <= offset:
            return rows, "resolved"
        offset = int(next_offset)
    return rows, "resolved_truncated"


def _openalex_mag_id(value: Any) -> str | None:
    identifier = str(value or "").rstrip("/").rsplit("/", maxsplit=1)[-1]
    if identifier.startswith("W") and identifier[1:].isdigit():
        return identifier[1:]
    return None


def _request_page(
    session: requests.Session, endpoint: str, offset: int
) -> tuple[dict[str, Any] | None, str]:
    for attempt in range(4):
        try:
            params: dict[str, str | int] = {
                "fields": CITATION_FIELDS,
                "limit": 100,
                "offset": offset,
            }
            response = session.get(
                endpoint,
                params=params,
                timeout=45,
            )
            if response.status_code == 429:
                time.sleep(2 ** (attempt + 1))
                continue
            if response.status_code == 404:
                return None, "not_found"
            response.raise_for_status()
            return response.json(), "resolved"
        except (requests.RequestException, ValueError) as exc:
            if attempt == 3:
                return None, f"error:{type(exc).__name__}"
            time.sleep(2 ** (attempt + 1))
    return None, "error:retry_exhausted"


def _context_rows(
    case: dict[str, Any],
    citation: dict[str, Any],
    *,
    cutoff: date,
    horizon_years: int,
) -> list[dict[str, Any]]:
    citing = citation.get("citingPaper") or {}
    year = citing.get("year")
    publication_date = _parse_date(citing.get("publicationDate"))
    resolved_year = (
        int(year)
        if year is not None
        else publication_date.year if publication_date is not None else None
    )
    if not _within_horizon(
        publication_date,
        resolved_year,
        cutoff=cutoff,
        horizon_years=horizon_years,
    ):
        return []
    if resolved_year is None:
        return []
    contexts = [str(value).strip() for value in citation.get("contexts") or []]
    output: list[dict[str, Any]] = []
    for index, context in enumerate(contexts):
        if not context:
            continue
        context_id = hashlib.sha256(
            f"{case.get('paper_id')}|{citing.get('paperId')}|{index}|{context}".encode()
        ).hexdigest()[:24]
        output.append(
            {
                "context_id": f"S2C-{context_id}",
                "paper_id": case.get("paper_id"),
                "citing_paper_id": citing.get("paperId"),
                "citing_title": citing.get("title"),
                "citing_year": resolved_year,
                "citing_publication_date": citing.get("publicationDate"),
                "citing_fields": citing.get("fieldsOfStudy") or [],
                "citing_s2_fields": citing.get("s2FieldsOfStudy") or [],
                "citing_external_ids": citing.get("externalIds") or {},
                "context": context,
                "intents": citation.get("intents") or [],
                "is_influential": bool(citation.get("isInfluential")),
                "data_role": "future_outcome_only",
            }
        )
    return output


def _filter_existing_rows(
    rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    *,
    horizon_years: int,
) -> list[dict[str, Any]]:
    cutoffs = {
        str(case["paper_id"]): date.fromisoformat(str(case["cutoff"])) for case in cases
    }
    return [
        row
        for row in rows
        if str(row.get("paper_id")) in cutoffs
        and _within_horizon(
            _parse_date(row.get("citing_publication_date")),
            int(row["citing_year"]) if row.get("citing_year") is not None else None,
            cutoff=cutoffs[str(row["paper_id"])],
            horizon_years=horizon_years,
        )
    ]


def _refresh_context_counts(
    papers: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        paper_id = str(row["paper_id"])
        counts[paper_id] = counts.get(paper_id, 0) + 1
    return [
        {**paper, "context_rows": counts.get(str(paper["paper_id"]), 0)}
        for paper in papers
    ]


def _within_horizon(
    publication_date: date | None,
    publication_year: int | None,
    *,
    cutoff: date,
    horizon_years: int,
) -> bool:
    end = _add_years(cutoff, horizon_years)
    if publication_date is not None:
        return cutoff < publication_date <= end
    if publication_year is None:
        return False
    return cutoff.year < publication_year < end.year


def _add_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)


def _parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


def _normalize_doi(value: Any) -> str:
    text = str(value or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.casefold().startswith(prefix):
            return text[len(prefix) :]
    return text


def _summary(
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    papers: list[dict[str, Any]],
    horizon_years: int,
) -> dict[str, Any]:
    covered = {str(row["paper_id"]) for row in rows}
    return {
        "contract": "gear_claim_adoption_contexts_v1",
        "data_role": "future_outcome_only",
        "retrieved_at": datetime.now(UTC).isoformat(),
        "horizon_years": horizon_years,
        "papers_requested": len(cases),
        "papers_resolved": sum(
            str(row["fetch_status"]).startswith("resolved") for row in papers
        ),
        "papers_with_contexts": len(covered),
        "citation_contexts": len(rows),
        "source": "Semantic Scholar Academic Graph citations endpoint",
        "time_window": "strictly_after_cutoff_through_exact_cutoff_plus_horizon",
        "missing_date_policy": "exclude_ambiguous_cutoff_and_horizon_boundary_years",
        "papers_truncated": sum(
            str(row["fetch_status"]).startswith("resolved_truncated") for row in papers
        ),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon-years", type=int, default=5)
    parser.add_argument("--max-citations-per-paper", type=int, default=1000)
    args = parser.parse_args()
    summary = acquire_contexts(
        args.benchmark_manifest,
        args.output_dir,
        horizon_years=args.horizon_years,
        max_citations_per_paper=args.max_citations_per_paper,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["acquire_contexts"]
