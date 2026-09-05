"""Prepare the fixed paper input consumed independently by Graph and GEAR."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

from gear.env import getenv, getenv_list

from .review_contracts import InnovationPaperInput


def normalize_doi(value: str) -> str:
    text = value.strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if text.casefold().startswith(prefix):
            text = text[len(prefix):]
    return text.strip()


def recover_abstract(index: object) -> str:
    if not isinstance(index, dict):
        return ""
    positioned: list[tuple[int, str]] = []
    for word, positions in index.items():
        if isinstance(positions, list):
            positioned.extend((int(position), str(word)) for position in positions)
    return " ".join(word for _, word in sorted(positioned))


def fetch_openalex(doi: str, api_key: str | None = None) -> dict[str, object]:
    identifier = urllib.parse.quote(f"https://doi.org/{normalize_doi(doi)}", safe="")
    suffix = f"?api_key={urllib.parse.quote(api_key)}" if api_key else ""
    request = urllib.request.Request(
        f"https://api.openalex.org/works/{identifier}{suffix}",
        headers={"User-Agent": "ASPR-GEAR innovation input preparation"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def prepare_input(
    *, paper_path: Path, paper_id: str, title: str, doi: str,
    publication_date: date, cutoff_date: date, venue: str | None = None,
    abstract_text: str | None = None,
) -> InnovationPaperInput:
    keys = getenv_list("OPENALEX_API_KEYS")
    api_key = keys[0] if keys else (getenv("OPENALEX_API_KEY") or None)
    work = fetch_openalex(doi, api_key)
    abstract = (abstract_text or recover_abstract(work.get("abstract_inverted_index"))).strip()
    if not abstract:
        raise ValueError(f"{paper_id} 没有可用摘要")
    return InnovationPaperInput(
        paper_id=paper_id, paper_path=paper_path.resolve(), title=title,
        doi=doi, venue=venue, publication_date=publication_date,
        cutoff_date=cutoff_date, abstract_text=abstract,
        abstract_source="provided" if abstract_text else "openalex",
        openalex_work_id=str(work.get("id") or "") or None,
        reference_work_ids=[str(x) for x in work.get("referenced_works", [])],
    )
