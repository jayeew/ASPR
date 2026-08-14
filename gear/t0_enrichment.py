"""Exact OpenAlex enrichment for online Full-text-16 materialization."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Iterable, Optional, Tuple
from urllib.parse import quote

import requests

from .contracts import PaperIR
from .env import getenv
from .nature_multihorizon.t0_runtime_v3 import ReferenceT0, TargetT0Record


def _openalex_id(value: object) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    suffix = text.rsplit("/", 1)[-1].upper()
    if not suffix.startswith("W") or not suffix[1:].isdigit():
        return None
    return f"https://openalex.org/{suffix}"


def _field_id(work: Dict[str, Any]) -> Optional[str]:
    topic = work.get("primary_topic")
    field = topic.get("field") if isinstance(topic, dict) else None
    # The frozen v6 reference view stores OpenAlex field display names (for
    # example, "Medicine"), not field entity URLs.
    value = field.get("display_name") if isinstance(field, dict) else None
    return str(value) if value else None


class OpenAlexT0Enricher:
    """Resolve exact IDs/DOIs without inventing identity from title similarity."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: int = 60,
    ) -> None:
        self.api_key = str(api_key or getenv("OPENALEX_API_KEY")).strip()
        self.base_url = str(
            base_url
            or getenv("ASPR_OPENALEX_WORKS_URL", "https://api.openalex.org/works")
        ).rstrip("/")
        self.timeout_seconds = int(timeout_seconds)

    def build_target(
        self,
        paper_ir: PaperIR,
        *,
        evidence_date: Optional[date] = None,
    ) -> TargetT0Record:
        """Build a T0 record from an exact work or exact manuscript reference DOIs."""
        work = self._target_work(paper_ir)
        publication_year = self._publication_year(paper_ir, work, evidence_date)
        if work:
            reference_ids = tuple(
                value
                for raw in work.get("referenced_works") or []
                if (value := _openalex_id(raw)) is not None
            )
            references = self._reference_records(reference_ids)
            authorships = [
                item for item in work.get("authorships") or [] if isinstance(item, dict)
            ]
            author_ids = tuple(
                str(author["id"])
                for item in authorships
                if isinstance((author := item.get("author")), dict) and author.get("id")
            )
            country_count = int(work.get("countries_distinct_count") or 0)
            location = work.get("primary_location")
            source = location.get("source") if isinstance(location, dict) else None
            source_id = (
                str(source.get("id"))
                if isinstance(source, dict) and source.get("id")
                else None
            )
            return TargetT0Record(
                paper_id=_openalex_id(work.get("id")) or paper_ir.paper_id,
                publication_year=publication_year,
                title=str(paper_ir.metadata.title or work.get("display_name") or ""),
                author_ids=author_ids,
                author_count=len(authorships) or len(paper_ir.metadata.authors) or None,
                country_codes=tuple(
                    f"COUNTRY_{index}" for index in range(country_count)
                ),
                metadata_observed=True,
                source_id=source_id,
                references=references,
            )
        exact_reference_ids = []
        for reference in paper_ir.references:
            if not reference.doi:
                continue
            resolved = self._fetch_one(f"https://doi.org/{reference.doi}")
            work_id = _openalex_id(resolved.get("id")) if resolved else None
            if work_id:
                exact_reference_ids.append(work_id)
        return TargetT0Record(
            paper_id=paper_ir.paper_id,
            publication_year=publication_year,
            title=paper_ir.metadata.title,
            author_count=len(paper_ir.metadata.authors) or None,
            metadata_observed=False,
            references=self._reference_records(
                tuple(dict.fromkeys(exact_reference_ids))
            ),
        )

    def _target_work(self, paper_ir: PaperIR) -> Dict[str, Any]:
        work_id = _openalex_id(paper_ir.metadata.openalex_id)
        if work_id:
            return self._fetch_one(work_id)
        if paper_ir.metadata.doi:
            return self._fetch_one(f"https://doi.org/{paper_ir.metadata.doi}")
        return {}

    @staticmethod
    def _publication_year(
        paper_ir: PaperIR,
        work: Dict[str, Any],
        evidence_date: Optional[date],
    ) -> int:
        value = work.get("publication_year") if work else None
        if value:
            return int(value)
        boundary = (
            paper_ir.metadata.publication_date
            or paper_ir.metadata.submission_date
            or evidence_date
            or date.today()
        )
        return int(boundary.year)

    def _reference_records(self, work_ids: Tuple[str, ...]) -> Tuple[ReferenceT0, ...]:
        rows: Dict[str, Dict[str, Any]] = {}
        for chunk in _chunks(work_ids, 50):
            short_ids = "|".join(value.rsplit("/", 1)[-1] for value in chunk)
            params: Dict[str, str | int] = {
                "filter": f"openalex_id:{short_ids}",
                "per-page": len(chunk),
                "select": "id,publication_year,primary_topic",
                **self._key_params(),
            }
            response = requests.get(
                self.base_url,
                params=params,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip, deflate",
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            for item in response.json().get("results", []):
                if isinstance(item, dict) and (work_id := _openalex_id(item.get("id"))):
                    rows[work_id] = item
        return tuple(
            ReferenceT0(
                reference_id=work_id,
                publication_year=(
                    int(rows[work_id]["publication_year"])
                    if work_id in rows and rows[work_id].get("publication_year")
                    else None
                ),
                field_id=_field_id(rows[work_id]) if work_id in rows else None,
            )
            for work_id in work_ids
        )

    def _fetch_one(self, identifier: str) -> Dict[str, Any]:
        suffix = quote(str(identifier), safe=":/")
        response = requests.get(
            f"{self.base_url}/{suffix}",
            params=self._key_params(),
            headers={"Accept": "application/json", "Accept-Encoding": "gzip, deflate"},
            timeout=self.timeout_seconds,
        )
        if response.status_code == 404:
            return {}
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def _key_params(self) -> Dict[str, str]:
        return {"api_key": self.api_key} if self.api_key else {}


def _chunks(values: Tuple[str, ...], size: int) -> Iterable[Tuple[str, ...]]:
    for start in range(0, len(values), int(size)):
        yield values[start : start + int(size)]


__all__ = ["OpenAlexT0Enricher"]
