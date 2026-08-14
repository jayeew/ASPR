"""Stateless bibliographic search adapter used by the GEAR evidence lane."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List, Optional

import requests

from .env import getenv, getenv_int


class OpenScholar:
    """Minimal Semantic Scholar/OpenAlex client without the legacy reviewer."""

    def __init__(self, args: Any) -> None:
        self.s2_api_key = str(getattr(args, "s2_api_key", "") or getenv("S2_API_KEY"))
        self.openalex_api_key = self._first_key(
            str(
                getattr(args, "openalex_api_key", "")
                or getenv("OPENALEX_API_KEY")
                or getenv("OPENALEX_API_KEYS")
            )
        )
        self.and_search = bool(getattr(args, "and_search", False))
        self.retrieval_provider = (
            str(
                getattr(args, "retrieval_provider", "")
                or getenv("ASPR_RETRIEVAL_PROVIDER", "semantic_scholar")
            )
            .strip()
            .casefold()
        )
        self.s2_url = getenv(
            "ASPR_S2_SEARCH_URL",
            "https://api.semanticscholar.org/graph/v1/paper/search",
        )
        self.openalex_url = getenv(
            "ASPR_OPENALEX_WORKS_URL", "https://api.openalex.org/works"
        )
        self.search_limit = max(1, min(getenv_int("ASPR_S2_SEARCH_LIMIT", 100), 100))
        self.openalex_limit = max(
            1, min(getenv_int("ASPR_OPENALEX_PER_PAGE", 100), 200)
        )
        self.last_query_audits: List[Dict[str, Any]] = []

    @staticmethod
    def _first_key(raw: str) -> str:
        values = [item for item in re.split(r"[,;\s]+", raw) if item]
        return values[0] if values else ""

    def search_query(
        self,
        query: str,
        *,
        provider: Optional[str] = None,
        date_to: Optional[date] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search with an API-level and local publication cutoff."""
        selected = str(provider or self.retrieval_provider).strip().casefold()
        if selected in {"openalex", "oa"}:
            rows = self._search_openalex(query, date_to=date_to, limit=limit)
            works = [self._format_openalex(item) for item in rows]
        else:
            rows = self._search_semantic_scholar(query, date_to=date_to, limit=limit)
            works = [self._format_semantic_scholar(item) for item in rows]
        return self._filter_cutoff(works, date_to)

    def fetch_work(self, work_id: str) -> Dict[str, Any]:
        """Fetch one work in the normalized GEAR retrieval schema."""
        identifier = str(work_id or "").strip()
        if not identifier:
            return {}
        if self._is_openalex(identifier):
            suffix = identifier.rsplit("/", 1)[-1]
            response = requests.get(
                f"{self.openalex_url}/{suffix}",
                params=self._openalex_key_params(),
                headers={"Accept-Encoding": "gzip, deflate"},
                timeout=60,
            )
            if response.status_code != 200:
                return {}
            return self._format_openalex(response.json())
        response = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{identifier}",
            params={"fields": self._semantic_fields(include_references=True)},
            headers=self._semantic_headers(),
            timeout=60,
        )
        if response.status_code != 200:
            return {}
        return self._format_semantic_scholar(response.json())

    def fetch_neighbors(
        self,
        work_id: str,
        direction: str = "references",
        *,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch one bounded citation hop."""
        maximum = min(100, max(1, int(limit or 12)))
        identifier = str(work_id or "").strip()
        if self._is_openalex(identifier):
            if direction == "citations":
                suffix = identifier.rsplit("/", 1)[-1]
                params: Dict[str, Any] = {
                    "filter": f"cites:{suffix}",
                    "per-page": maximum,
                    "sort": "publication_date:asc",
                    **self._openalex_key_params(),
                }
                response = requests.get(
                    self.openalex_url,
                    params=params,
                    headers={"Accept-Encoding": "gzip, deflate"},
                    timeout=60,
                )
                if response.status_code != 200:
                    return []
                return [
                    self._format_openalex(item)
                    for item in response.json().get("results", [])[:maximum]
                    if isinstance(item, dict)
                ]
            seed = self.fetch_work(identifier)
            identifiers = list(seed.get("referenced_works") or [])[:maximum]
            return [
                work for work in (self.fetch_work(item) for item in identifiers) if work
            ]
        edge = "citations" if direction == "citations" else "references"
        response = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{identifier}/{edge}",
            params={"fields": self._semantic_fields(), "limit": maximum},
            headers=self._semantic_headers(),
            timeout=60,
        )
        if response.status_code != 200:
            return []
        key = "citingPaper" if edge == "citations" else "citedPaper"
        return [
            self._format_semantic_scholar(item[key])
            for item in response.json().get("data", [])
            if isinstance(item, dict) and isinstance(item.get(key), dict)
        ]

    def _search_openalex(
        self, query: str, *, date_to: Optional[date], limit: Optional[int]
    ) -> List[Dict[str, Any]]:
        filters = ["from_publication_date:1800-01-01"]
        if date_to is not None:
            filters.append(f"to_publication_date:{date_to.isoformat()}")
        params: Dict[str, Any] = {
            "search": str(query).strip(),
            "filter": ",".join(filters),
            "sort": "cited_by_count:desc",
            "per-page": min(200, max(1, int(limit or self.openalex_limit))),
            **self._openalex_key_params(),
        }
        response = requests.get(
            self.openalex_url,
            params=params,
            headers={"Accept": "application/json", "Accept-Encoding": "gzip, deflate"},
            timeout=60,
        )
        self.last_query_audits.append(
            {"source": "openalex", "query": query, "status_code": response.status_code}
        )
        if response.status_code != 200:
            raise RuntimeError(f"OpenAlex request failed: {response.status_code}")
        rows = response.json().get("results", [])
        return rows if isinstance(rows, list) else []

    def _search_semantic_scholar(
        self, query: str, *, date_to: Optional[date], limit: Optional[int]
    ) -> List[Dict[str, Any]]:
        terms = [str(query).strip()]
        separator = " + " if self.and_search else " | "
        formatted = separator.join(f'"{term}"' for term in terms if term)
        params = {
            "query": formatted,
            "fields": self._semantic_fields(),
            "year": f"1800-{date_to.year}" if date_to else "1800-",
            "limit": min(100, max(1, int(limit or self.search_limit))),
            "sort": "citationCount:desc",
        }
        attempts = []
        if self.s2_api_key:
            attempts.append(self._semantic_headers())
        attempts.append({})
        last_status = 0
        for headers in attempts:
            response = requests.get(
                self.s2_url, params=params, headers=headers, timeout=60
            )
            last_status = response.status_code
            self.last_query_audits.append(
                {
                    "source": "semantic_scholar",
                    "query": formatted,
                    "status_code": response.status_code,
                    "used_key": bool(headers),
                }
            )
            if response.status_code == 200:
                rows = response.json().get("data", [])
                return rows if isinstance(rows, list) else []
            if response.status_code not in {401, 403}:
                break
        raise RuntimeError(f"Semantic Scholar request failed: {last_status}")

    @staticmethod
    def _semantic_fields(*, include_references: bool = False) -> str:
        fields = (
            "paperId,title,year,publicationDate,authors.name,abstract,venue,"
            "citationCount,url,externalIds,isOpenAccess,openAccessPdf,"
            "fieldsOfStudy,s2FieldsOfStudy"
        )
        return fields + (",references.paperId" if include_references else "")

    def _semantic_headers(self) -> Dict[str, str]:
        return {"x-api-key": self.s2_api_key} if self.s2_api_key else {}

    def _openalex_key_params(self) -> Dict[str, str]:
        return {"api_key": self.openalex_api_key} if self.openalex_api_key else {}

    @staticmethod
    def _is_openalex(identifier: str) -> bool:
        return "openalex.org/" in identifier.casefold() or bool(
            re.fullmatch(r"W\d+", identifier, re.I)
        )

    @staticmethod
    def _format_semantic_scholar(paper: Dict[str, Any]) -> Dict[str, Any]:
        external = paper.get("externalIds") or {}
        pdf = paper.get("openAccessPdf") or {}
        references = [
            str(item.get("paperId"))
            for item in paper.get("references") or []
            if isinstance(item, dict) and item.get("paperId")
        ]
        return {
            "paperId": paper.get("paperId") or "",
            "year": paper.get("year") or 0,
            "publication_date": paper.get("publicationDate") or "",
            "title": paper.get("title") or "",
            "authors": ", ".join(
                str(item.get("name") or "") for item in paper.get("authors") or []
            ),
            "venue": paper.get("venue") or "",
            "citationCount": paper.get("citationCount") or 0,
            "abstract": paper.get("abstract") or "",
            "isOpenAccess": bool(paper.get("isOpenAccess")),
            "url": pdf.get("url") or paper.get("url") or "",
            "externalIds": external,
            "doi": external.get("DOI") or external.get("doi") or "",
            "fieldsOfStudy": paper.get("fieldsOfStudy") or [],
            "s2FieldsOfStudy": paper.get("s2FieldsOfStudy") or [],
            "referenced_works": references,
            "retrieval_source": "semantic_scholar",
        }

    @classmethod
    def _format_openalex(cls, work: Dict[str, Any]) -> Dict[str, Any]:
        raw_ids = work.get("ids")
        raw_location = work.get("primary_location")
        raw_open_access = work.get("open_access")
        ids: Dict[str, Any] = raw_ids if isinstance(raw_ids, dict) else {}
        location: Dict[str, Any] = (
            raw_location if isinstance(raw_location, dict) else {}
        )
        raw_source = location.get("source")
        source: Dict[str, Any] = raw_source if isinstance(raw_source, dict) else {}
        open_access: Dict[str, Any] = (
            raw_open_access if isinstance(raw_open_access, dict) else {}
        )
        authors = []
        for authorship in work.get("authorships") or []:
            author = authorship.get("author") if isinstance(authorship, dict) else None
            if isinstance(author, dict) and author.get("display_name"):
                authors.append(str(author["display_name"]))
        abstract = cls._reconstruct_abstract(work.get("abstract_inverted_index"))
        doi = cls._strip_doi(work.get("doi") or ids.get("doi"))
        return {
            "paperId": ids.get("openalex") or work.get("id") or "",
            "year": work.get("publication_year") or 0,
            "publication_date": work.get("publication_date") or "",
            "title": work.get("display_name") or "",
            "authors": ", ".join(authors),
            "venue": source.get("display_name") or "",
            "citationCount": work.get("cited_by_count") or 0,
            "abstract": abstract,
            "isOpenAccess": bool(open_access.get("is_oa")),
            "url": location.get("pdf_url") or location.get("landing_page_url") or "",
            "externalIds": {
                "DOI": doi,
                "OpenAlex": ids.get("openalex") or work.get("id") or "",
            },
            "doi": doi,
            "fieldsOfStudy": [],
            "s2FieldsOfStudy": [],
            "referenced_works": list(work.get("referenced_works") or []),
            "retrieval_source": "openalex",
        }

    @staticmethod
    def _reconstruct_abstract(index: Any) -> str:
        if not isinstance(index, dict):
            return ""
        positioned = []
        for word, positions in index.items():
            if not isinstance(positions, list):
                continue
            for position in positions:
                try:
                    positioned.append((int(position), str(word)))
                except (TypeError, ValueError):
                    continue
        return " ".join(word for _, word in sorted(positioned))

    @staticmethod
    def _strip_doi(value: Any) -> str:
        text = str(value or "").strip()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if text.casefold().startswith(prefix):
                return text[len(prefix) :]
        return text

    @staticmethod
    def _filter_cutoff(
        works: List[Dict[str, Any]], cutoff: Optional[date]
    ) -> List[Dict[str, Any]]:
        if cutoff is None:
            return works
        eligible = []
        for work in works:
            raw_date = str(work.get("publication_date") or "").strip()
            if raw_date:
                try:
                    if date.fromisoformat(raw_date[:10]) >= cutoff:
                        continue
                except ValueError:
                    pass
            if not raw_date and int(work.get("year") or 0) > cutoff.year:
                continue
            eligible.append(work)
        return eligible


__all__ = ["OpenScholar"]
