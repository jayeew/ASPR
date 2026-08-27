"""Stateless bibliographic search adapter used by the GEAR evidence lane."""

from __future__ import annotations

import re
from datetime import date
from io import BytesIO
from typing import Any

import requests

from .env import getenv, getenv_int


class OpenScholar:
    """Minimal OpenAlex-first bibliographic client without the legacy reviewer."""

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
                or getenv("ASPR_RETRIEVAL_PROVIDER", "openalex")
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
        self.openalex_content_url = getenv(
            "ASPR_OPENALEX_CONTENT_URL", "https://content.openalex.org/works"
        ).rstrip("/")
        self.search_limit = max(1, min(getenv_int("ASPR_S2_SEARCH_LIMIT", 100), 100))
        self.openalex_limit = max(
            1, min(getenv_int("ASPR_OPENALEX_PER_PAGE", 100), 100)
        )
        self.last_query_audits: list[dict[str, Any]] = []

    def fetch_pdf_text(
        self,
        work_id: str,
        *,
        max_bytes: int,
        max_pages: int,
        max_characters: int,
    ) -> str:
        """Download one bounded OpenAlex Content PDF and extract plain text."""
        identifier = str(work_id or "").strip()
        if not self._is_openalex(identifier):
            return ""
        suffix = identifier.rsplit("/", 1)[-1]
        response = requests.get(
            f"{self.openalex_content_url}/{suffix}.pdf",
            params=self._openalex_key_params(),
            headers={"Accept": "application/pdf"},
            timeout=60,
            stream=True,
        )
        self.last_query_audits.append(
            {
                "source": "openalex_content",
                "work_id": identifier,
                "status_code": response.status_code,
            }
        )
        try:
            if response.status_code != 200:
                return ""
            declared_size = int(response.headers.get("Content-Length") or 0)
            if declared_size > max_bytes:
                return ""
            content = self._bounded_response_content(response, max_bytes=max_bytes)
            if not content.startswith(b"%PDF"):
                return ""
            return self._extract_pdf_text(
                content,
                max_pages=max_pages,
                max_characters=max_characters,
            )
        finally:
            response.close()

    @staticmethod
    def _bounded_response_content(response: Any, *, max_bytes: int) -> bytes:
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            size += len(chunk)
            if size > max_bytes:
                return b""
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _extract_pdf_text(
        content: bytes,
        *,
        max_pages: int,
        max_characters: int,
    ) -> str:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError

        try:
            reader = PdfReader(BytesIO(content), strict=False)
        except (OSError, PdfReadError, TypeError, ValueError):
            return ""
        parts: list[str] = []
        size = 0
        for page in reader.pages[:max_pages]:
            try:
                text = str(page.extract_text() or "").strip()
            except (KeyError, OSError, PdfReadError, TypeError, ValueError):
                continue
            if not text:
                continue
            remaining = max_characters - size
            if remaining <= 0:
                break
            parts.append(text[:remaining])
            size += min(len(text), remaining)
        return "\n\n".join(parts).strip()

    @staticmethod
    def _first_key(raw: str) -> str:
        values = [item for item in re.split(r"[,;\s]+", raw) if item]
        return values[0] if values else ""

    def search_query(
        self,
        query: str,
        *,
        provider: str | None = None,
        date_to: date | None = None,
        limit: int | None = None,
        search_mode: str = "text",
    ) -> list[dict[str, Any]]:
        """Search with an API-level and local publication cutoff."""
        selected = str(provider or self.retrieval_provider).strip().casefold()
        if selected in {"openalex", "oa"}:
            rows = self._search_openalex(
                query,
                date_to=date_to,
                limit=limit,
                search_mode=search_mode,
            )
            works = [self._format_openalex(item) for item in rows]
        else:
            rows = self._search_semantic_scholar(query, date_to=date_to, limit=limit)
            works = [self._format_semantic_scholar(item) for item in rows]
        return self._filter_cutoff(works, date_to)

    def fetch_work(self, work_id: str) -> dict[str, Any]:
        """Fetch one work in the normalized GEAR retrieval schema."""
        identifier = str(work_id or "").strip()
        if not identifier:
            return {}
        if self._is_openalex(identifier) or self._looks_like_doi(identifier):
            suffix = (
                identifier.rsplit("/", 1)[-1]
                if self._is_openalex(identifier)
                else f"https://doi.org/{self._strip_doi(identifier)}"
            )
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
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch one bounded citation hop."""
        maximum = min(100, max(1, int(limit or 12)))
        identifier = str(work_id or "").strip()
        if self._is_openalex(identifier):
            if direction == "citations":
                suffix = identifier.rsplit("/", 1)[-1]
                params: dict[str, str | int] = {
                    "filter": f"cites:{suffix}",
                    "per_page": maximum,
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
        neighbor_params: dict[str, str | int] = {
            "fields": self._semantic_fields(),
            "limit": maximum,
        }
        response = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/{identifier}/{edge}",
            params=neighbor_params,
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
        self,
        query: str,
        *,
        date_to: date | None,
        limit: int | None,
        search_mode: str,
    ) -> list[dict[str, Any]]:
        semantic = str(search_mode).strip().casefold() == "semantic"
        filters = ["has_abstract:true", "is_retracted:false"]
        if semantic:
            if date_to is not None:
                filters.append(f"publication_year:<{date_to.year + 1}")
        else:
            filters.append("from_publication_date:1800-01-01")
            if date_to is not None:
                filters.append(f"to_publication_date:{date_to.isoformat()}")
        params: dict[str, Any] = {
            "search.semantic" if semantic else "search": str(query).strip()[:2000],
            "filter": ",".join(filters),
            "per_page": min(
                50 if semantic else 100,
                max(1, int(limit or self.openalex_limit)),
            ),
            **self._openalex_key_params(),
        }
        if not semantic:
            params["sort"] = "relevance_score:desc"
        response = requests.get(
            self.openalex_url,
            params=params,
            headers={"Accept": "application/json", "Accept-Encoding": "gzip, deflate"},
            timeout=60,
        )
        self.last_query_audits.append(
            {
                "source": "openalex",
                "query": query,
                "search_mode": "semantic" if semantic else "text",
                "status_code": response.status_code,
            }
        )
        if response.status_code != 200:
            if semantic and response.status_code >= 500:
                self.last_query_audits.append(
                    {
                        "source": "openalex",
                        "query": query,
                        "search_mode": "semantic_to_text_fallback",
                        "status_code": response.status_code,
                    }
                )
                return self._search_openalex(
                    query,
                    date_to=date_to,
                    limit=limit,
                    search_mode="text",
                )
            raise RuntimeError(f"OpenAlex request failed: {response.status_code}")
        rows = response.json().get("results", [])
        return rows if isinstance(rows, list) else []

    def _search_semantic_scholar(
        self, query: str, *, date_to: date | None, limit: int | None
    ) -> list[dict[str, Any]]:
        terms = [str(query).strip()]
        separator = " + " if self.and_search else " | "
        formatted = separator.join(f'"{term}"' for term in terms if term)
        params: dict[str, str | int] = {
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

    def _semantic_headers(self) -> dict[str, str]:
        return {"x-api-key": self.s2_api_key} if self.s2_api_key else {}

    def _openalex_key_params(self) -> dict[str, str]:
        return {"api_key": self.openalex_api_key} if self.openalex_api_key else {}

    @staticmethod
    def _is_openalex(identifier: str) -> bool:
        return "openalex.org/" in identifier.casefold() or bool(
            re.fullmatch(r"W\d+", identifier, re.IGNORECASE)
        )

    @staticmethod
    def _looks_like_doi(identifier: str) -> bool:
        return bool(
            re.match(
                r"^(?:https?://doi\.org/|doi:)?10\.\d{4,9}/\S+$",
                identifier,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _format_semantic_scholar(paper: dict[str, Any]) -> dict[str, Any]:
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
    def _format_openalex(cls, work: dict[str, Any]) -> dict[str, Any]:
        raw_ids = work.get("ids")
        raw_location = work.get("primary_location")
        raw_open_access = work.get("open_access")
        ids: dict[str, Any] = raw_ids if isinstance(raw_ids, dict) else {}
        location: dict[str, Any] = (
            raw_location if isinstance(raw_location, dict) else {}
        )
        raw_source = location.get("source")
        source: dict[str, Any] = raw_source if isinstance(raw_source, dict) else {}
        open_access: dict[str, Any] = (
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
            "topics": [
                str(item.get("display_name") or "")
                for item in work.get("topics") or []
                if isinstance(item, dict) and item.get("display_name")
            ],
            "keywords": [
                str(item.get("display_name") or "")
                for item in work.get("keywords") or []
                if isinstance(item, dict) and item.get("display_name")
            ],
            "relevance_score": work.get("relevance_score"),
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
        works: list[dict[str, Any]], cutoff: date | None
    ) -> list[dict[str, Any]]:
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
