"""Stateless bibliographic search adapter used by the GEAR evidence lane."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import date
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from .cli_limiter import network_request_lease
from .env import getenv, getenv_bool, getenv_int
from .innovation.locking import stage_lock

_OPENALEX_KEY_LOCK = threading.Lock()
_OPENALEX_KEY_INDEX = 0
_OPENALEX_KEY_COOLDOWNS: dict[str, float] = {}


def _next_openalex_key(current: str = "") -> str:
    global _OPENALEX_KEY_INDEX
    values = [
        item
        for item in re.split(
            r"[,;\s]+",
            " ".join(
                [getenv("OPENALEX_API_KEYS"), getenv("OPENALEX_API_KEY"), current]
            ),
        )
        if item
    ]
    keys = list(dict.fromkeys(values))
    if not keys:
        return ""
    while True:
        with _OPENALEX_KEY_LOCK:
            now = time.monotonic()
            for offset in range(len(keys)):
                index = (_OPENALEX_KEY_INDEX + offset) % len(keys)
                key = keys[index]
                if _OPENALEX_KEY_COOLDOWNS.get(key, 0.0) <= now:
                    _OPENALEX_KEY_INDEX = index + 1
                    return key
            delay = min(_OPENALEX_KEY_COOLDOWNS[key] - now for key in keys)
        from .innovation.usage import log_progress

        log_progress("[OpenAlex冷却] 所有Key暂不可用，等待=%.1f秒", min(delay, 60.0))
        time.sleep(min(max(delay, 0.01), 60.0))


def _retry_after_seconds(value: str, *, epoch_allowed: bool = False) -> float:
    try:
        seconds = float(value)
        if epoch_allowed and seconds > 1_000_000_000:
            seconds -= time.time()
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0.0, seconds) if math.isfinite(seconds) else 0.0


def _budget_exhausted(headers: Mapping[str, str], body: str) -> bool:
    for name in ("X-RateLimit-Remaining-USD", "X-RateLimit-Remaining"):
        try:
            if name in headers and float(headers[name]) <= 0:
                return True
        except ValueError:
            continue
    text = body.casefold()
    return any(
        marker in text
        for marker in (
            "insufficient budget",
            "budget exhausted",
            "daily budget",
            "daily limit",
            "credits exhausted",
            "insufficient credits",
        )
    )


def _suspend_openalex_key(
    key: str, retry_after: str, response: requests.Response | None = None
) -> None:
    if not key:
        return
    seconds = _retry_after_seconds(retry_after)
    if response is not None and _budget_exhausted(
        response.headers, response.text[:4096]
    ):
        seconds = max(
            seconds,
            _retry_after_seconds(
                response.headers.get("X-RateLimit-Reset", ""), epoch_allowed=True
            ),
        )
    seconds = max(1.0, seconds or 60.0)
    with _OPENALEX_KEY_LOCK:
        _OPENALEX_KEY_COOLDOWNS[key] = max(
            _OPENALEX_KEY_COOLDOWNS.get(key, 0.0), time.monotonic() + seconds
        )


def _safe_request_error(error: requests.RequestException) -> str:
    return re.sub(
        r"(?i)(api_key=)[^&\s'\")]+",
        r"\1<redacted>",
        str(error),
    )


def _is_openalex_url(url: str) -> bool:
    return urlsplit(url).hostname in {"api.openalex.org", "content.openalex.org"}


def _openalex_cache_key(url: str, kwargs: dict[str, Any]) -> str:
    parts = urlsplit(url)
    if (
        parts.hostname != "api.openalex.org"
        or not (parts.path == "/works" or parts.path.startswith("/works/"))
        or kwargs.get("stream")
        or "pdf" in str(kwargs.get("headers", {}).get("Accept", "")).casefold()
    ):
        return ""
    prepared = requests.Request("GET", url, params=kwargs.get("params")).prepare()
    parts = urlsplit(str(prepared.url))
    query = sorted(
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.casefold() != "api_key"
    )
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def _read_openalex_cache(path: Path, url: str, ttl: int) -> requests.Response | None:
    try:
        max_bytes = getenv_int("GEAR_OPENALEX_CACHE_ENTRY_MAX_BYTES", 4 * 1024**2)
        if path.stat().st_size > max_bytes:
            return None
        entry = json.loads(path.read_text(encoding="utf-8"))
        age = time.time() - float(entry["saved_at"])
        if age < 0 or age > ttl:
            return None
        content = json.dumps(entry["payload"], ensure_ascii=False).encode("utf-8")
    except (OSError, ValueError, TypeError, KeyError):
        return None
    response = requests.Response()
    response.status_code = 200
    response.url = url
    response.encoding = "utf-8"
    response._content = content
    response.headers.update({"Content-Type": "application/json", "X-GEAR-Cache": "HIT"})
    return response


def _prune_openalex_cache(root: Path, incoming_size: int, ttl: int) -> None:
    max_entries = max(1, getenv_int("GEAR_OPENALEX_CACHE_MAX_ENTRIES", 10_000))
    max_bytes = max(1, getenv_int("GEAR_OPENALEX_CACHE_MAX_BYTES", 256 * 1024**2))
    now = time.time()
    entries: list[tuple[float, int, Path]] = []
    for path in root.glob("*.json"):
        try:
            stat = path.stat()
            if now - stat.st_mtime > ttl:
                path.unlink(missing_ok=True)
            else:
                entries.append((stat.st_mtime, stat.st_size, path))
        except OSError:
            continue
    entries.sort()
    total = sum(size for _, size, _ in entries) + incoming_size
    count = len(entries) + 1
    for _, size, path in entries:
        if count <= max_entries and total <= max_bytes:
            break
        path.unlink(missing_ok=True)
        count -= 1
        total -= size


def _write_openalex_cache(path: Path, response: requests.Response, ttl: int) -> None:
    if response.status_code != 200:
        return
    max_bytes = min(
        getenv_int("GEAR_OPENALEX_CACHE_ENTRY_MAX_BYTES", 4 * 1024**2),
        getenv_int("GEAR_OPENALEX_CACHE_MAX_BYTES", 256 * 1024**2),
    )
    if len(response.content) > max_bytes:
        return
    temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        payload = response.json()
        if not isinstance(payload, dict) or not (
            isinstance(payload.get("results"), list)
            or isinstance(payload.get("id"), str)
        ):
            return
        entry = json.dumps(
            {"saved_at": time.time(), "payload": payload}, ensure_ascii=False
        ).encode("utf-8")
        if len(entry) > max_bytes:
            return
        with stage_lock(path.parent / "capacity.lock"):
            _prune_openalex_cache(path.parent, len(entry), ttl)
            temporary.write_bytes(entry)
            temporary.replace(path)
    except (OSError, ValueError, TypeError):
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _limited_get(*args: Any, **kwargs: Any) -> requests.Response:
    url = str(args[0]) if args else str(kwargs.get("url", ""))
    cache_key = _openalex_cache_key(url, kwargs)
    ttl = getenv_int("GEAR_OPENALEX_CACHE_TTL_SECONDS", 7 * 86400)
    if not cache_key or ttl <= 0:
        return _network_get(*args, **kwargs)
    default_root = Path(__file__).resolve().parents[1] / "data/cache/openalex_requests"
    root = Path(getenv("GEAR_OPENALEX_CACHE_DIR", str(default_root))).expanduser()
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    path = root / f"{digest}.json"
    # A fixed set of lock files bounds inode use while serializing identical requests.
    with ExitStack() as stack:
        try:
            stack.enter_context(stage_lock(root / f"request_{digest[:2]}.lock"))
        except OSError:
            return _network_get(*args, **kwargs)
        cached = _read_openalex_cache(path, cache_key, ttl)
        if cached is not None:
            from .innovation.usage import log_progress

            log_progress("[OpenAlex缓存命中] 复用成功检索响应")
            return cached
        response = _network_get(*args, **kwargs)
        _write_openalex_cache(path, response, ttl)
        return response


def _network_get(*args: Any, **kwargs: Any) -> requests.Response:
    if getenv_bool("GEAR_SCHOLAR_BYPASS_PROXY", False):
        kwargs.setdefault("proxies", {"http": "", "https": ""})
    retries = max(0, min(getenv_int("GEAR_NETWORK_RETRIES", 2), 5))
    for attempt in range(retries + 1):
        try:
            url = str(args[0]) if args else str(kwargs.get("url", ""))
            params = kwargs.get("params") or {}
            if _is_openalex_url(url) and isinstance(params, dict):
                request_params = dict(params)
                key = _next_openalex_key(str(request_params.get("api_key", "")))
                if key:
                    request_params["api_key"] = key
                else:
                    request_params.pop("api_key", None)
                kwargs["params"] = request_params
            with network_request_lease():
                response = requests.get(*args, **kwargs)
            if response.status_code != 429:
                return response
            from .innovation.usage import log_progress

            retry_after = response.headers.get("Retry-After", "")
            used_key = str(kwargs.get("params", {}).get("api_key", ""))
            if _is_openalex_url(url):
                _suspend_openalex_key(used_key, retry_after, response)
            if attempt == retries:
                return response
            response.close()
            if used_key:
                delay = 0.25
            else:
                delay = max(
                    min(float(2 ** (attempt + 1)), 30.0),
                    _retry_after_seconds(retry_after),
                )
            log_progress(
                "[网络响应重试] attempt=%d/%d，status=%d，等待=%.1f秒",
                attempt + 1,
                retries,
                response.status_code,
                delay,
            )
            time.sleep(delay)
        except requests.RequestException as exc:
            if attempt == retries:
                raise requests.RequestException(_safe_request_error(exc)) from exc
            from .innovation.usage import log_progress

            delay = 2**attempt
            log_progress(
                "[网络请求重试] attempt=%d/%d，等待=%d秒，error=%s",
                attempt + 1,
                retries,
                delay,
                _safe_request_error(exc),
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")


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
        response = _limited_get(
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
            response = _limited_get(
                f"{self.openalex_url}/{suffix}",
                params=self._openalex_key_params(),
                headers={"Accept-Encoding": "gzip, deflate"},
                timeout=60,
            )
            if response.status_code != 200:
                return {}
            return self._format_openalex(response.json())
        response = _limited_get(
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
                response = _limited_get(
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
        response = _limited_get(
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
        response = _limited_get(
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
                "cache_hit": response.headers.get("X-GEAR-Cache") == "HIT",
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
            response = _limited_get(
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
            "open_access_locations": cls._openalex_locations(work),
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
    def _openalex_locations(work: dict[str, Any]) -> list[dict[str, Any]]:
        raw_locations = work.get("locations")
        candidates = [work.get("best_oa_location")]
        if isinstance(raw_locations, list):
            candidates.extend(raw_locations)
        candidates.append(work.get("primary_location"))
        locations: list[dict[str, Any]] = []
        for raw in candidates:
            if not isinstance(raw, dict):
                continue
            location: dict[str, Any] = {"is_oa": raw.get("is_oa") is True}
            for field in ("pdf_url", "landing_page_url", "version", "license"):
                value = raw.get(field)
                location[field] = value.strip() if isinstance(value, str) else ""
            if (
                location["pdf_url"] or location["landing_page_url"]
            ) and location not in locations:
                locations.append(location)
        return locations

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
