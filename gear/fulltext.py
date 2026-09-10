"""Bounded, identity-checked open-access full text with shared disk caching.

This module never uses OpenAlex Content or sends a bibliographic API key.
Only explicitly open-access locations and the Europe PMC public service are used.
"""

from __future__ import annotations

import fcntl
import hashlib
import ipaddress
import json
import re
import socket
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from datetime import datetime, timezone
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import requests
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

from .env import PROJECT_ROOT, getenv, getenv_bool, getenv_float

_EUROPE_PMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
_CACHE_VERSION = 1
_DOI = re.compile(r"10\.\d{4,9}/[^\s<>\"']+", re.IGNORECASE)


class _Unavailable(RuntimeError):
    """An expected external-source limitation, with a non-sensitive reason."""


def _result(status: str, error: str = "", **values: Any) -> dict[str, Any]:
    return {
        "text": "",
        "source_url": "",
        "provider": "",
        "format": "",
        "identity_verified": False,
        "status": status,
        "error": error,
        **values,
    }


def _doi(value: str) -> str:
    value = value.strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        value = value.removeprefix(prefix)
    return value.rstrip(".,;)")


def _title(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", value)
    return " ".join(re.findall(r"\w+", value.casefold()))


def _identity(text: str, doi: str, title: str) -> bool:
    if doi and any(_doi(match) == doi for match in _DOI.findall(text)):
        return True
    expected, actual = _title(title), _title(text)
    return len(expected) >= 20 and len(expected.split()) >= 4 and expected in actual


def _safe_url(value: str) -> str:
    """Reject unsupported, local, credential-bearing and OpenAlex endpoints."""
    parts = urlsplit(value)
    host = (parts.hostname or "").casefold()
    if (
        parts.scheme not in {"http", "https"}
        or not host
        or parts.username
        or parts.password
        or host == "localhost"
        or host.endswith((".localhost", ".local", ".openalex.org"))
        or host == "openalex.org"
    ):
        raise _Unavailable("unsafe_or_unsupported_source_url")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
        if re.fullmatch(r"[\d.]+", host) or re.fullmatch(r"0x[0-9a-f]+", host):
            raise _Unavailable("unsafe_or_unsupported_source_url")
    if address is not None and not address.is_global:
        raise _Unavailable("unsafe_or_unsupported_source_url")
    forbidden = {"api_key", "apikey", "access_token", "token", "authorization"}
    if any(key.casefold() in forbidden for key, _ in parse_qsl(parts.query)):
        raise _Unavailable("credential_bearing_source_url")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def _public_url(value: str) -> str:
    parts = urlsplit(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("fulltext_work_deadline_exceeded")
    return remaining


def _public_addresses(host: str, port: int, deadline: float | None = None) -> list[Any]:
    """Bound DNS waiting and validate every answer before any socket connects."""
    deadline = deadline or time.monotonic() + 10.0
    resolved: Future[list[Any]] = Future()

    def resolve() -> None:
        try:
            resolved.set_result(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except (OSError, ValueError):
            resolved.set_exception(_Unavailable("external_source_dns_failed"))

    threading.Thread(target=resolve, name="gear-fulltext-dns", daemon=True).start()
    try:
        addresses = resolved.result(timeout=_remaining(deadline))
    except FutureTimeoutError as exc:
        raise TimeoutError("fulltext_work_deadline_exceeded") from exc
    if not addresses or any(
        not ipaddress.ip_address(address[4][0]).is_global for address in addresses
    ):
        raise _Unavailable("source_resolves_to_non_public_address")
    return addresses


def _public_socket(connection: HTTPConnection) -> socket.socket:
    """Connect to a validated numeric address; retain the original TLS hostname."""
    deadline = time.monotonic() + float(connection.timeout or 10.0)
    addresses = _public_addresses(connection.host, connection.port or 80, deadline)
    last_error: OSError | None = None
    for family, socktype, proto, _, address in addresses:
        candidate = socket.socket(family, socktype, proto)
        try:
            candidate.settimeout(_remaining(deadline))
            for option in connection.socket_options or []:
                candidate.setsockopt(*option)
            if connection.source_address:
                candidate.bind(connection.source_address)
            candidate.connect(address)
            return candidate
        except OSError as exc:
            candidate.close()
            last_error = exc
    raise requests.ConnectionError("external_source_connection_failed") from last_error


class _PublicHTTPConnection(HTTPConnection):
    def _new_conn(self) -> socket.socket:
        return _public_socket(self)


class _PublicHTTPSConnection(HTTPSConnection):
    def _new_conn(self) -> socket.socket:
        return _public_socket(self)


class _PublicHTTPPool(HTTPConnectionPool):
    ConnectionCls = _PublicHTTPConnection


class _PublicHTTPSPool(HTTPSConnectionPool):
    ConnectionCls = _PublicHTTPSConnection


class _PublicAddressAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(
        self, connections: int, maxsize: int, block: bool = False, **kwargs: Any
    ) -> None:
        super().init_poolmanager(connections, maxsize, block, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {
            "http": _PublicHTTPPool,
            "https": _PublicHTTPSPool,
        }

    def send(
        self,
        request: requests.PreparedRequest,
        stream: bool = False,
        timeout: float | tuple[float | None, float | None] | None = None,
        verify: bool | str = True,
        cert: str | tuple[str, str] | None = None,
        proxies: dict[str, str] | None = None,
    ) -> requests.Response:
        # Explicit user-configured proxies are trusted transports. Validate their
        # target separately; direct connections pin the actual checked address.
        if requests.utils.select_proxy(str(request.url), proxies):
            parts = urlsplit(str(request.url))
            effective_timeout = timeout or 10.0
            connect_timeout = (
                effective_timeout[0]
                if isinstance(effective_timeout, tuple)
                else effective_timeout
            )
            _public_addresses(
                parts.hostname or "",
                parts.port or (443 if parts.scheme == "https" else 80),
                time.monotonic() + float(connect_timeout or 10.0),
            )
        return super().send(
            request,
            stream=stream,
            timeout=timeout,
            verify=verify,
            cert=cert,
            proxies=proxies,
        )


@contextmanager
def _locks(paths: list[Path], deadline: float) -> Iterator[None]:
    """Acquire one of the named cross-process slots, including across threads."""
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    handle = None
    while handle is None:
        _remaining(deadline)
        for path in paths:
            candidate = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                candidate.close()
                continue
            handle = candidate
            break
        if handle is None:
            time.sleep(min(0.05, _remaining(deadline)))
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _read_cache(path: Path, signature: list[int]) -> dict[str, Any] | None:
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("version") != _CACHE_VERSION or cached.get("limits") != signature:
            return None
        result = cached["result"]
        success = result.get("status") == "success"
        ttl = getenv_float(
            (
                "GEAR_FULLTEXT_POSITIVE_TTL_SECONDS"
                if success
                else "GEAR_FULLTEXT_NEGATIVE_TTL_SECONDS"
            ),
            30 * 86400 if success else 3600,
        )
        if time.time() - float(cached["saved_at"]) >= max(0.0, ttl):
            return None
        if success and not (result.get("identity_verified") and result.get("text")):
            return None
        return {**result, "cache_hit": True}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _write_cache(path: Path, signature: list[int], result: dict[str, Any]) -> None:
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    payload = {
        "version": _CACHE_VERSION,
        "limits": signature,
        "saved_at": time.time(),
        "result": result,
    }
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _download(
    session: requests.Session, url: str, max_bytes: int, deadline: float
) -> tuple[bytes, str, str]:
    """Stream once per URL; redirects get checked before another request."""
    for _ in range(5):
        url = _safe_url(url)
        timeout = min(
            max(0.1, getenv_float("GEAR_FULLTEXT_REQUEST_TIMEOUT_SECONDS", 10)),
            _remaining(deadline),
        )
        with session.get(
            url,
            timeout=timeout,
            stream=True,
            allow_redirects=False,
            headers={
                "User-Agent": "ASPR-GEAR/1.0",
                "Accept": "application/pdf, application/xml, text/html, application/json",
            },
        ) as response:
            if response.status_code in {301, 302, 303, 307, 308}:
                url = urljoin(url, response.headers.get("Location", ""))
                continue
            if response.status_code != 200:
                raise _Unavailable(f"http_{response.status_code}")
            try:
                size = int(response.headers.get("Content-Length", "0"))
            except ValueError:
                size = 0
            if size > max_bytes:
                raise _Unavailable("download_exceeds_byte_limit")
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                _remaining(deadline)
                size += len(chunk)
                if size > max_bytes:
                    raise _Unavailable("download_exceeds_byte_limit")
                chunks.append(chunk)
            return b"".join(chunks), url, response.headers.get("Content-Type", "")
    raise _Unavailable("redirect_limit_exceeded")


def _pdf_text(
    content: bytes,
    doi: str,
    title: str,
    max_pages: int,
    max_characters: int,
    deadline: float,
) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise _Unavailable("encrypted_pdf")
        pages: list[str] = []
        count = 0
        for page in reader.pages[:max_pages]:
            _remaining(deadline)
            text = page.extract_text() or ""
            if not pages:
                # A DOI in a cited reference is not the downloaded paper's identity.
                # Require the known title in front matter even when the DOI occurs.
                front = re.split(
                    r"\b(?:references|bibliography)\b",
                    text[:3000],
                    maxsplit=1,
                    flags=re.IGNORECASE,
                )[0]
                if not _identity(front, "", title):
                    raise _Unavailable("fulltext_identity_unverified")
            pages.append(text)
            count += len(text)
            if count >= max_characters:
                break
        result = "\n\n".join(pages).strip()[:max_characters]
    except (PdfReadError, ValueError, TypeError, KeyError) as exc:
        raise _Unavailable("pdf_parse_failed") from exc
    if not result:
        raise _Unavailable("pdf_text_unavailable")
    return result


def _xml_text(content: bytes, doi: str, title: str, max_characters: int) -> str:
    if re.search(rb"<!\s*(?:DOCTYPE|ENTITY)\b", content, re.IGNORECASE):
        raise _Unavailable("unsafe_xml_declaration")
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise _Unavailable("xml_parse_failed") from exc
    # JATS identity is read only from article front matter, never references.
    front = root.find("front/article-meta")
    if front is None:
        raise _Unavailable("xml_article_metadata_missing")
    article_dois = {
        _doi("".join(node.itertext()))
        for node in front.findall("article-id")
        if node.get("pub-id-type") == "doi"
    }
    article_title = " ".join(
        front.findtext("title-group/article-title", default="").split()
    )
    title_node = front.find("title-group/article-title")
    if title_node is not None:
        article_title = "".join(title_node.itertext())
    if doi and article_dois:
        verified = doi in article_dois
    else:
        verified = _identity(article_title, "", title)
    if not verified:
        raise _Unavailable("fulltext_identity_unverified")
    body = root.find("body")
    if body is None:
        raise _Unavailable("xml_fulltext_body_missing")
    parts = [article_title, " ".join(front.itertext()), " ".join(body.itertext())]
    return "\n\n".join(parts).strip()[:max_characters]


def _parse_document(
    content: bytes,
    fmt: str,
    doi: str,
    title: str,
    max_pages: int,
    max_characters: int,
    deadline: float,
) -> str:
    """Isolate untrusted parsers so the work deadline also bounds parsing."""
    metadata = json.dumps(
        {
            "format": fmt,
            "doi": doi,
            "title": title,
            "max_pages": max_pages,
            "max_characters": max_characters,
        }
    ).encode()
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from gear.fulltext import _parse_stdin; _parse_stdin()",
            ],
            input=metadata + b"\n" + content,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=_remaining(deadline),
            cwd=PROJECT_ROOT,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("fulltext_work_deadline_exceeded") from exc
    if completed.returncode != 0:
        raise _Unavailable("fulltext_parser_failed_or_memory_limit")
    try:
        result = json.loads(completed.stdout)
        if not result.get("text"):
            raise _Unavailable(str(result.get("error") or "fulltext_parse_failed"))
        return str(result["text"])
    except (ValueError, AttributeError, KeyError) as exc:
        raise _Unavailable("fulltext_parse_failed") from exc


def _parse_stdin() -> None:
    """Private subprocess entry; no network, model clients or file writes."""
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024**2, 768 * 1024**2))
    metadata = json.loads(sys.stdin.buffer.readline())
    content = sys.stdin.buffer.read()
    try:
        if metadata["format"] == "pdf":
            text = _pdf_text(
                content,
                metadata["doi"],
                metadata["title"],
                metadata["max_pages"],
                metadata["max_characters"],
                float("inf"),
            )
        else:
            text = _xml_text(
                content, metadata["doi"], metadata["title"], metadata["max_characters"]
            )
        result = {"text": text}
    except (MemoryError, _Unavailable):
        result = {"error": "fulltext_parse_or_identity_failed"}
    print(json.dumps(result))


class _CitationMeta(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "meta":
            return
        values = {key.casefold(): value or "" for key, value in attrs}
        name = (values.get("name") or values.get("property") or "").casefold()
        if name.startswith("citation_"):
            self.values[name] = values.get("content", "")


def _candidates(locations: list[dict[str, Any]]) -> list[str]:
    urls: list[str] = []
    for location in locations:
        if location.get("is_oa") is not True:
            continue
        for key in ("pdf_url", "landing_page_url"):
            value = str(location.get(key) or "").strip()
            if value:
                try:
                    urls.append(_safe_url(value))
                except (ValueError, _Unavailable):
                    continue
    return list(dict.fromkeys(urls))[:6]


def _pmc_id(url: str) -> str:
    host = (urlsplit(url).hostname or "").casefold()
    if host in {
        "pmc.ncbi.nlm.nih.gov",
        "www.ncbi.nlm.nih.gov",
        "europepmc.org",
        "www.europepmc.org",
    }:
        match = re.search(r"\bPMC\d+\b", url, re.IGNORECASE)
        return match[0].upper() if match else ""
    return ""


def _find_pmc(session: requests.Session, doi: str, deadline: float) -> str:
    query = urlencode(
        {"query": f'DOI:"{doi}"', "resultType": "core", "format": "json", "pageSize": 5}
    )
    content, _, _ = _download(
        session, f"{_EUROPE_PMC}/search?{query}", 2_000_000, deadline
    )
    try:
        rows = json.loads(content).get("resultList", {}).get("result", [])
        for row in rows:
            if (
                _doi(str(row.get("doi") or "")) == doi
                and row.get("isOpenAccess") == "Y"
            ):
                identifier = str(row.get("pmcid") or "")
                if re.fullmatch(r"PMC\d+", identifier):
                    return identifier
    except (ValueError, AttributeError, TypeError):
        return ""
    return ""


def _fetch_url(
    session: requests.Session,
    url: str,
    doi: str,
    title: str,
    max_bytes: int,
    max_pages: int,
    max_characters: int,
    deadline: float,
) -> dict[str, Any]:
    content, actual, content_type = _download(session, url, max_bytes, deadline)
    if content.lstrip().startswith(b"%PDF"):
        text = _parse_document(
            content, "pdf", doi, title, max_pages, max_characters, deadline
        )
        fmt = "pdf"
    elif actual.startswith(f"{_EUROPE_PMC}/") and actual.endswith("/fullTextXML"):
        text = _parse_document(
            content, "xml", doi, title, max_pages, max_characters, deadline
        )
        fmt = "xml"
    else:
        if (
            "html" not in content_type.casefold()
            and b"<html" not in content[:1000].lower()
        ):
            raise _Unavailable("unsupported_fulltext_format")
        metadata = _CitationMeta()
        metadata.feed(content.decode("utf-8", errors="replace"))
        values = metadata.values
        if not _identity(
            " ".join(
                [values.get("citation_doi", ""), values.get("citation_title", "")]
            ),
            doi,
            title,
        ):
            raise _Unavailable("landing_page_identity_unverified")
        pdf_url = values.get("citation_pdf_url", "")
        if not pdf_url:
            raise _Unavailable("landing_page_pdf_unavailable")
        content, actual, _ = _download(
            session, urljoin(actual, pdf_url), max_bytes, deadline
        )
        if not content.lstrip().startswith(b"%PDF"):
            raise _Unavailable("not_a_pdf")
        text = _parse_document(
            content, "pdf", doi, title, max_pages, max_characters, deadline
        )
        fmt = "pdf"
    return _result(
        "success",
        text=text,
        source_url=_public_url(actual),
        provider=urlsplit(actual).hostname or "external_oa",
        format=fmt,
        identity_verified=True,
        acquired_at=datetime.now(timezone.utc).isoformat(),
        fetched_at=time.time(),
        cache_hit=False,
    )


def _fetch(
    work_id: str,
    doi: str,
    title: str,
    locations: list[dict[str, Any]],
    max_bytes: int,
    max_pages: int,
    max_characters: int,
    deadline: float,
) -> dict[str, Any]:
    del work_id  # Bibliographic identifiers are never sent as authentication.
    candidates = _candidates(locations)
    pmc_ids = list(dict.fromkeys(filter(None, (_pmc_id(url) for url in candidates))))
    candidates = [
        f"{_EUROPE_PMC}/{identifier}/fullTextXML" for identifier in pmc_ids
    ] + candidates
    if doi and not pmc_ids:
        # Resolve DOI only after supplied OA locations fail. The sentinel is
        # processed at most once, and never repeats an already known PMC source.
        candidates.append("")
    errors: list[str] = []
    with requests.Session() as session:
        session.mount("http://", _PublicAddressAdapter())
        session.mount("https://", _PublicAddressAdapter())
        # Avoid netrc credentials being attached to external OA requests.
        session.trust_env = False
        if not getenv_bool("GEAR_SCHOLAR_BYPASS_PROXY", False):
            session.proxies.update(
                requests.utils.get_environ_proxies("https://www.ebi.ac.uk")
            )
        blocked_hosts: set[str] = set()
        for url in candidates:
            host = urlsplit(url).hostname or ""
            if host in blocked_hosts:
                continue
            _remaining(deadline)
            try:
                if not url:
                    pmcid = _find_pmc(session, doi, deadline)
                    if not pmcid:
                        continue
                    url = f"{_EUROPE_PMC}/{pmcid}/fullTextXML"
                    host = urlsplit(url).hostname or ""
                    if host in blocked_hosts:
                        continue
                return _fetch_url(
                    session,
                    url,
                    doi,
                    title,
                    max_bytes,
                    max_pages,
                    max_characters,
                    deadline,
                )
            except _Unavailable as exc:
                reason = str(exc)
                errors.append(reason)
                if reason in {"http_403", "http_429"}:
                    blocked_hosts.add(host)
            except requests.RequestException:
                errors.append("external_source_network_failed")
    return _result(
        "unavailable",
        ";".join(dict.fromkeys(errors)) or "no_open_access_fulltext_location",
    )


def fetch_external_fulltext(
    work_id: str,
    doi: str,
    title: str,
    locations: list[dict[str, Any]],
    *,
    max_bytes: int,
    max_pages: int,
    max_characters: int,
) -> dict[str, Any]:
    """Return verified OA text or an explicit limitation, within resource budgets.

    Positive results and short-lived negative results are shared by normalized
    DOI (or work ID). A per-work lock avoids duplicate downloads; two global
    file-lock slots bound downloads across all worker processes.
    """
    if min(max_bytes, max_pages, max_characters) <= 0:
        raise ValueError("Full-text byte, page and character budgets must be positive")
    normalized_doi = _doi(doi)
    identity = normalized_doi or work_id.strip()
    if not identity:
        return _result("unavailable", "work_identity_missing")
    root = Path(
        getenv(
            "GEAR_FULLTEXT_CACHE_DIR", str(PROJECT_ROOT / "data/cache/gear_fulltext")
        )
    ).expanduser()
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    path = root / f"{key}.json"
    signature = [max_bytes, max_pages, max_characters]
    queue_deadline = time.monotonic() + max(
        1.0, getenv_float("GEAR_FULLTEXT_QUEUE_TIMEOUT_SECONDS", 180)
    )
    try:
        with _locks([root / "locks" / f"{key}.lock"], queue_deadline):
            cached = _read_cache(path, signature)
            if cached is not None:
                return cached
            with _locks(
                [root / "locks" / f"download_{index}.lock" for index in range(2)],
                queue_deadline,
            ):
                deadline = time.monotonic() + max(
                    1.0, getenv_float("GEAR_FULLTEXT_TIMEOUT_SECONDS", 45)
                )
                try:
                    result = _fetch(
                        work_id,
                        normalized_doi,
                        title,
                        locations,
                        max_bytes,
                        max_pages,
                        max_characters,
                        deadline,
                    )
                except TimeoutError:
                    result = _result("failed", "fulltext_work_deadline_exceeded")
            _write_cache(path, signature, result)
            return result
    except TimeoutError:
        return _result("failed", "fulltext_queue_deadline_exceeded")
    except OSError:
        return _result("failed", "fulltext_cache_io_failed")
