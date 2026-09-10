from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, Self

import pytest
import requests

from gear import fulltext

DOI = "10.1234/example.2024"
TITLE = "A verified scientific investigation of model behavior"
XML = f"""<article><front><article-meta>
<article-id pub-id-type="doi">{DOI}</article-id>
<title-group><article-title>{TITLE}</article-title></title-group>
</article-meta></front><body><sec><title>Results</title>
<p>The measured result provides source bound scientific evidence.</p></sec></body></article>""".encode()


class Response:
    def __init__(self, content: bytes, status: int = 200, **headers: str) -> None:
        self.content = content
        self.status_code = status
        self.headers = headers
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True

    def iter_content(self, chunk_size: int) -> Any:
        yield self.content


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GEAR_FULLTEXT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GEAR_SCHOLAR_BYPASS_PROXY", "true")


def fetch(locations: list[dict[str, Any]], **values: Any) -> dict[str, Any]:
    return fulltext.fetch_external_fulltext(
        values.pop("work_id", "https://openalex.org/W1"),
        values.pop("doi", DOI),
        values.pop("title", TITLE),
        locations,
        max_bytes=values.pop("max_bytes", 100000),
        max_pages=values.pop("max_pages", 2),
        max_characters=values.pop("max_characters", 2000),
        **values,
    )


def route(monkeypatch: pytest.MonkeyPatch, responses: dict[str, Response]) -> list[str]:
    called: list[str] = []

    def fake_get(self: requests.Session, url: str, **kwargs: Any) -> Response:
        assert self.trust_env is False
        assert "params" not in kwargs and "Authorization" not in kwargs["headers"]
        assert "api_key" not in url
        assert kwargs["stream"] is True
        assert kwargs["allow_redirects"] is False
        if "/search?" in url and url not in responses:
            return Response(b'{"resultList":{"result":[]}}')
        called.append(url)
        return responses[url]

    monkeypatch.setattr(requests.Session, "get", fake_get)
    return called


def test_pmc_xml_is_verified_and_shared_by_doi(monkeypatch: pytest.MonkeyPatch) -> None:
    url = fulltext._EUROPE_PMC + "/PMC123/fullTextXML"
    response = Response(XML)
    called = route(monkeypatch, {url: response})
    location = [
        {
            "is_oa": True,
            "landing_page_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC123/",
        }
    ]
    first = fetch(location)
    second = fetch(
        [], work_id="https://openalex.org/W2", doi="https://doi.org/" + DOI.upper()
    )
    assert first["status"] == "success" and first["format"] == "xml"
    assert first["identity_verified"] and "measured result" in first["text"]
    assert second["cache_hit"] and second["text"] == first["text"]
    assert first["acquired_at"].endswith("+00:00")
    assert second["acquired_at"] == first["acquired_at"]
    assert called == [url] and response.closed


def test_doi_lookup_gets_open_pmc_body(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    def fake_get(self: requests.Session, url: str, **kwargs: Any) -> Response:
        called.append(url)
        if "/search?" in url:
            return Response(
                json.dumps(
                    {
                        "resultList": {
                            "result": [
                                {"doi": DOI, "pmcid": "PMC123", "isOpenAccess": "Y"}
                            ]
                        }
                    }
                ).encode()
            )
        return Response(XML)

    monkeypatch.setattr(requests.Session, "get", fake_get)
    result = fetch([])
    assert result["status"] == "success"
    assert len(called) == 2 and called[1].endswith("/PMC123/fullTextXML")


def test_publisher_failure_falls_back_to_doi_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []

    def fake_get(self: requests.Session, url: str, **kwargs: Any) -> Response:
        called.append(url)
        if "publisher.example" in url:
            return Response(b"", 403)
        if "/search?" in url:
            return Response(
                json.dumps(
                    {
                        "resultList": {
                            "result": [
                                {"doi": DOI, "pmcid": "PMC123", "isOpenAccess": "Y"}
                            ]
                        }
                    }
                ).encode()
            )
        return Response(XML)

    monkeypatch.setattr(requests.Session, "get", fake_get)
    result = fetch([{"is_oa": True, "pdf_url": "https://publisher.example/a.pdf"}])
    assert result["status"] == "success"
    assert len(called) == 3 and sum("/search?" in url for url in called) == 1


def test_rejected_identity_is_not_fulltext() -> None:
    with pytest.raises(fulltext._Unavailable, match="identity_unverified"):
        fulltext._xml_text(XML, "10.1234/another", TITLE, 2000)
    # Finding the requested DOI in references must not validate another article.
    wrong = XML.replace(DOI.encode(), b"10.1234/another").replace(
        b"</body>", DOI.encode() + b"</body>"
    )
    with pytest.raises(fulltext._Unavailable, match="identity_unverified"):
        fulltext._xml_text(wrong, DOI, TITLE, 2000)


def test_external_entities_and_abstract_only_xml_are_rejected() -> None:
    with pytest.raises(fulltext._Unavailable, match="unsafe_xml"):
        fulltext._xml_text(
            b'<!DOCTYPE article [<!ENTITY x SYSTEM "file:///etc/passwd">]>' + XML,
            DOI,
            TITLE,
            2000,
        )
    with pytest.raises(fulltext._Unavailable, match="body_missing"):
        fulltext._xml_text(
            XML[: XML.index(b"<body>")] + b"</article>", DOI, TITLE, 2000
        )


@pytest.mark.parametrize("status", [403, 429])
def test_no_retry_on_block_and_negative_cache(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    url = "https://publisher.example/paper.pdf"
    called = route(monkeypatch, {url: Response(b"", status)})
    locations = [
        {
            "is_oa": True,
            "pdf_url": url,
            "landing_page_url": "https://publisher.example/article",
        }
    ]
    first, second = fetch(locations), fetch(locations)
    assert first["status"] == "unavailable" and str(status) in first["error"]
    assert second["cache_hit"] and called == [url]
    assert not first["identity_verified"] and not first["text"]


def test_negative_cache_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://publisher.example/paper.pdf"
    called = route(monkeypatch, {url: Response(b"", 404)})
    locations = [{"is_oa": True, "pdf_url": url}]
    fetch(locations)
    monkeypatch.setenv("GEAR_FULLTEXT_NEGATIVE_TTL_SECONDS", "0")
    fetch(locations)
    assert called == [url, url]


def test_untrusted_or_closed_locations_are_not_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = route(monkeypatch, {})
    result = fetch(
        [
            {"is_oa": False, "pdf_url": "https://publisher.example/closed.pdf"},
            {"is_oa": True, "pdf_url": "https://content.openalex.org/works/W1.pdf"},
            {"is_oa": True, "pdf_url": "http://127.0.0.1/secret"},
            {
                "is_oa": True,
                "pdf_url": "https://publisher.example/a?api_key=not-for-external",
            },
        ],
        doi="",
    )
    assert result["status"] == "unavailable" and called == []


def test_redirect_to_openalex_content_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://publisher.example/paper.pdf"
    called = route(
        monkeypatch,
        {url: Response(b"", 302, Location="https://content.openalex.org/works/W1.pdf")},
    )
    result = fetch([{"is_oa": True, "pdf_url": url}])
    assert result["status"] == "unavailable" and called == [url]


@pytest.mark.parametrize("headers", [{"Content-Length": "200"}, {}])
def test_byte_budget_includes_chunked_downloads(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str]
) -> None:
    url = "https://publisher.example/paper.pdf"
    response = Response(b"x" * 200, **headers)
    route(monkeypatch, {url: response})
    result = fetch([{"is_oa": True, "pdf_url": url}], max_bytes=100)
    assert result["error"] == "download_exceeds_byte_limit" and response.closed


def test_html_only_follows_identity_bound_citation_pdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://publisher.example/article"
    pdf_url = "https://publisher.example/paper.pdf"
    html = f'<html><meta content="{DOI}" name="citation_doi"><meta name="citation_pdf_url" content="/paper.pdf"></html>'.encode()
    called = route(monkeypatch, {url: Response(html), pdf_url: Response(b"%PDF-stub")})
    monkeypatch.setattr(fulltext, "_parse_document", lambda *args: "Verified PDF text")
    result = fetch([{"is_oa": True, "landing_page_url": url}])
    assert result["status"] == "success" and called == [url, pdf_url]
    assert result["source_url"] == pdf_url and result["format"] == "pdf"


def test_login_html_is_not_treated_as_fulltext(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://publisher.example/article"
    called = route(monkeypatch, {url: Response(b"<html>Login required</html>")})
    result = fetch([{"is_oa": True, "landing_page_url": url}])
    assert result["status"] == "unavailable" and called == [url]
    assert not result["text"]


def test_concurrent_same_doi_downloads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_fetch(*args: Any) -> dict[str, Any]:
        calls.append(1)
        time.sleep(0.05)
        return fulltext._result("success", text="cached", identity_verified=True)

    monkeypatch.setattr(fulltext, "_fetch", fake_fetch)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: fetch([]), range(4)))
    assert len(calls) == 1 and len(results) == 4
    assert sum(bool(result.get("cache_hit")) for result in results) == 3


def test_global_download_slots_limit_parallelism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    maximum = 0
    lock = threading.Lock()

    def fake_fetch(*args: Any) -> dict[str, Any]:
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return fulltext._result("unavailable", "none")

    monkeypatch.setattr(fulltext, "_fetch", fake_fetch)
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(lambda n: fetch([], doi=f"10.1234/{n}"), range(5)))
    assert maximum == 2


def test_network_errors_do_not_expose_urls_or_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Response:
        raise requests.ConnectionError("failed api_key=EXAMPLE_SECRET")

    monkeypatch.setattr(requests.Session, "get", broken)
    result = fetch([{"is_oa": True, "pdf_url": "https://publisher.example/a.pdf"}])
    assert result["error"] == "external_source_network_failed"
    assert "EXAMPLE_SECRET" not in json.dumps(result)


def test_parser_timeout_is_work_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args: Any, **kwargs: Any) -> Any:
        raise fulltext.subprocess.TimeoutExpired("parser", 1)

    monkeypatch.setattr(fulltext.subprocess, "run", timeout)
    with pytest.raises(TimeoutError, match="deadline"):
        fulltext._parse_document(XML, "xml", DOI, TITLE, 3, 1000, time.monotonic() + 1)


def test_timed_out_work_has_negative_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def timeout(*args: Any) -> dict[str, Any]:
        calls.append(1)
        raise TimeoutError("deadline")

    monkeypatch.setattr(fulltext, "_fetch", timeout)
    first, second = fetch([]), fetch([])
    assert first["status"] == "failed" and second["cache_hit"]
    assert len(calls) == 1


def pdf_bytes(pages: tuple[str, ...]) -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 10 700 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_pdf_identity_and_page_budget() -> None:
    content = pdf_bytes((f"{TITLE} DOI: {DOI}", "SECOND PAGE MUST NOT BE PARSED"))
    result = fulltext._parse_document(
        content, "pdf", DOI, TITLE, 1, 2000, time.monotonic() + 10
    )
    assert DOI in result and "SECOND PAGE" not in result
    with pytest.raises(fulltext._Unavailable):
        fulltext._pdf_text(
            content,
            "10.1234/different",
            "Not the same title at all",
            1,
            2000,
            time.monotonic() + 10,
        )


def test_pdf_cited_doi_and_title_do_not_establish_identity() -> None:
    content = pdf_bytes((f"An unrelated article. References. {TITLE} DOI: {DOI}",))
    with pytest.raises(fulltext._Unavailable, match="identity_unverified"):
        fulltext._pdf_text(content, DOI, TITLE, 1, 2000, time.monotonic() + 10)


@pytest.mark.parametrize("host", ["127.1", "2130706433", "0x7f000001", "127.0.0.1"])
def test_noncanonical_loopback_urls_are_rejected(host: str) -> None:
    with pytest.raises(fulltext._Unavailable, match="unsafe"):
        fulltext._safe_url(f"http://{host}/private")


@pytest.mark.parametrize(
    "addresses",
    [
        ["127.0.0.1"],
        ["10.1.2.3"],
        ["192.168.1.1"],
        ["169.254.169.254"],
        ["::1"],
        ["8.8.8.8", "172.16.1.2"],
    ],
)
def test_dns_private_answers_never_connect(
    monkeypatch: pytest.MonkeyPatch, addresses: list[str]
) -> None:
    monkeypatch.setattr(
        fulltext.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (
                fulltext.socket.AF_INET,
                fulltext.socket.SOCK_STREAM,
                6,
                "",
                (address, 443),
            )
            for address in addresses
        ],
    )

    def forbidden_socket(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("No socket may be created for private DNS answers")

    monkeypatch.setattr(fulltext.socket, "socket", forbidden_socket)
    connection = fulltext.HTTPConnection("publisher.example", port=443, timeout=1)
    with pytest.raises(fulltext._Unavailable, match="non_public"):
        fulltext._public_socket(connection)


def test_public_dns_address_is_pinned_before_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = [
        (fulltext.socket.AF_INET, fulltext.socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443))
    ]
    queries: list[str] = []
    connected: list[tuple[str, int]] = []

    def resolve(host: str, *args: Any, **kwargs: Any) -> list[Any]:
        queries.append(host)
        return answers

    class FakeSocket:
        def settimeout(self, timeout: float) -> None:
            assert 0 < timeout <= 1

        def setsockopt(self, *args: Any) -> None:
            pass

        def connect(self, address: tuple[str, int]) -> None:
            connected.append(address)

    monkeypatch.setattr(fulltext.socket, "getaddrinfo", resolve)
    monkeypatch.setattr(fulltext.socket, "socket", lambda *args: FakeSocket())
    connection = fulltext.HTTPSConnection("publisher.example", port=443, timeout=1)
    fulltext._public_socket(connection)
    assert queries == ["publisher.example"] and connected == [("8.8.8.8", 443)]
    assert connection.host == "publisher.example"  # TLS still checks the original host.


def test_dns_resolution_is_deadline_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    def slow_dns(*args: Any, **kwargs: Any) -> list[Any]:
        time.sleep(0.1)
        return []

    monkeypatch.setattr(fulltext.socket, "getaddrinfo", slow_dns)
    with pytest.raises(TimeoutError, match="deadline"):
        fulltext._public_addresses("publisher.example", 443, time.monotonic() + 0.01)


def test_queue_wait_does_not_spend_request_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [100.0]
    waits: list[float] = []

    @contextmanager
    def queued(paths: list[Path], deadline: float) -> Any:
        waits.append(deadline)
        clock[0] += 40.0
        yield

    def fake_fetch(*args: Any) -> dict[str, Any]:
        assert args[-1] - clock[0] == 45.0
        return fulltext._result("unavailable", "none")

    monkeypatch.setenv("GEAR_FULLTEXT_QUEUE_TIMEOUT_SECONDS", "180")
    monkeypatch.setenv("GEAR_FULLTEXT_TIMEOUT_SECONDS", "45")
    monkeypatch.setattr(fulltext.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(fulltext, "_locks", queued)
    monkeypatch.setattr(fulltext, "_fetch", fake_fetch)
    monkeypatch.setattr(fulltext, "_write_cache", lambda *args: None)
    assert fetch([])["status"] == "unavailable"
    assert waits == [280.0, 280.0]


def test_queue_timeout_does_not_create_negative_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    @contextmanager
    def timed_out(paths: list[Path], deadline: float) -> Any:
        raise TimeoutError("queue_deadline")
        yield

    monkeypatch.setattr(fulltext, "_locks", timed_out)
    result = fetch([])
    assert result["error"] == "fulltext_queue_deadline_exceeded"
    assert list((tmp_path / "cache").glob("*.json")) == []
