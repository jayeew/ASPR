from __future__ import annotations

import json
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
import requests

from gear import scholar


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GEAR_OPENALEX_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GEAR_OPENALEX_CACHE_TTL_SECONDS", "604800")
    monkeypatch.setenv("GEAR_OPENALEX_CACHE_MAX_ENTRIES", "10000")
    monkeypatch.setenv("GEAR_OPENALEX_CACHE_MAX_BYTES", str(256 * 1024**2))
    monkeypatch.setenv("GEAR_OPENALEX_CACHE_ENTRY_MAX_BYTES", str(4 * 1024**2))
    monkeypatch.setenv("GEAR_NETWORK_RETRIES", "0")
    monkeypatch.setenv("OPENALEX_API_KEY", "")
    monkeypatch.setenv("OPENALEX_API_KEYS", "")
    monkeypatch.setattr(scholar, "_OPENALEX_KEY_COOLDOWNS", {})
    monkeypatch.setattr(scholar, "_OPENALEX_KEY_INDEX", 0)


def _response(
    status: int = 200, payload: dict[str, Any] | None = None
) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.headers["Content-Type"] = "application/json"
    response._content = json.dumps(payload or {"results": []}).encode("utf-8")
    return response


def test_different_keys_share_cache_without_persisting_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []

    def get(*args: Any, **kwargs: Any) -> requests.Response:
        calls.append(kwargs)
        return _response(payload={"results": [{"id": "https://openalex.org/W1"}]})

    monkeypatch.setattr(scholar.requests, "get", get)
    url = "https://api.openalex.org/works"
    first = scholar._limited_get(
        url, params={"search": "test", "api_key": "test-key-a"}
    )
    second = scholar._limited_get(
        url, params={"api_key": "test-key-b", "search": "test"}
    )
    assert len(calls) == 1
    assert second.json() == first.json()
    assert second.headers["X-GEAR-Cache"] == "HIT"
    assert "api_key" not in second.url
    for path in tmp_path.glob("*.json"):
        assert "test-key-" not in path.read_text(encoding="utf-8")


def test_all_query_parameters_distinguish_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def get(*args: Any, **kwargs: Any) -> requests.Response:
        calls.append(1)
        return _response()

    monkeypatch.setattr(scholar.requests, "get", get)
    url = "https://api.openalex.org/works"
    queries = [
        {"search": "test", "filter": "publication_year:2020", "per_page": 5},
        {"search": "test", "filter": "publication_year:2021", "per_page": 5},
        {"search": "test", "filter": "publication_year:2021", "per_page": 6},
        {"search.semantic": "test", "filter": "publication_year:2021", "per_page": 6},
    ]
    for params in queries:
        scholar._limited_get(url, params=params)
    scholar._limited_get(url, params=dict(reversed(list(queries[0].items()))))
    assert len(calls) == 4


@pytest.mark.parametrize("status", [429, 500, 403])
def test_error_response_is_not_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: int
) -> None:
    responses = iter([_response(status), _response()])
    monkeypatch.setattr(
        scholar.requests, "get", lambda *args, **kwargs: next(responses)
    )
    url = "https://api.openalex.org/works"
    assert scholar._limited_get(url).status_code == status
    assert list(tmp_path.glob("*.json")) == []
    assert scholar._limited_get(url).status_code == 200
    assert len(list(tmp_path.glob("*.json"))) == 1


@pytest.mark.parametrize(
    "url,kwargs",
    [
        ("https://content.openalex.org/works/W1.pdf", {}),
        ("https://api.openalex.org/works", {"stream": True}),
        ("https://api.openalex.org/works", {"headers": {"Accept": "application/pdf"}}),
        ("https://example.org/works", {}),
    ],
)
def test_pdf_and_other_hosts_are_not_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, url: str, kwargs: dict[str, Any]
) -> None:
    calls: list[int] = []

    def get(*args: Any, **kwargs: Any) -> requests.Response:
        calls.append(1)
        return _response()

    monkeypatch.setattr(scholar.requests, "get", get)
    scholar._limited_get(url, **kwargs)
    scholar._limited_get(url, **kwargs)
    assert len(calls) == 2
    assert list(tmp_path.glob("*.json")) == []


def test_concurrent_threads_fetch_identical_query_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def get(*args: Any, **kwargs: Any) -> requests.Response:
        calls.append(1)
        time.sleep(0.02)
        return _response()

    monkeypatch.setattr(scholar.requests, "get", get)
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(
            pool.map(
                lambda _: scholar._limited_get("https://api.openalex.org/works"),
                range(8),
            )
        )
    assert len(calls) == 1
    assert all(response.status_code == 200 for response in responses)


def test_concurrent_processes_fetch_identical_query_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = multiprocessing.get_context("fork")
    calls = context.Value("i", 0)

    def get(*args: Any, **kwargs: Any) -> requests.Response:
        with calls.get_lock():
            calls.value += 1
        time.sleep(0.02)
        return _response()

    def run() -> None:
        assert scholar._limited_get("https://api.openalex.org/works").status_code == 200

    monkeypatch.setattr(scholar.requests, "get", get)
    processes = [context.Process(target=run) for _ in range(3)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert calls.value == 1


def test_expired_cache_is_refetched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEAR_OPENALEX_CACHE_TTL_SECONDS", "10")
    now = [time.time()]
    monkeypatch.setattr(scholar.time, "time", lambda: now[0])
    responses = iter([_response(payload={"id": "W1"}), _response(payload={"id": "W2"})])
    monkeypatch.setattr(
        scholar.requests, "get", lambda *args, **kwargs: next(responses)
    )
    url = "https://api.openalex.org/works/W1"
    assert scholar._limited_get(url).json()["id"] == "W1"
    now[0] += 11
    assert scholar._limited_get(url).json()["id"] == "W2"


def test_cache_capacity_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GEAR_OPENALEX_CACHE_MAX_ENTRIES", "2")
    monkeypatch.setattr(scholar.requests, "get", lambda *args, **kwargs: _response())
    for index in range(3):
        scholar._limited_get(
            "https://api.openalex.org/works", params={"search": str(index)}
        )
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_oversized_and_invalid_success_payloads_are_not_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GEAR_OPENALEX_CACHE_ENTRY_MAX_BYTES", "100")
    responses = iter(
        [_response(payload={"id": "x" * 101}), _response(payload={"error": "bad"})]
    )
    monkeypatch.setattr(
        scholar.requests, "get", lambda *args, **kwargs: next(responses)
    )
    for _ in range(2):
        scholar._limited_get("https://api.openalex.org/works")
    assert list(tmp_path.glob("*.json")) == []


def test_open_access_locations_preserve_addresses_and_strict_boolean() -> None:
    best = {
        "is_oa": True,
        "pdf_url": "https://repository.example/paper.pdf",
        "version": "acceptedVersion",
        "license": "cc-by",
    }
    row = scholar.OpenScholar._format_openalex(
        {
            "id": "https://openalex.org/W1",
            "best_oa_location": best,
            "locations": [
                best,
                {"is_oa": "false", "pdf_url": "https://publisher.example/paper.pdf"},
                None,
            ],
            "primary_location": {
                "is_oa": False,
                "landing_page_url": "https://doi.org/10.1234/test",
            },
        }
    )
    locations = row["open_access_locations"]
    assert len(locations) == 3
    assert locations[0]["pdf_url"] == best["pdf_url"]
    assert locations[0]["version"] == "acceptedVersion"
    assert locations[0]["license"] == "cc-by"
    assert locations[1]["is_oa"] is False
    assert locations[2]["landing_page_url"] == "https://doi.org/10.1234/test"
    assert all(type(location["is_oa"]) is bool for location in locations)
