from __future__ import annotations

from datetime import datetime, timezone
from email.utils import format_datetime

import pytest
import requests

from gear import scholar


@pytest.fixture(autouse=True)
def clear_cooldowns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scholar, "_OPENALEX_KEY_INDEX", 0)
    monkeypatch.setattr(scholar, "_OPENALEX_KEY_COOLDOWNS", {})


def test_rotation_and_cooldown_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scholar,
        "getenv",
        lambda name: "key-a,key-b" if name == "OPENALEX_API_KEYS" else "",
    )
    now = [100.0]
    monkeypatch.setattr(scholar.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        scholar.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    assert scholar._next_openalex_key() == "key-a"
    assert scholar._next_openalex_key() == "key-b"
    scholar._suspend_openalex_key("key-a", "10")
    scholar._suspend_openalex_key("key-b", "20")
    assert scholar._next_openalex_key() == "key-a"
    assert now[0] == 110.0


def test_key_lists_and_current_key_are_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {"OPENALEX_API_KEYS": "key-a, key-b", "OPENALEX_API_KEY": "key-b"}
    monkeypatch.setattr(scholar, "getenv", lambda name: values.get(name, ""))
    assert [scholar._next_openalex_key("key-b; key-c") for _ in range(4)] == [
        "key-a",
        "key-b",
        "key-c",
        "key-a",
    ]


def test_rotation_order_stays_stable_when_one_key_cools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scholar,
        "getenv",
        lambda name: "key-a,key-b,key-c" if name == "OPENALEX_API_KEYS" else "",
    )
    monkeypatch.setattr(scholar.time, "monotonic", lambda: 100.0)
    assert scholar._next_openalex_key() == "key-a"
    scholar._suspend_openalex_key("key-b", "10")
    assert scholar._next_openalex_key() == "key-c"
    assert scholar._next_openalex_key() == "key-a"


def test_budget_reset_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scholar.time, "monotonic", lambda: 100.0)
    response = requests.Response()
    response._content = b'{"error":"Insufficient budget"}'
    response.headers["X-RateLimit-Reset"] = "7200"
    scholar._suspend_openalex_key("key-a", "10", response)
    assert scholar._OPENALEX_KEY_COOLDOWNS["key-a"] == 7300.0


def test_rate_limit_does_not_use_unrelated_daily_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scholar.time, "monotonic", lambda: 100.0)
    response = requests.Response()
    response._content = b'{"error":"Too many requests per second"}'
    response.headers["X-RateLimit-Reset"] = "7200"
    response.headers["X-RateLimit-Remaining-USD"] = "0.75"
    scholar._suspend_openalex_key("key-a", "10", response)
    assert scholar._OPENALEX_KEY_COOLDOWNS["key-a"] == 110.0


def test_retry_after_http_date_and_epoch_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    epoch = 1_800_000_000.0
    monkeypatch.setattr(scholar.time, "time", lambda: epoch)
    monkeypatch.setattr(scholar.time, "monotonic", lambda: 100.0)
    response = requests.Response()
    response._content = b"{}"
    response.headers["X-RateLimit-Remaining-USD"] = "0"
    response.headers["X-RateLimit-Reset"] = str(epoch + 3600)
    retry_after = format_datetime(datetime.fromtimestamp(epoch + 60, timezone.utc))
    assert scholar._retry_after_seconds(retry_after) == 60.0
    scholar._suspend_openalex_key("key-a", retry_after, response)
    assert scholar._OPENALEX_KEY_COOLDOWNS["key-a"] == 3700.0


def test_shorter_concurrent_cooldown_cannot_shorten_existing_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scholar.time, "monotonic", lambda: 100.0)
    scholar._suspend_openalex_key("key-a", "7200")
    scholar._suspend_openalex_key("key-a", "10")
    assert scholar._OPENALEX_KEY_COOLDOWNS["key-a"] == 7300.0
