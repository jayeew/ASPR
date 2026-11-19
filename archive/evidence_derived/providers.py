"""Provider clients with resumable, secret-safe OpenAlex scheduling."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests

try:
    from .core import ProtocolError
except ImportError:  # Direct execution from this directory.
    from core import ProtocolError  # type: ignore[no-redef]


@dataclass(frozen=True)
class ProviderPage:
    records: list[dict[str, Any]]
    next_cursor: str
    response_hash_source: dict[str, Any]
    key_slot: str


@dataclass(frozen=True)
class ProviderRecord:
    """Single-record lookup result with a secret-free key-slot label."""

    status: str
    record: dict[str, Any]
    key_slot: str


class SharedRateLimiter:
    """Process-wide limiter shared by all OpenAlex client instances."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_request_at = 0.0

    def wait(self, minimum_interval: float, sleep: Callable[[float], None]) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < minimum_interval:
                sleep(minimum_interval - elapsed)
            self._last_request_at = time.monotonic()


OPENALEX_RATE_LIMITER = SharedRateLimiter()


class OpenAlexClient:
    """Small OpenAlex client that persists slot labels, never API keys."""

    def __init__(
        self,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        minimum_interval: float = 0.11,
        max_attempts: int = 5,
        use_environment_proxy: bool = False,
    ) -> None:
        self.session = session or requests.Session()
        if session is None:
            self.session.trust_env = use_environment_proxy
        self.sleep = sleep
        self.minimum_interval = minimum_interval
        self.max_attempts = max_attempts
        self._next_slot = 0
        keys = [
            (slot, value)
            for slot, value in (
                ("A", os.environ.get("OPENALEX_API_KEY_A", "")),
                ("B", os.environ.get("OPENALEX_API_KEY_B", "")),
            )
            if value
        ]
        if not keys:
            combined = [
                value.strip()
                for value in os.environ.get("OPENALEX_API_KEYS", "").split(",")
                if value.strip()
            ]
            keys = [(slot, value) for slot, value in zip(("A", "B"), combined)]
        self._keys = keys

    @property
    def configured_slots(self) -> list[str]:
        return [slot for slot, _ in self._keys]

    def _slot(self) -> tuple[str, str]:
        if not self._keys:
            return "anonymous", ""
        value = self._keys[self._next_slot % len(self._keys)]
        self._next_slot += 1
        return value

    def fetch_page(
        self,
        filter_expression: str,
        cursor: str = "*",
        per_page: int = 200,
    ) -> ProviderPage:
        return self.fetch_search_page(
            search_expression="",
            filter_expression=filter_expression,
            cursor=cursor,
            per_page=per_page,
        )

    def fetch_search_page(
        self,
        search_expression: str,
        filter_expression: str,
        cursor: str = "*",
        per_page: int = 200,
    ) -> ProviderPage:
        """Fetch a cursor page, optionally using OpenAlex full-text search."""
        if not 1 <= per_page <= 200:
            raise ProtocolError("OpenAlex per-page must be in 1..200")
        last_error = ""
        for attempt in range(self.max_attempts):
            slot, key = self._slot()
            OPENALEX_RATE_LIMITER.wait(self.minimum_interval, self.sleep)
            params: dict[str, Any] = {
                "filter": filter_expression,
                "cursor": cursor,
                "per-page": per_page,
            }
            if search_expression:
                params["search"] = search_expression
            if key:
                params["api_key"] = key
            try:
                response = self.session.get(
                    "https://api.openalex.org/works", params=params, timeout=60
                )
            except requests.RequestException as error:
                last_error = type(error).__name__
                self.sleep(min(2**attempt, 30))
                continue
            if response.status_code == 429 or 500 <= response.status_code < 600:
                last_error = f"HTTP_{response.status_code}"
                retry_after = response.headers.get("Retry-After", "")
                delay = (
                    float(retry_after) if retry_after.isdigit() else min(2**attempt, 30)
                )
                self.sleep(delay)
                continue
            if response.status_code >= 400:
                raise ProtocolError(
                    f"OpenAlex request rejected: HTTP_{response.status_code}; key_slot={slot}"
                )
            try:
                payload = response.json()
            except ValueError as error:
                raise ProtocolError(
                    f"OpenAlex returned invalid JSON; key_slot={slot}"
                ) from error
            return ProviderPage(
                records=list(payload.get("results", [])),
                next_cursor=str(payload.get("meta", {}).get("next_cursor") or ""),
                response_hash_source={
                    "results": payload.get("results", []),
                    "meta": {"next_cursor": payload.get("meta", {}).get("next_cursor")},
                },
                key_slot=slot,
            )
        raise ProtocolError(f"OpenAlex request failed after retries: {last_error}")

    def fetch_doi(self, doi: str) -> ProviderRecord:
        """Resolve one DOI through the OpenAlex external-ID endpoint."""
        normalized = doi.strip().lower().removeprefix("https://doi.org/")
        if not normalized:
            raise ProtocolError("OpenAlex DOI lookup requires a DOI")
        if not self._keys:
            raise ProtocolError("OpenAlex DOI lookup requires a configured key slot")
        external_id = quote(f"https://doi.org/{normalized}", safe="")
        last_error = ""
        for attempt in range(self.max_attempts):
            slot, key = self._slot()
            OPENALEX_RATE_LIMITER.wait(self.minimum_interval, self.sleep)
            try:
                response = self.session.get(
                    f"https://api.openalex.org/works/{external_id}",
                    params={"api_key": key},
                    timeout=60,
                )
            except requests.RequestException as error:
                last_error = type(error).__name__
                self.sleep(min(2**attempt, 30))
                continue
            if response.status_code == 404:
                return ProviderRecord("not_found", {}, slot)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                last_error = f"HTTP_{response.status_code}"
                self.sleep(min(2**attempt, 30))
                continue
            if response.status_code >= 400:
                raise ProtocolError(
                    f"OpenAlex DOI lookup rejected: HTTP_{response.status_code}; key_slot={slot}"
                )
            try:
                return ProviderRecord("found", dict(response.json()), slot)
            except ValueError as error:
                last_error = type(error).__name__
                break
        raise ProtocolError(f"OpenAlex DOI lookup failed after retries: {last_error}")


class CrossrefClient:
    """DOI/bibliographic validation only; not a discovery provider."""

    def __init__(
        self,
        session: requests.Session | None = None,
        use_environment_proxy: bool = False,
    ) -> None:
        self.session = session or requests.Session()
        if session is None:
            self.session.trust_env = use_environment_proxy

    def validate_doi(self, doi: str) -> dict[str, Any]:
        normalized = doi.strip().lower().removeprefix("https://doi.org/")
        if not normalized:
            raise ProtocolError("Crossref validation requires a DOI")
        try:
            response = self.session.get(
                f"https://api.crossref.org/works/{normalized}", timeout=60
            )
        except requests.RequestException as error:
            raise ProtocolError(
                f"Crossref DOI lookup failed: {type(error).__name__}"
            ) from error
        if response.status_code == 404:
            return {"doi": normalized, "status": "not_found"}
        try:
            response.raise_for_status()
            message = response.json().get("message", {})
        except (requests.RequestException, ValueError) as error:
            raise ProtocolError(
                f"Crossref DOI lookup failed: {type(error).__name__}"
            ) from error
        title = message.get("title") or []
        year_parts = message.get("published", {}).get("date-parts") or []
        date_parts = year_parts[0] if year_parts else []
        publication_date = (
            f"{int(date_parts[0]):04d}-{int(date_parts[1]):02d}-{int(date_parts[2]):02d}"
            if len(date_parts) >= 3
            else ""
        )
        return {
            "doi": normalized,
            "status": "validated",
            "title": title[0] if title else "",
            "year": year_parts[0][0] if year_parts and year_parts[0] else None,
            "type": message.get("type", ""),
            "container_title": (message.get("container-title") or [""])[0],
            "publication_date": publication_date,
            "raw": message,
        }
