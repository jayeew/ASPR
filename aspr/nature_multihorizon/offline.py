"""Fail-closed network guard and local-only configuration validation."""

from __future__ import annotations

import socket
from contextlib import contextmanager
from typing import Any, Iterator, Mapping


class NetworkAccessForbidden(RuntimeError):
    """Raised when a v6 process attempts any socket connection."""


def _blocked_connection(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    raise NetworkAccessForbidden(
        "ASPR v6 is running under network_policy=forbidden"
    )


@contextmanager
def network_forbidden() -> Iterator[None]:
    """Block Python socket connections for the duration of a pipeline action."""
    original_socket = socket.socket
    original_create_connection = socket.create_connection

    class GuardedSocket(original_socket):
        def connect(self, address: Any) -> None:
            del address
            _blocked_connection()

        def connect_ex(self, address: Any) -> int:
            del address
            _blocked_connection()
            return 1

    socket.socket = GuardedSocket
    socket.create_connection = _blocked_connection
    try:
        yield
    finally:
        socket.socket = original_socket
        socket.create_connection = original_create_connection


def validate_local_only_config(config: Mapping[str, Any]) -> None:
    """Reject remote locations in fields that declare filesystem paths."""
    remote_prefixes = (
        "http://",
        "https://",
        "ftp://",
        "s3://",
        "gs://",
        "azure://",
        "ssh://",
    )

    def visit(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                visit(nested, path + (str(key),))
            return
        if isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                visit(nested, path + (str(index),))
            return
        if not isinstance(value, str) or not path:
            return
        key = path[-1].casefold()
        is_location = any(
            token in key
            for token in ("path", "dir", "root", "location", "file")
        )
        if is_location and value.casefold().startswith(remote_prefixes):
            raise ValueError(
                f"remote path is forbidden at {'.'.join(path)}: {value}"
            )

    visit(config, ())


__all__ = [
    "NetworkAccessForbidden",
    "network_forbidden",
    "validate_local_only_config",
]
