"""Per-run model call timing; unavailable provider token usage stays null."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_USAGE_PATH: ContextVar[Path | None] = ContextVar("gear_usage_path", default=None)
_PROGRESS_LOGGER: ContextVar[logging.Logger | None] = ContextVar(
    "gear_progress_logger", default=None
)
_PROGRESS_CONTEXT: ContextVar[str] = ContextVar("gear_progress_context", default="")


@contextmanager
def usage_log(path: Path) -> Iterator[None]:
    token = _USAGE_PATH.set(path)
    try:
        yield
    finally:
        _USAGE_PATH.reset(token)


@contextmanager
def progress_logging(logger: logging.Logger, context: str) -> Iterator[None]:
    logger_token = _PROGRESS_LOGGER.set(logger)
    context_token = _PROGRESS_CONTEXT.set(context)
    try:
        yield
    finally:
        _PROGRESS_CONTEXT.reset(context_token)
        _PROGRESS_LOGGER.reset(logger_token)


@contextmanager
def progress_scope(context: str) -> Iterator[None]:
    previous = _PROGRESS_CONTEXT.get()
    token = _PROGRESS_CONTEXT.set(
        " ".join(part for part in (previous, context) if part)
    )
    try:
        yield
    finally:
        _PROGRESS_CONTEXT.reset(token)


def log_progress(message: str, *args: object) -> None:
    logger = _PROGRESS_LOGGER.get()
    if logger is None:
        return
    context = _PROGRESS_CONTEXT.get()
    logger.info("%s %s", context, message % args if args else message)


def record_call(model: str, seconds: float, cached: bool, success: bool) -> None:
    path = _USAGE_PATH.get()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "model": model,
                        "seconds": seconds,
                        "cached": cached,
                        "success": success,
                        "provider_tokens": None,
                    }
                )
                + "\n"
            )
    log_progress(
        "[模型调用完成] model=%s，成功=%s，缓存=%s，耗时=%.1f秒",
        model,
        success,
        cached,
        seconds,
    )
