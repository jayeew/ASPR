from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILES = (PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local")
SYSTEM_ENV = dict(os.environ)
_LOADED = False


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    escaped = False
    for idx, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "#" and (idx == 0 or value[idx - 1].isspace()):
            return value[:idx].rstrip()
    return value.strip()


def _unquote(value: str) -> str:
    value = _strip_inline_comment(value.strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.replace("\\n", "\n").replace("\\t", "\t")


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a small dotenv-compatible KEY=VALUE file."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        values[key] = _unquote(value)
    return values


def load_env(
    paths: Iterable[Path] | None = None, override: bool = False
) -> dict[str, str]:
    """Load project .env files into os.environ.

    Existing process environment values win by default. `.env.local` is loaded
    after `.env`, so it can override `.env` values when they were not already
    exported by the shell.
    """
    global _LOADED
    loaded: dict[str, str] = {}
    original_env = set(os.environ)
    env_paths = tuple(paths or DEFAULT_ENV_FILES)
    for path in env_paths:
        for key, value in parse_env_file(Path(path)).items():
            loaded[key] = value
    for key, value in loaded.items():
        if override or key not in original_env:
            os.environ[key] = value
    _LOADED = True
    return loaded


def ensure_env_loaded() -> None:
    if not _LOADED:
        load_env()


def getenv(name: str, default: str = "") -> str:
    ensure_env_loaded()
    return os.getenv(name, default)


def getenv_system(name: str, default: str = "") -> str:
    """Return a variable from the process/system environment only.

    This intentionally ignores values loaded from project `.env` files. Use it
    for secrets that should be managed outside the repository workspace.
    """
    return SYSTEM_ENV.get(name, default)


def getenv_runtime(name: str, default: str = "") -> str:
    """Read an explicit process/test override while ignoring project dotenv values."""
    ensure_env_loaded()
    value = os.getenv(name)
    if value is None:
        return default
    dotenv_values: dict[str, str] = {}
    for path in DEFAULT_ENV_FILES:
        dotenv_values.update(parse_env_file(path))
    if name not in SYSTEM_ENV and dotenv_values.get(name) == value:
        return default
    return value


def getenv_int(name: str, default: int) -> int:
    value = getenv(name, "")
    if value == "":
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def getenv_float(name: str, default: float) -> float:
    value = getenv(name, "")
    if value == "":
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def getenv_bool(name: str, default: bool = False) -> bool:
    value = getenv(name, "")
    if value == "":
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def getenv_list(name: str, default: list[str] | None = None) -> list[str]:
    value = getenv(name, "")
    if not value:
        return list(default or [])
    return [part.strip() for part in re.split(r"[,;\s]+", value) if part.strip()]


load_env()
