from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gear.env as gear_env
from gear.env import (
    getenv,
    getenv_int,
    getenv_list,
    load_env,
    subprocess_environment,
)


def test_load_env_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        env_path = Path(tmp) / ".env"
        env_path.write_text(
            "ASPR_TEST_TOKEN=abc123\n"
            "ASPR_TEST_PORT=12345\n"
            "ASPR_TEST_KEYS=key1, key2 key3\n"
            "ASPR_TEST_QUOTED='value # not comment'\n"
            "ASPR_TEST_COMMENT=value # comment",
            encoding="utf-8",
        )

        load_env(paths=[env_path], override=True)
        assert getenv("ASPR_TEST_TOKEN") == "abc123"
        assert getenv_int("ASPR_TEST_PORT", 0) == 12345
        assert getenv_list("ASPR_TEST_KEYS") == ["key1", "key2", "key3"]
        assert getenv("ASPR_TEST_QUOTED") == "value # not comment"
        assert getenv("ASPR_TEST_COMMENT") == "value"


def test_existing_env_wins_by_default() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        env_path = Path(tmp) / ".env"
        env_path.write_text("ASPR_TEST_KEEP=from_file\n", encoding="utf-8")
        os.environ["ASPR_TEST_KEEP"] = "from_shell"
        try:
            load_env(paths=[env_path], override=False)
            assert getenv("ASPR_TEST_KEEP") == "from_shell"
        finally:
            os.environ.pop("ASPR_TEST_KEEP", None)


def test_later_env_file_overrides_earlier_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base_path = Path(tmp) / ".env"
        local_path = Path(tmp) / ".env.local"
        base_path.write_text("ASPR_TEST_LAYER=base\n", encoding="utf-8")
        local_path.write_text("ASPR_TEST_LAYER=local\n", encoding="utf-8")

        try:
            os.environ.pop("ASPR_TEST_LAYER", None)
            load_env(paths=[base_path, local_path], override=False)
            assert getenv("ASPR_TEST_LAYER") == "local"
        finally:
            os.environ.pop("ASPR_TEST_LAYER", None)


def test_subprocess_environment_does_not_promote_dotenv_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("DOTENV_ONLY_VALUE=not-an-override\n", encoding="utf-8")
    monkeypatch.setattr(gear_env, "DEFAULT_ENV_FILES", (env_path,))
    monkeypatch.setattr(gear_env, "SYSTEM_ENV", {"SYSTEM_ONLY_VALUE": "kept"})
    monkeypatch.setattr(gear_env, "_LOADED", False)
    monkeypatch.delenv("DOTENV_ONLY_VALUE", raising=False)
    monkeypatch.setenv("SYSTEM_ONLY_VALUE", "kept")

    child = subprocess_environment()

    assert "DOTENV_ONLY_VALUE" not in child
    assert child["SYSTEM_ONLY_VALUE"] == "kept"


if __name__ == "__main__":
    test_load_env_file()
    test_existing_env_wins_by_default()
    test_later_env_file_overrides_earlier_file()
    print("test_env.py passed")
