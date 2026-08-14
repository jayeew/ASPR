from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.env import getenv, getenv_int, getenv_list, load_env


def test_load_env_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        env_path = Path(tmp) / ".env"
        env_path.write_text(
            "\n".join(
                [
                    "ASPR_TEST_TOKEN=abc123",
                    "ASPR_TEST_PORT=12345",
                    "ASPR_TEST_KEYS=key1, key2 key3",
                    "ASPR_TEST_QUOTED='value # not comment'",
                    "ASPR_TEST_COMMENT=value # comment",
                ]
            ),
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


if __name__ == "__main__":
    test_load_env_file()
    test_existing_env_wins_by_default()
    test_later_env_file_overrides_earlier_file()
    print("test_env.py passed")
