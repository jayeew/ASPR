"""Watch a PhotoRec destination and validate recovered SQLite candidates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SQLITE_HEADER = b"SQLite format 3\x00"


def write_status(path: Path, payload: dict[str, Any]) -> None:
    """Persist a small machine-readable watcher status document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def is_sqlite(path: Path) -> bool:
    """Detect a SQLite candidate by its immutable file header."""
    try:
        with path.open("rb") as handle:
            return handle.read(len(SQLITE_HEADER)) == SQLITE_HEADER
    except OSError:
        return False


def validate(candidate: Path, exporter: Path, output_dir: Path) -> tuple[bool, str]:
    """Run the strict export validator against one recovered candidate."""
    result = subprocess.run(
        [
            sys.executable,
            str(exporter),
            "--database",
            str(candidate),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    detail = (result.stdout + result.stderr).strip()[-4_000:]
    return result.returncode == 0, detail


def main() -> None:
    """Watch until a fully validated candidate produces the three exports."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=30)
    args = parser.parse_args()
    checked: set[Path] = set()
    while True:
        candidates = [path for path in args.source_dir.rglob("*") if path.is_file()]
        for candidate in candidates:
            if candidate in checked or not is_sqlite(candidate):
                continue
            checked.add(candidate)
            ok, detail = validate(candidate, args.exporter, args.output_dir)
            write_status(
                args.status,
                {
                    "candidate": str(candidate),
                    "validated": ok,
                    "detail": detail,
                    "checked_candidates": len(checked),
                },
            )
            if ok:
                return
        write_status(
            args.status,
            {"validated": False, "checked_candidates": len(checked)},
        )
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
