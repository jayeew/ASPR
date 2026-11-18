from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(encoded(value).encode()).hexdigest()
