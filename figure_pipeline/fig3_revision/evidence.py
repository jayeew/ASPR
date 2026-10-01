"""Append raw evidence with ordinary identifiers; no hashes or binding gates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EvidenceStore:
    def __init__(self, directory: Path) -> None:
        self.path = directory/'evidence.jsonl'

    def add(self, key: str, kind: str, payload: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'key': key, 'kind': kind, 'payload': payload}, ensure_ascii=False)+'\n')
