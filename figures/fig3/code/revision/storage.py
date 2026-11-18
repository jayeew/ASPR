"""Ordinary files. Reuse depends only on existence; reruns are explicit."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Config


class MissingInput(ValueError):
    """Requested input is missing or unreadable."""


def read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise MissingInput(f'{path}: {exc}') from exc


def write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete JSON only after the write finishes; an interrupted .tmp is not a checkpoint.
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


class Store:
    def __init__(self, config: Config) -> None:
        self.config = config

    def path(self, stage: str, ident: str, variant: str = '') -> Path:
        area = ('inputs' if stage == 'prepare' else 'reports' if stage in {'direct_a', 'gear', 'graph', 'full', 'eacl', 'reviewgrounder'}
                else 'evidence' if stage in {'evidence_pool', 'reference'} else 'annotations')
        return self.config.output/area/stage/(variant or 'papers')/f'{ident}.json'

    def get(self, stage: str, ident: str, variant: str = '') -> dict[str, Any]:
        return read(self.path(stage, ident, variant))

    def existing(self, stage: str, ident: str, variant: str = '', overwrite: bool = False) -> bool:
        return not overwrite and self.path(stage, ident, variant).exists()

    def put(self, stage: str, ident: str, result: dict[str, Any], variant: str = '') -> None:
        write(self.path(stage, ident, variant), result)


def roster(config: Config) -> list[dict[str, Any]]:
    path = config.dataset/'papers.jsonl'
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise MissingInput(f'{path}: {exc}') from exc
