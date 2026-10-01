"""Ordinary files, per-object completion and explicit failures."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from figure_pipeline.fig3_revision.aggregate import write_csv
from figure_pipeline.fig3_revision.storage import read, write

__all__ = ['read', 'write', 'write_csv', 'jsonl', 'record', 'artifact', 'recover_calls']


def jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')


def artifact(root: Path, stage: str, paper: str, condition: str = 'papers') -> Path:
    return root / 'annotations' / stage / condition / f'{paper}.json'


def recover_calls(root: Path) -> int:
    """Reuse validated completed responses whose enclosing await was interrupted."""
    recovered = 0
    for path in (root/'logs/calls').glob('*/record.json'):
        call = read(path)
        response = path.with_name('response.json')
        if call['state'] != 'completed' or not response.is_file():
            continue
        target = root/'annotations/checkpoints'/call['stage']/(call.get('method') or 'papers')/call['paper_id']/(call['step']+'.json')
        if not target.is_file():
            write(target, read(response))
            recovered += 1
    return recovered


def record(root: Path, stage: str, paper: str, condition: str, state: str, **details: Any) -> None:
    row = {'time': datetime.now(timezone.utc).isoformat(), 'stage': stage, 'paper_id': paper,
           'condition': condition, 'state': state, **details}
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run_log.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False)+'\n')
