"""Optional exact-change hints; any untracked file mutation forces ordinary shard sync."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .publisher_delta_apply import stamp


def signature(paper_path: Path, decisions_path: Path) -> str:
    return 'ownership-queue-1:' + stamp(paper_path) + ':' + stamp(decisions_path)


def ensure_table(db: sqlite3.Connection) -> None:
    db.execute('CREATE TABLE IF NOT EXISTS metadata_task_deltas('
               'shard TEXT PRIMARY KEY,before_stamp TEXT,after_stamp TEXT,work_ids TEXT)')


def record(db: sqlite3.Connection, shard: str, before: str, after: str,
           work_ids: set[str], semantics_unchanged: bool) -> None:
    """Call under checkpoint lock after replacing a shard, before committing its ledger."""
    ensure_table(db)
    if not semantics_unchanged:
        db.execute('DELETE FROM metadata_task_deltas WHERE shard=?', (shard,))
        return
    previous = db.execute('SELECT before_stamp,after_stamp,work_ids FROM metadata_task_deltas WHERE shard=?',
                          (shard,)).fetchone()
    if previous and previous[1] == before:
        before = previous[0]
        work_ids = work_ids | set(json.loads(previous[2]))
    db.execute('INSERT OR REPLACE INTO metadata_task_deltas VALUES(?,?,?,?)',
               (shard, before, after, json.dumps(sorted(work_ids))))


def selected_ids(db: sqlite3.Connection, shard: str, synchronized: str | None,
                 current: str) -> set[str] | None:
    """Return None on any evidence gap so the caller performs its original full sync."""
    row = db.execute('SELECT before_stamp,after_stamp,work_ids FROM metadata_task_deltas WHERE shard=?',
                     (shard,)).fetchone()
    if not row or row[:2] != (synchronized, current):
        return None
    ids = json.loads(row[2])
    return set(ids) if ids and all(isinstance(w, str) and w for w in ids) else None
