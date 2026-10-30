"""New archive work must be confined to unique, unreviewed MDPI date gaps."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from experiments.fig8_extension import mdpi_primary_followup as module


def test_incremental_scope_excludes_reviewed_ambiguous_and_other_tasks(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / 'supplement/mdpi_oai/primary_followup'
    folder.mkdir(parents=True)
    path = folder / 'scope.json'
    path.write_text(json.dumps({'papers': [{'work_id': 'old', 'doi': '10.3390/old', 'pmcid': 'PMC1'}]}))
    database = tmp_path / 'tasks.sqlite'
    db = sqlite3.connect(database)
    db.execute('CREATE TABLE papers(work_id TEXT,doi TEXT,pmcids TEXT,status TEXT)')
    pending = 'pending_primary_online_date'
    db.executemany('INSERT INTO papers VALUES(?,?,?,?)', [
        ('old', '10.3390/old', '["PMC1"]', pending),
        ('new', '10.3390/new', '["PMC2"]', pending),
        ('same_archive', '10.3390/alias', '["PMC8"]', pending),
        ('same_archive2', '10.3390/alias2', '["PMC8"]', pending),
        ('ambiguous', '10.3390/ambiguous', '["PMC3","PMC4"]', pending),
        ('reviewed', '10.3390/reviewed', '["PMC5"]', pending),
        ('other', '10.1234/other', '["PMC6"]', pending),
        ('type_gap', '10.3390/type_gap', '["PMC7"]', 'pending_primary_type'),
    ])
    db.execute('CREATE TABLE current_evidence_index(work_id TEXT,kind TEXT,payload TEXT)')
    db.execute('INSERT INTO current_evidence_index VALUES(?,?,?)',
               ('reviewed', 'primary', json.dumps({'family': 'archived_jats', 'identity': 'matched'})))
    db.commit()
    db.close()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'connect', lambda: sqlite3.connect(database))
    assert module.extend_scope() == 1
    assert module.extend_scope() == 0
    assert [r['work_id'] for r in json.loads(path.read_text())['papers']] == ['old', 'new']
