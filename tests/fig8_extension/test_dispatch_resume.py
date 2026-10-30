import json
import sqlite3
from pathlib import Path

import pytest

from experiments.fig8_extension import cached_secondary_dispatch as dispatch


def test_discovery_preserves_reviewed_outcomes_and_adds_new_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dispatch, 'ROOT', tmp_path)
    folder = tmp_path / 'supplement/cached_secondary_publication_links'
    folder.mkdir(parents=True)
    hold = tmp_path / 'supplement/cached_primary_url_leads'
    hold.mkdir()
    (hold / 'host_hold_application.json').write_text('{"hosts":{}}')
    sqlite3.connect(tmp_path / 'supplement/tasks.sqlite').close()
    db = sqlite3.connect(folder / 'leads.sqlite')
    db.executescript('''
        CREATE TABLE leads(work_id TEXT,url TEXT,host TEXT);
        CREATE TABLE reviewed(work_id TEXT,doi TEXT);
        CREATE TABLE dispatch(work_id TEXT,url TEXT,status TEXT,evidence TEXT,
                              PRIMARY KEY(work_id,url));
    ''')
    db.executemany('INSERT INTO leads VALUES(?,?,?)', [
        ('w1', 'https://example.org/old', 'example.org'),
        ('w2', 'https://example.org/new', 'example.org'),
    ])
    db.executemany('INSERT INTO reviewed VALUES(?,?)', [('w1','10.1/old'), ('w2','10.1/new')])
    reviewed = ('w1', 'https://example.org/old', 'deferred_cached_subtype_requires_policy',
                json.dumps({'receipt':'review.json','pdf_visited':False}))
    db.execute('INSERT INTO dispatch VALUES(?,?,?,?)', reviewed)
    db.commit()
    dispatch.run()
    dispatch.run()
    assert db.execute("SELECT * FROM dispatch WHERE work_id='w1'").fetchone() == reviewed
    assert db.execute("SELECT status FROM dispatch WHERE work_id='w2'").fetchone()[0] == 'source_path_review_required'
    assert db.execute('SELECT count(*) FROM dispatch').fetchone()[0] == 2
    db.close()
