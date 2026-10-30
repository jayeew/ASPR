from __future__ import annotations

import fcntl
from pathlib import Path

from experiments.fig8_extension import publisher_delta_apply as module


def test_follow_waits_for_extension_handoff(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    assert not module.acquisition_chain_live()
    path = tmp_path / 'supplement/pleiades_route/dispatcher.lock'
    path.parent.mkdir(parents=True)
    with path.open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert module.acquisition_chain_live()
    assert not module.acquisition_chain_live()


def test_index_recovers_later_qualifications_without_revisiting_receipts() -> None:
    import sqlite3
    db = sqlite3.connect(':memory:')
    db.executescript('''CREATE TABLE papers(work_id,doi,shard,status);
        CREATE TABLE publisher_remaining_jobs(doi,receipt,route,status,applied);
        CREATE TABLE springer_author_processed(work_id,receipt,digest,status);''')
    for i in range(4):
        db.execute('INSERT INTO papers VALUES(?,?,?,?)',
                   (f'W{i}', f'D{i}', 'a.parquet', 'included_verified_research' if i != 2 else 'pending'))
        db.execute('INSERT INTO publisher_remaining_jobs VALUES(?,?,?,?,?)',
                   (f'D{i}', f'R{i}', 'springer', 'parsed', int(i != 3)))
    db.execute("INSERT INTO springer_author_processed VALUES('W0','R0','hash','no_explicit_byline')")
    assert module.unprocessed_author_jobs(db) == [('W1', 'a.parquet', 'R1')]
    db.execute("UPDATE publisher_remaining_jobs SET receipt='new_receipt' WHERE doi='D0'")
    assert set(module.unprocessed_author_jobs(db)) == {('W0', 'a.parquet', 'new_receipt'), ('W1', 'a.parquet', 'R1')}
    db.close()
