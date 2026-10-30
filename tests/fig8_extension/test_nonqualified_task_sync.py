"""Only proven unchanged eligible metadata can reuse existing task links."""
import sqlite3

from experiments.fig8_extension.metadata_task_sync import advance_nonqualified_only_sync


def test_nonqualified_changes_reuse_only_current_synchronized_links() -> None:
    db = sqlite3.connect(':memory:')
    assert not advance_nonqualified_only_sync(db, 's', 'old', 'new', False)
    db.execute('CREATE TABLE metadata_task_shards(shard TEXT PRIMARY KEY,stamp TEXT)')
    db.execute("INSERT INTO metadata_task_shards VALUES('s','old')")
    assert not advance_nonqualified_only_sync(db, 's', 'old', 'new', True)
    assert not advance_nonqualified_only_sync(db, 's', 'stale', 'new', False)
    assert db.execute('SELECT stamp FROM metadata_task_shards').fetchone() == ('old',)
    assert advance_nonqualified_only_sync(db, 's', 'old', 'new', False)
    assert db.execute('SELECT stamp FROM metadata_task_shards').fetchone() == ('new',)
    db.close()
