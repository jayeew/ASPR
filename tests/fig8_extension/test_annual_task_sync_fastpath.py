import sqlite3

from experiments.fig8_extension.metadata_task_sync import advance_annual_only_sync


def test_annual_only_sync_preserves_author_links_and_stale_changes() -> None:
    db = sqlite3.connect(':memory:')
    db.executescript('''CREATE TABLE metadata_task_shards(shard TEXT PRIMARY KEY,stamp TEXT);
        CREATE TABLE tasks(task_id TEXT,kind TEXT);
        CREATE TABLE task_papers(task_id TEXT,work_id TEXT,position INTEGER);
        INSERT INTO metadata_task_shards VALUES('a','old');
        INSERT INTO tasks VALUES('p','portfolio'),('i','institution');
        INSERT INTO task_papers VALUES('p','w',-1),('i','w',0),('p','u',-1);''')
    rows = [{'work_id': 'w', 'nature_portfolio_at_verified_online_year': 'False'},
            {'work_id': 'u', 'nature_portfolio_at_verified_online_year': ''}]
    assert not advance_annual_only_sync(db, 'a', 'stale', 'new', rows)
    assert db.execute('SELECT count(*) FROM task_papers').fetchone()[0] == 3
    assert advance_annual_only_sync(db, 'a', 'old', 'new', rows)
    assert set(db.execute('SELECT task_id,work_id FROM task_papers')) == {('i', 'w'), ('p', 'u')}
    assert db.execute('SELECT stamp FROM metadata_task_shards').fetchone()[0] == 'new'
    assert not advance_annual_only_sync(db, 'a', 'old', 'new', rows)
