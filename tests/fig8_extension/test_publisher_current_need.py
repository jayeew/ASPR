import sqlite3

from experiments.fig8_extension.springer_remaining_collect import (
    reopen_changed_needs,
    retire_satisfied_jobs,
)


def test_current_need_retirement_and_reopening() -> None:
    db = sqlite3.connect(':memory:')
    db.executescript('''CREATE TABLE publisher_remaining_jobs(
        doi TEXT,route TEXT,status TEXT,receipt TEXT);
        CREATE TABLE papers(work_id TEXT,doi TEXT,status TEXT,abstract_gap INTEGER);''')
    for doi, status, gap in [('done', 'included_verified_research', 0),
                             ('excluded', 'excluded_primary_nonresearch', 1),
                             ('pending', 'pending_primary_type', 0),
                             ('missing', 'included_verified_research', 1)]:
        db.execute('INSERT INTO papers VALUES(?,?,?,?)', (doi, doi, status, gap))
        db.execute('INSERT INTO publisher_remaining_jobs VALUES(?,?,?,?)',
                   (doi, 'springer', 'pending', 'retained-source'))
    assert retire_satisfied_jobs(db, 'springer') == 2
    assert retire_satisfied_jobs(db, 'springer') == 0
    assert reopen_changed_needs(db, 'springer') == 0
    db.execute("UPDATE papers SET status='pending_special_subtype_review' WHERE doi='done'")
    assert reopen_changed_needs(db, 'springer') == 1
    assert reopen_changed_needs(db, 'springer') == 0
    assert db.execute("SELECT receipt FROM publisher_remaining_jobs WHERE doi='done'").fetchone()[0] == 'retained-source'
    assert db.execute("SELECT count(*) FROM publisher_remaining_jobs WHERE status='pending'").fetchone()[0] == 3
    db.close()
