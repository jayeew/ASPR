"""Country-only signatures must not be dispatched as unknown institutions."""
import json
import sqlite3

from experiments.fig8_extension.supplement_report import institution_priorities


def test_field_routing_keeps_missing_requirements_explicit() -> None:
    db = sqlite3.connect(':memory:')
    db.executescript('''CREATE TABLE tasks(task_id TEXT,kind TEXT,active INTEGER,
        status TEXT,payload TEXT); CREATE TABLE task_papers(task_id TEXT,work_id TEXT);''')
    for ident, flags in [('country', (0, 1)), ('identity', (1, 0)),
                         ('both', (1, 1)), ('unknown', (None, None))]:
        db.execute('INSERT INTO tasks VALUES(?,?,?,?,?)', (ident, 'institution', 1,
            'pending', json.dumps({'name': ident, 'missing_identity': flags[0],
                                  'missing_country': flags[1]})))
        db.execute('INSERT INTO task_papers VALUES(?,?)', (ident, 'paper'))
    rows = {r['task_id']: r for r in institution_priorities(db)}
    db.close()
    assert rows['country']['next_action'] == 'signed_workplace_country_only'
    assert rows['identity']['next_action'] == 'institution_identity'
    assert rows['both']['next_action'] == 'identity_and_signed_country'
    assert rows['unknown']['next_action'] == 'refresh_slot_requirements_before_acquisition'
    assert all(r['affected_papers'] == r['author_records'] == 1 for r in rows.values())
