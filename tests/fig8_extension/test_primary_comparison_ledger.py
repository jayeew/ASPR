import json
import sqlite3

from experiments.fig8_extension import primary_comparison_ledger as module


def test_both_event_formats_and_idempotence(tmp_path, monkeypatch):
    folder = tmp_path / 'audit'
    folder.mkdir()
    review = {'work_id': 'W1', 'incoming_receipt': 'one.json', 'status': 'author_sequence_differs'}
    records = [{'review': review, 'attempt': {'status': 'no_additional_unique_exact_links'}},
               {'work_id': 'W2', 'incoming_receipt': 'two.json', 'status': 'source_hash_missing', 'application': {}}]
    (folder / 'review_events.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    path = tmp_path / 'ledger.sqlite'
    db = sqlite3.connect(path)
    db.executescript('CREATE TABLE tasks(task_id,kind,entity_key,attempt_n); CREATE TABLE attempts(task_id,source,status,receipt,detail);')
    db.executemany('INSERT INTO tasks VALUES(?,?,?,?)', [('T1', 'metadata', 'W1', 0), ('T2', 'metadata', 'W2', 0)])
    db.commit(); db.close()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'connect', lambda: sqlite3.connect(path))
    assert module.run(folder)['metadata_attempts_added'] == 2
    assert module.run(folder)['metadata_attempts_added'] == 0
    db = sqlite3.connect(path)
    assert dict(db.execute('SELECT task_id,status FROM attempts')) == {'T1': 'no_additional_unique_exact_links', 'T2': 'source_hash_missing'}
    assert list(db.execute('SELECT attempt_n FROM tasks')) == [(1,), (1,)]
    db.close()
