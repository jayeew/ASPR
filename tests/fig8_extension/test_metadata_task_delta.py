from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.fig8_extension import metadata_task_delta as delta
from experiments.fig8_extension import metadata_task_sync as sync
from experiments.fig8_extension.supplement_queue import connect


def test_chain_requires_complete_file_history() -> None:
    db = sqlite3.connect(':memory:')
    delta.record(db, 'p', 'a', 'b', {'W1'}, True)
    delta.record(db, 'p', 'b', 'c', {'W2'}, True)
    assert delta.selected_ids(db, 'p', 'a', 'c') == {'W1', 'W2'}
    assert delta.selected_ids(db, 'p', 'a', 'untracked') is None
    delta.record(db, 'p', 'untracked', 'd', {'W3'}, True)
    assert delta.selected_ids(db, 'p', 'a', 'd') is None
    delta.record(db, 'p', 'd', 'e', {'W4'}, False)
    assert delta.selected_ids(db, 'p', 'd', 'e') is None


def fixture(root: Path) -> None:
    (root / 'fields/papers').mkdir(parents=True)
    (root / 'fields/qualification/decisions').mkdir(parents=True)
    rows = []
    decisions = []
    db = connect(root)
    for i in range(50):
        wid = f'W{i}'
        author = {'name': 'Author', 'position': 0, 'countries': ['GB'],
                  'institution_ids_or_verified_local_entities': [],
                  'publisher_affiliation_slots': [{'text': f'Unit {i}', 'institution_ids': [], 'country_codes': ['GB']}]}
        rows.append({'work_id': wid, 'doi': f'10.1/{i}', 'source_id': 'S1', 'journal_name': 'Journal',
                     'abstract': 'Abstract', 'author_list_may_be_truncated': False, 'authors_json': json.dumps([author])})
        decisions.append({'work_id': wid, 'eligibility_status': 'included_verified_research',
                          'online_year': 2024, 'nature_portfolio_at_verified_online_year': 'False'})
        db.execute('INSERT INTO papers(work_id,shard,status) VALUES(?,?,?)', (wid, 'a.parquet', 'included_verified_research'))
    db.commit()
    db.close()
    pq.write_table(pa.Table.from_pylist(rows), root / 'fields/papers/a.parquet')
    pq.write_table(pa.Table.from_pylist(decisions), root / 'fields/qualification/decisions/a.parquet')


def change(root: Path, exact: bool) -> None:
    path = root / 'fields/papers/a.parquet'
    decisions = root / 'fields/qualification/decisions/a.parquet'
    before = delta.signature(path, decisions)
    rows = pq.read_table(path).to_pylist()
    author = json.loads(rows[0]['authors_json'])[0]
    author['institution_ids_or_verified_local_entities'] = ['I1']
    author['publisher_affiliation_slots'][0]['institution_ids'] = ['I1']
    rows[0]['authors_json'] = json.dumps([author])
    pq.write_table(pa.Table.from_pylist(rows), path)
    if exact:
        db = connect(root)
        delta.record(db, path.name, before, delta.signature(path, decisions), {'W0'}, True)
        db.commit()
        db.close()


def state(root: Path) -> dict:
    db = connect(root)
    result = {table: sorted(db.execute(f'SELECT * FROM {table}').fetchall())
              for table in ['task_papers', 'task_link_requirements']}
    result['tasks'] = sorted(db.execute('SELECT task_id,kind,entity_key,status,payload,active FROM tasks').fetchall())
    db.close()
    return result


def test_delta_matches_full_sync_without_reexamining_unmodified_papers(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sync, 'known_universities', lambda root: {'main_personal_direct': {'I1'}})
    monkeypatch.setattr(sync, 'official_relationships', lambda root: {})
    baseline, incremental = tmp_path / 'full', tmp_path / 'delta'
    fixture(baseline)
    sync.run(baseline)
    shutil.copytree(baseline, incremental)
    change(baseline, False)
    change(incremental, True)
    full_result = sync.run(baseline)
    delta_result = sync.run(incremental)
    assert full_result['qualified_papers_assessed'] == 50
    assert delta_result['qualified_papers_assessed'] == 1
    assert state(baseline) == state(incremental)
    assert sync.run(incremental)['unchanged_shards_skipped'] == 1
