from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

from experiments.fig8_extension import snapshot_location_dispatch as dispatch


def test_nullable_titles_markup_doi_and_replay(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dispatch, 'ROOT', tmp_path)
    out = sqlite3.connect(':memory:')
    out.executescript('CREATE TABLE consumed(receipt TEXT PRIMARY KEY);'
        'CREATE TABLE records(work_id TEXT,receipt TEXT,position INTEGER,identity TEXT,PRIMARY KEY(work_id,receipt));'
        'CREATE TABLE links(work_id TEXT,url TEXT,host TEXT,status TEXT,evidence TEXT,PRIMARY KEY(work_id,url));')
    current = sqlite3.connect(':memory:')
    current.execute('CREATE TABLE papers(work_id TEXT PRIMARY KEY,doi TEXT,title TEXT,status TEXT)')
    current.executemany('INSERT INTO papers VALUES(?,?,?,?)', [
        ('W1', '10.1/a', 'Expected title', 'pending_no_primary_type_date'),
        ('W2', '10.1/b', 'Protein 2', 'pending_no_primary_type_date'),
        ('W3', '10.1/c', 'Expected title', 'pending_no_primary_type_date')])
    records = [('W1', '10.1/a', None), ('W2', '10.1/b', 'Protein <sub>2</sub>'),
               ('W3', '10.1/other', 'Expected title')]
    with gzip.open(tmp_path / 'raw.jsonl.gz', 'wt') as stream:
        for position, (wid, doi, title) in enumerate(records):
            stream.write(json.dumps({'source_file': 'original.gz', 'source_line': position + 1,
                'source_record_sha256': 'test', 'identity': 'work_id_doi_match',
                'work': {'id': wid, 'doi': 'https://doi.org/' + doi, 'title': title,
                         'locations': [{'landing_page_url': 'https://doi.org/' + doi}]}}) + '\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'raw_receipt': 'raw.jsonl.gz'}))
    dispatch.consume(out, current, [], {}, manifest)
    assert dict(out.execute('SELECT work_id,status FROM links')) == {
        'W1': 'title_identity_review_required', 'W2': 'doi_resolver_existing_route',
        'W3': 'doi_identity_review_required'}
    dispatch.consume(out, current, [], {}, manifest)
    assert out.execute('SELECT count(*) FROM links').fetchone()[0] == 3
    assert out.execute('SELECT count(*) FROM consumed').fetchone()[0] == 1
