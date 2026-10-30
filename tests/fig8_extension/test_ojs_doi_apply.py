from __future__ import annotations

import csv
import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.fig8_extension import ojs_cached_evidence, ojs_doi_apply
from experiments.fig8_extension.supplement_queue import connect, task


def test_doi_fill_migrates_task_without_duplicate_attempts(tmp_path, monkeypatch):
    root = tmp_path
    folder = root / 'supplement/missing_doi'
    folder.mkdir(parents=True)
    for name in ['fields/papers', 'fields/qualification/decisions']:
        (root / name).mkdir(parents=True)
    wid, doi, sid = 'https://openalex.org/W1', '10.1234/test', 'https://openalex.org/S1'
    raw = '''<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/" xmlns:dc="http://purl.org/dc/elements/1.1/">
    <GetRecord><record><header><identifier>oai:test:1</identifier></header><metadata>
    <dc:title>Test paper</dc:title><dc:identifier>https://doi.org/10.1234/test</dc:identifier>
    <dc:source>Test journal; 1</dc:source><dc:source>1234-5679</dc:source>
    <dc:relation>https://doi.org/10.9999/reference-not-paper</dc:relation>
    </metadata></record></GetRecord></OAI-PMH>'''
    receipt = 'supplement/missing_doi/record.xml.gz'
    (root / receipt).write_bytes(gzip.compress(raw.encode()))
    with (root / 'fields/journal_registry.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['source_id', 'issn'])
        writer.writeheader()
        writer.writerow({'source_id': sid, 'issn': '["1234-5679"]'})
    pq.write_table(pa.Table.from_pylist([{'work_id': wid, 'doi': None}], schema=pa.schema([
        ('work_id', pa.string()), ('doi', pa.string())])), root / 'fields/papers/test.parquet')
    pq.write_table(pa.Table.from_pylist([{'work_id': wid, 'doi': None, 'evidence_ids': '[]'}], schema=pa.schema([
        ('work_id', pa.string()), ('doi', pa.string()), ('evidence_ids', pa.string())])), root / 'fields/qualification/decisions/test.parquet')
    proposal = {'work_id': wid, 'doi': doi, 'status': 'native_doi_identity_matched_pending_apply',
                'receipt': receipt, 'native_identifier': 'oai:test:1', 'source_url': 'https://example.org/oai'}
    (folder / 'native_doi_proposals.json').write_text(json.dumps({'records': [proposal]}))
    db = connect(root)
    db.execute('INSERT INTO papers(work_id,doi,title,journal,source_id,shard,route,status) VALUES(?,?,?,?,?,?,?,?)',
               (wid, None, 'Test paper', 'Test journal', sid, 'test.parquet', 'other', 'pending_primary_type'))
    old = task(db, 'qualification', wid, 'other', {'doi': None}, wid)
    db.commit()
    db.close()
    monkeypatch.setattr(ojs_doi_apply, 'ROOT', root)
    monkeypatch.setattr(ojs_cached_evidence, 'ROOT', root)
    monkeypatch.setattr(ojs_doi_apply, 'connect', lambda: connect(root))
    monkeypatch.setattr(ojs_doi_apply, 'refresh', lambda: None)
    ojs_cached_evidence.entries.cache_clear()
    assert ojs_doi_apply.run()['dois_filled'] == 1
    assert ojs_doi_apply.run()['already_applied'] == 1
    db = connect(root)
    assert db.execute('SELECT doi,status FROM papers').fetchone() == (doi, 'pending_primary_type')
    assert db.execute('SELECT active FROM tasks WHERE task_id=?', (old,)).fetchone() == (0,)
    assert db.execute('SELECT count(*) FROM task_papers').fetchone() == (1,)
    assert db.execute("SELECT count(*) FROM attempts WHERE source='native_ojs_doi'").fetchone() == (1,)
    assert db.execute("SELECT count(*) FROM changes WHERE field='doi'").fetchone() == (1,)
    assert pq.read_table(root / 'fields/papers/test.parquet').to_pylist()[0]['doi'] == doi
    assert pq.read_table(root / 'fields/qualification/decisions/test.parquet').to_pylist()[0]['doi'] == doi
    db.close()
    ojs_cached_evidence.entries.cache_clear()
