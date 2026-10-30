from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

from experiments.fig8_extension import plos_author_xml, publisher_delta_apply


def test_cached_xml_date_uses_epub_not_print_year(tmp_path: Path, monkeypatch) -> None:
    doi = '10.1371/journal.pone.0123456'
    base = tmp_path / 'supplement/plos_batch'
    author = base / 'author_xml'
    date = base / 'date_xml'
    author.mkdir(parents=True)
    date.mkdir()
    xml = f'''<article article-type="research-article"><front><journal-meta>
      <journal-title-group><journal-title>PLOS ONE</journal-title></journal-title-group>
      <issn pub-type="epub">1932-6203</issn></journal-meta><article-meta>
      <article-id pub-id-type="doi">{doi}</article-id>
      <title-group><article-title>A validated experimental study</article-title></title-group>
      <pub-date pub-type="ppub"><year>2026</year><month>1</month><day>2</day></pub-date>
      <pub-date pub-type="epub"><year>2025</year><month>12</month><day>15</day></pub-date>
      </article-meta></front></article>'''.encode()
    raw_path = author / 'source.raw.gz'
    raw_path.write_bytes(gzip.compress(xml))
    (author / 'source_validation.json').write_text(json.dumps({
        'doi': doi, 'http_status': 200, 'attempts': [{'http_status': 200}],
        'raw_receipt': str(raw_path.relative_to(tmp_path)), 'sha256': hashlib.sha256(xml).hexdigest()}))
    (date / 'scope.json').write_text(json.dumps([{'doi': doi, 'work_id': 'W1'}]))
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE papers(work_id,title,journal,source_id,shard,status)')
    db.execute('INSERT INTO papers VALUES(?,?,?,?,?,?)',
               ('W1', 'A validated experimental study', 'PLOS ONE', 'S1', 'a.parquet', 'pending_primary_online_date'))
    monkeypatch.setattr(plos_author_xml, 'ROOT', tmp_path)
    monkeypatch.setattr(plos_author_xml, 'connect', lambda: db)
    monkeypatch.setattr(plos_author_xml, 'read_csv', lambda path: [{'source_id': 'S1', 'issn': '["1932-6203"]'}])

    def unexpected_get(*args, **kwargs) -> None:
        raise AssertionError('Validated cached response must not be requested again')

    monkeypatch.setattr(plos_author_xml.requests.Session, 'get', unexpected_get)
    captured = []

    def apply_cached(*, supplied_jobs: list) -> dict:
        captured.extend(supplied_jobs)
        return {'applied_source_records': len(supplied_jobs)}

    monkeypatch.setattr(publisher_delta_apply, 'apply_available', apply_cached)
    assert plos_author_xml.run('online_date')['applied_source_records'] == 1
    received = json.loads((tmp_path / captured[0][1]).read_text())
    assert received['parsed']['online_date'] == '2025-12-15'
    assert received['parsed']['article_types'] == ['research-article']
    assert received['source'] == 'publisher_jats'
    assert captured[0][0] == doi

    # A generic research-article XML label must not clear a Lab Protocol hold.
    protocol = base / 'protocol_xml'
    protocol.mkdir()
    (protocol / 'scope.json').write_text(json.dumps([{'doi': doi, 'work_id': 'W1'}]))
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE papers(work_id,title,journal,source_id,shard,status)')
    db.execute('INSERT INTO papers VALUES(?,?,?,?,?,?)',
               ('W1', 'A validated experimental study', 'PLOS ONE', 'S1', 'a.parquet', 'pending_primary_type'))
    monkeypatch.setattr(plos_author_xml, 'connect', lambda: db)
    monkeypatch.setattr(plos_author_xml, 'apply_authors', unexpected_get)
    result = plos_author_xml.run('protocol_review')
    assert result['matched_content_records'] == 1
    assert result['qualification_changes'] == 0
    assert len(captured) == 1
