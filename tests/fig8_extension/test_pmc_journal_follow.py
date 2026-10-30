import json
import sqlite3
from pathlib import Path

import pytest


from experiments.fig8_extension.pmc_journal_follow import stage
from experiments.fig8_extension.publisher_delta_apply import publisher_items
from experiments.fig8_extension.supplement_oai import parse_oai

def test_native_type_and_unknown_online_date_preserved() -> None:
    raw = b'''<record><metadata><article article-type="editorial"><front>
    <journal-meta><journal-title-group><journal-title>Example</journal-title></journal-title-group>
    <issn>1234-5678</issn></journal-meta><article-meta><article-id pub-id-type="doi">10.1/x</article-id>
    <title-group><article-title>Example editorial</article-title></title-group>
    <pub-date pub-type="ppub"><year>2024</year></pub-date>
    <contrib-group><contrib contrib-type="author"><name><surname>Smith</surname></name></contrib></contrib-group>
    </article-meta></front></article></metadata></record>'''
    record = {'source': 'pmc_front_oai', 'status': 'parsed', 'doi': '10.1/x',
              'raw_receipt': 'oai_raw/PMC1.xml.gz', 'url': 'https://example.org', 'parsed': parse_oai(raw)}
    result = stage(record, 'oai_parsed/PMC1.json')
    assert result is not None
    item = publisher_items(result)[0]
    assert item['kind'] == 'editorial'
    assert not item['online_date']
    assert result['parsed']['authors'][0]['name'] == 'Smith'
    assert stage({**record, 'doi': '10.1/wrong'}, 'x') is None
    assert stage({**record, 'status': 'metadata_unavailable'}, 'x') is None


def test_attempt_links_idempotently_without_resolving_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.fig8_extension import pmc_journal_follow as module
    (tmp_path / 'supplement').mkdir()
    dbpath = tmp_path / 'tasks.sqlite'
    db = sqlite3.connect(dbpath)
    db.executescript("CREATE TABLE tasks(task_id TEXT,kind TEXT,entity_key TEXT,attempt_n INTEGER,status TEXT);"
                     "CREATE TABLE attempts(task_id TEXT,source TEXT,status TEXT,receipt TEXT,detail TEXT);"
                     "INSERT INTO tasks VALUES('t','qualification','10.1/x',0,'pending');")
    db.commit()
    db.close()
    (tmp_path / 'record.json').write_text(json.dumps({'doi': '10.1/x', 'pmcid': 'PMC1',
        'status': 'parsed', 'parsed': {'doi': ['10.1/wrong']}, 'batch_receipt': 'batch.xml.gz'}))
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'connect', lambda: sqlite3.connect(dbpath))
    module.record_attempts(['record.json'])
    module.record_attempts(['record.json'])
    db = sqlite3.connect(dbpath)
    assert db.execute('SELECT status,attempt_n FROM tasks').fetchone() == ('pending', 1)
    assert db.execute('SELECT status FROM attempts').fetchall() == [('doi_identity_mismatch',)]
    assert json.loads(db.execute('SELECT detail FROM attempts').fetchone()[0])['final_task_closure'] is False
    db.close()


def test_pending_uses_receipts_not_collector_run_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.fig8_extension import pmc_journal_follow as module
    folder = tmp_path / 'fields/qualification/oai_parsed'
    folder.mkdir(parents=True)
    for name, source in [('old', 'pmc_front_oai'), ('new', 'pmc_front_oai'), ('other', 'different')]:
        (folder / (name + '.json')).write_text(json.dumps({'source': source}))
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    done = {'fields/qualification/oai_parsed/old.json', 'historical_receipt_not_in_this_run'}
    assert module.pending_receipts(done) == 1
    done.add('fields/qualification/oai_parsed/new.json')
    assert module.pending_receipts(done) == 0
