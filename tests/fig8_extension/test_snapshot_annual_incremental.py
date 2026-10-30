"""Keep annual cache discovery inside the recovered records and study years."""
import json
import sqlite3

from experiments.fig8_extension import snapshot_annual_incremental as module


def test_only_recovered_in_window_native_evidence_is_applied(tmp_path, monkeypatch) -> None:
    folder = tmp_path / 'snapshot'
    annual = folder / 'annual_imprints'
    annual.mkdir(parents=True)
    (annual / 'review.json').write_text('{"records":[]}')
    (tmp_path / 'version.json').write_text(json.dumps({'paper_links': [{'work_id': 'recovered'}]}))
    archive = sqlite3.connect(folder / 'archive_jobs.sqlite')
    archive.execute('CREATE TABLE jobs(pmcid,status,receipt)')
    archive.execute('INSERT INTO jobs VALUES(?,?,?)', ('PMC123', 'metadata_and_available_xml_path_processed', 'version.json'))
    archive.commit(); archive.close()
    database = tmp_path / 'current.sqlite'
    db = sqlite3.connect(database)
    db.executescript('CREATE TABLE papers(work_id TEXT PRIMARY KEY,doi,title,journal,source_id);'
                    'CREATE TABLE current_evidence_index(work_id,evidence_key,kind,payload);'
                    'CREATE INDEX current_evidence_work_kind ON current_evidence_index(work_id,kind);')
    for wid in ('recovered', 'unrelated'):
        db.execute('INSERT INTO papers VALUES(?,?,?,?,?)', (wid, '10.test/' + wid, 'Title', 'Journal', 'S1'))
        for year in (2022, 2023, 2026):
            evidence = {'family': 'archived_jats', 'identity': 'matched', 'online_date': f'{year}-01-01',
                        'receipt': f'{wid}-{year}.xml.gz', 'url': 'https://example.org/PMC123'}
            db.execute('INSERT INTO current_evidence_index VALUES(?,?,?,?)',
                       (wid, str(year), 'primary', json.dumps(evidence)))
    db.commit(); db.close()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'FOLDER', folder)
    monkeypatch.setattr(module, 'connect', lambda: sqlite3.connect(database))
    monkeypatch.setattr(module, 'read_csv', lambda _: [
        {'source_id': 'S1', 'year': str(y), 'nature_portfolio_year': ''} for y in (2022, 2023, 2026)])
    monkeypatch.setattr(module.annual, 'FOLDER', annual)
    calls = []

    def apply() -> None:
        records = json.loads((module.annual.FOLDER / 'scope.json').read_text())['records']
        calls.extend(records)
        (module.annual.FOLDER / 'application_status.json').write_text('{"status":"fixed_annual_cache_applied"}')

    monkeypatch.setattr(module.annual, 'run', apply)
    module.run()
    assert [(r['work_id'], r['evidence']['online_date']) for r in calls] == [('recovered', '2023-01-01')]
    module.run()
    assert len(calls) == 1
