import json
import sqlite3

from experiments.fig8_extension import snapshot_publisher_incremental as module


def test_only_new_exact_urls_without_archive_or_prior_job(tmp_path, monkeypatch):
    links = sqlite3.connect(tmp_path / 'locations.sqlite')
    links.execute('CREATE TABLE links(work_id,url,host,status)')
    current = sqlite3.connect(tmp_path / 'current.sqlite')
    current.execute('CREATE TABLE papers(work_id,doi,journal,source_id,status,pmcids,abstract_gap)')
    current.execute('CREATE TABLE publisher_remaining_jobs(doi)')
    for i in range(7):
        doi = f'10.1038/test{i}'
        current.execute('INSERT INTO papers VALUES(?,?,?,?,?,?,?)',
                        (str(i), doi, 'Example', 'S1', 'pending_no_primary_type_date', '[]', 1))
        url = f'https://www.nature.com/articles/test{i}' + ('.pdf' if i == 3 else '')
        links.execute('INSERT INTO links VALUES(?,?,?,?)',
                      (str(i), url, 'www.nature.com', 'new_location_source_review_required'))
    links.execute('INSERT INTO links VALUES(?,?,?,?)', ('1', 'https://pmc/1', 'pmc', 'new_archive_metadata_lead'))
    current.execute('INSERT INTO publisher_remaining_jobs VALUES(?)', ('10.1038/test2',))
    current.execute('UPDATE papers SET pmcids=? WHERE work_id=?', (json.dumps(['PMC5']), '5'))
    current.execute('UPDATE papers SET doi=? WHERE work_id=?', ('10.1057/legacy', '6'))
    links.commit(); links.close(); current.commit(); current.close()
    monkeypatch.setattr(module, 'FOLDER', tmp_path)
    monkeypatch.setattr(module, 'connect', lambda: sqlite3.connect(tmp_path / 'current.sqlite'))
    monkeypatch.setattr(module, 'cached_page', lambda db, wid, url: {'receipt': 'old'} if wid == '4' else None)
    selected, skipped = module.discover()
    assert [r['work_id'] for r in selected] == ['0']
    assert skipped == {'archive_route_retained': 2, 'existing_publisher_queue_preserved': 1,
                       'different_format_or_url_preserved': 1, 'existing_native_page_preserved': 1,
                       'unsupported_publisher_doi_route_preserved': 1}
