import json
import sqlite3
from pathlib import Path

from experiments.fig8_extension.change_source_audit import run, source_paths


def test_capture_mapping_does_not_accept_an_uncached_url(tmp_path: Path) -> None:
    fields = tmp_path / 'fields'
    fields.mkdir()
    (fields / 'page.html').write_text('Publisher transfer notice')
    (fields / 'official_capture_index.json').write_text(json.dumps([
        {'url': 'https://example.org/journal', 'http_status': 200, 'raw_file': 'page.html'},
        {'url': 'https://example.org/absent', 'http_status': 200, 'raw_file': 'absent.html'}]))
    assert source_paths('https://example.org/journal', tmp_path)['status'] == 'source_paths_resolved'
    assert source_paths('https://example.org/absent', tmp_path)['status'] == 'unresolved_source_paths'
    assert source_paths('fields/page.html#notice', tmp_path)['status'] == 'source_paths_resolved'


def test_incremental_repeat_preserves_counts_and_missing_evidence(tmp_path: Path) -> None:
    (tmp_path / 'supplement').mkdir()
    (tmp_path / 'source').write_text('Evidence')
    db = sqlite3.connect(tmp_path / 'supplement/tasks.sqlite')
    db.execute('CREATE TABLE changes(id INTEGER PRIMARY KEY,work_id TEXT,field TEXT,receipt TEXT,reason TEXT,before_value TEXT,after_value TEXT)')
    receipt = json.dumps({'identifier_receipt': 'source', 'publisher_receipts': ['source']})
    db.execute('INSERT INTO changes VALUES(1,?,?,?,?,?,?)', ('w', 'doi', receipt, 'identity verified', 'old', 'new'))
    db.commit()
    assert run(tmp_path)['structurally_resolved_changes'] == 1
    assert run(tmp_path)['status'] == 'no_new_changes'
    db.execute('INSERT INTO changes VALUES(2,?,?,?,?,?,?)', ('w', 'abstract', 'absent', 'repair', '', 'new'))
    db.commit()
    db.close()
    result = run(tmp_path)
    assert result['reviewed_changes'] == 2
    assert result['changes_with_issues'] == 1


def test_orcid_review_receipt_requires_every_explicit_source(tmp_path):
    import json
    from experiments.fig8_extension.change_source_audit import source_paths
    for name in ['identifier.json', 'publisher.xml', 'review.json']:
        (tmp_path/name).write_text('source')
    receipt={'identifier_receipt':'identifier.json','publisher_receipts':['publisher.xml'],
             'review':'review.json'}
    result=source_paths(json.dumps(receipt),tmp_path)
    assert result['status']=='source_paths_resolved' and len(result['resolved_files'])==3
    (tmp_path/'review.json').unlink()
    assert source_paths(json.dumps(receipt),tmp_path)['status']=='unresolved_source_paths'
    assert source_paths(json.dumps({**receipt,'review':{}}),tmp_path)['status']=='unsupported_receipt_structure'
    assert source_paths(json.dumps({**receipt,'unexpected':'extra'}),tmp_path)['status']=='unsupported_receipt_structure'
