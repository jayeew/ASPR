import json
import sqlite3

from experiments.fig8_extension.springer_author_cache import already_processed


def test_raw_author_pointer_reuses_full_parsed_record_digest():
    from experiments.fig8_extension.springer_author_cache import source_digest
    full = {'source': 'pmc', 'route': 'pmc_archive_priority',
            'parsed': {'authors': [{'name': 'A'}], 'journal_titles': ['Example Journal']}}
    pointer = {'source': 'pmc', 'route': 'pmc_archive_priority',
               'parsed': {'journal_titles': ['Example Journal']}, 'parse_authors_from_raw': True}
    assert source_digest(full, 'raw') == source_digest(pointer, 'raw')
    pointer.pop('parse_authors_from_raw')
    assert source_digest(full, 'raw') != source_digest(pointer, 'raw')


def test_skip_only_same_work_receipt_and_current_raw_digest(tmp_path):
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE springer_author_processed(work_id TEXT,receipt TEXT,digest TEXT,status TEXT)')
    db.execute('INSERT INTO springer_author_processed VALUES(?,?,?,?)',
               ('W1', 'receipt.json', 'address-country-review-2:original', 'cache_processed'))
    path = tmp_path / 'receipt.json'
    path.write_text(json.dumps({'raw_sha256': 'original'}))
    assert already_processed(db, tmp_path, 'W1', 'receipt.json')
    assert not already_processed(db, tmp_path, 'W2', 'receipt.json')
    path.write_text(json.dumps({'raw_sha256': 'changed'}))
    assert not already_processed(db, tmp_path, 'W1', 'receipt.json')
    path.write_text('{}')
    assert not already_processed(db, tmp_path, 'W1', 'receipt.json')
