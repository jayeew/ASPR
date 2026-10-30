from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

from experiments.fig8_extension.supplement_apply import deposited_metadata


def test_unchanged_deposit_not_reparsed_and_updated_cache_is_applied(tmp_path: Path) -> None:
    folder = tmp_path / 'fields/crossref_raw'
    folder.mkdir(parents=True)
    path = folder / 'batch.gz'
    db = sqlite3.connect(':memory:')
    db.execute('CREATE TABLE source_metadata(receipt TEXT PRIMARY KEY,doi TEXT,kind TEXT,raw_hash TEXT,payload TEXT)')
    def write(title: str) -> None:
        record = {'status': 'success', 'url': 'https://api.crossref.org/works',
                  'response': {'message': {'items': [{'DOI': '10.1/example', 'title': [title]}]}}}
        with gzip.open(path, 'wt') as stream:
            json.dump(record, stream)
    write('Original')
    assert deposited_metadata(tmp_path, db) == 1
    with patch('experiments.fig8_extension.supplement_apply.gzip.open', side_effect=AssertionError('unchanged cache reread')):
        assert deposited_metadata(tmp_path, db) == 0
    write('Updated explicit source title')
    assert deposited_metadata(tmp_path, db) == 1
    assert json.loads(db.execute('SELECT payload FROM source_metadata').fetchone()[0])['title'] == ['Updated explicit source title']
    assert db.execute('SELECT count(*) FROM source_metadata').fetchone()[0] == 1
