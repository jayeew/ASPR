import gzip
import hashlib
import json
from pathlib import Path

import pytest

from experiments.fig8_extension import pmc_cloud_inventory as inventory


def test_failed_shard_does_not_skip_remaining_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / 'cloud'
    folder.mkdir()
    (folder / 'scope.json').write_text(json.dumps({'papers': [{'pmcids': ['PMC1']}]}))
    payload = gzip.compress(b'pmc-oa-opendata,metadata/PMC1.1.json,date,etag\n')
    (tmp_path / 'raw.gz').write_bytes(gzip.compress(payload))
    files = [{'key': str(i), 'size': len(payload), 'MD5checksum': hashlib.md5(payload).hexdigest()} for i in range(5)]
    (folder / 'manifest.json').write_text(json.dumps({'fileFormat': 'CSV', 'fileSchema': 'Bucket, Key, LastModifiedDate, ETag', 'files': files}))
    monkeypatch.setattr(inventory, 'FOLDER', folder)
    monkeypatch.setattr(inventory, 'ROOT', tmp_path)
    calls = []

    def fetch(url: str, params: dict) -> dict:
        calls.append(url)
        if url.endswith('/0'):
            return {'attempts': [{'error': 'ConnectionError'}] * 3}
        return {'http_status': 200, 'raw_receipt': 'raw.gz'}

    monkeypatch.setattr(inventory, 'fetch', fetch)
    inventory.run()
    state = json.loads((folder / 'inventory_status.json').read_text())
    assert len(calls) == 5
    assert state['status'] == 'fixed_inventory_partially_available'
    assert not state['inventory_complete']
    assert len(state['failed_files']) == 1
    inventory.run()
    assert len(calls) == 5
