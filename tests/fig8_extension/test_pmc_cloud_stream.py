import gzip
import hashlib

import pytest

from experiments.fig8_extension.pmc_cloud_stream import match_shard


def test_cached_shard_keeps_all_versions_only_within_fixed_scope() -> None:
    payload = gzip.compress(b'pmc-oa-opendata,metadata/PMC1.1.json,date,etag\npmc-oa-opendata,metadata/PMC1.2.json,date,etag\npmc-oa-opendata,metadata/PMC2.1.json,date,etag\n')
    item = {'size': len(payload), 'MD5checksum': hashlib.md5(payload).hexdigest()}
    result = match_shard(payload, item, {'PMC1'}, 'raw/cache')
    assert [r['version'] for r in result] == [1, 2]
    assert all(r['inventory_raw_receipt'] == 'raw/cache' for r in result)
    with pytest.raises(ValueError, match='checksum'):
        match_shard(payload + b'x', item, {'PMC1'}, 'raw/cache')
