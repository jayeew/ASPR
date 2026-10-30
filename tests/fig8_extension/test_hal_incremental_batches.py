from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

from experiments.fig8_extension.hal_fixed_metadata import request_params, stable_batches


def test_new_identifiers_keep_completed_and_failed_request_groups(tmp_path: Path) -> None:
    original = ['hal-200', 'hal-400']
    failed = ['hal-500']
    (tmp_path / 'a.json').write_text(json.dumps({'identifiers': original, 'http_status': 200}))
    (tmp_path / 'b.json').write_text(json.dumps({'identifiers': failed, 'http_status': 403}))
    groups = stable_batches(['hal-100', 'hal-200', 'hal-300', 'hal-400', 'hal-500'], tmp_path)
    assert groups == [original, failed, ['hal-100', 'hal-300']]
    assert sorted(v for batch in groups for v in batch) == ['hal-100', 'hal-200', 'hal-300', 'hal-400', 'hal-500']


def test_new_batches_respect_identifier_and_encoded_query_limits(tmp_path: Path) -> None:
    identifiers = [f'hal-{i:08d}' for i in range(501)]
    groups = stable_batches(identifiers, tmp_path)
    assert [v for batch in groups for v in batch] == identifiers
    assert max(len(batch) for batch in groups) <= 200
    assert all(len(urlencode(request_params(batch))) <= 6000 for batch in groups)
    assert len(groups) < 11  # The former 50-identifier grouping required 11 requests.
