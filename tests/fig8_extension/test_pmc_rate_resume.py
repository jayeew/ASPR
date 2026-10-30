import hashlib
import json
from datetime import datetime, timedelta, timezone

from experiments.fig8_extension import pmc_journal_harvest as harvest


class Response:
    status_code = 200
    content = b'<OAI-PMH/>'
    headers = {}


class Session:
    def __init__(self):
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        return Response()


def setup_receipt(tmp_path, monkeypatch, status=429, future=False, attempts=1):
    monkeypatch.setattr(harvest, 'ROOT', tmp_path)
    monkeypatch.setattr(harvest.time, 'sleep', lambda _: None)
    params = {'verb': 'ListRecords', 'resumptionToken': 'retained-token'}
    token = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    path = tmp_path/(token+'.json')
    path.write_text(json.dumps({'http_status': status, 'attempts': [{'http_status': status}]*attempts}))
    policy = {'request_token': token, 'not_before_utc': (datetime.now(timezone.utc)+timedelta(hours=1 if future else -1)).isoformat()}
    (tmp_path/'deferred_rate_retry.json').write_text(json.dumps(policy))
    return params, path


def test_delayed_retry_preserves_original_and_is_cached(tmp_path, monkeypatch):
    params, path = setup_receipt(tmp_path, monkeypatch)
    before = path.read_bytes()
    session = Session()
    result = harvest.fetch(session, params, tmp_path)
    assert result['http_status'] == 200 and result['previous_attempt_count'] == 1
    assert path.read_bytes() == before
    assert result['raw_receipt'].endswith('.deferred_retry.xml.gz')
    assert harvest.fetch(session, params, tmp_path) == result
    assert session.calls == 1


def test_no_early_permission_or_exhausted_retry(tmp_path, monkeypatch):
    for status, future, attempts in [(429, True, 1), (403, False, 1), (429, False, 3)]:
        params, _ = setup_receipt(tmp_path, monkeypatch, status, future, attempts)
        session = Session()
        assert harvest.fetch(session, params, tmp_path)['http_status'] == status
        assert session.calls == 0
