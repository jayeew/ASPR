from datetime import datetime, timedelta, timezone
import json

import pytest

from experiments.fig8_extension import cell_science_closure as closure


def test_request_cap_prevents_network(tmp_path, monkeypatch):
    monkeypatch.setattr(closure, 'AUDIT', tmp_path)
    monkeypatch.setattr(closure.bounded, 'init', lambda: {'network_stop_utc': (datetime.now(timezone.utc)+timedelta(minutes=1)).isoformat(), 'requests': 100, 'request_cap': 100})
    monkeypatch.setattr(closure.bounded, 'get', lambda *a: pytest.fail('Unexpected network call'))
    with pytest.raises(RuntimeError, match='Request cap'):
        closure.get('https://example.org')


def test_receipt_reuse_and_nonreceipt_json(tmp_path, monkeypatch):
    requests=tmp_path/'requests';requests.mkdir()
    record={'url':'https://example.org','params':None,'http_status':200}
    (requests/'001.json').write_text(json.dumps(record))
    (requests/'001.text.json').write_text('["page text"]')
    monkeypatch.setattr(closure, 'AUDIT', tmp_path)
    monkeypatch.setattr(closure.bounded, 'init', lambda: {'network_stop_utc': (datetime.now(timezone.utc)+timedelta(minutes=1)).isoformat(), 'requests': 1, 'request_cap': 100})
    monkeypatch.setattr(closure.bounded, 'get', lambda *a: pytest.fail('Repeated request'))
    assert closure.get('https://example.org')==record


def test_prepare_preserves_proposal_on_resume(tmp_path, monkeypatch):
    path=tmp_path/'decision_delta.json';path.write_text('[{"retained":true}]')
    monkeypatch.setattr(closure, 'AUDIT', tmp_path)
    monkeypatch.setattr(closure.bounded, 'load', lambda: pytest.fail('Proposal overwritten'))
    closure.prepare()
    assert json.loads(path.read_text())==[{'retained':True}]
