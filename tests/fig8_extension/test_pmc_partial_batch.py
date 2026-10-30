import json

import pytest

from experiments.fig8_extension import pmc_journal_follow as consumer


def test_startup_consumes_retained_partial_batch_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(consumer, 'FOLDER', tmp_path)
    (tmp_path/'status.json').write_text(json.dumps({'status':'running','new_front_records':123}))
    pending=iter([123,0]);called=[]
    monkeypatch.setattr(consumer,'pending_receipts',lambda done: next(pending))
    def consume(done):
        called.append(True)
        done.update(str(i) for i in range(123))
    monkeypatch.setattr(consumer,'consume',consume)
    def stop(_):
        raise InterruptedError('finish one loop')
    monkeypatch.setattr(consumer.time,'sleep',stop)
    with pytest.raises(InterruptedError):
        consumer.run()
    state=json.loads((tmp_path/'consumer_status.json').read_text())
    assert called==[True]
    assert state['consumed']==123 and state['pending_receipts']==0
    assert state['status']=='waiting_for_increment'
    assert state['max_batch_wait_seconds']==600
