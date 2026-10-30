"""Later native qualification triggers scoped cached metadata, with safe replay."""
import json

from experiments.fig8_extension import mdpi_new_qualified_metadata as worker


def test_terminal_flush_is_scoped_and_repeat_safe(tmp_path, monkeypatch):
    folder = tmp_path / 'supplement' / 'dates'
    target = tmp_path / 'supplement' / 'types'
    folder.mkdir(parents=True)
    target.mkdir()
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(worker, 'FOLDER', folder)
    monkeypatch.setattr(worker, 'TARGET', target)
    (folder / 'scope.json').write_text(json.dumps({'papers': [{'doi': '10.example/new'}]}))
    (folder / 'metadata_status.json').write_text(json.dumps({'status': 'qualified_metadata_applied'}))
    (target / 'status.json').write_text(json.dumps({'status': 'fixed_queue_processed'}))
    (target / 'consumer_status.json').write_text(json.dumps({'status': 'terminal_cache_processed'}))

    class Database:
        def execute(self, sql):
            assert "status='included_verified_research'" in sql
            return [('10.example/new',), ('10.example/outside',)]

        def close(self):
            pass

    calls = []
    monkeypatch.setattr(worker, 'connect', Database)
    monkeypatch.setattr(worker, 'apply', lambda **kwargs: calls.append(kwargs) or {'changed': 1})
    monkeypatch.setattr(worker, 'reuse_entities', lambda: None)
    monkeypatch.setattr(worker, 'refresh', lambda: None)
    worker.run()
    worker.run()
    assert calls == [{'scope_dois': {'10.example/new'}, 'index_sources': False}]
    state = json.loads((folder / 'new_qualified_metadata_status.json').read_text())
    assert state['pending'] == 0
    assert state['final_task_closure'] is False
    assert state['network_requests'] == 0
