"""A finished source phase must not be confused with completed Fig8 data."""
import json

from experiments.fig8_extension import mdpi_date_phase_finish as finish


def test_finish_requires_metadata_then_runs_once(tmp_path, monkeypatch):
    monkeypatch.setattr(finish, 'FOLDER', tmp_path)
    calls = []
    monkeypatch.setattr(finish, 'apply_annual', lambda: calls.append('annual') or {'filled': 2})
    monkeypatch.setattr(finish, 'reuse_entities', lambda: calls.append('entities'))
    monkeypatch.setattr(finish, 'refresh', lambda force: calls.append(('refresh', force)))
    (tmp_path / 'status.json').write_text(json.dumps({
        'status': 'fixed_deposit_path_processed_native_type_and_metadata_queues_continue'}))
    (tmp_path / 'metadata_status.json').write_text(json.dumps({'status': 'applying_qualified_metadata'}))
    finish.run()
    assert calls == []
    assert json.loads((tmp_path / 'phase_finish_status.json').read_text())['status'] == 'phase_incomplete_retained'
    (tmp_path / 'metadata_status.json').write_text(json.dumps({'status': 'qualified_metadata_applied'}))
    finish.run()
    finish.run()
    assert calls == ['annual', 'entities', ('refresh', True)]
    state = json.loads((tmp_path / 'phase_finish_status.json').read_text())
    assert state['status'] == 'phase_reports_refreshed'
    assert state['final_task_closure'] is False
    assert state['network_requests'] == 0
