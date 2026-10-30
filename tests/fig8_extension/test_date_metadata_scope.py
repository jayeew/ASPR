"""A queued metadata consumer must not read or overwrite another queue's state."""
import json

from experiments.fig8_extension import current_online_date_follow as consumer


def test_custom_queue_uses_own_completion_marker(tmp_path, monkeypatch):
    original = tmp_path / 'original'
    original.mkdir()
    prior = {'status': 'qualified_metadata_applied', 'qualified_dois': 7}
    (original / 'metadata_status.json').write_text(json.dumps(prior))
    selected = tmp_path / 'selected'
    selected.mkdir()
    (selected / 'status.json').write_text(json.dumps({'status': 'source_restricted'}))
    monkeypatch.setattr(consumer, 'FOLDER', original)
    consumer.run(selected, 'expected_success')
    assert json.loads((original / 'metadata_status.json').read_text()) == prior
    result = json.loads((selected / 'metadata_status.json').read_text())
    assert result['status'] == 'date_queue_not_complete_no_duplicate_acquisition'
    assert result['source_state']['status'] == 'source_restricted'
