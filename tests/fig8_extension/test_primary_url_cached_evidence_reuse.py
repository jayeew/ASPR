from pathlib import Path

from experiments.fig8_extension.primary_url_cached_evidence_reuse import usable_evidence


def test_reuse_requires_current_identity_and_existing_source(tmp_path: Path) -> None:
    folder = tmp_path / 'fields/qualification'
    folder.mkdir(parents=True)
    (folder / 'native.xml').write_text('<article/>')
    item = {'id': 'e1', 'doi': '10.1/example', 'identity': 'matched', 'receipt': 'native.xml'}
    assert usable_evidence(item, '10.1/example', tmp_path)
    assert not usable_evidence(item, '10.1/other', tmp_path)
    assert not usable_evidence({**item, 'identity': 'mismatch'}, '10.1/example', tmp_path)
    assert not usable_evidence({**item, 'receipt': 'missing.xml'}, '10.1/example', tmp_path)
    assert not usable_evidence({**item, 'receipt': ''}, '10.1/example', tmp_path)
