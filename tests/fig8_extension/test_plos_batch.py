import pytest

from experiments.fig8_extension.plos_batch import parameters
from experiments.fig8_extension.plos_batch_apply import metadata
from experiments.fig8_extension.publisher_delta_apply import publisher_items


def test_plos_batch_is_bounded_and_doi_scoped():
    p = parameters(['10.1371/journal.pone.1'])
    assert p['rows'] == 100 and p['fq'] == 'doc_type:full'
    with pytest.raises(ValueError):
        parameters(['10.1371/x'] * 51)


def test_plos_does_not_confuse_summary_date_or_flat_affiliations():
    doc = {'id': '10.1371/test', 'title_display': 'A title', 'article_type': 'Research Article',
           'publication_date': '2024-01-02T00:00:00Z', 'abstract': ['abstract plus lay summary'],
           'affiliate': ['Editor institute'], 'author_display': ['Author']}
    p = metadata(doc)
    assert p['abstract'] == '' and p['authors'] == [] and p['online_date'] == ''
    assert p['publication_dates'] == ['2024-01-02']
    doc['abstract_primary_display'] = ['Primary abstract']
    assert metadata(doc)['abstract'] == 'Primary abstract'
    item = publisher_items({'route': 'plos', 'url': 'https://api.plos.org/search', 'parsed': p})[0]
    assert item['online_date'] == '' and item['publication_dates'] == ['2024-01-02']


def test_staging_repeat_ignores_its_generated_outcome_file(tmp_path, monkeypatch):
    import gzip
    import hashlib
    import json
    from experiments.fig8_extension import plos_batch_apply as module
    folder = tmp_path / 'supplement/plos_batch'
    folder.mkdir(parents=True)
    doi = '10.1371/test'
    (folder / 'scope.json').write_text(json.dumps({'papers': [{'doi': doi}]}))
    raw = json.dumps({'response': {'numFound': 1, 'docs': [{'id': doi}]}}).encode()
    (folder / 'source.gz').write_bytes(gzip.compress(raw))
    (folder / 'batch_0000.json').write_text(json.dumps({
        'dois': [doi], 'http_status': 200, 'raw_receipt': 'supplement/plos_batch/source.gz',
        'sha256': hashlib.sha256(raw).hexdigest(), 'url': 'https://api.plos.org/search'}))
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    calls = []
    def apply(supplied_jobs):
        calls.append(supplied_jobs)
        return {'prepared': len(supplied_jobs)}
    monkeypatch.setattr(module, 'apply_available', apply)
    assert module.run() == module.run() == {'prepared': 1}
    assert calls[0] == calls[1]
    assert len(list((folder / 'records').glob('*.json'))) == 1
