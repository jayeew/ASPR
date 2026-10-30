"""Cache-first scheduling must preserve unresolved cases without duplicate requests."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from experiments.fig8_extension import supplement_acquisition as acquisition
from experiments.fig8_extension import supplement_sources as sources


def test_plan_retains_failed_evidence_and_defers_alternative_ids(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / 'fields/qualification/supplement_parsed'
    folder.mkdir(parents=True)
    (tmp_path / 'supplement').mkdir()
    (folder / 'failure.json').write_text(json.dumps({'pmcid': 'PMC1', 'source': 'pmc', 'status': 'retry_exhausted', 'attempts': [{}, {}, {}]}))
    rows = [{'doi': '10.1/a', 'work_id': 'W1', 'pmcids': '["PMC1","PMC2","PMC3"]', 'route': 'other', 'status': 'pending_primary_type'}]
    monkeypatch.setattr(acquisition, 'candidate_rows', lambda root: rows)
    jobs = acquisition.plan(tmp_path)
    assert [item['pmcid'] for item, _ in jobs] == ['PMC2']
    receipt = json.loads((tmp_path / 'supplement/acquisition_plan.json').read_text())
    assert receipt['cached_retry_exhausted'] == 1
    assert receipt['alternate_ids_deferred_until_replan'] == 1
    assert 'retry_exhausted' in (tmp_path / 'supplement/individual_acquisition_plan.csv').read_text()
    (folder / 'second.json').write_text(json.dumps({'pmcid': 'PMC2', 'source': 'pmc', 'status': 'not_available'}))
    assert acquisition.plan(tmp_path)[0][0]['pmcid'] == 'PMC3'
    rows.append({**rows[0], 'work_id': 'W2', 'pmcids': '["PMC4"]'})
    assert len(acquisition.plan(tmp_path)) == 1


def test_cache_index_distinguishes_no_match_from_unaccepted_truncation(tmp_path: Path) -> None:
    folder = tmp_path / 'fields/epmc_raw'
    folder.mkdir(parents=True)
    (tmp_path / 'supplement').mkdir()
    for name, status, records in [('empty', 'complete', []), ('truncated', 'truncated_not_accepted', [{'doi': '10.1/a'}])]:
        with gzip.open(folder / (name + '.gz'), 'wt') as stream:
            json.dump({'dois': ['10.1/a'], 'status': status, 'response': {'resultList': {'result': records}}}, stream)
    first = acquisition.cache_index(tmp_path)
    assert first['queried_dois'] == 1
    assert first['returned_dois'] == 0
    assert acquisition.cache_index(tmp_path)['changed_files'] == 0


def test_bounded_stop_drains_without_dispatching_queued_requests(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / 'supplement').mkdir()
    called = []
    def request(item: dict, root: Path, source: str) -> dict:
        called.append(item['doi'])
        (root / 'supplement/stop_acquisition').touch()
        return {'status': 'parsed'}
    monkeypatch.setattr(sources, 'request', request)
    jobs = [({'doi': str(i)}, 'pmc') for i in range(50)]
    results = list(sources.bounded_results(jobs, tmp_path, 1))
    assert len(results) == 1
    assert called == ['0']


def test_deadline_dispatches_no_new_request(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sources, 'request', lambda *args: pytest.fail('deadline must stop dispatch'))
    assert list(sources.bounded_results([({'doi': 'a'}, 'pmc')], tmp_path, 1, deadline=0)) == []


def test_retry_budget_is_total_across_resumes(tmp_path: Path, monkeypatch) -> None:
    import hashlib
    import requests
    folder=tmp_path/'fields/qualification/supplement_parsed';folder.mkdir(parents=True)
    url='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC1/fullTextXML'
    path=folder/(hashlib.sha256(url.encode()).hexdigest()+'.json')
    path.write_text(json.dumps({'status':'request_error','attempts':[{},{}]}))
    calls=[]
    class Session:
        def get(self, *args, **kwargs):
            calls.append(1)
            raise requests.ConnectionError('private diagnostic detail')
    monkeypatch.setattr(sources.LOCAL,'session',Session(),raising=False)
    monkeypatch.setattr(sources,'paced',lambda url: None)
    item={'doi':'10.1/a','pmcid':'PMC1','route':'pmc'}
    value=sources.request(item,tmp_path,'pmc')
    assert len(calls)==1 and len(value['attempts'])==3
    assert value['status']=='retry_exhausted'
    assert 'private diagnostic detail' not in path.read_text()
    assert sources.request(item,tmp_path,'pmc')['reused_receipt']
    assert len(calls)==1
