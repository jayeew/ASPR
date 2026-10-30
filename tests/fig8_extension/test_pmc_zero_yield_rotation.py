import csv
import gzip
import json

from experiments.fig8_extension import pmc_journal_harvest as harvest


def test_zero_yield_rotates_without_closing_unfinished_set(tmp_path, monkeypatch):
    monkeypatch.setattr(harvest, 'ROOT', tmp_path)
    monkeypatch.setattr(harvest, 'STOP_AFTER_PAGE', False)
    monkeypatch.setattr(harvest, 'wait_window', lambda *_: None)
    folder = tmp_path/'supplement';folder.mkdir()
    with (folder/'individual_acquisition_plan.csv').open('w') as f:
        f.write('pmcid,doi,action\nPMC1,10.1/a,fetch_missing_or_conflicting\n')
    with (folder/'pmc_journal_batch_plan.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['status','remaining_pmc_links','source_id','proposed_set','journal'])
        w.writeheader()
        for domain in ['first','second']:
            w.writerow(dict(status='candidate_set_requires_api_validation',remaining_pmc_links=20,
                            source_id=domain,proposed_set=domain,journal=domain))
    calls=[]
    def fetch(session, params, target):
        calls.append(params)
        raw=target/'page.gz';raw.write_bytes(gzip.compress(str(len(calls)).encode()))
        return {'http_status':200,'raw_receipt':str(raw.relative_to(tmp_path))}
    monkeypatch.setattr(harvest,'fetch',fetch)
    monkeypatch.setattr(harvest,'wanted_records',lambda data,targets: ([], 'continue' if int(data)<=100 else '', []))
    harvest.run()
    state=json.loads((folder/'pmc_journal_batches/status.json').read_text())
    assert len(calls)==101
    assert calls[-1]['set']=='second'
    assert state['completed_sets']==['second']
    assert state['deferred_sets']['first']['final_task_closure'] is False
    assert state['status']=='batch_pass_finished_deferred_sets_retained'
    checkpoint=tmp_path/state['deferred_sets']['first']['checkpoint']
    assert json.loads(checkpoint.read_text())['resumption_token']=='continue'
