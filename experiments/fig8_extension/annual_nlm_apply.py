"""Apply reviewed NLM intervals without rerunning qualification or acquisition."""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from datetime import datetime, timezone

import pyarrow.parquet as pq

from .fig8b_qualification_build import write_decisions
from .supplement_queue import ROOT, connect, task
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv, write_csv


def run(proposal_name: str = 'annual_proposals.json') -> None:
    folder=ROOT/'supplement/annual_nlm'
    if proposal_name not in {'annual_proposals.json', 'lookup_proposals.json', 'full_scope_proposals.json'}:
        raise ValueError('Unrecognized reviewed proposal receipt')
    marker=folder/('applied.json' if proposal_name=='annual_proposals.json' else proposal_name.replace('_proposals.json','_applied.json'))
    evidence_path='supplement/annual_nlm/'+proposal_name
    with (ROOT/'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        q=ROOT/'fields/qualification'
        if marker.exists():
            counts=json.loads(marker.read_text())
            if counts.get('derived_status')=='complete':
                print('Annual evidence and derived tables already current.',flush=True)
                return
            print('Resuming derived refresh; field updates already applied.',flush=True)
        else:
            proposals=json.loads((folder/proposal_name).read_text())
            rules={(r['source_id'],r['year']):r for r in proposals}
            registry_path=ROOT/'fields/journal_publisher_by_year.csv';registry=read_csv(registry_path)
            for row in registry:
                rule=rules.get((row['source_id'],int(row['year'])))
                if not rule:continue
                assert row['nature_portfolio_year']!='True'
                row.update(nature_portfolio_year='False',imprint=rule['publisher'],status=rule['status'],
                           comparison_group='non_nature_portfolio',transfer_evidence_url=evidence_path,membership_scope=rule['scope'])
            write_csv(registry_path,registry)
            q=ROOT/'fields/qualification';counts={'annual_rules':len(rules),'decision_rows':0,'included_labels_filled':0}
            for path in sorted((q/'decisions').glob('*.parquet')):
                rows=pq.read_table(path).to_pylist();changed=False
                for row in rows:
                    rule=rules.get((row['source_id'],row['online_year']))
                    if not rule or row['nature_portfolio_at_verified_online_year'] in ['True','False']:continue
                    row['nature_portfolio_at_verified_online_year']='False'
                    row['portfolio_evidence_ids']=json.dumps([f'nlm-imprint:{rule["nlm_id"]}:{rule["year"]}'])
                    counts['decision_rows']+=1;counts['included_labels_filled']+=row['eligibility_status']=='included_verified_research';changed=True
                if changed:write_decisions(path,rows,q/'decision_changes.jsonl.gz')
            db=connect(ROOT)
            for rule in proposals:
                tid=task(db,'portfolio',rule['source_id']+':'+str(rule['year']),'dated_nlm_catalogue',rule,status='resolved')
                db.execute('UPDATE tasks SET receipt=?,reason=? WHERE task_id=?',(evidence_path,'Exact ISSN and explicit publisher imprint interval; boundary years excluded',tid))
            db.commit();db.close()
            counts['applied_at']=datetime.now(timezone.utc).isoformat();counts['derived_status']='refreshing'
            atomic_json(marker,counts);print(json.dumps(counts),flush=True)
        for module,args in [('fig8b_qualification_recompute',[str(q),'--refresh']),('supplement_report',[]),('journal_readiness',['--lock-held'])]:
            log=folder/(module+'.log')
            with log.open('a') as out:
                subprocess.run([sys.executable,'-u','-m','experiments.fig8_extension.'+module,*args],stdout=out,stderr=subprocess.STDOUT,check=True)
            print('Refreshed',module,flush=True)
        counts['derived_status']='complete';atomic_json(marker,counts)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--proposal', default='annual_proposals.json')
    run(parser.parse_args().proposal)
