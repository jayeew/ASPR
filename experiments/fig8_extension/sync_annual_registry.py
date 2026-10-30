"""Keep candidate-year annotations aligned with the single dated annual registry."""
from __future__ import annotations

import fcntl
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .fig8b_qualification_build import write_decisions
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run(root: Path = ROOT) -> None:
    stats=Counter();conflicts=[];folder=root/'supplement';q=root/'fields/qualification'
    with (folder/'checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        annual={(r['source_id'],int(r['year'])):r for r in read_csv(root/'fields/journal_publisher_by_year.csv') if r['nature_portfolio_year'] in ['True','False']}
        db=connect(root)
        for path in sorted((root/'fields/papers').glob('*.parquet')):
            table=pq.read_table(path);rows=table.to_pylist();changed=False
            decisions=pq.read_table(q/'decisions'/path.name).to_pylist();by_id={r['work_id']:r for r in decisions};decision_changed=False
            for paper in rows:
                rule=annual.get((paper['source_id'],paper['year_openalex']))
                if not rule:continue
                label=rule['nature_portfolio_year'];old=paper['nature_portfolio_at_candidate_year'];d=by_id[paper['work_id']]
                if (old in ['True','False'] and old!=label) or (d['nature_portfolio_at_candidate_year'] in ['True','False'] and d['nature_portfolio_at_candidate_year']!=label):
                    conflicts.append({'work_id':paper['work_id'],'doi':paper['doi'],'paper_label':old,'decision_label':d['nature_portfolio_at_candidate_year'],'annual_label':label,'source_id':paper['source_id'],'candidate_year':paper['year_openalex']});continue
                evidence=rule.get('transfer_evidence_url') or rule.get('catalogue_evidence') or 'fields/journal_publisher_by_year.csv'
                values={'nature_portfolio_at_candidate_year':label,'annual_imprint':rule.get('imprint',''),'annual_publisher_evidence':evidence,'historical_portfolio_status':rule.get('status','')}
                row_changed=False
                for field,value in values.items():
                    if not value or paper[field]==value:continue
                    db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',(paper['work_id'],field,paper[field],value,evidence,'Synchronize candidate-year annotation from explicit annual registry; not a verified online-year or research-type claim'))
                    paper[field]=value;changed=True;row_changed=True;stats['changed_'+field]+=1
                stats['changed_papers']+=row_changed
                if d['nature_portfolio_at_candidate_year']!=label:
                    d['nature_portfolio_at_candidate_year']=label;decision_changed=True;stats['candidate_decision_labels_changed']+=1
            if changed:
                tmp=path.with_suffix('.parquet.tmp');pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),tmp,compression='zstd');tmp.replace(path)
            if decision_changed:write_decisions(q/'decisions'/path.name,decisions,q/'decision_changes.jsonl.gz')
            db.commit();atomic_json(folder/'annual_registry_sync_status.json',{'status':'running','shard':path.name,**stats});print(path.name,dict(stats),flush=True)
        db.close()
    atomic_json(folder/'annual_registry_sync_conflicts.json',conflicts)
    atomic_json(folder/'annual_registry_sync_status.json',{'status':'complete_with_explicit_conflicts' if conflicts else 'complete','conflicts':len(conflicts),**stats})


if __name__=='__main__':run()
