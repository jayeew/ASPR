"""Consume all unrequested open-access priority-journal archive leads, with finite retries."""
from __future__ import annotations

import fcntl
import hashlib
import json
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, request


def run() -> dict:
    folder=ROOT/'supplement/priority_archive';stats=Counter();stop=''
    with (folder/'collector.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior=folder/'status.json'
        if prior.exists() and json.loads(prior.read_text()).get('stop_reason'):
            return json.loads(prior.read_text())
        targets={r['pmcid']:r for r in json.loads((folder/'access_scope.json').read_text()) if r['access_path_status']=='public_xml_oa_lead'}
        db=connect(ROOT);db.execute('PRAGMA busy_timeout=900000');db.execute('CREATE TABLE IF NOT EXISTS priority_archive_jobs(pmcid TEXT PRIMARY KEY,doi TEXT,status TEXT,receipt TEXT)');db.commit()
        done={r[0] for r in db.execute('SELECT pmcid FROM priority_archive_jobs')};pending=iter([r for k,r in targets.items() if k not in done])
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs={}
            def submit() -> None:
                item=next(pending,None)
                if item:jobs[pool.submit(request,item,ROOT,'pmc')]=item
            for _ in range(8):submit()
            while jobs:
                completed,_=wait(jobs,return_when=FIRST_COMPLETED)
                for future in completed:
                    item=jobs.pop(future);record=future.result();status=record['status'];receipt='fields/qualification/'+record['parsed_receipt']
                    if status=='parsed' and item['doi'] not in record['parsed'].get('doi',[]):status='requested_doi_not_in_archived_front'
                    db.execute('INSERT OR REPLACE INTO priority_archive_jobs VALUES(?,?,?,?)',(item['pmcid'],item['doi'],status,receipt))
                    ident=hashlib.sha256(('qualification:'+item['doi']).encode()).hexdigest()[:32]
                    db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',(ident,'epmc_public_archive_priority',status,receipt,json.dumps({'pmcid':item['pmcid'],'identity_and_type_not_yet_applied':True})))
                    db.commit();stats[status]+=1;stats['http_attempts']+=0 if record.get('reused_receipt') else len(record.get('attempts',[]));stats['cached_receipts_reused']+=bool(record.get('reused_receipt'))
                    if record.get('http_status') in {401,403,429}:stop='Source access or rate restriction; remaining leads not requested'
                    atomic_json(prior,{'status':'draining' if stop else 'running','fixed_identifiers':len(targets),'previously_processed':len(done),'stop_reason':stop,**stats})
                    if not stop:submit()
        result={'status':'source_stopped_remaining_explicit' if stop else 'fixed_oa_archive_scope_processed','fixed_identifiers':len(targets),'previously_processed':len(done),'stop_reason':stop,**stats};atomic_json(prior,result);db.close();print(json.dumps(result),flush=True);return result


if __name__=='__main__':run()
