"""Consume retained publisher PDF first-page date evidence without inferring type."""
from __future__ import annotations

import fcntl
import json
import os
import time

from .publisher_delta_apply import apply_available
from .scoap3_pdf_dates import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> dict:
    staged=FOLDER/'staged';staged.mkdir(exist_ok=True)
    marker=FOLDER/'applied_results.json';done=json.loads(marker.read_text()) if marker.exists() else {}
    native={}
    for path in (FOLDER.parent/'staged').glob('*.json'):
        r=json.loads(path.read_text());native.setdefault(r['doi'],(r,path))
    jobs=[];processed={};held=[]
    for path in sorted((FOLDER/'results').glob('*.json')):
        rel=str(path.relative_to(ROOT))
        if rel in done:continue
        result=json.loads(path.read_text());doi=result['doi']
        if result['status']!='explicit_online_date_identified':
            processed[rel]=result;continue
        if doi not in native:
            held.append({'receipt':rel,'reason':'matching_native_identity_record_not_yet_available'});continue
        source,native_path=native[doi]
        proof=[e for e in result['evidence'] if e.get('status')=='explicit_online_date_identified']
        if len(proof)!=1:
            held.append({'receipt':rel,'reason':'nonunique_date_proof'});continue
        evidence=proof[0];receipt=json.loads((ROOT/evidence['receipt']).read_text())
        if receipt.get('doi')!=doi or source['parsed']['doi']!=[doi]:
            held.append({'receipt':rel,'reason':'native_or_pdf_doi_mismatch'});continue
        parsed={**source['parsed'],'article_types':[],'authors':[],'abstract':'',
                'online_date':result['online_date'],'date_precision':'day'}
        record={'source':'publisher_pdf_front','route':'elsevier','doi':doi,'url':receipt['url'],
                'raw_receipt':os.path.relpath(ROOT/receipt['raw_receipt'],ROOT/'fields/qualification'),
                'parsed':parsed,'date_evidence':evidence,'date_result_receipt':rel,
                'identity_receipt':str(native_path.relative_to(ROOT)),
                'journal_identity_receipts':source['journal_identity_receipts'],
                'identity_basis':'Native XML DOI/title and DOI-matched publisher deposit journal; PDF first page has the same unique standalone DOI',
                'type_not_inferred_from_pdf':True}
        target=staged/path.name;atomic_json(target,record)
        jobs.append((doi,str(target.relative_to(ROOT))));processed[rel]=result
    applied=apply_available(supplied_jobs=jobs) if jobs else {}
    db=connect();db.execute('PRAGMA busy_timeout=900000')
    for rel,result in processed.items():
        for tid, in db.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?",(result['doi'],)).fetchall():
            if not db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',(tid,'scoap3_publisher_pdf_date',rel)).fetchone():
                db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                           (tid,'scoap3_publisher_pdf_date',result['status'],rel,json.dumps({'final_task_closure':False,'no_original_research_inference':True})))
                db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?',(tid,))
        done[rel]=result['status']
    db.commit();db.close();atomic_json(marker,done)
    output={'results_recorded':len(processed),'date_evidence_records':len(jobs),'application':applied,'held':held,'network_requests':0}
    atomic_json(FOLDER/'date_application.json',output)
    return output


def follow() -> None:
    with (FOLDER/'consumer.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            marker=FOLDER/'applied_results.json';done=json.loads(marker.read_text()) if marker.exists() else {}
            pending=sum(str(p.relative_to(ROOT)) not in done for p in (FOLDER/'results').glob('*.json'))
            status=json.loads((FOLDER/'status.json').read_text())['status']
            terminal=status not in {'running','draining_inflight','waiting_for_native_xml_host_queue'}
            if pending>=100 or (terminal and pending):
                result=run();atomic_json(FOLDER/'consumer_status.json',{'status':'batch_processed',**result})
                if result['results_recorded']:continue
            if terminal:
                atomic_json(FOLDER/'consumer_status.json',{'status':'available_date_results_processed_unresolved_retained','unconsumed_results':pending,'source_status':status});return
            time.sleep(30)


if __name__=='__main__':follow()
