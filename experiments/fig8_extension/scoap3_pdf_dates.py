"""Finish a fixed PDF-date fallback queue after native XML collection releases the host."""
from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import httpx
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .publisher_pdf_online_date import extract
from .scoap3_native_collect import FOLDER as XML
from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json, paced

FOLDER = XML / 'elsevier/pdf_date'


def still_missing(doi: str) -> bool:
    db=sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro',uri=True)
    papers=db.execute("SELECT work_id FROM papers WHERE doi=? AND status LIKE 'pending%'",(doi,)).fetchall()
    result=any(not db.execute("SELECT 1 FROM current_evidence_index WHERE work_id=? AND kind='primary' AND json_extract(payload,'$.identity')='matched' AND coalesce(json_extract(payload,'$.online_date'),'')<>'' LIMIT 1",(wid,)).fetchone() for wid, in papers)
    db.close();return result


def fetch(job: dict) -> dict:
    token=hashlib.sha256(job['url'].encode()).hexdigest();path=FOLDER/(token+'.json');pdf=FOLDER/(token+'.pdf')
    if path.exists():return json.loads(path.read_text())
    record={**job,'route':'direct_trust_env_false','attempts':[]}
    with httpx.Client(trust_env=False,timeout=35,follow_redirects=False) as client:
        for attempt in range(3):
            paced(job['url'])
            try:
                response=client.get(job['url']);record['http_status']=response.status_code
                record['attempts'].append({'http_status':response.status_code})
                if response.status_code not in {500,502,503,504}:
                    if response.status_code==200 and response.content.startswith(b'%PDF'):
                        atomic_bytes(pdf,response.content);record.update(raw_receipt=str(pdf.relative_to(ROOT)),sha256=hashlib.sha256(response.content).hexdigest())
                    break
            except httpx.HTTPError as exc:
                record['attempts'].append({'error':type(exc).__name__,'detail':str(exc)})
            if attempt<2:time.sleep(2**attempt)
    atomic_json(path,record);return record


def process(doi: str, jobs: list[dict]) -> dict:
    target=FOLDER/'results'/(hashlib.sha256(doi.encode()).hexdigest()+'.json')
    if target.exists():return json.loads(target.read_text())
    result={'doi':doi,'status':'no_explicit_date_confirmed','evidence':[]}
    if not still_missing(doi):
        result['status']='current_eligibility_or_known_date_skipped';atomic_json(target,result);return result
    review=XML/'resource_restriction_review.json'
    restrictions=json.loads(review.read_text()) if review.exists() else {}
    for job in jobs:
        if restrictions and ('/media/files/' in job['url'] or doi in restrictions['denied_dois']):
            result['evidence'].append({'url':job['url'],'status':'reviewed_restricted_path_or_doi_not_requested'});continue
        receipt=fetch(job);token=hashlib.sha256(job['url'].encode()).hexdigest()
        evidence={'receipt':str((FOLDER/(token+'.json')).relative_to(ROOT))}
        if receipt.get('http_status') in {401,403,429}:
            result.update(status='source_permission_or_rate_restriction',stop_source=True);result['evidence'].append(evidence);break
        if receipt.get('raw_receipt'):
            pdf=ROOT/receipt['raw_receipt'];text_path=pdf.with_suffix('.txt')
            try:
                if text_path.exists():first=text_path.read_text()
                else:
                    first=PdfReader(pdf).pages[0].extract_text(extraction_mode='layout');atomic_bytes(text_path,first.encode())
                parsed=extract(first,doi);evidence.update(parsed,first_page_text=str(text_path.relative_to(ROOT)))
            except (PdfReadError,ValueError,IndexError,KeyError) as exc:
                evidence.update(status='pdf_first_page_parse_failed',error=type(exc).__name__)
        else:evidence['status']='pdf_not_available'
        result['evidence'].append(evidence)
        if evidence['status']=='explicit_online_date_identified':
            result.update(status=evidence['status'],online_date=evidence['online_date']);break
    atomic_json(target,result);return result


def run() -> None:
    (FOLDER/'results').mkdir(exist_ok=True)
    with (FOLDER/'collector.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(FOLDER/'status.json',{'status':'waiting_for_native_xml_host_queue','network_requests':0})
        with (XML/'collector.lock').open('a') as host:
            fcntl.flock(host,fcntl.LOCK_EX)
            state=json.loads((XML/'status.json').read_text())
            if state['status'] not in {'fixed_xml_queue_acquired_review_required','fixed_public_xml_queue_processed_restricted_path_retained'}:
                atomic_json(FOLDER/'status.json',{'status':'unattempted_native_source_not_successfully_finished','native_status':state});return
            groups=defaultdict(list)
            for job in json.loads((FOLDER/'fixed_scope.json').read_text())['jobs']:groups[job['doi']].append(job)
            done={json.loads(p.read_text())['doi'] for p in (FOLDER/'results').glob('*.json')}
            counts=Counter(json.loads(p.read_text())['status'] for p in (FOLDER/'results').glob('*.json'))
            pending=iter((doi,jobs) for doi,jobs in sorted(groups.items()) if doi not in done);stop=''
            with ThreadPoolExecutor(max_workers=8) as pool:
                active={}
                def submit() -> None:
                    item=next(pending,None)
                    if item:active[pool.submit(process,*item)]=item[0]
                for _ in range(8):submit()
                while active:
                    ready,_=wait(active,return_when=FIRST_COMPLETED)
                    for future in ready:
                        active.pop(future);result=future.result();counts[result['status']]+=1
                        if result.get('stop_source'):stop=result['status']
                        atomic_json(FOLDER/'status.json',{'status':'draining_inflight' if stop else 'running','workers':8,'request_start_interval_seconds':1.05,'fixed_dois':len(groups),'completed':sum(counts.values()),'outcomes':dict(counts),'stop_reason':stop,'qualification_not_implied':True})
                        if not stop:submit()
            atomic_json(FOLDER/'status.json',{'status':'source_stopped_unattempted_retained' if stop else 'fixed_pdf_date_queue_processed','fixed_dois':len(groups),'outcomes':dict(counts),'stop_reason':stop})


if __name__=='__main__':run()
