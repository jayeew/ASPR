"""Inspect fixed repository PDF fronts for explicit publisher metadata only."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import io
import json
import logging
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .supplement_queue import ROOT
from .supplement_sources import atomic_json

FOLDER = ROOT/'supplement/hal_metadata/pdf_fronts'
STOP = threading.Event()
PACE = threading.Lock()
LAST = 0.0
LOCAL = threading.local()
HOST_HOLDS: set[str] = set()


def fetch(job: dict) -> dict:
    global LAST
    url = job['url'];token = hashlib.sha256(url.encode()).hexdigest()
    path = FOLDER/'receipts'/(token+'.json');body = FOLDER/'raw'/(token+'.gz')
    state = json.loads(path.read_text()) if path.exists() else {'url':url,'attempts':[]}
    if state.get('review_status'):
        if (state['review_status'] == 'front_metadata_extracted_not_qualification'
                and state.get('fixed_papers') != job['papers']):
            state['fixed_papers'] = job['papers']
            atomic_json(path, state)
        return state
    if urlparse(url).hostname in HOST_HOLDS:
        return {'review_status':'unattempted_on_restricted_host','url':url}
    if STOP.is_set():
        return {'review_status':'unattempted_after_source_stop','url':url}
    if not hasattr(LOCAL,'session'):
        LOCAL.session = requests.Session();LOCAL.session.trust_env = False
    while state.get('http_status') != 200 and len(state['attempts']) < 3:
        if state.get('http_status') in {400,401,403,404,410,429} or STOP.is_set():break
        with PACE:
            time.sleep(max(0,0.5-(time.monotonic()-LAST)));LAST=time.monotonic()
        attempt = {'started_at':time.time()};state['attempts'].append(attempt)
        try:
            with LOCAL.session.get(url,timeout=(10,35),stream=True) as response:
                state['http_status'] = response.status_code;attempt['http_status'] = response.status_code
                state['content_type'] = response.headers.get('Content-Type','')
                if response.status_code in {401,403}:
                    HOST_HOLDS.add(urlparse(url).hostname)
                    atomic_json(FOLDER/'host_holds.json',{'hosts':sorted(HOST_HOLDS),'basis':'HTTP 401/403 on the exact host; no URL rewriting'})
                if response.status_code == 429:
                    STOP.set();atomic_json(FOLDER/'source_stop.json',{'url':url,'status':response.status_code})
                chunks = [];size = 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > 30_000_000:
                        state['review_status'] = 'file_exceeds_metadata_fetch_limit';break
                    chunks.append(chunk)
                if not state.get('review_status'):
                    body.write_bytes(gzip.compress(b''.join(chunks)));state['raw_receipt'] = str(body.relative_to(ROOT))
        except requests.RequestException as error:
            state.pop('http_status',None);attempt['error'] = str(error)
        atomic_json(path,state)
        if state.get('review_status'):return state
        if state.get('http_status') not in {None,200,500,502,503,504}:break
    if state.get('http_status') != 200:
        state['review_status'] = 'source_unavailable';atomic_json(path,state);return state
    raw = gzip.decompress((ROOT/state['raw_receipt']).read_bytes())
    if not raw.startswith(b'%PDF'):
        state['review_status'] = 'non_pdf_response'
        if any(v in raw[:200000].lower() for v in [b'anubis',b'captcha',b'challenge-platform']):
            STOP.set();atomic_json(FOLDER/'source_stop.json',{'url':url,'status':'interactive_access_challenge'})
        atomic_json(path,state);return state
    try:
        reader = PdfReader(io.BytesIO(raw))
        pages = [reader.pages[i].extract_text() or '' for i in range(min(3,len(reader.pages)))]
    except (PdfReadError,ValueError,IndexError,TypeError) as error:
        state.update(review_status='pdf_parse_unresolved',error=str(error));atomic_json(path,state);return state
    text = '\n\n'.join(pages)
    surrogate_count = sum(0xD800 <= ord(value) <= 0xDFFF for value in text)
    if surrogate_count:
        text = text.encode('utf-8',errors='replace').decode('utf-8')
        state['pdf_text_unpaired_surrogates_replaced'] = surrogate_count
    target = FOLDER/'text'/(token+'.txt');target.write_text(text)
    state.update(review_status='front_metadata_extracted_not_qualification',front_receipt=str(target.relative_to(ROOT)),
        inspected_pages=len(pages),fixed_papers=job['papers'],
        explicit_date_lines=[line for line in text.splitlines() if re.search(r'published|available online|publication date',line,re.I)],
        explicit_type_lines=[line.strip() for line in text.splitlines() if re.fullmatch(r'\s*(research article|original article|review|review article|editorial|letter|article|communication)\s*',line,re.I)])
    atomic_json(path,state);return state


def run() -> None:
    logging.getLogger('pypdf').setLevel(logging.ERROR)
    for part in [FOLDER,FOLDER/'raw',FOLDER/'receipts',FOLDER/'text']:part.mkdir(exist_ok=True)
    with (FOLDER/'collector.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (FOLDER/'host_holds.json').exists():
            HOST_HOLDS.update(json.loads((FOLDER/'host_holds.json').read_text())['hosts'])
        if (FOLDER/'source_stop.json').exists():
            prior = json.loads((FOLDER/'source_stop.json').read_text())
            if prior['status'] not in {401,403}:return
            HOST_HOLDS.add(urlparse(prior['url']).hostname)
            atomic_json(FOLDER/'host_holds.json',{'hosts':sorted(HOST_HOLDS),'basis':str((FOLDER/'source_stop.json').relative_to(ROOT))})
        jobs = {}
        for line in (FOLDER.parent/'identity_review.jsonl').read_text().splitlines():
            paper = json.loads(line)
            for record in paper['record_reviews']:
                if not record['identity_matched']:continue
                urls = record['repository_file_urls'] or [record['file_main']]
                for url in urls[:1]:
                    if url:
                        jobs.setdefault(url,{'url':url,'papers':[]})['papers'].append({'work_id':paper['work_id'],'doi':paper['doi'],'repository_record':record})
        atomic_json(FOLDER/'scope.json',{'jobs':list(jobs.values()),'qualification_unchanged':True})
        known = json.loads((FOLDER.parent/'known_pdf.json').read_text())
        if known.get('is_pdf') and known['url'] in jobs:
            path = FOLDER/'receipts'/(hashlib.sha256(known['url'].encode()).hexdigest()+'.json')
            if not path.exists():atomic_json(path,{**known,'attempts':[{'reused_existing_receipt':True}]})
        counts = Counter()
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(fetch,job) for job in jobs.values()]
            for future in as_completed(futures):
                counts[future.result()['review_status']] += 1
                atomic_json(FOLDER/'status.json',{'status':'collecting_front_metadata','fixed_urls':len(jobs),'processed':sum(counts.values()),'outcomes':dict(counts),'qualification_changes':0})
        atomic_json(FOLDER/'status.json',{'status':'source_stopped' if STOP.is_set() else 'fixed_fronts_processed_review_required',
            'fixed_urls':len(jobs),'processed':sum(counts.values()),'outcomes':dict(counts),'qualification_changes':0,'final_task_closure':False})


if __name__ == '__main__':
    run()
