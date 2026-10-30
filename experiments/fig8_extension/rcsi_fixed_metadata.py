"""Consume fixed cached RCSI identifiers, preferring productive journal OAI pages."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode

import requests
from lxml import etree

from .supplement_apply import signature_match
from .supplement_queue import ROOT
from .supplement_sources import atomic_json, jats_metadata
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/rcsi_source_review'
STOP = threading.Event()
PACE = threading.Lock()
LAST_START = 0.0
LOCAL = threading.local()
BULK_UNAVAILABLE = False


def fetch(url: str) -> tuple[dict, bytes | None]:
    global LAST_START
    path = FOLDER / 'requests' / (hashlib.sha256(url.encode()).hexdigest() + '.json')
    state = json.loads(path.read_text()) if path.exists() else {'url':url,'attempts':[]}
    if state.get('http_status') == 200:
        return state, gzip.decompress((ROOT / state['raw_receipt']).read_bytes())
    if state.get('http_status') in {401,403,429}:
        STOP.set()
    if state.get('http_status') in {400,401,403,404,410,429} or len(state['attempts']) >= 3:
        return state, None
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session(); LOCAL.session.trust_env = False
        LOCAL.session.headers['User-Agent'] = 'Fig8MetadataResearch/1.0 (public OAI metadata)'
    while len(state['attempts']) < 3 and not STOP.is_set():
        with PACE:
            time.sleep(max(0,1.05 - (time.monotonic()-LAST_START)))
            LAST_START = time.monotonic()
        if STOP.is_set():
            break
        attempt = {'started_at':time.time()}; state['attempts'].append(attempt)
        try:
            response = LOCAL.session.get(url,timeout=(10,35))
            attempt['http_status'] = response.status_code
            state['http_status'] = response.status_code
            raw = path.with_suffix('.xml.gz');raw.write_bytes(gzip.compress(response.content))
            state['raw_receipt'] = str(raw.relative_to(ROOT))
            if response.status_code in {401,403,429}:
                STOP.set();atomic_json(FOLDER/'source_stop.json',state)
        except requests.RequestException as error:
            attempt['error'] = str(error)
            state.pop('http_status',None)
        atomic_json(path,state)
        if state.get('http_status') == 200:
            return state,response.content
        if state.get('http_status') not in {None,500,502,503,504}:
            break
    return state,None


def consume(raw: bytes, evidence: dict, targets: dict, journals: dict) -> tuple[dict, str, int]:
    tree = etree.fromstring(raw,etree.XMLParser(resolve_entities=False,no_network=True))
    articles = tree.xpath('//*[local-name()="article"]');matched = {}
    for article in articles:
        data = etree.tostring(article).replace(b' xmlns="http://dtd.nlm.nih.gov/publishing/2.3"',b'')
        parsed = jats_metadata(data);parsed['doi'] = [d.lower() for d in parsed['doi']];parsed['issns'] = parsed['issn']
        for doi in set(parsed['doi']) & targets.keys():
            paper = targets[doi]
            valid = signature_match({'title':paper['title'],'journal_name':paper['journal']},parsed,journals[paper['source_id']])
            token = hashlib.sha256(doi.encode()).hexdigest();body = FOLDER/'records'/(token+'.xml.gz');body.write_bytes(gzip.compress(data))
            receipt = FOLDER/'records'/(token+'.json')
            atomic_json(receipt,{'source':'publisher_native_xml','route':'rcsi_oai_nlm','doi':doi,
                'parsed':parsed,'url':evidence['url'],'raw_receipt':'../../'+str(body.relative_to(ROOT)),
                'parent_oai_receipt':evidence['raw_receipt'],'identity_matched':valid,
                'transform':'Extract NLM article and remove default namespace; original OAI envelope retained'})
            if valid:
                matched[doi] = str(receipt.relative_to(ROOT))
    tokens = tree.xpath('//*[local-name()="resumptionToken"]/text()')
    return matched,tokens[0].strip() if tokens else '',len(articles)


def collect_set(code: str, papers: list[dict], journals: dict) -> dict:
    out = FOLDER/'sets'/(code+'.json')
    if out.exists() and json.loads(out.read_text()).get('status') == 'fixed_identifiers_reviewed':
        return json.loads(out.read_text())
    targets = {p['doi']:p for p in papers if p['doi']};matched = {};attempted = {};errors = []
    base = 'https://journals.rcsi.science/'+code+'/oai'
    url = base+'?'+urlencode({'verb':'ListRecords','metadataPrefix':'nlm','set':code,
                            'from':'2023-01-01T00:00:00Z','until':'2026-10-25T00:00:00Z'})
    tokens = set();pages = 0
    if BULK_UNAVAILABLE:
        errors.append({'stage':'bulk_skipped', 'receipt':str((FOLDER/'bulk_capability.json').relative_to(ROOT)),
                       'reason':'Existing exhausted ListRecords failures; use fixed exact identifiers'})
    while not STOP.is_set() and not BULK_UNAVAILABLE:
        receipt,raw = fetch(url);pages += 1
        if raw is None:
            errors.append({'stage':'bulk','receipt':receipt});break
        try:
            found,token,n = consume(raw,receipt,targets,journals)
        except (etree.XMLSyntaxError,ValueError,KeyError) as error:
            errors.append({'stage':'bulk_parse','error':str(error),'receipt':receipt});break
        new = set(found)-set(matched);matched.update(found)
        if len(matched)==len(targets) or not token:
            break
        # A sparse page is less efficient than the already-known exact article IDs.
        if len(new) < max(1,n//10) or token in tokens:
            break
        tokens.add(token);url = base+'?'+urlencode({'verb':'ListRecords','resumptionToken':token})
    for doi,paper in targets.items():
        if doi in matched or STOP.is_set():
            continue
        article_id = paper['url'].split('/article/download/')[1].split('/')[0]
        url = base+'?'+urlencode({'verb':'GetRecord','metadataPrefix':'nlm',
                                'identifier':'oai:journals.rcsi.science:article/'+article_id})
        receipt,raw = fetch(url);attempted[doi] = receipt
        if raw is None:
            continue
        try:
            found,_,_ = consume(raw,receipt,{doi:paper},journals);matched.update(found)
        except (etree.XMLSyntaxError,ValueError,KeyError) as error:
            errors.append({'stage':'record_parse','doi':doi,'error':str(error),'receipt':receipt})
    result = {'status':'source_stopped_unattempted_retained' if STOP.is_set() else 'fixed_identifiers_reviewed',
        'journal_set':code,'target_dois':len(targets),'bulk_pages':pages,'matched':matched,
        'unresolved_dois':sorted(set(targets)-set(matched)),'individual_receipts':attempted,
        'errors':errors,'final_task_closure':False}
    atomic_json(out,result);return result


def run() -> None:
    global BULK_UNAVAILABLE
    for name in ['requests','records','sets']:
        (FOLDER/name).mkdir(exist_ok=True)
    with (FOLDER/'collector.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (FOLDER/'source_stop.json').exists():
            return
        bulk_receipts = [(path, json.loads(path.read_text())) for path in (FOLDER/'requests').glob('*.json')]
        bulk_receipts = [(path, record) for path, record in bulk_receipts if 'verb=ListRecords' in record['url']]
        failures = [(path, record) for path, record in bulk_receipts
                    if record.get('http_status') == 500 and len(record.get('attempts', [])) >= 3]
        BULK_UNAVAILABLE = len(failures) >= 4 and not any(record.get('http_status') == 200 for _, record in bulk_receipts)
        if BULK_UNAVAILABLE:
            atomic_json(FOLDER/'bulk_capability.json', {'status':'bulk_unproductive_use_fixed_identifiers',
                'evidence':[str(path.relative_to(ROOT)) for path, _ in failures],
                'scope':'Current RCSI ListRecords request shape only; GetRecord remains available',
                'final_task_closure':False})
        groups = defaultdict(list)
        for paper in json.loads((FOLDER/'fixed_scope.json').read_text())['papers']:
            groups[paper['journal_set']].append(paper)
        journals = {r['source_id']:set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT/'fields/journal_registry.csv')}
        results = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(collect_set,code,rows,journals) for code,rows in sorted(groups.items(),key=lambda p:-len(p[1]))]
            for future in as_completed(futures):
                results.append(future.result())
                atomic_json(FOLDER/'status.json',{'status':'collecting' if not STOP.is_set() else 'draining_source_stop',
                    'sets_processed':len(results),'fixed_sets':len(groups),'matched_dois':sum(len(r['matched']) for r in results)})
        jobs = sorted({doi:receipt for r in results for doi,receipt in r['matched'].items()}.items())
        atomic_json(FOLDER/'jobs.json',jobs)
        atomic_json(FOLDER/'status.json',{'status':'source_stopped_cache_apply_required' if STOP.is_set() else 'fixed_source_queue_processed_apply_required',
            'sets_processed':len(results),'fixed_sets':len(groups),'matched_dois':len(jobs),
            'unresolved_dois':sum(len(r['unresolved_dois']) for r in results),'final_task_closure':False})


if __name__ == '__main__':
    run()
