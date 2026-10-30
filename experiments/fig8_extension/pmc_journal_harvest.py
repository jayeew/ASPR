"""Resume serial journal front-matter pages during official PMC bulk hours."""
from __future__ import annotations

import csv
import fcntl
import gzip
import hashlib
import json
import os
import signal
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from lxml import etree

from .supplement_oai import POLICY, URL, parse_oai, seconds_until_bulk_window
from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json

STOP_AFTER_PAGE = False


def request_checkpoint_stop(signum: int, frame: object) -> None:
    """Drain the current page before a requested collector restart."""
    global STOP_AFTER_PAGE
    STOP_AFTER_PAGE = True


def retain_front(raw_path: Path, raw: bytes) -> bool:
    """Recover an identical unparsed front; never overwrite conflicting evidence."""
    if raw_path.exists():
        return gzip.decompress(raw_path.read_bytes()) == raw
    atomic_bytes(raw_path, gzip.compress(raw, mtime=0))
    return True


def wanted_records(data: bytes, targets: set[str]) -> tuple[list[tuple[str, bytes]], str, list[str]]:
    root = etree.fromstring(data, etree.XMLParser(resolve_entities=False, no_network=True))
    errors = root.xpath('//*[local-name()="error"]/text()')
    records = []
    for node in root.xpath('//*[local-name()="ListRecords"]/*[local-name()="record"]'):
        identifiers = node.xpath('./*[local-name()="header"]/*[local-name()="identifier"]/text()')
        pmc = 'PMC'+identifiers[0].rsplit(':',1)[-1] if len(identifiers)==1 else ''
        if pmc in targets:
            records.append((pmc,etree.tostring(node)))
    tokens = root.xpath('//*[local-name()="resumptionToken"]/text()')
    return records, tokens[0].strip() if tokens else '', errors


def wait_window(state: dict, path: Path) -> None:
    while delay := seconds_until_bulk_window(datetime.now(timezone.utc)):
        state.update(status='waiting_off_peak',next_window_utc=(datetime.now(timezone.utc)+timedelta(seconds=delay)).isoformat())
        atomic_json(path,state)
        time.sleep(min(delay,60))


def fetch(session: requests.Session, params: dict, folder: Path) -> dict:
    token=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()
    path=folder/(token+'.json')
    prior_attempts = 0
    prior_receipt = ''
    if path.exists():
        prior = json.loads(path.read_text())
        policy_path = folder/'deferred_rate_retry.json'
        policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
        if (prior.get('http_status') != 429 or policy.get('request_token') != token
                or datetime.now(timezone.utc) < datetime.fromisoformat(policy['not_before_utc'])):
            return prior
        prior_attempts = len(prior.get('attempts', []))
        if prior_attempts >= 3:
            return prior
        prior_receipt = str(path.relative_to(ROOT))
        path = folder/(token+'.deferred_retry.json')
        if path.exists():
            return json.loads(path.read_text())
    r={'url':URL,'params':params,'retrieved_at':datetime.now(timezone.utc).isoformat(),'attempts':[]}
    if prior_receipt:
        r.update(previous_attempt_receipt=prior_receipt, previous_attempt_count=prior_attempts)
    for n in range(3-prior_attempts):
        started=time.monotonic()
        try:
            response=session.get(URL,params=params,timeout=(10,45),allow_redirects=False)
            r['attempts'].append({'http_status':response.status_code,'seconds':time.monotonic()-started})
            raw=path.with_suffix('.xml.gz');atomic_bytes(raw,gzip.compress(response.content,mtime=0))
            r.update(http_status=response.status_code,raw_receipt=str(raw.relative_to(ROOT)),raw_sha256=hashlib.sha256(response.content).hexdigest(),retry_after=response.headers.get('Retry-After',''))
            if response.status_code not in {500,502,503,504}:
                break
        except requests.RequestException as error:
            r['attempts'].append({'error':type(error).__name__,'seconds':time.monotonic()-started})
        if n<2-prior_attempts:time.sleep(2**n)
    atomic_json(path,r)
    time.sleep(1.05)
    return r


def run() -> None:
    folder=ROOT/'supplement/pmc_journal_batches';folder.mkdir(exist_ok=True)
    state_path=folder/'status.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'completed_sets':[], 'pages':0,'new_front_records':0,'until':datetime.now(timezone.utc).date().isoformat()}
    state.update(pid=os.getpid(),policy_url=POLICY,status='starting',request_interval_seconds=1.05)
    with (ROOT/'supplement/individual_acquisition_plan.csv').open(encoding='utf-8-sig') as f:
        targets={r['pmcid']:r['doi'] for r in csv.DictReader(f) if r['action']=='fetch_missing_or_conflicting'}
    target_ids=set(targets)
    with (ROOT/'supplement/pmc_journal_batch_plan.csv').open(encoding='utf-8-sig') as f:
        plans=[r for r in csv.DictReader(f) if r['status']=='candidate_set_requires_api_validation' and int(r['remaining_pmc_links'])>=20]
    priority = {'https://openalex.org/S3880285': 0, 'https://openalex.org/S110447773': 1,
                'https://openalex.org/S125754415': 2}
    plans.sort(key=lambda plan: priority.get(plan['source_id'], 3))
    session=requests.Session();session.trust_env=False
    session.headers.update({'User-Agent':'Fig8MetadataResearch/1.0 (serial journal front metadata)','Accept-Encoding':'gzip, deflate'})
    q=ROOT/'fields/qualification'
    deferred = state.setdefault('deferred_sets', {})
    plans.sort(key=lambda plan: (plan['proposed_set'] in deferred, priority.get(plan['source_id'], 3)))
    for plan in plans:
        domain=plan['proposed_set']
        if domain in state['completed_sets']:
            continue
        checkpoint=folder/(hashlib.sha256(domain.encode()).hexdigest()+'.progress.json')
        progress=json.loads(checkpoint.read_text()) if checkpoint.exists() else {'domain':domain,'pages':0,'matched':0,'resumption_token':''}
        if progress.get('terminal') or domain in deferred:
            continue
        while True:
            wait_window(state,state_path)
            params={'verb':'ListRecords','resumptionToken':progress['resumption_token']} if progress['resumption_token'] else {'verb':'ListRecords','metadataPrefix':'pmc_fm','set':domain,'from':progress.get('from_date', '2023-01-01'),'until':state['until']}
            state.update(status='running',journal=plan['journal'],domain=domain,updated_at=datetime.now(timezone.utc).isoformat());atomic_json(state_path,state)
            receipt=fetch(session,params,folder)
            if receipt.get('http_status')!=200:
                state.update(status='source_stopped',receipt=receipt,reason='No repeated permission/rate/failed-page requests; other goal queues continue')
                atomic_json(state_path,state)
                return
            data=gzip.decompress((ROOT/receipt['raw_receipt']).read_bytes())
            records, token, errors=wanted_records(data,target_ids)
            before_new = state['new_front_records']
            for pmc,raw in records:
                destination=q/'oai_parsed'/(pmc+'.json')
                if destination.exists():
                    continue
                record={'doi':targets[pmc],'pmcid':pmc,'source':'pmc_front_oai','url':URL,'source_url':URL,
                        'params':params,'batch_receipt':receipt['raw_receipt'],'retrieved_at':receipt['retrieved_at'],'attempts':[200]}
                raw_path=q/'oai_raw'/(pmc+'.xml.gz')
                if not retain_front(raw_path, raw):
                    record.update(status='metadata_unavailable', reason='Existing raw front differs; source conflict retained',
                                  raw_receipt=str(raw_path.relative_to(q)))
                    atomic_json(destination,record)
                    state['new_front_records'] += 1
                    continue
                record.update(raw_receipt=str(raw_path.relative_to(q)),raw_sha256=hashlib.sha256(raw).hexdigest())
                try:
                    parsed=parse_oai(raw)
                    record.update(status='parsed',parsed=parsed,requested_doi_matches=targets[pmc] in parsed['doi'])
                except (ValueError,etree.XMLSyntaxError) as error:
                    record.update(status='metadata_unavailable',reason=str(error)[:300])
                atomic_json(destination,record)
                state['new_front_records']+=1;progress['matched']+=1
            added = state['new_front_records'] - before_new
            progress['consecutive_pages_without_new_target'] = (0 if added else progress.get('consecutive_pages_without_new_target', 0) + 1)
            progress.update(pages=progress['pages']+1,resumption_token=token,errors=errors)
            state['pages']+=1
            if not token:
                progress.update(terminal=True,terminal_reason='oai_error' if errors else 'journal_page_sequence_complete')
                state['completed_sets'].append(domain)
            atomic_json(checkpoint,progress);atomic_json(state_path,state)
            if STOP_AFTER_PAGE:
                state.update(status='checkpointed_for_resume')
                atomic_json(state_path,state)
                return
            if not token:break
            if progress['consecutive_pages_without_new_target'] >= 100:
                deferred[domain] = {'reason': '100_consecutive_pages_without_new_target',
                                    'checkpoint': str(checkpoint.relative_to(ROOT)),
                                    'pages': progress['pages'], 'matched': progress['matched'],
                                    'final_task_closure': False, 'resumption_token_retained': True}
                atomic_json(state_path,state)
                break
    state.update(status='batch_pass_finished_deferred_sets_retained' if deferred else 'batch_paths_complete_remaining_requires_other_paths',updated_at=datetime.now(timezone.utc).isoformat())
    atomic_json(state_path,state)


if __name__=='__main__':
    signal.signal(signal.SIGTERM, request_checkpoint_stop)
    with (ROOT/'supplement/pmc_api.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
