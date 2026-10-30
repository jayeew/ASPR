"""Finite, cache-aware publisher fallback where bulk PMC identifiers are unavailable."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import sqlite3
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone

from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, publisher_url, request
from .three_journal_apply import read_csv


def retire_satisfied_jobs(db: sqlite3.Connection, source_route: str) -> int:
    """Retire only acquisition needs satisfied by current paper state elsewhere."""
    rows = db.execute("""SELECT doi,status,receipt FROM publisher_remaining_jobs j
        WHERE route=? AND status IN ('pending','cached_no_repeat')
        AND EXISTS(SELECT 1 FROM papers p WHERE p.doi=j.doi)
        AND NOT EXISTS(SELECT 1 FROM papers p WHERE p.doi=j.doi AND
            (p.status LIKE 'pending%' OR
             (p.status='included_verified_research' AND p.abstract_gap=1)))""",
        (source_route,)).fetchall()
    db.execute('''CREATE TABLE IF NOT EXISTS publisher_job_retirements(
        doi TEXT, route TEXT, previous_status TEXT, previous_receipt TEXT,
        paper_states TEXT, reason TEXT, retired_at TEXT)''')
    for doi, status, receipt in rows:
        states = [tuple(r) for r in db.execute(
            'SELECT work_id,status,abstract_gap FROM papers WHERE doi=?', (doi,))]
        db.execute('INSERT INTO publisher_job_retirements VALUES(?,?,?,?,?,?,?)',
            (doi, source_route, status, receipt, json.dumps(states),
             'Current qualification and abstract acquisition needs satisfied; other metadata tasks unchanged',
             datetime.now(timezone.utc).isoformat()))
        db.execute("UPDATE publisher_remaining_jobs SET status='no_current_qualification_or_abstract_need' WHERE doi=?", (doi,))
    db.commit()
    return len(rows)


def reopen_changed_needs(db: sqlite3.Connection, source_route: str) -> int:
    """New qualification/abstract gaps may reopen an unrequested retired job."""
    rows = db.execute("""SELECT doi FROM publisher_remaining_jobs j WHERE route=?
        AND status='no_current_qualification_or_abstract_need'
        AND EXISTS(SELECT 1 FROM papers p WHERE p.doi=j.doi AND
            (p.status LIKE 'pending%' OR
             (p.status='included_verified_research' AND p.abstract_gap=1)))""",
        (source_route,)).fetchall()
    db.executemany("UPDATE publisher_remaining_jobs SET status='pending' WHERE doi=?", rows)
    db.commit()
    return len(rows)


def run(source_route: str = 'springer', workers: int = 16,
        doi_prefixes: tuple[str, ...] | None = None,
        selected_dois: set[str] | None = None) -> None:
    if not 1 <= workers <= 16:
        raise ValueError('Worker count must be between 1 and 16')
    if source_route not in {'springer','frontiers','imr','nature','sciopen'}:raise ValueError('Source adapter not validated')
    folder=ROOT/'supplement'/(source_route+'_remaining');folder.mkdir(exist_ok=True)
    with (folder/'collector.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=folder/'status.json'
        if state_path.exists() and 'Source permission or rate restriction' in json.loads(state_path.read_text()).get('stop_reason',''):
            print('Source restriction retained; no automatic restart.');return
        db=connect(ROOT);db.execute('PRAGMA busy_timeout=900000')
        db.execute('CREATE TABLE IF NOT EXISTS publisher_remaining_jobs(doi TEXT PRIMARY KEY,route TEXT,status TEXT,receipt TEXT DEFAULT \'\')')
        marker=folder/'scope.json'
        if not marker.exists():
            db.execute("INSERT OR IGNORE INTO publisher_remaining_jobs(doi,route,status) SELECT DISTINCT doi,route,'pending' FROM papers WHERE route=? AND doi<>'' AND pmcids='[]' AND (status LIKE 'pending%' OR (status='included_verified_research' AND abstract_gap=1))",(source_route,))
            db.commit()
            atomic_json(marker,{'created_at':datetime.now(timezone.utc).isoformat(),'target_dois':db.execute('SELECT count(*) FROM publisher_remaining_jobs WHERE route=?',(source_route,)).fetchone()[0],
                'scope':'Existing candidate years/fields, unresolved qualification or qualified abstract gap, no PMC identifier; fixed scope, no repeated cached URLs',
                'bulk_api':'No configured Springer credential' if source_route=='springer' else 'No documented public batch endpoint found; existing PMC batch used for records with identifiers; private review API excluded',
                'source_policy':'https://www.springernature.com/jp/researchers/text-and-data-mining' if source_route=='springer' else 'https://www.frontiersin.org/about/open-access',
                'bulk_api_docs':'https://dev.springernature.com/docs/quick-start/api-access/' if source_route=='springer' else 'supplement/frontiers_source_review.json',
                'pacing':'Existing publisher limiter: at least 1.05 seconds between request starts; sixteen in flight maximum; stop permission/rate restrictions'})
        reopened = reopen_changed_needs(db, source_route)
        retired = retire_satisfied_jobs(db, source_route)
        atomic_json(folder / 'current_need_reconciliation.json', {
            'reopened_jobs': reopened, 'retired_jobs': retired,
            'at': datetime.now(timezone.utc).isoformat(),
            'network_requests': 0, 'other_metadata_tasks_unchanged': True})
        journals={r['source_id']:set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT/'fields/journal_registry.csv')}
        known={p.stem for p in (ROOT/'fields/qualification/supplement_parsed').glob('*.json')}
        pending=[]
        for doi, in db.execute("SELECT doi FROM publisher_remaining_jobs WHERE route=? AND (route<>'frontiers' OR doi LIKE '10.3389/%') AND status IN ('pending','cached_no_repeat') ORDER BY CASE WHEN doi LIKE '10.1007/%' THEN 0 WHEN doi LIKE '10.1186/%' THEN 1 WHEN doi LIKE '10.1140/%' THEN 2 ELSE 3 END,doi",(source_route,)).fetchall():
            if selected_dois is not None and doi not in selected_dois:
                continue
            if doi_prefixes is not None and not doi.startswith(doi_prefixes):
                continue
            url=publisher_url(doi,source_route);token=hashlib.sha256(url.encode()).hexdigest()
            if token in known:
                receipt='fields/qualification/supplement_parsed/'+token+'.json'
                cached=json.loads((ROOT/receipt).read_text());state=cached.get('status','cached_no_repeat')
                if state=='parsed':
                    payload=cached['parsed']
                    valid=doi in payload.get('doi',[]) and any(signature_match({'title':title,'journal_name':journal},payload,journals[sid]) for title,journal,sid in db.execute('SELECT title,journal,source_id FROM papers WHERE doi=?',(doi,)))
                    if not valid:state='publication_identity_mismatch'
                db.execute('UPDATE publisher_remaining_jobs SET status=?,receipt=? WHERE doi=?',(state,receipt,doi));db.commit()
            else:pending.append(doi)
        db.commit()
        journals={r['source_id']:set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT/'fields/journal_registry.csv')}
        stats=Counter();recent=deque(maxlen=100);stop='';iterator=iter(pending)
        def submit(pool: ThreadPoolExecutor, jobs: dict) -> None:
            doi=next(iterator,None)
            if doi:jobs[pool.submit(request,{'doi':doi,'route':source_route},ROOT,'publisher')]=doi
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs={}
            for _ in range(workers):submit(pool,jobs)
            while jobs:
                completed,_=wait(jobs,return_when=FIRST_COMPLETED)
                for future in completed:
                    doi=jobs.pop(future);record=future.result();state=record.get('status','missing_status')
                    receipt='fields/qualification/'+record.get('parsed_receipt','')
                    matched=False
                    if state=='parsed':
                        payload=record['parsed']
                        if doi not in payload.get('doi',[]):state='publication_identity_missing'
                        else:
                            for title,journal,sid in db.execute('SELECT title,journal,source_id FROM papers WHERE doi=?',(doi,)):
                                if signature_match({'title':title,'journal_name':journal},payload,journals[sid]):matched=True;break
                            if not matched:state='publication_identity_mismatch'
                    recent.append(matched);stats['new_requests_completed']+=not record.get('reused_receipt',False);stats['cached_receipts_reused']+=bool(record.get('reused_receipt',False));stats[state]+=1;stats['identity_matched']+=matched
                    stats['retained_bytes']+=record.get('retained_bytes',0);stats['http_attempts']+=0 if record.get('reused_receipt') else len(record.get('attempts',[]))
                    db.execute('UPDATE publisher_remaining_jobs SET status=?,receipt=? WHERE doi=?',(state,receipt,doi))
                    task_id=hashlib.sha256(('qualification:'+doi).encode()).hexdigest()[:32]
                    db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',(task_id,source_route+'_public_metadata_fallback',state,receipt,json.dumps({'identity_matched':matched,'not_yet_qualification_applied':True})))
                    db.execute('UPDATE tasks SET attempt_n=attempt_n+1,receipt=? WHERE task_id=?',(receipt,task_id))
                    db.commit()
                    if record.get('http_status') in {401,403,429} or state=='access_restricted':stop='Source permission or rate restriction; no further requests'
                    if len(recent)==100 and not any(recent):stop='No identities confirmed in 100 consecutive responses; parser/source review required'
                    atomic_json(folder/'status.json',{'status':'draining_inflight' if stop else 'running','stop_reason':stop,'fixed_pending_at_start':len(pending),'workers':workers,'request_start_interval_seconds':1.05,**stats})
                    if stats['new_requests_completed']%50==0:print(dict(stats),flush=True)
                    if not stop:submit(pool,jobs)
        final=('source_stopped_unattempted_retained' if stop else
               'fixed_selected_scope_processed_other_routes_pending' if selected_dois is not None else
               'fixed_selected_prefixes_processed_other_routes_pending' if doi_prefixes is not None else
               'fixed_acquisition_queue_processed_qualification_apply_required')
        atomic_json(folder/'status.json',{'status':final,'stop_reason':stop,'selected_doi_prefixes':doi_prefixes,'jobs':dict(db.execute("SELECT status,count(*) FROM publisher_remaining_jobs WHERE route=? GROUP BY status",(source_route,))),**stats});db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    run(workers=args.workers)
