"""Read the fixed public cloud version inventory with bounded concurrent GETs."""
from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone

import requests

from .pmc_cloud_inventory import BASE, FOLDER
from .pmc_cloud_metadata import assess_version
from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json

LOCAL = threading.local()
STOP = threading.Event()


def journal_priority(source_id: str, journal: str, nature_portfolio: bool = False) -> int:
    """Dispatch priority only; this does not assign a publisher brand."""
    comparison = {'https://openalex.org/S3880285': 0,
                  'https://openalex.org/S110447773': 1,
                  'https://openalex.org/S125754415': 2}
    if source_id in comparison:
        return comparison[source_id]
    return 3 if nature_portfolio or journal.casefold() == 'nature' or journal.casefold().startswith('nature ') else 4


def fetch_metadata(key: str, xml: bool = False) -> dict:
    pattern = r'(PMC\d+\.\d+)/\1\.xml' if xml else r'metadata/PMC\d+\.\d+\.json'
    if not re.fullmatch(pattern, key):
        raise ValueError('Object key outside fixed public dataset')
    folder = FOLDER / ('xml_raw' if xml else 'version_raw'); folder.mkdir(exist_ok=True)
    url = BASE + key; token = hashlib.sha256(url.encode()).hexdigest()
    path = folder / (token + '.json')
    if path.exists():
        return json.loads(path.read_text())
    old_token = hashlib.sha256((url + '{}').encode()).hexdigest()
    old = ROOT / 'supplement/missing_doi/records_raw' / (old_token + '.json')
    if old.exists():
        receipt = json.loads(old.read_text())
        atomic_json(path, receipt)
        return receipt
    if (FOLDER / 'source_stop.json').exists():
        STOP.set()
    if STOP.is_set():
        return {'status': 'not_requested_after_source_stop', 'url': url}
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session(); LOCAL.session.trust_env = False
        LOCAL.session.headers['User-Agent'] = 'Fig8MetadataResearch/1.0 (public PMC cloud metadata)'
    receipt = {'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat(), 'attempts': []}
    for attempt in range(3):
        try:
            response = LOCAL.session.get(url, timeout=(10, 35), allow_redirects=False)
            receipt['attempts'].append({'http_status': response.status_code})
            raw_path = path.with_suffix('.json.gz')
            atomic_bytes(raw_path, gzip.compress(response.content, mtime=0))
            receipt.update(http_status=response.status_code, raw_receipt=str(raw_path.relative_to(ROOT)),
                           retry_after=response.headers.get('Retry-After', ''), bytes=len(response.content),
                           etag=response.headers.get('ETag', ''))
            if response.status_code in {401, 403, 429}:
                STOP.set()
                atomic_json(FOLDER / 'source_stop.json', receipt)
                break
            if response.status_code not in {500, 502, 503, 504}:
                break
            if receipt['retry_after']:
                STOP.set(); break
        except requests.RequestException as error:
            receipt['attempts'].append({'error': type(error).__name__})
        if attempt < 2:
            time.sleep(2 ** attempt)
    atomic_json(path, receipt)
    if receipt.get('http_status') in {None, 500, 502, 503, 504}:
        STOP.set()
        atomic_json(FOLDER / 'source_stop.json', receipt)
    return receipt


def run(workers: int = 32, ready_versions: list[dict] | None = None) -> None:
    if not 1 <= workers <= 64:
        raise ValueError('Workers outside reviewed range 1..64')
    with (FOLDER / 'version_collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = FOLDER / 'version_status.json'
        if marker.exists() and json.loads(marker.read_text()).get('stop_reason'):
            return
        if ready_versions is None:
            atomic_json(marker, {'status': 'waiting_for_fixed_inventory', 'network_requests': 0})
            with (FOLDER / 'inventory.lock').open('a') as inventory:
                fcntl.flock(inventory, fcntl.LOCK_EX)
                state = json.loads((FOLDER / 'inventory_status.json').read_text())
                if state['status'] not in {'fixed_inventory_matched', 'fixed_inventory_partially_available'}:
                    atomic_json(marker, {'status': 'inventory_not_complete', 'inventory_state': state}); return
        papers = defaultdict(list)
        for paper in json.loads((FOLDER / 'scope.json').read_text())['papers']:
            for pmc in paper['pmcids']:
                papers[pmc].append(paper)
        # Titles are read from the current table; fixed Work/DOI/PMCID scope remains unchanged.
        from .supplement_queue import connect
        db = connect()
        db.execute('CREATE TEMP TABLE cloud_work_ids(work_id TEXT PRIMARY KEY)')
        db.executemany('INSERT OR IGNORE INTO cloud_work_ids VALUES(?)',
                       ((p['work_id'],) for group in papers.values() for p in group))
        with (ROOT / 'fields/journal_publisher_by_year.csv').open(encoding='utf-8-sig') as annual:
            nature_years = {(r['source_id'], int(r['year'])) for r in csv.DictReader(annual)
                            if r['nature_portfolio_year'] == 'True'}
        titles = {}; ranks = {}
        for wid, title, sid, journal, year in db.execute('SELECT p.work_id,p.title,p.source_id,p.journal,p.year FROM papers p JOIN cloud_work_ids c USING(work_id)'):
            titles[wid] = title; ranks[wid] = journal_priority(sid, journal, (sid, year) in nature_years)
        db.close()
        results = FOLDER / 'version_results'; results.mkdir(exist_ok=True)
        versions = ready_versions if ready_versions is not None else json.loads((FOLDER / 'matched_versions.json').read_text())
        versions = sorted({v['metadata_key']: v for v in versions}.values(),
                          key=lambda v: (min(ranks[p['work_id']] for p in papers[v['pmcid']]), v['metadata_key']))
        pending = iter(v for v in versions if not (results / v['metadata_key'].split('/')[-1]).exists())
        counts = Counter(); stop = ''; completed = 0; last_status = 0.0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            active = {}
            def submit() -> None:
                version = next(pending, None)
                if version and not STOP.is_set():
                    active[pool.submit(fetch_metadata, version['metadata_key'])] = version
            for _ in range(workers):
                submit()
            while active:
                ready, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in ready:
                    version = active.pop(future); receipt = future.result()
                    if receipt.get('status') == 'not_requested_after_source_stop':
                        continue
                    result = {'inventory': version, 'receipt': receipt, 'assessments': [],
                              'paper_links': [{'work_id': p['work_id'], 'doi': p['doi']}
                                              for p in papers[version['pmcid']]],
                              'final_task_closure': False}
                    if receipt.get('http_status') == 200:
                        try:
                            metadata = json.loads(gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes()))
                            if not isinstance(metadata, dict):
                                raise ValueError('Metadata must be an object')
                        except (ValueError, OSError):
                            result['status'] = 'invalid_metadata_retained'
                        else:
                            result['status'] = 'version_metadata_reviewed'
                            for paper in papers[version['pmcid']]:
                                assessment = assess_version(metadata, {**paper, 'pmcid': version['pmcid'], 'title': titles[paper['work_id']]})
                                result['assessments'].append({'work_id': paper['work_id'], 'doi': paper['doi'], **assessment})
                                counts[assessment['status']] += 1
                    else:
                        result['status'] = 'metadata_source_unavailable'
                    atomic_json(results / version['metadata_key'].split('/')[-1], result)
                    completed += 1
                    if receipt.get('http_status') in {401, 403, 429} or STOP.is_set():
                        stop = 'source_permission_or_rate_restriction'; STOP.set()
                    now = time.monotonic()
                    if stop or now - last_status >= 5.0:
                        # Receipts/results remain durable per item; this is only a progress view.
                        atomic_json(marker, {'status': 'draining' if stop else 'running', 'stop_reason': stop,
                                    'workers': workers, 'fixed_versions': len(versions), 'completed_this_run': completed,
                                    'assessments_this_run': dict(counts), 'qualification_applied': False})
                        last_status = now
                    if not stop:
                        submit()
        if STOP.is_set() and not stop:
            stop = 'shared_cloud_source_stop'
        atomic_json(marker, {'status': 'source_stopped' if stop else 'fixed_version_metadata_processed',
                    'stop_reason': stop, 'fixed_versions': len(versions), 'completed_this_run': completed,
                    'assessments_this_run': dict(counts), 'qualification_applied': False,
                    'final_task_closure': False, 'workers': workers})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=32)
    run(parser.parse_args().workers)
