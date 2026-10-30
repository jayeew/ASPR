"""Collect fixed PMID leads absent from the existing DOI metadata cache."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone

import requests

from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json

FOLDER = ROOT / 'supplement/snapshot_location_recovery/pubmed_cache_inventory'


def fetch(session: requests.Session, rows: list[dict]) -> dict:
    """Retain complete responses, identity conflicts and failed source attempts."""
    params = {'query': 'SRC:MED AND (' + ' OR '.join(
        f'EXT_ID:{pmid}' for pmid in sorted({r['pmid'] for r in rows})) + ')',
        'format': 'json', 'resultType': 'core', 'pageSize': 1000}
    token = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    path = FOLDER / 'raw' / (token + '.json.gz')
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    record = dict(url='https://www.ebi.ac.uk/europepmc/webservices/rest/search',
        params=params, dois=sorted({r['doi'] for r in rows}), leads=rows,
        retrieved_at=datetime.now(timezone.utc).isoformat(), attempts=[], proxy_used=False)
    for attempt in range(3):
        try:
            response = session.get(record['url'], params=params, timeout=(10, 45))
            record['attempts'].append({'http_status': response.status_code})
            record.update(http_status=response.status_code, status='http_error',
                          retry_after=response.headers.get('Retry-After', ''))
            if response.status_code == 200:
                record['response'] = response.json()
                items = record['response'].get('resultList', {}).get('result', [])
                record['status'] = ('complete' if len(items) == record['response'].get('hitCount')
                                    else 'truncated_not_accepted')
                break
            if response.status_code not in {500, 502, 503, 504} or record['retry_after']:
                break
        except (requests.RequestException, ValueError) as exc:
            record['attempts'].append({'error': type(exc).__name__})
            record['status'] = 'transport_or_parse_failure'
        if attempt < 2:
            time.sleep(2 ** attempt)
    record['receipt'] = str(path.relative_to(ROOT))
    atomic_bytes(path, gzip.compress(json.dumps(record).encode(), mtime=0))
    time.sleep(1.05)
    return record


def run() -> None:
    """Do not merge PMID-only identity leads into current papers automatically."""
    (FOLDER / 'raw').mkdir(exist_ok=True)
    with (FOLDER / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = FOLDER / 'collection_scope.json'
        if not path.exists():
            db = sqlite3.connect(FOLDER / 'inventory.sqlite')
            db.row_factory = sqlite3.Row
            db.execute('ATTACH DATABASE ? AS bulk', (str(ROOT / 'supplement/bulk_cache.sqlite'),))
            rows = [dict(r) for r in db.execute("SELECT work_id,url,doi,pmid FROM leads l "
                "WHERE status='pmid_cache_coverage_review_required' AND pmid<>'' AND NOT EXISTS "
                '(SELECT 1 FROM bulk.records r WHERE r.doi=l.doi) ORDER BY pmid,work_id')]
            db.close()
            atomic_json(path, dict(rows=rows, scope='Fixed PMID leads missing from DOI cache; '
                'not new candidates. Returned identity requires DOI/title/journal review.'))
        rows = json.loads(path.read_text())['rows']
        session = requests.Session()
        session.trust_env = False
        session.headers['User-Agent'] = 'Fig8MetadataResearch/1.0'
        summary = dict(target_leads=len(rows), completed_batches=0, matched_doi_pmid_leads=0,
                       retrieved_records=0, final_task_closure=False, qualification_changes=0)
        for offset in range(0, len(rows), 100):
            batch = rows[offset:offset + 100]
            record = fetch(session, batch)
            if record['status'] != 'complete':
                summary.update(status='source_stopped_unattempted_retained', receipt=record['receipt'])
                break
            items = record['response'].get('resultList', {}).get('result', [])
            identities = {(r.get('doi', '').lower(), r.get('id')) for r in items if r.get('source') == 'MED'}
            summary['matched_doi_pmid_leads'] += sum((r['doi'], r['pmid']) in identities for r in batch)
            summary['retrieved_records'] += len(items)
            summary['completed_batches'] += 1
            summary['status'] = 'running_identity_review_required'
            atomic_json(FOLDER / 'collection_status.json', summary)
            print(json.dumps(summary), flush=True)
        else:
            summary['status'] = 'fixed_pmid_batches_collected_identity_review_required'
        atomic_json(FOLDER / 'collection_status.json', summary)


if __name__ == '__main__':
    run()
