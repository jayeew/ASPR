"""Query every missing-abstract DOI in Europe PMC, retaining successes and no-matches."""
from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import requests

from .fig8b_collect import OUT
from .stage0 import write_json


def fetch(dois: list[str]) -> dict[str, Any]:
    params = {'query': ' OR '.join(f'DOI:"{d}"' for d in dois),
              'format': 'json', 'resultType': 'core', 'pageSize': 1000}
    token = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    path = OUT / 'epmc_raw' / f'{token}.json.gz'
    if path.exists():
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)
    receipt: dict[str, Any] = {'url': 'https://www.ebi.ac.uk/europepmc/webservices/rest/search',
        'params': params, 'dois': dois, 'retrieved_at': datetime.now(timezone.utc).isoformat()}
    session = requests.Session()
    session.trust_env = False
    for attempt in range(3):
        try:
            response = session.get(receipt['url'], params=params, timeout=(10, 45))
            receipt['http_status'] = response.status_code
            if response.ok:
                receipt['response'] = response.json()
                returned = len(receipt['response'].get('resultList', {}).get('result', []))
                receipt['status'] = 'complete' if returned == receipt['response'].get('hitCount') else 'truncated_not_accepted'
                break
            receipt['status'] = 'http_error'
        except (requests.RequestException, ValueError) as exc:
            receipt.update(status='request_error', error_type=type(exc).__name__)
        time.sleep(2 ** attempt)
    path.parent.mkdir(exist_ok=True)
    with gzip.open(path.with_suffix('.tmp'), 'wt', compresslevel=1) as stream:
        json.dump(receipt, stream, ensure_ascii=False)
    path.with_suffix('.tmp').replace(path)
    receipt['receipt_path'] = str(path.relative_to(OUT))
    return receipt


def main() -> None:
    connection = sqlite3.connect(OUT / 'epmc_queue.sqlite')
    connection.execute('CREATE TABLE IF NOT EXISTS queue (doi TEXT PRIMARY KEY, status TEXT, receipt TEXT)')
    connection.execute('CREATE TABLE IF NOT EXISTS seen_v2 (path TEXT PRIMARY KEY)')
    while True:
        seen = {r[0] for r in connection.execute('SELECT path FROM seen_v2')}
        files = sorted(p for p in (OUT / 'census_raw').glob('*/*.json.gz') if str(p) not in seen)
        for path in files:
            with gzip.open(path, 'rt') as stream:
                record = json.load(stream)
            dois = [(r['doi'].removeprefix('https://doi.org/').lower(), 'pending', None)
                    for r in record['response']['results'] if (not r.get('abstract_inverted_index') or not r.get('authorships')
                    or any(not a.get('institutions') or not a.get('countries') for a in r.get('authorships', [])))
                    and r.get('doi') and '"' not in r['doi']]
            connection.executemany('INSERT OR IGNORE INTO queue VALUES (?,?,?)', dois)
            connection.execute('INSERT INTO seen_v2 VALUES (?)', (str(path),))
        connection.commit()
        pending = [r[0] for r in connection.execute("SELECT doi FROM queue WHERE status='pending' ORDER BY doi")]
        with ThreadPoolExecutor(max_workers=10) as pool:
            for result in pool.map(fetch, [pending[i:i + 60] for i in range(0, len(pending), 60)]):
                token = hashlib.sha256(json.dumps(result['params'], sort_keys=True).encode()).hexdigest()
                connection.executemany('UPDATE queue SET status=?,receipt=? WHERE doi=?',
                    [(result['status'], f'epmc_raw/{token}.json.gz', d) for d in result['dois']])
                connection.commit()
        status = dict(connection.execute('SELECT status,count(*) FROM queue GROUP BY status'))
        write_json(OUT / 'epmc_status.json', {'doi_status': status, 'census_pages_scanned': len(seen) + len(files)})
        print(status, flush=True)
        states = [json.loads(p.read_text()) for p in (OUT / 'shards').glob('*.json')]
        if len(states) == 108 and all(s['status'] in ['complete', 'failed', 'reconciliation_failed'] for s in states):
            remaining = [p for p in (OUT / 'census_raw').glob('*/*.json.gz') if str(p) not in seen and p not in files]
            if not remaining:
                break
        time.sleep(5)
    connection.close()


if __name__ == '__main__':
    main()
