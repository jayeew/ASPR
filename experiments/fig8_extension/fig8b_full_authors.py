"""Retrieve individual-work authorships for every list response reaching 100 authors."""
from __future__ import annotations

import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import pyarrow.parquet as pq
import requests

from gear.env import getenv

from .fig8b_collect import OUT
from .stage0 import write_csv, write_json


def fetch(work_id: str) -> dict[str, Any]:
    path = OUT / 'full_authors_raw' / (work_id.rsplit('/', 1)[-1] + '.json.gz')
    if path.exists():
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)
    key = (getenv('OPENALEX_API_KEY') or getenv('OPENALEX_API_KEYS').split(',')[0]).strip()
    session = requests.Session()
    session.trust_env = False
    receipt: dict[str, Any] = {'url': 'https://api.openalex.org/works/' + work_id.rsplit('/', 1)[-1],
        'work_id': work_id, 'retrieved_at': datetime.now(timezone.utc).isoformat()}
    for attempt in range(3):
        try:
            response = session.get(receipt['url'], params={'api_key':key} if key else {}, timeout=(15,90))
            receipt['http_status'] = response.status_code
            if response.ok:
                receipt.update(status='retrieved',response=response.json())
                break
            receipt['status'] = 'http_error'
            if response.status_code not in [429,500,502,503,504]:
                break
        except (requests.RequestException,ValueError) as exc:
            receipt.update(status='request_error',error_type=type(exc).__name__)
        time.sleep(2 ** attempt)
    path.parent.mkdir(exist_ok=True)
    with gzip.open(path.with_suffix('.tmp'),'wt',compresslevel=1) as stream:
        json.dump(receipt,stream,ensure_ascii=False)
    path.with_suffix('.tmp').replace(path)
    return receipt


def main() -> None:
    targets = []
    for path in sorted((OUT / 'papers').glob('*.parquet')):
        columns = ['work_id','doi','title','field','before_author_records','before_any_institution','before_any_country']
        if 'listed_author_records' in pq.read_schema(path).names:
            columns.append('listed_author_records')
        rows = pq.read_table(path,columns=columns).to_pylist()
        targets.extend({**r,'before_author_records':r.get('listed_author_records',r['before_author_records']),'paper_shard':path.name}
                       for r in rows if r.get('listed_author_records',r['before_author_records']) >= 100)
    write_csv(OUT / 'full_author_targets.csv', targets)
    states = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        jobs = {pool.submit(fetch,row['work_id']):row for row in targets}
        for job in as_completed(jobs):
            receipt = job.result()
            row = jobs[job]
            authors = (receipt.get('response') or {}).get('authorships') or []
            states.append({'work_id':row['work_id'],'status':receipt['status'],
                'listed_authors':row['before_author_records'],'individual_endpoint_authors':len(authors),
                'any_institution_full':any(a.get('institutions') for a in authors),
                'any_country_full':any(a.get('countries') for a in authors)})
            if len(states) % 100 == 0:
                write_json(OUT / 'full_author_fetch_status.json', states)
                print('individual works',len(states),'/',len(targets),flush=True)
    write_json(OUT / 'full_author_fetch_status.json', states)


if __name__ == '__main__':
    main()
