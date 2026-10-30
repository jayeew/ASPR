"""Resumable full candidate census, with immutable credential-free API receipts."""
from __future__ import annotations

import argparse
import calendar
import gzip
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from gear.env import getenv

from .stage0 import FIELDS, SCREEN, write_json

OUT = Path(__file__).resolve().parents[2] / 'outputs/fig8_data/fields'
SELECT = ('id,doi,title,publication_date,publication_year,type,language,primary_topic,'
          'primary_location,abstract_inverted_index,authorships,updated_date,is_retracted,is_paratext')
LOCAL = threading.local()


def fetch(params: dict[str, Any], path: Path) -> dict[str, Any]:
    if path.exists():
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.trust_env = False
    key = (getenv('OPENALEX_API_KEY') or getenv('OPENALEX_API_KEYS').split(',')[0]).strip()
    status: int | str = 'network'
    for attempt in range(4):
        try:
            response = LOCAL.session.get('https://api.openalex.org/works',
                params={**params, **({'api_key': key} if key else {})}, timeout=(15, 90))
            status = response.status_code
            if response.ok:
                record = {'url': 'https://api.openalex.org/works', 'params': params,
                    'retrieved_at': datetime.now(timezone.utc).isoformat(), 'response': response.json()}
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix('.tmp')
                with gzip.open(temp, 'wt') as stream:
                    json.dump(record, stream, ensure_ascii=False)
                temp.replace(path)
                return record
            if status not in (429, 500, 502, 503, 504):
                break
        except (requests.RequestException, ValueError):
            status = 'network_or_json'
        time.sleep(2 ** attempt)
    raise RuntimeError(f'HTTP {status}; receipt {path.name}')


def collect(field: str, year: int, month: int) -> dict[str, Any]:
    shard = f'{field}_{year}_{month:02d}'
    directory = OUT / 'census_raw' / shard
    status_path = OUT / 'shards' / f'{shard}.json'
    query = (f'{SCREEN},primary_topic.field.id:{FIELDS[field]},'
        f'from_publication_date:{year}-{month:02d}-01,'
        f'to_publication_date:{year}-{month:02d}-{calendar.monthrange(year, month)[1]}')
    cursor, seen, counts, ids = '*', set(), set(), set()
    page, duplicates, total, cost = 0, 0, 0, 0.0
    state = {'shard': shard, 'field': field, 'year': year, 'month': month, 'filter': query}
    try:
        while cursor:
            if cursor in seen:
                raise RuntimeError('repeated_cursor')
            seen.add(cursor)
            params = {'filter': query, 'cursor': cursor, 'per_page': 200, 'select': SELECT}
            token = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
            record = fetch(params, directory / f'{page:05d}_{token}.json.gz')
            payload = record['response']
            counts.add(payload['meta']['count'])
            cost += payload['meta'].get('cost_usd', 0)
            for row in payload['results']:
                duplicates += row['id'] in ids
                ids.add(row['id'])
            total += len(payload['results'])
            page += 1
            cursor = payload['meta'].get('next_cursor')
            state.update(records=total, pages=page, reported_counts=sorted(counts), duplicate_ids=duplicates,
                         reported_query_cost_usd=cost, status='running')
            write_json(status_path, state)
        state['status'] = 'complete' if len(counts) == 1 and total == next(iter(counts)) and not duplicates else 'reconciliation_failed'
    except RuntimeError as exc:
        state.update(status='failed', error=str(exc))
    write_json(status_path, state)
    print(json.dumps(state), flush=True)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(collect, field, year, month) for field in FIELDS
                for year in (2023, 2024, 2025) for month in range(1, 13)]
        states = [job.result() for job in as_completed(jobs)]
    write_json(OUT / 'collection_status.json', states)


if __name__ == '__main__':
    main()
