"""Capture Nature Portfolio annual catalogues for the current candidate table."""
from __future__ import annotations

import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .nature_backfill_fetch import DirectPublisher
from .nature_types import parse_catalogue

BASE = Path('outputs/fig8_data/fields')
OUT = BASE / 'qualification'


def fetch(job: dict, publisher: DirectPublisher) -> dict:
    url = f"{job['journal_url'].rstrip('/')}/articles?year={job['year']}&page={job['page']}"
    token = hashlib.sha256(url.encode()).hexdigest()
    dest = OUT / 'catalogue_parsed' / f'{token}.json'
    if dest.exists():
        return json.loads(dest.read_text())
    result = {**job, 'url': url, 'raw_receipt': f'publisher_raw/{token}.html.gz'}
    try:
        result.update(parse_catalogue(publisher.get(url), url, job['year']), status='parsed')
    except (ValueError, RuntimeError) as error:
        result.update(status='unresolved', error=str(error))
    dest.write_text(json.dumps(result, ensure_ascii=False))
    return result


def main() -> None:
    (OUT / 'catalogue_parsed').mkdir(parents=True, exist_ok=True)
    with (BASE / 'journal_registry.csv').open(encoding='utf-8-sig') as stream:
        rows = {r['source_id']:r for r in csv.DictReader(stream)}
    with (BASE / 'journal_publisher_by_year.csv').open(encoding='utf-8-sig') as stream:
        annual = {r['source_id'] for r in csv.DictReader(stream) if r['year'] in ['2023','2024','2025'] and r['nature_portfolio_year']=='True'}
    scope = [r for r in rows.values() if r['nature_portfolio_official_current']=='True' or r['source_id'] in annual]
    (OUT / 'catalogue_scope.json').write_text(json.dumps(scope,ensure_ascii=False,indent=2))
    jobs = [{'source_id':r['source_id'],'journal_url':r['official_journal_url'],'year':y,'page':1}
            for r in scope if r['official_journal_url'].startswith('https://www.nature.com/') for y in [2023,2024,2025]]
    publisher = DirectPublisher(OUT)
    first = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch,j,publisher) for j in jobs]
        for i,f in enumerate(as_completed(futures),1):
            first.append(f.result())
            if i%30==0: print('first',i,len(jobs),flush=True)
    rest = [{k:r[k] for k in ['source_id','journal_url','year']}|{'page':p}
            for r in first if r['status']=='parsed' for p in range(2,r['last_page']+1)]
    print('remaining pages',len(rest),flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch,j,publisher) for j in rest]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            if i%100==0: print('pages',i,len(rest),flush=True)
    print('Catalogue fetch complete',flush=True)


if __name__=='__main__':
    main()
