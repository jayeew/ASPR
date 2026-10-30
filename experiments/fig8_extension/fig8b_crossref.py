"""Bounded publisher-deposited abstract recovery for every journal with >=1000 known gaps."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone

import requests

from .fig8b_collect import OUT
from .stage0 import write_json


def main() -> None:
    with (OUT / 'journal_registry.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    gaps: dict[str, int] = defaultdict(int)
    for row in rows:
        gaps[row['issn_l']] += int(row['candidate_n']) - int(row['abstract_n'])
    scope = [{'issn': k, 'baseline_abstract_gaps': v} for k,v in sorted(gaps.items()) if k and v >= 1000]
    write_json(OUT / 'crossref_scope.json', {'rule': 'all ISSN-L journals with >=1000 stage0 missing abstracts across three fields', 'journals': scope})
    session = requests.Session()
    session.trust_env = False
    states = []
    for journal in scope:
        cursor, seen, count = '*', set(), 0
        status = 'complete'
        while cursor:
            if cursor in seen:
                status = 'repeated_cursor'
                break
            seen.add(cursor)
            params = {'filter':'has-abstract:true,from-pub-date:2023-01-01,until-pub-date:2025-12-31',
                'rows':1000, 'cursor':cursor, 'select':'DOI,title,abstract,publisher,ISSN,published-online'}
            url = f'https://api.crossref.org/journals/{journal["issn"]}/works'
            token = hashlib.sha256(json.dumps([url,params],sort_keys=True).encode()).hexdigest()
            path = OUT / 'crossref_raw' / f'{token}.json.gz'
            if path.exists():
                with gzip.open(path,'rt') as stream:
                    receipt = json.load(stream)
            else:
                receipt = {'url':url, 'params':params,'retrieved_at':datetime.now(timezone.utc).isoformat()}
                for attempt in range(3):
                    try:
                        response = session.get(url,params=params,timeout=(15,60))
                        receipt['http_status'] = response.status_code
                        if response.ok:
                            receipt.update(status='success',response=response.json())
                            break
                        receipt['status'] = 'http_error'
                    except (requests.RequestException,ValueError) as exc:
                        receipt.update(status='request_error',error_type=type(exc).__name__)
                    time.sleep(2 ** attempt)
                path.parent.mkdir(exist_ok=True)
                with gzip.open(path,'wt',compresslevel=1) as stream:
                    json.dump(receipt,stream,ensure_ascii=False)
                # Public API advertises one concurrent request and one request/second.
                time.sleep(1.05)
            if receipt['status'] != 'success':
                status = receipt['status']
                break
            message = receipt['response']['message']
            count += len(message['items'])
            cursor = message.get('next-cursor') if message['items'] and count < message['total-results'] else None
        states.append({**journal,'status':status,'records':count,'pages':len(seen)})
        write_json(OUT / 'crossref_status.json', states)
        print(states[-1],flush=True)


if __name__ == '__main__':
    main()
