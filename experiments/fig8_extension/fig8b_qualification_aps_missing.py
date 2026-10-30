"""Resolve candidate APS DOIs absent from the windowed feed, including year boundaries."""
from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

OUT=Path('outputs/fig8_data/fields/qualification')


def fetch(doi: str) -> dict:
    url='https://harvest.aps.org/v2/journals/articles/'+doi
    path=OUT/'aps_single_raw'/(hashlib.sha256(doi.encode()).hexdigest()+'.json.gz')
    if path.exists():
        with gzip.open(path,'rt') as stream:return json.load(stream)
    session=requests.Session();session.trust_env=False
    receipt={'doi':doi,'url':url,'retrieved_at':datetime.now(timezone.utc).isoformat()}
    for attempt in range(2):
        try:
            response=session.get(url,timeout=(10,30));receipt.update(http_status=response.status_code,response=response.json())
            if response.status_code not in [429,500,502,503,504]:break
        except (requests.RequestException,ValueError) as error:receipt.update(http_status=0,error=str(error))
        time.sleep(2**attempt)
    with gzip.open(path,'wt') as stream:json.dump(receipt,stream)
    return receipt


def main() -> None:
    (OUT/'aps_single_raw').mkdir(exist_ok=True)
    target_path=OUT/'aps_missing_targets.json'
    if not target_path.exists():
        seen=set()
        for path in (OUT/'aps_all_raw').glob('*.gz'):
            with gzip.open(path,'rt') as stream:r=json.load(stream)
            seen.update(x.get('identifiers',{}).get('doi','').lower() for x in r.get('response',{}).get('data',[]))
        db=sqlite3.connect(OUT/'evidence.sqlite')
        targets=sorted({r[0] for r in db.execute("SELECT DISTINCT doi FROM candidates WHERE doi LIKE '10.1103/%'")}-seen)
        db.close();target_path.write_text(json.dumps(targets))
    targets=json.loads(target_path.read_text())
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(fetch,d) for d in targets]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            if i%200==0:print('APS single',i,len(targets),flush=True)
    print('APS single complete',len(targets),flush=True)


if __name__=='__main__':main()
