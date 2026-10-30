"""Capture all publicly returned APS metadata, with monthly server-linked pagination."""
from __future__ import annotations

import calendar
import gzip
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

OUT=Path('outputs/fig8_data/fields/qualification')


def month(job: tuple[int,int]) -> dict:
    year,month=job;folder=OUT/'aps_all_raw';folder.mkdir(exist_ok=True)
    session=requests.Session();session.trust_env=False
    url=f'https://harvest.aps.org/v2/journals/articles?from={year}-{month:02d}-01&until={year}-{month:02d}-{calendar.monthrange(year,month)[1]}&date=published&per_page=100'
    seen=set();count=0;status='complete'
    while url:
        if url in seen:status='repeated_next_link';break
        seen.add(url);path=folder/(hashlib.sha256(url.encode()).hexdigest()+'.json.gz')
        if path.exists():
            with gzip.open(path,'rt') as stream:receipt=json.load(stream)
        else:
            receipt={'url':url,'retrieved_at':datetime.now(timezone.utc).isoformat()}
            for attempt in range(3):
                try:
                    response=session.get(url,timeout=(15,45))
                    receipt.update(http_status=response.status_code,response=response.json(),next_url=response.links.get('next',{}).get('url'))
                    if response.status_code==200 or response.status_code in [401,403,404]:break
                except (requests.RequestException,ValueError) as error:
                    receipt.update(error=str(error),http_status=0)
                time.sleep(2**attempt)
            with gzip.open(path,'wt') as stream:json.dump(receipt,stream)
            time.sleep(.3)
        if receipt['http_status']!=200:status='unresolved';break
        count+=len(receipt['response']['data']);url=receipt.get('next_url')
    result={'year':year,'month':month,'status':status,'pages':len(seen),'records':count}
    print(json.dumps(result),flush=True);return result


def main() -> None:
    with ThreadPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(month,[(y,m) for y in [2023,2024,2025] for m in range(1,13)]))
    (OUT/'aps_all_status.json').write_text(json.dumps(result,indent=2))
    print('APS all complete',sum(r['records'] for r in result),flush=True)


if __name__=='__main__':main()
