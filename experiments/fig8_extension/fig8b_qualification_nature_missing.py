"""Read publisher pages for candidate Nature DOIs absent from annual catalogues."""
from __future__ import annotations

import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from lxml import html

from .fig8b_publisher import parse
from .fig8b_qualification_evidence import BASE, OUT
from .nature_backfill_fetch import DirectPublisher


def main() -> None:
    folder=OUT/'nature_single';(folder/'parsed').mkdir(parents=True,exist_ok=True)
    publisher=DirectPublisher(folder)
    with (BASE/'journal_registry.csv').open(encoding='utf-8-sig') as stream:
        sources={r['source_id']:r for r in csv.DictReader(stream)}
    targets=json.loads((OUT/'nature_missing_targets.json').read_text())
    def fetch(item: dict) -> dict:
        url='https://www.nature.com/articles/'+item['doi'].split('/',1)[1]
        token=hashlib.sha256(url.encode()).hexdigest();path=folder/'parsed'/(token+'.json')
        if path.exists():return json.loads(path.read_text())
        r={**item,'url':url,'retrieved_at':datetime.now(timezone.utc).isoformat()}
        try:
            text=publisher.get(url);r.update(parse(text,item['doi'],json.loads(sources[item['source_id']]['issn'])))
            titles=html.fromstring(text).xpath('//meta[@name="citation_title"]/@content')
            r['publisher_title']=titles[0] if len(titles)==1 else ''
            r['raw_receipt']='nature_single/publisher_raw/'+token+'.html.gz'
        except (ValueError,RuntimeError) as error:r.update(status='unresolved',error=str(error))
        path.write_text(json.dumps(r,ensure_ascii=False,indent=2));return r
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(fetch,targets))
    print([(r['doi'],r['status'],r.get('online_date')) for r in results])


if __name__=='__main__':main()
