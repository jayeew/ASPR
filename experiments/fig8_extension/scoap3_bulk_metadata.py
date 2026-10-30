"""Harvest public publisher-deposited metadata using SCOAP3's visible search pagination."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import time
from datetime import datetime, timezone

import requests
from lxml import html

from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json


def run() -> None:
    folder=ROOT/'supplement/scoap3';folder.mkdir(exist_ok=True)
    session=requests.Session();session.trust_env=False
    session.headers['User-Agent']='Fig8MetadataResearch/1.0 (public publisher metadata; serial requests)'
    page=1;total=None;seen=set();counts=[]
    params={'publication_year__gte':'2022-01-01','publication_year__lte':'2026-10-23','page_size':100,'ordering':'-publication_date'}
    atomic_json(folder/'bulk_scope.json',{'params':params,'scope':'Current target identities plus the requested 2022/2026 boundary-check leads; no automatic candidate additions',
        'authority':'Publisher-deposited SCOAP3 metadata; generic material=article and publication_date do not establish original research or online year',
        'source':'https://scoap3.org/scoap3-repository/','pagination_source':'search_client.js.gz; parameters exposed by public search UI','max_inflight':1})
    while total is None or page<=math.ceil(total/100):
        query={**params,'page':page};token=hashlib.sha256(json.dumps(query,sort_keys=True).encode()).hexdigest();receipt=folder/(token+'.json');raw_path=folder/(token+'.html.gz')
        if receipt.exists():record=json.loads(receipt.read_text())
        else:
            record={'url':'https://repo.scoap3.org/search','params':query,'retrieved_at':datetime.now(timezone.utc).isoformat(),'attempts':[]}
            for attempt in range(3):
                try:
                    response=session.get(record['url'],params=query,timeout=(10,45),allow_redirects=False)
                    record.update(http_status=response.status_code,raw_receipt=str(raw_path.relative_to(ROOT)),sha256=hashlib.sha256(response.content).hexdigest())
                    record['attempts'].append({'http_status':response.status_code});atomic_bytes(raw_path,gzip.compress(response.content,mtime=0))
                    if response.status_code not in {500,502,503,504}:break
                except requests.RequestException as e:record['attempts'].append({'error':type(e).__name__})
                if attempt<2:time.sleep(2**attempt)
            atomic_json(receipt,record);time.sleep(1.05)
        if record.get('http_status')!=200:
            atomic_json(folder/'bulk_status.json',{'status':'source_stopped','page':page,'receipt':str(receipt.relative_to(ROOT))});return
        raw=gzip.decompress(raw_path.read_bytes());tree=html.fromstring(raw, parser=html.HTMLParser(huge_tree=True))
        scripts=tree.xpath('//script[@id="__NEXT_DATA__"]/text()')
        if not scripts:
            atomic_json(folder/'bulk_status.json',{'status':'source_payload_unavailable','page':page,'receipt':str(receipt.relative_to(ROOT)),'pages_completed':page-1,'unique_records':len(seen)})
            return
        payload=json.loads(scripts[0])['props']['pageProps']
        if total is None:total=payload['count']
        if payload['query'].get('publication_year__gte')!=params['publication_year__gte']:raise ValueError('Search scope not echoed')
        outside=[r['id'] for r in payload['results'] if not params['publication_year__gte']<=r['publication_date'][:10]<=params['publication_year__lte']]
        if outside:
            atomic_json(folder/'bulk_status.json',{'status':'date_filter_mismatch','page':page,'outside_ids':outside});return
        counts.append(payload['count']);seen.update(r['id'] for r in payload['results'])
        atomic_json(folder/'bulk_status.json',{'status':'running','pages_completed':page,'planned_pages':math.ceil(total/100),'first_count':total,'unique_records':len(seen),'last_receipt':str(receipt.relative_to(ROOT))})
        if page%10==0:print('SCOAP3 metadata pages',page,'records',len(seen),'expected',total,flush=True)
        page+=1
    complete=len(set(counts))==1 and len(seen)==total
    atomic_json(folder/'bulk_status.json',{'status':'collected_identity_and_field_review_required' if complete else 'pagination_reconciliation_required','pages_completed':page-1,'first_count':total,'observed_counts':sorted(set(counts)),'unique_records':len(seen)})


if __name__=='__main__':run()
