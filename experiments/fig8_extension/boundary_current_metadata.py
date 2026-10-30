"""Batch current metadata for fixed, locally matched publisher-boundary clues."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone

import requests

from gear.env import getenv
from .fig8b_collect import SELECT
from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json


def run(all_directory: bool = False) -> dict:
    folder=ROOT/'supplement/boundary_review'
    if all_directory:
        scope_path=folder/'full_directory_scope.json'
        if scope_path.exists():dois=json.loads(scope_path.read_text())['dois']
        else:
            completed=set()
            for cached in (folder/'current_raw').glob('*.gz'):
                with gzip.open(cached,'rt') as stream:prior=json.load(stream)
                if prior.get('status')=='success':completed.update(prior['requested_dois'])
            with (ROOT/'supplement/boundary_candidates.csv').open(encoding='utf-8-sig') as stream:all_dois={r['doi'] for r in csv.DictReader(stream)}
            dois=sorted(all_dois-completed)
            atomic_json(scope_path,{'dois':dois,'already_successfully_queried':len(all_dois&completed),'scope':'Fixed current publisher-directory omissions; three existing primary fields only; no new source years/journals'})
        raw_folder=folder/'full_directory_raw';status_path=folder/'full_directory_status.json';batch_size=100
    else:
        targets=json.loads((folder/'current_metadata_targets.json').read_text());dois=sorted({r['doi'] for r in targets})
        raw_folder=folder/'current_raw';status_path=folder/'current_status.json';batch_size=50
    raw_folder.mkdir(exist_ok=True);stats=Counter()
    session=requests.Session();session.trust_env=False
    credential=(getenv('OPENALEX_API_KEY') or getenv('OPENALEX_API_KEYS').split(',')[0]).strip()
    for offset in range(0,len(dois),batch_size):
        group=dois[offset:offset+batch_size];params={'filter':'doi:'+'|'.join(group)+(',primary_topic.field.id:13|25|31' if all_directory else ''),'per_page':200,'select':SELECT}
        token=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest();path=raw_folder/(token+'.json.gz')
        if path.exists():
            with gzip.open(path,'rt') as stream:r=json.load(stream)
            stats['cached_batches']+=1
        else:
            r={'url':'https://api.openalex.org/works','params':params,'requested_dois':group,'attempts':[],'retrieved_at':datetime.now(timezone.utc).isoformat()}
            for attempt in range(3):
                try:
                    response=session.get(r['url'],params={**params,**({'api_key':credential} if credential else {})},timeout=(10,45))
                    r['http_status']=response.status_code;r['attempts'].append({'http_status':response.status_code})
                    if response.ok:
                        r['response']=response.json()
                        r['status']='success' if r['response'].get('meta',{}).get('count',0)<=len(r['response'].get('results',[])) else 'response_truncated'
                        break
                    if response.status_code not in {500,502,503,504}:break
                except (requests.RequestException,ValueError) as error:r['attempts'].append({'error':type(error).__name__})
                if attempt<2:time.sleep(2**attempt)
            r.setdefault('status','source_unresolved');atomic_bytes(path,gzip.compress(json.dumps(r).encode()));stats['new_batches']+=1;time.sleep(1.05)
        stats['returned_records']+=len(r.get('response',{}).get('results',[]))
        atomic_json(status_path,{'status':'running','fixed_dois':len(dois),**stats})
        if r.get('http_status') in {401,402,403,429}:
            result={'status':'source_access_restricted_or_rate_limited','fixed_dois':len(dois),**stats,'receipt':str(path)};atomic_json(status_path,result);print(json.dumps(result),flush=True);return result
    result={'status':'fixed_batches_processed_identity_scope_review_required','fixed_dois':len(dois),**stats};atomic_json(status_path,result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':run()
