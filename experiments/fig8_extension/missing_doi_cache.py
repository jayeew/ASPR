"""Recover existing landing-page leads for DOI-less candidates without any requests."""
from __future__ import annotations

import gzip
import json
import sqlite3
from collections import Counter, defaultdict
from urllib.parse import urlsplit

import pyarrow.parquet as pq

from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> dict:
    folder = ROOT / 'supplement/missing_doi'
    folder.mkdir(exist_ok=True)
    db = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
    groups = defaultdict(list)
    for wid, shard in db.execute("SELECT work_id,shard FROM papers WHERE (doi IS NULL OR doi='') AND status LIKE 'pending%'"):
        groups[shard].append(wid)
    db.close()
    by_receipt = defaultdict(dict)
    for shard, ids in groups.items():
        for row in pq.read_table(ROOT / 'fields/papers' / shard,
                                 columns=['work_id','title','source_id','raw_receipt','raw_position'],
                                 filters=[('work_id','in',ids)]).to_pylist():
            by_receipt[row['raw_receipt']][row['work_id']] = row
    hosts = Counter(); states = Counter(); source_types = Counter()
    temporary = folder / 'source_leads.jsonl.tmp'
    with temporary.open('w') as output:
        for receipt, wanted in by_receipt.items():
            document = json.loads(gzip.decompress((ROOT / 'fields' / receipt).read_bytes()))
            found = set()
            for raw in document.get('response',{}).get('results',[]):
                wid = raw['id']
                if wid not in wanted:
                    continue
                found.add(wid)
                source = raw.get('primary_location') or {}
                url = source.get('landing_page_url') or ''
                host = urlsplit(url).netloc.lower() if url else ''
                state = 'cached_landing_page_requires_primary_review' if host else 'no_primary_landing_page_in_cache'
                states[state] += 1; hosts[host or 'missing'] += 1
                source_types[raw.get('type') or 'missing'] += 1
                output.write(json.dumps({'work_id':wid,'title':wanted[wid]['title'],
                    'candidate_source_id':wanted[wid]['source_id'], 'raw_receipt':'fields/'+receipt,
                    'landing_page_url':url,'pdf_url':source.get('pdf_url'), 'host':host,
                    'raw_indexed_type':raw.get('type'), 'raw_indexed_language':raw.get('language'),
                    'status':state,'qualification_unchanged':True},ensure_ascii=False)+'\n')
            for wid in wanted.keys()-found:
                states['work_missing_from_recorded_cache'] += 1
                output.write(json.dumps({'work_id':wid,'raw_receipt':'fields/'+receipt,
                    'status':'work_missing_from_recorded_cache','qualification_unchanged':True})+'\n')
    temporary.replace(folder / 'source_leads.jsonl')
    result = {'candidate_n':sum(map(len,groups.values())), 'raw_files_reused':len(by_receipt),
              'network_requests':0,'states':dict(states),'hosts':dict(hosts.most_common()),
              'indexed_types_not_primary_confirmation':dict(source_types),
              'limitation':'Landing pages and indexed types are leads; no exclusion or original-research assertion from secondary metadata.'}
    atomic_json(folder / 'cache_summary.json',result)
    print(json.dumps({**result,'hosts':hosts.most_common(15)},ensure_ascii=False),flush=True)
    return result


if __name__ == '__main__':
    run()
