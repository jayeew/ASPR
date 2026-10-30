"""Reuse DOI-linked deposit caches for newly pending explicit online dates."""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from datetime import date

from .cached_date_apply import deposited_title, run as apply_dates
from .fig8b_qualification_build import plain
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> dict:
    folder = ROOT / 'supplement/indexed_online_date_reuse'
    folder.mkdir(exist_ok=True)
    db = connect()
    papers = db.execute("SELECT doi,work_id FROM papers WHERE status='pending_primary_online_date' AND doi<>''").fetchall()
    paths: dict[str, set[str]] = defaultdict(set)
    for doi, _ in papers:
        for receipt, in db.execute("SELECT receipt FROM source_metadata WHERE doi=? AND kind='deposit'", (doi,)):
            paths[receipt.split('#', 1)[0]].add(doi)
    db.close()
    previous_path = folder / 'reviewed_cache_pairs.json'
    previous = set(tuple(v) for v in json.loads(previous_path.read_text())) if previous_path.exists() else set()
    sources: dict[str, list[dict]] = defaultdict(list)
    reviewed = []
    for receipt, dois in paths.items():
        dois = {doi for doi in dois if (doi, receipt) not in previous}
        if not dois:
            continue
        with gzip.open(ROOT / receipt, 'rt') as stream:
            record = json.load(stream)
        items = {i['DOI'].lower(): i for i in record.get('response', {}).get('message', {}).get('items', [])}
        for doi in dois:
            item = items.get(doi, {})
            found = False
            for parts in item.get('published-online', {}).get('date-parts', []):
                if not parts or len(parts) > 3 or not all(isinstance(v, int) for v in parts):
                    continue
                try:
                    date(parts[0], parts[1] if len(parts)>1 else 1, parts[2] if len(parts)>2 else 1)
                except ValueError:
                    continue
                sources[doi].append({'family':'crossref_publisher_deposit',
                    'title':deposited_title(plain((item.get('title') or [''])[0])),
                    'issns':item.get('ISSN', []),'journal_titles':item.get('container-title', []),
                    'kind':item.get('type','unspecified'),
                    'online_date':'-'.join(str(v).zfill(4 if i==0 else 2) for i,v in enumerate(parts)),
                    'date_precision':['year','month','day'][len(parts)-1],
                    'receipt':'../'+receipt.removeprefix('fields/'),'url':record['url'],'categories':[]})
                found = True
            reviewed.append({'doi':doi,'receipt':receipt,
                             'status':'explicit_date_for_identity_review' if found else 'no_explicit_online_date',
                             'final_task_closure':False})
    atomic_json(folder / 'current_scope.json', {'papers':papers,'cache_reviews':reviewed,'network_requests':0})
    result = apply_dates(set(sources),folder,explicit_sources=dict(sources)) if sources else {'new_evidence':0}
    with (folder / 'reviews.jsonl').open('a') as stream:
        for row in reviewed:
            stream.write(json.dumps(row)+'\n')
    previous.update((r['doi'],r['receipt']) for r in reviewed)
    atomic_json(previous_path,sorted(previous))
    summary={'pending_date_papers':len(papers),'new_cache_pairs_reviewed':len(reviewed),
             'explicit_date_dois':len(sources),'application':result,'network_requests':0,
             'unlinked_or_field_absent_paths_not_closed':True}
    atomic_json(folder / 'status.json',summary)
    return summary


if __name__ == '__main__':
    print(json.dumps(run()),flush=True)
