"""Find current-publisher leads; historical brand needs independent dated evidence."""
from __future__ import annotations

import gzip
import json
import sqlite3
from collections import Counter, defaultdict

import pyarrow.parquet as pq

from .fig8b_qualification_build import key, plain
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> None:
    selected={}
    for path in (ROOT/'fields/qualification/decisions').glob('*.parquet'):
        for row in pq.read_table(path,columns=['work_id','doi','eligibility_status','nature_portfolio_at_verified_online_year','online_year']).to_pylist():
            if row['eligibility_status']=='included_verified_research' and row['nature_portfolio_at_verified_online_year'] not in ['True','False']:
                selected[row['doi']]=row
    db=sqlite3.connect(ROOT/'supplement/tasks.sqlite');proposals=defaultdict(list);stats=Counter()
    terms=['elsevier','wiley','american physical society','american chemical society','royal society of chemistry','iop publishing','frontiers','mdpi','multidisciplinary digital','oxford university press','optica','edp sciences','public library of science','american institute of physics','aip publishing','american association for the advancement of science','national academy of sciences']
    for path in (ROOT/'fields/crossref_raw').glob('*.gz'):
        with gzip.open(path,'rt') as stream:response=json.load(stream)
        if response.get('http_status')!=200:continue
        message=response.get('response',{}).get('message',{});items=message.get('items',[message])
        for item in items:
            doi=item.get('DOI','').lower();row=selected.get(doi)
            if not row:continue
            publisher=item.get('publisher','')
            if not any(t in publisher.casefold() for t in terms):continue
            parts=item.get('published-online',{}).get('date-parts',[[]])[0]
            if not parts or parts[0]!=row['online_year']:continue
            paper=db.execute('SELECT title,journal FROM papers WHERE work_id=?',(row['work_id'],)).fetchone()
            if not paper or key(paper[0]) not in {key(plain(v)) for v in item.get('title',[])}:continue
            if key(paper[1]) not in {key(v) for v in item.get('container-title',[])}:continue
            proposals[doi].append({'work_id':row['work_id'],'doi':doi,'online_year':parts[0],
                'publisher':publisher,'receipt':str(path.relative_to(ROOT)),
                'title':item.get('title',[]),'journal':item.get('container-title',[]),
                'status':'current_publisher_lead_requires_historical_evidence',
                'scope':'Exact identity and online year; Crossref publisher updates after transfers and cannot establish historical brand alone',
                'limitation_source':'https://www.crossref.org/documentation/register-maintain-records/maintaining-your-metadata/updating-after-title-transfer/'})
    db.close()
    rows=[v for values in proposals.values() for v in values]
    atomic_json(ROOT/'supplement/crossref_brand_candidates.json',rows)
    stats.update({'qualified_unknown_brand_dois':len(selected),'candidate_dois':len(proposals),'evidence_records':len(rows),'confirmed_historical_labels':0})
    atomic_json(ROOT/'supplement/crossref_brand_candidate_summary.json',dict(stats));print(json.dumps(dict(stats)),flush=True)


if __name__=='__main__':run()
