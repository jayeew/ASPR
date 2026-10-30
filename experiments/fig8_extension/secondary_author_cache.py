"""Reuse exact-matched Europe PMC author-affiliation statements with explicit authority."""
from __future__ import annotations

import csv
import fcntl
import gzip
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .fig8b_qualification_build import key
from .fig8b_qualification_recompute import metrics
from .nature_backfill_registry import normalized
from .nature_backfill_resolve import country_names
from .secondary_markup_recovery import decoded_title
from .supplement_apply import institution_ids, signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def metadata(item: dict, receipt: str) -> dict:
    journal=item.get('journalInfo',{}).get('journal',{})
    authors=[]
    for author in item.get('authorList',{}).get('author',[]):
        name=' '.join(filter(None,[author.get('firstName'),author.get('lastName')]))
        if not name:continue  # Abbreviations and collective names are not resolved by guessing.
        affiliations=[{'text':a['affiliation'],'countries':[],'identifiers':[],'names':[]}
            for a in author.get('authorAffiliationDetailsList',{}).get('authorAffiliation',[])
            if a.get('affiliation')]
        authors.append({'name':name,'affiliations':affiliations})
    return {'title':[decoded_title(item.get('title',''))],'issns':[journal.get('issn',''),journal.get('essn','')],
        'journal_titles':[journal.get('title','')],'authors':authors,'evidence_receipt':receipt,
        'source_url':'https://europepmc.org/article/'+item.get('source','MED')+'/'+item.get('id',''),
        'authority':'Europe PMC indexed author-affiliation statements; byline completeness not asserted'}


def enrich(current: list[dict], sources: list[dict], registry: dict, by_name: dict, names: dict,
           source_label: str = 'epmc', preserve_secondary: bool = False) -> tuple[list[dict], Counter]:
    result=json.loads(json.dumps(current));stats=Counter()
    counts=Counter(key(a['name']) for a in current)
    for author in result:
        if preserve_secondary and (author.get('secondary_affiliation_slots')
                                   or author.get('nature_v1_slots') is not None):
            continue
        if author.get('nature_v1_slots') or author.get('publisher_affiliation_slots'):continue
        if counts[key(author['name'])]!=1:continue
        candidates=[]
        for source in sources:
            matches=[a for a in source['authors'] if key(a['name'])==key(author['name'])]
            if len(matches)==1 and matches[0]['affiliations']:candidates.append((matches[0],source))
        if not candidates:continue
        variants={json.dumps(a['affiliations'],sort_keys=True) for a,_ in candidates}
        if len(variants)!=1:
            stats['secondary_affiliation_conflicts']+=1;continue
        matched,source=candidates[0]
        ids,slots=institution_ids(matched,registry,by_name,names)
        evidence={k:source[k] for k in ['evidence_receipt','source_url','authority']}
        author['secondary_affiliation_slots']=slots
        author['secondary_affiliation_evidence']=evidence
        if ids and not author['institution_ids_or_verified_local_entities']:
            author['institution_ids_or_verified_local_entities']=ids
            author['identity_source']=source_label+'_exact_author_name_and_declared_unit_site'
            stats['author_institution_filled']+=1
        site=sorted({c for s in slots for c in s['country_codes']}) if all(s['country_codes'] for s in slots) else []
        if site and not author['countries']:
            author['countries']=site;author['country_source']=source_label+'_explicit_affiliation_country'
            stats['author_country_filled']+=1
        elif site and set(site)!=set(author['countries']):
            author['secondary_country_conflict']=True
        stats['matched_author_statements']+=1
    return result,stats


def run(root: Path = ROOT, scope_dois: set[str] | None = None) -> None:
    folder=root/'supplement';q=root/'fields/qualification'
    db=connect(root);db.execute('PRAGMA busy_timeout=900000');cache=sqlite3.connect(folder/'bulk_cache.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS secondary_author_metadata(doi TEXT,receipt TEXT,payload TEXT,PRIMARY KEY(doi,receipt))')
    included=set();included_ids=set();shards=set()
    if scope_dois is None:
        for path in (q/'decisions').glob('*.parquet'):
            selected=pq.read_table(path,columns=['doi','work_id','eligibility_status'],filters=[('eligibility_status','=','included_verified_research')]).to_pylist()
            included.update(r['doi'] for r in selected);included_ids.update(r['work_id'] for r in selected)
    else:
        db.execute('CREATE TEMP TABLE secondary_scope(doi TEXT PRIMARY KEY)')
        db.executemany('INSERT INTO secondary_scope VALUES(?)', ((doi,) for doi in scope_dois))
        for doi,wid,shard in db.execute("SELECT p.doi,p.work_id,p.shard FROM papers p JOIN secondary_scope s USING(doi) WHERE p.status='included_verified_research'"):
            included.add(doi);included_ids.add(wid);shards.add(shard)
    db.commit()  # End temporary-scope reads before acquiring a main-database write transaction.
    cache.execute('CREATE TEMP TABLE selected(doi TEXT PRIMARY KEY)')
    cache.executemany('INSERT INTO selected VALUES(?)',[(d,) for d in included if d])
    paths=[r[0] for r in cache.execute('SELECT DISTINCT path FROM records JOIN selected USING(doi)')];cache.close()
    stats=Counter()
    # Index selected cached files afresh: new qualifications may select previously unindexed DOIs in the same batch.
    for i,rel in enumerate(paths,1):
        with gzip.open(root/rel,'rt') as stream:response=json.load(stream)
        if response.get('status')!='complete':continue
        for item in response.get('response',{}).get('resultList',{}).get('result',[]):
            doi=(item.get('doi') or '').lower()
            if doi not in included:continue
            value=metadata(item,rel)
            if not any(a['affiliations'] for a in value['authors']):continue
            db.execute('INSERT OR REPLACE INTO secondary_author_metadata VALUES(?,?,?)',(doi,rel,json.dumps(value,ensure_ascii=False)))
            stats['indexed_records']+=1
        if i%100==0:db.commit();print('Secondary cache files',i,'/',len(paths),flush=True)
    db.commit()
    registry=defaultdict(list);by_name=defaultdict(list)
    for record in json.loads((root/'fields/institution_registry.json').read_text()):
        if record.get('ror'):registry[record['ror'].rstrip('/')].append(record['id'])
        by_name[key(record['display_name'])].append(record)
    names=country_names()
    for ambiguous in ['korea','georgia','congo']:names.pop(normalized(ambiguous),None)
    with (root/'fields/journal_registry.csv').open(encoding='utf-8-sig') as f:
        journals={r['source_id']:set(json.loads(r['issn'] or '[]')) for r in csv.DictReader(f)}
    with (folder/'checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for path in sorted((root/'fields/papers').glob('*.parquet')):
            if scope_dois is not None and path.name not in shards:continue
            table=pq.read_table(path);rows=table.to_pylist();changed=False
            for paper in rows:
                if paper['work_id'] not in included_ids:continue
                sources=[json.loads(r[0]) for r in db.execute('SELECT payload FROM secondary_author_metadata WHERE doi=?',(paper['doi'],))]
                sources=[v for v in sources if signature_match(paper,v,journals[paper['source_id']])]
                if not sources:continue
                before=paper['authors_json'];authors,delta=enrich(json.loads(before),sources,registry,by_name,names)
                after=json.dumps(authors,ensure_ascii=False)
                if after==before:continue
                paper['authors_json']=after;changed=True;stats.update(delta);stats['changed_papers']+=1
                for metric,value in metrics(paper).items():
                    field='after_'+metric
                    if field in paper and paper[field]!=value:
                        db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',(paper['work_id'],field,json.dumps(paper[field]),json.dumps(value),sources[0]['evidence_receipt'],'Recomputed observed metadata flag; secondary source authority retained'))
                        paper[field]=value
                receipts=sorted({v['evidence_receipt'] for v in sources})
                db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                    (paper['work_id'],'authors_json',before,after,json.dumps(receipts),'DOI/title/journal and unique full author name matched; secondary source explicitly labelled; no byline completeness or new university ownership asserted'))
            if changed:
                temp=path.with_suffix('.parquet.tmp');pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),temp,compression='zstd');temp.replace(path)
            db.commit();print('Secondary statements applied',path.name,changed,flush=True)
    atomic_json(folder/('secondary_author_cache_summary.json' if scope_dois is None else 'secondary_author_incremental_summary.json'),{**dict(stats), 'selected_dois':len(included), 'network_requests':0});db.close()


if __name__=='__main__':run()
