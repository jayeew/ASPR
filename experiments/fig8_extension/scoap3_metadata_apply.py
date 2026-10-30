"""Reuse matched SCOAP3 cache statements without asserting complete publisher bylines."""
from __future__ import annotations

import fcntl
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .fig8b_qualification_build import key, plain
from .fig8b_qualification_recompute import metrics
from .nature_backfill_registry import normalized
from .nature_backfill_resolve import country_names
from .secondary_author_cache import enrich
from .scoap3_cache_index import journal_variants
from .supplement_apply import abstract_ok, signature_match
from .supplement_queue import ROOT, connect, task
from .supplement_sources import atomic_json


def metadata(row: dict, receipt: str) -> dict:
    return {'title':[row.get('title','')],
        'journal_titles':journal_variants([v.get('journal_title','') for v in row.get('publication_info',[])]),
        'abstract':plain(row.get('abstract') or ''),
        'authors':[{'name':a.get('name',''),'affiliations':[
            {'text':v['value'],'countries':[],'identifiers':[],'names':[]}
            for v in a.get('affiliations',[]) if v.get('value')]}
            for a in row.get('authors',[]) if a.get('name')],
        'evidence_receipt':receipt,'source_url':f'https://repo.scoap3.org/records/{row["id"]}',
        'authority':'SCOAP3 publisher-deposited archive metadata; source byline completeness not asserted; repository country/ROR annotations not independently verified'}


def merge(current: list[dict], sources: list[dict], registry: dict, by_name: dict,
          names: dict) -> tuple[list[dict], Counter]:
    result=json.loads(json.dumps(current));stats=Counter()
    # Existing secondary and primary statements are retained, not silently replaced.
    eligible=[a for a in result if not a.get('secondary_affiliation_slots')]
    updated,delta=enrich(eligible,sources,registry,by_name,names)
    # Duplicate names are checked against the entire current byline as well.
    counts=Counter(key(a['name']) for a in result)
    mapped={key(a['name']):a for a in updated if counts[key(a['name'])]==1}
    for i,author in enumerate(result):
        replacement=mapped.get(key(author['name']))
        if replacement is None or replacement==author:continue
        if replacement.get('identity_source')=='epmc_exact_author_name_and_declared_unit_site':
            replacement['identity_source']='scoap3_exact_author_name_and_declared_unit_site'
        if replacement.get('country_source')=='epmc_explicit_affiliation_country':
            replacement['country_source']='scoap3_explicit_affiliation_country'
        result[i]=replacement;stats['author_statements_added']+=1
        stats['countries_filled']+=not author['countries'] and bool(replacement['countries'])
        stats['institutions_filled']+=not author['institution_ids_or_verified_local_entities'] and bool(replacement['institution_ids_or_verified_local_entities'])
    stats['source_affiliation_conflicts']=delta['secondary_affiliation_conflicts']
    return result,stats


def run(root: Path = ROOT) -> None:
    folder=root/'supplement/scoap3';stats=Counter()
    cache=sqlite3.connect(folder/'cache_index.sqlite')
    selected=defaultdict(list)
    for wid,payload,receipt in cache.execute("SELECT work_id,payload,receipt FROM matches JOIN records USING(id,payload_hash) WHERE status='identity_matched'"):
        selected[wid].append(metadata(json.loads(payload),receipt))
    cache.close()
    registry=defaultdict(list);by_name=defaultdict(list)
    for row in json.loads((root/'fields/institution_registry.json').read_text()):
        if row.get('ror'):registry[row['ror'].rstrip('/')].append(row['id'])
        by_name[key(row['display_name'])].append(row)
    names=country_names()
    for ambiguous in ['korea','georgia','congo']:names.pop(normalized(ambiguous),None)
    with (root/'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=connect(root)
        for path in sorted((root/'fields/papers').glob('*.parquet')):
            included={r['work_id'] for r in pq.read_table(root/'fields/qualification/decisions'/path.name,
                columns=['work_id'],filters=[('eligibility_status','=','included_verified_research')]).to_pylist()}
            target=included & selected.keys()
            if not target:continue
            table=pq.read_table(path);rows=table.to_pylist();changed=False
            for paper in rows:
                if paper['work_id'] not in target:continue
                sources=[s for s in selected[paper['work_id']] if signature_match(paper,s,set())]
                if not sources:continue
                stats['qualified_identity_matches']+=1
                before={k:paper[k] for k in ['abstract','abstract_source','authors_json',*[k for k in paper if k.startswith('after_')]]}
                authors,delta=merge(json.loads(paper['authors_json']),sources,registry,by_name,names)
                paper['authors_json']=json.dumps(authors,ensure_ascii=False);stats.update(delta)
                abstracts={s['abstract'] for s in sources if abstract_ok(s['abstract'])}
                if not paper['abstract'] and len(abstracts)==1:
                    paper['abstract']=abstracts.pop();paper['abstract_source']='scoap3_matched_archive:'+sources[0]['evidence_receipt'];stats['abstracts_filled']+=1
                receipts=json.dumps(sorted({s['evidence_receipt'] for s in sources}))
                for metric,value in metrics(paper).items():
                    if 'after_'+metric in paper:paper['after_'+metric]=value
                for field,old in before.items():
                    if old==paper[field]:continue
                    db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                        (paper['work_id'],field,json.dumps(old),json.dumps(paper[field]),receipts,
                         'SCOAP3 cached DOI/title/journal match; exact author name; secondary authority retained, no complete byline or university ownership assertion'))
                    changed=True;stats['changed_'+field]+=1
                for author in authors:
                    if author.get('secondary_country_conflict') and author.get('secondary_affiliation_evidence',{}).get('authority','').startswith('SCOAP3'):
                        task(db,'country_conflict',paper['work_id']+':'+str(author['position'])+':secondary','scoap3_archive',
                             {'reason':'Archive signature country conflicts with existing country','evidence':author['secondary_affiliation_evidence']},paper['work_id'])
                values=metrics(paper)
                db.execute('UPDATE papers SET abstract_gap=?,institution_gap=?,country_gap=? WHERE work_id=?',
                           (not values['abstract'],not values['every_author_institution'],not values['every_author_country'],paper['work_id']))
            if changed:
                temporary=path.with_suffix('.parquet.tmp');pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),temporary,compression='zstd');temporary.replace(path)
            db.commit();print('Applied archive cache',path.name,dict(stats),flush=True)
        db.close()
    atomic_json(folder/'metadata_apply_status.json',dict(stats))


if __name__=='__main__':run()
