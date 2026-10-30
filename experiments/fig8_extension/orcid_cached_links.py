"""Recover publisher author-affiliation links using corroborated cached ORCIDs."""
from __future__ import annotations

import fcntl
import gzip
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .fig8b_qualification_build import key
from .fig8b_qualification_recompute import metrics
from .nature_backfill_registry import normalized
from .nature_backfill_resolve import country_names
from .supplement_apply import enrich_authors, signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def canonical_orcid(value: str) -> str:
    token=str(value or '').strip().rstrip('/').rsplit('/',1)[-1].upper()
    if not re.fullmatch(r'\d{4}-\d{4}-\d{4}-\d{3}[\dX]',token):return ''
    total=0
    for digit in token.replace('-','')[:-1]:total=(total+int(digit))*2
    checksum=(12-total%11)%11
    return token if token[-1]==('X' if checksum==10 else str(checksum)) else ''


@lru_cache(maxsize=8)
def raw_page(path: Path) -> dict:
    with gzip.open(path,'rt') as stream:return json.load(stream)


def identifiers(paper: dict, root: Path) -> dict[str,str]:
    receipt=paper.get('authorships_source_receipt') or paper['raw_receipt']
    path=root/'fields'/receipt
    raw=raw_page(path)
    response=raw.get('response',{})
    if 'results' in response:
        record=response['results'][paper['raw_position']]
    else:record=response
    if record.get('id')!=paper['work_id'] or (record.get('doi') or '').removeprefix('https://doi.org/').lower()!=paper['doi']:return {}
    values=defaultdict(set);counts=Counter()
    for author in record.get('authorships',[]):
        person=author.get('author',{});aid=person.get('id','');counts[aid]+=1
        raw_values=[person.get('orcid',''),*(person.get('observed_orcids') or [])]
        values[aid].update(v for raw in raw_values if (v:=canonical_orcid(raw)))
    return {aid:next(iter(v)) for aid,v in values.items() if aid and len(v)==1 and counts[aid]==1}


def link_authors(current: list[dict], sources: list[dict], ids: dict, registry: dict, by_name: dict, names: dict) -> tuple[list[dict],Counter]:
    result=json.loads(json.dumps(current));stats=Counter()
    current_orcids=[ids.get(a.get('author_id')) for a in current];counts=Counter(x for x in current_orcids if x)
    # Index each byline once; large collaborations otherwise repeatedly validate
    # every publisher ORCID for every missing author link.
    indexed_sources = []
    for source in sources:
        index = defaultdict(list)
        for candidate in source['authors']:
            identifier = canonical_orcid(candidate.get('orcid', ''))
            if identifier:
                index[identifier].append(candidate)
        indexed_sources.append((source, index))
    for index,author in enumerate(result):
        if author.get('nature_v1_slots') or author.get('publisher_affiliation_slots'):continue
        orcid=ids.get(author.get('author_id'))
        if not orcid or counts[orcid]!=1:continue
        matches=[];ambiguous=False
        for source, index_by_orcid in indexed_sources:
            candidates=index_by_orcid.get(orcid, [])
            if len(candidates)>1:ambiguous=True;break
            if len(candidates)==1:matches.append((candidates[0],source))
        if ambiguous or not matches:continue
        if len({json.dumps(a['affiliations'],sort_keys=True) for a,_ in matches})!=1:
            stats['orcid_affiliation_source_conflicts']+=1;continue
        matched,source=matches[0]
        if not matched.get('affiliations') or matched.get('unresolved_affiliation_refs'):continue
        # The internal alias is used only after ORCID agreement; neither displayed name changes.
        local_source={**source,'authors':[{**matched,'name':author['name']}]}
        enriched,delta=enrich_authors([author],local_source,registry,by_name,names)
        if not enriched[0].get('publisher_affiliation_slots'):continue
        value=enriched[0];value['publisher_author_identity_match']='unique_checksum_valid_orcid_in_both_paper_bylines'
        value['publisher_author_original_name']=matched['name'];value['corroborated_orcid']=orcid
        result[index]=value;stats.update(delta);stats['publisher_author_links_recovered']+=1
    return result,stats


def run(root: Path = ROOT) -> None:
    folder=root/'supplement';q=root/'fields/qualification';stats=Counter()
    registry=defaultdict(list);by_name=defaultdict(list)
    for record in json.loads((root/'fields/institution_registry.json').read_text()):
        if record.get('ror'):registry[record['ror'].rstrip('/')].append(record['id'])
        by_name[key(record['display_name'])].append(record)
    names=country_names()
    for name in ['korea','georgia','congo']:names.pop(normalized(name),None)
    journals={r['source_id']:set(json.loads(r['issn'] or '[]')) for r in read_csv(root/'fields/journal_registry.csv')}
    with (folder/'checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=connect(root)
        for path in sorted((root/'fields/papers').glob('*.parquet')):
            included={r['work_id'] for r in pq.read_table(q/'decisions'/path.name,columns=['work_id','eligibility_status'],filters=[('eligibility_status','=','included_verified_research')]).to_pylist()}
            table=pq.read_table(path);rows=table.to_pylist();changed=False
            for paper in rows:
                if paper['work_id'] not in included:continue
                authors=json.loads(paper['authors_json'])
                if all(a.get('publisher_affiliation_slots') or a.get('nature_v1_slots') for a in authors):continue
                sources=[json.loads(r[0]) for r in db.execute("SELECT payload FROM source_metadata WHERE doi=? AND kind='jats'",(paper['doi'],))]
                sources=[s for s in sources if signature_match(paper,s,journals[paper['source_id']]) and any(canonical_orcid(a.get('orcid','')) for a in s.get('authors',[]))]
                if not sources:continue
                # Publisher-only reconstructed bylines do not have an OpenAlex raw-position receipt.
                receipt=paper.get('authorships_source_receipt') or paper['raw_receipt']
                if not receipt.startswith(('census_raw/','full_authors_raw/')):
                    receipt=paper['raw_receipt']
                ids=identifiers({**paper,'authorships_source_receipt':receipt},root)
                updated,delta=link_authors(authors,sources,ids,registry,by_name,names)
                if updated==authors:continue
                before=paper['authors_json'];paper['authors_json']=json.dumps(updated,ensure_ascii=False);changed=True
                stats.update(delta);stats['changed_papers']+=1
                evidence=json.dumps({'identifier_receipt':'fields/'+receipt,'publisher_receipts':sorted({s['evidence_receipt'] for s in sources})})
                db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',(paper['work_id'],'authors_json',before,paper['authors_json'],evidence,'Unique checksum-valid ORCID agrees within exact-matched cached paper and publisher byline; no ownership inferred'))
                for metric,value in metrics(paper).items():
                    field='after_'+metric
                    if field in paper and paper[field]!=value:
                        db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',(paper['work_id'],field,json.dumps(paper[field]),json.dumps(value),evidence,'Recomputed observed metadata flag after corroborated author link'))
                        paper[field]=value
            if changed:
                tmp=path.with_suffix('.parquet.tmp');pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),tmp,compression='zstd');tmp.replace(path)
            db.commit();atomic_json(folder/'orcid_cached_links_status.json',{'status':'running','shard':path.name,**stats});print(path.name,dict(stats),flush=True)
        db.close()
    atomic_json(folder/'orcid_cached_links_status.json',{'status':'applied_derived_refresh_required',**stats})


if __name__=='__main__':run()
