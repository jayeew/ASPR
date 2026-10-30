"""Resolve lower-authority country assignments against agreeing complete paper signatures."""
from __future__ import annotations

import fcntl
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import metadata_task_delta
from .fig8b_qualification_build import key
from .orcid_cached_links import canonical_orcid
from .supplement_apply import countries, signature_match
from .nature_backfill_registry import normalized
from .nature_backfill_resolve import country_names
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, separate_affiliation_notes
from .three_journal_apply import read_csv

LOWER_AUTHORITY={'openalex_authorship_countries','all_observed_raw_addresses_explicit_terminal_country',
    'epmc_exact_doi_title_unique_full_author_name_explicit_country','epmc_explicit_affiliation_country'}


def index_authors(source: dict) -> dict:
    """Index one immutable source byline once, retaining ambiguous name matches."""
    by_name={};by_orcid={}
    for author in source.get('authors',[]):
        by_name.setdefault(key(author['name']),[]).append(author)
        orcid=canonical_orcid(author.get('orcid',''))
        if orcid:by_orcid.setdefault(orcid,[]).append(author)
    return {**source,'_by_name':by_name,'_by_orcid':by_orcid}


def resolve(author: dict, sources: list[dict], names: dict) -> tuple[dict,list[str]]:
    if (author.get('nature_v1_slots') is not None
            or not (author.get('country_conflict_retained') or author.get('secondary_country_conflict'))):
        return author,[]
    if author.get('country_source') not in LOWER_AUTHORITY:return author,[]
    slots=author.get('publisher_affiliation_slots')
    if not slots or not all(s.get('country_codes') for s in slots):return author,[]
    orcid=canonical_orcid(author.get('corroborated_orcid',''));agreed=[];receipts=[];present_notes=[]
    for source in sources:
        if '_by_name' in source:
            matches=source['_by_orcid'].get(orcid,[]) if orcid else source['_by_name'].get(key(author['name']),[])
        else:
            matches=[a for a in source.get('authors',[]) if canonical_orcid(a.get('orcid',''))==orcid] if orcid else [a for a in source.get('authors',[]) if key(a['name'])==key(author['name'])]
        if len(matches)>1:return author,[]
        if not matches:continue
        if (source.get('country_certification_scope') == 'explicit_individual_links_only'
                and not matches[0].get('country_link_certified')):
            return author,[]
        a=separate_affiliation_notes(matches[0])
        for note in a.get('excluded_present_addresses', []):
            if note not in present_notes:present_notes.append(note)
        if a.get('unresolved_affiliation_refs') or not a.get('affiliations'):return author,[]
        site=countries(a['affiliations'],names)
        if not site:return author,[]
        agreed.append(tuple(site));receipts.append(source['evidence_receipt'])
    if not agreed or len(set(agreed))!=1:return author,[]
    declared=sorted({c for s in slots for c in s['country_codes']})
    if declared!=list(agreed[0]):return author,[]
    updated={**author,'countries':declared,'country_source':'publisher_explicit_affiliation_countries',
        'country_conflict_retained':False,'country_conflict_disposition':'resolved_lower_authority_assignment_superseded_by_agreeing_complete_publisher_addresses',
        'country_resolution_receipts':sorted(set(receipts))}
    if present_notes:
        updated['country_resolution_excluded_present_addresses'] = present_notes
    if updated.get('secondary_country_conflict'):
        updated['secondary_country_conflict']=False
        updated['secondary_country_conflict_disposition']='resolved_by_agreeing_complete_publisher_addresses_secondary_statement_preserved'
    return updated,sorted(set(receipts))


def run(root: Path = ROOT, scope_dois: set[str] | None = None, status_path: Path | None = None,
        scope_positions: dict[str, set[int]] | None = None) -> None:
    folder=root/'supplement';q=root/'fields/qualification';stats=Counter()
    status_path = status_path or folder/'primary_country_resolution.json'
    names=country_names()
    for name in ['korea','georgia','congo']:names.pop(normalized(name),None)
    journals={r['source_id']:set(json.loads(r['issn'] or '[]')) for r in read_csv(root/'fields/journal_registry.csv')}
    with (folder/'checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);db=connect(root)
        selected_shards = None if scope_dois is None else {
            shard for doi in scope_dois for shard, in db.execute('SELECT shard FROM papers WHERE doi=?', (doi,))}
        for path in sorted((root/'fields/papers').glob('*.parquet')):
            if selected_shards is not None and path.name not in selected_shards:continue
            included={r['work_id'] for r in pq.read_table(q/'decisions'/path.name,columns=['work_id','eligibility_status'],filters=[('eligibility_status','=','included_verified_research')]).to_pylist()}
            decision_path=q/'decisions'/path.name
            before_signature=metadata_task_delta.signature(path,decision_path)
            changed_ids=set()
            table=pq.read_table(path);rows=table.to_pylist();changed=False
            for paper in rows:
                if scope_dois is not None and paper['doi'] not in scope_dois:continue
                if paper['work_id'] not in included:continue
                authors=json.loads(paper['authors_json'])
                if not any((a.get('country_conflict_retained') or a.get('secondary_country_conflict'))
                           and a.get('publisher_affiliation_slots') for a in authors):continue
                sources=[json.loads(r[0]) for r in db.execute(
                    "SELECT payload FROM source_metadata WHERE doi=? AND (kind='jats' OR "
                    "(kind='publisher' AND json_extract(payload,'$.native_country_resolution_certified')=1))",
                    (paper['doi'],))]
                sources=[index_authors(s) for s in sources if signature_match(paper,s,journals[paper['source_id']])]
                receipts=set();new=[];paper_delta=0
                for author in authors:
                    if scope_positions is not None and author['position'] not in scope_positions.get(paper['work_id'], set()):
                        new.append(author)
                        continue
                    updated,evidence=resolve(author,sources,names);new.append(updated)
                    if not evidence:continue
                    receipts.update(evidence);paper_delta+=1
                    entity=paper['work_id']+':'+str(author['position'])
                    db.execute("UPDATE tasks SET status='resolved',receipt=?,reason=? WHERE kind='country_conflict' AND entity_key IN (?,?)",
                        (json.dumps(evidence),'Complete agreeing publisher addresses supersede lower-authority assignment; before values preserved',entity,entity+':secondary'))
                if not paper_delta:continue
                before=paper['authors_json'];paper['authors_json']=json.dumps(new,ensure_ascii=False);changed=True
                changed_ids.add(paper['work_id'])
                stats['authors_resolved']+=paper_delta;stats['papers_changed']+=1
                db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                    (paper['work_id'],'authors_json',before,paper['authors_json'],json.dumps(sorted(receipts)),'All matched primary publication author-address country sets agree, all explicit references resolve; only lower-authority assignments replaced'))
            if changed:
                tmp=path.with_suffix('.parquet.tmp');pq.write_table(pa.Table.from_pylist(rows,schema=table.schema),tmp,compression='zstd');tmp.replace(path)
                metadata_task_delta.record(db,path.name,before_signature,
                    metadata_task_delta.signature(path,decision_path),changed_ids,True)
            db.commit();atomic_json(status_path,{'status':'running','shard':path.name,**stats});print(path.name,dict(stats),flush=True)
        db.close()
    atomic_json(status_path,{'status':'applied_derived_refresh_required',**stats})


if __name__=='__main__':run()
