"""Correct publisher-demonstrated author affiliation contamination; retain audit history."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import html

from . import nature_boundary_sources as sources
from . import nature_cutoff_review as prior
from .nature_backfill_resolve import author_addresses
from .nature_boundary_review import matching_addresses
from .nature_weighted_review import local_entity
from .stage0 import write_csv, write_json

PREV=prior.OUT
OUT=PREV.parent/'nature_publisher_conflict_review'
INPUT=PREV/'research_roster_cutoff_review.parquet'
FIELD,SLOTS='authorships_conflict_review_json','institution_conflict_review_entities'
CASES={
 ('s41586-023-05831-0',6):['Chinese Academy of Sciences','Institute of Biophysics','University of Chinese Academy of Sciences'],
 ('s41586-023-06525-3',18):['State Key Laboratory of Biocatalysis and Enzyme Engineering','Hubei University'],
 ('s41586-023-06525-3',22):['State Key Laboratory of Biocatalysis and Enzyme Engineering','Hubei University'],
 ('s41586-023-06783-1',1):['Chinese Academy of Sciences','Beijing Institute of Genomics','University of Chinese Academy of Sciences'],
 ('s41586-023-06783-1',29):['Beijing Hospital','Aging Biomarker Consortium, Beijing, China'],
 ('s41586-024-07152-2',8):['Hong Kong University of Science and Technology'],
 ('s41586-024-07286-3',3):['Westlake University'],
 ('s41586-024-07286-3',19):['Huazhong University of Science and Technology'],
 ('s41586-024-07438-5',11):['Chinese Academy of Sciences','Institute of Metal Research','Shenyang National Laboratory for Materials Science'],
 ('s41586-024-07438-5',25):['ShanghaiTech University','Liaoning Academy of Materials'],
 ('s41586-024-08159-5',34):['University of Toronto','Yunnan University'],
 ('s41586-025-09110-y',7):['Chinese Academy of Sciences','Institute of Zoology','State Key Laboratory of Integrated Management of Pest Insects and Rodents'],
}
RULES=[
 (r'^Xscape Photonics Inc\.,','Xscape Photonics, Inc.','US','https://www.xscapephotonics.com/about-us'),
 (r'^Unistellar, Marseille,','Unistellar','FR','https://shop.eu.unistellar.com/policies/legal-notice'),
 (r'^Revivicor, Blacksburg,','Revivicor, Inc.','US','https://www.govinfo.gov/content/pkg/FR-2021-04-01/pdf/2021-06704.pdf'),
]
OFFICIAL=[
 'https://eng.hubu.edu.cn/info/1030/2086.htm','https://cryo-em.hkust.edu.hk/',
 'https://english.phys.hust.edu.cn/index.htm','https://en-soe.westlake.edu.cn/OurSchool/',
 'https://www.ee.columbia.edu/news/xscape-photonics-secures-44-million-transform-ai-data-centers',
]


def publisher(doi: str) -> dict[str,Any]:
    suffix=doi.split('/')[1]+'.json'
    for folder in [OUT,PREV,prior.PREV,prior.prior.PREV,sources.PREV,sources.ROOT/'nature_metadata_backfill']:
        path=folder/'publisher_records'/suffix
        if path.exists():
            cached=json.loads(path.read_text())
            if any('excluded_present_address_ids' not in a for a in cached['authors']):
                raw=Path(cached['source_receipt']).with_suffix('.gz')
                parsed=sources.parse_page(gzip.decompress(raw.read_bytes()).decode('utf-8'),doi)
                parsed.update(source_url=cached['source_url'],source_receipt=cached['source_receipt'])
                write_json(OUT/'publisher_records'/suffix,parsed)
                return parsed
            return cached
    url='https://www.nature.com/articles/'+doi.split('/')[1]
    token=hashlib.sha256(url.encode()).hexdigest()
    folder=sources.BASE if (sources.BASE/'publisher_raw'/(token+'.html.gz')).exists() else OUT
    result=sources.parse_page(sources.DirectPublisher(folder).get(url),doi)
    result.update(source_url=url,source_receipt=str(folder/'publisher_raw'/(token+'.html.json')))
    write_json(OUT/'publisher_records'/suffix,result)
    return result


def normalized(value: str) -> str:
    return re.sub(r'[^a-z0-9]','',value.lower())


def corrected_slots(author: dict[str,Any], pub: dict[str,Any], keep: list[str]) -> tuple[list[dict[str,Any]],list[str]]:
    addresses,method=author_addresses(author,pub)
    if not pub['author_affiliation_parse_complete'] or method!='publisher_exact_author_name':
        raise ValueError('Correction requires complete publisher references and unique exact author match')
    slots=author[prior.SLOTS]
    selected=[copy.deepcopy(s) for s in slots if s['display_name'] in keep]
    if {s['display_name'] for s in selected}!=set(keep):
        raise ValueError('Curated retained identities missing')
    for name in keep:
        if not any(normalized(name) in normalized(ad) for ad in addresses):
            raise ValueError('Retained identity lacks explicit publisher address: '+name)
    return selected,addresses


def present_addresses(pub: dict[str,Any]) -> dict[str,str]:
    path=Path(pub['source_receipt']).with_suffix('.gz')
    root=html.fromstring(gzip.decompress(path.read_bytes()))
    return {node.get('id')[1:]: ' '.join(node.text_content().split())
            for node in root.xpath('//*[starts-with(@id,"nAff")]')}


def remove_present(author: dict[str,Any], pub: dict[str,Any], present: dict[str,str]) -> tuple[list[dict[str,Any]],dict[str,Any]]:
    addresses,method=author_addresses(author,pub)
    keys={normalized(author['author']['display_name']),normalized(author.get('raw_author_name',''))}-{''}
    matches=[n for n in pub['authors'] if normalized(n['name']) in keys]
    if method!='publisher_exact_author_name' or len(matches)!=1 or not pub['author_affiliation_parse_complete']:
        return author[SLOTS],{'status':'not_eligible_incomplete_or_ambiguous_link'}
    node=matches[0]
    secondary=[present[i] for i in node.get('excluded_present_address_ids',[]) if i in present]
    if not secondary:
        return author[SLOTS],{'status':'no_present_address'}
    removed=[s for s in author[SLOTS] if any(normalized(s['display_name']) in normalized(ad) for ad in secondary)
             and not any(normalized(s['display_name']) in normalized(ad) for ad in addresses)]
    retained=[s for s in author[SLOTS] if s not in removed]
    if not retained or not any(any(normalized(s['display_name']) in normalized(ad) for ad in addresses) for s in retained):
        return author[SLOTS],{'status':'unresolved_present_address','secondary':secondary,'primary':addresses}
    return retained,{'status':'removed_present_only' if removed else 'no_exact_present_only_slot',
        'secondary':secondary,'primary':addresses,'removed_slots':removed,'publisher_author_node':node}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    rows=pq.read_table(INPUT).to_pylist()
    selected={'10.1038/'+key[0] for key in CASES}
    boundary={r['doi'] for r in rows if any(s['display_name'] in ['Columbia University','University of California, Los Angeles']
        for a in json.loads(r[prior.FIELD]) for s in a[prior.SLOTS])}
    selected |= boundary
    for row in rows:
        if any(any(re.search(rule[0],s['display_name'],re.IGNORECASE) for rule in RULES)
               for a in json.loads(row[prior.FIELD]) for s in a[prior.SLOTS]):
            selected.add(row['doi'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(publisher,sorted(selected)))
    old=sources.OUT;sources.OUT=OUT
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            receipts=list(pool.map(sources.capture,OFFICIAL+[r[3] for r in RULES]))
    finally:
        sources.OUT=old
    write_json(OUT/'source_receipts.json',receipts)
    corrections,changes,attempts,present_audit=[],[],[],[]
    for row in rows:
        aa=json.loads(row[prior.FIELD])
        pub_boundary=publisher(row['doi']) if row['doi'] in boundary else None
        present=present_addresses(pub_boundary) if pub_boundary else {}
        for pos,a in enumerate(aa):
            a[SLOTS]=copy.deepcopy(a[prior.SLOTS])
            key=(row['doi'].split('/')[1],pos)
            if key in CASES:
                pub=publisher(row['doi'])
                a[SLOTS],addresses=corrected_slots(a,pub,CASES[key])
                removed=[s for s in a[prior.SLOTS] if s['display_name'] not in CASES[key]]
                ev={'doi':row['doi'],'author_position':pos,'author_name':a['author']['display_name'],
                    'publisher_addresses':addresses,'source_url':pub['source_url'],'source_receipt':pub['source_receipt'],
                    'publisher_author_nodes':[n for n in pub['authors'] if normalized(n['name']) in
                        {normalized(a['author']['display_name']),normalized(a.get('raw_author_name',''))}],
                    'before_slots':a[prior.SLOTS],'after_slots':a[SLOTS],'removed_slots':removed,
                    'policy':'Remove unsupported affiliations only from these reviewed authors; retain observed directly named units and explicit unknown units.'}
                corrections.append(ev)
                a['publisher_conflict_correction']=ev
            if pub_boundary:
                before_present=copy.deepcopy(a[SLOTS])
                a[SLOTS],finding=remove_present(a,pub_boundary,present)
                present_audit.append({'doi':row['doi'],'author_position':pos,'author_name':a['author']['display_name'],
                    'publisher_url':pub_boundary['source_url'],**finding})
                if finding['status']=='removed_present_only':
                    ev={'doi':row['doi'],'author_position':pos,'author_name':a['author']['display_name'],
                        'publisher_addresses':finding['primary'],'source_url':pub_boundary['source_url'],
                        'source_receipt':pub_boundary['source_receipt'],'before_slots':before_present,'after_slots':a[SLOTS],
                        'removed_slots':finding['removed_slots'],'policy':'Exclude publisher-labelled present addresses from research affiliation',
                        'present_address_evidence':finding}
                    corrections.append(ev)
                    a['present_address_correction']=ev
            for j,s in enumerate(a[SLOTS]):
                rule=next((r for r in RULES if s['identity_status']=='identity_unresolved' and re.search(r[0],s['display_name'],re.IGNORECASE)),None)
                if not rule:
                    continue
                pattern,name,country,url=rule
                ent=local_entity(name,'company',country,{'source_url':url})
                pub=publisher(row['doi'])
                matches,method=matching_addresses(a,pub,s['display_name'],{'pattern':pattern,'entity':ent})
                ev={'doi':row['doi'],'author_position':pos,'author_name':a['author']['display_name'],
                    'old_slot':s,'publisher_addresses':matches,'method':method,'publisher_url':pub['source_url'],'official_url':url}
                attempts.append(ev)
                if matches:
                    ent['country_code']=s.get('country_code') or country
                    if s.get('site_country_codes'):
                        ent['site_country_codes']=s['site_country_codes']
                    ent['conflict_pass_identity_evidence']=ev
                    a[SLOTS][j]=ent
                    changes.append({**ev,'new_slot':ent})
        row[FIELD]=json.dumps(aa,ensure_ascii=False)
    pq.write_table(pa.Table.from_pylist(rows),OUT/'research_roster_conflict_review.parquet',compression='zstd')
    for name,data in [('author_corrections',corrections),('identity_changes',changes),('identity_attempts',attempts)]:
        write_json(OUT/(name+'.json'),data);write_csv(OUT/(name+'.csv'),data)
    queue=list(__import__('csv').DictReader((PREV/'unresolved_branches.csv').open(encoding='utf-8-sig')))
    final_slots={(r['doi'],i):{s['review_entity_id'] for s in a[SLOTS]} for r in rows for i,a in enumerate(json.loads(r[FIELD]))}
    closed=[];remaining=[]
    for r in queue:
        if r['status']=='publisher_assignment_conflict' and (r['doi'].split('/')[1],int(r['author_position'])) in CASES:
            closed.append({**r,'resolution':'removed_unsupported_author_assignment'})
        elif r['entity_id'] not in final_slots[(r['doi'],int(r['author_position']))]:
            closed.append({**r,'resolution':'removed_explicit_present_address'})
        else:
            remaining.append(r)
    assert sum(r['resolution']=='removed_unsupported_author_assignment' for r in closed)==14
    assert len(closed)+len(remaining)==92
    write_csv(OUT/'resolved_14_conflicts.csv',closed)
    write_csv(OUT/'unresolved_branches.csv',remaining)
    write_json(OUT/'present_address_audit.json',present_audit)
    write_csv(OUT/'present_address_findings.csv',[r for r in present_audit if r['status']!='no_present_address'])
    write_json(OUT/'boundary_scope_dois.json',sorted(boundary))
    write_json(OUT/'review_rules.json',{'cases':[{'doi':k[0],'author_position':k[1],'retain':v} for k,v in CASES.items()],
                                      'identity_rules':RULES})
    print('corrected authors',len(corrections),'removed slots',sum(len(c['removed_slots']) for c in corrections),'identity repairs',len(changes))


if __name__=='__main__':
    main()
