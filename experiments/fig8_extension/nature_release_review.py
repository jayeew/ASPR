"""Close every requested record using publisher links and exact identity evidence."""
from __future__ import annotations

import copy
import csv
import json
import re
from collections import defaultdict
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from . import nature_clinical_review as prior
from .nature_backfill_registry import normalized
from .nature_backfill_resolve import (
    address_country,
    author_addresses,
    clean_address,
    country_names,
)
from .nature_boundary_rank import NONPERSONAL
from .nature_institution_apply import registry_entity
from .nature_institution_review import name_queries
from .nature_release_prepare import FIELD, INPUT, OUT, ROOT, SLOTS, publisher
from .nature_weighted_review import local_entity
from .stage0 import write_csv, write_json

COUNTRIES=country_names()
COUNTRIES.update({normalized(k):v for k,v in {'Great Britain':'GB','Turkey':'TR','St Helena Island':'SH'}.items()})
RETAIN={
 ('10.1038/s41586-024-08206-1',7):['China University of Petroleum, East China','State Key Laboratory of Heavy Oil'],
 ('10.1038/s41586-025-08853-y',0):['Harvard University'],
 ('10.1038/s41586-023-06717-x',86):['KU Leuven'],
 ('10.1038/s41586-023-06717-x',87):['Columbia University Irving Medical Center'],
 ('10.1038/s41586-025-08657-0',5):['Memorial Sloan Kettering Cancer Center'],
 ('10.1038/s41586-025-08657-0',8):['Memorial Sloan Kettering Cancer Center'],
 ('10.1038/s41586-024-08542-2',16):['University of California San Diego'],
 ('10.1038/s41586-025-09470-5',5):['University of Sussex'],
 ('10.1038/s41586-025-09588-6',14):['University of California San Diego'],
 ('10.1038/s41586-025-09588-6',17):['Columbia University','New York State Psychiatric Institute'],
}
OFFICIAL_RULES=[
 (r'^Lockheed Martin Space, Littleton', 'Lockheed Martin Space, Littleton', 'company','US','https://www.lockheedmartin.com/en-us/news/features/2021/mission-support-area---drivers-of-discovery.html'),
 (r'^Space Exploration and Mission Operations, Leidos, Houston', 'Leidos Space Exploration and Mission Operations, Houston','company','US','https://www.leidos.com/insights/recapping-big-weekend-leidos-space-city'),
 (r'^National Biodefense Analysis and Countermeasures Center', 'National Biodefense Analysis and Countermeasures Center','research_institute','US','https://www.cityoffrederickmd.gov/m/faq?cat=27'),
 (r'^Sony AI, Zurich', 'Sony AI Zurich','company','CH','https://ai.sony/about/labs-offices'),
 (r'^Seer Inc\., Redwood City', 'Seer, Inc.','company','US','https://seer.bio/locations-contact-us/'),
 (r'^CISTIM Leuven vzw, Leuven', 'CISTIM Leuven vzw','research_institute','BE','https://www.cistim.be/'),
 (r'^Zetta AI, (?:LLC, )?Sherrill', 'Zetta AI LLC','company','US','https://www.sbir.gov/portfolio/1414107'),
]


def key(text: str) -> str:
    return normalized(re.sub(r'https?://\S+','',clean_address(text))).strip()


def link(author: dict[str, Any], pub: dict[str, Any]) -> tuple[list[str], str, bool]:
    addresses, method=author_addresses(author,pub)
    # Validate the individual link rather than rejecting all authors because one consortium has no address.
    ok=method in ['publisher_exact_author_name','publisher_exact_affiliation_author_list'] and bool(addresses)
    return addresses, method, ok


def address_match(slot: dict[str, Any], addresses: list[str]) -> list[str]:
    name=key(slot['display_name']).split(' email ')[0]
    head=key(clean_address(slot['display_name']).split(',')[0])
    return [a for a in addresses if (name and name in key(a)) or
            (len(head)>=8 and (' '+head+' ') in (' '+key(a)+' '))]


def registry_index() -> tuple[dict[str, Any], dict[str, set[str]]]:
    records: dict[str, Any]={}
    paths=list((ROOT/'nature_institution_review'/'ror_queries').glob('*.json'))
    paths+=list((ROOT/'nature_university_boundary_review'/'ror_records').glob('*.json'))
    paths+=list((OUT/'ror_queries').glob('*.json'))
    for path in paths:
        d=json.loads(path.read_text())
        orgs=d.get('response',{}).get('items',[]) if 'response' in d else [d.get('organization',{})]
        for item in orgs:
            org=item.get('organization',item)
            if org.get('id') and org.get('names') and org.get('status')=='active':
                records.setdefault(org['id'],{'organization':org,'receipt':str(path.resolve())})
    aliases: dict[str,set[str]]=defaultdict(set)
    for ident,data in records.items():
        for name in data['organization']['names']:
            alias=key(name['value'])
            if len(alias)>=6 and 'acronym' not in name['types']:
                aliases[alias].add(ident)
    return records,aliases


def candidate(address: str, records: dict[str,Any], aliases: dict[str,set[str]],
              existing: dict[str,Any], country: str | None) -> tuple[dict[str,Any] | None,dict[str,Any]]:
    parts=[p.strip() for p in clean_address(address).split(',')]
    found=set()
    tail_n=3 if country in {'US','CA','AU'} and len(parts)>=4 else 2
    for part in parts[:-tail_n] if len(parts)>tail_n else parts[:1]:
        found.update(aliases.get(key(part),set()))
    accepted=[]
    for ident in sorted(found):
        org=records[ident]['organization']
        locations=org.get('locations',[])
        codes={x['geonames_details']['country_code'] for x in locations}
        cities={key(x['geonames_details'].get('name','')) for x in locations}-{''}
        if country and country in codes and cities.intersection({key(p) for p in parts[-4:]}):
            accepted.append(ident)
    evidence={'registry_exact_name_candidates':sorted(found),'accepted_country_city_candidates':accepted,
              'registry_receipts':[records[i]['receipt'] for i in sorted(found)]}
    if len(accepted)!=1:
        return None,evidence
    record=records[accepted[0]]
    ent=registry_entity(record['organization'],existing,country)
    ent['release_identity_source']=record['receipt']
    return ent,evidence


def priority_fix(slot: dict[str,Any], addresses: list[str], valid: bool,
                 existing_names: dict[str,Any]) -> tuple[dict[str,Any] | None,str]:
    if not valid:
        return slot,'unresolved_author_link'
    name=slot['display_name']
    if name=='Columbia University Irving Medical Center':
        if any(re.search(r'columbia university.*(?:medical cent|medical campus|vagelos)|vagelos.*columbia university',ad,re.IGNORECASE) for ad in addresses):
            result=copy.deepcopy(slot)
            result['clinical_rollup']={'kind':'academic','target_id':'https://openalex.org/I78577930','official_url':prior.CU}
            return result,'confirmed_columbia_academic_unit'
        if any('columbia university' in ad.lower() for ad in addresses):
            return copy.deepcopy(existing_names['Columbia University']),'corrected_to_explicit_columbia_signature'
        return slot,'unresolved_publisher_identity_conflict'
    rule=prior.rollup_candidate(name)
    if rule and any(re.search(rule[1],ad,re.IGNORECASE) for ad in addresses):
        result=copy.deepcopy(slot)
        result['clinical_rollup']={'kind':rule[2],'target_id':'https://openalex.org/'+rule[3],'official_url':rule[4]}
        return result,'confirmed_direct_unit'
    if name in prior.HOSPITALS and any(key(name) in key(ad) for ad in addresses):
        result=copy.deepcopy(slot);result.pop('clinical_link_unresolved',None)
        return result,'confirmed_existing_clinical_link'
    return slot,'unresolved_joint_branch' if name in ['Center for Life Sciences','Collaborative Innovation Center of Quantum Matter','Innovative Genomics Institute','QB3','State Key Laboratory of Membrane Biology'] else 'unresolved_publisher_identity_conflict'


def main() -> None:
    rows=pq.read_table(INPUT).to_pylist()
    scope=json.loads((OUT/'scope.json').read_text());selected=set(scope['union'])
    records,aliases=registry_index()
    known={s['review_entity_id']:s for r in rows for a in json.loads(r[prior.FIELD]) for s in a[prior.SLOTS] if s['identity_status']!='identity_unresolved'}
    existing={s['ror']:s for s in known.values() if s.get('ror')}
    existing_names={s['display_name']:s for s in known.values()}
    name_index: dict[str,set[str]]=defaultdict(set)
    for ident,ent in known.items():
        if len(key(ent['display_name']))>=8:
            name_index[key(ent['display_name'])].add(ident)
    queue=[{**r,'queue':f} for f in ['unresolved_branches.csv','additional_rollup_unresolved.csv']
        for r in csv.DictReader((prior.OUT/f).open(encoding='utf-8-sig'))]
    qkeys={(r['doi'],int(r['author_position']),r['entity_id']) for r in queue}
    query_log={r['query']:r for r in json.loads((OUT/'registry_query_log.json').read_text())}
    audit,changes,authors,paper_status=[],[],[],[]
    for row in rows:
        aa=json.loads(row[prior.FIELD]);pub=publisher(row['doi']) if row['doi'] in selected else None
        for pos,a in enumerate(aa):
            a[SLOTS]=copy.deepcopy(a[prior.SLOTS])
            if not pub:
                continue
            addresses,method,valid=link(a,pub)
            before_known=any(s['identity_status']!='identity_unresolved' for s in a[SLOTS])
            logs=[]
            for idx,s in enumerate(a[SLOTS]):
                original=copy.deepcopy(s)
                priority=(row['doi'],pos,s['review_entity_id']) in qkeys
                unknown=s['identity_status']=='identity_unresolved'
                no_country=not(s.get('country_code') or s.get('site_country_codes'))
                curated=(row['doi'],pos) in RETAIN
                if not(unknown or no_country or priority or curated):
                    continue
                ev={'doi':row['doi'],'author_position':pos,'author_name':a['author']['display_name'],
                    'slot_position':idx,'original_entity_id':s['review_entity_id'],'original_name':s['display_name'],
                    'publisher_url':pub['source_url'],'publisher_receipt':pub['source_receipt'],
                    'publisher_record':str(OUT/'publisher_records'/(row['doi'].split('/')[1]+'.json')),
                    'publisher_method':method,'publisher_addresses':addresses,'priority':priority,
                    'initial_identity_missing':unknown,'initial_country_missing':no_country,
                    'final_status':'unresolved_identity_evidence'}
                ev['registry_query_evidence']=[query_log[t] for t in name_queries(s['display_name']) if t in query_log] if unknown else []
                matched=address_match(s,addresses) if valid else []
                ev['matching_addresses']=matched
                # Two specifically inspected contamination/present-address cases, backed by complete author links.
                remove=curated and s['display_name'] not in RETAIN[(row['doi'],pos)]
                if remove and valid:
                    ev['final_status']='removed_unsupported_or_present_only_slot'
                    a[SLOTS][idx]=None
                elif curated and valid and not priority:
                    ev['final_status']='confirmed_publisher_retained_identity'
                elif priority:
                    s,status=priority_fix(s,addresses,valid,existing_names)
                    a[SLOTS][idx]=s
                    ev['final_status']=status
                # An exact, unique, same-country named identity already verified in the cohort.
                if not remove and unknown and matched:
                    ids=set()
                    components=clean_address(original['display_name']).split(',')
                    tail_n=3 if original.get('country_code') in {'US','CA','AU'} and len(components)>=4 else 2
                    for part in components[:-tail_n] if len(components)>tail_n else components[:1]:
                        ids.update(name_index.get(key(part),set()))
                    oldcountry=original.get('country_code')
                    ids={i for i in ids if oldcountry and known[i].get('country_code')==oldcountry}
                    # Same-country university names do not establish the campus in a contradictory city.
                    ids={i for i in ids if known[i]['review_entity_type'] not in ['university','education_proxy','education_other_or_unreviewed']}
                    ev['existing_exact_candidates']=sorted(ids)
                    if len(ids)==1:
                        s=copy.deepcopy(known[next(iter(ids))])
                        # Never transfer author-specific branch decisions from another author.
                        for k in ['clinical_rollup','clinical_review_evidence','clinical_link_unresolved','cutoff_branch','branch_review','release_review_evidence']:
                            s.pop(k,None)
                        ev['final_status']='resolved_existing_exact_identity'
                    else:
                        replacement,reg=candidate(original['display_name'],records,aliases,existing,oldcountry)
                        ev.update(reg)
                        if replacement:
                            s=replacement;ev['final_status']='resolved_registry_exact_name_country_city'
                    if s is not None:
                        a[SLOTS][idx]=s
                    rule=next((r for r in OFFICIAL_RULES if re.search(r[0],original['display_name'],re.IGNORECASE)),None)
                    if rule:
                        _,canon,kind,code,url=rule
                        s=local_entity(canon,kind,code,{'source_url':url})
                        a[SLOTS][idx]=s
                        ev.update(final_status='resolved_official_identity',official_url=url)
                        if 'Zetta AI' in canon and ', NJ,' in original['display_name']:
                            ev['remaining_location_caveat']='Publisher says NJ; federal firm record says NY. Country US agrees; raw state retained, not silently corrected.'
                if not remove and no_country and matched:
                    codes={address_country(ad,COUNTRIES) for ad in matched}-{None}
                    if len(codes)==1:
                        s['country_code']=next(iter(codes));ev['country_status']='resolved_publisher_explicit_country'
                        if ev['final_status']=='unresolved_identity_evidence':
                            ev['final_status']='country_resolved_identity_unresolved' if unknown else 'country_resolved'
                    if original['display_name']=='Deep Light SA':
                        s['country_code']='CH'
                        ev.update(country_status='resolved_official_named_legal_entity',official_url='https://deeplight.ai/contact-deeplight/',final_status='country_resolved')
                    if original['display_name'].startswith('Indonesian Association of Archaeologists'):
                        url='https://iaaipusat.wordpress.com/about/'
                        s=local_entity('Indonesian Association of Archaeologists, Sulawesi branch','professional_association','ID',{'source_url':url})
                        a[SLOTS][idx]=s
                        ev.update(country_status='resolved_official_regional_branch',official_url=url,final_status='resolved_official_regional_identity_country')
                if not remove and no_country and valid and original['display_name'].startswith('Centre for Atmospheric Research'):
                    ads=[ad for ad in addresses if 'Atmospheric Research' in ad and ad.endswith('Slovenia')]
                    if ads:
                        s['country_code']='SI';ev['country_status']='resolved_publisher_explicit_country';ev['final_status']='country_resolved';ev['matching_addresses']=ads
                if not remove and a.get('author_record_kind') in NONPERSONAL and not addresses:
                    ev['final_status']='unresolved_nonpersonal_no_individual_affiliation'
                elif not remove and not addresses:
                    ev['final_status']='unresolved_publisher_no_affiliation'
                elif not remove and not valid:
                    ev['final_status']='unresolved_author_link'
                elif not remove and unknown and not matched and not priority:
                    ev['final_status']='unresolved_raw_unit_not_in_primary_signature'
                new=a[SLOTS][idx]
                if new!=original:
                    if new is not None:
                        new['release_review_evidence']=copy.deepcopy(ev)
                    changes.append({**ev,'old_slot':original,'new_slot':copy.deepcopy(new)})
                ev['final_identity_missing']=new is not None and new['identity_status']=='identity_unresolved'
                ev['final_country_missing']=new is not None and not(new.get('country_code') or new.get('site_country_codes'))
                audit.append(ev);logs.append(ev)
            a[SLOTS]=[s for s in a[SLOTS] if s is not None]
            # Preserve observed-slot denominator for baseline. Canonical deduplication is an explicit sensitivity.
            assert a[SLOTS]
            after_known=any(s['identity_status']!='identity_unresolved' for s in a[SLOTS])
            authors.append({'doi':row['doi'],'author_position':pos,'author_name':a['author']['display_name'],
                'record_kind':a.get('author_record_kind'),'publisher_url':pub['source_url'],
                'publisher_receipt':pub['source_receipt'],'publisher_method':method,'publisher_addresses':addresses,
                'before_known':before_known,'after_known':after_known,
                'status':'resolved' if not before_known and after_known else 'unresolved' if not after_known else 'existing_identity_retained',
                'all_slots_known':all(s['identity_status']!='identity_unresolved' for s in a[SLOTS]),
                'all_slots_country':all(s.get('country_code') or s.get('site_country_codes') for s in a[SLOTS]),
                'slot_statuses':[e['final_status'] for e in logs]})
        row[FIELD]=json.dumps(aa,ensure_ascii=False)
        if pub:
            unresolved=sum(not any(s['identity_status']!='identity_unresolved' for s in a[SLOTS]) for a in aa)
            paper_status.append({'doi':row['doi'],'title':row['title'],'in_348':row['doi'] in scope['missing348'],
                'country_priority':row['doi'] in scope['country27'],'hierarchy_priority':row['doi'] in scope['priority48'],
                'publisher_url':pub['source_url'],'publisher_receipt':pub['source_receipt'],
                'final_status':'reviewed_with_explicit_unknowns' if unresolved else 'reviewed_all_authors_have_identity',
                'remaining_identity_authors':unresolved})
    pq.write_table(pa.Table.from_pylist(rows),OUT/'research_roster_release.parquet',compression='zstd')
    for name,data in [('slot_dispositions',audit),('changes',changes),('author_dispositions',authors),('paper_dispositions',paper_status)]:
        write_json(OUT/(name+'.json'),data);write_csv(OUT/(name+'.csv'),data)
    indexed={(r['doi'],r['author_position'],r['original_entity_id']):r for r in audit}
    final_queue=[{**r,'disposition':indexed[(r['doi'],int(r['author_position']),r['entity_id'])]} for r in queue]
    assert len(final_queue)==123
    write_json(OUT/'priority123_dispositions.json',final_queue)
    write_csv(OUT/'priority123_dispositions.csv',[{**r,**r['disposition']} for r in final_queue])
    assert len(paper_status)==400 and sum(r['in_348'] for r in paper_status)==348
    write_json(OUT/'review_rules.json',{'individual_publisher_links':'Exact name or explicit affiliation author list; no fuzzy-name assignment',
        'identity':'Unique exact organization name with known site country; registry additionally requires exact city',
        'hierarchy':'Existing clinical policy plus explicitly signed Columbia academic units; joint membership does not prove a branch',
        'scope':'All 400 scoped papers scanned; every originally missing/priority slot has a terminal disposition; unresolved is retained, not fabricated.'})
    print(f'Reviewed {len(paper_status)} papers, {len(audit)} targeted slots; changes {len(changes)}',flush=True)


if __name__=='__main__':
    main()
