"""Publisher-gated clinical and boundary rollups; previous layers stay immutable."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from . import nature_boundary_rank as rank
from . import nature_boundary_sources as sources
from . import nature_conflict_review as prior
from .nature_backfill_resolve import author_addresses
from .nature_boundary_hierarchy import HOSPITALS
from .nature_cutoff_review import CutoffHierarchy
from .nature_weighted_review import local_entity
from .stage0 import write_csv, write_json

PREV = prior.OUT
OUT = PREV.parent / 'nature_clinical_review'
INPUT = PREV / 'research_roster_conflict_review.parquet'
FIELD, SLOTS = 'authorships_clinical_review_json', 'institution_clinical_review_entities'
CU = 'https://www.cuimc.columbia.edu/about-us'
PKU = 'https://www.bjmu.edu.cn/ylfw/zsfsyy/index.htm'
LAB = 'https://www.puh3.net.cn/szyxzx/ksjj.htm'
IHN = 'https://www.ihn.cuimc.columbia.edu/about-us'
QB = 'https://qb3.berkeley.edu/category/news/core-research-facilities/'
QBI = 'https://qbi.ucsf.edu/about-qbi'
HUM = 'https://www.vir.bio/privacy-notice/'
INO = 'https://www.ino.cnr.it/?page_id=9664'
LABNAME = 'Beijing Key Laboratory of Reproductive Endocrinology and Assisted Reproductive Technology'
RULES = [
    ('Columbia University Irving Medical Center', r'columbia university (?:irving )?medical cent(?:er|re)', 'academic', 'I78577930', CU),
    (LABNAME, re.escape(LABNAME), 'health_system', 'I20231570', LAB),
    ('Institute of Human Nutrition', r'institute of human nutrition', 'academic', 'I78577930', IHN),
]
IDENTITIES = [
    (r'^Humabs BioMed, Bellinzona', 'Humabs BioMed SA', 'company', 'CH', HUM, r'humabs biomed'),
    (r'^CNR-INO, Sede di Pisa', 'CNR-INO Pisa Unit', 'research_institute', 'IT', INO, r'cnr.ino.*pisa'),
    (r'^Institute of Human Nutrition, New York', 'Institute of Human Nutrition', 'research_institute', 'US', IHN, r'institute of human nutrition'),
]


class ClinicalHierarchy(CutoffHierarchy):
    def targets(self, slot: dict[str, Any], scenario: str) -> dict[str, float]:
        if scenario == 'publisher_verified_clinical' and slot.get('clinical_link_unresolved'):
            return {}
        evidence = slot.get('clinical_rollup')
        if evidence:
            if evidence['kind'] == 'health_system' and scenario == 'academic_only':
                return {}
            if (scenario == 'exclude_all_reviewed_medical' and evidence['kind'] in ['academic', 'health_system']
                    and evidence['official_url'] in [CU, LAB, IHN]):
                return {}
            return {evidence['target_id']: 1.0}
        if scenario == 'exclude_all_reviewed_medical' and slot['display_name'] in HOSPITALS:
            return {}
        return super().targets(slot, scenario)


def eligible(author: dict[str, Any], pub: dict[str, Any]) -> tuple[list[str], bool, str]:
    addresses, method = author_addresses(author, pub)
    valid = pub['author_affiliation_parse_complete'] and method == 'publisher_exact_author_name'
    return addresses, bool(valid), method


def rollup_candidate(name: str) -> tuple[str, str, str, str, str] | None:
    return next((r for r in RULES if name == r[0] or name.startswith(r[0]+',')), None)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    input_hash = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    rows = pq.read_table(INPUT).to_pylist()
    queue = list(csv.DictReader((PREV/'unresolved_branches.csv').open(encoding='utf-8-sig')))
    queuekeys = {(r['doi'], int(r['author_position']), r['entity_id']) for r in queue}
    selected = {r['doi'] for r in queue}
    for row in rows:
        for author in json.loads(row[prior.FIELD]):
            if any(rollup_candidate(s['display_name']) or s['display_name'] in HOSPITALS or
                   any(re.search(r[0], s['display_name'], re.IGNORECASE) for r in IDENTITIES) for s in author[prior.SLOTS]):
                selected.add(row['doi'])
    # The prior parser is reused, but all reparsed receipts are saved in this new stage.
    old = prior.OUT
    prior.OUT = OUT
    with ThreadPoolExecutor(max_workers=4) as pool:
        pubs = dict(zip(sorted(selected), pool.map(prior.publisher, sorted(selected)), strict=True))
    prior.OUT = old
    old = sources.OUT
    sources.OUT = OUT
    urls = [CU, PKU, LAB, IHN, QB, QBI, HUM, INO,
        'https://facultyhandbook.columbia.edu/content/chronology',
        'https://www.cuimc.columbia.edu/news/irving-familys-700-million-bequest-columbia-and-nyp-sets-stage-dramatic-advances-cancer',
        'https://locator.lacounty.gov/lac/Location/3182121/harbor-ucla-medical-center',
        'https://sklffp.puh3.net.cn/info/1024/7382.htm',
        'https://www.columbia.edu/itc/hs/medical/nutrition/about/about.html',
        'https://www.sec.gov/Archives/edgar/data/1706431/000095017023005118/vir-ex10_17.htm']
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(sources.capture, urls))
    sources.OUT = old
    write_json(OUT/'source_receipts.json', receipts)
    audit, changes, branches, coverage = [], [], [], []
    for row in rows:
        aa = json.loads(row[prior.FIELD])
        pub = pubs.get(row['doi'])
        for pos, author in enumerate(aa):
            author[SLOTS] = copy.deepcopy(author[prior.SLOTS])
            addresses, valid, method = eligible(author, pub) if pub else ([], False, 'not_selected')
            for idx, slot in enumerate(author[SLOTS]):
                oldslot = copy.deepcopy(slot)
                name = slot['display_name']
                identity = next((r for r in IDENTITIES if slot['identity_status']=='identity_unresolved' and re.search(r[0],name,re.IGNORECASE)), None)
                candidate = rollup_candidate(name)
                branch = (row['doi'], pos, slot['review_entity_id']) in queuekeys
                if not (identity or candidate or branch or name in HOSPITALS):
                    continue
                ev = {'doi':row['doi'], 'author_position':pos, 'author':author['author']['display_name'],
                    'entity_id':slot['review_entity_id'], 'entity':name, 'weight':1/len(aa)/len(author[SLOTS]),
                    'publisher_addresses':addresses, 'publisher_method':method, 'complete_unique_match':valid,
                    'publisher_url':pub['source_url'], 'publisher_receipt':pub['source_receipt'],
                    'status':'unresolved_publisher_or_branch', 'official_url':candidate[4] if candidate else ''}
                if identity:
                    pattern, canon, kind, country, url, address_pattern = identity
                    ev['official_url'] = url
                    if valid and any(re.search(address_pattern, ad, re.IGNORECASE) for ad in addresses):
                        slot = local_entity(canon, kind, country, {'source_url':url})
                        slot['country_code'] = oldslot.get('country_code') or country
                        if oldslot.get('site_country_codes'):
                            slot['site_country_codes'] = oldslot['site_country_codes']
                        ev['status'] = 'identity_confirmed'
                        candidate = rollup_candidate(canon)
                if candidate:
                    _, pattern, kind, target, url = candidate
                    matches = [ad for ad in addresses if re.search(pattern,ad,re.IGNORECASE) and not re.search(r'new.?york.presbyterian',ad,re.IGNORECASE)]
                    if valid and matches:
                        ev.update(status='rollup_confirmed', kind=kind, target_id='https://openalex.org/'+target,
                                  official_url=url, matching_addresses=matches)
                        slot['clinical_rollup'] = copy.deepcopy(ev)
                if branch and name == 'QB3' and valid:
                    # City is usable only when the paper names the center and the same author's university corroborates it.
                    qb = any(re.search(r'(quantitative biosciences.*(?:qb3|berkeley)|qb3.*berkeley)',ad,re.IGNORECASE) for ad in addresses)
                    host = any(re.search(r'university of california(?: at|,)?\s+berkeley',ad,re.IGNORECASE) for ad in addresses)
                    qbi = any('quantitative biosciences institute' in ad.lower() and 'san francisco' in ad.lower() for ad in addresses)
                    if (qb and host) or qbi:
                        ev.update(status='branch_confirmed', kind='academic', target_id='https://openalex.org/'+('I180670191' if qbi else 'I95457486'),official_url=QBI if qbi else QB)
                        slot['clinical_rollup'] = copy.deepcopy(ev)
                if name in HOSPITALS:
                    ev.update(official_url=PKU,status='existing_clinical_rule_reconfirmed' if valid and any(prior.normalized(name) in prior.normalized(ad) for ad in addresses) else 'existing_rule_publisher_link_unresolved')
                    if ev['status'] == 'existing_rule_publisher_link_unresolved':
                        slot['clinical_link_unresolved'] = copy.deepcopy(ev)
                if slot != oldslot:
                    slot['clinical_review_evidence'] = copy.deepcopy(ev)
                    author[SLOTS][idx] = slot
                    changes.append({**ev,'old_slot':oldslot,'new_slot':slot})
                audit.append(ev)
                if branch:
                    branches.append(ev)
            before = any(s['identity_status']!='identity_unresolved' for s in author[prior.SLOTS])
            after = any(s['identity_status']!='identity_unresolved' for s in author[SLOTS])
            coverage.append({'doi':row['doi'],'author_position':pos,'before':before,'after':after,
                'all_slots_known':all(s['identity_status']!='identity_unresolved' for s in author[SLOTS]),
                'all_slots_country':all(s.get('site_country_codes') or s.get('country_code') for s in author[SLOTS])})
        row[FIELD] = json.dumps(aa,ensure_ascii=False)
    pq.write_table(pa.Table.from_pylist(rows), OUT/'research_roster_clinical_review.parquet', compression='zstd')
    for name, data in [('slot_audit',audit),('changes',changes),('branch_audit',branches),('author_coverage',coverage)]:
        write_json(OUT/(name+'.json'), data)
        write_csv(OUT/(name+'.csv'), data)
    closed = {(r['doi'],r['author_position'],r['entity_id']) for r in branches if r['status']=='branch_confirmed'}
    write_csv(OUT/'unresolved_branches.csv',[r for r in queue if (r['doi'],int(r['author_position']),r['entity_id']) not in closed])
    write_csv(OUT/'additional_rollup_unresolved.csv',[r for r in audit if r['status'] in ['unresolved_publisher_or_branch','existing_rule_publisher_link_unresolved'] and (r['doi'],r['author_position'],r['entity_id']) not in queuekeys])
    remaining = {r['doi'] for r in coverage if not r['after']}
    write_csv(OUT/'remaining_papers.csv',[{'doi':r['doi'],'title':r['title'],'year':r['publisher_year']} for r in rows if r['doi'] in remaining])
    write_csv(OUT/'remaining_identity_authors.csv',[r for r in coverage if not r['after']])
    summary = {'cohort_n':len(rows),'remaining_identity_papers':len(remaining),
        'remaining_identity_authors':sum(not r['after'] for r in coverage),
        'newly_resolved_authors':sum(not r['before'] and r['after'] for r in coverage),
        'remaining_branch_records':len(queue)-len(closed),'closed_branches':len(closed),
        'changed_slots':len(changes),'identity_changes':sum(c['old_slot']['identity_status']=='identity_unresolved' and c['new_slot']['identity_status']!='identity_unresolved' for c in changes),
        'reviewed_papers':len(selected),'reviewed_slots':len(audit),'input_sha256':input_hash}
    assert input_hash == hashlib.sha256(INPUT.read_bytes()).hexdigest()
    write_json(OUT/'review_summary.json',summary)
    analyze(rows)
    print(json.dumps(summary))


def analyze(rows: list[dict[str, Any]]) -> None:
    entities = {s['review_entity_id']:s for r in rows for a in json.loads(r[FIELD]) for s in a[SLOTS]}
    for row in rows:
        for author in json.loads(row[prior.FIELD]):
            for slot in author[prior.SLOTS]:
                entities.setdefault(slot['review_entity_id'], slot)
    h = ClinicalHierarchy(list(entities.values()))
    rank.FIELD, rank.SLOTS = FIELD, SLOTS
    scenarios = ['direct_units','academic_only','joint_equal_split','joint_two_partner_sensitivity',
        'university_deduplicated','exclude_nonpersonal','complete_case','official_only',
        'publisher_verified_clinical','exclude_all_reviewed_medical']
    results = {s:rank.calculate(rows,h,s) for s in scenarios}
    stability = {s:rank.bounds(d,h,s) for s,d in results.items()}
    for rec in stability.values():
        for kind in ['country','university']:
            names, ref = rec[kind]['members'], stability['direct_units'][kind]['members']
            rec[kind].update(same_members_as_direct=set(names)==set(ref),same_order_as_direct=names==ref,
                entered_vs_direct=[x for x in names if x not in ref],left_vs_direct=[x for x in ref if x not in names])
    write_json(OUT/'topk_stability.json',{'scenarios':stability,'limitations':'Bounds conditional on fixed slot mappings; unresolved hierarchy and author links excluded. Medical exclusion is a stress test, not a preferred definition.'})
    for category in ['countries','universities']:
        table=[]
        for scenario,result in results.items():
            for i,(key,value) in enumerate(sorted(((k,v) for k,v in result[category].items() if k!='unknown'),key=lambda x:-x[1])):
                table.append({'scenario':scenario,'rank':i+1,'id':key,'name':h.entities[key]['display_name'] if category=='universities' else key,'fractional_n':value})
        write_csv(OUT/(category+'_rank_scenarios.csv'),table)
    write_json(OUT/'fractional_summary.json',{s:d['summary'] for s,d in results.items()})
    write_json(OUT/'remaining_weighted_units.json',[{**u,'dois':sorted(u['dois']),'reasons':sorted(u['reasons'])} for u in sorted(results['direct_units']['units'].values(),key=lambda u:-u['weight'])])
    old = json.loads((PREV/'fractional_summary.json').read_text())['direct_units']
    assert abs(results['direct_units']['summary']['country_unknown']-old['country_unknown'])<1e-9
    rank.FIELD, rank.SLOTS = prior.FIELD, prior.SLOTS
    old_result = rank.calculate(rows, CutoffHierarchy(list(entities.values())), 'direct_units')
    rank.FIELD, rank.SLOTS = FIELD, SLOTS
    assert results['direct_units']['countries'] == old_result['countries']
    original = pq.read_table(INPUT).to_pylist()
    assert all(all(new[k] == value for k, value in old.items()) for old,new in zip(original,rows,strict=True))
    for row in rows:
        for a,b in zip(json.loads(row[prior.FIELD]),json.loads(row[FIELD]),strict=True):
            assert all(b[k] == value for k,value in a.items())
    for s in scenarios:
        if s not in ['complete_case','exclude_nonpersonal']:
            assert results[s]['countries']==results['direct_units']['countries']
    write_json(OUT/'validation.json',{'country_mass_conserved':True,'unknown_country_unchanged':True,
        'country_allocations_unchanged_vs_previous':True,'previous_fields_immutable':True,
        'country_invariant_across_hierarchy_scenarios':True,'unique_dois':len({r['doi'] for r in rows})==3683})


if __name__ == '__main__':
    main()
