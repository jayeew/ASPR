"""Evidence-preserving UCLA/Columbia cutoff review and missing identity repairs."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from . import nature_boundary_sources as sources
from . import nature_joint_branch_review as prior
from .nature_backfill_resolve import author_addresses
from .nature_boundary_review import matching_addresses
from .nature_joint_branch_rank import SensitivityHierarchy
from .nature_weighted_review import local_entity
from .stage0 import write_csv, write_json

PREV = prior.OUT
OUT = PREV.parent / 'nature_cutoff_review'
INPUT = PREV / 'research_roster_joint_review.parquet'
FIELD, SLOTS = 'authorships_cutoff_review_json', 'institution_cutoff_review_entities'
MEMBRANE = 'https://www.biomembrane.tsinghua.edu.cn/p/shiyanshijianjie.html'
PGEC = 'https://www.ars.usda.gov/pacific-west-area/albany-ca/pgec/docs/main/'
QB3 = 'https://qb3.org/about/'
IGI = 'https://innovativegenomics.org/about-us/'
BIOHUB = 'https://chanzuckerberg.com/newsroom/biohub-new-york-technologies-detect-disease/'
RULES = [
 (r'^Columbia Equine Hospital,', 'Columbia Equine Hospital', 'healthcare', 'US', 'https://www.columbiaequine.com/site/veterinarians-gresham'),
 (r'^Foundational AI Research \(FAIR\), Meta, Paris,', 'Meta FAIR Paris', 'company', 'FR', 'https://ai.meta.com/research/fair-paris/'),
 (r'^Psiphon Inc\., Toronto,', 'Psiphon Inc.', 'company', 'CA', 'https://psiphon.ca/en/about'),
 (r'^PROTEROS Biostructures GmbH,', 'Proteros biostructures GmbH', 'company', 'DE', 'https://www.proteros.com/terms-of-use'),
 (r'^Speech Graphics Ltd,', 'Speech Graphics Ltd', 'company', 'GB', 'https://www.speech-graphics.com/privacy-policy'),
 (r'^Autonomy Institute, London,', 'Autonomy Institute', 'research_institute', 'GB', 'https://autonomy.work/wp-content/uploads/2022/10/Consultant-JD-Autonomy.pdf'),
 (r'^Hamburger Sternwarte, Hamburg,', 'Hamburg Observatory', 'research_institute', 'DE', 'https://www.physik.uni-hamburg.de/hs/subsite---about-us.html'),
]
CENTERS = {'QB3': (r'qb3|california institute for quantitative biosciences', QB3),
           'Innovative Genomics Institute': (r'innovative genomics institute', IGI)}
HOSTS = {'berkeley': 'I95457486', 'san francisco': 'I180670191', 'santa cruz': 'I185103710',
         'davis': 'I84218800', 'san diego': 'I36258959'}


def publisher(doi: str) -> dict[str, Any]:
    suffix = doi.split('/')[1] + '.json'
    for folder in [OUT, PREV, prior.PREV, sources.PREV, sources.ROOT / 'nature_metadata_backfill']:
        path = folder / 'publisher_records' / suffix
        if path.exists():
            return json.loads(path.read_text())
    url = 'https://www.nature.com/articles/' + doi.split('/')[1]
    token = hashlib.sha256(url.encode()).hexdigest()
    folder = sources.BASE if (sources.BASE / 'publisher_raw' / (token + '.html.gz')).exists() else OUT
    result = sources.parse_page(sources.DirectPublisher(folder).get(url), doi)
    result.update(source_url=url, source_receipt=str(folder / 'publisher_raw' / (token + '.html.json')))
    write_json(OUT / 'publisher_records' / suffix, result)
    return result


def campus(address: str, name: str) -> str | None:
    pattern = CENTERS[name][0]
    text = address.lower()
    if not re.search(pattern, text):
        return None
    hosts = {v for k, v in HOSTS.items() if re.search(r'university of california,?\s+' + k, text)}
    return 'https://openalex.org/' + next(iter(hosts)) if len(hosts) == 1 else None


class CutoffHierarchy(SensitivityHierarchy):
    def manual_rules(self) -> dict[str, Any]:
        rules = super().manual_rules()
        for name, url in [('QB3', QB3), ('Plant Gene Expression Center', PGEC), ('Chan Zuckerberg Biohub New York', BIOHUB)]:
            entity = self.by_name(name)
            rules[entity['review_entity_id']] = {'action': 'joint',
                'target_ids': {'QB3': ['https://openalex.org/'+HOSTS[k] for k in ['berkeley','san francisco','santa cruz']],
                    'Plant Gene Expression Center': ['https://openalex.org/I95457486'],
                    'Chan Zuckerberg Biohub New York': ['https://openalex.org/I78577930','https://openalex.org/I83399316','https://openalex.org/I32971472']}[name],
                'divisor': 2 if name == 'Plant Gene Expression Center' else 3, 'source_url': url,
                'evidence_level': 'official_primary_override', 'note': 'Partnership, not exclusive university ownership; equal split is sensitivity only.'}
        matches = [e for e in self.entities.values() if e['display_name']=='Hamburg Observatory']
        if matches:
            rules[matches[0]['review_entity_id']] = {'action':'academic','target_ids':['https://openalex.org/I159176309'],
                'divisor':1,'source_url':RULES[-1][4],'evidence_level':'official_primary_override',
                'note':'University physics department observatory; retain source unit identity.'}
        return rules

    def targets(self, slot: dict[str, Any], scenario: str) -> dict[str, float]:
        if slot.get('cutoff_branch', {}).get('status') == 'verified_nonuniversity_branch':
            return {}
        if slot.get('cutoff_branch', {}).get('status') == 'publisher_assignment_conflict':
            return {}
        if slot.get('cutoff_branch', {}).get('target_id'):
            return {slot['cutoff_branch']['target_id']: 1.0}
        return super().targets(slot, scenario)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(INPUT).to_pylist()
    entities = {s['display_name']: s for r in rows for a in json.loads(r[prior.FIELD]) for s in a[prior.SLOTS]
                if s['identity_status'] != 'identity_unresolved'}
    oldbranches = json.loads((PREV/'branch_address_audit.json').read_text())
    oldqueue = {(r['doi'], int(r['author_position']), r['entity_id']) for r in oldbranches if r['status']=='unresolved_branch'}
    selected = {r[0] for r in oldqueue}
    for row in rows:
        if any(s['display_name'] in CENTERS or s['display_name']=='Plant Gene Expression Center' or
               any(re.search(rule[0], s['display_name'], re.IGNORECASE) for rule in RULES)
               for a in json.loads(row[prior.FIELD]) for s in a[prior.SLOTS]):
            selected.add(row['doi'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(publisher, sorted(selected)))
    urls = [r[4] for r in RULES] + [MEMBRANE, PGEC, QB3, IGI, BIOHUB]
    oldout = sources.OUT
    sources.OUT = OUT
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            receipts = list(pool.map(sources.capture, urls))
    finally:
        sources.OUT = oldout
    write_json(OUT/'source_receipts.json', receipts)
    changes, branches, attempts = [], [], []
    for row in rows:
        aa = json.loads(row[prior.FIELD])
        for pos, a in enumerate(aa):
            a[SLOTS] = copy.deepcopy(a[prior.SLOTS])
            for j, slot in enumerate(a[SLOTS]):
                key = (row['doi'], pos, slot['review_entity_id'])
                rule = next((r for r in RULES if slot['identity_status']=='identity_unresolved' and re.search(r[0],slot['display_name'],re.IGNORECASE)),None)
                if not rule and key not in oldqueue and slot['display_name'] not in CENTERS and slot['display_name']!='Plant Gene Expression Center':
                    continue
                pub = publisher(row['doi'])
                addresses, method = author_addresses(a,pub)
                ev = {'doi':row['doi'],'author_position':pos,'author':a['author']['display_name'],
                      'entity_id':slot['review_entity_id'],'entity':slot['display_name'],'addresses':addresses,
                      'publisher_method':method,'publisher_url':pub['source_url'],
                      'weight':1/len(aa)/len(a[SLOTS])}
                if rule:
                    pattern,name,kind,country,url = rule
                    ent = copy.deepcopy(entities.get(name) or local_entity(name,kind,country,{'source_url':url}))
                    matching, _linking = matching_addresses(a,pub,slot['display_name'],{'pattern':pattern,'entity':ent})
                    attempts.append({**ev,'candidate':name,'matched_addresses':matching,'official_url':url})
                    if matching:
                        ent.pop('site_country_codes',None)
                        ent['country_code'] = slot.get('country_code') or country
                        if slot.get('site_country_codes'):
                            ent['site_country_codes'] = slot['site_country_codes']
                        ent['cutoff_identity_evidence'] = {**ev,'matched_addresses':matching,'official_url':url}
                        a[SLOTS][j] = ent
                        changes.append({**ev,'new_entity':ent,'old_entity':slot})
                    continue
                status, target, url = 'unresolved_branch', None, None
                if method.startswith('publisher_'):
                    if key in oldqueue:
                        short = slot['review_entity_id'].split('/')[-1]
                        token = prior.JOINT[short][0].replace('center','cent(?:er|re)')
                        relevant = [ad for ad in addresses if re.search(token,ad,re.IGNORECASE)]
                        if short=='I4391768054' and any('Institute of Zoology, Chinese Academy of Sciences' in ad for ad in relevant):
                            status,url = 'verified_nonuniversity_branch',MEMBRANE
                        elif not relevant:
                            status = 'publisher_assignment_conflict'
                    elif slot['display_name']=='Plant Gene Expression Center':
                        if any('Plant Gene Expression Center' in ad and 'USDA' in ad for ad in addresses):
                            status,url = 'verified_nonuniversity_branch',PGEC
                    else:
                        targets = {campus(ad,slot['display_name']) for ad in addresses} - {None}
                        url = CENTERS[slot['display_name']][1]
                        if len(targets)==1:
                            target,status = next(iter(targets)),'confirmed_branch'
                ev.update(status=status,target_id=target,official_url=url,from_previous91=key in oldqueue)
                if status!='unresolved_branch':
                    slot['cutoff_branch'] = ev
                branches.append(ev)
        row[FIELD]=json.dumps(aa,ensure_ascii=False)
    pq.write_table(pa.Table.from_pylist(rows),OUT/'research_roster_cutoff_review.parquet',compression='zstd')
    for name, data in [('identity_changes',changes),('identity_attempts',attempts),('branch_audit',branches)]:
        write_json(OUT/(name+'.json'),data)
        write_csv(OUT/(name+'.csv'),data)
    write_json(OUT/'identity_rules.json',RULES)
    print('identity repairs',len(changes),'branch decisions',len(branches),flush=True)


if __name__=='__main__':
    main()
