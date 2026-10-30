"""Publication-address branch audit; immutable predecessor and per-slot evidence."""
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
from .nature_backfill_resolve import author_addresses
from .nature_boundary_hierarchy import Hierarchy
from .stage0 import write_csv, write_json

PREV = sources.OUT
OUT = PREV.parent / 'nature_joint_branch_review'
INPUT = PREV / 'research_roster_boundary_review.parquet'
FIELD, SLOTS = 'authorships_joint_review_json', 'institution_joint_review_entities'
JOINT = {
 'I4210127494': ('collaborative innovation center of quantum matter', 'https://110.phy.pku.edu.cn/info/1076/1222.htm'),
 'I4210160507': ('center for life sciences', 'https://cls.tsinghua.edu.cn/Introduction/'),
 'I4391768054': ('state key laboratory of membrane biology', 'https://www.biomembrane.tsinghua.edu.cn/p/shiyanshijianjie.html'),
 'I142207940': ('california nanosystems institute', 'https://cnsi.ucla.edu/about-us/'),
}
OFFICIAL = {
 'State Key Laboratory of Protein and Plant Gene Research': ('Peking University', 'https://www.bio.pku.edu.cn/homes/Index/news_cont/32/2991.html'),
 'State Key Laboratory of Artificial Microstructure and Mesoscopic Physics': ('Peking University', 'https://english.phy.pku.edu.cn/info/1259/3222.htm'),
 'State Key Laboratory of Nuclear Physics and Technology': ('Peking University', 'https://english.pku.edu.cn/labsCenters.html'),
 'Lamont-Doherty Earth Observatory': ('Columbia University', 'https://lamont.columbia.edu/news/celebrating-150-years-birth-thomas-w-lamont'),
 'Kennedy Institute of Rheumatology': ('University of Oxford', 'https://www.kennedy.ox.ac.uk/about/about-the-kennedy-institute'),
 'Centre for Human Genetics': ('University of Oxford', 'https://www.chg.ox.ac.uk/about-us/about-us'),
 'Space Sciences Laboratory': ('University of California, Berkeley', 'https://vcresearch.berkeley.edu/research-unit/space-sciences-laboratory'),
}
HOSTS = {'peking university': 'I20231570', 'tsinghua university': 'I99065089',
 'university of california, los angeles': 'I161318765', 'university of california los angeles': 'I161318765',
 'university of california, santa barbara': 'I154570441', 'university of california santa barbara': 'I154570441'}


def publisher(doi: str) -> dict[str, Any]:
    suffix = doi.split('/')[1] + '.json'
    for folder in [PREV, sources.PREV, sources.ROOT / 'nature_metadata_backfill', OUT]:
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



def branch(address: str, ident: str) -> str | None:
    """Require explicit separate host in the same address, never coaffiliation/email."""
    text = address.lower().split('@')[0].replace('centre for life sciences', 'center for life sciences')
    text = re.sub(r'(?:peking(?: university)?|tsinghua(?: university)?)\s*[-–—]\s*'
                  r'(?:peking(?: university)?|tsinghua(?: university)?)\s+(?:joint\s+)?center for life sciences',
                  'center for life sciences', text)
    token = JOINT[ident][0]
    if token not in text:
        return None
    # The label 'Peking-Tsinghua Center' itself cannot identify a branch.
    tail = text.split(token, 1)[1]
    hosts = {value for key, value in HOSTS.items() if key in tail}
    allowed = {'I161318765', 'I154570441'} if ident == 'I142207940' else {'I20231570', 'I99065089'}
    return next(iter(hosts)) if len(hosts) == 1 and hosts <= allowed else None


class ReviewedHierarchy(Hierarchy):
    def manual_rules(self) -> dict[str, Any]:
        rules = super().manual_rules()
        for name, (parent, url) in OFFICIAL.items():
            entity = self.by_name(name)
            rules[entity['review_entity_id']] = {'action': 'academic', 'target_ids': [self.by_name(parent)['review_entity_id']],
                'divisor': 1, 'source_url': url, 'evidence_level': 'official_primary_override',
                'note': 'Official university source confirms academic subunit; publisher address audit saved separately.'}
        rules['https://openalex.org/I4210127494'] = {'action': 'joint',
            'target_ids': ['https://openalex.org/I20231570', 'https://openalex.org/I99065089'], 'divisor': 3,
            'source_url': JOINT['I4210127494'][1], 'evidence_level': 'official_primary_override',
            'note': 'Historical three-partner sensitivity includes CAS Institute of Physics. Official 2023 page also anticipates two-school construction; timing unresolved; paired two-partner sensitivity required.'}
        return rules

    def targets(self, slot: dict[str, Any], scenario: str) -> dict[str, float]:
        if slot.get('branch_review') and scenario != 'unbranched_control':
            return {slot['branch_review']['target_id']: 1.0}
        return super().targets(slot, scenario)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(INPUT).to_pylist()
    selected = []
    for row in rows:
        authors = json.loads(row[sources.FIELD])
        if any(s['review_entity_id'].split('/')[-1] in JOINT or s['display_name'] in OFFICIAL
               for a in authors for s in a[sources.SLOTS]):
            selected.append(row['doi'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, _ in enumerate(pool.map(publisher, selected)):
            if i % 25 == 0:
                print('publisher', i, '/', len(selected), flush=True)
    urls = sorted({v[1] for v in JOINT.values()} | {v[1] for v in OFFICIAL.values()} |
                  {'https://www.cnsi.ucsb.edu/about'})
    sources.OUT = OUT
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            receipts = list(pool.map(sources.capture, urls))
    finally:
        sources.OUT = PREV
    write_json(OUT / 'primary_source_receipts.json', receipts)
    write_json(OUT / 'selected_publisher_dois.json', selected)
    print('prepared', len(selected), 'papers', flush=True)


def apply() -> None:
    rows = pq.read_table(INPUT).to_pylist()
    audits, rollups = [], []
    for row in rows:
        authors = json.loads(row[sources.FIELD])
        for pos, author in enumerate(authors):
            author[SLOTS] = copy.deepcopy(author[sources.SLOTS])
            for slot in author[SLOTS]:
                ident = slot['review_entity_id'].split('/')[-1]
                if ident not in JOINT and slot['display_name'] not in OFFICIAL:
                    continue
                pub = publisher(row['doi'])
                addresses, method = author_addresses(author, pub)
                receipt = {'doi': row['doi'], 'author_position': pos, 'author': author['author']['display_name'],
                    'entity_id': slot['review_entity_id'], 'entity': slot['display_name'], 'addresses': addresses,
                    'publisher_method': method, 'publisher_url': pub.get('source_url') or 'https://www.nature.com/articles/'+row['doi'].split('/')[1],
                    'weight': 1 / len(authors) / len(author[SLOTS])}
                if ident in JOINT:
                    matches = [(a, branch(a, ident)) for a in addresses] if method.startswith('publisher_') else []
                    targets = {t for _, t in matches if t}
                    if len(targets) == 1:
                        target = next(iter(targets))
                        slot['branch_review'] = {'target_id': 'https://openalex.org/'+target,
                            'matched_addresses': [a for a, t in matches if t == target],
                            'official_url': JOINT[ident][1], 'publisher_url': receipt['publisher_url'],
                            'method': method, 'rule': 'same_publisher_address_explicit_host'}
                        receipt.update(status='confirmed_branch', **slot['branch_review'])
                    else:
                        receipt.update(status='unresolved_branch', reason='no_unique_explicit_host_in_same_publisher_address')
                    audits.append(receipt)
                else:
                    receipt.update(official_url=OFFICIAL[slot['display_name']][1], target_name=OFFICIAL[slot['display_name']][0])
                    rollups.append(receipt)
        row[FIELD] = json.dumps(authors, ensure_ascii=False)
    pq.write_table(pa.Table.from_pylist(rows), OUT / 'research_roster_joint_review.parquet')
    write_json(OUT / 'branch_address_audit.json', audits)
    write_csv(OUT / 'branch_address_audit.csv', audits)
    write_csv(OUT / 'ror_rollup_publisher_audit.csv', rollups)
    write_json(OUT / 'official_rollup_rules.json', OFFICIAL)
    write_json(OUT / 'input_receipt.json', {'path': str(INPUT), 'sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest()})
    print('branch statuses', {s: sum(a['status'] == s for a in audits) for s in ['confirmed_branch', 'unresolved_branch']})


if __name__ == '__main__':
    prepare()
    apply()
