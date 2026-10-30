"""Resolve boundary-priority affiliations and retain an auditable new analysis layer."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .nature_backfill_resolve import author_addresses
from .nature_boundary_sources import (
    FIELD,
    OLD_FIELD,
    OLD_SLOTS,
    OUT,
    PREV,
    SLOTS,
    capture,
    load_rows,
    publisher,
)
from .nature_weighted_review import address_key, deduplicate, local_entity, site_country
from .stage0 import write_csv, write_json


def manual_rules() -> list[dict[str, Any]]:
    specs = [
        (r'^Lynxi Technologies,', 'Beijing Lynxi Technology Co., Ltd.', 'company', 'CN', 'https://github.com/LynxiTechnology'),
        (r'^Beijing StoneWise Technology,', 'Beijing StoneWise Technology Co., Ltd.', 'company', 'CN', 'https://wordpress-hk.stonewise.cn/protocal'),
        (r'^Trip\.com,', 'Trip.com', 'company', 'CN', 'https://www.nature.com/articles/s41586-024-07500-2'),
        (r'^Volastra Therapeutics', 'Volastra Therapeutics, Inc.', 'company', 'US', 'https://www.volastratx.com/'),
        (r'^GRIN Therapeutics,', 'GRIN Therapeutics, Inc.', 'company', 'US', 'https://grintherapeutics.com/about-grin-therapeutics/'),
        (r'^Fauna Robotics,', 'Fauna Robotics', 'company', 'US', 'https://faunarobotics.com/'),
        (r'^DP Technology,', 'DP Technology', 'company', 'CN', 'https://www.dp.tech/en/about'),
        (r'^Shuimu BioSciences,', 'Shuimu BioSciences', 'company', 'CN', 'https://shuimubio.com/about'),
        (r'Sloan Kettering Institute, New York', 'Memorial Sloan Kettering Cancer Center', 'research_institute', 'US', 'https://www.mskcc.org/research/ski'),
        (r'^Allen Discovery Cent(?:er|re) for Neuroimmune Interactions,', 'Allen Discovery Center for Neuroimmune Interactions', 'joint_program', 'US',
         'https://alleninstitute.org/news/the-paul-g-allen-frontiers-group-announces-allen-discovery-center-for-neuroimmune-interactions-at-icahn-school-of-medicine-at-mount-sinai'),
        (r'^Tri.institutional MD.PhD Program,', 'Tri-Institutional MD-PhD Program', 'joint_program', 'US', 'https://teach.weill.cornell.edu/mdphd'),
        (r'^Nanhu Laboratory,', 'Nanhu Laboratory', 'research_institute', 'CN',
         'https://www.nanhu.gov.cn/module/download/downfile.jsp?classid=0&filename=7c3c17fff14a42ab987bf93d0688b9e0.pdf'),
        (r'^National Energy Center for Coal to Clean Fuels, Synfuels China Technology',
         'Synfuels China Technology Co., Ltd.', 'company', 'CN', 'https://icame2019.dicp.ac.cn/Organizers.htm'),
        (r'^Institute for Advanced Algorithms Research,', 'Institute for Advanced Algorithms Research, Shanghai',
         'research_institute', 'CN', 'https://www.iaar.ac.cn/'),
        (r'^Key Laboratory of Multiscale Spin Physics,', 'Beijing Normal University', 'university', 'CN',
         'https://multiscale.bnu.edu.cn/'),
    ]
    return [{'pattern': p, 'canonical_name': n, 'entity_type': t, 'country': c, 'source_url': u}
            for p, n, t, c, u in specs]


def entity_catalog(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {s['review_entity_id']: s for r in rows for a in json.loads(r[OLD_FIELD]) for s in a[OLD_SLOTS]
            if s['identity_status'] != 'identity_unresolved'}


def prepare_mapping(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entities = entity_catalog(rows)
    rules = manual_rules()
    with ThreadPoolExecutor(max_workers=5) as pool:
        receipts = list(pool.map(capture, [r['source_url'] for r in rules]))
    by_name = {s['display_name']: s for s in entities.values()}
    aliases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in entities.values():
        if s['identity_status'] == 'standard_registry' and len(s['display_name']) >= 12:
            aliases[address_key(s['display_name'])].append(s)
    mapping = {}
    for unit in json.loads((PREV / 'remaining_weighted_units.json').read_text()):
        address = unit['address']
        if address == 'Department of Preventive Medicine, Northeastern University, Chicago, IL, USA':
            # Publisher spelling conflicts with official faculty/department records; do not auto-accept either school.
            continue
        rule_index = next((i for i, rule in enumerate(rules) if re.search(rule['pattern'], address, re.IGNORECASE)), None)
        if rule_index is not None:
            rule, receipt = rules[rule_index], receipts[rule_index]
            source = {**rule, 'receipt': receipt, 'web_fallback_dir': str(OUT / 'web_evidence')}
            ent = copy.deepcopy(by_name.get(rule['canonical_name']))
            if ent is None:
                ent = local_entity(rule['canonical_name'], rule['entity_type'], rule['country'], source)
            ent['boundary_identity_evidence'] = [source]
            if rule['canonical_name'].startswith('Allen Discovery'):
                ent.update(joint_university_targets=[by_name[n]['review_entity_id'] for n in
                    ['Cornell University', 'Icahn School of Medicine at Mount Sinai']], joint_partner_n=2)
            if rule['canonical_name'] == 'Tri-Institutional MD-PhD Program':
                ent.update(joint_university_targets=[by_name[n]['review_entity_id'] for n in
                    ['Cornell University', 'Rockefeller University']], joint_partner_n=3)
            mapping[address] = {'entity': ent, 'evidence': source, 'match_kind': 'targeted_primary_review', 'pattern': rule['pattern']}
            continue
        # Exact comma-delimited institutional names, NOT substring/nearest-name matching.
        # Joint programs and hospital addresses require separate hierarchy review.
        if re.search(r'hospital|joint|collaborative|consortium|centre for life sciences|center for life sciences', address, re.IGNORECASE):
            continue
        candidates = {s['review_entity_id']: s for token in address.split(',')
                      for s in aliases.get(address_key(token), [])
                      if s.get('country_code') and s['country_code'] == site_country(address)}
        if len(candidates) == 1:
            ent = copy.deepcopy(next(iter(candidates.values())))
            mapping[address] = {'entity': ent, 'evidence': {'canonical_name': ent['display_name'],
                'ror': ent.get('ror'), 'source_identity': 'previously_established_exact_registry_name'},
                'match_kind': 'exact_existing_canonical_name', 'pattern': None}
    write_json(OUT / 'identity_candidate_decisions.json', mapping)
    write_json(OUT / 'identity_manual_rules.json', rules)
    return mapping


def matching_addresses(author: dict[str, Any], pub: dict[str, Any], address: str,
                       decision: dict[str, Any]) -> tuple[list[str], str]:
    addresses, method = author_addresses(author, pub)
    if not method.startswith('publisher_'):
        return [], method
    key = address_key(address)
    matches = [a for a in addresses if key == address_key(a)]
    if not matches and decision.get('pattern'):
        matches = [a for a in addresses if re.search(decision['pattern'], a, re.IGNORECASE)
                   and site_country(a) == decision['entity'].get('country_code')]
    # Strip only the explicit trailing email, never use email domain as affiliation evidence.
    if not matches:
        without_email = re.split(r'\.\s+[\w.+-]+@', address)[0]
        matches = [a for a in addresses if address_key(without_email) == address_key(a)]
    return matches, method


def apply() -> None:
    before = load_rows()
    mapping = prepare_mapping(before)
    after, changes, audit, attempts = [], [], [], []
    for row in before:
        authors = json.loads(row[OLD_FIELD])
        for pos, author in enumerate(authors):
            slots = []
            for old in author[OLD_SLOTS]:
                new = copy.deepcopy(old)
                if old['display_name'] == 'Department of Preventive Medicine, Northeastern University, Chicago, IL, USA':
                    new['unresolved_reason'] = 'publisher_Northeastern_spelling_conflicts_with_Northwestern_faculty_records'
                    new['boundary_candidate_evidence'] = {'source_url':
                        'https://news.feinberg.northwestern.edu/2022/11/07/celebrating-50-years-of-the-department-of-preventive-medicine/',
                        'note': 'Do not automatically change the university solely from a name/city or current faculty match.'}
                if old['display_name'].startswith('Tsientang Institute for Advanced Study,'):
                    new['unresolved_reason'] = 'publication_2024_2025_vs_official_July_2025_registration_history_needs_reconciliation'
                    new['boundary_candidate_evidence'] = {'source_url': 'https://www.tias.ac.cn/about.html',
                        'note': 'Do not infer Peking parent from email domain; retain publication-site country.'}
                if old['identity_status'] == 'identity_unresolved' and old['display_name'] in mapping:
                    d = mapping[old['display_name']]
                    pub = publisher(row['doi'])
                    matches, method = matching_addresses(author, pub, old['display_name'], d)
                    evidence = {'doi': row['doi'], 'author_position': pos, 'address': old['display_name'],
                        'candidate_name': d['entity']['display_name'], 'method': method, 'publisher_addresses': matches,
                        'source_url': pub['source_url'], 'source_receipt': pub['source_receipt'], 'unit_evidence': d['evidence']}
                    attempts.append({**evidence, 'resolved': bool(matches)})
                    if matches:
                        new = copy.deepcopy(d['entity'])
                        new.pop('site_country_codes', None)
                        new.pop('weighted_review_evidence', None)
                        if old.get('site_country_codes'):
                            new['site_country_codes'] = old['site_country_codes']
                        new['country_code'] = old.get('country_code') or new.get('country_code')
                        new['resolved_address_aliases'] = [old['display_name']]
                        new['boundary_identity_evidence'] = [evidence]
                        changes.append({**evidence, 'old_slot': old, 'new_slot': new})
                    else:
                        new['unresolved_reason'] = 'unit_identity_verified_but_author_to_publisher_address_link_unconfirmed'
                        new['boundary_candidate_evidence'] = evidence
                slots.append(new)
            author[SLOTS] = deduplicate(slots)
            for slot in author[SLOTS]:
                proofs = [p for s in slots if s['review_entity_id'] == slot['review_entity_id']
                          for p in s.get('boundary_identity_evidence', [])]
                if proofs:
                    slot['boundary_identity_evidence'] = proofs
            old_known = any(s['identity_status'] != 'identity_unresolved' for s in author[OLD_SLOTS])
            known = any(s['identity_status'] != 'identity_unresolved' for s in author[SLOTS])
            audit.append({'doi': row['doi'], 'author_position': pos, 'author_name': author['author']['display_name'],
                'before_known': old_known, 'after_known': known,
                'standard_after': any(s['identity_status'] == 'standard_registry' for s in author[SLOTS]),
                'all_slots_known': all(s['identity_status'] != 'identity_unresolved' for s in author[SLOTS]),
                'all_slots_country': all(s.get('site_country_codes') or s.get('country_code') for s in author[SLOTS]),
                'remaining_slots': [s for s in author[SLOTS] if s['identity_status'] == 'identity_unresolved']})
        after.append({**row, FIELD: json.dumps(authors, ensure_ascii=False)})
    pq.write_table(pa.Table.from_pylist(after), OUT / 'research_roster_boundary_review.parquet', compression='zstd')
    write_json(OUT / 'identity_change_log.json', changes)
    write_csv(OUT / 'identity_change_log.csv', changes)
    write_csv(OUT / 'identity_resolution_attempts.csv', attempts)
    write_json(OUT / 'author_coverage_audit.json', audit)
    summarize(before, after, changes, audit)


def summarize(before: list[dict[str, Any]], after: list[dict[str, Any]], changes: list[dict[str, Any]],
              audit: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in audit:
        grouped[a['doi']].append(a)
    keys = ['after_known', 'standard_after', 'all_slots_known', 'all_slots_country']
    remaining = [{'doi': r['doi'], 'title': r['title'], 'year': r['publisher_year'],
                  'unresolved_author_n': sum(not a['after_known'] for a in grouped[r['doi']])}
                 for r in after if not all(a['after_known'] for a in grouped[r['doi']])]
    summary = {'cohort_n': len(after), 'before_unresolved_papers': len({a['doi'] for a in audit if not a['before_known']}),
        'before_unresolved_authors': sum(not a['before_known'] for a in audit),
        'newly_resolved_authors': sum(not a['before_known'] and a['after_known'] for a in audit),
        'remaining_identity_papers': len(remaining), 'remaining_identity_authors': sum(not a['after_known'] for a in audit),
        'coverage': {k: sum(all(a[k] for a in group) for group in grouped.values()) for k in keys},
        'changed_slot_n': len(changes), 'changed_paper_n': len({c['doi'] for c in changes})}
    write_json(OUT / 'review_summary.json', summary)
    write_csv(OUT / 'remaining_papers.csv', remaining)
    write_csv(OUT / 'remaining_identity_authors.csv', [a for a in audit if not a['after_known']])
    write_csv(OUT / 'coverage_by_year.csv', [{'year': y, 'paper_n': sum(r['publisher_year'] == y for r in after),
        **{k: sum(all(a[k] for a in grouped[r['doi']]) for r in after if r['publisher_year'] == y) for k in keys}}
        for y in [2023, 2024, 2025]])
    assert summary['before_unresolved_papers'] == 365
    assert all(all(new[k] == v for k, v in old.items()) for old, new in zip(before, after, strict=True))
    assert len({r['doi'] for r in after}) == 3683
    for r in after:
        aa, bb = json.loads(r[OLD_FIELD]), json.loads(r[FIELD])
        assert all(all(b[k] == v for k, v in a.items()) for a, b in zip(aa, bb, strict=True))
    write_json(OUT / 'identity_validation.json', {'cohort_unique': 3683, 'inherited_columns_unchanged': True,
        'inherited_author_fields_unchanged': True, 'input_sha256': hashlib.sha256(
            (PREV / 'research_roster_weighted_review.parquet').read_bytes()).hexdigest()})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    apply()
