"""Second affiliation review: evidence-backed edits, site countries, fixed-mass audits."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .nature_backfill_fetch import BASE, DirectPublisher, parse_page
from .nature_backfill_registry import normalized
from .nature_backfill_resolve import author_addresses, clean_address
from .nature_institution_apply import country_names
from .nature_backfill_resolve import address_country
from .stage0 import write_csv, write_json

ROOT = BASE.parent
PREV = ROOT / 'nature_institution_review'
OUT = ROOT / 'nature_weighted_review'
RULES = Path(__file__).parent / 'review_inputs/nature_weighted_rules.json'
FIELD = 'authorships_weighted_review_json'


def load_rows() -> list[dict[str, Any]]:
    return pq.read_table(PREV / 'research_roster_reviewed.parquet').to_pylist()


def conflicts() -> list[dict[str, Any]]:
    with (PREV / 'country_address_conflicts.csv').open(encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def inventory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units: dict[str, Any] = {}
    for row in rows:
        authors = json.loads(row['authorships_reviewed_json'])
        for pos, author in enumerate(authors):
            slots = author['institution_review_entities']
            for slot in slots:
                if slot['identity_status'] != 'identity_unresolved':
                    continue
                name = slot['display_name']
                unit = units.setdefault(name, {'address': name, 'weight': 0.0, 'instances': [],
                    'entity_type_before': slot['review_entity_type'], 'country': slot.get('country_code')})
                unit['weight'] += 1 / len(authors) / len(slots)
                unit['instances'].append({'doi': row['doi'], 'author_position': pos,
                    'author_name': author['author']['display_name'],
                    'previously_no_verified_entity': all(s['identity_status'] == 'identity_unresolved' for s in slots)})
    result = sorted(units.values(), key=lambda u: -u['weight'])
    for unit in result:
        unit['paper_n'] = len({i['doi'] for i in unit['instances']})
        unit['author_n'] = len(unit['instances'])
    write_json(OUT / 'weighted_priority_units.json', result)
    write_csv(OUT / 'weighted_priority_units.csv', result)
    return result


def publisher(doi: str) -> dict[str, Any]:
    suffix = doi.split('/')[1] + '.json'
    for folder in [ROOT / 'nature_metadata_backfill', OUT]:
        path = folder / 'publisher_records' / suffix
        if path.exists():
            return json.loads(path.read_text())
    url = 'https://www.nature.com/articles/' + doi.split('/')[1]
    token = hashlib.sha256(url.encode()).hexdigest()
    folder = BASE if (BASE / 'publisher_raw' / (token + '.html.gz')).exists() else OUT
    client = DirectPublisher(folder)
    result = parse_page(client.get(url), doi)
    result.update(source_url=url, source_receipt=str(folder / 'publisher_raw' / (token + '.html.json')))
    write_json(OUT / 'publisher_records' / suffix, result)
    return result


def prepare() -> None:
    rows = load_rows()
    inventory(rows)
    selected = {c['doi'] for c in conflicts() if c['cn_hk_mo_granularity'] == 'False'}
    with ThreadPoolExecutor(max_workers=5) as pool:
        pubs = list(pool.map(publisher, sorted(selected)))
    by_doi = {r['doi']: r for r in rows}
    review = []
    for c in conflicts():
        if c['cn_hk_mo_granularity'] != 'False':
            continue
        pub = next(p for p in pubs if p['doi'] == c['doi'])
        author = json.loads(by_doi[c['doi']]['authorships_reviewed_json'])[int(c['author_position'])]
        addresses, method = author_addresses(author, pub)
        review.append({**c, 'author_name': author['author']['display_name'],
            'publisher_addresses': addresses, 'publisher_match_method': method, 'source_receipt': pub['source_receipt']})
    write_json(OUT / 'country_conflict_evidence.json', review)
    write_csv(OUT / 'country_conflict_evidence.csv', review)
    print(f'Prepared {len(review)} non-granularity country links in {len(pubs)} papers')


def local_entity(name: str, kind: str, country: str | None, source: dict[str, Any],
                 status: str = 'verified_local_no_registry_id') -> dict[str, Any]:
    ident = 'reviewed-local:' + hashlib.sha256((name + (country or '')).encode()).hexdigest()[:20]
    return {'id': None, 'ror': None, 'display_name': name, 'review_entity_id': ident,
            'review_entity_type': kind, 'country_code': country, 'identity_status': status,
            'weighted_review_evidence': [source]}


def decision_map(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entities = {s['display_name']: s for r in rows for a in json.loads(r['authorships_reviewed_json'])
                for s in a['institution_review_entities'] if s['identity_status'] == 'standard_registry'}
    rules = json.loads(RULES.read_text())
    receipts = {r['url']: r for r in json.loads((OUT / 'source_receipts.json').read_text())}
    web_captures = {str(p): p.read_text() for p in (OUT / 'web_evidence').glob('*.json')}
    output = {}
    for unit in json.loads((OUT / 'weighted_priority_units.json').read_text()):
        matches = [r for r in rules if re.search(r['pattern'], unit['address'], re.IGNORECASE)]
        if len(matches) > 1:
            raise ValueError(f'Overlapping reviewed rules: {unit["address"]}')
        if not matches:
            continue
        rule = matches[0]
        country = unit['country']
        if rule['countries'] and country not in rule['countries']:
            continue
        evidence = {**rule, 'source_receipt': receipts[rule['source_url']]['receipt'],
                    'source_capture_status': receipts[rule['source_url']].get('status'),
                    'web_evidence_receipts': [p for p, text in web_captures.items() if rule['source_url'] in text],
                    'method': 'manual_official_source_and_publication_address'}
        if rule['existing_name']:
            inst = copy.deepcopy(entities[rule['existing_name']])
            inst.update(review_entity_type=rule['entity_type'], weighted_review_evidence=[evidence])
        else:
            inst = local_entity(rule['canonical_name'], rule['entity_type'], country, evidence)
        output[unit['address']] = {**unit, 'entity': inst, 'evidence': evidence}
    write_json(OUT / 'weighted_affiliation_decisions.json', list(output.values()))
    write_csv(OUT / 'weighted_affiliation_decisions.csv', list(output.values()))
    return output


def address_key(value: str) -> str:
    return normalized(clean_address(value).replace('’', "'").replace('—', '-'))


def site_country(value: str) -> str | None:
    # The study retains country/area codes (HK/MO/PR/SJ), not sovereign aggregation.
    if re.search(r',\s*Hong Kong(?: SAR)?(?:,\s*China)?\.?$', clean_address(value), re.I):
        return 'HK'
    if re.search(r',\s*Maca[ou](?:,\s*China)?\.?$', clean_address(value), re.I):
        return 'MO'
    return address_country(value, country_names())


def country_actions() -> dict[tuple[str, int, str], dict[str, Any]]:
    evidence = json.loads((OUT / 'country_conflict_evidence.json').read_text())
    actions: dict[tuple[str, int, str], Any] = {}
    rows = {r['doi']: r for r in load_rows()}
    for c in evidence:
        key = (c['doi'], int(c['author_position']), c['institution_id'])
        if key in actions:
            actions[key]['raw_link_n'] += 1
            continue
        a = json.loads(rows[c['doi']]['authorships_reviewed_json'])[key[1]]
        pubkeys = {address_key(x) for x in c['publisher_addresses']}
        linked = [f['raw_affiliation_string'] for f in a.get('affiliations', [])
                  if c['institution_id'] in f.get('institution_ids', []) and
                  address_key(f['raw_affiliation_string']) in pubkeys]
        sites = sorted({site_country(x) for x in linked} - {None})
        action = {**c, 'raw_link_n': 1, 'action': 'site_country_verified', 'sites': sites,
                  'reason': 'publisher_explicit_author_affiliation_instead_of_registry_headquarters'}
        suffix = c['doi'].split('/')[1]
        if suffix == 's41586-023-05791-5':
            action.update(action='unresolved_malformed_publisher_address', sites=[],
                          reason='Publisher itself combines Helsinki Finland with Cordoba Spain; retain FI and flag sensitivity.')
        elif {c['registry_country'], c['address_country']} in [{'SJ', 'NO'}, {'PR', 'US'}]:
            action.update(action='retain_country_area_granularity', sites=[], reason='Consistent separate ISO territory convention.')
        elif suffix == 's41586-024-07118-4' and c['registry_country'] == 'CD':
            action.update(action='retain_CD_ambiguous_Congo_suffix', sites=[], reason='Kisangani institution identity is CD; bare Congo is not evidence for CG.')
        elif suffix in ['s41586-023-06251-w', 's41586-025-09092-x']:
            action.update(action='remove_present_address_only_slot', sites=[],
                          reason='Publisher nAff present-address block explicitly excluded from primary affiliation refs.')
        elif suffix == 's41586-023-06529-z' and key[1] == 20:
            action.update(action='reject_unconfirmed_raw_extra_affiliation', sites=['SG'],
                          reason='Exact publisher author node has Singapore only; raw Chongqing extra not confirmed.')
        elif c['name'] == 'The Ocean Foundation':
            action.update(action='replace_wrong_identity', replacement_name='Voice of the Ocean Foundation',
                          replacement_type='nonprofit', sites=['SE'], official_url='https://voiceoftheocean.org/about-us/',
                          official_source_receipt=str(OUT / 'web_evidence/sources10.json'))
        elif c['name'] == 'TaiGen Biotechnology (Taiwan)':
            action.update(action='replace_wrong_identity', replacement_name='Theiagen Genomics',
                          replacement_type='company', sites=['US'], official_url='https://theiagen.com/',
                          official_source_receipt=str(OUT / 'web_evidence/sources9.json'))
        elif c['name'] == 'Instituto Tecnológico de Mazatlán':
            action.update(action='replace_noninstitutional_author_status', replacement_name=c['publisher_addresses'][0],
                          replacement_type='independent_researcher', sites=['US'])
        elif c['name'] in ['Institut de Recherche pour le Développement',
                            'Instituto Mixto de Investigación en Biodiversidad', 'UNESCO']:
            action.update(action='quarantine_wrong_identity_country_verified',
                          replacement_name=c['publisher_addresses'][0],
                          replacement_type='company' if 'EIRL' in c['publisher_addresses'][0] else 'unknown',
                          sites=[c['address_country']], reason='Publisher names a different entity; replacement registry identity remains unresolved.')
        elif not sites:
            action.update(action='unresolved_publisher_link', reason='No exact publisher-linked address confirmation.')
        actions[key] = action
    write_json(OUT / 'country_conflict_decisions.json', list(actions.values()))
    write_csv(OUT / 'country_conflict_decisions.csv', list(actions.values()))
    region = [dict(c, action='retain_CN_HK_MO_country_area_granularity') for c in conflicts()
              if c['cn_hk_mo_granularity'] == 'True']
    write_csv(OUT / 'country_area_granularity_retained.csv', region)
    return actions


def apply_country(slot: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    kind = action['action']
    if kind == 'remove_present_address_only_slot':
        return None
    if kind in ['replace_wrong_identity', 'quarantine_wrong_identity_country_verified', 'replace_noninstitutional_author_status']:
        status = {'quarantine_wrong_identity_country_verified': 'identity_unresolved',
                  'replace_noninstitutional_author_status': 'verified_noninstitutional_status'}.get(kind, 'verified_local_no_registry_id')
        result = local_entity(action['replacement_name'], action['replacement_type'], action['sites'][0], action, status)
        result['rejected_original_entity_id'] = slot['review_entity_id']
        return result
    result = copy.deepcopy(slot)
    result.setdefault('weighted_review_evidence', []).append(action)
    if action['sites']:
        result['site_country_codes'] = action['sites']
    return result


def deduplicate(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, Any] = {}
    for slot in slots:
        key = slot['review_entity_id']
        if key in result:
            target = result[key]
            target.setdefault('weighted_review_evidence', []).extend(slot.get('weighted_review_evidence', []))
            target.setdefault('resolved_address_aliases', []).extend(slot.get('resolved_address_aliases', []))
            if slot.get('site_country_codes'):
                target['site_country_codes'] = sorted(set(target.get('site_country_codes', [])) | set(slot['site_country_codes']))
        else:
            result[key] = copy.deepcopy(slot)
    return list(result.values())


def fill_missing_country(slot: dict[str, Any], author: dict[str, Any], doi: str) -> dict[str, Any]:
    if slot.get('country_code') or slot.get('site_country_codes') or slot['display_name'] == 'No affiliation evidence':
        return slot
    pub = publisher(doi)
    addresses, method = author_addresses(author, pub)
    # Manually reviewed raw fragments do not belong to the current primary affiliation.
    # Never transfer the author's other affiliation country to an unrelated unresolved slot.
    if slot['display_name'].startswith(('Pôle Sino-Français de Recherches',
            'Administration Bureau of Naban River Watershed', 'Imec, .')):
        slot['country_review_unresolved_reason'] = 'raw_fragment_not_confirmed_in_primary_publisher_affiliations'
        slot.setdefault('weighted_review_evidence', []).append({'method': 'country_not_transferred_from_other_author_address',
            'source_receipt': pub['source_receipt'], 'publisher_addresses': addresses})
        return slot
    codes = {site_country(x) for x in addresses}
    if method.startswith('publisher_') and len(codes) == 1 and None not in codes:
        slot['site_country_codes'] = sorted(codes)
        slot.setdefault('weighted_review_evidence', []).append({'method': 'all_exact_author_primary_addresses_agree_on_country',
            'publisher_addresses': addresses, 'source_receipt': pub['source_receipt'], 'source_url': pub['source_url']})
    return slot


def apply_review() -> None:
    rows = load_rows()
    mapping, actions = decision_map(rows), country_actions()
    enriched, changes, authors_audit = [], [], []
    for row in rows:
        authors = json.loads(row['authorships_reviewed_json'])
        revised = []
        for pos, author in enumerate(authors):
            slots = []
            for old in author['institution_review_entities']:
                slot = copy.deepcopy(old)
                if old['identity_status'] == 'identity_unresolved' and old['display_name'] in mapping:
                    decision = mapping[old['display_name']]
                    pub = publisher(row['doi'])
                    addresses, method = author_addresses(author, pub)
                    # Allow verified spelling/street variants only inside the same manual entity rule and country.
                    key = address_key(old['display_name'])
                    verified_addresses = [x for x in addresses if key == address_key(x) or
                        (re.search(decision['evidence'].get('publisher_pattern', decision['evidence']['pattern']), x, re.I) and
                         site_country(x) == (old.get('country_code') or decision['entity'].get('country_code')))]
                    if method.startswith('publisher_') and verified_addresses:
                        slot = copy.deepcopy(decision['entity'])
                        slot['resolved_address_aliases'] = [old['display_name']]
                        slot['weighted_review_evidence'].append({'source_url': pub['source_url'],
                            'source_receipt': pub['source_receipt'], 'address': old['display_name'],
                            'verified_publisher_addresses': verified_addresses, 'method': method})
                    else:
                        slot['weighted_review_candidate_entity'] = decision['entity']
                        slot['unresolved_reason'] = 'unit_identity_verified_but_author_to_publisher_address_link_unconfirmed'
                        slot['weighted_review_evidence'] = [{'source_url': pub['source_url'],
                            'source_receipt': pub['source_receipt'], 'publisher_match_method': method,
                            'publisher_addresses': addresses, 'unit_evidence': decision['evidence']}]
                action = actions.get((row['doi'], pos, old['review_entity_id']))
                if action:
                    slot = apply_country(slot, action)
                if slot is not None:
                    slot = fill_missing_country(slot, author, row['doi'])
                    slots.append(slot)
                if slot != old:
                    changes.append({'doi': row['doi'], 'author_position': pos, 'author_name': author['author']['display_name'],
                        'old_slot': old, 'new_slot': slot, 'country_action': action})
            slots = deduplicate(slots)
            if not slots:
                raise ValueError('Review cannot silently remove every author affiliation slot')
            out = copy.deepcopy(author)
            out['institution_weighted_review_entities'] = slots
            old_known = any(s['identity_status'] != 'identity_unresolved' for s in author['institution_review_entities'])
            new_known = any(s['identity_status'] != 'identity_unresolved' for s in slots)
            authors_audit.append({'doi': row['doi'], 'author_position': pos, 'author_name': author['author']['display_name'],
                'before_known': old_known, 'after_known': new_known,
                'source_roster': str(PREV / 'research_roster_reviewed.parquet'),
                'source_field': 'authorships_reviewed_json',
                'standard_after': any(s['identity_status'] == 'standard_registry' for s in slots),
                'all_slots_known': all(s['identity_status'] != 'identity_unresolved' for s in slots),
                'all_slots_country': all(s.get('site_country_codes') or s.get('country_code') for s in slots),
                'remaining_slots': [s for s in slots if s['identity_status'] == 'identity_unresolved']})
            revised.append(out)
        enriched.append({**row, FIELD: json.dumps(revised, ensure_ascii=False)})
    pq.write_table(pa.Table.from_pylist(enriched), OUT / 'research_roster_weighted_review.parquet', compression='zstd')
    write_json(OUT / 'mapping_change_log.json', changes)
    write_csv(OUT / 'mapping_change_log.csv', changes)
    write_json(OUT / 'author_coverage_audit.json', authors_audit)
    write_csv(OUT / 'remaining_identity_authors.csv', [a for a in authors_audit if not a['after_known']])
    summarize_review(rows, enriched, changes, authors_audit)


def rank_scenarios(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scenarios = ['before', 'reviewed', 'exclude_nonpersonal', 'registry_country_sensitivity',
                 'raw_country_sensitivity', 'complete_case']
    countries, education, summaries, paper_missing, unknown_units = [], [], {}, [], []
    for scenario in scenarios:
        cc, uu, unknown_by_country = defaultdict(float), defaultdict(float), defaultdict(float)
        names, totals, included = {}, Counter(), 0
        for row in rows:
            authors = json.loads(row['authorships_reviewed_json'] if scenario == 'before' else row[FIELD])
            slot_key = 'institution_review_entities' if scenario == 'before' else 'institution_weighted_review_entities'
            if scenario == 'complete_case' and any(s['identity_status'] == 'identity_unresolved' for a in authors for s in a[slot_key]):
                continue
            if scenario == 'exclude_nonpersonal':
                authors = [a for a in authors if a['author_record_kind'] not in
                    ['group_subdivision_heading', 'collective_author', 'acknowledged_consortium_not_byline']]
            included += 1
            pm = {'doi': row['doi'], 'year': row['publisher_year'], 'identity_unknown': 0.0, 'country_unknown': 0.0}
            for author in authors:
                slots = author[slot_key]
                for slot in slots:
                    weight = 1 / len(authors) / len(slots)
                    codes = slot.get('site_country_codes') or [slot.get('country_code') or 'unknown']
                    if scenario == 'registry_country_sensitivity':
                        codes = [slot.get('country_code') or 'unknown']
                    if scenario == 'raw_country_sensitivity':
                        raw = {site_country(a['raw_affiliation_string']) for a in author.get('affiliations', [])
                               if a.get('institution_ids') == [slot['review_entity_id']]} - {None}
                        if len(raw) == 1 and not (raw | set(codes)) <= {'CN', 'HK', 'MO'}:
                            codes = sorted(raw)
                    for code in codes:
                        cc[code] += weight / len(codes)
                    ident = slot['review_entity_id']
                    names[ident] = slot['display_name']
                    if slot['review_entity_type'] in ['university', 'education_proxy']:
                        uu[ident] += weight
                    if slot['identity_status'] == 'identity_unresolved':
                        totals['identity_unknown'] += weight
                        pm['identity_unknown'] += weight
                        if slot['review_entity_type'] == 'unknown':
                            totals['possible_university_unknown'] += weight
                            for code in codes:
                                unknown_by_country[code] += weight / len(codes)
                        if scenario == 'reviewed':
                            unknown_units.append({'doi': row['doi'], 'address': slot['display_name'],
                                'entity_type': slot['review_entity_type'], 'countries': codes, 'weight': weight,
                                'reason': slot.get('unresolved_reason', 'canonical_identity_or_parent_not_confirmed_from_available_evidence')})
                    if 'unknown' in codes:
                        pm['country_unknown'] += weight / len(codes)
            if scenario == 'reviewed':
                paper_missing.append(pm)
        assert abs(sum(cc.values()) - included) < 1e-7
        summaries[scenario] = dict(totals, paper_n=included, country_unknown=cc.get('unknown', 0),
            country_mass=sum(cc.values()), education_mass=sum(uu.values()), university_unknown_by_country=dict(unknown_by_country))
        countries.extend({'scenario': scenario, 'rank': j + 1, 'country': k, 'fractional_n': v}
                         for j, (k, v) in enumerate(sorted(cc.items(), key=lambda x: -x[1])) if k != 'unknown')
        education.extend({'scenario': scenario, 'rank': j + 1, 'institution_id': k, 'name': names[k], 'fractional_n': v}
                         for j, (k, v) in enumerate(sorted(uu.items(), key=lambda x: -x[1])))
    write_csv(OUT / 'country_rank_scenarios.csv', countries)
    write_csv(OUT / 'university_rank_scenarios.csv', education)
    write_csv(OUT / 'paper_missing_weight.csv', paper_missing)
    write_json(OUT / 'fractional_summary.json', summaries)
    grouped: dict[str, Any] = {}
    for unit in unknown_units:
        item = grouped.setdefault(unit['address'], {'address': unit['address'], 'weight': 0.0,
            'entity_type': unit['entity_type'], 'countries': unit['countries'], 'dois': set(), 'author_n': 0, 'reasons': set()})
        item['weight'] += unit['weight']
        item['dois'].add(unit['doi'])
        item['author_n'] += 1
        item['reasons'].add(unit['reason'])
    remaining = []
    for unit in sorted(grouped.values(), key=lambda x: -x['weight']):
        unit['dois'] = sorted(unit['dois'])
        unit['reasons'] = sorted(unit['reasons'])
        unit['paper_n'] = len(unit['dois'])
        unit['unresolved_reason'] = ('no_author_affiliation_evidence' if unit['address'] == 'No affiliation evidence' else ';'.join(unit['reasons']))
        remaining.append(unit)
    write_json(OUT / 'remaining_weighted_units.json', remaining)
    write_csv(OUT / 'remaining_weighted_units.csv', remaining)
    return stability(rows, countries, education, summaries)


def stability(rows: list[dict[str, Any]], countries: list[dict[str, Any]], universities: list[dict[str, Any]],
              summaries: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    membership = []
    for label, data, key, k in [('country', countries, 'country', 12), ('university', universities, 'institution_id', 15)]:
        reviewed = [r for r in data if r['scenario'] == 'reviewed']
        current = [r[key] for r in reviewed[:k]]
        comparisons = {}
        for scenario in summaries:
            roster = [r[key] for r in data if r['scenario'] == scenario][:k]
            comparisons[scenario] = {'same_members': set(roster) == set(current), 'same_order': roster == current,
                'entered_vs_reviewed': [x for x in roster if x not in current],
                'left_vs_reviewed': [x for x in current if x not in roster]}
        results[label] = {'k': k, 'cutoff': reviewed[k - 1], 'next': reviewed[k],
            'boundary_gap': reviewed[k - 1]['fractional_n'] - reviewed[k]['fractional_n'], 'scenarios': comparisons}
        if label == 'country':
            unknown = summaries['reviewed']['country_unknown']
            max_outside = max(r['fractional_n'] + unknown for r in reviewed[k:])
            results[label]['fixed_slot_missing_only_guaranteed_members'] = [r[key] for r in reviewed[:k] if r['fractional_n'] > max_outside]
            results[label]['unknown_mass'] = unknown
            for r in reviewed[:20]:
                membership.append(dict(category=label, name=r[key], rank=r['rank'], lower=r['fractional_n'],
                    upper=r['fractional_n'] + unknown, guaranteed_top_k=r[key] in results[label]['fixed_slot_missing_only_guaranteed_members']))
        else:
            entity_country = {s['review_entity_id']: s.get('country_code') for row in rows for a in json.loads(row[FIELD])
                for s in a['institution_weighted_review_entities'] if s['review_entity_type'] in ['university', 'education_proxy']}
            masses = summaries['reviewed']['university_unknown_by_country']
            bounds = [{**r, 'upper': r['fractional_n'] + masses.get(entity_country.get(r[key]), 0) + masses.get('unknown', 0)} for r in reviewed]
            max_outside = max(r['upper'] for r in bounds[k:])
            results[label]['fixed_slot_country_constrained_guaranteed_members'] = [r[key] for r in bounds[:k] if r['fractional_n'] > max_outside]
            results[label]['unknown_mass'] = summaries['reviewed']['possible_university_unknown']
            for r in bounds[:25]:
                membership.append(dict(category=label, name=r['name'], rank=r['rank'], lower=r['fractional_n'], upper=r['upper'],
                    guaranteed_top_k=r[key] in results[label]['fixed_slot_country_constrained_guaranteed_members']))
    results['bound_scope'] = ('Conditional screening bounds only: fixed author/entity slots, university country agrees with unresolved site; '
        'not confidence intervals and not robust to future deduplication, overseas university sites, systematic parent rollup, '
        'uncorrected existing mappings or malformed publisher country conflicts.')
    clean_country = [r for r in countries if r['scenario'] == 'exclude_nonpersonal']
    clean_gap = clean_country[11]['fractional_n'] - clean_country[12]['fractional_n']
    clean_unknown = summaries['exclude_nonpersonal']['country_unknown']
    results['country']['exclude_nonpersonal_missing_only_bound'] = {
        'boundary_gap': clean_gap, 'unknown_country_mass': clean_unknown,
        'top12_members_guaranteed_under_fixed_slot_missing_only_model': clean_gap > clean_unknown,
        'margin': clean_gap - clean_unknown}
    write_csv(OUT / 'conditional_membership_bounds.csv', membership)
    write_json(OUT / 'topk_stability.json', results)
    return results


def summarize_review(before: list[dict[str, Any]], after: list[dict[str, Any]], changes: list[dict[str, Any]],
                     audit: list[dict[str, Any]]) -> None:
    by_doi: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in audit:
        by_doi[a['doi']].append(a)
    remaining = []
    coverage = Counter()
    for row in after:
        aa = by_doi[row['doi']]
        for key in ['after_known', 'standard_after', 'all_slots_known', 'all_slots_country']:
            coverage[key] += all(a[key] for a in aa)
        if not all(a['after_known'] for a in aa):
            remaining.append({'doi': row['doi'], 'title': row['title'], 'year': row['publisher_year'],
                'unresolved_author_n': sum(not a['after_known'] for a in aa),
                'was_in_393_queue': any(not a['before_known'] for a in aa)})
    summary = {'cohort_n': len(after), 'before_unresolved_papers': len({a['doi'] for a in audit if not a['before_known']}),
        'before_unresolved_authors': sum(not a['before_known'] for a in audit),
        'newly_resolved_authors': sum(not a['before_known'] and a['after_known'] for a in audit),
        'newly_quarantined_authors': sum(a['before_known'] and not a['after_known'] for a in audit),
        'remaining_identity_papers': len(remaining), 'remaining_identity_authors': sum(not a['after_known'] for a in audit),
        'original_queue_remaining_papers': sum(r['was_in_393_queue'] for r in remaining),
        'coverage': dict(coverage), 'changed_slot_n': len(changes),
        'changed_paper_n': len({c['doi'] for c in changes}),
        'country_action_counts': dict(Counter(x['action'] for x in json.loads((OUT / 'country_conflict_decisions.json').read_text())))}
    write_csv(OUT / 'remaining_papers.csv', remaining)
    write_json(OUT / 'review_summary.json', summary)
    stability_result = rank_scenarios(after)
    # All inherited data columns and author identities remain byte-for-byte unchanged.
    assert all(all(new[k] == value for k, value in old.items()) for old, new in zip(before, after, strict=True))
    assert len({r['doi'] for r in after}) == 3683
    assert summary['before_unresolved_papers'] == 393
    for r in after:
        aa, bb = json.loads(r['authorships_reviewed_json']), json.loads(r[FIELD])
        assert len(aa) == len(bb)
        assert all(a['author'] == b['author'] and a['institutions'] == b['institutions'] for a, b in zip(aa, bb, strict=True))
    validation = {'passed': True, 'cohort_unique_n': 3683, 'all_inherited_columns_unchanged': True,
        'author_order_identity_and_original_registry_list_preserved': True,
        'local_entities_only_in_review_layer': True, 'fractional_conservation_all_scenarios': True,
        'input_sha256': hashlib.sha256((PREV / 'research_roster_reviewed.parquet').read_bytes()).hexdigest()}
    write_json(OUT / 'validation.json', validation)
    write_json(OUT / 'manual_rules.json', json.loads(RULES.read_text()))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v['boundary_gap'] for k, v in stability_result.items() if isinstance(v, dict)}, indent=2))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'apply'])
    args = parser.parse_args()
    {'prepare': prepare, 'apply': apply_review}[args.phase]()
