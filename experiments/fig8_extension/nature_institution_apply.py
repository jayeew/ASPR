"""Apply reviewed institution decisions and audit country/institution weight sensitivity."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from lxml import html

from .nature_backfill_registry import normalized
from .nature_backfill_resolve import address_country
from .nature_backfill_resolve import country_names as source_country_names
from .nature_institution_review import OUT, PREV, units
from .stage0 import write_csv, write_json

MANUAL = Path(__file__).parent / 'review_inputs/nature_manual_institutions.json'


@lru_cache(maxsize=1)
def country_names() -> dict[str, str]:
    return source_country_names()


def source_fetch(url: str) -> dict[str, Any]:
    token = hashlib.sha256(url.encode()).hexdigest()
    path = OUT / 'primary_sources' / (token + '.json')
    if path.exists():
        return {**json.loads(path.read_text()), 'receipt': str(path)}
    session = requests.Session()
    session.trust_env = False
    result: dict[str, Any] = {'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat()}
    try:
        response = session.get(url, timeout=(10, 25))
        result.update(status=response.status_code, final_url=response.url)
        if response.ok:
            result['sha256'] = hashlib.sha256(response.content).hexdigest()
            raw = path.with_suffix('.raw.gz')
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(gzip.compress(response.content))
            result['raw_path'] = str(raw)
            if 'api.ror.org' in url:
                result['organization'] = response.json()
    except (requests.RequestException, ValueError) as error:
        result['error'] = str(error)
    write_json(path, result)
    return {**result, 'receipt': str(path)}


def fetch_sources() -> None:
    decisions = json.loads(MANUAL.read_text())
    urls = list(dict.fromkeys([r['source_url'] for r in decisions] +
        ['https://api.ror.org/v2/organizations/' + r['ror'].rsplit('/', 1)[-1] for r in decisions if r['ror']]))
    with ThreadPoolExecutor(max_workers=5) as pool:
        receipts = list(pool.map(source_fetch, urls))
    write_json(OUT / 'source_receipts.json', receipts)
    print(f'Sources {len(receipts)}; successful raw captures {sum(x.get("status") == 200 for x in receipts)}')


def base_entities(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {i.get('ror') or i['id']: i for r in rows for a in json.loads(r['authorships_enriched_json'])
            for i in a['institutions']}


def registry_type(org: dict[str, Any]) -> str:
    types = org.get('types', [])
    if 'company' in types:
        return 'company'
    if 'healthcare' in types:
        return 'healthcare'
    if 'education' in types:
        return 'university' if any('university' in n['value'].casefold() for n in org['names']) else 'education_other_or_unreviewed'
    if 'facility' in types:
        return 'research_institute'
    return next((t for t in types if t != 'funder'), 'unknown')


def registry_entity(org: dict[str, Any], entities: dict[str, Any], country: str | None,
                    kind: str | None = None) -> dict[str, Any]:
    inst = copy.deepcopy(entities.get(org['id'], {}))
    inst.update(ror=org['id'], display_name=next(n['value'] for n in org['names'] if 'ror_display' in n['types']),
                country_code=country, registry_types=org['types'])
    inst.setdefault('id', None)
    inst.setdefault('type', org['types'][0])
    inst['review_entity_id'] = inst['id'] or inst['ror']
    inst['review_entity_type'] = kind or registry_type(org)
    inst['identity_status'] = 'standard_registry'
    return inst


def type_from_address(address: str) -> tuple[str, str]:
    # These are type-only labels. They NEVER assert canonical entity identity.
    if re.search(r'\b(inc\.?|llc|ltd\.?|gmbh|limited|sarl)\b', address, re.IGNORECASE):
        return 'company', 'publisher_explicit_legal_form'
    if re.search(r'^independent (researcher|scientist|scholar)', address, re.IGNORECASE):
        return 'independent_researcher', 'publisher_explicit_independent_status'
    if re.search(r'\b(indigenous people|citizen scientist)\b', address, re.IGNORECASE):
        return 'community_or_citizen_science', 'publisher_explicit_community_or_citizen_label'
    if re.search(r'\bmuseum\b', address, re.IGNORECASE) and not re.search(r'\buniversity\b', address, re.IGNORECASE):
        return 'museum', 'publisher_explicit_museum_label'
    return 'unknown', 'type_unresolved'


def decisions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    entities = base_entities(rows)
    manual = json.loads(MANUAL.read_text())
    receipts = {r['url']: r for r in json.loads((OUT / 'source_receipts.json').read_text())}
    proposals = {r['address']: r for r in json.loads((OUT / 'registry_proposals.json').read_text())}
    output = {}
    for unit in units():
        address = unit['address']
        matches = [r for r in manual if re.search(r['address_pattern'], address)]
        if len(matches) > 1:
            raise ValueError(f'Conflicting manual rules: {address}')
        result = {**unit, 'country_code': address_country(address, country_names()), 'entities': []}
        if matches:
            rule = matches[0]
            if rule['country_code'] != result['country_code']:
                raise ValueError(f'Manual country mismatch: {address}')
            evidence = {**rule, 'source_receipt': receipts[rule['source_url']]['receipt']}
            if rule['ror']:
                url = 'https://api.ror.org/v2/organizations/' + rule['ror'].rsplit('/', 1)[-1]
                org = receipts[url].get('organization')
                if not org or org['id'] != rule['ror']:
                    raise ValueError(f'Missing registry evidence: {url}')
                inst = registry_entity(org, entities, result['country_code'], rule['entity_type'])
                evidence['registry_receipt'] = receipts[url]['receipt']
            else:
                local_id = 'reviewed-local:' + hashlib.sha256((rule['canonical_name'] + rule['country_code']).encode()).hexdigest()[:20]
                inst = {'id': None, 'ror': None, 'display_name': rule['canonical_name'],
                        'review_entity_id': local_id, 'review_entity_type': rule['entity_type'],
                        'country_code': result['country_code'], 'identity_status': 'verified_local_no_registry_id'}
            result.update(entities=[inst], status=inst['identity_status'], evidence=evidence,
                          entity_type=inst['review_entity_type'])
        elif proposals[address]['status'] == 'accepted_exact_registry':
            proposal = proposals[address]['candidates'][0]
            inst = registry_entity(proposal['organization'], entities, result['country_code'])
            result.update(entities=[inst], status='standard_registry', entity_type=inst['review_entity_type'],
                          evidence={'method': proposal['method'], 'registry_receipt': proposal['receipt']})
        else:
            kind, reason = type_from_address(address)
            result.update(status='identity_unresolved', entity_type=kind, evidence={'method': reason},
                resolution_reason='no_unique_registry_identity_confirmed_after_name_country_city_review',
                query_receipts=proposals[address].get('query_receipts', []),
                plausible_rejected_candidates=proposals[address].get('plausible_rejected_candidates', []))
        key = normalized(address)
        result['address_aliases'] = [address]
        if key in output:
            previous = output[key]
            if previous['status'] != result['status'] or previous['entity_type'] != result['entity_type']:
                raise ValueError(f'Normalized address conflict: {address}')
            result['address_aliases'] += previous['address_aliases']
            result['dois'] = sorted(set(result['dois']) | set(previous['dois']))
            result['paper_n'] = len(result['dois'])
        output[key] = result
    author_keys: dict[str, set[tuple[str, int]]] = defaultdict(set)
    source_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in json.loads((PREV / 'unresolved_author_institutions.json').read_text()):
        for aff in row['affiliation_evidence']:
            key = normalized(aff['address'])
            author_keys[key].add((row['doi'], row['author_position']))
            ev = {'doi': row['doi'], 'address_source': aff['address_source'], 'source_receipt': aff.get('publisher_source_receipt')}
            if ev not in source_evidence[key]:
                source_evidence[key].append(ev)
    for key, result in output.items():
        result['author_n'] = len(author_keys[key])
        result['publisher_evidence'] = source_evidence[key]
    write_json(OUT / 'affiliation_decisions.json', list(output.values()))
    write_csv(OUT / 'affiliation_decisions.csv', list(output.values()))
    return output


def grain_review() -> dict[tuple[str, int], dict[str, Any]]:
    records = json.loads((PREV / 'unresolved_author_institutions.json').read_text())
    review = []
    for row in records:
        if row['affiliation_evidence']:
            continue
        pub = json.loads((PREV / 'publisher_records' / (row['doi'].split('/')[1] + '.json')).read_text())
        raw = Path(pub['source_receipt']).with_suffix('.gz')
        tree = html.fromstring(gzip.decompress(raw.read_bytes()).decode('utf-8'))
        key = normalized(row['author_name'])
        nodes = [x for x in tree.xpath('//*[self::h3 or self::h4 or self::li or self::a]')
                 if normalized(x.text_content()) == key]
        collective = any('institutional-author__name' in x.get('class', '') or
                         x.get('data-author-popup', '').startswith('group-') for x in nodes)
        heading = any('institutional-author__sub-division' in x.get('class', '') for x in nodes)
        heading |= key == 'primary authors' and bool(nodes)
        # ABCTB is explicitly a collective within a publisher member list, not an individual.
        collective |= key == 'abctb investigators' and bool(nodes)
        kind = 'collective_author' if collective else 'group_subdivision_heading' if heading else 'unresolved_person_or_absent_author'
        acknowledged = [p for p in tree.xpath('//p') if 'following consortia:' in p.text_content() and key in normalized(p.text_content())]
        if not nodes and acknowledged and key in {'cveda', 'harvard aging brain study', 'imagen', 'pond'}:
            kind = 'acknowledged_consortium_not_byline'
            nodes = acknowledged
        review.append({**{k: row[k] for k in ['doi', 'author_position', 'author_name']}, 'record_kind': kind,
            'source_receipt': pub['source_receipt'], 'evidence_nodes': [html.tostring(x, encoding='unicode')[:600] for x in nodes]})
    write_json(OUT / 'author_grain_review.json', review)
    write_csv(OUT / 'author_grain_review.csv', review)
    return {(r['doi'], r['author_position']): r for r in review}


def original_entity(inst: dict[str, Any]) -> dict[str, Any]:
    entity = copy.deepcopy(inst)
    kind = entity.get('review_entity_type') or {'education': 'education_proxy', 'facility': 'research_institute'}.get(inst['type'], inst['type'])
    entity.update(review_entity_id=inst.get('id') or inst.get('ror'), review_entity_type=kind,
                  identity_status='standard_registry')
    return entity


def unresolved_slot(address: str, kind: str = 'unknown') -> dict[str, Any]:
    return {'review_entity_id': 'unresolved:' + hashlib.sha256(normalized(address).encode()).hexdigest()[:20],
            'display_name': address, 'country_code': address_country(address, country_names()),
            'review_entity_type': kind, 'identity_status': 'identity_unresolved'}


def author_slots(author: dict[str, Any], mapping: dict[str, Any], reviewed: bool) -> list[dict[str, Any]]:
    institutions = {i.get('id') or i.get('ror'): original_entity(i) for i in author['institutions']}
    slots: dict[str, Any] = dict(institutions)
    for aff in author.get('backfill_affiliations', []):
        address = aff['address']
        decision = mapping.get(normalized(address)) if reviewed else None
        if aff.get('resolved_institution_ids'):
            continue
        additions = decision['entities'] if decision and decision['entities'] else [unresolved_slot(address, decision['entity_type'] if decision else 'unknown')]
        for inst in additions:
            slots[inst['review_entity_id']] = inst
    # Preserve explicit unparsed raw-affiliation fragments even in otherwise mapped authors.
    if not author.get('backfill_affiliations'):
        for aff in author.get('affiliations', []):
            if not aff.get('institution_ids'):
                address = aff['raw_affiliation_string']
                decision = mapping.get(normalized(address)) if reviewed else None
                additions = decision['entities'] if decision and decision['entities'] else [unresolved_slot(address, decision['entity_type'] if decision else 'unknown')]
                for inst in additions:
                    slots[inst['review_entity_id']] = inst
    if not slots:
        slots['unknown'] = {'review_entity_id': 'unknown', 'display_name': 'No affiliation evidence',
                            'country_code': None, 'review_entity_type': 'unknown', 'identity_status': 'identity_unresolved'}
    return list(slots.values())


def apply() -> None:
    rows = pq.read_table(PREV / 'research_roster_enriched.parquet').to_pylist()
    mapping, grain = decisions(rows), grain_review()
    audit, enriched = [], []
    target = {(r['doi'], r['author_position']) for r in json.loads((PREV / 'unresolved_author_institutions.json').read_text())}
    for row in rows:
        authors = json.loads(row['authorships_enriched_json'])
        revised = []
        for pos, author in enumerate(authors):
            out = copy.deepcopy(author)
            slots = author_slots(author, mapping, True)
            out['institution_review_entities'] = slots
            out['author_record_kind'] = grain.get((row['doi'], pos), {}).get('record_kind', 'not_flagged_as_nonpersonal')
            # Strict standard coverage remains comparable to previous phase.
            seen = {i.get('id') or i.get('ror') for i in out['institutions']}
            for inst in slots:
                if inst['identity_status'] == 'standard_registry' and inst['review_entity_id'] not in seen:
                    out['institutions'].append(copy.deepcopy(inst))
                    seen.add(inst['review_entity_id'])
            if (row['doi'], pos) in target:
                audit.append({'doi': row['doi'], 'title': row['title'], 'author_position': pos,
                    'author_name': author['author']['display_name'], 'record_kind': out['author_record_kind'],
                    'standard_resolved': bool(out['institutions']),
                    'identity_resolved': any(i['identity_status'] != 'identity_unresolved' for i in slots),
                    'all_slots_identity_resolved': all(i['identity_status'] != 'identity_unresolved' for i in slots),
                    'entities': slots, 'article_url': row.get('article_url')})
            revised.append(out)
        record = {**row, 'authorships_reviewed_json': json.dumps(revised, ensure_ascii=False),
                  'all_authors_standard_institution_reviewed': all(a['institutions'] for a in revised),
                  'all_authors_verified_entity_reviewed': all(any(i['identity_status'] != 'identity_unresolved' for i in a['institution_review_entities']) for a in revised),
                  'all_affiliation_slots_identity_resolved': all(i['identity_status'] != 'identity_unresolved' for a in revised for i in a['institution_review_entities']),
                  'all_affiliation_slots_country_resolved': all(i.get('country_code') for a in revised for i in a['institution_review_entities'])}
        enriched.append(record)
    pq.write_table(pa.Table.from_pylist(enriched), OUT / 'research_roster_reviewed.parquet', compression='zstd')
    write_json(OUT / 'author_review_audit.json', audit)
    write_csv(OUT / 'author_review_audit.csv', audit)
    write_csv(OUT / 'remaining_standard_mapping_authors.csv', [r for r in audit if not r['standard_resolved']])
    write_csv(OUT / 'remaining_identity_authors.csv', [r for r in audit if not r['identity_resolved']])
    summarize(rows, enriched, audit, mapping, grain)


def summarize(before: list[dict[str, Any]], after: list[dict[str, Any]], audit: list[dict[str, Any]],
              mapping: dict[str, Any], grain: dict[tuple[str, int], Any]) -> None:
    summary = {'cohort_n': len(after), 'target_papers_before': 422, 'target_authorships_before': len(audit),
        'new_standard_author_mappings': sum(r['standard_resolved'] for r in audit),
        'new_verified_identity_author_mappings': sum(r['identity_resolved'] for r in audit),
        'remaining_standard_authorships': sum(not r['standard_resolved'] for r in audit),
        'remaining_identity_authorships': sum(not r['identity_resolved'] for r in audit),
        'remaining_standard_papers': len({r['doi'] for r in audit if not r['standard_resolved']}),
        'remaining_identity_papers': len({r['doi'] for r in audit if not r['identity_resolved']}),
        'unit_status_counts': dict(Counter(r['status'] for r in mapping.values())),
        'grain_review_counts': dict(Counter(r['record_kind'] for r in grain.values())),
        'coverage': {k: sum(bool(r[k]) for r in after) for k in ['all_authors_standard_institution_reviewed',
            'all_authors_verified_entity_reviewed', 'all_affiliation_slots_identity_resolved', 'all_affiliation_slots_country_resolved']}}
    coverage = []
    for year in [2023, 2024, 2025]:
        subset = [r for r in after if r['publisher_year'] == year]
        coverage.append({'year': year, 'n': len(subset), **{k: sum(bool(r[k]) for r in subset) for k in summary['coverage']}})
    write_json(OUT / 'review_summary.json', summary)
    write_csv(OUT / 'coverage_by_year.csv', coverage)
    write_csv(OUT / 'remaining_papers.csv', [{'doi': r['doi'], 'title': r['title'], 'publisher_year': r['publisher_year'],
        'standard_complete': r['all_authors_standard_institution_reviewed'], 'verified_entity_complete': r['all_authors_verified_entity_reviewed'],
        'all_slots_country_complete': r['all_affiliation_slots_country_resolved']} for r in after if not r['all_authors_standard_institution_reviewed']])
    weights(before, after, mapping, grain)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def country_sensitivity(author: dict[str, Any], slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Diagnostic only: explicit raw address country for unambiguous single-ID links."""
    countries: dict[str, set[str]] = defaultdict(set)
    for aff in author.get('affiliations', []):
        ids = aff.get('institution_ids', [])
        country = address_country(aff['raw_affiliation_string'], country_names())
        if len(ids) == 1 and country:
            countries[ids[0]].add(country)
    result = copy.deepcopy(slots)
    for slot in result:
        options = countries.get(slot['review_entity_id'], set())
        if len(options) == 1:
            country = next(iter(options))
            # Preserve separate region codes; do not conflate granularity with wrong locations.
            if not {country, slot.get('country_code')} <= {'CN', 'HK', 'MO'}:
                slot['country_code'] = country
    return result


def weights(before: list[dict[str, Any]], after: list[dict[str, Any]], mapping: dict[str, Any],
            grain: dict[tuple[str, int], Any]) -> None:
    scenarios = ['before', 'reviewed', 'exclude_headings', 'exclude_headings_and_collectives', 'raw_country_sensitivity', 'complete_case']
    countries, universities, types, totals, paper_metrics = [], [], [], {}, []
    names: dict[str, str] = {}
    for scenario in scenarios:
        country = defaultdict(float)
        university = defaultdict(float)
        kinds = defaultdict(float)
        included = 0
        for old, new in zip(before, after, strict=True):
            original = json.loads(old['authorships_enriched_json'])
            positions = list(range(len(original)))
            if scenario.startswith('exclude_'):
                excluded = {'group_subdivision_heading'}
                if scenario.endswith('collectives'):
                    excluded.update({'collective_author', 'acknowledged_consortium_not_byline'})
                positions = [i for i in positions if grain.get((old['doi'], i), {}).get('record_kind') not in excluded]
            if scenario == 'complete_case' and not new['all_affiliation_slots_identity_resolved']:
                continue
            included += 1
            unknown_identity = unknown_country = university_unknown = 0.0
            for pos in positions:
                slots = author_slots(original[pos], mapping, scenario != 'before')
                if scenario == 'raw_country_sensitivity':
                    slots = country_sensitivity(original[pos], slots)
                weight = 1 / len(positions) / len(slots)
                for inst in slots:
                    country[inst.get('country_code') or 'unknown'] += weight
                    kind = inst['review_entity_type']
                    kinds[kind] += weight
                    ident = inst['review_entity_id']
                    names[ident] = inst['display_name']
                    if kind in ['university', 'education_proxy']:
                        university[ident] += weight
                    if inst['identity_status'] == 'identity_unresolved':
                        unknown_identity += weight
                        if kind == 'unknown':
                            university_unknown += weight
                    if not inst.get('country_code'):
                        unknown_country += weight
            if scenario == 'reviewed':
                paper_metrics.append({'doi': old['doi'], 'publisher_year': old['publisher_year'],
                    'institution_unknown_weight': unknown_identity, 'country_unknown_weight': unknown_country,
                    'university_unknown_upper_weight': university_unknown,
                    'had_missing_standard_author_before': not old['all_authors_standard_institution_enriched']})
            totals.setdefault(scenario, defaultdict(float))
            totals[scenario]['unknown_identity_weight'] += unknown_identity
            totals[scenario]['university_unknown_upper_weight'] += university_unknown
        if abs(sum(country.values()) - included) > 1e-7:
            raise ValueError('Country weights do not conserve paper mass')
        totals[scenario].update(paper_n=included, country_unknown_weight=country.get('unknown', 0),
                               education_weight=sum(university.values()), country_weight_sum=sum(country.values()))
        countries.extend({'scenario': scenario, 'country_code': k, 'fractional_n': v} for k, v in country.items())
        universities.extend({'scenario': scenario, 'institution_id': k, 'name': names[k], 'fractional_n': v} for k, v in university.items())
        types.extend({'scenario': scenario, 'entity_type': k, 'fractional_n': v} for k, v in kinds.items())
    write_csv(OUT / 'country_fractional_scenarios.csv', countries)
    write_csv(OUT / 'education_fractional_scenarios.csv', universities)
    write_csv(OUT / 'institution_type_fractional_scenarios.csv', types)
    write_csv(OUT / 'paper_missing_weight.csv', paper_metrics)
    write_json(OUT / 'fractional_summary.json', totals)


if __name__ == '__main__':
    import sys
    {'fetch': fetch_sources, 'apply': apply}[sys.argv[1]]()
