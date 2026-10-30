"""Resolve missing author affiliations conservatively and preserve unresolved cases."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .nature_backfill_fetch import BASE, OUT
from .nature_backfill_registry import normalized
from .stage0 import write_csv, write_json


def clean_address(value: str) -> str:
    return re.sub(r'\s*[^\s,;]+@[^\s,;]+', '', value).strip().rstrip('.')


def name_key(value: str) -> str:
    return normalized(value).replace(' ', '')


def country_names() -> dict[str, str]:
    path = OUT / 'iso_3166-1.json'
    if not path.exists():
        path.write_bytes(Path('/usr/share/iso-codes/json/iso_3166-1.json').read_bytes())
    entries = json.loads(path.read_text())['3166-1']
    names = {normalized(r[k]): r['alpha_2'] for r in entries for k in ['name', 'official_name', 'common_name'] if k in r}
    aliases = {'USA': 'US', 'U.S.A.': 'US', 'UK': 'GB', 'U.K.': 'GB', 'United Kingdom': 'GB', 'The Netherlands': 'NL',
               'South Korea': 'KR', 'Korea': 'KR', 'Republic of Korea': 'KR', 'Russia': 'RU',
               'Iran': 'IR', 'Taiwan ROC': 'TW', 'Taiwan': 'TW', 'P. R. China': 'CN', 'PR China': 'CN',
               'People’s Republic of China': 'CN', 'Hong Kong SAR': 'HK', 'Vietnam': 'VN', 'Türkiye': 'TR', 'Turkey': 'TR',
               'Czech Republic': 'CZ', 'Venezuela': 'VE', 'Tanzania': 'TZ', 'Bolivia': 'BO',
               'The Philippines': 'PH', 'Democratic Republic of the Congo': 'CD',
               'Democratic Republic of Congo': 'CD', 'Sint Maarten': 'SX'}
    names.update({normalized(k): v for k, v in aliases.items()})
    return names


def address_country(address: str, names: dict[str, str]) -> str | None:
    # Exact signed locations, corroborated by the universities' official addresses.
    # Evidence: supplement/korea_exact_locations/rules.json. Bare "Korea" stays ambiguous
    # in callers that remove that token from their country dictionary.
    if normalized(clean_address(address)) in {
        normalized('Kyungpook National University, Daegu, Korea'),
        normalized('Seoul National University, Seoul, Korea'),
        normalized('University of Seoul, Seoul, Korea'),
        normalized('Korea University, Seoul, Korea'),
    }:
        return 'KR'
    # Only an explicit final country/area token, never a city or author nationality.
    token = clean_address(address).rsplit(',', 1)[-1].strip()
    return names.get(normalized(token))


def source_index(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    entities = {i['id']: i for r in rows for a in json.loads(r['authorships_json']) for i in a['institutions']}
    index: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        for author in json.loads(row['authorships_json']):
            for aff in author.get('affiliations', []):
                key = normalized(clean_address(aff['raw_affiliation_string']))
                ids = sorted(i for i in aff.get('institution_ids', []) if i in entities)
                if key and ids:
                    index[key][json.dumps(ids)].add(row['work_id'])
    return entities, dict(index)


def publisher_for(row: dict[str, Any]) -> dict[str, Any]:
    path = OUT / 'publisher_records' / (row['doi'].split('/')[1] + '.json')
    return json.loads(path.read_text()) if path.exists() else {}


def author_addresses(author: dict[str, Any], publisher: dict[str, Any]) -> tuple[list[str], str]:
    keys = {name_key(author.get('raw_author_name') or ''), name_key(author['author'].get('display_name') or '')} - {''}
    matches = [a for a in publisher.get('authors', []) if name_key(a['name']) in keys]
    if len(matches) == 1:
        refs = matches[0]['affiliation_ids']
        addresses = [a['address'] for a in publisher['affiliations'] if a['affiliation_id'] in refs and a['address']]
        if addresses and len(addresses) == len(refs):
            return list(dict.fromkeys(addresses)), 'publisher_exact_author_name'
    if len(matches) <= 1:
        addresses = [aff['address'] for aff in publisher.get('affiliations', []) if aff['address'] and
            any(name_key(name) in keys for name in re.split(r',\s*|\s*&\s*', aff.get('authors_text', '')))]
        if addresses:
            return list(dict.fromkeys(addresses)), 'publisher_exact_affiliation_author_list'
    raw = list(dict.fromkeys(clean_address(a) for a in author.get('raw_affiliation_strings', []) if clean_address(a)))
    return raw, 'openalex_raw_affiliation' if raw else 'no_author_affiliation_evidence'


def exact_mapping(address: str, index: dict[str, Any]) -> tuple[list[str], list[str]]:
    choices = index.get(normalized(clean_address(address)), {})
    if len(choices) == 1:
        ids, work_ids = next(iter(choices.items()))
        return json.loads(ids), sorted(work_ids)
    return [], []


def prepare(rows: list[dict[str, Any]], index: dict[str, Any]) -> None:
    queries, evidence = set(), []
    for row in rows:
        if row['all_observed_authors_have_institution']:
            continue
        pub = publisher_for(row)
        for pos, author in enumerate(json.loads(row['authorships_json'])):
            if author['institutions']:
                continue
            addresses, source = author_addresses(author, pub)
            for address in addresses:
                ids, _ = exact_mapping(address, index)
                if not ids:
                    queries.add(address)
            evidence.append({'doi': row['doi'], 'author_position': pos, 'author_name': author['author']['display_name'],
                             'addresses': addresses, 'address_source': source})
    write_json(OUT / 'affiliation_query_inputs.json', sorted(queries))
    write_json(OUT / 'missing_author_affiliation_evidence.json', evidence)
    print(f'{len(evidence)} missing author mappings; {len(queries)} unique ROR affiliation queries')


def accepted_ror(address: str, result: dict[str, Any], names: dict[str, str]) -> tuple[dict[str, Any] | None, str]:
    chosen = result.get('chosen', [])
    if len(chosen) != 1:
        return None, 'ror_no_unique_chosen_match'
    organization = chosen[0]['organization']
    parts = [part.strip() for part in address.split(',')]
    # A region name in the location suffix is not an institution-name match.
    org_part = ', '.join(parts[:-2]) if len(parts) > 2 else parts[0]
    address_key = ' ' + normalized(org_part) + ' '
    aliases = [n['value'] for n in organization.get('names', [])
               if len(normalized(n['value'])) >= 4]
    if not any(' ' + normalized(alias) + ' ' in address_key for alias in aliases):
        return None, 'ror_chosen_without_exact_name_or_alias_in_address'
    countries = {x['geonames_details']['country_code'] for x in organization.get('locations', [])}
    country = address_country(address, names)
    if country and country not in countries:
        return None, 'ror_publisher_country_conflict'
    if not country and len(countries) != 1:
        return None, 'ror_country_ambiguous'
    cities = {normalized(x['geonames_details'].get('name', '')) for x in organization.get('locations', [])} - {''}
    address_locations = {normalized(p) for p in parts[-4:]}
    if cities and not cities.intersection(address_locations):
        return None, 'ror_location_requires_review'
    return organization, 'ror_chosen_exact_name_country_checked'


def resolve_address(address: str, index: dict[str, Any], entities: dict[str, Any],
                    queries: dict[str, Any], names: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ids, works = exact_mapping(address, index)
    if ids:
        return [copy.deepcopy(entities[i]) for i in ids], {'method': 'existing_exact_affiliation', 'evidence_work_ids': works}
    query = queries.get(address, {})
    organization, status = accepted_ror(address, query, names)
    evidence = {'method': status, 'source_receipt': query.get('source_receipt')}
    if organization:
        existing = [i for i in entities.values() if i.get('ror') == organization['id']]
        if len(existing) == 1:
            inst = copy.deepcopy(existing[0])
        else:
            inst = {'id': None, 'ror': organization['id'],
                    'display_name': next(n['value'] for n in organization['names'] if 'ror_display' in n['types']),
                    'type': organization.get('types', ['unknown'])[0], 'registry_types': organization.get('types', [])}
        countries = {x['geonames_details']['country_code'] for x in organization.get('locations', [])}
        inst['country_code'] = address_country(address, names) or (next(iter(countries)) if len(countries) == 1 else None)
        inst['country_source'] = query.get('source_receipt')
        return [inst], evidence
    return [], evidence


def enrich_author(author: dict[str, Any], pub: dict[str, Any], index: dict[str, Any], entities: dict[str, Any],
                  queries: dict[str, Any], names: dict[str, str], countries: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(author)
    output['backfill_affiliations'] = []
    if not output['institutions']:
        addresses, source = author_addresses(author, pub)
        for address in addresses:
            mapped, evidence = resolve_address(address, index, entities, queries, names)
            output['institutions'].extend(mapped)
            output['backfill_affiliations'].append({'address': address, 'address_source': source,
                'publisher_source_receipt': pub.get('source_receipt') if source.startswith('publisher') else None,
                'country_code': address_country(address, names),
                'local_affiliation_id': 'local-affiliation:' + hashlib.sha256(normalized(address).encode()).hexdigest()[:20],
                'resolved_institution_ids': [i.get('id') or i.get('ror') for i in mapped], **evidence})
    unique = {}
    for inst in output['institutions']:
        repair = countries.get(inst.get('id'), {})
        if not inst.get('country_code') and repair.get('country_code'):
            inst['country_code'], inst['country_source'] = repair['country_code'], repair['source_receipt']
        unique[inst.get('id') or inst.get('ror')] = inst
    output['institutions'] = list(unique.values())
    output['standard_institution_mapped'] = bool(unique)
    output['all_linked_institution_countries_present'] = bool(unique) and all(i.get('country_code') for i in unique.values())
    output['location_country_codes_enriched'] = sorted({i['country_code'] for i in unique.values() if i.get('country_code')} |
        {a['country_code'] for a in output['backfill_affiliations'] if a.get('country_code')})
    return output


def apply(rows: list[dict[str, Any]], entities: dict[str, Any], index: dict[str, Any]) -> None:
    query_path = OUT / 'affiliation_query_results.json'
    queries = {r['address']: r for r in json.loads(query_path.read_text())} if query_path.exists() else {}
    countries = {r['openalex_id']: r for r in json.loads((OUT / 'institution_country_repairs.json').read_text())}
    crosswalk = OUT / 'ror_openalex_crosswalk.json'
    if crosswalk.exists():
        for entity in json.loads(crosswalk.read_text()):
            if entity['id'] not in entities:
                entities[entity['id']] = {key: entity.get(key) for key in
                    ['id', 'ror', 'display_name', 'country_code', 'type', 'lineage', 'openalex_identity_source']}
    names, output, author_audit, unresolved, unresolved_affiliations = country_names(), [], [], [], []
    for row in rows:
        pub = publisher_for(row)
        authors = json.loads(row['authorships_json'])
        enriched = [enrich_author(a, pub, index, entities, queries, names, countries) for a in authors]
        abstract = row['abstract_text']
        source = 'openalex_existing' if abstract else 'unresolved'
        if not abstract and pub.get('abstract_status') == 'publisher_abstract_verified':
            abstract, source = pub['abstract_text'], 'nature_publisher'
        new = {**row, 'abstract_text_enriched': abstract, 'abstract_source': source,
               'abstract_source_receipt': pub.get('source_receipt') if source == 'nature_publisher' else None,
               'authorships_enriched_json': json.dumps(enriched, ensure_ascii=False),
               'has_abstract_enriched': bool(abstract),
               'has_standard_institution_enriched': any(a['standard_institution_mapped'] for a in enriched),
               'all_authors_standard_institution_enriched': bool(enriched) and all(a['standard_institution_mapped'] for a in enriched),
               'all_linked_countries_enriched': bool(enriched) and all(a['all_linked_institution_countries_present'] for a in enriched),
               'all_authors_at_least_one_country_enriched': bool(enriched) and all(a['location_country_codes_enriched'] for a in enriched),
               'publisher_author_n': len(pub['authors']) if pub else None,
               'publisher_affiliation_parse_complete': pub.get('author_affiliation_parse_complete'),
               'metadata_backfill_scope': 'original_openalex_observed_authors; ROR IDs accepted separately from OpenAlex IDs'}
        output.append(new)
        for pos, (before, after) in enumerate(zip(authors, enriched)):
            if not before['institutions']:
                record = {'doi': row['doi'], 'title': row['title'], 'author_position': pos,
                    'author_id': before['author'].get('id'), 'author_name': before['author'].get('display_name'),
                    'resolved': after['standard_institution_mapped'], 'affiliation_evidence': after['backfill_affiliations'],
                    'institutions_enriched': after['institutions'], 'article_url': row['article_url'],
                    'unresolved_reason': None if after['standard_institution_mapped'] else
                        ('no_author_affiliation_evidence' if not after['backfill_affiliations'] else 'standard_registry_mapping_unresolved')}
                author_audit.append(record)
                if not record['resolved']:
                    unresolved.append(record)
                for aff in after['backfill_affiliations']:
                    if not aff['resolved_institution_ids']:
                        unresolved_affiliations.append({'doi': row['doi'], 'title': row['title'],
                            'author_position': pos, 'author_name': before['author'].get('display_name'), **aff})
    pq.write_table(pa.Table.from_pylist(output), OUT / 'research_roster_enriched.parquet')
    write_csv(OUT / 'author_mapping_audit.csv', author_audit)
    write_json(OUT / 'author_mapping_audit.json', author_audit)
    write_csv(OUT / 'unresolved_author_institutions.csv', unresolved)
    write_json(OUT / 'unresolved_author_institutions.json', unresolved)
    write_csv(OUT / 'unresolved_affiliations.csv', unresolved_affiliations)
    print(f'Enriched {len(output)} papers; resolved {sum(a["resolved"] for a in author_audit)}/{len(author_audit)} missing authors')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'apply'], required=True)
    args = parser.parse_args()
    rows = pq.read_table(BASE / 'research_roster_locked.parquet').to_pylist()
    entities, index = source_index(rows)
    if args.phase == 'prepare':
        prepare(rows, index)
    else:
        apply(rows, entities, index)


if __name__ == '__main__':
    main()
