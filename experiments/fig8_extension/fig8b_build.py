"""Stream full candidate records into an auditable metadata release; never certify types by API label."""
from __future__ import annotations

import csv
import gzip
import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .fig8b_collect import OUT
from .fig8b_supplements import (
    html_text,
    load,
    select_epmc,
    supplementary_addresses,
    title_key,
)
from .nature_backfill_registry import normalized
from .nature_backfill_resolve import clean_address, country_names
from .stage0 import DEFAULT_OUT, write_csv, write_json
from .stage0_roster import abstract_text

NATURE = OUT.parent / 'nature'


def reference(source: Path, destination: Path) -> None:
    """Reference one canonical source instead of copying an inherited data layer."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        if destination.is_dir():
            raise ValueError(f'Migrate existing inherited directory before linking: {destination}')
        destination.unlink()
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()), target_is_directory=source.is_dir())


def doi_key(value: str | None) -> str:
    return re.sub(r'^https?://(?:dx\.)?doi\.org/', '', value or '', flags=re.IGNORECASE).lower().strip()


def name_key(value: str | None) -> str:
    return re.sub(r'\s+', ' ', value or '').strip().casefold()


@lru_cache(maxsize=250000)
def terminal_country_token(address: str) -> str:
    return normalized(clean_address(address).rsplit(',', 1)[-1].strip())


def explicit_countries(author: dict[str, Any], names: dict[str, str]) -> list[str]:
    addresses = list(dict.fromkeys(author.get('raw_affiliation_strings') or []))
    if not addresses:
        return []
    countries = []
    fragments = [fragment.strip() for address in addresses for fragment in address.split(';') if fragment.strip()]
    for address in fragments:
        if re.search(r'\b(present|current) (address|affiliation)\b', address, re.IGNORECASE):
            return []
        # Do not collapse explicit country/area combinations into an end-token country.
        if re.search(r'hong kong|macao|macau|taiwan', address, re.IGNORECASE):
            return []
        code = names.get(terminal_country_token(address))
        if not code:
            return []
        countries.append(code)
    return sorted(set(countries))


def enrich_authors(authors: list[dict[str, Any]], donor: dict[str, Any] | None,
                   names: dict[str, str], secondary: dict | None = None) -> tuple[list[dict[str, Any]], Counter]:
    counts: Counter = Counter()
    extra_addresses = supplementary_addresses((secondary or {}).get("record"))
    donor_names: dict[str, list] = defaultdict(list)
    for author in (donor or {}).get('authors', []):
        donor_names[name_key(author['name'])].append(author)
    observed_names = Counter(name_key(a.get('author', {}).get('display_name')) for a in authors)
    result = []
    for pos, author in enumerate(authors):
        name = author.get('author', {}).get('display_name')
        original_entities = [i['id'] for i in author.get('institutions') or [] if i.get('id')]
        countries = sorted(set(author.get('countries') or []))
        candidates = donor_names.get(name_key(name), [])
        verified = candidates[0] if len(candidates) == 1 and observed_names[name_key(name)] == 1 else None
        entity_ids, final_countries = original_entities[:], countries[:]
        status, country_source = 'inherited_openalex_not_independently_verified', 'openalex_authorship_countries'
        nonpersonal = False
        if verified:
            entity_ids = sorted({s['entity_id'] for s in verified['slots'] if not s['identity_missing']})
            final_countries = sorted({c for s in verified['slots'] for c in s['country_codes']})
            status, country_source = 'nature_v1_exact_doi_and_unique_author_name', 'nature_v1_slots'
            nonpersonal = verified['nonpersonal']
            counts['nature_v1_author_reused'] += 1
        explicit = explicit_countries(author, names)
        if not final_countries and explicit:
            final_countries = explicit
            country_source = 'all_observed_raw_addresses_explicit_terminal_country'
            counts['author_country_filled_from_explicit_address'] += 1
        epmc_addresses = extra_addresses.get(name_key(name), [])
        supplementary = []
        if not final_countries and len(epmc_addresses) == 1 and observed_names[name_key(name)] == 1:
            supplementary = epmc_addresses[0]
            extra_countries = explicit_countries({'raw_affiliation_strings': supplementary}, names)
            if extra_countries:
                final_countries = extra_countries
                country_source = 'epmc_exact_doi_title_unique_full_author_name_explicit_country'
                counts['author_country_filled_epmc'] += 1
        conflict = bool(explicit and countries and set(explicit) != set(countries))
        counts['raw_country_vs_openalex_conflict'] += conflict
        counts['author_identity_filled'] += bool(entity_ids) and not original_entities
        counts['author_country_filled'] += bool(final_countries) and not countries
        result.append({'position': pos, 'name': name, 'author_id': author.get('author', {}).get('id'),
            'original_institution_ids': original_entities, 'original_countries': countries,
            'institution_ids_or_verified_local_entities': entity_ids, 'countries': final_countries,
            'nonpersonal_verified': nonpersonal, 'identity_source': status, 'country_source': country_source,
            'country_conflict_retained': conflict and not bool(verified), 'explicit_raw_countries': explicit,
            'country_conflict_disposition': 'superseded_by_nature_v1' if conflict and verified else 'unresolved' if conflict else 'not_detected',
            'raw_affiliation_strings': author.get('raw_affiliation_strings') or [],
            'status': 'identity_unresolved' if not entity_ids else 'inherited_or_evidenced_identity',
            'supplementary_addresses': supplementary,
            'supplementary_source': (secondary or {}).get('receipt') if supplementary else None,
            'nature_v1_slots': verified['slots'] if verified else None})
    return result, counts


def metrics(authors: list[dict[str, Any]], abstract: str | None) -> dict[str, int]:
    return {'papers': 1, 'abstract': int(bool(abstract)), 'author_records': len(authors),
        'no_authors': int(not authors),
        'any_institution': int(any(a['institution_ids_or_verified_local_entities'] for a in authors)),
        'every_author_institution': int(bool(authors) and all(a['institution_ids_or_verified_local_entities'] for a in authors)),
        'authors_with_institution': sum(bool(a['institution_ids_or_verified_local_entities']) for a in authors),
        'any_country': int(any(a['countries'] for a in authors)),
        'every_author_country': int(bool(authors) and all(a['countries'] for a in authors)),
        'authors_with_country': sum(bool(a['countries']) for a in authors)}


def main() -> None:
    states = [json.loads(p.read_text()) for p in sorted((OUT / 'shards').glob('*.json'))]
    if len(states) != 108 or any(s['status'] != 'complete' for s in states):
        raise RuntimeError('All 108 census shards must complete before release construction')
    donors = {r['doi']: r for r in map(json.loads, (NATURE / 'papers.jsonl').open())}
    with (OUT.parent / 'nature_type_verification/publisher_catalogue.csv').open(encoding='utf-8-sig') as stream:
        nature_catalogue = {r['doi']: r for r in csv.DictReader(stream)}
    names = country_names()
    for ambiguous in ['korea', 'georgia', 'congo']:
        names.pop(ambiguous, None)
    epmc, publisher, crossref = load()
    with sqlite3.connect(f'file:{OUT / "epmc_queue.sqlite"}?mode=ro', uri=True) as queue_db:
        secondary_status = dict(queue_db.execute('SELECT doi,status FROM queue'))
    with (OUT / 'journal_registry.csv').open(encoding='utf-8-sig') as stream:
        journals = {(r['cohort'], r['source_id']): r for r in csv.DictReader(stream)}
    with (OUT / 'journal_publisher_by_year.csv').open(encoding='utf-8-sig') as stream:
        annual_publishers = {(r['source_id'], int(r['year'])): r for r in csv.DictReader(stream)}
    database = OUT / 'identity_audit.sqlite'
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    connection.execute('CREATE TABLE identities (work_id TEXT PRIMARY KEY, doi TEXT, field TEXT, source_id TEXT)')
    totals: dict[tuple, Counter] = defaultdict(Counter)
    journal_totals: dict[tuple, Counter] = defaultdict(Counter)
    repairs: Counter = Counter()
    institutional_entities: dict[str, dict] = {}
    (OUT / 'papers').mkdir(exist_ok=True)
    (OUT / 'inherited_evidence').mkdir(exist_ok=True)
    reference(OUT.parent / 'nature_type_verification/publisher_catalogue.csv', OUT / 'inherited_evidence/nature_publisher_catalogue.csv')
    reference(OUT.parent / 'nature_type_verification/publisher_raw', OUT / 'inherited_evidence/nature_catalogue_raw')
    reference(NATURE / 'papers.jsonl', OUT / 'inherited_evidence/nature_v1_papers.jsonl')
    reference(NATURE / 'protocol.json', OUT / 'inherited_evidence/nature_v1_protocol.json')
    reference(NATURE / 'evidence', OUT / 'inherited_evidence/nature_v1_evidence')
    inventory = json.loads((NATURE / 'source_inventory.json').read_text())
    for item in inventory:
        if item.get('portable_path'):
            item['portable_path'] = item['portable_path'].replace('evidence/', 'inherited_evidence/nature_v1_evidence/', 1)
    write_json(OUT / 'inherited_evidence/nature_v1_source_inventory.json', inventory)
    for filename in ['source_entities.json','source_manifest.json','taxonomy.json','snapshot_audit.json']:
        reference(DEFAULT_OUT / filename, OUT / 'inherited_evidence' / ('stage0_' + filename))
    reference(DEFAULT_OUT / 'coverage_by_cohort_year.csv', OUT / 'inherited_evidence/stage0_coverage.csv')
    reference(DEFAULT_OUT / 'journal_registry.csv', OUT / 'inherited_evidence/stage0_journal_registry.csv')
    with gzip.open(OUT / 'repair_queue.jsonl.gz', 'wt', compresslevel=1) as queue, gzip.open(OUT / 'changes.jsonl.gz', 'wt', compresslevel=1) as changes, gzip.open(OUT / 'paper_status.csv.gz', 'wt', newline='', compresslevel=1) as status_stream:
        status_fields = ['work_id','doi','field','year_openalex','year_publisher_verified','source_id','journal_name',
                         'type_status','publisher_type','date_status','abstract_source','after_abstract',
                         'after_every_author_institution','after_every_author_country','metadata_final_status',
                         'publisher_comparison_group','historical_portfolio_status','epmc_attempt_status','publisher_attempt_status']
        status_writer = csv.DictWriter(status_stream, fieldnames=status_fields)
        status_writer.writeheader()
        for state in states:
            shard, field = state['shard'], state['field']
            writer = None
            for receipt_path in sorted((OUT / 'census_raw' / shard).glob('*.json.gz')):
                with gzip.open(receipt_path, 'rt') as stream:
                    receipt = json.load(stream)
                batch = []
                for raw_position, record in enumerate(receipt['response']['results']):
                    doi = doi_key(record.get('doi'))
                    source = (record.get('primary_location') or {}).get('source') or {}
                    donor = donors.get(doi) if source.get('id') == 'https://openalex.org/S137773608' else None
                    catalogue = nature_catalogue.get(doi) if source.get('id') == 'https://openalex.org/S137773608' else None
                    if donor and title_key(donor['title']) != title_key(record.get('title')):
                        donor = None
                        repairs['nature_v1_doi_title_conflict'] += 1
                    if catalogue and title_key(catalogue['title']) != title_key(record.get('title')):
                        catalogue = None
                        repairs['nature_catalogue_doi_title_conflict'] += 1
                    for author in record.get('authorships') or []:
                        for institution in author.get('institutions') or []:
                            if institution.get('id'):
                                institutional_entities[institution['id']] = institution
                    old_abstract = abstract_text(record.get('abstract_inverted_index'))
                    abstract = old_abstract or (donor or {}).get('abstract')
                    abstract_source = 'openalex_inverted_index' if old_abstract else 'nature_v1_exact_doi' if abstract else 'unresolved'
                    secondary = select_epmc(record, epmc.get(doi, []))
                    pub = publisher.get(doi, {})
                    pub_verified = pub.get('status') == 'publisher_doi_issn_verified'
                    pub_type_known = pub_verified and len(pub.get('publisher_type', [])) == 1 and bool(pub['publisher_type'][0])
                    supplement_source = None
                    if not abstract and pub_verified and pub.get('abstract'):
                        abstract, abstract_source = pub['abstract'], 'publisher_doi_issn_verified'
                        supplement_source = pub['raw_receipt']
                    deposited = select_epmc(record, [c for c in crossref.get(doi, []) if set(c['record']['issns']) & set(source.get('issn') or [source.get('issn_l')])])
                    if not abstract and deposited:
                        abstract = html_text(deposited['record']['abstractText'])
                        abstract_source = 'crossref_publisher_deposit_exact_doi_title_issn'
                        supplement_source = deposited['receipt']
                    if not abstract and secondary and secondary['record'].get('abstractText'):
                        abstract = html_text(secondary['record']['abstractText'])
                        abstract_source = 'epmc_exact_doi_title_secondary'
                        supplement_source = secondary['receipt']
                    authors, updates = enrich_authors(record.get('authorships') or [], donor, names, secondary)
                    updates['abstract_filled_publisher'] += abstract_source == 'publisher_doi_issn_verified'
                    updates['abstract_filled_crossref'] += abstract_source == 'crossref_publisher_deposit_exact_doi_title_issn'
                    updates['abstract_filled_epmc'] += abstract_source == 'epmc_exact_doi_title_secondary' 
                    updates['abstract_filled_nature_v1'] += abstract_source == 'nature_v1_exact_doi'
                    repairs.update(updates)
                    online_date = catalogue['publisher_date'] if catalogue else None
                    if pub_verified and len(pub.get('online_date', [])) == 1:
                        candidate_date = pub['online_date'][0].replace('/', '-')
                        try:
                            online_date = date.fromisoformat(candidate_date).isoformat()
                        except ValueError:
                            pass
                    publisher_year = donor['year'] if donor else int(online_date[:4]) if online_date else None
                    old_authors = [{**a, 'institution_ids_or_verified_local_entities': a['original_institution_ids'],
                                    'countries': a['original_countries']} for a in authors]
                    before, after = metrics(old_authors, old_abstract), metrics(authors, abstract)
                    journal = journals.get((field, source.get('id')), {})
                    membership_year = publisher_year or record['publication_year']
                    membership = annual_publishers.get((source.get('id'), membership_year), {})
                    row = {'work_id': record['id'], 'doi': doi or None, 'field': field,
                        'field_id': (record.get('primary_topic') or {}).get('field', {}).get('id'),
                        'topic_id': (record.get('primary_topic') or {}).get('id'),
                        'title': record.get('title'), 'publication_date_openalex': record.get('publication_date'),
                        'year_openalex': record['publication_year'], 'year_publisher_verified': publisher_year, 'publisher_online_date': online_date,
                        'source_id': source.get('id'), 'journal_name': source.get('display_name'),
                        'issn_l': source.get('issn_l'), 'abstract': abstract, 'abstract_source': abstract_source,
                        'supplement_source': supplement_source, 'publisher_evidence': 'inherited_evidence/nature_publisher_catalogue.csv' if catalogue else pub.get('raw_receipt') if pub_verified else None,
                        'authors_json': json.dumps(authors, ensure_ascii=False),
                        'epmc_attempt_status': secondary_status.get(doi, 'no_doi' if not doi else 'not_queued_without_initial_gap'),
                        'epmc_match_status': 'exact_doi_title_accepted' if secondary else 'conflict_or_title_mismatch' if epmc.get(doi) else 'no_accepted_record',
                        'publisher_attempt_status': pub.get('status', 'inherited_nature_v1' if donor or catalogue else 'outside_publisher_fetch_scope'),
                        'type_status': 'nature_v1_original_article_verified' if donor else 'nature_catalogue_type_verified' if catalogue else 'publisher_type_verified' if pub_type_known else 'unverified_candidate_not_final_research',
                        'publisher_type': 'Article' if donor else catalogue['publisher_type'] if catalogue else pub['publisher_type'][0] if pub_type_known else None,
                        'date_status': 'nature_v1_online_verified' if donor else 'nature_catalogue_online_verified' if catalogue else 'publisher_online_date_verified' if online_date else 'openalex_date_not_online_verified',
                        'nature_portfolio_official_current': journal.get('nature_portfolio_official_current') == 'True',
                        'publisher_label_status': journal.get('publisher_label_status', 'journal_not_in_prior_registry'),
                        'historical_portfolio_status': membership.get('status', 'unverified'),
                        'nature_portfolio_at_candidate_year': membership.get('nature_portfolio_year') or 'unknown',
                        'publisher_comparison_group': membership.get('comparison_group', 'unresolved'),
                        'annual_imprint': membership.get('imprint'), 'annual_publisher_evidence': membership.get('catalogue_file'),
                        'canonical_journal_key': membership.get('canonical_journal_key', source.get('id')),
                        'raw_receipt': str(receipt_path.relative_to(OUT)), 'raw_position': raw_position,
                        'author_list_may_be_truncated': len(authors) >= 100,
                        'openalex_updated_date': record.get('updated_date'),
                        'metadata_final_status': 'gaps_remain' if not after['abstract'] or not after['every_author_institution'] or not after['every_author_country'] else 'observed_fields_present_not_publisher_verified',
                        **{'before_' + k: v for k, v in before.items()}, **{'after_' + k: v for k, v in after.items()}}
                    for phase, values in [('before', before), ('after', after)]:
                        for year in (0, record['publication_year']):
                            totals[(field, year, phase)].update(values)
                        journal_totals[(field, source.get('id'), phase)].update(values)
                    connection.execute('INSERT INTO identities VALUES (?,?,?,?)', (record['id'], doi or None, field, source.get('id')))
                    if row['metadata_final_status'] == 'gaps_remain':
                        queue.write(json.dumps({'work_id': record['id'], 'doi': doi, 'field': field,
                            'missing_abstract': not bool(abstract), 'missing_author_identity': not after['every_author_institution'],
                            'missing_author_country': not after['every_author_country'],
                            'status': 'available_sources_checked_unresolved',
                            'epmc_attempt_status': row['epmc_attempt_status'], 'epmc_match_status': row['epmc_match_status'],
                            'publisher_attempt_status': row['publisher_attempt_status'], 'publisher_type': row['publisher_type'],
                            'raw_receipt': row['raw_receipt'], 'raw_position': raw_position}) + '\n')
                    if any(updates[k] for k in ['abstract_filled_nature_v1', 'abstract_filled_publisher', 'abstract_filled_epmc', 'abstract_filled_crossref', 'author_identity_filled', 'author_country_filled']):
                        changes.write(json.dumps({'work_id': record['id'], 'doi': doi, 'changes': dict(updates),
                            'raw_receipt': row['raw_receipt'], 'supplement_source': supplement_source, 'donor': 'inherited_evidence/nature_v1_papers.jsonl' if donor else None}) + '\n')
                    status_writer.writerow({k: row[k] for k in status_fields})
                    batch.append(row)
                if batch:
                    if set(batch[0]) != set(paper_schema().names):
                        raise RuntimeError('Paper schema would omit or invent fields')
                    table = pa.Table.from_pylist(batch, schema=paper_schema())
                    if writer is None:
                        writer = pq.ParquetWriter(OUT / 'papers' / f'{shard}.parquet', table.schema, compression='zstd')
                    writer.write_table(table)
            if writer:
                writer.close()
            connection.commit()
            print('built', shard, flush=True)
    duplicates = [dict(zip(['doi', 'work_count'], r)) for r in connection.execute('SELECT doi,count(*) FROM identities WHERE doi IS NOT NULL GROUP BY doi HAVING count(*)>1')]
    write_csv(OUT / 'duplicate_doi_queue.csv', duplicates)
    write_json(OUT / 'repairs_summary.json', dict(repairs))
    write_json(OUT / 'institution_registry.json', list(institutional_entities.values()))
    write_csv(OUT / 'coverage.csv', [{'field': k[0], 'year': k[1], 'phase': k[2], **v} for k, v in sorted(totals.items())])
    write_csv(OUT / 'journal_coverage.csv', [{'field': k[0], 'source_id': k[1], 'phase': k[2], **v} for k, v in sorted(journal_totals.items())])
    write_json(OUT / 'identity_audit.json', {'work_n': connection.execute('SELECT count(*) FROM identities').fetchone()[0],
        'duplicate_work_ids': 0, 'duplicate_doi_groups': len(duplicates),
        'duplicate_doi_excess': sum(r['work_count'] - 1 for r in duplicates),
        'missing_doi': connection.execute('SELECT count(*) FROM identities WHERE doi IS NULL').fetchone()[0]})
    connection.close()


def paper_schema() -> pa.Schema:
    strings = ['work_id','doi','field','field_id','topic_id','title','publication_date_openalex','source_id',
               'journal_name','issn_l','abstract','abstract_source','authors_json','type_status','publisher_type',
               'date_status','publisher_label_status','historical_portfolio_status','raw_receipt',
               'openalex_updated_date','metadata_final_status','supplement_source','publisher_evidence',
               'nature_portfolio_at_candidate_year','annual_imprint','annual_publisher_evidence','canonical_journal_key',
               'epmc_attempt_status','epmc_match_status','publisher_attempt_status','publisher_comparison_group','publisher_online_date']
    integers = ['year_openalex','year_publisher_verified','raw_position']
    integers += [phase + '_' + k for phase in ['before','after'] for k in metrics([], None)]
    return pa.schema([(k, pa.string()) for k in strings] + [(k, pa.int64()) for k in integers]
                     + [('nature_portfolio_official_current', pa.bool_()), ('author_list_may_be_truncated', pa.bool_())])


if __name__ == '__main__':
    main()
