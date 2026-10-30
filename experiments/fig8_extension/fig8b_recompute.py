"""Standalone offline coverage check. Requires pyarrow; no repository imports."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

import pyarrow.parquet as pq


@contextmanager
def deterministic_gzip_text(path: Path) -> Iterator[TextIO]:
    with (path.open('wb') as raw,
          gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed,
          io.TextIOWrapper(compressed, encoding='utf-8', newline='') as stream):
        yield stream


def recompute(root: Path) -> dict:
    counts: dict[tuple, Counter] = defaultdict(Counter)
    types: Counter = Counter()
    statuses: Counter = Counter()
    sources: Counter = Counter()
    truncated, cap_remaining, extra_authors = 0, 0, 0
    repairs: Counter = Counter()
    date_conflicts, outside_window = 0, 0
    publisher_groups: Counter = Counter()
    remaining: dict[str, Counter] = defaultdict(Counter)
    country_conflicts = []
    conflict_papers = set()
    unmatched_authors: Counter = Counter()
    unmatched_papers: Counter = Counter()
    examples = {}
    seen = set()
    duplicate_ids = []
    with deterministic_gzip_text(root / 'repair_queue.jsonl.gz') as queue, deterministic_gzip_text(root / 'changes.jsonl.gz') as changes:
        for path in sorted((root / 'papers').glob('*.parquet')):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=500):
                for row in batch.to_pylist():
                    if row['work_id'] in seen:
                        duplicate_ids.append(row['work_id'])
                    seen.add(row['work_id'])
                    authors = json.loads(row['authors_json'])
                    paper_addresses = set()
                    paper_fills: Counter = Counter()
                    for author in authors:
                        identity_added = bool(author['institution_ids_or_verified_local_entities']) and not author['original_institution_ids']
                        country_added = bool(author['countries']) and not author['original_countries']
                        paper_fills['author_identity_filled'] += identity_added
                        paper_fills['author_country_filled'] += country_added
                        repairs['nature_v1_author_reused'] += author['identity_source'] == 'nature_v1_exact_doi_and_unique_author_name'
                        repairs['author_country_filled_from_explicit_address'] += country_added and author['country_source'] == 'all_observed_raw_addresses_explicit_terminal_country'
                        repairs['author_country_filled_epmc'] += country_added and author['country_source'] == 'epmc_exact_doi_title_unique_full_author_name_explicit_country'
                        repairs['raw_country_vs_openalex_conflict'] += bool(author['explicit_raw_countries'] and author['original_countries'] and set(author['explicit_raw_countries']) != set(author['original_countries']))
                        if not author['institution_ids_or_verified_local_entities']:
                            addresses = {re.sub(r'\s*[^\s,;]+@[^\s,;]+', '', a).strip() for a in (author['raw_affiliation_strings'] or author.get('supplementary_addresses', []))}
                            addresses.discard('')
                            unmatched_authors.update(addresses)
                            paper_addresses.update(addresses)
                            for address in addresses:
                                examples.setdefault(address, (row['work_id'], row['doi']))
                        if author['country_conflict_retained']:
                            conflict_papers.add(row['work_id'])
                            country_conflicts.append({'work_id':row['work_id'],'doi':row['doi'],'author_position':author['position'],
                                'retained_countries':json.dumps(author['countries']), 'explicit_raw_countries':json.dumps(author['explicit_raw_countries']),
                                'raw_receipt':row.get('authorships_source_receipt') or row['raw_receipt'],'status':'source_conflict_requires_verification'})
                    unmatched_papers.update(paper_addresses)
                    gaps = {'abstract': not bool(row['abstract']),
                            'author_identity': not authors or any(not a['institution_ids_or_verified_local_entities'] for a in authors),
                            'author_country': not authors or any(not a['countries'] for a in authors)}
                    if any(gaps.values()):
                        queue.write(json.dumps({'work_id':row['work_id'],'doi':row['doi'],'field':row['field'],
                            'missing_abstract':gaps['abstract'],'missing_author_identity':gaps['author_identity'],'missing_author_country':gaps['author_country'],
                            'status':'available_sources_checked_unresolved','epmc_attempt_status':row['epmc_attempt_status'],
                            'publisher_attempt_status':row['publisher_attempt_status'],'publisher_type':row['publisher_type'],
                            'raw_receipt':row.get('authorships_source_receipt') or row['raw_receipt']})+'\n')
                    added_abstract = bool(row['abstract']) and row['abstract_source'] != 'openalex_inverted_index'
                    source_key = {'publisher_doi_issn_verified':'abstract_filled_publisher','epmc_exact_doi_title_secondary':'abstract_filled_epmc',
                        'nature_v1_exact_doi':'abstract_filled_nature_v1','crossref_publisher_deposit_exact_doi_title_issn':'abstract_filled_crossref'}.get(row['abstract_source'])
                    if added_abstract and source_key:
                        repairs[source_key] += 1
                    repairs.update(paper_fills)
                    if added_abstract or any(paper_fills.values()):
                        changes.write(json.dumps({'work_id':row['work_id'],'doi':row['doi'],'changes':dict(paper_fills),
                            'abstract_added':added_abstract,'abstract_source':row['abstract_source'],'supplement_source':row['supplement_source'],
                            'authorships_source_receipt':row.get('authorships_source_receipt') or row['raw_receipt'],'paper_shard':str(path.relative_to(root))})+'\n')
                    remaining[row['field']].update({**gaps, 'any_metadata_gap': any(gaps.values())})
                    types[row['publisher_type'] or 'unverified'] += 1
                    statuses[row['type_status']] += 1
                    sources[row['abstract_source']] += 1
                    publisher_groups[row['publisher_comparison_group']] += 1
                    if row['year_publisher_verified']:
                        date_conflicts += row['year_publisher_verified'] != row['year_openalex']
                        outside_window += row['year_publisher_verified'] not in [2023,2024,2025]
                    truncated += row.get('listed_author_records',row['before_author_records']) >= 100
                    cap_remaining += row['author_list_may_be_truncated']
                    extra_authors += row.get('full_endpoint_authors_added',0)
                    for phase in ['before', 'after']:
                        institutions = [a['original_institution_ids'] if phase == 'before' else a['institution_ids_or_verified_local_entities'] for a in authors]
                        countries = [a['original_countries'] if phase == 'before' else a['countries'] for a in authors]
                        abstract = row['abstract_source'] == 'openalex_inverted_index' if phase == 'before' else bool(row['abstract'])
                        values = {'papers': 1, 'abstract': int(abstract), 'author_records': len(authors), 'no_authors': int(not authors),
                            'any_institution': int(any(institutions)), 'every_author_institution': int(bool(authors) and all(institutions)),
                            'authors_with_institution': sum(map(bool, institutions)), 'any_country': int(any(countries)),
                            'every_author_country': int(bool(authors) and all(countries)), 'authors_with_country': sum(map(bool, countries))}
                        for year in [0, row['year_openalex']]:
                            counts[(row['field'], year, phase)].update(values)
    with (root / 'coverage.csv').open(encoding='utf-8-sig') as stream:
        expected = {(r['field'], int(r['year']), r['phase']): {k: int(v) for k, v in r.items() if k not in ['field','year','phase']}
                    for r in csv.DictReader(stream)}
    mismatches = [{'key': key, 'expected': expected.get(key), 'actual': dict(value)}
                  for key, value in counts.items() if expected.get(key) != dict(value)]
    with deterministic_gzip_text(root / 'country_conflicts.csv.gz') as stream:
        writer = csv.DictWriter(stream, fieldnames=['work_id','doi','author_position','retained_countries','explicit_raw_countries','raw_receipt','status'])
        writer.writeheader()
        writer.writerows(country_conflicts)
    with deterministic_gzip_text(root / 'unmatched_affiliation_frequency.csv.gz') as stream:
        writer = csv.DictWriter(stream, fieldnames=['address','author_records','paper_records','example_work_id','example_doi','status'])
        writer.writeheader()
        for address, count in sorted(unmatched_authors.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow({'address':address,'author_records':count,'paper_records':unmatched_papers[address],
                'example_work_id':examples[address][0],'example_doi':examples[address][1],'status':'institution_identity_unresolved'})
    result = {'paper_count': len(seen), 'duplicate_ids': duplicate_ids, 'coverage_equal': not mismatches and set(expected) == set(counts),
        'mismatches': mismatches, 'publisher_types': dict(types), 'type_verification_statuses': dict(statuses),
        'abstract_sources': dict(sources), 'author_lists_at_least_100': truncated,
        'author_list_cap_risk_remaining':cap_remaining,'additional_author_records':extra_authors,
        'publisher_online_year_disagreements': date_conflicts, 'publisher_year_outside_window': outside_window,
        'publisher_comparison_groups': dict(publisher_groups),
        'remaining_by_field': {k:dict(v) for k,v in remaining.items()},
        'country_conflict_author_records': len(country_conflicts), 'country_conflict_papers': len(conflict_papers),
        'coverage': [{'field': k[0], 'year': k[1], 'phase': k[2], **v} for k, v in sorted(counts.items())]}
    previous = root / 'list_response_repairs_summary.json'
    if previous.exists():
        repairs['nature_catalogue_doi_title_conflict'] = json.loads(previous.read_text()).get('nature_catalogue_doi_title_conflict',0)
    repairs.setdefault('abstract_filled_crossref',0)
    (root / 'repairs_summary.json').write_text(json.dumps(dict(repairs),ensure_ascii=False,indent=2)+'\n')
    (root / 'recomputed_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    if duplicate_ids or not result['coverage_equal']:
        raise RuntimeError('Coverage or identity validation failed')
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path, nargs='?', default=Path('.'))
    args = parser.parse_args()
    result = recompute(args.root)
    print(json.dumps({k: v for k, v in result.items() if k != 'coverage'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
