"""Lock a publisher-defined Nature research roster and enumerate metadata gaps."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .nature_types import OUT
from .stage0 import DEFAULT_OUT, NATURE, Client, write_csv, write_json
from .stage0_roster import abstract_text


def doi_key(value: Any) -> str:
    return str(value or '').lower().removeprefix('https://doi.org/').removeprefix('http://doi.org/').removeprefix('doi:')


def fetch_metadata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    client = Client(OUT / 'openalex')
    dois = sorted({row['doi'] for row in rows})
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(client.get, 'works', {'filter': 'doi:' + '|'.join(dois[start:start + 100]),
                   'per_page': 200}) for start in range(0, len(dois), 100)]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result['meta']['count'] > len(result['results']):
                raise RuntimeError('DOI batch exceeded one result page; must not truncate matches')
            records.extend(result['results'])
            print(f'Research metadata batches {i}/{len(futures)}', flush=True)
    unique = {record['id']: record for record in records}
    return list(unique.values())


def research_rows(catalogue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only publisher Article/Letter, never generic OpenAlex article or correspondence."""
    output = {}
    for row in catalogue:
        if row['publisher_type'] not in ['Article', 'Letter']:
            continue
        if row['publisher_year'] not in [2023, 2024, 2025]:
            continue
        previous = output.get(row['doi'])
        if previous and any(previous[key] != row[key] for key in ['publisher_date', 'publisher_type', 'title']):
            raise RuntimeError('Conflicting publisher research records')
        output[row['doi']] = row
    return sorted(output.values(), key=lambda r: (r['publisher_date'], r['doi']))


def enrich(row: dict[str, Any], matches: list[dict[str, Any]]) -> dict[str, Any]:
    records = [r for r in matches if ((r.get('primary_location') or {}).get('source') or {}).get('id') == 'https://openalex.org/' + NATURE]
    record = records[0] if len(records) == 1 else {}
    authors = record.get('authorships') or []
    author_count = record.get('authors_count')
    truncated = bool(record.get('is_authors_truncated')) or bool(author_count and author_count > len(authors))
    text = abstract_text(record.get('abstract_inverted_index'))
    institution_n = sum(bool(a.get('institutions')) for a in authors)
    countries_n = sum(bool(a.get('institutions')) and all(i.get('country_code') for i in a['institutions'])
                      for a in authors)
    status = 'unique_nature_doi_match' if record else ('ambiguous_doi_match' if matches else 'missing_openalex')
    if matches and not records:
        status = 'doi_source_mismatch'
    reasons = []
    if not record:
        reasons.append(status)
    if not text:
        reasons.append('abstract_missing')
    if institution_n == 0:
        reasons.append('all_institutions_missing')
    elif institution_n < len(authors) or truncated:
        reasons.append('partial_institutions_or_truncated_authors')
    if countries_n < len(authors) or not authors or truncated:
        reasons.append('incomplete_institution_country')
    if not record.get('primary_topic'):
        reasons.append('topic_missing')
    return {**row, 'article_type_verified': True, 'metadata_match_status': status,
        'work_id': record.get('id'), 'candidate_work_ids': [r['id'] for r in matches],
        'abstract_text': text, 'has_abstract': bool(text), 'has_institution': institution_n > 0,
        'openalex_language': record.get('language'),
        'observed_author_n': len(authors), 'reported_author_n': author_count,
        'authors_with_institution_n': institution_n, 'authors_with_institution_country_n': countries_n,
        'authors_truncated': truncated,
        'all_observed_authors_have_institution': bool(authors) and institution_n == len(authors) and not truncated,
        'complete_institution_country': bool(authors) and countries_n == len(authors) and not truncated,
        'is_retracted_openalex': record.get('is_retracted'),
        'include_primary': record.get('is_retracted') is not True,
        'retraction_status': 'api_reported' if isinstance(record.get('is_retracted'), bool) else 'unverified_missing_metadata',
        'openalex_type': record.get('type'), 'openalex_date': record.get('publication_date'),
        'openalex_year': record.get('publication_year'),
        'openalex_updated_date': record.get('updated_date'),
        'authorships_json': json.dumps(authors, ensure_ascii=False),
        'primary_topic_json': json.dumps(record.get('primary_topic'), ensure_ascii=False),
        'missing_reasons': reasons}


def attach_publisher_status(rows: list[dict[str, Any]]) -> None:
    notices = json.loads((OUT / 'publisher_notices.json').read_text())
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for notice in notices:
        by_target[notice['target_doi']].append(notice)
    for row in rows:
        links = by_target.get(row['doi'], [])
        retracted = row['title'].startswith('RETRACTED ARTICLE:') or any(n['notice_type'] == 'Retraction' for n in links)
        row['publisher_retracted'] = retracted
        row['publisher_notice_urls'] = [n['notice_url'] for n in links]
        row['include_primary'] = row['include_primary'] and not retracted
        if retracted:
            row['retraction_status'] = 'publisher_confirmed'


def save_rows(name: str, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows) if rows else pa.table({'doi': pa.array([], type=pa.string())})
    pq.write_table(table, OUT / (name + '.parquet'))
    if not rows:
        (OUT / (name + '.csv')).write_text('doi\n')
        return
    write_csv(OUT / (name + '.csv'), [{key: value for key, value in row.items()
        if key not in ['abstract_text', 'authorships_json', 'primary_topic_json']} for row in rows])


def summarize(rows: list[dict[str, Any]], name: str = 'research_coverage_by_year') -> list[dict[str, Any]]:
    summary = []
    for year in [0, 2023, 2024, 2025]:
        group = [r for r in rows if year == 0 or r['publisher_year'] == year]
        n = len(group)
        counts = {'research_n': n, 'metadata_matched_n': sum(r['metadata_match_status'] == 'unique_nature_doi_match' for r in group),
            'abstract_n': sum(r['has_abstract'] for r in group),
            'institution_n': sum(r['has_institution'] for r in group),
            'abstract_and_institution_n': sum(r['has_abstract'] and r['has_institution'] for r in group),
            'all_authors_institution_n': sum(r['all_observed_authors_have_institution'] for r in group),
            'all_authors_country_n': sum(r['complete_institution_country'] for r in group),
            'authors_truncated_n': sum(r['authors_truncated'] for r in group)}
        summary.append({'year': year, **counts, **{key.removesuffix('_n') + '_coverage': value / n if n else None
                                                  for key, value in counts.items() if key != 'research_n'}})
    write_csv(OUT / (name + '.csv'), summary)
    return summary


def reconcile_prior(catalogue: list[dict[str, Any]], research: list[dict[str, Any]]) -> dict[str, Any]:
    by_doi = {r['doi']: r for r in catalogue}
    research_dois = {r['doi'] for r in research}
    old = pq.read_table(DEFAULT_OUT / 'nature_candidate_roster.parquet').to_pylist()
    prior_dois = {doi_key(r['doi']) for r in old}
    resolution_path = OUT / 'prior_candidate_resolutions.json'
    resolutions = {r['doi']: r for r in json.loads(resolution_path.read_text())} if resolution_path.exists() else {}
    rows = []
    for row in old:
        doi = doi_key(row['doi'])
        official = by_doi.get(doi)
        decision = ('included_publisher_research' if doi in research_dois else
                    'analysis_separate_sensitivity_cohort' if official and official['publisher_type'] == 'Analysis' else
                    'excluded_nonprimary_type' if official and official['publisher_type'] not in ['Article', 'Letter']
                    else 'not_in_locked_roster_requires_date_or_identity_check')
        resolution = resolutions.get(doi, {})
        rows.append({'doi': doi, 'work_id': row['work_id'], 'title': row['title'],
            'old_api_year': row['year'], 'publisher_type': official['publisher_type'] if official else None,
            'publisher_date': official['publisher_date'] if official else resolution.get('publisher_date'),
            'decision': resolution.get('decision', decision),
            'canonical_doi': resolution.get('canonical_doi'), 'identity_evidence_url': resolution.get('evidence_url'),
            'catalogue_url': official['catalogue_url'] if official else None})
    write_csv(OUT / 'prior_candidate_decisions.csv', rows)
    save_rows('prior_candidates_unresolved', [r for r in rows if r['decision'].startswith('not_in')])
    write_csv(OUT / 'research_added_beyond_prior_candidates.csv', [{k: v for k, v in r.items()
        if k not in ['abstract_text', 'authorships_json', 'primary_topic_json']}
        for r in research if r['doi'] not in prior_dois])
    return {'old_candidate_n': len(old), 'decisions': dict(Counter(r['decision'] for r in rows)),
            'research_added_beyond_prior_candidates_n': len(research_dois - prior_dois)}


def main() -> None:
    catalogue = pq.read_table(OUT / 'publisher_catalogue.parquet').to_pylist()
    selected = research_rows(catalogue)
    analyses = [r for r in catalogue if r['publisher_type'] == 'Analysis']
    records = fetch_metadata(selected + analyses)
    by_doi: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_doi[doi_key(row.get('doi'))].append(row)
    enriched = [enrich(row, by_doi.get(row['doi'], [])) for row in selected]
    analysis_rows = [enrich(row, by_doi.get(row['doi'], [])) for row in analyses]
    attach_publisher_status(enriched + analysis_rows)
    save_rows('analysis_sensitivity_roster', analysis_rows)
    save_rows('publisher_research_all', enriched)
    included = [r for r in enriched if r['include_primary']]
    save_rows('research_roster_locked', included)
    save_rows('excluded_retracted_research', [r for r in enriched if not r['include_primary']])
    subsets = {'missing_abstract': [r for r in included if not r['has_abstract']],
        'missing_all_institutions': [r for r in included if not r['has_institution']],
        'incomplete_author_institutions': [r for r in included if not r['all_observed_authors_have_institution']],
        'incomplete_institution_country': [r for r in included if not r['complete_institution_country']],
        'missing_or_ambiguous_openalex': [r for r in included if r['metadata_match_status'] != 'unique_nature_doi_match'],
        'any_metadata_gap': [r for r in included if r['missing_reasons']]}
    for name, subset in subsets.items():
        save_rows(name, subset)
    coverage = summarize(included)
    summarize(included + [r for r in analysis_rows if r['include_primary']], 'article_plus_analysis_coverage_by_year')
    audit = {'publisher_research_before_retraction_filter_n': len(enriched), 'locked_research_n': len(included),
        'excluded_retracted_n': len(enriched) - len(included),
        'api_retracted_n': sum(r['is_retracted_openalex'] is True for r in enriched),
        'publisher_retracted_n': sum(r['publisher_retracted'] for r in enriched),
        'type_counts': dict(Counter(r['publisher_type'] for r in enriched)),
        'metadata_match_counts': dict(Counter(r['metadata_match_status'] for r in enriched)),
        'missing_list_sizes': {key: len(value) for key, value in subsets.items()},
        'coverage': coverage, 'prior_reconciliation': reconcile_prior(catalogue, included)}
    write_json(OUT / 'research_audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
