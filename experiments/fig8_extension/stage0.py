"""Audit local snapshot and materialize source-versioned Fig.8 metadata censuses."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sqlite3
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from gear.env import getenv

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = Path('/mnt/d/FabCitationData/openalex-snapshot')
DEFAULT_OUT = ROOT / 'outputs/fig8_data/metadata'
NATURE = 'S137773608'
PORTFOLIO = 'https://openalex.org/P4310319908'
FIELDS = {'biochemistry': '13', 'materials': '25', 'physics': '31'}
SCREEN = 'primary_location.source.type:journal,type:article|letter,is_retracted:false,is_paratext:false,language:en'
COHORTS = {'nature': f'primary_location.source.id:{NATURE}', **{
    name: f'primary_topic.field.id:{ident}' for name, ident in FIELDS.items()}}
FLAGS = {'candidate': '', 'abstract': 'has_abstract:true',
         'institution': 'authorships.institutions.id:!null',
         'country': 'authorships.countries:!null',
         'abstract_institution': 'has_abstract:true,authorships.institutions.id:!null',
         'doi': 'doi:!null'}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                         else value for key, value in row.items()} for row in rows)


def entity_id(value: Any, kind: str) -> str | None:
    """Preserve distinct OpenAlex namespaces, including numeric taxonomy IDs."""
    text = str(value or '').strip().rstrip('/')
    suffix = text.rsplit('/', 1)[-1]
    if kind in ('fields', 'subfields', 'domains'):
        return f'https://openalex.org/{kind}/{suffix}' if suffix.isdigit() else None
    prefix = {'works': 'W', 'sources': 'S', 'topics': 'T'}[kind]
    return f'https://openalex.org/{suffix}' if re.fullmatch(prefix + r'\d+', suffix) else None


class Client:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.key = (getenv('OPENALEX_API_KEY') or getenv('OPENALEX_API_KEYS').split(',')[0]).strip()
        self.local = threading.local()

    def get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        query = {'endpoint': endpoint, 'params': params}
        token = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()
        path = self.out / 'api_raw' / f'{token}.json'
        if path.exists():
            return json.loads(path.read_text())['response']
        session = getattr(self.local, 'session', None)
        if session is None:
            session = self.local.session = requests.Session()
        status: int | str = 'network'
        for attempt in range(4):
            try:
                response = session.get('https://api.openalex.org/' + endpoint,
                                       params={**params, **({'api_key': self.key} if self.key else {})}, timeout=45)
                status = response.status_code
                if response.ok:
                    payload = response.json()
                    write_json(path, {**query, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                                      'response': payload})
                    return payload
                if status not in (429, 500, 502, 503, 504):
                    break
            except (requests.RequestException, ValueError):
                status = 'network_or_json'
            time.sleep(2 ** attempt)
        raise RuntimeError(f'OpenAlex {endpoint} failed: {status}; query cache token {token}')

    def groups(self, filters: str, group: str) -> tuple[int, list[dict[str, Any]]]:
        cursor, seen = '*', set()
        rows: list[dict[str, Any]] = []
        counts_seen = set()
        while cursor:
            if cursor in seen:
                raise RuntimeError('Repeated API cursor')
            seen.add(cursor)
            payload = self.get('works', {'filter': filters, 'group_by': group, 'per_page': 200, 'cursor': cursor})
            counts_seen.add(payload['meta']['count'])
            rows.extend(payload.get('group_by') or [])
            cursor = payload['meta'].get('next_cursor')
        if len(counts_seen) != 1 or len(rows) != len({row['key'] for row in rows}):
            raise RuntimeError('Inconsistent group pagination; inspect cached responses')
        return counts_seen.pop(), rows


def taxonomy() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for kind in ['fields', 'subfields', 'topics']:
        result[kind] = []
        for path in sorted((SNAPSHOT / 'data' / kind).rglob('*.gz')):
            with gzip.open(path, 'rt') as stream:
                result[kind].extend(json.loads(line) for line in stream)
    return result


def snapshot_schema_probes(out: Path) -> None:
    files = sorted((SNAPSHOT / 'data/works').rglob('*.gz'))
    result = []
    for index in sorted({int(k * (len(files) - 1) / 11) for k in range(12)}):
        path = files[index]
        with gzip.open(path, 'rt') as stream:
            records = [json.loads(line) for _, line in zip(range(3), stream)]
        result.append({'file': str(path), 'first_three_years': [r.get('publication_year') for r in records],
            'field_present': [bool(r.get('primary_topic')) for r in records],
            'abstract_present': [bool(r.get('abstract_inverted_index')) for r in records],
            'authorships_present': [bool(r.get('authorships')) for r in records],
            'raw_record_keys': list(records[0]), 'scope': 'prefix schema probe; not representative coverage sample'})
    write_json(out / 'snapshot_schema_probes.json', result)


def snapshot_audit(out: Path) -> None:
    root = SNAPSHOT / 'data/works'
    manifest = json.loads((root / 'manifest').read_text())
    rows = []
    for entry in manifest['entries']:
        relative = entry['url'].split('s3://openalex/', 1)[1]
        path = SNAPSHOT / relative
        size = path.stat().st_size if path.exists() else None
        rows.append({'path': str(path), 'expected_bytes': entry['meta']['content_length'],
                     'actual_bytes': size, 'records_manifest': entry['meta']['record_count'],
                     'size_matches': size == entry['meta']['content_length']})
    actual = {str(path) for path in root.rglob('*.gz')}
    expected = {row['path'] for row in rows}
    db = Path('/home/jayee/workspace/FabCitation/openalex_snapshot_reference_check_results/analysis_state.db')
    with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as connection:
        prior = dict(connection.execute('SELECT key,value FROM metadata'))
    result = {'snapshot_path': str(SNAPSHOT), 'manifest_sha256': hashlib.sha256((root / 'manifest').read_bytes()).hexdigest(),
              'files': len(rows), 'compressed_bytes': sum(row['actual_bytes'] or 0 for row in rows),
              'manifest_records': sum(row['records_manifest'] for row in rows),
              'mismatched_files': sum(not row['size_matches'] for row in rows),
              'unlisted_files': sorted(actual - expected),
              'updated_partition_min': min(Path(row['path']).parent.name for row in rows),
              'updated_partition_max': max(Path(row['path']).parent.name for row in rows),
              'prior_index_metadata': {key: json.loads(value) for key, value in prior.items()},
              'scope': 'All manifest file sizes checked; no full decompression/CRC or row-level census performed.',
              'source_entities_present': (SNAPSHOT / 'data/sources').exists(),
              'institution_entities_present': (SNAPSHOT / 'data/institutions').exists()}
    write_csv(out / 'snapshot_files.csv', rows)
    write_json(out / 'snapshot_audit.json', result)
    write_json(out / 'taxonomy.json', taxonomy())


def exact_source(client: Client, name: str) -> dict[str, Any]:
    payload = client.get('sources', {'search': name, 'per_page': 100})
    matches = [row for row in payload['results'] if row['display_name'] == name and row['type'] == 'journal']
    if len(matches) != 1:
        raise RuntimeError(f'Nonunique journal mapping for {name}: {len(matches)}')
    return matches[0]


def restore_ids(client: Client) -> None:
    out = client.out
    tax = json.loads((out / 'taxonomy.json').read_text())
    maps = {kind: defaultdict(list) for kind in ['fields', 'topics']}
    for kind in maps:
        for row in tax[kind]:
            maps[kind][row['display_name']].append(row)
    path = ROOT / 'data/claim_graph/canonical_target_works.parquet'
    rows = pq.read_table(path).to_pylist()
    sources = {name: exact_source(client, name) for name in sorted({row['source_name'] for row in rows})}
    issues = []
    for row in rows:
        for kind, column, label in [('fields', 'field_id', 'field_name'), ('topics', 'primary_topic_id', 'primary_topic_name')]:
            matches = maps[kind].get(row[label], [])
            row[column] = entity_id(matches[0]['id'], kind) if len(matches) == 1 else None
            if row[column] is None:
                issues.append({'work_id': row['work_id'], 'issue': 'unmatched_' + column, 'value': row[label]})
        topic = maps['topics'].get(row['primary_topic_name'], [])
        row['subfield_id'] = topic[0]['subfield']['id'] if len(topic) == 1 else None
        if topic and topic[0]['field']['id'] != row['field_id']:
            issues.append({'work_id': row['work_id'], 'issue': 'topic_field_conflict', 'value': topic[0]['field']['id']})
        row['source_id'] = entity_id(sources[row['source_name']]['id'], 'sources')
        row['id_recovery_method'] = 'unique exact taxonomy name; unique exact API journal name'
        row['id_recovery_status'] = 'label_join_not_raw_work_revalidation'
    pq.write_table(pa.Table.from_pylist(rows), out / 'canonical_target_works_ids_restored.parquet')
    write_csv(out / 'restored_id_mapping.csv', [{key: row.get(key) for key in [
        'work_id', 'doi', 'source_name', 'source_id', 'field_name', 'field_id',
        'primary_topic_name', 'primary_topic_id', 'subfield_id', 'id_recovery_status']} for row in rows])
    write_csv(out / 'metadata_conflicts.csv', issues)
    write_json(out / 'id_recovery_audit.json', {'rows': len(rows), 'issues': len(issues),
               'original_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'original_modified': False,
               'missing_after': {key: sum(row[key] is None for row in rows) for key in ['source_id', 'field_id', 'primary_topic_id']},
               'sources': sources, 'limitation': 'IDs recovered through unique exact labels, not per-work API validation.'})


def filters(cohort: str, extra: str = '') -> str:
    return ','.join(value for value in ['publication_year:2023-2025', COHORTS[cohort], SCREEN, extra] if value)


def collect_aggregates(client: Client, workers: int) -> None:
    jobs = []
    for cohort, cohort_filter in COHORTS.items():
        jobs.extend((cohort, flag, filters(cohort, suffix), 'publication_year') for flag, suffix in FLAGS.items())
        broad = f'publication_year:2023-2025,{cohort_filter},primary_location.source.type:journal'
        jobs.append((cohort, 'all_types', broad, 'type'))
        jobs.append((cohort, 'field_breakdown', filters(cohort), 'primary_topic.field.id'))
        for flag in ['candidate', 'abstract', 'institution', 'abstract_institution']:
            jobs.append((cohort, 'journals_' + flag, filters(cohort, FLAGS[flag]), 'primary_location.source.id'))
    for year in range(2018, 2023):
        for flag in ['candidate', 'abstract']:
            query = ','.join(value for value in [f'publication_year:{year}', SCREEN, FLAGS[flag]] if value)
            jobs.append(('history', str(year) + '_' + flag, query, 'primary_topic.field.id'))
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(client.groups, query, group): (cohort, metric, query, group)
                   for cohort, metric, query, group in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            cohort, metric, query, group = futures[future]
            total, rows = future.result()
            results.append({'cohort': cohort, 'metric': metric, 'filter': query, 'group_by': group,
                            'total': total, 'rows': rows, 'group_sum': sum(row['count'] for row in rows)})
            write_json(client.out / 'aggregates_partial.json', results)
            print(f'aggregates {index}/{len(jobs)}: {cohort}/{metric}: {total}', flush=True)
    write_json(client.out / 'aggregates.json', results)


def collect_sources(client: Client, workers: int) -> None:
    aggregates = json.loads((client.out / 'aggregates.json').read_text())
    ids = sorted({row['key'].rsplit('/', 1)[-1] for item in aggregates if item['metric'] == 'journals_candidate'
                  for row in item['rows'] if row['key'] and row['key'] != 'unknown'})
    sources = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(client.get, 'sources', {'filter': 'openalex:' + '|'.join(ids[start:start + 100]),
                   'per_page': 100}) for start in range(0, len(ids), 100)]
        for index, future in enumerate(as_completed(futures), 1):
            sources.extend({key: row.get(key) for key in [
                'id', 'display_name', 'issn_l', 'issn', 'type', 'host_organization',
                'host_organization_name', 'host_organization_lineage',
                'host_organization_lineage_names', 'homepage_url', 'updated_date']}
                for row in future.result()['results'])
            if index % 20 == 0 or index == len(futures):
                print(f'source metadata {index}/{len(futures)}', flush=True)
    write_json(client.out / 'source_entities.json', sources)
    write_json(client.out / 'source_entity_coverage.json', {'requested': len(ids), 'returned': len(sources),
               'missing_ids': sorted(set(ids) - {row['id'].rsplit('/', 1)[-1] for row in sources})})


def collect_samples(client: Client, workers: int) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(client.get, 'works', {'filter': filters(cohort), 'sample': 100,
                   'seed': 20261021, 'per_page': 100}): cohort for cohort in COHORTS}
        for future in as_completed(futures):
            cohort = futures[future]
            write_json(client.out / f'sample_{cohort}.json', future.result())


def tables(client: Client) -> None:
    items = json.loads((client.out / 'aggregates.json').read_text())
    lookup = {(item['cohort'], item['metric']): item for item in items}
    coverage = []
    for cohort in COHORTS:
        for year in [0, 2023, 2024, 2025]:
            row: dict[str, Any] = {'cohort': cohort, 'year': year, 'status': 'API_screened_candidates_not_publisher_verified'}
            for metric in FLAGS:
                item = lookup[cohort, metric]
                row[metric + '_n'] = item['total'] if year == 0 else sum(r['count'] for r in item['rows'] if r['key'] == str(year))
            for metric in ['abstract', 'institution', 'country', 'abstract_institution', 'doi']:
                row[metric + '_coverage'] = row[metric + '_n'] / row['candidate_n'] if row['candidate_n'] else None
                row['missing_' + metric + '_n'] = row['candidate_n'] - row[metric + '_n']
            coverage.append(row)
    write_csv(client.out / 'coverage_by_cohort_year.csv', coverage)
    source_map = {row['id']: row for row in json.loads((client.out / 'source_entities.json').read_text())}
    registry = []
    for cohort in COHORTS:
        metrics = {flag: {r['key']: r['count'] for r in lookup[cohort, 'journals_' + flag]['rows']}
                   for flag in ['candidate', 'abstract', 'institution', 'abstract_institution']}
        for source_id, count in metrics['candidate'].items():
            source = source_map.get(source_id, {})
            lineage = source.get('host_organization_lineage') or []
            registry.append({'cohort': cohort, 'source_id': source_id, 'journal_name': source.get('display_name'),
                'issn_l': source.get('issn_l'), 'issn': source.get('issn'), 'source_type': source.get('type'),
                'host_organization': source.get('host_organization'), 'host_name': source.get('host_organization_name'),
                'publisher_lineage': lineage, 'nature_portfolio_current': PORTFOLIO in lineage if lineage else None,
                'nature_main': source_id == 'https://openalex.org/' + NATURE,
                'portfolio_at_publication': 'unverified', 'valid_from': None, 'valid_to': None,
                'evidence_url': source_id, 'candidate_n': count,
                **{flag + '_n': metrics[flag].get(source_id, 0) for flag in metrics if flag != 'candidate'}})
    write_csv(client.out / 'journal_registry.csv', registry)
    tax = json.loads((client.out / 'taxonomy.json').read_text())
    write_csv(client.out / 'field_registry.csv', [{
        'field_id': row['id'], 'field_name': row['display_name'], 'domain_id': row['domain']['id'],
        'panel_b': row['id'].rsplit('/', 1)[-1] in FIELDS.values()} for row in tax['fields']])
    write_csv(client.out / 'cohort_flow.csv', [{'cohort': item['cohort'], 'work_type': row['key'],
              'n_before_language_paratext_retraction_screen': row['count']}
              for item in items if item['metric'] == 'all_types' for row in item['rows']])
    write_json(client.out / 'group_reconciliation.json', [{key: item[key] for key in ['cohort', 'metric', 'total', 'group_sum']}
               for item in items if item['total'] != item['group_sum']])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--stage', choices=['local', 'aggregates', 'sources', 'samples', 'tables', 'all'], default='all')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    client = Client(args.out)
    if args.stage in ['local', 'all']:
        snapshot_audit(args.out)
        snapshot_schema_probes(args.out)
        restore_ids(client)
    if args.stage in ['aggregates', 'all']:
        collect_aggregates(client, args.workers)
    if args.stage in ['sources', 'all']:
        collect_sources(client, args.workers)
    if args.stage in ['samples', 'all']:
        collect_samples(client, args.workers)
    if args.stage in ['tables', 'all']:
        tables(client)


if __name__ == '__main__':
    main()
