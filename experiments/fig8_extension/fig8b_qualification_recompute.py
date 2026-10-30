"""Offline qualification and coverage from one mutable paper table and its decisions."""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import io
import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

import pyarrow.parquet as pq


@contextmanager
def gzip_text(path: Path) -> Iterator[TextIO]:
    with path.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed, io.TextIOWrapper(compressed, encoding='utf-8', newline='') as stream:
        yield stream


def metrics(row: dict) -> dict:
    authors = json.loads(row['authors_json'])
    institutions = [a['institution_ids_or_verified_local_entities'] for a in authors]
    countries = [a['countries'] for a in authors]
    return {'papers': 1, 'abstract': int(bool(row['abstract'])), 'author_records': len(authors), 'no_authors': int(not authors),
            'any_institution': int(any(institutions)), 'every_author_institution': int(bool(authors) and all(institutions)),
            'authors_with_institution': sum(map(bool, institutions)), 'any_country': int(any(countries)),
            'every_author_country': int(bool(authors) and all(countries)), 'authors_with_country': sum(map(bool, countries))}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text('')
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def iter_papers(root: Path, research_only: bool = False) -> Iterator[tuple[Path, dict, dict]]:
    """Join current metadata and qualification without storing another paper copy."""
    paper_paths = sorted((root.parent / 'papers').glob('*.parquet'))
    decisions = {p.name for p in (root / 'decisions').glob('*.parquet')}
    if not paper_paths or {p.name for p in paper_paths} != decisions:
        raise ValueError('Paper/decision shards differ; rebuild qualification after adding papers')
    for path in paper_paths:
        rows = pq.read_table(root / 'decisions' / path.name).to_pylist()
        lookup = {r['work_id']: r for r in rows}
        if len(lookup) != len(rows):
            raise ValueError(f'Duplicate decision identity: {path.name}')
        for batch in pq.ParquetFile(path).iter_batches(batch_size=500):
            for paper in batch.to_pylist():
                decision = lookup.pop(paper['work_id'])
                if (paper['doi'], paper['source_id'], paper['field'], paper['title']) != (decision['doi'], decision['source_id'], decision['field'], decision['title']):
                    raise ValueError(f'Stale qualification identity: {paper["work_id"]}')
                if not research_only or decision['eligibility_status'] == 'included_verified_research':
                    yield path, paper, decision
        if lookup:
            raise ValueError(f'Decisions without papers: {path.name}')


def run(root: Path, refresh: bool = False) -> dict:
    spec = importlib.util.spec_from_file_location('fig8b_decide', root / 'decide.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evidence = defaultdict(list)
    secondary = {}
    with gzip.open(root / 'decision_evidence.jsonl.gz', 'rt') as stream:
        for line in stream:
            item = json.loads(line)
            evidence[item['work_id']].append(item)
    with gzip.open(root / 'secondary_evidence.jsonl.gz', 'rt') as stream:
        for line in stream:
            item = json.loads(line)
            secondary[item['work_id']] = item['records']
    counts = defaultdict(Counter)
    journals = defaultdict(Counter)
    publishers = defaultdict(Counter)
    with (root.parent/'journal_publisher_by_year.csv').open(encoding='utf-8-sig') as stream:
        annual={(r['source_id'],int(r['year'])):r['nature_portfolio_year'] for r in csv.DictReader(stream)}
    decision_errors, errors, roster = [], [], []
    ids, dois, classified = set(), set(), set()
    statuses = Counter()
    cap = yearshifts = unresolved = 0
    gap_counts = Counter(dict.fromkeys(['missing_abstract', 'missing_institution', 'missing_country', 'any_metadata_gap'], 0))
    gap_path = root / 'research_metadata_gaps.csv.gz'
    gap_temporary = gap_path.with_suffix('.gz.tmp')
    gap_columns = ['work_id', 'doi', 'field', 'missing_abstract', 'missing_institution', 'missing_country', 'paper_shard', 'metadata_root', 'abstract_source', 'author_source_receipt']
    with gzip_text(root / 'unresolved.csv.gz') as stream, gzip_text(gap_temporary) as gap_stream:
        gap_writer = csv.DictWriter(gap_stream, fieldnames=gap_columns)
        gap_writer.writeheader()
        writer = None
        for path, paper, decision in iter_papers(root):
            wid = paper['work_id']
            if wid in classified:
                errors.append('duplicate_paper:' + wid)
            classified.add(wid)
            status = decision['eligibility_status']
            statuses[status] += 1
            rerun = module.decide(evidence.get(wid, []), secondary.get(wid, []), decision['duplicate_of'])
            if rerun['status'] != status or rerun['online_year'] != decision['online_year']:
                decision_errors.append(wid)
            label,_=module.portfolio_label(annual.get((paper['source_id'],decision['online_year']),''),evidence.get(wid,[]),decision['online_year'])
            if label!=decision['nature_portfolio_at_verified_online_year']:decision_errors.append('portfolio:'+wid)
            values = metrics(paper)
            current = {**decision, **values, 'base_shard': '../papers/' + path.name}
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(current))
                writer.writeheader()
            if status.startswith('pending'):
                writer.writerow(current)
                unresolved += 1
            group = 'verified_research' if status == 'included_verified_research' else 'excluded' if status.startswith('excluded') else 'pending'
            year = decision['online_year'] if group == 'verified_research' else paper['year_openalex']
            for y in [0, year]:
                counts[(paper['field'], group, y)].update(values)
            if group != 'verified_research':
                continue
            if wid in ids or not paper['doi'] or paper['doi'] in dois:
                errors.append(wid)
            gaps = {'missing_abstract': not values['abstract'],
                    'missing_institution': not values['every_author_institution'],
                    'missing_country': not values['every_author_country']}
            gap_counts.update(gaps)
            if any(gaps.values()):
                gap_counts['any_metadata_gap'] += 1
                gap_writer.writerow({**{k: paper[k] for k in ['work_id', 'doi', 'field']}, **gaps,
                    'paper_shard': '../papers/' + path.name, 'metadata_root': '..',
                    'abstract_source': paper.get('abstract_source', ''),
                    'author_source_receipt': paper.get('authorships_source_receipt') or paper.get('raw_receipt', '')})
            ids.add(wid)
            dois.add(paper['doi'])
            cap += paper['author_list_may_be_truncated']
            yearshifts += year != paper['year_openalex']
            journals[(paper['field'], paper['source_id'], paper['journal_name'])].update(values)
            publishers[(paper['field'], decision['nature_portfolio_at_verified_online_year'])].update(values)
            roster.append({**{k: paper[k] for k in ['work_id', 'doi', 'field', 'source_id', 'journal_name', 'title']},
                           **{k: decision.get(k,'') for k in ['online_date', 'online_year', 'publisher_types', 'evidence_ids', 'nature_portfolio_at_verified_online_year','portfolio_evidence_ids']}})
    with (root / 'coverage.csv').open(encoding='utf-8-sig') as stream:
        expected = {(r['field'], r['population'], int(r['year'])): {k: int(v) for k, v in r.items() if k not in ['field', 'population', 'year']} for r in csv.DictReader(stream)}
    mismatches = [str(k) for k in set(expected) | set(counts) if expected.get(k) != (dict(counts[k]) if k in counts else None)]
    if decision_errors or errors:
        (root / 'recompute_failure.json').write_text(json.dumps({
            'status': 'failed_validation', 'decision_errors': decision_errors,
            'identity_errors': errors, 'coverage_not_published': True}, ensure_ascii=False, indent=2))
        raise ValueError(f'Qualification errors: {len(decision_errors)}; identity errors: {len(errors)}')
    if refresh:
        write_csv(root / 'coverage.csv', [{'field': k[0], 'population': k[1], 'year': k[2], **v} for k, v in sorted(counts.items())])
    write_csv(root / 'research_roster.csv', roster)
    write_csv(root / 'research_journal_coverage.csv', [{'field': k[0], 'source_id': k[1], 'journal_name': k[2], **v} for k, v in sorted(journals.items())])
    write_csv(root / 'research_publisher_coverage.csv', [{'field': k[0], 'nature_portfolio_at_online_year': k[1], **v} for k, v in sorted(publishers.items())])
    result = {'candidate_count': len(classified), 'verified_research_count': len(ids), 'unique_research_dois': len(dois), 'statuses': dict(statuses),
              'eligibility_equal': not decision_errors, 'decision_errors': decision_errors, 'coverage_equal': not mismatches or refresh,
              'mismatches': [] if refresh else mismatches, 'refreshed_coverage_groups': mismatches if refresh else [], 'identity_errors': errors,
              'unresolved_count': unresolved, 'verified_author_cap_remaining': cap, 'verified_online_year_shifts': yearshifts,
              'metadata_table': '../papers', 'update_policy': 'One current dataset; append/merge/replace metadata, then refresh derived outputs'}
    gap_temporary.replace(gap_path)
    (root / 'research_metadata_gaps_summary.json').write_text(json.dumps(
        {**gap_counts, 'qualified_denominator': len(ids)}, indent=2))
    (root / 'recomputed_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path, nargs='?', default=Path('.'))
    parser.add_argument('--refresh', action='store_true', help='Refresh coverage after an intentional metadata update')
    args = parser.parse_args()
    result = run(args.root, args.refresh)
    print(json.dumps(result))
    if not result['coverage_equal']:
        raise SystemExit('Coverage changed; review metadata updates and use --refresh')


if __name__ == '__main__':
    main()
