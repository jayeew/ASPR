"""Cache-first acquisition planning and measured batch capability, without weaker eligibility."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

from .fig8b_qualification_build import match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_bytes, atomic_json, paced


def cache_index(root: Path) -> dict:
    """Index pointers and completed queries; keep raw evidence in its existing location."""
    started = time.monotonic()
    db = sqlite3.connect(root / 'supplement/bulk_cache.sqlite')
    db.executescript('''CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY,stamp TEXT,status TEXT);
        CREATE TABLE IF NOT EXISTS queries(doi TEXT,path TEXT,status TEXT,PRIMARY KEY(doi,path));
        CREATE TABLE IF NOT EXISTS records(doi TEXT,path TEXT,position INTEGER,PRIMARY KEY(doi,path,position));
        CREATE INDEX IF NOT EXISTS records_doi ON records(doi);
        CREATE INDEX IF NOT EXISTS queries_path ON queries(path);
        CREATE INDEX IF NOT EXISTS records_path ON records(path);''')
    seen = dict(db.execute('SELECT path,stamp FROM files'))
    changed = 0
    folders = [root / 'fields/epmc_raw', root / 'fields/qualification/epmc_raw']
    for folder in folders:
        for path in sorted(folder.glob('*.gz')):
            relative = str(path.relative_to(root))
            stat = path.stat()
            stamp = f'{stat.st_size}:{stat.st_mtime_ns}'
            if seen.get(relative) == stamp:
                continue
            with gzip.open(path, 'rt') as stream:
                receipt = json.load(stream)
            status = receipt.get('status', 'unknown')
            if relative in seen:
                db.execute('DELETE FROM queries WHERE path=?', (relative,))
                db.execute('DELETE FROM records WHERE path=?', (relative,))
            db.executemany('INSERT OR REPLACE INTO queries VALUES(?,?,?)',
                           [(d.lower(), relative, status) for d in receipt.get('dois', [])])
            if status == 'complete':
                requested = {d.lower() for d in receipt.get('dois', [])}
                db.executemany('INSERT OR IGNORE INTO records VALUES(?,?,?)',
                    [(item['doi'].lower(), relative, i) for i, item in enumerate(receipt.get('response', {}).get('resultList', {}).get('result', []))
                     if item.get('doi', '').lower() in requested])
            db.execute('INSERT OR REPLACE INTO files VALUES(?,?,?)', (relative, stamp, status))
            changed += 1
            if changed % 1000 == 0:
                db.commit()
    db.commit()
    result = {'files': db.execute('SELECT count(*) FROM files').fetchone()[0], 'changed_files': changed,
              'queried_dois': db.execute('SELECT count(DISTINCT doi) FROM queries').fetchone()[0],
              'returned_dois': db.execute('SELECT count(DISTINCT doi) FROM records').fetchone()[0],
              'seconds': time.monotonic() - started}
    db.close()
    atomic_json(root / 'supplement/bulk_cache_index.json', result)
    return result


def candidate_rows(root: Path) -> list[dict]:
    db = connect(root)
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute('''SELECT p.* FROM papers p WHERE
        (p.status LIKE 'pending%' OR (p.status='included_verified_research' AND EXISTS
        (SELECT 1 FROM task_papers l JOIN tasks t USING(task_id) WHERE l.work_id=p.work_id
         AND t.kind='metadata' AND t.active=1))) AND p.pmcids<>'[]'
        ORDER BY CASE WHEN p.status LIKE 'pending%' THEN 0 ELSE 1 END,p.priority''')]
    db.close()
    return rows


def plan(root: Path = ROOT) -> list[tuple[dict, str]]:
    """Never reschedule terminal receipts; choose one unread archive per DOI per pass."""
    q = root / 'fields/qualification'
    cached = {}
    for folder in ['jats_parsed', 'oai_parsed', 'supplement_parsed']:
        for path in (q / folder).glob('*.json'):
            r = json.loads(path.read_text())
            pmc = r.get('pmcid') or (path.stem if folder == 'jats_parsed' else '')
            if not pmc or r.get('source') == 'publisher':
                continue
            status = r.get('status', '')
            if status not in ['retryable_error', 'request_error'] or len(r.get('attempts', [])) >= 3:
                old = cached.get(pmc)
                if not old or status == 'parsed':
                    cached[pmc] = (status, str(path.relative_to(root)))
    counts = Counter()
    jobs, trace, used = [], [], set()
    chosen_dois = set()
    for row in candidate_rows(root):
        doi = row['doi'] or row['work_id']
        pmcs = sorted(set(json.loads(row['pmcids'])))
        selected = False
        for pmc in pmcs:
            if pmc in used:
                counts['duplicate_identifier_links'] += 1
                continue
            used.add(pmc)
            if pmc in cached:
                status, receipt = cached[pmc]
                counts['cached_' + status] += 1
                trace.append({'pmcid': pmc, 'doi': doi, 'action': 'reuse_or_preserve_unresolved', 'reason': status, 'receipt': receipt})
                continue
            if selected or doi in chosen_dois:
                counts['alternate_ids_deferred_until_replan'] += 1
                trace.append({'pmcid': pmc, 'doi': doi, 'action': 'deferred_alternate', 'reason': 'one source per DOI per pass', 'receipt': ''})
                continue
            selected = True
            chosen_dois.add(doi)
            counts['individual_requests_needed'] += 1
            trace.append({'pmcid': pmc, 'doi': doi, 'action': 'fetch_missing_or_conflicting', 'reason': row['status'], 'receipt': ''})
            jobs.append(({'doi': row['doi'], 'work_id': row['work_id'], 'pmcid': pmc, 'route': row['route']}, 'pmc'))
    path = root / 'supplement/individual_acquisition_plan.csv'
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['pmcid', 'doi', 'action', 'reason', 'receipt'])
        writer.writeheader()
        writer.writerows(trace)
    atomic_json(root / 'supplement/acquisition_plan.json', {'at': datetime.now(timezone.utc).isoformat(), **counts,
        'unique_candidate_pmcids': len(used), 'policy': 'Current eligibility and metadata gaps only; terminal source receipts retained without another attempt. Generic bulk types never resolve eligibility.'})
    return jobs


def benchmark(root: Path = ROOT, sample_size: int = 120) -> dict:
    """Compare retained individual-source timings with fresh bulk retrieval on the same DOIs."""
    db = connect(root)
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute('''SELECT p.*, x.pmcid FROM pilot x JOIN papers p USING(work_id)
        WHERE x.arm='pmc' ORDER BY p.field,p.year,p.priority''')]
    # Round robin over field/year strata, preserving the original fixed-seed pilot order.
    strata = {}
    for row in rows:
        strata.setdefault((row['field'], row['year']), []).append(row)
    sample = []
    while strata and len(sample) < sample_size:
        for group in list(strata):
            sample.append(strata[group].pop(0))
            if not strata[group]:
                del strata[group]
            if len(sample) == sample_size:
                break
    db.close()
    with (root / 'fields/journal_registry.csv').open(encoding='utf-8-sig') as stream:
        journals = {r['source_id']: r for r in csv.DictReader(stream)}
    index = sqlite3.connect(root / 'supplement/bulk_cache.sqlite')
    start = time.monotonic()
    pointers = {r['doi']: list(index.execute('SELECT path,position FROM records WHERE doi=?', (r['doi'],))) for r in sample}
    lookup_seconds = time.monotonic() - start
    index.close()
    q = root / 'fields/qualification'
    old = []
    for r in sample:
        url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/{r["pmcid"]}/fullTextXML'
        path = q / 'supplement_parsed' / (hashlib.sha256(url.encode()).hexdigest() + '.json')
        old.append(json.loads(path.read_text()))
    session = requests.Session()
    session.trust_env = False
    session.headers['User-Agent'] = 'Fig8MetadataResearch/1.0 (bounded batch capability benchmark)'
    counters = Counter()
    receipts = []
    recorded_network_seconds = 0.0
    started = time.monotonic()
    for offset in range(0, len(sample), 60):
        group = sample[offset:offset + 60]
        params = {'query': ' OR '.join(f'DOI:"{r["doi"]}"' for r in group), 'format': 'json', 'resultType': 'core', 'pageSize': 1000}
        url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
        token = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        path = root / 'supplement/batch_benchmark_raw' / (token + '.json.gz')
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            with gzip.open(path, 'rt') as stream:
                receipt = json.load(stream)
            counters['reused_benchmark_requests'] += 1
        else:
            paced(url)
            tick = time.monotonic()
            try:
                response = session.get(url, params=params, timeout=(10, 45))
                payload = response.json() if response.ok else {}
                receipt = {'url': url, 'params': params, 'dois': [r['doi'] for r in group], 'http_status': response.status_code,
                    'response': payload, 'seconds': time.monotonic() - tick, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                    'status': 'complete' if response.ok and payload.get('hitCount') == len(payload.get('resultList', {}).get('result', [])) else 'failed_or_truncated'}
            except (requests.RequestException, ValueError) as error:
                receipt = {'url': url, 'params': params, 'dois': [r['doi'] for r in group], 'status': 'request_error', 'error': str(error), 'seconds': time.monotonic() - tick}
            atomic_bytes(path, gzip.compress(json.dumps(receipt).encode()))
            counters['new_http_requests'] += 1
        if receipt['status'] == 'complete':
            destination = q / 'epmc_raw' / path.name
            if not destination.exists():
                os.link(path, destination)
        recorded_network_seconds += receipt.get('seconds', 0)
        receipts.append(str(path.relative_to(root)))
        if receipt['status'] != 'complete':
            continue
        returned = {}
        for item in receipt['response']['resultList']['result']:
            returned.setdefault(item.get('doi', '').lower(), []).append(item)
        for row in group:
            hits = returned.get(row['doi'], [])
            valid = []
            for item in hits:
                journal = item.get('journalInfo', {}).get('journal', {})
                evidence = {'family': 'secondary_epmc', 'title': item.get('title', ''), 'kind': 'Journal Article',
                    'issns': [journal.get('issn', ''), journal.get('essn', '')], 'journal_titles': [journal.get('title', '')]}
                if match({**row, 'journal_name': row['journal']}, evidence, journals[row['source_id']])['identity'] == 'matched':
                    valid.append(item)
            counters['identity_matched'] += bool(valid)
            counters['abstract_available'] += any(i.get('abstractText') for i in valid)
            counters['explicit_electronic_date_available'] += any(i.get('electronicPublicationDate') for i in valid)
            counters['author_affiliation_available'] += any(any(a.get('authorAffiliationDetailsList', {}).get('authorAffiliation') for a in i.get('authorList', {}).get('author', [])) for i in valid)
    result = {'at': datetime.now(timezone.utc).isoformat(), 'sample_dois': len(sample), **counters,
        'bulk_wall_seconds': time.monotonic() - started if counters['new_http_requests'] else None,
        'bulk_retained_network_seconds': recorded_network_seconds, 'cache_lookup_seconds': lookup_seconds,
        'already_cached_dois': sum(bool(v) for v in pointers.values()),
        'individual_retained_request_seconds': sum(r.get('elapsed_seconds', 0) for r in old),
        'individual_retained_http_attempts': sum(len(r.get('attempts', [])) for r in old),
        'individual_parsed': sum(r.get('status') == 'parsed' for r in old),
        'bulk_primary_research_confirmations': 0, 'receipts': receipts,
        'interpretation': 'Paired DOI sample; individual baseline uses retained timings, not new duplicate requests. Serial request-seconds are not old wall time. Secondary bulk type is not primary research confirmation; no end-to-end speedup claimed.'}
    previous_path = root / 'supplement/batch_benchmark.json'
    if previous_path.exists() and not counters['new_http_requests']:
        previous = json.loads(previous_path.read_text())
        result['initial_fresh_bulk_wall_seconds'] = previous.get('initial_fresh_bulk_wall_seconds', previous.get('bulk_wall_seconds'))
    atomic_json(previous_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['index', 'plan', 'benchmark'])
    args = parser.parse_args()
    if args.action == 'index':
        result = cache_index(ROOT)
    elif args.action == 'benchmark':
        result = benchmark(ROOT)
    else:
        result = {'individual_jobs': len(plan(ROOT))}
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
