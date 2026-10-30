"""Recover newly qualified short-format authors from already retained primary pages."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os

from lxml import etree

from .springer_author_cache import already_processed, parse_authors, source_digest, run as apply_authors
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata, parse_publisher
from .three_journal_apply import read_csv


def run(all_cached_native: bool = False, all_cached_jats: bool = False,
        selected_work_ids: set[str] | None = None) -> dict:
    folder = ROOT / 'supplement/journal_type_policies/author_cache'
    folder.mkdir(exist_ok=True)
    q = ROOT / 'fields/qualification'
    records = json.loads((folder.parent / 'application_audit.json').read_text())
    registry = {r['source_id']: r for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    db = connect()
    if all_cached_native or all_cached_jats:
        scope_path = folder / 'expanded_scope.json'
        if not scope_path.exists():
            selected = {r['work_id'] for r in records}
            selected.update(r[0] for r in db.execute("""SELECT DISTINCT p.work_id FROM papers p
                JOIN tasks t ON t.entity_key=p.work_id AND t.kind='metadata'
                JOIN current_evidence_index e ON e.work_id=p.work_id AND e.kind='primary'
                WHERE p.status='included_verified_research' AND t.active=1
                AND t.payload LIKE '%publisher_affiliation_structure%'
                AND json_extract(e.payload,'$.family')='nature_page'
                AND json_extract(e.payload,'$.identity')='matched'"""))
            atomic_json(scope_path, {'work_ids': sorted(selected), 'network_requests': 0,
                        'basis': 'Current qualified metadata tasks lacking primary signature structure, with matched native Nature pages already cached; original short-format queue retained'})
        scope = json.loads(scope_path.read_text())
        families = scope.setdefault('discovered_source_families', ['nature_page'])
        if all_cached_jats and 'archived_jats' not in families:
            additional = {r[0] for r in db.execute("""SELECT DISTINCT p.work_id FROM papers p
                JOIN tasks t ON t.entity_key=p.work_id AND t.kind='metadata'
                JOIN current_evidence_index e ON e.work_id=p.work_id AND e.kind='primary'
                WHERE p.status='included_verified_research' AND t.active=1
                AND t.payload LIKE '%publisher_affiliation_structure%'
                AND json_extract(e.payload,'$.family')='archived_jats'
                AND json_extract(e.payload,'$.identity')='matched'""")}
            scope['jats_candidates_at_discovery'] = len(additional)
            scope['jats_added_work_ids'] = sorted(additional - set(scope['work_ids']))
            scope['work_ids'] = sorted(set(scope['work_ids']) | additional)
            families.append('archived_jats')
            atomic_json(scope_path, scope)
        additions = {r['work_id'] for r in records} - set(scope['work_ids'])
        if additions:
            scope['work_ids'] = sorted(set(scope['work_ids']) | additions)
            scope.setdefault('newly_qualified_rule_additions', []).extend(sorted(additions))
            atomic_json(scope_path, scope)
        records = [{'work_id': wid} for wid in scope['work_ids']]
    if selected_work_ids is not None:
        records = [{'work_id': wid} for wid in sorted(selected_work_ids)]
    jobs = []
    audit = []
    for changed in records:
        wid = changed['work_id']
        row = db.execute('SELECT doi,title,journal,source_id,shard,status FROM papers WHERE work_id=?', (wid,)).fetchone()
        doi, title, journal, sid, shard, status = row
        if status != 'included_verified_research':
            continue
        paper = {'doi': doi, 'title': title, 'journal_name': journal, 'source_id': sid}
        candidates = [json.loads(v[0]) for v in db.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (wid,))]
        candidates.sort(key=lambda e: e.get('family') != 'nature_page')
        seen = set()
        attempts = []
        for evidence in candidates:
            family = evidence.get('family')
            if family not in {'nature_page', 'archived_jats'} or evidence.get('identity') != 'matched':
                continue
            original = evidence['receipt']
            if original in seen:
                continue
            seen.add(original)
            token = hashlib.sha256((wid + original).encode()).hexdigest()
            receipt = folder / (token + '.json')
            if receipt.exists() and already_processed(db, ROOT, wid, str(receipt.relative_to(ROOT))):
                attempts.append({'receipt': original, 'status': 'previously_processed_primary_byline'})
                break
            path = q / original
            if not path.exists():
                attempts.append({'receipt': original, 'status': 'raw_path_missing'})
                continue
            try:
                raw = gzip.decompress(path.read_bytes())
                native_text = raw.decode('utf-8') if family == 'nature_page' else raw
                parsed = parse_publisher(native_text) if family == 'nature_page' else jats_metadata(raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw)
                authors = parse_authors(native_text) if family == 'nature_page' else parsed
            except (OSError, EOFError, ValueError, etree.XMLSyntaxError) as error:
                attempts.append({'receipt': original, 'status': type(error).__name__})
                continue
            if doi not in parsed.get('doi', []) or not signature_match(paper, parsed, set(json.loads(registry[sid]['issn']))):
                attempts.append({'receipt': original, 'status': 'paper_identity_not_confirmed'})
                continue
            if not authors.get('authors'):
                attempts.append({'receipt': original, 'status': 'no_explicit_author_links'})
                continue
            record = {'doi': doi, 'source': 'publisher' if family == 'nature_page' else 'pmc',
                        'route': 'nature_cached_page' if family == 'nature_page' else 'pmc_archive_priority',
                        'url': evidence['url'], 'parsed': parsed, 'raw_receipt': os.path.relpath(path, q),
                        'raw_sha256': hashlib.sha256(raw).hexdigest(), 'qualification_evidence_id': evidence['id'],
                        'network_requests': 0}
            if family == 'nature_page':
                record['raw_text_encoding'] = 'utf-8'
            # Reuse prior processing of identical evidence even under another receipt path.
            digest = source_digest(record, record['raw_sha256'])
            prior = db.execute("SELECT receipt,status FROM springer_author_processed WHERE work_id=? AND digest=? AND status IN ('cache_processed','nature_review_preserved') LIMIT 1", (wid, digest)).fetchone()
            if prior:
                attempts.append({'receipt': original, 'status': 'previously_processed_primary_byline',
                                 'processed_receipt': prior[0], 'application_status': prior[1]})
                break
            if family == 'archived_jats':
                record['parsed'] = {k: v for k, v in parsed.items() if k in {'doi', 'title', 'issns', 'issn', 'journal_titles'}}
                record['parse_authors_from_raw'] = True
            atomic_json(receipt, record)
            jobs.append((wid, shard, str(receipt.relative_to(ROOT))))
            attempts.append({'receipt': original, 'status': 'matched_primary_byline_available', 'authors': len(authors['authors'])})
            break
        state = 'ready_for_author_application' if jobs and jobs[-1][0] == wid else 'no_confirmed_cached_primary_byline'
        if attempts and attempts[-1]['status'] == 'previously_processed_primary_byline':
            state = 'previously_processed_primary_byline'
        audit.append({'work_id': wid, 'doi': doi, 'attempts': attempts, 'status': state})
    db.close()
    atomic_json(folder / 'scope.json', {'jobs': jobs, 'audit': audit, 'network_requests': 0})
    result = apply_authors(selected_jobs=jobs)
    db = connect()
    for entry in audit:
        if not entry['attempts'] or entry['status'] == 'no_confirmed_cached_primary_byline':
            continue
        if entry['attempts'][-1].get('processed_receipt'):
            entry['application_status'] = entry['attempts'][-1]['application_status']
            entry['fields_complete_not_implied'] = True
            continue
        token = hashlib.sha256((entry['work_id'] + entry['attempts'][-1]['receipt']).encode()).hexdigest()
        receipt = folder / (token + '.json')
        record = json.loads(receipt.read_text())
        processed = db.execute('SELECT status FROM springer_author_processed WHERE work_id=? AND receipt=? AND digest=?',
                               (entry['work_id'], str(receipt.relative_to(ROOT)), source_digest(record, record['raw_sha256']))).fetchone()
        entry['application_status'] = processed[0] if processed else 'not_processed'
        entry['fields_complete_not_implied'] = True
    db.close()
    atomic_json(folder / 'scope.json', {'jobs': jobs, 'audit': audit, 'network_requests': 0})
    atomic_json(folder / 'status.json', {'scoped_papers': len(audit), 'cached_jobs': len(jobs), **result})
    return result


if __name__ == '__main__':
    from .derived_refresh import refresh
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-cached-native', action='store_true')
    parser.add_argument('--all-cached-jats', action='store_true')
    args = parser.parse_args()
    run(all_cached_native=args.all_cached_native, all_cached_jats=args.all_cached_jats)
    refresh()
