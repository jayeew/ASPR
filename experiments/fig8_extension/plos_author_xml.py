"""Fetch fixed PLOS byline gaps without duplicating the existing PMC queue."""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import requests
from lxml import etree

from .springer_author_cache import run as apply_authors
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_bytes, atomic_json, jats_metadata
from .three_journal_apply import read_csv


def run(mode: str = 'byline') -> dict:
    if mode not in {'byline', 'online_date', 'protocol_review', 'content_review'}:
        raise ValueError('Only existing byline or explicit online-date gaps are supported')
    folder = ROOT / 'supplement/plos_batch' / {
        'byline': 'author_xml', 'online_date': 'date_xml', 'protocol_review': 'protocol_xml', 'content_review': 'content_xml'}[mode]
    folder.mkdir(exist_ok=True)
    with (folder.parent / 'article_xml_collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scope_path = folder / 'scope.json'
        if not scope_path.exists():
            if mode == 'byline':
                prior = json.loads((folder.parent / 'author_cache/scope.json').read_text())
                selected = [r for r in prior['audit']
                            if r['next_source_status'] == 'primary_byline_source_discovery_required']
            elif mode == 'content_review':
                prior = json.loads((folder.parent / 'remaining_review_queue.json').read_text())
                reviewed = json.loads((folder.parent / 'protocol_scope_unresolved.json').read_text())
                closed = {r['work_id'] for r in reviewed['records']}
                selected = [r for r in prior if r['work_id'] not in closed]
            else:
                prior = json.loads((folder.parent / 'remaining_review_queue.json').read_text())
                selected = [r for r in prior if (
                    r['qualification_status'] == 'pending_primary_online_date' if mode == 'online_date'
                    else r['publisher_article_type'] == 'Lab Protocol')]
            atomic_json(scope_path, selected)
        scope = json.loads(scope_path.read_text())
        journals = {r['source_id']: r for r in read_csv(ROOT / 'fields/journal_registry.csv')}
        session = requests.Session()
        session.trust_env = False
        session.headers['User-Agent'] = 'Fig8MetadataResearch/1.0 (fixed missing byline metadata)'
        db = connect()
        jobs, audit = [], []
        last_request = 0.0
        validation = json.loads((folder.parent / 'author_xml/source_validation.json').read_text())
        for item in scope:
            doi, wid = item['doi'], item['work_id']
            row = db.execute('SELECT status,title,journal,source_id,shard FROM papers WHERE work_id=?', (wid,)).fetchone()
            expected = {'byline': 'included_verified_research', 'online_date': 'pending_primary_online_date',
                        'protocol_review': 'pending_primary_type', 'content_review': row[0] if row[0].startswith('pending') else 'pending'}[mode]
            if row[0] != expected:
                audit.append({'work_id': wid, 'status': 'no_longer_matches_requested_gap'})
                continue
            token = hashlib.sha256(doi.encode()).hexdigest()
            receipt = folder / (token + '.json')
            journal = {'pbio': 'plosbiology', 'pcbi': 'ploscompbiol', 'pgen': 'plosgenetics',
                       'ppat': 'plospathogens', 'pntd': 'plosntds', 'pone': 'plosone',
                       'pgph': 'globalpublichealth', 'pmed': 'plosmedicine',
                       'pdig': 'digitalhealth', 'pclm': 'climate', 'pcsy': 'complexsystems',
                       'pwat': 'water', 'pstr': 'sustainabilitytransformation'}.get(doi.split('.')[2])
            if not journal:
                audit.append({'work_id': wid, 'status': 'journal_download_route_unknown'})
                continue
            url = f'https://journals.plos.org/{journal}/article/file?id={doi}&type=manuscript'
            record = json.loads(receipt.read_text()) if receipt.exists() else (
                dict(validation) if doi == validation['doi'] else
                {'doi': doi, 'url': url, 'attempts': [], 'retrieved_at': datetime.now(timezone.utc).isoformat()})
            while record.get('http_status') != 200 and len(record['attempts']) < 3:
                if record.get('http_status') and record['http_status'] not in {500, 502, 503, 504}:
                    break
                time.sleep(max(0, 12.1 - (time.monotonic() - last_request)))
                last_request = time.monotonic()
                try:
                    response = session.get(url, timeout=(10, 45))
                    raw_path = folder / (token + f'.attempt{len(record["attempts"])+1}.xml.gz')
                    atomic_bytes(raw_path, gzip.compress(response.content, mtime=0))
                    record.update(http_status=response.status_code, raw_receipt=str(raw_path.relative_to(ROOT)),
                                  sha256=hashlib.sha256(response.content).hexdigest())
                    record['attempts'].append({'http_status': response.status_code, 'raw_receipt': record['raw_receipt']})
                except requests.RequestException as error:
                    record['attempts'].append({'error': type(error).__name__})
                atomic_json(receipt, record)
            atomic_json(receipt, record)
            entry = {'work_id': wid, 'doi': doi, 'receipt': str(receipt.relative_to(ROOT)), 'status': 'source_unavailable'}
            if record.get('http_status') in {401, 403, 429}:
                entry['status'] = 'source_restricted_remaining_unattempted'
                audit.append(entry)
                break
            if record.get('http_status') == 200:
                raw = gzip.decompress((ROOT / record['raw_receipt']).read_bytes())
                if hashlib.sha256(raw).hexdigest() != record['sha256']:
                    raise ValueError('PLOS raw source hash changed')
                try:
                    parsed = jats_metadata(raw)
                except (ValueError, etree.XMLSyntaxError) as error:
                    entry.update(status='parse_failed', reason=type(error).__name__)
                else:
                    paper = {'title': row[1], 'journal_name': row[2]}
                    if doi not in parsed.get('doi', []) or not signature_match(paper, parsed, set(json.loads(journals[row[3]]['issn']))):
                        entry['status'] = 'identity_not_confirmed'
                    else:
                        wrapper = folder / (token + '.author.json')
                        atomic_json(wrapper, {'doi': doi, 'source': 'publisher_jats', 'route': 'plos',
                            'url': url, 'parsed': parsed, 'raw_sha256': record['sha256'],
                            'raw_receipt': os.path.relpath(ROOT / record['raw_receipt'], ROOT / 'fields/qualification')})
                        jobs.append((wid, row[4], str(wrapper.relative_to(ROOT))))
                        entry['status'] = ('identity_confirmed_requires_author_application' if mode == 'byline'
                                           else 'identity_confirmed_requires_qualification_application')
                        if mode in {'protocol_review', 'content_review'}:
                            entry.update(status='primary_content_ready_requires_study_design_review',
                                         has_article_body=b'<body' in raw)
            audit.append(entry)
            atomic_json(folder / 'acquisition_status.json', {'fixed_scope': len(scope), 'processed': len(audit), 'audit': audit})
        db.close()
        atomic_json(folder / 'acquisition_status.json', {'fixed_scope': len(scope), 'processed': len(audit),
                    'unattempted': len(scope)-len(audit), 'audit': audit, 'phase': 'acquisition_finished'})
        if mode in {'protocol_review', 'content_review'}:
            result = {'fixed_scope': len(scope), 'matched_content_records': len(jobs),
                      'qualification_changes': 0, 'status': 'content_ready_requires_review'}
        elif mode == 'online_date':
            from .fig8b_qualification_build import electronic_date
            from .publisher_delta_apply import apply_available
            supplied = []
            for _, _, receipt in jobs:
                record = json.loads((ROOT / receipt).read_text())
                parsed = record['parsed']
                online, precision = electronic_date(parsed['dates'])
                record['parsed'] = {**parsed, 'issns': parsed['issn'],
                    'article_types': [parsed['article_type']], 'online_date': online,
                    'date_precision': precision}
                target = ROOT / receipt.replace('.author.json', '.qualification.json')
                atomic_json(target, record)
                supplied.append((record['doi'], str(target.relative_to(ROOT))))
            result = apply_available(supplied_jobs=supplied)
        else:
            result = apply_authors(selected_jobs=jobs)
        atomic_json(folder / 'application_status.json', result)
        return result


if __name__ == '__main__':
    from .derived_refresh import refresh
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['byline', 'online_date', 'protocol_review', 'content_review'], default='byline')
    print(json.dumps(run(parser.parse_args().mode)), flush=True)
    refresh()
