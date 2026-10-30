"""Recover exact DOI/title archive leads obscured by encoded inline markup."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import html as entities
import json
import re
import sqlite3
from collections import Counter, defaultdict

from lxml import html

from .fig8b_qualification_decide import key
from .supplement_queue import ROOT
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/secondary_markup_recovery'


def decoded_title(value: str) -> str:
    for _ in range(2):
        decoded = entities.unescape(value)
        if decoded == value:
            break
        value = decoded
    if '<' in value:
        value = html.fragment_fromstring(value, create_parent='div').text_content()
    return value


def title_key(value: str) -> str:
    return key(decoded_title(value))


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    with (FOLDER / 'discovery.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parts = FOLDER / 'parts'; parts.mkdir(exist_ok=True)
        scope_path = FOLDER / 'scope.json'
        if not scope_path.exists():
            db = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
            db.execute('ATTACH DATABASE ? AS bulk', (str(ROOT / 'supplement/bulk_cache.sqlite'),))
            grouped = defaultdict(list)
            query = """SELECT p.work_id,p.doi,p.title,p.journal,r.path,r.position FROM papers p
                JOIN bulk.records r ON r.doi=p.doi WHERE p.pmcids='[]'
                AND (p.status LIKE 'pending%' OR (p.status='included_verified_research'
                AND (p.abstract_gap=1 OR p.institution_gap=1 OR p.country_gap=1 OR p.author_cap=1)))"""
            for wid, doi, title, journal, path, position in db.execute(query):
                grouped[path].append({'work_id': wid, 'doi': doi, 'title': title,
                                      'journal': journal, 'position': position})
            db.close()
            atomic_json(scope_path, {'files': dict(grouped), 'scope': 'Existing current candidates with no associated PMCID and unresolved qualification or metadata only; cached records only.'})
        scope = json.loads(scope_path.read_text()); counts = Counter()
        for path, candidates in scope['files'].items():
            target = parts / (hashlib.sha256(path.encode()).hexdigest() + '.json')
            if target.exists():
                result = json.loads(target.read_text())
            else:
                payload = json.loads(gzip.decompress((ROOT / path).read_bytes()))
                records = payload.get('response', {}).get('resultList', {}).get('result', [])
                requested = set(payload.get('dois', [])); found = []; stats = Counter()
                for paper in candidates:
                    if paper['position'] >= len(records):
                        stats['index_position_unresolved'] += 1; continue
                    record = records[paper['position']]
                    doi = (record.get('doi') or '').lower(); pmcid = record.get('pmcid') or ''
                    if doi != paper['doi'] or doi not in requested or record.get('source') not in {'MED', 'PMC'}:
                        stats['identity_or_request_scope_unresolved'] += 1; continue
                    if not re.fullmatch(r'PMC\d+', pmcid):
                        stats['no_pmcid_in_cached_record'] += 1; continue
                    if not title_key(paper['title']) or title_key(record.get('title') or '') != title_key(paper['title']):
                        stats['substantive_title_difference_retained'] += 1; continue
                    found.append({**paper, 'pmcid': pmcid, 'secondary_title': record['title'],
                        'receipt': path, 'secondary_source': record['source'],
                        'pubtypes': record.get('pubTypeList', {}).get('pubType', []),
                        'secondary_online_date': record.get('electronicPublicationDate', ''),
                        'identity_basis': 'Exact DOI and exact alphanumeric title after HTML entity/inline markup decoding; not fuzzy matching',
                        'qualification_unchanged': True})
                    stats['archive_lead_recovered'] += 1
                result = {'receipt': path, 'stats': dict(stats), 'leads': found}
                atomic_json(target, result)
            counts.update(result['stats'])
            atomic_json(FOLDER / 'status.json', {'status': 'reading_cached_records',
                'fixed_files': len(scope['files']), 'counts': dict(counts), 'network_requests': 0,
                'qualification_changes': 0, 'final_task_closure': False})
        atomic_json(FOLDER / 'status.json', {'status': 'fixed_cache_discovery_processed',
            'fixed_files': len(scope['files']), 'counts': dict(counts), 'network_requests': 0,
            'qualification_changes': 0, 'final_task_closure': False})


def register_leads() -> None:
    """Attach exact cache associations to existing tasks without resolving them."""
    from .supplement_queue import connect

    with (FOLDER / 'discovery.lock').open('a') as discovery:
        fcntl.flock(discovery, fcntl.LOCK_EX)
        if json.loads((FOLDER / 'status.json').read_text())['status'] != 'fixed_cache_discovery_processed':
            raise ValueError('Cache discovery incomplete')
    attached = 0
    with (ROOT / 'supplement/checkpoint.lock').open('a') as checkpoint:
        fcntl.flock(checkpoint, fcntl.LOCK_EX)
        db = connect()
        try:
            for path in sorted((FOLDER / 'parts').glob('*.json')):
                for paper in json.loads(path.read_text())['leads']:
                    receipt = str(path.relative_to(ROOT))
                    detail = json.dumps({'work_id': paper['work_id'], 'doi': paper['doi'],
                        'pmcid': paper['pmcid'], 'cache_record_position': paper['position'],
                        'original_receipt': paper['receipt'], 'identity_basis': paper['identity_basis'],
                        'qualification_unchanged': True, 'final_task_closure': False}, sort_keys=True)
                    tasks = db.execute("SELECT task_id FROM tasks WHERE "
                        "(kind='qualification' AND entity_key=?) OR (kind='metadata' AND entity_key=?)",
                        (paper['doi'], paper['work_id'])).fetchall()
                    for tid, in tasks:
                        found = db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=? AND detail=?',
                            (tid, 'cached_secondary_exact_identity_lead', receipt, detail)).fetchone()
                        if found:
                            continue
                        db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                            (tid, 'cached_secondary_exact_identity_lead', 'archive_lead_identified_not_qualified', receipt, detail))
                        db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (tid,))
                        attached += 1
            db.commit()
        finally:
            db.close()
    atomic_json(FOLDER / 'ledger_status.json', {'status': 'recovered_leads_attached_to_existing_tasks',
        'new_attempt_records': attached, 'qualification_changes': 0, 'network_requests': 0,
        'final_task_closure': False})


if __name__ == '__main__':
    run()
