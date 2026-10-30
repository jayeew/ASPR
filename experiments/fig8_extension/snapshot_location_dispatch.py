"""Incrementally deduplicate snapshot locations against prior source work."""
from __future__ import annotations

import fcntl
import gzip
import json
import sqlite3
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .cached_secondary_dispatch import archive_pmcid
from .fig8b_qualification_decide import key
from .fig8b_qualification_build import plain
from .snapshot_location_recovery import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def consume(out: sqlite3.Connection, current: sqlite3.Connection,
            prior: list[sqlite3.Connection], holds: dict, manifest: Path) -> None:
    result = json.loads(manifest.read_text()); receipt = result['raw_receipt']
    if out.execute('SELECT 1 FROM consumed WHERE receipt=?', (receipt,)).fetchone():
        return
    with gzip.open(ROOT / receipt, 'rt') as stream:
        for position, line in enumerate(stream):
            record = json.loads(line); work = record['work']; wid = work['id']
            paper = current.execute('SELECT doi,title,status FROM papers WHERE work_id=?', (wid,)).fetchone()
            identity = record['identity']
            if not paper or not (paper[2].startswith('pending') or paper[2] == 'included_verified_research'):
                identity = 'outside_current_metadata_scope'
            elif (work.get('doi') or '').lower().removeprefix('https://doi.org/').removeprefix('http://doi.org/') != paper[0]:
                identity = 'doi_identity_review_required'
            elif (not work.get('title') or not paper[1]
                  or key(plain(work['title'])) != key(plain(paper[1]))):
                identity = 'title_identity_review_required'
            out.execute('INSERT OR IGNORE INTO records VALUES(?,?,?,?)', (wid, receipt, position, identity))
            locations = [*work.get('locations', []), work.get('best_oa_location'), work.get('primary_location')]
            for location in locations:
                if not isinstance(location, dict):
                    continue
                for name in ['landing_page_url', 'pdf_url']:
                    url = location.get(name) or ''; parsed = urlsplit(url)
                    if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
                        continue
                    if out.execute('SELECT 1 FROM links WHERE work_id=? AND url=?', (wid, url)).fetchone():
                        continue
                    host = parsed.hostname; pmcid = archive_pmcid(url, host)
                    evidence = {'raw_receipt': receipt, 'raw_position': position,
                        'source_file': record['source_file'], 'source_line': record['source_line'],
                        'source_record_sha256': record['source_record_sha256'], 'location_kind': name,
                        'is_oa': location.get('is_oa'), 'version': location.get('version'),
                        'visited': False, 'final_task_closure': False}
                    state = 'new_location_source_review_required'
                    if identity != 'work_id_doi_match':
                        state = identity
                    elif host in {'doi.org', 'dx.doi.org', 'www.doi.org'}:
                        matches = unquote(parsed.path).lstrip('/').casefold() == paper[0]
                        state = 'doi_resolver_existing_route' if matches else 'resolver_doi_identity_review_required'
                    elif any(db.execute('SELECT 1 FROM leads WHERE work_id=? AND url=?', (wid, url)).fetchone() for db in prior):
                        state = 'existing_location_queue_reused'
                    elif host in holds:
                        state = 'existing_host_hold_alternatives_pending'; evidence['hold'] = holds[host]
                    elif pmcid:
                        cache = ROOT / 'supplement/pmc_cloud/version_results' / (pmcid + '.1.json')
                        state = 'existing_archive_association_review_required' if cache.exists() else 'new_archive_metadata_lead'
                        evidence.update(pmcid=pmcid, version_receipt=str(cache.relative_to(ROOT)) if cache.exists() else '')
                    out.execute('INSERT INTO links VALUES(?,?,?,?,?)', (wid, url, host, state, json.dumps(evidence)))
    out.execute('INSERT INTO consumed VALUES(?)', (receipt,)); out.commit()


def run() -> None:
    with (FOLDER / 'dispatch.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out = sqlite3.connect(FOLDER / 'locations.sqlite', timeout=120)
        out.execute('PRAGMA journal_mode=WAL')
        out.executescript('CREATE TABLE IF NOT EXISTS consumed(receipt TEXT PRIMARY KEY);'
            'CREATE TABLE IF NOT EXISTS records(work_id TEXT,receipt TEXT,position INTEGER,identity TEXT,PRIMARY KEY(work_id,receipt));'
            'CREATE TABLE IF NOT EXISTS links(work_id TEXT,url TEXT,host TEXT,status TEXT,evidence TEXT,PRIMARY KEY(work_id,url));')
        current = connect()
        prior = [sqlite3.connect(f'file:{ROOT}/supplement/{name}/leads.sqlite?mode=ro', uri=True)
                 for name in ['cached_primary_url_leads', 'cached_secondary_publication_links']]
        holds = {}
        for rel in ['cached_primary_url_leads/host_hold_application.json', 'remaining_publisher_link_access/host_holds.json', 'hal_metadata/pdf_fronts/host_holds.json']:
            path = ROOT / 'supplement' / rel
            if path.exists():
                stored = json.loads(path.read_text())['hosts']
                pairs = stored.items() if isinstance(stored, dict) else ((host, 'Retained host restriction') for host in stored)
                for host, detail in pairs:
                    holds[host] = {'receipt': str(path.relative_to(ROOT)), 'detail': detail}
        while True:
            manifests = sorted((FOLDER / 'files').glob('*.json'))
            for path in manifests:
                consume(out, current, prior, holds, path)
            scanner = json.loads((FOLDER / 'status.json').read_text())
            completed = out.execute('SELECT count(*) FROM consumed').fetchone()[0]
            done = scanner['status'] == 'fixed_snapshot_scope_processed_locations_need_review' and completed == scanner['fixed_files']
            atomic_json(FOLDER / 'dispatch_status.json', {'status': 'fixed_locations_dispatched' if done else 'consuming_completed_snapshot_files',
                'completed_files': completed, 'outcomes': dict(out.execute('SELECT status,count(*) FROM links GROUP BY status')),
                'identity_outcomes': dict(out.execute('SELECT identity,count(*) FROM records GROUP BY identity')),
                'network_requests': 0, 'qualification_changes': 0, 'final_task_closure': False})
            if done:
                break
            time.sleep(20)
        current.close(); out.close()
        for db in prior:
            db.close()


if __name__ == '__main__':
    run()
