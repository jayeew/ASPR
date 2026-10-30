"""Incrementally apply published cloud JATS while preserving version-review holds."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from lxml import etree

from .derived_refresh import refresh
from .pmc_cloud_inventory import FOLDER
from .pmc_cloud_versions import STOP, fetch_metadata
from .publisher_delta_apply import apply_available
from .springer_author_cache import run as apply_authors
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata


def parsed_publication(raw: bytes, assessment: dict) -> dict:
    if hashlib.md5(raw).hexdigest() != assessment['md5']:
        raise ValueError('cloud_xml_checksum_conflict')
    parsed = jats_metadata(raw)
    if assessment['doi'].lower() not in {d.lower() for d in parsed['doi']}:
        raise ValueError('cloud_xml_doi_mismatch')
    return {**parsed, 'issns': parsed['issn']}


def acquire(item: tuple[Path, dict, dict]) -> tuple[str, dict]:
    from .pmc_cloud_front import acquire_front

    return acquire_front(item)


def reader_live() -> bool:
    for name in ['version_dispatch.lock', 'version_collector.lock']:
        with (FOLDER / name).open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
    return False


def apply_batch(paths: list[Path]) -> list[str]:
    started = time.monotonic()
    work = []; held = []; used = set()
    for path in paths:
        version = json.loads(path.read_text())
        candidates = [a for a in version['assessments']
                      if a['status'] == 'published_xml_candidate_identity_check_required']
        # Multiple matching Works for one version need explicit identity adjudication.
        if len(candidates) != 1:
            held.append((path, version)); continue
        candidate = candidates[0]
        if candidate['doi'] in used:
            continue
        used.add(candidate['doi']); work.append((path, version, candidate))
    with ThreadPoolExecutor(max_workers=64) as pool:
        outcomes = list(pool.map(acquire, work))
    jobs = [(doi, result['parsed_receipt']) for doi, result in outcomes if result.get('parsed_receipt')]
    fetched = time.monotonic()
    qualification = apply_available(supplied_jobs=jobs) if jobs else {}
    qualified = time.monotonic()
    db = connect()
    selected = [(wid, shard, receipt) for doi, receipt in jobs for wid, shard in db.execute(
        "SELECT work_id,shard FROM papers WHERE doi=? AND status='included_verified_research'", (doi,))]
    db.close()
    authors = apply_authors(selected_jobs=selected, workers=8, fill_abstract=True) if selected else {}
    if authors.get('papers_changed'):
        from .official_rule_reuse import run as reuse_entities
        reuse_entities()
        from .country_incremental import run as reuse_countries
        reuse_countries()
    handled = []
    for _, result in outcomes:
        if result['status'] == 'unrequested_source_stop':
            continue
        origin = result['version_result']; handled.append(origin)
        atomic_json(FOLDER / 'xml_results' / Path(origin).name, result)
    for path, version in held:
        handled.append(str(path.relative_to(ROOT)))
        atomic_json(FOLDER / 'xml_results' / path.name,
                    {'status': 'version_not_promoted_to_published_xml', 'version_result': str(path.relative_to(ROOT)),
                     'assessments': version['assessments'], 'final_task_closure': False})
    # Attach every reviewed version to its exact paper tasks; no final closure implied.
    db = connect(); db.execute('PRAGMA busy_timeout=900000')
    for origin in handled:
        version = json.loads((ROOT / origin).read_text())
        receipt = str((FOLDER / 'xml_results' / Path(origin).name).relative_to(ROOT))
        outcome = json.loads((ROOT / receipt).read_text())
        for assessment in version['assessments'] or version.get('paper_links', []):
            for tid, in db.execute("SELECT task_id FROM tasks WHERE (kind='qualification' AND entity_key=?) OR (kind='metadata' AND entity_key=?)",
                                   (assessment['doi'], assessment['work_id'])).fetchall():
                if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',
                              (tid, 'pmc_cloud_published_version_path', receipt)).fetchone():
                    continue
                db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                           (tid, 'pmc_cloud_published_version_path', outcome['status'], receipt,
                            json.dumps({'work_id': assessment['work_id'],
                                        'version_assessment': assessment.get('status', version['status']),
                                        'final_task_closure': False})))
                db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (tid,))
        db.commit()
    db.close()
    with (FOLDER / 'xml_events.jsonl').open('a') as out:
        out.write(json.dumps({'reviewed_versions': len(handled), 'qualification': qualification,
                              'authors': authors, 'source_stopped': STOP.is_set(),
                              'xml_workers': 64,
                              'elapsed_seconds': time.monotonic() - started,
                              'acquisition_seconds': fetched - started,
                              'qualification_seconds': qualified - fetched,
                              'authors_and_ledger_seconds': time.monotonic() - qualified}) + '\n')
    if jobs:
        refresh()
    return handled


def run(batch_size: int = 2000) -> None:
    with (FOLDER / 'xml_collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = FOLDER / 'xml_status.json'
        if marker.exists() and json.loads(marker.read_text()).get('source_stopped'):
            return
        for name in ['xml_parsed', 'xml_results']:
            (FOLDER / name).mkdir(exist_ok=True)
        done_path = FOLDER / 'xml_consumed.json'
        done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
        while True:
            pending = [p for p in (FOLDER / 'version_results').glob('*.json') if str(p.relative_to(ROOT)) not in done]
            pending.sort(key=lambda p: p.stat().st_mtime_ns)
            live = reader_live()
            version_marker = FOLDER / 'version_status.json'
            shard_done = version_marker.exists() and json.loads(version_marker.read_text()).get('status') == 'fixed_version_metadata_processed'
            if pending and (len(pending) >= batch_size or not live or shard_done):
                done.update(apply_batch(pending[:batch_size]))
                atomic_json(done_path, sorted(done))
                if STOP.is_set():
                    atomic_json(marker, {'status': 'source_stopped', 'source_stopped': True, 'reviewed_versions': len(done)})
                    return
                continue
            state = 'waiting_for_version_metadata' if live else 'available_version_metadata_consumed'
            atomic_json(marker, {'status': state, 'reviewed_versions': len(done), 'pending_versions': len(pending),
                                 'xml_workers': 64, 'final_task_closure': False})
            if not live:
                return
            time.sleep(30)


if __name__ == '__main__':
    run()
