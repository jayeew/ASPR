"""Process newly observed, supported publisher pages after cache and archive routing."""
from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
from collections import Counter

from . import snapshot_publisher_fallback as fallback
from .snapshot_location_recovery import FOLDER
from .snapshot_publisher_reuse import cached_page, canonical
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, publisher_url

HOSTS = {'www.nature.com': 'nature', 'link.springer.com': 'springer',
         'www.frontiersin.org': 'frontiers'}


def discover() -> tuple[list[dict], dict]:
    links = sqlite3.connect(f'file:{FOLDER}/locations.sqlite?mode=ro', uri=True)
    current = connect()
    rows = links.execute("""SELECT work_id,url,host FROM links
        WHERE status='new_location_source_review_required'
        AND host IN ('www.nature.com','link.springer.com','www.frontiersin.org')""").fetchall()
    archive_ids = {r[0] for r in links.execute("""SELECT DISTINCT work_id FROM links
        WHERE status IN ('new_archive_metadata_lead','existing_archive_association_review_required',
        'existing_archive_outcome_reused')""")}
    links.close()
    selected, counts, seen = [], Counter(), set()
    for wid, url, host in rows:
        route = HOSTS[host]
        paper = current.execute('SELECT doi,journal,source_id,status,pmcids,abstract_gap FROM papers WHERE work_id=?', (wid,)).fetchone()
        if not paper or not (paper[3].startswith('pending') or
                             (paper[3] == 'included_verified_research' and paper[5])):
            counts['no_current_qualification_or_abstract_gap'] += 1
            continue
        doi, journal, sid, _, pmcids, _ = paper
        if ((route == 'nature' and not doi.startswith('10.1038/'))
                or (route == 'frontiers' and not doi.startswith('10.3389/'))):
            counts['unsupported_publisher_doi_route_preserved'] += 1
            continue
        if wid in archive_ids or json.loads(pmcids or '[]'):
            counts['archive_route_retained'] += 1
            continue
        if canonical(url) != canonical(publisher_url(doi, route)):
            counts['different_format_or_url_preserved'] += 1
            continue
        if current.execute('SELECT 1 FROM publisher_remaining_jobs WHERE doi=?', (doi,)).fetchone():
            counts['existing_publisher_queue_preserved'] += 1
            continue
        if cached_page(current, wid, url):
            counts['existing_native_page_preserved'] += 1
            continue
        if doi in seen:
            continue
        seen.add(doi)
        selected.append({'work_id': wid, 'doi': doi, 'journal': journal,
                         'source_id': sid, 'route': route, 'observed_url': url})
    current.close()
    return sorted(selected, key=lambda r: (r['route'], r['source_id'], r['doi'])), dict(counts)


def run() -> None:
    folder = FOLDER / 'publisher_incremental'
    folder.mkdir(exist_ok=True)
    with (folder / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for scope in sorted(folder.glob('*/*/publisher_fallback/scope.json')):
            status = scope.with_name('status.json')
            if status.exists() and json.loads(status.read_text()).get('status') == 'fixed_observed_publisher_scope_processed':
                continue
            fallback.FOLDER = scope.parent.parent
            fallback.run()
        papers, skipped = discover()
        token = hashlib.sha256(json.dumps(papers, sort_keys=True).encode()).hexdigest()[:20]
        run_folder = folder / token
        run_folder.mkdir(exist_ok=True)
        atomic_json(run_folder / 'discovery.json', {'papers': papers, 'skipped': skipped,
            'network_requests': 0, 'scope': 'Observed exact publisher URLs in fixed snapshot; current qualification or abstract gaps; no existing publisher job/native page/archive route.'})
        results = []
        # Operational chunks permit continuous application; every discovered DOI remains in scope.
        for route in sorted({r['route'] for r in papers}):
            selected = [r for r in papers if r['route'] == route]
            for start in range(0, len(selected), 500):
                state_path = ROOT / 'supplement' / (route + '_remaining') / 'status.json'
                if state_path.exists() and json.loads(state_path.read_text()).get('stop_reason'):
                    results.append({'route': route, 'unattempted': len(selected) - start,
                                    'status': 'existing_source_stop_preserved',
                                    'receipt': str(state_path.relative_to(ROOT))})
                    break
                batch = run_folder / f'{route}_{start:07d}'
                (batch / 'publisher_fallback').mkdir(parents=True, exist_ok=True)
                atomic_json(batch / 'publisher_fallback/scope.json', {'papers': selected[start:start + 500]})
                fallback.FOLDER = batch
                fallback.run()
                results.append({'route': route, 'papers': len(selected[start:start + 500]),
                                'receipt': str((batch / 'publisher_fallback/status.json').relative_to(ROOT))})
                atomic_json(folder / 'status.json', {'status': 'applying_fixed_discovered_scope',
                    'selected_dois': len(papers), 'batches': results, 'final_task_closure': False})
        atomic_json(folder / 'status.json', {'status': 'fixed_discovered_scope_processed_or_explicitly_held',
            'selected_dois': len(papers), 'batches': results, 'skipped': skipped,
            'discovery_receipt': str((run_folder / 'discovery.json').relative_to(ROOT)),
            'final_task_closure': False})


if __name__ == '__main__':
    run()
