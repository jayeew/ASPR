"""Consume exact observed publisher URLs without reopening unrelated source queues."""
from __future__ import annotations

import fcntl
import json
from collections import Counter

from .derived_refresh import refresh
from .publisher_delta_apply import apply_available
from .snapshot_location_recovery import FOLDER
from .snapshot_publisher_reuse import canonical
from .springer_author_cache import run as apply_authors
from .springer_remaining_collect import run as collect
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, publisher_url


def run() -> None:
    folder = FOLDER / 'publisher_fallback'
    with (folder / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scope = json.loads((folder / 'scope.json').read_text())
        rows = scope['papers']
        for row in rows:
            assert canonical(row['observed_url']) == canonical(publisher_url(row['doi'], row['route']))
        db = connect()
        for row in rows:
            current = db.execute('SELECT doi FROM papers WHERE work_id=?', (row['work_id'],)).fetchone()
            if current != (row['doi'],):
                raise ValueError('Candidate identity changed; scope requires review')
            db.execute("INSERT OR IGNORE INTO publisher_remaining_jobs(doi,route,status) VALUES(?,?,'pending')", (row['doi'], row['route']))
        db.commit(); db.close()
        results = {}
        for route in sorted({row['route'] for row in rows}):
            selected = {row['doi'] for row in rows if row['route'] == route}
            collect(route, selected_dois=selected)
            db = connect()
            jobs = []
            outcomes = Counter()
            for doi in sorted(selected):
                status, receipt = db.execute('SELECT status,receipt FROM publisher_remaining_jobs WHERE doi=?', (doi,)).fetchone()
                outcomes[status] += 1
                if status == 'parsed':
                    jobs.append((doi, receipt))
            db.close()
            author_jobs = []
            applied = apply_available(supplied_jobs=jobs, author_jobs_out=author_jobs)
            authors = apply_authors(selected_jobs=author_jobs, workers=8, fill_abstract=True) if author_jobs else {}
            if authors.get('papers_changed'):
                from .official_rule_reuse import run as reuse_entities
                reuse_entities()
            results[route] = {'fixed_dois': len(selected), 'outcomes': dict(outcomes),
                              'application': applied, 'author_application': authors}
            atomic_json(folder / 'status.json', {'routes': results, 'final_task_closure': False})
        refresh()
        atomic_json(folder / 'status.json', {'status': 'fixed_observed_publisher_scope_processed',
            'routes': results, 'final_task_closure': False, 'unresolved_retained': True})


if __name__ == '__main__':
    run()
