"""Drain the fixed Nature batch once after its collector releases the lock."""
from __future__ import annotations

import fcntl
import json

from .derived_refresh import refresh
from .publisher_delta_apply import apply_available
from .springer_author_cache import run as apply_authors
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run(source_route: str = 'nature') -> dict:
    if source_route not in {'nature', 'sciopen'}:
        raise ValueError('Unreviewed phase route')
    folder = ROOT / 'supplement' / (source_route + '_remaining')
    with (folder / 'finisher.lock').open('a') as finisher:
        fcntl.flock(finisher, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (folder / 'collector.lock').open('a') as collector:
            fcntl.flock(collector, fcntl.LOCK_EX)
            state = json.loads((folder / 'status.json').read_text())
            if state['status'] in {'running', 'draining_inflight'}:
                raise RuntimeError('Collector ended without a terminal receipt; review before applying')
            if source_route == 'sciopen':
                from .sciopen_cache_reparse import run as reparse_sciopen
                reparse_sciopen()
            db = connect(ROOT)
            jobs = db.execute("SELECT doi,receipt FROM publisher_remaining_jobs "
                              "WHERE route=? AND status='parsed' AND applied=0", (source_route,)).fetchall()
            db.close()
            result = apply_available(supplied_jobs=jobs)
            db = connect(ROOT)
            db.executemany("UPDATE publisher_remaining_jobs SET applied=1 "
                           "WHERE doi=? AND receipt=? AND route=?", [(doi, receipt, source_route) for doi, receipt in jobs])
            db.commit()
            authors = db.execute("SELECT p.work_id,p.shard,j.receipt FROM papers p "
                "JOIN publisher_remaining_jobs j USING(doi) "
                "WHERE j.route=? AND j.status='parsed' AND j.applied=1 "
                "AND p.status='included_verified_research' "
                "AND NOT EXISTS (SELECT 1 FROM springer_author_processed a "
                "WHERE a.work_id=p.work_id AND a.receipt=j.receipt)", (source_route,)).fetchall()
            db.close()
            author_result = apply_authors(selected_jobs=authors, workers=8, fill_abstract=True) if authors else {}
            if author_result.get('papers_changed'):
                from .official_rule_reuse import run as reuse_entities
                entities = reuse_entities()
            else:
                entities = {}
            refresh()
            report = {'qualification': result, 'authors': author_result,
                      'shared_entities': entities, 'full_refresh': 'scheduled_only',
                      'network_requests': 0, 'status': 'fixed_' + source_route + '_batch_applied'}
            atomic_json(folder / 'application.json', report)
            print(json.dumps(report), flush=True)
            return report


if __name__ == '__main__':
    run()
