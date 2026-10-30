"""Apply previously unindexed cached affiliations to a fixed qualified-gap scope."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path

from .derived_refresh import refresh
from .secondary_author_cache import run as apply_cache
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/secondary_author_incremental'


def run(folder: Path = FOLDER, exclude_scope: Path | None = None) -> None:
    folder.mkdir(exist_ok=True)
    with (folder / 'consumer.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / 'application.json').exists():
            return
        scope_path = folder / 'scope.json'
        db = connect(); db.execute('PRAGMA busy_timeout=900000')
        if not scope_path.exists():
            db.execute('ATTACH DATABASE ? AS bulk', (str(ROOT / 'supplement/bulk_cache.sqlite'),))
            db.execute('CREATE TEMP TABLE prior_scope(doi TEXT PRIMARY KEY)')
            if exclude_scope is not None:
                previous = json.loads(exclude_scope.read_text())['papers']
                db.executemany('INSERT OR IGNORE INTO prior_scope VALUES(?)',
                               ((p['doi'],) for p in previous))
            rows = db.execute("""SELECT DISTINCT p.doi,p.work_id,b.path FROM papers p
                JOIN bulk.records b USING(doi) WHERE p.status='included_verified_research'
                AND (institution_gap=1 OR country_gap=1)
                AND NOT EXISTS (SELECT 1 FROM secondary_author_metadata s WHERE s.doi=p.doi)
                AND NOT EXISTS (SELECT 1 FROM attempts a JOIN tasks t USING(task_id)
                    WHERE t.kind='metadata' AND t.entity_key=p.work_id
                    AND a.source='epmc_incremental_author_cache' AND a.receipt=b.path)
                AND NOT EXISTS (SELECT 1 FROM prior_scope s WHERE s.doi=p.doi)""").fetchall()
            atomic_json(scope_path, {'papers': [{'doi': d, 'work_id': w, 'cache_receipt': p} for d,w,p in rows],
                        'excluded_previous_scope': str(exclude_scope.relative_to(ROOT)) if exclude_scope else None,
                        'scope': 'Fixed currently qualified gaps with previously unindexed cached Europe PMC records'})
        scope = json.loads(scope_path.read_text())['papers']; db.close()
        atomic_json(folder / 'status.json', {'status': 'applying_fixed_cache_scope',
                    'dois': len({p['doi'] for p in scope}), 'network_requests': 0})
        apply_cache(scope_dois={p['doi'] for p in scope})
        db = connect(); db.execute('PRAGMA busy_timeout=900000')
        for paper in scope:
            indexed = bool(db.execute('SELECT 1 FROM secondary_author_metadata WHERE doi=? AND receipt=?',
                                      (paper['doi'], paper['cache_receipt'])).fetchone())
            state = 'cached_affiliations_indexed_identity_and_author_match_required' if indexed else 'cache_no_usable_full_name_affiliation_statement'
            for tid, in db.execute("SELECT task_id FROM tasks WHERE kind='metadata' AND entity_key=?", (paper['work_id'],)).fetchall():
                if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',
                              (tid, 'epmc_incremental_author_cache', paper['cache_receipt'])).fetchone():
                    continue
                db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                           (tid, 'epmc_incremental_author_cache', state, paper['cache_receipt'],
                            json.dumps({'work_id': paper['work_id'], 'doi': paper['doi'], 'final_task_closure': False,
                                        'scope_receipt': str(scope_path.relative_to(ROOT)), 'network_requests': 0})))
                db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (tid,))
        db.commit(); db.close()
        result = json.loads((ROOT / 'supplement/secondary_author_incremental_summary.json').read_text())
        atomic_json(folder / 'application.json', result)
        if result.get('changed_papers'):
            from .official_rule_reuse import run as reuse
            reuse()
            refresh()
        atomic_json(folder / 'status.json', {'status': 'fixed_cache_scope_processed',
                    'application': result, 'final_task_closure': False})


if __name__ == '__main__':
    run()
