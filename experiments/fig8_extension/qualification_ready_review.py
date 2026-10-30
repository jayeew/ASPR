"""Clarify cached evidence tasks using current decisions without closing acquisition paths."""
from __future__ import annotations

import fcntl
import json
from collections import Counter
from datetime import datetime, timezone

from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

NEEDS = {
    'pending_primary_type': ['primary_research_type'],
    'pending_primary_online_date': ['explicit_online_date'],
    'pending_no_primary_type_date': ['primary_research_type', 'explicit_online_date'],
    'pending_special_subtype_review': ['journal_specific_type_or_content_scope'],
    'pending_primary_secondary_type_conflict': ['primary_secondary_type_conflict'],
    'pending_primary_identity_mismatch': ['publication_identity'],
    'pending_integrity_review': ['integrity_review'],
    'pending_conflicting_primary_types': ['primary_type_conflict'],
    'pending_conflicting_primary_online_years': ['online_year_conflict'],
    'pending_secondary_nonprimary_or_integrity_flag': ['secondary_type_or_integrity_flag'],
}


def run() -> dict:
    folder = ROOT / 'supplement/qualification_ready_review'
    folder.mkdir(exist_ok=True)
    counts = Counter()
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        rows = db.execute("SELECT task_id,entity_key,status,payload,reason,receipt FROM tasks WHERE kind='qualification' AND status='evidence_ready' AND active=1").fetchall()
        with (folder / 'events.jsonl').open('a') as log:
            for tid, doi, old_status, raw, reason, receipt in rows:
                states = {r[0] for r in db.execute('SELECT status FROM papers WHERE doi=?', (doi,))}
                if not states or not states <= NEEDS.keys():
                    counts['other_or_mixed_decisions_retained'] += 1
                    continue
                payload = json.loads(raw)
                fields = sorted({field for state in states for field in NEEDS[state]})
                new = {**payload, 'qualification_decision_states': sorted(states), 'missing_fields': fields,
                       'cache_review_state': 'Cached evidence already represented in current pending decision; not proof of qualification or exhausted alternatives'}
                after_reason = 'Current qualification remains pending: ' + ', '.join(fields) + '; acquisition paths not closed'
                event = {'task_id': tid, 'doi': doi, 'before': {'status': old_status, 'payload': payload, 'reason': reason},
                         'after': {'status': 'cached_evidence_review_required', 'payload': new, 'reason': after_reason},
                         'receipt': receipt, 'basis': 'Current papers.status joined by exact normalized DOI; no eligibility inference',
                         'at': datetime.now(timezone.utc).isoformat()}
                log.write(json.dumps(event, ensure_ascii=False) + '\n')
                log.flush()
                db.execute('UPDATE tasks SET status=?,payload=?,reason=?,updated_at=CURRENT_TIMESTAMP WHERE task_id=?',
                           ('cached_evidence_review_required', json.dumps(new), after_reason, tid))
                counts['tasks_clarified'] += 1
                counts.update(states)
            db.commit()
        db.close()
    result = {**counts, 'qualification_changes': 0, 'network_requests': 0, 'tasks_marked_complete': 0}
    atomic_json(folder / 'status.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    run()
