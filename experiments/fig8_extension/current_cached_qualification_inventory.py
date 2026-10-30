"""Rebuild pending primary-field priorities from the current ledger, without fetching."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone

from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def reviewed_constraint(group: dict) -> dict:
    """Attach retained source limits without closing tasks or suppressing new evidence."""
    journals = set(group['journals'])
    if journals == {'Quantum'} and group['kind'] == 'Paper':
        receipt = 'supplement/quantum_source_review/type_policy_review.json'
        action = 'Review article content or explicit native type; generic Paper includes reviews and software showcases.'
    elif journals == {'Physics Letters B'} and group['kind'] == 'Short communication':
        receipt = 'supplement/plb_short_communication_policy/assessment.json'
        action = 'New journal-specific type policy or article content needed; do not reuse FLA definition for SCO or repeat documented 404.'
    elif journals == {'Nuclear Physics B'} and group['status'] == 'pending_primary_online_date':
        receipt = 'supplement/elsevier_native_full_length/date_review_application.json'
        action = 'Inspect per-DOI retained date review before any request; existing Crossref response and PDF date checks must not be repeated unchanged.'
    elif journals == {'Acta Physica Sinica | 物理学报'} and group['status'] == 'pending_primary_online_date':
        receipt = 'supplement/acta_physica_native/online_date_field_review.json'
        action = 'Inspect per-DOI date review; print issue, acceptance/preprint and release timestamps cannot substitute for final online date.'
    else:
        return {}
    if not (ROOT / receipt).is_file():
        return {}
    return dict(review_receipt=receipt, next_action=action, final_task_closure=False,
                scope='Source-specific restriction, not exhaustion of all alternatives. Check individual records before dispatch.')


def run() -> dict:
    folder = ROOT / 'supplement/remaining_qualification_plan'
    db = sqlite3.connect(f'file:{ROOT / "supplement/tasks.sqlite"}?mode=ro', uri=True)
    db.execute('BEGIN')
    groups: dict[tuple, dict] = {}
    records = 0
    matched = set()
    for work_id, status, payload in db.execute("""SELECT p.work_id,p.status,e.payload
        FROM current_evidence_index e JOIN papers p USING(work_id)
        WHERE p.status LIKE 'pending%' AND e.kind='primary'"""):
        records += 1
        item = json.loads(payload)
        if item.get('identity') != 'matched':
            continue
        matched.add(work_id)
        key = (status, item.get('family', ''), item.get('kind', ''),
               tuple(item.get('categories') or []))
        group = groups.setdefault(key, dict(status=status, family=key[1], kind=key[2],
            categories=list(key[3]), ids=set(), journals=Counter(), example=item))
        if work_id not in group['ids']:
            group['journals'][' | '.join(item.get('journal_titles') or ['unknown'])] += 1
        group['ids'].add(work_id)
    pending = db.execute("SELECT count(*) FROM papers WHERE status LIKE 'pending%'").fetchone()[0]
    db.close()
    rows = []
    for group in groups.values():
        ids = group.pop('ids')
        group.update(papers=len(ids), work_ids=sorted(ids), journals=dict(group['journals']))
        group['retained_source_constraint'] = reviewed_constraint(group)
        rows.append(group)
    rows.sort(key=lambda item: (-item['papers'], item['family'], item['kind']))
    result = dict(updated_at=datetime.now(timezone.utc).isoformat(), pending_candidates=pending,
        primary_evidence_records=records, pending_with_matched_primary=len(matched), groups=rows,
        network_requests=0, qualification_changes=0, final_task_closure=False,
        scope='Current pending ledger snapshot; groups overlap across evidence. No original-inventory restriction. Inventory is not source closure or permission to repeat source requests.')
    atomic_json(folder / 'current_cached_label_inventory.json', result)
    return {key: value for key, value in result.items() if key != 'groups'}


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
