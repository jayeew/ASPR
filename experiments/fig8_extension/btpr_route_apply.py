"""Correct Biotechnology Progress acquisition routing without changing eligibility."""
from __future__ import annotations

import fcntl
import json
from collections import Counter
from datetime import datetime, timezone

from .supplement_queue import ROOT, attempt, connect, route
from .supplement_sources import atomic_json, publisher_url

FOLDER = ROOT / 'supplement/btpr_route'
RULE = 'supplement/btpr_route/rule.json'


def run() -> dict:
    FOLDER.mkdir(exist_ok=True)
    rule = {'scope': 'Existing 2023–2025 candidates in Biotechnology Progress with 10.1002/btpr. DOI',
        'official_sources': ['https://www.aiche.org/sbe/resources/publications/biotechnology-progress',
                             'https://aiche.onlinelibrary.wiley.com/journal/15206033',
                             'https://aiche.onlinelibrary.wiley.com/hub/journal/15206033/homepage/forauthors.html'],
        'source_review': 'Society states publication by Wiley on its behalf; official journal and author-guideline pages identify the Wiley host. The journal publishes both research and reviews, so host correction does not establish paper eligibility.',
        'retrieved_via': 'Web search returned official society and publisher page content, reviewed 2026-10-26',
        'route': 'wiley', 'host': 'aiche.onlinelibrary.wiley.com',
        'not_inferred': ['annual Nature Portfolio brand', 'original research type', 'online date', 'successful page access'],
        'failure_rule': 'A request to pubs.acs.org for this DOI is a wrong-host attempt and cannot close the Wiley publication path.'}
    atomic_json(FOLDER / 'rule.json', rule)
    result, records = Counter(), []
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        papers = db.execute("SELECT work_id,doi,journal,route,status FROM papers WHERE doi LIKE '10.1002/btpr.%'").fetchall()
        for wid, doi, journal, before, status in papers:
            if journal != 'Biotechnology Progress':
                raise ValueError('Unexpected journal within scoped route')
            target = publisher_url(doi, route('', doi))
            if before != 'wiley':
                db.execute('UPDATE papers SET route=? WHERE work_id=?', ('wiley', wid))
                db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                           (wid, 'acquisition_route', before, 'wiley', RULE, 'Official society and publisher host; no change to source publisher name or eligibility'))
                result['paper_routes_changed'] += 1
            task_ids = [r[0] for r in db.execute('SELECT task_id FROM task_papers WHERE work_id=?', (wid,))]
            wrong = []
            for ident in task_ids:
                task_row = db.execute('SELECT route,status FROM tasks WHERE task_id=?', (ident,)).fetchone()
                if task_row and task_row[0] == 'acs':
                    db.execute("UPDATE tasks SET route='wiley',updated_at=CURRENT_TIMESTAMP WHERE task_id=?", (ident,))
                    attempt(db, ident, 'publisher_route_correction', 'correct_host_identified_not_requested', RULE,
                            {'before_route': 'acs', 'after_route': 'wiley', 'url': target,
                             'status_preserved': task_row[1], 'request_made': False})
                    result['task_routes_changed'] += 1
                for receipt, in db.execute("SELECT receipt FROM attempts WHERE task_id=? AND source='publisher' AND status='access_restricted'", (ident,)):
                    source = json.loads((ROOT / receipt).read_text())
                    if source.get('doi') == doi and source.get('url', '').startswith('https://pubs.acs.org/'):
                        wrong.append(receipt)
            records.append({'work_id': wid, 'doi': doi, 'qualification_status': status, 'url': target,
                            'wrong_host_failure_receipts': sorted(set(wrong)),
                            'next_action': 'Assess cached Wiley/archive evidence and documented host access; do not count ACS failure as Wiley failure',
                            'final_task_closure': False})
        db.commit()
        db.close()
    result.update(scope=len(records), papers_with_wrong_host_failures=sum(bool(r['wrong_host_failure_receipts']) for r in records),
                  qualification_changes=0, network_requests=0)
    event = {'updated_at': datetime.now(timezone.utc).isoformat(), **result}
    with (FOLDER / 'applications.jsonl').open('a') as stream:
        stream.write(json.dumps(event) + '\n')
    atomic_json(FOLDER / 'current_scope.json', {'summary': event, 'records': records})
    print(json.dumps(event), flush=True)
    return event


if __name__ == '__main__':
    run()
