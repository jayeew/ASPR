"""Correct AIP/IOP name collisions without inferring article eligibility or imprint."""
from __future__ import annotations

import fcntl
import json
import sqlite3
from collections import Counter
from urllib.parse import urlsplit

from .supplement_queue import ROOT, connect, route
from .supplement_sources import atomic_json, publisher_url

FOLDER = ROOT / 'supplement/aip_route'
RECEIPT = 'supplement/aip_route/rule.json'


def prepare() -> list[dict]:
    FOLDER.mkdir(exist_ok=True)
    atomic_json(FOLDER / 'rule.json', {
        'status': 'reviewed', 'scope': 'Existing candidates with DOI prefix 10.1063 or exact '
            'publisher names American Institute of Physics / AIP Publishing; acquisition dispatch only',
        'official_sources': [
            'https://publishing.aip.org/wp-content/uploads/2023/03/AIPP-Library-FAQs-03072023.pdf',
            'https://publishing.aip.org/about/news/aip-publishing-launches-new-digital-content-platform/'],
        'review': 'Official 2023 platform migration FAQ states pubs.aip.org replaces Scitation and '
            'Crossref destinations were updated so DOI links continue resolving. The April 19, 2023 '
            'announcement corroborates the platform. Read via official web search results 2026-10-26.',
        'route': 'aip', 'url_strategy': 'DOI resolver; never construct an IOP URL for an AIP DOI',
        'not_inferred': ['annual publisher membership', 'original-research status', 'online year',
                         'successful access at the correct publisher'],
        'wrong_host_rule': 'A failed IOP request does not establish an AIP access restriction; '
            'retain failure and assess applicable correct-host/batch/archive paths separately.'})
    if (FOLDER / 'scope.json').exists():
        return json.loads((FOLDER / 'scope.json').read_text())['records']
    db = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
    rows = db.execute("SELECT work_id,doi,publisher,route FROM papers WHERE doi LIKE '10.1063/%' "
        "OR lower(trim(publisher)) IN ('american institute of physics','aip publishing')").fetchall()
    records = []
    for wid, doi, publisher, previous in rows:
        if route(publisher, doi) != 'aip':
            raise ValueError('Prepared routing scope disagrees with dispatcher')
        records.append({'work_id': wid, 'doi': doi, 'publisher_name_preserved': publisher,
            'previous_route': previous, 'target_route': 'aip',
            'target_url': publisher_url(doi, 'aip'), 'final_task_closure': False})
    db.close()
    atomic_json(FOLDER / 'scope.json', {'records': records, 'network_requests': 0})
    return records


def wrong_host_receipt(receipt: str, doi: str) -> bool:
    if not receipt:
        return False
    path = ROOT / receipt
    if not path.is_file() or path.suffix != '.json':
        return False
    record = json.loads(path.read_text())
    return (record.get('doi', '').casefold() == doi.casefold()
            and urlsplit(record.get('url', '')).hostname == 'iopscience.iop.org')


def run() -> dict:
    records = prepare()
    atomic_json(FOLDER / 'status.json', {'status': 'waiting_for_writer', 'scope': len(records)})
    counts = Counter()
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        for r in records:
            wid, doi = r['work_id'], r['doi']
            paper = db.execute('SELECT doi,route FROM papers WHERE work_id=?', (wid,)).fetchone()
            if not paper or paper[0] != doi:
                raise ValueError('Candidate identity changed')
            if paper[1] != 'aip':
                db.execute('UPDATE papers SET route=? WHERE work_id=?', ('aip', wid))
                db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) '
                    'VALUES(?,?,?,?,?,?)', (wid, 'acquisition_route', paper[1], 'aip', RECEIPT,
                    'AIP is not IOP; correct dispatch using official DOI-resolution policy'))
                counts['paper_routes_changed'] += 1
            tasks = db.execute("SELECT t.task_id,t.route,t.status,t.receipt FROM tasks t "
                "JOIN task_papers l USING(task_id) WHERE l.work_id=? AND t.kind='qualification'",
                (wid,)).fetchall()
            for tid, old_route, old_status, old_receipt in tasks:
                if old_route == 'aip':
                    continue
                wrong = wrong_host_receipt(old_receipt, doi)
                status = 'wrong_host_attempt_correct_route_unchecked' if wrong else old_status
                detail = {'before_route': old_route, 'after_route': 'aip',
                    'before_status': old_status, 'after_status': status,
                    'previous_receipt': old_receipt, 'wrong_host_attempt': wrong,
                    'url': r['target_url'], 'request_made': False, 'final_task_closure': False}
                db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                    (tid, 'aip_route_correction', status, RECEIPT, json.dumps(detail)))
                db.execute('UPDATE tasks SET route=?,status=?,attempt_n=attempt_n+1,'
                    'updated_at=CURRENT_TIMESTAMP WHERE task_id=?', ('aip', status, tid))
                if wrong:
                    db.execute('UPDATE tasks SET reason=? WHERE task_id=?',
                        ('IOP request was routed to wrong host; correct AIP/source paths remain unconfirmed', tid))
                counts['qualification_task_routes_changed'] += 1
                counts['wrong_host_statuses_reopened'] += wrong
        db.commit()
        db.close()
    result = {'status': 'routes_corrected_other_source_paths_pending', 'scope': len(records),
        **counts, 'qualification_changes': 0, 'network_requests': 0, 'final_task_closure': False}
    atomic_json(FOLDER / 'status.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
