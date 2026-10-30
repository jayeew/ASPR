"""Apply cached native fronts and attach each fixed HAL path outcome to its task."""
from __future__ import annotations

import fcntl
import hashlib
import json
from collections import Counter
from urllib.parse import urlparse

from .hal_publisher_front_apply import run as apply_fronts
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    folder=ROOT/'supplement/hal_metadata/pdf_fronts'
    with (folder/'finisher.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (folder/'collector.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=json.loads((folder/'status.json').read_text())
            if state['status'] not in {'fixed_fronts_processed_review_required','source_stopped'}:
                raise RuntimeError('Collector not terminal; retain incomplete paths')
        apply_fronts()
        holds=json.loads((folder/'host_holds.json').read_text()) if (folder/'host_holds.json').exists() else {'hosts':[]}
        results=[]
        for job in json.loads((folder/'scope.json').read_text())['jobs']:
            path=folder/'receipts'/(hashlib.sha256(job['url'].encode()).hexdigest()+'.json')
            receipt=json.loads(path.read_text()) if path.exists() else {}
            status=receipt.get('review_status') or ('unattempted_on_restricted_host' if urlparse(job['url']).hostname in holds['hosts'] else 'unattempted_source_stop')
            for paper in job['papers']:
                results.append({'work_id':paper['work_id'],'doi':paper['doi'],'url':job['url'],'status':status,
                    'receipt':str(path.relative_to(ROOT)) if path.exists() else '',
                    'host_hold_receipt':str((folder/'host_holds.json').relative_to(ROOT)) if urlparse(job['url']).hostname in holds['hosts'] else '',
                    'final_task_closure':False})
        report=folder/'path_outcomes.jsonl'
        with report.open('w') as output:
            for row in results:output.write(json.dumps(row)+'\n')
        inserted=0
        with (ROOT/'supplement/checkpoint.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);db=connect()
            for row in results:
                detail=json.dumps(row,sort_keys=True)
                for tid, in db.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?",(row['doi'],)).fetchall():
                    if not db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=? AND detail=?',
                        (tid,'hal_pdf_metadata',str(report.relative_to(ROOT)),detail)).fetchone():
                        db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                            (tid,'hal_pdf_metadata',row['status'],str(report.relative_to(ROOT)),detail));inserted+=1
            db.commit();db.close()
        atomic_json(folder/'ledger_status.json',{'status':'fixed_source_outcomes_recorded','paper_paths':len(results),
            'outcomes':dict(Counter(row['status'] for row in results)),'new_attempts':inserted,'final_task_closure':False})


if __name__ == '__main__':
    run()
