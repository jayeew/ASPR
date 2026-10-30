"""Fill fixed missing native-source routes after unresolved archive evidence."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .publisher_delta_apply import apply_available
from .springer_remaining_collect import run
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

F = ROOT/'supplement/target_500k/springer_unrouted'


def collect(route: str) -> None:
    folder = F/route
    folder.mkdir(exist_ok=True)
    with (folder/'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder/'application.json').exists():
            return
        rows = json.loads((F/'other_series_inventory.json').read_text())['records']
        selected = {r['doi'] for r in rows if r['route'] == route
                    and r['journal'] != 'Scientific American'}
        atomic_json(folder/'scope.json', {'dois': sorted(selected), 'count': len(selected),
            'source_rule': 'Existing public native adapter; source identity and type/year validation unchanged',
            'scope': 'Only existing candidates with no cached native or prior route receipt; unresolved PMC identifier does not satisfy source evidence requirement',
            'held': [r for r in rows if r['route'] == route and r['doi'] not in selected],
            'publisher_annual_labels_unchanged': True})
        if route == 'springer':
            process = Path('/proc/167954/cmdline')
            while process.exists() and b'springer_unrouted/collect_apply.py' in process.read_bytes():
                atomic_json(folder/'status.json', {'status':'waiting_for_current_native_collector', 'pid':os.getpid(), 'collector_pid':167954})
                time.sleep(5)
            if not (F/'application.json').exists():
                atomic_json(folder/'status.json', {'status':'prior_source_not_applied_review_required'})
                return
        with (ROOT/'supplement/checkpoint.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            db = connect()
            db.executemany('INSERT OR IGNORE INTO publisher_remaining_jobs(doi,route,status) VALUES(?,?,\'pending\')',[(d,route) for d in sorted(selected)])
            db.commit()
            db.close()
        atomic_json(folder/'status.json', {'status':'collecting', 'pid':os.getpid(), 'fixed_dois':len(selected)})
        run(route, 16, selected_dois=selected)
        db = connect()
        jobs = []
        counts = Counter()
        for doi in selected:
            status, receipt = db.execute('select status,receipt from publisher_remaining_jobs where doi=?',(doi,)).fetchone()
            counts[status] += 1
            if status == 'parsed' and receipt:
                jobs.append((doi,receipt))
        db.close()
        atomic_json(folder/'status.json', {'status':'applying', 'pid':os.getpid(), 'fixed_dois':len(selected), 'outcomes':dict(counts)})
        result = apply_available(supplied_jobs=jobs) if jobs else {}
        atomic_json(folder/'application.json', {'at':datetime.now(timezone.utc).isoformat(), 'fixed_dois':len(selected), 'outcomes':dict(counts), 'application':result, 'validation_pending':True})
        atomic_json(folder/'status.json', {'status':'fixed_source_applied_validation_required','outcomes':dict(counts)})
        print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('route',choices=['springer','nature'])
    collect(parser.parse_args().route)
