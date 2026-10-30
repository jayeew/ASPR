"""Continue proven journal URL series after the current fixed platform batch."""
from __future__ import annotations

import fcntl
import json
import time

from .springer_remaining_collect import run as collect
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

PREFIXES = ('10.1557/s43579-', '10.1557/s43580-', '10.1245/s10434-', '10.1208/', '10.3103/')
FOLDER = ROOT / 'supplement/pleiades_route'


def run() -> None:
    with (FOLDER / 'dispatcher.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = connect()
        proofs = []
        counts = {}
        for prefix in PREFIXES:
            counts[prefix] = db.execute("SELECT count(*) FROM publisher_remaining_jobs WHERE route='springer' AND status='pending' AND doi LIKE ?", (prefix+'%',)).fetchone()[0]
            for doi, receipt in db.execute("SELECT doi,receipt FROM publisher_remaining_jobs WHERE route='springer' AND status='parsed' AND doi LIKE ?", (prefix+'%',)):
                record = json.loads((ROOT / receipt).read_text())
                assert record['http_status'] == 200 and doi in record['parsed']['doi']
                proofs.append({'doi':doi,'receipt':receipt,'journal_titles':record['parsed']['journal_titles']})
        db.close()
        routing = json.loads((FOLDER / 'additional_series_routing_proofs.json').read_text())
        proofs.extend(routing['proofs'])
        assert all(any(p['doi'].startswith(prefix) for p in proofs) for prefix in PREFIXES)
        atomic_json(FOLDER / 'verified_series_scope.json', {'pending_by_series':counts,'proofs':proofs,
                    'routing_only_not_annual_publisher_rule':True,'no_roster_expansion':True})
        atomic_json(FOLDER / 'verified_series_status.json', {'status':'waiting_for_current_collector','network_requests':0})
        while True:
            with (ROOT / 'supplement/springer_remaining/collector.lock').open('a') as current:
                try:
                    fcntl.flock(current,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    busy = True
                else:
                    busy = False
            if not busy:
                break
            time.sleep(30)
        state = json.loads((ROOT / 'supplement/springer_remaining/status.json').read_text())
        if state['status'] != 'fixed_selected_prefixes_processed_other_routes_pending':
            atomic_json(FOLDER / 'verified_series_status.json', {'status':'source_stop_preserved','source_state':state})
            return
        atomic_json(FOLDER / 'verified_series_status.json', {'status':'collecting_verified_journal_series'})
        collect('springer',16,doi_prefixes=PREFIXES)
        atomic_json(FOLDER / 'verified_series_status.json', {'status':'selected_series_collector_returned',
                    'source_state':json.loads((ROOT / 'supplement/springer_remaining/status.json').read_text()),
                    'final_task_closure':False})


if __name__ == '__main__':
    run()
