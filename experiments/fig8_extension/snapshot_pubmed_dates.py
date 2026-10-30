"""Route recovered PMID date leads through existing batch/cache date rules."""
from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict

from .pubmed_date_apply import run as apply_dates
from .pubmed_date_collect import run as collect
from .pubmed_date_review import run as review
from .snapshot_location_recovery import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    folder = FOLDER / 'pubmed_dates'
    folder.mkdir(exist_ok=True)
    scope = folder / 'scope.json'
    if not scope.exists():
        previous = defaultdict(list)
        old_scopes = [ROOT / 'supplement' / name / 'scope.json' for name in
                      ['pubmed_date_followup', 'pubmed_remaining_dates', 'pubmed_cached_identifier_dates']]
        old_scopes += list((ROOT / 'supplement/mdpi_oai/primary_followup/pubmed_incremental').glob('*/scope.json'))
        stops = []
        for path in old_scopes:
            if not path.exists():
                continue
            status = path.parent / 'status.json'
            if status.exists() and json.loads(status.read_text()).get('status') == 'source_stopped_unattempted_retained':
                stops.append(str(status.relative_to(ROOT)))
            for row in json.loads(path.read_text()).get('candidates', []):
                if row.get('status') == 'ready':
                    for pmid in row['pmids']:
                        previous[pmid].append(str(path.relative_to(ROOT)))
        current = connect()
        targets = dict(current.execute("SELECT work_id,doi FROM papers WHERE status='pending_primary_online_date'"))
        current.close()
        links = sqlite3.connect(f'file:{FOLDER}/locations.sqlite?mode=ro', uri=True)
        found = defaultdict(set)
        for wid, url in links.execute("SELECT work_id,url FROM links WHERE host='pubmed.ncbi.nlm.nih.gov' AND status='new_location_source_review_required'"):
            match = re.fullmatch(r'https?://pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/?(?:[?#].*)?', url)
            if wid in targets and match:
                found[wid].add(match[1])
        links.close()
        candidates, held = [], []
        for wid, pmids in found.items():
            row = {'doi': targets[wid], 'work_id': wid, 'pmids': sorted(pmids),
                   'receipts': ['supplement/snapshot_location_recovery/locations.sqlite']}
            reused = [r for pmid in pmids for r in previous[pmid]]
            if len(pmids) != 1 or reused or stops:
                held.append({**row, 'reason': 'previous_pmid_scope_reused' if reused else
                             'existing_source_stop_retained' if stops else 'ambiguous_pmid',
                             'previous_receipts': reused, 'source_stops': stops, 'final_task_closure': False})
            else:
                candidates.append({**row, 'status': 'ready'})
        by_pmid = defaultdict(set)
        for row in candidates:
            by_pmid[row['pmids'][0]].add(row['doi'])
        unique = []
        for row in candidates:
            if len(by_pmid[row['pmids'][0]]) != 1:
                held.append({**row, 'reason': 'pmid_shared_by_multiple_dois', 'final_task_closure': False})
            else:
                unique.append(row)
        atomic_json(scope, {'candidates': unique, 'eligibility_unchanged': True,
                    'scope': 'Recovered exact-host PMID leads for current primary-online-date gaps only; identity rechecked against returned DOI/title/journal before merge'})
        atomic_json(folder / 'held.json', {'records': held, 'final_task_closure': False})
    if (folder / 'completion.json').exists():
        return
    collect(folder=folder)
    if json.loads((folder / 'status.json').read_text())['status'] != 'collected_identity_and_date_review_required':
        return
    reviewed = review(folder=folder)
    applied = apply_dates(folder=folder)
    atomic_json(folder / 'completion.json', {'review': reviewed, 'application': applied,
                'final_task_closure': False})


if __name__ == '__main__':
    run()
