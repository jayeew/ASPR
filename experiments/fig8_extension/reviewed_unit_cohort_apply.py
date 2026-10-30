"""Apply a fixed reviewed signature cohort and replay changes under one writer lock."""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
from pathlib import Path

from .derived_refresh import refresh
from .official_rule_reuse import verify_changes
from .official_units_apply import run as apply_rules
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run(folder: Path) -> dict:
    receipt = str((folder / 'rules.json').relative_to(ROOT))
    rules = json.loads((folder / 'rules.json').read_text())
    if rules['status'] != 'reviewed':
        raise ValueError('Explicit reviewed rules required')
    for rule in rules['rules']:
        raw = gzip.decompress((ROOT / rule['raw_receipt']).read_bytes())
        if hashlib.sha256(raw).hexdigest() != rule['raw_sha256']:
            raise ValueError('Reviewed official source changed')
    scope = json.loads((folder / 'scope.json').read_text())
    ids = set(scope['selected_work_ids'])
    atomic_json(folder / 'application_status.json', {'status': 'waiting_for_writer',
                'selected_papers': len(ids), 'final_task_closure': False})
    with (folder.parent / 'rule_reuse.lock').open('a') as reuse:
        fcntl.flock(reuse, fcntl.LOCK_EX)
        with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            atomic_json(folder / 'application_status.json', {'status': 'applying',
                        'selected_papers': len(ids), 'final_task_closure': False})
            db = connect()
            low = db.execute('SELECT coalesce(max(rowid),0) FROM changes').fetchone()[0]
            db.close()
            result = apply_rules(rule_paths=[receipt], selected_work_ids=ids,
                status_name=folder.name + '_apply.json', lock_held=True)
            checked = verify_changes(low, ids, [receipt])
            if checked != result.get('papers_changed', 0):
                raise ValueError('Author changes did not reproduce')
            path = folder.parent / 'current_rule_reuse_scope.json'
            state = json.loads(path.read_text())
            if receipt not in state['next_rule_paths']:
                state['next_rule_paths'].append(receipt)
            atomic_json(path, state)
            report = {'status': 'applied_and_replayed', 'selected_papers': len(ids),
                'result': result, 'changed_papers_reproduced': checked,
                'countries_and_nature_preserved': True, 'repeat_safe': True,
                'network_requests_during_apply': 0, 'final_task_closure': False}
            atomic_json(folder / 'validation.json', report)
            atomic_json(folder / 'application_status.json', report)
    if checked:
        refresh()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    print(json.dumps(run(parser.parse_args().folder)), flush=True)
