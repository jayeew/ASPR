"""Finite priority acquisition followed by an offline checkpoint; no automatic fallback loop."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

from .supplement_acquisition import plan
from .supplement_queue import ROOT, attempt, connect
from .supplement_sources import atomic_json, bounded_results


def ordered_jobs(jobs: list[tuple[dict, str]], priorities: dict[str, int]) -> list[tuple[dict, str]]:
    return sorted(jobs, key=lambda pair: (priorities.get(pair[0]['doi'], 999), pair[0]['doi']))


def run() -> None:
    folder = ROOT / 'supplement'
    config = json.loads((folder / 'journal_goal_budget.json').read_text())
    global_remaining = datetime.fromisoformat(config['deadline_utc']).timestamp() - time.time()
    deadline = time.monotonic() + max(0, min(config['acquisition_seconds'], global_remaining - 1800))
    priorities = json.loads((folder / 'priority_dois.json').read_text())
    jobs = ordered_jobs(plan(ROOT), priorities)
    scope = jobs[:config['max_source_items']]
    state = {'started_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(), 'status': 'running',
             'eligible_uncached_jobs': len(jobs), 'bounded_jobs': len(scope), 'completed': 0,
             'priority_jobs': sum(item['doi'] in priorities for item, _ in scope), 'source_counts': {}, 'windows': []}
    state_path = folder / 'journal_goal_status.json'
    atomic_json(state_path, state)
    db = connect(ROOT)
    workers = config['workers']
    counts: Counter = Counter()
    # First 40 measure response quality; later windows amortize scheduling and adapt on limits.
    offset = 0
    while offset < len(scope) and time.monotonic() < deadline:
        size = 40 if offset == 0 else 200
        batch = scope[offset:offset + size]
        started = time.monotonic()
        window: Counter = Counter()
        for item, source, result in bounded_results(batch, ROOT, workers, deadline):
            ident = hashlib.sha256(('qualification:' + (item['doi'] or item['work_id'])).encode()).hexdigest()[:32]
            if not result.get('reused_receipt'):
                attempt(db, ident, source, 'evidence_ready' if result['status'] == 'parsed' else result['status'],
                        'fields/qualification/' + result['parsed_receipt'],
                        {'reason': result.get('error', result['status']), 'priority_run': True, 'elapsed_seconds': result.get('elapsed_seconds')})
            counts[result['status']] += 1
            window[result['status']] += 1
            window['rate_limited'] += result.get('http_status') == 429
            state['completed'] += 1
            state['source_counts'] = dict(counts)
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            atomic_json(state_path, state)
        elapsed = time.monotonic() - started
        state['windows'].append({'n': sum(v for k, v in window.items() if k != 'rate_limited'), 'seconds': elapsed,
                                'counts': dict(window), 'workers': workers,
                                'parsed_per_hour': window['parsed'] * 3600 / max(elapsed, 1)})
        if window['rate_limited']:
            workers = max(1, workers // 2)
        if window['parsed'] == 0 and len(state['windows']) >= 3 and all(w['counts'].get('parsed', 0) == 0 for w in state['windows'][-3:]):
            state['stop_reason'] = 'three_zero_yield_windows_no_more_same_source_requests'
            break
        offset += len(batch)
    db.close()
    state['status'] = 'offline_checkpoint'
    state.setdefault('stop_reason', 'item_cap_or_scope_complete' if state['completed'] == len(scope) else 'time_or_stop_marker')
    atomic_json(state_path, state)
    modules = [('fig8b_qualification_build', []), ('supplement_acquisition', ['index']), ('supplement_apply', []),
               ('fig8b_qualification_recompute', [str(ROOT / 'fields/qualification'), '--refresh']),
               ('fig8b_qualification_report', []), ('supplement_queue', ['discover'])]
    logs = folder / 'logs'
    with (folder / 'checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for module, args in modules:
            state['checkpoint_module'] = module
            atomic_json(state_path, state)
            with (logs / ('journal_goal_' + module + '.log')).open('a') as stream:
                result = subprocess.run([sys.executable, '-m', 'experiments.fig8_extension.' + module, *args], stdout=stream, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                state.update(status='checkpoint_failed', failed_module=module, returncode=result.returncode)
                atomic_json(state_path, state)
                return
        fcntl.flock(lock, fcntl.LOCK_UN)
    subprocess.run([sys.executable, '-m', 'experiments.fig8_extension.journal_readiness'], check=True)
    state.update(status='complete_with_unresolved', finished_at=datetime.now(timezone.utc).isoformat())
    atomic_json(state_path, state)


if __name__ == '__main__':
    run()
