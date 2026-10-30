"""Serialize current-data coverage/report refreshes without rebuilding qualification."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json


def refresh(force: bool = False, resume: bool = False) -> None:
    folder=ROOT/'supplement'
    pending=folder/'derived_refresh_pending.json'
    if not force:
        atomic_json(pending,{'status':'pending_coalesced_phase_refresh','requested_at':datetime.now(timezone.utc).isoformat(),'reason':'Incremental updates continue; full report and sensitivity recomputation deferred to phase completion','run':'python3 -m experiments.fig8_extension.derived_refresh --force'})
        return
    with (folder/'checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        from .qa_type_apply import reconcile_current_aliases
        reconcile_current_aliases(lock_held=True)
        marker=folder/'derived_refresh_status.json'
        previous = json.loads(marker.read_text()) if resume and marker.exists() else {}
        completed = list(previous.get('completed_modules', [])) if previous.get('status') == 'failed' else []
        state={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'completed_modules':completed}
        if completed:
            state['resumed_failure'] = previous
        atomic_json(marker,state)
        for source_name, target_name in [('fig8b_qualification_decide.py', 'decide.py'),
                                         ('fig8b_qualification_recompute.py', 'recompute.py')]:
            source = Path(__file__).with_name(source_name)
            target = ROOT / 'fields/qualification' / target_name
            current = source.read_bytes()
            before = target.read_bytes()
            if before != current:
                atomic_bytes(target, current)
                with (folder / 'offline_rule_sync_events.jsonl').open('a') as stream:
                    stream.write(json.dumps({'source': str(source), 'target': str(target),
                        'before_sha256': hashlib.sha256(before).hexdigest(),
                        'after_sha256': hashlib.sha256(current).hexdigest(),
                        'reason': 'Offline replay must use the same current qualification and metadata gap rules as incremental updates',
                        'updated_at': datetime.now(timezone.utc).isoformat()}) + '\n')
        modules=[('metadata_task_sync',['--lock-held']),('fig8b_qualification_recompute',[str(ROOT/'fields/qualification'),'--refresh']),
                 ('metadata_gap_flags',['--lock-held']),
                 ('supplement_report',[]),('journal_readiness',['--lock-held']),
                 ('change_source_audit',[]),('qualification_source_audit',[]),
                 ('journal_goal_audit',[])]
        for name,args in modules:
            if name in completed:
                continue
            with (folder/'logs'/('current_'+name+'.log')).open('a') as stream:
                result=subprocess.run([sys.executable,'-u','-m','experiments.fig8_extension.'+name,*args],
                                      stdout=stream,stderr=subprocess.STDOUT,check=False)
            if result.returncode:
                state.update(status='failed',failed_module=name,returncode=result.returncode)
                atomic_json(marker,state)
                raise RuntimeError(f'Derived refresh failed: {name}; see retained log')
            state['completed_modules'].append(name);atomic_json(marker,state)
        state.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(marker,state)
        pending.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    refresh(force=True, resume=args.resume)
