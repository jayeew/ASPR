"""Apply a completed acquisition wave once to the single current Fig8 dataset."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT / 'supplement'
    state_path = folder / 'full_preparation_checkpoint.json'
    stages = [
        ('fig8b_qualification_build', []),
        ('supplement_acquisition', ['index']),
        ('supplement_apply', []),
        ('fig8b_qualification_recompute', [str(ROOT/'fields/qualification'), '--refresh']),
        ('fig8b_qualification_report', []),
        ('supplement_queue', ['discover']),
        ('supplement_report', []),
        ('journal_readiness', ['--lock-held']),
    ]
    state = {'pid': os.getpid(), 'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
             'stages_completed': [], 'purpose': 'Apply all retained evidence after completed batch acquisition; not goal completion'}
    with (folder/'checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for module, arguments in stages:
            state.update(stage=module, updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(state_path, state)
            log = folder/'logs'/('full_preparation_'+module+'.log')
            with log.open('a') as stream:
                result = subprocess.run([sys.executable, '-u', '-m', 'experiments.fig8_extension.'+module, *arguments],
                                        stdout=stream, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                state.update(status='failed', returncode=result.returncode, log=str(log))
                atomic_json(state_path, state)
                raise RuntimeError(f'{module} failed; {log}')
            state['stages_completed'].append(module)
            print(json.dumps({'completed': module}), flush=True)
        state.update(status='complete', updated_at=datetime.now(timezone.utc).isoformat())
        atomic_json(state_path, state)


if __name__ == '__main__':
    run()
