"""Execute supplementation against the one current dataset, with resumable progress."""
from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .supplement_queue import ROOT
from .supplement_sources import parse_publisher


def state(phase: str, status: str, detail: dict | None = None) -> None:
    name=os.environ.get('FIG8_SUPPLEMENT_JOB','')
    path = ROOT/'supplement'/('execution_status_'+name+'.json' if name else 'execution_status.json')
    old = json.loads(path.read_text()) if path.exists() else {'steps': []}
    old.update(phase=phase, status=status, updated_at=datetime.now(timezone.utc).isoformat())
    old['steps'].append({'phase': phase, 'status': status, 'detail': detail or {}, 'at': old['updated_at']})
    path.write_text(json.dumps(old, ensure_ascii=False, indent=2))


def command(module: str, arguments: list[str] | None = None) -> None:
    state(module, 'running')
    logs = ROOT / 'supplement/logs'
    logs.mkdir(exist_ok=True)
    destination = logs / (module.rsplit('.', 1)[-1] + '.log')
    with destination.open('a') as stream:
        result = subprocess.run([sys.executable, '-m', 'experiments.fig8_extension.' + module, *(arguments or [])],
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    state(module, 'complete' if result.returncode == 0 else 'failed', {'returncode': result.returncode, 'log': str(destination)})
    if result.returncode:
        raise RuntimeError(f'{module} failed; see {destination}')


def classify_soft_blocks() -> None:
    q = ROOT / 'fields/qualification'
    counts: dict[str, int] = {}
    for path in (q / 'supplement_parsed').glob('*.json'):
        r = json.loads(path.read_text())
        if r.get('source') != 'publisher' or r.get('status') != 'parsed':continue
        raw = gzip.decompress((q/r['raw_receipt']).read_bytes()).decode('utf-8',errors='replace')
        if r.get('parsed',{}).get('doi'):
            reparsed=parse_publisher(raw.encode())
            if reparsed!=r['parsed']:
                r['parsed']=reparsed;path.write_text(json.dumps(r,ensure_ascii=False))
            continue
        if any(s in raw for s in ['triggerInterstitialChallenge', 'Radware Bot Manager Captcha']):
            r.update(status='access_restricted', reason='HTTP 200 browser verification/CAPTCHA page; no publication identity; no bypass attempted')
        else:
            r.update(status='publication_metadata_missing', reason='HTTP 200 page has no extractable publication DOI; not evidence of article absence')
        path.write_text(json.dumps(r, ensure_ascii=False))
        counts[r['status']] = counts.get(r['status'], 0) + 1
    (ROOT / 'supplement/soft_block_classification.json').write_text(json.dumps(counts, indent=2))


def companions() -> None:
    code = Path('experiments/fig8_extension')
    q = ROOT / 'fields/qualification'
    for src, dest in [('fig8b_qualification_decide.py', 'decide.py'), ('fig8b_qualification_recompute.py', 'recompute.py')]:
        shutil.copy2(code / src, q / dest)


def checkpoint_steps() -> None:
    classify_soft_blocks()
    companions()
    command('fig8b_qualification_build')
    command('supplement_crossref')
    crossref=json.loads((ROOT/'supplement/crossref_status.json').read_text())
    if crossref.get('new_success_batches'):command('fig8b_qualification_build')
    command('supplement_acquisition', ['index'])
    command('supplement_apply')
    command('fig8b_qualification_recompute', [str(ROOT / 'fields/qualification'), '--refresh'])
    command('fig8b_qualification_report')
    command('supplement_queue', ['discover'])
    command('supplement_report')
    command('nature_release_recompute', [str(ROOT / 'nature')])
    state('checkpoint', 'complete', {'scope': 'Current acquired evidence applied and independently recomputed; unresolved tasks retained'})


def checkpoint() -> None:
    # One dataset writer at a time; source collection may continue with atomic receipts.
    with (ROOT/'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        checkpoint_steps()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['pilot', 'checkpoint', 'continue-pmc', 'all'])
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if args.action in ['pilot', 'all']:
        command('supplement_queue', ['discover'])
        command('supplement_queue', ['pilot'])
        command('supplement_sources', ['pilot', '--workers', str(args.workers)])
    if args.action in ['pilot', 'checkpoint', 'all']:
        checkpoint()
    if args.action in ['continue-pmc', 'all']:
        if args.action == 'continue-pmc':
            checkpoint()
        command('supplement_sources', ['pmc', '--workers', str(args.workers),'--checkpoint-every','10000'])
        if (ROOT / 'supplement/stop_acquisition').exists():
            state('acquisition', 'stopped_after_inflight', {'reason': 'Stop marker retained; no fallback requests started'})
            return
        command('supplement_oai')
        command('supplement_run',['checkpoint'])
        state('available_pmc_paths', 'complete_with_unresolved', {'remaining': 'Other publisher/official branch verification may still be required; see task ledger and panel readiness'})


if __name__ == '__main__':
    main()
