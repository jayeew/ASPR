"""One resumable native-snapshot pass for current candidate publication locations."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import multiprocessing
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/snapshot_location_recovery'
TARGETS: dict[str, str] = {}
IDENTITY = re.compile(rb'\s*\{\s*"id"\s*:\s*"([^"]+)"')


def scan_file(value: dict) -> dict:
    path = Path(value['path']); token = hashlib.sha256(str(path).encode()).hexdigest()
    receipt = FOLDER / 'files' / (token + '.json')
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old['source_bytes'] != value['bytes']:
            raise ValueError('Snapshot file changed after scope was fixed')
        return old
    if path.stat().st_size != value['bytes']:
        raise ValueError('Snapshot file size differs from fixed scope')
    start = time.monotonic(); lines = matched = conflicts = 0
    output = FOLDER / 'files' / (token + '.jsonl.gz')
    temporary = output.with_suffix('.tmp')
    with gzip.open(path, 'rb') as source, gzip.open(temporary, 'wt', compresslevel=1) as target:
        for line in source:
            lines += 1
            match = IDENTITY.match(line[:160])
            if not match:
                raise ValueError(f'Unrecognized work identifier at {path}:{lines}')
            wid = match[1].decode()
            if wid not in TARGETS:
                continue
            item = json.loads(line)
            doi = (item.get('doi') or '').lower().removeprefix('https://doi.org/').removeprefix('http://doi.org/')
            identity = 'work_id_doi_match' if doi and doi == TARGETS[wid] else 'doi_identity_review_required'
            conflicts += identity != 'work_id_doi_match'
            matched += identity == 'work_id_doi_match'
            target.write(json.dumps({'source_file': str(path), 'source_line': lines,
                'source_record_sha256': hashlib.sha256(line).hexdigest(),
                'identity': identity, 'work': item}, ensure_ascii=False) + '\n')
    temporary.replace(output)
    result = {'status': 'source_file_processed', 'source_file': str(path),
        'source_bytes': value['bytes'], 'records_read': lines, 'matched_candidates': matched,
        'identity_review_records': conflicts, 'raw_receipt': str(output.relative_to(ROOT)),
        'output_bytes': output.stat().st_size, 'elapsed_seconds': time.monotonic() - start}
    atomic_json(receipt, result)
    return result


def run(workers: int = 4) -> None:
    global TARGETS
    FOLDER.mkdir(exist_ok=True); (FOLDER / 'files').mkdir(exist_ok=True)
    with (FOLDER / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        targets = FOLDER / 'targets.json'
        if not targets.exists():
            db = connect()
            selected = dict(db.execute("SELECT work_id,doi FROM papers WHERE status LIKE 'pending%' OR status='included_verified_research'"))
            db.close(); atomic_json(targets, selected)
        TARGETS = json.loads(targets.read_text())
        manifest = FOLDER / 'scope.json'
        if not manifest.exists():
            paths = sorted(Path('/mnt/d/FabCitationData/openalex-snapshot/data/works').glob('updated_date=*/*.gz'), reverse=True)
            atomic_json(manifest, {'files': [{'path': str(p), 'bytes': p.stat().st_size} for p in paths],
                'candidate_identities': len(TARGETS), 'target_scope_sha256': hashlib.sha256(targets.read_bytes()).hexdigest(),
                'scope': 'Fixed current pending and included candidates only. Additional locations are retrieval leads, never research-type or institutional-ownership confirmation.'})
        scope = json.loads(manifest.read_text())
        assert scope['target_scope_sha256'] == hashlib.sha256(targets.read_bytes()).hexdigest()
        state = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
            'fixed_files': len(scope['files']), 'completed_files': 0, 'compressed_bytes_processed': 0,
            'records_read': 0, 'matched_candidates': 0, 'identity_review_records': 0,
            'output_bytes': 0, 'workers': workers, 'network_requests': 0,
            'qualification_changes': 0, 'final_task_closure': False}
        atomic_json(FOLDER / 'status.json', state)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
            futures = [pool.submit(scan_file, row) for row in scope['files']]
            for future in as_completed(futures):
                result = future.result(); state['completed_files'] += 1
                state['compressed_bytes_processed'] += result['source_bytes']
                for key in ['records_read', 'matched_candidates', 'identity_review_records', 'output_bytes']:
                    state[key] += result[key]
                state['last_completed_file'] = result['source_file']
                atomic_json(FOLDER / 'status.json', state)
        state.update(status='fixed_snapshot_scope_processed_locations_need_review', finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(FOLDER / 'status.json', state)


if __name__ == '__main__':
    run()
