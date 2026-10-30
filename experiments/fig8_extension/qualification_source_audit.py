"""Audit shared source references in the current qualification evidence index."""
from __future__ import annotations

import fcntl
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone

from .change_source_audit import source_paths
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> dict:
    folder = ROOT / 'supplement/qualification_source_audit'
    folder.mkdir(exist_ok=True)
    with (folder / 'runner.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = sqlite3.connect(f'file:{ROOT / "supplement/tasks.sqlite"}?mode=ro', uri=True)
        # One consistent read snapshot; the captured index may be extended afterwards.
        db.execute('BEGIN')
        counts: Counter = Counter()
        sources = {}
        with (folder / 'unresolved.jsonl').open('w') as unresolved:
            for work, key, kind, payload in db.execute('SELECT work_id,evidence_key,kind,payload FROM current_evidence_index'):
                values = json.loads(payload)
                records = values if isinstance(values, list) else [values]
                counts[kind + '_index_rows'] += 1
                for record in records:
                    receipt = record.get('receipt') or ''
                    encoded = receipt if isinstance(receipt, str) else json.dumps(receipt)
                    if encoded not in sources:
                        sources[encoded] = {**source_paths(encoded, ROOT), 'reference_count': 0}
                    result = sources[encoded]
                    result['reference_count'] += 1
                    counts[kind + '_references'] += 1
                    counts[result['status']] += 1
                    if result['status'] != 'source_paths_resolved':
                        unresolved.write(json.dumps({'work_id': work, 'evidence_key': key,
                            'kind': kind, 'family': record.get('family'), 'receipt': receipt,
                            'status': result['status']}, ensure_ascii=False) + '\n')
        db.close()
        with (folder / 'source_results.jsonl').open('w') as stream:
            for receipt, result in sources.items():
                stream.write(json.dumps({'receipt': receipt, **result}, ensure_ascii=False) + '\n')
        result = {'at': datetime.now(timezone.utc).isoformat(), 'status': 'fixed_index_snapshot_checked',
                  'counts': dict(counts), 'distinct_source_references': len(sources), 'network_requests': 0,
                  'scope': 'Current primary/secondary evidence index at read snapshot; later records require subsequent audit',
                  'limitations': ['File resolution does not prove source-content validity or applicability.',
                                  'Original evidence log and active decision-ID linkage are separate checks.'],
                  'goal_complete': False}
        atomic_json(folder / 'status.json', result)
        return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
