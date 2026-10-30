"""Apply explicitly typed native Springer XML through existing qualification guards."""
from __future__ import annotations

import gzip
import json
import os
from collections import Counter

from .publisher_delta_apply import apply_available
from .scoap3_native_collect import FOLDER
from .springer_native_xml import parse
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> dict:
    staged = FOLDER / 'staged'; staged.mkdir(exist_ok=True)
    done_path = FOLDER / 'applied_receipts.json'
    done = json.loads(done_path.read_text()) if done_path.exists() else {}
    jobs = []; selected = {}; held = []; dois = set()
    for path in sorted((FOLDER / 'receipts').glob('*.json')):
        rel = str(path.relative_to(ROOT))
        if rel in done:continue
        receipt = json.loads(path.read_text())
        if receipt.get('status') != 'xml_cached_identity_review_required':
            held.append({'receipt': rel, 'reason': receipt['status']}); continue
        if receipt.get('root_tag') != 'Publisher':
            held.append({'receipt': rel, 'reason': 'separate_native_schema_adapter_required'}); continue
        parsed = parse(gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes()))
        if not parsed or parsed['doi'] != [receipt['doi']]:
            held.append({'receipt': rel, 'reason': 'unsupported_schema_or_native_doi_mismatch'}); continue
        if receipt['doi'] in dois:continue
        dois.add(receipt['doi'])
        record = {'source': 'publisher_native_xml', 'route': 'springer', 'status': 'parsed',
                  'doi': receipt['doi'], 'url': receipt['url'], 'raw_sha256': receipt['sha256'],
                  'raw_receipt': os.path.relpath(ROOT / receipt['raw_receipt'], ROOT / 'fields/qualification'),
                  'parsed': parsed, 'acquisition_receipt': rel,
                  'native_schema': 'Springer Publisher/ArticleInfo; publisher-deposited SCOAP3 XML'}
        target = staged / path.name; atomic_json(target, record)
        jobs.append((receipt['doi'], str(target.relative_to(ROOT)))); selected[rel] = receipt
    result = apply_available(supplied_jobs=jobs) if jobs else {}
    db = connect(); db.execute('PRAGMA busy_timeout=900000')
    for rel, receipt in selected.items():
        for task_id, in db.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?", (receipt['doi'],)).fetchall():
            if not db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?', (task_id, 'scoap3_native_xml', rel)).fetchone():
                db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                           (task_id, 'scoap3_native_xml', 'native_evidence_processed_identity_and_conflict_guards_retained', rel,
                            json.dumps({'final_task_closure': False, 'source_url': receipt['url']})))
                db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (task_id,))
        done[rel] = 'qualification_evidence_processed'
    db.commit(); db.close()
    atomic_json(done_path, done)
    output = {'selected_receipts': len(selected), 'application': result, 'held': held,
              'held_reasons': dict(Counter(r['reason'] for r in held)), 'network_requests': 0}
    atomic_json(FOLDER / 'application.json', output)
    print(json.dumps({k: v for k, v in output.items() if k != 'held'}), flush=True)
    return output


if __name__ == '__main__':run()
