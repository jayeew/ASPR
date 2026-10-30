"""Refresh fixed SciOpen receipts from cached HTML before qualification application."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter

from .sciopen_metadata import enrich_metadata
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, parse_publisher


def run() -> dict:
    folder = ROOT / 'supplement/sciopen_remaining/native_reparse'
    folder.mkdir(exist_ok=True)
    db = connect(ROOT)
    jobs = db.execute("SELECT doi,receipt FROM publisher_remaining_jobs "
                      "WHERE route='sciopen' AND status='parsed' AND applied=0").fetchall()
    stats = Counter()
    for doi, receipt in jobs:
        record = json.loads((ROOT / receipt).read_text())
        if record.get('sciopen_parser_revision') == 2:
            continue
        raw = gzip.decompress((ROOT / 'fields/qualification' / record['raw_receipt']).read_bytes())
        parsed = enrich_metadata(raw, parse_publisher(raw.decode('utf-8')))
        target = folder / (hashlib.sha256(doi.encode()).hexdigest() + '.json')
        revised = {**record, 'parsed': parsed, 'raw_text_encoding': 'utf-8',
                   'sciopen_parser_revision': 2, 'prior_receipt': receipt,
                   'network_requests': 0, 'reason': 'Parse explicit online date with unpadded month/day'}
        atomic_json(target, revised)
        relative = str(target.relative_to(ROOT))
        db.execute("UPDATE publisher_remaining_jobs SET receipt=? WHERE doi=? AND receipt=?",
                   (relative, doi, receipt))
        task_id = hashlib.sha256(('qualification:' + doi).encode()).hexdigest()[:32]
        db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                   (task_id, 'sciopen_cached_native_reparse', 'parsed_not_qualification_applied', relative,
                    json.dumps({'prior_receipt': receipt, 'before_online_date': record['parsed'].get('online_date'),
                                'after_online_date': parsed['online_date']})))
        stats['receipts_reparsed'] += 1
        stats['explicit_online_dates'] += bool(parsed['online_date'])
    db.commit()
    db.close()
    result = {**stats, 'network_requests': 0}
    atomic_json(folder / 'status.json', result)
    return result
