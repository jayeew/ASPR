"""Repair only cached parse classification; retain raw bytes and before/after evidence."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from lxml import etree

from .native_xml_document import document
from .scoap3_native_collect import FOLDER
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> dict:
    ledger = FOLDER / 'namespace_repairs'
    ledger.mkdir(exist_ok=True)
    count = 0
    held = []
    for path in (FOLDER / 'receipts').glob('*.json'):
        before = json.loads(path.read_text())
        if before.get('status') != 'invalid_xml':
            continue
        try:
            tree, changes = document(gzip.decompress((ROOT / before['raw_receipt']).read_bytes()))
        except etree.XMLSyntaxError as exc:
            held.append({'receipt': str(path.relative_to(ROOT)), 'reason': str(exc)[:250]})
            continue
        if not changes:
            continue
        after = {**before, 'status': 'xml_cached_identity_review_required',
                 'root_tag': tree.tag, 'xml_normalization': changes,
                 'normalization_receipt': str((ledger / path.name).relative_to(ROOT))}
        atomic_json(ledger / path.name, {'before': before, 'after': after,
                    'raw_bytes_unchanged': True, 'network_requests': 0})
        atomic_json(path, after)
        count += 1
    result = {'reclassified': count, 'held': held, 'network_requests': 0}
    atomic_json(FOLDER / 'namespace_repair_latest.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run()))
