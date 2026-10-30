"""Repair known XML label pollution and retry the fixed unresolved country scope."""
from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

from lxml import etree

from . import terminal_country_apply as application
from .affiliation_text_repair import replacements, run as repair
from .postal_country_cache import FOLDER
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def remaining_papers(papers: list[dict], audit: list[dict]) -> set[str]:
    expected = Counter((p['work_id'], s['receipt']) for p in papers for s in p['slots'])
    filled = Counter()
    for row in audit:
        filled[(row['work_id'], row['receipt'])] += row.get('slots_filled', 0)
    return {wid for (wid, receipt), count in expected.items() if filled[(wid, receipt)] < count}


def run(base: Path = FOLDER) -> None:
    folder = base / 'cleaned_native'
    folder.mkdir(exist_ok=True)
    if (folder / 'application.json').exists():
        return
    original = json.loads((base / 'scope.json').read_text())
    audit = json.loads((base / 'application.json').read_text())['audit']
    ids = remaining_papers(original['papers'], audit)
    result = repair(selected_work_ids=ids, folder_name=str(base.relative_to(ROOT / 'supplement') / 'text_repair'))
    papers, unresolved = [], []
    for paper in original['papers']:
        if paper['work_id'] not in ids:
            continue
        revised = json.loads(json.dumps(paper))
        cache = {}
        for slot in revised['slots']:
            receipt = slot['receipt']
            if receipt not in cache:
                try:
                    cache[receipt] = replacements(gzip.decompress((ROOT / receipt).read_bytes()), paper['doi'])
                except (OSError, EOFError, etree.XMLSyntaxError) as error:
                    cache[receipt] = {}
                    unresolved.append({'work_id': paper['work_id'], 'receipt': receipt, 'error': type(error).__name__})
            slot['address'] = cache[receipt].get(slot['address'], slot['address'])
        papers.append(revised)
    scope = folder / 'scope.json'
    atomic_json(scope, {**original, 'papers': papers, 'text_repair': result,
        'source_read_unresolved': unresolved, 'parent_scope': str((base / 'scope.json').relative_to(ROOT)),
        'retry_basis': 'New native address match after removal of XML structural label/identifier elements; no new requests.'})
    application.FOLDER = folder
    application.RULE = str(scope.relative_to(ROOT))
    application.run()


if __name__ == '__main__':
    run()
