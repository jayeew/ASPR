"""Recheck only cached rejected identities affected by official ISSN additions."""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict

from .cached_date_apply import run as merge_evidence
from .derived_refresh import refresh
from .fig8b_qualification_build import match
from .supplement_apply import apply
from .supplement_queue import ROOT
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run() -> dict:
    folder = ROOT / 'supplement/annual_nlm/issn_recheck'
    folder.mkdir(exist_ok=True)
    repaired = {json.loads(line)['source_id'] for line in (folder.parent / 'issn_changes.jsonl').read_text().splitlines()}
    journals = {r['source_id']: r for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    db = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
    sources = defaultdict(list)
    for wid, doi, title, journal, sid in db.execute("SELECT work_id,doi,title,journal,source_id FROM papers WHERE status IN ('pending_primary_identity_mismatch','pending_primary_online_date')"):
        if sid not in repaired or not doi:
            continue
        paper = {'work_id': wid, 'doi': doi, 'title': title, 'journal_name': journal, 'source_id': sid}
        for payload, in db.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (wid,)):
            old = json.loads(payload)
            if old['identity'] == 'matched':
                continue
            item = {k: v for k, v in old.items() if k not in {'id', 'work_id', 'doi', 'identity', 'class'}}
            item['journal_registry_repair_receipt'] = '../../supplement/annual_nlm/issn_changes.jsonl'
            if match(paper, item, journals[sid])['identity'] == 'matched':
                sources[doi].append(item)
    db.close()
    atomic_json(folder / 'scope.json', {'dois': sorted(sources), 'source_count': len(repaired), 'network_requests': 0,
                                      'method': 'Only pending records with rejected cached evidence now matching the officially enriched current registry'})
    result = merge_evidence(set(sources), folder, explicit_sources=dict(sources))
    metadata = apply(scope_dois=set(sources), index_sources=False)
    atomic_json(folder / 'metadata_status.json', metadata)
    refresh()
    return result


if __name__ == '__main__':
    run()
