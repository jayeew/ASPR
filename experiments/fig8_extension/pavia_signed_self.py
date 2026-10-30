"""Reuse official Pavia identity for an exact standalone university signature."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict

import pyarrow.parquet as pq
from lxml import html

from .official_units_apply import run as apply_rules
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .terminal_country_apply import native_metadata
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/official_entities/pavia_signed_self'
NAME = 'università di pavia, pavia, italy'


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    if (FOLDER / 'application.json').exists():
        return
    original = json.loads((ROOT / 'supplement/official_entities/high_frequency_university_self_rules.json').read_text())
    parent = next(r for r in original['rules'] if r['name'] == 'University of Pavia')
    source = json.loads((ROOT / parent['source']).read_text())
    raw = gzip.decompress((ROOT / source['raw_receipt']).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == source['sha256']
    text = ' '.join(html.fromstring(raw).text_content().split())
    assert 'Università degli Studi di Pavia' in text and '27100 Pavia - Italy' in text
    rule = {**parent, 'exact_names': [NAME],
            'locator': 'Official department header Università di Pavia and footer Università degli Studi di Pavia, 27100 Pavia - Italy',
            'scope': 'Exact standalone university signature; cached native DOI/title/journal and author-affiliation statement checked. No branch/hospital/joint-center inference.'}
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    db = connect(); groups = defaultdict(set)
    for wid, shard in db.execute("""SELECT DISTINCT p.work_id,p.shard FROM tasks t
        JOIN task_papers l USING(task_id) JOIN papers p USING(work_id)
        WHERE t.kind='institution' AND t.entity_key=? AND p.status='included_verified_research'""", (NAME,)):
        groups[shard].add(wid)
    db.close(); confirmed = set(); checks = []
    for shard, ids in groups.items():
        for paper in pq.read_table(ROOT / 'fields/papers' / shard, filters=[('work_id', 'in', sorted(ids))]).to_pylist():
            receipts = {a['publisher_affiliation_source'] for a in json.loads(paper['authors_json'])
                        if a.get('publisher_affiliation_source') and any(' '.join(s.get('text', '').casefold().split()) == NAME
                        for s in a.get('publisher_affiliation_slots') or [])}
            for receipt in receipts:
                native = native_metadata(receipt, paper['doi'])
                names = {' '.join(s['text'].casefold().split()) for a in native.get('authors', []) for s in a.get('affiliations', [])}
                matched = (paper['doi'] in native.get('doi', []) and signature_match(paper, native, journals[paper['source_id']]) and NAME in names)
                checks.append({'work_id': paper['work_id'], 'receipt': receipt, 'native_identity_and_signature_confirmed': matched})
                if matched:
                    confirmed.add(paper['work_id'])
    atomic_json(FOLDER / 'rules.json', {'status': 'reviewed', 'rules': [rule], 'native_checks': checks,
                                      'network_requests': 0, 'final_task_closure': False})
    result = apply_rules(rules_path=str((FOLDER / 'rules.json').relative_to(ROOT)),
                         status_name='pavia_signed_self/application.json', selected_work_ids=confirmed)
    atomic_json(FOLDER / 'result.json', {'selected_papers': sum(map(len, groups.values())),
                'confirmed_papers': len(confirmed), 'native_checks': checks, 'application': result,
                'network_requests': 0, 'final_task_closure': False})


if __name__ == '__main__':
    run()
