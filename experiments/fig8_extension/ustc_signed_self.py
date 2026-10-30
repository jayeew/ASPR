"""Confirm USTC self signatures using its official identity and native bylines."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import defaultdict

import pyarrow.parquet as pq
from lxml import html

from .official_units_apply import run as apply_rules
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .terminal_country_apply import native_metadata
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/official_entities/ustc_signed_self'


def classify(name: str) -> str | None:
    value = name.casefold().replace('’', "'")
    head = r'(?:the )?university of science and technology of china(?: \(ustc\))?'
    address = r'hefei(?: \d{6})?|anhui(?: province)?|china|p\.?r\.? ?china|people\'s republic of china|\d{6}'
    return 'ustc' if re.fullmatch(head + r'(?:, (?:' + address + r'))*', value) else None


def run() -> None:
    if (FOLDER / 'application.json').exists():
        return
    source = json.loads((FOLDER / 'official.json').read_text())
    raw = gzip.decompress((ROOT / source['raw_receipt']).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == source['sha256']
    text = ' '.join(html.fromstring(raw.decode('utf-8')).text_content().split())
    assert 'University of Science and Technology of China' in text and 'Hefei' in text
    sources = {'ustc': source}
    base = {'ustc': ('https://openalex.org/I126520041', 'University of Science and Technology of China')}
    db = connect(); names = defaultdict(set); groups = defaultdict(set)
    for tid, name in db.execute("SELECT task_id,entity_key FROM tasks WHERE kind='institution' AND active=1 AND (entity_key LIKE 'university of science and technology of china%' OR entity_key LIKE 'the university of science and technology of china%')"):
        label = classify(name)
        if label:
            names[label].add(name)
            for wid, shard in db.execute("SELECT p.work_id,p.shard FROM task_papers l JOIN papers p USING(work_id) WHERE l.task_id=? AND p.status='included_verified_research'", (tid,)):
                groups[shard].add(wid)
    db.close()
    rules = [{'entity_id': eid, 'name': title, 'exact_names': sorted(names[label]),
        'entity_type': 'education',
        'ownership_status': 'officially_confirmed_university_self',
        'university_targets': {eid: 1.0},
        'source': str((FOLDER / 'official.json').relative_to(ROOT)),
        'official_url': sources[label]['url'], 'raw_receipt': sources[label]['raw_receipt'],
        'raw_sha256': sources[label]['sha256'],
        'scope': 'Exact standalone full-name signature only; native paper establishes association in 2023–2025. No department, branch, hospital, joint organization or country inferred.'}
        for label, (eid, title) in base.items()]
    aliases = {name for values in names.values() for name in values}
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    checks, selected = [], set()
    for shard, ids in groups.items():
        for paper in pq.read_table(ROOT / 'fields/papers' / shard, filters=[('work_id', 'in', sorted(ids))]).to_pylist():
            receipts = defaultdict(set)
            for author in json.loads(paper['authors_json']):
                if author.get('nature_v1_slots') is not None or not author.get('publisher_affiliation_source'):
                    continue
                for slot in author.get('publisher_affiliation_slots') or []:
                    name = ' '.join(slot.get('text', '').casefold().split())
                    if name in aliases:
                        receipts[author['publisher_affiliation_source']].add(name)
            matched = set()
            for receipt, expected in receipts.items():
                native = native_metadata(receipt, paper['doi'])
                native_names = {' '.join(s['text'].casefold().split()) for a in native.get('authors', []) for s in a.get('affiliations', [])}
                valid = (paper['doi'] in native.get('doi', []) and signature_match(paper, native, journals[paper['source_id']]))
                found = expected & native_names if valid else set()
                matched.update(found)
                checks.append({'work_id': paper['work_id'], 'receipt': receipt, 'expected_names': sorted(expected), 'confirmed_names': sorted(found)})
            # All applicable signatures must have native support before paper-level application.
            if receipts and set.union(*receipts.values()) <= matched:
                selected.add(paper['work_id'])
    atomic_json(FOLDER / 'rules.json', {'status': 'reviewed', 'rules': rules, 'native_checks': checks,
                                      'final_task_closure': False})
    result = apply_rules(rules_path=str((FOLDER / 'rules.json').relative_to(ROOT)),
                         status_name='ustc_signed_self/application.json', selected_work_ids=selected)
    atomic_json(FOLDER / 'result.json', {'candidate_papers': sum(map(len, groups.values())),
                'native_confirmed_papers': len(selected), 'native_checks': checks, 'application': result,
                'paper_network_requests': 0, 'final_task_closure': False})


if __name__ == '__main__':
    run()
