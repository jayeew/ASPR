"""Confirm finite explicit Moscow university signatures against official sources."""
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

FOLDER = ROOT / 'supplement/official_entities/russian_signed_self'


def classify(name: str) -> str | None:
    value = name.casefold().replace('“', '').replace('”', '').replace('‘‘', '').replace('’’', '')
    msu = r'(?:faculty of physics, )?moscow state university(?:, faculty of physics)?'
    mephi = r'national research nuclear university (?:mephi(?: \(moscow engineering physics institute\))?|moscow engineering physics institute(?: mephi)?)'
    for label, head, postal in [('msu', msu, '119991'), ('mephi', mephi, '115409')]:
        tail = r'(?:, | )?(?:' + postal + r',? )?moscow(?:,? ' + postal + r')?,? (?:russia|russian federation)(?:,? ' + postal + r')?'
        if re.fullmatch(head + tail, value):
            return label
    return None


def run() -> None:
    if (FOLDER / 'application.json').exists():
        return
    mephi = json.loads((FOLDER / 'mephi.json').read_text())
    raw = gzip.decompress((ROOT / mephi['raw_receipt']).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == mephi['sha256']
    text = ' '.join(html.fromstring(raw).text_content().split())
    assert all(x in text for x in ['National Research Nuclear University', '115409', 'Moscow Engineering Physics Institute'])
    atomic_json(FOLDER / 'msu_web_review.json', {
        'url': 'https://www.phys.msu.ru/eng/about/contacts/', 'access_method': 'web primary-page indexed text',
        'reviewed_on': '2026-10-25', 'locator': 'Official Address and Contact Information, lines 182–188',
        'source_excerpt': 'Faculty of Physics; M.V.Lomonosov Moscow State University; Leninskie Gory, Moscow 119991; Russia',
        'direct_request': 'ConnectTimeout retained in msu.json; primary web text available',
        'scope': 'University self and explicitly named physics faculty only; no other faculty, subsidiary or branch inference'})
    base = {
        'msu': ('https://openalex.org/I19880235', 'Lomonosov Moscow State University', 'msu_web_review.json', 'https://www.phys.msu.ru/eng/about/contacts/'),
        'mephi': ('https://openalex.org/I887846188', 'National Research Nuclear University MEPhI', 'mephi.json', mephi['url'])}
    db = connect(); names = defaultdict(set); groups = defaultdict(set)
    for tid, name in db.execute("SELECT task_id,entity_key FROM tasks WHERE kind='institution' AND active=1 AND (entity_key LIKE '%moscow state university%' OR entity_key LIKE 'national research nuclear university%')"):
        label = classify(name)
        if label:
            names[label].add(name)
            for wid, shard in db.execute("SELECT p.work_id,p.shard FROM task_papers l JOIN papers p USING(work_id) WHERE l.task_id=? AND p.status='included_verified_research'", (tid,)):
                groups[shard].add(wid)
    db.close()
    rules = [{'entity_id': eid, 'name': title, 'exact_names': sorted(names[label]), 'entity_type': 'education',
        'ownership_status': 'officially_confirmed_university_self', 'university_targets': {eid: 1.0},
        'source': str((FOLDER / src).relative_to(ROOT)), 'official_url': url,
        'scope': 'Exact signed university self or explicitly documented MSU physics faculty in Moscow; paper signature establishes 2023–2025 association. Country unchanged; no branches, hospitals or joint centers inferred.'}
        for label, (eid, title, src, url) in base.items()]
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
                         status_name='russian_signed_self/application.json', selected_work_ids=selected)
    atomic_json(FOLDER / 'result.json', {'candidate_papers': sum(map(len, groups.values())),
                'native_confirmed_papers': len(selected), 'native_checks': checks, 'application': result,
                'paper_network_requests': 0, 'final_task_closure': False})


if __name__ == '__main__':
    run()
