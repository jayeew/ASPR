"""Confirm USTC identity from official evidence and exact deposited author links."""
from __future__ import annotations

import fcntl
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .crossref_country_metadata import load
from .fig8b_qualification_build import key
from .official_units_apply import run as apply_rules
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv

BASE = ROOT / 'supplement/official_entities/ustc_signed_self'
FOLDER = BASE / 'deposited'


def confirmed_links(authors: list[dict], source: dict, receipt: str, aliases: set[str]) -> bool:
    old_counts = Counter(key(a['name']) for a in authors)
    native = defaultdict(list)
    for author in source.get('authors', []):
        native[key(author['name'])].append(author)
    found = False
    for author in authors:
        matching = [s for s in author.get('publisher_affiliation_slots') or []
                    if ' '.join(s.get('text', '').casefold().split()) in aliases]
        if not matching or author.get('nature_v1_slots') is not None:
            continue
        found = True
        if author.get('publisher_affiliation_source') != receipt:
            return False
        name = key(author['name'])
        if old_counts[name] != 1 or len(native[name]) != 1:
            return False
        addresses = {' '.join(a['text'].casefold().split()) for a in native[name][0]['affiliations']}
        if any(' '.join(s['text'].casefold().split()) not in addresses for s in matching):
            return False
    return found


def run(base: Path = BASE) -> None:
    folder = base / 'deposited'
    folder.mkdir(exist_ok=True)
    if (folder / 'application.json').exists():
        return
    routes = base / 'remaining_source_routes.json'
    if routes.exists():
        scope = json.loads(routes.read_text())['records']
    else:
        checks = json.loads((base / 'result.json').read_text())['native_checks']
        scope = [r for r in checks if r['receipt'].startswith('fields/crossref_raw/')
                 and not r.get('confirmed_names', r.get('native_identity_and_signature_confirmed', False))]
    rules = json.loads((base / 'rules.json').read_text())['rules']
    for rule in rules:
        rule['scope'] = 'Exact reviewed institution identity corroborated by its retained official source. Author-affiliation link verified in publisher-deposited Crossref metadata; no complete native byline, country or external person identity asserted.'
    aliases = {n for rule in rules for n in rule['exact_names']}
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    groups = defaultdict(dict); checks = []; selected = set()
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        for item in scope:
            row = db.execute("SELECT shard FROM papers WHERE work_id=? AND status='included_verified_research'", (item['work_id'],)).fetchone()
            if row:
                groups[row[0]][item['work_id']] = item['receipt']
        db.close()
        for shard, targets in groups.items():
            for paper in pq.read_table(ROOT / 'fields/papers' / shard, filters=[('work_id', 'in', sorted(targets))]).to_pylist():
                receipt = targets[paper['work_id']]; source = load(receipt, paper['doi'])
                identity = paper['doi'] in source.get('doi', []) and signature_match(paper, source, journals[paper['source_id']])
                matched = identity and confirmed_links(json.loads(paper['authors_json']), source, receipt, aliases)
                checks.append({'work_id': paper['work_id'], 'receipt': receipt,
                               'identity_matched': identity, 'unique_author_address_links_matched': matched,
                               'source_authority': 'Crossref publisher-deposited author-affiliation metadata',
                               'complete_native_byline_confirmed': False})
                if matched:
                    selected.add(paper['work_id'])
        atomic_json(folder / 'rules.json', {'status': 'reviewed', 'rules': rules,
                    'deposited_link_checks': checks, 'network_requests': 0, 'final_task_closure': False})
        result = apply_rules(rules_path=str((folder / 'rules.json').relative_to(ROOT)),
                             status_name=str((folder / 'application.json').relative_to(ROOT / 'supplement/official_entities')),
                             selected_work_ids=selected, lock_held=True)
    atomic_json(folder / 'result.json', {'scope_records': len(scope), 'checks': checks,
                'confirmed_papers': len(selected), 'application': result,
                'network_requests': 0, 'final_task_closure': False})


if __name__ == '__main__':
    run()
