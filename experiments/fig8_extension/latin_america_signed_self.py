"""Confirm explicit Latin American university/research-unit signatures."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import unicodedata
from collections import defaultdict

import pyarrow.parquet as pq
from lxml import html

from .official_units_apply import run as apply_rules
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .terminal_country_apply import native_metadata
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/official_entities/latin_america_signed_self'


def classify(name: str) -> str | None:
    value = ''.join(c for c in unicodedata.normalize('NFKD', name.casefold()) if not unicodedata.combining(c))
    definitions = [
        ('sonora', r'universidad de sonora(?: \(unison\))?', r'hermosillo|sonora|mexico|83000'),
        ('ufrj', r'universidade federal do rio de janeiro(?: \(ufrj\))?', r'rio de janeiro|rj|brazil|brasil|\d{5}-?\d{3}'),
        ('cbpf', r'centro brasileiro de pesquisas fisicas(?: \(cbpf\))?', r'rio de janeiro|rj|brazil|brasil|\d{5}-?\d{3}')]
    for label, head, address in definitions:
        if re.fullmatch(head + r'(?:, (?:' + address + r'))*', value):
            return label
    return None


def run() -> None:
    if (FOLDER / 'application.json').exists():
        return
    sources = {}
    for label, phrase in [('sonora', 'Universidad de Sonora (UNISON)'),
                          ('ufrj', 'Universidade Federal do Rio de Janeiro'),
                          ('cbpf', 'Centro Brasileiro de Pesquisas Físicas')]:
        source = json.loads((FOLDER / (label + '.json')).read_text())
        raw = gzip.decompress((ROOT / source['raw_receipt']).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == source['sha256']
        text = ' '.join(html.fromstring(raw.decode('utf-8')).text_content().split())
        assert phrase in text
        if label == 'cbpf':
            assert 'vinculado ao Ministério da Ciência' in text
        sources[label] = source
    base = {
        'sonora': ('https://openalex.org/I4061448', 'Universidad de Sonora'),
        'ufrj': ('https://openalex.org/I122140584', 'Universidade Federal do Rio de Janeiro'),
        'cbpf': ('https://openalex.org/I4210125245', 'Centro Brasileiro de Pesquisas Físicas')}
    db = connect(); names = defaultdict(set); groups = defaultdict(set)
    for tid, name in db.execute("SELECT task_id,entity_key FROM tasks WHERE kind='institution' AND active=1 AND (entity_key LIKE 'universidad de sonora%' OR entity_key LIKE 'universidade federal do rio de janeiro%' OR entity_key LIKE 'centro brasileiro de pesquisas f%')"):
        label = classify(name)
        if label:
            names[label].add(name)
            for wid, shard in db.execute("SELECT p.work_id,p.shard FROM task_papers l JOIN papers p USING(work_id) WHERE l.task_id=? AND p.status='included_verified_research'", (tid,)):
                groups[shard].add(wid)
    db.close()
    rules = [{'entity_id': eid, 'name': title, 'exact_names': sorted(names[label]),
        'entity_type': 'research_organization' if label == 'cbpf' else 'education',
        'ownership_status': 'officially_confirmed_nonuniversity' if label == 'cbpf' else 'officially_confirmed_university_self',
        'university_targets': {} if label == 'cbpf' else {eid: 1.0},
        'source': str((FOLDER / (label + '.json')).relative_to(ROOT)),
        'official_url': sources[label]['url'], 'raw_receipt': sources[label]['raw_receipt'],
        'raw_sha256': sources[label]['sha256'],
        'scope': 'Exact standalone full-name signature only; native paper establishes association in 2023–2025. No department, branch, hospital, joint organization or country inferred. CBPF research institute is not assigned to a university.'}
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
                         status_name='latin_america_signed_self/application.json', selected_work_ids=selected)
    atomic_json(FOLDER / 'result.json', {'candidate_papers': sum(map(len, groups.values())),
                'native_confirmed_papers': len(selected), 'native_checks': checks, 'application': result,
                'paper_network_requests': 0, 'final_task_closure': False})


if __name__ == '__main__':
    run()
