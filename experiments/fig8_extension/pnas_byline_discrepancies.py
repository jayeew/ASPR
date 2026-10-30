"""Identify inherited duplicate bylines against identity-matched cached PNAS XML."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import pyarrow.parquet as pq

from .fig8b_qualification_build import key
from .springer_author_cache import prepare_source
from .supplement_queue import ROOT
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run() -> None:
    folder = ROOT / 'supplement/pnas_cached_byline_gaps'
    audit = json.loads((folder / 'audit.json').read_text())
    groups = defaultdict(dict)
    for entry in audit['entries']:
        if entry['remaining_link_reasons'].get('current_name_ambiguous'):
            groups[entry['shard']][entry['work_id']] = entry
    registry = {r['source_id']: set(json.loads(r['issn'] or '[]'))
                for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    records = []
    for shard, jobs in groups.items():
        for paper in pq.read_table(ROOT / 'fields/papers' / shard).to_pylist():
            if paper['work_id'] not in jobs:
                continue
            entry = jobs[paper['work_id']]
            receipt, digest, native = prepare_source(
                ROOT, paper, entry['receipt'], registry[paper['source_id']])
            current = json.loads(paper['authors_json'])
            counts = Counter(key(a['name']) for a in current)
            declared = Counter(key(a['name']) for a in native.get('authors', []))
            duplicate = [a for a in current if counts[key(a['name'])] > 1]
            confirmed = bool(declared) and not native.get('byline_has_etal')
            excess = sum(max(0, n - declared[k]) for k, n in counts.items() if n > 1)
            records.append({
                'work_id': paper['work_id'], 'doi': paper['doi'], 'shard': shard,
                'receipt': entry['receipt'], 'raw_receipt': receipt.get('raw_receipt'),
                'digest': digest, 'native_complete': confirmed,
                'status': 'inherited_duplicate_disagrees_with_native_byline'
                if confirmed and excess else 'requires_byline_review',
                'current_count': len(current), 'native_count': sum(declared.values()),
                'duplicate_entries': [{'name': a['name'], 'author_id': a.get('author_id'),
                    'position': a.get('position'),
                    'has_primary_slots': bool(a.get('publisher_affiliation_slots'))}
                    for a in duplicate],
                'native_names_absent_from_current': [a['name'] for a in native.get('authors', [])
                                                     if key(a['name']) not in counts],
                'excess_duplicate_entries': excess,
                'native_byline': native.get('authors', []),
                'repair_status': 'pending_author_roster_reconciliation',
                'cross_source_person_identity_inferred': False})
    summary = {'papers': len(records), 'outcomes': dict(Counter(r['status'] for r in records)),
               'excess_duplicate_entries': sum(r['excess_duplicate_entries'] for r in records),
               'network_requests': 0, 'main_table_modified': False,
               'scope': 'Duplicate-name subset of existing cached PNAS gap audit; not whole-cohort census',
               'records': records}
    atomic_json(folder / 'duplicate_byline_evidence.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    run()
