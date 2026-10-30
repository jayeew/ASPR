"""Find complete cached native rosters that can replace differing inherited lists."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import pyarrow.parquet as pq

from .fig8b_qualification_build import key
from .pnas_byline_repair import reconcile
from .springer_author_cache import prepare_source
from .supplement_apply import byline_fingerprint, signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/cached_native_roster_reconciliation'


def run(use_native_adapters: bool = False) -> None:
    FOLDER.mkdir(exist_ok=True)
    db = connect()
    scope_path = FOLDER / 'fixed_sources.json'
    if not scope_path.exists():
        rows = db.execute("""SELECT DISTINCT p.work_id,p.shard,s.receipt
            FROM tasks t JOIN papers p ON t.entity_key=p.work_id
            JOIN springer_author_processed s ON s.work_id=p.work_id
            WHERE t.kind='metadata' AND t.active=1 AND p.status='included_verified_research'
            AND s.status='cache_processed' AND (t.payload LIKE '%publisher_affiliation_structure%'
                OR t.payload LIKE '%complete_byline%')""").fetchall()
        atomic_json(scope_path, {'jobs': rows, 'scope': 'Current qualified missing-structure tasks with previously applied native cache; fixed discovery, not all-source closure'})
    db.close()
    groups = defaultdict(lambda: defaultdict(list))
    for wid, shard, receipt in json.loads(scope_path.read_text())['jobs']:
        groups[shard][wid].append(receipt)
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]'))
                for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    counts = Counter()
    selected = []
    columns = ['work_id', 'doi', 'source_id', 'title', 'journal_name', 'authors_json']
    with (FOLDER / 'reviews.jsonl').open('w') as log:
        for shard, targets in groups.items():
            for batch in pq.ParquetFile(ROOT / 'fields/papers' / shard).iter_batches(batch_size=100, columns=columns):
                for paper in batch.to_pylist():
                    wid = paper['work_id']
                    if wid not in targets:
                        continue
                    current = json.loads(paper['authors_json'])
                    outcomes = []
                    eligible = []
                    for receipt in targets[wid]:
                        path = ROOT / receipt
                        if not path.is_file():
                            outcomes.append((receipt, 'cached_receipt_missing'))
                            continue
                        if path.suffix == '.gz' and not use_native_adapters:
                            outcomes.append((receipt, 'raw_cache_requires_source_specific_adapter'))
                            continue
                        if use_native_adapters:
                            record, _, native = prepare_source(ROOT, paper, receipt, journals[paper['source_id']],
                                aps_only=receipt.startswith('fields/qualification/aps_all_raw/'))
                        else:
                            record = json.loads(path.read_text())
                            native = record.get('parsed', {})
                        if (not use_native_adapters and record.get('source') not in {'pmc', 'publisher_jats', 'publisher_native_xml'}) or not native.get('authors'):
                            outcomes.append((receipt, 'different_adapter_or_missing_native_byline'))
                            continue
                        identity = record.get('parsed', {}) if use_native_adapters else native
                        if not signature_match(paper, identity, journals[paper['source_id']]) or paper['doi'] not in {v.lower() for v in identity.get('doi', [])}:
                            outcomes.append((receipt, 'identity_unconfirmed'))
                            continue
                        if any(a.get('unresolved_affiliation_refs') for a in native['authors']):
                            outcomes.append((receipt, 'incomplete_native_affiliation_references'))
                            continue
                        source = {**native, 'evidence_receipt': 'fields/qualification/' + record['raw_receipt']}
                        rebuilt, status = reconcile(current, source)
                        if status == 'publisher_roster_reconciled':
                            if [key(a['name']) for a in rebuilt] == [key(a['name']) for a in current]:
                                status = 'roster_already_agrees_no_rebuild'
                            else:
                                eligible.append((receipt, byline_fingerprint(native['authors'])))
                        outcomes.append((receipt, status))
                    if eligible and len({v[1] for v in eligible}) == 1:
                        selected.append({'work_id': wid, 'doi': paper['doi'], 'shard': shard,
                                         'receipt': eligible[0][0], 'basis': 'Complete matched native roster; existing Nature/primary/group safeguards retained; no inferred person IDs'})
                        result = 'native_roster_repair_selected'
                    else:
                        result = 'native_roster_conflict' if eligible else 'no_eligible_roster_change'
                    counts[result] += 1
                    log.write(json.dumps({'work_id': wid, 'doi': paper['doi'], 'status': result,
                                          'source_outcomes': outcomes, 'final_task_closure': False}) + '\n')
            atomic_json(FOLDER / 'discovery_status.json', {'status': 'running', 'last_shard': shard,
                        'reviewed_papers': sum(counts.values()), 'outcomes': dict(counts), 'network_requests': 0})
    atomic_json(FOLDER / 'duplicate_byline_evidence.json', {'records': selected, 'network_requests': 0})
    atomic_json(FOLDER / 'discovery_status.json', {'status': 'fixed_cache_scope_reviewed',
                'reviewed_papers': sum(counts.values()), 'outcomes': dict(counts), 'network_requests': 0,
                'application_required': True, 'final_task_closure': False})
    print(json.dumps(dict(counts)), flush=True)


if __name__ == '__main__':
    run()
