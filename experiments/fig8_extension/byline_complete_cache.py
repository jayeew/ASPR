"""Confirm complete declared rosters from previously reviewed native equivalents."""
from __future__ import annotations

import fcntl
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .arrow_row_patch import patch_rows, selected_rows
from .derived_refresh import refresh
from .primary_byline_comparison import compare, incoming_metadata, load_primary
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/native_equivalent_byline'
CONFIRMED = {'publisher_byline_reconstructed_unique_exact_links',
             'publisher_complete_roster_reconciled_unmatched_person_ids_unknown'}


def validate(folder: Path) -> None:
    receipt = str((folder / 'review.json').relative_to(ROOT))
    review = json.loads((ROOT / receipt).read_text())
    proofs = {r['work_id']: r for r in review['records']}
    db = connect(); changed = set()
    for wid, field, before, after in db.execute(
            'SELECT work_id,field,before_value,after_value FROM changes WHERE receipt=?', (receipt,)):
        assert field in {'full_authorships_status', 'author_list_may_be_truncated', 'authorships_source_receipt'}
        assert json.loads(before) == proofs[wid]['before'][field]
        assert json.loads(after) == proofs[wid]['after'][field]
        changed.add(wid)
    db.close()
    assert changed == set(proofs)
    assert all((ROOT / r['incoming_receipt']).is_file() for r in proofs.values())
    atomic_json(FOLDER / 'validation.json', {'changed_papers': len(changed),
        'only_roster_status_and_source_fields_changed': True,
        'authors_institutions_countries_and_external_person_ids_unchanged': True,
        'source_records_exist': True, 'application_review': receipt, 'final_task_closure': False})


def run(name_typography: bool = False) -> None:
    compare_native = compare
    if name_typography:
        from .byline_typography import compare_typography
        compare_native = compare_typography
    FOLDER.mkdir(exist_ok=True)
    with (FOLDER / 'runner.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = FOLDER / 'state.json'
        low = json.loads(state_path.read_text())['event_offset'] if state_path.exists() else 0
        events = ROOT / 'supplement/snapshot_location_recovery/byline_comparison/review_events.jsonl'
        candidates = {}
        with events.open() as stream:
            stream.seek(low)
            for line in stream:
                item = json.loads(line)
                if item['status'] == ('author_sequence_differs' if name_typography else 'equivalent_complete_declared_links_entity_mapping_not_asserted'):
                    candidates[item['work_id']] = item
            high = stream.tell()
        if high == low:
            print(json.dumps({'new_events': 0}), flush=True)
            return
        folder = FOLDER / f'{low}-{high}'; folder.mkdir(exist_ok=True)
        journals = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT / 'fields/journal_registry.csv')}
        previous_path = folder / 'review.json'
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
        counts = Counter(previous.get('counts', {})); audit = previous.get('records', [])
        with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            db = connect(); groups = defaultdict(set)
            for wid in candidates:
                row = db.execute("SELECT shard FROM papers WHERE work_id=? AND status='included_verified_research'", (wid,)).fetchone()
                if row:
                    groups[row[0]].add(wid)
            for shard, ids in groups.items():
                path = ROOT / 'fields/papers' / shard; table = pq.read_table(path); updates = {}
                for paper in selected_rows(table, ids):
                    wid = paper['work_id']; proof = candidates[wid]
                    if paper['full_authorships_status'] in CONFIRMED:
                        counts['existing_publisher_roster_preserved'] += 1
                        continue
                    authors = json.loads(paper['authors_json'])
                    if any(a.get('nature_v1_slots') is not None for a in authors):
                        counts['nature_preserved'] += 1
                        continue
                    if hashlib.sha256(paper['authors_json'].encode()).hexdigest() != proof['retained_authors_sha256']:
                        counts['current_author_state_changed_review_preserved'] += 1
                        continue
                    record, raw = load_primary(proof['incoming_receipt'], paper['doi'])
                    if hashlib.sha256(raw).hexdigest() != proof.get('incoming_raw_sha256'):
                        counts['raw_source_changed_review_preserved'] += 1
                        continue
                    native = incoming_metadata(record, raw)
                    if (paper['doi'] not in native.get('doi', []) or not signature_match(paper, native, journals[paper['source_id']])
                            or compare_native(authors, native) != 'equivalent_complete_declared_links_entity_mapping_not_asserted'):
                        counts['native_equivalence_not_reproduced'] += 1
                        continue
                    values = {'full_authorships_status': 'publisher_byline_reconstructed_unique_exact_links',
                              'author_list_may_be_truncated': False,
                              'authorships_source_receipt': proof['incoming_receipt']}
                    receipt = str((folder / 'review.json').relative_to(ROOT))
                    for field, value in values.items():
                        if paper[field] != value:
                            db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                                       (wid, field, json.dumps(paper[field]), json.dumps(value), receipt,
                                        'Complete native declared byline has unique ordered names under the recorded name-comparison rule and exact affiliation links; roster, person IDs, countries and institutions unchanged'))
                    updates[wid] = values
                    audit.append({**proof, 'before': {k: paper[k] for k in values}, 'after': values,
                                  'authors_json_unchanged_sha256': proof['retained_authors_sha256'],
                                  'external_person_ids_not_confirmed': True,
                                  'name_comparison': 'Hyphens as spaces and periods omitted; word boundaries and order retained' if name_typography else 'Exact casefolded whitespace-normalized names',
                                  'native_name_differences': [(a['name'], b['name']) for a, b in zip(authors, native['authors']) if a['name'] != b['name']]})
                    counts['papers_roster_confirmed'] += 1
                if updates:
                    temp = path.with_suffix('.parquet.tmp'); pq.write_table(patch_rows(table, updates), temp, compression='zstd');temp.replace(path)
                # Persist source records before committing their change-log references.
                atomic_json(folder / 'review.json', {'records': audit, 'counts': dict(counts), 'network_requests': 0, 'final_task_closure': False})
                db.commit()
            db.close()
        atomic_json(folder / 'review.json', {'records': audit, 'counts': dict(counts), 'network_requests': 0, 'final_task_closure': False})
        validate(folder)
        atomic_json(state_path, {'event_offset': high, 'last_review': str((folder / 'review.json').relative_to(ROOT)),
                                'counts': dict(counts), 'network_requests': 0, 'final_task_closure': False})
        if counts['papers_roster_confirmed']:
            refresh()
        print(json.dumps(dict(counts)), flush=True)


if __name__ == '__main__':
    run()
    from .byline_typography import run as confirm_typography
    confirm_typography()
