"""Apply individually reviewed postal variants of exact DOE laboratory signatures."""
from __future__ import annotations

import fcntl
import json
from collections import defaultdict

import pyarrow.parquet as pq

from .official_rule_reuse import verify_changes
from .official_units_apply import run as apply_rules
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .terminal_country_apply import native_metadata
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/official_entities/doe_lab_self/signed_addresses'


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    if (FOLDER / 'application.json').exists():
        return
    reviewed = json.loads((FOLDER.parent / 'signed_address_review.json').read_text())['records']
    selected_names = [r for r in reviewed if r['decision'] == 'native_signature_check_required']
    original = json.loads((FOLDER.parent / 'rules.json').read_text())['rules']
    names = defaultdict(set)
    for row in selected_names:
        names[row['entity_id']].add(row['name'])
    rules = [{**r, 'exact_names': sorted(names[r['entity_id']]),
              'scope': 'Individually reviewed whole laboratory name plus postal address, corroborated in native paper. Country unchanged; no subunit, operator, joint organization or campus inference.'}
             for r in original if names[r['entity_id']]]
    aliases = {n for r in rules for n in r['exact_names']}
    db = connect()
    groups = defaultdict(set)
    for row in selected_names:
        for wid, shard in db.execute('SELECT p.work_id,p.shard FROM task_papers l JOIN papers p USING(work_id) '
                                    "WHERE l.task_id=? AND p.status='included_verified_research'", (row['task_id'],)):
            groups[shard].add(wid)
    db.close()
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    checks, confirmed = [], set()
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
            found = set()
            for receipt, expected in receipts.items():
                native = native_metadata(receipt, paper['doi'])
                valid = (paper['doi'] in {v.lower() for v in native.get('doi', [])}
                         and signature_match(paper, native, journals[paper['source_id']]))
                native_names = {' '.join(s['text'].casefold().split())
                                for a in native.get('authors', []) for s in a.get('affiliations', [])}
                matched = expected & native_names if valid else set()
                found.update(matched)
                checks.append({'work_id': paper['work_id'], 'receipt': receipt,
                               'expected_names': sorted(expected), 'confirmed_names': sorted(matched)})
            if receipts and set.union(*receipts.values()) <= found:
                confirmed.add(paper['work_id'])
    receipt = str((FOLDER / 'rules.json').relative_to(ROOT))
    atomic_json(FOLDER / 'rules.json', {'status': 'reviewed', 'rules': rules, 'native_checks': checks})
    db = connect()
    low = db.execute('SELECT coalesce(max(rowid),0) FROM changes').fetchone()[0]
    db.close()
    result = apply_rules(rule_paths=[receipt], selected_work_ids=confirmed,
                        status_name='doe_lab_self/signed_addresses/application.json')
    checked = verify_changes(low, confirmed, [receipt])
    assert checked == result.get('papers_changed', 0)
    atomic_json(FOLDER / 'result.json', {'candidate_papers': sum(map(len, groups.values())),
        'native_confirmed_papers': len(confirmed), 'native_checks': checks, 'application': result,
        'changed_papers_reproduced': checked, 'country_and_nature_preserved': True,
        'network_requests': 0, 'final_task_closure': False})
    with (FOLDER.parent.parent / 'rule_reuse.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        path = FOLDER.parent.parent / 'current_rule_reuse_scope.json'
        state = json.loads(path.read_text())
        if receipt not in state['next_rule_paths']:
            state['next_rule_paths'].append(receipt)
        atomic_json(path, state)


if __name__ == '__main__':
    run()
