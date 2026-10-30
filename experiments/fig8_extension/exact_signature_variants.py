"""Reuse reviewed whole-signature identities across typography and label variants."""
from __future__ import annotations

import fcntl
import json
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

from .derived_refresh import refresh
from .official_rule_reuse import verify_changes
from .official_units_apply import run as apply_rules
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/official_entities/exact_signature_variants'
RECEIPT = 'supplement/official_entities/exact_signature_variants/rules.json'


def normalized_signature(value: str) -> str:
    """Remove only typography and a leading numerical affiliation label."""
    value = re.sub(r'^\d+\s*', '', value)
    value = ''.join(c for c in unicodedata.normalize('NFKD', value.casefold())
                    if not unicodedata.combining(c))
    return ''.join(c for c in value if c.isalnum())


def prepare(inventory: dict) -> tuple[dict, list[dict]]:
    """Recheck proposed exact aliases against their unchanged reviewed rule files."""
    groups: dict[tuple[str, str], dict] = {}
    unresolved: list[dict] = []
    for row in inventory['records']:
        document = json.loads((ROOT / row['rule_path']).read_text())
        candidates = [r for r in document['rules'] if r['entity_id'] == row['entity_id']
                      and row['matched_alias'] in r['exact_names']]
        if document['status'] != 'reviewed' or len(candidates) != 1:
            raise ValueError('Original exact rule changed: ' + row['signature'])
        rule = candidates[0]
        if rule != row['rule']:
            raise ValueError('Original exact rule payload changed')
        if normalized_signature(row['signature']) != normalized_signature(row['matched_alias']):
            raise ValueError('Variant changes signature words or digits')
        if not (ROOT / rule['source']).is_file():
            unresolved.append({**row, 'reason': 'reviewed_source_receipt_missing'})
            continue
        # No expansion of identities already reopened for invalid official support.
        reopen = ROOT / 'supplement/official_entities/nature_official_reference_reopen.json'
        reopened_ids = {r['entity_id'] for r in json.loads(reopen.read_text())['records']} if reopen.exists() else set()
        if rule['entity_id'] in reopened_ids:
            unresolved.append({**row, 'reason': 'official_source_support_reopened'})
            continue
        key = (row['rule_path'], row['entity_id'])
        target = groups.setdefault(key, {**rule, 'exact_names': [],
            'inherited_rule_receipt': row['rule_path'],
            'variant_rule': 'Exact normalized full signature only; punctuation/diacritics/leading numeric labels; no synonym, country, branch, hospital or parent substitution'})
        target['exact_names'].append(row['signature'])
    return {'status': 'reviewed', 'rules': list(groups.values()), 'network_requests': 0}, unresolved


def run(inventory_path: Path) -> dict:
    start = time.monotonic()
    inventory = json.loads(inventory_path.read_text())
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        FOLDER.mkdir(exist_ok=True)
        rules, unresolved = prepare(inventory)
        atomic_json(FOLDER / 'inventory.json', inventory)
        atomic_json(FOLDER / 'rules.json', rules)
        atomic_json(FOLDER / 'unresolved.json', {'records': unresolved})
        db = connect()
        low = db.execute('SELECT max(rowid) FROM changes').fetchone()[0]
        db.close()
        ids = set(inventory['paper_ids'])
        result = apply_rules(rule_paths=[RECEIPT], selected_work_ids=ids,
                             status_name='exact_signature_variants_apply.json', lock_held=True)
        checked = verify_changes(low, ids, [RECEIPT])
        if checked != result.get('papers_changed', 0):
            raise ValueError('Changed author records fail replay')
        state_path = ROOT / 'supplement/official_entities/current_rule_reuse_scope.json'
        state = json.loads(state_path.read_text())
        if RECEIPT not in state['next_rule_paths']:
            state['next_rule_paths'].append(RECEIPT)
        atomic_json(state_path, state)
        summary = {'candidate_signatures': len(inventory['records']), 'selected_papers': len(ids),
                   'rules': len(rules['rules']), 'unresolved_signatures': len(unresolved),
                   'changed_papers_reproduced': checked, 'repeat_safe': True,
                   'countries_and_nature_preserved': True, 'result': result,
                   'elapsed_seconds': time.monotonic() - start, 'network_requests': 0,
                   'final_task_closure': False}
        atomic_json(FOLDER / 'validation.json', summary)
    refresh()
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('inventory', type=Path)
    print(json.dumps(run(parser.parse_args().inventory)), flush=True)
