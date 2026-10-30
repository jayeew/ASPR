"""Apply cached US official degree-granting institution identities, never parents."""
from __future__ import annotations

import fcntl
import hashlib
import json

from .derived_refresh import refresh
from .england_university_self import record_attempts
from .official_rule_reuse import verify_changes
from .official_units_apply import affected_papers, run as apply_rules
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/official_entities/college_scorecard'


def run() -> None:
    with (FOLDER / 'application.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_and_apply()


def prepare_and_apply() -> None:
    acquisition = json.loads((FOLDER / 'acquisition.json').read_text())
    assert hashlib.sha256((FOLDER / 'bulk.zip').read_bytes()).hexdigest() == acquisition['sha256']
    candidates = json.loads((FOLDER / 'candidate_review.json').read_text())['candidates']
    receipt = str((FOLDER / 'rules.json').relative_to(ROOT))
    rules = json.loads((FOLDER / 'rules.json').read_text())['rules'] if (FOLDER / 'rules.json').exists() else []
    known = {r['entity_id'] for r in rules}
    held = []
    db = connect()
    for candidate in candidates:
        row = candidate['official_row']; entity = candidate['entity_id']
        if entity in known:
            continue
        if row['HIGHDEG'] not in {'3', '4'} or row['MAIN'] != '1' or row['CURROPER'] != '1':
            held.append({**candidate, 'reason': 'outside_operating_main_bachelor_or_graduate_scope'})
            continue
        current = db.execute('SELECT name,entity_type,countries,ownership_status FROM identities WHERE entity_id=?', (entity,)).fetchone()
        if current != (candidate['name'], 'education', '["US"]', 'unverified'):
            held.append({**candidate, 'reason': 'current_identity_or_reviewed_status_preserved'})
            continue
        rules.append({'entity_id': entity, 'name': candidate['name'],
            'exact_names': sorted({' '.join(n.casefold().split()) for n in [candidate['name'], row['INSTNM']]}),
            'entity_type': 'education', 'ownership_status': 'officially_confirmed_university_self',
            'university_targets': {entity: 1.0},
            'source': str((FOLDER / 'candidate_review.json').relative_to(ROOT)),
            'official_url': acquisition['url'], 'raw_receipt': acquisition['raw_receipt'],
            'raw_sha256': acquisition['sha256'],
            'locator': f"{candidate['member']}: CSV line {row['csv_line']}, UNITID={row['UNITID']}",
            'scope': 'Existing US higher-education self identity only, corroborated by exact whole primary signature. Current official directory is not historical parent ownership evidence. No hospital, department, branch, country or headquarters inference.'})
    atomic_json(FOLDER / 'rules.json', {'status': 'reviewed', 'rules': rules,
        'dictionary_receipt': 'supplement/official_entities/college_scorecard/dictionary_receipt.json',
        'dictionary_locator': 'Institution_Data_Dictionary: HIGHDEG rows 22-26, MAIN 14-15, CURROPER 558-559',
        'selection': 'Unique existing US education identity; main operating institution granting bachelor/graduate degrees. Code meanings do not establish historical ownership.'})
    aliases = {name: rule for rule in rules for name in rule['exact_names']}
    ids = set().union(*affected_papers(db, aliases).values()) if aliases else set()
    low = db.execute('SELECT coalesce(max(id),0) FROM changes').fetchone()[0]
    db.close()
    atomic_json(FOLDER / 'scope.json', {'selected_work_ids': sorted(ids), 'held': held, 'rules': len(rules)})
    result = apply_rules(rule_paths=[receipt], selected_work_ids=ids, status_name='college_scorecard_status.json')
    checked = verify_changes(low, ids, [receipt])
    assert checked == result.get('papers_changed', 0)
    with (FOLDER.parent / 'rule_reuse.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        path = FOLDER.parent / 'current_rule_reuse_scope.json'
        state = json.loads(path.read_text())
        if receipt not in state['next_rule_paths']:
            state['next_rule_paths'].append(receipt)
        atomic_json(path, state)
    record_attempts(rules, receipt, 'college_scorecard_official_self')
    atomic_json(FOLDER / 'validation.json', {'result': result, 'changed_papers_reproduced': checked,
        'countries_and_nature_preserved': True, 'final_task_closure': False})
    refresh()


if __name__ == '__main__':
    run()
