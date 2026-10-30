"""Apply reviewed official England university names to exact primary signatures."""
from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone

from .derived_refresh import refresh
from .official_rule_reuse import verify_changes
from .official_units_apply import affected_papers, run as apply_rules
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/official_entities/england_recognised_bodies'
# Reviewed PDF schedule, pages 4–6. Only leading definite article normalized.
NAMES = ['The University of ' + n for n in (
    'Bath|Birmingham|Bradford|Buckingham|Chichester|Cumbria|East Anglia|Essex|'
    'Huddersfield|Hull|Kent|Leeds|Leicester|Liverpool|Manchester|Reading|'
    'Sheffield|Surrey|Warwick|West London|Westminster').split('|')]
NAMES += ['University of ' + n for n in (
    'Bedfordshire|Brighton|Bristol|Cambridge|Chester|Derby|East London|Exeter|'
    'Gloucestershire|Greenwich|Lincoln|Southampton|Suffolk|Sunderland|Sussex|'
    'Winchester|Wolverhampton|Worcester|York').split('|')]


def normalized(name: str) -> str:
    return ' '.join(name.casefold().split()).removeprefix('the ')


def record_attempts(rules: list[dict], receipt: str, source_name: str = 'england_official_university_self') -> None:
    db = connect()
    for rule in rules:
        tasks = db.execute("SELECT task_id FROM tasks WHERE kind='university_ownership' AND entity_key=?",
                           (rule['entity_id'],)).fetchall()
        for task_id, in tasks:
            if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',
                          (task_id, source_name, receipt)).fetchone():
                continue
            db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                       (task_id, source_name, 'exact_university_identity_reviewed', receipt,
                        json.dumps({'entity_id': rule['entity_id'], 'final_task_closure': False,
                                    'scope': rule['scope']})))
            db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (task_id,))
    db.commit()
    db.close()


def run() -> None:
    source = str((FOLDER / 'official_review.json').relative_to(ROOT))
    receipt = str((FOLDER / 'rules.json').relative_to(ROOT))
    url = 'https://www.legislation.gov.uk/uksi/2020/573/pdfs/uksi_20200573_en.pdf'
    atomic_json(FOLDER / 'official_review.json', {
        'url': url, 'reviewed_at': datetime.now(timezone.utc).isoformat(),
        'access': 'web tool PDF text, 8 pages, schedule pages 4–6',
        'raw_pdf_available': False, 'direct_pdf_status': 202, 'direct_pdf_bytes': 0,
        'effective_date': '2020-06-18', 'reviewed_names': NAMES,
        'scope': 'Named university self identity only, corroborated by exact 2023–2025 primary signature. No claim that all recognised degree bodies are universities or that the 2020 list proves current parent relationships.',
        'excluded': 'Colleges, legal suffix normalization, renames, campuses, hospitals, joint centers and nonunique matches.'})
    db = connect()
    entities = db.execute('SELECT entity_id,name,entity_type,countries,ownership_status FROM identities').fetchall()
    rule_file = FOLDER / 'rules.json'
    rules = json.loads(rule_file.read_text())['rules'] if rule_file.exists() else []
    existing = {r['entity_id'] for r in rules}
    held = []
    for official in NAMES:
        matches = [r for r in entities if normalized(r[1]) == normalized(official)
                   and r[2] == 'education' and json.loads(r[3]) == ['GB']]
        if len(matches) != 1:
            held.append({'name': official, 'reason': 'no_unique_existing_GB_education_entity'}); continue
        entity, name, _, _, status = matches[0]
        if entity in existing:
            continue
        if status != 'unverified':
            held.append({'name': official, 'reason': 'existing_reviewed_status_preserved', 'status': status}); continue
        rules.append({'entity_id': entity, 'name': name,
                      'exact_names': sorted({normalized(official), official.casefold(), name.casefold()}),
                      'entity_type': 'education', 'ownership_status': 'officially_confirmed_university_self',
                      'university_targets': {entity: 1.0}, 'source': source, 'official_url': url,
                      'locator': 'Schedule, PDF pages 4–6',
                      'scope': 'Exact whole university-self signature only; leading The optional; no country changes or subsidiary affiliations inferred.'})
    atomic_json(FOLDER / 'rules.json', {'status': 'reviewed', 'rules': rules})
    aliases = {n: r for r in rules for n in r['exact_names']}
    ids = set().union(*affected_papers(db, aliases).values()) if aliases else set()
    low = db.execute('SELECT coalesce(max(rowid),0) FROM changes').fetchone()[0]
    db.close()
    atomic_json(FOLDER / 'scope.json', {'selected_work_ids': sorted(ids), 'held': held, 'rules': len(rules)})
    result = apply_rules(rule_paths=[receipt], selected_work_ids=ids, status_name='england_self_apply_status.json')
    checked = verify_changes(low, ids, [receipt])
    assert checked == result.get('papers_changed', 0)
    with (FOLDER.parent / 'rule_reuse.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        state_path = FOLDER.parent / 'current_rule_reuse_scope.json'
        state = json.loads(state_path.read_text())
        if receipt not in state['next_rule_paths']:
            state['next_rule_paths'].append(receipt)
        atomic_json(state_path, state)
    atomic_json(FOLDER / 'validation.json', {'result': result, 'changed_papers_reproduced': checked,
        'idempotent_slot_updates': True, 'country_and_nature_preserved': True,
        'final_task_closure': False})
    record_attempts(rules, receipt)
    if rules:
        refresh()


if __name__ == '__main__':
    run()
