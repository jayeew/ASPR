"""Confirm IFCA signed identities while retaining joint-owner allocation uncertainty."""
from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone

from .derived_refresh import refresh
from .official_rule_reuse import verify_changes
from .official_units_apply import affected_papers, run as apply_rules
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT / 'supplement/official_entities/ifca_identity'; folder.mkdir(exist_ok=True)
    receipt = str((folder/'rules.json').relative_to(ROOT))
    url = 'https://www.boe.es/diario_boe/txt.php?id=BOE-A-2024-26102'
    source = str((folder/'official_review.json').relative_to(ROOT))
    atomic_json(folder/'official_review.json', {'url': url, 'reviewed_at': datetime.now(timezone.utc).isoformat(),
        'locator': 'Preamble paragraphs 6–7; clauses 1 and 3; official signature provision',
        'access': 'Web tool official gazette text; raw HTTP not retained',
        'publication_date': '2024-12-14',
        'findings': 'IFCA established in 1995, jointly held by CSIC and Universidad de Cantabria; 2024 agreement continues joint status. Not an independent university and not exclusively attributable to either owner.',
        'temporal_scope': 'Historical institute identity and continuing joint status; no back-projection of new detailed 2024 governance conditions.',
        'allocation_limitation': 'Joint ownership confirmed; individual author employer and contribution fractions not established. Do not assign all papers to Cantabria or invent 50:50 weights.'})
    db = connect()
    names = [r[0] for r in db.execute("SELECT entity_key FROM tasks WHERE kind='institution' AND (entity_key LIKE '%física de cantabria%' OR entity_key LIKE '%fisica de cantabria%')")]
    accepted = [n for n in names if len(n)<220 and (n.startswith('instituto de física de cantabria') or n.startswith('instituto de fisica de cantabria') or n.startswith('112instituto de física de cantabria'))]
    if (folder/'scope.json').exists():
        accepted = json.loads((folder/'scope.json').read_text())['exact_names']
    rule = {'entity_id': 'https://openalex.org/I4210152024', 'name': 'Instituto de Física de Cantabria',
        'exact_names': sorted(set(accepted)), 'entity_type': 'research_organization',
        'ownership_status': 'unverified', 'university_targets': {}, 'source': source,
        'official_url': url, 'locator': 'Preamble 6–7 and clauses 1,3',
        'confirmed_joint_owner_names': ['CSIC','Universidad de Cantabria'],
        'allocation_status': 'joint_ownership_confirmed_author_allocation_unresolved',
        'scope': 'Exact whole IFCA signatures including CSIC-UC naming; no bare CSIC-UC, IBBTEC or hospital inference. Country preserved.'}
    atomic_json(folder/'rules.json', {'status': 'reviewed', 'rules': [rule]})
    aliases = {n:rule for n in accepted}; ids = set().union(*affected_papers(db, aliases).values())
    low = db.execute('SELECT coalesce(max(id),0) FROM changes').fetchone()[0]; db.close()
    atomic_json(folder/'scope.json', {'selected_work_ids': sorted(ids), 'exact_names': accepted,
        'held_names': [n for n in names if n not in accepted], 'final_task_closure': False})
    result = apply_rules(rule_paths=[receipt], selected_work_ids=ids, status_name='ifca_identity_status.json')
    checked = verify_changes(low, ids, [receipt]); assert checked == result.get('papers_changed',0)
    with (folder.parent/'rule_reuse.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        path = folder.parent/'current_rule_reuse_scope.json'; state = json.loads(path.read_text())
        if receipt not in state['next_rule_paths']: state['next_rule_paths'].append(receipt)
        atomic_json(path,state)
    db = connect(); added = 0
    for name in accepted:
        for tid, in db.execute("SELECT task_id FROM tasks WHERE kind='institution' AND entity_key=?",(name,)).fetchall():
            if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',(tid,'ifca_official_gazette',receipt)).fetchone(): continue
            db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',(tid,'ifca_official_gazette','identity_confirmed_joint_allocation_unresolved',receipt,json.dumps({'exact_name':name,'final_task_closure':False})))
            db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?',(tid,)); added += 1
    db.commit(); db.close()
    atomic_json(folder/'validation.json',{'result':result,'changed_papers_reproduced':checked,
        'repeat_safe':True,'countries_and_nature_preserved':True,'ledger_attempts_added':added,
        'joint_university_allocation_remains_unknown':True})
    if checked: refresh()


if __name__ == '__main__':
    run()
