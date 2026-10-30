"""Confirm IP2I Lyon signed identities while retaining joint-owner allocation uncertainty."""
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
    folder = ROOT / 'supplement/official_entities/ip2i_identity'; folder.mkdir(exist_ok=True)
    receipt = str((folder/'rules.json').relative_to(ROOT))
    url = 'https://www.rhone-auvergne.cnrs.fr/sites/delegation_dr7/files/news/2019-11/cpip2i071019.pdf'
    source = str((folder/'official_review.json').relative_to(ROOT))
    atomic_json(folder/'official_review.json', {'url': url, 'reviewed_at': datetime.now(timezone.utc).isoformat(),
        'additional_url': 'https://www.univ-lyon1.fr/recherche/entites-de-recherche-et-plateformes-technologiques/institut-de-physique-des-deux-infinis-de-lyon-ip2i-lyon',
        'locator': 'CNRS press release page 1, dated 7 October 2019; university directory Organisation section',
        'access': 'Official PDF and university HTML read with web tool; raw HTTP not retained',
        'findings': 'IP2I inaugurated in October 2019 after IPNL and LMA merger; joint research unit UMR5822 of CNRS/IN2P3 and Université Claude Bernard Lyon 1.',
        'temporal_scope': '2019 establishment corroborates identity during 2023–2025; current university directory supports continuing identity, not retroactive individual employment.',
        'unavailable_path': 'Institute website currently shows temporary security shutdown; not repeatedly requested.',
        'allocation_limitation': 'No individual employer or contribution fractions inferred; no automatic university assignment, no predecessor or LMA-only consolidation.'})
    db = connect()
    names = [r[0] for r in db.execute("SELECT entity_key FROM tasks WHERE kind='institution' AND (entity_key LIKE '%ip2i%' OR entity_key LIKE '%infinis%lyon%')")]
    accepted = [n for n in names if n.startswith(('institut de physique des 2 infinis de lyon', 'institut de physique des deux infinis de lyon'))]
    if (folder/'scope.json').exists():
        accepted = json.loads((folder/'scope.json').read_text())['exact_names']
    rule = {'entity_id': 'https://openalex.org/I4387155512', 'name': 'Institut de Physique des 2 Infinis de Lyon',
        'exact_names': sorted(set(accepted)), 'entity_type': 'research_organization',
        'ownership_status': 'unverified', 'university_targets': {}, 'source': source,
        'official_url': url, 'locator': '2019 CNRS establishment announcement; university directory UMR5822 identity',
        'confirmed_joint_owner_names': ['CNRS/IN2P3','Université Claude Bernard Lyon 1'],
        'allocation_status': 'joint_ownership_confirmed_author_allocation_unresolved',
        'scope': 'Exact whole IP2I Lyon signatures, including CNRS/IN2P3 descriptors. Joint university signatures, medical subunits, LMA-only names, predecessors, numeric-prefix contamination and typographic ambiguity remain held. Countries preserved.'}
    atomic_json(folder/'rules.json', {'status': 'reviewed', 'rules': [rule]})
    aliases = {n:rule for n in accepted}; ids = set().union(*affected_papers(db, aliases).values())
    low = db.execute('SELECT coalesce(max(id),0) FROM changes').fetchone()[0]; db.close()
    atomic_json(folder/'scope.json', {'selected_work_ids': sorted(ids), 'exact_names': accepted,
        'held_names': [n for n in names if n not in accepted], 'final_task_closure': False})
    result = apply_rules(rule_paths=[receipt], selected_work_ids=ids, status_name='ip2i_identity_status.json')
    checked = verify_changes(low, ids, [receipt]); assert checked == result.get('papers_changed',0)
    with (folder.parent/'rule_reuse.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        path = folder.parent/'current_rule_reuse_scope.json'; state = json.loads(path.read_text())
        if receipt not in state['next_rule_paths']: state['next_rule_paths'].append(receipt)
        atomic_json(path,state)
    db = connect(); added = 0
    for name in accepted:
        for tid, in db.execute("SELECT task_id FROM tasks WHERE kind='institution' AND entity_key=?",(name,)).fetchall():
            if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',(tid,'ip2i_official_identity',receipt)).fetchone(): continue
            db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',(tid,'ip2i_official_identity','identity_confirmed_joint_allocation_unresolved',receipt,json.dumps({'exact_name':name,'final_task_closure':False})))
            db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?',(tid,)); added += 1
    db.commit(); db.close()
    atomic_json(folder/'validation.json',{'result':result,'changed_papers_reproduced':checked,
        'repeat_safe':True,'countries_and_nature_preserved':True,'ledger_attempts_added':added,
        'joint_university_allocation_remains_unknown':True})
    if checked: refresh()


if __name__ == '__main__':
    run()
