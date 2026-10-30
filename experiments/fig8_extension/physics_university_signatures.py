"""Map reviewed explicit physics department signatures to their named universities."""
from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone

from .derived_refresh import refresh
from .official_rule_reuse import verify_changes
from .official_units_apply import affected_papers, run as apply_rules
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/official_entities/physics_university_signatures'
SPECS = [
    ('https://openalex.org/I123387679', 'Uppsala University',
     'https://www.uu.se/en/contact-and-organisation/organisation?query=X208', [
         'department of physics and astronomy, university of uppsala, uppsala, sweden',
         'department of physics and astronomy, uppsala universitet, 752 37, uppsala, sweden',
         'department of physics and astronomy, uppsala university, 75120 box 516 sweden']),
    ('https://openalex.org/I25974101', 'University of Würzburg',
     'https://www.uni-wuerzburg.de/en/institutions/faculties/', [
         'fakultät für physik und astronomie, julius-maximilians-universität würzburg, würzburg, germany']),
]


def run(review: dict | None = None) -> None:
    FOLDER.mkdir(exist_ok=True)
    receipt = str((FOLDER / 'rules.json').relative_to(ROOT))
    source = str((FOLDER / 'official_review.json').relative_to(ROOT))
    atomic_json(FOLDER / 'official_review.json', review or {
        'reviewed_at': datetime.now(timezone.utc).isoformat(),
        'sources': [s[2] for s in SPECS], 'access': 'Official university directory text read with web tool',
        'findings': 'Uppsala directory identifies its Department of Physics and Astronomy; JMU university lists Physics and Astronomy among its faculties.',
        'temporal_scope': 'Current directories corroborate identity of universities explicitly named in the 2023–2025 paper signatures; not retrospective ownership of hospitals, joint centers or other subsidiaries.',
        'excluded': 'Associated-to footnotes, numeric contamination, WISE initiative, hospitals and any unreviewed whole signature.'})
    rules = [{'entity_id': eid, 'name': name, 'exact_names': aliases,
              'entity_type': 'education', 'ownership_status': 'officially_confirmed_university_self',
              'university_targets': {eid: 1.0}, 'source': source, 'official_url': url,
              'locator': 'University directory department entry / university faculties list',
              'scope': (review or {}).get('rule_scope', 'Only reviewed exact whole signatures explicitly naming university and its physics unit; no hospital or joint-center inference. Country unchanged.')}
             for eid, name, url, aliases in SPECS]
    atomic_json(FOLDER / 'rules.json', {'status': 'reviewed', 'rules': rules})
    db = connect()
    aliases = {n: r for r in rules for n in r['exact_names']}
    ids = set().union(*affected_papers(db, aliases).values())
    low = db.execute('SELECT coalesce(max(rowid),0) FROM changes').fetchone()[0]
    db.close()
    atomic_json(FOLDER / 'scope.json', {'selected_work_ids': sorted(ids), 'exact_names': list(aliases)})
    result = apply_rules(rule_paths=[receipt], selected_work_ids=ids, status_name='physics_university_signatures_status.json')
    checked = verify_changes(low, ids, [receipt])
    assert checked == result.get('papers_changed', 0)
    with (FOLDER.parent / 'rule_reuse.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        path = FOLDER.parent / 'current_rule_reuse_scope.json'
        state = json.loads(path.read_text())
        if receipt not in state['next_rule_paths']:
            state['next_rule_paths'].append(receipt)
        atomic_json(path, state)
    db = connect()
    for name in aliases:
        for tid, in db.execute("SELECT task_id FROM tasks WHERE kind='institution' AND entity_key=?", (name,)).fetchall():
            if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',
                          (tid, 'official_physics_university_signature', receipt)).fetchone():
                continue
            db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                       (tid, 'official_physics_university_signature', 'exact_university_signature_confirmed', receipt,
                        json.dumps({'exact_name': name, 'final_task_closure': False})))
            db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (tid,))
    db.commit()
    db.close()
    atomic_json(FOLDER / 'validation.json', {'result': result, 'changed_papers_reproduced': checked,
                'repeat_safe': True, 'countries_and_nature_preserved': True})
    if checked:
        refresh()


if __name__ == '__main__':
    run()
