"""Corroborate academy self signatures without allocating them to universities."""
from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone

from .derived_refresh import refresh
from .official_rule_reuse import verify_changes
from .official_units_apply import affected_papers, run as apply_rules
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

SOURCES = [
    ('https://openalex.org/I19820366', 'Chinese Academy of Sciences',
     'https://english.cas.cn/about-cas/profile/',
     'National science academy and research organization. A signature naming the academy '
     'alone does not name UCAS, USTC or any other constituent university.'),
    ('https://openalex.org/I84653119', 'Academia Sinica',
     'https://www.sinica.edu.tw/en/cp/712',
     'National research institution. Its graduate programs cooperate with universities; '
     'academy self signatures do not identify a participating university.'),
]


def record_reviews(rules: list[dict], receipt: str, source: str) -> int:
    """Record entity review without claiming all linked signatures are resolved."""
    db = connect()
    count = 0
    for rule in rules:
        for task_id, in db.execute("SELECT task_id FROM tasks WHERE kind='university_ownership' "
                                  "AND entity_key=?", (rule['entity_id'],)).fetchall():
            if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',
                          (task_id, source, receipt)).fetchone():
                continue
            db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                       (task_id, source, 'exact_research_self_nonuniversity_confirmed', receipt,
                        json.dumps({'entity_id': rule['entity_id'], 'scope': rule['scope'],
                                    'final_task_closure': False})))
            db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (task_id,))
            count += 1
    db.commit()
    db.close()
    return count


def run(sources: list[tuple[str, str, str, str]] | None = None,
        folder_name: str = 'academy_self', source_receipts: dict[str, str] | None = None) -> None:
    sources = SOURCES if sources is None else sources
    folder = ROOT / 'supplement/official_entities' / folder_name
    folder.mkdir(exist_ok=True)
    receipt = str((folder / 'rules.json').relative_to(ROOT))
    proof = str((folder / 'official_review.json').relative_to(ROOT))
    rules = []
    db = connect()
    for entity, name, url, summary in sources:
        found = db.execute('SELECT name FROM identities WHERE entity_id=?', (entity,)).fetchone()
        assert found and found[0] == name
        rules.append({'entity_id': entity, 'name': name, 'exact_names': [name.casefold()],
            'entity_type': 'research_organization', 'ownership_status': 'officially_confirmed_nonuniversity',
            'university_targets': {}, 'source': proof, 'official_url': url,
            'scope': 'Exact whole signed research-organization name only; no inferred abbreviation, institute, joint center, '
                     'graduate-program or subsidiary inference; retain country and Nature reviewed records.'})
    atomic_json(folder / 'official_review.json', {'reviewed_at': datetime.now(timezone.utc).isoformat(),
        'sources': [{'entity_id': e, 'name': n, 'url': u, 'summary': s,
                     **({'receipt': source_receipts[e]} if source_receipts and e in source_receipts else {})}
                    for e, n, u, s in sources],
        'access': 'Official raw HTTP receipt retained for each listed receipt' if source_receipts else 'Official web text read; raw HTTP not retained',
        'interpretation': 'Named entity self is a non-university research organization; separate university '
                          'signatures remain independently attributable. Not an assertion that no '
                          'academy constituent has university status.'})
    atomic_json(folder / 'rules.json', {'status': 'reviewed', 'rules': rules})
    aliases = {name: rule for rule in rules for name in rule['exact_names']}
    ids = set().union(*affected_papers(db, aliases).values())
    low = db.execute('SELECT coalesce(max(rowid),0) FROM changes').fetchone()[0]
    db.close()
    atomic_json(folder / 'scope.json', {'selected_work_ids': sorted(ids), 'rules': len(rules)})
    result = apply_rules(rule_paths=[receipt], selected_work_ids=ids, status_name=folder_name+'_status.json')
    checked = verify_changes(low, ids, [receipt])
    assert checked == result.get('papers_changed', 0)
    with (folder.parent / 'rule_reuse.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        path = folder.parent / 'current_rule_reuse_scope.json'
        state = json.loads(path.read_text())
        if receipt not in state['next_rule_paths']:
            state['next_rule_paths'].append(receipt)
        atomic_json(path, state)
    atomic_json(folder / 'validation.json', {'result': result, 'changed_papers_reproduced': checked,
        'repeat_safe': True, 'country_and_nature_preserved': True, 'final_task_closure': False})
    atomic_json(folder / 'ledger_status.json', {'new_review_attempts': record_reviews(
        rules, receipt, folder_name+'_official_research_self'), 'final_task_closure': False})
    refresh()


if __name__ == '__main__':
    run()
