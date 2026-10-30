"""Apply finite Siena self signatures corroborated by official indexed identity."""
from __future__ import annotations

import fcntl
import json

import pyarrow.parquet as pq
from lxml import etree

from .official_rule_reuse import verify_changes
from .official_units_apply import affected_papers, run as apply_rules
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .terminal_country_apply import native_metadata
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/official_entities/siena_spbu_self'
NAMES = ['università degli studi di siena', 'università di siena',
         'università di siena, siena, italy']


def run() -> None:
    """Check native signatures, apply once, and retain all unsuccessful checks."""
    if (FOLDER / 'application.json').exists():
        return
    receipt = str((FOLDER / 'rules.json').relative_to(ROOT))
    proof = str((FOLDER / 'official_review.json').relative_to(ROOT))
    rule = dict(entity_id='https://openalex.org/I102064193', name='University of Siena',
                exact_names=NAMES, entity_type='education',
                ownership_status='officially_confirmed_university_self',
                university_targets={'https://openalex.org/I102064193': 1.0},
                source=proof, official_url='https://www.unisi.it/comunicazione/identita-visiva',
                scope='Finite standalone signed university names only, 2023–2025; official indexed '
                      'identity and matched native paper signature. No hospital, department, joint '
                      'center, branch or country inference. Direct source access restricted.')
    db = connect()
    assert db.execute('SELECT name FROM identities WHERE entity_id=?',
                      (rule['entity_id'],)).fetchone()[0] == rule['name']
    groups = affected_papers(db, {name: rule for name in NAMES})
    low = db.execute('SELECT coalesce(max(rowid),0) FROM changes').fetchone()[0]
    db.close()
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]'))
                for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    confirmed, checks = set(), []
    for shard, ids in groups.items():
        papers = pq.read_table(ROOT / 'fields/papers' / shard,
                               filters=[('work_id', 'in', sorted(ids))]).to_pylist()
        for paper in papers:
            receipts = {a['publisher_affiliation_source'] for a in json.loads(paper['authors_json'])
                        if a.get('publisher_affiliation_source') and any(
                            ' '.join(s.get('text', '').casefold().split()) in NAMES
                            for s in a.get('publisher_affiliation_slots') or [])}
            for source in receipts:
                check = dict(work_id=paper['work_id'], receipt=source, confirmed=False)
                try:
                    native = native_metadata(source, paper['doi'])
                    names = {' '.join(s['text'].casefold().split()) for a in native.get('authors', [])
                             for s in a.get('affiliations', [])}
                    matched = (paper['doi'] in native.get('doi', []) and
                               signature_match(paper, native, journals[paper['source_id']]))
                    check['confirmed'] = bool(matched and set(NAMES) & names)
                    if check['confirmed']:
                        confirmed.add(paper['work_id'])
                except (OSError, ValueError, etree.XMLSyntaxError) as exc:
                    check['error'] = str(exc)
                checks.append(check)
    atomic_json(FOLDER / 'rules.json', dict(status='reviewed', rules=[rule], native_checks=checks))
    result = apply_rules(rule_paths=[receipt], selected_work_ids=confirmed,
                         status_name='siena_spbu_self/application.json')
    checked = verify_changes(low, confirmed, [receipt])
    assert checked == result.get('papers_changed', 0)
    with (FOLDER.parent / 'rule_reuse.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        path = FOLDER.parent / 'current_rule_reuse_scope.json'
        state = json.loads(path.read_text())
        if receipt not in state['next_rule_paths']:
            state['next_rule_paths'].append(receipt)
        atomic_json(path, state)
    atomic_json(FOLDER / 'validation.json', dict(application=result, native_checks=checks,
        selected_papers=sum(map(len, groups.values())), native_confirmed_papers=len(confirmed),
        changed_papers_reproduced=checked, repeat_safe=True, countries_and_nature_preserved=True,
        final_task_closure=False, full_coverage_refresh='pending_coalesced_refresh'))


if __name__ == '__main__':
    run()
