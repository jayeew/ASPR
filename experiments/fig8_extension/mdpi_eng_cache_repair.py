"""Apply the verified Eng alias from retained responses without restarting acquisition."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json

from lxml import etree

from .cached_date_apply import run as apply_sources
from .derived_refresh import refresh
from .mdpi_cache_apply import evidence
from .mdpi_targeted_collect import FOLDER, NS, TARGET, extract
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run() -> dict:
    folder = TARGET / 'eng_cache_application'
    folder.mkdir(exist_ok=True)
    with (folder / 'repair.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scope = {p['doi']: p for p in json.loads((TARGET / 'scope.json').read_text())['papers']
                 if p['journal'] == 'Eng—Advances in Engineering'}
        registry = {r['source_id']: set(json.loads(r['issn'] or '[]'))
                    for r in read_csv(ROOT / 'fields/journal_registry.csv')}
        sources, reviewed = {}, []
        for doi, paper in scope.items():
            path = TARGET / 'results' / (hashlib.sha256(doi.encode()).hexdigest() + '.json')
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            if result['status'] != 'identity_or_type_unconfirmed':
                continue
            receipt = result['receipt']
            raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
            tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
            nodes = tree.findall('o:GetRecord/o:record', NS)
            proof = (evidence(extract(nodes[0]), paper, registry[paper['source_id']], receipt,
                              hashlib.sha256(raw).hexdigest()) if len(nodes) == 1 else None)
            if proof:
                sources[doi] = [proof]
            reviewed.append({'doi': doi, 'work_id': paper['work_id'], 'receipt': receipt['raw_receipt'],
                             'acquisition_status': result['status'],
                             'review_status': 'identity_and_native_type_confirmed' if proof else 'unresolved'})
        rule = TARGET / 'eng_alias_review.json'
        if not rule.exists():
            atomic_json(rule, {'rule': 'Eng—Advances in Engineering corresponds to Eng only with exact registered ISSN 2673-4117, matching DOI and title.',
                              'official_source_receipt': 'supplement/mdpi_oai/targeted/eng_official_source.json',
                              'date_not_inferred': True, 'qualification_not_implied': True})
        atomic_json(folder / 'reviewed.json', reviewed)
        applied = apply_sources(set(sources), folder, explicit_sources=sources) if sources else {}
        applied.pop('with_explicit_deposited_date', None)
        applied['online_dates_asserted'] = 0
        atomic_json(folder / 'date_apply_status.json', applied)
        db = connect(); db.execute('PRAGMA busy_timeout=900000')
        for record in reviewed:
            for task_id, in db.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?", (record['doi'],)).fetchall():
                receipt = 'supplement/mdpi_oai/targeted/eng_cache_application/reviewed.json'
                if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',
                              (task_id, 'mdpi_eng_cached_alias', receipt)).fetchone():
                    continue
                db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                           (task_id, 'mdpi_eng_cached_alias', record['review_status'], receipt,
                            json.dumps({**record, 'final_task_closure': False, 'network_requests': 0})))
                db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (task_id,))
        db.commit(); db.close()
        result = {'reviewed_dois': len(reviewed), 'confirmed_identity_type': len(sources),
                  'unresolved': len(reviewed)-len(sources), 'application': applied,
                  'network_requests': 0, 'collector_state_unchanged': True}
        if not (folder / 'initial_application.json').exists():
            atomic_json(folder / 'initial_application.json', result)
        atomic_json(folder / 'status.json', result)
        if applied.get('new_evidence'):
            refresh()
        return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
