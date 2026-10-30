"""Apply exact retained Analytica identities without changing live acquisition state."""
from __future__ import annotations

import gzip
import hashlib
import json

from lxml import etree

from .cached_date_apply import run as apply_sources
from .mdpi_cache_apply import evidence
from .mdpi_targeted_collect import TARGET, NS, extract
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run() -> dict:
    review_path = TARGET / 'analytica_alias_review.json'
    if review_path.exists() and json.loads(review_path.read_text()).get('application_finished'):
        return {'status': 'already_applied_no_new_evidence'}
    previous = json.loads((TARGET/'carbon_alias_review.json').read_text())
    scope = {p['doi']: p for p in json.loads((TARGET/'scope.json').read_text())['papers']}
    registry = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT/'fields/journal_registry.csv')}
    sources, reviewed = {}, []
    for row in previous['unresolved']:
        paper = scope[row['doi']]
        if paper['journal'] != 'Analytica—A Journal of Analytical Chemistry and Chemical Analysis':
            continue
        raw = gzip.decompress((ROOT/row['receipt']).read_bytes())
        tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
        item = extract(tree.find('o:GetRecord/o:record', NS))
        proof = evidence(item, paper, registry[paper['source_id']], {'raw_receipt': row['receipt']}, hashlib.sha256(raw).hexdigest())
        if proof:
            sources[paper['doi']] = [proof]
            reviewed.append({**row, 'status': 'exact_doi_title_issn_and_reviewed_journal_alias', 'native_type': proof['kind']})
    review = {'rule': 'Only Analytica versus Analytica—A Journal of Analytical Chemistry and Chemical Analysis, exact registered ISSN 2673-4532, DOI and title',
              'source': 'Retained official MDPI OAI GetRecord header and dc:source', 'reviewed': reviewed,
              'network_requests': 0, 'online_dates_asserted': 0, 'application_finished': False}
    atomic_json(review_path, review)
    (TARGET/'analytica_application').mkdir(exist_ok=True)
    result = apply_sources(set(sources), TARGET/'analytica_application', explicit_sources=sources) if sources else {}
    db = connect()
    for row in reviewed:
        for tid, in db.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?", (row['doi'],)).fetchall():
            rel = str(review_path.relative_to(ROOT))
            if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?', (tid, 'mdpi_cached_journal_alias', rel)).fetchone():
                continue
            db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                       (tid, 'mdpi_cached_journal_alias', row['status'], rel, json.dumps({**row, 'final_task_closure': False})))
            db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (tid,))
    db.commit()
    db.close()
    atomic_json(review_path, {**review, 'application_finished': True, 'application': result})
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
