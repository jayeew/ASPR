"""Recover annual non-Portfolio rules from qualified Wiley papers' cached permissions."""
from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from collections import defaultdict

import pyarrow.dataset as ds
from lxml import etree

from .cell_press_annual_cache import apply_proposals
from .fig8b_qualification_evidence import parse_front
from .jats_annual_copyright import named_publisher
from .supplement_apply import signature_match
from .supplement_queue import ROOT
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/wiley_dated_permissions'


def run() -> dict:
    FOLDER.mkdir(exist_ok=True)
    db = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
    ids = {r[0] for r in db.execute("SELECT work_id FROM papers WHERE route='wiley' AND status='included_verified_research'")}
    dataset = ds.dataset(ROOT / 'fields/qualification/decisions', format='parquet')
    rows = dataset.to_table(columns=['work_id', 'doi', 'title', 'source_id', 'online_year', 'online_date', 'evidence_ids'],
        filter=ds.field('work_id').isin(ids) & (ds.field('eligibility_status') == 'included_verified_research') &
        (~ds.field('nature_portfolio_at_verified_online_year').isin(['True', 'False']))).to_pylist()
    journals = {r['source_id']: r for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    groups, audit = defaultdict(list), []
    for row in rows:
        attempts = []
        for payload, in db.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (row['work_id'],)):
            evidence = json.loads(payload)
            if evidence.get('family') != 'archived_jats' or evidence.get('identity') != 'matched':
                continue
            path = ROOT / 'fields/qualification' / evidence['receipt']
            raw = gzip.decompress(path.read_bytes())
            document = raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw
            tree = etree.fromstring(document, etree.XMLParser(resolve_entities=False, no_network=True))
            native = parse_front(document)
            paper = {**row, 'journal_name': journals[row['source_id']]['journal_name']}
            native['issns'] = native['issn']
            identity = row['doi'].strip().casefold() in {doi.strip().casefold() for doi in native['doi']} and signature_match(paper, native, set(json.loads(journals[row['source_id']]['issn'] or '[]')))
            named = named_publisher(tree, row['online_year']) if identity else None
            attempts.append({'receipt': str(path.relative_to(ROOT)), 'status': 'confirmed_dated_wiley_publication_statement' if named and named[0] == 'Wiley' else 'no_agreeing_dated_wiley_statement_or_identity'})
            if not named or named[0] != 'Wiley':
                continue
            proof = {**row, 'publication_date': row['online_date'], 'statement': named[1],
                     'raw_receipt': str(path.relative_to(ROOT)), 'raw_sha256': hashlib.sha256(raw).hexdigest(),
                     'source_url': evidence['url'], 'native_evidence_id': evidence['id'],
                     'locator': 'front/article-meta/permissions/copyright-statement',
                     'online_date_basis': 'Current qualified explicit online date and evidence IDs; copyright year must equal online year'}
            groups[(row['source_id'], row['online_year'])].append(proof)
            break
        audit.append({'work_id': row['work_id'], 'doi': row['doi'], 'attempts': attempts,
                      'final_task_closure': False})
    db.close()
    proposals, held = [], []
    for (sid, year), items in groups.items():
        proofs = sorted({r['doi']: r for r in items}.values(), key=lambda r: (r['online_date'], r['doi']))
        if len(proofs) < 2:
            held.append({'source_id': sid, 'year': year, 'reason': 'fewer_than_two_distinct_dated_paper_proofs'})
            continue
        first, last = proofs[0], proofs[-1]
        proposals.append({'source_id': sid, 'year': year, 'label': 'False',
            'journal': journals[sid]['journal_name'], 'publisher': 'Wiley / John Wiley & Sons',
            'source_url': first['source_url'], 'raw_receipt': first['raw_receipt'], 'raw_sha256': first['raw_sha256'],
            'first_dated_proof': first, 'last_dated_proof': last,
            'scope': 'Two distinct matched native papers explicitly published by Wiley/John Wiley & Sons; copyright year equals verified online year. Working same-journal/year rule only; no cross-year extrapolation; reopen on transfer or conflicting Portfolio evidence.'})
    atomic_json(FOLDER / 'review.json', {'qualified_unknown_scope': len(rows), 'papers': audit,
                'annual_proposals': len(proposals), 'held_pairs': held, 'network_requests': 0})
    prior_path = FOLDER / 'rules.json'
    prior = {(r['source_id'], r['year']): r for r in json.loads(prior_path.read_text())} if prior_path.exists() else {}
    prior.update({(r['source_id'], r['year']): r for r in proposals})
    return apply_proposals(list(prior.values()), FOLDER, held,
        rule_status='dated_native_wiley_publication_statement',
        reason='Two DOI/title/journal-matched native published-by declarations with copyright year agreeing with verified online year',
        route='wiley_dated_permissions', apply_pairs={(r['source_id'], r['year']) for r in proposals})


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
