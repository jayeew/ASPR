"""Corroborate dated PLOS publisher records against verified electronic dates."""
from __future__ import annotations

import gzip
import hashlib
import json

import pyarrow.parquet as pq

from .cell_press_annual_cache import apply_proposals
from .fig8b_qualification_build import key, plain
from .supplement_queue import ROOT, connect
from .three_journal_apply import read_csv


def matched_record(doc: dict, paper: dict, issns: set[str]) -> bool:
    return (doc.get('id', '').lower() == paper['doi']
            and doc.get('publisher') == 'Public Library of Science'
            and key(plain(doc.get('title_display', ''))) == key(paper['title'])
            and doc.get('eissn') in issns
            and len(paper['online_date']) == 10
            and doc.get('publication_date', '')[:10] == paper['online_date']
            and paper['online_year'] in {2023, 2024, 2025}
            and paper['online_date'] in paper.get('independent_dates', []))


def run() -> dict:
    folder = ROOT / 'supplement/plos_annual_cache'
    folder.mkdir(exist_ok=True)
    db = connect()
    targets = {r[0]: dict(zip(['work_id', 'doi', 'title', 'journal', 'source_id', 'shard'], r)) for r in db.execute(
        "SELECT work_id,doi,title,journal,source_id,shard FROM papers WHERE doi LIKE '10.1371/%' AND status='included_verified_research'")}
    for paper in targets.values():
        evidence = [json.loads(r[0]) for r in db.execute(
            "SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (paper['work_id'],))]
        paper['independent_dates'] = [e['online_date'] for e in evidence
            if e.get('identity') == 'matched' and e.get('online_date') and e.get('url')
            and 'api.plos.org' not in e['url'] and e.get('receipt')]
    db.close()
    for shard in {p['shard'] for p in targets.values()}:
        for row in pq.read_table(ROOT / 'fields/qualification/decisions' / shard,
                columns=['work_id', 'online_date', 'online_year', 'eligibility_status']).to_pylist():
            if row['work_id'] in targets:
                targets[row['work_id']].update(row)
    by_doi = {}
    for paper in targets.values():
        if paper['eligibility_status'] == 'included_verified_research':
            by_doi.setdefault(paper['doi'], []).append(paper)
    registry = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    prior = folder / 'rules.json'
    rules = {(r['source_id'], r['year']): r for r in json.loads(prior.read_text())} if prior.exists() else {}
    unmatched = []
    for path in sorted((ROOT / 'supplement/plos_batch').glob('batch_[0-9][0-9][0-9][0-9].json')):
        receipt = json.loads(path.read_text())
        if receipt.get('http_status') != 200:
            continue
        raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
        if hashlib.sha256(raw).hexdigest() != receipt['sha256']:
            raise ValueError('PLOS batch source hash mismatch')
        for doc in json.loads(raw)['response']['docs']:
            doi = doc.get('id', '').lower()
            if doi not in by_doi:
                continue
            matching = [p for p in by_doi[doi] if matched_record(doc, p, registry[p['source_id']])]
            if len(matching) != 1:
                unmatched.append({'doi': doi, 'status': 'identity_or_independent_electronic_date_not_confirmed', 'receipt': str(path.relative_to(ROOT))})
                continue
            paper = matching[0]
            pair = paper['source_id'], paper['online_year']
            proof = {'doi': doi, 'work_id': paper['work_id'], 'publication_date': paper['online_date'],
                     'source_url': receipt['url'], 'raw_receipt': receipt['raw_receipt'], 'raw_sha256': receipt['sha256'],
                     'locator': 'response.docs[id=' + doi + ']', 'publisher': doc['publisher'],
                     'eissn': doc['eissn'], 'title': doc['title_display'],
                     'corroborating_decision_shard': 'fields/qualification/decisions/' + paper['shard']}
            if pair not in rules:
                rules[pair] = {'source_id': pair[0], 'year': pair[1], 'label': 'False', 'journal': paper['journal'],
                    'publisher': 'Public Library of Science', 'source_url': receipt['url'],
                    'raw_receipt': receipt['raw_receipt'], 'raw_sha256': receipt['sha256'],
                    'first_dated_proof': proof, 'last_dated_proof': proof,
                    'scope': 'Dated native publisher record with exact DOI/title/eISSN and publication day independently corroborated by verified electronic date; annual working rule, no cross-year extrapolation; reopen on transfer evidence'}
            rule = rules[pair]
            if proof['publication_date'] < rule['first_dated_proof']['publication_date']:
                rule['first_dated_proof'] = proof
            if proof['publication_date'] > rule['last_dated_proof']['publication_date']:
                rule['last_dated_proof'] = proof
    return apply_proposals(list(rules.values()), folder, unmatched,
                           rule_status='dated_native_plos_corroborated_publication',
                           reason='Native named publisher and publication date independently corroborated by qualified electronic date and exact identity',
                           route='native_plos_corroborated')


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
