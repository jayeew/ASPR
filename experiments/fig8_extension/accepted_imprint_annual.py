"""Promote already-accepted dated native imprints to scoped journal/year rules."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from functools import lru_cache

import pyarrow.dataset as ds

from .cell_press_annual_cache import apply_proposals
from .fig8b_qualification_decide import portfolio_label
from .supplement_queue import ROOT, connect
from .three_journal_apply import read_csv


@lru_cache(maxsize=128)
def source_hash(receipt: str) -> str:
    path = ROOT / 'fields/qualification' / receipt
    raw = path.read_bytes()
    if path.suffix == '.gz':
        raw = gzip.decompress(raw)
    return hashlib.sha256(raw).hexdigest()


def run() -> dict:
    folder = ROOT / 'supplement/accepted_imprint_annual'
    folder.mkdir(exist_ok=True)
    annual = {(r['source_id'], int(r['year'])): r for r in read_csv(ROOT / 'fields/journal_publisher_by_year.csv')}
    fields = ['work_id', 'doi', 'title', 'source_id', 'online_year', 'online_date', 'nature_portfolio_at_verified_online_year']
    rows = ds.dataset(ROOT / 'fields/qualification/decisions', format='parquet').to_table(
        columns=fields, filter=ds.field('eligibility_status') == 'included_verified_research').to_pylist()
    true_pairs = {(r['source_id'], r['online_year']) for r in rows if r['nature_portfolio_at_verified_online_year'] == 'True'}
    groups = defaultdict(list)
    for row in rows:
        pair = row['source_id'], row['online_year']
        if (pair in annual and annual[pair]['nature_portfolio_year'] not in {'True', 'False'}
                and pair not in true_pairs and row['nature_portfolio_at_verified_online_year'] == 'False'):
            groups[pair].append(row)
    prior = folder / 'rules.json'
    rules = {(r['source_id'], r['year']): r for r in json.loads(prior.read_text())} if prior.exists() else {}
    db = connect()
    unresolved = []
    for pair, candidates in groups.items():
        # At most the earliest and latest usable dated sources are needed per annual rule.
        proofs = []
        ordered = sorted(candidates, key=lambda r: r['online_date'])
        for sequence in [ordered, list(reversed(ordered))]:
            for row in sequence:
                primary = [json.loads(r[0]) for r in db.execute(
                    "SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (row['work_id'],))]
                label, used_ids = portfolio_label('', primary, pair[1])
                used = [e for e in primary if e['id'] in used_ids]
                native = [e for e in used if e['family'] == 'archived_jats' and e.get('publishers')]
                if label != 'False' or not native:
                    continue
                if any(not e.get('receipt') for e in used):
                    unresolved.append({'work_id': row['work_id'], 'status': 'accepted_evidence_receipt_missing'})
                    continue
                sources = [{**e, 'checked_raw_sha256': source_hash(e['receipt'])} for e in used]
                proof = {'work_id': row['work_id'], 'doi': row['doi'], 'title': row['title'],
                         'source_id': pair[0], 'year': pair[1], 'publication_date': row['online_date'],
                         'evidence': sources, 'raw_receipt': str((ROOT / 'fields/qualification' / native[0]['receipt']).relative_to(ROOT)),
                         'raw_sha256': source_hash(native[0]['receipt']), 'source_url': native[0].get('url', ''),
                         'publishers': sorted({p for e in native for p in e['publishers']})}
                proofs.append(proof)
                break
        if len(proofs) != 2:
            continue
        rules[pair] = {'source_id': pair[0], 'year': pair[1], 'label': 'False',
                       'journal': annual[pair]['journal_name'], 'publisher': ' | '.join(sorted({p for proof in proofs for p in proof['publishers']})),
                       'source_url': proofs[0]['source_url'], 'raw_receipt': proofs[0]['raw_receipt'],
                       'raw_sha256': proofs[0]['raw_sha256'], 'first_dated_proof': proofs[0], 'last_dated_proof': proofs[1],
                       'scope': 'Working journal/year rule reusing the existing dated-native-imprint decision policy; exact current journal identity and verified online year; raw source and date corroboration preserved; no cross-year extrapolation; reopen on transfer/conflicting Portfolio evidence'}
    db.close()
    return apply_proposals(list(rules.values()), folder, unresolved,
                           rule_status='existing_accepted_dated_native_imprint',
                           reason='Already-accepted dated native publisher evidence reused for the same verified journal/year; no candidate-year inference',
                           route='accepted_native_imprint', apply_pairs=set(groups) & set(rules))


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
