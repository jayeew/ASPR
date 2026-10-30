"""Annual MDPI identity from exact native publisher records and corroborated online dates."""
from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pyarrow.dataset as ds
from lxml import etree

from .cell_press_annual_cache import apply_proposals
from .mdpi_cache_apply import evidence
from .mdpi_oai_harvest import NS, extract
from .supplement_queue import ROOT
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/mdpi_oai/dated_annual'
PUBLISHERS = {'Multidisciplinary Digital Publishing Institute', 'MDPI', 'MDPI AG'}


@lru_cache(maxsize=128)
def native_records(path: str) -> tuple[str, list[dict]]:
    raw = gzip.decompress(Path(path).read_bytes())
    tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
    return hashlib.sha256(raw).hexdigest(), [extract(n) for n in tree.findall('.//o:record', NS)]


def dated_publisher(item: dict, online_dates: set[str]) -> bool:
    publishers = set(item['fields']['publisher'])
    return bool(publishers) and publishers <= PUBLISHERS and bool(set(item['fields']['date']) & online_dates)


def run() -> dict:
    FOLDER.mkdir(exist_ok=True)
    scope_path = ROOT / 'supplement/mdpi_oai/scope.json'
    scope = {p['work_id'] for p in json.loads(scope_path.read_text())['papers']}
    atomic_json(FOLDER / 'application_scope.json', {'source_scope': str(scope_path.relative_to(ROOT)),
                'fixed_existing_work_ids': len(scope), 'scope_expansion': 'All existing MDPI candidates; only currently qualified unknown annual labels reviewed', 'network_requests': 0})
    columns = ['work_id', 'doi', 'title', 'source_id', 'online_year', 'online_date', 'nature_portfolio_at_verified_online_year']
    rows = ds.dataset(ROOT / 'fields/qualification/decisions', format='parquet').to_table(
        columns=columns, filter=ds.field('work_id').isin(scope) & (ds.field('eligibility_status') == 'included_verified_research')).to_pylist()
    journals = {r['source_id']: r for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    annual = {(r['source_id'], int(r['year'])): r for r in read_csv(ROOT / 'fields/journal_publisher_by_year.csv')}
    db = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    held = []
    for row in rows:
        pair = row['source_id'], row['online_year']
        if row['nature_portfolio_at_verified_online_year'] in {'True', 'False'} or annual.get(pair, {}).get('nature_portfolio_year') in {'True', 'False'}:
            continue
        primary = [json.loads(v) for v, in db.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (row['work_id'],))]
        dates = [e for e in primary if e.get('identity') == 'matched' and e.get('family') == 'crossref_publisher_deposit'
                 and e.get('online_date') == row['online_date'] and len(e.get('online_date', '')) == 10]
        found = False
        for entry in primary:
            if entry.get('identity') != 'matched' or entry.get('family') != 'publisher_mdpi_oai' or not dates:
                continue
            path = (ROOT / 'fields/qualification' / entry['receipt']).resolve()
            digest, items = native_records(str(path))
            if digest != entry['raw_sha256']:
                raise ValueError('Native evidence hash changed: ' + str(path))
            for item in items:
                if row['doi'] not in item['dois'] or not dated_publisher(item, {e['online_date'] for e in dates}):
                    continue
                paper = {**row, 'journal': journals[row['source_id']]['journal_name']}
                receipt = {'raw_receipt': str(path.relative_to(ROOT.resolve()))}
                if not evidence(item, paper, set(json.loads(journals[row['source_id']]['issn'])), receipt, digest):
                    continue
                if not all((ROOT / 'fields/qualification' / e['receipt']).exists() for e in dates):
                    raise ValueError('Missing corroborating online-date receipt')
                grouped[pair].append({**row, 'publishers': item['fields']['publisher'], 'native_dc_dates': item['fields']['date'],
                    'raw_receipt': str(path.relative_to(ROOT.resolve())), 'raw_sha256': digest,
                    'native_evidence_id': entry['id'], 'online_date_evidence': dates, 'source_url': entry['url']})
                found = True
                break
            if found:
                break
        if not found:
            held.append({'work_id': row['work_id'], 'doi': row['doi'], 'reason': 'no_exact_native_publisher_and_corresponding_explicit_online_date_pair'})
    db.close()
    proposals = []
    for (sid, year), proofs in grouped.items():
        proofs = sorted({p['doi']: p for p in proofs}.values(), key=lambda p: (p['online_date'], p['doi']))
        if len(proofs) < 2:
            held.append({'source_id': sid, 'year': year, 'reason': 'fewer_than_two_independent_dated_article_proofs'})
            continue
        first, last = proofs[0], proofs[-1]
        proposals.append({'source_id': sid, 'year': year, 'label': 'False', 'journal': journals[sid]['journal_name'],
            'publisher': 'MDPI', 'source_url': first['source_url'], 'raw_receipt': first['raw_receipt'],
            'raw_sha256': first['raw_sha256'], 'first_dated_proof': first, 'last_dated_proof': last,
            'scope': 'Same-journal/year working rule from two distinct DOI/title/ISSN-matched official MDPI OAI publisher declarations, each native date corroborated by an exact matched Crossref published-online date. DC date alone never establishes online date. No cross-year extrapolation; reopen on transfer or conflicting Portfolio evidence.'})
    atomic_json(FOLDER / 'current_unmatched_review.json', held)
    atomic_json(FOLDER / 'review_summary.json', {'qualified_scope': len(rows), 'unknown_papers_reviewed': sum(map(len, grouped.values())) + sum('work_id' in r for r in held),
                'annual_proposals': len(proposals), 'held': len(held), 'network_requests': 0})
    apply_pairs = {(r['source_id'], r['year']) for r in proposals}
    prior_path = FOLDER / 'rules.json'
    retained = {(r['source_id'], r['year']): r for r in json.loads(prior_path.read_text())} if prior_path.exists() else {}
    retained.update({(r['source_id'], r['year']): r for r in proposals})
    if not apply_pairs:
        return {'new_scope_pairs': 0, 'retained_rules': len(retained), 'network_requests': 0}
    return apply_proposals(list(retained.values()), FOLDER, held, rule_status='dated_mdpi_official_oai_publisher_with_explicit_online_date',
                           reason='Exact native MDPI publisher identity and independently corroborated online date; same journal/year only',
                           route='mdpi_dated_native_publisher', apply_pairs=apply_pairs)


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
