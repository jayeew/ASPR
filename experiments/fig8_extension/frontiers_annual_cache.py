"""Reuse exact Frontiers native journal identities and explicit publication histories."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from urllib.parse import urlparse

import pyarrow.dataset as ds

from .cell_press_annual_cache import apply_proposals
from .fig8b_qualification_build import key
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, parse_publisher
from .three_journal_apply import read_csv


def matches(parsed: dict, paper: dict, journal: dict) -> bool:
    return (paper['doi'] in parsed['doi']
            and key(paper['title']) in {key(v) for v in parsed['title']}
            and key(journal['journal_name']) in {key(v) for v in parsed['journal_titles']}
            and bool(set(json.loads(journal['issn'] or '[]')) & set(parsed['issns']))
            and parsed['publisher'] == ['Frontiers']
            and parsed['online_date'] == paper['online_date']
            and parsed['date_field_review'].get('explicit_article_history', {}).get('Published') == [paper['online_date']])


def run() -> dict:
    folder = ROOT / 'supplement/frontiers_annual_cache'
    folder.mkdir(exist_ok=True)
    registry = {r['source_id']: r for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    rows = ds.dataset(ROOT / 'fields/qualification/decisions', format='parquet').to_table(
        columns=['work_id', 'doi', 'title', 'source_id', 'online_date', 'online_year', 'nature_portfolio_at_verified_online_year'],
        filter=ds.field('eligibility_status') == 'included_verified_research').to_pylist()
    groups = defaultdict(list)
    for row in rows:
        if row['doi'].startswith('10.3389/') and row['nature_portfolio_at_verified_online_year'] not in {'True', 'False'}:
            groups[(row['source_id'], row['online_year'])].append(row)
    path = folder / 'rules.json'
    rules = {(r['source_id'], r['year']): r for r in json.loads(path.read_text())} if path.exists() else {}
    db = connect()
    unresolved = []
    for (sid, year), candidates in groups.items():
        proofs = []
        ordered = sorted(candidates, key=lambda r: r['online_date'])
        for sequence in [ordered, list(reversed(ordered))]:
            for row in sequence:
                found = False
                for payload, in db.execute("SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (row['work_id'],)):
                    e = json.loads(payload)
                    if (e.get('family') != 'publisher_frontiers' or e.get('identity') != 'matched'
                            or urlparse(e.get('url', '')).hostname != 'www.frontiersin.org'):
                        continue
                    raw_path = ROOT / 'fields/qualification' / e['receipt']
                    raw = gzip.decompress(raw_path.read_bytes())
                    parsed = parse_publisher(raw)
                    if not matches(parsed, row, registry[sid]):
                        unresolved.append({'work_id': row['work_id'], 'receipt': e['receipt'], 'status': 'native_identity_publisher_or_publication_history_mismatch'})
                        continue
                    proofs.append({**row, 'journal': registry[sid]['journal_name'], 'registry_issns': json.loads(registry[sid]['issn']),
                                   'raw_receipt': str(raw_path.relative_to(ROOT)), 'raw_sha256': hashlib.sha256(raw).hexdigest(),
                                   'source_url': e['url'], 'date_field_review': parsed['date_field_review']})
                    found = True
                    break
                if found:
                    break
        if len(proofs) != 2 or year not in {2023, 2024, 2025}:
            continue
        rules[(sid, year)] = {'source_id': sid, 'year': year, 'label': 'False', 'journal': registry[sid]['journal_name'],
                             'publisher': 'Frontiers', 'source_url': proofs[0]['source_url'], 'raw_receipt': proofs[0]['raw_receipt'],
                             'raw_sha256': proofs[0]['raw_sha256'], 'proofs': proofs,
                             'scope': 'Working same-journal/year native Frontiers imprint rule; exact DOI/title/ISSN and explicit Published history; no cross-year extrapolation; reopen on transfer or conflicting Portfolio evidence'}
    db.close()
    # Independently reparse the retained proof records before publishing rules.
    for rule in rules.values():
        for proof in rule['proofs']:
            raw = gzip.decompress((ROOT / proof['raw_receipt']).read_bytes())
            assert hashlib.sha256(raw).hexdigest() == proof['raw_sha256']
            assert matches(parse_publisher(raw), proof, registry[rule['source_id']])
    atomic_json(folder / 'proof_validation.json', {'rules': len(rules), 'proofs_reparsed': 2 * len(rules), 'network_requests': 0})
    return apply_proposals(list(rules.values()), folder, unresolved,
                           rule_status='native_frontiers_explicit_published_history',
                           reason='Exact native Frontiers journal imprint and explicit Published article history; Accepted metadata not used',
                           route='frontiers_native_history')


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
