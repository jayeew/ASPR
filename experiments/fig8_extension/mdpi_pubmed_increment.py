"""Batch only newly identified MDPI date gaps, preserving immutable request scopes."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter

from lxml import etree

from .pubmed_date_apply import run as apply_dates
from .pubmed_date_collect import run as collect
from .pubmed_date_review import run as review
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def pmid_lead(doi: str, raw: bytes) -> list[str]:
    tree = etree.fromstring(raw, etree.XMLParser(recover=True, resolve_entities=False, no_network=True))
    if tree is None:
        return []
    meta = tree.xpath('//*[local-name()="article-meta"]')
    if len(meta) != 1:
        return []
    ids = meta[0].xpath('./*[local-name()="article-id"][@pub-id-type="doi"]/text()')
    if {v.strip().lower() for v in ids} != {doi}:
        return []
    return sorted({v.strip() for v in meta[0].xpath(
        './*[local-name()="article-id"][@pub-id-type="pmid"]/text()') if v.strip().isdigit()})


def run(records: dict) -> dict:
    folder = ROOT / 'supplement/mdpi_oai/primary_followup/pubmed_incremental'
    folder.mkdir(exist_ok=True)
    seen_dois, seen_pmids, stopped = set(), set(), []
    for path in (ROOT / 'supplement').rglob('scope.json'):
        if 'pubmed' not in str(path).lower():
            continue
        for row in json.loads(path.read_text()).get('candidates', []):
            if row.get('status') == 'ready':
                seen_dois.add(row['doi'])
                seen_pmids.update(row.get('pmids', []))
        status_path = path.parent / 'status.json'
        if status_path.exists() and json.loads(status_path.read_text()).get('status') == 'source_stopped_unattempted_retained':
            stopped.append(str(status_path.relative_to(ROOT)))
    if stopped:
        result = {'status': 'existing_source_stop_retained', 'receipts': stopped, 'network_requests': 0}
        atomic_json(folder / 'dispatch_status.json', result)
        return result
    db = connect()
    candidates, held = [], []
    for record in records.values():
        doi = record['doi']
        if record['status'] != 'parsed' or doi in seen_dois:
            continue
        if not db.execute("SELECT 1 FROM papers WHERE doi=? AND status='pending_primary_online_date'", (doi,)).fetchone():
            continue
        receipt = 'fields/qualification/' + record['raw_receipt']
        ids = pmid_lead(doi, gzip.decompress((ROOT / receipt).read_bytes()))
        if len(ids) != 1 or ids[0] in seen_pmids:
            held.append({'doi': doi, 'pmids': ids, 'receipt': receipt,
                         'reason': 'missing_ambiguous_or_previously_queried_pmid'})
            continue
        candidates.append({'doi': doi, 'pmids': ids, 'receipts': [receipt], 'status': 'ready'})
    db.close()
    pmid_counts = Counter(r['pmids'][0] for r in candidates)
    duplicate_pmids = {pmid for pmid, count in pmid_counts.items() if count > 1}
    held.extend({**r, 'reason': 'pmid_shared_by_multiple_dois'} for r in candidates if r['pmids'][0] in duplicate_pmids)
    candidates = sorted([r for r in candidates if r['pmids'][0] not in duplicate_pmids], key=lambda r: r['doi'])
    for offset in range(0, len(candidates), 100):
        rows = candidates[offset:offset + 100]
        token = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:24]
        batch = folder / token
        batch.mkdir(exist_ok=True)
        atomic_json(batch / 'scope.json', {'candidates': rows, 'eligibility_unchanged': True,
                    'reason': 'New native archive PMID after primary type confirmation; prior ready scopes excluded'})
    atomic_json(folder / 'held.json', {'records': held, 'final_task_closure': False})
    completed = 0
    for scope in sorted(folder.glob('*/scope.json')):
        batch = scope.parent
        if (batch / 'completion.json').exists():
            continue
        collect(folder=batch)
        if json.loads((batch / 'status.json').read_text())['status'] != 'collected_identity_and_date_review_required':
            return {'status': 'source_stopped_unattempted_retained', 'batch': str(batch)}
        reviewed = review(folder=batch)
        applied = apply_dates(folder=batch)
        atomic_json(batch / 'completion.json', {'review': reviewed, 'application': applied})
        completed += 1
    result = {'new_candidates': len(candidates), 'batches_processed': completed, 'held': len(held)}
    atomic_json(folder / 'dispatch_status.json', result)
    return result
