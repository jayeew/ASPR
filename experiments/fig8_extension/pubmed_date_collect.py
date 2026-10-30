"""Fetch a fixed remaining date cohort in PMID batches; never change eligibility."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from lxml import etree

from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json


def run(remaining: bool = False, folder: Path | None = None) -> None:
    folder = folder if folder is not None else ROOT / 'supplement' / ('pubmed_remaining_dates' if remaining else 'pubmed_date_followup')
    folder.mkdir(exist_ok=True)
    scope_path = folder / 'scope.json'
    if not scope_path.exists():
        original = set(json.loads((ROOT / 'supplement/cached_journal_date_followup/scope.json').read_text())['dois'])
        db = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
        candidates = sorted({d for d, in db.execute("SELECT doi FROM papers WHERE status='pending_primary_online_date'") if remaining or d in original})
        if remaining:
            processed = json.loads((ROOT / 'supplement/pubmed_date_followup/scope.json').read_text())['candidates']
            candidates = sorted(set(candidates) - {r['doi'] for r in processed})
        rows = []
        for doi in candidates:
            ids = set()
            receipts = []
            for payload, in db.execute("SELECT payload FROM source_metadata WHERE doi=? AND kind='jats'", (doi,)):
                value = json.loads(payload)
                path = ROOT / value['evidence_receipt']
                tree = etree.fromstring(gzip.decompress(path.read_bytes()), etree.XMLParser(recover=True, resolve_entities=False, no_network=True))
                native_dois = tree.xpath('//*[local-name()="article-id"][@pub-id-type="doi"]/text()')
                if doi not in [d.strip().lower() for d in native_dois]:
                    continue
                ids.update(p.strip() for p in tree.xpath('//*[local-name()="article-id"][@pub-id-type="pmid"]/text()') if p.strip().isdigit())
                receipts.append(value['evidence_receipt'])
            rows.append({'doi': doi, 'pmids': sorted(ids), 'receipts': receipts,
                         'status': 'ready' if len(ids) == 1 else 'missing_or_conflicting_pmid'})
        db.close()
        atomic_json(scope_path, {'candidates': rows, 'eligibility_unchanged': True,
            'date_semantics_source': 'https://www.nlm.nih.gov/bsd/licensee/elements_article_source.html',
            'rule': 'Only ArticleDate DateType=Electronic is a date lead; exact DOI/title/journal and existing primary type require review before application. Generic PubDate and history/indexing dates are not substitutes.'})
    rows = json.loads(scope_path.read_text())['candidates']
    ids = sorted({r['pmids'][0] for r in rows if r['status'] == 'ready'})
    session = requests.Session()
    session.trust_env = False
    session.headers['User-Agent'] = 'Fig8MetadataResearch/1.0 (fixed PMID metadata batches)'
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
    status = {'target_papers': len(rows), 'target_pmids': len(ids), 'batches_completed': 0}
    for offset in range(0, len(ids), 100):
        params = {'db': 'pubmed', 'id': ','.join(ids[offset:offset + 100]), 'retmode': 'xml'}
        token = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        receipt_path = folder / (token + '.json')
        if receipt_path.exists():
            record = json.loads(receipt_path.read_text())
        else:
            record = {'url': url, 'params': params, 'retrieved_at': datetime.now(timezone.utc).isoformat(), 'attempts': []}
            for n in range(3):
                try:
                    response = session.get(url, params=params, timeout=(10, 45), allow_redirects=False)
                    record['attempts'].append({'http_status': response.status_code})
                    raw = folder / (token + '.xml.gz')
                    atomic_bytes(raw, gzip.compress(response.content, mtime=0))
                    record.update(http_status=response.status_code, raw_receipt=str(raw.relative_to(ROOT)),
                                  sha256=hashlib.sha256(response.content).hexdigest(), retry_after=response.headers.get('Retry-After', ''))
                    if response.status_code not in {500, 502, 503, 504} or record['retry_after']:
                        break
                except requests.RequestException as error:
                    record['attempts'].append({'error': type(error).__name__})
                if n < 2:
                    time.sleep(2 ** n)
            atomic_json(receipt_path, record)
            time.sleep(1.05)
        if record.get('http_status') != 200:
            atomic_json(folder / 'status.json', {**status, 'status': 'source_stopped_unattempted_retained', 'receipt': str(receipt_path.relative_to(ROOT))})
            return
        status['batches_completed'] += 1
        atomic_json(folder / 'status.json', {**status, 'status': 'running'})
    atomic_json(folder / 'status.json', {**status, 'status': 'collected_identity_and_date_review_required'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--remaining', action='store_true')
    run(parser.parse_args().remaining)
