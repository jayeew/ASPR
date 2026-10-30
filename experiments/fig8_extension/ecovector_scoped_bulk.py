"""Resume one verified OAI journal set and apply only the fixed existing DOI queue."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlencode

import requests
from lxml import etree

from .publisher_delta_apply import apply_available
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/ecovector_bulk'
BASE = 'https://journals.eco-vector.com/0367-6765/oai'


def fetch(url: str, page: int, session: requests.Session) -> Path:
    receipt = FOLDER / f'page_{page}.json'
    state = json.loads(receipt.read_text()) if receipt.exists() else {'url': url, 'attempts': []}
    if state.get('http_status') == 200:
        return ROOT / state['raw_receipt']
    if state.get('http_status') in {401, 403, 429}:
        raise RuntimeError('Retained source restriction; no repeat request')
    while len(state['attempts']) < 3:
        attempt = {'started_at': time.time()}
        state['attempts'].append(attempt)
        try:
            response = session.get(url, timeout=(10, 90))
            state['http_status'] = response.status_code
            attempt['http_status'] = response.status_code
            raw = FOLDER / f'page_{page}.attempt_{len(state["attempts"])}.xml.gz'
            raw.write_bytes(gzip.compress(response.content))
            state['raw_receipt'] = str(raw.relative_to(ROOT))
        except requests.RequestException as error:
            attempt['error'] = str(error)
        atomic_json(receipt, state)
        if state.get('http_status') == 200:
            return ROOT / state['raw_receipt']
        if state.get('http_status') in {401, 403, 429, 404, 410}:
            break
        time.sleep(2)
    raise RuntimeError(f'OAI page {page} unresolved; receipts retained')


def run() -> None:
    with (FOLDER / 'collector.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = connect(ROOT)
        scope_path = FOLDER / 'fixed_scope.json'
        if not scope_path.exists():
            rows = db.execute("SELECT doi FROM publisher_remaining_jobs WHERE route='springer' "
                              "AND status='pending' AND doi LIKE '10.31857/s03676765%'").fetchall()
            atomic_json(scope_path, {'dois': [r[0] for r in rows], 'set': '0367-6765',
                                    'source_record_count': 796, 'max_pages': 8})
        scope = json.loads(scope_path.read_text())
        targets = set(scope['dois'])
        journals = {r['source_id']: set(json.loads(r['issn'] or '[]'))
                    for r in read_csv(ROOT / 'fields/journal_registry.csv')}
        session = requests.Session()
        session.trust_env = False
        session.headers['User-Agent'] = 'Fig8MetadataResearch/1.0 (public metadata; paced requests)'
        raw_path = FOLDER / 'scoped_first_page_attempt2.xml.gz'
        matched = {}
        observed = []
        seen_tokens = set()
        for page in range(scope['max_pages']):
            tree = etree.fromstring(gzip.decompress(raw_path.read_bytes()))
            for article in tree.xpath('//*[local-name()="article"]'):
                dois = [v.lower() for v in article.xpath('.//*[local-name()="article-id"][@pub-id-type="doi"]/text()')]
                selected = targets.intersection(dois)
                if not selected:
                    continue
                raw = etree.tostring(article).replace(b' xmlns="http://dtd.nlm.nih.gov/publishing/2.3"', b'')
                parsed = jats_metadata(raw)
                parsed['doi'], parsed['issns'] = dois, parsed['issn']
                for doi in selected:
                    rows = db.execute('SELECT title,journal,source_id FROM papers WHERE doi=?', (doi,)).fetchall()
                    valid = any(signature_match({'title':t, 'journal_name':j}, parsed, journals[s]) for t,j,s in rows)
                    token = hashlib.sha256(doi.encode()).hexdigest()
                    article_path = FOLDER / f'{token}.article.xml.gz'
                    article_path.write_bytes(gzip.compress(raw))
                    target = FOLDER / f'{token}.receipt.json'
                    atomic_json(target, {'source':'publisher_native_xml', 'route':'ecovector_oai_nlm',
                        'doi':doi, 'parsed':parsed, 'url':BASE, 'raw_receipt':'../../' + str(article_path.relative_to(ROOT)),
                        'raw_sha256':hashlib.sha256(raw).hexdigest(), 'parent_oai_receipt':str(raw_path.relative_to(ROOT)),
                        'transform':'Extract NLM article and remove default namespace; original envelope retained'})
                    observed.append({'doi':doi, 'identity_matched':valid, 'receipt':str(target.relative_to(ROOT))})
                    if valid:
                        matched[doi] = str(target.relative_to(ROOT))
            tokens = tree.xpath('//*[local-name()="resumptionToken"]/text()')
            atomic_json(FOLDER / 'status.json', {'status':'collecting', 'pages_processed':page+1,
                        'matched_dois':len(matched), 'fixed_target_dois':len(targets)})
            if not tokens or not tokens[0].strip():
                break
            if page + 1 == scope['max_pages'] or tokens[0] in seen_tokens:
                raise RuntimeError('OAI pagination exceeds declared fixed set; review before continuing')
            seen_tokens.add(tokens[0])
            try:
                raw_path = fetch(BASE + '?' + urlencode({'verb':'ListRecords', 'resumptionToken':tokens[0]}), page+1, session)
            except RuntimeError as error:
                atomic_json(FOLDER / 'status.json', {
                    'status': 'source_stopped_cached_evidence_apply_required',
                    'stop_reason': str(error), 'failed_page': page + 1,
                    'pages_processed': page + 1, 'matched_dois': len(matched),
                    'fixed_target_dois': len(targets), 'final_task_closure': False,
                })
                db.close()
                raise
        db.close()
        atomic_json(FOLDER / 'record_review.json', {'observed':observed,
                    'not_found_in_fixed_set':sorted(targets - {r['doi'] for r in observed}),
                    'not_found_is_not_final_closure':True})
        result = apply_available(supplied_jobs=sorted(matched.items()))
        atomic_json(FOLDER / 'application.json', result)
        atomic_json(FOLDER / 'status.json', {'status':'fixed_set_applied', 'matched_dois':len(matched),
                    'fixed_target_dois':len(targets), 'qualification':result})


if __name__ == '__main__':
    run()
