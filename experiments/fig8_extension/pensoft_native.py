"""Fetch native XML only for unresolved fixed-scope MODS matches, with cached receipts."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from lxml import etree

from . import ojs_metadata_harvest as transport
from .fig8b_qualification_decide import electronic_date, imprint_dates, NONPRIMARY, SPECIAL, key
from .pensoft_bulk import FOLDER as BULK
from .publisher_delta_apply import apply_available
from .springer_author_cache import run as apply_authors
from .batch_memory import release_batch_memory
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata

FOLDER = BULK / 'native'


def fetch(job: dict) -> dict:
    doi = job['doi']
    token = hashlib.sha256(doi.encode()).hexdigest()
    target = FOLDER / 'results' / (token + '.json')
    if target.exists():
        return json.loads(target.read_text())
    urls = job['parsed']['landing_urls']
    valid = [u for u in urls if urlparse(u).hostname and urlparse(u).hostname.endswith('.pensoft.net')
             and urlparse(u).path.rstrip('/').split('/')[-1].isdigit()]
    result = {'doi': doi, 'status': 'no_unique_validated_native_route', 'mods_receipt': job['_receipt']}
    if len(valid) == 1 or job.get('validated_xml_url'):
        url = job.get('validated_xml_url') or valid[0].rstrip('/') + '/download/xml/'
        receipt = transport.fetch(url, {})
        result.update(receipt=receipt, url=url, status='native_source_unavailable')
        if receipt.get('http_status') == 200:
            try:
                parsed = jats_metadata(gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes()))
                if [d.lower().strip() for d in parsed['doi']] == [doi]:
                    online, precision = electronic_date(parsed['dates'])
                    tree = etree.fromstring(gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes()), etree.XMLParser(resolve_entities=False, no_network=True))
                    headings = tree.xpath('./front/article-meta/article-categories/subj-group[@subj-group-type="heading"]/subject/text()')
                    allowed = NONPRIMARY | SPECIAL | {'researcharticle', 'originalresearch', 'originalresearcharticle', 'conferenceabstract'}
                    if not headings or any(key(h) not in allowed for h in headings):
                        headings.append('pensoft_unreviewed_heading')
                    parsed.update(title=[t.lstrip('\ufeff') for t in parsed['title']],
                                  issns=parsed['issn'], article_types=[parsed['article_type'], *headings], publisher_headings=headings,
                                  online_date=online, date_precision=precision,
                                  publication_dates=imprint_dates(parsed['dates']))
                    record = {'doi': doi, 'source': 'publisher_jats', 'route': 'pensoft', 'url': url,
                              'parsed': parsed, 'raw_receipt': os.path.relpath(ROOT / receipt['raw_receipt'], ROOT / 'fields/qualification'),
                              'acquisition_receipt': str(target.relative_to(ROOT))}
                    staged = FOLDER / 'staged' / target.name
                    atomic_json(staged, record)
                    result.update(status='native_identity_evidence_requires_application', staged_receipt=str(staged.relative_to(ROOT)))
                else:
                    result['status'] = 'native_doi_mismatch'
            except (etree.XMLSyntaxError, ValueError) as exc:
                result.update(status='native_parse_failed', error=str(exc)[:300])
    atomic_json(target, result)
    return result


def run() -> None:
    for directory in [FOLDER, FOLDER / 'results', FOLDER / 'staged']:
        directory.mkdir(exist_ok=True)
    with (FOLDER / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = FOLDER / 'status.json'
        if status_path.exists() and json.loads(status_path.read_text()).get('status') == 'source_restricted_unattempted_retained':
            return
        transport.FOLDER, transport.RAW = BULK, BULK / 'raw'
        marker = FOLDER / 'processed.json'
        done = json.loads(marker.read_text()) if marker.exists() else {}
        while True:
            jobs = []
            db = connect()
            for path in (BULK / 'matched').glob('*.json'):
                item = json.loads(path.read_text())
                doi = item['doi']
                if doi in done:
                    continue
                missing = db.execute("SELECT 1 FROM papers WHERE doi=? AND (status LIKE 'pending%' OR (status='included_verified_research' AND (abstract_gap OR institution_gap OR country_gap OR author_cap))) LIMIT 1", (doi,)).fetchone()
                if not missing:
                    done[doi] = 'current_status_or_observed_fields_require_no_fetch'
                    continue
                jobs.append({**item, '_receipt': str(path.relative_to(ROOT))})
            db.close()
            batch = jobs[:100]
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = []
                # Only four outstanding requests; stop new dispatch on source restriction.
                for start in range(0, len(batch), 4):
                    group = list(pool.map(fetch, batch[start:start+4]))
                    results.extend(group)
                    if any(r.get('receipt', {}).get('http_status') in {401,403,429} for r in group):
                        break
            supplied = [(r['doi'], r['staged_receipt']) for r in results if r.get('staged_receipt')]
            application = apply_available(supplied_jobs=supplied) if supplied else {}
            db = connect()
            author_jobs = [(wid, shard, receipt) for doi, receipt in supplied
                           for wid, shard in db.execute("SELECT work_id,shard FROM papers WHERE doi=? AND status='included_verified_research'", (doi,))]
            db.close()
            authors = apply_authors(selected_jobs=author_jobs, workers=4, fill_abstract=True) if author_jobs else {}
            for result in results:
                done[result['doi']] = result['status']
            atomic_json(marker, done)
            blocked = any(r.get('receipt', {}).get('http_status') in {401,403,429} for r in results)
            terminal = json.loads((BULK / 'status.json').read_text())['status'] != 'running'
            status = 'source_restricted_unattempted_retained' if blocked else ('available_native_queue_processed_unknowns_retained' if terminal and not jobs else 'running')
            atomic_json(FOLDER / 'status.json', {'status': status, 'processed': len(done), 'application': application, 'authors': authors})
            if blocked or (terminal and not jobs):
                return
            release_batch_memory()
            if not jobs:
                time.sleep(30)


if __name__ == '__main__':
    run()
