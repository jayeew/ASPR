"""Resolve cached archive-link identity conflicts against retained native front matter."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone

from lxml import etree

from .primary_byline_comparison import incoming_metadata, load_primary
from .snapshot_location_recovery import FOLDER
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> None:
    scope = json.loads((FOLDER / 'archive_association_conflicts.json').read_text())['records']
    reviews = []
    for item in scope:
        result = dict(item)
        pmcid = re.search(r'PMC\d+', item['source_receipt']).group()
        outcome = ROOT / 'supplement/pmc_cloud/xml_results' / (pmcid + '.1.json')
        saved = json.loads(outcome.read_text()) if outcome.exists() else {}
        receipt = saved.get('parsed_receipt')
        result['status'] = 'native_cache_unavailable_no_new_request'
        if receipt:
            try:
                record, raw = load_primary(receipt, item['current_doi'])
                digest = hashlib.sha256(raw).hexdigest()
                result.update(native_parsed_receipt=receipt, native_raw_sha256=digest)
                if not record.get('raw_sha256') or digest != record['raw_sha256']:
                    result['status'] = 'native_integrity_unconfirmed'
                else:
                    native = incoming_metadata(record, raw)
                    dois = sorted(set(native.get('doi') or []))
                    result['native_article_dois'] = dois
                    result['status'] = ('native_doi_matches_requires_full_identity_review'
                        if item['current_doi'] in dois else 'native_article_doi_mismatch' if dois
                        else 'native_article_doi_missing')
            except (OSError, ValueError, etree.XMLSyntaxError) as error:
                result.update(status='native_read_or_parse_unresolved', error=type(error).__name__)
        reviews.append(result)
    receipt = FOLDER / 'archive_association_native_review.json'
    atomic_json(receipt, {'records': reviews, 'outcomes': dict(Counter(r['status'] for r in reviews)),
        'network_requests': 0, 'qualification_changes': 0, 'final_task_closure': False})
    db = sqlite3.connect(FOLDER / 'locations.sqlite', timeout=120)
    changed = 0
    for item in reviews:
        if item['status'] != 'native_article_doi_mismatch':
            continue
        row = db.execute("SELECT evidence FROM links WHERE work_id=? AND url=? AND status='existing_archive_association_review_required'", (item['work_id'], item['url'])).fetchone()
        if not row:
            continue
        evidence = json.loads(row[0])
        evidence['native_identity_review'] = {'receipt': str(receipt.relative_to(ROOT)),
            'status': item['status'], 'native_article_dois': item['native_article_dois'],
            'current_doi': item['current_doi'], 'final_task_closure': False}
        target = 'cached_archive_native_doi_mismatch'
        db.execute('UPDATE links SET status=?,evidence=? WHERE work_id=? AND url=?',
                   (target, json.dumps(evidence), item['work_id'], item['url']))
        db.execute('INSERT OR IGNORE INTO routing_changes VALUES(?,?,?,?,?,?)',
            (item['work_id'], item['url'], 'existing_archive_association_review_required', target,
             'Retained native article DOI differs; reject this source association only, preserve paper and other sources',
             datetime.now(timezone.utc).isoformat()))
        changed += 1
    db.commit();db.close()
    print(json.dumps({'native_reviewed_links':len(reviews), 'mismatched_links_reclassified':changed,
                     'outcomes':dict(Counter(r['status'] for r in reviews)), 'network_requests':0}),flush=True)


if __name__ == '__main__':
    run()
