"""Apply explicit v8 identities; retain version aliases for separate duplicate adjudication."""
from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from collections import Counter

from .cached_date_apply import run as apply_sources
from .derived_refresh import refresh
from .elife_batch import FOLDER
from .elife_evidence import evidence
from .elife_version_details import OUT
from .elife_text import plain
from .fig8b_qualification_build import key
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> dict:
    db = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
    sources, reviews = {}, []
    stats: Counter = Counter()
    for identifier in json.loads((OUT / 'scope.json').read_text())['ids']:
        path = OUT / (identifier + '.json')
        if not path.exists():
            continue
        receipt = json.loads(path.read_text())
        if receipt.get('http_status') != 200 or 'version=8' not in receipt.get('content_type', '').replace(' ', ''):
            stats['unsuccessful_or_unconfirmed_representation'] += 1
            continue
        raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
        item = json.loads(raw)
        canonical, version = item.get('doi', '').lower(), item.get('doiVersion', '').lower()
        if not canonical or not version or item.get('status') != 'vor':
            stats['version_identity_not_explicit'] += 1
            continue
        candidates = [dict(zip(['work_id', 'doi', 'title', 'journal', 'source_id'], row))
                      for row in db.execute('SELECT work_id,doi,title,journal,source_id FROM papers WHERE doi IN (?,?)', (canonical, version))]
        matched = [p for p in candidates if key(p['title']) == key(plain(item['title'])) and key(p['journal']) == 'elife']
        retained = [p for p in matched if p['doi'] == canonical]
        if len(retained) > 1 or (not retained and len(matched) != 1):
            stats['ambiguous_or_missing_candidate_identity'] += 1
            continue
        chosen = retained[0] if retained else matched[0]
        for candidate in matched:
            if candidate['work_id'] != chosen['work_id']:
                reviews.append({'work_id': candidate['work_id'], 'retained_work_id': chosen['work_id'],
                                'canonical_doi': canonical, 'version_doi': version,
                                'raw_receipt': receipt['raw_receipt'], 'status': 'explicit_alias_pending_duplicate_apply'})
        record = {'native': item, 'candidate': chosen, 'raw_receipt': receipt['raw_receipt'],
                  'verified_v8': True, 'raw_sha256': hashlib.sha256(raw).hexdigest()}
        parsed = evidence(record)
        if parsed:
            sources.setdefault(chosen['doi'], []).append(parsed)
            stats['matched_retained_papers'] += 1
    db.close()
    folder = OUT / 'application'
    folder.mkdir(exist_ok=True)
    atomic_json(folder / 'review.json', {'sources': sources, 'aliases': reviews, 'summary': dict(stats)})
    result = apply_sources(set(sources), folder, explicit_sources=sources) if sources else {}
    refresh()
    return result


if __name__ == '__main__':
    run()
