"""Reuse identity-matched archived MED abstracts for explicit review exclusions."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from lxml import html

from .cached_date_apply import run as apply_sources
from .derived_refresh import refresh
from .fig8b_qualification_decide import cached_author_review, key
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/cached_author_review_scope'


def plain(value: str) -> str:
    return ' '.join(html.fromstring('<div>' + value + '</div>').text_content().split())


def prepare() -> dict[str, list[dict]]:
    records = json.loads((FOLDER / 'inventory_records.json').read_text())['records']
    db = connect()
    sources: dict[str, list[dict]] = {}
    outcomes: list[dict] = []
    previous_path = ''
    items: list[dict] = []
    digest = ''
    for row in sorted(records, key=lambda r: r['raw_receipt']):
        path = row['raw_receipt']
        if path != previous_path:
            raw = gzip.decompress((ROOT / path).read_bytes())
            payload = json.loads(raw)
            items = payload.get('response', payload).get('resultList', {}).get('result', [])
            digest = hashlib.sha256(raw).hexdigest()
            previous_path = path
        if digest != row['raw_sha256']:
            raise ValueError('Archived response changed: ' + path)
        matches = [r for r in items if (r.get('doi') or '').casefold() == row['doi']
                   and r.get('source') == 'MED' and str(r.get('id')) == str(row['pmid'])
                   and key(plain(r.get('title', ''))) == key(row['title'])]
        if len(matches) != 1 or plain(matches[0].get('abstractText') or '') != row['abstract']:
            raise ValueError('Archived record identity/content changed: ' + row['doi'])
        item = matches[0]
        journal = (item.get('journalInfo') or {}).get('journal') or {}
        issns = sorted({journal.get('issn'), journal.get('essn')} - {None, ''})
        types = (item.get('pubTypeList') or {}).get('pubType', [])
        if issns != row['journal_issns'] or types != row['indexed_types']:
            raise ValueError('Archived journal/types changed: ' + row['doi'])
        secondary: list[dict] = []
        primary: list[dict] = []
        for kind, payload in db.execute('SELECT kind,payload FROM current_evidence_index WHERE work_id=?', (row['work_id'],)):
            if kind == 'secondary':
                secondary = json.loads(payload)
            else:
                primary.append(json.loads(payload))
        receipt = str(Path('../..') / path)
        backing = [s for s in secondary if s.get('receipt') and 'Review' in s.get('types', [])
                   and (ROOT / 'fields/qualification' / s['receipt']).resolve() == (ROOT / path).resolve()]
        source = {'family': 'cached_author_review_scope', 'doi': row['doi'], 'title': row['title'],
                  'issns': issns, 'journal_titles': [row['journal']], 'kind': 'archived-author-review',
                  'categories': [], 'online_date': '', 'date_precision': 'not_a_date_source',
                  'review_rule': 'med_author_review_no_primary_1', 'source': 'MED', 'pmid': row['pmid'],
                  'indexed_types': types, 'review_abstract': row['abstract'],
                  'abstract_sha256': row['abstract_sha256'], 'raw_sha256': digest,
                  'secondary_receipt': backing[0]['receipt'] if backing else '',
                  'content_locator': row['evidence_locator'] + '.abstractText + .pubTypeList',
                  'receipt': receipt, 'url': 'https://europepmc.org/article/MED/' + row['pmid'],
                  'review_rule_receipt': 'supplement/cached_author_review_scope/rule.json',
                  'provenance_limit': 'MED index and archived author abstract; not a verified publisher article-type or online date'}
        native = [p for p in primary if p.get('family') != 'cached_author_review_scope']
        accepted = not native and cached_author_review({**source, 'identity': 'matched'}, secondary)
        status = 'accepted_archived_review_scope' if accepted else 'retained_pending_specific_scope_or_source_conflict'
        outcomes.append({'work_id': row['work_id'], 'doi': row['doi'], 'shard': row['shard'],
                         'status': status, 'raw_receipt': path})
        if accepted:
            sources[row['doi']] = [source]
    db.close()
    atomic_json(FOLDER / 'scope.json', {'records': [r for r in outcomes if r['status'].startswith('accepted')]})
    atomic_json(FOLDER / 'outcomes.json', {'records': outcomes})
    atomic_json(FOLDER / 'preparation.json', {'counts': dict(Counter(r['status'] for r in outcomes)),
                                           'network_requests': 0, 'date_fields_promoted': 0})
    return sources


def run() -> dict:
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sources = prepare()
        atomic_json(FOLDER / 'sources.json', sources)
    result = apply_sources(set(sources), FOLDER, explicit_sources=sources)
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        shutil.copy2(Path(__file__).with_name('fig8b_qualification_decide.py'), ROOT / 'fields/qualification/decide.py')
    refresh()
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
