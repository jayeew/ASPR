"""Retain explicit native image/commentary conflicts for study-design adjudication."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter

from .cached_date_apply import run as apply_sources
from .derived_refresh import refresh
from .fig8b_qualification_build import key
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata
from .three_journal_apply import read_csv

CATEGORIES = {'innovationsimages', 'imageincardiovascularmedicine', 'imagesincardiovascularmedicine',
              'eyeimages', 'image', 'imageletter', 'researchperspective', 'commentaryandperspective',
              'lettertoeditor', 'lettertotheeditor'}
TITLE = re.compile(r'^(?:letter to (?:the )?editor\b|comments? on\s*[“"\'])', re.I)


def flagged(title: str, categories: list[str]) -> list[str]:
    return [v for v in categories if key(v) in CATEGORIES] + (['Explicit commentary title'] if TITLE.search(title) else [])


def run() -> dict:
    folder = ROOT / 'supplement/native_subtype_holds'
    folder.mkdir(exist_ok=True)
    registry = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    db = connect()
    sources, reviewed, rejected = {}, [], []
    for wid, doi, title, journal, sid, payload in db.execute("""SELECT p.work_id,p.doi,p.title,p.journal,p.source_id,s.payload
            FROM papers p JOIN source_metadata s USING(doi)
            WHERE p.status='included_verified_research' AND s.kind='jats' AND p.journal<>'Nature'"""):
        cached = json.loads(payload)
        flags = flagged(title, cached.get('categories', []))
        if not flags:
            continue
        if db.execute("SELECT 1 FROM current_evidence_index WHERE work_id=? AND kind='primary' AND json_extract(payload,'$.family')='reviewed_study_design'", (wid,)).fetchone():
            reviewed.append({'work_id': wid, 'doi': doi, 'status': 'existing_study_design_review_preserved'})
            continue
        receipt = cached['evidence_receipt']
        raw = gzip.decompress((ROOT / receipt).read_bytes())
        native = jats_metadata(raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw)
        native['issns'] = native.get('issn', [])
        paper = {'title': title, 'journal_name': journal, 'source_id': sid}
        native_flags = flagged(native.get('title', [''])[0], native.get('categories', []))
        if doi not in {v.lower() for v in native.get('doi', [])} or not native_flags or not signature_match(paper, native, registry[sid]):
            rejected.append({'work_id': wid, 'doi': doi, 'receipt': receipt, 'status': 'raw_identity_or_subtype_not_confirmed'})
            continue
        sources.setdefault(doi, []).append({
            'family': 'publisher_native_subtype_review', 'source_id': sid, 'title': title,
            'journal_titles': [journal], 'kind': native.get('article_type', 'research-article'),
            'categories': native.get('categories', []), 'online_date': '', 'date_precision': '',
            'receipt': '../../' + receipt, 'raw_sha256': hashlib.sha256(raw).hexdigest(),
            'url': cached['source_url'], 'scope_review_required': True,
            'scope_note': 'Explicit native image/commentary format conflicts with generic research label; individual original-study adjudication required',
            'native_subtype_flags': native_flags})
    db.close()
    atomic_json(folder / 'sources.json', sources)
    atomic_json(folder / 'preserved_reviews.json', reviewed)
    atomic_json(folder / 'unresolved_identity.json', rejected)
    result = apply_sources(set(sources), folder, explicit_sources=sources) if sources else {'new_sources': 0}
    result.pop('with_explicit_deposited_date', None)
    result.update(network_requests=0, candidate_dois=len(sources), preserved_reviews=len(reviewed), identity_rejections=len(rejected))
    atomic_json(folder / 'status.json', result)
    if sources:
        refresh()
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    run()
