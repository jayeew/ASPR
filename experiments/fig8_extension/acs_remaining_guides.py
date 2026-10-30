"""Fetch official journal-level guides for the remaining fixed ACS short-format queue."""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit

import requests
from lxml import html

from .fig8b_qualification_decide import key
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT / 'supplement/acs_rapid_formats/remaining_guides'
    folder.mkdir(exist_ok=True)
    scope_path = folder / 'scope.json'
    if not scope_path.exists():
        tree = html.fromstring(gzip.decompress((folder.parent / 'jacsat.html.gz').read_bytes()))
        codes = {key(a.text_content()): urlsplit(a.get('href')).path.rsplit('/', 1)[-1]
                 for a in tree.xpath('//a[contains(@href,"pubs.acs.org/journal/")]')}
        db = connect()
        rows = db.execute("SELECT DISTINCT p.work_id,p.doi,p.journal,e.payload FROM papers p "
            "JOIN current_evidence_index e USING(work_id) WHERE p.status='pending_primary_type' "
            "AND e.kind='primary' AND p.doi LIKE '10.1021/%' "
            "AND json_extract(e.payload,'$.kind')='rapid-communication'").fetchall()
        db.close()
        journals = {}
        for wid, doi, journal, payload in rows:
            journals.setdefault(journal, {'journal': journal, 'coden': codes.get(key(journal)), 'papers': {}})['papers'][wid] = doi
        atomic_json(scope_path, {'journals': list(journals.values()),
            'code_mapping_source': 'supplement/acs_rapid_formats/jacsat.html.gz',
            'scope': 'Existing unresolved ACS rapid-communication candidates only; no new journals or years'})
    scope = json.loads(scope_path.read_text())
    session = requests.Session(); session.trust_env = False
    outcomes = []
    for journal in sorted(scope['journals'], key=lambda j: -len(j['papers'])):
        code = journal['coden']
        if not code:
            outcomes.append({'journal': journal['journal'], 'status': 'official_code_mapping_missing'})
            continue
        path = folder / (code + '.json')
        if path.exists():
            receipt = json.loads(path.read_text())
        else:
            url = 'https://researcher-resources.acs.org/publish/author_guidelines?coden=' + code
            receipt = {'url': url, 'journal': journal['journal'], 'trust_env': False,
                       'at': datetime.now(timezone.utc).isoformat()}
            try:
                response = session.get(url, timeout=(10, 30))
                raw = response.content
                raw_path = folder / (code + '.html.gz'); raw_path.write_bytes(gzip.compress(raw))
                receipt.update(http_status=response.status_code, final_url=response.url,
                    raw_receipt=str(raw_path.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
            except requests.RequestException as error:
                receipt.update(error=type(error).__name__, detail=str(error))
            atomic_json(path, receipt)
            time.sleep(1.1)
        outcomes.append({'journal': journal['journal'], 'papers': len(journal['papers']),
                         'receipt': str(path.relative_to(ROOT)), 'http_status': receipt.get('http_status'),
                         'error': receipt.get('error'), 'status': 'guide_requires_content_review'})
        state = {'status': 'running', 'outcomes': outcomes, 'qualification_changes': 0,
                 'final_task_closure': False}
        if receipt.get('http_status') in {401, 403, 429} or receipt.get('error'):
            state['status'] = 'source_stopped_unattempted_retained'
            atomic_json(folder / 'status.json', state)
            return
        atomic_json(folder / 'status.json', state)
    atomic_json(folder / 'status.json', {'status': 'fixed_guides_cached_content_review_required',
        'outcomes': outcomes, 'qualification_changes': 0, 'final_task_closure': False})
    print(json.dumps({'guides': len(outcomes)}), flush=True)


if __name__ == '__main__':
    run()
