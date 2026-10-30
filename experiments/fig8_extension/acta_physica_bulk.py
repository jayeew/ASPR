"""Reuse the official journal's public issue endpoint for its current fixed roster."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter

import requests

from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, paced

FOLDER = ROOT / 'supplement/acta_physica_native'
URL = 'https://wulixb.iphy.ac.cn/data/tplData'


def run(journal_name: str = 'Acta Physica Sinica', issues_per_year: int = 24) -> None:
    folder = FOLDER / 'issues'; folder.mkdir(exist_ok=True)
    db = connect()
    rows = db.execute('SELECT work_id,doi,title,source_id,journal,status FROM papers WHERE journal=?',(journal_name,)).fetchall()
    db.close()
    atomic_json(FOLDER / 'scope.json', {'records': [dict(zip(
        ['work_id', 'doi', 'title', 'source_id', 'journal', 'status'], r)) for r in rows],
        'years': [2023, 2024, 2025], 'issues_per_year': issues_per_year, 'candidate_additions': False,
        'endpoint_discovery': 'supplement/acta_physica_native/init.js:getIssueArticleList',
        'date_limitation': 'Do not promote issue publication, preprint or lastReleaseTime to explicit final online publication date'})
    session = requests.Session(); session.trust_env = False; counts = Counter(); seen = set()
    wanted = {r[1] for r in rows}; requests_n = 0
    for year in [2023, 2024, 2025]:
        for issue in range(1, issues_per_year + 1):
            path = folder / f'{year}-{issue:02d}.json'; raw_path = path.with_suffix('.json.gz')
            params = {'dataModel': 'article', 'current': '0', 'year': str(year), 'issue': str(issue), 'max': '999'}
            if path.exists():
                receipt = json.loads(path.read_text())
            elif journal_name == 'Acta Physica Sinica' and (year, issue) == (2023, 11):
                raw_path.write_bytes((FOLDER / 'bulk_probe.json.gz').read_bytes())
                receipt = {'http_status': 200, 'raw_receipt': str(raw_path.relative_to(ROOT)),
                           'url': URL, 'data': params, 'reused_probe': True}
                atomic_json(path, receipt)
            else:
                receipt = {'url': URL, 'data': params, 'attempts': []}
                for number in range(1, 4):
                    paced(URL)
                    try:
                        r = session.post(URL, data=params, timeout=(10, 30)); requests_n += 1
                        receipt['attempts'].append({'number': number, 'http_status': r.status_code})
                        receipt.update(http_status=r.status_code, retry_after=r.headers.get('Retry-After', ''))
                        if r.status_code == 200:
                            raw_path.write_bytes(gzip.compress(r.content))
                            receipt.update(raw_receipt=str(raw_path.relative_to(ROOT)), sha256=hashlib.sha256(r.content).hexdigest())
                        if r.status_code < 500:
                            break
                    except requests.RequestException as exc:
                        receipt['attempts'].append({'number': number, 'error': type(exc).__name__})
                atomic_json(path, receipt)
            if receipt.get('http_status') in {401, 403, 429}:
                atomic_json(FOLDER / 'status.json', {'status': 'source_hold', 'receipt': str(path), 'completed': sum(counts.values())})
                session.close(); return
            if receipt.get('http_status') == 200:
                data = json.loads(gzip.decompress(raw_path.read_bytes()))
                native = data.get('data') or {}; catalog = native.get('catalog') or {}
                records = (native.get('pagerFilter') or {}).get('records') or []
                valid = data.get('result') == 'success' and str(catalog.get('year')) == str(year) and str(catalog.get('issue')) == str(issue)
                if valid:
                    seen.update(a.get('doi', '').casefold() for a in records if a.get('doi', '').casefold() in wanted)
                counts['matched_issue' if valid else 'issue_identity_unconfirmed'] += 1
            else:
                counts['request_unresolved'] += 1
            atomic_json(FOLDER / 'status.json', {'status': 'complete' if sum(counts.values()) == 3 * issues_per_year else 'running',
                'completed': sum(counts.values()), 'fixed_issues': 3 * issues_per_year, 'fixed_candidates': len(wanted),
                'cached_candidate_dois': len(seen), 'outcomes': dict(counts), 'new_requests': requests_n,
                'qualification_changes': 0, 'final_task_closure': False})
            print(sum(counts.values()), f'/{3 * issues_per_year}, cached candidate DOIs', len(seen), flush=True)
    session.close()


if __name__ == '__main__':
    run()
