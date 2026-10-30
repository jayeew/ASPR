"""Consume the official XML list within the existing Wellcome candidate scope."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lxml import etree

from .derived_refresh import refresh
from .fig8b_qualification_decide import electronic_date, key, NONPRIMARY
from .ojs_metadata_harvest import fetch
from .publisher_delta_apply import apply_available
from .springer_author_cache import run as apply_authors
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata

FOLDER = ROOT / 'supplement/wellcome_xml'


def stage(raw: bytes, receipt: dict, candidates: list[dict]) -> dict:
    parsed = jats_metadata(raw)
    matched = [p for p in candidates if p['doi'].lower() in {d.lower() for d in parsed['doi']}
               and key(p['title']) in {key(t) for t in parsed['title']}]
    if not matched or {key(j) for j in parsed['journal_titles']} != {'wellcomeopenresearch'}:
        return {'status': 'identity_unresolved', 'receipt': receipt}
    if not matched[0]['doi'].endswith('.1'):
        return {'status': 'version_relationship_and_first_online_date_unresolved',
                'receipt': receipt, 'parsed': parsed, 'matched_work_ids': [p['work_id'] for p in matched],
                'reason': 'Noninitial version DOI: retain XML but do not count revision date as first publication or duplicate article as a new study.'}
    categories = [c for c in parsed['categories'] if key(c) not in {'article', 'articles'}]
    # Explicit journal subtype takes precedence over a generic XML research-article tag.
    kinds = categories if len(categories) == 1 else ['unreviewed_wellcome_subtype']
    if key(parsed['article_type']) in NONPRIMARY:
        kinds = list(dict.fromkeys(kinds + [parsed['article_type']]))
    online, precision = electronic_date(parsed['dates'])
    return {'status': 'parsed', 'source': 'publisher_jats', 'route': 'wellcome_xml',
            'doi': matched[0]['doi'], 'matched_work_ids': [p['work_id'] for p in matched],
            'url': receipt['url'], 'raw_receipt': '../../' + receipt['raw_receipt'],
            'raw_sha256': hashlib.sha256(raw).hexdigest(),
            'parsed': {**parsed, 'issns': parsed['issn'], 'article_types': kinds,
                       'online_date': online, 'date_precision': precision}}


def acquire(job: tuple[str, list[dict]]) -> dict:
    url, candidates = job
    token = hashlib.sha256(url.encode()).hexdigest()
    path = FOLDER / 'results' / (token + '.json')
    if path.exists():
        return json.loads(path.read_text())
    receipt = fetch(url, {})
    result = {'status': 'source_unavailable', 'receipt': receipt}
    if receipt.get('http_status') == 200:
        try:
            result = stage(gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes()), receipt, candidates)
        except (ValueError, OSError, etree.XMLSyntaxError) as error:
            result = {'status': 'parse_unresolved', 'error': str(error), 'receipt': receipt}
    if result['status'] == 'parsed':
        atomic_json(FOLDER / 'staged' / path.name, result)
    result.update(candidate_dois=sorted({p['doi'] for p in candidates}),
                  fetch_receipt=receipt, staged_receipt=str((FOLDER / 'staged' / path.name).relative_to(ROOT)))
    atomic_json(path, result)
    return result



def apply_current_authors(selected: list[tuple[str, str]]) -> dict:
    db = connect()
    jobs = [(wid, shard, receipt) for doi, receipt in selected
            for wid, shard in db.execute("SELECT work_id,shard FROM papers WHERE doi=? AND status='included_verified_research'", (doi,))]
    db.close()
    result = apply_authors(selected_jobs=jobs, workers=8, fill_abstract=True) if jobs else {}
    if result.get('papers_changed'):
        from .official_rule_reuse import run as reuse_entities
        reuse_entities()
    return result


def run() -> None:
    for name in ['results', 'staged']:
        (FOLDER / name).mkdir(exist_ok=True)
    with (FOLDER / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        papers = json.loads((FOLDER / 'scope.json').read_text())['papers']
        jobs: dict[str, list[dict]] = {}
        unmatched = []
        for paper in papers:
            version = re.search(r'\.(\d+)$', paper['doi'])
            urls = [u for u in paper['xml_urls'] if version and re.fullmatch(
                r'https://wellcomeopenresearch\.org/articles/\d+-\d+/v' + version[1] + '/xml', u)]
            if not urls:
                unmatched.append(paper['work_id'])
            for url in urls:
                jobs.setdefault(url, []).append(paper)
        atomic_json(FOLDER / 'dispatch_scope.json', {'urls': list(jobs), 'unmatched_version_work_ids': unmatched})
        outcomes = []; stop = ''; entries = list(jobs.items())
        with ThreadPoolExecutor(max_workers=8) as pool:
            for start in range(0, len(entries), 8):
                batch = list(pool.map(acquire, entries[start:start + 8])); outcomes.extend(batch)
                if any(r['fetch_receipt'].get('http_status') in {401, 403, 429}
                       or r['fetch_receipt'].get('retry_after') for r in batch):
                    stop = 'access_or_rate_restriction'
                atomic_json(FOLDER / 'status.json', {'status': 'source_stopped' if stop else 'collecting',
                    'fixed_urls': len(entries), 'completed': len(outcomes), 'stop_reason': stop,
                    'outcomes': dict(Counter(r['status'] for r in outcomes)), 'final_task_closure': False})
                if stop:
                    break
        with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            db = connect()
            for result in outcomes:
                receipt = result['staged_receipt'].replace('/staged/', '/results/')
                for doi in result['candidate_dois']:
                    for task_id, in db.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?", (doi,)).fetchall():
                        if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?', (task_id, 'wellcome_official_xml', receipt)).fetchone():
                            continue
                        db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                            (task_id, 'wellcome_official_xml', result['status'], receipt, json.dumps({'final_task_closure': False})))
                        db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?', (task_id,))
            db.commit(); db.close()
        selected = [(r['doi'], r['staged_receipt']) for r in outcomes if r['status'] == 'parsed']
        author_jobs = []
        qualification = apply_available(supplied_jobs=selected, author_jobs_out=author_jobs) if selected else {}
        authors = apply_current_authors(selected)
        atomic_json(FOLDER / 'application.json', {'qualification': qualification, 'authors': authors,
                                                'stop_reason': stop, 'final_task_closure': False})
        state = json.loads((FOLDER / 'status.json').read_text())
        state.update(status='source_stopped_application_processed' if stop else 'fixed_xml_path_application_processed',
                     application_receipt='supplement/wellcome_xml/application.json')
        atomic_json(FOLDER / 'status.json', state)
        refresh()


if __name__ == '__main__':
    run()
