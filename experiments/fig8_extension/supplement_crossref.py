"""Public batched DOI deposits for explicit online dates, abstracts and affiliations."""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import pyarrow.parquet as pq
import requests

from .supplement_queue import ROOT, attempt, connect

SELECT = 'DOI,title,type,published-online,publisher,ISSN,container-title,abstract,author'


def batches(dois: list[str]) -> list[list[str]]:
    result, current = [], []
    for doi in dois:
        if ',' in doi:
            continue
        proposed = current + [doi]
        params = {'filter': ','.join('doi:' + d for d in proposed), 'select': SELECT, 'rows': len(proposed)}
        if current and (len(proposed) > 60 or len(urlencode(params)) > 6500):
            result.append(current)
            current = [doi]
        else:
            current = proposed
    if current:
        result.append(current)
    return result


def run(root: Path = ROOT, all_candidates: bool = False, apply_cache: bool = True, requested_dois: set[str] | None = None) -> dict:
    """Serialize source acquisition so separately prepared scoped batches cannot overlap."""
    folder = root / 'supplement'
    folder.mkdir(exist_ok=True)
    with (folder / 'crossref_collector.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(root, all_candidates, apply_cache, requested_dois)


def _run_locked(root: Path = ROOT, all_candidates: bool = False, apply_cache: bool = True, requested_dois: set[str] | None = None) -> dict:
    # Apply retained primary and indexed secondary evidence before choosing new network work.
    if root == ROOT and apply_cache:
        for module in ['supplement_acquisition', 'supplement_apply']:
            arguments = ['index'] if module == 'supplement_acquisition' else []
            subprocess.run([sys.executable, '-m', 'experiments.fig8_extension.' + module, *arguments], check=True)
    db = connect(root)
    folder=root/'fields/crossref_raw'
    completed=set()
    # Scoped batches retain their raw receipts in the shared ledger, not duplicate cache files.
    for doi, detail in db.execute("SELECT t.entity_key,a.detail FROM attempts a JOIN tasks t USING(task_id) "
            "WHERE a.source='crossref_public_deposit_bounded_three_journals' AND a.status='evidence_ready'"):
        if json.loads(detail).get('returned_doi_match'):
            completed.add(doi)
    for path in folder.glob('*.gz'):
        with gzip.open(path,'rt') as stream:prior=json.load(stream)
        if prior.get('dois') and prior.get('status') != 'retry_deferred':
            completed.update(prior['dois'])
        selected = prior.get('params', {}).get('select')
        if prior.get('status') == 'success' and (not selected or set(SELECT.split(',')) <= set(selected.split(','))):
            completed.update(i['DOI'].lower() for i in prior.get('response', {}).get('message', {}).get('items', []) if i.get('DOI'))
    if requested_dois is not None:
        requested=set(requested_dois)
    elif all_candidates:
        requested={r[0] for r in db.execute("SELECT doi FROM papers WHERE doi<>''")}
    else:
        metadata_gaps = set()
        for path in (root / 'fields/papers').glob('*.parquet'):
            for row in pq.read_table(path, columns=['doi', 'after_abstract', 'after_every_author_institution', 'after_every_author_country', 'author_list_may_be_truncated']).to_pylist():
                if not row['after_abstract'] or not row['after_every_author_institution'] or not row['after_every_author_country'] or row['author_list_may_be_truncated']:
                    metadata_gaps.add(row['doi'])
        requested=set()
        for path in (root/'fields/qualification/decisions').glob('*.parquet'):
            for r in pq.read_table(path,columns=['doi','eligibility_status']).to_pylist():
                if r['doi'] and (r['eligibility_status'] in ['pending_primary_online_date','pending_primary_type'] or (r['eligibility_status']=='included_verified_research' and r['doi'] in metadata_gaps)):
                    requested.add(r['doi'])
    targets=sorted(requested-completed)
    scope = batches(targets)
    (root / 'supplement/crossref_scope.json').write_text(json.dumps({'target_dois': len(targets), 'batches': len(scope),'previously_completed_dois':len(completed),'current_requested_dois':len(requested),
        'skipped_filter_separator_dois': sum(',' in d for d in targets), 'fields': SELECT,
        'scope': 'Explicit fixed DOI set; previously queried deposits skipped' if requested_dois is not None else 'All current DOI candidates' if all_candidates else 'Primary online-date/type gaps plus qualified abstract, author-list, institution and country gaps',
        'authority': 'Publisher-deposited explicit published-online and text; generic journal-article is not research confirmation',
        'method_source': 'https://github.com/CrossRef/rest-api-doc#filter-names'}, indent=2))
    session = requests.Session()
    session.trust_env = False
    session.headers['User-Agent'] = 'Fig8MetadataResearch/1.0 (public deposited metadata; 1 request/second)'
    folder = root / 'fields/crossref_raw'
    folder.mkdir(exist_ok=True)
    counts = {'batches': 0, 'success': 0, 'failed': 0, 'matched_dois': 0,'new_success_batches':0,'target_dois':len(targets),'target_batches':len(scope)}
    for group in scope:
        params = {'filter': ','.join('doi:' + d for d in group), 'select': SELECT, 'rows': len(group)}
        token = hashlib.sha256(json.dumps(['supplement_doi_batch', params], sort_keys=True).encode()).hexdigest()
        path = folder / (token + '.json.gz')
        if path.exists():
            with gzip.open(path, 'rt') as stream:
                r = json.load(stream)
        else:
            r = {'url': 'https://api.crossref.org/works', 'params': params, 'dois': group,
                 'retrieved_at': datetime.now(timezone.utc).isoformat(), 'attempts': []}
            for n in range(3):
                started = time.monotonic()
                try:
                    response = session.get(r['url'], params=params, timeout=(10, 45))
                    r['http_status'] = response.status_code
                    r['attempts'].append({'http_status': response.status_code, 'seconds': time.monotonic() - started})
                    if response.ok:
                        r.update(status='success', response=response.json())
                        break
                    r['status'] = 'access_restricted' if response.status_code in [401, 403] else 'http_error'
                    if response.status_code==429:
                        r.update(status='retry_deferred',retry_after=response.headers.get('Retry-After',''));break
                    if response.status_code not in [500, 502, 503, 504]:
                        break
                    retry = response.headers.get('Retry-After', '')
                    if retry.isdigit() and int(retry) > 60:
                        r.update(status='retry_deferred', retry_after_seconds=int(retry))
                        break
                    time.sleep(max(2 ** n, int(retry) if retry.isdigit() else 0))
                except (requests.RequestException, ValueError) as error:
                    r.update(status='request_error', error=type(error).__name__)
                    r['attempts'].append({'error': type(error).__name__, 'seconds': time.monotonic() - started})
                    time.sleep(2 ** n)
                time.sleep(1.05)
            temporary = path.with_suffix('.gz.tmp')
            with gzip.open(temporary, 'wt') as stream:
                json.dump(r, stream, ensure_ascii=False)
            temporary.replace(path)
            time.sleep(1.05)
            if r.get('status')=='success':counts['new_success_batches']+=1
            found={i['DOI'].lower() for i in r.get('response',{}).get('message',{}).get('items',[])}
            for doi in group:
                ident=hashlib.sha256(('qualification:'+doi).encode()).hexdigest()[:32]
                detail={'reason':'Retrieved publisher deposit requires DOI/title/journal validation' if doi in found else 'No matching deposit or source retrieval failed; not evidence of nonresearch','batch_dois':len(group),'record_returned':doi in found}
                attempt(db,ident,'crossref_public_deposit','evidence_ready' if doi in found else 'no_match_or_retrieval_failure',str(path.relative_to(root)),detail)
        counts['batches'] += 1
        counts['success' if r.get('status') == 'success' else 'failed'] += 1
        if r.get('status') == 'success':
            counts['matched_dois'] += len({i['DOI'].lower() for i in r['response']['message']['items']} & set(group))
        (root / 'supplement/crossref_status.json').write_text(json.dumps({**counts, 'target_batches': len(scope)}))
        if counts['batches'] % 20 == 0:
            print(json.dumps(counts), flush=True)
        if r.get('http_status') in [401, 403, 429]:
            counts['source_stopped_http_status'] = r['http_status']
            counts['unattempted_batches'] = len(scope) - counts['batches']
            break
    (root/'supplement/crossref_status.json').write_text(json.dumps(counts))
    db.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-candidates', action='store_true')
    parser.add_argument('--reuse-applied-state', action='store_true',
                        help='Use the already applied current tables without repeating global cache application')
    args = parser.parse_args()
    print(json.dumps(run(all_candidates=args.all_candidates, apply_cache=not args.reuse_applied_state)))


if __name__ == '__main__':
    main()
