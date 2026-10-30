"""Batch known HAL identifiers from cached links; preserve repository provenance."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlencode

import requests

from .supplement_queue import ROOT
from .supplement_sources import atomic_json

FOLDER = ROOT/'supplement/hal_metadata'


def request_params(batch: list[str]) -> dict:
    return {'q': 'halId_s:(' + ' OR '.join('"' + value + '"' for value in batch) + ')',
            'wt': 'json', 'rows': 1000,
            'fl': 'halId_s,doiId_s,title_s,journalTitle_s,docType_s,producedDate_s,'
                  'publicationDate_s,fileMain_s,files_s,version_i,uri_s,en_abstract_s'}


def stable_batches(identifiers: list[str], folder: Path) -> list[list[str]]:
    """Keep original request groups so appended identifiers do not refetch caches."""
    wanted = set(identifiers)
    covered: set[str] = set()
    batches = []
    for path in sorted(folder.glob('*.json')):
        batch = json.loads(path.read_text()).get('identifiers', [])
        if batch and set(batch) <= wanted:
            batches.append(batch)
            covered.update(batch)
    remaining = sorted(wanted - covered)
    current: list[str] = []
    for identifier in remaining:
        proposed = current + [identifier]
        if current and (len(proposed) > 200 or len(urlencode(request_params(proposed))) > 6000):
            batches.append(current)
            current = []
        current.append(identifier)
    if current:
        batches.append(current)
    return batches


def scope() -> dict:
    path = FOLDER/'scope.json'
    if path.exists():
        return json.loads(path.read_text())
    secondary = ROOT/'supplement/cached_secondary_publication_links'
    with (secondary/'collector.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not (secondary/'status.json').exists():
            raise RuntimeError('Secondary link indexing must finish before fixed HAL scope')
    groups = defaultdict(list); unresolved = []
    for index in [secondary,ROOT/'supplement/cached_primary_url_leads']:
        db = sqlite3.connect(f'file:{index}/leads.sqlite?mode=ro',uri=True)
        for wid,url,receipt in db.execute("SELECT work_id,url,receipt FROM leads WHERE host='hal.science' OR host LIKE '%.hal.science' OR host LIKE '%.archives-ouvertes.fr'"):
            ids = re.findall(r'/(\w+[-_]\d{6,})(?:v\d+)?(?:/|$|[?#])',url)
            evidence = {'work_id':wid,'url':url,'receipt':receipt}
            if ids:
                groups[ids[0]].append(evidence)
            else:
                unresolved.append(evidence)
        db.close()
    result = {'identifiers':dict(groups),'unparsed_links':unresolved,'qualification_unchanged':True}
    atomic_json(path,result);return result


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    with (FOLDER/'collector.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        data = scope();ids = sorted(data['identifiers']);folder = FOLDER/'batches';folder.mkdir(exist_ok=True)
        session = requests.Session();session.trust_env = False
        completed = 0;docs = 0;stopped = False;stop_reason = None
        for batch in stable_batches(ids, folder):
            token = hashlib.sha256(json.dumps(batch).encode()).hexdigest()
            path = folder/(token+'.json');body = folder/(token+'.response.gz')
            state = json.loads(path.read_text()) if path.exists() else {'identifiers':batch,'attempts':[]}
            while state.get('http_status') != 200 and len(state['attempts']) < 3:
                if state.get('http_status') in {400,401,403,404,429}:
                    break
                attempt = {'started_at':time.time()};state['attempts'].append(attempt)
                params = request_params(batch)
                try:
                    response = session.get('https://api.hal.science/search/',params=params,timeout=(10,35))
                    state.update(http_status=response.status_code,url=response.url)
                    attempt['http_status'] = response.status_code
                    body.write_bytes(gzip.compress(response.content))
                    state['raw_receipt'] = str(body.relative_to(ROOT))
                except requests.RequestException as error:
                    state.pop('http_status',None);attempt['error'] = str(error)
                atomic_json(path,state);time.sleep(1)
            if state.get('http_status') in {401,403,429}:
                stopped = True;stop_reason = 'access_restriction_or_rate_limit';break
            if state.get('http_status') == 200:
                result = json.loads(gzip.decompress(body.read_bytes()))['response']
                docs += len(result['docs'])
                state['returned_records'] = len(result['docs']);state['num_found'] = result['numFound']
                state['truncated'] = result['numFound'] > len(result['docs']);atomic_json(path,state)
                if state['truncated']:
                    stopped = True;stop_reason = 'truncated_batch_requires_pagination';break
            completed += len(batch)
            atomic_json(FOLDER/'status.json',{'status':'batch_metadata_collecting','fixed_identifiers':len(ids),
                'processed_identifiers':completed,'returned_records':docs,'qualification_changes':0})
        atomic_json(FOLDER/'status.json',{'status':'source_stopped' if stopped else 'fixed_metadata_collected_identity_review_required',
            'fixed_identifiers':len(ids),'processed_identifiers':completed,'returned_records':docs,
            'qualification_changes':0,'final_task_closure':False,
            'stop_reason':stop_reason,
            'limitation':'Repository dates and generic ART type are not publisher original-research evidence'})


if __name__ == '__main__':
    run()
