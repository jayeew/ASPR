"""Retrieve fixed-scope publisher deposit XML, respecting shared unauthenticated quota."""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import time
from collections import Counter

import requests

from .supplement_queue import ROOT
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/joss_native'


def fetch(client: requests.Session, row: dict, item: dict) -> tuple[dict, bytes]:
    result = {'doi': row['doi'], 'work_id': row['work_id'], 'url': item['url'],
              'git_blob_sha': item['sha'], 'network_route': 'direct_trust_env_false', 'attempts': []}
    data = b''
    for number in range(1, 4):
        try:
            response = client.get(item['url'], timeout=(10, 25))
            result['attempts'].append({'number': number, 'status': response.status_code})
            result.update(status=response.status_code, remaining=response.headers.get('X-RateLimit-Remaining'),
                          reset=response.headers.get('X-RateLimit-Reset'), retry_after=response.headers.get('Retry-After'))
            if response.status_code == 200:
                data = base64.b64decode(response.json()['content'], validate=False)
                return result, data
            if response.status_code < 500:
                break
        except requests.RequestException as exc:
            result['attempts'].append({'number': number, 'error': type(exc).__name__})
            result['status'] = 'transport_error'
        if number < 3:
            time.sleep(number * 2)
    return result, data


def run() -> None:
    for name in ['api_collection_status.json', 'deposit_collection_status.json']:
        path = FOLDER / name
        if path.exists():
            state = json.loads(path.read_text())
            if int(state.get('last_rate_remaining') or 999) <= 5 and time.time() < int(state.get('reset') or 0):
                print('Documented rate-reset wait; no request made.', flush=True)
                return
    folder = FOLDER / 'publisher_deposits'
    folder.mkdir(exist_ok=True)
    scope = json.loads((FOLDER / 'source_index.json').read_text())['records']
    commit = json.loads(gzip.decompress((FOLDER / 'tree.json.gz').read_bytes()))['sha']
    client = requests.Session()
    client.trust_env = False
    results, requests_n, last = [], 0, {}
    try:
        for row in scope:
            key = row['doi'].split('/')[-1]
            receipt = folder / (key + '.json')
            if receipt.exists():
                results.append(json.loads(receipt.read_text()))
                continue
            matches = row['publisher_crossref']
            if len(matches) != 1:
                results.append({'doi': row['doi'], 'status': 'ambiguous_source_index'})
                continue
            item = matches[0]
            if key == 'joss.04693':
                result = {**json.loads((FOLDER / 'crossref_control_receipt.json').read_text()),
                          'doi': row['doi'], 'work_id': row['work_id'], 'reused_control': True,
                          'git_blob_sha': item['sha']}
                data = gzip.decompress((FOLDER / 'control.crossref.xml.gz').read_bytes())
            else:
                result, data = fetch(client, row, item)
                requests_n += len(result['attempts'])
                last = result
            result['commit'] = commit
            if result['status'] == 200:
                actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
                result.update(sha256=hashlib.sha256(data).hexdigest(), git_blob_verified=actual == item['sha'])
                if actual != item['sha']:
                    raise ValueError('Publisher blob content hash mismatch')
                target = folder / (key + '.xml.gz')
                target.write_bytes(gzip.compress(data))
                result.update(bytes=len(data), raw_receipt=str(target.relative_to(ROOT)))
            atomic_json(receipt, result)
            results.append(result)
            if result['status'] != 200 or (not result.get('reused_control') and int(result.get('remaining') or 0) <= 5):
                break
            time.sleep(.5)
    finally:
        client.close()
    state = {'status': 'source_batch_finished', 'scope': len(scope), 'processed': len(results),
             'new_requests': requests_n, 'outcomes': dict(Counter(str(r['status']) for r in results)),
             'unrequested': len(scope) - len(results), 'last_rate_remaining': last.get('remaining'),
             'reset': last.get('reset'), 'qualification_changes': 0, 'final_task_closure': False}
    atomic_json(FOLDER / 'deposit_collection_status.json', state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    run()
