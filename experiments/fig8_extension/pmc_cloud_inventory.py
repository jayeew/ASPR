"""Match the official bulk cloud inventory to the fixed unresolved PMC-ID scope."""
from __future__ import annotations

import csv
import fcntl
import gzip
import hashlib
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote

from .ojs_metadata_harvest import fetch
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/pmc_cloud'
BASE = 'https://pmc-oa-opendata.s3.amazonaws.com/'


def run() -> None:
    with (FOLDER / 'inventory.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = FOLDER / 'inventory_status.json'
        if state_path.exists():
            previous = json.loads(state_path.read_text())
            if previous.get('stop_reason') not in ('', None, 'inventory_source_unavailable_no_automatic_retry') or previous.get('status') in {'fixed_inventory_matched', 'fixed_inventory_partially_available'}:
                print(json.dumps(previous)); return
        scope_path = FOLDER / 'scope.json'
        if not scope_path.exists():
            db = connect(); papers = []
            for wid, doi, ids in db.execute("SELECT work_id,doi,pmcids FROM papers WHERE pmcids<>'[]' AND (status LIKE 'pending%' OR (status='included_verified_research' AND (abstract_gap=1 OR institution_gap=1 OR country_gap=1 OR author_cap=1)))"):
                papers.append({'work_id': wid, 'doi': doi, 'pmcids': json.loads(ids)})
            db.close()
            atomic_json(scope_path, {'papers': papers, 'scope': 'Existing pending qualification or qualified metadata gaps only; no roster expansion'})
        scope = json.loads(scope_path.read_text())
        wanted = {pmc for p in scope['papers'] for pmc in p['pmcids']}
        manifest = json.loads((FOLDER / 'manifest.json').read_text())
        if manifest['fileFormat'] != 'CSV' or manifest['fileSchema'] != 'Bucket, Key, LastModifiedDate, ETag':
            raise ValueError('Unreviewed inventory schema')
        receipts = []; matches = []; scanned = 0; stop = ''; failures = []
        files = manifest['files']
        with ThreadPoolExecutor(max_workers=4) as pool:
            for offset in range(0, len(files), 4):
                batch = files[offset:offset + 4]
                results = list(pool.map(lambda item: fetch(BASE + item['key'], {}), batch))
                for item, receipt in zip(batch, results):
                    receipts.append({'inventory_file': item, 'receipt': receipt})
                    if receipt.get('http_status') != 200:
                        failures.append({'key': item['key'], 'receipt': receipt})
                        if receipt.get('http_status') in {401, 403, 429} or receipt.get('retry_after'):
                            stop = 'inventory_source_access_or_rate_stop'
                        continue
                    payload = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
                    if len(payload) != item['size'] or hashlib.md5(payload).hexdigest() != item['MD5checksum']:
                        stop = 'inventory_checksum_or_length_conflict'; continue
                    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as compressed:
                        for row in csv.reader(io.TextIOWrapper(compressed, encoding='utf-8')):
                            scanned += 1
                            if len(row) != 4 or row[0] != 'pmc-oa-opendata':
                                raise ValueError('Unexpected inventory row')
                            key = unquote(row[1])
                            match = re.fullmatch(r'metadata/(PMC\d+)\.(\d+)\.json', key)
                            if match and match.group(1) in wanted:
                                matches.append({'pmcid': match.group(1), 'version': int(match.group(2)),
                                                'metadata_key': key, 'last_modified': row[2], 'etag': row[3],
                                                'inventory_raw_receipt': receipt['raw_receipt']})
                atomic_json(FOLDER / 'inventory_receipts.json', receipts)
                atomic_json(state_path, {'status': 'reading_inventory', 'stop_reason': stop,
                            'fixed_pmids': len(wanted), 'files_processed': len(receipts),
                            'rows_scanned': scanned, 'matched_versions': len(matches)})
                if stop:
                    break
        atomic_json(FOLDER / 'matched_versions.json', matches)
        matched = {m['pmcid'] for m in matches}
        atomic_json(FOLDER / 'inventory_unmatched.json', {'pmcids': sorted(wanted - matched),
                    'interpretation': 'Not found in this dated inventory only; not proof of permanent unavailability',
                    'inventory_complete': not stop and not failures})
        result = {'status': 'source_stopped' if stop else ('fixed_inventory_partially_available' if failures else 'fixed_inventory_matched'),
                  'failed_files': failures, 'inventory_complete': not stop and not failures,
                  'stop_reason': stop, 'fixed_papers': len(scope['papers']), 'fixed_pmcids': len(wanted),
                  'files_processed': len(receipts), 'rows_scanned': scanned,
                  'matched_pmcids': len(matched), 'matched_versions': len(matches),
                  'unmatched_pmcids': len(wanted - matched), 'qualification_applied': False,
                  'next_step': 'Read matched version metadata; distinguish manuscripts, retractions and licenses before JATS integration',
                  'final_task_closure': False}
        atomic_json(state_path, result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
