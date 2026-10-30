"""Consume checksum-verified cached inventory shards before the full download ends."""
from __future__ import annotations

import csv
import fcntl
import gzip
import hashlib
import io
import json
import re
import time
from urllib.parse import unquote

from .ojs_metadata_harvest import RAW
from .pmc_cloud_inventory import BASE, FOLDER
from .pmc_cloud_versions import STOP, run as consume
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def match_shard(payload: bytes, item: dict, wanted: set[str], receipt: str) -> list[dict]:
    if len(payload) != item['size'] or hashlib.md5(payload).hexdigest() != item['MD5checksum']:
        raise ValueError('inventory_checksum_or_length_conflict')
    matches = []
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as compressed:
        for row in csv.reader(io.TextIOWrapper(compressed, encoding='utf-8')):
            if len(row) != 4 or row[0] != 'pmc-oa-opendata':
                raise ValueError('Unexpected inventory row')
            key = unquote(row[1]); match = re.fullmatch(r'metadata/(PMC\d+)\.(\d+)\.json', key)
            if match and match.group(1) in wanted:
                matches.append({'pmcid': match.group(1), 'version': int(match.group(2)),
                                'metadata_key': key, 'last_modified': row[2], 'etag': row[3],
                                'inventory_raw_receipt': receipt})
    return matches


def run() -> None:
    with (FOLDER / 'version_dispatch.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scope = json.loads((FOLDER / 'scope.json').read_text())
        wanted = {p for paper in scope['papers'] for p in paper['pmcids']}
        manifest = json.loads((FOLDER / 'manifest.json').read_text())
        if manifest['fileSchema'] != 'Bucket, Key, LastModifiedDate, ETag':
            raise ValueError('Unexpected inventory schema')
        cache = FOLDER / 'stream_shards'; cache.mkdir(exist_ok=True)
        handled = set()
        while not STOP.is_set() and not (FOLDER / 'source_stop.json').exists():
            for item in manifest['files']:
                token = hashlib.sha256((BASE + item['key'] + '{}').encode()).hexdigest()
                if token in handled:
                    continue
                receipt_path = RAW / (token + '.json')
                if not receipt_path.exists():
                    continue
                receipt = json.loads(receipt_path.read_text())
                if receipt.get('http_status') != 200:
                    handled.add(token); continue
                parsed_path = cache / (token + '.json')
                if parsed_path.exists():
                    matches = json.loads(parsed_path.read_text())
                else:
                    payload = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
                    matches = match_shard(payload, item, wanted, receipt['raw_receipt'])
                    atomic_json(parsed_path, matches)
                atomic_json(FOLDER / 'stream_status.json', {'status': 'processing_cached_shard',
                            'matched_versions': len(matches), 'inventory_files_seen': len(handled) + 1,
                            'inventory_download_requests': 0, 'final_task_closure': False})
                consume(64, ready_versions=matches)
                handled.add(token)
                if STOP.is_set():
                    break
            with (FOLDER / 'inventory.lock').open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    live = True
                else:
                    live = False
            if not live:
                # Complete inventory is authoritative; consume() skips earlier version receipts.
                consume(64)
                return
            atomic_json(FOLDER / 'stream_status.json', {'status': 'waiting_for_additional_cached_shards',
                        'inventory_files_seen': len(handled), 'inventory_download_requests': 0,
                        'final_task_closure': False})
            time.sleep(30)


if __name__ == '__main__':
    run()
