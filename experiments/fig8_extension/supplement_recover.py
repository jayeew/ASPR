"""Recover interrupted receipt writes without inventing missing primary evidence."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import zlib
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from lxml import etree

from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata


def quarantine(path: Path, root: Path, reason: str, events: list[dict]) -> None:
    destination = root / 'supplement/interrupted_receipts' / path.relative_to(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination = destination.with_name(destination.name + '.' + hashlib.sha256(path.read_bytes()).hexdigest()[:12])
    events.append({'path': str(path.relative_to(root)), 'reason': reason, 'preserved_as': str(destination.relative_to(root))})
    shutil.move(path, destination)
    with (root/'supplement/recovery_events.jsonl').open('a') as stream:
        stream.write(json.dumps({**events[-1],'at':datetime.now(timezone.utc).isoformat()})+'\n');stream.flush();os.fsync(stream.fileno())


def recover(root: Path = ROOT) -> dict:
    q = root / 'fields/qualification'
    events, invalid_pmcs = [], set()
    for folder in ['jats_parsed', 'supplement_parsed', 'oai_parsed']:
        for path in (q / folder).glob('*.json'):
            try:
                r = json.loads(path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                if folder == 'jats_parsed':
                    invalid_pmcs.add(path.stem)
                quarantine(path, root, type(error).__name__, events)
                continue
            if r.get('status') != 'parsed' or not r.get('raw_receipt'):
                continue
            raw_path = q / r['raw_receipt']
            try:
                data = gzip.decompress(raw_path.read_bytes())
                if r.get('raw_sha256') and hashlib.sha256(data).hexdigest() != r['raw_sha256']:
                    raise ValueError('Raw content hash differs from recorded receipt')
            except (FileNotFoundError, OSError, EOFError, ValueError,zlib.error) as error:
                if r.get('pmcid'):
                    invalid_pmcs.add(r['pmcid'])
                quarantine(path, root, 'Primary raw receipt unavailable: ' + str(error)[:200], events)
    restored = 0
    # One native marker and one generic marker can reference the same retained bytes.
    for path in (q / 'supplement_parsed').glob('*.json'):
        r = json.loads(path.read_text())
        if r.get('source') != 'pmc' or r.get('status') != 'parsed':
            continue
        native = q / 'jats_parsed' / (r['pmcid'] + '.json')
        if not native.exists():
            atomic_json(native, r)
            restored += 1
    for path in (q / 'jats_parsed').glob('*.json'):
        r = json.loads(path.read_text())
        if r.get('status') != 'parsed' or not r.get('parsed_receipt'):
            continue
        generic = q / r['parsed_receipt']
        if not generic.exists():
            atomic_json(generic, r)
            restored += 1
    # If both markers were lost but a validated raw prefix survived, regenerate parsed metadata only.
    for path in (q / 'jats_raw').glob('PMC*.front.xml.gz'):
        pmc = path.name.split('.')[0]
        marker = q / 'jats_parsed' / (pmc + '.json')
        if marker.exists():
            continue
        try:
            raw = gzip.decompress(path.read_bytes())
            parsed = jats_metadata(raw + b'</article>')
        except (OSError, EOFError, ValueError,zlib.error,etree.XMLSyntaxError) as error:
            invalid_pmcs.add(pmc)
            if path.exists():
                quarantine(path, root, 'Unrecoverable interrupted prefix: ' + str(error)[:200], events)
            continue
        r = {'doi': parsed['doi'][0] if parsed['doi'] else '', 'pmcid': pmc, 'source': 'pmc', 'status': 'parsed',
             'url': f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmc}/fullTextXML',
             'retrieved_at': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
             'retrieval_time_source': 'Preserved raw-file mtime; original client metadata interrupted',
             'raw_receipt': str(path.relative_to(q)), 'raw_sha256': hashlib.sha256(raw).hexdigest(),
             'parsed': parsed, 'recovery': 'Reparsed retained source bytes; no source XML synthesized'}
        atomic_json(marker, r)
        restored += 1
    for path in [*(root / 'fields/papers').glob('*.parquet'), *(q / 'decisions').glob('*.parquet')]:
        _ = pq.ParquetFile(path).metadata
    db = connect(root)
    integrity = db.execute('PRAGMA quick_check').fetchone()[0]
    if integrity != 'ok':
        raise ValueError('Task ledger integrity failure: ' + integrity)
    db.close()
    unresolved = [p for p in invalid_pmcs if not (q / 'jats_parsed' / (p + '.json')).exists()]
    result = {'checked_at': datetime.now(timezone.utc).isoformat(), 'interrupted_files': events,
              'restored_parsed_markers': restored, 'pmcids_requiring_source_refetch': unresolved,
              'parquet_footers_valid': True, 'task_ledger_quick_check': integrity,
              'source_policy': 'Corrupt evidence preserved in interrupted_receipts; missing originals are refetched, never fabricated'}
    atomic_json(root / 'supplement/recovery_receipt.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'interrupted_files'}), flush=True)
    return result


if __name__ == '__main__':
    recover()
