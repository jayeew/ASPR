"""Replay only unchanged raw JATS behind explicit Turkey country-slot gaps."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict

import pyarrow.parquet as pq
from lxml import etree

from .derived_refresh import refresh
from .springer_author_cache import run as apply_authors
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata
from .three_journal_apply import read_csv


def run() -> None:
    folder = ROOT / 'supplement/turkey_country_alias'
    output = folder / 'parsed'
    output.mkdir(exist_ok=True)
    scope = json.loads((folder / 'scope.json').read_text())
    groups = defaultdict(list)
    for row in scope['unavailable']:
        groups[row['shard']].append(row['work_id'])
    registry = {r['source_id']: r for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    jobs, audit = [], []
    for shard, ids in groups.items():
        rows = pq.read_table(ROOT / 'fields/papers' / shard,
                            columns=['work_id', 'doi', 'title', 'journal_name', 'source_id', 'authors_json'],
                            filters=[('work_id', 'in', ids)]).to_pylist()
        for paper in rows:
            sources = {a.get('publisher_affiliation_source') for a in json.loads(paper['authors_json'])
                       if not a.get('nature_v1_slots') and any(
                           s.get('text', '').strip().casefold().endswith(', turkey') and not s.get('country_codes')
                           for s in a.get('publisher_affiliation_slots') or [])}
            if not sources:
                continue
            raw_receipt = next(iter(sources)) if len(sources) == 1 else None
            entry = {'work_id': paper['work_id'], 'sources': sorted(s or '' for s in sources),
                     'status': 'no_single_compatible_jats_receipt'}
            audit.append(entry)
            if not raw_receipt or not raw_receipt.startswith('fields/qualification/') or not raw_receipt.endswith('.xml.gz'):
                continue
            raw = gzip.decompress((ROOT / raw_receipt).read_bytes())
            try:
                native = jats_metadata(raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw)
            except (ValueError, etree.XMLSyntaxError) as error:
                entry.update(status='invalid_cached_jats', error=type(error).__name__)
                continue
            native['issns'] = native.get('issn', [])
            if paper['doi'] not in {v.lower() for v in native['doi']} or not signature_match(
                    paper, native, set(json.loads(registry[paper['source_id']]['issn']))):
                entry['status'] = 'cached_identity_not_confirmed'
                continue
            token = hashlib.sha256((paper['work_id'] + raw_receipt).encode()).hexdigest()
            path = output / (token + '.json')
            record = {'doi': paper['doi'], 'source': 'pmc', 'route': 'pmc_archive_priority',
                      'parsed': {k: native[k] for k in ['doi', 'title', 'issns', 'journal_titles']},
                      'raw_receipt': str((ROOT / raw_receipt).relative_to(ROOT / 'fields/qualification')),
                      'raw_sha256': hashlib.sha256(raw).hexdigest(), 'parse_authors_from_raw': True,
                      'network_requests': 0, 'reason': 'Explicit terminal Turkey alias repair in same primary source'}
            atomic_json(path, record)
            jobs.append((paper['work_id'], shard, str(path.relative_to(ROOT))))
            entry.update(status='same_source_jats_replay_ready', receipt=str(path.relative_to(ROOT)))
    atomic_json(folder / 'other_cached_scope.json', {'jobs': jobs, 'audit': audit, 'network_requests': 0})
    result = apply_authors(selected_jobs=jobs, workers=8, reprocess_selected=True)
    atomic_json(folder / 'other_cached_status.json', result)
    refresh()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
