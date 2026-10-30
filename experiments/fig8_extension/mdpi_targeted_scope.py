"""Recover fixed MDPI OAI identifiers from existing publisher URL evidence."""
from __future__ import annotations

import gzip
import json
import re
from collections import Counter, defaultdict
from urllib.parse import urlparse

import pyarrow.parquet as pq

from .mdpi_oai_harvest import FOLDER
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def identifier(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.hostname not in {'mdpi.com', 'www.mdpi.com'}:
        return None
    found = re.fullmatch(r'/(\d{4}-\d{3}[\dX])/(\d+)/(\d+)/(\d+)(?:/pdf)?/?', parsed.path)
    return 'oai:mdpi.com:/' + '/'.join(found.groups()) + '/' if found else None


def run() -> dict:
    folder = FOLDER / 'targeted'
    folder.mkdir(exist_ok=True)
    scope_path = folder / 'scope.json'
    if scope_path.exists():
        return json.loads(folder.joinpath('scope_status.json').read_text())
    native_scope = {p['doi']: p for p in json.loads((FOLDER / 'scope.json').read_text())['papers']}
    applied = json.loads((FOLDER / 'type_application/applied_fingerprints.json').read_text())
    known = {k.split(':native:', 1)[0] for k in applied}
    db = connect()
    selected = defaultdict(dict)
    for wid, doi, shard, status in db.execute("SELECT work_id,doi,shard,status FROM papers WHERE route='mdpi'"):
        if doi in native_scope and doi not in known and status in {
            'pending_no_primary_type_date', 'pending_primary_type',
            'pending_secondary_nonprimary_or_integrity_flag', 'pending_special_subtype_review'}:
            selected[shard][wid] = doi
    db.close()
    raw_groups = defaultdict(list)
    for shard, works in selected.items():
        for p in pq.read_table(ROOT / 'fields/papers' / shard,
                               columns=['work_id', 'raw_receipt']).to_pylist():
            if p['work_id'] in works:
                raw_groups[p['raw_receipt']].append((p['work_id'], works[p['work_id']]))
    jobs, unresolved = {}, []
    for receipt, pairs in raw_groups.items():
        body = json.loads(gzip.decompress((ROOT / 'fields' / receipt).read_bytes()))
        records = {p['id']: p for p in body['response']['results']}
        for wid, doi in pairs:
            record = records.get(wid)
            if not record or (record.get('doi') or '').removeprefix('https://doi.org/').lower() != doi:
                unresolved.append({'work_id': wid, 'doi': doi, 'status': 'cached_identity_not_matched', 'receipt': receipt})
                continue
            locations = [record.get('primary_location') or {}, *(record.get('locations') or [])]
            identifiers = {found for location in locations for k in ['landing_page_url', 'pdf_url']
                           if (found := identifier(location.get(k) or ''))}
            if len(identifiers) != 1:
                unresolved.append({'work_id': wid, 'doi': doi, 'status': 'no_unique_cached_publisher_url', 'receipt': receipt})
                continue
            jobs[doi] = {**native_scope[doi], 'oai_identifier': identifiers.pop(),
                         'discovery_receipt': 'fields/' + receipt, 'discovery_work_id': wid}
    identities = Counter(j['oai_identifier'] for j in jobs.values())
    for doi, job in list(jobs.items()):
        if identities[job['oai_identifier']] > 1:
            unresolved.append({**job, 'status': 'identifier_shared_by_multiple_dois'})
            del jobs[doi]
    atomic_json(scope_path, {'scope': 'Fixed existing candidate DOI cohort; only current type gaps with exact cached publisher URL. No DOI-to-URL guessing or new candidates.',
                            'papers': sorted(jobs.values(), key=lambda p: p['doi'])})
    atomic_json(folder / 'unresolved_identifiers.json', unresolved)
    result = {'target_dois': len(jobs), 'unresolved_identifier_papers': len(unresolved),
              'known_type_dois_skipped': len(known), 'network_requests': 0}
    atomic_json(folder / 'scope_status.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
