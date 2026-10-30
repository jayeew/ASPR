"""Reuse existing publisher HTML and native OAI adapters for the alias-repair tail."""
from __future__ import annotations

import gzip
import hashlib
import json

from .springer_author_cache import run as apply_authors
from .supplement_apply import signature_match
from .supplement_oai import parse_oai
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, parse_publisher
from .three_journal_apply import read_csv


def run() -> None:
    folder = ROOT / 'supplement/turkey_country_alias'
    cached = json.loads((folder / 'other_cached_scope.json').read_text())
    aps = json.loads((folder / 'aps_scope.json').read_text())
    done = {j[0] for j in cached['jobs'] + aps['jobs']}
    db = connect()
    registry = {r['source_id']: r for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    jobs, audit = [], []
    for item in cached['audit']:
        if item['work_id'] in done:
            continue
        wid = item['work_id']
        doi, title, journal, sid, shard = db.execute('SELECT doi,title,journal,source_id,shard FROM papers WHERE work_id=?', (wid,)).fetchone()
        source = item['sources'][0]
        if not source.startswith('fields/qualification/'):
            audit.append({**item, 'status': 'secondary_deposit_requires_distinct_authority_adapter'})
            continue
        raw = gzip.decompress((ROOT / source).read_bytes())
        xml = source.endswith('.xml.gz')
        native = parse_oai(raw) if xml else parse_publisher(raw)
        if xml:
            native['issns'] = native.get('issn', [])
        if doi not in native['doi'] or not signature_match({'title': title, 'journal_name': journal, 'source_id': sid}, native, set(json.loads(registry[sid]['issn']))):
            audit.append({**item, 'status': 'identity_not_confirmed'})
            continue
        path = folder / 'parsed' / (hashlib.sha256((wid + source).encode()).hexdigest() + '.json')
        atomic_json(path, {'doi': doi, 'source': 'pmc' if xml else 'publisher',
            'route': 'pmc_archive_priority' if xml else 'nature_cached_page', 'parsed': native,
            'raw_receipt': source.removeprefix('fields/qualification/'), 'raw_sha256': hashlib.sha256(raw).hexdigest(),
            'network_requests': 0, 'locator': 'Single native OAI article, namespaces normalized by existing adapter' if xml else 'Native publisher author byline'})
        jobs.append((wid, shard, str(path.relative_to(ROOT))))
        audit.append({**item, 'status': 'compatible_cached_source_ready'})
    db.close()
    atomic_json(folder / 'tail_scope.json', {'jobs': jobs, 'audit': audit, 'network_requests': 0})
    result = apply_authors(selected_jobs=jobs, workers=4, reprocess_selected=True)
    atomic_json(folder / 'tail_status.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
