"""Reuse v8 abstracts and complete bylines only for currently qualified identities."""
from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3

from .derived_refresh import refresh
from .elife_authors import metadata
from .elife_version_details import OUT
from .springer_author_cache import run as apply_authors
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> dict:
    folder = OUT / 'metadata_adapters'
    folder.mkdir(exist_ok=True)
    sources = json.loads((OUT / 'application/review.json').read_text())['sources']
    db = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
    jobs = []
    for doi, records in sources.items():
        candidates = db.execute("SELECT work_id,shard FROM papers WHERE doi=? AND status='included_verified_research'", (doi,)).fetchall()
        if len(candidates) != 1 or len(records) != 1:
            continue
        work_id, shard = candidates[0]
        source = records[0]
        raw = gzip.decompress((ROOT / 'fields/qualification' / source['receipt']).read_bytes())
        if hashlib.sha256(raw).hexdigest() != source['raw_sha256']:
            raise ValueError('eLife source hash changed')
        article = json.loads(raw)
        if doi not in {article.get('doi', '').lower(), article.get('doiVersion', '').lower()}:
            raise ValueError('eLife explicit DOI identity changed')
        path = folder / (hashlib.sha256(work_id.encode()).hexdigest() + '.json')
        atomic_json(path, {'source': 'publisher_api', 'route': 'elife_api',
                           'raw_receipt': source['receipt'], 'raw_sha256': source['raw_sha256'],
                           'parsed': metadata(article), 'url': source['url']})
        jobs.append((work_id, shard, str(path.relative_to(ROOT))))
    db.close()
    result = apply_authors(selected_jobs=jobs, fill_abstract=True)
    atomic_json(folder / 'status.json', result)
    refresh()
    return result


if __name__ == '__main__':
    run()
