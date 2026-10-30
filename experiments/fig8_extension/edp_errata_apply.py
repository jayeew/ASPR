"""Apply explicit EDP errata subject evidence, retaining conflicting primary types."""
from __future__ import annotations

import hashlib
import json

from .cached_date_apply import run as apply_sources
from .derived_refresh import refresh
from .edp_oai_harvest import FOLDER, URL
from .edp_oai_review import run as review_cache
from .supplement_sources import atomic_json


def run() -> dict:
    review_cache()
    review = json.loads((FOLDER / 'cache_review.json').read_text())
    sources = {}
    for record in review['records']:
        if record['status'] != 'explicit_errata_subject_review':
            continue
        fields = record['fields']
        doi = record['candidate']['doi']
        sources[doi] = [{
            'family': 'publisher_edp_oai', 'title': fields['title'][0],
            'journal_titles': [record['set_title']],
            'source_id': record['candidate']['source_id'],
            'journal_identity_rule': 'Exact official set title matches candidate journal after ampersand/and normalization; supplement/edp_oai/sets.json and cache_review.json',
            'kind': 'erratum', 'categories': fields['subject'],
            'online_date': '', 'date_precision': '',
            'receipt': '../../' + record['raw_receipt'],
            'raw_sha256': record['raw_sha256'],
            'url': URL, 'oai_identifier': record['oai_identifier'],
            'type_rule': 'Publisher DC subject explicitly states errata or errata, addenda; generic Text is not used',
            'date_limitation': 'DC date not interpreted as online publication',
        }]
    folder = FOLDER / 'errata_application'
    folder.mkdir(exist_ok=True)
    atomic_json(folder / 'sources.json', sources)
    applied_path = folder / 'applied_fingerprints.json'
    applied = json.loads(applied_path.read_text()) if applied_path.exists() else {}
    fingerprints = {doi: hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()
                    for doi, items in sources.items()}
    pending = {doi: items for doi, items in sources.items()
               if applied.get(doi) != fingerprints[doi]}
    if not pending:
        return {'new_sources': 0, 'previously_applied': len(sources)}
    result = apply_sources(set(pending), folder, explicit_sources=pending)
    applied.update({doi: fingerprints[doi] for doi in pending})
    atomic_json(applied_path, applied)
    refresh()
    return result


if __name__ == '__main__':
    run()
