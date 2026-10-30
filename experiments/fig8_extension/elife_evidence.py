"""Review official eLife list records without conflating preprint and journal dates."""
from __future__ import annotations

import json
from datetime import date

from .cached_date_apply import run as apply_sources
from .derived_refresh import refresh
from .elife_batch import FOLDER, URL
from .elife_text import plain
from .fig8b_qualification_build import key
from .supplement_sources import atomic_json


def evidence(record: dict) -> dict | None:
    item, paper = record['native'], record['candidate']
    declared_dois = {item.get('doi', '').lower()}
    if record.get('verified_v8'):
        declared_dois.add(item.get('doiVersion', '').lower())
    if paper['doi'].lower() not in declared_dois or key(paper['journal']) != 'elife':
        return None
    if item.get('status') not in {'vor', 'poa'} or item.get('stage') != 'published':
        return None
    raw_date = item.get('published', '')[:10]
    try:
        online_date = date.fromisoformat(raw_date).isoformat()
    except ValueError:
        online_date = ''
    reviewed_preprint_path = item.get('elocationId', '').startswith('RP') and not record.get('verified_v8')
    type_policy = ('insight_policy.json' if item['type'] == 'insight' else
                   'format_policy.json' if item['type'] in {'short-report', 'research-advance'} else '')
    return {
        'type_policy_receipt': '../../supplement/elife_official/' + type_policy if type_policy else '',
        'family': 'publisher_elife_api', 'title': plain(item['title']),
        'source_id': paper['source_id'], 'journal_titles': ['eLife'],
        'kind': item['type'], 'categories': [],
        'online_date': '' if reviewed_preprint_path else online_date,
        'date_precision': '' if reviewed_preprint_path or not online_date else 'day',
        'receipt': '../../' + record['raw_receipt'], 'url': URL,
        'native_api_version': 8 if record.get('verified_v8') else 0,
        'native_version_doi': item.get('doiVersion', '') if record.get('verified_v8') else '',
        'raw_sha256': record.get('raw_sha256', ''),
        'native_doi': item['doi'], 'native_status': item['status'],
        'native_published': item.get('published'), 'native_status_date': item.get('statusDate'),
        'schema_receipt': '../../supplement/elife_official/schema/article.v1.yaml',
        'schema_commit': '33062ea5ce7ce424bda26f5aaada9c05652710ae',
        'date_rule': 'Publisher API published means first article publication; versionDate is not used',
        'scope_review_required': reviewed_preprint_path,
        'scope_note': ('Reviewed-preprint pathway requires historical journal publication date review'
                       if reviewed_preprint_path else 'Published journal article API; exact title checked by shared matcher'),
    }


def run() -> dict:
    state = json.loads((FOLDER / 'batch_progress.json').read_text())
    sources = {}
    unresolved = []
    for doi, records in state['matched'].items():
        items = [e for record in records if (e := evidence(record)) is not None]
        if items:
            sources[doi] = items
        else:
            unresolved.append({'doi': doi, 'reason': 'API identity or publication stage unsupported'})
    folder = FOLDER / 'qualification_application'
    folder.mkdir(exist_ok=True)
    atomic_json(folder / 'review.json', {'sources': sources, 'unresolved': unresolved,
                                        'harvest_status': state['status']})
    result = apply_sources(set(sources), folder, explicit_sources=sources)
    refresh()
    return result


if __name__ == '__main__':
    run()
