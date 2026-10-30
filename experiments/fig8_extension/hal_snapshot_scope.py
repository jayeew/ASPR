"""Extend HAL acquisition once across unresolved candidates in the fixed snapshot."""
from __future__ import annotations

import fcntl
import json
import re
import sqlite3
from collections import defaultdict
from urllib.parse import unquote

from .archive_mapping_route import assess
from .supplement_queue import ROOT
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/hal_metadata'


def run() -> dict:
    checkpoint = FOLDER / 'snapshot_scope_extension.json'
    if checkpoint.exists():
        return json.loads(checkpoint.read_text())
    db = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
    db.execute('ATTACH DATABASE ? AS loc',
        (str(ROOT / 'supplement/snapshot_location_recovery/locations.sqlite'),))
    query = """SELECT l.work_id,l.url,l.status,l.evidence,p.doi,p.title,p.journal,
        p.source_id,p.shard,p.status FROM loc.links l JOIN papers p USING(work_id)
        WHERE (l.host='hal.science' OR l.host LIKE '%.hal.science'
          OR l.host LIKE '%.archives-ouvertes.fr')
        AND (p.status LIKE 'pending%' OR (p.status='included_verified_research'
          AND (p.abstract_gap OR p.institution_gap OR p.country_gap OR p.author_cap)))"""
    groups, held = defaultdict(dict), []
    for wid, url, status, payload, doi, title, journal, sid, shard, eligibility in db.execute(query):
        if status in {'title_identity_review_required', 'doi_identity_review_required'}:
            held.append({'work_id': wid, 'url': url, 'reason': status})
            continue
        identifiers = re.findall(r'/(\w+[-_]\d{6,})(?:v\d+)?(?:/|$|[?#])', unquote(url))
        if len(identifiers) != 1:
            held.append({'work_id': wid, 'url': url, 'reason': 'identifier_parse_unresolved'})
            continue
        item = groups[identifiers[0]].setdefault(wid, {'work_id': wid, 'doi': doi,
            'title': title, 'journal': journal, 'source_id': sid, 'shard': shard,
            'eligibility_status': eligibility, 'links': []})
        item['links'].append({'work_id': wid, 'url': url,
                              'receipt': json.loads(payload)['raw_receipt']})
    db.close()
    atomic_json(FOLDER / 'snapshot_scope_preparation.json', {'status': 'checking_existing_native_fields',
        'discovered_identifiers': len(groups), 'held_links': len(held), 'network_requests': 0})
    mapping = assess([(identifier, list(papers.values())) for identifier, papers in groups.items()])
    with (FOLDER / 'collector.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = FOLDER / 'scope.json'
        scope = json.loads(path.read_text())
        added = []
        for identifier, papers in groups.items():
            if identifier in mapping:
                continue
            if identifier not in scope['identifiers']:
                scope['identifiers'][identifier] = []
                added.append(identifier)
            target = scope['identifiers'][identifier]
            for paper in papers.values():
                for link in paper['links']:
                    if link not in target:
                        target.append(link)
        atomic_json(path, scope)
        report = {'status': 'fixed_snapshot_remaining_hal_identifiers_prepared',
            'discovered_identifiers': len(groups), 'new_identifiers': added,
            'mapping_only_existing_native_evidence': mapping, 'held_links': held,
            'total_hal_identifiers': len(scope['identifiers']), 'network_requests': 0,
            'scope': 'Existing 2023–2025 candidates only; complete native metadata goes to '
                'entity review, not repeated acquisition. Repository identifiers remain leads.',
            'final_task_closure': False}
        atomic_json(checkpoint, report)
        with (FOLDER / 'scope_additions.jsonl').open('a') as stream:
            stream.write(json.dumps({'source': str(checkpoint.relative_to(ROOT)),
                'new_identifiers': added, 'reason': report['scope']}) + '\n')
    return report


if __name__ == '__main__':
    result = run()
    print(json.dumps({k: len(v) if isinstance(v, (dict, list)) else v
                      for k, v in result.items()}), flush=True)
