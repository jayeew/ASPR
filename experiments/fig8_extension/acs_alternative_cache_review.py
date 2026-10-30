"""Resolve ACS cached alternative-source paths without new network requests."""
from __future__ import annotations

import gzip
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

from .secondary_markup_recovery import title_key
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT / 'supplement/acs_remaining_paths'
    papers = [json.loads(line) for line in (folder/'path_applicability.jsonl').read_text().splitlines()]
    db = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
    db.execute('ATTACH DATABASE ? AS bulk', (str(ROOT/'supplement/bulk_cache.sqlite'),))
    grouped: dict[str, list[dict]] = defaultdict(list)
    outcomes = []
    for paper in papers:
        current = db.execute('SELECT title,status FROM papers WHERE work_id=?', (paper['work_id'],)).fetchone()
        base = {**paper, 'current_status':current[1], 'qualification_changed':False}
        if not current[1].startswith('pending'):
            outcomes.append({**base, 'cache_review':'already_decided_skip'});continue
        cached = db.execute('SELECT path,position FROM bulk.records WHERE doi=?', (paper['doi'],)).fetchone()
        if cached:
            grouped[cached[0]].append({**base, 'title':current[0], 'position':cached[1]})
        else:
            outcomes.append({**base, 'cache_review':'no_secondary_record_in_completed_query'})
    db.close()
    for path, selected in grouped.items():
        records = json.loads(gzip.decompress((ROOT/path).read_bytes()))['response']['resultList']['result']
        for paper in selected:
            item = records[paper['position']]
            matched = ((item.get('doi') or '').lower() == paper['doi'] and
                       bool(title_key(paper['title'])) and title_key(item.get('title') or '') == title_key(paper['title']))
            result = {**paper, 'secondary_record_receipt':path, 'secondary_identity_matched':matched}
            if not matched:
                outcomes.append({**result, 'cache_review':'identity_conflict_do_not_reuse'});continue
            pmcid = item.get('pmcid') or ''
            cloud = ROOT/'supplement/pmc_cloud/version_results'/f'{pmcid}.1.json'
            archive_state = json.loads(cloud.read_text()) if pmcid and cloud.exists() else {}
            urls = [value for value in item.get('fullTextUrlList', {}).get('fullTextUrl', [])
                    if urlparse(value.get('url', '')).hostname not in {'doi.org','dx.doi.org',None}]
            result.update(cache_review='matched_secondary_record_reviewed',
                          secondary_types=item.get('pubTypeList', {}).get('pubType', []),
                          secondary_online_date=item.get('electronicPublicationDate', ''),
                          pmcid_in_cache=pmcid, alternative_url_records=urls,
                          existing_archive_result=str(cloud.relative_to(ROOT)) if archive_state else '',
                          existing_archive_status=archive_state.get('status', ''),
                          next_path_state='archive_or_external_url_review' if urls or pmcid else 'indexed_official_publisher_metadata',
                          limitation='Secondary type/date are leads; generic Journal Article does not establish original research.')
            outcomes.append(result)
    with (folder/'alternative_cache_review.jsonl').open('w') as output:
        for result in outcomes:
            output.write(json.dumps(result, ensure_ascii=False)+'\n')
    atomic_json(folder/'alternative_cache_status.json', {'fixed_records':len(papers), 'reviewed':len(outcomes),
        'cache_files_read':len(grouped), 'outcomes':dict(Counter(row['cache_review'] for row in outcomes)),
        'matched_records_with_archive_or_external_url':sum(row.get('next_path_state')=='archive_or_external_url_review' for row in outcomes),
        'network_requests':0, 'qualification_changes':0, 'final_task_closure':False})


if __name__ == '__main__':
    run()
