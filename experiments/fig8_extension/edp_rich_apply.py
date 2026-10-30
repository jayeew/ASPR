"""Apply fixed EDP native JATS records with standard identity/date/conflict guards."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os

from .derived_refresh import refresh
from .edp_oai_harvest import FOLDER, URL
from .fig8b_qualification_decide import electronic_date, imprint_dates
from .publisher_delta_apply import apply_available
from .springer_author_cache import run as apply_authors
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def staged_record(result: dict) -> dict:
    parsed = result['parsed']
    if parsed['doi'] != [result['candidate']['doi'].lower()]:
        raise ValueError('Native DOI does not match candidate')
    date, precision = electronic_date(parsed['dates'])
    receipt = result['receipt']
    raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
    return {'source': 'publisher_jats', 'route': 'edp_native', 'status': 'parsed',
            'doi': parsed['doi'][0], 'url': URL, 'raw_sha256': hashlib.sha256(raw).hexdigest(),
            'raw_receipt': os.path.relpath(ROOT / receipt['raw_receipt'], ROOT / 'fields/qualification'),
            'acquisition_receipt': receipt,
            'parsed': {**parsed, 'issns': parsed['issn'], 'article_types': [parsed['article_type']],
                       'online_date': date, 'date_precision': precision,
                       'evidence_categories': parsed.get('categories', []),
                       'publication_dates': imprint_dates(parsed['dates'])}}


def run() -> None:
    folder = FOLDER / 'rich_set_review'
    with (folder / 'apply.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (folder / 'candidate_collector.lock').open('a') as collector:
            fcntl.flock(collector, fcntl.LOCK_EX)
        staged = folder / 'staged'
        staged.mkdir(exist_ok=True)
        jobs, held = [], []
        for path in sorted((folder / 'candidates').glob('*.json')):
            result = json.loads(path.read_text())
            if not result.get('parsed'):
                held.append({'receipt': str(path.relative_to(ROOT)), 'reason': result['status']})
                continue
            try:
                record = staged_record(result)
            except ValueError as error:
                held.append({'receipt': str(path.relative_to(ROOT)), 'reason': str(error)})
                continue
            record['candidate_receipt'] = str(path.relative_to(ROOT))
            target = staged / path.name
            if not target.exists():
                atomic_json(target, record)
            jobs.append((record['doi'], str(target.relative_to(ROOT))))
        qualification = apply_available(supplied_jobs=jobs) if jobs else {}
        db = connect()
        selected = [(wid, shard, receipt) for doi, receipt in jobs for wid, shard in db.execute(
            "SELECT work_id,shard FROM papers WHERE doi=? AND status='included_verified_research'", (doi,))]
        db.close()
        authors = apply_authors(selected_jobs=selected, workers=4, fill_abstract=True) if selected else {}
        if authors.get('papers_changed'):
            from .official_rule_reuse import run as reuse_rules
            reuse_rules()
        result = {'staged_records': len(jobs), 'qualification': qualification,
                  'authors': authors, 'held': held, 'network_requests': 0, 'final_task_closure': False}
        atomic_json(folder / 'application.json', result)
        if qualification.get('changed_decisions') or authors.get('papers_changed'):
            refresh()
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
