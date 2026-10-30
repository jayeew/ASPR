"""Apply retained publisher HTML author links to the fixed qualified RRPharm set."""
from __future__ import annotations

import gzip
import json
import os

from .rrpharm_authors import parse_authors
from .rrpharm_pdf_metadata import OUT
from .springer_author_cache import run as apply_authors
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> dict:
    folder = OUT / 'author_cache'
    folder.mkdir(exist_ok=True)
    sources = {}
    for relative in ['staged', 'parse_corrections', 'cached_review']:
        for path in (OUT / relative).glob('*.json'):
            record = json.loads(path.read_text())
            if 'doi' in record and 'parsed' in record:
                sources[record['doi']] = (path, record)
    db = connect()
    jobs, reviews = [], []
    for path in sorted((OUT / 'results').glob('*.json')):
        result = json.loads(path.read_text())
        doi = result['doi']
        papers = db.execute("SELECT work_id,shard FROM papers WHERE doi=? AND status='included_verified_research'", (doi,)).fetchall()
        if not papers:
            continue
        if doi not in sources:
            reviews.append({'doi': doi, 'status': 'qualified_identity_receipt_not_found'})
            continue
        identity_path, identity = sources[doi]
        receipt = result['landing_receipt']
        raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
        authors = parse_authors(raw, doi)
        target = folder / (identity_path.parent.name + "_" + path.name)
        atomic_json(target, {**identity, 'source': 'publisher_html', 'route': 'rrpharmacology_html_byline',
                    'raw_receipt': os.path.relpath(ROOT / receipt['raw_receipt'], ROOT / 'fields/qualification'),
                    'identity_source_receipt': str(identity_path.relative_to(ROOT)),
                    'parsed': {**identity['parsed'], **authors},
                    'byline_evidence': 'Author name and affiliation are direct children of the same publisher byline item; no country inference'})
        jobs.extend((wid, shard, str(target.relative_to(ROOT))) for wid, shard in papers)
        reviews.append({'doi': doi, 'authors': len(authors['authors']),
                        'complete_byline_confirmed': not authors['byline_has_etal'], 'receipt': str(target.relative_to(ROOT))})
    db.close()
    applied = apply_authors(selected_jobs=jobs, workers=4) if jobs else {}
    output = {'network_requests': 0, 'qualified_papers_selected': len(jobs), 'reviews': reviews, 'application': applied}
    atomic_json(folder / 'status.json', output)
    return output


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
