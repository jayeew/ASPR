"""Apply DOI/title-bound AIP type and online dates from cached publication fronts."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections import Counter

from pypdf import PdfReader

from .aip_pdf_front import extract
from .cached_date_apply import run as apply_sources
from .derived_refresh import refresh
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/hal_metadata/pdf_fronts'


def run() -> dict:
    db, sources, reviews = connect(), {}, []
    for path in sorted((FOLDER / 'receipts').glob('*.json')):
        item = json.loads(path.read_text())
        papers = [p for p in item.get('fixed_papers', []) if p['doi'].startswith('10.1063/')]
        if not papers or item.get('review_status') != 'front_metadata_extracted_not_qualification':
            continue
        raw = gzip.decompress((ROOT / item['raw_receipt']).read_bytes())
        reader = PdfReader(io.BytesIO(raw))
        pages = [reader.pages[i].extract_text() or '' for i in range(min(3, len(reader.pages)))]
        for paper in papers:
            title, journal = db.execute('SELECT title,journal FROM papers WHERE work_id=?',
                                       (paper['work_id'],)).fetchone()
            proof = extract(pages, paper['doi'], title, journal)
            review = {'doi': paper['doi'], 'work_id': paper['work_id'],
                'native_receipt': item['raw_receipt'], 'raw_sha256': hashlib.sha256(raw).hexdigest(),
                'repository_identity_receipt': paper['repository_record'], 'proof': proof,
                'status': proof['status']}
            reviews.append(review)
            if proof['status'] != 'native_aip_front_confirmed':
                continue
            evidence = {'family': 'archived_aip_publisher_pdf', 'doi': paper['doi'],
                'title': title, 'journal_titles': [journal], 'issns': [], 'kind': proof['kind'],
                'categories': [], 'online_date': proof['online_date'], 'date_precision': 'day',
                'receipt': '../../' + item['raw_receipt'], 'url': item['url'],
                'raw_sha256': review['raw_sha256'], 'native_front_proof': proof,
                'repository_identity_receipt': paper['repository_record'],
                'provenance_limit': 'Primary type/date from original publisher cover/front; '
                    'DOI/title bound to matched repository journal identity. Repository submission '
                    'dates and generic ART labels not used. Conflicts remain subject to qualification rules.'}
            sources.setdefault(paper['doi'], []).append(evidence)
    db.close()
    folder = FOLDER / 'aip_fronts'
    folder.mkdir(exist_ok=True)
    atomic_json(folder / 'source_review.json', {'records': reviews, 'network_requests': 0,
        'outcomes': dict(Counter(r['status'] for r in reviews))})
    atomic_json(folder / 'sources.json', sources)
    atomic_json(folder / 'status.json', {'status': 'waiting_for_writer', 'source_dois': len(sources)})
    result = apply_sources(set(sources), folder, explicit_sources=sources) if sources else {}
    report = {'qualification': result, 'network_requests': 0, 'final_task_closure': False}
    atomic_json(folder / 'application.json', report)
    atomic_json(folder / 'status.json', {'status': 'applied', **report})
    if sources:
        refresh()
    return report


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
