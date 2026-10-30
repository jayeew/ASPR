"""Reuse cached HAL-hosted ACS publisher fronts, retaining literal type labels."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections import Counter

from pypdf import PdfReader

from .acs_pdf_front import extract
from .cached_date_apply import run as apply_sources
from .derived_refresh import refresh
from .fig8b_qualification_decide import key
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/hal_metadata/pdf_fronts'


def run() -> dict:
    db, sources, reviews = connect(), {}, []
    for path in sorted((FOLDER / 'receipts').glob('*.json')):
        item = json.loads(path.read_text())
        papers = [p for p in item.get('fixed_papers', []) if p['doi'].startswith('10.1021/')]
        if not papers or item.get('review_status') != 'front_metadata_extracted_not_qualification':
            continue
        raw = gzip.decompress((ROOT / item['raw_receipt']).read_bytes())
        reader = PdfReader(io.BytesIO(raw))
        pages = [reader.pages[i].extract_text() or '' for i in range(min(3, len(reader.pages)))]
        for paper in papers:
            title, journal = db.execute('SELECT title,journal FROM papers WHERE work_id=?', (paper['work_id'],)).fetchone()
            proofs = []
            for number, text in enumerate(pages, 1):
                proof = extract(text, paper['doi'])
                if proof['status'] == 'native_acs_front_confirmed' and key(title) in key(text) and key(journal) in key(text):
                    proofs.append({**proof, 'pdf_page': number})
            review = {'doi': paper['doi'], 'work_id': paper['work_id'], 'native_receipt': item['raw_receipt'],
                      'raw_sha256': hashlib.sha256(raw).hexdigest(), 'proofs': proofs,
                      'status': 'matched_native_publisher_front' if len(proofs) == 1 else 'no_unique_native_front_identity_type_date'}
            reviews.append(review)
            if len(proofs) != 1:
                continue
            proof = proofs[0]
            sources[paper['doi']] = [{'family': 'archived_acs_publisher_pdf', 'doi': paper['doi'],
                'title': title, 'journal_titles': [journal], 'issns': [], 'kind': proof['kind'],
                'categories': [], 'online_date': proof['online_date'], 'date_precision': 'day',
                'receipt': '../../' + item['raw_receipt'], 'url': item['url'],
                'raw_sha256': review['raw_sha256'], 'native_front_proof': proof,
                'provenance_limit': 'Native publisher front archived by HAL; literal Article/Letter labels remain unresolved without an applicable type rule. Repository dates are excluded.'}]
    db.close()
    folder = FOLDER / 'acs_fronts'
    folder.mkdir(exist_ok=True)
    atomic_json(folder / 'source_review.json', {'records': reviews, 'network_requests': 0,
                'outcomes': dict(Counter(r['status'] for r in reviews))})
    atomic_json(folder / 'sources.json', sources)
    result = apply_sources(set(sources), folder, explicit_sources=sources) if sources else {}
    atomic_json(folder / 'application.json', {'qualification': result, 'network_requests': 0, 'final_task_closure': False})
    refresh()
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
