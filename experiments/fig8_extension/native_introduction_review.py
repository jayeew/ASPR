"""Resolve publication-level introductions from retained native XML evidence."""
from __future__ import annotations

import gzip
import hashlib
import json

from lxml import etree

from .terminal_country_apply import native_metadata
from .fig8b_qualification_decide import key, native_nonresearch_format
from .qa_type_apply import run as apply_review
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT / 'supplement/native_introduction_review'
    folder.mkdir(exist_ok=True)
    if (folder / 'status.json').exists():
        return
    db = connect()
    query = """SELECT p.work_id,p.doi,p.title,p.shard,p.status,e.payload FROM papers p
        JOIN current_evidence_index e USING(work_id) WHERE p.status LIKE 'pending%'
        AND e.kind='primary' AND json_extract(e.payload,'$.family')='archived_jats'
        AND json_extract(e.payload,'$.kind')='introduction'"""
    papers, checks, unresolved = {}, [], []
    for wid, doi, title, shard, status, payload in db.execute(query):
        item = json.loads(payload)
        if not native_nonresearch_format(item):
            unresolved.append({'work_id': wid, 'reason': 'Identity, category or scope hold retained'})
            continue
        receipt = 'fields/qualification/' + item['receipt']
        try:
            record = native_metadata(receipt, doi)
        except (OSError, ValueError, etree.XMLSyntaxError) as error:
            unresolved.append({'work_id': wid, 'receipt': receipt, 'reason': str(error)})
            continue
        titles = record.get('title') or []
        titles = [titles] if isinstance(titles, str) else titles
        if (doi not in {d.lower() for d in record.get('doi', [])}
                or key(title) not in {key(t) for t in titles}
                or key(record.get('article_type')) != 'introduction'
                or set(record.get('journal_titles', [])) != set(item.get('journal_titles', []))):
            unresolved.append({'work_id': wid, 'receipt': receipt, 'reason': 'Native identity/type not reproduced'})
            continue
        raw = gzip.decompress((ROOT / receipt).read_bytes())
        checks.append({'work_id': wid, 'evidence_id': item['id'], 'receipt': receipt,
                       'sha256': hashlib.sha256(raw).hexdigest(), 'locator': 'article/@article-type'})
        papers.setdefault(wid, {'work_id': wid, 'doi': doi, 'shard': shard,
                               'before_status': status, 'evidence': []})['evidence'].append(item)
    db.close()
    atomic_json(folder / 'review.json', {'papers': list(papers.values()), 'native_checks': checks,
        'unresolved': unresolved, 'policy_url': 'https://jats.nlm.nih.gov/archiving/tag-library/1.3/attribute/article-type.html',
        'policy_interpretation': 'Publication-level introduction denotes prefatory material for a publication or article series. Body section headings are never used. Conflicting research categories and other primary sources remain held.',
        'network_requests': 0, 'final_task_closure': False})
    print(json.dumps({'selected': len(papers), 'checks': len(checks), 'unresolved': len(unresolved)}), flush=True)
    print(json.dumps(apply_review(folder.name)), flush=True)


if __name__ == '__main__':
    run()
