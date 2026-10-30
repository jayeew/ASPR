"""Exclude identity-checked ASO companion visual abstracts using the official guide."""
from __future__ import annotations

import hashlib
import json

from .fig8b_qualification_decide import aso_visual_abstract, key
from .qa_type_apply import run as apply_review
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .terminal_country_apply import native_metadata
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/aso_visual_abstract'


def run() -> None:
    if (FOLDER / 'status.json').exists():
        return
    policy = FOLDER / 'official_instructions.pdf'
    assert policy.read_bytes().startswith(b'%PDF')
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]'))
                for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    db = connect()
    papers, checked, unresolved = {}, set(), []
    query = """SELECT p.work_id,p.doi,p.title,p.journal,p.source_id,p.shard,p.status,e.payload
        FROM papers p JOIN current_evidence_index e USING(work_id)
        WHERE p.journal='Annals of Surgical Oncology' AND e.kind='primary'"""
    for wid, doi, title, journal, sid, shard, status, payload in db.execute(query):
        evidence = json.loads(payload)
        if not aso_visual_abstract(evidence):
            continue
        receipt = 'fields/qualification/' + evidence['receipt']
        paper = {'work_id': wid, 'doi': doi, 'title': title, 'journal_name': journal,
                 'source_id': sid, 'shard': shard, 'before_status': status}
        pair = (wid, receipt)
        if pair not in checked:
            native = native_metadata(receipt, doi)
            labels = {key(v) for v in native.get('article_types', [])}
            if (doi not in {d.lower() for d in native.get('doi', [])}
                    or not signature_match(paper, native, journals[sid])
                    or 'asovisualabstract' not in labels):
                unresolved.append({'work_id': wid, 'receipt': receipt,
                                   'status': 'native_type_or_identity_unconfirmed'})
                continue
            checked.add(pair)
        row = papers.setdefault(wid, {**paper, 'evidence': []})
        row['evidence'].append(evidence)
    db.close()
    atomic_json(FOLDER / 'review.json', {'papers': list(papers.values()),
        'native_checks': [{'work_id': w, 'receipt': r} for w, r in sorted(checked)],
        'unresolved': unresolved, 'policy': {
            'source_url': 'https://surgonc.org/wp-content/uploads/2021/02/ASOInstructions_FEB9.2021_web_links.pdf',
            'receipt': str(policy.relative_to(ROOT)),
            'sha256': hashlib.sha256(policy.read_bytes()).hexdigest(),
            'locator': 'Page 2, Original Article Manuscript Preparation, New as of January 2021',
            'interpretation': 'Visual/video abstracts are separate companion publications. Exclude explicitly identified visual abstracts from original-paper denominator; original articles and other special types unaffected.'},
        'paper_network_requests': 0, 'final_task_closure': False})
    print(json.dumps({'selected_papers': len(papers), 'checked_native_pages': len(checked),
                      'unresolved': len(unresolved)}), flush=True)
    print(json.dumps(apply_review(FOLDER.name)), flush=True)


if __name__ == '__main__':
    run()
