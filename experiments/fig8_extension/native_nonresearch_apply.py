"""Apply explicit native nonresearch formats with source-level validation."""
from __future__ import annotations

import gzip
import json

from .fig8b_qualification_decide import key, native_nonresearch_format
from .qa_type_apply import run as apply_review
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, parse_publisher
from .terminal_country_apply import native_metadata
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/native_nonresearch_formats'


def run() -> None:
    if (FOLDER / 'status.json').exists():
        return
    policies = [json.loads((FOLDER / name).read_text()) for name in
                ['jats_article_types.html.source.json', 'polymer_instructions.pdf.source.json']]
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]'))
                for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    db = connect()
    selected, checks, unresolved = {}, [], []
    query = """SELECT p.work_id,p.doi,p.title,p.journal,p.source_id,p.shard,p.status,e.payload
        FROM current_evidence_index e JOIN papers p USING(work_id)
        WHERE e.kind='primary' AND json_extract(e.payload,'$.kind') IN
        ('discussion','in-brief','reviewer-report','editor-report','community-comment','ReviewPaper','Focus Review')"""
    for wid, doi, title, journal, sid, shard, status, payload in db.execute(query):
        evidence = json.loads(payload)
        if not native_nonresearch_format(evidence):
            continue
        receipt = 'fields/qualification/' + evidence['receipt']
        paper = {'work_id': wid, 'doi': doi, 'title': title, 'journal_name': journal,
                 'source_id': sid, 'shard': shard, 'before_status': status}
        native = native_metadata(receipt, doi)
        if evidence.get('family') == 'nature_page' and receipt.endswith('.html.gz'):
            raw = gzip.decompress((ROOT / receipt).read_bytes())
            # Current Nature HTML is UTF-8; do not let an absent HTML charset
            # make lxml reinterpret Greek letters or typographic punctuation.
            try:
                decoded = raw.decode('utf-8')
            except UnicodeDecodeError:
                decoded = None
            if decoded is not None and decoded.encode('utf-8') == raw:
                native.update(parse_publisher(decoded))
        labels = {key(native.get('article_type'))} | {key(v) for v in native.get('article_types', [])}
        if (doi not in {v.lower() for v in native.get('doi', [])}
                or not signature_match(paper, native, journals[sid])
                or key(evidence['kind']) not in labels):
            unresolved.append({'work_id': wid, 'receipt': receipt, 'status': 'native_identity_or_type_unconfirmed'})
            continue
        selected.setdefault(wid, {**paper, 'evidence': []})['evidence'].append(evidence)
        checks.append({'work_id': wid, 'receipt': receipt, 'kind': evidence['kind']})
    db.close()
    atomic_json(FOLDER / 'review.json', {'papers': list(selected.values()), 'native_checks': checks,
        'unresolved': unresolved, 'policy_sources': policies,
        'interpretation': 'JATS article-level discussion/in-brief/peer-review reports describe commentary, issue summaries or peer review, not the target research. Polymer Focus Review/ReviewPaper are review formats. Research discussion sections and other journals are unaffected.',
        'network_requests': 0, 'final_task_closure': False})
    print(json.dumps({'selected_papers': len(selected), 'native_checks': len(checks), 'unresolved': len(unresolved)}), flush=True)
    print(json.dumps(apply_review(FOLDER.name)), flush=True)


if __name__ == '__main__':
    run()
