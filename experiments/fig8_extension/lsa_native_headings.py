"""Resolve explicit LSA nonresearch headings from the existing native cache."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from collections import defaultdict

from .cached_date_apply import run as apply_evidence
from .fig8b_qualification_decide import key
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, jats_metadata

FOLDER = ROOT / 'supplement/lsa_native_headings'
LABELS = {'newsviews': 'News & Views', 'researchhighlight': 'Research Highlight',
          'lightpeople': 'Interview', 'correspondence': 'Correspondence',
          'perspective': 'Perspective'}


def run() -> None:
    FOLDER.mkdir(exist_ok=True)
    with (FOLDER / 'application.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        policy = {'journal': 'Light: Science & Applications',
                  'source_id': 'https://openalex.org/S2498964232',
                  'issns': ['2095-5545', '2047-7538'], 'native_labels': LABELS,
                  'official_sources': ['https://www.nature.com/lsa/articles?year=2025',
                      'https://www.nature.com/articles/s41377-022-00716-9'],
                  'basis': 'Exact editorial column labels on native articles. The journal editorial identifies Light People as an interview column.',
                  'scope': 'Matched archived brief-report with one explicit listed heading only. Generic brief reports, missing headings, research-type conflicts and integrity flags remain unresolved.'}
        atomic_json(FOLDER / 'policy.json', policy)
        db = connect()
        rows = db.execute("SELECT p.work_id,p.doi,p.title,p.journal,e.payload FROM papers p "
                          "JOIN current_evidence_index e USING(work_id) WHERE "
                          "p.source_id=? AND p.status LIKE 'pending%' AND e.kind='primary'",
                          (policy['source_id'],)).fetchall()
        db.close()
        sources = defaultdict(list); reviews = []; seen = set()
        for wid, doi, title, journal, payload in rows:
            evidence = json.loads(payload)
            if evidence.get('family') != 'archived_jats' or evidence.get('identity') != 'matched':
                continue
            receipt = evidence['receipt']
            if (doi, receipt) in seen:
                continue
            seen.add((doi, receipt))
            raw = gzip.decompress((ROOT / 'fields/qualification' / receipt).read_bytes())
            native = jats_metadata(raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw)
            native['issns'] = native.get('issn', [])
            labels = {LABELS[key(v)] for v in native.get('categories', []) if key(v) in LABELS}
            paper = {'title': title, 'journal_name': journal, 'source_id': policy['source_id']}
            if (key(native.get('article_type')) != 'briefreport' or len(labels) != 1
                    or doi not in {v.lower() for v in native.get('doi', [])}
                    or not signature_match(paper, native, set(policy['issns']))):
                reviews.append({'work_id': wid, 'receipt': receipt, 'status': 'scope_or_identity_not_confirmed'})
                continue
            source = {k: v for k, v in evidence.items() if k not in {'id', 'class', 'identity', 'work_id', 'doi'}}
            source.update(family='reviewed_native_journal_heading', kind=next(iter(labels)),
                          issns=native['issns'], journal_titles=native['journal_titles'],
                          categories=native['categories'], native_article_type=native['article_type'],
                          native_heading_sha256=hashlib.sha256(raw).hexdigest(),
                          type_policy_receipt=str((FOLDER / 'policy.json').relative_to(ROOT)))
            sources[doi].append(source)
            reviews.append({'work_id': wid, 'doi': doi, 'receipt': receipt,
                            'status': 'explicit_native_nonresearch_heading', 'kind': source['kind']})
        atomic_json(FOLDER / 'review.json', {'records': reviews, 'network_requests': 0})
        atomic_json(FOLDER / 'application_status.json', {'status': 'waiting_for_writer', 'selected_dois': len(sources)})
        result = apply_evidence(set(sources), FOLDER, explicit_sources=dict(sources)) if sources else {}
        atomic_json(FOLDER / 'application_status.json', {'status': 'fixed_native_headings_applied',
                    'result': result, 'network_requests': 0, 'final_task_closure': False})


if __name__ == '__main__':
    run()
