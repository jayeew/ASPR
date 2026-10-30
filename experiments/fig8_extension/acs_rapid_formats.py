"""Reuse native ACS short-format records and explicit official journal definitions."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from lxml import html

from .fig8b_qualification_decide import acs_reviewed_rapid_format, electronic_date, key
from .qa_type_apply import run as apply_review
from .springer_author_cache import run as apply_metadata
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .terminal_country_apply import native_metadata
from .three_journal_apply import read_csv

GUIDES = [
    ('jacsat', 'Journal of the American Chemical Society', 'Communication', 'Communications are restricted to reports'),
    ('amlccd', 'ACS Macro Letters', 'Letter', 'predominantly publishes research Letters'),
    ('orlef7', 'Organic Letters', 'Letter', 'invites original reports of fundamental research'),
]


def run(output_folder: Path | None = None) -> None:
    folder = output_folder or ROOT / 'supplement/acs_rapid_formats'
    folder.mkdir(exist_ok=True)
    if (folder / 'completion.json').exists():
        return
    policies = (json.loads((folder.parent / 'reviewed_rules.json').read_text())['rules']
                if output_folder is not None else [])
    for code, journal, category, required in ([] if output_folder is not None else GUIDES):
        receipt = json.loads((folder / (code + '.json')).read_text())
        raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
        assert receipt['http_status'] == 200 and hashlib.sha256(raw).hexdigest() == receipt['sha256']
        assert required in ' '.join(html.fromstring(raw).text_content().split())
        policies.append({'journal': journal, 'native_heading': category,
                         'source': f'supplement/acs_rapid_formats/{code}.json',
                         'scope': 'Native 2023–2025 publication heading corroborated by current format definition; '
                                  'no claim about historical word limits or publisher ownership'})
    db = connect()
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]'))
                for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    rows = db.execute("SELECT p.work_id,p.doi,p.title,p.journal,p.source_id,p.shard,p.status,e.payload "
                      "FROM papers p JOIN current_evidence_index e USING(work_id) WHERE e.kind='primary' "
                      "AND json_extract(e.payload,'$.kind')='rapid-communication' "
                      "AND p.journal IN (" + ','.join('?' for _ in policies) + ")",
                      [p['journal'] for p in policies]).fetchall()
    selected, checked, unresolved, jobs = {}, [], [], {}
    staged = folder / 'staged'; staged.mkdir(exist_ok=True)
    for wid, doi, title, journal, sid, shard, status, payload in rows:
        evidence = json.loads(payload)
        if not acs_reviewed_rapid_format(evidence):
            continue
        paper = {'work_id': wid, 'doi': doi, 'title': title, 'journal_name': journal,
                 'source_id': sid, 'shard': shard, 'before_status': status}
        receipt = 'fields/qualification/' + evidence['receipt']
        native = native_metadata(receipt, doi)
        labels = {key(v) for v in native.get('categories', [])}
        if (doi not in {d.lower() for d in native.get('doi', [])}
                or not signature_match(paper, native, journals[sid])
                or electronic_date(native.get('dates', []))[0] != evidence.get('online_date', '')
                or key(native.get('article_type')) != 'rapidcommunication'
                or not {key(v) for v in evidence['categories']} <= labels):
            unresolved.append({'work_id': wid, 'receipt': receipt, 'reason': 'Native identity or heading not reproduced'})
            continue
        selected.setdefault(wid, {**paper, 'evidence': []})['evidence'].append(evidence)
        raw = gzip.decompress((ROOT / receipt).read_bytes())
        digest = hashlib.sha256(raw).hexdigest()
        checked.append({'work_id': wid, 'receipt': receipt, 'sha256': digest, 'evidence_id': evidence['id']})
        path = staged / (hashlib.sha256(doi.encode()).hexdigest() + '.json')
        atomic_json(path, {'doi': doi, 'source': 'pmc', 'route': 'pmc', 'status': 'parsed',
                          'url': evidence['url'], 'raw_receipt': evidence['receipt'],
                          'raw_sha256': digest, 'parsed': native})
        jobs[wid] = (wid, shard, str(path.relative_to(ROOT)))
    db.close()
    atomic_json(folder / 'review.json', {'papers': list(selected.values()), 'native_checks': checked,
        'unresolved': unresolved, 'policies': policies, 'network_requests': 0,
        'qualification_scope': 'All existing evidence and conflicts remain in the decision', 'final_task_closure': False})
    print(json.dumps({'selected': len(selected), 'native_checks': len(checked), 'unresolved': len(unresolved)}), flush=True)
    if unresolved:
        raise ValueError('Inspect native unresolved records before enabling batch application')
    result = apply_review(str(folder.relative_to(ROOT / 'supplement')))
    db = connect()
    eligible = [job for wid, job in jobs.items() if db.execute(
        'SELECT status FROM papers WHERE work_id=?', (wid,)).fetchone()[0] == 'included_verified_research']
    db.close()
    metadata = apply_metadata(selected_jobs=eligible, workers=8, fill_abstract=True) if eligible else {}
    atomic_json(folder / 'completion.json', {'qualification': result, 'metadata': metadata,
        'network_requests_for_papers': 0, 'final_task_closure': False})
    print(json.dumps({'qualification': result, 'metadata': metadata}), flush=True)


if __name__ == '__main__':
    run()
