"""Apply literal Nature news/review type aliases after native cache checks."""
from __future__ import annotations

import gzip
import hashlib
import json

from lxml import html

from .fig8b_qualification_decide import key, literal_nonresearch_alias
from .qa_type_apply import run as apply_review
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, parse_publisher


def native_check(raw: bytes, evidence: dict, doi: str, title: str) -> bool:
    decoded = raw.decode('utf-8')
    if evidence['family'] == 'nature_catalogue':
        for card in html.fromstring(decoded).xpath('//article'):
            links = card.xpath('.//h3/a[@href=$href]', href='/articles/' + doi.removeprefix('10.1038/'))
            types = card.xpath('.//*[@data-test="article.type"]')
            if (links and key(links[0].text_content()) == key(title)
                    and any(key(t.text_content()) == key(evidence['kind']) for t in types)):
                return True
        return False
    value = parse_publisher(decoded)
    titles = value.get('title') or []
    titles = [titles] if isinstance(titles, str) else titles
    labels = {key(value.get('article_type'))} | {key(x) for x in value.get('article_types', [])}
    return (doi.lower() in {d.lower() for d in value.get('doi', [])}
            and key(evidence['kind']) in labels and key(title) in {key(t) for t in titles})


def run() -> None:
    folder = ROOT / 'supplement/nature_literal_formats'
    folder.mkdir(exist_ok=True)
    if (folder / 'status.json').exists():
        return
    db = connect()
    selected, checks, unresolved = {}, [], []
    query = """SELECT p.work_id,p.doi,p.title,p.shard,p.status,e.payload FROM papers p
        JOIN current_evidence_index e USING(work_id) WHERE e.kind='primary'
        AND json_extract(e.payload,'$.kind') IN ('Career Q&A','News Q&A','ReviewPaper','Product News')
        AND json_extract(e.payload,'$.family') IN ('nature_page','nature_catalogue')"""
    for wid, doi, title, shard, status, payload in db.execute(query):
        evidence = json.loads(payload)
        if not literal_nonresearch_alias(evidence):
            continue
        path = ROOT / 'fields/qualification' / evidence['receipt']
        valid = False
        if path.exists() and str(path).endswith('.html.gz'):
            raw = gzip.decompress(path.read_bytes())
            valid = native_check(raw, evidence, doi, title)
            if valid:
                checks.append({'work_id': wid, 'evidence_id': evidence['id'],
                               'receipt': str(path.relative_to(ROOT)),
                               'sha256': hashlib.sha256(raw).hexdigest(), 'type': evidence['kind']})
        if not valid:
            unresolved.append({'work_id': wid, 'receipt': evidence['receipt'],
                               'reason': 'Native DOI/title/type not reproduced by this cache parser'})
        selected.setdefault(wid, {'work_id': wid, 'doi': doi, 'shard': shard,
                                 'before_status': status, 'evidence': []})['evidence'].append(evidence)
    db.close()
    verified = {r['work_id'] for r in checks}
    unchecked = set(selected) - verified
    atomic_json(folder / 'review.json', {'papers': list(selected.values()), 'native_checks': checks,
        'unresolved_receipts': unresolved, 'unverified_work_ids': sorted(unchecked),
        'rule': 'Literal ReviewPaper is review; Product News is news; Career/News Q&A is an interview format. '
                'Publisher-labelled type only, not title keywords. Conflicting research sources remain pending.',
        'network_requests': 0, 'final_task_closure': False})
    print(json.dumps({'papers': len(selected), 'native_confirmed': len(verified), 'unchecked': sorted(unchecked)}), flush=True)
    if unchecked:
        raise ValueError('Unchecked native sources; inspect before application')
    print(json.dumps(apply_review(folder.name)), flush=True)


if __name__ == '__main__':
    run()
