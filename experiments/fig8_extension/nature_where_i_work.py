"""Resolve Nature's photo-profile format from retained publisher catalogue cards."""
from __future__ import annotations

import gzip
import hashlib
import json

from lxml import html

from .fig8b_qualification_decide import key, literal_nonresearch_alias
from .qa_type_apply import run as apply_review
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT / 'supplement/nature_where_i_work'
    if (folder / 'status.json').exists():
        return
    db = connect()
    rows = db.execute("SELECT p.work_id,p.doi,p.title,p.shard,p.status,e.payload "
                      "FROM papers p JOIN current_evidence_index e USING(work_id) "
                      "WHERE e.kind='primary' AND json_extract(e.payload,'$.kind')='Where I Work'").fetchall()
    db.close()
    selected, checks, unresolved = {}, [], []
    trees = {}
    for wid, doi, title, shard, status, payload in rows:
        item = json.loads(payload)
        if not literal_nonresearch_alias(item) or item.get('family') != 'nature_catalogue':
            continue
        path = ROOT / 'fields/qualification' / item['receipt']
        if not str(path).endswith('.html.gz') or not path.exists():
            unresolved.append({'work_id': wid, 'receipt': item['receipt'], 'reason': 'No raw catalogue HTML at this receipt'})
            continue
        if path not in trees:
            raw = gzip.decompress(path.read_bytes())
            trees[path] = (html.fromstring(raw.decode('utf-8')), hashlib.sha256(raw).hexdigest())
        tree, digest = trees[path]
        suffix = doi.removeprefix('10.1038/')
        cards = [card for card in tree.xpath('//article')
                 if card.xpath('.//a[@href=$href]', href='/articles/' + suffix)]
        valid = []
        for card in cards:
            links = card.xpath('.//h3/a[@href=$href]', href='/articles/' + suffix)
            types = card.xpath('.//*[@data-test="article.type"]')
            dates = card.xpath('.//time/@datetime')
            if (links and key(links[0].text_content()) == key(title)
                    and any(key(t.text_content()) == 'whereiwork' for t in types)
                    and item['online_date'] in dates):
                valid.append(card)
        if len(valid) != 1:
            unresolved.append({'work_id': wid, 'receipt': item['receipt'], 'reason': 'Native DOI/title/type/date card not uniquely matched'})
            continue
        selected.setdefault(wid, {'work_id': wid, 'doi': doi, 'shard': shard,
                                 'before_status': status, 'evidence': []})['evidence'].append(item)
        checks.append({'work_id': wid, 'evidence_id': item['id'], 'receipt': str(path.relative_to(ROOT)),
                       'raw_sha256': digest, 'doi': doi, 'type': 'Where I Work', 'online_date': item['online_date']})
    atomic_json(folder / 'review.json', {'papers': list(selected.values()), 'native_checks': checks,
        'unresolved_receipts': unresolved, 'policy': 'supplement/nature_where_i_work/policy.json',
        'scope': 'Exact Nature photo-profile type; all other primary evidence retained for conflicts',
        'network_requests': 0, 'final_task_closure': False})
    print(json.dumps({'selected': len(selected), 'native_checks': len(checks)}), flush=True)
    print(json.dumps(apply_review(folder.name)), flush=True)


if __name__ == '__main__':
    run()
