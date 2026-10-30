"""Reassess Frontiers Perspective labels using the retained official policy."""
from __future__ import annotations

import json

from .fig8b_qualification_decide import frontiers_perspective_hold
from .qa_type_apply import run as apply_review
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/frontiers_perspective_policy'


def run() -> None:
    review = json.loads((FOLDER / 'review.json').read_text())
    db = connect(); selected = {}
    for wid, shard, payload in db.execute("""SELECT p.work_id,p.shard,e.payload
            FROM papers p JOIN current_evidence_index e USING(work_id)
            WHERE p.doi LIKE '10.3389/%' AND e.kind='primary'"""):
        item = json.loads(payload)
        if frontiers_perspective_hold(item):
            selected.setdefault(wid, {'work_id': wid, 'shard': shard, 'evidence': []})['evidence'].append(item)
    db.close()
    review['papers'] = list(selected.values())
    atomic_json(FOLDER / 'review.json', review)
    print(json.dumps(apply_review(FOLDER.name)), flush=True)


if __name__ == '__main__':
    run()
