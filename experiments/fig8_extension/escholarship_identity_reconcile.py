"""Resolve repository formatting differences using exact title and registry ISSNs."""
from __future__ import annotations

import csv
import json
from collections import Counter

from .cached_date_apply import deposited_title
from .escholarship_metadata import FOLDER, record_outcomes
from .fig8b_qualification_decide import key
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def run() -> None:
    data = json.loads((FOLDER/'review.json').read_text())
    journals = {}
    with (ROOT/'fields/journal_registry.csv').open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream): journals[row['source_id']] = set(json.loads(row['issn'] or '[]'))
    db = connect(); audit = []
    for record in data['records']:
        if record['status'] != 'repository_identity_conflict': continue
        native = record['metadata']
        sid = db.execute('SELECT source_id FROM papers WHERE work_id=?',(record['work_id'],)).fetchone()[0]
        title_ok = key(deposited_title(native['title'])) == key(deposited_title(record['title']))
        journal_ok = key(native.get('journal')) == key(record['journal']) or native.get('issn') in journals[sid]
        if native['id'] == record['ark'] and title_ok and journal_ok:
            before = dict(record)
            record.update(status='matched_repository_metadata_not_publisher_qualification',
                          identity_rule='Exact DOI lookup and ARK; decoded title; exact journal name or registry ISSN',
                          registry_source_id=sid, registry_issns=sorted(journals[sid]))
            audit.append({'work_id':record['work_id'],'before':before,'after':dict(record)})
    db.close()
    if audit:
        with (FOLDER/'identity_changes.jsonl').open('a') as stream:
            for change in audit: stream.write(json.dumps(change)+'\n')
    data['outcomes'] = dict(Counter(p['status'] for p in data['records']))
    atomic_json(FOLDER/'review.json',data)
    atomic_json(FOLDER/'identity_reconciliation.json',{'changed_records':len(audit),'outcomes':data['outcomes'],'network_requests':0,'qualification_changes':0})
    record_outcomes()
    print(json.dumps({'changed_records':len(audit),'outcomes':data['outcomes']}))


if __name__ == '__main__':
    run()
