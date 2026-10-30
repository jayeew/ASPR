"""Validate source-bound country-only changes and record individual task outcomes."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .nature_backfill_resolve import clean_address
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def validate_change(before: list[dict], after: list[dict], mapping: dict, rule: str) -> Counter:
    counts = Counter()
    allowed = {'publisher_affiliation_slots','countries','country_source',
               'country_conflict_retained','country_conflict_disposition'}
    assert len(before) == len(after)
    for old, new in zip(before, after):
        assert {k:v for k,v in old.items() if k not in allowed} == {k:v for k,v in new.items() if k not in allowed}
        if old.get('nature_v1_slots') is not None:
            assert old == new
        if old.get('countries'):
            assert old['countries'] == new['countries']
        elif new.get('countries'):
            counts['empty_author_countries_filled'] += 1
        old_slots, new_slots = old.get('publisher_affiliation_slots') or [], new.get('publisher_affiliation_slots') or []
        assert len(old_slots) == len(new_slots)
        for a, b in zip(old_slots, new_slots):
            assert {k:v for k,v in a.items() if k not in {'country_codes','country_evidence'}} == {k:v for k,v in b.items() if k not in {'country_codes','country_evidence'}}
            if a.get('country_codes'):
                assert a == b
            elif b.get('country_codes'):
                suffix = clean_address(b['text']).rsplit(',',1)[-1].strip().rstrip('.').casefold()
                assert b['country_codes'] == [mapping[suffix]]
                assert b['country_evidence']['rule'] == rule
                assert b['country_evidence']['raw_signature'] == new['publisher_affiliation_source']
                counts['empty_country_slots_filled'] += 1
    return counts


def run(folder: Path) -> dict:
    rule = str((folder/'scope.json').relative_to(ROOT))
    scope = json.loads((ROOT/rule).read_text())
    application = json.loads((folder/'application.json').read_text())
    db = connect(); counts = Counter(); papers = set()
    for wid, before, after in db.execute("SELECT work_id,before_value,after_value FROM changes WHERE field='authors_json' AND receipt=?", (rule,)):
        counts.update(validate_change(json.loads(before), json.loads(after), scope['mapping'], rule))
        papers.add(wid)
    assert len(papers) == application.get('papers_changed',0)
    assert counts['empty_country_slots_filled'] == application.get('country_slots_filled',0)
    receipt = str((folder/'application.json').relative_to(ROOT))
    for row in application['audit']:
        assert (ROOT/row['receipt']).is_file()
        for tid, in db.execute("SELECT task_id FROM tasks WHERE kind='metadata' AND entity_key=?", (row['work_id'],)).fetchall():
            source = 'native_explicit_postal_country'
            if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?',(tid,source,receipt)).fetchone():
                continue
            db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                (tid,source,row['status'],receipt,json.dumps({**row,'rule':rule,'final_task_closure':False})))
            db.execute('UPDATE tasks SET attempt_n=attempt_n+1 WHERE task_id=?',(tid,))
            counts['ledger_attempts_added'] += 1
            if counts['ledger_attempts_added'] % 250 == 0:
                db.commit()
    db.commit();db.close()
    result={'papers_with_country_changes':len(papers),**counts,'existing_country_values_preserved':True,
            'identity_and_ownership_preserved':True,'nature_preserved':True,'source_receipts_resolve':True,
            'full_coverage_refresh_pending':True,'network_requests':0,'final_task_closure':False}
    atomic_json(folder/'validation_and_ledger.json',result);print(json.dumps(result),flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args();run(args.folder)
