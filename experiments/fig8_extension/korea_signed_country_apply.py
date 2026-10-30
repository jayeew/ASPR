"""Fill exact reviewed signed-campus countries without changing entity ownership."""
from __future__ import annotations
import fcntl
import json
from collections import Counter, defaultdict
import pyarrow as pa
import pyarrow.parquet as pq
from .derived_refresh import refresh
from .fig8b_qualification_recompute import metrics
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

RULE = 'supplement/korea_exact_locations/rules.json'


def fill(author: dict, addresses: set[str]) -> tuple[dict, int]:
    if author.get('nature_v1_slots') is not None:
        return author, 0
    receipt = author.get('publisher_affiliation_source')
    if not receipt or not (ROOT / receipt).is_file():
        return author, 0
    new = json.loads(json.dumps(author))
    changed = 0
    for slot in new.get('publisher_affiliation_slots') or []:
        if slot.get('text') not in addresses or slot.get('country_codes'):
            continue
        slot['country_codes'] = ['KR']
        slot['country_evidence'] = {'rule': RULE, 'author_address_receipt': receipt,
                                    'basis': 'exact_complete_signed_university_and_city_with_official_campus_country'}
        changed += 1
    slots = new.get('publisher_affiliation_slots') or []
    if changed and not new.get('countries') and slots and all(s.get('country_codes') for s in slots):
        new['countries'] = sorted({c for s in slots for c in s['country_codes']})
        new['country_source'] = 'all_signed_slots_with_reviewed_country_evidence'
    return new, changed


def run() -> None:
    rules = json.loads((ROOT / RULE).read_text())
    addresses = {r['exact_address'] for r in rules['sources'] if r.get('raw_receipt')}
    counts = Counter(); records = []
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); db = connect(); jobs = defaultdict(set)
        for address in addresses:
            for wid, shard in db.execute("SELECT DISTINCT p.work_id,p.shard FROM tasks t JOIN task_papers l USING(task_id) JOIN papers p ON p.work_id=l.work_id WHERE t.kind='institution' AND t.entity_key=? AND p.status='included_verified_research'", (' '.join(address.casefold().split()),)):
                jobs[shard].add(wid)
        for shard, ids in jobs.items():
            path = ROOT / 'fields/papers' / shard; table = pq.read_table(path); rows = table.to_pylist(); changed = False
            for paper in rows:
                if paper['work_id'] not in ids:continue
                before = dict(paper); original = json.loads(paper['authors_json']); authors = []; slots = 0
                for author in original:
                    updated, delta = fill(author, addresses); authors.append(updated); slots += delta
                    assert fill(updated, addresses)[0] == updated
                if not slots:continue
                paper['authors_json'] = json.dumps(authors, ensure_ascii=False)
                values = metrics(paper)
                for metric, value in values.items():
                    if 'after_' + metric in paper:paper['after_' + metric] = value
                for field in paper:
                    if paper[field] != before[field]:
                        db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)', (paper['work_id'], field, json.dumps(before[field]), json.dumps(paper[field]), RULE, 'Exact signed campus and official Republic of Korea address; only empty countries filled'))
                db.execute('UPDATE papers SET institution_gap=?,country_gap=? WHERE work_id=?', (not values['every_author_institution'], not values['every_author_country'], paper['work_id']))
                counts['papers_changed'] += 1; counts['empty_slot_countries_filled'] += slots
                records.append({'work_id': paper['work_id'], 'slots_filled': slots, 'rule': RULE}); changed = True
            if changed:
                tmp = path.with_suffix('.parquet.tmp'); pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), tmp, compression='zstd'); tmp.replace(path); db.commit()
        db.close()
    atomic_json(ROOT/'supplement/korea_exact_locations/signed_country_application.json', {'counts': dict(counts), 'records': records, 'application_network_requests': 0})
    refresh(); print(json.dumps(dict(counts)), flush=True)


if __name__ == '__main__':run()
