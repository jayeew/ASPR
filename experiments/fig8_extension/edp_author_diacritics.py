"""Fill four reviewed EDP author links differing only in combining accents."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import unicodedata
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

from .derived_refresh import refresh
from .edp_rich_candidates import extract
from .fig8b_qualification_build import key
from .fig8b_qualification_recompute import metrics
from .nature_backfill_registry import normalized as country_key
from .nature_backfill_resolve import country_names
from .primary_byline_comparison import fill_missing_links
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv

PAIRS = {
    '10.1051/m2an/2024044': ('Élise Grosjean', 'Elise Grosjean'),
    '10.1051/ro/2023052': ('Elói Araújo', 'Eloi Araujo'),
    '10.1051/m2an/2023013': ('Agustín Somacal', 'Agustin Somacal'),
    '10.1051/m2an/2023023': ('Élie Bretin', 'Elie Bretin'),
}


def fold(value: str) -> str:
    return ' '.join(''.join(c for c in unicodedata.normalize('NFD', value)
                           if not unicodedata.combining(c)).casefold().split())


def run() -> dict:
    folder = ROOT / 'supplement/edp_oai/rich_set_review'
    registry, by_name = defaultdict(list), defaultdict(list)
    for entity in json.loads((ROOT / 'fields/institution_registry.json').read_text()):
        if entity.get('ror'):
            registry[entity['ror'].rstrip('/')].append(entity['id'])
        by_name[key(entity['display_name'])].append(entity)
    names = country_names()
    for ambiguous in ['korea', 'georgia', 'congo']:
        names.pop(country_key(ambiguous), None)
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]'))
                for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    changes, checked = [], []
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        for doi, (old_name, native_name) in PAIRS.items():
            assert fold(old_name) == fold(native_name)
            candidates = db.execute("SELECT work_id,shard FROM papers WHERE doi=? AND status='included_verified_research'", (doi,)).fetchall()
            assert len(candidates) == 1
            wid, shard = candidates[0]
            token = hashlib.sha256(doi.encode()).hexdigest()
            source_path = folder / 'staged' / (token + '.json')
            record = json.loads(source_path.read_text())
            raw = gzip.decompress((ROOT / 'fields/qualification' / record['raw_receipt']).read_bytes())
            assert hashlib.sha256(raw).hexdigest() == record['raw_sha256']
            source = extract(raw, doi)
            assert source is not None
            source['issns'] = source['issn']
            path = ROOT / 'fields/papers' / shard
            table = pq.read_table(path)
            rows = table.to_pylist()
            paper = next(r for r in rows if r['work_id'] == wid)
            assert signature_match(paper, source, journals[paper['source_id']])
            old = json.loads(paper['authors_json'])
            assert sum(fold(a['name']) == fold(old_name) for a in old) == 1
            matching = [a for a in source['authors'] if fold(a['name']) == fold(native_name)]
            assert len(matching) == 1 and matching[0]['name'] == native_name
            assert any(a['name'] == old_name for a in old)
            receipt = str((folder / ('author_alias_' + token + '.json')).relative_to(ROOT))
            atomic_json(ROOT / receipt, {'doi':doi, 'work_id':wid, 'retained_name':old_name,
                'native_name':native_name, 'native_receipt':str(source_path.relative_to(ROOT)),
                'raw_sha256':record['raw_sha256'], 'basis':'Unique complete name differs only by combining accents on the same identity-matched paper; no initials expansion'})
            source['authors'] = [{**matching[0], 'name':old_name}]
            source['evidence_receipt'] = receipt
            authors, count = fill_missing_links(old, source, registry, by_name, names)
            assert fill_missing_links(authors, source, registry, by_name, names)[1] == 0
            checked.append(wid)
            if not count:
                continue
            before = dict(paper)
            paper['authors_json'] = json.dumps(authors, ensure_ascii=False)
            values = metrics(paper)
            for metric, value in values.items():
                if 'after_' + metric in paper:
                    paper['after_' + metric] = value
            for field, value in paper.items():
                if value != before[field]:
                    db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                        (wid, field, json.dumps(before[field]), json.dumps(value), receipt,
                         'Reviewed unique same-paper full-name accent alias; only missing primary affiliation links filled'))
            db.execute('UPDATE papers SET institution_gap=?,country_gap=? WHERE work_id=?',
                       (not values['every_author_institution'], not values['every_author_country'], wid))
            temporary = path.with_suffix('.parquet.tmp')
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary, compression='zstd')
            temporary.replace(path)
            db.commit()
            changes.append({'work_id':wid, 'authors_filled':count, 'receipt':receipt})
        db.close()
    result = {'checked_papers':len(checked), 'changes':changes, 'repeat_safe':True,
              'network_requests':0, 'university_ownership_not_asserted':True}
    with (folder / 'author_alias_events.jsonl').open('a') as stream:
        stream.write(json.dumps(result) + '\n')
    atomic_json(folder / 'author_alias_status.json', result)
    if changes:
        refresh()
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
