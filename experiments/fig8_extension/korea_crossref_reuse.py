"""Replay two reviewed signed-location rules against the remaining cached deposit."""
from __future__ import annotations

import fcntl
import gzip
import json
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run() -> dict:
    folder = ROOT / 'supplement/korea_exact_locations'
    rule_path = 'supplement/korea_exact_locations/rules.json'
    rules = json.loads((ROOT / rule_path).read_text())
    addresses = {r['exact_address'] for r in rules['sources']}
    receipt = 'fields/crossref_raw/b2c9ad974876ab574aa022e030da32845aad0b4522f35bb64ecdeb2ad77a2b60.json.gz'
    doi = '10.1103/physrevlett.132.021803'
    raw = json.loads(gzip.decompress((ROOT / receipt).read_bytes()))
    items = [i for i in raw['response']['message']['items'] if i['DOI'].lower() == doi]
    assert len(items) == 1
    item = items[0]
    names = Counter(' '.join(filter(None, [a.get('given'), a.get('family')])) for a in item['author'])
    links = {( ' '.join(filter(None, [a.get('given'), a.get('family')])), f['name'])
             for a in item['author'] for f in a.get('affiliation', [])}
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ROOT / 'fields/papers/physics_2024_01.parquet'
        table = pq.read_table(path)
        rows = table.to_pylist()
        paper = next(r for r in rows if r['doi'] == doi)
        journals = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT / 'fields/journal_registry.csv')}
        assert signature_match(paper, {'title': item['title'], 'issns': item['ISSN'], 'journal_titles': item['container-title']}, journals[paper['source_id']])
        before = paper['authors_json']
        authors = json.loads(before)
        current_names = Counter(a['name'] for a in authors)
        filled = 0
        for author in authors:
            if author.get('nature_v1_slots') is not None or author.get('publisher_affiliation_source') != receipt:
                continue
            if names[author['name']] != 1 or current_names[author['name']] != 1:
                continue
            for slot in author.get('publisher_affiliation_slots', []):
                if slot['text'] in addresses and (author['name'], slot['text']) in links and not slot.get('country_codes'):
                    slot['country_codes'] = ['KR']
                    slot['country_evidence'] = {'rule': rule_path, 'author_address_receipt': receipt,
                                                'match': 'unique_exact_author_name_and_original_deposited_address'}
                    filled += 1
        original = json.loads(before)
        for old, new in zip(original, authors, strict=True):
            for prior, after in zip(old.get('publisher_affiliation_slots', []), new.get('publisher_affiliation_slots', []), strict=True):
                check = dict(after)
                if prior != after:
                    assert not prior.get('country_codes') and after['country_codes'] == ['KR']
                    check['country_codes'] = prior.get('country_codes', [])
                    check.pop('country_evidence')
                    assert check == prior
            new_check = dict(new)
            if 'publisher_affiliation_slots' in old:
                new_check['publisher_affiliation_slots'] = old['publisher_affiliation_slots']
            assert new_check == old
        if filled:
            paper['authors_json'] = json.dumps(authors, ensure_ascii=False)
            db = connect()
            db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                       (paper['work_id'], 'authors_json', before, paper['authors_json'], rule_path,
                        'Exact cached Crossref author-address link and reviewed signed-city rule; only empty slot country filled'))
            temporary = path.with_suffix('.parquet.tmp')
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary, compression='zstd')
            temporary.replace(path)
            db.commit()
            db.close()
        result = {'doi': doi, 'empty_country_slots_filled': filled, 'network_requests': 0,
                  'only_reviewed_empty_slot_countries_changed': True, 'source': receipt,
                  'author_countries_and_structure_unchanged': True}
        atomic_json(folder / 'crossref_apply_status.json', result)
        return result


if __name__ == '__main__':
    print(json.dumps(run()))
