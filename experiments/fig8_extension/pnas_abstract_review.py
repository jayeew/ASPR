"""Repair a confirmed notice-as-abstract using identity-matched cached native JATS."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

from .fig8b_qualification_evidence import parse_front
from .supplement_apply import signature_match
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv

SCOPE = {
    '10.1073/pnas.2304643121': '20c309ddd386994936cd61cb74edd6ccca1f8097169879318641911fae044f0a',
    '10.1073/pnas.2219827120': '6b758b7b536582624041a310f77582b033c2dac225b54e0c2d443f6d8fe06562',
}


def run() -> dict:
    folder = ROOT / 'supplement/pnas_abstract_review'
    folder.mkdir(exist_ok=True)
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]'))
                for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    audit = []
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        for doi, token in SCOPE.items():
            wid, shard = db.execute('SELECT work_id,shard FROM papers WHERE doi=?', (doi,)).fetchone()
            path = ROOT / 'fields/papers' / shard
            table = pq.read_table(path)
            rows = table.to_pylist()
            paper = next(r for r in rows if r['work_id'] == wid)
            receipt = f'fields/qualification/supplement_raw/{token}.xml.gz'
            raw = gzip.decompress((ROOT / receipt).read_bytes())
            raw = raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw
            parsed = parse_front(raw)
            if doi not in parsed['doi'] or not signature_match(paper, parsed, journals[paper['source_id']]):
                raise ValueError('Native abstract identity mismatch')
            root = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
            nodes = root.xpath('./front/article-meta/abstract[not(@abstract-type)]')
            if len(nodes) != 1:
                raise ValueError('Ambiguous main abstract; no automatic replacement')
            value = ' '.join(''.join(nodes[0].itertext()).split())
            old = paper['abstract']
            identical = ' '.join(old.split()) == value
            defect = 'this preprint be removed from Research Square' in old
            entry = {'work_id': wid, 'doi': doi, 'receipt': receipt,
                     'raw_sha256': hashlib.sha256(raw).hexdigest(),
                     'abstract_sha256': hashlib.sha256(value.encode()).hexdigest(),
                     'main_abstract_only': True, 'network_requests': 0,
                     'status': 'native_text_confirmed' if identical else 'repaired_notice_as_abstract' if defect else 'conflict_retained'}
            if defect:
                updates = {'abstract': value, 'abstract_source': 'matched_publisher:' + receipt}
                for field, after in updates.items():
                    db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                               (wid, field, json.dumps(paper[field]), json.dumps(after), receipt,
                                'Identity-matched native main abstract replaces unrelated preprint removal notice; executive summary excluded'))
                    paper[field] = after
                temporary = path.with_suffix('.parquet.tmp')
                pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary, compression='zstd')
                temporary.replace(path)
                db.commit()
            audit.append(entry)
        db.close()
        with (folder / 'audit.jsonl').open('a') as stream:
            for item in audit:
                stream.write(json.dumps(item) + '\n')
        atomic_json(folder / 'current.json', {'papers': audit, 'network_requests': 0,
                    'limitation': 'Short native abstract remains separately source-reviewed; generic length flag not globally relaxed.'})
    from .derived_refresh import refresh
    refresh()
    return {'assessed': len(audit), 'repaired': sum(r['status'] == 'repaired_notice_as_abstract' for r in audit)}


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
