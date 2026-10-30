"""Review fixed unmatched-journal EDP responses without guessing journal aliases."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import sqlite3
from collections import Counter

from lxml import etree

from .edp_oai_harvest import FOLDER, NS
from .edp_oai_review import dc_fields
from .fig8b_qualification_build import key
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json


def record_reviews(records: list[dict]) -> int:
    source = 'edp_unmapped_native_identity_review'
    added = 0
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        for row in records:
            for tid, in db.execute("SELECT task_id FROM tasks WHERE kind='qualification' AND entity_key=?", (row['doi'],)).fetchall():
                receipt = row['raw_receipt'] or 'supplement/edp_oai/unmapped_record_attempts.json'
                if db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=?', (tid, source, receipt)).fetchone():
                    continue
                detail = {**row, 'review_receipt': 'supplement/edp_oai/unmapped_identity_review.json'}
                db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                           (tid, source, row['status'], receipt, json.dumps(detail, ensure_ascii=False)))
                db.execute('UPDATE tasks SET attempt_n=attempt_n+1,updated_at=CURRENT_TIMESTAMP WHERE task_id=?', (tid,))
                added += 1
        db.commit()
        db.close()
    return added


def run() -> dict:
    with (FOLDER / 'collector.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        attempts = json.loads((FOLDER / 'unmapped_record_attempts.json').read_text())
        db = sqlite3.connect(f'file:{ROOT}/supplement/tasks.sqlite?mode=ro', uri=True)
        records = []
        for doi, item in attempts.items():
            record = {'doi': doi, 'candidate': item['candidate'], 'source_outcome': item['status'],
                      'raw_receipt': item['receipt'].get('raw_receipt'), 'journal_mapping_confirmed': False,
                      'qualification_unchanged': True, 'final_task_closure': False}
            if item['status'] != 'native_record_returned_identity_and_journal_review_required':
                record.update(status=item['status'], errors=item.get('errors', []))
            else:
                raw = gzip.decompress((ROOT / record['raw_receipt']).read_bytes())
                tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
                nodes = tree.findall('o:GetRecord/o:record', NS)
                assert len(nodes) == 1
                assert nodes[0].findtext('o:header/o:identifier', namespaces=NS) == 'oai:edpsciences.org:dkey/' + doi
                fields = dc_fields(nodes[0])
                paper = db.execute('SELECT doi,title FROM papers WHERE work_id=?', (item['candidate']['work_id'],)).fetchone()
                matched = bool(paper and paper[0] == doi and key(paper[1]) in {key(t) for t in fields['title']})
                record.update(status='doi_title_matched_journal_type_date_unresolved' if matched else 'title_identity_unresolved',
                              doi_title_matched=matched, fields=fields,
                              native_sets=nodes[0].xpath('./o:header/o:setSpec/text()', namespaces=NS),
                              raw_sha256=hashlib.sha256(raw).hexdigest(),
                              limitation='Exact native DOI is not alone sufficient. Truncated titles remain unresolved; journal alias unconfirmed; generic DC type/date not research/online evidence.')
            records.append(record)
        db.close()
        result = {'records': records, 'summary': dict(Counter(r['status'] for r in records)),
                  'network_requests': 0, 'qualification_changes': 0, 'final_task_closure': False}
        atomic_json(FOLDER / 'unmapped_identity_review.json', result)
        added = record_reviews(records)
        atomic_json(FOLDER / 'unmapped_review_ledger_status.json', {'events_added': added, 'reviewed_dois': len(records),
                                                                'qualification_changes': 0, 'network_requests': 0})
        print(json.dumps(result['summary']), flush=True)
        return result


if __name__ == '__main__':
    run()
