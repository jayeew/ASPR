"""Record field-specific native abstract absence without closing other metadata gaps."""
from __future__ import annotations

import fcntl
import gzip
import json
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

from .supplement_queue import ROOT, connect, task
from .supplement_sources import atomic_json


def run() -> None:
    folder = ROOT / 'supplement/native_abstract_availability'
    folder.mkdir(exist_ok=True)
    policy = {'journal': 'Genes & Diseases', 'format': 'Rapid Communication',
              'source_url': 'https://www.keaipublishing.com/en/journals/genes-and-diseases/guide-for-authors',
              'locator': 'Types of Articles > Rapid Communication > Abstract',
              'excerpt': 'Abstract: None', 'retrieved_at': '2026-10-25',
              'scope': 'Current guide corroborates individual historical native JATS absence; no blanket historical assumption'}
    atomic_json(folder / 'policy.json', policy)
    records = []; counts = Counter()
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect()
        selected = dict((doi, (wid, shard)) for wid, doi, shard in db.execute(
            "SELECT work_id,doi,shard FROM papers WHERE source_id='https://openalex.org/S2764642421' AND status='included_verified_research'"))
        papers = {}
        for shard in {s for _, s in selected.values()}:
            for row in pq.read_table(ROOT / 'fields/papers' / shard,
                                     columns=['work_id', 'title', 'abstract', 'abstract_source']).to_pylist():
                papers[row['work_id']] = row
        repairs = {}
        for path in sorted((ROOT / 'supplement/named_native_formats/staged').glob('*.json')):
            data = json.loads(path.read_text()); doi = data['doi']; parsed = data['parsed']
            if doi not in selected or 'Rapid Communication' not in parsed.get('categories', []):
                continue
            wid, _ = selected[doi]
            raw_path = ROOT / 'fields/qualification' / data['raw_receipt']
            raw = gzip.decompress(raw_path.read_bytes())
            if raw.rstrip().endswith(b'</front>'):
                raw += b'</article>'
            xml = etree.fromstring(raw, parser=etree.XMLParser(resolve_entities=False, no_network=True))
            native_dois = [str(v).strip().lower() for v in xml.xpath('./front/article-meta/article-id[@pub-id-type="doi"]/text()')]
            if doi not in native_dois:
                raise ValueError(f'Native DOI mismatch: {doi}')
            nodes = xml.xpath('./front/article-meta/abstract | ./front/article-meta/trans-abstract')
            paper = papers[wid]
            text = (paper['abstract'] or '').strip()
            invalid = (text in {'No abstract available', 'International audience'}
                       or (doi == '10.1016/j.gendis.2025.101543' and text.startswith('This work was supported by grants'))
                       or (doi == '10.1016/j.gendis.2024.101238' and text.startswith('Voß S, Walhorn V, Holler S, et al.'))
                       or (text and text.casefold() == (paper['title'] or '').strip().casefold()))
            if not nodes and invalid and paper['abstract_source'] == 'openalex_inverted_index':
                repairs[wid] = {'abstract': '', 'abstract_source': ''}
            status = ('invalid_secondary_text_removed' if wid in repairs else
                      'native_abstract_present_review' if nodes else
                      'existing_text_without_native_abstract_review' if paper['abstract'] else
                      'native_source_does_not_provide_abstract')
            record = {'work_id': wid, 'doi': doi, 'status': status,
                      'raw_receipt': str(raw_path.relative_to(ROOT)), 'parsed_receipt': str(path.relative_to(ROOT)),
                      'policy_receipt': 'supplement/native_abstract_availability/policy.json',
                      'native_abstract_nodes': len(nodes), 'existing_abstract_source': paper['abstract_source'],
                      'existing_abstract_preserved': bool(paper['abstract']) and wid not in repairs,
                      'invalid_text_retained_in_change_ledger': wid in repairs, 'qualified_denominator_preserved': True,
                      'next_action': 'No repeat fetch of this native source without new evidence; other metadata gaps remain open' if not nodes and not paper['abstract'] else 'Review existing text provenance against native source; do not delete automatically'}
            records.append(record); counts[status] += 1
            receipt = str((folder / 'review.json').relative_to(ROOT))
            ident = task(db, 'abstract_availability', wid, 'native_jats_and_journal_guide', record, wid,
                         status='source_unavailable' if status == 'native_source_does_not_provide_abstract' else 'pending')
            prior = db.execute('SELECT 1 FROM attempts WHERE task_id=? AND source=? AND receipt=? AND status=?',
                               (ident, 'native_abstract_availability', receipt, status)).fetchone()
            if not prior:
                db.execute('INSERT INTO attempts(task_id,source,status,receipt,detail) VALUES(?,?,?,?,?)',
                           (ident, 'native_abstract_availability', status, receipt, json.dumps(record)))
        for shard in {s for w, s in selected.values() if w in repairs}:
            target = ROOT / 'fields/papers' / shard
            table = pq.read_table(target); rows = table.to_pylist()
            for paper in rows:
                wid = paper['work_id']
                if wid not in repairs:
                    continue
                for field, after in repairs[wid].items():
                    db.execute('INSERT INTO changes(work_id,field,before_value,after_value,receipt,reason) VALUES(?,?,?,?,?,?)',
                               (wid, field, json.dumps(paper[field]), json.dumps(after), receipt,
                                'Invalid secondary placeholder, funding, citation or title text is not an abstract; native JATS has no abstract'))
                    paper[field] = after
                if 'after_abstract' in paper: paper['after_abstract'] = 0
                db.execute('UPDATE papers SET abstract_gap=1 WHERE work_id=?', (wid,))
            temporary = target.with_suffix('.parquet.tmp')
            pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary, compression='zstd')
            temporary.replace(target)
        atomic_json(folder / 'review.json', {'records': records, 'outcomes': dict(counts),
                    'network_article_requests': 0, 'metadata_tasks_closed': 0, 'abstract_values_modified': len(repairs)})
        db.commit(); db.close()
    if repairs:
        from .derived_refresh import refresh
        refresh()
    print(json.dumps(dict(counts)), flush=True)


if __name__ == '__main__':
    run()
