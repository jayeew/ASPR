"""Discover unlinked single-group affiliations in an existing fixed cache scope."""
from __future__ import annotations

import gzip
import json
import os
from collections import Counter
from functools import lru_cache

from lxml import etree

from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json, shared_group_affiliation


@lru_cache(maxsize=1024)
def is_native_aps_batch(receipt: str) -> bool:
    if not receipt.startswith('fields/qualification/aps_all_raw/') or not receipt.endswith('.json.gz'):
        return False
    document = json.loads(gzip.decompress((ROOT / receipt).read_bytes()))
    return (document.get('http_status') == 200
            and isinstance(document.get('response', {}).get('data'), list))


def run() -> None:
    folder = ROOT / 'supplement/group_affiliation_cache'
    folder.mkdir(exist_ok=True)
    scope_path = folder / 'scope.json'
    if not scope_path.exists():
        db = connect()
        rows = db.execute("SELECT DISTINCT a.work_id,a.receipt FROM springer_author_processed a "
                          "JOIN papers p USING(work_id) WHERE a.status='cache_processed' "
                          "AND p.status='included_verified_research' ORDER BY a.work_id,a.receipt").fetchall()
        db.close()
        atomic_json(scope_path, {'records': rows, 'scope': 'Existing qualified cached author sources only',
                                 'network_requests': 0})
    records = json.loads(scope_path.read_text())['records']
    output = folder / 'review.jsonl'
    latest = {}
    if output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            latest[(row['work_id'], row['receipt'])] = row
    reopen = {pair for pair, row in latest.items()
              if row['status'] == 'cache_review_unresolved' and row.get('error') == 'UnicodeDecodeError'
              and row['receipt'].startswith('fields/qualification/aps_all_raw/')}
    seen = set(latest) - reopen
    counts = Counter(row['status'] for pair, row in latest.items() if pair not in reopen)
    with output.open('a') as stream:
        for wid, receipt in records:
            if (wid, receipt) in seen:
                continue
            row = {'work_id': wid, 'receipt': receipt}
            try:
                native = is_native_aps_batch(receipt)
                source = {} if native else json.loads((ROOT / receipt).read_text())
                if native:
                    row.update(status='not_jats_source', source_format='native_aps_batch_json',
                               correction_of='cache_review_unresolved:UnicodeDecodeError',
                               reason='Validated gzip JSON with native authors/affiliationIds; JATS group rule not applicable')
                elif source.get('source') not in {'pmc', 'publisher_jats'}:
                    row['status'] = 'not_jats_source'
                else:
                    raw = gzip.decompress((ROOT / 'fields/qualification' / source['raw_receipt']).read_bytes())
                    if raw.rstrip().endswith(b'</front>'):
                        raw += b'</article>'
                    doc = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
                    authors = doc.xpath('./front/article-meta/contrib-group/contrib[@contrib-type="author"] | '
                                        './front/article-meta/contrib-group[@content-type="author"]/contrib[not(@contrib-type)]')
                    count = sum(shared_group_affiliation(author) is not None for author in authors)
                    row.update(status='group_affiliation_candidate' if count else 'no_shared_group_pattern',
                               declared_authors_with_group_address=count, raw_receipt=source['raw_receipt'],
                               raw_sha256=source.get('raw_sha256'))
            except (OSError, EOFError, ValueError, KeyError, etree.XMLSyntaxError) as error:
                row.update(status='cache_review_unresolved', error=type(error).__name__)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            counts[row['status']] += 1
            seen.add((wid, receipt))
            if len(seen) % 500 == 0:
                stream.flush()
                atomic_json(folder / 'status.json', {'status': 'running', 'pid': os.getpid(),
                            'reviewed': len(seen), 'fixed_records': len(records), 'counts': dict(counts),
                            'network_requests': 0, 'main_table_modified': False})
        stream.flush()
    atomic_json(folder / 'status.json', {'status': 'cache_scope_review_complete_application_required',
                'pid': os.getpid(), 'reviewed': len(seen), 'fixed_records': len(records),
                'counts': dict(counts), 'network_requests': 0, 'main_table_modified': False})
    print(dict(counts), flush=True)


if __name__ == '__main__':
    run()
