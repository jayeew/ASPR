"""Reparse lossless EDP batches and review exact identities without HTTP requests."""
from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from collections import Counter

from lxml import etree

from .edp_oai_harvest import FOLDER, NS, title_key
from .fig8b_qualification_build import key
from .supplement_queue import ROOT
from .supplement_sources import atomic_json

FIELDS = ('title', 'creator', 'subject', 'date', 'publisher', 'source',
          'identifier', 'type', 'language', 'description')


def dc_fields(node: etree._Element) -> dict[str, list[str]]:
    """Preserve inline mathematical/italic text and its following tail text."""
    return {name: [''.join(element.itertext()).strip()
                   for element in node.xpath('.//dc:' + name, namespaces=NS)]
            for name in FIELDS}


def run() -> dict:
    records = [record for path in sorted(FOLDER.glob('*_progress.json'))
               for record in json.loads(path.read_text()).get('matched', {}).values()]
    batches: dict[str, list[dict]] = {}
    for record in records:
        batches.setdefault(record['raw_receipt'], []).append(record)
    db = sqlite3.connect(f'file:{ROOT / "supplement/tasks.sqlite"}?mode=ro', uri=True)
    stats: Counter = Counter()
    output = []
    for receipt, targets in batches.items():
        raw = gzip.decompress((ROOT / receipt).read_bytes())
        digest = hashlib.sha256(raw).hexdigest()
        tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
        nodes = {node.findtext('o:header/o:identifier', namespaces=NS): node
                 for node in tree.findall('o:ListRecords/o:record', NS) + tree.findall('o:GetRecord/o:record', NS)}
        for record in targets:
            if digest != record['raw_sha256']:
                raise ValueError('EDP cached batch digest mismatch')
            node = nodes[record['oai_identifier']]
            fields = dc_fields(node)
            candidate = record['candidate']
            paper = db.execute('SELECT doi,title,journal FROM papers WHERE work_id=?',
                               (candidate['work_id'],)).fetchone()
            matched = bool(paper and paper[0] == candidate['doi'] and
                           key(paper[1]) in {key(t) for t in fields['title']} and
                           title_key(paper[2]) == title_key(record['set_title']) and
                           (tree.find('o:GetRecord', NS) is None or record['set'] in
                            node.xpath('./o:header/o:setSpec/text()', namespaces=NS)))
            subjects = {key(value) for value in fields['subject']}
            status = ('identity_conflict' if not matched else
                      'explicit_errata_subject_review' if subjects & {'errata', 'errataaddenda'}
                      else 'identity_matched_type_and_online_date_unresolved')
            stats[status] += 1
            stats['with_description'] += bool(fields['description'])
            previous_fields = {name: [value.strip() for value in record['fields'].get(name, [])]
                               for name in FIELDS}
            stats['inline_text_restored_records'] += fields != previous_fields
            output.append({**record, 'fields': fields, 'status': status,
                           'identity_rule': 'exact DOI, normalized title and official journal set title',
                           'date_limitation': 'DC date semantics not confirmed as online publication',
                           'abstract_status': 'cached_not_applied_pending_qualification',
                           'author_limitation': 'DC creators do not establish author-affiliation links'})
    db.close()
    atomic_json(FOLDER / 'cache_review.json', {'records': output, 'network_requests': 0,
                                             'qualification_unchanged': True, 'summary': dict(stats)})
    result = {'reviewed_records': len(output), 'raw_batches': len(batches), **stats}
    atomic_json(FOLDER / 'cache_review_status.json', result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    run()
