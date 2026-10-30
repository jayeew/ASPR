"""Fetch exact native identifiers for the fixed unmatched EDP candidate remainder."""
from __future__ import annotations

import fcntl
import gzip
import json
from collections import Counter

from lxml import etree

from . import ojs_metadata_harvest as transport
from .edp_oai_harvest import FOLDER, NS, URL
from .edp_oai_review import dc_fields
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> None:
    with (FOLDER / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transport.FOLDER, transport.RAW = FOLDER, FOLDER / 'raw'
        path = FOLDER / 'unmapped_record_attempts.json'
        attempts = json.loads(path.read_text()) if path.exists() else {}
        scope = json.loads((FOLDER / 'plan.json').read_text())['unmatched_candidates']
        for candidate in scope:
            doi = candidate['doi']
            if doi in attempts:
                if attempts[doi]['status'] == 'source_request_failed':
                    break
                continue
            identifier = 'oai:edpsciences.org:dkey/' + doi
            receipt = transport.fetch(URL, {'verb': 'GetRecord', 'metadataPrefix': 'oai_dc', 'identifier': identifier})
            item = {'candidate': candidate, 'receipt': receipt, 'qualification_unchanged': True,
                    'journal_mapping_confirmed': False}
            if receipt.get('http_status') != 200:
                item['status'] = 'source_request_failed'
            else:
                raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
                tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
                nodes = tree.findall('o:GetRecord/o:record', NS)
                item['errors'] = [n.get('code') for n in tree.findall('o:error', NS)]
                if len(nodes) == 1 and nodes[0].findtext('o:header/o:identifier', namespaces=NS) == identifier:
                    item.update(status='native_record_returned_identity_and_journal_review_required',
                                fields=dc_fields(nodes[0]), native_sets=nodes[0].xpath('./o:header/o:setSpec/text()', namespaces=NS))
                else:
                    item['status'] = 'native_identifier_not_returned'
            attempts[doi] = item
            atomic_json(path, attempts)
            atomic_json(FOLDER / 'unmapped_status.json', {
                'scope_candidates': len(scope), 'attempted_dois': len(attempts),
                'outcomes': dict(Counter(v['status'] for v in attempts.values())),
                'qualification_changes': 0, 'final_task_closure': False})
            if item['status'] == 'source_request_failed':
                break
        print(json.dumps({'attempted_dois': len(attempts)}), flush=True)


if __name__ == '__main__':
    run()
