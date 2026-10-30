"""Finish fixed EDP batch omissions by exact native identifier, preserving evidence."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json

from lxml import etree

from . import ojs_metadata_harvest as transport
from .edp_oai_harvest import FOLDER, NS, URL
from .edp_oai_review import dc_fields
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def run() -> None:
    with (FOLDER / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transport.FOLDER = FOLDER
        transport.RAW = FOLDER / 'raw'
        state = json.loads((FOLDER / 'harvest_status.json').read_text())
        if state['status'] == 'source_stopped_unattempted_retained':
            return
        attempts_path = FOLDER / 'exact_tail_attempts.json'
        attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
        attempted = {a['doi'] for a in attempts}
        stopped = False
        for plan in json.loads((FOLDER / 'plan.json').read_text())['sets']:
            path = FOLDER / (plan['set'] + '_progress.json')
            progress = json.loads(path.read_text()) if path.exists() else {'set': plan['set'], 'pages': 0, 'matched': {}, 'tokens': []}
            targets = [p for p in plan['targets'] if p['doi'] not in progress['matched'] and p['doi'] not in attempted]
            if not targets:
                continue
            for candidate in targets:
                identifier = 'oai:edpsciences.org:dkey/' + candidate['doi']
                receipt = transport.fetch(URL, {'verb': 'GetRecord', 'metadataPrefix': 'oai_dc', 'identifier': identifier})
                item = {'doi': candidate['doi'], 'set': plan['set'], 'receipt': receipt}
                attempts.append(item)
                if receipt.get('http_status') != 200:
                    item['status'] = 'request_failed'; stopped = True
                    break
                raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
                tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
                nodes = tree.findall('o:GetRecord/o:record', NS)
                if (len(nodes) != 1 or nodes[0].findtext('o:header/o:identifier', namespaces=NS) != identifier
                        or plan['set'] not in nodes[0].xpath('./o:header/o:setSpec/text()', namespaces=NS)):
                    item.update(status='native_identifier_not_returned', errors=[n.get('code') for n in tree.findall('o:error', NS)])
                    continue
                node = nodes[0]
                progress['matched'][candidate['doi']] = {
                    'candidate': candidate, 'fields': dc_fields(node), 'oai_identifier': identifier,
                    'raw_receipt': receipt['raw_receipt'], 'raw_sha256': hashlib.sha256(raw).hexdigest(),
                    'set': plan['set'], 'set_title': plan['title'],
                    'status': 'publisher_metadata_cached_identity_type_date_review_required',
                    'limitation': 'Generic DC type and date do not establish original research or electronic date'}
                item['status'] = 'native_identifier_returned_review_required'
            progress.update(terminal=not stopped, terminal_reason='exact_remaining_identifiers_processed' if not stopped else 'source_request_failed')
            atomic_json(path, progress)
            if stopped:
                break
            if plan['set'] not in state['completed_sets']:
                state['completed_sets'].append(plan['set'])
        atomic_json(FOLDER / 'exact_tail_attempts.json', attempts)
        state.update(status='source_stopped_unattempted_retained' if stopped else 'fixed_set_harvest_complete_review_required',
                     tail_strategy='Exact GetRecord for fixed-scope DOIs omitted from batches; prior responses never repeated')
        atomic_json(FOLDER / 'harvest_status.json', state)
        print(json.dumps({'requests_reviewed': len(attempts), 'source_stopped': stopped}), flush=True)


if __name__ == '__main__':
    run()
