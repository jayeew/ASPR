"""Harvest official EDP Dublin Core batches for a fixed existing candidate scope."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import re
from collections import Counter

from lxml import etree

from . import ojs_metadata_harvest as transport
from .fig8b_qualification_build import key
from .supplement_queue import ROOT
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/edp_oai'
URL = 'http://oai.edpsciences.org/'
NS = {'o': 'http://www.openarchives.org/OAI/2.0/', 'dc': 'http://purl.org/dc/elements/1.1/'}


def title_key(text: str) -> str:
    return key(text.replace('&', ' and '))


def run() -> None:
    transport.RAW = FOLDER / 'raw'
    transport.RAW.mkdir(exist_ok=True)
    transport.FOLDER = FOLDER
    scope = json.loads((FOLDER / 'scope.json').read_text())['papers']
    receipt = json.loads((FOLDER / 'sets.json').read_text())
    tree = etree.fromstring(gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes()))
    sets = [(n.findtext('o:setSpec', namespaces=NS), n.findtext('o:setName', namespaces=NS))
            for n in tree.findall('o:ListSets/o:set', NS)]
    plans = []
    for spec, name in sets:
        targets = [p for p in scope if title_key(p['journal']) == title_key(name)]
        if targets:
            plans.append({'set': spec, 'title': name, 'targets': targets})
    covered = {p['work_id'] for plan in plans for p in plan['targets']}
    atomic_json(FOLDER / 'plan.json', {'sets': plans, 'unmatched_candidates': [p for p in scope if p['work_id'] not in covered],
                                     'qualification_unchanged': True, 'scope_not_expanded': True})
    state = {'status': 'running', 'sets_planned': len(plans), 'candidates_in_sets': len(covered), 'completed_sets': []}
    atomic_json(FOLDER / 'harvest_status.json', state)
    for plan in sorted(plans, key=lambda p: -len(p['targets'])):
        checkpoint = FOLDER / (plan['set'] + '_progress.json')
        progress = json.loads(checkpoint.read_text()) if checkpoint.exists() else {'set': plan['set'], 'pages': 0, 'matched': {}, 'tokens': []}
        if progress.get('terminal'):
            state['completed_sets'].append(plan['set'])
            continue
        targets = {p['doi'].lower(): p for p in plan['targets'] if p['doi']}
        while True:
            params = ({'verb': 'ListRecords', 'resumptionToken': progress['next_token']} if progress.get('next_token') else
                      {'verb': 'ListRecords', 'metadataPrefix': 'oai_dc', 'set': plan['set'], 'from': '2023-01-01', 'until': '2026-10-23'})
            if plan['set'] == 'aa' and progress['pages'] == 0:
                initial = json.loads((FOLDER / 'aa_dc_first.json').read_text())
                token = hashlib.sha256((URL + json.dumps(params, sort_keys=True)).encode()).hexdigest()
                atomic_json(transport.RAW / (token + '.json'), initial)
            receipt = transport.fetch(URL, params)
            if receipt.get('http_status') != 200:
                state.update(status='source_stopped_unattempted_retained', stop_receipt=receipt)
                atomic_json(FOLDER / 'harvest_status.json', state)
                return
            raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
            tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
            errors = tree.findall('o:error', NS)
            for node in tree.findall('o:ListRecords/o:record', NS):
                identifier = node.findtext('o:header/o:identifier', namespaces=NS) or ''
                doi = identifier.removeprefix('oai:edpsciences.org:dkey/').lower()
                if doi not in targets:
                    continue
                fields = {field: [''.join(element.itertext()).strip() for element in node.xpath('.//dc:' + field, namespaces=NS)]
                          for field in ['title', 'creator', 'subject', 'date', 'publisher', 'source', 'identifier', 'type', 'language', 'description']}
                progress['matched'][doi] = {'candidate': targets[doi], 'fields': fields, 'oai_identifier': identifier,
                                            'raw_receipt': receipt['raw_receipt'], 'raw_sha256': hashlib.sha256(raw).hexdigest(),
                                            'set': plan['set'], 'set_title': plan['title'],
                                            'status': 'publisher_metadata_cached_identity_type_date_review_required',
                                            'limitation': 'Dublin Core Text is not original research; OAI datestamp is not publication date'}
            progress['pages'] += 1
            next_token = tree.findtext('o:ListRecords/o:resumptionToken', namespaces=NS) or ''
            terminal = bool(errors) or not next_token or next_token in progress['tokens'] or set(progress['matched']) >= set(targets)
            progress.update(next_token=next_token, terminal=terminal)
            if terminal:
                progress['terminal_reason'] = ('oai_error:' + ','.join(e.get('code', '') for e in errors) if errors else
                                               'all_target_dois_returned' if set(progress['matched']) >= set(targets) else
                                               'repeated_token' if next_token in progress['tokens'] and next_token else 'provider_page_sequence_exhausted')
            progress['tokens'].append(next_token)
            atomic_json(checkpoint, progress)
            state.update(current_set=plan['set'], current_pages=progress['pages'], current_matched=len(progress['matched']))
            atomic_json(FOLDER / 'harvest_status.json', state)
            if terminal:
                state['completed_sets'].append(plan['set'])
                break
    state.update(status='fixed_set_harvest_complete_review_required')
    atomic_json(FOLDER / 'harvest_status.json', state)


if __name__ == '__main__':
    with (FOLDER / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
