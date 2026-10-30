"""Finite official MDPI OAI harvest, retaining only existing candidate matches."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from pathlib import Path

from lxml import etree

from . import ojs_metadata_harvest as transport
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/mdpi_oai'
URL = 'https://oai.mdpi.com/oai/oai2.php'
NS = {'o': 'http://www.openarchives.org/OAI/2.0/', 'dc': 'http://purl.org/dc/elements/1.1/'}
PARAMS = {'verb': 'ListRecords', 'metadataPrefix': 'oai_dc', 'from': '2023-01-01', 'until': '2026-10-23'}


def extract(node: etree._Element) -> dict:
    fields = {name: [''.join(n.itertext()) for n in node.findall('.//dc:' + name, NS)]
              for name in ['title', 'creator', 'description', 'publisher', 'date', 'source', 'identifier', 'type', 'subject']}
    dois = [v.split('doi.org/', 1)[1].lower().rstrip('/') for v in fields['identifier'] if 'doi.org/' in v]
    return {'dois': dois, 'fields': fields,
            'oai_identifier': node.findtext('o:header/o:identifier', namespaces=NS),
            'native_type_sets': node.findall('o:header/o:setSpec', NS) and [n.text for n in node.findall('o:header/o:setSpec', NS)],
            'oai_datestamp': node.findtext('o:header/o:datestamp', namespaces=NS)}


def run() -> None:
    transport.FOLDER = FOLDER
    transport.RAW = FOLDER / 'raw'
    scope_path = FOLDER / 'scope.json'
    if not scope_path.exists():
        db = connect()
        papers = [dict(zip(['work_id', 'doi', 'title', 'journal', 'source_id'], row))
                  for row in db.execute("SELECT work_id,doi,title,journal,source_id FROM papers WHERE route='mdpi' AND doi<>''")]
        db.close()
        atomic_json(scope_path, {'papers': papers, 'scope': 'Existing candidate roster only; no discovered DOI appended'})
    papers = json.loads(scope_path.read_text())['papers']
    targets = {p['doi']: p for p in papers}
    state_path = FOLDER / 'harvest_status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'status': 'running', 'pages': 0, 'records_seen': 0, 'matched_records': 0, 'scope_dois': len(targets)}
    if state['status'] != 'running':
        return
    pages = FOLDER / 'matched_pages'
    pages.mkdir(exist_ok=True)
    seen_tokens = set()
    while True:
        params = {'verb': 'ListRecords', 'resumptionToken': state['next_token']} if state.get('next_token') else state.get('initial_params', PARAMS)
        receipt = transport.fetch(URL, params)
        if receipt.get('http_status') != 200:
            state.update(status='source_stopped_unattempted_retained', failure=receipt)
            atomic_json(state_path, state)
            return
        raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
        tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
        errors = tree.findall('o:error', NS)
        nodes = tree.findall('o:ListRecords/o:record', NS)
        matches = []
        for node in nodes:
            item = extract(node)
            for doi in item['dois']:
                if doi in targets:
                    matches.append({**item, 'candidate': targets[doi], 'raw_receipt': receipt['raw_receipt'],
                                    'raw_sha256': hashlib.sha256(raw).hexdigest(),
                                    'status': 'cached_identity_type_date_review_required'})
        token_node = tree.find('o:ListRecords/o:resumptionToken', NS)
        token = token_node.text if token_node is not None else ''
        total = int(token_node.get('completeListSize', '0')) if token_node is not None else 0
        state.setdefault('initial_provider_record_count', total)
        known_dois = set(state.get('matched_dois', []))
        new_dois = {m['candidate']['doi'] for m in matches} - known_dois
        state['matched_dois'] = sorted(known_dois | new_dois)
        state['pages_without_new_match'] = 0 if new_dois else state.get('pages_without_new_match', 0) + 1
        atomic_json(pages / f'{state["pages"]:06d}.json', {'receipt': receipt, 'matches': matches})
        state.update(pages=state['pages'] + 1, records_seen=state['records_seen'] + len(nodes),
                     matched_records=state['matched_records'] + len(matches), next_token=token,
                     qualification_unchanged=True, network_route='direct_trust_env_false')
        if errors or not token or token in seen_tokens:
            state.update(status='provider_sequence_ended_review_required', terminal_reason=
                         ','.join(e.get('code', '') for e in errors) or ('repeated_token' if token else 'no_next_token'))
        elif state['records_seen'] - state.get('window_records_offset', 0) > state['initial_provider_record_count'] + 50:
            state.update(status='provider_count_boundary_review_required')
        elif state['pages_without_new_match'] >= 100:
            state.update(status='no_scope_matches_source_review_required')
        seen_tokens.add(token)
        atomic_json(state_path, state)
        if state['pages'] % 25 == 0:
            print(json.dumps(state), flush=True)
        if state['status'] != 'running':
            return


if __name__ == '__main__':
    with (FOLDER / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
