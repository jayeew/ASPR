"""Harvest official MODS in resumable pages, retaining only fixed candidate matches."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from datetime import datetime, timezone

from lxml import etree

from . import ojs_metadata_harvest as transport
from .supplement_queue import ROOT
from .supplement_sources import atomic_json

FOLDER = ROOT / 'supplement/pensoft_bulk'
URL = 'https://oai.pensoft.net/oai.php'
NS = {'o': 'http://www.openarchives.org/OAI/2.0/', 'm': 'http://www.loc.gov/mods/v3'}


def extract(record: etree._Element) -> dict:
    node = record.find('o:metadata/m:mods', NS)
    if node is None:
        return {}
    def values(path: str) -> list[str]:
        return [' '.join(''.join(n.itertext()).split()) for n in node.findall(path, NS)]
    authors = []
    for name in node.findall('m:name', NS):
        authors.append({'name': name.findtext('m:namePart', default='', namespaces=NS),
                        'affiliations': [' '.join(''.join(n.itertext()).split()) for n in name.findall('m:affiliation', NS)],
                        'roles': name.xpath('./m:role/m:roleTerm/text()', namespaces=NS)})
    return {'doi': values('m:identifier[@type="doi"]'), 'title': values('m:titleInfo/m:title'),
            'journal_titles': values('m:relatedItem[@type="host"]/m:titleInfo/m:title'),
            'issns': values('m:relatedItem[@type="host"]/m:identifier[@type="eissn"]'),
            'article_types': values('m:genre'), 'abstract': '\n\n'.join(values('m:abstract')),
            'authors': authors, 'date_available': values('m:extension/m:dateAvailable'),
            'date_issued': values('m:originInfo/m:dateIssued'),
            'landing_urls': values('m:identifier[@type="uri"]'),
            'online_date': '', 'date_limitation': 'MODS dateAvailable/dateIssued semantics require publisher validation'}


def run() -> None:
    with (FOLDER / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transport.FOLDER, transport.RAW = FOLDER, FOLDER / 'raw'
        matches = FOLDER / 'matched'
        matches.mkdir(exist_ok=True)
        targets = {p['doi'] for p in json.loads((FOLDER / 'scope.json').read_text())['papers'] if p['doi']}
        checkpoint = FOLDER / 'checkpoint.json'
        state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {
            'params': {'verb': 'ListRecords', 'metadataPrefix': 'mods', 'from': '2023-01-01'},
            'pages': 0, 'seen_tokens': [], 'status': 'running'}
        if state['status'] != 'running':
            return
        while state['params']:
            receipt = transport.fetch(URL, state['params'])
            if receipt.get('http_status') != 200:
                state.update(status='source_failed_unattempted_retained', failure_receipt=receipt)
                break
            raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
            try:
                tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
            except etree.XMLSyntaxError as exc:
                state.update(status='invalid_xml_unattempted_retained', error=str(exc), failure_receipt=receipt)
                break
            errors = tree.findall('o:error', NS)
            if errors:
                state.update(status='protocol_error_unattempted_retained', errors=[dict(e.attrib) for e in errors], failure_receipt=receipt)
                break
            for record in tree.findall('o:ListRecords/o:record', NS):
                parsed = extract(record)
                dois = [d.lower().strip() for d in parsed.get('doi', [])]
                if len(dois) != 1 or dois[0] not in targets:
                    continue
                doi = dois[0]
                token = hashlib.sha256(doi.encode()).hexdigest()
                atomic_json(matches / (token + '.json'), {'doi': doi, 'parsed': parsed,
                    'source': 'official_pensoft_mods', 'page_receipt': receipt,
                    'record_identifier': record.findtext('o:header/o:identifier', namespaces=NS),
                    'qualification_not_implied': True})
            token = tree.findtext('o:ListRecords/o:resumptionToken', default='', namespaces=NS).strip()
            state['pages'] += 1
            state['matched_dois'] = len(list(matches.glob('*.json')))
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            if token and token in state['seen_tokens']:
                state.update(status='repeated_token_unattempted_retained');break
            if token:
                state['seen_tokens'].append(token)
            state['params'] = {'verb': 'ListRecords', 'resumptionToken': token} if token else {}
            if not token or state['matched_dois'] == len(targets):
                state['status'] = 'fixed_scope_bulk_acquired_review_required'
            atomic_json(checkpoint, state)
            atomic_json(FOLDER / 'status.json', {k:v for k,v in state.items() if k not in {'seen_tokens', 'params'}})
            if state['status'] != 'running':
                break
        atomic_json(checkpoint, state)
        atomic_json(FOLDER / 'status.json', {k:v for k,v in state.items() if k not in {'seen_tokens', 'params'}})


if __name__ == '__main__':
    run()
