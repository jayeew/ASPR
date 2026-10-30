"""Resolve fixed DOI identifiers absent from the completed Pensoft bulk traversal."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

from lxml import etree

from . import ojs_metadata_harvest as transport
from .pensoft_bulk import FOLDER, URL, NS, extract
from .pensoft_bulk_apply import run as apply
from .supplement_sources import atomic_json
from .supplement_queue import ROOT


def fetch(doi: str) -> dict:
    target = FOLDER / 'missing_identifiers' / (hashlib.sha256(doi.encode()).hexdigest()+'.json')
    if target.exists():
        return json.loads(target.read_text())
    receipt = transport.fetch(URL, {'verb': 'GetRecord', 'metadataPrefix': 'mods', 'identifier': doi})
    result = {'doi': doi, 'receipt': receipt, 'status': 'source_unavailable'}
    if receipt.get('http_status') == 200:
        tree = etree.fromstring(gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes()), etree.XMLParser(resolve_entities=False, no_network=True))
        nodes = tree.findall('o:GetRecord/o:record', NS)
        result.update(status='identifier_absent_or_nonunique', protocol_errors=[dict(e.attrib) for e in tree.findall('o:error', NS)])
        if len(nodes) == 1:
            parsed = extract(nodes[0])
            if [d.lower().strip() for d in parsed.get('doi', [])] == [doi]:
                atomic_json(FOLDER / 'matched' / target.name, {'doi': doi, 'parsed': parsed, 'source': 'official_pensoft_mods',
                            'page_receipt': receipt, 'record_identifier': doi, 'qualification_not_implied': True})
                result['status'] = 'native_metadata_matched'
    atomic_json(target, result)
    return result


def run() -> None:
    folder = FOLDER / 'missing_identifiers'
    folder.mkdir(exist_ok=True)
    with (FOLDER / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transport.FOLDER, transport.RAW = FOLDER, FOLDER / 'raw'
        scope = folder / 'scope.json'
        if not scope.exists():
            known = {json.loads(p.read_text())['doi'] for p in (FOLDER / 'matched').glob('*.json')}
            missing = sorted({r['doi'] for r in json.loads((FOLDER / 'scope.json').read_text())['papers'] if r['doi']} - known)
            atomic_json(scope, {'dois': missing, 'reason': 'Fixed candidates absent after complete bulk traversal; exact OAI DOI identifiers, no date filter'})
        missing = json.loads(scope.read_text())['dois']
        results = []
        blocked = False
        with ThreadPoolExecutor(max_workers=4) as pool:
            for start in range(0, len(missing), 4):
                group = list(pool.map(fetch, missing[start:start+4]))
                results.extend(group)
                if any(r['receipt'].get('http_status') in {401,403,429} for r in group):
                    blocked = True
                    break
        applied = apply()
        atomic_json(folder / 'status.json', {'status': 'source_restriction_retained' if blocked else 'fixed_identifier_queue_processed',
                    'fixed_dois': len(missing), 'attempted_dois': len(results),
                    'matched': sum(r['status']=='native_metadata_matched' for r in results), 'application': applied})


if __name__ == '__main__':
    run()
