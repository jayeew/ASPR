"""Reuse rich batch pages, then fetch only fixed EDP candidates in supported sets."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json

from lxml import etree

from . import ojs_metadata_harvest as transport
from .edp_oai_harvest import FOLDER, NS, URL
from .supplement_oai import parse_oai
from .supplement_queue import ROOT
from .supplement_sources import atomic_json


def extract(raw: bytes, doi: str) -> dict | None:
    tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
    records = tree.findall('o:ListRecords/o:record', NS) + tree.findall('o:GetRecord/o:record', NS)
    selected = [r for r in records if (r.findtext('o:header/o:identifier', namespaces=NS) or '').lower()
                == 'oai:edpsciences.org:dkey/' + doi]
    if len(selected) != 1:
        return None
    parsed = parse_oai(etree.tostring(selected[0]))
    return parsed if parsed.get('doi') == [doi] else None


def run() -> None:
    folder = FOLDER / 'rich_set_review'
    with (folder / 'candidate_collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (folder / 'collector.lock').open('a') as review:
            fcntl.flock(review, fcntl.LOCK_EX)
            reviews = json.loads((folder / 'results.json').read_text())
        transport.FOLDER, transport.RAW = FOLDER, FOLDER / 'raw'
        plans = json.loads((FOLDER / 'plan.json').read_text())['sets']
        out = folder / 'candidates'
        out.mkdir(exist_ok=True)
        counts = {'processed': 0, 'matched': 0, 'cache_batch_reused': 0}
        for plan in plans:
            source = reviews.get(plan['set'], {})
            if not source.get('records'):
                continue
            for candidate in plan['targets']:
                doi = candidate['doi'].lower()
                path = out / (hashlib.sha256(doi.encode()).hexdigest() + '.json')
                if path.exists():
                    continue
                receipt = source['receipt']
                raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
                parsed = extract(raw, doi)
                if parsed:
                    counts['cache_batch_reused'] += 1
                else:
                    receipt = transport.fetch(URL, {'verb': 'GetRecord', 'metadataPrefix': 'pmc',
                        'identifier': 'oai:edpsciences.org:dkey/' + doi})
                    if receipt.get('http_status') == 200:
                        raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
                        parsed = extract(raw, doi)
                result = {'candidate': candidate, 'set': plan['set'], 'receipt': receipt,
                          'parsed': parsed, 'status': 'matched_native_identity_review_required' if parsed else
                          'no_unique_doi_matched_native_record', 'qualification_changes': 0}
                atomic_json(path, result)
                counts['processed'] += 1
                counts['matched'] += bool(parsed)
                atomic_json(folder / 'candidate_status.json', counts)
                if receipt.get('http_status') != 200:
                    return
        print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    run()
