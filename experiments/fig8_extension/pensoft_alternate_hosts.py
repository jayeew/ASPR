"""Discover same-host native links from official deposited landing pages."""
from __future__ import annotations

import fcntl
import gzip
import json
import time
from urllib.parse import urljoin, urlparse

from lxml import html

from . import ojs_metadata_harvest as transport
from . import pensoft_native as native
from .pensoft_bulk import FOLDER as BULK
from .publisher_delta_apply import apply_available
from .springer_author_cache import run as apply_authors
from .batch_memory import release_batch_memory
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json

ORIGINAL = BULK / 'native'
FOLDER = BULK / 'alternate_hosts'


def resolve(record: dict) -> dict:
    source = json.loads((ROOT / record['mods_receipt']).read_text())
    urls = source['parsed']['landing_urls']
    output = {'doi': record['doi'], 'original_result': record, 'status': 'nonunique_landing_url'}
    if len(urls) != 1:
        return output
    url = urls[0]
    receipt = transport.fetch(url, {})
    output.update(landing_receipt=receipt, status='landing_unavailable')
    if receipt.get('http_status') != 200:
        return output
    tree = html.fromstring(gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes()))
    dois = tree.xpath('//meta[@name="citation_doi"]/@content')
    xml = tree.xpath('//meta[@name="citation_xml_url"]/@content')
    output['status'] = 'landing_doi_or_native_link_unconfirmed'
    if [d.lower().strip() for d in dois] != [record['doi']] or len(xml) != 1:
        return output
    xml_url = urljoin(url, xml[0])
    if urlparse(xml_url).scheme != 'https' or urlparse(xml_url).hostname != urlparse(url).hostname:
        return output
    result = native.fetch({**source, '_receipt': record['mods_receipt'], 'validated_xml_url': xml_url})
    return {**output, 'status': result['status'], 'native_result': result}


def run() -> None:
    for path in [FOLDER, FOLDER / 'results', FOLDER / 'staged', FOLDER / 'reviews']:
        path.mkdir(exist_ok=True)
    with (FOLDER / 'collector.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = FOLDER / 'status.json'
        if status_path.exists() and json.loads(status_path.read_text()).get('status') == 'source_restriction_retained':
            return
        native.FOLDER = FOLDER
        transport.FOLDER, transport.RAW = BULK, BULK / 'raw'
        while True:
            selected = []
            for path in (ORIGINAL / 'results').glob('*.json'):
                if (FOLDER / 'reviews' / path.name).exists():
                    continue
                record = json.loads(path.read_text())
                if record['status'] == 'no_unique_validated_native_route':
                    selected.append((path, record))
            jobs = []
            blocked = False
            for path, record in selected:
                result = resolve(record)
                atomic_json(FOLDER / 'reviews' / path.name, result)
                nr = result.get('native_result', {})
                if nr.get('staged_receipt'):
                    jobs.append((record['doi'], nr['staged_receipt']))
                if any(r.get('http_status') in {401,403,429} for r in [result.get('landing_receipt', {}), nr.get('receipt', {})]):
                    blocked = True
                    break
            applied = apply_available(supplied_jobs=jobs) if jobs else {}
            db = connect()
            aj = [(wid, shard, receipt) for doi, receipt in jobs for wid, shard in db.execute("SELECT work_id,shard FROM papers WHERE doi=? AND status='included_verified_research'", (doi,))]
            db.close()
            authors = apply_authors(selected_jobs=aj, workers=4, fill_abstract=True) if aj else {}
            parent = ORIGINAL / 'status.json'
            terminal = parent.exists() and json.loads(parent.read_text())['status'] != 'running'
            atomic_json(FOLDER / 'status.json', {'status': 'source_restriction_retained' if blocked else ('available_alternate_hosts_reviewed' if terminal else 'running'), 'reviewed': len(list((FOLDER / 'reviews').glob('*.json'))), 'application': applied, 'authors': authors})
            if terminal or blocked:
                return
            release_batch_memory()
            time.sleep(30)


if __name__ == '__main__':
    run()
