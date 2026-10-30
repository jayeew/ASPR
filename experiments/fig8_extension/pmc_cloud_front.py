"""Read complete JATS front matter using conditional public S3 byte ranges."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from lxml import etree

from .pmc_cloud_inventory import BASE, FOLDER
from .pmc_cloud_versions import LOCAL, STOP, fetch_metadata
from .supplement_queue import ROOT
from .supplement_sources import atomic_bytes, atomic_json, jats_metadata

LIMIT = 65536


def range_matches(raw_size: int, receipt: dict, expected: str) -> bool:
    match = re.fullmatch(r'bytes 0-(\d+)/(\d+)', receipt.get('content_range', ''))
    return bool(match and int(match[1]) + 1 == raw_size <= int(match[2])
                and receipt.get('etag', '').strip('"') == expected
                and receipt.get('if_match') == '"' + expected + '"')


def fetch_front(assessment: dict) -> dict:
    key = urlparse(assessment['url']).path.lstrip('/')
    if not re.fullmatch(r'(PMC\d+\.\d+)/\1\.xml', key):
        raise ValueError('Unexpected public XML object key')
    url = BASE + key
    token = hashlib.sha256(url.encode()).hexdigest()
    full = FOLDER / 'xml_raw' / (token + '.json')
    if full.exists():
        return json.loads(full.read_text())
    folder = FOLDER / 'xml_front_raw'; folder.mkdir(exist_ok=True)
    path = folder / (token + '.json')
    if path.exists():
        return json.loads(path.read_text())
    if (FOLDER / 'source_stop.json').exists():
        STOP.set()
    if STOP.is_set():
        return {'status': 'not_requested_after_source_stop', 'url': url}
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session(); LOCAL.session.trust_env = False
        LOCAL.session.headers['User-Agent'] = 'Fig8MetadataResearch/1.0 (public PMC front metadata)'
    expected = assessment['md5']
    headers = {'Range': f'bytes=0-{LIMIT - 1}', 'If-Match': '"' + expected + '"', 'Accept-Encoding': 'identity'}
    receipt = {'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat(), 'attempts': [],
               'requested_range': headers['Range'], 'if_match': headers['If-Match'],
               'full_object_md5_expected': expected, 'front_capture_only': True}
    for attempt in range(3):
        try:
            response = LOCAL.session.get(url, headers=headers, timeout=(10, 35), allow_redirects=False)
            body = response.content
            receipt['attempts'].append({'http_status': response.status_code})
            receipt.update(http_status=response.status_code, etag=response.headers.get('ETag', ''),
                content_range=response.headers.get('Content-Range', ''), response_bytes=len(body),
                retry_after=response.headers.get('Retry-After', ''))
            match = re.fullmatch(r'bytes 0-(\d+)/(\d+)', receipt['content_range'])
            whole = response.status_code == 200 or bool(match and int(match[2]) == len(body))
            end = body.find(b'</front>') + len(b'</front>')
            valid_range = response.status_code == 206 and range_matches(len(body), receipt, expected)
            if valid_range and not whole and end >= len(b'</front>'):
                body = body[:end]
                receipt.update(source_extent='complete_front_only', retained_prefix_bytes=len(body),
                               integrity_basis='Conditional S3 ETag and retained-prefix SHA256; full-object MD5 not computed')
            elif whole:
                receipt.update(source_extent='full_object', integrity_basis='Full-object MD5 checked at parse')
            else:
                receipt.update(source_extent='prefix_incomplete_or_unconfirmed')
            raw_path = path.with_suffix('.xml.gz')
            atomic_bytes(raw_path, gzip.compress(body, mtime=0))
            receipt.update(raw_receipt=str(raw_path.relative_to(ROOT)), retained_sha256=hashlib.sha256(body).hexdigest(),
                           retained_bytes=len(body), receipt_path=str(path.relative_to(ROOT)))
            if response.status_code in {401, 403, 429}:
                STOP.set(); atomic_json(FOLDER / 'source_stop.json', receipt); break
            if response.status_code not in {500, 502, 503, 504}:
                break
            if receipt['retry_after']:
                STOP.set(); atomic_json(FOLDER / 'source_stop.json', receipt); break
        except requests.RequestException as error:
            receipt['attempts'].append({'error': type(error).__name__})
        if attempt < 2:
            time.sleep(2 ** attempt)
    atomic_json(path, receipt)
    if receipt.get('http_status') in {None, 500, 502, 503, 504}:
        STOP.set(); atomic_json(FOLDER / 'source_stop.json', receipt)
    return receipt


def parse_front_capture(raw: bytes, receipt: dict, assessment: dict) -> dict:
    if receipt.get('source_extent') == 'complete_front_only':
        if (receipt.get('http_status') != 206 or not range_matches(receipt['response_bytes'], receipt, assessment['md5'])
                or hashlib.sha256(raw).hexdigest() != receipt.get('retained_sha256')
                or not raw.endswith(b'</front>')):
            raise ValueError('Front range or object identity not confirmed')
        document = raw + b'</article>'
    else:
        if hashlib.md5(raw).hexdigest() != assessment['md5']:
            raise ValueError('Full XML checksum conflict')
        document = raw
    tree = etree.fromstring(document, etree.XMLParser(resolve_entities=False, no_network=True))
    if tree.tag != 'article' or tree.find('./front/article-meta') is None:
        raise ValueError('Missing native article front matter')
    parsed = jats_metadata(document)
    if assessment['doi'].lower() not in {d.lower() for d in parsed['doi']}:
        raise ValueError('XML DOI mismatch')
    return {**parsed, 'issns': parsed['issn']}


def acquire_front(item: tuple) -> tuple[str, dict]:
    origin, version, assessment = item
    receipt = fetch_front(assessment)
    result = {'version_result': str(origin.relative_to(ROOT)), 'assessment': assessment,
              'receipt': receipt, 'final_task_closure': False}
    if receipt.get('status') == 'not_requested_after_source_stop':
        result['status'] = 'unrequested_source_stop'; return assessment['doi'], result
    if receipt.get('http_status') not in {200, 206}:
        result['status'] = 'xml_source_unavailable'; return assessment['doi'], result
    raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
    if (receipt.get('source_extent') == 'prefix_incomplete_or_unconfirmed'
            and range_matches(receipt.get('response_bytes', 0), receipt, assessment['md5'])):
        result['front_attempt_receipt'] = receipt
        receipt = fetch_metadata(urlparse(assessment['url']).path.lstrip('/'), xml=True)
        result['receipt'] = receipt
        if receipt.get('http_status') != 200:
            result['status'] = ('unrequested_source_stop' if receipt.get('status') == 'not_requested_after_source_stop'
                                else 'xml_source_unavailable')
            return assessment['doi'], result
        raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
    try:
        parsed = parse_front_capture(raw, receipt, assessment)
    except (ValueError, OSError, etree.XMLSyntaxError) as error:
        result.update(status='xml_validation_unresolved', error=str(error))
        return assessment['doi'], result
    record = {'source': 'pmc', 'route': 'pmc_cloud', 'status': 'parsed', 'parsed': parsed,
        'url': assessment['url'], 'raw_receipt': '../../' + receipt['raw_receipt'],
        'cloud_version_result': str(origin.relative_to(ROOT)), 'pmcid': version['inventory']['pmcid'],
        'source_extent': receipt.get('source_extent', 'full_object'), 'acquisition_receipt': receipt}
    path = FOLDER / 'xml_parsed' / origin.name
    atomic_json(path, record)
    result.update(status='xml_parsed_identity_recheck_at_application', parsed_receipt=str(path.relative_to(ROOT)))
    return assessment['doi'], result
