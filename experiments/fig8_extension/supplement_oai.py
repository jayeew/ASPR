"""Sequential official PMC front-matter fallback during permitted bulk-retrieval hours."""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from lxml import etree

from .supplement_queue import ROOT, attempt, connect
from .supplement_sources import atomic_json, jats_metadata

URL = 'https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/'
POLICY = 'https://pmc.ncbi.nlm.nih.gov/tools/oai'


def seconds_until_bulk_window(now: datetime) -> float:
    local = now.astimezone(ZoneInfo('America/New_York'))
    if local.weekday() >= 5 or local.hour < 5 or local.hour >= 21:
        return 0
    end = local.replace(hour=21, minute=0, second=0, microsecond=0)
    return max(0, (end - local).total_seconds())


def parse_oai(data: bytes) -> dict:
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    root = etree.fromstring(data, parser)
    articles = root.xpath('//*[local-name()="metadata"]/*[local-name()="article"]')
    if len(articles) != 1:
        errors = root.xpath('//*[local-name()="error"]/text()')
        raise ValueError('No unique JATS front record: ' + ' | '.join(errors))
    article = etree.fromstring(etree.tostring(articles[0]), parser)
    for node in article.iter():
        if isinstance(node.tag, str):
            node.tag = etree.QName(node).localname
    etree.cleanup_namespaces(article)
    return jats_metadata(etree.tostring(article))


def run(root: Path = ROOT) -> dict:
    q = root / 'fields/qualification'
    previous = {}
    for path in (q / 'supplement_parsed').glob('*.json'):
        r = json.loads(path.read_text())
        if r.get('source') == 'pmc' and r.get('status') in ['retry_exhausted', 'not_available', 'http_error', 'access_restricted']:
            previous[r['pmcid']] = r
    folder = q / 'oai_parsed'
    raw_folder = q / 'oai_raw'
    folder.mkdir(exist_ok=True)
    raw_folder.mkdir(exist_ok=True)
    pending = [r for pmc, r in previous.items() if not (folder / (pmc + '.json')).exists()]
    db = connect(root)
    session = requests.Session()
    session.trust_env = False
    session.headers['User-Agent'] = 'Fig8MetadataResearch/1.0 (sequential public front metadata; policy-compliant bulk hours)'
    counts = {'targets': len(pending), 'completed': 0, 'parsed': 0, 'unresolved': 0}
    for old in pending:
        while True:
            delay = seconds_until_bulk_window(datetime.now(timezone.utc))
            if not delay:
                break
            atomic_json(root / 'supplement/oai_status.json', {**counts, 'status': 'waiting_for_permitted_bulk_window',
                'not_before_utc': (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat(), 'policy_url': POLICY})
            time.sleep(min(60, delay))
        pmc = old['pmcid']
        r = {'doi': old['doi'], 'pmcid': pmc, 'source': 'pmc_front_oai', 'url': URL,
             'params': {'verb': 'GetRecord', 'identifier': 'oai:pubmedcentral.nih.gov:' + pmc.removeprefix('PMC'), 'metadataPrefix': 'pmc_fm'},
             'retrieved_at': datetime.now(timezone.utc).isoformat(), 'attempts': []}
        for n in range(3):
            try:
                response = session.get(URL, params=r['params'], timeout=(10, 45))
                r['http_status'] = response.status_code
                r['source_url']=response.url
                r['attempts'].append(response.status_code)
                raw_path = raw_folder / (pmc + '.xml.gz')
                raw_path.write_bytes(gzip.compress(response.content, mtime=0))
                r.update(raw_receipt=str(raw_path.relative_to(q)), raw_sha256=hashlib.sha256(response.content).hexdigest())
                if response.ok:
                    try:
                        r.update(status='parsed', parsed=parse_oai(response.content))
                    except (ValueError, etree.XMLSyntaxError) as error:
                        r.update(status='metadata_unavailable', reason=str(error)[:400])
                    break
                r['status'] = 'access_restricted' if response.status_code in [401, 403] else 'http_error'
                if response.status_code not in [429, 500, 502, 503, 504]:
                    break
                retry = response.headers.get('Retry-After', '')
                if retry.isdigit() and int(retry) > 60:
                    r.update(status='retry_deferred', retry_after_seconds=int(retry))
                    break
                time.sleep(max(2 ** n, int(retry) if retry.isdigit() else 0))
            except requests.RequestException as error:
                r.update(status='request_error', reason=type(error).__name__)
                time.sleep(2 ** n)
            time.sleep(.35)
        atomic_json(folder / (pmc + '.json'), r)
        ident = hashlib.sha256(('qualification:' + old['doi']).encode()).hexdigest()[:32]
        attempt(db, ident, 'official_pmc_front_oai', 'evidence_ready' if r['status'] == 'parsed' else r['status'],
                str((folder / (pmc + '.json')).relative_to(root)), {'reason': r.get('reason', r['status']), 'policy_url': POLICY})
        counts['completed'] += 1
        counts['parsed' if r['status'] == 'parsed' else 'unresolved'] += 1
        atomic_json(root / 'supplement/oai_status.json', {**counts, 'status': 'running', 'policy_url': POLICY})
        time.sleep(.35)
    db.close()
    atomic_json(root / 'supplement/oai_status.json', {**counts, 'status': 'complete_with_explicit_unresolved', 'policy_url': POLICY})
    return counts


def main() -> None:
    print(json.dumps(run()))


if __name__ == '__main__':
    main()
