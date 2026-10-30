"""At most 100 serial official front-matter requests on priority EPMC failures."""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import time
from datetime import datetime, timezone

import requests
from lxml import etree

from .supplement_oai import POLICY, URL, parse_oai, seconds_until_bulk_window
from .supplement_queue import ROOT, attempt, connect
from .supplement_sources import atomic_bytes, atomic_json


def run(expanded: bool = False) -> None:
    folder = ROOT / 'supplement'
    config = json.loads((folder / 'journal_goal_budget.json').read_text())
    priorities = json.loads((folder / 'priority_dois.json').read_text())
    q = ROOT / 'fields/qualification'
    target = q / 'oai_parsed'
    raw = q / 'oai_raw'
    target.mkdir(exist_ok=True)
    raw.mkdir(exist_ok=True)
    db = connect(ROOT)
    candidates = []
    seen = set()
    for receipt, in db.execute("SELECT receipt FROM attempts WHERE source='pmc' AND status IN ('retry_exhausted','not_available') ORDER BY id DESC"):
        path = ROOT / receipt
        if not path.is_file():
            continue
        r = json.loads(path.read_text())
        pmc = r.get('pmcid', '')
        if r.get('doi') not in priorities or not pmc or pmc in seen or (target / (pmc + '.json')).exists():
            continue
        seen.add(pmc)
        candidates.append(r)
        if len(candidates) >= 100:
            break
    limit = 100
    if expanded:
        previous = json.loads((folder / 'journal_goal_status.json').read_text())
        limit = max(0, 3000 - previous['completed'] - 100 - 16)
        # Reuse original priority cohort; one archive identity per DOI and no repeated OAI receipt.
        for doi, pmcids in db.execute("SELECT doi,pmcids FROM papers WHERE status LIKE 'pending%' AND pmcids<>'[]'"):
            if doi not in priorities:
                continue
            for pmc in sorted(set(json.loads(pmcids))):
                if pmc in seen or (target / (pmc + '.json')).exists():
                    continue
                native = q / 'jats_parsed' / (pmc + '.json')
                if native.exists() and json.loads(native.read_text()).get('status') == 'parsed':
                    continue
                seen.add(pmc)
                candidates.append({'doi': doi, 'pmcid': pmc})
                break
    candidates.sort(key=lambda r: (priorities[r['doi']], r['doi']))
    unique = {}
    for item in candidates:
        unique.setdefault(item['doi'], item)
    candidates = list(unique.values())[:limit]
    status_file = folder / ('journal_oai_priority_status.json' if expanded else 'journal_oai_probe_status.json')
    acquisition_deadline = min(datetime.fromisoformat(config['deadline_utc']).timestamp(), datetime.fromisoformat(config['created_at']).timestamp() + 2700)
    status = {'started_at': datetime.now(timezone.utc).isoformat(), 'targets': len(candidates), 'completed': 0, 'parsed': 0, 'identity_matched': 0, 'policy': POLICY, 'status': 'running'}
    session = requests.Session()
    session.trust_env = False
    session.headers.update({'User-Agent': 'Fig8MetadataResearch/1.0 (bounded serial front-matter probe)', 'Accept-Encoding': 'gzip, deflate'})
    for old in candidates:
        if time.time() >= acquisition_deadline or seconds_until_bulk_window(datetime.now(timezone.utc)):
            status['stop_reason'] = 'deadline_or_policy_window'
            break
        if status['completed'] == 10 and status['identity_matched'] < 5:
            status['stop_reason'] = 'pilot_identity_yield_below_half'
            break
        pmc = old['pmcid']
        params = {'verb': 'GetRecord', 'identifier': 'oai:pubmedcentral.nih.gov:' + pmc.removeprefix('PMC'), 'metadataPrefix': 'pmc_fm'}
        r = {'doi': old['doi'], 'pmcid': pmc, 'source': 'pmc_front_oai', 'url': URL, 'params': params, 'retrieved_at': datetime.now(timezone.utc).isoformat(), 'attempts': []}
        started = time.monotonic()
        try:
            response = session.get(URL, params=params, timeout=(10, 25))
            r.update(http_status=response.status_code, source_url=response.url, attempts=[response.status_code])
            raw_path = raw / (pmc + '.xml.gz')
            atomic_bytes(raw_path, gzip.compress(response.content, mtime=0))
            r.update(raw_receipt=str(raw_path.relative_to(q)), raw_sha256=hashlib.sha256(response.content).hexdigest())
            if response.ok:
                r.update(status='parsed', parsed=parse_oai(response.content))
                status['parsed'] += 1
                # The builder performs the complete DOI/title/journal identity checks later.
                found = r['parsed'].get('doi', [])
                found = [found] if isinstance(found, str) else found
                status['identity_matched'] += old['doi'] in found
            else:
                r['status'] = 'access_restricted' if response.status_code in [401, 403] else 'http_error'
        except (requests.RequestException, ValueError, etree.XMLSyntaxError) as error:
            r.update(status='metadata_unavailable', reason=str(error)[:250])
        r['elapsed_seconds'] = time.monotonic() - started
        atomic_json(target / (pmc + '.json'), r)
        ident = hashlib.sha256(('qualification:' + old['doi']).encode()).hexdigest()[:32]
        attempt(db, ident, 'official_pmc_front_oai', 'evidence_ready' if r['status'] == 'parsed' else r['status'], str((target / (pmc + '.json')).relative_to(ROOT)), {'reason': r.get('reason', r['status']), 'bounded_priority_probe': True})
        status['completed'] += 1
        atomic_json(status_file, status)
        if r.get('http_status') in [401, 403, 429]:
            status['stop_reason'] = 'access_or_rate_limit_no_retry_this_run'
            break
        time.sleep(.35)
    db.close()
    status.update(status='complete_with_unresolved', finished_at=datetime.now(timezone.utc).isoformat())
    atomic_json(status_file, status)
    print(json.dumps(status))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expanded', action='store_true')
    args = parser.parse_args()
    if args.expanded:
        # Hold shared read lock until all new evidence is durable; the writer then applies it once.
        with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            (ROOT / 'supplement/stop_acquisition').write_text('Measured priority handoff to bounded serial OAI; offline checkpoint follows.\n')
            try:
                run(True)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    else:
        run()
