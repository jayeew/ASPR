"""Bounded three-journal delta work; no global discovery or collection entry points."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import requests
from lxml import etree

from .fig8b_qualification_build import match, plain
from .fig8b_qualification_decide import decide, electronic_date, key
from .supplement_crossref import SELECT, batches
from .supplement_oai import URL, parse_oai, seconds_until_bulk_window

ROOT = Path('outputs/fig8_data')
Q = ROOT / 'fields/qualification'
OUT = ROOT / 'analysis/journal_readiness'
AUDIT = ROOT / 'supplement/three_journal_round'
SIDS = ['https://openalex.org/' + s for s in ['S3880285', 'S110447773', 'S125754415']]
ALIASES = {SIDS[0]: 'Science (New York, N.Y.)', SIDS[2]: 'Proceedings of the National Academy of Sciences of the United States of America'}


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def init() -> dict:
    path = AUDIT / 'budget.json'
    if not path.exists():
        save(path, {'start_utc': '2026-10-23T05:49:29+00:00', 'deadline_utc': '2026-10-23T07:19:29+00:00',
                    'request_cap': 300, 'doi_cap': 200, 'requests': 0, 'dois': [], 'blocked_hosts': [], 'status': 'running'})
    return json.loads(path.read_text())


def get(url: str, params: dict | None = None, dois: list[str] | None = None) -> dict:
    """Reserve each HTTP request before sending; redirects never happen implicitly."""
    from urllib.parse import urlsplit
    state = init()
    host = urlsplit(url).netloc
    ids = sorted(set(state['dois']) | set(dois or []))
    if time.time() >= datetime.fromisoformat(state['deadline_utc']).timestamp() - 1200:
        raise RuntimeError('Acquisition deadline: reserve final 20 minutes')
    if state['status'] != 'running' or state['requests'] >= 300 or len(state['dois']) >= 200 or len(ids) > 200 or host in state['blocked_hosts']:
        raise RuntimeError('Request/DOI cap or blocked source')
    state.update(requests=state['requests'] + 1, dois=ids)
    save(AUDIT / 'budget.json', state)
    receipt = AUDIT / 'requests' / f"{state['requests']:03d}.json"
    record = {'url': url, 'params': params, 'dois': dois or [], 'started_at': datetime.now(timezone.utc).isoformat()}
    save(receipt, record)
    session = requests.Session()
    session.trust_env = False
    session.headers['User-Agent'] = 'Fig8MetadataResearch/1.0 (bounded serial metadata requests)'
    started = time.monotonic()
    try:
        response = session.get(url, params=params, timeout=(10, 30), allow_redirects=False)
        raw_path = receipt.with_suffix('.body.gz')
        raw_path.write_bytes(gzip.compress(response.content, mtime=0))
        record.update(http_status=response.status_code, final_url=response.url, raw_receipt=str(raw_path.relative_to(ROOT)),
                      sha256=hashlib.sha256(response.content).hexdigest(), retry_after=response.headers.get('Retry-After', ''),
                      location=response.headers.get('Location', ''))
        if response.status_code in [401, 403, 429]:
            state['blocked_hosts'].append(host)
            save(AUDIT / 'budget.json', state)
    except requests.RequestException as error:
        record.update(error=type(error).__name__)
    record['elapsed_seconds'] = time.monotonic() - started
    save(receipt, record)
    time.sleep(1.05)
    return record


def load() -> tuple[dict, dict, dict, dict]:
    decisions = {}
    for path in sorted((Q / 'decisions').glob('*.parquet')):
        for row in pq.read_table(path, filters=[('source_id', 'in', SIDS)]).to_pylist():
            decisions[row['work_id']] = row
    evidence, secondary = defaultdict(list), defaultdict(list)
    # Stream existing evidence indices once; no global rebuilding or network discovery.
    for filename, dest, secondary_mode in [('decision_evidence.jsonl.gz', evidence, False), ('secondary_evidence.jsonl.gz', secondary, True)]:
        with gzip.open(Q / filename, 'rt') as stream:
            for line in stream:
                item = json.loads(line)
                if item['work_id'] in decisions:
                    dest[item['work_id']].extend(item['records'] if secondary_mode else [item])
    with (ROOT / 'fields/journal_registry.csv').open(encoding='utf-8-sig') as stream:
        sources = {r['source_id']: r for r in csv.DictReader(stream) if r['source_id'] in SIDS}
    return decisions, evidence, secondary, sources


def row_for(d: dict, sources: dict) -> dict:
    return {**d, 'journal_name': sources[d['source_id']].get('journal_name') or sources[d['source_id']].get('display_name') or ''}


def cached_aliases(decisions: dict, evidence: dict, sources: dict) -> dict:
    proofs = defaultdict(list)
    for wid, items in evidence.items():
        sid = decisions[wid]['source_id']
        for e in items:
            if e['identity'] == 'matched' and set(e.get('issns', [])) & set(json.loads(sources[sid]['issn'])) and ALIASES.get(sid) in e.get('journal_titles', []):
                if len(proofs[sid]) < 3:
                    proofs[sid].append({'id': e['id'], 'receipt': e['receipt'], 'issns': e['issns'], 'journal_titles': e['journal_titles']})
    rules = {sid: {'alias': ALIASES[sid], 'proofs': items, 'scope': 'Exact title + DOI; missing ISSN only; journal alias validated by archived ISSNs'} for sid, items in proofs.items()}
    save(AUDIT / 'journal_alias_rules.json', rules)
    return rules


def repair_alias(d: dict, e: dict, rules: dict) -> dict:
    rule = rules.get(d['source_id'])
    if not rule or e['identity'] == 'matched' or e.get('issns'):
        return e
    if key(plain(d['title'])) != key(plain(e['title'])) or rule['alias'] not in e.get('journal_titles', []):
        return e
    # The original evidence index is keyed by returned DOI, not requested DOI.
    if e.get('doi') != d['doi']:
        return e
    result = {**e, 'source_id': d['source_id'], 'identity': 'matched', 'identity_rule': 'three_journal_exact_title_doi_archived_journal_alias',
              'supersedes': e['id'], 'rule_receipt': '../../supplement/three_journal_round/journal_alias_rules.json'}
    result['id'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:24]
    return result


def cache() -> None:
    init()
    decisions, evidence, secondary, sources = load()
    rules = cached_aliases(decisions, evidence, sources)
    changes, opportunities = [], []
    for wid, d in decisions.items():
        if not d['eligibility_status'].startswith('pending'):
            continue
        fixed = [repair_alias(d, e, rules) for e in evidence[wid]]
        for old, new in zip(evidence[wid], fixed):
            if old['id'] != new['id']:
                changes.append(new)
        result = decide(fixed, secondary[wid], d['duplicate_of'])
        if result['status'] == 'pending_primary_online_date':
            opportunities.append({'work_id': wid, 'doi': d['doi'], 'source_id': d['source_id'], 'field': d['field'],
                                  'year': d['year_openalex'], 'pmcids': json.loads(d['secondary_pmcids']), 'before_status': d['eligibility_status']})
    save(AUDIT / 'cache_identity_delta.json', changes)
    save(AUDIT / 'date_opportunities.json', opportunities)
    print(json.dumps({'alias_repaired': len(changes), 'date_opportunities': dict(Counter(r['source_id'] for r in opportunities))}))


def crossref() -> None:
    opportunities = json.loads((AUDIT / 'date_opportunities.json').read_text())
    # Reuse cached deposit responses; do not repeat successful deposits lacking online dates.
    known = set()
    for path in (ROOT / 'fields/crossref_raw').glob('*.gz'):
        with gzip.open(path, 'rt') as stream:
            prior = json.load(stream)
        if prior.get('status') == 'success':
            known.update(i['DOI'].lower() for i in prior.get('response', {}).get('message', {}).get('items', []))
    selected = []
    quotas = {SIDS[0]: 30, SIDS[1]: 50, SIDS[2]: 120}
    for sid, quota in quotas.items():
        pool = [r for r in opportunities if r['source_id'] == sid and r['doi'] not in known]
        groups = defaultdict(list)
        for r in pool:
            groups[(r['field'], r['year'])].append(r)
        for group in groups.values():
            group.sort(key=lambda r: hashlib.sha256(('20261021:' + r['doi']).encode()).hexdigest())
        ordered = []
        while any(groups.values()):
            for group in groups.values():
                if group:
                    ordered.append(group.pop(0))
        selected.extend(ordered[:quota])
    save(AUDIT / 'network_targets.json', selected)
    for group in batches([r['doi'] for r in selected]):
        record = get('https://api.crossref.org/works', {'filter': ','.join('doi:' + d for d in group), 'select': SELECT, 'rows': len(group)}, group)
        if record.get('http_status') != 200:
            print('Crossref stopped', record, flush=True)
            break
        print('crossref batch', len(group), record['elapsed_seconds'], flush=True)


def official() -> None:
    # Three direct public publisher pages; redirects/blocks retained, never bypassed.
    for url in ['https://www.science.org/content/page/about-science', 'https://www.cell.com/cell/about', 'https://www.pnas.org/about']:
        record = get(url)
        print(url, record.get('http_status', record.get('error')), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['cache', 'crossref', 'official', 'prepare'])
    args = parser.parse_args()
    {'cache': cache, 'crossref': crossref, 'official': official, 'prepare': prepare}[args.action]()


def deposit_evidence(item: dict, receipt: str, url: str) -> list[dict]:
    from datetime import date
    result = []
    for parts in item.get('published-online', {}).get('date-parts', []):
        if not parts or not all(isinstance(x, int) for x in parts):
            continue
        try:
            date(parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1)
        except (ValueError, TypeError):
            continue
        result.append({'family': 'crossref_publisher_deposit', 'title': plain((item.get('title') or [''])[0]),
                       'issns': item.get('ISSN', []), 'journal_titles': item.get('container-title', []),
                       'kind': item.get('type', 'unspecified'), 'online_date': '-'.join(f'{x:04d}' if i == 0 else f'{x:02d}' for i, x in enumerate(parts)),
                       'date_precision': ['year', 'month', 'day'][min(len(parts), 3)-1],
                       'receipt': receipt, 'url': url, 'categories': []})
    return result


def prepare() -> None:
    decisions, evidence, secondary, sources = load()
    delta = json.loads((AUDIT / 'cache_identity_delta.json').read_text())
    by_doi = {d['doi']: wid for wid, d in decisions.items()}
    for path in sorted((AUDIT / 'requests').glob('*.json')):
        record = json.loads(path.read_text())
        if record.get('http_status') != 200 or 'crossref' not in record['url']:
            continue
        raw = json.loads(gzip.decompress((ROOT / record['raw_receipt']).read_bytes()))
        for item in raw.get('message', {}).get('items', []):
            doi = item.get('DOI', '').lower()
            wid = by_doi.get(doi)
            if not wid or doi not in record['dois']:
                continue
            for e in deposit_evidence(item, '../../' + record['raw_receipt'], record['final_url']):
                delta.append({'work_id': wid, 'doi': doi, **match(row_for(decisions[wid], sources), e, sources[decisions[wid]['source_id']])})
    for e in delta:
        evidence[e['work_id']].append(e)
    # Annual PNAS imprints: exact-identity archived publications with explicit online year.
    annual = []
    for sid in SIDS:
        for year in [2023, 2024, 2025]:
            proofs = []
            for wid, d in decisions.items():
                if d['source_id'] != sid:
                    continue
                for e in evidence[wid]:
                    if e['identity'] != 'matched' or e['family'] != 'archived_jats' or not e.get('online_date', '').startswith(str(year)):
                        continue
                    if 'National Academy of Sciences' not in e.get('publishers', []):
                        continue
                    proofs.append({'work_id': wid, 'doi': d['doi'], 'id': e['id'], 'receipt': e['receipt'], 'date': e['online_date'], 'publisher': e['publishers']})
            proofs = sorted(proofs, key=lambda e: e['date'])
            # Beginning, middle, end of observed annual publishing range; all support is retained in trace.
            sample = [proofs[i] for i in sorted({0, len(proofs)//2, len(proofs)-1})] if proofs else []
            annual.append({'source_id': sid, 'year': year, 'label': 'False' if proofs else 'unresolved',
                           'status': 'dated_archived_journal_imprint' if proofs else 'no_confirmed_dated_imprint_in_cache',
                           'proof_count': len(proofs), 'proofs': sample,
                           'scope': 'Journal annual non-Nature label from dated publisher imprints; future conflicting evidence reopens rule; no inference from missing Nature list'})
    override_path = AUDIT / 'annual_overrides.json'
    if override_path.exists():
        overrides = {(r['source_id'], r['year']): r for r in json.loads(override_path.read_text())}
        annual = [overrides.get((r['source_id'], r['year']), r) for r in annual]
    save(AUDIT / 'annual_rules.json', annual)
    amap = {(r['source_id'], r['year']): r for r in annual}
    proposals = []
    from .fig8b_qualification_decide import portfolio_label
    for wid, d in decisions.items():
        result = decide(evidence[wid], secondary[wid], d['duplicate_of'])
        updated = dict(d)
        updated.update(online_year=result['online_year'], online_date=result['online_date'], eligibility_status=result['status'], reason=result['reason'],
                       publisher_types=result['type'], evidence_ids=json.dumps(result['evidence_ids']), rejected_evidence_ids=json.dumps(result.get('rejected_evidence_ids', [])))
        rule = amap.get((d['source_id'], result['online_year']))
        inherited = rule['label'] if rule else ''
        # Preserve pre-existing confirmed labels when no new annual rule is available.
        if inherited not in ['True', 'False'] and d['online_year'] == result['online_year']:
            inherited = d['nature_portfolio_at_verified_online_year']
        label, ids = portfolio_label(inherited, evidence[wid], result['online_year'])
        updated['nature_portfolio_at_verified_online_year'] = label
        if label == d['nature_portfolio_at_verified_online_year'] and result['online_year'] == d['online_year']:
            ids = json.loads(d['portfolio_evidence_ids'])
        elif rule and rule['label'] == 'False':
            ids = [p['id'] for p in rule['proofs']]
        elif not ids:
            ids = json.loads(d['portfolio_evidence_ids'])
        updated['portfolio_evidence_ids'] = json.dumps(ids)
        if updated != d:
            proposals.append({'work_id': wid, 'before': d, 'after': updated})
    save(AUDIT / 'evidence_delta.json', delta)
    save(AUDIT / 'decision_delta.json', proposals)
    transitions = Counter((r['before']['source_id'], r['before']['eligibility_status'], r['after']['eligibility_status']) for r in proposals if r['before']['eligibility_status'] != r['after']['eligibility_status'])
    print(json.dumps({'transitions': [{'source': k[0], 'before': k[1], 'after': k[2], 'n': n} for k,n in transitions.items()], 'proposals': len(proposals), 'evidence_delta': len(delta)}))


if __name__ == '__main__':
    main()
