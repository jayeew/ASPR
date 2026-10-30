"""Apply exact MDPI OAI identities and native types; keep DC date semantics unresolved."""
from __future__ import annotations

import gzip
import hashlib
import html
import json
from collections import Counter

from lxml import etree

from .cached_date_apply import run as apply_sources
from .derived_refresh import refresh
from .fig8b_qualification_build import key, plain
from .mdpi_oai_harvest import FOLDER, NS, URL, extract
from .supplement_queue import ROOT
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def evidence(item: dict, paper: dict, issns: set[str], receipt: dict, digest: str) -> dict | None:
    fields = item['fields']
    titles = [html.unescape(plain(v)) for v in fields['title']]
    journals = fields['source']
    identifier = item['oai_identifier'] or ''
    parts = identifier.split('/')
    issn = parts[1] if len(parts) > 1 else ''
    carbon_alias = (issn == '2311-5629' and issn in issns
                    and key(paper['journal']) == key('C – Journal of Carbon Research')
                    and 'C' in journals)
    analytica_alias = (issn == '2673-4532' and issn in issns
                       and key(paper['journal']) == key('Analytica—A Journal of Analytical Chemistry and Chemical Analysis')
                       and 'Analytica' in journals)
    eng_alias = (issn == '2673-4117' and issn in issns
                 and key(paper['journal']) == key('Eng—Advances in Engineering')
                 and 'Eng' in journals)
    if (paper['doi'] not in item['dois'] or key(paper['title']) not in {key(v) for v in titles}
            or (key(paper['journal']) not in {key(v) for v in journals} and not (carbon_alias or analytica_alias or eng_alias))
            or issn not in issns):
        return None
    kinds = sorted(set(item['native_type_sets']))
    if len(kinds) != 1:
        return None
    return {'family': 'publisher_mdpi_oai', 'title': titles[0], 'source_id': paper['source_id'],
            'journal_titles': [paper['journal']], 'issns': [issn], 'kind': kinds[0], 'categories': [],
            'online_date': '', 'date_precision': '', 'receipt': '../../' + receipt['raw_receipt'],
            'raw_sha256': digest, 'url': URL, 'oai_identifier': identifier,
            'native_dc_dates': fields['date'], 'native_oai_datestamp': item['oai_datestamp'],
            'journal_identity_rule_receipt': ('../../supplement/mdpi_oai/targeted/carbon_alias_review.json'
                                              if carbon_alias else '../../supplement/mdpi_oai/targeted/eng_alias_review.json'
                                              if eng_alias else '../../supplement/mdpi_oai/targeted/analytica_alias_review.json'
                                              if analytica_alias else ''),
            'type_policy_receipt': '../../supplement/mdpi_oai/ListSets.json',
            'type_rule': 'Official OAI header setSpec article type; generic DC Text ignored',
            'date_limitation': 'Neither OAI datestamp nor untyped DC date establishes electronic publication date'}


def run() -> dict:
    folder = FOLDER / 'type_application'
    folder.mkdir(exist_ok=True)
    applied_path = folder / 'applied_fingerprints.json'
    applied = json.loads(applied_path.read_text()) if applied_path.exists() else {}
    scope = {p['doi']: p for p in json.loads((FOLDER / 'scope.json').read_text())['papers']}
    journals = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    receipts = [json.loads(p.read_text())['receipt'] for p in sorted((FOLDER / 'matched_pages').glob('*.json'))
                if json.loads(p.read_text())['matches']]
    for name in ['candidate_control.json', 'date_filter_control.json']:
        control = FOLDER / name
        if control.exists():
            receipts.append(json.loads(control.read_text()))
    sources, fingerprints = {}, {}
    unresolved, stats = [], Counter()
    for receipt in receipts:
        if receipt.get('http_status') != 200:
            continue
        raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
        digest = hashlib.sha256(raw).hexdigest()
        tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
        for node in tree.findall('.//o:record', NS):
            item = extract(node)
            for doi in item['dois']:
                if doi not in scope:
                    continue
                proof = evidence(item, scope[doi], journals[scope[doi]['source_id']], receipt, digest)
                if proof is None:
                    unresolved.append({'doi': doi, 'receipt': receipt['raw_receipt'], 'status': 'identity_or_native_type_not_unique'})
                    continue
                fingerprint = hashlib.sha256(json.dumps(
                    {'native_record': item, 'source_id': scope[doi]['source_id']}, sort_keys=True).encode()).hexdigest()
                entry_key = doi + ':native:' + fingerprint
                if applied.get(entry_key) == fingerprint or entry_key in fingerprints:
                    stats['unchanged_sources_skipped'] += 1
                    continue
                sources.setdefault(doi, []).append(proof)
                fingerprints[entry_key] = fingerprint
    atomic_json(folder / 'unresolved.json', unresolved)
    result = apply_sources(set(sources), folder, explicit_sources=sources) if sources else {'new_sources': 0}
    result.pop('with_explicit_deposited_date', None)
    result['type_evidence_dois'] = len(sources)
    result['online_dates_asserted'] = 0
    applied.update(fingerprints)
    atomic_json(applied_path, applied)
    result.update(network_requests=0, **stats)
    atomic_json(folder / 'status.json', result)
    if sources:
        refresh()
    return result


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
