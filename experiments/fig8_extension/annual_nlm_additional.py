"""Reuse additional explicit non-Portfolio imprint intervals from cached NLM records."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict

from lxml import etree

from .annual_nlm import URL, imprint_interval
from .cell_press_annual_cache import apply_proposals
from .supplement_queue import ROOT
from .three_journal_apply import read_csv

ENTITIES = {
    'Lippincott Williams & Wilkins', 'Lippincott Williams & Wilkins, Inc.',
    'Thieme', 'De Gruyter', 'De Gruyter Open', 'Walter de Gruyter GmbH',
    'De Gruyter Poland', 'Cell Press',
}



def run() -> dict:
    folder = ROOT / 'supplement/annual_nlm/additional_imprints'
    folder.mkdir(exist_ok=True)
    registry = {r['source_id']: r for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    annual = {(r['source_id'], int(r['year'])): r for r in read_csv(ROOT / 'fields/journal_publisher_by_year.csv')}
    queue = defaultdict(list)
    for row in read_csv(ROOT / 'supplement/annual_nlm/queue.csv'):
        queue[row['nlm_id']].append(row)
    prior = folder / 'rules.json'
    proposals = {(r['source_id'], r['year']): r for r in json.loads(prior.read_text())} if prior.exists() else {}
    unresolved = []
    for path in sorted((ROOT / 'supplement/annual_nlm/raw').glob('*.json')):
        receipt = json.loads(path.read_text())
        if receipt.get('http_status') != 200:
            continue
        raw = gzip.decompress((ROOT / receipt['raw_receipt']).read_bytes())
        if hashlib.sha256(raw).hexdigest() != receipt['sha256']:
            raise ValueError('NLM receipt hash mismatch')
        tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
        for record in tree.findall('NLMCatalogRecord'):
            identifier = record.findtext('NlmUniqueID')
            issns = {n.text for n in record.findall('ISSN') if n.get('ValidYN') == 'Y'}
            for imprint in record.findall('./PublicationInfo/Imprint[@ImprintType="Current"]'):
                entity = imprint.findtext('Entity') or ''
                interval = imprint_interval(imprint.findtext('DateIssued') or '') or imprint_interval(imprint.findtext('ImprintFull') or '')
                if entity not in ENTITIES or not interval:
                    continue
                for item in queue[identifier]:
                    sid = item['source_id']
                    if not issns & set(json.loads(registry[sid]['issn'] or '[]')):
                        unresolved.append({'source_id': sid, 'nlm_id': identifier, 'status': 'issn_identity_not_confirmed'})
                        continue
                    for year in (2023, 2024, 2025):
                        old = annual.get((sid, year))
                        if not old or old['nature_portfolio_year'] in {'True', 'False'} or not interval[0] < year < interval[1]:
                            continue
                        proposals[(sid, year)] = {
                            'source_id': sid, 'year': year, 'label': 'False', 'journal': item['journal'],
                            'publisher': entity, 'nlm_id': identifier, 'issns': sorted(issns),
                            'source_url': URL, 'raw_receipt': receipt['raw_receipt'], 'raw_sha256': receipt['sha256'],
                            'imprint_full': imprint.findtext('ImprintFull'), 'interval': interval,
                            'locator': f'NLMCatalogRecord[NlmUniqueID={identifier}]/PublicationInfo/Imprint[@ImprintType="Current"]',
                            'scope': 'Exact valid ISSN; explicit NLM publisher imprint interval; exclude both boundary years; reopen on contradictory transfer evidence'}
    return apply_proposals(list(proposals.values()), folder, unresolved,
                           rule_status='official_nlm_additional_explicit_imprint_interval',
                           reason='Cached official NLM dated non-Portfolio imprint; exact ISSN; interval boundary excluded',
                           route='dated_nlm_catalogue')


if __name__ == '__main__':
    print(json.dumps(run()), flush=True)
