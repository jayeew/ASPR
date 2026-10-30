"""Review retained MDPI failures, apply scoped alias, and unlock the fixed queue."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter

from lxml import etree

from .mdpi_cache_apply import evidence
from .mdpi_targeted_collect import FOLDER, NS, TARGET, extract
from .supplement_queue import ROOT
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run(review_name: str = 'carbon_alias_review.json',
        rule_description: str = 'Only C – Journal of Carbon Research versus publisher C, with exact ISSN 2311-5629, DOI and title') -> None:
    review_receipt = 'supplement/mdpi_oai/targeted/' + review_name
    state = json.loads((TARGET/'status.json').read_text())
    if state.get('stop_reason') != '100_consecutive_unmatched_records_review_required':
        raise ValueError('Only the reviewed identity-mismatch stop can be reset')
    scope = {p['doi']: p for p in json.loads((TARGET/'scope.json').read_text())['papers']}
    registry = {r['source_id']: set(json.loads(r['issn'] or '[]')) for r in read_csv(ROOT/'fields/journal_registry.csv')}
    repaired, held = [], []
    for path in (TARGET/'results').glob('*.json'):
        result = json.loads(path.read_text())
        if result['status'] != 'identity_or_type_unconfirmed':
            continue
        receipt = result['receipt']
        raw = gzip.decompress((ROOT/receipt['raw_receipt']).read_bytes())
        tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
        item = extract(tree.find('o:GetRecord/o:record', NS))
        paper = scope[result['doi']]
        proof = evidence(item, paper, registry[paper['source_id']], receipt, hashlib.sha256(raw).hexdigest())
        if not proof:
            held.append({'doi': paper['doi'], 'source_dois': item['dois'], 'receipt': receipt['raw_receipt']})
            continue
        repaired.append({'doi': paper['doi'], 'before_status': result['status'], 'after_status': 'matched_native_type',
                         'receipt': receipt['raw_receipt'], 'work_id': paper['work_id']})
        atomic_json(FOLDER/'matched_pages'/('targeted_'+path.name),
                    {'receipt': receipt, 'matches': [{'doi': paper['doi'], 'oai_identifier': item['oai_identifier']}],
                     'discovery_receipt': paper['discovery_receipt']})
        atomic_json(path, {**result, 'status': 'matched_native_type',
                          'identity_repair_receipt': review_receipt})
    atomic_json(TARGET/review_name, {
        'rule': rule_description,
        'evidence_basis': 'Official OAI GetRecord header embeds the matching registered ISSN; dc:source supplies C',
        'network_requests': 0, 'previous_collector_state': state, 'repaired': repaired, 'unresolved': held,
        'date_not_inferred': True, 'qualification_not_implied': True})
    counts = Counter(json.loads(p.read_text())['status'] for p in (TARGET/'results').glob('*.json'))
    atomic_json(TARGET/'status.json', {**state, 'status': 'reviewed_resume_pending', 'stop_reason': '',
                'matched_records': counts['matched_native_type'], 'outcomes': dict(counts),
                'review_receipt': review_receipt})
    print(json.dumps({'cache_repaired': len(repaired), 'held': len(held), 'remaining_unattempted': state['fixed_dois']-state['completed']}))


if __name__ == '__main__':
    run()
