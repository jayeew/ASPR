"""Apply dated publication statements from the saved fixed native-cache scope."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from lxml import etree

from .cell_press_annual_cache import apply_proposals
from .fig8b_qualification_decide import electronic_date
from .jats_annual_copyright import named_publisher
from .supplement_apply import signature_match
from .supplement_queue import ROOT
from .supplement_sources import atomic_json, jats_metadata
from .three_journal_apply import read_csv

FOLDER = ROOT / 'supplement/jats_annual_nonresearch_cache'


def prepare(folder: Path = FOLDER) -> tuple[list[dict], list[dict]]:
    scope = json.loads((folder / 'scope.json').read_text())['records']
    registry = {r['source_id']: set(json.loads(r['issn'] or '[]'))
                for r in read_csv(ROOT / 'fields/journal_registry.csv')}
    rules, reviews, seen, conflicts = {}, [], set(), set()
    for item in scope:
        e = item['evidence']
        year, sid = int(e['online_date'][:4]), item['source_id']
        if year not in {2023, 2024, 2025}:
            raise ValueError('Saved scope contains a year outside the goal')
        path = ROOT / 'fields/qualification' / e['receipt']
        identity = (item['doi'], str(path))
        if identity in seen:
            continue
        seen.add(identity)
        review = {'work_id': item['work_id'], 'doi': item['doi'], 'source_id': sid,
                  'year': year, 'receipt': str(path.relative_to(ROOT)),
                  'status': 'no_explicit_dated_named_publisher'}
        raw = gzip.decompress(path.read_bytes())
        document = raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw
        tree = etree.fromstring(document, etree.XMLParser(resolve_entities=False, no_network=True))
        if etree.QName(tree).localname != 'article':
            articles = tree.xpath('//*[local-name()="article"]')
            if len(articles) != 1:
                review['status'] = 'ambiguous_article_envelope'
                reviews.append(review)
                continue
            tree = articles[0]
            document = etree.tostring(tree)
        named = named_publisher(tree, year)
        if named is not None:
            native = jats_metadata(document)
            native['issns'] = native.get('issn', [])
            date, _ = electronic_date(native['dates'])
            if (date != e['online_date'] or item['doi'] not in {v.lower() for v in native['doi']}
                    or not signature_match({'title': item['title'], 'journal_name': item['journal'],
                                            'source_id': sid}, native, registry[sid])):
                review['status'] = 'native_identity_or_date_conflict'
            else:
                publisher, statement = named
                proof = {'doi': item['doi'], 'publication_date': date, 'statement': statement,
                         'raw_receipt': review['receipt'], 'raw_sha256': hashlib.sha256(raw).hexdigest(),
                         'source_url': e['url'], 'locator': 'front/article-meta/permissions',
                         'journal_titles': native['journal_titles'], 'issns': native['issns']}
                if publisher == 'Korean Society for Microbiology and Biotechnology':
                    proof['publisher_alias_policy'] = 'supplement/jmb_annual_cache/policy.json'
                pair = (sid, year)
                if pair not in rules:
                    rules[pair] = {'source_id': sid, 'year': year, 'label': 'False',
                        'journal': item['journal'], 'publisher': publisher, 'source_url': e['url'],
                        'raw_receipt': review['receipt'], 'raw_sha256': proof['raw_sha256'],
                        'first_dated_proof': proof, 'last_dated_proof': proof,
                        'scope': 'Annual working rule from exact native publication identity, agreeing electronic/copyright year and explicit named publisher. Research eligibility is independent. No extrapolation; reopen on transfer evidence.'}
                rule = rules[pair]
                if rule['publisher'] != publisher:
                    conflicts.add(pair)
                    review['status'] = 'different_named_publishers_same_year_review'
                else:
                    review['status'] = 'dated_publisher_identity_confirmed'
                    if date < rule['first_dated_proof']['publication_date']:
                        rule['first_dated_proof'] = proof
                    if date > rule['last_dated_proof']['publication_date']:
                        rule['last_dated_proof'] = proof
        reviews.append(review)
    proposals = [r for p, r in rules.items() if p not in conflicts]
    atomic_json(folder / 'review.json', {'records': reviews, 'conflicting_pairs': sorted(conflicts),
                                       'network_requests': 0, 'annual_changes': 0})
    atomic_json(folder / 'discovery_status.json', {'status': 'cached_annual_proposals_prepared',
        'proposed_journal_years': len(proposals), 'reviewed_receipts': len(reviews),
        'outcomes': dict(Counter(r['status'] for r in reviews)), 'conflicting_pairs': len(conflicts),
        'network_requests': 0, 'annual_changes': 0, 'final_task_closure': False})
    return proposals, reviews


def run() -> None:
    with (FOLDER / 'application.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        proposals, reviews = prepare()
        atomic_json(FOLDER / 'application_status.json', {'status': 'waiting_for_writer',
                    'proposed_journal_years': len(proposals), 'network_requests': 0})
        result = apply_proposals(proposals, FOLDER,
            [r for r in reviews if r['status'] != 'dated_publisher_identity_confirmed'],
            rule_status='dated_native_jats_publication_statement',
            reason='Exact native identity, explicit named publisher and agreeing electronic/copyright year; independent of research eligibility',
            route='native_jats_copyright', apply_pairs={(r['source_id'], r['year']) for r in proposals})
        atomic_json(FOLDER / 'application_status.json', {'status': 'fixed_annual_cache_applied',
                    'result': result, 'network_requests': 0, 'final_task_closure': False})


if __name__ == '__main__':
    run()
