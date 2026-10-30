"""Validate dated native copyright proofs and their current annual/decision labels."""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json

import pyarrow.parquet as pq
from lxml import etree

from .aps_annual_cache import copyright_year
from .jats_annual_copyright import named_publisher
from .supplement_queue import ROOT, connect
from .supplement_sources import atomic_json
from .three_journal_apply import read_csv


def run(name: str, selected_pairs: set[tuple[str, int]] | None = None) -> dict:
    if name not in {'aps_annual_cache', 'jats_annual_copyright', 'plos_annual_cache', 'accepted_imprint_annual', 'frontiers_annual_cache', 'annual_nlm/additional_imprints', 'additional_dated_imprints'}:
        raise ValueError('Unsupported evidence format')
    folder = ROOT / 'supplement' / name
    with (ROOT / 'supplement/checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rules = {(r['source_id'], r['year']): r for r in json.loads((folder / 'rules.json').read_text())}
        if selected_pairs is not None:
            assert selected_pairs <= set(rules)
            rules = {pair:rule for pair,rule in rules.items() if pair in selected_pairs}
        annual = {(r['source_id'], int(r['year'])): r for r in read_csv(ROOT / 'fields/journal_publisher_by_year.csv')}
        journals = {r['source_id']: r for r in read_csv(ROOT / 'fields/journal_registry.csv')}
        checked_proofs = 0
        for pair, rule in rules.items():
            assert annual[pair]['nature_portfolio_year'] == 'False'
            proofs = ([rule] if name == 'annual_nlm/additional_imprints' else rule['proofs']
                      if name in {'frontiers_annual_cache', 'additional_dated_imprints'} else [rule['first_dated_proof'], rule['last_dated_proof']])
            for proof in proofs:
                raw = gzip.decompress((ROOT / proof['raw_receipt']).read_bytes())
                assert hashlib.sha256(raw).hexdigest() == proof['raw_sha256']
                if name != 'annual_nlm/additional_imprints':
                    assert int(proof.get('publication_date', proof.get('online_date', ''))[:4]) == pair[1]
                if name == 'aps_annual_cache':
                    articles = json.loads(raw)['response']['data']
                    matched = [a for a in articles if a.get('identifiers', {}).get('doi', '').lower() == proof['doi']]
                    assert len(matched) == 1 and copyright_year(matched[0]) == pair[1]
                    assert matched[0]['rights'] == proof['copyright']
                    assert matched[0]['date'] == proof['publication_date']
                elif name == 'annual_nlm/additional_imprints':
                    from .annual_nlm import imprint_interval
                    from .annual_nlm_additional import ENTITIES
                    tree = etree.fromstring(raw, etree.XMLParser(resolve_entities=False, no_network=True))
                    records = [r for r in tree.findall('NLMCatalogRecord') if r.findtext('NlmUniqueID') == rule['nlm_id']]
                    assert len(records) == 1 and rule['publisher'] in ENTITIES
                    imprints = records[0].findall('./PublicationInfo/Imprint[@ImprintType="Current"]')
                    assert any(i.findtext('Entity') == rule['publisher'] and i.findtext('ImprintFull') == rule['imprint_full']
                               and (imprint_interval(i.findtext('DateIssued') or '') or imprint_interval(i.findtext('ImprintFull') or '')) == tuple(rule['interval']) for i in imprints)
                    assert rule['interval'][0] < pair[1] < rule['interval'][1]
                    registry = journals[pair[0]]
                    valid_issns = {n.text for n in records[0].findall('ISSN') if n.get('ValidYN') == 'Y'}
                    assert valid_issns & set(json.loads(registry['issn'] or '[]'))
                elif name == 'additional_dated_imprints':
                    from .additional_dated_imprints import ENTITIES, date_supported, native_imprint
                    from .supplement_sources import jats_metadata
                    from .supplement_apply import signature_match
                    assert hashlib.sha256((ROOT / rule['context_receipt']).read_bytes()).hexdigest() == rule['context_sha256']
                    native = native_imprint(raw, proof['doi'], proof.get('native_family', 'archived_jats'))
                    native['issns'] = native['issn']
                    assert set(native['publisher']) == set(proof['publishers']) and set(proof['publishers']) <= ENTITIES
                    assert proof['doi'] in {v.lower() for v in native['doi']}
                    assert signature_match({'title':proof['title'],'journal_name':proof['journal'],'source_id':pair[0]}, native, set(proof['registry_issns']))
                    assert date_supported(native, proof['online_date'], proof['date_evidence'])
                    for source in proof['date_evidence']:
                        p = ROOT / 'fields/qualification' / source['receipt']
                        b = p.read_bytes()
                        if p.suffix == '.gz': b = gzip.decompress(b)
                        assert hashlib.sha256(b).hexdigest() == source['checked_raw_sha256']
                elif name == 'frontiers_annual_cache':
                    from .frontiers_annual_cache import matches
                    from .supplement_sources import parse_publisher
                    journal = {'journal_name': rule['journal'], 'issn': json.dumps(proof['registry_issns'])}
                    assert matches(parse_publisher(raw), proof, journal)
                elif name == 'accepted_imprint_annual':
                    from .fig8b_qualification_decide import portfolio_label
                    label, evidence_ids = portfolio_label('', proof['evidence'], pair[1])
                    assert label == 'False' and evidence_ids
                    for source in proof['evidence']:
                        source_path = ROOT / 'fields/qualification' / source['receipt']
                        source_raw = source_path.read_bytes()
                        if source_path.suffix == '.gz':
                            source_raw = gzip.decompress(source_raw)
                        assert hashlib.sha256(source_raw).hexdigest() == source['checked_raw_sha256']
                elif name == 'plos_annual_cache':
                    docs = json.loads(raw)['response']['docs']
                    matched = [d for d in docs if d.get('id', '').lower() == proof['doi']]
                    assert len(matched) == 1
                    native = matched[0]
                    assert native['publisher'] == proof['publisher'] == 'Public Library of Science'
                    assert native['publication_date'][:10] == proof['publication_date']
                    assert native['eissn'] == proof['eissn'] and native['title_display'] == proof['title']
                    evidence_db = connect()
                    primary = [json.loads(p[0]) for p in evidence_db.execute(
                        "SELECT payload FROM current_evidence_index WHERE work_id=? AND kind='primary'", (proof['work_id'],))]
                    evidence_db.close()
                    assert any(e.get('identity') == 'matched' and e.get('online_date') == proof['publication_date']
                               and e.get('url') and 'api.plos.org' not in e['url']
                               and e.get('receipt') for e in primary)
                else:
                    document = raw + b'</article>' if raw.rstrip().endswith(b'</front>') else raw
                    tree = etree.fromstring(document, etree.XMLParser(resolve_entities=False, no_network=True))
                    actual = named_publisher(tree, pair[1])
                    assert actual == (rule['publisher'], proof['statement'])
                    assert proof['doi'] in {v.lower() for v in tree.xpath('./front/article-meta/article-id[@pub-id-type="doi"]/text()')}
                checked_proofs += 1
        db = connect()
        sids = sorted({k[0] for k in rules})
        shards = [r[0] for r in db.execute('SELECT DISTINCT shard FROM papers WHERE source_id IN (' + ','.join('?' for _ in sids) + ')', sids)]
        db.close()
        verified = included = 0
        for shard in shards:
            for row in pq.read_table(ROOT / 'fields/qualification/decisions' / shard, columns=[
                    'source_id', 'online_year', 'nature_portfolio_at_verified_online_year', 'eligibility_status']).to_pylist():
                if (row['source_id'], row['online_year']) in rules:
                    assert row['nature_portfolio_at_verified_online_year'] == 'False'
                    verified += 1
                    included += row['eligibility_status'] == 'included_verified_research'
        result = {'annual_rules_checked': len(rules), 'raw_dated_proofs_checked': checked_proofs,
                  'verified_year_decisions_checked': verified, 'included_decisions_checked': included,
                  'inconsistent_labels': 0, 'network_requests': 0,
                  'scope': name + '; not a global qualification or formal-comparison completeness claim'}
        atomic_json(folder / ('validation.json' if selected_pairs is None else 'incremental_validation.json'), result)
        print(json.dumps(result), flush=True)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', choices=['aps_annual_cache', 'jats_annual_copyright', 'plos_annual_cache', 'accepted_imprint_annual', 'frontiers_annual_cache', 'annual_nlm/additional_imprints', 'additional_dated_imprints'])
    run(parser.parse_args().name)
