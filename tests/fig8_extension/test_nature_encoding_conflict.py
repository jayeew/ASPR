from __future__ import annotations

from copy import deepcopy

from experiments.fig8_extension.nature_encoding_conflict import repair


def test_decoding_conflict_keeps_official_identity_and_real_conflicts():
    raw = '''<html><meta name="citation_author" content="Liao, Junhao"><ul data-test="authors-list"><li><a data-test="author-name">Junhao Liao</a><sup><a href="#Aff1">1</a></sup></li></ul><li id="Aff1"><p class="c-article-author-affiliation__address">University, People’s Republic of China</p></li></html>'''.encode('utf-8')
    official = {'text': 'University, Peopleâ\x80\x99s Republic of China', 'institution_ids': ['I1'], 'official_identity_evidence': 'official.json', 'university_ownership_status': 'officially_confirmed_university_self'}
    incoming = {'text': 'University, People’s Republic of China', 'institution_ids': ['I1'], 'country_codes': ['CN']}
    author = {'name': 'Junhao Liao', 'countries': ['CN'], 'publisher_affiliation_source': 'raw.gz', 'publisher_affiliation_slots': [official], 'official_identity_source_conflict': True, 'conflicting_publisher_slots': {'receipt': 'raw.gz', 'slots': [incoming]}}
    result = repair(author, raw, 'raw.gz')
    assert result['official_identity_source_conflict'] is False
    assert result['publisher_affiliation_slots'][0]['official_identity_evidence'] == 'official.json'
    assert result['countries'] == author['countries']
    assert repair(result, raw, 'raw.gz') == result
    conflict = deepcopy(author)
    conflict['conflicting_publisher_slots']['slots'][0]['institution_ids'] = ['I2']
    assert repair(conflict, raw, 'raw.gz') == conflict
    assert repair(author, raw, 'other.gz') == author
