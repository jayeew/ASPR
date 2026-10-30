from experiments.fig8_extension.aps_authors import parse_authors
from experiments.fig8_extension.supplement_apply import complete_byline, enrich_authors


def test_native_links_ror_and_group_author_without_parent_inference():
    article = {'authors': [
        {'type': 'Person', 'name': 'Jane Doe', 'affiliationIds': ['a1']},
        {'type': 'Collaboration', 'name': 'Example Collaboration', 'affiliationIds': []}],
        'affiliations': [{'id': 'a1', 'name': 'Physics, <a href="https://ror.org/example">Example Institute</a>, France'}]}
    source = {**parse_authors(article), 'evidence_receipt': 'native.json.gz'}
    authors, complete = complete_byline([], source)
    enriched, _ = enrich_authors(authors, source, {'https://ror.org/example': ['I1']}, {}, {'france': 'FR'})
    assert complete and len(enriched) == 2
    assert enriched[0]['countries'] == ['FR']
    assert enriched[0]['institution_ids_or_verified_local_entities'] == ['I1']
    assert enriched[0]['publisher_affiliation_slots'][0]['university_ownership_status'] == 'not_inferred_from_registry'
    assert enriched[1]['nonpersonal_verified']
    article['authors'][0]['affiliationIds'] = ['missing']
    parsed = parse_authors(article)
    assert parsed['byline_has_etal'] and parsed['authors'][0]['unresolved_affiliation_refs'] == ['missing']
    article['authors'][0]['type'] = 'Unknown'
    assert parse_authors(article)['byline_has_etal']


def test_single_record_envelope_and_entity_title(tmp_path) -> None:
    import gzip
    import json
    from experiments.fig8_extension.aps_authors import source_record
    article = {'identifiers': {'doi': '10.1103/example'}, 'title': {'value': 'S&amp;P500 &lt; signal'},
               'journal': {'id': 'PRB', 'name': 'Physical Review B'}, 'authors': [], 'affiliations': []}
    path = tmp_path / 'receipt.json.gz'
    path.write_bytes(gzip.compress(json.dumps({'http_status': 200, 'response': {'data': article}}).encode()))
    record, raw = source_record(tmp_path, 'receipt.json.gz', '10.1103/example')
    assert record['parsed']['title'] == ['S&P500 < signal']
    assert record['parsed']['doi'] == ['10.1103/example']
    assert raw
