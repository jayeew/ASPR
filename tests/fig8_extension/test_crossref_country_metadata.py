from experiments.fig8_extension.crossref_country_metadata import parse


def test_batch_identity_and_deposited_links_only() -> None:
    item = {'DOI': '10.1/a', 'title': ['Paper'], 'ISSN': ['1234'],
            'container-title': ['Journal'], 'author': [
                {'given': 'Jane', 'family': 'Doe', 'affiliation': [{'name': 'Lab, City, 12345 France'}]},
                {'name': 'Collaboration', 'affiliation': [{'name': 'Other'}]}]}
    record = {'http_status': 200, 'response': {'message': {'items': [item, {'DOI': '10.1/b'}]}}}
    parsed = parse(record, '10.1/a')
    assert parsed['doi'] == ['10.1/a']
    assert parsed['authors'] == [{'name': 'Jane Doe', 'affiliations': [{'text': 'Lab, City, 12345 France', 'countries': []}]}]
    assert 'byline_complete' not in parsed
    assert parse(record, '10.1/absent') == {}
    record['response']['message']['items'].append(item)
    assert parse(record, '10.1/a') == {}
    record['http_status'] = 403
    assert parse(record, '10.1/b') == {}
