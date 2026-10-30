from copy import deepcopy

from experiments.fig8_extension.epmc_explicit_dates import date_source


def test_explicit_electronic_field_only_and_no_type_promotion() -> None:
    item = {'doi': '10.1/test', 'source': 'MED', 'title': 'Title',
            'electronicPublicationDate': '2023-12-31', 'firstPublicationDate': '2024-01-01',
            'journalInfo': {'journal': {'title': 'Journal', 'issn': '1234-5678'}}}
    record = {'url': 'source', 'http_status': 200, 'response': {'resultList': {'result': [item]}}}
    result = date_source(record, '10.1/test', 'cache')
    assert result['online_date'] == '2023-12-31'
    assert result['kind'] == 'unspecified'
    markup = deepcopy(record); markup['response']['resultList']['result'][0]['title'] = 'CD8<sup>+</sup> T cells'
    assert date_source(markup, '10.1/test', 'cache')['title'] == 'CD8+ T cells'
    for change in [{'electronicPublicationDate': ''}, {'electronicPublicationDate': '2023-02-31'},
                   {'electronicPublicationDate': '2023'}, {'source': 'PPR'}]:
        altered = deepcopy(record); altered['response']['resultList']['result'][0].update(change)
        assert date_source(altered, '10.1/test', 'cache') is None
    duplicate = deepcopy(record); duplicate['response']['resultList']['result'].append(item)
    assert date_source(duplicate, '10.1/test', 'cache') is None
