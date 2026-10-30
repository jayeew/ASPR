from experiments.fig8_extension.scipost_bulk_apply import stage
from experiments.fig8_extension.publisher_delta_apply import publisher_items
from experiments.fig8_extension.fig8b_qualification_decide import type_class


def test_api_date_does_not_invent_original_research_type() -> None:
    record = {'doi': '10.21468/Example', 'title': 'Example', 'url': '/Example',
              'in_volume': 'SciPost Physics Vol. 18', 'publication_date': '2025-02-03'}
    staged = stage({'record': record}, 'supplement/receipt.gz')
    assert staged is not None
    item = publisher_items(staged)[0]
    assert item['online_date'] == '2025-02-03'
    assert item['kind'] == 'unspecified'
    assert type_class(item['kind'], item['family']) == 'unresolved_type'
    record['in_volume'] = 'SciPost Physics Core Vol. 8'
    assert stage({'record': record}, 'supplement/receipt.gz')['parsed']['online_date'] == ''
    record['in_volume'] = None
    assert stage({'record': record}, 'supplement/receipt.gz') is None
