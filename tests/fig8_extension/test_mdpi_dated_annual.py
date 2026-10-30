from experiments.fig8_extension.mdpi_dated_annual import dated_publisher


def test_publisher_name_and_same_full_date_both_required():
    item = {'fields': {'publisher': ['Multidisciplinary Digital Publishing Institute'], 'date': ['2023-01-26']}}
    assert dated_publisher(item, {'2023-01-26'})
    assert not dated_publisher(item, {'2024-01-26'})
    assert not dated_publisher(item, {'2023'})
    assert not dated_publisher({'fields': {'publisher': ['Springer Nature'], 'date': ['2023-01-26']}}, {'2023-01-26'})
    assert not dated_publisher({'fields': {'publisher': ['MDPI', 'Springer Nature'], 'date': ['2023-01-26']}}, {'2023-01-26'})
