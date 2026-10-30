from experiments.fig8_extension.nature_backfill_resolve import address_country, country_names
from experiments.fig8_extension.nature_backfill_registry import normalized


def test_korea_requires_the_reviewed_complete_signed_location() -> None:
    names=country_names();names.pop(normalized('Korea'),None)
    assert address_country('Kyungpook National University, Daegu, Korea',names)=='KR'
    assert address_country('Seoul National University, Seoul, Korea',names)=='KR'
    assert address_country('Unknown Institute, Korea',names) is None
    assert address_country('Seoul National University',names) is None
    assert address_country('Seoul National University, foreign branch, Korea',names) is None
