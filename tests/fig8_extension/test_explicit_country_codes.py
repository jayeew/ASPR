from experiments.fig8_extension.nature_backfill_registry import normalized
from experiments.fig8_extension.nature_backfill_resolve import country_names
from experiments.fig8_extension.supplement_apply import countries


def test_explicit_iso_code_survives_ambiguous_name_removal() -> None:
    names = country_names()
    names.pop(normalized('Georgia'), None)
    address = {'text': 'Institution, Georgia', 'countries': [{'text': 'Georgia', 'code': 'GE'}]}
    assert countries([address], names) == ['GE']
    address['countries'][0]['code'] = ''
    assert countries([address], names) == []
    address['countries'][0]['code'] = 'ZZ'
    assert countries([address], names) == []


def test_explicit_code_does_not_bypass_territory_conflict_guard() -> None:
    assert countries([{'text': 'Institution, Hong Kong, China',
                       'countries': [{'text': 'China', 'code': 'CN'}]}], country_names()) == []
