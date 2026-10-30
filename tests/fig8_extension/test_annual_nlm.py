from experiments.fig8_extension.annual_nlm import imprint_interval


def test_explicit_dates_only():
    assert imprint_interval('<Oct. 4, 1991- > : Washington')==(1991,9999)
    assert imprint_interval('[2013]-')==(2013,9999)
    assert imprint_interval('<2016-2024> : Thousand Oaks')==(2016,2024)
    assert imprint_interval('Cambridge, Ma : Cell Press') is None
    assert imprint_interval('Cambridge : Publisher, founded 1974') is None


def test_applied_additional_evidence_is_not_overwritten(tmp_path):
    import pytest
    from experiments.fig8_extension.annual_nlm import propose
    folder=tmp_path/'supplement/annual_nlm';folder.mkdir(parents=True)
    (folder/'lookup_applied.json').write_text('{}')
    with pytest.raises(RuntimeError,match='immutable'):
        propose(tmp_path,'lookup_proposals.json')
