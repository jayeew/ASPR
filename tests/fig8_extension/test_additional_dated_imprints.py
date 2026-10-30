from experiments.fig8_extension.additional_dated_imprints import date_supported


def test_print_date_needs_exact_matched_electronic_corroboration() -> None:
    native = {'dates': [{'attributes': {'pub-type': 'ppub'}, 'year': '2023', 'month': '1', 'day': '2'}]}
    assert not date_supported(native, '2023-01-02', [])
    evidence = {'identity':'matched', 'online_date':'2023-01-02', 'receipt':'pubmed.xml.gz'}
    assert date_supported(native, '2023-01-02', [evidence])
    assert not date_supported(native, '2023-01-03', [evidence])
    assert not date_supported(native, '2023-01-02', [{**evidence, 'identity':'unmatched'}])
    assert not date_supported(native, '2023-01-02', [{**evidence, 'receipt':''}])


def test_copernicus_front_month_name_keeps_date_corroboration_requirement():
    import pytest
    from experiments.fig8_extension.additional_dated_imprints import native_imprint
    raw = b'<article xmlns="urn:nlm"><front><journal-meta><publisher><publisher-name>Copernicus Publications</publisher-name></publisher></journal-meta><article-meta><article-id pub-id-type="doi">10.5194/test</article-id><pub-date><year>2023</year><month>January</month><day>6</day></pub-date></article-meta></front>'
    native = native_imprint(raw,'10.5194/test','publisher_copernicus')
    assert native['publisher'] == ['Copernicus Publications']
    assert not date_supported(native,'2023-01-06',[])
    assert date_supported(native,'2023-01-06',[{'identity':'matched','online_date':'2023-01-06','receipt':'existing'}])
    with pytest.raises(ValueError):
        native_imprint(raw,'10.5194/other','publisher_copernicus')
