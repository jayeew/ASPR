from experiments.fig8_extension.acs_pdf_front import extract


def test_native_front_date_requires_publisher_identity() -> None:
    text = 'Articlepubs.acs.org/biochemistry\n© 2025 American Chemical Society\nhttps://doi.org/10.1021/acs.biochem.5c00082\nReceived: February 10, 2025\nPublished: May 2, 2025\n'
    result = extract(text, '10.1021/acs.biochem.5c00082')
    assert result['online_date'] == '2025-05-02'
    assert result['kind'] == 'Article'  # No generic-to-research promotion.
    assert extract(text, '10.1021/other')['status'] == 'native_doi_unconfirmed'
    assert extract(text.replace('Published: May 2, 2025', 'Submitted on 7 Aug 2025'), '10.1021/acs.biochem.5c00082')['status'] == 'native_published_date_absent_or_conflicting'
    assert extract(text.replace('Articlepubs.acs.org/biochemistry', 'HAL repository'), '10.1021/acs.biochem.5c00082')['status'] == 'native_publisher_header_unconfirmed'
