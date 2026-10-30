from experiments.fig8_extension.rrpharm_pdf_metadata import extract


def test_explicit_pdf_header_type_and_published_date_only():
    text='DOI 10.18413/rrpharmacolog y.11.759\n\n Research Article\nAccepted 27 October 2025 Published 29 December 2025\n'
    assert extract(text,'10.18413/rrpharmacology.11.759')['online_date']=='2025-12-29'
    assert extract(text,'10.1/other')['status']=='pdf_header_doi_unconfirmed'
    assert extract(text.replace('Published','Received'),'10.18413/rrpharmacology.11.759')['status']=='pdf_type_or_published_date_unconfirmed'
    assert extract(text.replace('Research Article','Experimental Pharmacology'),'10.18413/rrpharmacology.11.759')['status']=='pdf_type_or_published_date_unconfirmed'


def test_pdf_header_can_have_large_layout_indentation():
    text=(' '*350+'Journal\n'+' '*350+'UDC\n'+' '*350+'DOI 10.18413/rrpharmacology. 1 0 . 4 9 8\n\nResearch Article\nPublished 29 December 2024\n')
    assert extract(text,'10.18413/rrpharmacology.10.498')['status']=='pdf_front_explicit_type_date'


def test_review_label_whitespace_is_not_missing_type():
    text='DOI 10.1/review\n\nReview  Article\nPublished 25 December 2025\n'
    assert extract(text,'10.1/review')['article_type']=='Review Article'
