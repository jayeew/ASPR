from experiments.fig8_extension.publisher_pdf_online_date import extract


def test_publication_date_not_acceptance_or_generic_online_banner():
    text='Available online at www.sciencedirect.com\nAccepted 1 January 2023\nAvailable online 8 February 2023\nhttps://doi.org/10.1/test\n'
    assert extract(text,'10.1/test')['online_date']=='2023-02-08'
    assert extract(text.replace('Available online 8 February 2023',''),'10.1/test')['status']=='explicit_online_date_absent_or_conflicting'
    assert extract(text,'10.1/other')['status']=='first_page_standalone_doi_not_unique_or_mismatched'
    assert extract(text.replace('8 February','31 February'),'10.1/test')['status']=='invalid_explicit_online_date'
