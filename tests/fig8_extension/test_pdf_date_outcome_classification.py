from experiments.fig8_extension.scoap3_pdf_date_audit import classify


def test_unread_pdf_not_classified_as_missing_date():
    assert classify({'status': 'no_explicit_date_confirmed', 'evidence': [{'status': 'pdf_not_available'}]}) == 'retrieval_or_parse_unresolved'
    assert classify({'status': 'no_explicit_date_confirmed', 'evidence': [{'status': 'reviewed_restricted_path_or_doi_not_requested'}]}) == 'not_requested_previously_restricted_path_or_doi'
    assert classify({'status': 'no_explicit_date_confirmed', 'evidence': [{'status': 'explicit_online_date_absent_or_conflicting'}]}) == 'read_first_page_date_absent_or_conflicting'
