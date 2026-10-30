from experiments.fig8_extension.aip_pdf_front import extract


def test_native_aip_type_and_date_ignore_repository_submission() -> None:
    pages = ['HAL Submitted on 20 July 2025\nJournal of Applied Physics',
             'RESEARCH ARTICLE | MARCH 17 2025\nTarget title\nhttps://doi.org/10.1063/5.12345',
             'Journal of Applied Physics ARTICLE pubs.aip.org/aip/jap\nTarget title\n'
             'Cite as: doi: 10.1063/5.12345\nPublished Online: 17 March 2025']
    result = extract(pages, '10.1063/5.12345', 'Target title', 'Journal of Applied Physics')
    assert result['online_date'] == '2025-03-17'
    assert result['kind'] == 'RESEARCH ARTICLE'
    assert extract(pages, '10.1063/5.99999', 'Target title', 'Journal of Applied Physics')['status'] != 'native_aip_front_confirmed'
    assert extract(pages[:2], '10.1063/5.12345', 'Target title', 'Journal of Applied Physics')['status'] != 'native_aip_front_confirmed'
    bad = [*pages[:2], pages[2].replace('17 March', '18 March')]
    assert extract(bad, '10.1063/5.12345', 'Target title', 'Journal of Applied Physics')['status'] != 'native_aip_front_confirmed'
    bad = [*pages[:2], 'Accepted Manuscript\n' + pages[2]]
    assert extract(bad, '10.1063/5.12345', 'Target title', 'Journal of Applied Physics')['status'] != 'native_aip_front_confirmed'
