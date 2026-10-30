from experiments.fig8_extension.supplement_sources import parse_publisher


def test_frontiers_acceptance_meta_is_not_publication_date():
    raw = b'''<html><meta name="citation_doi" content="10.3389/test.2022.1">
    <meta name="citation_online_date" content="2022/12/20">
    <div><p>Accepted</p><p>20 December 2022</p></div>
    <div><p>Published</p><p>06 January 2023</p></div></html>'''
    result = parse_publisher(raw)
    assert result['online_date'] == '2023-01-06'
    assert result['date_field_review']['citation_online_date'] == ['2022/12/20']
    assert parse_publisher(raw.replace(b'<p>Published</p>', b'<p>Other</p>'))['online_date'] == ''
