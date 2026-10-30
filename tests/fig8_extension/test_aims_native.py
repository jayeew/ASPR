"""AIMS header parsing must not promote untyped dates or navigation labels."""
from experiments.fig8_extension.aims_native_collect import parse
from experiments.fig8_extension.aims_native_metadata import metadata


def test_native_header_keeps_date_boundary_and_original_abstract() -> None:
    raw = b'''<html><head>
    <meta name="citation_doi" content="10.3934/example">
    <meta name="citation_title" content="Example">
    <meta name="dc.description" content="&lt;p&gt;Original English abstract.&lt;/p&gt;">
    </head><body><span class="columnli">Research article</span>
    <span class="columnli">Special Issues</span><span class="columnli"></span>
    <span><b>Published:</b>14 November 2023</span></body></html>'''
    result = parse(raw, '10.3934/example')
    assert result['doi_matched']
    assert result['literal_types'] == ['Research article']
    assert result['abstract'] == 'Original English abstract.'
    assert result['online_date'] == ''
    assert '14 November 2023' in result['published_labels_not_online_evidence'][0]
    assert not parse(raw, '10.3934/another')['doi_matched']


def test_native_address_links_never_invent_missing_edges() -> None:
    article = {'doi': '10.3934/test', 'authors': [
        {'contribType': 'author', 'authorNameEn': 'A Person', 'addressTagIds': ['aff2']},
        {'contribType': 'author', 'authorNameEn': 'B Person', 'addressTagIds': ['missing']}],
        'affiliations': [{'tagId': 'aff1', 'labelId': '1', 'affliEn': 'University One'},
                         {'tagId': 'aff2', 'labelId': '2', 'affliEn': 'University Two'}]}
    result = metadata(article, 'raw.json.gz')
    assert result['authors'][0]['affiliations'][0]['text'] == 'University Two'
    assert result['authors'][1]['affiliations'] == []
    assert result['authors'][1]['unresolved_affiliation_refs'] == ['missing']
    article['authors'][0]['addressTagIds'] = []
    article['authors'][1]['addressTagIds'] = []
    assert all(a['unresolved_affiliation_refs'] for a in metadata(article, 'raw')['authors'])
