from experiments.fig8_extension.aimsciences_parse import native_metadata


def test_explicit_online_history_and_numbered_links() -> None:
    page = b'''<html><meta name="citation_doi" content="10.3934/test.1">
    <meta name="citation_date" content="2024-01-01">
    <meta name="citation_issn" content="">
    <meta name="dc.description" content="Original abstract">
    <div class="article-date"><div><b>Early access:</b>August 2, 2023</div>
    <div><b>Published online:</b>September 11, 2023</div></div>
    <ul class="article-author"><li><a type="authors.authorNameEn">A Name</a>
    <sup><a data-tagval="2">2</a></sup></li>
    <li><a type="authors.authorNameEn">B Name</a><sup><a data-tagval="3">3</a></sup></li></ul>
    <li class="article-author-address"><span class="aff-nmuber">1.</span><p>Unit one</p></li>
    <li class="article-author-address"><span class="aff-nmuber">2.</span><p>Unit two</p></li></html>'''
    result = native_metadata(page)
    assert result['online_date'] == '2023-09-11'
    assert result['abstract'] == 'Original abstract'
    assert result['article_types'] == []
    assert result['issns'] == []
    assert [a['text'] for a in result['authors'][0]['affiliations']] == ['Unit two']
    assert result['authors'][1]['unresolved_affiliation_refs'] == ['3']
    assert result['authors'][1]['affiliations'] == []
    assert native_metadata(page.replace(b'Published online:', b'Published:'))['online_date'] == ''
    month = native_metadata(page.replace(b'September 11, 2023', b'September 2023'))
    assert (month['online_date'], month['date_precision']) == ('2023-09', 'month')
