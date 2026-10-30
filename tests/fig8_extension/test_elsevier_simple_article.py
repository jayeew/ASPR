from experiments.fig8_extension.elsevier_native_xml import parse


def test_explicit_comment_header_and_target_doi_exclude_related_doi():
    raw=b'''<simple-article xmlns="http://www.elsevier.com/xml/ja/dtd" xmlns:ce="http://www.elsevier.com/xml/common/dtd" docsubtype="dis">
    <item-info><ce:doi>10.1/comment</ce:doi><related-article><ce:doi>10.1/original</ce:doi></related-article></item-info>
    <simple-head><ce:dochead><ce:textfn>Comment</ce:textfn></ce:dochead><ce:title>A comment</ce:title>
    <ce:date-accepted year="2025" month="7" day="25"/></simple-head></simple-article>'''
    result=parse(raw)
    assert result['doi']==['10.1/comment']
    assert result['article_types']==['Comment']
    assert result['native_article_type']=='dis'
    assert result['online_date']==''


def test_unknown_code_not_inferred_from_title():
    raw=b'''<simple-article xmlns="http://www.elsevier.com/xml/ja/dtd" xmlns:ce="http://www.elsevier.com/xml/common/dtd" docsubtype="dis">
    <item-info><ce:doi>10.1/a</ce:doi></item-info><simple-head><ce:title>Comment on something</ce:title></simple-head></simple-article>'''
    assert parse(raw)['article_types']==['dis']
