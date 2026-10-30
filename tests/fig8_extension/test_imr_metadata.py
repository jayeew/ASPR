from experiments.fig8_extension.supplement_sources import parse_publisher, publisher_url


def test_article_bound_original_type_does_not_use_unrelated_jsonld():
    raw=b'''<html><meta name="citation_doi" content="10.31083/j.fbe1504024">
    <meta name="citation_publisher" content="IMR Press"><meta name="citation_article_type" content="Article">
    <script type="application/ld+json">{"@type":"ScholarlyArticle","@id":"https://www.imrpress.com/journal/FBE/15/4/10.31083/j.fbe1504024","genre":"Original Research"}</script></html>'''
    assert parse_publisher(raw)['article_types']==['Original Research']
    assert parse_publisher(raw.replace(b'/10.31083/j.fbe1504024"',b'/wrong"'))['article_types']==['Article']
    assert publisher_url('10.31083/j.fbe1504024','imr')=='https://www.imrpress.com/journal/FBE/15/4/10.31083/j.fbe1504024'
