from experiments.fig8_extension.wellcome_xml import stage
from experiments.fig8_extension.fig8b_qualification_decide import type_class


def test_explicit_subtype_does_not_inherit_generic_research_type() -> None:
    raw = b'''<article article-type="research-article"><front><journal-meta><journal-title-group><journal-title>Wellcome Open Research</journal-title></journal-title-group><issn>2398-502X</issn></journal-meta><article-meta><article-id pub-id-type="doi">10.12688/wellcomeopenres.123.1</article-id><title-group><article-title>Target</article-title></title-group><article-categories><subj-group><subject>Genome Note</subject><subject>Articles</subject></subj-group></article-categories><pub-date pub-type="epub"><year>2024</year><month>2</month><day>3</day></pub-date></article-meta></front></article>'''
    candidates = [{'doi': '10.12688/wellcomeopenres.123.1', 'title': 'Target', 'work_id': 'W1'}]
    receipt = {'url': 'https://wellcomeopenresearch.org/articles/9-1/v1/xml', 'raw_receipt': 'cache.xml.gz'}
    result = stage(raw, receipt, candidates)
    assert result['status'] == 'parsed'
    assert result['parsed']['article_types'] == ['Genome Note']
    assert type_class('Genome Note', 'publisher_wellcome_xml') != 'research'
    assert result['parsed']['online_date'] == '2024-02-03'
    research = stage(raw.replace(b'Genome Note', b'Research Article'), receipt, candidates)
    assert type_class(research['parsed']['article_types'][0], 'publisher_wellcome_xml') == 'research'
    wrong_version = [{**candidates[0], 'doi': '10.12688/wellcomeopenres.123.2'}]
    assert stage(raw, receipt, wrong_version)['status'] == 'identity_unresolved'
    revised = stage(raw.replace(b'123.1', b'123.2'), receipt, wrong_version)
    assert revised['status'] == 'version_relationship_and_first_online_date_unresolved'
