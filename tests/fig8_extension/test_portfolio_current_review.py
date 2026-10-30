from __future__ import annotations

from experiments.fig8_extension.portfolio_current_review import catalogue_rule


def catalogue(comment: str, imprint: str = 'Nature Portfolio') -> dict:
    return {2024:{'index':{'2948-197X':[{'Title':'npj Imaging','row_number':2,
            'Imprint':imprint,'Comments':comment}]},'url':'https://example.org/catalogue',
            'receipt':'catalogue.xlsx','sha256':'hash'}}


def test_explicit_new_start_supports_previous_year_membership() -> None:
    registry={'journal_name':'npj Imaging','issn':'["2948-197X"]'}
    result=catalogue_rule('S1',2023,registry,catalogue('New Start in 2023.'))
    assert result['label']=='True'
    assert result['catalogue_year']==2024


def test_current_membership_without_start_note_is_not_backdated() -> None:
    registry={'journal_name':'npj Imaging','issn':'["2948-197X"]'}
    assert catalogue_rule('S1',2023,registry,catalogue('')) is None


def test_name_or_issn_mismatch_is_not_membership_evidence() -> None:
    registry={'journal_name':'Other imaging','issn':'["2948-197X"]'}
    assert catalogue_rule('S1',2024,registry,catalogue('')) is None


def test_springer_group_is_not_nature_portfolio() -> None:
    registry={'journal_name':'npj Imaging','issn':'["2948-197X"]'}
    assert catalogue_rule('S1',2024,registry,catalogue('', 'Springer'))['label']=='False'


def test_namespaced_oai_selects_exact_article_and_preserves_math() -> None:
    from lxml import etree
    from experiments.fig8_extension.portfolio_current_review import native_article
    raw = b'''<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><record><article xmlns="https://jats.nlm.nih.gov/ns/archiving/1.4/" xmlns:m="http://www.w3.org/1998/Math/MathML"><front><article-meta><article-id pub-id-type="doi">10.1/right</article-id><m:math><m:mi>x</m:mi></m:math></article-meta></front></article></record><record><article><front><article-meta><article-id pub-id-type="doi">10.1/other</article-id></article-meta></front></article></record></OAI-PMH>'''
    article = native_article(raw, '10.1/right')
    assert article.findtext('./front/article-meta/article-id') == '10.1/right'
    assert article.xpath('count(//*[namespace-uri()="http://www.w3.org/1998/Math/MathML"])') == 2
    assert b'10.1/other' not in etree.tostring(article)
    assert etree.fromstring(etree.tostring(article)).findtext('./front/article-meta/article-id') == '10.1/right'


def test_ambiguous_native_doi_is_not_used() -> None:
    import pytest
    from experiments.fig8_extension.portfolio_current_review import native_article
    article = b'<article><article-id pub-id-type="doi">10.1/same</article-id></article>'
    with pytest.raises(ValueError, match='unique'):
        native_article(b'<root>' + article + article + b'</root>', '10.1/same')
