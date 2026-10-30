from experiments.fig8_extension.affiliation_text_repair import replacements


def test_repeat_does_not_chain_two_distinct_raw_affiliations() -> None:
    from experiments.fig8_extension.affiliation_text_repair import clean_slot
    slot = {'text': '12Institute', 'institution_ids': ['one'], 'country_codes': ['CH']}
    changes = {'12Institute': '2Institute', '2Institute': 'Institute'}
    assert clean_slot(slot, changes, 'source.json')
    assert not clean_slot(slot, changes, 'source.json')
    assert slot == {'text': '2Institute', 'institution_ids': ['one'], 'country_codes': ['CH'],
                    'text_cleanup_source': 'source.json'}


def test_repair_requires_source_doi_and_structural_tags() -> None:
    raw = b'''<article><front><article-meta><article-id pub-id-type="doi">10.1/example</article-id>
    <aff><label>123</label><institution-id institution-id-type="ror">https://ror.org/id</institution-id>CERN, Switzerland</aff>
    <aff>123 Science Road, Institute, Country</aff></article-meta></front></article>'''
    assert replacements(raw, '10.1/other') == {}
    assert replacements(raw, '10.1/example') == {
        '123https://ror.org/idCERN, Switzerland': 'CERN, Switzerland'}


def test_repair_preserves_cooperation_agreement_and_joint_units() -> None:
    raw = b'''<article><front><article-meta><article-id pub-id-type="doi">10.1/example</article-id>
    <aff><label>2</label>Institute covered by a cooperation agreement with CERN</aff>
    <aff><label>3</label>CERN and University, Switzerland</aff></article-meta></front>'''
    assert replacements(raw, '10.1/example') == {
        '2Institute covered by a cooperation agreement with CERN': 'Institute covered by a cooperation agreement with CERN',
        '3CERN and University, Switzerland': 'CERN and University, Switzerland'}


def test_oai_envelope_requires_unique_article_doi() -> None:
    article=b'<article><front><article-meta><article-id pub-id-type="doi">10.1/x</article-id><aff><label>1</label>University</aff></article-meta></front></article>'
    assert replacements(b'<OAI><record>'+article+b'</record></OAI>', '10.1/x') == {'1University':'University'}
    assert replacements(b'<OAI>'+article+article+b'</OAI>', '10.1/x') == {}
    assert replacements(b'<OAI>'+article+b'</OAI>', '10.1/y') == {}
