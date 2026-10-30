from lxml import etree
from experiments.fig8_extension.pensoft_bulk import extract


def test_available_date_not_automatically_online_and_author_links_retained():
    raw=b'''<record xmlns="http://www.openarchives.org/OAI/2.0/"><metadata><mods xmlns="http://www.loc.gov/mods/v3"><identifier type="doi">10.1/abc</identifier><titleInfo><title>A title</title></titleInfo><genre>Review</genre><extension><dateAvailable>2024-01-02</dateAvailable></extension><name><namePart>Smith, A</namePart><affiliation>University, UK</affiliation><role><roleTerm>author</roleTerm></role></name></mods></metadata></record>'''
    parsed = extract(etree.fromstring(raw))
    assert parsed['online_date'] == ''
    assert parsed['date_available'] == ['2024-01-02']
    assert parsed['article_types'] == ['Review']
    assert parsed['authors'][0]['affiliations'] == ['University, UK']
    assert parsed['authors'][0]['roles'] == ['author']
    assert extract(etree.fromstring(b'<record/>')) == {}
