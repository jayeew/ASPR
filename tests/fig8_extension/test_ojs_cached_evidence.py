from lxml import etree
from experiments.fig8_extension import ojs_cached_evidence as module
from experiments.fig8_extension.fig8b_qualification_build import match


def test_dc_date_not_online_and_generic_article_not_research(monkeypatch):
    entry=etree.fromstring(b'''<record xmlns="http://www.openarchives.org/OAI/2.0/" xmlns:dc="http://purl.org/dc/elements/1.1/"><metadata><dc:title>Paper</dc:title><dc:source>Journal; Vol 2</dc:source><dc:source>1234-5678</dc:source><dc:date>2023-01-01</dc:date><dc:type>Peer-reviewed Article</dc:type></metadata></record>''')
    monkeypatch.setattr(module,'entries',lambda _: {'native':entry})
    value=module.dc_evidence({'raw_receipt':'supplement/raw.gz','identifier':'native'},'https://example.org/oai','Paper')[0]
    assert value['online_date']==''
    assert value['declared_dc_dates_not_verified_online']==['2023-01-01']
    matched=match({'work_id':'W1','title':'Paper','journal_name':'Journal','source_id':'S1'},value,{'issn':'["1234-5678"]'})
    assert matched['identity']=='matched' and matched['class']=='unresolved_type'
    assert value['receipt']=='../../supplement/raw.gz'
