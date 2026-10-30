from lxml import etree
from experiments.fig8_extension.copernicus_bulk import parse_article


def test_nlm_namespace_and_publisher_rid_preserve_links():
    raw=b'''<article xmlns="http://dtd.nlm.nih.gov/2.0/xsd/archivearticle" article-type="research-article"><front><journal-meta><journal-title-group><journal-title>Journal</journal-title></journal-title-group></journal-meta><article-meta><article-id pub-id-type="doi">10.1/test</article-id><title-group><article-title>Title</article-title></title-group><contrib-group><contrib contrib-type="author" rid="aff1 missing"><name><given-names>Jane</given-names><surname>Doe</surname></name></contrib></contrib-group><aff id="aff1">Lab, United Kingdom</aff><pub-date><year>2024</year></pub-date></article-meta></front></article>'''
    original=etree.fromstring(raw)
    parsed=parse_article(original)
    assert parsed['doi']==['10.1/test']
    assert parsed['authors'][0]['affiliations'][0]['text']=='Lab, United Kingdom'
    assert parsed['authors'][0]['unresolved_affiliation_refs']==['missing']
    assert parsed['dates'][0]['attributes']=={}  # No invented online-date designation.
    assert etree.tostring(original)==raw


def test_invalid_source_is_retained_without_recovery(tmp_path):
    import gzip
    import json
    from experiments.fig8_extension.copernicus_bulk import read_xml
    path=tmp_path/'source.xml.gz'
    path.write_bytes(gzip.compress(b'<root>A & B</root>'))
    record={'raw_receipt':str(path)}
    assert read_xml(record,tmp_path,{'set':'test'}) is None
    event=json.loads((tmp_path/'parse_errors.jsonl').read_text())
    assert event['qualification_not_resolved']
    assert event['receipt']==record


def test_only_dc_creator_ampersands_are_normalized(tmp_path):
    import gzip
    from experiments.fig8_extension.copernicus_bulk import read_xml
    path=tmp_path/'raw.xml.gz'
    raw=b'<root xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:creator>A & B</dc:creator><x><![CDATA[A & B]]></x></root>'
    path.write_bytes(gzip.compress(raw))
    tree=read_xml({'raw_receipt':str(path)},tmp_path,{'set':'gi'})
    assert tree[0].text=='A & B'
    assert tree[1].text=='A & B'
    assert gzip.decompress(path.read_bytes())==raw
    assert (tmp_path/'normalizations.jsonl').exists()
