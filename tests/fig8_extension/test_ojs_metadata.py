from lxml import etree
from experiments.fig8_extension.ojs_metadata_harvest import record_info


def test_native_metadata_is_not_qualification_and_deleted_record_is_explicit():
    raw = b'''<record xmlns="http://www.openarchives.org/OAI/2.0/" xmlns:dc="http://purl.org/dc/elements/1.1/"><header><identifier>oai:example:article/9</identifier></header><metadata><dc:title>Original title</dc:title><dc:identifier>https://example.org/index.php/x/article/view/9</dc:identifier><dc:identifier>https://doi.org/10.1234/example</dc:identifier><dc:type>Journal Article</dc:type></metadata></record>'''
    result = record_info(etree.fromstring(raw))
    assert result['declared_dois']==['10.1234/example']
    assert result['titles']==['Original title']
    assert result['status']=='metadata_found_type_requires_review'
    deleted = etree.fromstring(b'<record xmlns="http://www.openarchives.org/OAI/2.0/"><header status="deleted"><identifier>old</identifier></header></record>')
    assert record_info(deleted)['status']=='deleted_or_no_metadata'
