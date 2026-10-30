from lxml import etree

from experiments.fig8_extension.mdpi_oai_harvest import extract


def test_native_type_is_distinct_from_generic_dc_type_and_datestamp() -> None:
    node = etree.fromstring(b'''<record xmlns="http://www.openarchives.org/OAI/2.0/" xmlns:dc="http://purl.org/dc/elements/1.1/">
      <header><identifier>oai:mdpi.com:/1/2/3/4/</identifier><datestamp>2026-01-01</datestamp><setSpec>review-article</setSpec></header>
      <metadata><dc:title>A &amp; B</dc:title><dc:identifier>https://dx.doi.org/10.3390/EXAMPLE</dc:identifier><dc:type>Text</dc:type><dc:date>2023-04-05</dc:date><dc:description>A <i>nested</i> paragraph.</dc:description></metadata>
    </record>''')
    item = extract(node)
    assert item['dois'] == ['10.3390/example']
    assert item['native_type_sets'] == ['review-article']
    assert item['fields']['type'] == ['Text']
    assert item['fields']['date'] == ['2023-04-05']
    assert item['oai_datestamp'] == '2026-01-01'
    assert item['fields']['description'] == ['A nested paragraph.']


def test_type_evidence_requires_doi_title_journal_and_issn() -> None:
    from experiments.fig8_extension.mdpi_cache_apply import evidence

    paper = {'doi': '10.3390/example', 'title': 'A study', 'journal': 'Materials', 'source_id': 'S1'}
    item = {'dois': [paper['doi']], 'fields': {'title': ['A study'], 'source': ['Materials'], 'date': ['2023-04-05']},
            'oai_identifier': 'oai:mdpi.com:/1996-1944/1/2/3/', 'native_type_sets': ['review-article'], 'oai_datestamp': '2024-05-06'}
    receipt = {'raw_receipt': 'test.xml.gz'}
    proof = evidence(item, paper, {'1996-1944'}, receipt, 'hash')
    assert proof['kind'] == 'review-article'
    assert proof['online_date'] == ''
    assert evidence(item, paper, {'0000-0000'}, receipt, 'hash') is None
    assert evidence(item, {**paper, 'title': 'Different study'}, {'1996-1944'}, receipt, 'hash') is None
