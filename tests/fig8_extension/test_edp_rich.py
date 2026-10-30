from experiments.fig8_extension.edp_rich_candidates import extract
from experiments.fig8_extension.fig8b_qualification_decide import electronic_date


def test_edp_oai_requires_both_header_and_native_doi() -> None:
    raw = b'''<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><GetRecord><record>
    <header><identifier>oai:edpsciences.org:dkey/10.1051/test</identifier></header><metadata>
    <article article-type="research-article"><front><journal-meta><journal-title>Test</journal-title>
    <issn>1234-5678</issn></journal-meta><article-meta><article-id pub-id-type="doi">10.1051/test</article-id>
    <title-group><article-title>Target</article-title></title-group>
    <pub-date publication-format="electronic" date-type="pub"><year>2024</year><month>04</month><day>04</day></pub-date>
    <history><date date-type="accepted"><year>2023</year></date></history>
    </article-meta></front></article></metadata></record></GetRecord></OAI-PMH>'''
    parsed = extract(raw, '10.1051/test')
    assert parsed is not None
    assert electronic_date(parsed['dates'])[0] == '2024-04-04'
    assert extract(raw, '10.1051/absent') is None
    wrong = raw.replace(b'>10.1051/test</article-id>', b'>10.1051/other</article-id>')
    assert extract(wrong, '10.1051/test') is None


def test_publisher_evidence_keeps_native_categories() -> None:
    from experiments.fig8_extension.publisher_delta_apply import publisher_items
    record = {'source': 'publisher_jats', 'route': 'edp_native', 'url': 'http://oai.edpsciences.org/',
              'parsed': {'title': ['Target'], 'issns': ['1234-5678'], 'journal_titles': ['Test'],
                         'article_types': ['research-article'], 'online_date': '2024-04-04',
                         'date_precision': 'day', 'evidence_categories': ['Review']}}
    item = publisher_items(record)[0]
    assert item['categories'] == ['Review']
    assert item['kind'] == 'research-article'
    del record['parsed']['evidence_categories']
    assert publisher_items(record)[0]['categories'] == ['research-article']


def test_scoped_author_accent_alias_does_not_expand_initials() -> None:
    from experiments.fig8_extension.edp_author_diacritics import PAIRS, fold
    assert len(PAIRS) == 4
    assert all(fold(a) == fold(b) for a, b in PAIRS.values())
    assert fold('Paola Francesca Antonietti') != fold('Paola F. Antonietti')
    assert fold('Tony Lelièvre A') != fold('Tony Lelièvre')
