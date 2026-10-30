from lxml import etree

from experiments.fig8_extension.pubmed_author_cache import metadata
from experiments.fig8_extension.secondary_author_cache import enrich


def test_pubmed_explicit_links_and_secondary_authority() -> None:
    article = etree.fromstring(b'''<PubmedArticle><MedlineCitation><PMID>123</PMID><Article>
    <ArticleTitle>A <i>test</i></ArticleTitle><Journal><Title>Example</Title><ISSN>1234-5678</ISSN></Journal>
    <AuthorList><Author><ForeName>Alice</ForeName><LastName>Li</LastName><AffiliationInfo>
    <Affiliation>Example University, Canada</Affiliation></AffiliationInfo></Author>
    <Author><Initials>B</Initials><LastName>Li</LastName></Author>
    <Author><CollectiveName>Research Group</CollectiveName></Author></AuthorList></Article></MedlineCitation>
    <PubmedData><ArticleIdList><ArticleId IdType="doi">10.test/123</ArticleId></ArticleIdList></PubmedData></PubmedArticle>''')
    source = metadata(article, 'cache.xml.gz')
    assert source['title'] == ['A test']
    assert source['doi'] == ['10.test/123']
    assert len(source['authors']) == 1
    person = {'name': 'Alice Li', 'countries': [], 'institution_ids_or_verified_local_entities': []}
    updated, stats = enrich([person], [source], {}, {}, {'canada': 'CA'}, source_label='pubmed', preserve_secondary=True)
    assert updated[0]['countries'] == ['CA']
    assert updated[0]['country_source'] == 'pubmed_explicit_affiliation_country'
    assert updated[0]['secondary_affiliation_evidence']['authority'].startswith('PubMed')
    assert 'publisher_affiliation_slots' not in updated[0]
    assert stats['author_country_filled'] == 1
    assert enrich(updated, [source], {}, {}, {'canada': 'CA'}, preserve_secondary=True)[0] == updated


def test_preserved_secondary_does_not_hide_duplicate_author_identity() -> None:
    existing = {'name': 'Alice Li', 'secondary_affiliation_slots': [{'text': 'Existing'}]}
    empty = {'name': 'Alice Li', 'countries': [], 'institution_ids_or_verified_local_entities': []}
    source = {'authors': [{'name': 'Alice Li', 'affiliations': [{'text': 'Canada'}]}]}
    assert enrich([existing, empty], [source], {}, {}, {}, preserve_secondary=True)[0] == [existing, empty]
