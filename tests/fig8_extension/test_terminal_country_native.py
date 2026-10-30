import gzip
from pathlib import Path

import pytest

from experiments.fig8_extension import terminal_country_apply as module


@pytest.mark.parametrize('receipt,envelope', [
    ('fields/qualification/oai_raw/test.xml.gz', True),
    ('supplement/pmc_cloud/xml_raw/test.json.gz', False),
])
def test_native_xml_identity_survives_oai_namespace_and_cache_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, receipt: str, envelope: bool,
) -> None:
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    article = '''<article xmlns="https://jats.nlm.nih.gov/ns/archiving/1.4/">
      <front><journal-meta><journal-title-group><journal-title>Test</journal-title></journal-title-group></journal-meta>
      <article-meta><article-id pub-id-type="doi">10.1234/test</article-id>
      <title-group><article-title>Sample</article-title></title-group>
      <contrib-group><contrib contrib-type="author"><name><surname>Author</surname><given-names>A</given-names></name>
      <xref ref-type="aff" rid="a1">1</xref></contrib></contrib-group>
      <aff id="a1">University, City, Japan</aff></article-meta></front></article>'''
    raw = ('<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><metadata>' + article + '</metadata></OAI-PMH>') if envelope else article
    path = tmp_path / receipt
    path.parent.mkdir(parents=True)
    path.write_bytes(gzip.compress(raw.encode()))
    result = module.native_metadata(receipt, '10.1234/test')
    assert result['doi'] == ['10.1234/test']
    assert result['authors'][0]['affiliations'][0]['text'] == 'University, City, Japan'


def test_springer_native_author_links_use_existing_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    raw=b'''<Publisher><Journal><JournalInfo><JournalTitle>Journal</JournalTitle><JournalElectronicISSN>1234-5678</JournalElectronicISSN></JournalInfo><Article><ArticleInfo ArticleType="OriginalPaper"><ArticleDOI>10.1/ABC</ArticleDOI><ArticleTitle>Title</ArticleTitle></ArticleInfo><AuthorGroup><Author AffiliationIDS="A"><AuthorName><GivenName>First</GivenName><FamilyName>Last</FamilyName></AuthorName></Author><Affiliation ID="A"><OrgName>University</OrgName><OrgAddress><City>City</City><Country Code="CN">People's Republic of China</Country></OrgAddress></Affiliation></AuthorGroup></Article></Journal></Publisher>'''
    path=tmp_path/'raw.xml.gz';path.write_bytes(gzip.compress(raw))
    result=module.native_metadata('raw.xml.gz','10.1/abc')
    assert result['doi']==['10.1/abc']
    assert result['authors'][0]['name']=='First Last'
    assert result['authors'][0]['affiliations'][0]['countries']==[{'text':"People's Republic of China",'code':'CN'}]


def test_batch_selects_unique_native_article_doi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module,'ROOT',tmp_path)
    def article(doi: str) -> str:
        return f'<article><front><article-meta><article-id pub-id-type="doi">{doi}</article-id><title-group><article-title>Title</article-title></title-group><contrib-group><contrib contrib-type="author"><name><surname>A</surname></name></contrib></contrib-group></article-meta></front></article>'
    p=tmp_path/'batch.xml.gz'
    p.write_bytes(gzip.compress(('<OAI>'+article('10.1/one')+article('10.1/two')+'</OAI>').encode()))
    assert module.native_metadata('batch.xml.gz','10.1/two')['doi']==['10.1/two']
    assert module.native_metadata('batch.xml.gz','10.1/missing')=={}
    p.write_bytes(gzip.compress(('<OAI>'+article('10.1/two')+article('10.1/two')+'</OAI>').encode()))
    assert module.native_metadata('batch.xml.gz','10.1/two')=={}
