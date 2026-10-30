from experiments.fig8_extension.springer_native_xml import parse


def test_explicit_fields_and_author_refs():
    raw=b'''<Publisher><Journal><JournalInfo><JournalTitle>Journal</JournalTitle><JournalElectronicISSN>1234-5678</JournalElectronicISSN></JournalInfo><Article><ArticleInfo ArticleType="OriginalPaper"><ArticleDOI>10.1/ABC</ArticleDOI><ArticleTitle>Title</ArticleTitle><ArticleHistory><OnlineDate><Year>2024</Year><Month>2</Month><Day>29</Day></OnlineDate></ArticleHistory></ArticleInfo><AuthorGroup><Author AffiliationIDS="A missing"><AuthorName><GivenName>First</GivenName><FamilyName>Last</FamilyName></AuthorName></Author><Affiliation ID="A"><OrgName>University</OrgName><OrgAddress><City>City</City><Country Code="GB">UK</Country></OrgAddress></Affiliation></AuthorGroup></Article></Journal></Publisher>'''
    r=parse(raw)
    assert r['doi']==['10.1/abc'] and r['online_date']=='2024-02-29'
    assert r['authors'][0]['name']=='First Last'
    assert r['authors'][0]['unresolved_affiliation_refs']==['missing']
    assert r['authors'][0]['affiliations'][0]['countries']==[{'text':'UK','code':'GB'}]
    assert parse(raw.replace(b'OnlineDate',b'RegistrationDate'))['online_date']==''
    assert parse(raw.replace(b'<Day>29',b'<Day>31'))['date_error']
    assert parse(b'<article/>')=={}


def test_equation_alternatives_are_not_concatenated():
    from lxml import etree
    from experiments.fig8_extension.springer_native_xml import text
    node=etree.fromstring(b'<ArticleTitle>at <InlineEquation><EquationSource Format="MATHML"><math><msqrt><mi>s</mi></msqrt></math></EquationSource><EquationSource Format="TEX">$$ \\sqrt{s} $$</EquationSource></InlineEquation> = 13 TeV</ArticleTitle>')
    assert text(node)=='at $$ \\sqrt{s} $$ = 13 TeV'
    assert len(node.findall('.//EquationSource'))==2
