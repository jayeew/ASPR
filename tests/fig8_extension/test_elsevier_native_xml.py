from experiments.fig8_extension.elsevier_native_xml import parse


def test_footnote_is_not_affiliation_and_acceptance_is_not_online():
    raw=b'''<article xmlns="http://www.elsevier.com/xml/ja/dtd" xmlns:ce="http://www.elsevier.com/xml/common/dtd" xmlns:sa="http://www.elsevier.com/xml/common/struct-aff/dtd" docsubtype="rev"><item-info><ce:doi>10.1/XYZ</ce:doi></item-info><head><ce:title>Review</ce:title><ce:date-accepted year="2024" month="1" day="2"/><ce:author-group><ce:author><ce:given-name>A</ce:given-name><ce:surname>B</ce:surname><ce:cross-ref refid="aff1"/><ce:cross-ref refid="fn1"/></ce:author><ce:affiliation id="aff1"><ce:label>a</ce:label><ce:textfn>University, China</ce:textfn><sa:affiliation><sa:country>China</sa:country></sa:affiliation></ce:affiliation></ce:author-group></head></article>'''
    r=parse(raw)
    assert r['online_date']=='' and r['article_types']==['Review article']
    assert r['doi']==['10.1/xyz']
    assert r['authors'][0]['unresolved_affiliation_refs']==[]
    assert r['authors'][0]['affiliations'][0]['text']=='University, China'
