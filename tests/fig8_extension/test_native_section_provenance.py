from experiments.fig8_extension.publisher_delta_apply import publisher_items


def test_section_interpretation_keeps_absent_native_attribute_explicit() -> None:
    record = {'route': 'rcsi_oai_nlm', 'raw_receipt': 'native.xml.gz', 'url': 'https://example.org/oai',
              'parsed': {'title': ['Title'], 'issn': ['0026-3656'], 'journal_titles': ['Microbiology'],
                         'article_type': 'research-article', 'dates': [], 'categories': ['Experimental articles']},
              'native_section_type_evidence': {'native_article_type': '', 'native_section': 'Experimental articles',
                                               'rule_receipt': 'policy.json'}}
    evidence = publisher_items(record)[0]
    assert evidence['kind'] == 'research-article'
    assert evidence['native_section_type_evidence']['native_article_type'] == ''
    assert evidence['receipt'] == 'native.xml.gz'
    record.pop('native_section_type_evidence')
    record['parsed']['article_type'] = ''
    record['parsed']['categories'] = ['Articles']
    assert publisher_items(record)[0]['kind'] == ''
    record['parsed']['categories'] = ['Experimental articles']
    assert publisher_items(record)[0]['kind'] == 'research-article'
    assert publisher_items(record)[0]['native_section_type_evidence']['native_article_type'] == ''
    record['parsed']['issn'] = ['unreviewed-journal']
    assert publisher_items(record)[0]['kind'] == ''
    record['parsed']['categories'] = ['Reviews']
    assert publisher_items(record)[0]['kind'] == 'review'
    record['route'] = 'ecovector_oai_nlm'
    assert publisher_items(record)[0]['kind'] == ''
