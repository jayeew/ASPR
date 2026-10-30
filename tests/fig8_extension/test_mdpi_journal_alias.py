from experiments.fig8_extension.mdpi_cache_apply import evidence


def test_carbon_alias_requires_doi_title_and_registered_issn():
    item = {'fields': {'title': ['Example title'], 'source': ['C'], 'date': ['2023-01-01']},
            'oai_identifier': 'oai:mdpi.com:/2311-5629/9/1/1/', 'dois': ['10.3390/example'],
            'native_type_sets': ['research-article'], 'oai_datestamp': '2023-01-01'}
    paper = {'doi': '10.3390/example', 'title': 'Example title', 'journal': 'C – Journal of Carbon Research', 'source_id': 'S1'}
    receipt = {'raw_receipt': 'retained.xml.gz'}
    result = evidence(item, paper, {'2311-5629'}, receipt, 'digest')
    assert result and result['online_date'] == '' and result['journal_identity_rule_receipt']
    assert evidence(item, {**paper, 'doi': '10.3390/wrong'}, {'2311-5629'}, receipt, 'digest') is None
    assert evidence(item, {**paper, 'title': 'Other paper'}, {'2311-5629'}, receipt, 'digest') is None
    assert evidence(item, paper, {'0000-0000'}, receipt, 'digest') is None
    assert evidence(item, {**paper, 'journal': 'Other journal'}, {'2311-5629'}, receipt, 'digest') is None


def test_analytica_alias_preserves_review_type_and_date_uncertainty():
    item = {'fields': {'title': ['Example review'], 'source': ['Analytica'], 'date': ['2025-02-16']},
            'oai_identifier': 'oai:mdpi.com:/2673-4532/6/1/5/', 'dois': ['10.3390/analytica6010005'],
            'native_type_sets': ['review-article'], 'oai_datestamp': '2025-02-16'}
    paper = {'doi': item['dois'][0], 'title': 'Example review',
             'journal': 'Analytica—A Journal of Analytical Chemistry and Chemical Analysis', 'source_id': 'S2'}
    result = evidence(item, paper, {'2673-4532'}, {'raw_receipt': 'retained.xml.gz'}, 'digest')
    assert result['kind'] == 'review-article'
    assert result['online_date'] == ''
    assert 'analytica_alias_review.json' in result['journal_identity_rule_receipt']
    assert evidence(item, paper, {'2311-5629'}, {'raw_receipt': 'retained.xml.gz'}, 'digest') is None


def test_eng_alias_never_relaxes_other_identity_checks():
    item = {'fields': {'title': ['Example'], 'source': ['Eng'], 'date': ['2025-01-01']},
            'oai_identifier': 'oai:mdpi.com:/2673-4117/6/1/1/', 'dois': ['10.3390/eng6010001'],
            'native_type_sets': ['review-article'], 'oai_datestamp': '2025-01-01'}
    paper = {'doi': item['dois'][0], 'title': 'Example', 'journal': 'Eng—Advances in Engineering', 'source_id': 'S3'}
    receipt = {'raw_receipt': 'retained.xml.gz'}
    result = evidence(item, paper, {'2673-4117'}, receipt, 'digest')
    assert result['kind']=='review-article' and result['online_date']==''
    assert 'eng_alias_review.json' in result['journal_identity_rule_receipt']
    for field, value in [('doi','10.3390/other'),('title','Other title'),('journal','Other journal')]:
        assert evidence(item,{**paper,field:value},{'2673-4117'},receipt,'digest') is None
    assert evidence(item,paper,{'2673-4109'},receipt,'digest') is None
