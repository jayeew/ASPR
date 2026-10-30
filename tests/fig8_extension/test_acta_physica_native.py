"""Literal Chinese publisher type and date boundaries are source-bound."""
from experiments.fig8_extension.acta_physica_apply import native_source


def test_mixed_type_and_release_timestamp_are_not_research_or_online_date() -> None:
    journal = {'titleEn': 'Acta Physica Sinica', 'titleCn': '物理学报', 'issnEpub': '1000-3290'}
    article = {'titleEn': 'Example', 'titleCn': '例子', 'type': '综述/研究论文',
               'releaseProgress': {'lastReleaseTime': '2023-06-08 14:09'}}
    paper = {'title': 'Example', 'doi': '10.7498/test'}
    receipt = {'raw_receipt': 'example.json.gz', 'url': 'https://wulixb.iphy.ac.cn/data/tplData'}
    value = native_source(article, paper, receipt, journal)
    assert value['kind'] == '综述/研究论文'
    assert value['online_date'] == ''
    article['type'] = '研究论文'
    assert native_source(article, paper, receipt, journal)['kind'] == 'Research article'
    article['type'] = '综述'
    assert native_source(article, paper, receipt, journal)['kind'] == 'Review'
