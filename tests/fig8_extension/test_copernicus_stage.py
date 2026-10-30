from experiments.fig8_extension.copernicus_stage import adapted
from experiments.fig8_extension.publisher_delta_apply import publisher_items
from experiments.fig8_extension.fig8b_qualification_decide import type_class


def test_no_online_date_inference_and_subtype_retained():
    native={'article_type':'research-article','categories':['Review'],
        'dates':[{'attributes':{},'year':'2024','month':'1','day':'1'}],
        'issn':['1234'],'title':['Title'],'journal_titles':['Journal']}
    result=adapted(native)
    assert result['online_date']==''
    evidence=publisher_items({'parsed':result,'route':'copernicus','url':'official'})
    assert all(type_class(v['kind'],v['family'],v['categories'])!='research' for v in evidence)
