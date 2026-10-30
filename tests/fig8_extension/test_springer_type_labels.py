from experiments.fig8_extension.fig8b_qualification_build import match
from experiments.fig8_extension.fig8b_qualification_decide import type_class


def test_scoped_original_labels_and_review():
    assert type_class('OriginalPaper','publisher_springer')=='research'
    assert type_class('OriginalPaper','publisher_other')=='unresolved_type'
    assert type_class('ReviewPaper','publisher_springer')=='nonprimary'
    assert type_class('OriginalPaper','publisher_springer',['Case Report'])=='subtype_review_required'
    assert type_class('OriginalPaper','publisher_springer',['Review Article'])=='subtype_review_required'


def test_review_title_retains_hold_even_with_original_label():
    title='A systematic review of materials'
    row={'work_id':'W1','doi':'10.1007/example','title':title,'source_id':'S1','journal_name':'Example'}
    item={'title':title,'source_id':'S1','family':'publisher_springer','kind':'OriginalPaper','receipt':'cached','online_date':'2024-01-01','categories':[],'url':'https://link.springer.com/article/10.1007/example'}
    value=match(row,item,{'issn':'[]'})
    assert value['class']=='subtype_review_required'
    assert value['policy_url'].startswith('https://www.springernature.com/')
