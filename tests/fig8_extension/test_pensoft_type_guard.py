from experiments.fig8_extension.fig8b_qualification_decide import decide, type_class


def test_generic_research_cannot_override_pensoft_heading():
    for heading in ['Monograph', 'Taxonomy & Inventories', 'Notes on Geographic Distribution', 'Checklist', 'FSKX (Food Safety Knowledge)', 'pensoft_unreviewed_heading']:
        assert type_class('research-article','publisher_pensoft',[heading]) == 'subtype_review_required'
    assert type_class('research-article','publisher_pensoft',['Conference Abstract']) == 'nonprimary'
    assert type_class('research-article','publisher_pensoft',['Research Article']) == 'research'


def test_cached_wrong_class_is_recomputed():
    evidence={'id':'test','identity':'matched','class':'research','kind':'research-article','family':'publisher_pensoft',
              'categories':['research-article','Conference Abstract'], 'online_date':'2024-01-01', 'title':'Study', 'receipt':'raw'}
    assert decide([evidence],[])['status']=='excluded_primary_nonresearch'
    evidence['categories']=['Monograph']
    assert decide([evidence],[])['status']=='pending_special_subtype_review'


def test_bdj_taxonomy_rule_is_journal_bound_and_keeps_holds():
    item={'id':'tax','identity':'matched','class':'subtype_review_required','kind':'research-article',
          'family':'publisher_pensoft','issns':['1314-2836'], 'title':'New species and taxonomy',
          'categories':['research-article','Taxonomy & Inventories','pensoft_unreviewed_heading'],
          'online_date':'2024-01-01','receipt':'native.xml'}
    assert decide([item],[])['status']=='included_verified_research'
    assert decide([{**item,'issns':['other']}],[])['status']=='pending_special_subtype_review'
    assert decide([{**item,'online_date':''}],[])['status']=='pending_primary_online_date'
    assert decide([{**item,'scope_review_required':True}],[])['status']=='pending_special_subtype_review'
    assert decide([{**item,'categories':['Conference Abstract']}],[])['status']=='excluded_primary_nonresearch'
    assert decide([{**item,'categories':['Data Paper (Biosciences)','pensoft_unreviewed_heading']}],[])['status']=='pending_special_subtype_review'
