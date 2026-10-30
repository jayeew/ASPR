from experiments.fig8_extension.fig8b_qualification_decide import decide


def test_prior_nature_review_survives_title_only_flag_but_not_new_conflict():
    item = {'id':'a','identity':'matched','family':'nature_v1','kind':'Article',
            'title':'GWAS and meta-analysis identifies variants', 'class':'subtype_review_required',
            'title_scope_flag':'Explicit title phrase requires subtype adjudication',
            'online_date':'2023-05-01','categories':[]}
    assert decide([item],[])['status']=='included_verified_research'
    assert decide([{**item,'family':'nature_catalogue'}],[])['status']=='pending_special_subtype_review'
    assert decide([item,{**item,'id':'b','kind':'Review','class':'nonprimary'}],[])['status']!='included_verified_research'
    assert decide([item],[{'types':['Retracted Publication']}])['status']!='included_verified_research'
    assert decide([{**item,'scope_review_required':True}],[])['status']=='pending_special_subtype_review'
    assert decide([{**item,'title':'Systematic review and meta-analysis'}],[])['status']=='pending_special_subtype_review'


def test_pnas_alias_is_exact_and_issn_bound():
    from experiments.fig8_extension.fig8b_qualification_build import pnas_journal_alias
    short = 'Proceedings of the National Academy of Sciences'
    full = short + ' of the United States of America'
    assert pnas_journal_alias(short,[full],{'1091-6490'})
    assert not pnas_journal_alias(short,[full],{'wrong'})
    assert not pnas_journal_alias('PNAS Nexus',[full],{'1091-6490'})
