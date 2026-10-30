from experiments.fig8_extension.fig8b_qualification_decide import portfolio_label


def test_retained_jats_date_objects_replay_like_normalized_imprint_dates() -> None:
    date_source={'id':'date','identity':'matched','family':'crossref_publisher_deposit','online_date':'2024-03-04'}
    native={'id':'native','identity':'matched','family':'archived_jats','online_date':'','publishers':['Elsevier'],'publication_dates':['2024-03-04']}
    expected=portfolio_label('',[date_source,native],2024)
    structured={**native,'publication_dates':[{'attributes':{'pub-type':'epub'},'year':'2024','month':'3','day':'4'}]}
    assert portfolio_label('',[date_source,structured],2024)==expected
    assert expected[0]=='False'
    issue={**structured,'publication_dates':[{'attributes':{'pub-type':'issue'},'year':'2024','month':'3','day':'4'}]}
    assert portfolio_label('',[date_source,issue],2024)[0]=='unresolved'
